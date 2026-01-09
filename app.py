# ============================================================
# Futebol Odds Platform — UI (Ligas/Temporadas/Times) + Poisson + ML
# Arquivo único: app.py
#
# - Seleção: Campeonato (liga), Temporada, Time Casa, Time Fora
# - Importa dados automaticamente (mmz4281/<season>/<league>.csv)
# - Usa 2 temporadas: 2526 e 2425 (para ligas listadas)
# - Poisson: 1X2 FT/HT, BTTS FT, Over/Under FT (0.5..5.5), Over/Under HT (0.5,1.5,2.5),
#           Placar exato FT (0–5) e HT (0–3) + "Outros"
# - ML calibrador: melhora probas quando aprovado por validação temporal (2425->2526)
# - Dicas: mercados onde Poisson e ML concordam mais
# ============================================================

import time
from math import exp, factorial
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import requests_cache
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

# ----------------------------
# Pastas
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data/raw/mmz4281"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Cache + throttling
# ----------------------------
requests_cache.install_cache(
    cache_name=str(BASE_DIR / "data/http_cache"),
    backend="sqlite",
    expire_after=60 * 60 * 24,
)
THROTTLE_S = 0.20  # respeitar fornecedor

# ----------------------------
# Dados fixos (2 temporadas)
# ----------------------------
SEASONS = ["2526", "2425"]  # mais recente primeiro
MMZ_BASE = "https://www.football-data.co.uk/mmz4281"

LEAGUES: Dict[str, str] = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "SC0": "Scotland Premiership",
    "D1": "Bundesliga",
    "D2": "2. Bundesliga",
    "I1": "Serie A",
    "I2": "Serie B",
    "SP1": "La Liga",
    "SP2": "La Liga 2",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "B1": "Belgium First Division A",
    "P1": "Portugal Primeira Liga",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
    "EC": "Euro Competitions",
}

# ----------------------------
# Poisson config
# ----------------------------
HOME_ADV = 1.10
MAX_FT = 10
MAX_HT = 6

FT_OU_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
HT_OU_LINES = [0.5, 1.5, 2.5]

SHOW_CS_FT_MAX = 5
SHOW_CS_HT_MAX = 3

# Shrinkage (estabiliza)
K_SHRINK = 40

# ============================================================
# Helpers: download / load
# ============================================================

def csv_url(season: str, league: str) -> str:
    return f"{MMZ_BASE}/{season}/{league}.csv"

def csv_path(season: str, league: str) -> Path:
    return RAW_DIR / season / f"{league}.csv"

def ensure_csv(season: str, league: str) -> Path:
    p = csv_path(season, league)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size > 200:
        return p

    url = csv_url(season, league)
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    p.write_bytes(r.content)
    time.sleep(THROTTLE_S)
    return p

def load_league_data(league: str, seasons: List[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = ensure_csv(s, league)
        df = pd.read_csv(p).copy()
        df["season"] = s
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["FTHG", "FTAG"]).copy()
    return out

# ============================================================
# Poisson core
# ============================================================

def pmf(k: int, lam: float) -> float:
    return exp(-lam) * lam**k / factorial(k)

def score_matrix(lh: float, la: float, max_g: int) -> Dict[Tuple[int, int], float]:
    m: Dict[Tuple[int, int], float] = {}
    for h in range(max_g + 1):
        ph = pmf(h, lh)
        for a in range(max_g + 1):
            m[(h, a)] = ph * pmf(a, la)
    s = sum(m.values())
    if s <= 0:
        return {(0, 0): 1.0}
    for k in list(m.keys()):
        m[k] /= s
    return m

def odd(p: float) -> float:
    return round(1.0 / p, 2) if p > 0 else 9999.0

def markets_from_matrix(M: Dict[Tuple[int, int], float], lines: List[float]) -> Dict[str, float]:
    p_home = sum(p for (h, a), p in M.items() if h > a)
    p_draw = sum(p for (h, a), p in M.items() if h == a)
    p_away = sum(p for (h, a), p in M.items() if h < a)
    p_btts = sum(p for (h, a), p in M.items() if h > 0 and a > 0)

    totals = {}
    for line in lines:
        totals[line] = sum(p for (h, a), p in M.items() if (h + a) > line)

    out = {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_btts": p_btts,
    }
    for line, pv in totals.items():
        out[f"p_over_{line}"] = pv
    return out

# ============================================================
# Estimar forças (Poisson)
# ============================================================

def fit_strengths(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    avg_h = df["FTHG"].mean()
    avg_a = df["FTAG"].mean()
    teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))

    att_h, def_h, att_a, def_a = {}, {}, {}, {}
    n_home, n_away = {}, {}

    for t in teams:
        dh = df[df["HomeTeam"] == t]
        da = df[df["AwayTeam"] == t]
        n_home[t] = int(len(dh))
        n_away[t] = int(len(da))

        mh_sc = dh["FTHG"].mean() if len(dh) else avg_h
        mh_co = dh["FTAG"].mean() if len(dh) else avg_a
        ma_sc = da["FTAG"].mean() if len(da) else avg_a
        ma_co = da["FTHG"].mean() if len(da) else avg_h

        sh_h = len(dh) / (len(dh) + K_SHRINK) if len(dh) else 0.0
        sh_a = len(da) / (len(da) + K_SHRINK) if len(da) else 0.0

        mh_sc = sh_h * mh_sc + (1 - sh_h) * avg_h
        mh_co = sh_h * mh_co + (1 - sh_h) * avg_a
        ma_sc = sh_a * ma_sc + (1 - sh_a) * avg_a
        ma_co = sh_a * ma_co + (1 - sh_a) * avg_h

        att_h[t] = (mh_sc / avg_h) if avg_h > 0 else 1.0
        def_h[t] = (mh_co / avg_a) if avg_a > 0 else 1.0
        att_a[t] = (ma_sc / avg_a) if avg_a > 0 else 1.0
        def_a[t] = (ma_co / avg_h) if avg_h > 0 else 1.0

    return {
        "avg_h": avg_h,
        "avg_a": avg_a,
        "att_h": att_h,
        "def_h": def_h,
        "att_a": att_a,
        "def_a": def_a,
        "n_home": n_home,
        "n_away": n_away,
        "teams": teams,
    }

def lambdas_ft(model: Dict[str, Dict[str, float]], home: str, away: str) -> Tuple[float, float]:
    avg_h = model["avg_h"]
    avg_a = model["avg_a"]
    lh = avg_h * model["att_h"][home] * model["def_a"][away] * HOME_ADV
    la = avg_a * model["att_a"][away] * model["def_h"][home]
    return lh, la

def lambdas_ht(df_all: pd.DataFrame, lh_ft: float, la_ft: float) -> Tuple[float, float, str]:
    if ("HTHG" in df_all.columns) and ("HTAG" in df_all.columns):
        df_ht = df_all.dropna(subset=["HTHG", "HTAG"])
        ft_sum = float(df_all["FTHG"].sum() + df_all["FTAG"].sum())
        if len(df_ht) > 50 and ft_sum > 0:
            frac = float(df_ht["HTHG"].sum() + df_ht["HTAG"].sum()) / ft_sum
            return lh_ft * frac, la_ft * frac, f"HT por fração calibrada da liga (frac={frac:.3f})"
        return lh_ft * 0.45, la_ft * 0.45, "HT fallback (colunas HT insuficientes)"
    return lh_ft * 0.45, la_ft * 0.45, "HT fallback (sem colunas HT)"

# ============================================================
# ML (aprovado por validação temporal)
# ============================================================

def train_ml_models(df_all: pd.DataFrame) -> Dict[str, dict]:
    if "season" not in df_all.columns:
        return {"_status": {"ok": False, "reason": "sem coluna season"}}

    df_train = df_all[df_all["season"] == "2425"].copy()
    df_val = df_all[df_all["season"] == "2526"].copy()

    if len(df_train) < 80 or len(df_val) < 80:
        return {"_status": {"ok": False, "reason": "precisa de 2425 e 2526 com volume suficiente"}}

    poisson_train = fit_strengths(df_train)

    def build_binary(df_part: pd.DataFrame, kind: str, line: Optional[float] = None):
        X, y, pbase = [], [], []
        for _, r in df_part.iterrows():
            ht, at = r["HomeTeam"], r["AwayTeam"]
            if ht not in poisson_train["att_h"] or at not in poisson_train["att_a"]:
                continue
            lh, la = lambdas_ft(poisson_train, ht, at)
            M = score_matrix(lh, la, MAX_FT)
            mk = markets_from_matrix(M, FT_OU_LINES)

            dist = abs((lh + la) - (poisson_train["avg_h"] + poisson_train["avg_a"]))

            if kind == "btts":
                pb = mk["p_btts"]
                yt = 1 if (r["FTHG"] > 0 and r["FTAG"] > 0) else 0
            elif kind == "over_ft":
                assert line is not None
                pb = mk[f"p_over_{line}"]
                yt = 1 if (r["FTHG"] + r["FTAG"]) > line else 0
            else:
                continue

            X.append([pb, lh, la, poisson_train["att_h"][ht], poisson_train["att_a"][at], dist])
            y.append(yt)
            pbase.append(pb)

        if len(X) < 150:
            return None
        return np.array(X, float), np.array(y, int), np.array(pbase, float)

    out: Dict[str, dict] = {"_status": {"ok": True, "train": "2425", "val": "2526"}}

    # BTTS
    tr = build_binary(df_train, "btts")
    va = build_binary(df_val, "btts")
    if tr and va:
        Xtr, ytr, _ = tr
        Xva, yva, pbase = va
        m = LogisticRegression(max_iter=300)
        m.fit(Xtr, ytr)
        pml = m.predict_proba(Xva)[:, 1]
        ll_b = float(log_loss(yva, pbase))
        ll_m = float(log_loss(yva, pml))
        br_b = float(brier_score_loss(yva, pbase))
        br_m = float(brier_score_loss(yva, pml))
        out["btts"] = {"approved": (ll_m <= ll_b) and (br_m <= br_b),
                       "ll_poisson": ll_b, "ll_ml": ll_m,
                       "brier_poisson": br_b, "brier_ml": br_m,
                       "model": m}

    # Overs FT
    for line in FT_OU_LINES:
        tr = build_binary(df_train, "over_ft", line=line)
        va = build_binary(df_val, "over_ft", line=line)
        if not tr or not va:
            continue
        Xtr, ytr, _ = tr
        Xva, yva, pbase = va
        m = LogisticRegression(max_iter=300)
        m.fit(Xtr, ytr)
        pml = m.predict_proba(Xva)[:, 1]
        ll_b = float(log_loss(yva, pbase))
        ll_m = float(log_loss(yva, pml))
        br_b = float(brier_score_loss(yva, pbase))
        br_m = float(brier_score_loss(yva, pml))
        out[f"over_ft_{line}"] = {"approved": (ll_m <= ll_b) and (br_m <= br_b),
                                  "ll_poisson": ll_b, "ll_ml": ll_m,
                                  "brier_poisson": br_b, "brier_ml": br_m,
                                  "model": m}
    return out

# ============================================================
# UI helpers
# ============================================================

def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def select_options(values: List[str], selected: str) -> str:
    return "\n".join([f'<option value="{html_escape(v)}" {"selected" if v==selected else ""}>{html_escape(v)}</option>' for v in values])

def league_options(selected: str) -> str:
    out = []
    for code, name in LEAGUES.items():
        out.append(f'<option value="{code}" {"selected" if code==selected else ""}>{code} — {html_escape(name)}</option>')
    return "\n".join(out)

def correct_score_table(M: Dict[Tuple[int, int], float], max_show: int) -> Tuple[List[Tuple[str, float, float]], float]:
    rows = []
    shown = 0.0
    for h in range(max_show + 1):
        for a in range(max_show + 1):
            p = float(M.get((h, a), 0.0))
            shown += p
            rows.append((f"{h}-{a}", p, odd(p)))
    other = max(0.0, 1.0 - shown)
    return rows, other

# ============================================================
# APP
# ============================================================

app = FastAPI(title="Futebol Odds Platform", version="2.0.1")

@app.get("/", response_class=HTMLResponse)
def view(
    league: str = "E0",
    season: str = "2526",
    home_team: str = "",
    away_team: str = "",
):
    # carregar dados (2 temporadas)
    try:
        df_all = load_league_data(league, SEASONS)
    except Exception as e:
        return HTMLResponse(f"<pre>Erro ao carregar dados: {html_escape(str(e))}</pre>", status_code=500)

    teams = sorted(set(df_all["HomeTeam"]).union(df_all["AwayTeam"]))
    if not teams:
        return HTMLResponse("<pre>Sem times encontrados.</pre>", status_code=500)

    if home_team not in teams:
        home_team = teams[0]
    if away_team not in teams:
        away_team = teams[1] if len(teams) > 1 else teams[0]
    if home_team == away_team and len(teams) > 1:
        away_team = teams[1]

    poisson = fit_strengths(df_all)
    lh_ft, la_ft = lambdas_ft(poisson, home_team, away_team)

    # FT
    M_FT = score_matrix(lh_ft, la_ft, MAX_FT)
    mk_ft = markets_from_matrix(M_FT, FT_OU_LINES)

    # HT
    lh_ht, la_ht, ht_note = lambdas_ht(df_all, lh_ft, la_ft)
    M_HT = score_matrix(lh_ht, la_ht, MAX_HT)
    mk_ht = markets_from_matrix(M_HT, HT_OU_LINES)

    # ML
    ml = train_ml_models(df_all)
    ml_status = ml.get("_status", {})
    ml_ok = bool(ml_status.get("ok"))

    def apply_ml_binary(key: str, p_poi: float, feat: List[float]) -> Tuple[float, str]:
        if ml_ok and key in ml and ml[key].get("approved") and ml[key].get("model") is not None:
            model = ml[key]["model"]
            p_ml = float(model.predict_proba(np.array([feat], float))[0, 1])
            return p_ml, "ML (aprovado)"
        return float(p_poi), "Poisson (ML rejeitado)"

    dist = abs((lh_ft + la_ft) - (poisson["avg_h"] + poisson["avg_a"]))
    feat_bin = [0.0, lh_ft, la_ft, poisson["att_h"][home_team], poisson["att_a"][away_team], dist]

    # 1X2 (Poisson apenas nesta versão — podemos calibrar depois)
    pH_p, pD_p, pA_p = mk_ft["p_home"], mk_ft["p_draw"], mk_ft["p_away"]

    # BTTS
    feat_bin[0] = mk_ft["p_btts"]
    p_btts_ml, tag_btts = apply_ml_binary("btts", mk_ft["p_btts"], feat_bin)

    # Overs FT
    overs_ft_rows = []
    tips = []
    for line in FT_OU_LINES:
        p_poi = mk_ft[f"p_over_{line}"]
        feat_bin[0] = p_poi
        key = f"over_ft_{line}"
        p_ml, tag = apply_ml_binary(key, p_poi, feat_bin)
        overs_ft_rows.append((line, p_poi, p_ml, tag))
        tips.append((abs(p_poi - p_ml), f"Over FT {line}", p_poi, p_ml, odd(p_poi), odd(p_ml), tag))

    # Overs HT (Poisson)
    overs_ht_rows = []
    for line in HT_OU_LINES:
        p_poi = mk_ht[f"p_over_{line}"]
        overs_ht_rows.append((line, p_poi, odd(p_poi)))

    # placar exato
    cs_ft_rows, cs_ft_other = correct_score_table(M_FT, SHOW_CS_FT_MAX)
    cs_ht_rows, cs_ht_other = correct_score_table(M_HT, SHOW_CS_HT_MAX)

    # dicas: incluir BTTS e 1X2
    tips.append((abs(mk_ft["p_btts"] - p_btts_ml), "BTTS FT (Sim)", mk_ft["p_btts"], p_btts_ml, odd(mk_ft["p_btts"]), odd(p_btts_ml), tag_btts))
    tips.append((0.0, "1X2 FT Home (Poisson)", pH_p, pH_p, odd(pH_p), odd(pH_p), "Poisson"))
    tips.append((0.0, "1X2 FT Draw (Poisson)", pD_p, pD_p, odd(pD_p), odd(pD_p), "Poisson"))
    tips.append((0.0, "1X2 FT Away (Poisson)", pA_p, pA_p, odd(pA_p), odd(pA_p), "Poisson"))

    tips_sorted = sorted(tips, key=lambda x: x[0])
    top_tips = tips_sorted[:8]

    # montar HTML das dicas fora do f-string (corrige o erro do Python)
    tips_cards = []
    for diff, name, p1, p2, o1, o2, tag in top_tips:
        card = (
            '<div class="border rounded p-3">'
            f'<div class="font-semibold">{html_escape(name)}</div>'
            f'<div class="text-xs text-gray-500">Método: {html_escape(tag)}</div>'
            f'<div class="mt-1">Poisson: p={p1:.4f} (odd {o1})</div>'
            f'<div>ML/Final: p={p2:.4f} (odd {o2})</div>'
            f'<div class="text-xs text-gray-500 mt-1">Diferença: {diff:.4f}</div>'
            '</div>'
        )
        tips_cards.append(card)
    tips_html = "\n".join(tips_cards)

    league_select = league_options(league)
    season_select = select_options(SEASONS, season)
    home_select = select_options(teams, home_team)
    away_select = select_options(teams, away_team)

    ml_line = "ML: indisponível."
    if ml_ok:
        ml_line = f"Treino ML: {ml_status.get('train')} → Validação: {ml_status.get('val')} (aplica só se melhorar)"

    return f"""
    <!doctype html>
    <html lang="pt-br">
    <head>
      <meta charset="utf-8"/>
      <title>Futebol Odds Platform</title>
      <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100">
      <div class="max-w-6xl mx-auto p-6">
        <div class="bg-white rounded-2xl shadow p-6 mb-6">
          <h1 class="text-2xl font-bold mb-2">⚽ Futebol Odds Platform</h1>
          <p class="text-sm text-gray-600">Odds estatísticas (Poisson) + calibração ML quando aprovada — sem promessa de lucro.</p>
        </div>

        <div class="bg-white rounded-2xl shadow p-6 mb-6">
          <h2 class="text-lg font-semibold mb-4">Seleção</h2>

          <form method="get" class="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label class="text-sm font-medium">Campeonato</label>
              <select name="league" class="w-full mt-1 border rounded p-2">{league_select}</select>
            </div>
            <div>
              <label class="text-sm font-medium">Temporadas (base)</label>
              <select name="season" class="w-full mt-1 border rounded p-2">{season_select}</select>
              <p class="text-xs text-gray-500 mt-1">Modelo usa 2526 + 2425 sempre (conforme pedido).</p>
            </div>
            <div>
              <label class="text-sm font-medium">Time 1 (Casa)</label>
              <select name="home_team" class="w-full mt-1 border rounded p-2">{home_select}</select>
            </div>
            <div>
              <label class="text-sm font-medium">Time 2 (Fora)</label>
              <select name="away_team" class="w-full mt-1 border rounded p-2">{away_select}</select>
            </div>

            <div class="md:col-span-4 flex gap-2 mt-2">
              <button class="px-4 py-2 rounded bg-black text-white">Gerar Odds</button>
              <a class="px-4 py-2 rounded border" href="/">Reset</a>
            </div>
          </form>

          <div class="mt-4 text-sm text-gray-700">
            <b>Dados:</b> baixa automaticamente 2526 e 2425 da liga escolhida.
            <br/>
            <b>{html_escape(ml_line)}</b>
          </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div class="bg-white rounded-2xl shadow p-6">
            <h2 class="text-lg font-semibold mb-2">Jogo</h2>
            <p class="text-sm mb-3"><b>{html_escape(home_team)}</b> (Casa) x <b>{html_escape(away_team)}</b> (Fora)</p>
            <p class="text-xs text-gray-500">HT: {html_escape(ht_note)}</p>

            <div class="mt-4">
              <h3 class="font-semibold">1X2 (FT)</h3>
              <div class="text-sm mt-2">
                <div>Poisson: Home <b>{odd(pH_p)}</b> | Draw <b>{odd(pD_p)}</b> | Away <b>{odd(pA_p)}</b></div>
              </div>
            </div>

            <div class="mt-4">
              <h3 class="font-semibold">BTTS (FT)</h3>
              <div class="text-sm mt-2">
                <div>Poisson (Sim): <b>{odd(mk_ft["p_btts"])}</b></div>
                <div>{html_escape(tag_btts)} (Sim): <b>{odd(p_btts_ml)}</b></div>
              </div>
            </div>

            <div class="mt-4">
              <h3 class="font-semibold">Over/Under (FT)</h3>
              <div class="text-sm mt-2">
                {''.join([f"<div>Over {line}: Poisson <b>{odd(pp)}</b> | Final <b>{odd(pm)}</b> <span class='text-xs text-gray-500'>({html_escape(tag)})</span></div>"
                          for (line, pp, pm, tag) in overs_ft_rows])}
              </div>
            </div>

            <div class="mt-4">
              <h3 class="font-semibold">Over/Under (HT)</h3>
              <div class="text-sm mt-2">
                {''.join([f"<div>Over {line}: Poisson <b>{o}</b></div>"
                          for (line, pp, o) in overs_ht_rows])}
              </div>
            </div>
          </div>

          <div class="bg-white rounded-2xl shadow p-6">
            <h2 class="text-lg font-semibold mb-2">Placar Exato</h2>

            <h3 class="font-semibold mt-3">FT (0–{SHOW_CS_FT_MAX})</h3>
            <div class="grid grid-cols-2 gap-2 text-sm mt-2">
              {''.join([f"<div class='flex justify-between border rounded p-2'><span>{s}</span><b>{o}</b></div>"
                        for (s, p, o) in cs_ft_rows])}
            </div>
            <p class="text-xs text-gray-500 mt-2">Outros (tail): prob={cs_ft_other:.4f} → odd≈{odd(cs_ft_other) if cs_ft_other>0 else 9999}</p>

            <h3 class="font-semibold mt-6">HT (0–{SHOW_CS_HT_MAX})</h3>
            <div class="grid grid-cols-2 gap-2 text-sm mt-2">
              {''.join([f"<div class='flex justify-between border rounded p-2'><span>{s}</span><b>{o}</b></div>"
                        for (s, p, o) in cs_ht_rows])}
            </div>
            <p class="text-xs text-gray-500 mt-2">Outros (tail): prob={cs_ht_other:.4f} → odd≈{odd(cs_ht_other) if cs_ht_other>0 else 9999}</p>
          </div>
        </div>

        <div class="bg-white rounded-2xl shadow p-6 mt-6">
          <h2 class="text-lg font-semibold mb-2">Dicas (onde Poisson e ML concordam mais)</h2>
          <p class="text-sm text-gray-600 mb-4">
            Critério: menor diferença |P(Poisson) − P(ML)|. Se ML foi rejeitado, a dica indica estabilidade do Poisson.
          </p>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
            {tips_html}
          </div>

          <p class="text-xs text-gray-500 mt-6">
            Observação: este produto fornece estimativas estatísticas e calibração quando aprovada por validação temporal.
            Não há promessa de lucro.
          </p>
        </div>
      </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
