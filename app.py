# ============================================================
# Futebol Odds Platform — UI completa + Poisson + ML + Placar Exato FT/HT
# Ajustes:
# - 5 dicas (somente mercados tradicionais)
# - Under FT/HT
# - BTTS (Ambas Marcam) SIM/NÃO
# - Dupla chance 1X / 12 / 2X
# - Cards: 5 placares FT mais prováveis e 5 menos prováveis
# - NOVO: limitar cards de placar a no máximo 5x5
# - NOVO: Badge do Backtest Avançado nas 5 melhores apostas
# ============================================================

import time
from math import exp, factorial
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import requests_cache
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
import json

# =========================
# Backtest Avançado (badge)
# =========================

# Mapa (mercados da HOME/UI -> mercados do backtest_avancado)
MARKET_UI_TO_ADV = {
    # BTTS FT
    "btts_ft_yes": "btts_yes",
    "btts_ft_no": "btts_no",

    # 1X2 FT
    "1x2_home": "1x2_home",
    "1x2_draw": "1x2_draw",
    "1x2_away": "1x2_away",

    # Dupla chance FT
    "dc_1x": "dc_1x",
    "dc_12": "dc_12",
    "dc_2x": "dc_2x",

    # Totais FT (principais)
    "over_0.5": "over_0.5",
    "over_1.5": "over_1.5",
    "over_2.5": "over_2.5",
    "over_3.5": "over_3.5",
    "under_0.5": "under_0.5",
    "under_1.5": "under_1.5",
    "under_2.5": "under_2.5",
    "under_3.5": "under_3.5",
}

_ADV_SUMMARY_CACHE = {"path": None, "data": None}

def _load_adv_summary() -> dict:
    """
    Lê static/backtest_adv/summary.json com cache simples.
    Retorna {} se não existir/der erro.
    """
    try:
        root = Path(__file__).resolve().parent
        p = root / "static" / "backtest_adv" / "summary.json"
        if not p.exists():
            return {}
        if _ADV_SUMMARY_CACHE["path"] == str(p) and _ADV_SUMMARY_CACHE["data"] is not None:
            return _ADV_SUMMARY_CACHE["data"]
        data = json.loads(p.read_text(encoding="utf-8"))
        _ADV_SUMMARY_CACHE["path"] = str(p)
        _ADV_SUMMARY_CACHE["data"] = data
        return data
    except Exception:
        return {}

def adv_badge_for(league_code: str, market_ui_key: str) -> dict:
    """
    Retorna um badge simples baseado no summary.json do backtest avançado.
    status: good | bad | neutral | missing
    """
    market_adv = MARKET_UI_TO_ADV.get(market_ui_key)
    if not market_adv:
        return {"status": "missing", "label": "Sem histórico", "detail": "Mercado não mapeado."}

    data = _load_adv_summary()
    if not data:
        return {"status": "missing", "label": "Sem histórico", "detail": "summary.json não encontrado."}

    all_rows = []
    all_rows.extend(data.get("best", []) or [])
    all_rows.extend(data.get("worst", []) or [])

    row = next((r for r in all_rows if r.get("league") == league_code and r.get("market") == market_adv), None)
    if not row:
        return {"status": "missing", "label": "Sem histórico", "detail": "Liga/mercado não aparece no resumo."}

    dlog = ((row.get("delta") or {}).get("logloss"))
    try:
        dlog = float(dlog)
    except Exception:
        dlog = None

    if dlog is None:
        return {"status": "neutral", "label": "Neutro", "detail": "Sem delta numérico."}
    if dlog > 0:
        return {"status": "good", "label": "📈 Valor", "detail": f"ML melhorou (ΔLogLoss={dlog:.4f})"}
    if dlog < 0:
        return {"status": "bad", "label": "⚠️ Cautela", "detail": f"ML piorou (ΔLogLoss={dlog:.4f})"}
    return {"status": "neutral", "label": "Neutro", "detail": "ΔLogLoss≈0"}

# ----------------------------
# Pastas
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent

# ----------------------------
# Templates (Jinja) + Static
# ----------------------------
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

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
THROTTLE_S = 0.20

# ----------------------------
# Temporadas (3) + pesos
# ----------------------------
SEASONS = ["2526", "2425", "2324"]
SEASON_WEIGHTS: Dict[str, float] = {"2526": 1.0, "2425": 0.7, "2324": 0.5}

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
    "EC": "Conference",
}

# ----------------------------
# Config Poisson
# ----------------------------
HOME_ADV = 1.10
MAX_FT = 10
MAX_HT = 6

FT_OU_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
HT_OU_LINES = [0.5, 1.5, 2.5]

SHOW_CS_FT_MAX = 5
SHOW_CS_HT_MAX = 3

# Limite dos cards de placar (mais/menos provável)
CARDS_CS_FT_MAX = 5  # <= pedido do usuário: máximo 5x5

# shrinkage
K_SHRINK = 40

# Defaults das faixas (aparecem na tela)
DEFAULT_ODD_MIN = 1.55
DEFAULT_ODD_MAX = 4.50
DEFAULT_P_MIN = 0.22
DEFAULT_P_MAX = 0.75

# ============================================================
# Download / Load
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
        df["season_w"] = float(SEASON_WEIGHTS.get(s, 1.0))
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    for col in ["FTHG", "FTAG", "HTHG", "HTAG"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

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

    totals_over = {}
    for line in lines:
        totals_over[line] = sum(p for (h, a), p in M.items() if (h + a) > line)

    out = {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_btts": p_btts,
    }
    for line, pv in totals_over.items():
        out[f"p_over_{line}"] = pv
        out[f"p_under_{line}"] = max(0.0, 1.0 - pv)
    return out

# ============================================================
# Estimar forças — ponderado por recência
# ============================================================

def wmean(series: pd.Series, weights: pd.Series, default: float) -> float:
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = s.notna() & w.notna()
    s = s[mask]
    w = w[mask]
    if len(s) == 0:
        return float(default)
    ws = float(w.sum())
    if ws <= 0:
        return float(default)
    return float((s * w).sum() / ws)

def fit_strengths(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    avg_h = wmean(df["FTHG"], df["season_w"], default=float(df["FTHG"].mean()))
    avg_a = wmean(df["FTAG"], df["season_w"], default=float(df["FTAG"].mean()))
    teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))

    att_h, def_h, att_a, def_a = {}, {}, {}, {}

    for t in teams:
        dh = df[df["HomeTeam"] == t]
        da = df[df["AwayTeam"] == t]

        mh_sc = wmean(dh["FTHG"], dh["season_w"], default=avg_h)
        mh_co = wmean(dh["FTAG"], dh["season_w"], default=avg_a)
        ma_sc = wmean(da["FTAG"], da["season_w"], default=avg_a)
        ma_co = wmean(da["FTHG"], da["season_w"], default=avg_h)

        n_eff_h = float(dh["season_w"].sum()) if len(dh) else 0.0
        n_eff_a = float(da["season_w"].sum()) if len(da) else 0.0

        sh_h = n_eff_h / (n_eff_h + K_SHRINK) if n_eff_h > 0 else 0.0
        sh_a = n_eff_a / (n_eff_a + K_SHRINK) if n_eff_a > 0 else 0.0

        mh_sc = sh_h * mh_sc + (1 - sh_h) * avg_h
        mh_co = sh_h * mh_co + (1 - sh_h) * avg_a
        ma_sc = sh_a * ma_sc + (1 - sh_a) * avg_a
        ma_co = sh_a * ma_co + (1 - sh_a) * avg_h

        att_h[t] = (mh_sc / avg_h) if avg_h > 0 else 1.0
        def_h[t] = (mh_co / avg_a) if avg_a > 0 else 1.0
        att_a[t] = (ma_sc / avg_a) if avg_a > 0 else 1.0
        def_a[t] = (ma_co / avg_h) if avg_h > 0 else 1.0

    return {
        "avg_h": float(avg_h),
        "avg_a": float(avg_a),
        "att_h": att_h,
        "def_h": def_h,
        "att_a": att_a,
        "def_a": def_a,
        "teams": teams,
    }

def lambdas_ft(model: Dict[str, Dict[str, float]], home: str, away: str) -> Tuple[float, float]:
    avg_h = model["avg_h"]
    avg_a = model["avg_a"]
    lh = avg_h * model["att_h"][home] * model["def_a"][away] * HOME_ADV
    la = avg_a * model["att_a"][away] * model["def_h"][home]
    return float(lh), float(la)

def lambdas_ht(df_all: pd.DataFrame, lh_ft: float, la_ft: float) -> Tuple[float, float, str, bool]:
    if ("HTHG" in df_all.columns) and ("HTAG" in df_all.columns):
        df_ht = df_all.dropna(subset=["HTHG", "HTAG"]).copy()
        if len(df_ht) > 50:
            ft_sum = float(((df_all["FTHG"] + df_all["FTAG"]) * df_all["season_w"]).sum())
            ht_sum = float(((df_ht["HTHG"] + df_ht["HTAG"]) * df_ht["season_w"]).sum())
            if ft_sum > 0 and ht_sum > 0:
                frac = ht_sum / ft_sum
                return lh_ft * frac, la_ft * frac, f"HT por fração calibrada da liga (frac={frac:.3f})", True
        return lh_ft * 0.45, la_ft * 0.45, "HT fallback (colunas HT insuficientes)", False
    return lh_ft * 0.45, la_ft * 0.45, "HT fallback (sem colunas HT)", False

# ============================================================
# ML (FT) — Treino: 2324+2425 / Val: 2526
# ============================================================

def train_ml_models(df_all: pd.DataFrame) -> Dict[str, dict]:
    if "season" not in df_all.columns:
        return {"_status": {"ok": False, "reason": "sem coluna season"}}

    df_train = df_all[df_all["season"].isin(["2324", "2425"])].copy()
    df_val = df_all[df_all["season"] == "2526"].copy()

    if len(df_train) < 120 or len(df_val) < 80:
        return {"_status": {"ok": False, "reason": "dados insuficientes (treino/val)"}}

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

        if len(X) < 180:
            return None
        return np.array(X, float), np.array(y, int), np.array(pbase, float)

    out: Dict[str, dict] = {"_status": {"ok": True, "train": "2324+2425", "val": "2526"}}

    tr = build_binary(df_train, "btts")
    va = build_binary(df_val, "btts")
    if tr and va:
        Xtr, ytr, _ = tr
        Xva, yva, pbase = va
        m = LogisticRegression(max_iter=400)
        m.fit(Xtr, ytr)
        pml = m.predict_proba(Xva)[:, 1]
        ll_b = float(log_loss(yva, pbase))
        ll_m = float(log_loss(yva, pml))
        br_b = float(brier_score_loss(yva, pbase))
        br_m = float(brier_score_loss(yva, pml))
        out["btts"] = {
            "approved": (ll_m <= ll_b) and (br_m <= br_b),
            "ll_poisson": ll_b, "ll_ml": ll_m,
            "brier_poisson": br_b, "brier_ml": br_m,
            "model": m
        }

    for line in FT_OU_LINES:
        tr = build_binary(df_train, "over_ft", line=line)
        va = build_binary(df_val, "over_ft", line=line)
        if not tr or not va:
            continue
        Xtr, ytr, _ = tr
        Xva, yva, pbase = va
        m = LogisticRegression(max_iter=400)
        m.fit(Xtr, ytr)
        pml = m.predict_proba(Xva)[:, 1]
        ll_b = float(log_loss(yva, pbase))
        ll_m = float(log_loss(yva, pml))
        br_b = float(brier_score_loss(yva, pbase))
        br_m = float(brier_score_loss(yva, pml))
        out[f"over_ft_{line}"] = {
            "approved": (ll_m <= ll_b) and (br_m <= br_b),
            "ll_poisson": ll_b, "ll_ml": ll_m,
            "brier_poisson": br_b, "brier_ml": br_m,
            "model": m
        }
    return out

# ============================================================
# UI helpers
# ============================================================

def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def select_options(values: List[str], selected: str) -> str:
    return "\n".join([f'<option value="{html_escape(v)}" {"selected" if v==selected else ""}>{html_escape(v)}</option>' for v in values])

def clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

def parse_float(x: Optional[float], default: float) -> float:
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)

def top_and_bottom_scores_limited(
    M: Dict[Tuple[int, int], float],
    k: int = 5,
    max_score: int = 5
) -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float, float]]]:
    items = [((h, a), float(p)) for (h, a), p in M.items() if (0 <= h <= max_score and 0 <= a <= max_score)]
    items.sort(key=lambda x: x[1], reverse=True)

    top = items[:k]
    bottom = sorted(items[-k:], key=lambda x: x[1])

    top_rows = [(f"{h}-{a}", p, odd(p)) for ((h, a), p) in top]
    bottom_rows = [(f"{h}-{a}", p, odd(p)) for ((h, a), p) in bottom]
    return top_rows, bottom_rows

# ============================================================
# App
# ============================================================

app = FastAPI(title="Futebol Odds Platform", version="2.3.2")

# Static files (imagens, css, etc.)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def view(
    league: str = "E0",
    season: str = "2526",
    home_team: str = "",
    away_team: str = "",
    odd_min: Optional[float] = DEFAULT_ODD_MIN,
    odd_max: Optional[float] = DEFAULT_ODD_MAX,
    p_min: Optional[float] = DEFAULT_P_MIN,
    p_max: Optional[float] = DEFAULT_P_MAX,
):
    odd_min = parse_float(odd_min, DEFAULT_ODD_MIN)
    odd_max = parse_float(odd_max, DEFAULT_ODD_MAX)
    p_min = parse_float(p_min, DEFAULT_P_MIN)
    p_max = parse_float(p_max, DEFAULT_P_MAX)

    if odd_min > odd_max:
        odd_min, odd_max = odd_max, odd_min
    if p_min > p_max:
        p_min, p_max = p_max, p_min

    odd_min = clamp_float(odd_min, 1.01, 1000.0)
    odd_max = clamp_float(odd_max, 1.01, 1000.0)
    p_min = clamp_float(p_min, 0.01, 0.99)
    p_max = clamp_float(p_max, 0.01, 0.99)

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
    lh_ht, la_ht, ht_note, ht_calibrated = lambdas_ht(df_all, lh_ft, la_ft)
    M_HT = score_matrix(lh_ht, la_ht, MAX_HT)
    mk_ht = markets_from_matrix(M_HT, HT_OU_LINES)

    # ML
    ml = train_ml_models(df_all)
    ml_status = ml.get("_status", {})
    ml_ok = bool(ml_status.get("ok"))

    def apply_ml_over(line: float, p_over_poi: float, feat: List[float]) -> Tuple[float, str, bool]:
        key = f"over_ft_{line}"
        if ml_ok and key in ml and ml[key].get("approved") and ml[key].get("model") is not None:
            model = ml[key]["model"]
            p_over_ml = float(model.predict_proba(np.array([feat], float))[0, 1])
            return p_over_ml, "ML (aprovado)", True
        return float(p_over_poi), "Poisson (ML rejeitado)", False

    def apply_ml_btts(p_poi: float, feat: List[float]) -> Tuple[float, str, bool]:
        if ml_ok and "btts" in ml and ml["btts"].get("approved") and ml["btts"].get("model") is not None:
            model = ml["btts"]["model"]
            p_ml = float(model.predict_proba(np.array([feat], float))[0, 1])
            return p_ml, "ML (aprovado)", True
        return float(p_poi), "Poisson (ML rejeitado)", False

    dist = abs((lh_ft + la_ft) - (poisson["avg_h"] + poisson["avg_a"]))
    feat = [0.0, lh_ft, la_ft, poisson["att_h"][home_team], poisson["att_a"][away_team], dist]

    # 1X2 FT (Poisson)
    p_home = mk_ft["p_home"]
    p_draw = mk_ft["p_draw"]
    p_away = mk_ft["p_away"]

    # Dupla chance FT (Poisson)
    p_1x = max(0.0, min(1.0, p_home + p_draw))
    p_12 = max(0.0, min(1.0, p_home + p_away))
    p_2x = max(0.0, min(1.0, p_away + p_draw))

    # BTTS FT (ML opcional) — SIM e NÃO
    p_btts_ft_poi = mk_ft["p_btts"]
    feat[0] = p_btts_ft_poi
    p_btts_ft_final, tag_btts_ft, btts_ml_used = apply_ml_btts(p_btts_ft_poi, feat)
    p_btts_ft_no_poi = max(0.0, 1.0 - p_btts_ft_poi)
    p_btts_ft_no_final = max(0.0, 1.0 - p_btts_ft_final)

    # BTTS HT (Poisson) — SIM e NÃO
    p_btts_ht_sim = mk_ht["p_btts"]
    p_btts_ht_no = max(0.0, 1.0 - p_btts_ht_sim)

    # Over/Under FT (ML opcional no Over; Under = 1-Over)
    ou_ft_rows = []
    for line in FT_OU_LINES:
        p_over_poi = mk_ft[f"p_over_{line}"]
        feat[0] = p_over_poi
        p_over_final, tag, used = apply_ml_over(line, p_over_poi, feat)
        p_under_poi = max(0.0, 1.0 - p_over_poi)
        p_under_final = max(0.0, 1.0 - p_over_final)
        ou_ft_rows.append((line, p_over_poi, p_over_final, p_under_poi, p_under_final, tag, used))

    # Over/Under HT (Poisson)
    ou_ht_rows = []
    for line in HT_OU_LINES:
        p_over = mk_ht[f"p_over_{line}"]
        p_under = mk_ht[f"p_under_{line}"]
        ou_ht_rows.append((line, p_over, p_under))

    # Placar exato FT/HT (tabelas)
    cs_ft_table = [
        (f"{h}-{a}", float(M_FT.get((h, a), 0.0)), odd(float(M_FT.get((h, a), 0.0))))
        for h in range(SHOW_CS_FT_MAX + 1) for a in range(SHOW_CS_FT_MAX + 1)
    ]
    shown_ft = sum(p for _, p, _ in cs_ft_table)
    cs_ft_other = max(0.0, 1.0 - shown_ft)

    cs_ht_table = [
        (f"{h}-{a}", float(M_HT.get((h, a), 0.0)), odd(float(M_HT.get((h, a), 0.0))))
        for h in range(SHOW_CS_HT_MAX + 1) for a in range(SHOW_CS_HT_MAX + 1)
    ]
    shown_ht = sum(p for _, p, _ in cs_ht_table)
    cs_ht_other = max(0.0, 1.0 - shown_ht)

    # Cards TOP/BOTTOM 5 FT limitados a 0..5 x 0..5
    top5_ft, bottom5_ft = top_and_bottom_scores_limited(M_FT, k=5, max_score=CARDS_CS_FT_MAX)

    def score_cards(rows: List[Tuple[str, float, float]], title: str) -> str:
        cards = []
        for placar, prob, o in rows:
            cards.append(
                '<div class="border rounded-xl p-4 bg-gray-50">'
                f'<div class="text-xs text-gray-500">{html_escape(title)}</div>'
                f'<div class="text-lg font-semibold">{html_escape(placar)}</div>'
                f'<div class="text-sm">Prob.: <b>{prob:.4f}</b></div>'
                f'<div class="text-sm">Odd justa: <b>{o}</b></div>'
                '</div>'
            )
        return "\n".join(cards)

    top5_cards_html = score_cards(top5_ft, "Placar FT (mais provável)")
    bottom5_cards_html = score_cards(bottom5_ft, "Placar FT (menos provável)")

    # ============================================================
    # DICAS: 5 e APENAS mercados tradicionais
    # ============================================================

    candidates = []

    def add_tip(name: str, p_poi: float, p_final: float, tag: str, ml_used: bool):
        o_final = odd(p_final)
        if not (p_min <= p_final <= p_max):
            return
        if not (odd_min <= o_final <= odd_max):
            return
        diff = abs(p_poi - p_final)
        score = (0 if ml_used else 1, diff, abs(p_final - 0.5))
        candidates.append({
            "name": name,
            "p_poi": float(p_poi),
            "p_final": float(p_final),
            "odd_min": o_final,
            "tag": tag,
            "ml_used": ml_used,
            "diff": diff,
            "score": score
        })

    add_tip("1X2 (FT) — Casa", p_home, p_home, "Poisson", False)
    add_tip("1X2 (FT) — Empate", p_draw, p_draw, "Poisson", False)
    add_tip("1X2 (FT) — Visitante", p_away, p_away, "Poisson", False)

    add_tip("Dupla Chance (FT) — 1X (Casa/Empate)", p_1x, p_1x, "Poisson", False)
    add_tip("Dupla Chance (FT) — 12 (Casa/Fora)", p_12, p_12, "Poisson", False)
    add_tip("Dupla Chance (FT) — 2X (Fora/Empate)", p_2x, p_2x, "Poisson", False)

    add_tip("BTTS (Ambas Marcam) FT — SIM", p_btts_ft_poi, p_btts_ft_final, tag_btts_ft, btts_ml_used)
    add_tip("BTTS (Ambas Marcam) FT — NÃO", p_btts_ft_no_poi, p_btts_ft_no_final, tag_btts_ft, btts_ml_used)

    add_tip("BTTS (Ambas Marcam) HT — SIM", p_btts_ht_sim, p_btts_ht_sim, "Poisson", False)
    add_tip("BTTS (Ambas Marcam) HT — NÃO", p_btts_ht_no, p_btts_ht_no, "Poisson", False)

    for (line, p_over_poi, p_over_final, p_under_poi, p_under_final, tag, used) in ou_ft_rows:
        add_tip(f"Over (FT) {line}", p_over_poi, p_over_final, tag, used)
        add_tip(f"Under (FT) {line}", p_under_poi, p_under_final, tag, used)

    for (line, p_over, p_under) in ou_ht_rows:
        add_tip(f"Over (HT) {line}", p_over, p_over, "Poisson", False)
        add_tip(f"Under (HT) {line}", p_under, p_under, "Poisson", False)

    best5 = sorted(candidates, key=lambda x: x["score"])[:5]

    # ===== Badge do Backtest Avançado por dica (só FT tradicionais) =====
    def tip_market_ui_key(name: str) -> Optional[str]:
        n = (name or "").strip()

        # 1X2 FT
        if n.startswith("1X2 (FT) — Casa"):
            return "1x2_home"
        if n.startswith("1X2 (FT) — Empate"):
            return "1x2_draw"
        if n.startswith("1X2 (FT) — Visitante"):
            return "1x2_away"

        # Dupla chance FT
        if n.startswith("Dupla Chance (FT) — 1X"):
            return "dc_1x"
        if n.startswith("Dupla Chance (FT) — 12"):
            return "dc_12"
        if n.startswith("Dupla Chance (FT) — 2X"):
            return "dc_2x"

        # BTTS FT
        if n.startswith("BTTS (Ambas Marcam) FT — SIM"):
            return "btts_ft_yes"
        if n.startswith("BTTS (Ambas Marcam) FT — NÃO"):
            return "btts_ft_no"

        # Over/Under FT (ex: "Over (FT) 2.5")
        if n.startswith("Over (FT) "):
            line = n.replace("Over (FT) ", "").strip()
            return f"over_{line}"
        if n.startswith("Under (FT) "):
            line = n.replace("Under (FT) ", "").strip()
            return f"under_{line}"

        # HT e outros não entram no backtest avançado
        return None

    def render_adv_badge(b: dict) -> str:
        status = (b or {}).get("status") or "missing"
        label = (b or {}).get("label") or "Sem histórico"
        detail = (b or {}).get("detail") or ""

        if status == "good":
            return (
                "<span class='inline-flex items-center px-2 py-0.5 rounded-full "
                "border text-xs bg-green-50 text-green-800 border-green-200'>"
                f"{html_escape(label)} — {html_escape(detail)}</span>"
            )
        if status == "bad":
            return (
                "<span class='inline-flex items-center px-2 py-0.5 rounded-full "
                "border text-xs bg-red-50 text-red-800 border-red-200'>"
                f"{html_escape(label)} — {html_escape(detail)}</span>"
            )
        if status == "neutral":
            return (
                "<span class='inline-flex items-center px-2 py-0.5 rounded-full "
                "border text-xs bg-slate-50 text-slate-700 border-slate-200'>"
                f"{html_escape(label)} — {html_escape(detail)}</span>"
            )
        return (
            "<span class='inline-flex items-center px-2 py-0.5 rounded-full "
            "border text-xs bg-slate-50 text-slate-700 border-slate-200'>"
            f"➖ Backtest Avançado: neutro/sem dado</span>"
        )

    tips_cards = []
    for i, c in enumerate(best5, 1):
        k_ui = tip_market_ui_key(c["name"])
        badge_html = ""
        if k_ui:
            b = adv_badge_for(league, k_ui)
            badge_html = render_adv_badge(b)
        else:
            badge_html = (
                "<span class='inline-flex items-center px-2 py-0.5 rounded-full "
                "border text-xs bg-slate-50 text-slate-700 border-slate-200'>"
                "➖ Backtest Avançado: não aplicável</span>"
            )

        tips_cards.append(
            '<div class="border rounded-xl p-4 bg-gray-50">'
            f'<div class="text-xs text-gray-500">Dica #{i}</div>'
            f'<div class="text-lg font-semibold">{html_escape(c["name"])}</div>'
            f'<div class="text-sm text-gray-600">Método: {html_escape(c["tag"])}</div>'
            f'<div class="mt-2">{badge_html}</div>'
            f'<div class="mt-2 text-sm">Prob. final: <b>{c["p_final"]:.4f}</b></div>'
            f'<div class="text-sm">Odd mínima (justa): <b>{c["odd_min"]}</b></div>'
            f'<div class="text-xs text-gray-500 mt-2">Concordância |Poisson-ML|: {c["diff"]:.4f}</div>'
            "</div>"
        )

    best5_html = "\n".join(tips_cards) if tips_cards else (
        "<div class='text-sm text-gray-600'>Não achei 5 mercados dentro das faixas. "
        "Tente ampliar odd/prob nas configurações.</div>"
    )

    # selects
    league_select = "\n".join(
        [f'<option value="{code}" {"selected" if code==league else ""}>{code} — {html_escape(name)}</option>'
         for code, name in LEAGUES.items()]
    )
    season_select = select_options(SEASONS, season)
    home_select = select_options(teams, home_team)
    away_select = select_options(teams, away_team)

    ml_line = "ML: indisponível."
    if ml_ok:
        ml_line = f"Treino ML: {ml_status.get('train')} → Validação: {ml_status.get('val')} (aplica só se melhorar)"

    ht_aviso = "HT fallback (sem HT real suficiente)." if not ht_calibrated else "HT calibrado com HTHG/HTAG (quando disponível)."

    cs_ft_html = "".join([f"<div class='flex justify-between border rounded p-2'><span>{s}</span><b>{o}</b></div>" for (s, p, o) in cs_ft_table])
    cs_ht_html = "".join([f"<div class='flex justify-between border rounded p-2'><span>{s}</span><b>{o}</b></div>" for (s, p, o) in cs_ht_table])

    ou_ft_html = "".join([
        f"<div class='mb-1'>"
        f"<b>Linha {line}</b>: "
        f"Over — Poisson <b>{odd(p_over_poi)}</b> | Final <b>{odd(p_over_final)}</b> "
        f"<span class='text-xs text-gray-500'>({html_escape(tag)})</span>"
        f"<br/>"
        f"Under — Poisson <b>{odd(p_under_poi)}</b> | Final <b>{odd(p_under_final)}</b>"
        f"</div>"
        for (line, p_over_poi, p_over_final, p_under_poi, p_under_final, tag, used) in ou_ft_rows
    ])

    ou_ht_html = "".join([
        f"<div class='mb-1'><b>Linha {line}</b>: "
        f"Over <b>{odd(p_over)}</b> | Under <b>{odd(p_under)}</b></div>"
        for (line, p_over, p_under) in ou_ht_rows
    ])

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
          <p class="text-sm text-gray-600">
            Odds estatísticas (Poisson) + calibração ML quando aprovada — sem promessa de lucro.
          </p>
          <p class="text-xs text-gray-500 mt-2">
            Temporadas: 2324 + 2425 + 2526 (pesos 0.5 / 0.7 / 1.0)
          </p>
        </div>

        <div class="bg-white rounded-2xl shadow p-6 mb-6">
          <h2 class="text-lg font-semibold mb-4">Seleção</h2>

          <form method="get" class="grid grid-cols-1 md:grid-cols-6 gap-4 items-end">
            <div class="md:col-span-2">
              <label class="text-sm font-medium">Campeonato</label>
              <select name="league" class="w-full mt-1 border rounded p-2">{league_select}</select>
            </div>

            <div>
              <label class="text-sm font-medium">Temporadas</label>
              <select name="season" class="w-full mt-1 border rounded p-2">{season_select}</select>
            </div>

            <div class="md:col-span-1">
              <label class="text-sm font-medium">Casa</label>
              <select name="home_team" class="w-full mt-1 border rounded p-2">{home_select}</select>
            </div>

            <div class="md:col-span-1">
              <label class="text-sm font-medium">Fora</label>
              <select name="away_team" class="w-full mt-1 border rounded p-2">{away_select}</select>
            </div>

            <div class="md:col-span-6 grid grid-cols-2 md:grid-cols-4 gap-3 mt-2">
              <div>
                <label class="text-xs text-gray-600">Odd mínima (dicas)</label>
                <input name="odd_min" value="{odd_min}" step="0.01" type="number" class="w-full mt-1 border rounded p-2"/>
              </div>
              <div>
                <label class="text-xs text-gray-600">Odd máxima (dicas)</label>
                <input name="odd_max" value="{odd_max}" step="0.01" type="number" class="w-full mt-1 border rounded p-2"/>
              </div>
              <div>
                <label class="text-xs text-gray-600">Prob mínima (dicas)</label>
                <input name="p_min" value="{p_min}" step="0.01" type="number" class="w-full mt-1 border rounded p-2"/>
              </div>
              <div>
                <label class="text-xs text-gray-600">Prob máxima (dicas)</label>
                <input name="p_max" value="{p_max}" step="0.01" type="number" class="w-full mt-1 border rounded p-2"/>
              </div>
            </div>

            <div class="md:col-span-6 flex gap-2 mt-3">
              <div class="flex gap-2 flex-wrap">
                <button class="px-4 py-2 rounded bg-black text-white">Gerar Odds</button>
                <a href="/modelos-inline" class="px-4 py-2 rounded bg-slate-800 text-white hover:bg-slate-700 transition">📊 Modelos</a>
                <a href="/backtest-inline" class="px-4 py-2 rounded bg-slate-800 text-white hover:bg-slate-700 transition">🧪 Backtest</a>
                <a href="/backtest-avancado" class="px-4 py-2 rounded bg-slate-800 text-white hover:bg-slate-700 transition">📈 Avançado</a>
              </div>
              <a class="px-4 py-2 rounded border" href="/">Reset</a>
            </div>
          </form>

          <div class="mt-4 text-sm text-gray-700">
            <b>{html_escape(LEAGUES.get(league, league))}</b> • <b>{html_escape(home_team)}</b> x <b>{html_escape(away_team)}</b>
            <br/>
            <b>{html_escape(ml_line)}</b>
            <br/>
            <span class="text-xs text-gray-500">HT: {html_escape(ht_aviso)} ({html_escape(ht_note)})</span>
          </div>
        </div>

        <!-- LINHA 1: 5 melhores apostas -->
        <div class="bg-white rounded-2xl shadow p-6 mb-6">
          <h2 class="text-lg font-semibold mb-2">🔥 5 Melhores apostas (com odd mínima)</h2>
          <p class="text-sm text-gray-600 mb-4">
            Apenas mercados tradicionais: 1X2, Dupla Chance, Over/Under, BTTS (Sim/Não).
          </p>
          <div class="grid grid-cols-1 md:grid-cols-5 gap-4">
            {best5_html}
          </div>
        </div>

        <!-- LINHA 2: 5 placares mais prováveis (FT) -->
        <div class="bg-white rounded-2xl shadow p-6 mb-6">
          <h2 class="text-lg font-semibold mb-2">🎯 5 placares FT mais prováveis</h2>
          <p class="text-sm text-gray-600 mb-4">
            Considera apenas placares de <b>0–{CARDS_CS_FT_MAX} gols</b> por time (máx. {CARDS_CS_FT_MAX}x{CARDS_CS_FT_MAX}).
          </p>
          <div class="grid grid-cols-1 md:grid-cols-5 gap-4">
            {top5_cards_html}
          </div>
        </div>

        <!-- LINHA 3: 5 placares menos prováveis (FT) -->
        <div class="bg-white rounded-2xl shadow p-6 mb-6">
          <h2 class="text-lg font-semibold mb-2">🧊 5 placares FT menos prováveis</h2>
          <p class="text-sm text-gray-600 mb-4">
            Considera apenas placares de <b>0–{CARDS_CS_FT_MAX} gols</b> por time (máx. {CARDS_CS_FT_MAX}x{CARDS_CS_FT_MAX}).
          </p>
          <div class="grid grid-cols-1 md:grid-cols-5 gap-4">
            {bottom5_cards_html}
          </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div class="bg-white rounded-2xl shadow p-6">
            <h2 class="text-lg font-semibold mb-2">Jogo</h2>
            <p class="text-sm mb-3"><b>{html_escape(home_team)}</b> (Casa) x <b>{html_escape(away_team)}</b> (Fora)</p>

            <div class="mt-3">
              <h3 class="font-semibold">1X2 (FT) — Poisson</h3>
              <div class="text-sm mt-1">
                Casa <b>{odd(p_home)}</b> | Empate <b>{odd(p_draw)}</b> | Fora <b>{odd(p_away)}</b>
              </div>

              <h3 class="font-semibold mt-3">Dupla Chance (FT) — Poisson</h3>
              <div class="text-sm mt-1">
                1X (Casa/Empate) <b>{odd(p_1x)}</b> |
                12 (Casa/Fora) <b>{odd(p_12)}</b> |
                2X (Fora/Empate) <b>{odd(p_2x)}</b>
              </div>
            </div>

            <div class="mt-4">
              <h3 class="font-semibold">BTTS (Ambas Marcam)</h3>
              <div class="text-sm mt-2">
                <div class="mb-2">
                  <b>FT</b> — SIM: Poisson <b>{odd(p_btts_ft_poi)}</b> | Final <b>{odd(p_btts_ft_final)}</b>
                  <span class='text-xs text-gray-500'>({html_escape(tag_btts_ft)})</span>
                  <br/>
                  <b>FT</b> — NÃO: Poisson <b>{odd(p_btts_ft_no_poi)}</b> | Final <b>{odd(p_btts_ft_no_final)}</b>
                </div>
                <div>
                  <b>HT</b> — SIM: Poisson <b>{odd(p_btts_ht_sim)}</b> | NÃO: Poisson <b>{odd(p_btts_ht_no)}</b>
                </div>
              </div>
            </div>

            <div class="mt-4">
              <h3 class="font-semibold">Over/Under (FT)</h3>
              <div class="text-sm mt-2">
                {ou_ft_html}
              </div>
            </div>

            <div class="mt-4">
              <h3 class="font-semibold">Over/Under (HT)</h3>
              <div class="text-sm mt-2">
                {ou_ht_html}
              </div>
            </div>
          </div>

          <div class="bg-white rounded-2xl shadow p-6">
            <h2 class="text-lg font-semibold mb-2">Placar Exato</h2>

            <h3 class="font-semibold mt-3">FT (0–{SHOW_CS_FT_MAX})</h3>
            <div class="grid grid-cols-2 gap-2 text-sm mt-2">
              {cs_ft_html}
            </div>
            <p class="text-xs text-gray-500 mt-2">
              Outros (tail): prob={cs_ft_other:.4f} → odd≈{odd(cs_ft_other) if cs_ft_other>0 else 9999}
            </p>

            <h3 class="font-semibold mt-6">HT (0–{SHOW_CS_HT_MAX})</h3>
            <div class="grid grid-cols-2 gap-2 text-sm mt-2">
              {cs_ht_html}
            </div>
            <p class="text-xs text-gray-500 mt-2">
              Outros (tail): prob={cs_ht_other:.4f} → odd≈{odd(cs_ht_other) if cs_ht_other>0 else 9999}
            </p>
          </div>
        </div>

        <div class="bg-white rounded-2xl shadow p-6 mt-6">
          <p class="text-xs text-gray-500">
            Observação: estimativas estatísticas. Não há promessa de lucro.
          </p>
        </div>

      </div>
    </body>
    </html>
    """

# ----------------------------
# Página "Modelos" (Calibração) - INLINE (mantida)
# ----------------------------
@app.get("/modelos-inline", response_class=HTMLResponse)
def page_modelos():
    metrics_path = str(BASE_DIR / "docs/calibracao/metrics.txt")
    try:
        metrics_text = Path(metrics_path).read_text(encoding="utf-8").strip()
    except Exception:
        metrics_text = "metrics.txt não encontrado. Rode: python scripts/gerar_calibracao.py"

    html = f"""
    <!doctype html>
    <html lang="pt-br">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width,initial-scale=1" />
      <title>Modelos — Calibração (Poisson vs ML)</title>
      <script src="https://cdn.tailwindcss.com"></script>
    </head>

    <body class="bg-slate-950 text-slate-100">
      <div class="max-w-6xl mx-auto px-4 py-8">
        <div class="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 class="text-3xl font-bold">Modelos</h1>
            <p class="text-slate-300 mt-1">
              Calibração e validação temporal (treino: 2023/24 + 2024/25 • teste: 2025/26).
            </p>
          </div>

          <div class="flex gap-2 flex-wrap">
            <a href="/" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">
              🏠 Odds (Home)
            </a>
            <span class="px-4 py-2 rounded-lg bg-slate-700 text-white cursor-default">
              📊 Modelos
            </span>
            <a href="/backtest-inline" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">
              🧪 Backtest
            </a>
          </div>
        </div>

        <div class="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div class="rounded-2xl bg-slate-900 border border-slate-800 p-5">
            <h2 class="text-xl font-semibold">O que é calibração?</h2>
            <p class="text-slate-300 mt-2 leading-relaxed">
              Um modelo bem calibrado significa: quando ele diz “60%”, esse evento acontece
              perto de 60% das vezes no longo prazo (em dados fora da amostra).
            </p>

            <h3 class="text-lg font-semibold mt-5">Métricas</h3>
            <ul class="text-slate-300 mt-2 list-disc ml-5 space-y-1">
              <li><b>LogLoss</b>: penaliza probabilidades erradas (menor é melhor)</li>
              <li><b>Brier Score</b>: erro quadrático médio das probabilidades (menor é melhor)</li>
            </ul>

            <div class="mt-5">
              <h3 class="text-lg font-semibold">Resultado (teste)</h3>
              <pre class="mt-2 whitespace-pre-wrap text-slate-200 bg-slate-950/50 border border-slate-800 rounded-xl p-3 text-sm">{metrics_text}</pre>
            </div>

            <p class="text-slate-400 text-sm mt-5">
              ⚠️ Projeto educacional: não recomenda apostas e não promete lucro.
            </p>
          </div>

          <div class="rounded-2xl bg-slate-900 border border-slate-800 p-5 lg:col-span-2">
            <h2 class="text-xl font-semibold">Curvas de calibração</h2>
            <p class="text-slate-300 mt-2">
              Comparação visual entre Poisson e ML (Platt scaling) no evento <b>Vitória do Mandante</b>.
            </p>

            <div class="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div class="rounded-xl bg-slate-950/40 border border-slate-800 p-3">
                <div class="font-semibold mb-2">Poisson</div>
                <img src="/static/calibracao/reliability_poisson.png" alt="Calibração Poisson"
                     class="w-full rounded-lg border border-slate-800" />
              </div>

              <div class="rounded-xl bg-slate-950/40 border border-slate-800 p-3">
                <div class="font-semibold mb-2">ML (Platt)</div>
                <img src="/static/calibracao/reliability_ml.png" alt="Calibração ML"
                     class="w-full rounded-lg border border-slate-800" />
              </div>
            </div>

            <div class="mt-6 rounded-xl bg-slate-950/40 border border-slate-800 p-3">
              <div class="font-semibold mb-2">Comparação de métricas</div>
              <img src="/static/calibracao/compare_logloss_brier.png" alt="Comparação LogLoss e Brier"
                   class="w-full rounded-lg border border-slate-800" />
            </div>
          </div>
        </div>

        <div class="mt-8 text-slate-500 text-sm">
          <p>
            Dica: se as curvas ficarem próximas da diagonal, a calibração está melhor.
            Quanto menor LogLoss/Brier, melhor a qualidade das probabilidades.
          </p>
        </div>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# ----------------------------
# Página "Backtest" (educacional) - INLINE (mantida)
# ----------------------------
@app.get("/backtest-inline", response_class=HTMLResponse)
def page_backtest():
    metrics_path = str(BASE_DIR / "docs/backtest/metrics.txt")
    try:
        metrics_text = Path(metrics_path).read_text(encoding="utf-8").strip()
    except Exception:
        metrics_text = "metrics.txt não encontrado. Rode: python scripts/backtest_educacional.py"

    html = f"""
    <!doctype html>
    <html lang="pt-br">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width,initial-scale=1" />
      <title>Backtest — Educação (Poisson vs ML)</title>
      <script src="https://cdn.tailwindcss.com"></script>
    </head>

    <body class="bg-slate-950 text-slate-100">
      <div class="max-w-6xl mx-auto px-4 py-8">
        <div class="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 class="text-3xl font-bold">Backtest (Educacional)</h1>
            <p class="text-slate-300 mt-1">
              Avaliação da qualidade das probabilidades por mercado (LogLoss / Brier / calibração).
            </p>
          </div>

          <div class="flex gap-2 flex-wrap">
            <a href="/" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">
              🏠 Odds (Home)
            </a>
            <a href="/modelos-inline" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">
              📊 Modelos
            </a>
            <span class="px-4 py-2 rounded-lg bg-slate-700 text-white cursor-default">
              🧪 Backtest
            </span>
          </div>
        </div>

        <div class="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div class="rounded-2xl bg-slate-900 border border-slate-800 p-5">
            <h2 class="text-xl font-semibold">O que este backtest mede?</h2>
            <p class="text-slate-300 mt-2 leading-relaxed">
              Este backtest é <b>educacional</b>: ele avalia se as probabilidades previstas
              são <b>boas</b> e <b>bem calibradas</b> em dados fora da amostra.
            </p>

            <h3 class="text-lg font-semibold mt-5">Validação temporal</h3>
            <ul class="text-slate-300 mt-2 list-disc ml-5 space-y-1">
              <li>Treino: temporadas 2023/24 + 2024/25</li>
              <li>Teste: temporada 2025/26</li>
            </ul>

            <h3 class="text-lg font-semibold mt-5">Métricas</h3>
            <ul class="text-slate-300 mt-2 list-disc ml-5 space-y-1">
              <li><b>LogLoss</b> (menor é melhor)</li>
              <li><b>Brier</b> (menor é melhor)</li>
              <li><b>Curva de calibração</b> (quanto mais perto da diagonal, melhor)</li>
            </ul>

            <div class="mt-5">
              <h3 class="text-lg font-semibold">Relatório (teste)</h3>
              <pre class="mt-2 whitespace-pre-wrap text-slate-200 bg-slate-950/50 border border-slate-800 rounded-xl p-3 text-sm">{metrics_text}</pre>
            </div>

            <p class="text-slate-400 text-sm mt-5">
              ⚠️ Sem recomendação de apostas / sem promessa de lucro.
            </p>
          </div>

          <div class="rounded-2xl bg-slate-900 border border-slate-800 p-5 lg:col-span-2">
            <h2 class="text-xl font-semibold">Comparação por mercado</h2>
            <p class="text-slate-300 mt-2">
              Barras com LogLoss e Brier para Poisson vs ML (Platt).
            </p>

            <div class="mt-4 rounded-xl bg-slate-950/40 border border-slate-800 p-3">
              <img src="/static/backtest/compare_markets.png" alt="Comparação de mercados"
                   class="w-full rounded-lg border border-slate-800" />
            </div>

            <h3 class="text-lg font-semibold mt-6">Curvas por mercado</h3>

            <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div class="rounded-xl bg-slate-950/40 border border-slate-800 p-3">
                <div class="font-semibold mb-2">Home Win (FT) — Poisson</div>
                <img src="/static/backtest/reliability_home_win_poisson.png" class="w-full rounded-lg border border-slate-800" />
              </div>
              <div class="rounded-xl bg-slate-950/40 border border-slate-800 p-3">
                <div class="font-semibold mb-2">Home Win (FT) — ML</div>
                <img src="/static/backtest/reliability_home_win_ml.png" class="w-full rounded-lg border border-slate-800" />
              </div>

              <div class="rounded-xl bg-slate-950/40 border border-slate-800 p-3">
                <div class="font-semibold mb-2">BTTS (FT) SIM — Poisson</div>
                <img src="/static/backtest/reliability_btts_ft_yes_poisson.png" class="w-full rounded-lg border border-slate-800" />
              </div>
              <div class="rounded-xl bg-slate-950/40 border border-slate-800 p-3">
                <div class="font-semibold mb-2">BTTS (FT) SIM — ML</div>
                <img src="/static/backtest/reliability_btts_ft_yes_ml.png" class="w-full rounded-lg border border-slate-800" />
              </div>

              <div class="rounded-xl bg-slate-950/40 border border-slate-800 p-3">
                <div class="font-semibold mb-2">Over 2.5 (FT) — Poisson</div>
                <img src="/static/backtest/reliability_over25_ft_poisson.png" class="w-full rounded-lg border border-slate-800" />
              </div>
              <div class="rounded-xl bg-slate-950/40 border border-slate-800 p-3">
                <div class="font-semibold mb-2">Over 2.5 (FT) — ML</div>
                <img src="/static/backtest/reliability_over25_ft_ml.png" class="w-full rounded-lg border border-slate-800" />
              </div>
            </div>
          </div>
        </div>

        <div class="mt-8 text-slate-500 text-sm">
          <p>
            Interpretação: se ML reduz LogLoss/Brier e melhora a calibração, ele é preferível.
            Caso contrário, manter Poisson é a escolha mais segura.
          </p>
        </div>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# --- FORCE TEMPLATES ROUTES (AUTO) ---
# Se existir HTML inline para /modelos e /backtest, isso garante que as páginas usem templates/*.html
# (última rota definida no FastAPI vence)

try:
    from fastapi import Request as _Request
except Exception:
    _Request = None

def _template_exists(name: str) -> bool:
    try:
        return (BASE_DIR / "templates" / name).exists()
    except Exception:
        return False

@app.get("/backtest-avancado", response_class=HTMLResponse)
def backtest_avancado_page(request: Request):
    return templates.TemplateResponse("backtest_avancado.html", {"request": request})

# --- END FORCE TEMPLATES ROUTES (AUTO) ---
