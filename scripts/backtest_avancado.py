from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import requests_cache
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw" / "mmz4281"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DOCS_OUT = BASE_DIR / "docs" / "backtest_adv"
PLOTS_OUT = DOCS_OUT / "plots"
DOCS_OUT.mkdir(parents=True, exist_ok=True)
PLOTS_OUT.mkdir(parents=True, exist_ok=True)

MMZ_BASE = "https://www.football-data.co.uk/mmz4281"

LEAGUES: Dict[str, str] = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "EC": "Conference",
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
}

TRAIN_SEASONS = ["2324", "2425"]
TEST_SEASON = "2526"

THROTTLE_S = 0.2
requests_cache.install_cache(
    cache_name=str(BASE_DIR / "data" / "http_cache"),
    backend="sqlite",
    expire_after=60 * 60 * 24,
)

# ----------------------------
# Utils
# ----------------------------
def download_csv(season: str, league: str) -> Path:
    out = RAW_DIR / season / f"{league}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1000:
        return out

    url = f"{MMZ_BASE}/{season}/{league}.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out.write_bytes(r.content)
    time.sleep(THROTTLE_S)
    return out


def load_csv(season: str, league: str) -> pd.DataFrame:
    p = download_csv(season, league)
    df = pd.read_csv(p)
    df["Season"] = season
    df["League"] = league
    # campos mínimos
    need = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{season}/{league} sem coluna {c}")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]).copy()
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"]).copy()
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def score_matrix(lam_h: float, lam_a: float, max_goals: int = 10) -> np.ndarray:
    ph = np.array([poisson_pmf(i, lam_h) for i in range(max_goals + 1)])
    pa = np.array([poisson_pmf(j, lam_a) for j in range(max_goals + 1)])
    m = np.outer(ph, pa)
    s = m.sum()
    if s > 0:
        m = m / s
    return m


@dataclass
class LeagueModel:
    teams: List[str]
    avg_hg: float
    avg_ag: float
    att_home: Dict[str, float]
    def_home: Dict[str, float]
    att_away: Dict[str, float]
    def_away: Dict[str, float]
    home_adv: float


def fit_league_model(df: pd.DataFrame) -> LeagueModel:
    # médias liga
    avg_hg = float(df["FTHG"].mean())
    avg_ag = float(df["FTAG"].mean())
    # home advantage simples
    home_adv = (avg_hg / max(avg_ag, 1e-9))

    teams = sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))
    # forças por time (ponderadas iguais)
    home_for = df.groupby("HomeTeam")["FTHG"].mean().to_dict()
    home_against = df.groupby("HomeTeam")["FTAG"].mean().to_dict()
    away_for = df.groupby("AwayTeam")["FTAG"].mean().to_dict()
    away_against = df.groupby("AwayTeam")["FTHG"].mean().to_dict()

    # normalizar por média da liga
    att_home = {t: (home_for.get(t, avg_hg) / max(avg_hg, 1e-9)) for t in teams}
    def_home = {t: (home_against.get(t, avg_ag) / max(avg_ag, 1e-9)) for t in teams}

    att_away = {t: (away_for.get(t, avg_ag) / max(avg_ag, 1e-9)) for t in teams}
    def_away = {t: (away_against.get(t, avg_hg) / max(avg_hg, 1e-9)) for t in teams}

    return LeagueModel(
        teams=teams,
        avg_hg=avg_hg,
        avg_ag=avg_ag,
        att_home=att_home,
        def_home=def_home,
        att_away=att_away,
        def_away=def_away,
        home_adv=home_adv,
    )


def poisson_probs_for_match(model: LeagueModel, home: str, away: str) -> Dict[str, float]:
    # lambdas
    lam_h = model.avg_hg * model.att_home.get(home, 1.0) * model.def_away.get(away, 1.0) * model.home_adv
    lam_a = model.avg_ag * model.att_away.get(away, 1.0) * model.def_home.get(home, 1.0)

    mat = score_matrix(lam_h, lam_a, max_goals=10)

    # 1X2
    p_home = float(np.tril(mat, -1).sum())  # i>j
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())   # i<j

    # DC
    p_1x = p_home + p_draw
    p_12 = p_home + p_away
    p_2x = p_away + p_draw

    # BTTS
    p_h0 = float(mat[0, :].sum())
    p_a0 = float(mat[:, 0].sum())
    p_00 = float(mat[0, 0])
    p_btts_no = p_h0 + p_a0 - p_00
    p_btts_yes = 1.0 - p_btts_no

    # Totais
    goals = np.add.outer(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
    def over_line(line: float) -> float:
        return float(mat[goals > line].sum())
    def under_line(line: float) -> float:
        return float(mat[goals < line].sum())

    probs = {
        "1x2_home": p_home,
        "1x2_draw": p_draw,
        "1x2_away": p_away,
        "dc_1x": p_1x,
        "dc_12": p_12,
        "dc_2x": p_2x,
        "btts_yes": p_btts_yes,
        "btts_no": p_btts_no,
        "over_0.5": over_line(0.5),
        "over_1.5": over_line(1.5),
        "over_2.5": over_line(2.5),
        "over_3.5": over_line(3.5),
        "under_0.5": under_line(0.5),
        "under_1.5": under_line(1.5),
        "under_2.5": under_line(2.5),
        "under_3.5": under_line(3.5),
    }
    # clamp
    for k in list(probs.keys()):
        probs[k] = float(min(max(probs[k], 1e-9), 1 - 1e-9))
    return probs


def outcome_labels(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    ftr = df["FTR"].astype(str).values
    y = {}
    y["1x2_home"] = (ftr == "H").astype(int)
    y["1x2_draw"] = (ftr == "D").astype(int)
    y["1x2_away"] = (ftr == "A").astype(int)
    # DC
    y["dc_1x"] = ((ftr == "H") | (ftr == "D")).astype(int)
    y["dc_12"] = ((ftr == "H") | (ftr == "A")).astype(int)
    y["dc_2x"] = ((ftr == "A") | (ftr == "D")).astype(int)
    # BTTS
    y["btts_yes"] = ((df["FTHG"].values > 0) & (df["FTAG"].values > 0)).astype(int)
    y["btts_no"] = 1 - y["btts_yes"]
    # Totais
    tot = (df["FTHG"].values + df["FTAG"].values).astype(int)
    for line in [0.5, 1.5, 2.5, 3.5]:
        y[f"over_{line}"] = (tot > line).astype(int)
        y[f"under_{line}"] = (tot < line).astype(int)
    return y


def fit_platt(p: np.ndarray, y: np.ndarray) -> LogisticRegression:
    # Platt scaling: logistic regression on logit(p)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(x, y)
    return lr


def apply_platt(lr: LogisticRegression, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    return lr.predict_proba(x)[:, 1]


def plot_bins(league: str, market: str, p: np.ndarray, y: np.ndarray, out: Path) -> None:
    # bins por probabilidade prevista
    bins = np.linspace(0, 1, 11)
    ids = np.digitize(p, bins) - 1
    rows = []
    for b in range(10):
        mask = ids == b
        if mask.sum() < 20:
            continue
        avg_p = float(p[mask].mean())
        avg_y = float(y[mask].mean())
        gap = abs(avg_p - avg_y)
        rows.append((b, avg_p, avg_y, gap, int(mask.sum())))
    if not rows:
        return
    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    ns = [r[4] for r in rows]

    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.scatter(xs, ys)
    for x, yy, n in zip(xs, ys, ns):
        plt.text(x, yy, str(n), fontsize=8)
    plt.title(f"{league} - {market} (calibração por bins)")
    plt.xlabel("Prob prevista")
    plt.ylabel("Prob real")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    print("🔄 Gerando backtest avançado (treino 2324+2425 / teste 2526) ...")

    records = []

    for lg, lg_name in LEAGUES.items():
        # carrega treino+teste
        dfs_train = []
        for s in TRAIN_SEASONS:
            dfs_train.append(load_csv(s, lg))
        df_train = pd.concat(dfs_train, ignore_index=True)

        df_test = load_csv(TEST_SEASON, lg)

        model = fit_league_model(df_train)

        # predições poisson em teste
        p_rows = []
        for _, r in df_test.iterrows():
            p_rows.append(poisson_probs_for_match(model, str(r["HomeTeam"]), str(r["AwayTeam"])))
        df_p = pd.DataFrame(p_rows)

        ys = outcome_labels(df_test)

        markets = list(ys.keys())

        # treino de calibradores por mercado usando TREINO (poisson no treino)
        # precisamos gerar probs do treino também
        p_rows_tr = []
        for _, r in df_train.iterrows():
            p_rows_tr.append(poisson_probs_for_match(model, str(r["HomeTeam"]), str(r["AwayTeam"])))
        df_ptr = pd.DataFrame(p_rows_tr)
        ys_tr = outcome_labels(df_train)

        for m in markets:
            p_te = df_p[m].values.astype(float)
            y_te = ys[m].astype(int)
            p_tr = df_ptr[m].values.astype(float)
            y_tr = ys_tr[m].astype(int)

            # Platt
            lr = fit_platt(p_tr, y_tr)
            p_ml = apply_platt(lr, p_te)

            # métricas
            ll_p = float(log_loss(y_te, p_te, labels=[0, 1]))
            br_p = float(brier_score_loss(y_te, p_te))
            ll_m = float(log_loss(y_te, p_ml, labels=[0, 1]))
            br_m = float(brier_score_loss(y_te, p_ml))

            # plot bins (gap) usando ML vs real (mais útil)
            out_plot = PLOTS_OUT / f"{lg}_{m}_gap_bins.png"
            plot_bins(lg, m, p_ml, y_te, out_plot)

            records.append(
                {
                    "league": lg,
                    "league_name": lg_name,
                    "market": m,
                    "n": int(len(df_test)),
                    "logloss_poisson": ll_p,
                    "brier_poisson": br_p,
                    "logloss_ml": ll_m,
                    "brier_ml": br_m,
                }
            )

        print(f"✅ {lg} ({lg_name}) ok: {len(df_test)} jogos / {len(markets)} mercados")

    df_out = pd.DataFrame(records).sort_values(["league", "market"]).reset_index(drop=True)
    out_csv = DOCS_OUT / "metrics_by_league.csv"
    out_md = DOCS_OUT / "metrics_by_league.md"
    df_out.to_csv(out_csv, index=False)
    df_out.to_markdown(out_md, index=False)

    print(f"\n✅ CSV salvo em: {out_csv}")
    print(f"✅ Markdown salvo em: {out_md}")
    print(f"✅ Plots (bins) em: {PLOTS_OUT}")


if __name__ == "__main__":
    main()
