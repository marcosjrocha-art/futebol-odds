import os
import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

import requests_cache

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "docs/backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = str(BASE_DIR / "data/http_cache.sqlite")

TRAIN_SEASONS = ["2324", "2425"]
TEST_SEASON = "2526"

LEAGUES = [
    "E0","E1","E2","E3","EC",
    "SC0","D1","D2",
    "I1","I2",
    "SP1","SP2",
    "F1","F2",
    "N1","B1","P1","T1","G1"
]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# mercados do backtest (tradicionais)
MARKETS = {
    "Home Win (FT)": "home_win",
    "BTTS (FT) - SIM": "btts_ft_yes",
    "Over 2.5 (FT)": "over25_ft",
}

# sessão HTTP com cache (usa o mesmo sqlite)
session = requests_cache.CachedSession(
    cache_name=CACHE_PATH,
    backend="sqlite",
    expire_after=24 * 3600,
)
session.headers.update({"User-Agent": "futebol-odds-backtest/1.0"})

# =========================
# HELPERS
# =========================
def fetch_csv(season: str, league: str) -> pd.DataFrame | None:
    url = BASE_URL.format(season=season, league=league)
    try:
        r = session.get(url, timeout=40)
        if r.status_code != 200:
            return None
        text = r.text
        if not text or len(text) < 50:
            return None
        df = pd.read_csv(io.StringIO(text))
        df["Season"] = season
        df["League"] = league
        return df
    except Exception:
        return None

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # precisa: HomeTeam, AwayTeam, FTHG, FTAG
    if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        return pd.DataFrame()
    if "FTHG" not in df.columns or "FTAG" not in df.columns:
        return pd.DataFrame()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    df = df.dropna(subset=["FTHG","FTAG"]).copy()
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG","FTAG"]).copy()

    # targets
    df["home_win"] = (df["FTHG"] > df["FTAG"]).astype(int)
    df["btts_ft_yes"] = ((df["FTHG"] >= 1) & (df["FTAG"] >= 1)).astype(int)
    df["over25_ft"] = ((df["FTHG"] + df["FTAG"]) >= 3).astype(int)

    if "Date" in df.columns and df["Date"].notna().any():
        df = df.sort_values(["Season","League","Date"])
    else:
        df = df.sort_values(["Season","League"]).reset_index(drop=True)

    return df.reset_index(drop=True)

def shrink(x: float, w: float = 0.35) -> float:
    if not np.isfinite(x) or x <= 0:
        return 1.0
    return (1 - w) * x + w * 1.0

def poisson_lambdas(train: pd.DataFrame, league: str, home: str, away: str) -> tuple[float,float]:
    # restringe por liga, fallback global se liga pequena
    dfl = train[train["League"] == league]
    if len(dfl) < 50:
        dfl = train

    avg_h = float(dfl["FTHG"].mean())
    avg_a = float(dfl["FTAG"].mean())
    if not np.isfinite(avg_h) or avg_h <= 0: avg_h = 1.35
    if not np.isfinite(avg_a) or avg_a <= 0: avg_a = 1.10

    ht = dfl["HomeTeam"] == home
    at = dfl["AwayTeam"] == away

    home_att = (float(dfl.loc[ht, "FTHG"].mean()) / avg_h) if ht.any() else 1.0
    home_def = (float(dfl.loc[ht, "FTAG"].mean()) / avg_a) if ht.any() else 1.0
    away_att = (float(dfl.loc[at, "FTAG"].mean()) / avg_a) if at.any() else 1.0
    away_def = (float(dfl.loc[at, "FTHG"].mean()) / avg_h) if at.any() else 1.0

    home_att = shrink(home_att)
    home_def = shrink(home_def)
    away_att = shrink(away_att)
    away_def = shrink(away_def)

    HOME_ADV = 1.08
    lam_h = avg_h * home_att * away_def * HOME_ADV
    lam_a = avg_a * away_att * home_def
    return lam_h, lam_a

def poisson_probs_from_lambdas(lam_h: float, lam_a: float, max_goals: int = 10) -> np.ndarray:
    # matriz de placares (0..max_goals)
    probs_h = np.array([math.exp(-lam_h) * (lam_h ** k) / math.factorial(k) for k in range(max_goals + 1)])
    probs_a = np.array([math.exp(-lam_a) * (lam_a ** k) / math.factorial(k) for k in range(max_goals + 1)])
    mat = np.outer(probs_h, probs_a)
    s = mat.sum()
    if s > 0:
        mat = mat / s
    return mat

def market_prob_from_matrix(mat: np.ndarray, market_key: str) -> float:
    # mat[i,j] = P(home=i, away=j)
    if market_key == "home_win":
        return float(np.triu(mat, 1).sum())  # i>j
    if market_key == "btts_ft_yes":
        # 1 - P(home=0) - P(away=0) + P(0,0)
        p_h0 = float(mat[0, :].sum())
        p_a0 = float(mat[:, 0].sum())
        p_00 = float(mat[0, 0])
        return float(1.0 - p_h0 - p_a0 + p_00)
    if market_key == "over25_ft":
        # total >=3
        p = 0.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if (i + j) >= 3:
                    p += mat[i, j]
        return float(p)
    raise ValueError("Mercado não suportado no backtest")

def build_dataset(df_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize(df_all)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    train = df[df["Season"].isin(TRAIN_SEASONS)].copy()
    test = df[df["Season"] == TEST_SEASON].copy()
    if train.empty or test.empty:
        return pd.DataFrame(), pd.DataFrame()

    # features Poisson por mercado
    for market_name, market_key in MARKETS.items():
        p_list_tr = []
        for _, r in train.iterrows():
            lam_h, lam_a = poisson_lambdas(train, r["League"], r["HomeTeam"], r["AwayTeam"])
            mat = poisson_probs_from_lambdas(lam_h, lam_a)
            p = market_prob_from_matrix(mat, market_key)
            p_list_tr.append(p)
        train[f"p_poisson_{market_key}"] = np.clip(np.array(p_list_tr), 1e-6, 1 - 1e-6)

        p_list_te = []
        for _, r in test.iterrows():
            lam_h, lam_a = poisson_lambdas(train, r["League"], r["HomeTeam"], r["AwayTeam"])
            mat = poisson_probs_from_lambdas(lam_h, lam_a)
            p = market_prob_from_matrix(mat, market_key)
            p_list_te.append(p)
        test[f"p_poisson_{market_key}"] = np.clip(np.array(p_list_te), 1e-6, 1 - 1e-6)

    return train, test

def plot_reliability(y_true, p_pred, title, out_path):
    frac_pos, mean_pred = calibration_curve(y_true, p_pred, n_bins=10, strategy="uniform")
    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Modelo")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfeito")
    plt.xlabel("Probabilidade prevista")
    plt.ylabel("Frequência observada")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_market_compare(results, out_path):
    markets = list(results.keys())
    ll_p = [results[m]["poisson"]["logloss"] for m in markets]
    bs_p = [results[m]["poisson"]["brier"] for m in markets]
    ll_m = [results[m]["ml"]["logloss"] for m in markets]
    bs_m = [results[m]["ml"]["brier"] for m in markets]

    x = np.arange(len(markets))
    w = 0.18

    plt.figure(figsize=(11, 5))
    plt.bar(x - 1.5*w, ll_p, width=w, label="Poisson LogLoss")
    plt.bar(x - 0.5*w, bs_p, width=w, label="Poisson Brier")
    plt.bar(x + 0.5*w, ll_m, width=w, label="ML LogLoss")
    plt.bar(x + 1.5*w, bs_m, width=w, label="ML Brier")
    plt.xticks(x, markets, rotation=10, ha="right")
    plt.title("Backtest (teste: 2025/26) — comparação por mercado (menor é melhor)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# =========================
# MAIN
# =========================
def main():
    print("Baixando CSVs (com cache)...")
    dfs = []
    for season in TRAIN_SEASONS + [TEST_SEASON]:
        for league in LEAGUES:
            df = fetch_csv(season, league)
            if df is not None and not df.empty:
                dfs.append(df)
                print(f"OK {season}/{league} ({len(df)} linhas)")
            else:
                print(f"SKIP {season}/{league}")

    if not dfs:
        print("Nenhum dado carregado. Abortando.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    train, test = build_dataset(df_all)

    if train.empty or test.empty:
        print("Treino ou teste vazio. Abortando.")
        return

    print(f"\nTreino: {len(train)} jogos | Teste: {len(test)} jogos")

    results = {}
    report_lines = []
    report_lines.append("Backtest educacional — validação temporal")
    report_lines.append(f"Treino: {', '.join(TRAIN_SEASONS)} | Teste: {TEST_SEASON}")
    report_lines.append("Métricas: LogLoss e Brier (menor é melhor)")
    report_lines.append("")

    for market_name, market_key in MARKETS.items():
        y_tr = train[market_key].values
        y_te = test[market_key].values

        p_tr = train[f"p_poisson_{market_key}"].values
        p_te = test[f"p_poisson_{market_key}"].values

        # ML: calibração (Platt) por mercado
        X_tr = p_tr.reshape(-1, 1)
        X_te = p_te.reshape(-1, 1)

        clf = LogisticRegression(max_iter=200)
        clf.fit(X_tr, y_tr)
        p_te_ml = clf.predict_proba(X_te)[:, 1]
        p_te_ml = np.clip(p_te_ml, 1e-6, 1 - 1e-6)

        poisson_metrics = {
            "logloss": float(log_loss(y_te, p_te)),
            "brier": float(brier_score_loss(y_te, p_te)),
        }
        ml_metrics = {
            "logloss": float(log_loss(y_te, p_te_ml)),
            "brier": float(brier_score_loss(y_te, p_te_ml)),
        }

        results[market_name] = {"poisson": poisson_metrics, "ml": ml_metrics}

        report_lines.append(f"== {market_name} ==")
        report_lines.append(f"Poisson: LogLoss={poisson_metrics['logloss']:.4f} | Brier={poisson_metrics['brier']:.4f}")
        report_lines.append(f"ML(Platt): LogLoss={ml_metrics['logloss']:.4f} | Brier={ml_metrics['brier']:.4f}")
        report_lines.append("")

        # gráficos de calibração
        plot_reliability(
            y_te, p_te,
            f"Calibração — Poisson ({market_name})",
            str(OUT_DIR / f"reliability_{market_key}_poisson.png")
        )
        plot_reliability(
            y_te, p_te_ml,
            f"Calibração — ML (Platt) ({market_name})",
            str(OUT_DIR / f"reliability_{market_key}_ml.png")
        )

    # gráfico comparativo geral
    plot_market_compare(results, str(OUT_DIR / "compare_markets.png"))

    # salvar relatório
    (OUT_DIR / "metrics.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\n✅ Backtest finalizado. Arquivos em: {OUT_DIR}")
    print("✅ Gerados:")
    print("- compare_markets.png")
    for _, k in MARKETS.items():
        print(f"- reliability_{k}_poisson.png")
        print(f"- reliability_{k}_ml.png")
    print("- metrics.txt")

if __name__ == "__main__":
    main()
