import os
import math
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

import requests
import requests_cache

# -------------------------
# CONFIG
# -------------------------
CACHE_PATH = "data/http_cache.sqlite"
OUT_DIR = "docs/calibracao"
os.makedirs(OUT_DIR, exist_ok=True)

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

# -------------------------
# HTTP session com cache
# -------------------------
session = requests_cache.CachedSession(
    cache_name=CACHE_PATH,
    backend="sqlite",
    expire_after=24 * 3600,  # 1 dia
)
session.headers.update({"User-Agent": "futebol-odds-calibration/1.0"})

def fetch_csv(season: str, league: str) -> pd.DataFrame | None:
    url = BASE_URL.format(season=season, league=league)
    try:
        r = session.get(url, timeout=40)
        if r.status_code != 200:
            print(f"HTTP {r.status_code} {season}/{league}")
            return None

        # alguns arquivos podem vir com encoding diferente
        text = r.text
        if not text or len(text) < 50:
            print(f"EMPTY {season}/{league}")
            return None

        # leitura robusta
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception:
            # fallback: tenta ; (às vezes acontece em outros provedores, mas ok tentar)
            df = pd.read_csv(io.StringIO(text), sep=";")

        if df is None or df.empty:
            print(f"EMPTY_DF {season}/{league}")
            return None

        df["Season"] = season
        df["League"] = league
        return df
    except Exception as e:
        print(f"ERR {season}/{league}: {type(e).__name__}: {e}")
        return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "HomeTeam" not in df.columns and "Home" in df.columns:
        rename_map["Home"] = "HomeTeam"
    if "AwayTeam" not in df.columns and "Away" in df.columns:
        rename_map["Away"] = "AwayTeam"
    df = df.rename(columns=rename_map)

    needed = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    for col in needed:
        if col not in df.columns:
            return pd.DataFrame()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # remover jogos sem placar final
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"]).copy()

    df["home_win"] = (df["FTHG"] > df["FTAG"]).astype(int)
    df["y"] = df["home_win"].astype(int)

    if "Date" in df.columns and df["Date"].notna().any():
        df = df.sort_values(["Season", "League", "Date"])
    else:
        df = df.sort_values(["Season", "League"]).reset_index(drop=True)

    return df.reset_index(drop=True)

def poisson_homewin_prob(df_train: pd.DataFrame, row: pd.Series, max_goals: int = 10) -> float:
    home = row["HomeTeam"]
    away = row["AwayTeam"]
    league = row["League"]

    dfl = df_train[df_train["League"] == league]
    if len(dfl) < 50:
        dfl = df_train

    avg_h = float(dfl["FTHG"].mean())
    avg_a = float(dfl["FTAG"].mean())
    if not np.isfinite(avg_h) or avg_h <= 0: avg_h = 1.35
    if not np.isfinite(avg_a) or avg_a <= 0: avg_a = 1.10

    # médias por time (com fallback)
    ht_mask = dfl["HomeTeam"] == home
    at_mask = dfl["AwayTeam"] == away

    home_att = (float(dfl.loc[ht_mask, "FTHG"].mean()) / avg_h) if ht_mask.any() else 1.0
    home_def = (float(dfl.loc[ht_mask, "FTAG"].mean()) / avg_a) if ht_mask.any() else 1.0
    away_att = (float(dfl.loc[at_mask, "FTAG"].mean()) / avg_a) if at_mask.any() else 1.0
    away_def = (float(dfl.loc[at_mask, "FTHG"].mean()) / avg_h) if at_mask.any() else 1.0

    def shrink(x: float, w: float = 0.35) -> float:
        if not np.isfinite(x) or x <= 0:
            return 1.0
        return (1 - w) * x + w * 1.0

    home_att = shrink(home_att)
    home_def = shrink(home_def)
    away_att = shrink(away_att)
    away_def = shrink(away_def)

    HOME_ADV = 1.08
    lam_h = avg_h * home_att * away_def * HOME_ADV
    lam_a = avg_a * away_att * home_def

    probs_h = np.array([math.exp(-lam_h) * (lam_h ** k) / math.factorial(k) for k in range(max_goals + 1)])
    probs_a = np.array([math.exp(-lam_a) * (lam_a ** k) / math.factorial(k) for k in range(max_goals + 1)])

    mat = np.outer(probs_h, probs_a)
    s = mat.sum()
    if s > 0:
        mat = mat / s

    hw = np.triu(mat, 1).sum()  # i>j
    return float(hw)

def build_dataset(df_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(df_all)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    train = df[df["Season"].isin(TRAIN_SEASONS)].copy()
    test = df[df["Season"] == TEST_SEASON].copy()

    if train.empty or test.empty:
        return pd.DataFrame(), pd.DataFrame()

    print(f"Treino: {len(train)} jogos | Teste: {len(test)} jogos")

    # feature Poisson P(HomeWin)
    train["p_poisson"] = train.apply(lambda r: poisson_homewin_prob(train, r), axis=1)
    test["p_poisson"] = test.apply(lambda r: poisson_homewin_prob(train, r), axis=1)

    train["p_poisson"] = train["p_poisson"].clip(1e-6, 1 - 1e-6)
    test["p_poisson"] = test["p_poisson"].clip(1e-6, 1 - 1e-6)

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

def plot_compare_metrics(metrics, out_path):
    labels = list(metrics.keys())
    ll = [metrics[k]["logloss"] for k in labels]
    bs = [metrics[k]["brier"] for k in labels]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - w/2, ll, width=w, label="LogLoss (menor melhor)")
    plt.bar(x + w/2, bs, width=w, label="Brier (menor melhor)")
    plt.xticks(x, labels)
    plt.title("Comparação de métricas no TESTE (temporada mais recente)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

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
        print("Treino ou teste vazio. Verifique temporadas disponíveis/placares.")
        return

    y_te = test["y"].values
    p_te = test["p_poisson"].values

    # ML: Platt scaling (calibração)
    X_tr = train["p_poisson"].values.reshape(-1, 1)
    y_tr = train["y"].values
    X_te = p_te.reshape(-1, 1)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_tr, y_tr)
    p_te_ml = clf.predict_proba(X_te)[:, 1]
    p_te_ml = np.clip(p_te_ml, 1e-6, 1 - 1e-6)

    metrics = {
        "Poisson": {
            "logloss": float(log_loss(y_te, p_te)),
            "brier": float(brier_score_loss(y_te, p_te)),
        },
        "ML (Platt)": {
            "logloss": float(log_loss(y_te, p_te_ml)),
            "brier": float(brier_score_loss(y_te, p_te_ml)),
        }
    }

    print("\nMétricas (teste):")
    for k, v in metrics.items():
        print(f"- {k}: LogLoss={v['logloss']:.4f} | Brier={v['brier']:.4f}")

    plot_reliability(y_te, p_te, "Calibração — Poisson (Home Win)", os.path.join(OUT_DIR, "reliability_poisson.png"))
    plot_reliability(y_te, p_te_ml, "Calibração — ML (Platt) (Home Win)", os.path.join(OUT_DIR, "reliability_ml.png"))
    plot_compare_metrics(metrics, os.path.join(OUT_DIR, "compare_logloss_brier.png"))

    report_path = os.path.join(OUT_DIR, "metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Métricas no teste (temporada mais recente)\n")
        for k, v in metrics.items():
            f.write(f"{k}: LogLoss={v['logloss']:.6f} | Brier={v['brier']:.6f}\n")

    print(f"\n✅ Gráficos salvos em: {OUT_DIR}")
    print(f"✅ Relatório salvo em: {report_path}")

if __name__ == "__main__":
    main()
