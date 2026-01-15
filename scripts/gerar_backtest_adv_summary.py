from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
CSV = BASE_DIR / "static" / "backtest_adv" / "metrics_by_league.csv"
OUT = BASE_DIR / "static" / "backtest_adv" / "summary.json"

def score_row(r):
    # score: melhora combinada (logloss + brier)
    d_ll = float(r["logloss_poisson"] - r["logloss_ml"])
    d_br = float(r["brier_poisson"] - r["brier_ml"])
    return 0.6 * d_ll + 0.4 * d_br

def main():
    if not CSV.exists():
        print(f"❌ Não achei o CSV em: {CSV}")
        return

    df = pd.read_csv(CSV)
    df = df.dropna()
    df["score"] = df.apply(score_row, axis=1)

    # melhores = score alto (ML melhora)
    best = df.sort_values("score", ascending=False).head(30)
    # piores = score negativo (ML piora)
    worst = df.sort_values("score", ascending=True).head(30)

    def pack(dfx):
        out = []
        for _, r in dfx.iterrows():
            out.append({
                "league": r["league"],
                "league_name": r["league_name"],
                "market": r["market"],
                "n": int(r["n"]),
                "poisson": {"logloss": float(r["logloss_poisson"]), "brier": float(r["brier_poisson"])},
                "ml": {"logloss": float(r["logloss_ml"]), "brier": float(r["brier_ml"])},
                "delta": {"logloss": float(r["logloss_poisson"] - r["logloss_ml"]),
                          "brier": float(r["brier_poisson"] - r["brier_ml"])},
                "score": float(r["score"]),
            })
        return out

    payload = {
        "generated_from": str(CSV),
        "best": pack(best),
        "worst": pack(worst),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ summary.json gerado em: {OUT}")

if __name__ == "__main__":
    main()
