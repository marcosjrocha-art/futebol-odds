from pathlib import Path
import re

APP = Path("app.py")

ROUTE_BLOCK = r'''
# ----------------------------
# Página "Backtest" (educacional)
# ----------------------------
@app.get("/backtest", response_class=HTMLResponse)
def page_backtest():
    """
    Exibe backtest educacional por mercado (Poisson vs ML) com validação temporal.
    """
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

          <div class="flex gap-2">
            <a href="/" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">← Voltar</a>
            <a href="/modelos" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">📊 Modelos</a>
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
'''.lstrip("\n")

def main():
    txt = APP.read_text(encoding="utf-8")

    if re.search(r"@app\.get\(\"/backtest\"", txt):
        print("Rota /backtest já existe. Nada a fazer.")
        return

    m_main = re.search(r"^if\s+__name__\s*==\s*[\"']__main__[\"']\s*:", txt, flags=re.M)
    if m_main:
        idx = m_main.start()
        txt = txt[:idx] + ROUTE_BLOCK + "\n" + txt[idx:]
    else:
        txt = txt + ("\n" if not txt.endswith("\n") else "") + ROUTE_BLOCK + "\n"

    APP.write_text(txt, encoding="utf-8")
    print("✅ Patch aplicado: rota /backtest adicionada no app.py")

if __name__ == "__main__":
    main()
