from pathlib import Path
import re

APP = Path("app.py")

ROUTE_BLOCK = r'''
# ----------------------------
# Página "Modelos" (Calibração)
# ----------------------------
@app.get("/modelos", response_class=HTMLResponse)
def page_modelos():
    """
    Exibe gráficos de calibração (Poisson vs ML) e métricas do arquivo metrics.txt.
    """
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

          <div class="flex gap-2">
            <a href="/" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">
              ← Voltar
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
'''.lstrip("\n")

def ensure_import(txt: str, import_line: str) -> str:
    if re.search(rf"^\s*{re.escape(import_line)}\s*$", txt, flags=re.M):
        return txt
    # inserir após o bloco inicial de imports
    lines = txt.splitlines()
    insert_at = 0
    for i, line in enumerate(lines[:120]):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, import_line)
    return "\n".join(lines) + ("\n" if not txt.endswith("\n") else "")

def main():
    txt = APP.read_text(encoding="utf-8")

    # Imports necessários
    txt = ensure_import(txt, "import os")
    txt = ensure_import(txt, "from fastapi.staticfiles import StaticFiles")

    # Garantir mount do static (após app = FastAPI(...))
    if "app.mount(\"/static\"" not in txt and "app.mount('/static'" not in txt:
        m = re.search(r"^app\s*=\s*FastAPI\([^\n]*\)\s*$", txt, flags=re.M)
        if not m:
            raise SystemExit("Não encontrei a linha 'app = FastAPI(...)' no app.py. Me envie o trecho onde o app é criado.")
        insert_pos = m.end()
        mount_block = "\n\n# Static files (imagens, css, etc.)\napp.mount(\"/static\", StaticFiles(directory=str(BASE_DIR / \"static\")), name=\"static\")\n"
        txt = txt[:insert_pos] + mount_block + txt[insert_pos:]

    # Adicionar rota /modelos antes do if __main__ (se existir) ou no final
    if re.search(r"@app\.get\(\"/modelos\"", txt):
        print("Rota /modelos já existe. Nada a fazer.")
        return

    m_main = re.search(r"^if\s+__name__\s*==\s*[\"']__main__[\"']\s*:", txt, flags=re.M)
    if m_main:
        idx = m_main.start()
        txt = txt[:idx] + ROUTE_BLOCK + "\n" + txt[idx:]
    else:
        txt = txt + ("\n" if not txt.endswith("\n") else "") + ROUTE_BLOCK + "\n"

    APP.write_text(txt, encoding="utf-8")
    print("✅ Patch aplicado: StaticFiles + rota /modelos adicionados no app.py")

if __name__ == "__main__":
    main()
