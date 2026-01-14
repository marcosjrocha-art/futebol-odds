from pathlib import Path

APP = Path("app.py")

def main():
    txt = APP.read_text(encoding="utf-8")

    # ----- MODELOS: troca bloco de botões -----
    old_modelos = """          <div class="flex gap-2">
            <a href="/" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">
              ← Voltar
            </a>
          </div>"""

    new_modelos = """          <div class="flex gap-2 flex-wrap">
            <a href="/" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">
              🏠 Odds (Home)
            </a>
            <span class="px-4 py-2 rounded-lg bg-slate-700 text-white cursor-default">
              📊 Modelos
            </span>
            <a href="/backtest" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">
              🧪 Backtest
            </a>
          </div>"""

    if old_modelos in txt:
        txt = txt.replace(old_modelos, new_modelos)
        print("✅ Menu atualizado em /modelos")
    else:
        print("⚠️ Não encontrei o bloco antigo de botões em /modelos (talvez já tenha sido alterado).")

    # ----- BACKTEST: troca bloco de botões -----
    old_backtest = """          <div class="flex gap-2">
            <a href="/" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">← Voltar</a>
            <a href="/modelos" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">📊 Modelos</a>
          </div>"""

    new_backtest = """          <div class="flex gap-2 flex-wrap">
            <a href="/" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">
              🏠 Odds (Home)
            </a>
            <a href="/modelos" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition">
              📊 Modelos
            </a>
            <span class="px-4 py-2 rounded-lg bg-slate-700 text-white cursor-default">
              🧪 Backtest
            </span>
          </div>"""

    if old_backtest in txt:
        txt = txt.replace(old_backtest, new_backtest)
        print("✅ Menu atualizado em /backtest")
    else:
        print("⚠️ Não encontrei o bloco antigo de botões em /backtest (talvez já tenha sido alterado).")

    APP.write_text(txt, encoding="utf-8")
    print("✅ Patch de navegação aplicado no app.py")

if __name__ == "__main__":
    main()
