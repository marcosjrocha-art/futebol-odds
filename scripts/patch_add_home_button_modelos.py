from pathlib import Path

APP = Path("app.py")

OLD = '              <button class="px-4 py-2 rounded bg-black text-white">Gerar Odds</button>'
NEW = """              <div class="flex gap-2 flex-wrap">
                <button class="px-4 py-2 rounded bg-black text-white">Gerar Odds</button>
                <a href="/modelos" class="px-4 py-2 rounded bg-slate-800 text-white hover:bg-slate-700 transition">📊 Modelos</a>
              </div>"""

def main():
    txt = APP.read_text(encoding="utf-8")
    if '/modelos' in txt and '📊 Modelos' in txt:
        print("Já existe botão de Modelos na Home. Nada a fazer.")
        return

    if OLD not in txt:
        raise SystemExit("Não encontrei a linha exata do botão 'Gerar Odds'. Me mande 10 linhas antes/depois da linha 706 (use: sed -n '696,716p' app.py).")

    txt = txt.replace(OLD, NEW)
    APP.write_text(txt, encoding="utf-8")
    print("✅ Botão '📊 Modelos' adicionado na Home.")

if __name__ == "__main__":
    main()
