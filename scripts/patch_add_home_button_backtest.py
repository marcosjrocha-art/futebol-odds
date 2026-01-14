from pathlib import Path

APP = Path("app.py")

def main():
    txt = APP.read_text(encoding="utf-8")

    if 'href="/backtest"' in txt:
        print("Já existe botão de Backtest na Home. Nada a fazer.")
        return

    marker = 'href="/modelos"'
    if marker not in txt:
        raise SystemExit("Não encontrei o botão de Modelos na Home. Rode: grep -n \"href=\\\"/modelos\\\"\" app.py | head")

    # Insere o botão Backtest logo após o botão Modelos (na Home)
    txt = txt.replace(
        'href="/modelos" class="px-4 py-2 rounded bg-slate-800 text-white hover:bg-slate-700 transition">📊 Modelos</a>',
        'href="/modelos" class="px-4 py-2 rounded bg-slate-800 text-white hover:bg-slate-700 transition">📊 Modelos</a>\n'
        '                <a href="/backtest" class="px-4 py-2 rounded bg-slate-800 text-white hover:bg-slate-700 transition">🧪 Backtest</a>'
    )

    APP.write_text(txt, encoding="utf-8")
    print("✅ Botão '🧪 Backtest' adicionado na Home.")

if __name__ == "__main__":
    main()
