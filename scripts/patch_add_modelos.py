from pathlib import Path
import re

APP = Path("app.py")

def main():
    txt = APP.read_text(encoding="utf-8")

    # 1) garantir import os
    if re.search(r"^\s*import\s+os\s*$", txt, flags=re.M) is None:
        # coloca import os depois do primeiro bloco de imports (bem no topo)
        lines = txt.splitlines()
        insert_at = 0
        for i, line in enumerate(lines[:80]):
            if line.startswith("import ") or line.startswith("from "):
                insert_at = i + 1
        lines.insert(insert_at, "import os")
        txt = "\n".join(lines) + ("\n" if not txt.endswith("\n") else "")

    # 2) achar o objeto templates (Jinja2Templates)
    # se não achar, aborta com mensagem
    if "Jinja2Templates" not in txt or "templates =" not in txt:
        raise SystemExit("Não encontrei 'templates = Jinja2Templates(...)' no app.py. Me envie um print do topo do app.py (imports e inicialização).")

    # 3) não duplicar rota
    if re.search(r"@app\.get\(\"/modelos\"", txt):
        print("Rota /modelos já existe. Nada a fazer.")
        return

    # 4) inserir rota depois do bloco de rotas existentes (após a primeira ocorrência de app.get("/"))
    route_code = r'''

@app.get("/modelos", response_class=HTMLResponse)
def modelos(request: Request):
    """
    Página de modelos: exibe gráficos de calibração (Poisson vs ML) e métricas.
    """
    metrics_path = os.path.join("docs", "calibracao", "metrics.txt")
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_text = f.read().strip()
    except Exception:
        metrics_text = "metrics.txt não encontrado. Rode: python scripts/gerar_calibracao.py"

    return templates.TemplateResponse(
        "modelos.html",
        {"request": request, "metrics_text": metrics_text},
    )
'''.lstrip("\n")

    # Estratégia: inserir antes do "if __name__ == '__main__'" se existir; senão, no fim.
    m = re.search(r"^if\s+__name__\s*==\s*[\"']__main__[\"']\s*:", txt, flags=re.M)
    if m:
        idx = m.start()
        txt = txt[:idx] + route_code + "\n" + txt[idx:]
    else:
        txt = txt + ("\n" if not txt.endswith("\n") else "") + route_code + "\n"

    APP.write_text(txt, encoding="utf-8")
    print("✅ Patch aplicado: rota /modelos adicionada no app.py")

if __name__ == "__main__":
    main()
