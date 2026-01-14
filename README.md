# Futebol Odds Platform ⚽ (Poisson + ML)

Plataforma educacional para gerar **odds justas** (sem margem) a partir de dados históricos do Football-Data.

> ⚠️ Sem promessa de lucro. É análise estatística e calibração.

## Recursos
- Seleção de **liga / temporada / time casa / time fora**
- 1X2 FT + **Dupla chance (1X / 12 / 2X)**
- **BTTS (Ambas Marcam) FT** (SIM/NÃO) com ML quando aprovado
- **BTTS (Ambas Marcam) HT** (SIM/NÃO) Poisson
- **Over/Under FT e HT** (Over e Under)
- Placar exato FT/HT + cards: **Top 5 mais prováveis** e **Top 5 menos prováveis** (limitado a 0–5 gols por time)

## Dados
Fonte: Football-Data (mmz4281).  
Temporadas usadas: 2324, 2425, 2526 (ponderação por recência).

## Rodar local (Linux)
```bash
cd ~/futebol-odds
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
