# Script `generate_table`

Consolida métricas em uma planilha Excel (média, desvio padrão, mediana e moda) para recompensa, tempo efetivo de entrega e distância.

```bash
python -m scripts.generate_table [opções]
```

Requer `openpyxl` (incluído em `requirements.txt`).

## Opções

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `-r` / `--results-dir` | Raiz com pastas `obj_N/` | `./data/runs/execucoes` |
| `-out` / `--output` | Arquivo Excel de saída | `./data/runs/tabelas/objective_table.xlsx` |
| `-o` / `--objectives` | Objetivos | todos |
| `-s` / `--scenarios` | Cenários | `simple medium complex` |

## Exemplos

```bash
python -m scripts.generate_table

python -m scripts.generate_table \
  -r ./data/runs/experimento_exemplo \
  -out ./data/runs/experimento_exemplo/report/objective_table.xlsx \
  -o 3
```

O script descobre agentes automaticamente e destaca o melhor resultado por cenário e objetivo.

Para experimentos com `run.json`, prefira [report](report.md).
