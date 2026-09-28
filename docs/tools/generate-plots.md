# Script `generate_plots`

Gera gráficos por episódio e/ou agregados a partir de `metrics_data.npz` ou `.json`, sem reexecutar simulações.

```bash
python -m scripts.generate_plots [opções]
```

## Opções

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `-r` / `--results-dir` | Raiz com pastas `obj_N/` | `./data/runs/execucoes` |
| `-o` / `--objectives` | Objetivos | 1 a 13 |
| `-s` / `--scenarios` | Cenários | padrão dos cenários principais |
| `--only-episode` | Somente gráficos por episódio | (ausente) |
| `--only-batch` | Somente gráficos agregados | (ausente) |

As flags `--only-episode` e `--only-batch` são mutuamente exclusivas.

## Exemplos

```bash
# Layout legado do artigo
python -m scripts.generate_plots -r ./data/runs/execucoes -o 3

# Experimento nomeado
python -m scripts.generate_plots -r ./data/runs/experimento_exemplo -o 3 --only-batch
```

## Saída

As figuras ficam em `figs/` dentro de cada pasta de agente (nomes como `mean_results_*_other_metrics.png`; pastas `run_*_results_*/` para episódios).

Para fluxos com `run.json`, prefira [report](report.md) com `--episodes`.
