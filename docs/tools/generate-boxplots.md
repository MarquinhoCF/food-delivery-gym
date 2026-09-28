# Script `generate_boxplots`

Gera boxplots comparativos entre agentes nas métricas `rewards`, `delivery_time` e `distance`.

```bash
python -m scripts.generate_boxplots [opções]
```

## Dados

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `-r` / `--results-dir` | Raiz com pastas `obj_N/` | `./data/runs/execucoes` |
| `-o` / `--objective` | Um objetivo | `3` |
| `-s` / `--scenarios` | Cenários | simple, medium, complex |
| `-a` / `--agents` | Ordem ou filtro de agentes | descoberta automática |
| `--exclude-agents` | Agentes a excluir | (nenhum) |

## Métricas

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `-M` / `--metrics` | `rewards`, `delivery_time`, `distance` | todas |

## Layout

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--by-scenario` / `--no-by-scenario` | Agentes no eixo X | ligado |
| `--split-scenarios` / `--no-split-scenarios` | Um painel por cenário | ligado |
| `--split` | Um arquivo por métrica, sem painéis | (ausente) |
| `--no-fliers` | Oculta outliers | (ausente) |
| `--show-means` / `--no-show-means` | Marca a média com losango | ligado |
| `--show-mean-values` / `--no-show-mean-values` | Valor numérico da média | ligado |
| `--annotate-n` | Anota o tamanho amostral N | (ausente) |
| `--legend-stats` | Mediana e média na legenda | (ausente) |
| `--suptitle` | Título geral da figura | (nenhum) |

## Saída

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `-od` / `--output-dir` | Diretório de saída | `./data/runs/figuras` |
| `--prefix` | Prefixo do nome do arquivo | `boxplot` |
| `--fmt` | `png`, `pdf`, `svg` | `png` |
| `--figsize` | Largura e altura | `16 5` |
| `--dpi` | DPI | `300` |
| `--font-size` | Tamanho da fonte base | `9` |

## Exemplos

```bash
python -m scripts.generate_boxplots

python -m scripts.generate_boxplots -r data/runs/experimento_exemplo -o 3 \
  -od data/runs/experimento_exemplo/report/boxplots --prefix boxplot_obj3

python -m scripts.generate_boxplots --metrics rewards distance --no-fliers --annotate-n
python -m scripts.generate_boxplots --no-split-scenarios --no-by-scenario --fmt pdf
```

Para experimentos com `run.json`, prefira [report](report.md).
