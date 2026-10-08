# Script `fit_terminal_cost`

Ajusta a regressão linear do custo terminal a partir de `samples.npz` gerados por [`collect_terminal_cost`](collect-terminal-cost.md). O artefato `linear_model.npz` é usado por `RolloutOptimizerGym` quando `terminal=model`.

```bash
python -m scripts.fit_terminal_cost [opções]
```

## Opções

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--input-dir` | Diretório raiz com `samples.npz` | `data/terminal_cost` |
| `-s` / `--scenarios` | Filtra cenários | todos encontrados |
| `-o` / `--objectives` | Filtra objetivos | todos encontrados |
| `-b` / `--bases` | Filtra políticas de base | todas encontradas |
| `-c` / `--cost-functions` | Funções de custo quando a base é `lowest` | todas |

## Exemplo

```bash
python -m scripts.fit_terminal_cost \
  --scenarios simple \
  --objectives 3 \
  --bases lowest \
  --cost-functions weighted_score
```

## Saída

```
data/terminal_cost/<scenario>/obj_<N>/<base_key>/
  linear_model.npz
```

Consulte também [collect_terminal_cost](collect-terminal-cost.md).
