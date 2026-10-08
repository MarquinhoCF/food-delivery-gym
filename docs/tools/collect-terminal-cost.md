# Script `collect_terminal_cost`

Coleta o retorno restante da política de base do rollout e grava `samples.npz` por variante. O ajuste da regressão linear fica em [`fit_terminal_cost`](fit-terminal-cost.md).

```bash
python -m scripts.collect_terminal_cost [opções]
```

## Opções

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `-s` / `--scenarios` | Cenários | padrão (`simple`, `medium`, `complex`) |
| `-o` / `--objectives` | Objetivos | `[3]` |
| `-b` / `--bases` | Políticas de base do rollout | todas |
| `-c` / `--cost-functions` | Funções de custo quando a base é `lowest` | todas |
| `-e` / `--episodes` | Episódios por variante | `1000` |
| `--alpha` | Fator de desconto do retorno | padrão do catálogo |
| `--seed` | Semente raiz | `123456789` |
| `--output-dir` | Diretório raiz de saída | `data/terminal_cost` |

## Exemplo

```bash
python -m scripts.collect_terminal_cost \
  --scenarios simple \
  --objectives 3 \
  --bases lowest \
  --cost-functions weighted_score \
  --episodes 2000

python -m scripts.fit_terminal_cost \
  --scenarios simple \
  --objectives 3 \
  --bases lowest \
  --cost-functions weighted_score
```

## Saída

```
data/terminal_cost/<scenario>/obj_<N>/<base_key>/
  samples.npz
```

Depois de ajustar com `fit_terminal_cost` (gera `linear_model.npz`), use `terminal: model` no YAML de rollout ou inclua `terminal=model` na flag `--rollout` do `test_runner` / `run_batch_eval`.

Consulte também [fit_terminal_cost](fit-terminal-cost.md), [visualize_rollout_decisions](visualize-rollout-decisions.md) e [Otimizadores](../framework/optimizers.md).
