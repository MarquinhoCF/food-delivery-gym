# Script `visualize_rollout_decisions`

Executa um episódio com `RolloutOptimizerGym`, grava o `decision_log` (Q por motorista, trajetória da política de base e custo terminal) e salva PNGs e JSON.

```bash
python -m scripts.visualize_rollout_decisions [opções]
```

## Opções

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--scenario` | Arquivo JSON do cenário | `medium.json` |
| `--seed` | Semente | `5434` |
| `--objective` | Objetivo de 1 a 13 | `1` |
| `--base-optimizer` | Política de base do rollout | padrão do catálogo |
| `--cost-function` | Obrigatório se a base for `lowest` | (nenhum) |
| `--horizon` | Horizonte após a ação candidata | padrão do catálogo |
| `--alpha` | Fator de desconto | padrão do catálogo |
| `--terminal-cost` | `0` ou `model` | padrão do catálogo |
| `--out-dir` | Diretório de saída | `data/visualization/rollout_viz` |
| `--max-steps` | Limite de passos | `10000` |

## Exemplo

```bash
python -m scripts.visualize_rollout_decisions \
  --scenario medium.json \
  --objective 3 \
  --base-optimizer lowest \
  --cost-function route \
  --horizon 5 \
  --terminal-cost 0
```

Consulte também [collect_terminal_cost](collect-terminal-cost.md) e [test_runner](test-runner.md).
