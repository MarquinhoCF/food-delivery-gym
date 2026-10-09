# Script `visualize_rollout_decisions`

Executa um episódio com `RolloutOptimizerGym`, grava o `decision_log` (Q por motorista, trajetória da política de base e custo terminal) e salva o JSON completo mais **um** PNG da decisão pedida.

```bash
python -m scripts.visualize_rollout_decisions --decision 0 [opções]
```

## Opções

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--decision` | Índice da decisão (`decision_idx`, 0-based); **obrigatório** | — |
| `--from-json` | `decisions.json` já coletado; pula o episódio | (nenhum) |
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
  --decision 0 \
  --scenario medium.json \
  --objective 3 \
  --base-optimizer lowest \
  --cost-function route \
  --horizon 5 \
  --terminal-cost 0
```

O JSON completo fica em `decisions.json` (para inspecionar índices). O PNG sai em `rollout_decisions/decision_001.png` quando `--decision 0`.

Para só re-renderizar a partir de um JSON já gravado:

```bash
python -m scripts.visualize_rollout_decisions \
  --decision 2 \
  --from-json data/visualization/rollout_viz/.../decisions.json
```

Com `--from-json`, o PNG vai para a pasta do JSON (ou `--out-dir` se passado).

Consulte também [visualize_mcts_decisions](visualize-mcts-decisions.md), [collect_terminal_cost](collect-terminal-cost.md), [fit_terminal_cost](fit-terminal-cost.md) e [test_runner](test-runner.md).
