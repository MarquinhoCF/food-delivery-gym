# Script `visualize_mcts_decisions`

Executa um episódio com `MonteCarloTreeSearchOptimizerGym`, grava o `decision_log` (árvore explorada e valores na raiz) e salva o JSON completo mais **um** PNG da decisão pedida.

```bash
python -m scripts.visualize_mcts_decisions --decision 0 [opções]
```

## Opções

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--decision` | Índice da decisão (`decision_idx`, 0-based); **obrigatório** | — |
| `--from-json` | `decisions.json` já coletado; pula o episódio | (nenhum) |
| `--scenario` | Arquivo JSON do cenário | `medium.json` |
| `--seed` | Semente | `5434` |
| `--objective` | Objetivo de 1 a 13 | `1` |
| `--base-optimizer` | Política de base nas folhas | padrão do catálogo |
| `--cost-function` | Obrigatório se a base for `lowest` | (nenhum) |
| `--horizon` | Horizonte do rollout nas folhas | padrão do catálogo |
| `--alpha` | Fator de desconto | padrão do catálogo |
| `--terminal-cost` | `0` ou `model` | padrão do catálogo |
| `--iterations` | Iterações MCTS | padrão do catálogo |
| `--exploration-weight` | Peso UCT | padrão do catálogo |
| `--depth` | Profundidade máxima da árvore | igual ao `horizon` |
| `--max-outcomes` | Máx. futuros W por ação | padrão do catálogo |
| `--max-expanded-actions` | Limiar d_thr (máx. ações expandidas por nó) | todos os motoristas |
| `--out-dir` | Diretório de saída | `data/visualization/mcts_viz` |
| `--max-steps` | Limite de passos | `10000` |

## Exemplo

```bash
python -m scripts.visualize_mcts_decisions \
  --decision 0 \
  --scenario medium.json \
  --objective 3 \
  --base-optimizer nearest \
  --horizon 5 \
  --iterations 8 \
  --depth 2 \
  --terminal-cost 0
```

O JSON completo fica em `decisions.json` (para inspecionar índices). O PNG sai em `mcts_decisions/decision_001.png` quando `--decision 0`.

Para só re-renderizar a partir de um JSON já gravado:

```bash
python -m scripts.visualize_mcts_decisions \
  --decision 2 \
  --from-json data/visualization/mcts_viz/.../decisions.json
```

Com `--from-json`, o PNG vai para a pasta do JSON (ou `--out-dir` se passado).

Consulte também [visualize_rollout_decisions](visualize-rollout-decisions.md) e [Otimizadores](../framework/optimizers.md).
