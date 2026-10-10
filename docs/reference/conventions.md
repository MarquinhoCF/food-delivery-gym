# Convenções

## IDs Gymnasium

```
FoodDelivery-{cenário}-obj{N}-v1
```

- `{cenário}`: stem do arquivo JSON (exemplo: `medium` a partir de `medium.json`)
- `{N}`: inteiro de 1 a 13

## Nomes de agentes em resultados

| Origem | Pasta / chave típica |
|--------|----------------------|
| `random` | `random` |
| `first` / `first_driver` | `first_driver` |
| `nearest` / `nearest_driver` | `nearest_driver` |
| `lowest` + `route` | `lowest_route_cost` |
| `lowest` + `marginal_route` | `lowest_marginal_route_cost` |
| `lowest` + `weighted_score` | `lowest_weighted_score` |
| rollout | `rollout_<base_variant>_h..._a..._tc...` |
| mcts | `mcts_<base_variant>_h..._a..._tc..._i..._ew..._d..._o...[_maxexpK]` |
| modelo PPO | nome da pasta sob `treinamento/obj_N/` (exemplo: `18M_steps`) |

Detalhes de aliases, specs e expansão de variantes: [Catálogo de otimizadores](../framework/optimizer-catalog.md).

## Caminhos

| Caminho | Uso |
|---------|-----|
| `data/runs/<name>/` | Experimentos nomeados (atual) |
| `data/runs/execucoes/` | Layout legado / defaults de alguns scripts |
| `data/ppo_training/` | Modelos treinados |
| `experiments/example.yaml` | Exemplo versionado de experimento |
| `requirements-freeze.txt` | Pinagem exata (hífen no nome do arquivo) |

## Invocação de scripts

Sempre a partir da raiz do repositório:

```bash
python -m scripts.<modulo>
```
