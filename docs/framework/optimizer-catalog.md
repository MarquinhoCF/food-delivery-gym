# Catálogo de otimizadores

Referência do catálogo em [`food_delivery_gym/main/optimizer/catalog.py`](../../food_delivery_gym/main/optimizer/catalog.py). Os scripts (`test_runner`, `run_batch_eval`, relatórios) leem chaves, aliases, labels e regras de instanciação daqui. Para cadastrar um otimizador novo, implemente a classe e acrescente um `OptimizerSpec` em `_SPECS`; em geral não é preciso alterar cada script.

Guia de uso e extensão: [Otimizadores](optimizers.md).

## Papel do catálogo

| Conceito | Função |
|----------|--------|
| `OptimizerSpec` | Entrada canônica: chave, aliases CLI, labels, classe, builder, requisitos |
| `CostFunctionSpec` | Funções de custo do `lowest` e nomes de pasta de resultado |
| `LowestVariantSpec` / `RolloutVariantSpec` | Variantes explícitas usadas na CLI e no YAML |
| `EvalVariant` | Par `(spec, result_key, extras)` expandido para uma execução |
| `discover_rl_models` | Descoberta de modelos sob `data/ppo_training/` |

A **ordem** das entradas em `_SPECS` é a ordem canônica usada em plots, tabelas e boxplots (heurísticas conhecidas primeiro).

## Entradas cadastradas

| Chave | Aliases CLI | Classe | Heurística | Pode ser base do rollout | Requisitos |
|-------|-------------|--------|------------|--------------------------|------------|
| `random` | `random` | `RandomDriverOptimizerGym` | sim | sim | (nenhum) |
| `first_driver` | `first`, `first_driver` | `FirstDriverOptimizerGym` | sim | sim | (nenhum) |
| `nearest_driver` | `nearest`, `nearest_driver` | `NearestDriverOptimizerGym` | sim | sim | (nenhum) |
| `lowest` | `lowest` | `LowestCostDriverOptimizerGym` | sim | sim | `cost_function` |
| `rollout` | `rollout` | `RolloutOptimizerGym` | sim | **não** | variantes `--rollout` |
| `rl` | `rl` | `RLModelOptimizerGym` | não | **não** | `model` (ou descoberta / `--model-path`) |

Comportamento resumido:

| Chave | Decisão |
|-------|----------|
| `random` | Motorista aleatório entre os elegíveis |
| `first_driver` | Primeiro motorista elegível |
| `nearest_driver` | Motorista mais próximo do próximo segmento da rota |
| `lowest` | Minimiza a função de custo informada |
| `rollout` | Avalia ações candidatas com lookahead a partir de uma política base |
| `rl` | Política SB3 (`predict`); carregada por caminho ou descoberta |

## Funções de custo (`lowest`)

| Chave CLI | Pasta de resultado (`result_key`) | Usa objetivo do ambiente |
|-----------|-----------------------------------|--------------------------|
| `route` | `lowest_route_cost` | sim |
| `marginal_route` | `lowest_marginal_route_cost` | sim |
| `weighted_score` | `lowest_weighted_score` | não |

Na CLI e no YAML, a especificação usa o formato `cost=...` (também aceito `cost_function=...` no parser).

Exemplos:

```bash
# test_runner: uma variante
python -m scripts.test_runner --mode auto --optimizer lowest --lowest cost=route

# run_batch_eval: uma ou mais variantes
python -m scripts.run_batch_eval --name costs --agents lowest --no-rl \
  --lowest cost=route --lowest cost=weighted_score
```

No YAML:

```yaml
agents:
  - lowest
lowest:
  - cost: route
  - cost: weighted_score
```

## Rollout

### Spec CLI / YAML

Formato: `base=...,cost=...,horizon=...,alpha=...,terminal=...`

| Chave | Obrigatório | Descrição |
|-------|-------------|-----------|
| `base` | não (default `nearest`) | Política base; deve ter `rollout_base=True` no catálogo |
| `cost` | só se a base for `lowest` | Função de custo da base |
| `horizon` | não (default `5`) | Passos de lookahead; use `inf` / `none` / `null` para horizonte infinito |
| `alpha` | não (default `0.9`) | Fator de desconto |
| `terminal` | não (default `0`) | `0` ou `model` |

Bases válidas hoje: `random`, `first_driver` / `first`, `nearest_driver` / `nearest`, `lowest`.  
`rollout` e `rl` **não** podem ser base.

Exemplos:

```bash
python -m scripts.test_runner --mode auto --optimizer rollout \
  --rollout base=lowest,cost=route,horizon=5,alpha=0.9,terminal=0

python -m scripts.run_batch_eval --name rollout_smoke --agents rollout --no-rl \
  --rollout base=lowest,cost=route,horizon=5,terminal=0 \
  --rollout base=nearest,horizon=10,terminal=model
```

### Nome da pasta de resultado

Formato novo:

```
rollout_<base_variant>_h<H>_a<alpha>_tc<terminal>
```

Exemplos:

| Spec | `result_key` |
|------|----------------|
| `base=nearest,horizon=5,alpha=0.9,terminal=0` | `rollout_nearest_driver_h5_a0p9_tc0` |
| `base=lowest,cost=route,horizon=10,alpha=0.9,terminal=model` | `rollout_lowest_route_cost_h10_a0p9_tcmodel` |
| `base=lowest,cost=weighted_score,horizon=inf,terminal=0` | `rollout_lowest_weighted_score_hinf_a0p9_tc0` |

Pastas antigas só com `rollout_<base_variant>` ainda são reconhecidas (defaults de horizonte/alpha/terminal).

Calibração do modo `terminal=model`: [collect_terminal_cost](../tools/collect-terminal-cost.md).  
Visualização de decisões: [visualize_rollout_decisions](../tools/visualize-rollout-decisions.md).

## Modelos RL

A entrada `rl` no catálogo é o builder genérico. Na prática, os scripts descobrem modelos em disco:

```
<data/ppo_training>/<cenário>/treinamento/obj_<N>/<nome>/best_model.zip
```

| Campo | Origem |
|-------|--------|
| `name` | Nome da pasta (`18M_steps`) |
| `key` | Prefixo de algoritmo + nome (`ppo_18M_steps`) |
| `path` | Caminho do `best_model.zip` |

O `test_runner` e o `run_batch_eval` aceitam a chave (`ppo_18M_steps`), o nome da pasta (`18M_steps`) ou `--optimizer rl` com `--model-path`. O `vecnormalize.pkl` é buscado recursivamente sob a pasta do modelo.

Organização após o treino no Zoo: [Artefatos](../rl-baselines3-zoo/artifacts.md).

## Expansão de variantes (`expand_evaluations`)

Quando a lista de agentes inclui `lowest` ou `rollout`, o batch **exige** variantes explícitas (`--lowest` / `--rollout` ou blocos YAML). Cada variante vira um `EvalVariant` com:

- `result_key`: nome da pasta em `data/runs/<name>/obj_N/<cenário>/`
- `extras`: kwargs passados ao builder (`cost_function`, `base_optimizer`, `horizon`, …)

Heurísticas simples (`random`, `first_driver`, `nearest_driver`) geram uma única pasta com o mesmo nome da chave.

## API útil (programática)

```python
from food_delivery_gym.main.optimizer import catalog as optimizer_catalog

optimizer_catalog.keys(is_heuristic=True)
optimizer_catalog.cli_choices()
optimizer_catalog.resolve_key("nearest")          # -> "nearest_driver"
optimizer_catalog.parse_lowest_cli("cost=route")
optimizer_catalog.parse_rollout_cli(
    "base=lowest,cost=route,horizon=5,terminal=0"
)
optimizer_catalog.build("lowest", env, objective=3, cost_function="route")
optimizer_catalog.discover_rl_models(
    "./data/ppo_training", "medium", 3
)
```

## Cadastrar um otimizador novo

1. Implemente uma subclasse de `OptimizerGym` com `select_driver` (e `get_title` se quiser).
2. Acrescente um `OptimizerSpec` em `_SPECS` (chave, aliases, labels, `builder`, `requires`, `rollout_base`, `is_heuristic`).
3. Se for variante parametrizada (como `lowest`), defina o parsing CLI e a expansão em `expand_evaluations` / `result_key`.

Detalhes de implementação e exemplo completo: [Otimizadores](optimizers.md).

## Documentação relacionada

- [Otimizadores](optimizers.md)
- [Convenções](../reference/conventions.md)
- [test_runner](../tools/test-runner.md)
- [run_batch_eval](../tools/run-batch-eval.md)
- [YAML de experimento](../experiments/yaml.md)
