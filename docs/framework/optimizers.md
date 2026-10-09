# Otimizadores

Agentes de decisão herdam de `OptimizerGym` e implementam `select_driver`. As chaves CLI, aliases, funções de custo, variantes de `lowest`/`rollout`/`mcts` e a descoberta de modelos RL estão documentadas no [catálogo de otimizadores](optimizer-catalog.md) (`food_delivery_gym/main/optimizer/catalog.py`).

Este guia cobre o uso prático e como criar um otimizador customizado.

## Uso rápido

```bash
# Heurística simples
python -m scripts.test_runner --mode auto --optimizer nearest

# Lowest (mesmo formato do run_batch_eval)
python -m scripts.test_runner --mode auto --optimizer lowest --lowest cost=route

# Rollout
python -m scripts.test_runner --mode auto --optimizer rollout \
  --rollout base=lowest,cost=route,horizon=5,terminal=0

# MCTS
python -m scripts.test_runner --mode auto --optimizer mcts \
  --mcts base=nearest,horizon=5,iterations=8,depth=2

# Batch com variantes explícitas
python -m scripts.run_batch_eval --name costs --agents lowest --no-rl \
  --lowest cost=route --lowest cost=weighted_score
```

Referência completa de chaves, pastas de resultado e specs: [Catálogo de otimizadores](optimizer-catalog.md).

## Modelos de aprendizado por reforço

Modelos são descobertos sob `./data/ppo_training/` (ou `--model-base-dir`). Layout e organização após o Zoo: [Artefatos](../rl-baselines3-zoo/artifacts.md).

```bash
python -m scripts.test_runner --mode agent --optimizer rl \
  --model-path data/ppo_training/medium/treinamento/obj_3/18M_steps/best_model.zip

python -m scripts.test_runner --mode agent --optimizer ppo_18M_steps --objective 3
```

## Implementando um otimizador customizado

```python
from typing import List
from food_delivery_gym.main.optimizer.optimizer_gym.optmizer_gym import OptimizerGym
from food_delivery_gym.main.route.route import Route
from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.map.map import Map

class NearestDriverOptimizerGym(OptimizerGym):
    def get_title(self):
        return "Otimizador do Motorista Mais Próximo"

    def compare_distance(self, map: Map, driver: Driver, route: Route):
        return map.distance(
            driver.get_last_valid_coordinate(),
            route.route_segments[0].coordinate,
        )

    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        nearest = min(
            drivers,
            key=lambda d: self.compare_distance(self.gym_env.simpy_env.map, d, route),
        )
        return drivers.index(nearest)
```

Execução programática:

```python
optimizer = NearestDriverOptimizerGym(env)
optimizer.run_simulations(
    num_runs=10,
    dir_path="./resultados/",
    seed=42,
    save_individual_plots=True,
    save_mean_plots=True,
    metrics_fmt="npz",
)
```

O nome do módulo `optmizer_gym` (grafia atual no código) deve ser respeitado nos imports. Para registrar a classe no catálogo e usá-la nos scripts, veja a seção correspondente em [Catálogo de otimizadores](optimizer-catalog.md).

## Documentação relacionada

- [Catálogo de otimizadores](optimizer-catalog.md)
- [test_runner](../tools/test-runner.md)
- [run_batch_eval](../tools/run-batch-eval.md)
- [Convenções de nomes](../reference/conventions.md)
