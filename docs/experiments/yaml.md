# YAML de experimento

Experimentos em lote são definidos por um arquivo YAML versionável e/ou por flags na CLI. Flags explícitas **sobrescrevem** o YAML. Sem YAML, `--name` é obrigatório.

Arquivo de referência versionado: [`experiments/example.yaml`](../../experiments/example.yaml).

> O diretório `experiments/` ignora a maior parte dos arquivos no Git; apenas `example.yaml` é versionado por padrão.

## Precedência

```
CLI explícita > YAML > defaults do código
```

Defaults importantes:

| Campo | Default |
|-------|---------|
| `objectives` | 1 a 13 |
| `scenarios` | `simple`, `medium`, `complex` |
| `num_runs` | `20` |
| `num_workers` | `1` |
| `seed` | `123456789` |
| `experiment_mode` | `cross_scenario` |
| `train_scenario` | `medium` |
| `model_base_dir` | `./data/ppo_training/` |
| `metrics_fmt` | `npz` |

## Campos permitidos

| Campo YAML | Equivalente CLI | Descrição |
|------------|-----------------|-----------|
| `name` | `--name` | Nome do experimento; saída em `data/runs/<name>/` |
| `objectives` | `-o` / `--objectives` | Lista de objetivos (1 a 13) |
| `scenarios` | `-s` / `--scenarios` | Lista de cenários |
| `agents` | `-a` / `--agents` | Heurísticas e/ou modelos |
| `heuristics` | `--heuristics` | Filtro de heurísticas |
| `models` | `-m` / `--models` | Filtro de modelos RL |
| `no_rl` | `--no-rl` | Não executa modelos RL |
| `no_heuristics` | `--no-heuristics` | Não executa heurísticas |
| `lowest` | `--lowest` | Variantes `cost=...` |
| `rollout` | `--rollout` | Variantes com `base`, `cost`, `horizon`, `alpha`, `terminal` |
| `rollout_record_decisions` | `--rollout-record-decisions` | Grava log de decisões |
| `num_runs` | `-n` / `--num-runs` | Episódios por agente |
| `num_workers` | `--num-workers` | Paralelismo por agente |
| `seed` | `--seed` | Semente |
| `experiment_mode` | `--experiment-mode` | `cross_scenario` ou `same_scenario` |
| `train_scenario` | `--train-scenario` | Cenário de treino no modo cross |
| `model_base_dir` | `--model-base-dir` | Raiz dos modelos PPO |
| `metrics_fmt` | `--metrics-fmt` | `npz` ou `json` |
| `batch_plots` | `--batch-plots` | Gráficos agregados |
| `all_plots` | `--all-plots` | Gráficos agregados + por episódio |

## Exemplo mínimo

```yaml
name: smoke_obj3
objectives: [3]
scenarios: [simple]
agents:
  - random
  - nearest_driver
  - lowest
lowest:
  - cost: route
no_rl: true
num_runs: 5
seed: 42
```

## Exemplo com rollout e PPO

Consulte [`experiments/example.yaml`](../../experiments/example.yaml): objetivo 3, três cenários, heurísticas, várias variantes de rollout e modelos `ppo_18M_steps` / `ppo_18M_steps_otimizado`.

## Regras de validação

- Se `lowest` ou `rollout` estiverem em `agents`, as variantes correspondentes devem ser declaradas (`lowest:` / `rollout:` no YAML ou flags `--lowest` / `--rollout`).
- `num_workers > 1` usa processos paralelos por agente.

## Execução

```bash
python -m scripts.run_batch_eval experiments/example.yaml
python -m scripts.run_batch_eval experiments/example.yaml --num-runs 2
python -m scripts.run_batch_eval --name smoke --objectives 3 --scenarios simple \
  --agents random --no-rl --num-runs 5
```

Documentação completa da CLI: [run_batch_eval](../tools/run-batch-eval.md).
