# Script `run_batch_eval`

Avalia em lote heurísticas e modelos de RL em combinações de cenários e objetivos.

```bash
python -m scripts.run_batch_eval [EXPERIMENT.yaml] [opções]
```

Schema do YAML: [YAML de experimento](../experiments/yaml.md).  
Fluxo recomendado: [Avaliação em lote](../experiments/batch-eval.md).

## O que faz

- Executa agentes do catálogo e modelos descobertos em `--model-base-dir`
- Aceita YAML e/ou CLI (a CLI sobrescreve o YAML)
- Grava `run.json`, `summary.csv` e, por agente, `episodes.csv`, `metrics_data.*` e `summary.json`
- Retoma automaticamente quando as métricas do agente já existem e são legíveis

## Opções principais

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `EXPERIMENT.yaml` | Arquivo YAML (posicional, opcional) | (nenhum) |
| `--name` | Nome do experimento (obrigatório sem YAML) | (nenhum) |
| `-o` / `--objectives` | Objetivos de 1 a 13 | todos |
| `-s` / `--scenarios` | Cenários | `simple medium complex` |
| `-a` / `--agents` | Agentes | todos disponíveis |
| `--heuristics` | Filtro de heurísticas | (nenhum) |
| `-m` / `--models` | Filtro de modelos RL | descoberta automática |
| `--lowest` | Variante `cost=...` (repetível) | (nenhum) |
| `--rollout` | Variante `base=...,cost=...,horizon=...` (repetível) | (nenhum) |
| `--rollout-record-decisions` | Grava log de decisões | desligado |
| `--no-rl` | Exclui modelos RL | (ausente) |
| `--no-heuristics` | Exclui heurísticas | (ausente) |
| `-n` / `--num-runs` | Episódios por agente | `20` |
| `--num-workers` | Processos paralelos | `1` |
| `--seed` | Semente | `123456789` |
| `--experiment-mode` | `cross_scenario` ou `same_scenario` | `cross_scenario` |
| `--train-scenario` | Cenário de treino (modo cross) | `medium` |
| `--model-base-dir` | Raiz dos modelos | `./data/ppo_training/` |
| `--metrics-fmt` | `npz` ou `json` | `npz` |
| `--batch-plots` | Gera gráficos agregados | (ausente) |
| `--all-plots` | Gera agregados e por episódio | (ausente) |

## Layout de saída

```
data/runs/<name>/
  run.json
  summary.csv
  obj_<N>/<scenario>/<agent>/
    episodes.csv
    metrics_data.npz
    summary.json
    figs/                  # se --batch-plots / --all-plots
```

## Exemplos

```bash
# Via YAML
python -m scripts.run_batch_eval experiments/example.yaml

# Sobrescrevendo parâmetros
python -m scripts.run_batch_eval experiments/example.yaml --num-runs 2

# Sem YAML
python -m scripts.run_batch_eval --name smoke --objectives 3 --scenarios simple \
  --agents random nearest_driver --lowest cost=route --no-rl --num-runs 5

# Apenas PPO
python -m scripts.run_batch_eval --name rl_only --no-heuristics --experiment-mode same_scenario

# Cross-scenario
python -m scripts.run_batch_eval --name cross_med \
  --experiment-mode cross_scenario --train-scenario medium

# Rollout
python -m scripts.run_batch_eval --name costs --agents lowest rollout --no-rl \
  --lowest cost=route \
  --rollout base=lowest,cost=route,horizon=5,terminal=0
```

## Modos RL e estrutura de modelos

Consulte [Layout de dados](../experiments/data-layout.md), [Artefatos do Zoo](../rl-baselines3-zoo/artifacts.md) e o [catálogo de otimizadores](../framework/optimizer-catalog.md).

Após a execução:

```bash
python -m scripts.report data/runs/<name>
```
