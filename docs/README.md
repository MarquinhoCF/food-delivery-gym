# Documentação do Food Delivery Gym

Este é o índice central da documentação. O [README principal](../README.md) concentra a visão geral e o anúncio da publicação no SBPO 2026. Os guias abaixo detalham instalação, uso do framework, experimentos, ferramentas e treinamento com RL Baselines3 Zoo.

## Começando

| Documento | Descrição |
|-----------|-----------|
| [Instalação](getting-started/installation.md) | Ambientes do gym e do Zoo, `requirements.txt`, freeze e Python 3.10.12 |
| [Início rápido](getting-started/quickstart.md) | Primeiros comandos para validar a instalação |

## Framework

| Documento | Descrição |
|-----------|-----------|
| [Visão geral](framework/overview.md) | Arquitetura do simulador e fluxo básico |
| [Cenários](framework/scenarios.md) | Schema JSON, geradores de pedidos e registro Gymnasium |
| [Objetivos de recompensa](framework/reward-objectives.md) | Objetivos 1 a 13 |
| [Otimizadores](framework/optimizers.md) | Uso, rollout, RL e extensão de `OptimizerGym` |
| [Catálogo de otimizadores](framework/optimizer-catalog.md) | Chaves, aliases, custos, variantes e descoberta RL |

## Experimentos

| Documento | Descrição |
|-----------|-----------|
| [YAML de experimento](experiments/yaml.md) | Schema, precedência CLI/YAML e exemplo |
| [Avaliação em lote](experiments/batch-eval.md) | Fluxo recomendado de execução e pós-processamento |
| [Layout de dados](experiments/data-layout.md) | Estrutura de `data/runs/`, modelos e artefatos |
| [Reprodução SBPO](experiments/reproducing-sbpo.md) | Dados do Drive (tag `v1.2.8`), layout legado e regeneração de figuras |

## Ferramentas

| Documento | Descrição |
|-----------|-----------|
| [test_runner](tools/test-runner.md) | Execução interativa, automática e com agente |
| [run_batch_eval](tools/run-batch-eval.md) | Avaliação em lote de heurísticas e modelos |
| [report](tools/report.md) | Planilha e boxplots a partir do `run.json` |
| [generate_plots](tools/generate-plots.md) | Gráficos por episódio e agregados |
| [generate_table](tools/generate-table.md) | Planilha Excel consolidada |
| [generate_boxplots](tools/generate-boxplots.md) | Boxplots comparativos |
| [convert_metrics](tools/convert-metrics.md) | Conversão NPZ ↔ JSON |
| [collect_terminal_cost](tools/collect-terminal-cost.md) | Coleta de custo terminal para rollout |
| [visualize_rollout_decisions](tools/visualize-rollout-decisions.md) | Visualização de decisões do rollout |

## RL Baselines3 Zoo

| Documento | Descrição |
|-----------|-----------|
| [Visão geral](rl-baselines3-zoo/README.md) | Dois repositórios, papéis e fluxo |
| [Configuração](getting-started/installation.md) | Seguir seção 7 em diante |
| [Ajuste de hiperparâmetros](rl-baselines3-zoo/hyperparameter-tuning.md) | Optuna e `extract_best_trial.py` |
| [Treinamento](rl-baselines3-zoo/training.md) | Treino PPO e visualização de curvas |
| [Artefatos](rl-baselines3-zoo/artifacts.md) | Layout de modelos e uso na avaliação |
| [Problemas comuns](rl-baselines3-zoo/troubleshooting.md) | Erros frequentes e versões pinadas |

## Referência

| Documento | Descrição |
|-----------|-----------|
| [Cheatsheet CLI](reference/cli-cheatsheet.md) | Comandos principais em uma página |
| [Convenções](reference/conventions.md) | IDs de ambiente, nomes de agentes e caminhos |
| [Legado](reference/legacy.md) | Scripts e layouts antigos |

## Trilhas por perfil

### Reprodução dos resultados do artigo SBPO 2026 sem retreinar

1. [Reprodução SBPO](experiments/reproducing-sbpo.md) (dados do Drive = tag `v1.2.8`, layout legado)
2. [Layout de dados](experiments/data-layout.md)
3. [generate_table](tools/generate-table.md), [generate_boxplots](tools/generate-boxplots.md), [generate_plots](tools/generate-plots.md), [convert_metrics](tools/convert-metrics.md)

Não use [report](tools/report.md) sobre os dados do artigo: ele exige `run.json` (pipeline atual).

### Desenvolvimento do simulador

1. [Instalação](getting-started/installation.md)
2. [Início rápido](getting-started/quickstart.md)
3. [Cenários](framework/scenarios.md)
4. [test_runner](tools/test-runner.md)

### Treinamento e avaliação PPO

1. [Cenários](framework/scenarios.md)
2. [RL Baselines3 Zoo](rl-baselines3-zoo/README.md)
3. [run_batch_eval](tools/run-batch-eval.md)
4. [report](tools/report.md)
