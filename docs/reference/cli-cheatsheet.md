# Cheatsheet CLI

Todos os comandos assumem o ambiente virtual ativado e o diretório raiz do `food-delivery-gym`, salvo indicação contrária.

## Instalação

```bash
python -m venv venv && source venv/bin/activate
python -m pip install -r requirements.txt && python -m pip install -e .
# Reprodução exata (Python 3.10.12):
# python -m pip install -r requirements-freeze.txt && python -m pip install -e .
cp .env.example .env
```

## Simulador

```bash
python -m scripts.test_runner --mode interactive --scenario medium.json --render
python -m scripts.test_runner --mode auto --optimizer lowest --lowest cost=route --objective 3
```

## Experimentos

```bash
python -m scripts.run_batch_eval experiments/example.yaml
python -m scripts.report data/runs/experimento_exemplo
python -m scripts.report data/runs/experimento_exemplo --episodes
```

## Figuras e tabela (manual / legado)

```bash
python -m scripts.generate_table -r data/runs/execucoes -o 3
python -m scripts.generate_boxplots -r data/runs/execucoes -o 3
python -m scripts.generate_plots -r data/runs/execucoes -o 3
python -m scripts.convert_metrics data/runs/execucoes --dry-run
```

## Rollout (pesquisa)

```bash
python -m scripts.collect_terminal_cost --help
python -m scripts.fit_terminal_cost --help
python -m scripts.visualize_rollout_decisions --decision 0 --help
python -m scripts.visualize_mcts_decisions --decision 0 --help
```

## Testes

```bash
pytest
```

## RL Baselines3 Zoo (no fork)

```bash
python scripts/update_ppo_envs.py
python train.py --algo ppo --env FoodDelivery-medium-obj3-v1 --n-timesteps 18000000 --log-folder logs/training/
python scripts/extract_best_trial.py --exp-dir logs/ppo/FoodDelivery-medium-obj3-v1_1
python scripts/plot_train.py -a ppo -e FoodDelivery-medium-obj3-v1 -f logs/training/
```
