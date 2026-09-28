# Treinamento PPO

Execute na raiz do fork `rl-baselines3-zoo`.

## Hiperparâmetros

**Opção A: padrões do PPO**

Garanta que o ambiente existe em `hyperparams/ppo.yml`:

```bash
python scripts/update_ppo_envs.py
```

**Opção B: melhores hiperparâmetros do Optuna**

Gere o YAML com [extract_best_trial](hyperparameter-tuning.md) e use `--conf-file`.

## Comandos

```bash
# Padrão PPO
python train.py \
  --algo ppo \
  --env FoodDelivery-medium-obj1-v1 \
  --n-timesteps 18000000 \
  --log-folder logs/training/

# Com YAML otimizado (ajuste a versão v1.2.x conforme o pacote instalado)
python train.py \
  --algo ppo \
  --env FoodDelivery-medium-obj1-v1 \
  --conf-file hyperparams/best_params_for_food_delivery_gym/v1.2.x/ppo.yml \
  --n-timesteps 18000000 \
  --log-folder logs/training/
```

### Com avaliação frequente e seed

```bash
python train.py \
  --algo ppo \
  --env FoodDelivery-medium-obj1-v1 \
  --conf-file hyperparams/best_params_for_food_delivery_gym/v1.2.x/ppo.yml \
  --n-timesteps 18000000 \
  --eval-freq 10000 \
  --eval-episodes 10 \
  --seed 42 \
  --log-folder logs/training/
```

### Checkpoints intermediários

```bash
python train.py \
  --algo ppo \
  --env FoodDelivery-medium-obj1-v1 \
  --conf-file hyperparams/best_params_for_food_delivery_gym/v1.2.x/ppo.yml \
  --n-timesteps 18000000 \
  --save-freq 500000 \
  --log-folder logs/training/
```

## Parâmetros úteis

| Parâmetro | Descrição |
|-----------|-----------|
| `--conf-file` | YAML de hiperparâmetros |
| `--n-timesteps` | Total de passos de treino |
| `--log-folder` | Destino de logs e modelos |
| `--eval-episodes` | Episódios de avaliação durante o treino |
| `--eval-freq` | Frequência de avaliação (em passos) |
| `--save-freq` | Frequência de checkpoints |
| `--seed` | Semente para reprodutibilidade |
| `--verbose` | `0` ou `1` |

## Curva de aprendizado

```bash
python scripts/plot_train.py -a ppo -e FoodDelivery-medium-obj1-v1 -f logs/training/
```

## Após o treino

Organize ou copie os artefatos para o layout esperado pelo simulador e avalie com `run_batch_eval`. Consulte [Artefatos](artifacts.md).
