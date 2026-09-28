# Ajuste de hiperparâmetros (Optuna)

Execute os comandos na **raiz do fork** `rl-baselines3-zoo`, com o venv ativado.

## Pré-requisito

Se cenários ou objetivos mudaram desde a última atualização do YAML `rl-baselines3-zoo/hyperparams/ppo.yml`:

```bash
python scripts/update_ppo_envs.py
```

## Comando básico com SQLite

```bash
python train.py \
  --algo ppo \
  --env FoodDelivery-medium-obj1-v1 \
  --n-timesteps 1000000 \
  --optimize-hyperparameters \
  --max-total-trials 200 \
  --n-jobs 2 \
  --storage sqlite:///optuna_studies.db \
  --study-name ppo_medium_obj1
```

Para retomar, execute o **mesmo comando** com o mesmo `--storage` e `--study-name`.

## Parâmetros

| Parâmetro | Descrição |
|-----------|-----------|
| `--algo` | Algoritmo (`ppo` e outros suportados pelo Zoo) |
| `--env` | ID Gymnasium registrado |
| `--n-timesteps` | Passos por trial |
| `--optimize-hyperparameters` | Ativa a busca com Optuna |
| `--max-total-trials` | Número máximo de trials |
| `--n-jobs` | Trials em paralelo |
| `--storage` | URI do banco (ex.: `sqlite:///optuna_studies.db`) |
| `--study-name` | Nome do estudo |
| `--optimization-log-path` | Opcional; se omitido, fica em `<exp-dir>/optimization/` |

Recomenda-se omitir `--optimization-log-path` para manter a estrutura esperada por `extract_best_trial.py`.

## Dashboard

```bash
optuna-dashboard sqlite:///optuna_studies.db
```

A interface fica em `http://localhost:8080` (o pacote `optuna-dashboard` pode exigir instalação separada).

## Extrair o melhor trial

```bash
python scripts/extract_best_trial.py \
  --exp-dir logs/ppo/FoodDelivery-medium-obj1-v1_1
```

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--exp-dir` / `-e` | Diretório do experimento no Zoo | obrigatório |
| `--env` | ID do ambiente | inferido do diretório |
| `--n-timesteps` | Timesteps no YAML gerado | `18000000` |
| `--n-envs` | Ambientes paralelos no YAML | `4` |
| `--output-dir` | Base de saída | `hyperparams/best_params_for_food_delivery_gym` |

O script escolhe o trial com melhor recompensa média no último checkpoint em `optimization/trial_*/evaluations.npz` e **anexa** o bloco ao `ppo.yml` da versão do pacote (ex.: `v1.2.x`).

Exemplo de bloco gerado:

```yaml
FoodDelivery-medium-obj1-v1:
  n_timesteps: 18000000
  policy: 'MultiInputPolicy'
  n_envs: 4
  learning_rate: 0.0009742009357947689
  normalize: true
  # ... demais hiperparâmetros
```

## Próximo passo

- [Treinamento](training.md)
