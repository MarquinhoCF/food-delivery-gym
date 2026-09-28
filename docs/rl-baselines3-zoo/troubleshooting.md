# Problemas comuns (RL Baselines3 Zoo)

## `ModuleNotFoundError: No module named 'pkg_resources'`

Versões recentes do `setuptools` (81 ou superior) podem não incluir `pkg_resources`:

```bash
pip install "setuptools<81"
```

## Incompatibilidade entre SB3, sb3-contrib e rl_zoo3

Use versões alinhadas. Para reprodução, consulte [`requirements-freeze.txt`](../../requirements-freeze.txt) do food-delivery-gym (Python **3.10.12**):

| Pacote | Versão no freeze |
|--------|------------------|
| `stable_baselines3` | 2.7.0 |
| `sb3_contrib` | 2.7.0 |
| `rl_zoo3` | 2.7.0 |
| `gymnasium` | 1.2.1 |
| `torch` | 2.8.0 |

```bash
pip install stable_baselines3==2.7.0 sb3-contrib==2.7.0 rl_zoo3==2.7.0 gymnasium==1.2.1
```

## Erros de API do Gymnasium

Evite misturar o pacote legado `gym` com `gymnasium`. Confirme a instalação:

```bash
python -c "import gymnasium; print(gymnasium.__version__)"
```

## CUDA e GPU

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Se `torch.cuda.is_available()` for `False`, reinstale o PyTorch pelo [seletor oficial](https://pytorch.org/get-started/locally/). O treino em CPU funciona, porém é mais lento.

O freeze inclui wheels Linux/CUDA; em outras plataformas, veja [Instalação](../getting-started/installation.md).

## Ambiente não encontrado no `train.py`

1. Execute `pip install -e .` no food-delivery-gym no **mesmo** venv.
2. Execute `python scripts/update_ppo_envs.py` no Zoo.
3. Confirme o ID: `FoodDelivery-{cenário}-obj{N}-v1`.

## Marcador do `ppo.yml`

O script `update_ppo_envs.py` localiza a seção Food Delivery por um comentário marcador. Não remova o marcador `# ========== FOOD DELIVERY GYM ENVS ==========` (ou equivalente) do `hyperparams/ppo.yml`.

## `optuna-dashboard` não encontrado

```bash
pip install optuna-dashboard
```
