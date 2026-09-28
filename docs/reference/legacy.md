# Conteúdo legado

Itens mantidos no repositório, mas **não** recomendados como fluxo principal.

## Scripts em `scripts/legacy/`

Exemplos: treino SB3 direto sem Zoo (`train_ppo_model.py`), visualização de grid, experimentos OR-Tools, plots antigos.

Use apenas para referência histórica ou depuração pontual.

## Layout `data/runs/execucoes`

Formato da tag **`v1.2.8`**, usado nos dados do artigo SBPO (sufixo `_scenario` nas pastas, `results.txt`). Os scripts `generate_*` ainda usam esse caminho como default. Experimentos novos devem usar `data/runs/<name>/` via `run_batch_eval`. Detalhes: [Reprodução SBPO](../experiments/reproducing-sbpo.md).

## Entry point `python -m food_delivery_gym.main`

Demo antiga baseada em SimPy, fora do fluxo Gymnasium moderno.

## Treino sem Zoo

É tecnicamente possível treinar com Stable-Baselines3 puro (`scripts/legacy/train_ppo_model.py`). O fluxo documentado e usado no artigo é o [RL Baselines3 Zoo](../rl-baselines3-zoo/README.md).
