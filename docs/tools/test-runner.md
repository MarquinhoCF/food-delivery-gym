# Script `test_runner`

Executa um episódio de forma interativa ou automática, útil para desenvolvimento e depuração.

```bash
python -m scripts.test_runner [opções]
```

Quando usar `--render`, é necessário ter o `.env` configurado (consulte [Instalação](../getting-started/installation.md)).

## Modos

| `--mode` | Descrição |
|----------|-----------|
| `interactive` | Execução passo a passo (padrão) |
| `auto` | Execução automática com o otimizador escolhido |
| `agent` | Uso com modelo de RL (recomendado com `--optimizer rl` ou nome descoberto) |

## Opções

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--scenario` | Arquivo JSON do cenário | `medium.json` |
| `--mode` | `interactive`, `auto` ou `agent` | `interactive` |
| `--optimizer` | Chave do catálogo ou modelo descoberto | `random` |
| `--lowest` | Variante de lowest no formato `cost=...` (obrigatório com `--optimizer lowest`) | (nenhum) |
| `--rollout` | Variante de rollout no formato `base=...,cost=...,horizon=...,alpha=...,terminal=...` (obrigatório com `--optimizer rollout`) | (nenhum) |
| `--mcts` | Variante de MCTS no formato `base=...,iterations=...,depth=...,max_expanded_actions=...` (obrigatório com `--optimizer mcts`) | (nenhum) |
| `--model-path` | Caminho para `best_model.zip` (com `--optimizer rl`) | (nenhum) |
| `--objective` | Objetivo de 1 a 13 | `1` |
| `--seed` | Semente aleatória | padrão do script |
| `--render` | Ativa visualização gráfica | desligado |
| `--max-steps` | Limite de passos no episódio | `10000` |
| `--save-log` | Redireciona a saída para `log.txt` | desligado |

As flags `--lowest`, `--rollout` e `--mcts` usam o **mesmo formato** do [`run_batch_eval`](run-batch-eval.md). No `test_runner` basta uma variante por execução.

## Exemplos

```bash
# Interativo com render
python -m scripts.test_runner --mode interactive --scenario medium.json --render

# Heurística nearest
python -m scripts.test_runner --mode auto --optimizer nearest --scenario simple.json

# Lowest com custo de rota
python -m scripts.test_runner --mode auto --optimizer lowest --lowest cost=route --objective 3

# Rollout
python -m scripts.test_runner --mode auto --optimizer rollout \
  --rollout base=lowest,cost=route,horizon=5,terminal=0

# MCTS
python -m scripts.test_runner --mode auto --optimizer mcts \
  --mcts base=nearest,horizon=5,iterations=8,depth=2

# Modelo PPO por caminho
python -m scripts.test_runner --mode agent --optimizer rl \
  --model-path caminho/para/best_model.zip --render
```

## Regras de uso

- `--optimizer lowest` exige `--lowest` (ex.: `--lowest cost=route`).
- `--optimizer rollout` exige `--rollout` (ex.: `--rollout base=lowest,cost=route,horizon=5,terminal=0`).
- `--optimizer mcts` exige `--mcts` (ex.: `--mcts base=nearest,horizon=5,iterations=8,depth=2`).
- `--lowest` só pode ser usado com `--optimizer lowest`.
- `--rollout` só pode ser usado com `--optimizer rollout`.
- `--mcts` só pode ser usado com `--optimizer mcts`.
- `--optimizer rl` exige `--model-path`.
- O arquivo `vecnormalize.pkl` é buscado automaticamente perto do modelo.

Consulte também [Otimizadores](../framework/optimizers.md) e o [catálogo](../framework/optimizer-catalog.md).
