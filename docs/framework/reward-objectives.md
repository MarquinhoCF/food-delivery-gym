# Objetivos de recompensa

O ambiente define **13 objetivos** de recompensa (`FoodDeliveryGymEnv.REWARD_OBJECTIVES = 1..13`).

## Lista

| Objetivo | Resumo |
|----------|--------|
| 1 | Minimizar tempo de entrega a partir da expectativa de tempo (penalidade por passo) |
| 2 | Minimizar custo operacional a partir da expectativa de distância (penalidade por passo) |
| 3 | Minimizar tempo efetivo gasto pelos motoristas (penalidade por passo); **objetivo principal do artigo SBPO** |
| 4 | Minimizar custo a partir da distância efetiva (penalidade por passo) |
| 5 | Como o 1, mas com recompensa ao final do episódio |
| 6 | Como o 2, mas ao final do episódio |
| 7 | Como o 3, mas ao final do episódio |
| 8 | Como o 4, mas ao final do episódio |
| 9 | Como o 3, com penalização 5× para pedidos não coletados (por passo) |
| 10 | Como o 9, ao final do episódio |
| 11 | Maximizar número de pedidos entregues (recompensa positiva por entrega) |
| 12 | Penaliza pelo tempo total de cada pedido entregue no passo |
| 13 | Penaliza pelo tempo de pedidos na pipeline e pelos entregues no passo |

## Uso

```bash
# test_runner
python -m scripts.test_runner --objective 3 --mode auto --scenario medium.json

# batch eval
python -m scripts.run_batch_eval --name obj3 --objectives 3 --scenarios simple medium complex --no-rl
```

No YAML:

```yaml
objectives: [3]
```

## Adicionar um novo objetivo

1. Atualize `REWARD_OBJECTIVES` em `FoodDeliveryGymEnv`.
2. Implemente a lógica em `_calculate_reward`.
3. Documente neste mesmo arquivo `docs/framework/reward-objectives.md`.
4. Reinstale o pacote e, se usar o Zoo, rode `python scripts/update_ppo_envs.py` no fork.
