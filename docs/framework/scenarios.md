# Cenários experimentais

Os cenários definem a cidade simulada: demanda, frota, estabelecimentos e duração. Cada arquivo `.json` em `food_delivery_gym/main/scenarios/` é descoberto automaticamente.

## Cenários disponíveis

Arquivos presentes no repositório:

| Arquivo | Uso típico |
|---------|------------|
| `simple.json` | Cenário de menor demanda (artigo) |
| `medium.json` | Cenário intermediário (artigo); padrão em vários scripts |
| `complex.json` | Cenário de maior demanda (artigo) |

Os arquivos restantes em `food_delivery_gym/main/scenarios/` foram e são utilizados para testes apenas.

## Parâmetros principais

| Campo | Descrição |
|-------|-----------|
| `order_generator` | Processo de chegada de pedidos |
| `simpy_env.max_time_step` | Tempo máximo da simulação (minutos) |
| `grid_map.size` | Tamanho do grid (ex.: 50) |
| `drivers` | Frota: quantidade, velocidade, capacidade, tolerância |
| `establishments` | Restaurantes: preparo, raio, capacidade, momento de alocação |

## Geração de pedidos

| Tipo | Descrição |
|------|-----------|
| `poisson` | Taxa constante λ |
| `non_homogeneous_poisson` | Taxa variável λ(t) |

### Campos do gerador

| Campo | Obrigatório | Descrição |
|-------|-------------|-----------|
| `type` | Sim | `poisson` ou `non_homogeneous_poisson` |
| `estimated_num_orders` | Sim | Total estimado de pedidos |
| `time_window` | Sim | Janela de geração (minutos) |
| `lambda_rate` | Não | λ do Poisson homogêneo; se omitido: `estimated_num_orders / time_window` |
| `rate_function` | Sim (não homogêneo) | Expressão Python de λ(t) |
| `max_rate` | Não | Taxa máxima para thinning |

### Exemplo: Poisson homogêneo

```json
"order_generator": {
  "type": "poisson",
  "estimated_num_orders": 288,
  "time_window": 1440,
  "lambda_rate": 0.2
}
```

### Exemplo: Poisson não homogêneo

```json
"order_generator": {
  "type": "non_homogeneous_poisson",
  "estimated_num_orders": 576,
  "time_window": 960,
  "rate_function": "lambda t: 0.3115 + 0.9345 * (np.exp(-((t - 330)**2) / 7000) + np.exp(-((t - 630)**2) / 7000))"
}
```

### Poisson Tuner

Para calibrar λ(t) visualmente, use o [Poisson Tuner](https://github.com/MarquinhoCF/poisson-tuner): configuração de picos, escala automática e exportação da `rate_function` pronta para o JSON.

## Motoristas

| Campo | Descrição |
|-------|-----------|
| `num` | Quantidade de motoristas |
| `vel` | Intervalo `[mínimo, máximo]` de velocidade |
| `tolerance_percentage` | Tolerância de reordenação de rota |
| `max_capacity` | Capacidade máxima de pedidos simultâneos |
| `type` | Variante de política de rota (quando aplicável) |

## Estabelecimentos

| Campo | Descrição |
|-------|-----------|
| `num` | Quantidade de estabelecimentos |
| `prepare_time` | Tempo de preparo `[mín, máx]` (minutos) |
| `operating_radius` | Raio de operação `[mín, máx]` |
| `production_capacity` | Capacidade de produção `[mín, máx]` |
| `percentage_allocation_driver` | Fração do preparo (0-1) que dispara a alocação do motorista |

Exemplo: `0.7` solicita o motorista quando 70% do preparo estiver concluído.

## Exemplo completo

```json
{
  "order_generator": {
    "type": "poisson",
    "estimated_num_orders": 288,
    "time_window": 1440
  },
  "simpy_env": {
    "max_time_step": 2880
  },
  "grid_map": {
    "size": 50
  },
  "drivers": {
    "num": 10,
    "vel": [3, 5],
    "tolerance_percentage": 50,
    "max_capacity": 2
  },
  "establishments": {
    "num": 10,
    "prepare_time": [20, 60],
    "operating_radius": [5, 30],
    "production_capacity": [4, 4],
    "percentage_allocation_driver": 0.7
  }
}
```

A validação do schema está em `food_delivery_gym/main/scenarios/spec.py`.

## Registro Gymnasium

O registro ocorre em `food_delivery_gym/__init__.py`:

```
FoodDelivery-{stem_do_json}-obj{N}-v1
```

Para adicionar um cenário: coloque um novo `.json` em `food_delivery_gym/main/scenarios/` e reinstale o pacote (`pip install -e .`). Para usá-lo no RL Baselines3 Zoo, atualize o `ppo.yml` com `scripts/update_ppo_envs.py` no fork (após a [instalação do Zoo](../getting-started/installation.md#7-treinamento-com-rl-baselines3-zoo)).

## Documentação relacionada

- [Objetivos de recompensa](reward-objectives.md)
- [Visão geral](overview.md)
- [Convenções](../reference/conventions.md)
