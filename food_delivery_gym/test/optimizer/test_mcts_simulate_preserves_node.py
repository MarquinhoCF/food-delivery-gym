"""Simulação MCTS não muta o env guardado no nó da árvore."""

from food_delivery_gym.main.optimizer.optimizer_gym.first_driver_optimizer_gym import (
    FirstDriverOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.monte_carlo_tree_search_optimizer_gym import (
    MonteCarloTreeSearchOptimizerGym,
    _TreeNode,
)
from food_delivery_gym.test.conftest import TINY, make_env


def test_simulate_leaf_does_not_mutate_stored_node_env():
    env = make_env(TINY, seed=9)
    optimizer = MonteCarloTreeSearchOptimizerGym(
        env,
        base_optimizer_cls=FirstDriverOptimizerGym,
        horizon=2,
        iterations=1,
        depth=2,
        max_outcomes=1,
        record_decisions=False,
    )
    parent = env.clone(future="resample", scenario_seed=1)
    order = parent.get_current_order()
    assert order is not None
    obs_after, _reward, terminated, truncated, _info = parent.step(0)
    node = _TreeNode(
        env=parent,
        obs=obs_after,
        done=bool(terminated),
        truncated=bool(truncated),
        depth=1,
    )
    order_id_before = (
        None if parent.get_current_order() is None else parent.get_current_order().order_id
    )
    now_before = parent.simpy_env.now

    optimizer._simulate_leaf(node)

    assert parent.simpy_env.now == now_before
    order_after = parent.get_current_order()
    order_id_after = None if order_after is None else order_after.order_id
    assert order_id_after == order_id_before


def test_mcts_depth_two_with_many_iterations_does_not_crash():
    """Revisita filhos após explorar a raiz (iterations > num_drivers)."""
    env = make_env(TINY, seed=11)
    num_drivers = env.num_drivers
    optimizer = MonteCarloTreeSearchOptimizerGym(
        env,
        base_optimizer_cls=FirstDriverOptimizerGym,
        horizon=2,
        iterations=num_drivers + 4,
        exploration_weight=1.0,
        depth=2,
        max_outcomes=1,
        record_decisions=False,
    )
    # Avança quase ao fim: rollout curto costumava zerar current_order no nó.
    for _ in range(max(env.orders_generated - 3, 0)):
        if env.current_order is None:
            break
        obs = env.get_observation()
        order = env.get_current_order()
        action = optimizer.assign_driver_to_order(obs, order)
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break

    if env.current_order is None:
        return

    obs = env.get_observation()
    order = env.get_current_order()
    action = optimizer.assign_driver_to_order(obs, order)
    assert action in range(num_drivers)
