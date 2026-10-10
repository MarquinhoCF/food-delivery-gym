import pytest

from food_delivery_gym.main.optimizer.optimizer_gym.first_driver_optimizer_gym import (
    FirstDriverOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.monte_carlo_tree_search_optimizer_gym import (
    MonteCarloTreeSearchOptimizerGym,
    _TreeNode,
)
from food_delivery_gym.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import (
    NearestDriverOptimizerGym,
)
from food_delivery_gym.test.conftest import TINY, make_env
from food_delivery_gym.test.optimizer.test_mcts_myopic_expansion import (
    _stub_expand_outcome,
)


def test_heuristic_expansion_order_follows_nearest_ranking():
    env = make_env(TINY, seed=9)
    num_drivers = env.num_drivers
    rewards = [float(i) for i in range(num_drivers)]
    call_log: list[int] = []
    optimizer = MonteCarloTreeSearchOptimizerGym(
        env,
        base_optimizer_cls=NearestDriverOptimizerGym,
        horizon=1,
        iterations=num_drivers,
        depth=1,
        max_outcomes=1,
        max_expanded_actions=num_drivers,
        expansion_order="heuristic",
        record_decisions=False,
    )
    optimizer._expand_outcome = _stub_expand_outcome(rewards, call_log)

    root = _TreeNode(env=None, obs=None, done=False, truncated=False, depth=0)
    optimizer._order_untried_actions(root, num_drivers)

    base = NearestDriverOptimizerGym(env)
    expected = base.ranked_actions(
        env.get_observation(),
        env.get_drivers(),
        optimizer._route_for_order(env),
    )
    assert root.untried_actions == expected
    assert root.pending == {}
    assert call_log == []

    for _ in range(optimizer.iterations):
        optimizer._run_iteration(root, num_drivers)

    assert list(root.actions.keys()) == expected
    assert call_log == expected


def test_heuristic_does_not_preexpand_all_actions():
    env = make_env(TINY, seed=9)
    num_drivers = env.num_drivers
    rewards = [float(i) for i in range(num_drivers)]
    call_log: list[int] = []
    optimizer = MonteCarloTreeSearchOptimizerGym(
        env,
        base_optimizer_cls=NearestDriverOptimizerGym,
        horizon=1,
        iterations=1,
        depth=1,
        max_outcomes=1,
        max_expanded_actions=1,
        expansion_order="heuristic",
        record_decisions=False,
    )
    optimizer._expand_outcome = _stub_expand_outcome(rewards, call_log)

    root = _TreeNode(env=None, obs=None, done=False, truncated=False, depth=0)
    optimizer._run_iteration(root, num_drivers)

    assert len(call_log) == 1
    assert len(root.actions) == 1
    assert root.pending == {}


def test_heuristic_with_first_base_raises_on_constructor():
    env = make_env(TINY, seed=9)
    with pytest.raises(ValueError, match="ranked_actions"):
        MonteCarloTreeSearchOptimizerGym(
            env,
            base_optimizer_cls=FirstDriverOptimizerGym,
            expansion_order="heuristic",
        )


def test_invalid_expansion_order_raises():
    env = make_env(TINY, seed=9)
    with pytest.raises(ValueError, match="expansion_order"):
        MonteCarloTreeSearchOptimizerGym(
            env,
            base_optimizer_cls=NearestDriverOptimizerGym,
            expansion_order="myopic",  # type: ignore[arg-type]
        )


def test_expansion_order_in_title_hyperparameters_and_decision_log():
    env = make_env(TINY, seed=9)
    optimizer = MonteCarloTreeSearchOptimizerGym(
        env,
        base_optimizer_cls=NearestDriverOptimizerGym,
        horizon=1,
        iterations=2,
        depth=1,
        max_outcomes=1,
        expansion_order="heuristic",
        record_decisions=True,
    )
    assert "ord=heuristic" in optimizer.get_title()
    assert optimizer.get_hyperparameters()["expansion_order"] == "heuristic"

    obs = env.get_observation()
    order = env.get_current_order()
    action = optimizer.assign_driver_to_order(obs, order)
    assert action in range(env.num_drivers)
    assert optimizer.decision_log[0]["expansion_order"] == "heuristic"
