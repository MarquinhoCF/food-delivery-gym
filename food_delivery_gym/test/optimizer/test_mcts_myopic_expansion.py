import pytest

from food_delivery_gym.main.optimizer.optimizer_gym.first_driver_optimizer_gym import (
    FirstDriverOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.monte_carlo_tree_search_optimizer_gym import (
    MonteCarloTreeSearchOptimizerGym,
    _OutcomeChild,
    _TreeNode,
)
from food_delivery_gym.test.conftest import TINY, make_env


def _stub_expand_outcome(rewards: list[float], call_log: list[int] | None = None):
    """Dublê: recompensa fixa e filho terminal (sem depender do ambiente)."""

    def _expand_outcome(parent_env, action: int, child_depth: int):
        if call_log is not None:
            call_log.append(action)
        outcome = _OutcomeChild(
            scenario_seed=len(call_log) if call_log is not None else action,
            env=parent_env,
            obs={},
            done=True,
            truncated=False,
            node=_TreeNode(
                env=parent_env,
                obs={},
                done=True,
                truncated=False,
                depth=child_depth,
            ),
        )
        return outcome, float(rewards[action])

    return _expand_outcome


def _stub_expand_live(rewards: list[float], call_log: list[int] | None = None):
    """Dublê com filho vivo (permite profundidade > 1 na árvore)."""

    def _expand_outcome(parent_env, action: int, child_depth: int):
        if call_log is not None:
            call_log.append(action)
        outcome = _OutcomeChild(
            scenario_seed=len(call_log) if call_log is not None else action,
            env=parent_env,
            obs={},
            done=False,
            truncated=False,
            node=_TreeNode(
                env=parent_env,
                obs={},
                done=False,
                truncated=False,
                depth=child_depth,
            ),
        )
        return outcome, float(rewards[action])

    return _expand_outcome


def _walk_action_counts(node: _TreeNode) -> list[int]:
    counts = [len(node.actions)]
    for action_stats in node.actions.values():
        for outcome in action_stats.outcomes:
            if outcome.node is not None:
                counts.extend(_walk_action_counts(outcome.node))
    return counts


def test_myopic_expansion_order_and_tiebreak():
    env = make_env(TINY, seed=9)
    rewards = [1.0, 10.0, 3.0, 10.0, 0.0]
    num_drivers = len(rewards)
    optimizer = MonteCarloTreeSearchOptimizerGym(
        env,
        base_optimizer_cls=FirstDriverOptimizerGym,
        horizon=1,
        iterations=2,
        depth=1,
        max_outcomes=1,
        record_decisions=False,
    )
    optimizer._expand_outcome = _stub_expand_outcome(rewards)

    root = _TreeNode(env=None, obs=None, done=False, truncated=False, depth=0)
    for _ in range(optimizer.iterations):
        optimizer._run_iteration(root, num_drivers)

    assert list(root.actions.keys()) == [1, 3]
    assert root.untried_actions == [2, 0, 4]


def test_max_expanded_actions_limits_every_node():
    env = make_env(TINY, seed=9)
    rewards = [1.0, 5.0, 3.0, 4.0, 0.0]
    num_drivers = len(rewards)
    optimizer = MonteCarloTreeSearchOptimizerGym(
        env,
        base_optimizer_cls=FirstDriverOptimizerGym,
        horizon=2,
        iterations=20,
        depth=2,
        max_outcomes=1,
        max_expanded_actions=2,
        record_decisions=False,
    )
    optimizer._expand_outcome = _stub_expand_live(rewards)
    optimizer._simulate_leaf = lambda node: 0.0

    root = _TreeNode(env=None, obs=None, done=False, truncated=False, depth=0)
    for _ in range(optimizer.iterations):
        optimizer._run_iteration(root, num_drivers)

    for count in _walk_action_counts(root):
        assert count <= 2


def test_pending_cache_reused_no_duplicate_expand():
    env = make_env(TINY, seed=9)
    rewards = [1.0, 10.0, 3.0, 8.0, 0.0]
    num_drivers = len(rewards)
    call_log: list[int] = []
    optimizer = MonteCarloTreeSearchOptimizerGym(
        env,
        base_optimizer_cls=FirstDriverOptimizerGym,
        horizon=1,
        iterations=num_drivers,
        depth=1,
        max_outcomes=1,
        record_decisions=False,
    )
    optimizer._expand_outcome = _stub_expand_outcome(rewards, call_log)

    root = _TreeNode(env=None, obs=None, done=False, truncated=False, depth=0)
    optimizer._order_untried_actions(root, num_drivers)
    assert len(call_log) == num_drivers
    assert sorted(call_log) == list(range(num_drivers))
    # Melhor ação = 1 (recompensa 10); o desfecho expandido é o mesmo do pending.
    cached_outcome, cached_reward = root.pending[1]
    optimizer._run_iteration(root, num_drivers)

    assert 1 in root.actions
    assert 1 not in root.pending
    assert root.actions[1].outcomes[0] is cached_outcome
    assert root.actions[1].immediate_reward == cached_reward
    assert len(call_log) == num_drivers

    for _ in range(num_drivers - 1):
        optimizer._run_iteration(root, num_drivers)

    assert len(call_log) == num_drivers
    assert len(root.actions) == num_drivers
    assert root.pending == {}
    for action_stats in root.actions.values():
        assert len(action_stats.outcomes) == 1


def test_max_expanded_actions_invalid_raises():
    env = make_env(TINY, seed=9)
    with pytest.raises(ValueError, match="max_expanded_actions"):
        MonteCarloTreeSearchOptimizerGym(
            env,
            base_optimizer_cls=FirstDriverOptimizerGym,
            max_expanded_actions=0,
        )
    with pytest.raises(ValueError, match="max_expanded_actions"):
        MonteCarloTreeSearchOptimizerGym(
            env,
            base_optimizer_cls=FirstDriverOptimizerGym,
            max_expanded_actions=-1,
        )


def test_max_expanded_actions_in_title_hyperparameters_and_decision_log():
    env = make_env(TINY, seed=9)
    optimizer_all = MonteCarloTreeSearchOptimizerGym(
        env,
        base_optimizer_cls=FirstDriverOptimizerGym,
        horizon=1,
        iterations=2,
        depth=1,
        max_outcomes=1,
        max_expanded_actions=None,
        record_decisions=False,
    )
    assert "maxexp=all" in optimizer_all.get_title()
    assert optimizer_all.get_hyperparameters()["max_expanded_actions"] is None

    optimizer = MonteCarloTreeSearchOptimizerGym(
        env,
        base_optimizer_cls=FirstDriverOptimizerGym,
        horizon=1,
        iterations=2,
        depth=1,
        max_outcomes=1,
        max_expanded_actions=1,
        record_decisions=True,
    )
    assert "maxexp=1" in optimizer.get_title()
    assert optimizer.get_hyperparameters()["max_expanded_actions"] == 1

    obs = env.get_observation()
    order = env.get_current_order()
    action = optimizer.assign_driver_to_order(obs, order)
    assert action in range(env.num_drivers)
    assert len(optimizer.decision_log) == 1
    assert optimizer.decision_log[0]["max_expanded_actions"] == 1
