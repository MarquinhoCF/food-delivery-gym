"""Serialização da árvore MCTS no decision_log."""

from food_delivery_gym.main.optimizer.optimizer_gym.first_driver_optimizer_gym import (
    FirstDriverOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.monte_carlo_tree_search_optimizer_gym import (
    MonteCarloTreeSearchOptimizerGym,
)
from food_delivery_gym.test.conftest import TINY, make_env


def test_mcts_decision_log_includes_serialized_tree():
    env = make_env(TINY, seed=9)
    optimizer = MonteCarloTreeSearchOptimizerGym(
        env,
        base_optimizer_cls=FirstDriverOptimizerGym,
        horizon=1,
        iterations=4,
        exploration_weight=1.0,
        depth=1,
        max_outcomes=1,
        record_decisions=True,
    )
    obs = env.get_observation()
    order = env.get_current_order()
    action = optimizer.assign_driver_to_order(obs, order)

    assert len(optimizer.decision_log) == 1
    entry = optimizer.decision_log[0]
    assert entry["chosen_action"] == action
    assert "tree" in entry
    tree = entry["tree"]
    assert tree["depth"] == 0
    assert tree["visit_count"] >= 1
    assert tree["actions"]
    assert {a["action"] for a in tree["actions"]} == {
        c["action"] for c in entry["candidates"]
    }
    for action_node in tree["actions"]:
        assert "driver_id" in action_node
        assert action_node["outcomes"]
        for outcome in action_node["outcomes"]:
            assert "scenario_seed" in outcome
            assert outcome["node"] is not None
            assert outcome["node"]["depth"] == 1
