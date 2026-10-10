"""Parse de variante MCTS e smoke de select_driver."""

import pytest

from food_delivery_gym.main.optimizer import catalog as optimizer_catalog
from food_delivery_gym.main.optimizer.optimizer_gym.first_driver_optimizer_gym import (
    FirstDriverOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.monte_carlo_tree_search_optimizer_gym import (
    MonteCarloTreeSearchOptimizerGym,
)
from food_delivery_gym.test.conftest import TINY, make_env


def test_parse_mcts_cli_and_result_key_roundtrip():
    variant = optimizer_catalog.parse_mcts_cli(
        "base=nearest,horizon=5,alpha=0.9,terminal=0,"
        "iterations=8,exploration_weight=1,depth=2,max_outcomes=1"
    )
    assert variant.base_optimizer == "nearest_driver"
    assert variant.horizon == 5
    assert variant.iterations == 8
    assert variant.exploration_weight == 1.0
    assert variant.resolved_depth() == 2
    assert variant.max_outcomes == 1

    key = optimizer_catalog.mcts_result_key(
        variant.base_optimizer,
        variant.cost_function,
        horizon=variant.horizon,
        alpha=variant.alpha,
        terminal=variant.terminal,
        iterations=variant.iterations,
        exploration_weight=variant.exploration_weight,
        depth=variant.depth,
        max_outcomes=variant.max_outcomes,
    )
    assert key == "mcts_nearest_driver_h5_a0p9_tc0_i8_ew1_d2_o1"

    parsed = optimizer_catalog.parse_mcts_result_key(key)
    assert parsed is not None
    assert parsed.base_optimizer == "nearest_driver"
    assert parsed.iterations == 8
    assert parsed.resolved_depth() == 2


def test_parse_mcts_cli_with_lowest_base():
    variant = optimizer_catalog.parse_mcts_cli(
        "base=lowest,cost=route,horizon=5,iterations=4,"
        "exploration_weight=50,depth=1,max_outcomes=1"
    )
    key = optimizer_catalog.mcts_result_key(
        variant.base_optimizer,
        variant.cost_function,
        horizon=variant.horizon,
        alpha=variant.alpha,
        terminal=variant.terminal,
        iterations=variant.iterations,
        exploration_weight=variant.exploration_weight,
        depth=variant.depth,
        max_outcomes=variant.max_outcomes,
    )
    assert key == "mcts_lowest_route_cost_h5_a0p9_tc0_i4_ew50_d1_o1"
    parsed = optimizer_catalog.parse_mcts_result_key(key)
    assert parsed is not None
    assert parsed.cost_function == "route"


def test_expand_evaluations_mcts_variant():
    specs = [optimizer_catalog.get("mcts")]
    variants = optimizer_catalog.expand_evaluations(
        specs,
        mcts_variants=[
            optimizer_catalog.parse_mcts_cli(
                "base=nearest,horizon=5,iterations=8,depth=2"
            )
        ],
    )
    assert len(variants) == 1
    assert variants[0].result_key == "mcts_nearest_driver_h5_a0p9_tc0_i8_ew1_d2_o1"
    assert variants[0].extras["iterations"] == 8
    assert variants[0].extras["depth"] == 2
    assert "max_actions" not in variants[0].extras


def test_mcts_select_driver_smoke():
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
    now_before = env.simpy_env.now
    order_id_before = None if order is None else order.order_id

    action = optimizer.assign_driver_to_order(obs, order)

    assert action in range(env.num_drivers)
    # Ambiente real intacto: busca só usa clones resample.
    assert env.simpy_env.now == now_before
    assert env.current_order is not None
    assert env.current_order.order_id == order_id_before
    assert len(optimizer.decision_log) == 1
    assert optimizer.decision_log[0]["chosen_action"] == action
    assert optimizer.decision_log[0]["candidates"]

    env.step(action)
    assert env.simpy_env.now >= now_before


def test_mcts_depth_zero_falls_back_to_rollout_without_advancing_real():
    env = make_env(TINY, seed=11)
    optimizer = MonteCarloTreeSearchOptimizerGym(
        env,
        base_optimizer_cls=FirstDriverOptimizerGym,
        horizon=1,
        iterations=2,
        depth=0,
        record_decisions=False,
    )
    now_before = env.simpy_env.now
    obs = env.get_observation()
    order = env.get_current_order()
    action = optimizer.assign_driver_to_order(obs, order)
    assert action in range(env.num_drivers)
    assert env.simpy_env.now == now_before


def test_parse_mcts_cli_max_expanded_actions_and_result_key():
    default_variant = optimizer_catalog.parse_mcts_cli(
        "base=nearest,horizon=5,iterations=8,depth=2"
    )
    assert default_variant.max_expanded_actions is None
    default_key = optimizer_catalog.mcts_result_key(
        default_variant.base_optimizer,
        default_variant.cost_function,
        horizon=default_variant.horizon,
        alpha=default_variant.alpha,
        terminal=default_variant.terminal,
        iterations=default_variant.iterations,
        exploration_weight=default_variant.exploration_weight,
        depth=default_variant.depth,
        max_outcomes=default_variant.max_outcomes,
        max_expanded_actions=default_variant.max_expanded_actions,
    )
    assert default_key == "mcts_nearest_driver_h5_a0p9_tc0_i8_ew1_d2_o1"
    assert "_maxexp" not in default_key

    variant = optimizer_catalog.parse_mcts_cli(
        "base=nearest,horizon=5,iterations=8,depth=2,max_expanded_actions=4"
    )
    assert variant.max_expanded_actions == 4
    key = optimizer_catalog.mcts_result_key(
        variant.base_optimizer,
        variant.cost_function,
        horizon=variant.horizon,
        alpha=variant.alpha,
        terminal=variant.terminal,
        iterations=variant.iterations,
        exploration_weight=variant.exploration_weight,
        depth=variant.depth,
        max_outcomes=variant.max_outcomes,
        max_expanded_actions=variant.max_expanded_actions,
    )
    assert key == "mcts_nearest_driver_h5_a0p9_tc0_i8_ew1_d2_o1_maxexp4"
    parsed = optimizer_catalog.parse_mcts_result_key(key)
    assert parsed is not None
    assert parsed.max_expanded_actions == 4

    legacy = optimizer_catalog.parse_mcts_result_key(
        "mcts_nearest_driver_h5_a0p9_tc0_i8_ew1_d2_o1"
    )
    assert legacy is not None
    assert legacy.max_expanded_actions is None

    for token in ("all", "none", "null"):
        cleared = optimizer_catalog.parse_mcts_cli(
            f"base=nearest,depth=2,max_expanded_actions={token}"
        )
        assert cleared.max_expanded_actions is None


def test_expand_evaluations_mcts_includes_max_expanded_actions():
    variants = optimizer_catalog.expand_evaluations(
        [optimizer_catalog.get("mcts")],
        mcts_variants=[
            optimizer_catalog.parse_mcts_cli(
                "base=nearest,horizon=5,iterations=8,depth=2,max_expanded_actions=3"
            )
        ],
    )
    assert len(variants) == 1
    assert variants[0].extras["max_expanded_actions"] == 3
    assert variants[0].result_key.endswith("_maxexp3")


def test_parse_mcts_cli_expansion_order_and_result_key():
    default_variant = optimizer_catalog.parse_mcts_cli(
        "base=nearest,horizon=5,iterations=8,depth=2"
    )
    assert default_variant.expansion_order == "immediate"
    default_key = optimizer_catalog.mcts_result_key(
        default_variant.base_optimizer,
        default_variant.cost_function,
        horizon=default_variant.horizon,
        alpha=default_variant.alpha,
        terminal=default_variant.terminal,
        iterations=default_variant.iterations,
        exploration_weight=default_variant.exploration_weight,
        depth=default_variant.depth,
        max_outcomes=default_variant.max_outcomes,
        max_expanded_actions=default_variant.max_expanded_actions,
        expansion_order=default_variant.expansion_order,
    )
    assert default_key == "mcts_nearest_driver_h5_a0p9_tc0_i8_ew1_d2_o1"
    assert "_ordheur" not in default_key

    variant = optimizer_catalog.parse_mcts_cli(
        "base=nearest,horizon=5,iterations=8,depth=2,"
        "max_expanded_actions=4,expansion_order=heuristic"
    )
    assert variant.expansion_order == "heuristic"
    key = optimizer_catalog.mcts_result_key(
        variant.base_optimizer,
        variant.cost_function,
        horizon=variant.horizon,
        alpha=variant.alpha,
        terminal=variant.terminal,
        iterations=variant.iterations,
        exploration_weight=variant.exploration_weight,
        depth=variant.depth,
        max_outcomes=variant.max_outcomes,
        max_expanded_actions=variant.max_expanded_actions,
        expansion_order=variant.expansion_order,
    )
    assert key == "mcts_nearest_driver_h5_a0p9_tc0_i8_ew1_d2_o1_maxexp4_ordheur"
    parsed = optimizer_catalog.parse_mcts_result_key(key)
    assert parsed is not None
    assert parsed.expansion_order == "heuristic"
    assert parsed.max_expanded_actions == 4

    legacy = optimizer_catalog.parse_mcts_result_key(
        "mcts_nearest_driver_h5_a0p9_tc0_i8_ew1_d2_o1"
    )
    assert legacy is not None
    assert legacy.expansion_order == "immediate"


def test_parse_mcts_cli_heuristic_with_first_raises():
    with pytest.raises(ValueError, match="ranked_actions"):
        optimizer_catalog.parse_mcts_cli(
            "base=first_driver,horizon=5,iterations=8,depth=2,"
            "expansion_order=heuristic"
        )


def test_expand_evaluations_mcts_includes_expansion_order():
    variants = optimizer_catalog.expand_evaluations(
        [optimizer_catalog.get("mcts")],
        mcts_variants=[
            optimizer_catalog.parse_mcts_cli(
                "base=nearest,horizon=5,iterations=8,depth=2,"
                "expansion_order=heuristic"
            )
        ],
    )
    assert len(variants) == 1
    assert variants[0].extras["expansion_order"] == "heuristic"
    assert variants[0].result_key.endswith("_ordheur")
