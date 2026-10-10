import numpy as np
import pytest

from food_delivery_gym.main.optimizer import catalog as optimizer_catalog
from food_delivery_gym.main.optimizer.optimizer_gym.lowest_cost_driver_optimizer_gym import (
    LowestCostDriverOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import (
    NearestDriverOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.random_driver_optimizer_gym import (
    RandomDriverOptimizerGym,
)
from food_delivery_gym.main.route.delivery_route_segment import DeliveryRouteSegment
from food_delivery_gym.main.route.pickup_route_segment import PickupRouteSegment
from food_delivery_gym.main.route.route import Route
from food_delivery_gym.test.conftest import TINY, make_env


def _route_for_current_order(env) -> Route:
    order = env.get_current_order()
    return Route(
        env.get_simpy_env(),
        [PickupRouteSegment(order), DeliveryRouteSegment(order)],
    )


def test_nearest_ranked_actions_best_first_with_index_tiebreak():
    env = make_env(TINY, seed=9)
    opt = NearestDriverOptimizerGym(env)
    obs = env.get_observation()
    drivers = env.get_drivers()
    route = _route_for_current_order(env)
    map_ = env.simpy_env.map

    ranked = opt.ranked_actions(obs, drivers, route)
    assert set(ranked) == set(range(len(drivers)))
    distances = [
        opt.compare_distance(map_, drivers[i], route) for i in ranked
    ]
    assert distances == sorted(distances)
    for i in range(len(ranked) - 1):
        if distances[i] == distances[i + 1]:
            assert ranked[i] < ranked[i + 1]

    assert ranked[0] == opt.select_driver(obs, drivers, route)


def test_lowest_ranked_actions_best_first_with_index_tiebreak():
    env = make_env(TINY, seed=9, reward_objective=3)
    cost_fn = optimizer_catalog.make_cost_function("route", objective=3)
    opt = LowestCostDriverOptimizerGym(env, cost_function=cost_fn)
    obs = env.get_observation()
    drivers = env.get_drivers()
    route = _route_for_current_order(env)

    ranked = opt.ranked_actions(obs, drivers, route)
    assert set(ranked) == set(range(len(drivers)))
    costs = [opt.get_cost_for_driver(drivers[i], route) for i in ranked]
    assert costs == sorted(costs)
    for i in range(len(ranked) - 1):
        if costs[i] == costs[i + 1]:
            assert ranked[i] < ranked[i + 1]

    assert ranked[0] == opt.select_driver(obs, drivers, route)


def test_random_ranked_actions_is_permutation_from_rng():
    env = make_env(TINY, seed=9)
    opt = RandomDriverOptimizerGym(env)
    obs = env.get_observation()
    drivers = env.get_drivers()
    route = _route_for_current_order(env)
    rng = np.random.default_rng(42)

    ranked = opt.ranked_actions(obs, drivers, route, rng=rng)
    assert sorted(ranked) == list(range(len(drivers)))

    expected = [int(i) for i in np.random.default_rng(42).permutation(len(drivers))]
    assert ranked == expected


def test_random_ranked_actions_requires_rng():
    env = make_env(TINY, seed=9)
    opt = RandomDriverOptimizerGym(env)
    with pytest.raises(ValueError, match="rng"):
        opt.ranked_actions(
            env.get_observation(),
            env.get_drivers(),
            _route_for_current_order(env),
        )
