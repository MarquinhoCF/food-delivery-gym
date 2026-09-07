from pathlib import Path

import numpy as np
import pytest

from food_delivery_gym.main.base.dimensions import Dimensions
from food_delivery_gym.main.customer.customer import Customer
from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.driver.driver_status import DriverStatus
from food_delivery_gym.main.driver.dynamic_route_driver import DynamicRouteDriver
from food_delivery_gym.main.driver.reactive_driver import ReactiveDriver
from food_delivery_gym.main.environment.env_clone import capture_wakes
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.establishment.catalog import Catalog
from food_delivery_gym.main.establishment.establishment import Establishment
from food_delivery_gym.main.establishment.establishment_order_rate import EstablishmentOrderRate
from food_delivery_gym.main.map.grid_map import GridMap
from food_delivery_gym.main.order.item import Item
from food_delivery_gym.main.order.order import Order
from food_delivery_gym.main.route.delivery_route_segment import DeliveryRouteSegment
from food_delivery_gym.main.route.pickup_route_segment import PickupRouteSegment
from food_delivery_gym.main.route.route import Route
from food_delivery_gym.main.utils.rng_factory import RngFactory


FIXTURES = Path(__file__).parent / "fixtures"
TINY = FIXTURES / "tiny_scenario.json"
STRESS = FIXTURES / "stress_scenario.json"


@pytest.fixture(autouse=True)
def _clear_scenario_cache():
    yield
    FoodDeliveryGymEnv.SCENARIO = None


def make_env(scenario_path: Path = TINY, seed: int = 123, reward_objective: int = 3) -> FoodDeliveryGymEnv:
    env = FoodDeliveryGymEnv(scenario_json_file_path=str(scenario_path), reward_objective=reward_objective)
    env.reset(seed=seed)
    return env


def run_episode(env: FoodDeliveryGymEnv, action_fn, max_steps: int = 400):
    """Avança o episódio com action_fn(step_idx, env) até terminated/truncated."""
    transitions = []
    for step_idx in range(max_steps):
        if env.current_order is None:
            break
        action = action_fn(step_idx, env)
        obs, reward, terminated, truncated, info = env.step(action)
        transitions.append((obs, reward, terminated, truncated, info))
        if terminated or truncated:
            break
    else:
        raise AssertionError(f"episódio não terminou em {max_steps} steps")
    return transitions


def bare_simpy_env(seed: int = 0, map_size: int = 20) -> FoodDeliverySimpyEnv:
    rng_factory = RngFactory(seed=seed)
    return FoodDeliverySimpyEnv(
        map=GridMap(map_size, rng=rng_factory.next()),
        generators=[],
        optimizer=None,
        rng_factory=rng_factory,
        initialize=False,
    )


def _default_catalog() -> Catalog:
    dim = Dimensions(1, 1, 1, 1)
    return Catalog([Item(f"item_{i}", dim, 4) for i in range(5)])


def make_driver(
    env: FoodDeliverySimpyEnv | None = None,
    *,
    driver_id: int = 1,
    coordinate=(0, 0),
    available: bool = True,
    movement_rate: float = 5,
    reward_objective: int = 1,
    status: DriverStatus = DriverStatus.AVAILABLE,
) -> Driver:
    env = env or bare_simpy_env()
    return Driver(
        id=driver_id,
        environment=env,
        coordinate=coordinate,
        available=available,
        status=status,
        movement_rate=movement_rate,
        reward_objective=reward_objective,
        start_processes=False,
    )


def make_dynamic_driver(
    env: FoodDeliverySimpyEnv | None = None,
    *,
    driver_id: int = 1,
    coordinate=(0, 0),
    available: bool = True,
    movement_rate: float = 5,
    tolerance_percentage: float = 0.5,
    max_capacity: int = 2,
    reward_objective: int = 1,
) -> DynamicRouteDriver:
    env = env or bare_simpy_env()
    return DynamicRouteDriver(
        id=driver_id,
        environment=env,
        coordinate=coordinate,
        available=available,
        tolerance_percentage=tolerance_percentage,
        max_capacity=max_capacity,
        movement_rate=movement_rate,
        reward_objective=reward_objective,
        start_processes=False,
    )


def make_reactive_driver(
    env: FoodDeliverySimpyEnv | None = None,
    *,
    driver_id: int = 1,
    coordinate=(0, 0),
    available: bool = True,
    max_distance: float = 10,
    movement_rate: float = 5,
) -> ReactiveDriver:
    env = env or bare_simpy_env()
    return ReactiveDriver(
        id=driver_id,
        environment=env,
        coordinate=coordinate,
        available=available,
        max_distance=max_distance,
        movement_rate=movement_rate,
        start_processes=False,
    )


def make_establishment(
    env: FoodDeliverySimpyEnv | None = None,
    *,
    establishment_id: int = 1,
    coordinate=(5, 5),
    available: bool = True,
    production_capacity: int = 2,
    catalog: Catalog | None = None,
    use_estimate: bool = False,
) -> Establishment:
    env = env or bare_simpy_env()
    return Establishment(
        id=establishment_id,
        environment=env,
        coordinate=coordinate,
        available=available,
        catalog=catalog or _default_catalog(),
        production_capacity=production_capacity,
        use_estimate=use_estimate,
        start_processes=False,
    )


def make_establishment_order_rate(
    env: FoodDeliverySimpyEnv | None = None,
    *,
    establishment_id: int = 1,
    coordinate=(5, 5),
    available: bool = True,
    production_capacity: int = 2,
    order_production_time_rate: float = 30,
    operating_radius: int = 10,
    min_prepare_time: float = 20,
    max_prepare_time: float = 60,
    catalog: Catalog | None = None,
) -> EstablishmentOrderRate:
    env = env or bare_simpy_env()
    return EstablishmentOrderRate(
        id=establishment_id,
        environment=env,
        coordinate=coordinate,
        available=available,
        catalog=catalog or _default_catalog(),
        production_capacity=production_capacity,
        order_production_time_rate=order_production_time_rate,
        operating_radius=operating_radius,
        min_prepare_time=min_prepare_time,
        max_prepare_time=max_prepare_time,
        use_estimate=True,
        start_processes=False,
    )


def make_customer(
    env: FoodDeliverySimpyEnv,
    *,
    customer_id: int = 1,
    coordinate=(10, 10),
) -> Customer:
    return Customer(
        id=customer_id,
        environment=env,
        coordinate=coordinate,
        available=True,
        single_order=True,
    )


def make_order(
    env: FoodDeliverySimpyEnv,
    establishment: Establishment,
    customer: Customer | None = None,
    *,
    order_id: int = 1,
    request_date: int = 0,
) -> Order:
    customer = customer or make_customer(env)
    items = list(establishment.catalog.items[:2])
    return Order(
        id=order_id,
        customer=customer,
        establishment=establishment,
        request_date=request_date,
        items=items,
    )


def make_pickup_delivery_route(env: FoodDeliverySimpyEnv, order: Order) -> Route:
    return Route(env, [PickupRouteSegment(order), DeliveryRouteSegment(order)])


def assert_obs_equal(obs_a: dict, obs_b: dict) -> None:
    assert obs_a.keys() == obs_b.keys()
    for key in obs_a:
        np.testing.assert_array_equal(obs_a[key], obs_b[key], err_msg=key)


def _canonicalize(value):
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return tuple(_canonicalize(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_canonicalize(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _rng_fingerprint(rng) -> tuple:
    return _canonicalize(rng.bit_generator.state)


def rng_snapshot(env: FoodDeliveryGymEnv) -> dict:
    simpy = env.simpy_env
    snapshot = {
        "factory": _canonicalize(simpy.rng_factory._seed_sequence.state) if simpy.rng_factory else None,
        "map": _rng_fingerprint(simpy.map.rng),
    }
    for index, generator in enumerate(simpy.generators):
        snapshot[f"generator[{index}]"] = _rng_fingerprint(generator.rng)
    for driver in simpy.state.drivers:
        snapshot[f"driver[{driver.driver_id}]"] = _rng_fingerprint(driver.rng)
    for establishment in simpy.state.establishments:
        snapshot[f"establishment[{establishment.establishment_id}]"] = _rng_fingerprint(establishment.rng)
    for customer in simpy.state.customers:
        snapshot[f"customer[{customer.customer_id}]"] = _rng_fingerprint(customer.rng)
    return snapshot


def wake_snapshot(env: FoodDeliveryGymEnv) -> list[tuple]:
    items = [
        (
            wake.co_name,
            round(wake.remaining, 10),
            tuple(sorted((key, repr(value)) for key, value in wake.extras.items())),
        )
        for wake in capture_wakes(env.simpy_env)
    ]
    return sorted(items)


def structural_snapshot(env: FoodDeliveryGymEnv) -> dict:
    simpy = env.simpy_env
    drivers = [
        (
            d.driver_id,
            d.coordinate,
            d.status,
            d.get_number_of_orders_in_list(),
            len(d.route_requests),
            d.total_distance,
            getattr(d, "current_load", None),
        )
        for d in simpy.state.drivers
    ]
    establishments = [
        (
            e.establishment_id,
            e.coordinate,
            e.orders_in_preparation,
            [cook.get_length_orders_accepted() for cook in e.cooks],
            [None if cook.current_order is None else cook.current_order.order_id for cook in e.cooks],
            tuple(sorted(e._processing_order_ids)),
        )
        for e in simpy.state.establishments
    ]
    orders = [(o.order_id, o.status, o.isReady) for o in simpy.state.orders]
    return {
        "now": simpy.now,
        "orders_generated": env.orders_generated,
        "last_decision_time": env._last_decision_time,
        "current_order_id": None if env.current_order is None else env.current_order.order_id,
        "drivers": drivers,
        "establishments": establishments,
        "orders": orders,
        "core_events": len(simpy.core_events),
        "orders_delivered": simpy.state.orders_delivered,
        "queue_len": len(wake_snapshot(env)),
        "wakes": wake_snapshot(env),
        "rng": rng_snapshot(env),
    }


def assert_envs_consistent(env_a: FoodDeliveryGymEnv, env_b: FoodDeliveryGymEnv, context: str = "") -> None:
    prefix = f"{context}: " if context else ""
    assert env_a.simpy_env is not env_b.simpy_env, f"{prefix}clone deve ser outro Environment"
    assert env_a.simpy_env.rng_factory is not env_b.simpy_env.rng_factory, f"{prefix}RngFactory compartilhada"
    snap_a = structural_snapshot(env_a)
    snap_b = structural_snapshot(env_b)
    if snap_a != snap_b:
        differing = [key for key in snap_a if snap_a.get(key) != snap_b.get(key)]
        details = []
        for key in differing:
            if key == "rng":
                rng_diff = [rk for rk in snap_a["rng"] if snap_a["rng"].get(rk) != snap_b["rng"].get(rk)]
                sample = rng_diff[0] if rng_diff else None
                details.append(
                    f"rng keys={rng_diff} sample {sample}: {snap_a['rng'].get(sample)!r} vs {snap_b['rng'].get(sample)!r}"
                    if sample
                    else f"rng keys={rng_diff}"
                )
            elif key == "wakes":
                only_a = [w for w in snap_a["wakes"] if w not in snap_b["wakes"]]
                only_b = [w for w in snap_b["wakes"] if w not in snap_a["wakes"]]
                details.append(f"wakes only_a={only_a} only_b={only_b}")
            else:
                details.append(f"{key}: {snap_a[key]!r} != {snap_b[key]!r}")
        raise AssertionError(f"{prefix}estado estrutural/RNG/waits diverge: " + "; ".join(details))
