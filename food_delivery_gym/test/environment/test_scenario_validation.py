import copy
import json
from pathlib import Path

import pytest

from food_delivery_gym.main.driver.dynamic_route_driver import DynamicRouteDriver
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.establishment.establishment_order_rate import EstablishmentOrderRate
from food_delivery_gym.main.scenarios.spec import parse_scenario
from food_delivery_gym.test.conftest import TINY


def _write_scenario(tmp_path: Path, scenario: dict) -> Path:
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")
    return path


def _base_scenario() -> dict:
    return json.loads(TINY.read_text(encoding="utf-8"))


def test_set_scenario_missing_file():
    with pytest.raises(FileNotFoundError):
        FoodDeliveryGymEnv.set_scenario("/nonexistent/path/scenario.json")


def test_constructor_without_path_or_cache():
    FoodDeliveryGymEnv.SCENARIO = None
    with pytest.raises(ValueError, match="cenário"):
        FoodDeliveryGymEnv(scenario_json_file_path="", reward_objective=1)


@pytest.mark.parametrize(
    "section",
    ["order_generator", "simpy_env", "grid_map", "drivers", "establishments"],
)
def test_missing_required_section(tmp_path, section):
    scenario = _base_scenario()
    del scenario[section]
    path = _write_scenario(tmp_path, scenario)
    FoodDeliveryGymEnv.SCENARIO = None
    with pytest.raises(ValueError, match=section):
        FoodDeliveryGymEnv(scenario_json_file_path=str(path))


def test_invalid_order_generator_type(tmp_path):
    scenario = _base_scenario()
    scenario["order_generator"]["type"] = "gaussian"
    path = _write_scenario(tmp_path, scenario)
    FoodDeliveryGymEnv.SCENARIO = None
    with pytest.raises(ValueError, match="poisson"):
        FoodDeliveryGymEnv(scenario_json_file_path=str(path))


def test_non_positive_estimated_num_orders(tmp_path):
    scenario = _base_scenario()
    scenario["order_generator"]["estimated_num_orders"] = 0
    path = _write_scenario(tmp_path, scenario)
    FoodDeliveryGymEnv.SCENARIO = None
    with pytest.raises(ValueError, match="estimated_num_orders"):
        FoodDeliveryGymEnv(scenario_json_file_path=str(path))


def test_non_positive_max_time_step(tmp_path):
    scenario = _base_scenario()
    scenario["simpy_env"]["max_time_step"] = 0
    path = _write_scenario(tmp_path, scenario)
    FoodDeliveryGymEnv.SCENARIO = None
    with pytest.raises(ValueError, match="max_time_step"):
        FoodDeliveryGymEnv(scenario_json_file_path=str(path))


def test_percentage_allocation_driver_out_of_range(tmp_path):
    scenario = _base_scenario()
    scenario["establishments"]["percentage_allocation_driver"] = 1.5
    path = _write_scenario(tmp_path, scenario)
    FoodDeliveryGymEnv.SCENARIO = None
    with pytest.raises(ValueError, match="percentage_allocation_driver"):
        FoodDeliveryGymEnv(scenario_json_file_path=str(path))


def test_valid_tiny_scenario_loads(tmp_path):
    path = _write_scenario(tmp_path, copy.deepcopy(_base_scenario()))
    FoodDeliveryGymEnv.SCENARIO = None
    env = FoodDeliveryGymEnv(scenario_json_file_path=str(path), reward_objective=1)
    assert env.num_drivers == 2


def test_json_requires_driver_and_establishment_type():
    scenario = _base_scenario()
    del scenario["drivers"]["type"]
    with pytest.raises(ValueError, match="drivers.*type"):
        parse_scenario(scenario)

    scenario = _base_scenario()
    del scenario["establishments"]["type"]
    with pytest.raises(ValueError, match="establishments.*type"):
        parse_scenario(scenario)


def test_unknown_driver_type_rejected():
    scenario = _base_scenario()
    scenario["drivers"]["type"] = "teleporter"
    with pytest.raises(ValueError, match="drivers.type"):
        parse_scenario(scenario)


def test_driver_type_without_max_capacity_is_valid():
    scenario = _base_scenario()
    scenario["drivers"] = {"type": "driver", "num": 2, "vel": [3, 4]}
    spec = parse_scenario(scenario)
    assert spec.driver_type == "driver"
    assert spec.max_capacity is None
    assert spec.tolerance_percentage is None


def test_dynamic_route_without_max_capacity_rejected():
    scenario = _base_scenario()
    del scenario["drivers"]["max_capacity"]
    with pytest.raises(ValueError, match="max_capacity"):
        parse_scenario(scenario)


def test_factory_creates_types_from_json(tmp_path):
    path = _write_scenario(tmp_path, copy.deepcopy(_base_scenario()))
    FoodDeliveryGymEnv.SCENARIO = None
    env = FoodDeliveryGymEnv(scenario_json_file_path=str(path), reward_objective=1)
    env.reset(seed=0)
    assert env.scenario_spec.driver_type == "dynamic_route"
    assert env.scenario_spec.establishment_type == "order_rate"
    assert all(isinstance(d, DynamicRouteDriver) for d in env.simpy_env.state.drivers)
    assert all(isinstance(e, EstablishmentOrderRate) for e in env.simpy_env.state.establishments)


def test_gym_orders_have_no_required_capacity_and_items_without_dimensions(tmp_path):
    path = _write_scenario(tmp_path, copy.deepcopy(_base_scenario()))
    FoodDeliveryGymEnv.SCENARIO = None
    env = FoodDeliveryGymEnv(scenario_json_file_path=str(path), reward_objective=1)
    env.reset(seed=0)
    assert env.current_order is not None
    order = env.current_order
    assert not hasattr(order, "required_capacity")
    assert len(order.items) == 1
    for item in order.items:
        assert item.dimensions is None
        assert item.preparation_time is None
