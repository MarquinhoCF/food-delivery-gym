import copy
import json
from pathlib import Path

import pytest

from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
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
