__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from gymnasium.envs.registration import register
from importlib.resources import files
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.scenarios import get_all_scenarios


def get_scenario_path(filename: str) -> str:
    return str(files("food_delivery_gym.main.scenarios").joinpath(filename))


def _make_env(scenario_json_file_path: str, reward_objective: int) -> FoodDeliveryGymEnv:
    return FoodDeliveryGymEnv(scenario_json_file_path=scenario_json_file_path, reward_objective=reward_objective)


SCENARIOS = get_all_scenarios()
OBJECTIVES = FoodDeliveryGymEnv.REWARD_OBJECTIVES

for _scenario in SCENARIOS:
    for _obj in OBJECTIVES:
        register(
            id=f"FoodDelivery-{_scenario}-obj{_obj}-v1",
            entry_point="food_delivery_gym:_make_env",
            kwargs={
                "scenario_json_file_path": get_scenario_path(f"{_scenario}.json"),
                "reward_objective": _obj,
            },
        )