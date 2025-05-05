from gymnasium.envs.registration import register
from importlib.resources import files
import os

def get_scenario_path(filename):
    return str(files("src.main.scenarios").joinpath(filename))

register(
    id='FoodDelivery-initial-v0',
    entry_point='src.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("initial.json"),
    }
)

register(
    id='FoodDelivery-medium-v0',
    entry_point='src.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("medium.json"),
    }
)

register(
    id='FoodDelivery-complex-v0',
    entry_point='src.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("complex.json"),
    }
)
