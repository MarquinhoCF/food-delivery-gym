__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from gymnasium.envs.registration import register
from importlib.resources import files

def get_scenario_path(filename):
    return str(files("food_delivery_gym.main.scenarios").joinpath(filename))

register(
    id='food_delivery_gym/FoodDelivery-initial-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    # kwargs={
    #     "scenario_json_file_path": get_scenario_path("initial.json"),
    # }
)

register(
    id='food_delivery_gym/FoodDelivery-medium-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("medium.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-complex-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("complex.json"),
    }
)