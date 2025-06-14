__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from gymnasium.envs.registration import register
from importlib.resources import files

def get_scenario_path(filename):
    return str(files("food_delivery_gym.main.scenarios").joinpath(filename))

register(
    id='food_delivery_gym/FoodDelivery-initial-obj1-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("initial_obj1.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-initial-obj2-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("initial_obj2.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-initial-obj3-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("initial_obj3.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-initial-obj4-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("initial_obj4.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-initial-obj7-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("initial_obj7.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-medium-obj1-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("medium_obj1.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-medium-obj2-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("medium_obj2.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-medium-obj3-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("medium_obj3.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-medium-obj4-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("medium_obj4.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-medium-obj7-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("medium_obj7.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-complex-obj1-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("complex_obj1.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-complex-obj2-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("complex_obj2.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-complex-obj3-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("complex_obj2.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-complex-obj4-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("complex_obj4.json"),
    }
)

register(
    id='food_delivery_gym/FoodDelivery-complex-obj7-v0',
    entry_point='food_delivery_gym.main.environment.food_delivery_gym_env:FoodDeliveryGymEnv',
    kwargs={
        "scenario_json_file_path": get_scenario_path("complex_obj7.json"),
    }
)