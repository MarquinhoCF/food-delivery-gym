import json
from src.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv

DIR_PATH = "./scenarios"

def load_scenario(file_name: str = "scenario.json", dir_path: str = DIR_PATH) -> FoodDeliveryGymEnv:
    file_path = dir_path + "/" + file_name
    with open(file_path, "r", encoding="utf-8") as f:
        scenario = json.load(f)

    scenario["function"] = eval(scenario["function_code"])  # Converte a função salva como string para lambda

    gym_env = FoodDeliveryGymEnv(
        num_drivers=scenario["num_drivers"],
        num_establishments=scenario["num_establishments"],
        num_orders=scenario["num_orders"],
        num_costumers=scenario["num_costumers"],
        grid_map_size=scenario["grid_map_size"],
        vel_drivers=scenario["vel_drivers"],
        prepare_time=scenario["prepare_time"],
        operating_radius=scenario["operating_radius"],
        production_capacity=scenario["production_capacity"],
        percentage_allocation_driver=scenario["percentage_allocation_driver"],
        use_estimate=scenario["use_estimate"],
        desconsider_capacity=scenario["desconsider_capacity"],
        max_time_step=scenario["max_time_step"],
        reward_objective=scenario["reward_objective"],
        function=scenario["function"],
        lambda_code=scenario["function_code"],
        time_shift=scenario["time_shift"],
        normalize=scenario["normalize"],
    )

    return gym_env
