from importlib.resources import files

from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.optimizer.optimizer_gym.random_driver_optimizer_gym import RandomDriverOptimizerGym

SEED = 101010

def main():
    scenario_path = str(files("food_delivery_gym.main.scenarios").joinpath("complex.json"))
 
    # Carrega o cenário no cache de classe antes de instanciar.
    gym_env = FoodDeliveryGymEnv(scenario_json_file_path=scenario_path, reward_objective=1, mode=EnvMode.TRAINING)
 
    optimizer = RandomDriverOptimizerGym(gym_env)
    optimizer.initialize(seed=SEED)

    for i in range(200):
        optimizer.run()
        optimizer.reset_env()


if __name__ == '__main__':
    main()