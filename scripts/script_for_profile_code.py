from importlib.resources import files

from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.optimizer.optimizer_gym.random_driver_optimizer_gym import RandomDriverOptimizerGym

SEED = 101010

RESULTS_DIR = "./data/runs/teste/"

def main():
    scenario_path = str(files("food_delivery_gym.main.scenarios").joinpath("complex.json"))
    gym_env = FoodDeliveryGymEnv.from_file(scenario_path)

    optimizer = RandomDriverOptimizerGym(gym_env)
    optimizer.set_gym_env_mode(EnvMode.TRAINING)
    optimizer.initialize(seed=SEED)

    for i in range(200):
        optimizer.run()
        optimizer.reset_env()


if __name__ == '__main__':
    main()