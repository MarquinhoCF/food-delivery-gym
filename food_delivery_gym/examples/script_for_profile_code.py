import sys

from stable_baselines3 import PPO
from food_delivery_gym.main.utils.load_scenarios import load_scenario
from food_delivery_gym.main.cost.objective_based_cost_function import ObjectiveBasedCostFunction
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.optimizer.optimizer_gym.first_driver_optimizer_gym import FirstDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.lowest_cost_driver_optimizer_gym import LowestCostDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import NearestDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.random_driver_optimizer_gym import RandomDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.rl_model_optimizer_gym import RLModelOptimizerGym

SEED = 101010

# Escolha se deseja salvar o log em um arquivo
SAVE_LOG_TO_FILE = False

RESULTS_DIR = "./data/runs/teste/"

if SAVE_LOG_TO_FILE:
    log_file = open(RESULTS_DIR + "log.txt", "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file

def main():
    gym_env: FoodDeliveryGymEnv = load_scenario("complex.json")

    optimizer = RandomDriverOptimizerGym(gym_env)
    optimizer.set_gym_env_mode(EnvMode.TRAINING)
    optimizer.initialize(seed=SEED)

    for i in range(200):
        optimizer.run()
        optimizer.reset_env()



if __name__ == '__main__':
    main()

if SAVE_LOG_TO_FILE:
    log_file.close()