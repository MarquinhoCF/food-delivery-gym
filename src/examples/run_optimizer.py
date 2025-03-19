import sys

from stable_baselines3 import PPO
from src.main.utils.load_scenarios import load_scenario
from src.main.cost.objective_based_cost_function import ObjectiveBasedCostFunction
from src.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from src.main.optimizer.optimizer_gym.first_driver_optimizer_gym import FirstDriverOptimizerGym
from src.main.optimizer.optimizer_gym.lowest_cost_driver_optimizer_gym import LowestCostDriverOptimizerGym
from src.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import NearestDriverOptimizerGym
from src.main.optimizer.optimizer_gym.random_driver_optimizer_gym import RandomDriverOptimizerGym
from src.main.optimizer.optimizer_gym.rl_model_optimizer_gym import RLModelOptimizerGym

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

    num_runs = 10

    # optimizer = RandomDriverOptimizerGym(gym_env)
    # optimizer.run_simulations(num_runs, RESULTS_DIR + "random_heuristic/", seed=SEED)

    optimizer = FirstDriverOptimizerGym(gym_env)
    optimizer.run_simulations(num_runs, RESULTS_DIR + "first_driver_heuristic/", seed=SEED)

    # optimizer = NearestDriverOptimizerGym(gym_env)
    # optimizer.run_simulations(num_runs, RESULTS_DIR + "nearest_driver_heuristic/", seed=SEED)

    # optimizer = LowestCostDriverOptimizerGym(gym_env, cost_function=ObjectiveBasedCostFunction(objective=gym_env.reward_objective))
    # optimizer.run_simulations(num_runs, RESULTS_DIR + "lowest_cost_driver_heuristic/", seed=SEED)

    # optimizer = RLModelOptimizerGym(gym_env, PPO.load("./data/ppo_training/complex_scenario/13000000_total_time_steps/best_model/best_model.zip"))
    # optimizer.run_simulations(num_runs, RESULTS_DIR + "ppo_agent_trained_13000000/", seed=SEED)


if __name__ == '__main__':
    main()

if SAVE_LOG_TO_FILE:
    log_file.close()