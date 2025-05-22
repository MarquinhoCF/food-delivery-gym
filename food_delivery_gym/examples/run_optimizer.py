from importlib.resources import files
import sys

from stable_baselines3 import PPO
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.cost.objective_based_cost_function import ObjectiveBasedCostFunction
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer.optimizer_gym.first_driver_optimizer_gym import FirstDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.lowest_cost_driver_optimizer_gym import LowestCostDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import NearestDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.random_driver_optimizer_gym import RandomDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.rl_model_optimizer_gym import RLModelOptimizerGym

SEED = 101010

# Escolha se deseja salvar o log em um arquivo
SAVE_LOG_TO_FILE = False

RESULTS_DIR = "./data/runs/obj_9/complex_scenario_ant/"

if SAVE_LOG_TO_FILE:
    log_file = open(RESULTS_DIR + "log.txt", "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file

def main():
    scenario_path = str(files("food_delivery_gym.main.scenarios").joinpath("complex_obj2.json"))
    gym_env: FoodDeliveryGymEnv = FoodDeliveryGymEnv(scenario_json_file_path=scenario_path)
    gym_env.set_mode(EnvMode.EVALUATING)
    gym_env.set_reward_objective(9)

    num_runs = 20

    optimizer = RandomDriverOptimizerGym(gym_env)
    optimizer.run_simulations(num_runs, RESULTS_DIR + "random_heuristic/", seed=SEED)

    optimizer = FirstDriverOptimizerGym(gym_env)
    optimizer.run_simulations(num_runs, RESULTS_DIR + "first_driver_heuristic/", seed=SEED)

    optimizer = NearestDriverOptimizerGym(gym_env)
    optimizer.run_simulations(num_runs, RESULTS_DIR + "nearest_driver_heuristic/", seed=SEED)

    optimizer = LowestCostDriverOptimizerGym(gym_env, cost_function=ObjectiveBasedCostFunction(objective=1))
    optimizer.run_simulations(num_runs, RESULTS_DIR + "lowest_cost_driver_heuristic/", seed=SEED)

    optimizer = RLModelOptimizerGym(gym_env, PPO.load("./data/ppo_training/obj_9/complex_scenario_ant/30000000_time_steps/best_model/best_model.zip"))
    optimizer.run_simulations(num_runs, RESULTS_DIR + "ppo_agent_trained_30000000/", seed=SEED)

if __name__ == '__main__':
    main()

if SAVE_LOG_TO_FILE:
    log_file.close()