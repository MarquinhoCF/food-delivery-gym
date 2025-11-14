from importlib.resources import files
import sys
import os
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.cost.objective_based_cost_function import ObjectiveBasedCostFunction
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer.optimizer_gym.first_driver_optimizer_gym import FirstDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.lowest_cost_driver_optimizer_gym import LowestCostDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import NearestDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.random_driver_optimizer_gym import RandomDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.rl_model_optimizer_gym import RLModelOptimizerGym

# Escolha se deseja salvar o log em um arquivo
SAVE_LOG_TO_FILE = False

SCENARIOS = ["initial", "medium", "complex"]
TIMESTEPS_OPTIONS = ["18M_steps", "50M_steps", "100M_steps"]
MODEL_BASE_DIR = "./data/ppo_training/otimizacao_1M_steps_200_trials/treinamento"

BASE_RESULTS_DIR = "./data/teste/runs/execucoes/obj_{}/{}_scenario/"

def setup_logging(results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file

def create_environment(reward_objective: int, scenario_name: str):
    if reward_objective not in range(1, 11):
        raise ValueError("reward_objective deve ser um valor entre 1 e 10.")

    scenario_file = scenario_name + ".json"
    scenario_path = str(files("food_delivery_gym.main.scenarios").joinpath(scenario_file))
    gym_env = FoodDeliveryGymEnv(scenario_json_file_path=scenario_path)
    gym_env.set_mode(EnvMode.EVALUATING)
    gym_env.set_reward_objective(reward_objective)
    return gym_env

def create_normalized_environment(reward_objective: int = 1):
    base_env = create_environment(reward_objective=reward_objective)
    vec_env = DummyVecEnv([lambda: base_env])
    
    normalized_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    
    return normalized_env

def load_with_separate_vecnormalize(model_path: str, vecnormalize_path: str, reward_objective: int, scenario_name: str):
    model = PPO.load(model_path)

    base_env = create_environment(reward_objective=reward_objective, scenario_name=scenario_name)
    vec_env = DummyVecEnv([lambda: base_env])
    env = VecNormalize.load(vecnormalize_path, vec_env)
    env.training = False
    env.norm_reward = False

    return model, env

def main():
    num_runs = 20
    seed = 123456789

    print("=== Executando Agentes Otimizadores ao Ambiente de Entrega de Última Milha ===")

    for i in range(1, 2):  # objetivos de 1 a 10
        for scenario in SCENARIOS:
            results_dir = BASE_RESULTS_DIR.format(i, scenario)
            if SAVE_LOG_TO_FILE:
                log_file = setup_logging(results_dir)

            print(f"\n\n=== Iniciando simulações para Objetivo {i} no cenário '{scenario}' ===")
            base_env = create_environment(reward_objective=i, scenario_name=scenario)

            print("\n=== Executando Heurísticas ===")

            # # Heurísticas
            # print(f"\n=== Executando simulações com o Agente aleatório no cenário '{scenario}' ===")
            # RandomDriverOptimizerGym(base_env).run_simulations(num_runs, results_dir + "random_heuristic/", seed=seed)

            # print(f"\n=== Executando simulações com o Agente do Primeiro Motorista no cenário '{scenario}' ===")
            # FirstDriverOptimizerGym(base_env).run_simulations(num_runs, results_dir + "first_driver_heuristic/", seed=seed)

            print(f"\n=== Executando simulações com o Agente do Motorista mais Próximo no cenário '{scenario}' ===")
            NearestDriverOptimizerGym(base_env).run_simulations(num_runs, results_dir + "nearest_driver_heuristic/", seed=seed)

            # if i in [1, 3, 5, 7, 9, 10]:
            #     # Seleciona a função de custo baseada em tempo de entrega
            #     objective_for_cost_function = 1
            # elif i in [2, 4, 6, 8]:
            #     # Seleciona a função de custo baseada em custo de operação (distância)
            #     objective_for_cost_function = 2
            # else:
            #     raise ValueError(f"Objetivo {i} não reconhecido.")

            # print(f"\n=== Executando simulações com o Agente do Motorista de Menor Custo no cenário '{scenario}' ===")
            # LowestCostDriverOptimizerGym(base_env, cost_function=ObjectiveBasedCostFunction(objective=objective_for_cost_function))\
            #     .run_simulations(num_runs, results_dir + "lowest_cost_driver_heuristic/", seed=seed)

            # # RL - tenta os 3 time steps possíveis
            # print("\n=== Tentando executar modelos de Aprendizado por Reforço ===")
            # for timestep in TIMESTEPS_OPTIONS:
            #     model_dir = f"{MODEL_BASE_DIR}/obj_{i}/medium/{timestep}/"
            #     model_path = model_dir + "best_model.zip"
            #     vecnormalize_path = model_dir + f"food_delivery_gym-FoodDelivery-medium-obj{i}-v0/vecnormalize.pkl"

            #     if not os.path.exists(model_path):
            #         print(f"\n[AVISO] Modelo não encontrado: {model_path}")
            #         continue
            #     if not os.path.exists(vecnormalize_path):
            #         print(f"\n[AVISO] VecNormalize não encontrado: {vecnormalize_path}")
            #         continue

            #     print(f"\nExecutando PPO para Objetivo {i}, cenário '{scenario}', timestep {timestep}")
            #     try:
            #         # OPÇÃO 1: Recontruir o modelo com VecNormalize separado
            #         model, rl_env = load_with_separate_vecnormalize(model_path, vecnormalize_path, reward_objective=i, scenario_name=scenario)
                    
            #         # # OPÇÃO 2: Configurar o modelo e o ambiente diretamente:
            #         # model = PPO.load(model_path)
            #         # rl_env = create_normalized_environment(reward_objective=i)

            #         rl_optimizer = RLModelOptimizerGym(rl_env, model)
            #         rl_optimizer.run_simulations(
            #             num_runs,
            #             results_dir + f"ppo_otimizado_trained_{timestep}/",
            #             seed=seed
            #         )
            #     except Exception as e:
            #         print(f"Erro ao executar PPO para objetivo {i}, cenário {scenario}, timestep {timestep}: {e}")
            #         traceback.print_exc()

            if SAVE_LOG_TO_FILE:
                log_file.close()

    print("=== Execução concluída ===")

    if SAVE_LOG_TO_FILE:
        log_file.close()

if __name__ == '__main__':
    main()