from importlib.resources import files
import sys
import os

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

SEED = 101010

# Escolha se deseja salvar o log em um arquivo
SAVE_LOG_TO_FILE = False

BASE_RESULTS_DIR = "./data/runs/obj_{}/medium_scenario/"

def setup_logging(results_dir: str):
    """Redireciona stdout/stderr para um arquivo de log."""
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file

def create_environment(reward_objective: int = 1):
    """Cria o ambiente base para ser usado pelos otimizadores."""
    if reward_objective is None:
        raise ValueError("reward_objective não pode ser None. Deve ser um valor entre 1 e 10.")
    elif reward_objective not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        raise ValueError("reward_objective deve ser um valor entre 1 e 10.")

    scenario_path = str(files("food_delivery_gym.main.scenarios").joinpath("medium_obj1.json"))
    gym_env: FoodDeliveryGymEnv = FoodDeliveryGymEnv(scenario_json_file_path=scenario_path)
    gym_env.set_mode(EnvMode.EVALUATING)
    gym_env.set_reward_objective(reward_objective)
    return gym_env

def create_normalized_environment(reward_objective: int = 1):
    """
    Cria ambiente normalizado similar ao usado no treinamento.
    IMPORTANTE: Use exatamente a mesma configuração que foi usada no treinamento!
    """
    base_env = create_environment(reward_objective=reward_objective)
    vec_env = DummyVecEnv([lambda: base_env])
    
    # Configuração similar ao treinamento
    # ATENÇÃO: norm_reward deve ser igual ao usado no treinamento!
    normalized_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
    
    return normalized_env

def load_with_separate_vecnormalize(model_path: str, vecnormalize_path: str = None, reward_objective: int = None):
    """
    Carrega modelo e VecNormalize separadamente.
    Use se você salvou o VecNormalize separadamente durante o treinamento.
    """
    model = PPO.load(model_path)
    env = create_normalized_environment(reward_objective=reward_objective)
    
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        print(f"Carregando VecNormalize de: {vecnormalize_path}")
        env = VecNormalize.load(vecnormalize_path, env)
    else:
        print("AVISO: VecNormalize não encontrado, usando configuração padrão")
        print("Isso pode causar problemas se as estatísticas de normalização forem diferentes!")
        input("    Pressione Enter para continuar...")
    
    return model, env

def main():
    num_runs = 20
    
    base_path = "./data/ppo_training/obj_1/medium_scenario/otimization/normalized/1M5k-timesteps_50-max-trials/18000000_time_steps_best_params/"

    # Caminho para o modelo treinado
    model_path = base_path + "best_model.zip"
    
    # Opcional: caminho para VecNormalize salvo separadamente
    vecnormalize_path = base_path + "food_delivery_gym-FoodDelivery-medium-obj1-v0/vecnormalize.pkl"
    
    # ===== EXECUÇÃO DOS OTIMIZADORES HEURÍSTICOS =====
    print("=== Executando otimizadores heurísticos ===")
    
    for i in range(1, 2):
        results_dir = BASE_RESULTS_DIR.format(i)
        if SAVE_LOG_TO_FILE:
            log_file = setup_logging(results_dir)
        
        print(f"\n=== Iniciando simulações para Objetivo {i} ===")
        
        # # Para heurísticas, use ambiente original
        # base_env = create_environment(reward_objective=i)
        
        # optimizer = RandomDriverOptimizerGym(base_env)
        # optimizer.run_simulations(num_runs, results_dir + "random_heuristic/", seed=SEED)

        # optimizer = FirstDriverOptimizerGym(base_env)
        # optimizer.run_simulations(num_runs, results_dir + "first_driver_heuristic/", seed=SEED)

        # optimizer = NearestDriverOptimizerGym(base_env)
        # optimizer.run_simulations(num_runs, results_dir + "nearest_driver_heuristic/", seed=SEED)

        # if i in [1, 3, 5, 7, 9, 10]:
        #     # Seleciona a função de custo baseada em tempo de entrega
        #     objective_for_cost_function = 1
        # elif i in [2, 4, 6, 8]:
        #     # Seleciona a função de custo baseada em custo de operação (distância)
        #     objective_for_cost_function = 2

        # optimizer = LowestCostDriverOptimizerGym(base_env, cost_function=ObjectiveBasedCostFunction(objective=objective_for_cost_function))
        # optimizer.run_simulations(num_runs, results_dir + "lowest_cost_driver_heuristic/", seed=SEED)

        # # ===== EXECUÇÃO DO OTIMIZADOR RL =====
        # print("=== Executando otimizador RL ===")
        
        try:
            # OPÇÃO 1: Configurar o modelo e o ambiente diretamente:
            model = PPO.load(model_path)
            rl_env = create_normalized_environment(reward_objective=i)
            
            # OPÇÃO 2: Recontruir o modelo com VecNormalize separado --> Gera resultados muito ruins!!!
            # Hipótese: O VecNormalize carregado com o arquivo ".pkl" é específico para aquele cenário e gera complicações
            # model, rl_env = load_with_separate_vecnormalize(model_path, vecnormalize_path, reward_objective=i)
            
            print(f"Configuração do ambiente RL:")
            print(f"  Tipo: {type(rl_env).__name__}")
            print(f"  É vectorizado: {hasattr(rl_env, 'num_envs')}")
            
            # Cria e executa o otimizador RL
            rl_optimizer = RLModelOptimizerGym(rl_env, model)
            rl_optimizer.run_simulations(
                num_runs, 
                results_dir + "ppo_teste_training_false/", 
                seed=SEED
            )
            
        except Exception as e:
            print(f"Erro ao executar otimizador RL: {e}")
            import traceback
            traceback.print_exc()
    
    print("=== Execução concluída ===")

    if SAVE_LOG_TO_FILE:
        log_file.close()

if __name__ == '__main__':
    main()