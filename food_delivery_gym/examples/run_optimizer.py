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

RESULTS_DIR = "./data/runs/obj_1/medium_scenario/"

if SAVE_LOG_TO_FILE:
    log_file = open(RESULTS_DIR + "log.txt", "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file

def create_environment():
    """Cria o ambiente base para ser usado pelos otimizadores."""
    scenario_path = str(files("food_delivery_gym.main.scenarios").joinpath("medium_obj1.json"))
    gym_env: FoodDeliveryGymEnv = FoodDeliveryGymEnv(scenario_json_file_path=scenario_path)
    gym_env.set_mode(EnvMode.EVALUATING)
    gym_env.set_reward_objective(1)
    return gym_env

def create_normalized_environment():
    """
    Cria ambiente normalizado similar ao usado no treinamento.
    IMPORTANTE: Use exatamente a mesma configuração que foi usada no treinamento!
    """
    base_env = create_environment()
    vec_env = DummyVecEnv([lambda: base_env])
    
    # Configuração similar ao treinamento
    # ATENÇÃO: norm_reward deve ser igual ao usado no treinamento!
    normalized_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    
    return normalized_env

def load_model_with_environment(model_path: str):
    """
    Carrega o modelo e tenta usar o ambiente que vem com ele.
    Esta é a abordagem mais segura.
    """
    print(f"Carregando modelo de: {model_path}")
    
    # Carrega o modelo
    model = PPO.load(model_path)
    
    # Tenta obter o ambiente do modelo
    model_env = model.get_env()
    
    if model_env is not None:
        print(f"Usando ambiente do modelo: {type(model_env).__name__}")
        return model, model_env
    else:
        print("Modelo não tem ambiente associado, criando ambiente normalizado...")
        env = create_normalized_environment()
        return model, env

def load_with_separate_vecnormalize(model_path: str, vecnormalize_path: str = None):
    """
    Carrega modelo e VecNormalize separadamente.
    Use se você salvou o VecNormalize separadamente durante o treinamento.
    """
    model = PPO.load(model_path)
    env = create_normalized_environment()
    
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        print(f"Carregando VecNormalize de: {vecnormalize_path}")
        env = VecNormalize.load(vecnormalize_path, env)
    else:
        print("AVISO: VecNormalize não encontrado, usando configuração padrão")
        print("Isso pode causar problemas se as estatísticas de normalização forem diferentes!")
    
    return model, env

def main():
    num_runs = 20
    
    # Caminho para o modelo treinado
    model_path = "./data/ppo_training/obj_1/medium_scenario/otimization/unnormalized/1M-timesteps_50-max-trials/10000000_time_steps_best_params/best_model.zip"
    
    # Opcional: caminho para VecNormalize salvo separadamente
    # vecnormalize_path = "./data/ppo_training/obj_1/medium_scenario/otimization/unnormalized/1M-timesteps_50-max-trials/10000000_time_steps_best_params/food_delivery_gym-FoodDelivery-medium-obj1-v0/vecnormalize.pkl"
    
    # ===== EXECUÇÃO DOS OTIMIZADORES HEURÍSTICOS =====
    print("=== Executando otimizadores heurísticos ===")
    
    # Para heurísticas, use ambiente original
    base_env = create_environment()
    
    # optimizer = RandomDriverOptimizerGym(base_env)
    # optimizer.run_simulations(num_runs, RESULTS_DIR + "random_heuristic/", seed=SEED)

    # optimizer = FirstDriverOptimizerGym(base_env)
    # optimizer.run_simulations(num_runs, RESULTS_DIR + "first_driver_heuristic/", seed=SEED)

    # optimizer = NearestDriverOptimizerGym(base_env)
    # optimizer.run_simulations(num_runs, RESULTS_DIR + "nearest_driver_heuristic/", seed=SEED)

    # optimizer = LowestCostDriverOptimizerGym(base_env, cost_function=ObjectiveBasedCostFunction(objective=1))
    # optimizer.run_simulations(num_runs, RESULTS_DIR + "lowest_cost_driver_heuristic/", seed=SEED)

    # ===== EXECUÇÃO DO OTIMIZADOR RL =====
    print("=== Executando otimizador RL ===")
    
    try:
        # OPÇÃO 1: Usar ambiente do modelo
        model, rl_env = load_model_with_environment(model_path)
        
        # OPÇÃO 2: Recontruir o modelo com VecNormalize separado
        # model, rl_env = load_with_separate_vecnormalize(model_path, vecnormalize_path)
        
        # OPÇÃO 3: Configurar o modelo e o ambiente diretamente:
        # model = PPO.load(model_path)
        # rl_env = create_normalized_environment()
        
        print(f"Configuração do ambiente RL:")
        print(f"  Tipo: {type(rl_env).__name__}")
        print(f"  É vectorizado: {hasattr(rl_env, 'num_envs')}")
        
        # Cria e executa o otimizador RL
        rl_optimizer = RLModelOptimizerGym(rl_env, model)
        rl_optimizer.run_simulations(
            num_runs, 
            RESULTS_DIR + "ppo_agent_trained_18000000_best_params_normalized/", 
            seed=SEED
        )
        
    except Exception as e:
        print(f"Erro ao executar otimizador RL: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== Execução concluída ===")

if __name__ == '__main__':
    main()

if SAVE_LOG_TO_FILE:
    log_file.close()