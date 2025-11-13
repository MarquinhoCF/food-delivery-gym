from importlib.resources import files
import argparse
import os
import sys
import textwrap

from dotenv import load_dotenv
from stable_baselines3 import PPO
from food_delivery_gym.main.cost.objective_based_cost_function import ObjectiveBasedCostFunction
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv

# Imports dos otimizadores
from food_delivery_gym.main.optimizer.optimizer_gym.first_driver_optimizer_gym import FirstDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.lowest_cost_driver_optimizer_gym import LowestCostDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.random_driver_optimizer_gym import RandomDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import NearestDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.rl_model_optimizer_gym import RLModelOptimizerGym

# --- Config padrão ---
DEFAULT_SEED = 5434
DEFAULT_OBJECTIVE = 1

load_dotenv()

DRAW_GRID = os.getenv("DRAW_GRID") == "True"
WINDOW_WIDTH = int(os.getenv("WINDOW_WIDTH"))
WINDOW_HEIGHT = int(os.getenv("WINDOW_HEIGHT"))
FPS = int(os.getenv("FPS"))

def prepare_env(scenario_filename: str, seed: int, render: bool) -> FoodDeliveryGymEnv:
    """Prepara e retorna o ambiente configurado."""
    scenario_path = str(files("food_delivery_gym.main.scenarios").joinpath(scenario_filename))
    env = FoodDeliveryGymEnv(scenario_json_file_path=scenario_path, reward_objective=DEFAULT_OBJECTIVE)
    env.set_mode(EnvMode.TESTING)
    
    # Reset com compatibilidade
    reset_options = {
        "render_mode": "human", 
        "draw_grid": DRAW_GRID, 
        "window_size": (WINDOW_WIDTH, WINDOW_HEIGHT), 
        "fps": FPS
    } if render else None

    if reset_options:
        env.reset(seed=seed, options=reset_options)
    else:
        env.reset(seed=seed)
    
    return env


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Runner para FoodDeliveryGymEnv com otimizadores.

            Modos:
             - auto: executa até o fim automaticamente
             - interactive: passo-a-passo controlado pelo usuário
             - agent: usa um modelo PPO salvo para decidir ações
             
            Otimizadores:
             - random: escolhe motoristas aleatoriamente
             - nearest: escolhe o motorista mais próximo
             - rl: usa um modelo de aprendizado por reforço (requer --model-path)
            """
        ),
    )

    parser.add_argument("--scenario", default="medium.json", 
                       help="Arquivo de cenário dentro de food_delivery_gym.main.scenarios")
    parser.add_argument("--mode", choices=("auto", "interactive", "agent"), default="interactive",
                       help="Modo de execução")
    parser.add_argument("--optimizer", choices=("random", "first", "nearest", "lowest", "rl"), default="random",
                       help="Tipo de otimizador a usar")
    parser.add_argument("--model-path", default=None, 
                       help="Caminho para um modelo PPO (necessário para --optimizer rl)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--objective", type=int, default=1,
                       help="Objetivo de recompensa para o ambiente (1-10)")
    parser.add_argument("--render", action="store_true", 
                       help="Passar render_mode='human' no reset")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--save-log", action="store_true", 
                       help="Redirecionar stdout/stderr para log.txt")

    args = parser.parse_args()

    if args.save_log:
        log_file = open("log.txt", "w", encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file
    else:
        log_file = None

    try:
        # Prepara o ambiente
        env = prepare_env(args.scenario, seed=args.seed, render=args.render)
        
        # Cria o otimizador apropriado
        if args.optimizer == "random":
            optimizer = RandomDriverOptimizerGym(env)
        elif args.optimizer == "first":
            optimizer = FirstDriverOptimizerGym(env)
        elif args.optimizer == "nearest":
            optimizer = NearestDriverOptimizerGym(env)
        elif args.optimizer == "lowest":
            if args.objective in [1, 3, 5, 7, 9, 10]:
                # Seleciona a função de custo baseada em tempo de entrega
                objective_for_cost_function = 1
            elif args.objective in [2, 4, 6, 8]:
                # Seleciona a função de custo baseada em custo de operação (distância)
                objective_for_cost_function = 2
            else:
                raise ValueError(f"Objetivo {args.objective} não reconhecido.")
            optimizer = LowestCostDriverOptimizerGym(env, cost_function=ObjectiveBasedCostFunction(objective=objective_for_cost_function))
        elif args.optimizer == "rl":
            # TODO: Testar pra ver se isso vai funcionar mesmo
            # if not args.model_path:
            #     raise RuntimeError("--optimizer rl requer --model-path com o arquivo do modelo PPO.")
            # model = PPO.load(args.model_path)
            # print(f"Modelo PPO carregado de: {args.model_path}")
            # optimizer = RLModelOptimizerGym(env, model)
            raise NotImplementedError("O otimizador RL ainda não está implementado neste runner.")
        else:
            raise ValueError(f"Otimizador '{args.optimizer}' não reconhecido")
        
        print(f"=== Ambiente pronto com otimizador: {optimizer.get_title()} ===")
        print(f"Action space: {env.action_space}")
        print("Iniciando...\n")
        
        # Executa baseado no modo
        if args.mode == "auto":
            optimizer.run_auto(max_steps=args.max_steps)
        elif args.mode == "interactive":
            optimizer.run_interactive(max_steps=args.max_steps)
        elif args.mode == "agent":
            # No modo agent, usa o otimizador RL
            if args.optimizer != "rl":
                print("AVISO: Modo 'agent' funciona melhor com --optimizer rl")
            optimizer.run_auto(max_steps=args.max_steps)
        
        # Mostra estatísticas finais
        print("\n== FIM DA EXECUÇÃO ==")
        try:
            env.print_enviroment_state()
            print(f"Observação final: {env.get_observation()}")
            print(f"Quantidade de rotas criadas = {env.simpy_env.state.get_length_orders()}")
            print(f"Quantidade de rotas entregues = {env.simpy_env.state.orders_delivered}")
            optimizer.show_statistics_board()
        except Exception as e:
            print(f"Erro ao mostrar estatísticas: {e}")

    except Exception as e:
        print(f"Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()