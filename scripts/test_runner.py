from importlib.resources import files
import argparse
import os
import sys
import textwrap

from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from food_delivery_gym.main.cost.marginal_route_cost_function import MarginalRouteCostFunction
from food_delivery_gym.main.cost.route_cost_function import RouteCostFunction
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv

from food_delivery_gym.main.optimizer.optimizer_gym.first_driver_optimizer_gym import FirstDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.lowest_cost_driver_optimizer_gym import LowestCostDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.random_driver_optimizer_gym import RandomDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import NearestDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.rl_model_optimizer_gym import RLModelOptimizerGym
from food_delivery_gym.main.scenarios import get_all_scenarios
from food_delivery_gym.main.statistic.statistics_view.board import Board

# --- Config padrão ---
DEFAULT_SEED = 5434
DEFAULT_OBJECTIVE = 1
ALL_SCENARIOS = get_all_scenarios()
ALL_OBJECTIVES = FoodDeliveryGymEnv.REWARD_OBJECTIVES

# Carrega variáveis de ambiente do .env
load_dotenv()

DRAW_GRID = os.getenv("DRAW_GRID") == "True"
WINDOW_WIDTH = int(os.getenv("WINDOW_WIDTH"))
WINDOW_HEIGHT = int(os.getenv("WINDOW_HEIGHT"))
FPS = int(os.getenv("FPS"))


def prepare_env(scenario_filename: str, reward_objective: int, seed: int, render: bool) -> FoodDeliveryGymEnv:
    """Prepara e retorna o ambiente configurado."""
    scenario_path = str(files("food_delivery_gym.main.scenarios").joinpath(scenario_filename))

    env = FoodDeliveryGymEnv(scenario_json_file_path=scenario_path, reward_objective=reward_objective, mode=EnvMode.TESTING)

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


def find_vecnormalize(model_dir: str) -> str | None:
    """Procura o vecnormalize.pkl no diretório do modelo.

    O rl_zoo3 salva o arquivo em um subdiretório com o nome do ambiente
    registrado. Como o nome pode variar, a busca é feita recursivamente
    para ser resiliente a variações no nome do subdiretório.
    """
    for root, _dirs, files_found in os.walk(model_dir):
        if "vecnormalize.pkl" in files_found:
            return os.path.join(root, "vecnormalize.pkl")
    return None


def load_rl_model(model_path: str, scenario_filename: str, reward_objective: int, seed: int, render: bool):
    """Carrega o modelo PPO e monta o ambiente adequado.

    Casos tratados:
    - Com vecnormalize.pkl  → VecNormalize carregado do arquivo (ambiente normalizado)
    - Sem vecnormalize.pkl  → DummyVecEnv simples (ambiente sem normalização)
    """
    model_dir = os.path.dirname(model_path)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    print(f"  [PPO] Carregando modelo: {model_path}")
    model = PPO.load(model_path)

    base_env = prepare_env(scenario_filename, reward_objective, seed=seed, render=render)

    # Captura o valor atual de base_env no default do argumento para evitar
    # o bug clássico de closure em loop.
    vec_env = DummyVecEnv([lambda env=base_env: env])

    vecnormalize_path = find_vecnormalize(model_dir)

    if vecnormalize_path:
        print(f"  [VecNormalize] Carregando: {vecnormalize_path}")
        rl_env = VecNormalize.load(vecnormalize_path, vec_env)
        rl_env.training = False
        rl_env.norm_reward = False
    else:
        print("  [VecNormalize] Não encontrado — usando ambiente sem normalização.")
        rl_env = vec_env

    return model, rl_env


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
             - first: escolhe sempre o primeiro motorista
             - nearest: escolhe o motorista mais próximo
             - lowest: escolhe pelo menor custo de rota (requer --cost-function)
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
    parser.add_argument("--cost-function", choices=("route", "marginal_route"), default=None,
                        help="Função de custo usada pelo LowestCostDriverOptimizerGym (apenas quando --optimizer lowest)")
    parser.add_argument("--model-path", default=None,
                        help=(
                            "Caminho para o arquivo best_model.zip do PPO (necessário para --optimizer rl).\n"
                            "O vecnormalize.pkl será procurado automaticamente no mesmo diretório."
                        ))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--objective", type=int, default=1,
                        help="Objetivo de recompensa para o ambiente ({})".format(", ".join(map(str, ALL_OBJECTIVES))))
    parser.add_argument("--render", action="store_true",
                        help="Passar render_mode='human' no reset")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--save-log", action="store_true",
                        help="Redirecionar stdout/stderr para log.txt")

    args = parser.parse_args()

    if args.objective not in ALL_OBJECTIVES:
        parser.error("O argumento --objective deve ser um inteiro entre {} e {}.".format(min(ALL_OBJECTIVES), max(ALL_OBJECTIVES)))

    if args.cost_function and args.optimizer != "lowest":
        parser.error("Erro: --cost-function só pode ser usado com --optimizer lowest")

    if not args.cost_function and args.optimizer == "lowest":
        parser.error("Erro: --optimizer lowest requer --cost-function")

    if args.optimizer == "rl" and not args.model_path:
        parser.error("Erro: --optimizer rl requer --model-path com o caminho para best_model.zip")

    if args.save_log:
        log_file = open("log.txt", "w", encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file
    else:
        log_file = None

    try:
        # Cria o otimizador apropriado
        if args.optimizer == "rl":
            model, rl_env = load_rl_model(
                model_path=args.model_path,
                scenario_filename=args.scenario,
                reward_objective=args.objective,
                seed=args.seed,
                render=args.render,
            )
            optimizer = RLModelOptimizerGym(rl_env, model)
            env = optimizer.gym_env
        else:
            env = prepare_env(args.scenario, args.objective, seed=args.seed, render=args.render)

            if args.optimizer == "random":
                optimizer = RandomDriverOptimizerGym(env)
            elif args.optimizer == "first":
                optimizer = FirstDriverOptimizerGym(env)
            elif args.optimizer == "nearest":
                optimizer = NearestDriverOptimizerGym(env)
            elif args.optimizer == "lowest":
                if args.cost_function == "route":
                    cost_obj = RouteCostFunction.get_cost_objective(args.objective)
                    cost_function = RouteCostFunction(objective=cost_obj)
                elif args.cost_function == "marginal_route":
                    cost_obj = MarginalRouteCostFunction.get_cost_objective(args.objective)
                    cost_function = MarginalRouteCostFunction(objective=cost_obj)
                else:
                    raise ValueError(f"Cost function '{args.cost_function}' inválida")

                optimizer = LowestCostDriverOptimizerGym(env, cost_function=cost_function)
            else:
                raise ValueError(f"Otimizador '{args.optimizer}' não reconhecido")

        print(f"=== Ambiente pronto com otimizador: {optimizer.get_title()} ===")
        print()
        print(env.get_description())
        print()
        print(f"Action space: {env.action_space}")
        print("Iniciando...\n")

        # Executa baseado no modo
        board: Board = None
        if args.mode in ("auto", "agent"):
            if args.mode == "agent" and args.optimizer != "rl":
                print("AVISO: Modo 'agent' funciona melhor com --optimizer rl")
            board = optimizer.run_auto(max_steps=args.max_steps)
        elif args.mode == "interactive":
            board = optimizer.run_interactive(max_steps=args.max_steps)

        # Mostra estatísticas finais
        print("\n== FIM DA EXECUÇÃO ==")
        try:
            env.print_enviroment_state()
            print(f"Observação final: {env.get_observation()}")
            print(f"Quantidade de rotas criadas = {env.simpy_env.state.get_length_orders()}")
            print(f"Quantidade de rotas entregues = {env.simpy_env.state.get_orders_delivered()}")
            if board:
                board.view()
        except Exception as e:
            print(f"Erro ao mostrar estatísticas: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()