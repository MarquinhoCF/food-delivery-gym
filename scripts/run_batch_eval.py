from importlib.resources import files
import sys
import os
import traceback
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.cost.route_cost_function import RouteCostFunction
from food_delivery_gym.main.cost.marginal_route_cost_function import MarginalRouteCostFunction
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer.optimizer_gym.first_driver_optimizer_gym import FirstDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.lowest_cost_driver_optimizer_gym import LowestCostDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import NearestDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.random_driver_optimizer_gym import RandomDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.rl_model_optimizer_gym import RLModelOptimizerGym

ALL_SCENARIOS = ["initial", "medium", "complex"]
ALL_OBJECTIVES = list(range(1, 14))

# Chave = identificador do argumento --heuristics
#   "dir"   = subdiretório de saída (results_dir/<dir>/)
#   "label" = nome legível usado nos logs
ALL_HEURISTICS = {
    "random": {
        "dir":   "random",
        "label": "Agente Aleatório",
    },
    "first_driver": {
        "dir":   "first_driver",
        "label": "Agente do Primeiro Motorista",
    },
    "nearest_driver": {
        "dir":   "nearest_driver",
        "label": "Agente do Motorista mais Próximo",
    },
    "lowest_route_cost": {
        "dir":   "lowest_route_cost",
        "label": "Agente do Motorista de Menor Custo de Rota",
    },
    "lowest_marginal_route_cost": {
        "dir":   "lowest_marginal_route_cost",
        "label": "Agente do Motorista de Menor Custo de Rota Marginal",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avalia agentes otimizadores (heurísticas e PPO) no ambiente de entrega de última milha.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--objectives", "-o",
        nargs="+",
        type=int,
        choices=ALL_OBJECTIVES,
        default=ALL_OBJECTIVES,
        metavar="N",
        help=(
            "Objetivos de recompensa a executar (1-13). Aceita múltiplos valores.\n"
            "Padrão: todos (1 a 13)\n"
            "Exemplo: --objectives 1 3 5"
        ),
    )

    parser.add_argument(
        "--scenarios", "-s",
        nargs="+",
        choices=ALL_SCENARIOS,
        default=ALL_SCENARIOS,
        metavar="SCENARIO",
        help=(
            "Cenários a executar. Aceita múltiplos valores.\n"
            f"Opções: {ALL_SCENARIOS}\n"
            "Padrão: todos\n"
            "Exemplo: --scenarios initial medium"
        ),
    )

    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=None,
        metavar="MODEL_NAME",
        help=(
            "Nomes dos modelos RL a executar (subdiretórios de obj_N/ que contêm best_model.zip).\n"
            "Padrão: descoberta automática a partir de --model-base-dir\n"
            "Exemplo: --models 18M_steps meu_experimento_v2"
        ),
    )

    parser.add_argument(
        "--heuristics",
        nargs="+",
        choices=list(ALL_HEURISTICS),
        default=list(ALL_HEURISTICS),
        metavar="HEURISTIC",
        help=(
            "Heurísticas a executar. Aceita múltiplos valores.\n"
            f"Opções: {list(ALL_HEURISTICS)}\n"
            "Padrão: todas\n"
            "Exemplo: --heuristics random nearest_driver"
        ),
    )

    parser.add_argument(
        "--no-rl",
        action="store_true",
        help="Desativa a execução dos modelos de Aprendizado por Reforço (PPO).",
    )

    parser.add_argument(
        "--no-heuristics",
        action="store_true",
        help="Desativa a execução de todas as heurísticas.",
    )

    parser.add_argument(
        "--num-runs", "-n",
        type=int,
        default=20,
        help="Número de simulações por agente. Padrão: 20.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123456789,
        help="Seed para reprodutibilidade. Padrão: 123456789.",
    )

    parser.add_argument(
        "--model-base-dir",
        type=str,
        default="./data/ppo_training/medium/otimizacao_1M_steps_200_trials/treinamento",
        help="Diretório base dos modelos PPO treinados.",
    )

    parser.add_argument(
        "--results-base-dir",
        type=str,
        default="./data/runs/execucoes/obj_{}/{}_scenario/",
        help=(
            "Diretório base para salvar resultados.\n"
            "Use '{}' como placeholder para objetivo e cenário respectivamente.\n"
            "Padrão: ./data/runs/execucoes/obj_{}/{}_scenario/"
        ),
    )

    parser.add_argument(
        "--save-log",
        action="store_true",
        help="Salva o log de saída em arquivo (log.txt) dentro do diretório de resultados.",
    )

    return parser.parse_args()


def setup_logging(results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file


def create_environment(reward_objective: int, scenario_name: str):
    if reward_objective not in range(1, 14):
        raise ValueError("reward_objective deve ser um valor entre 1 e 13.")

    scenario_file = scenario_name + ".json"
    scenario_path = str(files("food_delivery_gym.main.scenarios").joinpath(scenario_file))
    gym_env = FoodDeliveryGymEnv(scenario_json_file_path=scenario_path)
    gym_env.set_mode(EnvMode.EVALUATING)
    gym_env.set_reward_objective(reward_objective)
    return gym_env


def find_vecnormalize(model_dir: str, objective: int) -> str | None:
    """Procura o vecnormalize.pkl no diretório do modelo.

    O rl_zoo3 salva o arquivo em um subdiretório com o nome do ambiente
    registrado. Como o nome pode variar (ex: diferentes versões ou cenários
    de treino), a busca é feita de forma recursiva para ser resiliente a
    variações no nome do subdiretório.
    """
    for root, _dirs, files_found in os.walk(model_dir):
        if "vecnormalize.pkl" in files_found:
            return os.path.join(root, "vecnormalize.pkl")
    return None


def load_rl_model(model_path: str, model_dir: str, reward_objective: int, scenario_name: str):
    """Carrega o modelo PPO e monta o ambiente adequado.

    Casos tratados:
    - Com vecnormalize.pkl  → VecNormalize carregado do arquivo (ambiente normalizado)
    - Sem vecnormalize.pkl  → DummyVecEnv simples (ambiente sem normalização)
    """
    model = PPO.load(model_path)

    base_env = create_environment(reward_objective=reward_objective, scenario_name=scenario_name)
    vec_env = DummyVecEnv([lambda: base_env])

    vecnormalize_path = find_vecnormalize(model_dir, reward_objective)

    if vecnormalize_path:
        print(f"  [VecNormalize] Carregando: {vecnormalize_path}")
        env = VecNormalize.load(vecnormalize_path, vec_env)
        env.training = False
        env.norm_reward = False
    else:
        print("  [VecNormalize] Não encontrado — usando ambiente sem normalização.")
        env = vec_env

    return model, env


def get_cost_objective(objective: int) -> int:
    if objective in [1, 3, 5, 7, 9, 10, 11, 12, 13]:
        return 1  # baseado em tempo de entrega
    elif objective in [2, 4, 6, 8]:
        return 2  # baseado em custo de operação (distância)
    else:
        raise ValueError(f"Objetivo {objective} não reconhecido.")


def discover_models(model_base_dir: str, objective: int) -> list:
    """Descobre automaticamente subdiretórios que contêm best_model.zip
    dentro de model_base_dir/obj_{objective}/."""
    search_root = os.path.join(model_base_dir, f"obj_{objective}")
    found = []
    if not os.path.isdir(search_root):
        return found
    for entry in sorted(os.scandir(search_root), key=lambda e: e.name):
        if entry.is_dir() and os.path.isfile(os.path.join(entry.path, "best_model.zip")):
            found.append(entry.name)
    return found


def build_heuristic_optimizer(key: str, base_env, cost_obj: int):
    """Instancia o otimizador correspondente à chave da heurística."""
    if key == "random":
        return RandomDriverOptimizerGym(base_env)
    if key == "first_driver":
        return FirstDriverOptimizerGym(base_env)
    if key == "nearest_driver":
        return NearestDriverOptimizerGym(base_env)
    if key == "lowest_route_cost":
        return LowestCostDriverOptimizerGym(base_env, cost_function=RouteCostFunction(objective=cost_obj))
    if key == "lowest_marginal_route_cost":
        return LowestCostDriverOptimizerGym(base_env, cost_function=MarginalRouteCostFunction(objective=cost_obj))
    raise ValueError(f"Heurística desconhecida: '{key}'")


def run_heuristics(base_env, scenario: str, heuristics: list, objective: int, results_dir: str, num_runs: int, seed: int):
    cost_obj = get_cost_objective(objective)

    for key in heuristics:
        meta = ALL_HEURISTICS[key]
        output_dir = os.path.join(results_dir, meta["dir"]) + "/"
        print(f"\n=== Executando simulações com o {meta['label']} no cenário '{scenario}' ===")
        build_heuristic_optimizer(key, base_env, cost_obj).run_simulations(num_runs, output_dir, seed=seed)


def run_rl_models(objective: int, scenario: str, models: list, model_base_dir: str, results_dir: str, num_runs: int, seed: int):
    print("\n=== Tentando executar modelos de Aprendizado por Reforço ===")

    if not models:
        print(f"[AVISO] Nenhum modelo encontrado em '{model_base_dir}/obj_{objective}/'.")
        return

    for model_name in models:
        model_dir = os.path.join(model_base_dir, f"obj_{objective}", model_name)
        model_path = os.path.join(model_dir, "best_model.zip")

        if not os.path.exists(model_path):
            print(f"\n[AVISO] Modelo não encontrado: {model_path}")
            continue

        print(f"\nExecutando PPO — Objetivo {objective}, cenário '{scenario}', modelo '{model_name}'")
        try:
            model, rl_env = load_rl_model(
                model_path, model_dir,
                reward_objective=objective,
                scenario_name=scenario,
            )
            rl_optimizer = RLModelOptimizerGym(rl_env, model)
            rl_optimizer.run_simulations(
                num_runs,
                os.path.join(results_dir, f"ppo_{model_name}") + "/",
                seed=seed,
            )
        except Exception as e:
            print(f"Erro ao executar PPO — objetivo {objective}, cenário '{scenario}', modelo '{model_name}': {e}")
            traceback.print_exc()


def main():
    args = parse_args()

    print("=== Avaliando Agentes no Ambiente de Entrega de Última Milha ===")
    print(f"  Objetivos    : {args.objectives}")
    print(f"  Cenários     : {args.scenarios}")
    print(f"  Modelos RL   : {args.models if args.models else 'descoberta automática'}")
    print(f"  Heurísticas  : {args.heuristics if not args.no_heuristics else 'desativadas'}")
    print(f"  RL (PPO)     : {'desativado' if args.no_rl else 'ativado'}")
    print(f"  Runs         : {args.num_runs} | Seed: {args.seed}")
    print(f"  Model base   : {args.model_base_dir}")
    print(f"  Results base : {args.results_base_dir}")

    for objective in args.objectives:
        for scenario in args.scenarios:
            results_dir = args.results_base_dir.format(objective, scenario)

            log_file = None
            if args.save_log:
                log_file = setup_logging(results_dir)

            print(f"\n\n=== Iniciando avaliações para Objetivo {objective} no cenário '{scenario}' ===")
            base_env = create_environment(reward_objective=objective, scenario_name=scenario)

            if not args.no_heuristics:
                print("\n=== Executando Heurísticas ===")
                run_heuristics(base_env, scenario, args.heuristics, objective, results_dir, args.num_runs, args.seed)

            if not args.no_rl:
                models = args.models if args.models else discover_models(args.model_base_dir, objective)
                run_rl_models(objective, scenario, models, args.model_base_dir, results_dir, args.num_runs, args.seed)

            if log_file:
                log_file.close()

    print("\n=== Avaliação concluída ===")


if __name__ == "__main__":
    main()