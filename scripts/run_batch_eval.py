from importlib.resources import files
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
from food_delivery_gym.main.scenarios import get_all_scenarios, get_defaults_scenarios

ALL_SCENARIOS = get_all_scenarios()
DEFAULT_SCENARIOS = get_defaults_scenarios()
ALL_OBJECTIVES = FoodDeliveryGymEnv.REWARD_OBJECTIVES
DEFAULT_NUM_RUNS = 20
DEFAULT_SEED = 123456789
DEFAULT_MODEL_BASE_DIR = "./data/ppo_training/"
DEFAULT_MODEL_SUBDIR = "treinamento"
EXPERIMENT_MODES = ["cross_scenario", "same_scenario"]
DEFAULT_EXPERIMENT_MODE = "cross_scenario"
DEFAULT_TRAIN_SCENARIO = "medium"
DEFAULT_RESULTS_BASE_DIR = "./data/runs/execucoes/obj_{}/{}_scenario/"
METRICS_FMT_OPTIONS = ["npz", "json"]
DEFAULT_METRICS_FMT = "npz"

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
        default=DEFAULT_SCENARIOS,
        metavar="SCENARIO",
        help=(
            "Cenários a executar. Aceita múltiplos valores.\n"
            f"Opções: {ALL_SCENARIOS}\n"
            f"Padrão: {DEFAULT_SCENARIOS}\n"
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
        default=DEFAULT_NUM_RUNS,
        help=f"Número de simulações por agente. Padrão: {DEFAULT_NUM_RUNS}.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed para reprodutibilidade. Padrão: {DEFAULT_SEED}.",
    )

    parser.add_argument(
        "--model-base-dir",
        type=str,
        default=DEFAULT_MODEL_BASE_DIR,
        help=(
            "Diretório raiz dos modelos PPO treinados.\n"
            f"Estrutura esperada: <model-base-dir>/<scenario>/{DEFAULT_RESULTS_BASE_DIR}/obj_N/<model_name>/best_model.zip\n"
            f"Padrão: {DEFAULT_MODEL_BASE_DIR}"
        ),
    )

    parser.add_argument(
        "--experiment-mode",
        choices=EXPERIMENT_MODES,
        default=DEFAULT_EXPERIMENT_MODE,
        help=(
            "Modo de seleção dos modelos RL:\n"
            "  cross_scenario  – usa modelos treinados em --train-scenario para avaliar\n"
            "                    em todos os cenários selecionados (comportamento anterior).\n"
            "  same_scenario   – usa o modelo treinado no próprio cenário de avaliação.\n"
            f"Padrão: {DEFAULT_EXPERIMENT_MODE}"
        ),
    )

    parser.add_argument(
        "--train-scenario",
        type=str,
        default=DEFAULT_TRAIN_SCENARIO,
        help=(
            "Cenário cujos modelos serão usados no modo cross_scenario.\n"
            "Ignorado no modo same_scenario.\n"
            f"Padrão: {DEFAULT_TRAIN_SCENARIO}"
        ),
    )

    parser.add_argument(
        "--results-base-dir",
        type=str,
        default=DEFAULT_RESULTS_BASE_DIR,
        help=(
            "Diretório base para salvar resultados.\n"
            "Use '{}' como placeholder para objetivo e cenário respectivamente.\n"
            f"Padrão: {DEFAULT_RESULTS_BASE_DIR}"
        ),
    )

    parser.add_argument(
        "--metrics-fmt",
        choices=METRICS_FMT_OPTIONS,
        default=DEFAULT_METRICS_FMT,
        help=(
            "Formato do arquivo de métricas gerado por simulação.\n"
            "  npz  – comprimido, menor tamanho em disco (padrão)\n"
            "  json – legível, estrutura orientada por simulação\n"
            f"Padrão: {DEFAULT_METRICS_FMT}"
        ),
    )

    parser.add_argument(
        "--batch-plots",
        action="store_true",
        help="Ativa a geração de gráficos agregados.",
    )

    parser.add_argument(
        "--all-plots",
        action="store_true",
        help="Ativa todos os gráficos (equivale a --batch-plots e ativa a geração de gráficos por episódio).",
    )

    return parser.parse_args()

def create_environment(reward_objective: int, scenario_name: str):
    if reward_objective not in range(1, 14):
        raise ValueError("reward_objective deve ser um valor entre 1 e 13.")

    scenario_file = scenario_name + ".json"
    scenario_path = str(files("food_delivery_gym.main.scenarios").joinpath(scenario_file))

    # Atualiza o cache ANTES de instanciar — workers herdarão o dict via fork.
    FoodDeliveryGymEnv.set_scenario(scenario_path)
 
    gym_env = FoodDeliveryGymEnv(reward_objective=reward_objective, mode=EnvMode.EVALUATING)
    return gym_env


def find_vecnormalize(model_dir: str) -> str | None:
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
    
    # Captura o valor atual de base_env no default do argumento para evitar o bug de closure em loop.
    vec_env = DummyVecEnv([lambda env=base_env: env])

    vecnormalize_path = find_vecnormalize(model_dir)

    if vecnormalize_path:
        print(f"  [VecNormalize] Carregando: {vecnormalize_path}")
        env = VecNormalize.load(vecnormalize_path, vec_env)
        env.training = False
        env.norm_reward = False
    else:
        print("  [VecNormalize] Não encontrado — usando ambiente sem normalização.")
        env = vec_env

    return model, env


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


def build_heuristic_optimizer(key: str, base_env, objective):
    """Instancia o otimizador correspondente à chave da heurística."""
    if key == "random":
        return RandomDriverOptimizerGym(base_env)
    if key == "first_driver":
        return FirstDriverOptimizerGym(base_env)
    if key == "nearest_driver":
        return NearestDriverOptimizerGym(base_env)
    if key == "lowest_route_cost":
        cost_obj = RouteCostFunction.get_cost_objective(objective)
        return LowestCostDriverOptimizerGym(base_env, cost_function=RouteCostFunction(objective=cost_obj))
    if key == "lowest_marginal_route_cost":
        cost_obj = MarginalRouteCostFunction.get_cost_objective(objective)
        return LowestCostDriverOptimizerGym(base_env, cost_function=MarginalRouteCostFunction(objective=cost_obj))
    raise ValueError(f"Heurística desconhecida: '{key}'")


def run_heuristics(
    base_env, scenario: str, heuristics: list, objective: int,
    results_dir: str, num_runs: int, seed: int,
    save_individual_plots: bool, save_mean_plots: bool,
    metrics_fmt: str,
):
    for key in heuristics:
        meta = ALL_HEURISTICS[key]
        output_dir = os.path.join(results_dir, meta["dir"]) + "/"
        print(f"\n=== Executando simulações com o {meta['label']} no cenário '{scenario}' ===")
        build_heuristic_optimizer(key, base_env, objective).run_simulations(
            num_runs, output_dir, seed=seed,
            save_individual_plots=save_individual_plots,
            save_mean_plots=save_mean_plots,
            metrics_fmt=metrics_fmt,
        )


def run_rl_models(
    objective: int, scenario: str, models: list, model_base_dir: str,
    results_dir: str, num_runs: int, seed: int,
    save_individual_plots: bool, save_mean_plots: bool,
    metrics_fmt: str,
):
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
                save_individual_plots=save_individual_plots,
                save_mean_plots=save_mean_plots,
                metrics_fmt=metrics_fmt,
            )
        except Exception as e:
            print(f"Erro ao executar PPO — objetivo {objective}, cenário '{scenario}', modelo '{model_name}': {e}")
            traceback.print_exc()


def main():
    args = parse_args()

    if args.all_plots:
        args.batch_plots = True
        save_individual_plots = True
    else:
        save_individual_plots = False

    save_mean_plots       = args.batch_plots

    print("=== Avaliando Agentes no Ambiente de Entrega de Última Milha ===")
    print(f"  Objetivos    : {args.objectives}")
    print(f"  Cenários     : {args.scenarios}")
    print(f"  Modelos RL   : {args.models if args.models else 'descoberta automática'}")
    print(f"  Heurísticas  : {args.heuristics if not args.no_heuristics else 'desativadas'}")
    print(f"  RL (PPO)     : {'desativado' if args.no_rl else 'ativado'}")
    print(f"  Runs         : {args.num_runs} | Seed: {args.seed}")
    print(f"  Modo experim.: {args.experiment_mode}")
    print(f"  Model base   : {args.model_base_dir}")
    if args.experiment_mode == "cross_scenario":
        print(f"  Train scenario: {args.train_scenario}")
    print(f"  Results base : {args.results_base_dir}")
    print(f"  Plots indiv. : {'desativados' if not save_individual_plots else 'ativados'}")
    print(f"  Plot médias  : {'desativado' if not save_mean_plots else 'ativado'}")
    print(f"  Formato métr.: {args.metrics_fmt}")

    for objective in args.objectives:
        for scenario in args.scenarios:
            results_dir = args.results_base_dir.format(objective, scenario)

            print(f"\n\n=== Iniciando avaliações para Objetivo {objective} no cenário '{scenario}' ===")
            base_env = create_environment(reward_objective=objective, scenario_name=scenario)

            if not args.no_heuristics:
                print("\n=== Executando Heurísticas ===")
                run_heuristics(
                    base_env, scenario, args.heuristics, objective, results_dir,
                    args.num_runs, args.seed,
                    save_individual_plots=save_individual_plots,
                    save_mean_plots=save_mean_plots,
                    metrics_fmt=args.metrics_fmt,
                )

            if not args.no_rl:
                # Resolve o diretório de modelos de acordo com o modo de experimento.
                if args.experiment_mode == "same_scenario":
                    train_scenario = scenario          # modelo treinado no próprio cenário
                else:                                  # cross_scenario
                    train_scenario = args.train_scenario

                effective_model_dir = os.path.join(
                    args.model_base_dir, train_scenario, DEFAULT_MODEL_SUBDIR
                )

                models = args.models if args.models else discover_models(effective_model_dir, objective)
                run_rl_models(
                    objective, scenario, models, effective_model_dir, results_dir,
                    args.num_runs, args.seed,
                    save_individual_plots=save_individual_plots,
                    save_mean_plots=save_mean_plots,
                    metrics_fmt=args.metrics_fmt,
                )

    print("\n=== Avaliação concluída ===")


if __name__ == "__main__":
    main()