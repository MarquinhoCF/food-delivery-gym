from importlib.resources import files
import os
import traceback
import argparse

from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer import catalog as optimizer_catalog
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avalia agentes otimizadores (heurísticas e modelos de RL) no ambiente de entrega de última milha.",
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
        "--agents", "-a",
        nargs="+",
        default=None,
        metavar="AGENT",
        help=(
            "Agentes a executar: heurísticas do catálogo ou modelos descobertos.\n"
            f"Heurísticas: {optimizer_catalog.keys(is_heuristic=True)}\n"
            "Modelos: chaves descobertas em --model-base-dir (ex.: ppo_18M_steps, sac_1M).\n"
            "Padrão: todos os disponíveis para o objetivo e o cenário de treino."
        ),
    )

    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=None,
        metavar="MODEL_NAME",
        help=(
            "Filtra os modelos RL (chave ppo_18M_steps ou pasta 18M_steps).\n"
            "Padrão: todos os descobertos, salvo se --agents já restringir a lista."
        ),
    )

    parser.add_argument(
        "--heuristics",
        nargs="+",
        default=None,
        metavar="HEURISTIC",
        help=(
            "Filtra as heurísticas.\n"
            f"Opções: {optimizer_catalog.keys(is_heuristic=True)}\n"
            "Padrão: todas, salvo se --agents já restringir a lista."
        ),
    )

    parser.add_argument(
        "--lowest-cost-functions",
        nargs="+",
        choices=optimizer_catalog.COST_FUNCTION_CHOICES,
        default=None,
        metavar="COST",
        help=(
            "Cost functions do lowest. Vale para a heurística lowest e para o rollout\n"
            "quando a base é lowest. Não há lista separada para a base.\n"
            f"Opções: {optimizer_catalog.COST_FUNCTION_CHOICES}\n"
            "Padrão: todas."
        ),
    )

    parser.add_argument(
        "--rollout-base-optimizers",
        nargs="+",
        default=None,
        metavar="OPTIMIZER",
        help=(
            "Políticas de base do rollout (apenas se rollout estiver entre os agentes).\n"
            f"Opções: {optimizer_catalog.cli_choices(rollout_base=True)}\n"
            "Padrão: todas."
        ),
    )

    parser.add_argument(
        "--rollout-alpha",
        type=float,
        default=optimizer_catalog.DEFAULT_ROLLOUT_ALPHA,
        help=f"Fator de desconto do rollout. Padrão: {optimizer_catalog.DEFAULT_ROLLOUT_ALPHA}.",
    )

    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=optimizer_catalog.DEFAULT_ROLLOUT_HORIZON,
        help=(
            "Passos de rollout após a ação candidata.\n"
            f"Padrão: {optimizer_catalog.DEFAULT_ROLLOUT_HORIZON}."
        ),
    )

    parser.add_argument(
        "--rollout-record-decisions",
        action="store_true",
        help="Grava o decision_log do rollout (desativado por padrão).",
    )

    parser.add_argument(
        "--no-rl",
        action="store_true",
        help="Desativa a execução dos modelos de Aprendizado por Reforço.",
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
            "Diretório raiz dos modelos treinados.\n"
            f"Estrutura esperada: <model-base-dir>/<scenario>/{DEFAULT_MODEL_SUBDIR}/obj_N/<model_name>/best_model.zip\n"
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
        choices=ALL_SCENARIOS,
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

    return parser, parser.parse_args()

def create_environment(reward_objective: int, scenario_name: str):
    if reward_objective not in range(1, 14):
        raise ValueError("reward_objective deve ser um valor entre 1 e 13.")

    scenario_file = scenario_name + ".json"
    scenario_path = str(files("food_delivery_gym.main.scenarios").joinpath(scenario_file))

    # Atualiza o cache ANTES de instanciar — workers herdarão o dict via fork.
    FoodDeliveryGymEnv.set_scenario(scenario_path)
 
    gym_env = FoodDeliveryGymEnv(reward_objective=reward_objective, mode=EnvMode.EVALUATING)
    return gym_env


def select_agents_for_run(
    available: list,
    agents: list[str] | None,
    heuristics: list[str] | None,
    models: list[str] | None,
    no_heuristics: bool,
    no_rl: bool,
) -> list:
    """Aplica --agents, --heuristics, --models, --no-heuristics e --no-rl sobre a lista unificada."""
    selected = list(available)
    if agents:
        selected = optimizer_catalog.select_agents(available, agents)

    if heuristics:
        wanted = {
            spec.key
            for spec in optimizer_catalog.select_agents(
                [spec for spec in available if spec.is_heuristic],
                heuristics,
            )
        }
        selected = [
            spec for spec in selected
            if not spec.is_heuristic or spec.key in wanted
        ]

    if models:
        wanted_models = {
            spec.key
            for spec in optimizer_catalog.select_agents(
                [spec for spec in available if not spec.is_heuristic],
                models,
            )
        }
        selected = [
            spec for spec in selected
            if spec.is_heuristic or spec.key in wanted_models
        ]

    if no_heuristics:
        selected = [spec for spec in selected if not spec.is_heuristic]
    if no_rl:
        selected = [spec for spec in selected if spec.is_heuristic]
    return selected


def run_agents(
    scenario: str, variants: list, objective: int,
    results_dir: str, num_runs: int, seed: int,
    save_individual_plots: bool, save_mean_plots: bool,
    metrics_fmt: str,
):
    for variant in variants:
        output_dir = os.path.join(results_dir, variant.result_key) + "/"
        print(f"\n=== Executando {variant.spec.label} ({variant.result_key}) no cenário '{scenario}' ===")
        try:
            env = create_environment(reward_objective=objective, scenario_name=scenario)
            optimizer = optimizer_catalog.instantiate(
                variant.spec,
                env,
                objective,
                **variant.extras,
            )
            optimizer.run_simulations(
                num_runs, output_dir, seed=seed,
                save_individual_plots=save_individual_plots,
                save_mean_plots=save_mean_plots,
                metrics_fmt=metrics_fmt,
            )
        except Exception as e:
            print(
                f"Erro ao executar {variant.result_key} — "
                f"objetivo {objective}, cenário '{scenario}': {e}"
            )
            traceback.print_exc()


def main():
    parser, args = parse_args()

    if args.all_plots:
        args.batch_plots = True
        save_individual_plots = True
    else:
        save_individual_plots = False

    save_mean_plots       = args.batch_plots

    print("=== Avaliando Agentes no Ambiente de Entrega de Última Milha ===")
    print(f"  Objetivos    : {args.objectives}")
    print(f"  Cenários     : {args.scenarios}")
    print(f"  Agentes      : {args.agents if args.agents else 'todos os disponíveis'}")
    print(f"  Heurísticas  : {args.heuristics if args.heuristics else 'todas'}"
          f"{' (desativadas)' if args.no_heuristics else ''}")
    print(f"  Modelos RL   : {args.models if args.models else 'descoberta automática'}"
          f"{' (desativados)' if args.no_rl else ''}")
    cost_functions = list(
        args.lowest_cost_functions or optimizer_catalog.COST_FUNCTION_CHOICES
    )
    base_optimizers = list(
        args.rollout_base_optimizers
        or optimizer_catalog.keys(rollout_base=True)
    )
    print(f"  Cost functions: {cost_functions}")
    print(f"  Bases rollout : {base_optimizers}")
    print(f"  Rollout alpha : {args.rollout_alpha}")
    print(f"  Rollout horiz.: {args.rollout_horizon}")
    print(f"  Runs         : {args.num_runs} | Seed: {args.seed}")
    print(f"  Modo experim.: {args.experiment_mode}")
    print(f"  Model base   : {args.model_base_dir}")
    if args.experiment_mode == "cross_scenario":
        print(f"  Train scenario: {args.train_scenario}")
    print(f"  Results base : {args.results_base_dir}")
    print(f"  Plots indiv. : {'desativados' if not save_individual_plots else 'ativados'}")
    print(f"  Plot médias  : {'desativado' if not save_mean_plots else 'ativado'}")
    print(f"  Formato métr.: {args.metrics_fmt}")

    heuristics_filter = set(args.heuristics or [])
    agents_filter = set(args.agents or [])

    def _heuristic_selected(name: str) -> bool:
        return (
            not args.no_heuristics
            and (not agents_filter or name in agents_filter)
            and (not heuristics_filter or name in heuristics_filter)
        )

    rollout_selected = _heuristic_selected("rollout")
    lowest_selected = _heuristic_selected("lowest")
    try:
        bases = optimizer_catalog.normalize_base_optimizers(base_optimizers)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    lowest_base_selected = rollout_selected and "lowest" in bases

    if args.rollout_base_optimizers and not rollout_selected:
        parser.error(
            "Erro: --rollout-base-optimizers só pode ser usado se rollout estiver entre os agentes"
        )
    if args.lowest_cost_functions and not (lowest_selected or lowest_base_selected):
        parser.error(
            "Erro: --lowest-cost-functions só pode ser usado se lowest estiver na seleção "
            "ou se lowest for base do rollout"
        )

    heuristic_names = set(optimizer_catalog.keys(is_heuristic=True)) | set(
        optimizer_catalog.cli_choices(is_heuristic=True)
    )
    may_run_rl = not args.no_rl and (
        not args.agents or any(name not in heuristic_names for name in args.agents)
    )
    if may_run_rl and args.experiment_mode == "cross_scenario":
        expected_dir = os.path.join(
            args.model_base_dir, args.train_scenario, DEFAULT_MODEL_SUBDIR
        )
        if not os.path.isdir(expected_dir):
            parser.error(
                f"--train-scenario '{args.train_scenario}': diretório de modelos não encontrado.\n"
                f"  Esperado: {expected_dir}\n"
                "  Verifique --model-base-dir e --train-scenario."
            )

    for objective in args.objectives:
        for scenario in args.scenarios:
            results_dir = args.results_base_dir.format(objective, scenario)

            print(f"\n\n=== Iniciando avaliações para Objetivo {objective} no cenário '{scenario}' ===")

            if args.experiment_mode == "same_scenario":
                train_scenario = scenario
            else:
                train_scenario = args.train_scenario

            try:
                available = optimizer_catalog.available_agents(
                    args.model_base_dir,
                    train_scenario,
                    objective,
                )
                agents = select_agents_for_run(
                    available,
                    args.agents,
                    args.heuristics,
                    args.models,
                    args.no_heuristics,
                    args.no_rl,
                )
            except KeyError as exc:
                parser.error(str(exc))

            if not agents:
                print("[AVISO] Nenhum agente selecionado para este objetivo.")
                continue

            try:
                variants = optimizer_catalog.expand_evaluations(
                    agents,
                    cost_functions=cost_functions,
                    base_optimizers=bases,
                    rollout_alpha=args.rollout_alpha,
                    rollout_horizon=args.rollout_horizon,
                    record_decisions=args.rollout_record_decisions,
                )
            except (KeyError, ValueError) as exc:
                parser.error(str(exc))

            print(f"  Execuções neste objetivo: {[variant.result_key for variant in variants]}")
            run_agents(
                scenario, variants, objective, results_dir,
                args.num_runs, args.seed,
                save_individual_plots=save_individual_plots,
                save_mean_plots=save_mean_plots,
                metrics_fmt=args.metrics_fmt,
            )

    print("\n=== Avaliação concluída ===")


if __name__ == "__main__":
    main()