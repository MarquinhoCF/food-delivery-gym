from importlib.resources import files
import argparse
import os
import sys
import textwrap

from dotenv import load_dotenv
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer import catalog as optimizer_catalog
from food_delivery_gym.main.scenarios import get_all_scenarios
from food_delivery_gym.main.statistics.boards.board import Board

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


def _scenario_name(scenario_filename: str) -> str:
    return scenario_filename[:-5] if scenario_filename.endswith(".json") else scenario_filename


def _is_catalog_optimizer(name: str) -> bool:
    try:
        optimizer_catalog.requires(name)
        return True
    except KeyError:
        return False


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Runner para FoodDeliveryGymEnv com otimizadores.

            Modos:
             - auto: executa até o fim automaticamente
             - interactive: passo-a-passo controlado pelo usuário
             - agent: usa um modelo salvo para decidir ações

            Otimizadores (cadastrados em optimizer/catalog.py):
            {optimizer_lines}

            Modelos RL descobertos em {model_root}/<cenário>/treinamento/obj_N/
            também são aceitos em --optimizer (ex.: ppo_18M_steps, sac_1M).
            --optimizer rl ainda aceita --model-path com um zip avulso.
            """
        ).format(optimizer_lines="\n".join(
            f" - {name}: {optimizer_catalog.cli_label(name)}"
            + (
                " (requer --lowest cost=...)"
                if optimizer_catalog.resolve_key(name) == "lowest"
                else " (requer --rollout base=...,...)"
                if optimizer_catalog.resolve_key(name) == "rollout"
                else " (requer --mcts base=...,...)"
                if optimizer_catalog.resolve_key(name) == "mcts"
                else " (aceita --model-path)"
                if "model" in optimizer_catalog.requires(name)
                else ""
            )
            for name in optimizer_catalog.cli_choices()
        ), model_root=optimizer_catalog.DEFAULT_MODEL_ROOT),
    )

    parser.add_argument("--scenario", default="medium.json",
                        help="Arquivo de cenário dentro de food_delivery_gym.main.scenarios")
    parser.add_argument("--mode", choices=("auto", "interactive", "agent"), default="interactive",
                        help="Modo de execução")
    parser.add_argument("--optimizer", default="random",
                        help="Otimizador do catálogo ou chave de modelo descoberta (ex.: ppo_18M_steps)")
    parser.add_argument(
        "--lowest",
        default=None,
        metavar="SPEC",
        help=(
            "Variante de lowest (mesmo formato do run_batch_eval). "
            f"Formato: cost=route. Opções de cost: {optimizer_catalog.COST_FUNCTION_CHOICES}. "
            "Ex.: --lowest cost=route"
        ),
    )
    parser.add_argument(
        "--rollout",
        default=None,
        metavar="SPEC",
        help=(
            "Variante de rollout (mesmo formato do run_batch_eval). "
            "Formato: base=...,cost=...,horizon=...,alpha=...,terminal=0|model. "
            f"Defaults: base={optimizer_catalog.DEFAULT_ROLLOUT_BASE}, "
            f"horizon={optimizer_catalog.DEFAULT_ROLLOUT_HORIZON}, "
            f"alpha={optimizer_catalog.DEFAULT_ROLLOUT_ALPHA}, "
            f"terminal={optimizer_catalog.DEFAULT_ROLLOUT_TERMINAL}. "
            "Ex.: --rollout base=lowest,cost=route,horizon=5,terminal=0"
        ),
    )
    parser.add_argument(
        "--mcts",
        default=None,
        metavar="SPEC",
        help=(
            "Variante de MCTS (mesmo formato do run_batch_eval). "
            "Formato: base=...,cost=...,horizon=...,alpha=...,terminal=0|model,"
            "iterations=...,exploration_weight=...,depth=...,"
            "max_outcomes=...,max_expanded_actions=.... "
            f"Defaults: base={optimizer_catalog.DEFAULT_ROLLOUT_BASE}, "
            f"horizon={optimizer_catalog.DEFAULT_ROLLOUT_HORIZON}, "
            f"alpha={optimizer_catalog.DEFAULT_ROLLOUT_ALPHA}, "
            f"terminal={optimizer_catalog.DEFAULT_ROLLOUT_TERMINAL}, "
            f"iterations={optimizer_catalog.DEFAULT_MCTS_ITERATIONS}, "
            f"exploration_weight={optimizer_catalog.DEFAULT_MCTS_EXPLORATION_WEIGHT}, "
            f"max_outcomes={optimizer_catalog.DEFAULT_MCTS_MAX_OUTCOMES}, "
            "max_expanded_actions=all. "
            "Ex.: --mcts base=nearest,horizon=5,iterations=8,depth=2,"
            "max_expanded_actions=4"
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Caminho para um best_model.zip avulso (atalho com --optimizer rl).\n"
            "O algoritmo é detectado automaticamente. O vecnormalize.pkl é "
            "procurado no mesmo diretório."
        )
    )
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

    scenario_name = _scenario_name(args.scenario)
    discovered = optimizer_catalog.discover_rl_models(
        optimizer_catalog.DEFAULT_MODEL_ROOT,
        scenario_name,
        args.objective,
    )
    rl_model = optimizer_catalog.match_rl_model(discovered, args.optimizer)
    is_catalog = _is_catalog_optimizer(args.optimizer)
    uses_model_path = args.optimizer == "rl" or (args.model_path and not is_catalog and rl_model is None)

    if not is_catalog and rl_model is None and not args.model_path:
        known = ", ".join(
            optimizer_catalog.cli_choices()
            + [model.key for model in discovered]
        )
        parser.error(f"Otimizador '{args.optimizer}' não reconhecido. Opções: {known}")

    optimizer_key = (
        optimizer_catalog.resolve_key(args.optimizer) if is_catalog else None
    )
    is_lowest = optimizer_key == "lowest"
    is_rollout = optimizer_key == "rollout"
    is_mcts = optimizer_key == "mcts"

    if args.lowest and not is_lowest:
        parser.error("Erro: --lowest só pode ser usado com --optimizer lowest")
    if args.rollout and not is_rollout:
        parser.error("Erro: --rollout só pode ser usado com --optimizer rollout")
    if args.mcts and not is_mcts:
        parser.error("Erro: --mcts só pode ser usado com --optimizer mcts")
    if is_lowest and not args.lowest:
        parser.error(
            "Erro: --optimizer lowest exige --lowest "
            "(ex.: --lowest cost=route)"
        )
    if is_rollout and not args.rollout:
        parser.error(
            "Erro: --optimizer rollout exige --rollout "
            "(ex.: --rollout base=lowest,cost=route,horizon=5,terminal=0)"
        )
    if is_mcts and not args.mcts:
        parser.error(
            "Erro: --optimizer mcts exige --mcts "
            "(ex.: --mcts base=nearest,horizon=5,iterations=8,depth=2)"
        )

    lowest_variant = None
    rollout_variant = None
    mcts_variant = None
    try:
        if args.lowest:
            lowest_variant = optimizer_catalog.parse_lowest_cli(args.lowest)
        if args.rollout:
            rollout_variant = optimizer_catalog.parse_rollout_cli(args.rollout)
        if args.mcts:
            mcts_variant = optimizer_catalog.parse_mcts_cli(args.mcts)
    except ValueError as exc:
        parser.error(str(exc))

    if uses_model_path and not args.model_path:
        parser.error("Erro: --optimizer rl requer --model-path com o caminho para best_model.zip")

    if args.model_path and not uses_model_path and rl_model is None and args.optimizer != "rl":
        parser.error("Erro: --model-path só pode ser usado com --optimizer rl")

    if args.save_log:
        log_file = open("log.txt", "w", encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file
    else:
        log_file = None

    try:
        # Cria o otimizador apropriado
        if rl_model is not None or uses_model_path:
            env = prepare_env(args.scenario, args.objective, seed=args.seed, render=args.render)
            model_path = args.model_path if uses_model_path else rl_model.path
            optimizer = optimizer_catalog.load_rl_optimizer(
                env,
                model_path,
                search_root=optimizer_catalog.DEFAULT_MODEL_ROOT,
            )
            env = optimizer.gym_env
        else:
            env = prepare_env(args.scenario, args.objective, seed=args.seed, render=args.render)
            build_kwargs = {}
            if lowest_variant is not None:
                build_kwargs["cost_function"] = lowest_variant.cost_function
            if rollout_variant is not None:
                build_kwargs.update(
                    base_optimizer=rollout_variant.base_optimizer,
                    alpha=rollout_variant.alpha,
                    horizon=rollout_variant.horizon,
                    terminal_cost_mode=rollout_variant.terminal,
                )
                if rollout_variant.cost_function:
                    build_kwargs["cost_function"] = rollout_variant.cost_function
            if mcts_variant is not None:
                build_kwargs.update(
                    base_optimizer=mcts_variant.base_optimizer,
                    alpha=mcts_variant.alpha,
                    horizon=mcts_variant.horizon,
                    terminal_cost_mode=mcts_variant.terminal,
                    iterations=mcts_variant.iterations,
                    exploration_weight=mcts_variant.exploration_weight,
                    depth=mcts_variant.resolved_depth(),
                    max_outcomes=mcts_variant.max_outcomes,
                    max_expanded_actions=mcts_variant.max_expanded_actions,
                )
                if mcts_variant.cost_function:
                    build_kwargs["cost_function"] = mcts_variant.cost_function
            optimizer = optimizer_catalog.build(
                args.optimizer,
                env,
                args.objective,
                **build_kwargs,
            )

        print(f"=== Ambiente pronto com otimizador: {optimizer.get_title()} ===")
        print()
        print(env.get_description())
        print()
        print(f"Action space: {env.action_space}")
        print("Iniciando...\n")

        # Executa baseado no modo
        board: Board = None
        if args.mode in ("auto", "agent"):
            if args.mode == "agent" and rl_model is None and not uses_model_path:
                print("AVISO: Modo 'agent' funciona melhor com --optimizer rl")
            board = optimizer.run_auto(max_steps=args.max_steps)
        elif args.mode == "interactive":
            board = optimizer.run_interactive(max_steps=args.max_steps)

        # Mostra estatísticas finais
        print("\n== FIM DA EXECUÇÃO ==")
        try:
            env.print_environment_state()
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