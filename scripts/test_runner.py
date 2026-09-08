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
                " (requer --cost-function)"
                if "cost_function" in optimizer_catalog.requires(name)
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
    parser.add_argument("--cost-function", choices=optimizer_catalog.COST_FUNCTION_CHOICES, default=None,
                        help="Função de custo (obrigatória com --optimizer lowest, ou com --optimizer rollout e --base-optimizer lowest)")
    parser.add_argument("--base-optimizer", choices=optimizer_catalog.cli_choices(rollout_base=True), default=None,
                        help=f"Política de base do rollout (apenas com --optimizer rollout; padrão: {optimizer_catalog.DEFAULT_ROLLOUT_BASE})")
    parser.add_argument("--alpha", type=float, default=optimizer_catalog.DEFAULT_ROLLOUT_ALPHA,
                        help=f"Fator de desconto do rollout (padrão: {optimizer_catalog.DEFAULT_ROLLOUT_ALPHA})")
    parser.add_argument("--horizon", type=int, default=optimizer_catalog.DEFAULT_ROLLOUT_HORIZON,
                        help=f"Passos de rollout após a ação candidata (padrão: {optimizer_catalog.DEFAULT_ROLLOUT_HORIZON})")
    parser.add_argument("--model-path", default=None,
                        help=(
                            "Caminho para um best_model.zip avulso (atalho com --optimizer rl).\n"
                            "O algoritmo é detectado automaticamente. O vecnormalize.pkl é "
                            "procurado no mesmo diretório."
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

    if is_catalog:
        needed = optimizer_catalog.requires(args.optimizer)
        is_rollout = optimizer_catalog.resolve_key(args.optimizer) == "rollout"
    else:
        needed = ()
        is_rollout = False

    base_name = args.base_optimizer or optimizer_catalog.DEFAULT_ROLLOUT_BASE
    base_needs_cost = is_rollout and "cost_function" in optimizer_catalog.requires(base_name)

    if args.base_optimizer and not is_rollout:
        parser.error("Erro: --base-optimizer só pode ser usado com --optimizer rollout")

    if args.cost_function and "cost_function" not in needed and not base_needs_cost:
        parser.error("Erro: --cost-function só pode ser usado com --optimizer lowest, ou com --optimizer rollout e --base-optimizer lowest")

    if "cost_function" in needed and not args.cost_function:
        parser.error("Erro: --optimizer lowest requer --cost-function")

    if base_needs_cost and not args.cost_function:
        parser.error("Erro: --base-optimizer lowest requer --cost-function")

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
            optimizer = optimizer_catalog.build(
                args.optimizer,
                env,
                args.objective,
                cost_function=args.cost_function,
                base_optimizer=base_name,
                alpha=args.alpha,
                horizon=args.horizon,
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