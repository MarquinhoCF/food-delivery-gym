from __future__ import annotations

import argparse
import os
import textwrap
from importlib.resources import files

from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer import catalog as optimizer_catalog
from food_delivery_gym.main.optimizer.optimizer_gym.rollout_optimizer_gym import (
    RolloutOptimizerGym,
)
from food_delivery_gym.main.statistics.lookahead.rollout_decision_board import (
    RolloutDecisionBoard,
)

DEFAULT_SEED = 5434
DEFAULT_OBJECTIVE = 1
DEFAULT_ALPHA = optimizer_catalog.DEFAULT_ROLLOUT_ALPHA
DEFAULT_BASE_OPTIMIZER = optimizer_catalog.DEFAULT_ROLLOUT_BASE
DEFAULT_HORIZON = optimizer_catalog.DEFAULT_ROLLOUT_HORIZON
DEFAULT_OUT_DIR = "data/visualization/rollout_viz"
ALL_OBJECTIVES = FoodDeliveryGymEnv.REWARD_OBJECTIVES
BASE_OPTIMIZER_CHOICES = optimizer_catalog.cli_choices(rollout_base=True)


def prepare_env(scenario_filename: str, reward_objective: int, seed: int) -> FoodDeliveryGymEnv:
    scenario_path = str(
        files("food_delivery_gym.main.scenarios").joinpath(scenario_filename)
    )
    env = FoodDeliveryGymEnv(
        scenario_json_file_path=scenario_path,
        reward_objective=reward_objective,
        mode=EnvMode.TESTING,
    )
    env.reset(seed=seed)
    return env


def resolve_base_optimizer(
    name: str,
    objective: int,
    cost_function_name: str | None,
):
    """Retorna (classe, kwargs) da política de base do rollout."""
    try:
        return optimizer_catalog.constructor_args(
            name,
            objective=objective,
            cost_function=cost_function_name,
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Visualização pós-episódio da seleção de ações do RolloutOptimizerGym.

            Executa um episódio, grava o decision_log (Q por motorista + trajetória
            da política base) e salva PNGs + JSON em --out-dir.

            Políticas de base (--base-optimizer), cadastradas em optimizer/catalog.py:
              {base_lines}
            """
        ).format(base_lines="\n".join(
            f"  {name:<8} {optimizer_catalog.cli_label(name)}"
            + (
                " (requer --cost-function)"
                if "cost_function" in optimizer_catalog.requires(name)
                else ""
            )
            for name in BASE_OPTIMIZER_CHOICES
        )),
    )
    parser.add_argument(
        "--scenario",
        default="medium.json",
        help="Arquivo de cenário em food_delivery_gym.main.scenarios",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--objective",
        type=int,
        default=DEFAULT_OBJECTIVE,
        help=f"Objetivo de recompensa ({ALL_OBJECTIVES})",
    )
    parser.add_argument(
        "--base-optimizer",
        choices=BASE_OPTIMIZER_CHOICES,
        default=DEFAULT_BASE_OPTIMIZER,
        help="Política de base usada no rollout (default: nearest)",
    )
    parser.add_argument(
        "--cost-function",
        choices=optimizer_catalog.COST_FUNCTION_CHOICES,
        default=None,
        help="Função de custo (obrigatório se --base-optimizer lowest)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Horizonte de rollout após a ação candidata (default: 5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Fator de desconto do rollout (default: 0.9)",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Diretório de saída (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Limite de passos do episódio",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.objective not in ALL_OBJECTIVES:
        raise SystemExit(
            f"--objective deve ser um inteiro em {ALL_OBJECTIVES}"
        )

    needed = optimizer_catalog.requires(args.base_optimizer)
    if args.cost_function and "cost_function" not in needed:
        raise SystemExit("--cost-function só pode ser usado com --base-optimizer lowest")

    if "cost_function" in needed and not args.cost_function:
        raise SystemExit("--base-optimizer lowest requer --cost-function")

    base_optimizer_cls, base_optimizer_kwargs = resolve_base_optimizer(
        args.base_optimizer,
        objective=args.objective,
        cost_function_name=args.cost_function,
    )

    env = prepare_env(args.scenario, args.objective, seed=args.seed)
    optimizer = RolloutOptimizerGym(
        env,
        base_optimizer_cls=base_optimizer_cls,
        base_optimizer_kwargs=base_optimizer_kwargs,
        alpha=args.alpha,
        horizon=args.horizon,
        record_decisions=True,
    )
    # prepare_env já fez reset; sincroniza o estado do otimizador
    optimizer.state = env.get_observation()
    optimizer.done = False
    optimizer.truncated = False

    print(f"=== {optimizer.get_title()} ===")
    out_dir = os.path.join(
        args.out_dir,
        args.scenario.split(".")[0],
        optimizer_catalog.rollout_result_key(args.base_optimizer, args.cost_function),
    )
    print(f"scenario={args.scenario} seed={args.seed} objective={args.objective}")
    print(f"base_optimizer={args.base_optimizer} cost_function={args.cost_function}")
    print(f"out_dir={out_dir}\n")

    step = 0
    sum_reward = 0.0
    while step < args.max_steps and not (optimizer.done or optimizer.truncated):
        step += 1
        order = env.get_current_order()
        if order is None:
            break
        action = optimizer.assign_driver_to_order(optimizer.state, order)
        obs, reward, terminated, truncated, _info = env.step(action)
        optimizer.state = obs
        optimizer.done = terminated
        optimizer.truncated = truncated
        sum_reward += reward
        if step % 10 == 0:
            print(
                f"Step {step}: reward_sum={sum_reward:.2f} "
                f"decisions={len(optimizer.decision_log)}"
            )

    print(f"\nEpisódio finalizado em {step} passos | reward_sum={sum_reward:.2f}")
    print(f"Decisões gravadas: {len(optimizer.decision_log)}")

    os.makedirs(out_dir, exist_ok=True)
    board = RolloutDecisionBoard(optimizer.decision_log)
    json_path = os.path.join(out_dir, "decisions.json")
    board.dump_json(json_path)
    board.save(out_dir)

    print(f"JSON: {json_path}")
    print(f"Figuras: {os.path.join(out_dir, 'rollout_decisions')}/")


if __name__ == "__main__":
    main()
