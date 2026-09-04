from __future__ import annotations

import argparse
import os
import textwrap
from importlib.resources import files
from typing import Type

from food_delivery_gym.main.cost.marginal_route_cost_function import MarginalRouteCostFunction
from food_delivery_gym.main.cost.route_cost_function import RouteCostFunction
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer.optimizer_gym.first_driver_optimizer_gym import (
    FirstDriverOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.lowest_cost_driver_optimizer_gym import (
    LowestCostDriverOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import (
    NearestDriverOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.optmizer_gym import OptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.random_driver_optimizer_gym import (
    RandomDriverOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.rollout_optimizer_gym import (
    RolloutOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.weighted_score_driver_optimizer_gym import (
    WeightedScoreDriverOptimizerGym,
)
from food_delivery_gym.main.statistics.lookahead.rollout_decision_board import (
    RolloutDecisionBoard,
)

DEFAULT_SEED = 5434
DEFAULT_OBJECTIVE = 1
DEFAULT_ALPHA = 0.9
DEFAULT_BASE_OPTIMIZER = "nearest"
DEFAULT_HORIZON = 5
DEFAULT_OUT_DIR = "data/visualization/rollout_viz"
ALL_OBJECTIVES = FoodDeliveryGymEnv.REWARD_OBJECTIVES
BASE_OPTIMIZER_CHOICES = ("nearest", "first", "random", "weighted", "lowest")


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
) -> tuple[Type[OptimizerGym], dict]:
    """Retorna (classe, kwargs) da política de base do rollout."""
    if name == "nearest":
        return NearestDriverOptimizerGym, {}
    if name == "first":
        return FirstDriverOptimizerGym, {}
    if name == "random":
        return RandomDriverOptimizerGym, {}
    if name == "weighted":
        return WeightedScoreDriverOptimizerGym, {}
    if name == "lowest":
        if cost_function_name == "route":
            cost_obj = RouteCostFunction.get_cost_objective(objective)
            cost_function = RouteCostFunction(objective=cost_obj)
        elif cost_function_name == "marginal_route":
            cost_obj = MarginalRouteCostFunction.get_cost_objective(objective)
            cost_function = MarginalRouteCostFunction(objective=cost_obj)
        else:
            raise SystemExit(
                "--base-optimizer lowest requer --cost-function "
                "{route|marginal_route}"
            )
        return LowestCostDriverOptimizerGym, {"cost_function": cost_function}

    raise SystemExit(f"base-optimizer desconhecido: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Visualização pós-episódio da seleção de ações do RolloutOptimizerGym.

            Executa um episódio, grava o decision_log (Q por motorista + trajetória
            da política base) e salva PNGs + JSON em --out-dir.

            Políticas de base (--base-optimizer):
              nearest   NearestDriverOptimizerGym (default)
              first     FirstDriverOptimizerGym
              random    RandomDriverOptimizerGym
              weighted  WeightedScoreDriverOptimizerGym
              lowest    LowestCostDriverOptimizerGym (requer --cost-function)
            """
        ),
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
        choices=("route", "marginal_route"),
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

    if args.cost_function and args.base_optimizer != "lowest":
        raise SystemExit("--cost-function só pode ser usado com --base-optimizer lowest")

    if args.base_optimizer == "lowest" and not args.cost_function:
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
    out_dir = os.path.join(args.out_dir, args.scenario.split(".")[0], args.base_optimizer if args.base_optimizer != "lowest" else args.base_optimizer + "_" + args.cost_function)
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
