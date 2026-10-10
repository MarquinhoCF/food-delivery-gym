from __future__ import annotations

import argparse
import os
import textwrap
from datetime import datetime
from importlib.resources import files

from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer import catalog as optimizer_catalog
from food_delivery_gym.main.optimizer.optimizer_gym.monte_carlo_tree_search_optimizer_gym import (
    MonteCarloTreeSearchOptimizerGym,
)
from food_delivery_gym.main.statistics.lookahead.mcts_decision_board import (
    MCTSDecisionBoard,
)

DEFAULT_SEED = 5434
DEFAULT_OBJECTIVE = 1
DEFAULT_ALPHA = optimizer_catalog.DEFAULT_ROLLOUT_ALPHA
DEFAULT_BASE_OPTIMIZER = optimizer_catalog.DEFAULT_ROLLOUT_BASE
DEFAULT_HORIZON = optimizer_catalog.DEFAULT_ROLLOUT_HORIZON
DEFAULT_OUT_DIR = "data/visualization/mcts_viz"
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
            Visualização pós-episódio da árvore do MonteCarloTreeSearchOptimizerGym.

            Executa um episódio, grava o decision_log (árvore explorada + valores
            na raiz) e salva JSON completo + um PNG da decisão pedida em --decision.

            Com --from-json, pula o episódio e só renderiza o PNG a partir do JSON.

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
        "--decision",
        type=int,
        required=True,
        help="Índice da decisão (decision_idx, 0-based); gera só esse PNG",
    )
    parser.add_argument(
        "--from-json",
        default=None,
        help="Caminho de decisions.json já coletado; pula o episódio",
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
        help="Política de base nas folhas (default: nearest)",
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
        help="Horizonte do rollout nas folhas (default: 5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Fator de desconto (default: 0.9)",
    )
    parser.add_argument(
        "--terminal-cost",
        choices=optimizer_catalog.TERMINAL_COST_MODES,
        default=optimizer_catalog.DEFAULT_ROLLOUT_TERMINAL,
        help=(
            "Custo terminal: '0' força zero; 'model' exige o linear model "
            f"(default: {optimizer_catalog.DEFAULT_ROLLOUT_TERMINAL})"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=optimizer_catalog.DEFAULT_MCTS_ITERATIONS,
        help=f"Iterações MCTS (default: {optimizer_catalog.DEFAULT_MCTS_ITERATIONS})",
    )
    parser.add_argument(
        "--exploration-weight",
        type=float,
        default=optimizer_catalog.DEFAULT_MCTS_EXPLORATION_WEIGHT,
        help=(
            "Peso de exploração UCT "
            f"(default: {optimizer_catalog.DEFAULT_MCTS_EXPLORATION_WEIGHT})"
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Profundidade máxima da árvore (default: igual ao horizon)",
    )
    parser.add_argument(
        "--max-outcomes",
        type=int,
        default=optimizer_catalog.DEFAULT_MCTS_MAX_OUTCOMES,
        help=(
            "Máx. futuros W por ação "
            f"(default: {optimizer_catalog.DEFAULT_MCTS_MAX_OUTCOMES})"
        ),
    )
    parser.add_argument(
        "--max-expanded-actions",
        type=int,
        default=None,
        help=(
            "Máx. ações expandidas por nó (max_expanded_actions) "
            "(default: todos os motoristas)"
        ),
    )
    parser.add_argument(
        "--expansion-order",
        choices=list(optimizer_catalog.MCTS_EXPANSION_ORDER_CHOICES),
        default=optimizer_catalog.DEFAULT_MCTS_EXPANSION_ORDER,
        help=(
            "Ordem das ações não tentadas: immediate (C(S,x)) ou "
            f"heuristic (ranked_actions da base; default: "
            f"{optimizer_catalog.DEFAULT_MCTS_EXPANSION_ORDER})"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            f"Diretório de saída (default: {DEFAULT_OUT_DIR}; "
            "com --from-json, pasta do JSON)"
        ),
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

    if args.from_json is not None:
        json_path = args.from_json
        if not os.path.isfile(json_path):
            raise SystemExit(f"--from-json não encontrado: {json_path}")
        board = MCTSDecisionBoard.from_json(json_path)
        out_dir = args.out_dir or (os.path.dirname(os.path.abspath(json_path)) or ".")
        print(f"=== MCTS from JSON ({len(board.decision_log)} decisões) ===")
        print(f"from_json={json_path}")
        print(f"decision={args.decision} out_dir={out_dir}\n")
        os.makedirs(out_dir, exist_ok=True)
        board.save(out_dir, decision_idx=args.decision)
        print(
            f"Figura: {os.path.join(out_dir, 'mcts_decisions', f'decision_{args.decision + 1:03d}.png')}"
        )
        return

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
    optimizer = MonteCarloTreeSearchOptimizerGym(
        env,
        base_optimizer_cls=base_optimizer_cls,
        base_optimizer_kwargs=base_optimizer_kwargs,
        alpha=args.alpha,
        horizon=args.horizon,
        record_decisions=True,
        terminal_cost_mode=args.terminal_cost,
        iterations=args.iterations,
        exploration_weight=args.exploration_weight,
        depth=args.depth,
        max_outcomes=args.max_outcomes,
        max_expanded_actions=args.max_expanded_actions,
        expansion_order=args.expansion_order,
    )
    optimizer.state = env.get_observation()
    optimizer.done = False
    optimizer.truncated = False

    print(f"=== {optimizer.get_title()} ===")
    out_base = args.out_dir or DEFAULT_OUT_DIR
    out_dir = os.path.join(
        out_base,
        args.scenario.split(".")[0],
        optimizer_catalog.mcts_result_key(
            args.base_optimizer,
            args.cost_function,
            horizon=args.horizon,
            alpha=args.alpha,
            terminal=args.terminal_cost,
            iterations=args.iterations,
            exploration_weight=args.exploration_weight,
            depth=args.depth,
            max_outcomes=args.max_outcomes,
            max_expanded_actions=args.max_expanded_actions,
            expansion_order=args.expansion_order,
        ),
        f"obj_{args.objective}",
        datetime.now().strftime("%d_%m_%Y-%H_%M_%S"),
    )
    print(f"scenario={args.scenario} seed={args.seed} objective={args.objective}")
    print(f"base_optimizer={args.base_optimizer} cost_function={args.cost_function}")
    print(f"decision={args.decision} out_dir={out_dir}\n")

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
    board = MCTSDecisionBoard(optimizer.decision_log)
    json_path = os.path.join(out_dir, "decisions.json")
    board.dump_json(json_path)
    board.save(out_dir, decision_idx=args.decision)

    print(f"JSON: {json_path}")
    print(
        f"Figura: {os.path.join(out_dir, 'mcts_decisions', f'decision_{args.decision + 1:03d}.png')}"
    )


if __name__ == "__main__":
    main()
