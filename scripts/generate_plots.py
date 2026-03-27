from __future__ import annotations

import argparse
import json
import os
import traceback

import numpy as np

from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.scenarios import get_defaults_scenarios
from food_delivery_gym.main.statistic.simulation_stats import SimulationStats

DEFAULT_RESULTS_DIR = "./data/runs/execucoes"
ALL_OBJECTIVES      = FoodDeliveryGymEnv.REWARD_OBJECTIVES
DEFAULT_SCENARIOS   = get_defaults_scenarios()


def _load_from_npz(npz_path: str) -> SimulationStats:
    return SimulationStats.load(npz_path)


def _load_from_json(json_path: str) -> SimulationStats:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sim_list = data.get("sim", [])
    n        = len(sim_list)

    stats                    = SimulationStats.__new__(SimulationStats)
    stats._sim               = None
    stats._raw_episodes      = []
    stats.num_drivers        = None
    stats.num_establishments = None
    stats._num_runs          = n
    stats.aggregate          = data.get("aggregate", {})

    if n == 0:
        stats.episodes       = {}
        stats.drivers        = {}
        stats.establishments = {}
        return stats

    stats.episodes = {
        "rewards":          [ep.get("reward")           for ep in sim_list],
        "lengths":          [ep.get("episode_length")   for ep in sim_list],
        "simpy_times":      [ep.get("simpy_final_time") for ep in sim_list],
        "truncated":        [ep.get("truncated")        for ep in sim_list],
        "orders_generated": [ep.get("orders_generated") for ep in sim_list],
        "delivery_time":    [ep.get("delivery_time")    for ep in sim_list],
        "distance":         [ep.get("distance")         for ep in sim_list],
        "events":           [ep.get("events", [])       for ep in sim_list],
    }

    stats.drivers  = {}
    driver_ids     = list(sim_list[0].get("driver", {}).keys())
    for did in driver_ids:
        sample = sim_list[0]["driver"].get(did, {})
        stats.drivers[did] = {
            metric: [ep.get("driver", {}).get(did, {}).get(metric) for ep in sim_list]
            for metric in sample
        }

    stats.establishments = {}
    est_ids              = list(sim_list[0].get("establishment", {}).keys())
    for eid in est_ids:
        sample = sim_list[0]["establishment"].get(eid, {})
        stats.establishments[eid] = {
            metric: [ep.get("establishment", {}).get(eid, {}).get(metric) for ep in sim_list]
            for metric in sample
        }

    return stats


def _ensure_raw_episodes(stats: SimulationStats) -> None:
    if stats._raw_episodes or stats._num_runs == 0:
        return

    for ep in stats.sim:
        stats._raw_episodes.append({
            "reward":           ep.get("reward"),
            "length":           ep.get("episode_length"),
            "simpy_time":       ep.get("simpy_final_time"),
            "truncated":        ep.get("truncated"),
            "orders_generated": ep.get("orders_generated"),
            "events":           ep.get("events", []),
            # EpisodeStatsBoard acessa ep["drivers"] e ep["establishments"]
            # internamente em get_episode_sim(), que por sua vez devolve
            # as chaves "driver" e "establishment" (singular).
            "drivers":          ep.get("driver", {}),
            "establishments":   ep.get("establishment", {}),
        })


def load_sim_stats(agent_dir: str) -> SimulationStats | None:
    npz_path  = os.path.join(agent_dir, "metrics_data.npz")
    json_path = os.path.join(agent_dir, "metrics_data.json")

    if os.path.exists(npz_path):
        try:
            stats = _load_from_npz(npz_path)
            _ensure_raw_episodes(stats)
            print(f"    [NPZ] {npz_path}")
            return stats
        except Exception as e:
            print(f"    [Erro NPZ] {npz_path}: {e}")

    if os.path.exists(json_path):
        try:
            stats = _load_from_json(json_path)
            _ensure_raw_episodes(stats)
            print(f"    [JSON] {json_path}")
            return stats
        except Exception as e:
            print(f"    [Erro JSON] {json_path}: {e}")

    return None


def _has_metrics_file(agent_dir: str) -> bool:
    return (
        os.path.isfile(os.path.join(agent_dir, "metrics_data.npz")) or
        os.path.isfile(os.path.join(agent_dir, "metrics_data.json"))
    )

# ── Descoberta de agentes ─────────────────────────────────────────────────────

def discover_agent_dirs(results_dir: str, objectives: list, scenarios: list) -> list[str]:
    KNOWN_ORDER = [
        "random", "first_driver", "nearest_driver",
        "lowest_route_cost", "lowest_marginal_route_cost",
    ]

    dirs: list[str] = []

    for obj in objectives:
        for scenario in scenarios:
            base = os.path.join(results_dir, f"obj_{obj}", f"{scenario}_scenario")
            if not os.path.isdir(base):
                continue

            entries = [e for e in os.scandir(base) if e.is_dir() and _has_metrics_file(e.path)]

            known = [e for k in KNOWN_ORDER for e in entries if e.name == k]
            ppo   = sorted([e for e in entries if e.name not in KNOWN_ORDER], key=lambda e: e.name)

            for entry in known + ppo:
                dirs.append(entry.path)

    return dirs

def generate_episode_plots(stats: SimulationStats, agent_dir: str) -> None:
    for idx in range(stats._num_runs):
        try:
            board = stats.get_episode_board(episode_idx=idx)
            board.save(agent_dir)
        except Exception as e:
            print(f"      ⚠ Episódio {idx + 1}: {e}")
            traceback.print_exc()


def generate_batch_plots(stats: SimulationStats, agent_dir: str) -> None:
    try:
        board = stats.get_batch_board()
        board.save(agent_dir)
    except Exception as e:
        print(f"      ⚠ Batch board: {e}")
        traceback.print_exc()


def process_agent(agent_dir: str, do_episode: bool, do_batch: bool) -> None:
    agent_name = os.path.basename(agent_dir)
    print(f"  → {agent_name}")

    stats = load_sim_stats(agent_dir)
    if stats is None:
        print(f"    ✗ Nenhum arquivo de métricas encontrado em: {agent_dir}")
        return

    if stats._num_runs == 0:
        print(f"    ✗ Nenhum episódio registrado.")
        return

    if do_episode:
        print(f"    Gerando {stats._num_runs} gráficos de episódio...")
        generate_episode_plots(stats, agent_dir)

    if do_batch:
        print(f"    Gerando gráficos de lote...")
        generate_batch_plots(stats, agent_dir)

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera gráficos de episódios e/ou lote a partir dos arquivos de métricas\n"
            "(metrics_data.npz ou metrics_data.json) de cada agente avaliado."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--results-dir", "-r",
        default=DEFAULT_RESULTS_DIR,
        help=f"Diretório raiz com os resultados (obj_N/). Padrão: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--objectives", "-o",
        nargs="+",
        type=int,
        default=ALL_OBJECTIVES,
        metavar="N",
        help="Objetivos a processar (1–13). Padrão: todos.",
    )
    parser.add_argument(
        "--scenarios", "-s",
        nargs="+",
        choices=DEFAULT_SCENARIOS,
        default=DEFAULT_SCENARIOS,
        metavar="SCENARIO",
        help=f"Cenários a processar. Opções: {DEFAULT_SCENARIOS}. Padrão: todos.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--only-episode",
        action="store_true",
        help="Gera somente os gráficos de episódios individuais.",
    )
    mode.add_argument(
        "--only-batch",
        action="store_true",
        help="Gera somente os gráficos agregados de lote.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    do_episode = not args.only_batch
    do_batch   = not args.only_episode

    mode_label = (
        "episódios + lote" if (do_episode and do_batch) else
        "somente episódios" if do_episode else
        "somente lote"
    )

    print("=== Geração de Gráficos ===")
    print(f"  Diretório  : {args.results_dir}")
    print(f"  Objetivos  : {args.objectives}")
    print(f"  Cenários   : {args.scenarios}")
    print(f"  Modo       : {mode_label}")
    print()

    agent_dirs = discover_agent_dirs(args.results_dir, args.objectives, args.scenarios)

    if not agent_dirs:
        print(
            "[AVISO] Nenhum agente encontrado. "
            "Verifique --results-dir e se os arquivos metrics_data.npz/.json existem."
        )
        return

    print(f"{len(agent_dirs)} diretório(s) de agente encontrado(s).\n")

    prev_scenario_key = None
    for agent_dir in agent_dirs:
        # Extrai obj_N/scenario para exibir cabeçalhos de seção
        parts = agent_dir.replace("\\", "/").split("/")
        try:
            scenario_key = f"{parts[-3]}/{parts[-2]}"
        except IndexError:
            scenario_key = agent_dir

        if scenario_key != prev_scenario_key:
            print(f"\n── {scenario_key} ──")
            prev_scenario_key = scenario_key

        process_agent(agent_dir, do_episode=do_episode, do_batch=do_batch)

    print("\n=== Concluído ===")


if __name__ == "__main__":
    main()