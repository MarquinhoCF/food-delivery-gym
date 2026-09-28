from __future__ import annotations

import argparse
import sys
from pathlib import Path

from food_delivery_gym.main.eval import experiment as exp
from scripts import generate_boxplots, generate_plots, generate_table


def run(
    experiment_dir: str | Path,
    *,
    episodes: bool = False,
) -> dict:
    experiment_dir = Path(experiment_dir).resolve()
    payload = exp.load_run_json(experiment_dir)
    spec = payload["spec"]
    objectives = [int(o) for o in spec["objectives"]]
    scenarios = [str(s) for s in spec["scenarios"]]
    results_dir = str(experiment_dir)

    report_dir = experiment_dir / "report"
    boxplots_dir = report_dir / "boxplots"
    report_dir.mkdir(parents=True, exist_ok=True)
    boxplots_dir.mkdir(parents=True, exist_ok=True)

    print("=== Relatório do experimento ===")
    print(f"  Pasta      : {experiment_dir}")
    print(f"  Nome       : {spec.get('name', experiment_dir.name)}")
    print(f"  Objetivos  : {objectives}")
    print(f"  Cenários   : {scenarios}")
    print(f"  Episódios  : {'sim' if episodes else 'não'}")
    print()

    table_path = report_dir / "objective_table.xlsx"
    print("── Planilha ──")
    saved_table = generate_table.run(
        results_dir,
        objectives,
        scenarios,
        str(table_path),
    )
    print()

    saved_boxplots: list[str] = []
    print("── Boxplots ──")
    for objective in objectives:
        print(f"\n  Objetivo {objective}")
        paths = generate_boxplots.run(
            results_dir,
            objective,
            scenarios,
            str(boxplots_dir),
            prefix=f"boxplot_obj{objective}",
        )
        if not paths:
            print(
                f"  [AVISO] Nenhum boxplot gerado para objetivo {objective} "
                "(sem agentes no disco)."
            )
        saved_boxplots.extend(paths)
    print()

    episodes_count = 0
    if episodes:
        print("── Gráficos por episódio ──")
        episodes_count = generate_plots.run(
            results_dir,
            objectives,
            scenarios,
            only_episode=True,
        )
        print()

    print("=== Relatório concluído ===")
    if saved_table:
        print(f"  Planilha   : {saved_table}")
    print(f"  Boxplots   : {len(saved_boxplots)} arquivo(s) em {boxplots_dir}")
    for path in saved_boxplots:
        print(f"               {path}")
    if episodes:
        print(f"  Episódios  : {episodes_count} agente(s) processado(s)")

    return {
        "table": saved_table,
        "boxplots": saved_boxplots,
        "episodes_agents": episodes_count,
        "report_dir": str(report_dir),
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Gera planilha e boxplots a partir do run.json de um experimento.\n"
            "Objetivos e cenários vêm da spec; agentes são descobertos no disco."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "experiment_dir",
        metavar="EXPERIMENT_DIR",
        help="Pasta do experimento (ex.: data/runs/obj3_rollout).",
    )
    parser.add_argument(
        "--episodes",
        action="store_true",
        help=(
            "Também gera gráficos por episódio em cada pasta de agente\n"
            "(obj_N/<cenário>/<agente>/figs/)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args.experiment_dir, episodes=args.episodes)
    except FileNotFoundError as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
