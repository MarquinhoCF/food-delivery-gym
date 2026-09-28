from food_delivery_gym.main.eval.silence import in_worker_process, silence_eval_noise

if in_worker_process():
    silence_eval_noise(filter_stderr=True)

import os
import time
import traceback
import argparse
from datetime import datetime, timezone

from food_delivery_gym.main.eval import experiment as exp
from food_delivery_gym.main.eval.eval_parallel import (  # noqa: E402
    EvalJobSpec,
    create_eval_environment,
)
from food_delivery_gym.main.optimizer import catalog as optimizer_catalog  # noqa: E402
from food_delivery_gym.main.scenarios import get_all_scenarios  # noqa: E402

ALL_SCENARIOS = get_all_scenarios()
DEFAULT_MODEL_SUBDIR = "treinamento"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Avalia agentes otimizadores (heurísticas e modelos de RL) "
            "no ambiente de entrega de última milha.\n"
            "Aceita um YAML de experimento (posicional) e/ou flags CLI.\n"
            "Flags explícitas sobrescrevem o YAML. Sem YAML, --name é obrigatório."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "experiment_file",
        nargs="?",
        default=None,
        metavar="EXPERIMENT.yaml",
        help="Arquivo YAML do experimento (opcional).",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nome do experimento (obrigatório sem YAML). Saída em data/runs/<name>/.",
    )

    parser.add_argument(
        "--objectives", "-o",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Objetivos de recompensa (1-13). Padrão: todos.",
    )

    parser.add_argument(
        "--scenarios", "-s",
        nargs="+",
        choices=ALL_SCENARIOS,
        default=None,
        metavar="SCENARIO",
        help=f"Cenários a executar. Opções: {ALL_SCENARIOS}",
    )

    parser.add_argument(
        "--agents", "-a",
        nargs="+",
        default=None,
        metavar="AGENT",
        help=(
            "Agentes a executar: heurísticas do catálogo ou modelos descobertos.\n"
            f"Heurísticas: {optimizer_catalog.keys(is_heuristic=True)}"
        ),
    )

    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=None,
        metavar="MODEL_NAME",
        help="Filtra os modelos RL. Padrão: todos os descobertos.",
    )

    parser.add_argument(
        "--heuristics",
        nargs="+",
        default=None,
        metavar="HEURISTIC",
        help=(
            "Filtra as heurísticas.\n"
            f"Opções: {optimizer_catalog.keys(is_heuristic=True)}"
        ),
    )

    parser.add_argument(
        "--lowest",
        action="append",
        default=None,
        metavar="SPEC",
        help=(
            "Variante explícita de lowest (repetível). Formato: cost=route\n"
            f"Opções de cost: {optimizer_catalog.COST_FUNCTION_CHOICES}\n"
            "Ex.: --lowest cost=route --lowest cost=weighted_score"
        ),
    )

    parser.add_argument(
        "--rollout",
        action="append",
        default=None,
        metavar="SPEC",
        help=(
            "Variante explícita de rollout (repetível). Formato key=value,...\n"
            "Chaves: base, cost, horizon, alpha, terminal (0|model).\n"
            f"Defaults: base={optimizer_catalog.DEFAULT_ROLLOUT_BASE}, "
            f"horizon={optimizer_catalog.DEFAULT_ROLLOUT_HORIZON}, "
            f"alpha={optimizer_catalog.DEFAULT_ROLLOUT_ALPHA}, "
            f"terminal={optimizer_catalog.DEFAULT_ROLLOUT_TERMINAL}.\n"
            "Ex.: --rollout base=lowest,cost=weighted_score,horizon=5,terminal=0,alpha=0.9"
        ),
    )

    parser.add_argument(
        "--rollout-record-decisions",
        action="store_true",
        default=None,
        help="Grava o decision_log do rollout (desativado por padrão).",
    )

    parser.add_argument(
        "--no-rl",
        action="store_true",
        default=None,
        help="Desativa a execução dos modelos de Aprendizado por Reforço.",
    )

    parser.add_argument(
        "--no-heuristics",
        action="store_true",
        default=None,
        help="Desativa a execução de todas as heurísticas.",
    )

    parser.add_argument(
        "--num-runs", "-n",
        type=int,
        default=None,
        help=f"Número de simulações por agente. Padrão: {exp.DEFAULT_NUM_RUNS}.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Processos paralelos por agente. 1 = serial (padrão).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Seed para reprodutibilidade. Padrão: {exp.DEFAULT_SEED}.",
    )

    parser.add_argument(
        "--model-base-dir",
        type=str,
        default=None,
        help=(
            "Diretório raiz dos modelos treinados.\n"
            f"Padrão: {exp.DEFAULT_MODEL_BASE_DIR}"
        ),
    )

    parser.add_argument(
        "--experiment-mode",
        choices=list(exp.EXPERIMENT_MODES),
        default=None,
        help=(
            "Modo de seleção dos modelos RL:\n"
            "  cross_scenario  – modelos de --train-scenario em todos os cenários\n"
            "  same_scenario   – modelo do próprio cenário de avaliação\n"
            f"Padrão: {exp.DEFAULT_EXPERIMENT_MODE}"
        ),
    )

    parser.add_argument(
        "--train-scenario",
        type=str,
        choices=ALL_SCENARIOS,
        default=None,
        help=(
            "Cenário cujos modelos serão usados no modo cross_scenario.\n"
            f"Padrão: {exp.DEFAULT_TRAIN_SCENARIO}"
        ),
    )

    parser.add_argument(
        "--metrics-fmt",
        choices=list(exp.METRICS_FMT_OPTIONS),
        default=None,
        help=f"Formato das métricas: npz ou json. Padrão: {exp.DEFAULT_METRICS_FMT}",
    )

    parser.add_argument(
        "--batch-plots",
        action="store_true",
        default=None,
        help="Ativa a geração de gráficos agregados.",
    )

    parser.add_argument(
        "--all-plots",
        action="store_true",
        default=None,
        help="Ativa todos os gráficos (individuais + agregados).",
    )

    return parser, parser.parse_args()


def _cli_overrides(args) -> dict:
    """Coleta só os campos efetivamente passados na linha de comando."""
    overrides: dict = {}
    mapping = {
        "name": args.name,
        "objectives": args.objectives,
        "scenarios": args.scenarios,
        "agents": args.agents,
        "heuristics": args.heuristics,
        "models": args.models,
        "lowest": args.lowest,
        "rollout": args.rollout,
        "num_runs": args.num_runs,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "model_base_dir": args.model_base_dir,
        "experiment_mode": args.experiment_mode,
        "train_scenario": args.train_scenario,
        "metrics_fmt": args.metrics_fmt,
    }
    for key, value in mapping.items():
        if value is not None:
            overrides[key] = value

    # store_true: None = ausente, True = passado
    if args.no_rl is not None:
        overrides["no_rl"] = bool(args.no_rl)
    if args.no_heuristics is not None:
        overrides["no_heuristics"] = bool(args.no_heuristics)
    if args.rollout_record_decisions is not None:
        overrides["rollout_record_decisions"] = bool(args.rollout_record_decisions)
    if args.batch_plots is not None:
        overrides["batch_plots"] = bool(args.batch_plots)
    if args.all_plots is not None:
        overrides["all_plots"] = bool(args.all_plots)
    return overrides


def create_environment(reward_objective: int, scenario_name: str):
    return create_eval_environment(reward_objective, scenario_name)


def build_eval_job_spec(variant, scenario: str, objective: int) -> EvalJobSpec:
    """Monta o payload serializável para workers a partir de uma EvalVariant."""
    return EvalJobSpec(
        scenario=scenario,
        objective=objective,
        optimizer_key=variant.spec.key,
        extras=dict(variant.extras),
        model_path=variant.spec.model_path,
        model_search_root=variant.spec.model_search_root,
    )


def select_agents_for_run(
    available: list,
    agents: list[str] | None,
    heuristics: list[str] | None,
    models: list[str] | None,
    no_heuristics: bool,
    no_rl: bool,
) -> list:
    """Aplica --agents, --heuristics, --models, --no-heuristics e --no-rl."""
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
    *,
    experiment_name: str,
    scenario: str,
    variants: list,
    objective: int,
    num_runs: int,
    seed: int,
    save_individual_plots: bool,
    save_mean_plots: bool,
    metrics_fmt: str,
    num_workers: int,
    completed_result_keys: list[str],
):
    for variant in variants:
        output_dir = str(
            exp.agent_dir(experiment_name, objective, scenario, variant.result_key)
        ) + "/"

        if exp.agent_complete(output_dir):
            print(
                f"\n=== Pulando {variant.spec.label} ({variant.result_key}) "
                f"— já completo em {output_dir} ==="
            )
            if variant.result_key not in completed_result_keys:
                completed_result_keys.append(variant.result_key)
            continue

        print(
            f"\n=== Executando {variant.spec.label} ({variant.result_key}) "
            f"no cenário '{scenario}' ==="
        )
        try:
            env = create_environment(reward_objective=objective, scenario_name=scenario)
            optimizer = optimizer_catalog.instantiate(
                variant.spec,
                env,
                objective,
                **variant.extras,
            )
            eval_spec = build_eval_job_spec(variant, scenario, objective)
            optimizer.run_simulations(
                num_runs, output_dir, seed=seed,
                save_individual_plots=save_individual_plots,
                save_mean_plots=save_mean_plots,
                metrics_fmt=metrics_fmt,
                num_workers=num_workers,
                eval_spec=eval_spec,
            )
            if variant.result_key not in completed_result_keys:
                completed_result_keys.append(variant.result_key)
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário — encerrando avaliação.")
            raise
        except Exception as e:
            print(
                f"Erro ao executar {variant.result_key} — "
                f"objetivo {objective}, cenário '{scenario}': {e}"
            )
            traceback.print_exc()


def main():
    parser, args = parse_args()

    try:
        spec = exp.resolve_experiment(
            yaml_path=args.experiment_file,
            cli_overrides=_cli_overrides(args),
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))

    save_individual_plots = bool(spec["all_plots"])
    save_mean_plots = bool(spec["batch_plots"])
    lowest_variants = spec["_lowest_variants"]
    rollout_variants = spec["_rollout_variants"]

    print("=== Avaliando Agentes no Ambiente de Entrega de Última Milha ===")
    print(f"  Experimento  : {spec['name']}")
    print(f"  Saída        : {exp.experiment_dir(spec['name'])}/")
    print(f"  Objetivos    : {spec['objectives']}")
    print(f"  Cenários     : {spec['scenarios']}")
    print(f"  Agentes      : {spec['agents'] if spec['agents'] else 'todos os disponíveis'}")
    print(
        f"  Heurísticas  : {spec['heuristics'] if spec['heuristics'] else 'todas'}"
        f"{' (desativadas)' if spec['no_heuristics'] else ''}"
    )
    print(
        f"  Modelos RL   : {spec['models'] if spec['models'] else 'descoberta automática'}"
        f"{' (desativados)' if spec['no_rl'] else ''}"
    )
    if lowest_variants:
        print("  Lowest:")
        for variant in lowest_variants:
            print(f"    - {optimizer_catalog.lowest_result_key(variant.cost_function)}")
    else:
        print("  Lowest       : (nenhum --lowest)")
    if rollout_variants:
        print("  Rollouts:")
        for variant in rollout_variants:
            key = optimizer_catalog.rollout_result_key(
                variant.base_optimizer,
                variant.cost_function,
                horizon=variant.horizon,
                alpha=variant.alpha,
                terminal=variant.terminal,
            )
            print(f"    - {key}")
    else:
        print("  Rollouts     : (nenhum --rollout)")
    print(f"  Runs         : {spec['num_runs']} | Seed: {spec['seed']}")
    print(f"  Workers      : {spec['num_workers']}")
    print(f"  Modo experim.: {spec['experiment_mode']}")
    print(f"  Model base   : {spec['model_base_dir']}")
    if spec["experiment_mode"] == "cross_scenario":
        print(f"  Train scenario: {spec['train_scenario']}")
    print(f"  Plots indiv. : {'desativados' if not save_individual_plots else 'ativados'}")
    print(f"  Plot médias  : {'desativado' if not save_mean_plots else 'ativado'}")
    print(f"  Formato métr.: {spec['metrics_fmt']}")

    heuristics_filter = set(spec["heuristics"] or [])
    agents_filter = set(spec["agents"] or [])

    def _heuristic_selected(name: str) -> bool:
        return (
            not spec["no_heuristics"]
            and (not agents_filter or name in agents_filter)
            and (not heuristics_filter or name in heuristics_filter)
        )

    rollout_selected = _heuristic_selected("rollout")
    lowest_selected = _heuristic_selected("lowest")

    explicit_lowest = (
        ("lowest" in agents_filter) or ("lowest" in heuristics_filter)
    )
    explicit_rollout = (
        ("rollout" in agents_filter) or ("rollout" in heuristics_filter)
    )

    if spec["rollout"] and not rollout_selected:
        parser.error(
            "Erro: --rollout só pode ser usado se rollout estiver entre os agentes"
        )
    if explicit_rollout and not rollout_variants:
        parser.error(
            "Erro: rollout selecionado exige ao menos um --rollout "
            "(ex.: --rollout base=lowest,cost=weighted_score,horizon=5,terminal=0)"
        )
    if spec["lowest"] and not lowest_selected:
        parser.error(
            "Erro: --lowest só pode ser usado se lowest estiver na seleção"
        )
    if explicit_lowest and not lowest_variants:
        parser.error(
            "Erro: lowest selecionado exige ao menos um --lowest "
            "(ex.: --lowest cost=route)"
        )

    heuristic_names = set(optimizer_catalog.keys(is_heuristic=True)) | set(
        optimizer_catalog.cli_choices(is_heuristic=True)
    )
    may_run_rl = not spec["no_rl"] and (
        not spec["agents"] or any(name not in heuristic_names for name in spec["agents"])
    )
    if may_run_rl and spec["experiment_mode"] == "cross_scenario":
        expected_dir = os.path.join(
            spec["model_base_dir"], spec["train_scenario"], DEFAULT_MODEL_SUBDIR
        )
        if not os.path.isdir(expected_dir):
            parser.error(
                f"--train-scenario '{spec['train_scenario']}': "
                "diretório de modelos não encontrado.\n"
                f"  Esperado: {expected_dir}\n"
                "  Verifique --model-base-dir e --train-scenario."
            )

    run_json_path = exp.write_run_json(spec)
    print(f"  run.json     : {run_json_path}")

    wall_t0 = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    completed_result_keys: list[str] = []

    try:
        for objective in spec["objectives"]:
            for scenario in spec["scenarios"]:
                print(
                    f"\n\n=== Iniciando avaliações para Objetivo {objective} "
                    f"no cenário '{scenario}' ==="
                )

                if spec["experiment_mode"] == "same_scenario":
                    train_scenario = scenario
                else:
                    train_scenario = spec["train_scenario"]

                try:
                    available = optimizer_catalog.available_agents(
                        spec["model_base_dir"],
                        train_scenario,
                        objective,
                    )
                    agents = select_agents_for_run(
                        available,
                        spec["agents"],
                        spec["heuristics"],
                        spec["models"],
                        spec["no_heuristics"],
                        spec["no_rl"],
                    )
                except KeyError as exc:
                    parser.error(str(exc))

                if not agents:
                    print("[AVISO] Nenhum agente selecionado para este objetivo.")
                    continue

                # lowest/rollout só entram com variantes explícitas
                if not lowest_variants:
                    agents = [a for a in agents if a.key != "lowest"]
                if not rollout_variants:
                    agents = [a for a in agents if a.key != "rollout"]

                if not agents:
                    print("[AVISO] Nenhum agente selecionado para este objetivo.")
                    continue

                try:
                    variants = optimizer_catalog.expand_evaluations(
                        agents,
                        lowest_variants=lowest_variants,
                        rollout_variants=rollout_variants,
                        record_decisions=spec["rollout_record_decisions"],
                    )
                except (KeyError, ValueError) as exc:
                    parser.error(str(exc))

                print(
                    f"  Execuções neste objetivo: "
                    f"{[variant.result_key for variant in variants]}"
                )
                run_agents(
                    experiment_name=spec["name"],
                    scenario=scenario,
                    variants=variants,
                    objective=objective,
                    num_runs=spec["num_runs"],
                    seed=spec["seed"],
                    save_individual_plots=save_individual_plots,
                    save_mean_plots=save_mean_plots,
                    metrics_fmt=spec["metrics_fmt"],
                    num_workers=spec["num_workers"],
                    completed_result_keys=completed_result_keys,
                )

                # Atualiza summary.csv após cada cenário
                all_keys = list(dict.fromkeys(
                    completed_result_keys
                    + [v.result_key for v in variants]
                ))
                exp.rebuild_summary_csv(
                    name=spec["name"],
                    objectives=spec["objectives"],
                    scenarios=spec["scenarios"],
                    agent_keys=all_keys,
                )

        # Summary final
        exp.rebuild_summary_csv(
            name=spec["name"],
            objectives=spec["objectives"],
            scenarios=spec["scenarios"],
            agent_keys=completed_result_keys,
        )
    finally:
        duration = time.perf_counter() - wall_t0
        finished_path = exp.finalize_run_json(
            spec["name"],
            started_at=started_at,
            duration_seconds=duration,
        )
        print(
            f"\n=== Avaliação concluída — {exp.experiment_dir(spec['name'])}/ "
            f"(duração={duration:.2f}s) ==="
        )
        print(f"  run.json     : {finished_path}")


if __name__ == "__main__":
    main()
