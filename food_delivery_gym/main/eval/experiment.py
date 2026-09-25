"""Spec de experimento nomeado: YAML, layout de pastas, run.json e summary.csv."""

from __future__ import annotations

# Workers podem importar este módulo cedo via run_batch_eval → catalog.
from food_delivery_gym.main.eval.silence import in_worker_process, silence_eval_noise

if in_worker_process():
    silence_eval_noise(filter_stderr=True)

import csv
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer import catalog as optimizer_catalog
from food_delivery_gym.main.scenarios import get_all_scenarios, get_defaults_scenarios

DEFAULT_RUNS_ROOT = "./data/runs"
DEFAULT_NUM_RUNS = 20
DEFAULT_SEED = 123456789
DEFAULT_MODEL_BASE_DIR = "./data/ppo_training/"
DEFAULT_EXPERIMENT_MODE = "cross_scenario"
DEFAULT_TRAIN_SCENARIO = "medium"
DEFAULT_METRICS_FMT = "npz"
EXPERIMENT_MODES = ("cross_scenario", "same_scenario")
METRICS_FMT_OPTIONS = ("npz", "json")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Campos aceitos no YAML (além de name).
ALLOWED_YAML_KEYS = frozenset({
    "name",
    "objectives",
    "scenarios",
    "agents",
    "heuristics",
    "models",
    "no_rl",
    "no_heuristics",
    "lowest",
    "rollout",
    "rollout_record_decisions",
    "num_runs",
    "num_workers",
    "seed",
    "experiment_mode",
    "train_scenario",
    "model_base_dir",
    "metrics_fmt",
    "batch_plots",
    "all_plots",
})

# Defaults aplicados quando nem CLI nem YAML informam o campo.
FIELD_DEFAULTS: dict[str, Any] = {
    "objectives": None,  # resolvido para ALL_OBJECTIVES
    "scenarios": None,   # resolvido para get_defaults_scenarios()
    "agents": None,
    "heuristics": None,
    "models": None,
    "no_rl": False,
    "no_heuristics": False,
    "lowest": None,
    "rollout": None,
    "rollout_record_decisions": False,
    "num_runs": DEFAULT_NUM_RUNS,
    "num_workers": 1,
    "seed": DEFAULT_SEED,
    "experiment_mode": DEFAULT_EXPERIMENT_MODE,
    "train_scenario": DEFAULT_TRAIN_SCENARIO,
    "model_base_dir": DEFAULT_MODEL_BASE_DIR,
    "metrics_fmt": DEFAULT_METRICS_FMT,
    "batch_plots": False,
    "all_plots": False,
}


def package_version() -> str:
    try:
        return pkg_version("food_delivery_gym")
    except PackageNotFoundError:
        return "unknown"


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    commit = out.stdout.strip()
    return commit or None


def experiment_dir(name: str, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    return Path(runs_root) / name


def agent_dir(
    name: str,
    objective: int,
    scenario: str,
    agent: str,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> Path:
    return experiment_dir(name, runs_root) / f"obj_{objective}" / scenario / agent


def agent_complete(dir_path: str | Path) -> bool:
    """True se metrics_data.npz existe e abre com numpy.load."""
    npz_path = Path(dir_path) / "metrics_data.npz"
    if not npz_path.is_file():
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as raw:
            _ = list(raw.files)
        return True
    except Exception:
        return False


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name do experimento é obrigatório")
    name = name.strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            f"name inválido: '{name}'. Use letras, dígitos, '.', '_' ou '-' "
            "(até 128 caracteres, começando com alfanumérico)."
        )
    return name


def load_yaml_spec(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de experimento não encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML de experimento deve ser um mapeamento; recebido {type(data).__name__}")
    unknown = sorted(set(data) - ALLOWED_YAML_KEYS)
    if unknown:
        raise ValueError(f"Chaves desconhecidas no YAML: {unknown}")
    if "name" not in data:
        raise ValueError("YAML de experimento exige o campo 'name'")
    data["name"] = _validate_name(data["name"])
    return data


def _normalize_lowest_list(raw: Any) -> list[str] | None:
    """Converte YAML/CLI de lowest em lista de strings 'cost=...'."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("lowest deve ser uma lista de strings ou mapas {cost: ...}")
    specs: list[str] = []
    for item in raw:
        if isinstance(item, str):
            specs.append(item)
        elif isinstance(item, dict):
            if "cost" in item:
                specs.append(f"cost={item['cost']}")
            elif "cost_function" in item:
                specs.append(f"cost={item['cost_function']}")
            else:
                raise ValueError(
                    f"mapa de lowest inválido: {item} (esperado chave 'cost')"
                )
        else:
            raise ValueError(f"item de lowest inválido: {item!r}")
    return specs


def _normalize_rollout_list(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("rollout deve ser uma lista de strings ou mapas")
    specs: list[str] = []
    for item in raw:
        if isinstance(item, str):
            specs.append(item)
        elif isinstance(item, dict):
            parts = []
            for key, value in item.items():
                k = "cost" if key == "cost_function" else key
                parts.append(f"{k}={value}")
            specs.append(",".join(parts))
        else:
            raise ValueError(f"item de rollout inválido: {item!r}")
    return specs


def resolve_experiment(
    *,
    yaml_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resolve a spec final: CLI explícito > YAML > defaults.

    `cli_overrides` deve conter apenas chaves presentes na linha de comando
    (valores nenhuma/omitidos não devem ser incluídos, exceto store_true
    quando a flag foi passada).
    """
    yaml_data: dict[str, Any] = {}
    if yaml_path is not None:
        yaml_data = load_yaml_spec(yaml_path)

    overrides = dict(cli_overrides or {})
    # name: CLI --name, senão YAML, senão erro
    name = overrides.pop("name", None) or yaml_data.get("name")
    if name is None:
        raise ValueError(
            "name do experimento é obrigatório "
            "(passe um YAML com 'name' ou use --name)"
        )
    name = _validate_name(str(name))

    resolved: dict[str, Any] = {"name": name}

    for key, default in FIELD_DEFAULTS.items():
        if key in overrides and overrides[key] is not None:
            resolved[key] = overrides[key]
        elif key in yaml_data and yaml_data[key] is not None:
            resolved[key] = yaml_data[key]
        else:
            resolved[key] = default

    # Listas especiais
    resolved["lowest"] = _normalize_lowest_list(resolved.get("lowest"))
    resolved["rollout"] = _normalize_rollout_list(resolved.get("rollout"))

    all_objectives = list(FoodDeliveryGymEnv.REWARD_OBJECTIVES)
    all_scenarios = get_all_scenarios()
    default_scenarios = get_defaults_scenarios()

    if resolved["objectives"] is None:
        resolved["objectives"] = all_objectives
    else:
        objs = [int(o) for o in resolved["objectives"]]
        invalid = [o for o in objs if o not in all_objectives]
        if invalid:
            raise ValueError(f"objetivos inválidos: {invalid}")
        resolved["objectives"] = objs

    if resolved["scenarios"] is None:
        resolved["scenarios"] = list(default_scenarios)
    else:
        scenarios = [str(s) for s in resolved["scenarios"]]
        invalid = [s for s in scenarios if s not in all_scenarios]
        if invalid:
            raise ValueError(f"cenários inválidos: {invalid}. Opções: {all_scenarios}")
        resolved["scenarios"] = scenarios

    if resolved["experiment_mode"] not in EXPERIMENT_MODES:
        raise ValueError(
            f"experiment_mode inválido: '{resolved['experiment_mode']}'. "
            f"Opções: {EXPERIMENT_MODES}"
        )
    if resolved["metrics_fmt"] not in METRICS_FMT_OPTIONS:
        raise ValueError(
            f"metrics_fmt inválido: '{resolved['metrics_fmt']}'. "
            f"Opções: {METRICS_FMT_OPTIONS}"
        )
    if resolved["train_scenario"] not in all_scenarios:
        raise ValueError(
            f"train_scenario inválido: '{resolved['train_scenario']}'. "
            f"Opções: {all_scenarios}"
        )

    resolved["num_runs"] = int(resolved["num_runs"])
    resolved["num_workers"] = int(resolved["num_workers"])
    resolved["seed"] = int(resolved["seed"])
    resolved["no_rl"] = bool(resolved["no_rl"])
    resolved["no_heuristics"] = bool(resolved["no_heuristics"])
    resolved["rollout_record_decisions"] = bool(resolved["rollout_record_decisions"])
    resolved["batch_plots"] = bool(resolved["batch_plots"])
    resolved["all_plots"] = bool(resolved["all_plots"])
    if resolved["all_plots"]:
        resolved["batch_plots"] = True
    resolved["model_base_dir"] = str(resolved["model_base_dir"])

    if resolved["num_workers"] < 1:
        raise ValueError("--num-workers deve ser >= 1")
    if resolved["num_runs"] < 0:
        raise ValueError("num_runs deve ser >= 0")

    # Parseia variantes (falha cedo se formato inválido)
    lowest_variants = [
        optimizer_catalog.parse_lowest_cli(raw)
        for raw in (resolved["lowest"] or [])
    ]
    rollout_variants = [
        optimizer_catalog.parse_rollout_cli(raw)
        for raw in (resolved["rollout"] or [])
    ]
    resolved["_lowest_variants"] = lowest_variants
    resolved["_rollout_variants"] = rollout_variants

    return resolved


def write_run_json(
    spec: dict[str, Any],
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    started_at: datetime | None = None,
) -> Path:
    """Grava data/runs/<name>/run.json com a spec resolvida e metadados."""
    name = spec["name"]
    out_dir = experiment_dir(name, runs_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run.json"

    serializable = {
        k: v
        for k, v in spec.items()
        if not k.startswith("_")
    }
    # Serializa variantes parseadas de forma legível
    serializable["lowest"] = [
        {"cost": v.cost_function} for v in spec.get("_lowest_variants", [])
    ] or None
    serializable["rollout"] = [
        {
            "base": v.base_optimizer,
            "cost": v.cost_function,
            "horizon": v.horizon,
            "alpha": v.alpha,
            "terminal": v.terminal,
        }
        for v in spec.get("_rollout_variants", [])
    ] or None

    started = started_at or datetime.now(timezone.utc)
    payload = {
        "spec": serializable,
        "package_version": package_version(),
        "git_commit": git_commit(),
        "started_at": started.isoformat(),
        "finished_at": None,
        "duration_seconds": None,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def finalize_run_json(
    name: str,
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    duration_seconds: float | None = None,
) -> Path:
    """Atualiza run.json com finished_at e duration_seconds ao fim da avaliação."""
    path = experiment_dir(name, runs_root) / "run.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {"spec": {"name": name}}

    finished = finished_at or datetime.now(timezone.utc)
    payload["finished_at"] = finished.isoformat()

    if duration_seconds is not None:
        payload["duration_seconds"] = round(float(duration_seconds), 4)
    elif started_at is not None:
        payload["duration_seconds"] = round(
            (finished - started_at).total_seconds(), 4
        )
    elif payload.get("started_at"):
        try:
            started = datetime.fromisoformat(payload["started_at"])
            payload["duration_seconds"] = round(
                (finished - started).total_seconds(), 4
            )
        except ValueError:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def _agg_stat(aggregate: dict, metric: str, stat: str) -> float | int | None:
    block = aggregate.get(metric) or {}
    val = block.get(stat)
    if val is None:
        return None
    if stat == "n":
        return int(val)
    return float(val)


def summary_row_from_aggregate(
    *,
    objective: int,
    scenario: str,
    agent: str,
    aggregate: dict,
    truncated: int = 0,
) -> dict[str, Any]:
    n = _agg_stat(aggregate, "rewards", "n")
    return {
        "objective": objective,
        "scenario": scenario,
        "agent": agent,
        "n": n if n is not None else 0,
        "rewards_avg": _agg_stat(aggregate, "rewards", "avg"),
        "rewards_std": _agg_stat(aggregate, "rewards", "std_dev"),
        "delivery_time_avg": _agg_stat(aggregate, "delivery_time", "avg"),
        "delivery_time_std": _agg_stat(aggregate, "delivery_time", "std_dev"),
        "distance_avg": _agg_stat(aggregate, "distance", "avg"),
        "distance_std": _agg_stat(aggregate, "distance", "std_dev"),
        "truncated": truncated,
    }


def load_agent_summary_row(
    *,
    objective: int,
    scenario: str,
    agent: str,
    dir_path: str | Path,
) -> dict[str, Any] | None:
    """Lê summary.json (ou aggregate do npz) e monta uma linha do summary.csv."""
    dir_path = Path(dir_path)
    summary_path = dir_path / "summary.json"
    aggregate: dict | None = None
    truncated = 0

    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        aggregate = data.get("aggregate", data)
        truncated = int(data.get("truncated", 0) or 0)
    else:
        npz_path = dir_path / "metrics_data.npz"
        if not npz_path.is_file():
            return None
        try:
            with np.load(npz_path, allow_pickle=False) as raw:
                aggregate = {}
                for key in raw.files:
                    parts = key.split("__")
                    if parts[0] == "agg" and len(parts) == 3:
                        _, metric, stat = parts
                        val = raw[key]
                        aggregate.setdefault(metric, {})[stat] = (
                            int(val[0]) if stat == "n" else float(val[0])
                        )
                if "ep__truncated" in raw.files:
                    truncated = int(np.sum(raw["ep__truncated"]))
        except Exception:
            return None

    if not aggregate:
        return None
    return summary_row_from_aggregate(
        objective=objective,
        scenario=scenario,
        agent=agent,
        aggregate=aggregate,
        truncated=truncated,
    )


SUMMARY_CSV_FIELDS = [
    "objective",
    "scenario",
    "agent",
    "n",
    "rewards_avg",
    "rewards_std",
    "delivery_time_avg",
    "delivery_time_std",
    "distance_avg",
    "distance_std",
    "truncated",
]


def write_summary_csv(
    rows: list[dict[str, Any]],
    *,
    name: str,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> Path:
    out_dir = experiment_dir(name, runs_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in SUMMARY_CSV_FIELDS})
    return path


def resolve_agent_dir(
    results_dir: str | Path,
    objective: int,
    scenario: str,
    agent: str,
) -> Path | None:
    """
    Resolve o diretório do agente no layout novo ou no legado.

    Novo:  <results_dir>/obj_<N>/<scenario>/<agent>/
    Legado: <results_dir>/obj_<N>/<scenario>_scenario/<agent>/
    """
    results_dir = Path(results_dir)
    candidates = [
        results_dir / f"obj_{objective}" / scenario / agent,
        results_dir / f"obj_{objective}" / f"{scenario}_scenario" / agent,
    ]
    for path in candidates:
        if path.is_dir() and (
            (path / "metrics_data.npz").is_file()
            or (path / "metrics_data.json").is_file()
            or (path / "episodes.csv").is_file()
            or (path / "summary.json").is_file()
        ):
            return path
    return None


def discover_agent_names(
    results_dir: str | Path,
    objectives: list[int],
    scenarios: list[str],
) -> list[str]:
    """Descobre nomes de agentes nos layouts novo e legado."""
    results_dir = Path(results_dir)
    found: set[str] = set()
    for objective in objectives:
        for scenario in scenarios:
            bases = [
                results_dir / f"obj_{objective}" / scenario,
                results_dir / f"obj_{objective}" / f"{scenario}_scenario",
            ]
            for base in bases:
                if not base.is_dir():
                    continue
                for entry in base.iterdir():
                    if not entry.is_dir():
                        continue
                    if (
                        (entry / "metrics_data.npz").is_file()
                        or (entry / "metrics_data.json").is_file()
                        or (entry / "episodes.csv").is_file()
                        or (entry / "summary.json").is_file()
                    ):
                        found.add(entry.name)
    return optimizer_catalog.sort_discovered_result_dirs(found)


def rebuild_summary_csv(
    *,
    name: str,
    objectives: list[int],
    scenarios: list[str],
    agent_keys: list[str],
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> Path:
    """Reescreve summary.csv a partir dos agentes já presentes no disco."""
    rows: list[dict[str, Any]] = []
    for objective in objectives:
        for scenario in scenarios:
            for agent in agent_keys:
                path = agent_dir(name, objective, scenario, agent, runs_root)
                if not path.is_dir():
                    continue
                row = load_agent_summary_row(
                    objective=objective,
                    scenario=scenario,
                    agent=agent,
                    dir_path=path,
                )
                if row is not None:
                    rows.append(row)
    return write_summary_csv(rows, name=name, runs_root=runs_root)
