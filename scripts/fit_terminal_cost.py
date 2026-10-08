import argparse
import re
import time
from pathlib import Path

import numpy as np

from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer import catalog
from food_delivery_gym.main.optimizer.terminal_cost.linear_model import (
    DEFAULT_ROOT,
    fit_linear_model,
    save_linear_model,
)
from food_delivery_gym.main.scenarios import get_all_scenarios
from scripts.collect_terminal_cost import expand_variants

ALL_SCENARIOS = get_all_scenarios()
_OBJ_RE = re.compile(r"^obj_(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ajusta a regressão linear do custo terminal a partir de samples.npz.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_ROOT),
        help=f"Diretório raiz com samples.npz. Padrão: {DEFAULT_ROOT}.",
    )
    parser.add_argument(
        "--scenarios", "-s",
        nargs="+",
        choices=ALL_SCENARIOS,
        default=None,
        metavar="SCENARIO",
        help=f"Filtra cenários. Opções: {ALL_SCENARIOS}. Padrão: todos encontrados.",
    )
    parser.add_argument(
        "--objectives", "-o",
        nargs="+",
        type=int,
        choices=FoodDeliveryGymEnv.REWARD_OBJECTIVES,
        default=None,
        metavar="N",
        help="Filtra objetivos. Padrão: todos encontrados.",
    )
    parser.add_argument(
        "--bases", "-b",
        nargs="+",
        choices=catalog.cli_choices(rollout_base=True),
        default=None,
        metavar="BASE",
        help=(
            "Filtra políticas de base.\n"
            f"Opções: {catalog.cli_choices(rollout_base=True)}. Padrão: todas encontradas.\n"
            "'lowest' expande nas cost functions de --cost-functions."
        ),
    )
    parser.add_argument(
        "--cost-functions", "-c",
        nargs="+",
        choices=catalog.COST_FUNCTION_CHOICES,
        default=None,
        metavar="COST",
        help=(
            "Cost functions usadas quando a base é lowest.\n"
            f"Opções: {list(catalog.COST_FUNCTION_CHOICES)}. Padrão: todas."
        ),
    )
    return parser.parse_args()


def parse_sample_path(path: Path, root: Path) -> tuple[str, int, str] | None:
    """Extrai (scenario, objective, base_key) de .../scenario/obj_N/base_key/samples.npz."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) != 4 or relative.name != "samples.npz":
        return None
    scenario, obj_dir, base_key, _ = relative.parts
    match = _OBJ_RE.match(obj_dir)
    if match is None:
        return None
    return scenario, int(match.group(1)), base_key


def discover_samples(
    root: Path,
    scenarios: list[str] | None,
    objectives: list[int] | None,
    base_keys: set[str] | None,
) -> list[tuple[Path, str, int, str]]:
    found: list[tuple[Path, str, int, str]] = []
    for path in sorted(root.rglob("samples.npz")):
        parsed = parse_sample_path(path, root)
        if parsed is None:
            continue
        scenario, objective, base_key = parsed
        if scenarios is not None and scenario not in scenarios:
            continue
        if objectives is not None and objective not in objectives:
            continue
        if base_keys is not None and base_key not in base_keys:
            continue
        found.append((path, scenario, objective, base_key))
    return found


def r_squared(features: np.ndarray, returns: np.ndarray, coef: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    standardized = (features - mean) / std
    predictions = coef[0] + standardized @ coef[1:]
    residual = returns - predictions
    total = returns - returns.mean()
    return float(1.0 - (residual @ residual) / max((total @ total), 1e-12))


def fit_one(samples_file: Path, scenario: str, objective: int, base_key: str) -> tuple[Path, int, float]:
    with np.load(samples_file) as data:
        features = np.asarray(data["features"], dtype=np.float64)
        returns = np.asarray(data["returns"], dtype=np.float64)
        alpha = float(data["alpha"])

    coef, mean, std = fit_linear_model(features, returns)
    out_model = samples_file.with_name("linear_model.npz")
    save_linear_model(
        out_model, coef, mean, std,
        alpha=alpha, scenario=scenario, objective=objective, base_key=base_key,
    )
    return out_model, len(returns), r_squared(features, returns, coef, mean, std)


def main():
    args = parse_args()
    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"Diretório não encontrado: {root}")

    base_keys = None
    if args.bases is not None or args.cost_functions is not None:
        base_keys = {key for _, _, key in expand_variants(args.bases, args.cost_functions)}

    samples = discover_samples(root, args.scenarios, args.objectives, base_keys)
    if not samples:
        raise SystemExit(f"Nenhum samples.npz encontrado em {root}")

    print(f"Ajustando {len(samples)} variante(s) em {root}\n")
    for samples_file, scenario, objective, base_key in samples:
        start = time.perf_counter()
        out_model, n, score = fit_one(samples_file, scenario, objective, base_key)
        elapsed = time.perf_counter() - start
        print(
            f"[{scenario} | obj {objective} | {base_key}] "
            f"{n} amostras, R²={score:.4f}, {elapsed:.1f}s -> {out_model}"
        )


if __name__ == "__main__":
    main()
