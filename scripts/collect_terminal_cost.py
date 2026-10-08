from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer import catalog
from food_delivery_gym.main.optimizer.terminal_cost.features import FEATURE_NAMES, extract_features
from food_delivery_gym.main.optimizer.terminal_cost.linear_model import (
    discounted_returns,
    samples_path,
)
from food_delivery_gym.main.scenarios import get_all_scenarios, get_defaults_scenarios

ALL_SCENARIOS = get_all_scenarios()
DEFAULT_SCENARIOS = get_defaults_scenarios()
DEFAULT_OBJECTIVES = [3]
DEFAULT_EPISODES = 1000
DEFAULT_ALPHA = catalog.DEFAULT_ROLLOUT_ALPHA
DEFAULT_SEED = 123456789
DEFAULT_OUTPUT_DIR = "data/terminal_cost"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coleta dados de custo terminal (retorno restante) das políticas de base do rollout.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--scenarios", "-s",
        nargs="+",
        choices=ALL_SCENARIOS,
        default=DEFAULT_SCENARIOS,
        metavar="SCENARIO",
        help=f"Cenários a coletar. Opções: {ALL_SCENARIOS}. Padrão: {DEFAULT_SCENARIOS}.",
    )
    parser.add_argument(
        "--objectives", "-o",
        nargs="+",
        type=int,
        choices=FoodDeliveryGymEnv.REWARD_OBJECTIVES,
        default=DEFAULT_OBJECTIVES,
        metavar="N",
        help=f"Objetivos de recompensa. Padrão: {DEFAULT_OBJECTIVES}.",
    )
    parser.add_argument(
        "--bases", "-b",
        nargs="+",
        choices=catalog.cli_choices(rollout_base=True),
        default=None,
        metavar="BASE",
        help=(
            "Políticas de base do rollout a coletar.\n"
            f"Opções: {catalog.cli_choices(rollout_base=True)}. Padrão: todas.\n"
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
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Episódios por variante. Padrão: {DEFAULT_EPISODES}.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Fator de desconto do retorno (o mesmo do rollout). Padrão: {DEFAULT_ALPHA}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Semente raiz; cada episódio recebe uma semente derivada. Padrão: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Diretório raiz dos artefatos. Padrão: {DEFAULT_OUTPUT_DIR}.",
    )
    return parser.parse_args()


def scenario_path(name: str) -> str:
    return str(files("food_delivery_gym.main.scenarios") / f"{name}.json")


def expand_variants(
    bases: list[str] | None,
    cost_functions: list[str] | None,
) -> list[tuple[str, str | None, str]]:
    """[(base, cost_function|None, base_key)] com lowest filtrado por cost function."""
    base_keys = catalog.normalize_base_optimizers(bases) if bases else catalog.keys(rollout_base=True)
    selected_costs = list(cost_functions) if cost_functions else catalog.cost_function_choices()
    has_lowest = any(catalog.needs_cost_function(base) for base in base_keys)
    if cost_functions and not has_lowest:
        raise SystemExit("--cost-functions só se aplica quando a base inclui lowest")

    variants: list[tuple[str, str | None, str]] = []
    for base in base_keys:
        if catalog.needs_cost_function(base):
            for cost_name in selected_costs:
                variants.append((base, cost_name, catalog.base_variant_key(base, cost_name)))
        else:
            variants.append((base, None, catalog.base_variant_key(base)))
    return variants


def run_episode(env: FoodDeliveryGymEnv, optimizer, seed: int, alpha: float):
    """Roda um episódio da base e retorna (features, observações, recompensas, retornos)."""
    obs, _ = env.reset(seed=seed)
    features: list[np.ndarray] = []
    raw_obs: list[dict] = []
    rewards: list[float] = []

    terminated = truncated = False
    while not (terminated or truncated):
        features.append(extract_features(env))
        raw_obs.append(obs)

        order = env.get_current_order()
        action = optimizer.assign_driver_to_order(obs, order)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))

    return features, raw_obs, rewards, discounted_returns(rewards, alpha)


def collect_variant(
    env: FoodDeliveryGymEnv,
    base: str,
    cost_function: str | None,
    objective: int,
    episodes: int,
    alpha: float,
    seed_rng: np.random.Generator,
):
    extras = {"cost_function": cost_function} if cost_function else {}
    optimizer = catalog.build(base, env, objective, **extras)

    all_features: list[np.ndarray] = []
    all_returns: list[np.ndarray] = []
    all_episode_ids: list[np.ndarray] = []
    all_steps: list[np.ndarray] = []
    all_seeds: list[np.ndarray] = []
    obs_stacks: dict[str, list[np.ndarray]] = defaultdict(list)

    for episode_id in range(episodes):
        episode_seed = int(seed_rng.integers(0, 2**31 - 1))
        features, raw_obs, rewards, returns = run_episode(env, optimizer, episode_seed, alpha)
        if not rewards:
            continue

        n = len(rewards)
        all_features.append(np.stack(features))
        all_returns.append(returns)
        all_episode_ids.append(np.full(n, episode_id, dtype=np.int64))
        all_steps.append(np.arange(n, dtype=np.int64))
        all_seeds.append(np.full(n, episode_seed, dtype=np.int64))
        for key in raw_obs[0]:
            obs_stacks[f"obs_{key}"].append(np.stack([step_obs[key] for step_obs in raw_obs]))

    if not all_features:
        raise RuntimeError(f"Nenhuma decisão coletada para base={base} objetivo={objective}")

    dataset = {
        "features": np.concatenate(all_features),
        "feature_names": np.array(FEATURE_NAMES),
        "returns": np.concatenate(all_returns),
        "episode_id": np.concatenate(all_episode_ids),
        "step": np.concatenate(all_steps),
        "seed": np.concatenate(all_seeds),
        "alpha": np.float64(alpha),
    }
    for key, chunks in obs_stacks.items():
        dataset[key] = np.concatenate(chunks)
    return dataset


def main():
    args = parse_args()
    root = Path(args.output_dir)
    variants = expand_variants(args.bases, args.cost_functions)

    print(f"Cenários: {args.scenarios} | Objetivos: {args.objectives}")
    print(f"Variantes de base: {[key for _, _, key in variants]}")
    print(f"Episódios por variante: {args.episodes} | alpha={args.alpha}\n")

    for scenario in args.scenarios:
        FoodDeliveryGymEnv.set_scenario(scenario_path(scenario))
        for objective in args.objectives:
            env = FoodDeliveryGymEnv(reward_objective=objective, mode=EnvMode.TESTING)
            for base, cost_function, base_key in variants:
                seed_rng = np.random.default_rng(args.seed)
                start = time.perf_counter()
                dataset = collect_variant(
                    env, base, cost_function, objective, args.episodes, args.alpha, seed_rng
                )

                out_samples = samples_path(scenario, objective, base_key, root)
                out_samples.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_samples, **dataset)
                elapsed = time.perf_counter() - start

                print(
                    f"[{scenario} | obj {objective} | {base_key}] "
                    f"{len(dataset['returns'])} amostras, {elapsed:.1f}s -> {out_samples}"
                )


if __name__ == "__main__":
    main()
