"""Catálogo único de optimizers usados pelos scripts.

Para cadastrar um optimizer novo: implemente a classe e acrescente um
`OptimizerSpec` em `_SPECS`. Chave, aliases de CLI, labels e instanciação
vêm daqui, os scripts não precisam ser atualizados um a um.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from food_delivery_gym.main.cost.marginal_route_cost_function import MarginalRouteCostFunction
from food_delivery_gym.main.cost.route_cost_function import RouteCostFunction
from food_delivery_gym.main.cost.weighted_score_cost_function import WeightedScoreCostFunction
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
from food_delivery_gym.main.optimizer.optimizer_gym.rl_model_optimizer_gym import (
    RLModelOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.rollout_optimizer_gym import (
    RolloutOptimizerGym,
)

DEFAULT_ROLLOUT_ALPHA = 0.9
DEFAULT_ROLLOUT_HORIZON = 5
DEFAULT_ROLLOUT_RECORD_DECISIONS = False
DEFAULT_ROLLOUT_BASE = "nearest"

Builder = Callable[..., OptimizerGym]


@dataclass(frozen=True)
class CostFunctionSpec:
    key: str
    label: str
    short_label: str
    cls: type
    result_key: str
    result_label: str
    result_short_label: str
    uses_objective: bool = True


_COST_FUNCTIONS: tuple[CostFunctionSpec, ...] = (
    CostFunctionSpec(
        key="route",
        label="Custo de Rota",
        short_label="Custo Rota",
        cls=RouteCostFunction,
        result_key="lowest_route_cost",
        result_label="Motorista de Menor Custo de Rota",
        result_short_label="Menor Custo",
    ),
    CostFunctionSpec(
        key="marginal_route",
        label="Custo Marginal de Rota",
        short_label="Custo Marg.",
        cls=MarginalRouteCostFunction,
        result_key="lowest_marginal_route_cost",
        result_label="Motorista de Menor Custo Marginal de Rota",
        result_short_label="Menor Custo Marg.",
    ),
    CostFunctionSpec(
        key="weighted_score",
        label="Score Ponderado",
        short_label="Score Ponderado",
        cls=WeightedScoreCostFunction,
        result_key="lowest_weighted_score",
        result_label="Motorista de Score Ponderado",
        result_short_label="Score Ponderado",
        uses_objective=False,
    ),
)

COST_FUNCTION_CHOICES = tuple(spec.key for spec in _COST_FUNCTIONS)
_COST_BY_KEY: dict[str, CostFunctionSpec] = {spec.key: spec for spec in _COST_FUNCTIONS}


def cost_function_choices() -> list[str]:
    return list(COST_FUNCTION_CHOICES)


def get_cost_function(name: str) -> CostFunctionSpec:
    try:
        return _COST_BY_KEY[name]
    except KeyError as exc:
        raise ValueError(
            f"Cost function '{name}' inválida. Opções: {COST_FUNCTION_CHOICES}"
        ) from exc


def make_cost_function(name: str, objective: int | None = None):
    """Instancia a cost function a partir do nome de CLI e do objetivo do ambiente."""
    spec = get_cost_function(name)
    if spec.uses_objective:
        if objective is None:
            raise ValueError(
                f"objective é obrigatório para a cost function '{spec.key}'"
            )
        instance = spec.cls(objective=spec.cls.get_cost_objective(objective))
    else:
        instance = spec.cls()
    instance.label = spec.label
    return instance


def _env_only(cls: type[OptimizerGym]) -> Builder:
    def build(env, objective: int | None = None, **_extras) -> OptimizerGym:
        return cls(env)

    return build


def _lowest(env, objective: int | None = None, **extras) -> OptimizerGym:
    cost_function_name = extras.get("cost_function")
    if not cost_function_name:
        raise ValueError("LowestCostDriverOptimizerGym requer cost_function")
    return LowestCostDriverOptimizerGym(
        env,
        cost_function=make_cost_function(cost_function_name, objective),
    )


def _rollout(env, objective: int | None = None, **extras) -> OptimizerGym:
    base_name = extras.get("base_optimizer", DEFAULT_ROLLOUT_BASE)
    cost_function = extras.get("cost_function")
    base_cls, base_kwargs = constructor_args(
        base_name,
        objective=objective,
        cost_function=cost_function,
    )
    horizon = extras["horizon"] if "horizon" in extras else DEFAULT_ROLLOUT_HORIZON
    return RolloutOptimizerGym(
        env,
        base_optimizer_cls=base_cls,
        base_optimizer_kwargs=base_kwargs,
        alpha=extras.get("alpha", DEFAULT_ROLLOUT_ALPHA),
        horizon=horizon,
        record_decisions=extras.get("record_decisions", DEFAULT_ROLLOUT_RECORD_DECISIONS),
    )


def _rl(env, objective: int | None = None, **extras) -> OptimizerGym:
    model = extras.get("model")
    if model is None:
        raise ValueError("RLModelOptimizerGym requer model")
    return RLModelOptimizerGym(env, model)


@dataclass(frozen=True)
class OptimizerSpec:
    key: str
    label: str
    short_label: str
    aliases: tuple[str, ...] = ()
    cls: type[OptimizerGym] | None = None
    builder: Builder | None = None
    requires: tuple[str, ...] = ()
    is_heuristic: bool = True
    rollout_base: bool = True


@dataclass(frozen=True)
class EvalVariant:
    spec: OptimizerSpec
    result_key: str
    extras: dict[str, Any]


# Ordem de inserção é a ordem canônica de relatórios (plots, tabelas, boxplots).
_SPECS: tuple[OptimizerSpec, ...] = (
    OptimizerSpec(
        key="random",
        aliases=("random",),
        label="Motorista Aleatório",
        short_label="Aleatório",
        cls=RandomDriverOptimizerGym,
        builder=_env_only(RandomDriverOptimizerGym),
    ),
    OptimizerSpec(
        key="first_driver",
        aliases=("first", "first_driver"),
        label="Primeiro Motorista",
        short_label="Primeiro Mot.",
        cls=FirstDriverOptimizerGym,
        builder=_env_only(FirstDriverOptimizerGym),
    ),
    OptimizerSpec(
        key="nearest_driver",
        aliases=("nearest", "nearest_driver"),
        label="Motorista mais Próximo",
        short_label="Mot. Próximo",
        cls=NearestDriverOptimizerGym,
        builder=_env_only(NearestDriverOptimizerGym),
    ),
    OptimizerSpec(
        key="lowest",
        aliases=("lowest",),
        label="Motorista de Menor Custo",
        short_label="Menor Custo",
        cls=LowestCostDriverOptimizerGym,
        builder=_lowest,
        requires=("cost_function",),
    ),
    OptimizerSpec(
        key="rollout",
        aliases=("rollout",),
        label="Rollout",
        short_label="Rollout",
        cls=RolloutOptimizerGym,
        builder=_rollout,
        rollout_base=False,
    ),
    OptimizerSpec(
        key="rl",
        aliases=("rl",),
        label="Aprendizado por Reforço",
        short_label="PPO",
        cls=RLModelOptimizerGym,
        builder=_rl,
        requires=("model",),
        is_heuristic=False,
        rollout_base=False,
    ),
)

CATALOG: dict[str, OptimizerSpec] = {spec.key: spec for spec in _SPECS}


def _matching(name: str) -> list[OptimizerSpec]:
    return [
        spec
        for spec in CATALOG.values()
        if name == spec.key or name in spec.aliases
    ]


def resolve_key(name: str, cost_function: str | None = None) -> str:
    """Traduz um nome de CLI ou alias para a chave canônica do catálogo.

    `cost_function` é ignorado: `lowest` é um único optimizer e a função
    de custo é parâmetro de instanciação, não de seleção do spec.
    """
    del cost_function
    matches = _matching(name)
    if not matches:
        known = ", ".join(cli_choices())
        raise KeyError(f"Otimizador '{name}' não reconhecido. Opções: {known}")
    if len(matches) > 1:
        raise KeyError(f"Otimizador '{name}' é ambíguo: {[spec.key for spec in matches]}")
    return matches[0].key


def get(name: str, cost_function: str | None = None) -> OptimizerSpec:
    return CATALOG[resolve_key(name, cost_function)]


def cli_label(name: str) -> str:
    """Label para help de CLI."""
    return get(name).label


def requires(name: str) -> tuple[str, ...]:
    """Dependências da CLI para `name` (alias ou chave)."""
    return get(name).requires


def instantiate(spec: OptimizerSpec, env, objective: int | None = None, **extras) -> OptimizerGym:
    """Instancia um spec do catálogo ou um spec de modelo descoberto."""
    if spec.builder is None:
        raise ValueError(f"Otimizador '{spec.key}' não é instanciável pelo catálogo")
    return spec.builder(env, objective, **extras)


def build(name: str, env, objective: int | None = None, **extras) -> OptimizerGym:
    spec = get(name, cost_function=extras.get("cost_function"))
    return instantiate(spec, env, objective, **extras)


def constructor_args(
    name: str,
    objective: int | None = None,
    cost_function: str | None = None,
) -> tuple[type[OptimizerGym], dict[str, Any]]:
    """Retorna (classe, kwargs) sem o ambiente, usado como política de base do rollout."""
    spec = get(name)
    if spec.cls is None:
        raise ValueError(f"Otimizador '{spec.key}' não tem classe registrada")
    kwargs: dict[str, Any] = {}
    if "cost_function" in spec.requires:
        if not cost_function:
            raise ValueError(f"Otimizador '{spec.key}' requer cost_function")
        kwargs["cost_function"] = make_cost_function(cost_function, objective)
    return spec.cls, kwargs


def keys(*, is_heuristic: bool | None = None, rollout_base: bool | None = None) -> list[str]:
    selected = []
    for spec in CATALOG.values():
        if is_heuristic is not None and spec.is_heuristic != is_heuristic:
            continue
        if rollout_base is not None and spec.rollout_base != rollout_base:
            continue
        selected.append(spec.key)
    return selected


def labels(*, short: bool = False) -> dict[str, str]:
    return {
        spec.key: spec.short_label if short else spec.label
        for spec in CATALOG.values()
    }


def cli_choices(
    *,
    is_heuristic: bool | None = None,
    rollout_base: bool | None = None,
) -> list[str]:
    """Nomes aceitos na CLI, na ordem do catálogo, sem aliases duplicados."""
    seen: list[str] = []
    for key in keys(is_heuristic=is_heuristic, rollout_base=rollout_base):
        spec = CATALOG[key]
        name = spec.aliases[0] if spec.aliases else spec.key
        if name not in seen:
            seen.append(name)
    return seen


def needs_cost_function(name: str) -> bool:
    return "cost_function" in requires(name)


def lowest_result_key(cost_function: str) -> str:
    return get_cost_function(cost_function).result_key


def rollout_result_key(base_optimizer: str, cost_function: str | None = None) -> str:
    base = get(base_optimizer)
    if needs_cost_function(base.key):
        if not cost_function:
            raise ValueError(f"Base '{base.key}' requer cost_function")
        return f"rollout_{lowest_result_key(cost_function)}"
    return f"rollout_{base.key}"


def base_variant_key(base_optimizer: str, cost_function: str | None = None) -> str:
    """Chave da variante da política de base (ex.: nearest_driver, lowest_weighted_score)."""
    base = get(base_optimizer)
    if needs_cost_function(base.key):
        if not cost_function:
            raise ValueError(f"Base '{base.key}' requer cost_function")
        return lowest_result_key(cost_function)
    return base.key


def base_variant_key_for_instance(base_cls: type, base_kwargs: dict[str, Any]) -> str | None:
    """
    Chave da variante a partir da classe/kwargs usados pelo rollout.

    Retorna None se a classe não estiver no catálogo ou a cost function
    não for reconhecida.
    """
    spec = next((s for s in CATALOG.values() if s.cls is base_cls), None)
    if spec is None:
        return None
    if not needs_cost_function(spec.key):
        return spec.key
    cost_function = base_kwargs.get("cost_function")
    cost_spec = next(
        (c for c in _COST_FUNCTIONS if cost_function is not None and type(cost_function) is c.cls),
        None,
    )
    if cost_spec is None:
        return None
    return cost_spec.result_key


def _normalize_cost_functions(names: list[str]) -> list[str]:
    selected: list[str] = []
    for name in names:
        spec = get_cost_function(name)
        if spec.key not in selected:
            selected.append(spec.key)
    return selected


def normalize_base_optimizers(names: list[str]) -> list[str]:
    selected: list[str] = []
    known = ", ".join(cli_choices(rollout_base=True))
    for name in names:
        spec = get(name)
        if not spec.rollout_base:
            raise ValueError(
                f"Otimizador '{name}' não pode ser base do rollout. Opções: {known}"
            )
        if spec.key not in selected:
            selected.append(spec.key)
    return selected


def expand_evaluations(
    specs: list[OptimizerSpec],
    *,
    cost_functions: list[str],
    base_optimizers: list[str],
    rollout_alpha: float = DEFAULT_ROLLOUT_ALPHA,
    rollout_horizon: int | None = DEFAULT_ROLLOUT_HORIZON,
    record_decisions: bool = DEFAULT_ROLLOUT_RECORD_DECISIONS,
) -> list[EvalVariant]:
    """Expande lowest e rollout em uma execução por cost function e/ou base."""
    cost_names = _normalize_cost_functions(cost_functions)
    bases = normalize_base_optimizers(base_optimizers)
    variants: list[EvalVariant] = []

    for spec in specs:
        if spec.key == "lowest":
            for cost_name in cost_names:
                variants.append(
                    EvalVariant(
                        spec=spec,
                        result_key=lowest_result_key(cost_name),
                        extras={"cost_function": cost_name},
                    )
                )
            continue

        if spec.key == "rollout":
            rollout_extras = {
                "alpha": rollout_alpha,
                "horizon": rollout_horizon,
                "record_decisions": record_decisions,
            }
            for base_key in bases:
                if needs_cost_function(base_key):
                    for cost_name in cost_names:
                        variants.append(
                            EvalVariant(
                                spec=spec,
                                result_key=rollout_result_key(base_key, cost_name),
                                extras={
                                    **rollout_extras,
                                    "base_optimizer": base_key,
                                    "cost_function": cost_name,
                                },
                            )
                        )
                else:
                    variants.append(
                        EvalVariant(
                            spec=spec,
                            result_key=rollout_result_key(base_key),
                            extras={**rollout_extras, "base_optimizer": base_key},
                        )
                    )
            continue

        variants.append(EvalVariant(spec=spec, result_key=spec.key, extras={}))

    return variants


def result_labels(*, short: bool = False) -> dict[str, str]:
    """Chaves de pasta de resultado → label. Inclui variantes de lowest e rollout."""
    labels_by_key: dict[str, str] = {}
    for spec in CATALOG.values():
        if spec.key == "lowest":
            for cost in _COST_FUNCTIONS:
                labels_by_key[cost.result_key] = (
                    cost.result_short_label if short else cost.result_label
                )
            continue
        if spec.key == "rollout":
            for base in CATALOG.values():
                if not base.rollout_base:
                    continue
                if needs_cost_function(base.key):
                    for cost in _COST_FUNCTIONS:
                        key = rollout_result_key(base.key, cost.key)
                        cost_label = (
                            cost.result_short_label if short else cost.result_label
                        )
                        labels_by_key[key] = f"Rollout ({cost_label})"
                else:
                    base_label = base.short_label if short else base.label
                    labels_by_key[rollout_result_key(base.key)] = f"Rollout ({base_label})"
            continue
        labels_by_key[spec.key] = spec.short_label if short else spec.label
    return labels_by_key


def result_keys() -> list[str]:
    return list(result_labels().keys())


# ── Modelos de aprendizado por reforço ───────────────────────────────────────

DEFAULT_MODEL_ROOT = "./data/ppo_training"
DEFAULT_MODEL_SUBDIR = "treinamento"
RL_ALGOS = ("ppo", "sac", "a2c", "td3", "dqn")
_ALGO_CONFIG_NAMES = ("config.yml", "config.yaml", "args.yml", "args.yaml")
_ALGO_FIELD = re.compile(r"(?m)^algo:\s*([A-Za-z0-9_]+)\s*$")


@dataclass(frozen=True)
class DiscoveredModel:
    name: str
    algo: str
    key: str
    path: str
    model_dir: str
    scenario: str
    objective: int


def model_key(algo: str, name: str) -> str:
    """Chave de resultado: 18M_steps + ppo → ppo_18M_steps."""
    prefix = f"{algo}_"
    if name.startswith(prefix):
        return name
    return f"{prefix}{name}"


def rl_result_label(dir_name: str) -> str | None:
    """Label de pasta de resultado de modelo (ppo_18M_steps, sac_1M, ...)."""
    for algo in RL_ALGOS:
        prefix = f"{algo}_"
        if dir_name.startswith(prefix):
            return f"{algo.upper()} — {dir_name[len(prefix):]}"
    return None


def _algo_classes() -> dict[str, type]:
    from stable_baselines3 import A2C, DQN, PPO, SAC, TD3

    return {
        "ppo": PPO,
        "sac": SAC,
        "a2c": A2C,
        "td3": TD3,
        "dqn": DQN,
    }


def _parse_algo_field(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    match = _ALGO_FIELD.search(text)
    if not match:
        return None
    algo = match.group(1).lower()
    return algo if algo in RL_ALGOS else None


def _algo_from_config(start_dir: str, stop_at: str | None) -> str | None:
    current = os.path.abspath(start_dir)
    stop = os.path.abspath(stop_at) if stop_at else None
    while True:
        for filename in _ALGO_CONFIG_NAMES:
            algo = _parse_algo_field(os.path.join(current, filename))
            if algo:
                return algo
        if stop is not None and current == stop:
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        if stop is not None and not (
            current == stop or current.startswith(stop + os.sep)
        ):
            break
        current = parent
    return None


def _algo_from_name(name: str) -> str | None:
    lower = name.lower()
    for algo in RL_ALGOS:
        if lower == algo or lower.startswith(f"{algo}_"):
            return algo
    return None


def _under_ppo_training(path: str, search_root: str | None) -> bool:
    root_name = os.path.basename(os.path.abspath(search_root).rstrip(os.sep)) if search_root else ""
    if root_name == "ppo_training":
        return True
    return "ppo_training" in os.path.abspath(path).split(os.sep)


def detect_algo(model_dir: str, search_root: str | None = None) -> str:
    """Descobre o algoritmo SB3 de um diretório de modelo."""
    algo = _algo_from_config(model_dir, search_root)
    if algo:
        return algo
    algo = _algo_from_name(os.path.basename(model_dir))
    if algo:
        return algo
    if _under_ppo_training(model_dir, search_root):
        return "ppo"
    raise ValueError(
        f"Não foi possível detectar o algoritmo de '{model_dir}'. "
        "Inclua algo: ppo|sac|a2c|td3|dqn em config.yml/args.yml "
        "ou prefixe a pasta (ex.: sac_1M)."
    )


def find_vecnormalize(model_dir: str) -> str | None:
    """Procura vecnormalize.pkl de forma recursiva (rl-zoo3 salva num subdiretório)."""
    for root, _dirs, files_found in os.walk(model_dir):
        if "vecnormalize.pkl" in files_found:
            return os.path.join(root, "vecnormalize.pkl")
    return None


def load_rl_optimizer(env, model_path: str, search_root: str | None = None) -> RLModelOptimizerGym:
    """Carrega qualquer modelo SB3, aplica VecNormalize se existir e devolve o optimizer."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    model_dir = os.path.dirname(model_path)
    algo = detect_algo(model_dir, search_root=search_root)
    algo_cls = _algo_classes()[algo]
    print(f"  [{algo.upper()}] Carregando modelo: {model_path}")
    model = algo_cls.load(model_path)

    vec_env = DummyVecEnv([lambda bound=env: bound])
    vecnormalize_path = find_vecnormalize(model_dir)
    if vecnormalize_path:
        print(f"  [VecNormalize] Carregando: {vecnormalize_path}")
        rl_env = VecNormalize.load(vecnormalize_path, vec_env)
        rl_env.training = False
        rl_env.norm_reward = False
    else:
        print("  [VecNormalize] Não encontrado — usando ambiente sem normalização.")
        rl_env = vec_env

    return RLModelOptimizerGym(rl_env, model)


def discover_rl_models(
    root: str,
    scenario: str,
    objective: int,
    subdir: str = DEFAULT_MODEL_SUBDIR,
) -> list[DiscoveredModel]:
    """Varre root/<cenário>/treinamento/obj_N/<nome>/best_model.zip."""
    search_root = os.path.join(root, scenario, subdir, f"obj_{objective}")
    found: list[DiscoveredModel] = []
    if not os.path.isdir(search_root):
        return found

    for entry in sorted(os.scandir(search_root), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        model_path = os.path.join(entry.path, "best_model.zip")
        if not os.path.isfile(model_path):
            continue
        algo = detect_algo(entry.path, search_root=root)
        found.append(
            DiscoveredModel(
                name=entry.name,
                algo=algo,
                key=model_key(algo, entry.name),
                path=model_path,
                model_dir=entry.path,
                scenario=scenario,
                objective=objective,
            )
        )
    return found


def match_rl_model(models: list[DiscoveredModel], name: str) -> DiscoveredModel | None:
    """Aceita a chave (ppo_18M_steps) ou o nome da pasta (18M_steps)."""
    for model in models:
        if name in (model.key, model.name):
            return model
    return None


def filter_rl_models(
    models: list[DiscoveredModel],
    names: list[str] | None,
) -> list[DiscoveredModel]:
    if not names:
        return models
    selected = []
    known = {model.key for model in models} | {model.name for model in models}
    for name in names:
        model = match_rl_model(models, name)
        if model is None:
            raise KeyError(
                f"Modelo '{name}' não encontrado. Opções: {sorted(known)}"
            )
        selected.append(model)
    return selected


def _rl_discovered_builder(model_path: str, search_root: str) -> Builder:
    def build(env, objective: int | None = None, **_extras) -> OptimizerGym:
        return load_rl_optimizer(env, model_path, search_root=search_root)

    return build


def spec_from_discovered(model: DiscoveredModel, search_root: str) -> OptimizerSpec:
    """Spec executável de um zip descoberto. Não entra no catálogo estático."""
    return OptimizerSpec(
        key=model.key,
        aliases=(model.name, model.key),
        label=rl_result_label(model.key) or model.key,
        short_label=model.key,
        cls=RLModelOptimizerGym,
        builder=_rl_discovered_builder(model.path, search_root),
        is_heuristic=False,
        rollout_base=False,
    )


def available_agents(root: str, scenario: str, objective: int) -> list[OptimizerSpec]:
    """Heurísticas do catálogo e modelos RL encontrados para o cenário/objetivo."""
    agents = [spec for spec in CATALOG.values() if spec.is_heuristic]
    for model in discover_rl_models(root, scenario, objective):
        agents.append(spec_from_discovered(model, root))
    return agents


def select_agents(agents: list[OptimizerSpec], names: list[str]) -> list[OptimizerSpec]:
    """Filtra specs por chave ou alias. Aceita pasta do modelo (18M_steps) ou chave (ppo_18M_steps)."""
    selected: list[OptimizerSpec] = []
    known = [spec.key for spec in agents]
    for name in names:
        matches = [spec for spec in agents if name == spec.key or name in spec.aliases]
        if not matches:
            raise KeyError(f"Agente '{name}' não encontrado. Opções: {known}")
        selected.append(matches[0])
    return selected
