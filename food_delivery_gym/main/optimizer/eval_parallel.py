"""Avaliação de episódios em processos filhos (sem pickle do SimPy).

O processo pai envia apenas primitivas (`EvalJobSpec` + índice/seed).
Cada worker remonta ambiente + otimizador via catálogo, roda um episódio
e devolve um dict serializável com as métricas.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback
import types
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import numpy as np

# Silêncio ANTES de imports do projeto que puxam SB3/gym via outras rotas.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def silence_eval_noise() -> None:
    """
    Reduz spam de TensorFlow/oneDNN e do aviso legado do Gym.

    O aviso "Gym has been unmaintained..." vem de `gym_notices` via
    `print(..., file=sys.stderr)` no import de `gym` — filterwarnings não
    pega. Stubamos o pacote antes do primeiro import de gym.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    if "gym_notices.notices" not in sys.modules:
        pkg = types.ModuleType("gym_notices")
        notices_mod = types.ModuleType("gym_notices.notices")
        notices_mod.notices = {}  # type: ignore[attr-defined]
        sys.modules["gym_notices"] = pkg
        sys.modules["gym_notices.notices"] = notices_mod

    warnings.filterwarnings(
        "ignore",
        message=r".*Gym has been unmaintained since 2022.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The environment .* is out of date.*",
    )
    try:
        import gym.logger as gym_logger

        gym_logger.set_level(gym_logger.ERROR)
    except Exception:
        pass


silence_eval_noise()

from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.statistics.simulation_stats import SimulationStats


@dataclass(frozen=True)
class EvalJobSpec:
    """Payload serializável para reconstruir um otimizador no worker."""

    scenario: str
    objective: int
    optimizer_key: str
    extras: dict[str, Any] = field(default_factory=dict)
    model_path: str | None = None
    model_search_root: str | None = None
    # Caminho absoluto/relativo opcional (ex.: fixtures de teste). Se None,
    # resolve `scenario` dentro de food_delivery_gym.main.scenarios.
    scenario_path: str | None = None


@dataclass(frozen=True)
class EpisodeJob:
    """Um episódio a executar em paralelo."""

    spec: EvalJobSpec
    episode_idx: int
    seed: int


def derive_episode_seeds(seed: int | None, num_runs: int) -> list[int]:
    """Deriva uma semente estável por episódio via SeedSequence.spawn."""
    if num_runs < 0:
        raise ValueError(f"num_runs deve ser >= 0; recebido {num_runs}")
    if num_runs == 0:
        return []
    base = np.random.SeedSequence(seed)
    children = base.spawn(num_runs)
    return [int(child.generate_state(1)[0]) for child in children]


def create_eval_environment(
    reward_objective: int,
    scenario_name: str,
    scenario_path: str | None = None,
):
    """Cria um FoodDeliveryGymEnv em modo EVALUATING para o cenário dado."""
    if reward_objective not in FoodDeliveryGymEnv.REWARD_OBJECTIVES:
        raise ValueError(
            f"reward_objective deve ser um valor em {FoodDeliveryGymEnv.REWARD_OBJECTIVES}."
        )

    if scenario_path is None:
        scenario_file = f"{scenario_name}.json"
        scenario_path = str(
            files("food_delivery_gym.main.scenarios").joinpath(scenario_file)
        )
    FoodDeliveryGymEnv.set_scenario(scenario_path)
    return FoodDeliveryGymEnv(
        reward_objective=reward_objective,
        mode=EnvMode.EVALUATING,
    )


def _build_optimizer(env, spec: EvalJobSpec):
    from food_delivery_gym.main.optimizer import catalog as optimizer_catalog

    if spec.model_path:
        return optimizer_catalog.load_rl_optimizer(
            env,
            spec.model_path,
            search_root=spec.model_search_root,
        )
    return optimizer_catalog.build(
        spec.optimizer_key,
        env,
        spec.objective,
        **spec.extras,
    )


def run_episode_job(job: EpisodeJob) -> dict[str, Any]:
    """
    Worker top-level (pickleável com spawn/forkserver).

    Remonta ambiente + otimizador, roda um episódio e devolve métricas.
    """
    silence_eval_noise()
    idx = job.episode_idx + 1
    print(
        f"-> Iniciando episódio {idx} (pid={os.getpid()})...",
        flush=True,
    )
    try:
        env = create_eval_environment(
            job.spec.objective,
            job.spec.scenario,
            scenario_path=job.spec.scenario_path,
        )
        optimizer = _build_optimizer(env, job.spec)
        optimizer.prepare_episode(job.seed)
        resultado = optimizer.run()

        simpy_env = optimizer.gym_env.get_simpy_env()
        orders_generated = optimizer.gym_env.get_num_orders_generated()
        episode = SimulationStats.snapshot_episode(
            simpy_env=simpy_env,
            reward=resultado["sum_reward"],
            length=resultado["steps"],
            truncated=resultado["truncated"],
            orders_generated=orders_generated,
        )
        return {
            "episode_idx": job.episode_idx,
            "ok": True,
            "error": None,
            "episode": episode,
        }
    except Exception as exc:
        return {
            "episode_idx": job.episode_idx,
            "ok": False,
            "error": f"{exc}\n{traceback.format_exc()}",
            "episode": None,
        }


def _mp_context():
    """Prefere forkserver; cai para spawn se indisponível."""
    for method in ("forkserver", "spawn"):
        try:
            return mp.get_context(method)
        except ValueError:
            continue
    return mp.get_context()


def _terminate_executor(pool: ProcessPoolExecutor) -> None:
    """Encerra o pool sem esperar jobs: cancela futures e mata workers."""
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        # Python < 3.9 não tem cancel_futures
        pool.shutdown(wait=False)

    processes = getattr(pool, "_processes", None) or {}
    for proc in list(processes.values()):
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
    for proc in list(processes.values()):
        try:
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)
        except Exception:
            pass


def run_episodes_parallel(
    jobs: list[EpisodeJob],
    num_workers: int,
) -> list[dict[str, Any]]:
    """
    Executa jobs em um ProcessPool e devolve resultados ordenados por episode_idx.

    Imprime progresso em tempo real a cada episódio concluído.
    Em Ctrl+C / SIGTERM, mata os workers imediatamente (não deixa órfãos).
    """
    if not jobs:
        return []
    workers = max(1, min(num_workers, len(jobs)))
    total = len(jobs)
    ctx = _mp_context()
    results: dict[int, dict[str, Any]] = {}

    # Garante quietude no pai antes de subir o forkserver/spawn.
    silence_eval_noise()
    print(
        f"-> Paralelizando {total} episódios com {workers} worker(s)...",
        flush=True,
    )

    # Não usar `with ProcessPoolExecutor`: o __exit__ chama shutdown(wait=True)
    # e, no Ctrl+C, espera os workers terminarem em vez de matá-los.
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=silence_eval_noise,
    )
    try:
        futures = {
            pool.submit(run_episode_job, job): job.episode_idx for job in jobs
        }
        done = 0
        for future in as_completed(futures):
            result = future.result()
            results[result["episode_idx"]] = result
            done += 1
            idx = result["episode_idx"] + 1
            if result["ok"]:
                ep = result["episode"]
                print(
                    f"-> Concluído {done}/{total} "
                    f"(episódio {idx}): "
                    f"retorno={ep['reward']:.4f} | "
                    f"passos={ep['length']} | "
                    f"truncada={ep['truncated']}",
                    flush=True,
                )
            else:
                print(
                    f"-> Concluído {done}/{total} (episódio {idx}): ERRO",
                    flush=True,
                )
        pool.shutdown(wait=True)
    except BaseException:
        print("\n-> Interrompido — encerrando workers...")
        _terminate_executor(pool)
        raise

    return [results[i] for i in range(total)]
