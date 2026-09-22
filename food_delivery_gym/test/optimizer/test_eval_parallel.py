"""Testes da avaliação paralela de episódios."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from food_delivery_gym.main.optimizer import catalog as optimizer_catalog
from food_delivery_gym.main.optimizer.eval_parallel import (
    EpisodeJob,
    EvalJobSpec,
    create_eval_environment,
    derive_episode_seeds,
    run_episode_job,
)
from food_delivery_gym.test.conftest import TINY


SEED = 42
OBJECTIVE = 3
NUM_RUNS = 2
TINY_PATH = str(TINY.resolve())


def _tiny_eval_spec(optimizer_key: str = "nearest_driver", **extras) -> EvalJobSpec:
    return EvalJobSpec(
        scenario="tiny_scenario",
        objective=OBJECTIVE,
        optimizer_key=optimizer_key,
        extras=extras,
        scenario_path=TINY_PATH,
    )


def _run_batch(num_workers: int, tmp_path: Path):
    env = create_eval_environment(
        OBJECTIVE, "tiny_scenario", scenario_path=TINY_PATH
    )
    optimizer = optimizer_catalog.build("nearest_driver", env, OBJECTIVE)
    out = tmp_path / f"workers_{num_workers}"
    out.mkdir()
    stats = optimizer.run_simulations(
        NUM_RUNS,
        str(out) + "/",
        seed=SEED,
        save_individual_plots=False,
        save_mean_plots=False,
        num_workers=num_workers,
        eval_spec=_tiny_eval_spec() if num_workers > 1 else None,
    )
    return [
        (ep["reward"], ep["length"], ep["truncated"], ep["orders_generated"])
        for ep in stats._raw_episodes
    ]


def test_derive_episode_seeds_stable_and_distinct():
    a = derive_episode_seeds(SEED, 5)
    b = derive_episode_seeds(SEED, 5)
    assert a == b
    assert len(a) == 5
    assert len(set(a)) == 5


def test_derive_episode_seeds_empty():
    assert derive_episode_seeds(SEED, 0) == []


def test_num_workers_requires_eval_spec(tmp_path):
    env = create_eval_environment(
        OBJECTIVE, "tiny_scenario", scenario_path=TINY_PATH
    )
    optimizer = optimizer_catalog.build("nearest_driver", env, OBJECTIVE)
    with pytest.raises(ValueError, match="eval_spec"):
        optimizer.run_simulations(
            1,
            str(tmp_path) + "/",
            seed=SEED,
            save_individual_plots=False,
            save_mean_plots=False,
            num_workers=2,
            eval_spec=None,
        )


def test_serial_vs_parallel_nearest_driver_same_results(tmp_path):
    serial = _run_batch(1, tmp_path)
    parallel = _run_batch(2, tmp_path)
    assert serial == parallel
    assert len(serial) == NUM_RUNS


def test_rollout_job_pickleable_and_runs():
    spec = _tiny_eval_spec(
        "rollout",
        base_optimizer="nearest",
        horizon=1,
        alpha=0.9,
        terminal_cost_mode="0",
        record_decisions=False,
    )
    job = EpisodeJob(spec=spec, episode_idx=0, seed=SEED)
    # Garante que o payload viaja por pickle (spawn/forkserver).
    job = pickle.loads(pickle.dumps(job))
    result = run_episode_job(job)
    assert result["ok"] is True, result["error"]
    assert result["episode_idx"] == 0
    assert result["episode"]["length"] > 0
