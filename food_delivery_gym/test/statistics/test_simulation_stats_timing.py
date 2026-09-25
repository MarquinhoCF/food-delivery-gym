from __future__ import annotations

import csv
import json
from pathlib import Path

from food_delivery_gym.main.statistics.simulation_stats import SimulationStats


def _minimal_episode(*, seed: int, eval_seconds: float) -> dict:
    return {
        "reward": 1.0,
        "length": 10,
        "simpy_time": 100.0,
        "truncated": False,
        "orders_generated": 2,
        "seed": seed,
        "eval_seconds": eval_seconds,
        "drivers": {
            "1": {"time_spent_on_delivery": 5.0, "total_distance": 3.0},
        },
        "establishments": {"1": {"orders_produced": 2}},
        "events": [],
    }


def test_episodes_csv_and_summary_json_timing(tmp_path: Path):
    stats = SimulationStats()
    stats.register_episode_dict(_minimal_episode(seed=42, eval_seconds=1.23456))
    stats.register_episode_dict(_minimal_episode(seed=43, eval_seconds=2.0))
    stats.finalize()
    stats.duration_seconds = 3.5

    out = tmp_path / "agent"
    stats.save(dir_path=str(out), fmt="npz")

    with (out / "episodes.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["eval_seconds"] for r in rows] == ["1.2346", "2.0"]
    assert rows[0]["seed"] == "42"

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["duration_seconds"] == 3.5
    assert summary["eval_seconds_sum"] == 3.2346
