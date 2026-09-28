"""Testes do comando scripts.report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from food_delivery_gym.main.eval import experiment as exp
from scripts import report


def _write_agent_artifacts(agent_dir: Path) -> None:
    agent_dir.mkdir(parents=True)
    episodes = [
        {
            "episode": 1,
            "seed": 1,
            "reward": -100.0,
            "length": 10,
            "simpy_time": 100.0,
            "delivery_time": 50.0,
            "distance": 12.0,
            "orders_generated": 5,
            "truncated": False,
            "eval_seconds": 0.1,
        },
        {
            "episode": 2,
            "seed": 2,
            "reward": -120.0,
            "length": 12,
            "simpy_time": 110.0,
            "delivery_time": 60.0,
            "distance": 14.0,
            "orders_generated": 5,
            "truncated": False,
            "eval_seconds": 0.2,
        },
    ]
    fields = list(episodes[0].keys())
    with (agent_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episodes)

    summary = {
        "aggregate": {
            "rewards": {
                "avg": -110.0,
                "std_dev": 10.0,
                "median": -110.0,
                "mode": -110.0,
                "n": 2,
            },
            "delivery_time": {
                "avg": 55.0,
                "std_dev": 5.0,
                "median": 55.0,
                "mode": 55.0,
                "n": 2,
            },
            "distance": {
                "avg": 13.0,
                "std_dev": 1.0,
                "median": 13.0,
                "mode": 13.0,
                "n": 2,
            },
        },
        "truncated": 0,
        "num_runs": 2,
        "hyperparameters": {},
        "duration_seconds": 0.3,
    }
    (agent_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _make_experiment(tmp_path: Path) -> Path:
    """Experimento com objetivo 7 e cenário simple (fora dos defaults comuns)."""
    name = "unit_report"
    root = tmp_path / name
    root.mkdir()
    payload = {
        "spec": {
            "name": name,
            "objectives": [7],
            "scenarios": ["simple"],
            "num_runs": 2,
            "seed": 42,
        },
        "package_version": "0.0.0-test",
        "git_commit": None,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00",
        "duration_seconds": 60.0,
    }
    (root / "run.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    agent_dir = root / "obj_7" / "simple" / "random"
    _write_agent_artifacts(agent_dir)
    return root


def test_load_run_json(tmp_path: Path):
    root = _make_experiment(tmp_path)
    payload = exp.load_run_json(root)
    assert payload["spec"]["objectives"] == [7]
    assert payload["spec"]["scenarios"] == ["simple"]


def test_load_run_json_missing(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="run.json"):
        exp.load_run_json(empty)


def test_report_writes_table_and_boxplots(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    root = _make_experiment(tmp_path)

    result = report.run(root, episodes=False)

    table = Path(result["table"])
    assert table.is_file()
    assert table.name == "objective_table.xlsx"
    assert table.parent == root / "report"

    boxplots = [Path(p) for p in result["boxplots"]]
    assert boxplots
    for path in boxplots:
        assert path.is_file()
        assert path.parent == root / "report" / "boxplots"
        assert path.name.startswith("boxplot_obj7_")
        assert path.suffix == ".png"
        assert "_scenarios" in path.name

    # Sem --episodes: nenhum figs/ de episódio
    agent_figs = root / "obj_7" / "simple" / "random" / "figs"
    assert not agent_figs.exists()
    assert result["episodes_agents"] == 0


def test_report_main_missing_run_json(tmp_path: Path):
    empty = tmp_path / "no_run"
    empty.mkdir()
    code = report.main([str(empty)])
    assert code == 1
