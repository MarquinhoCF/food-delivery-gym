"""Testes do módulo de experimento nomeado."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from food_delivery_gym.main.eval import experiment as exp
from food_delivery_gym.main.optimizer import catalog as optimizer_catalog


def test_parse_lowest_cli():
    spec = optimizer_catalog.parse_lowest_cli("cost=route")
    assert spec.cost_function == "route"
    assert optimizer_catalog.lowest_result_key(spec.cost_function) == "lowest_route_cost"


def test_parse_lowest_cli_rejects_unknown_key():
    with pytest.raises(ValueError, match="chave desconhecida"):
        optimizer_catalog.parse_lowest_cli("foo=bar")


def test_expand_evaluations_lowest_variants():
    specs = [optimizer_catalog.get("lowest"), optimizer_catalog.get("nearest_driver")]
    variants = optimizer_catalog.expand_evaluations(
        specs,
        lowest_variants=[
            optimizer_catalog.parse_lowest_cli("cost=route"),
            optimizer_catalog.parse_lowest_cli("cost=weighted_score"),
        ],
    )
    keys = [v.result_key for v in variants]
    assert keys == ["lowest_route_cost", "lowest_weighted_score", "nearest_driver"]


def test_expand_evaluations_lowest_requires_variants():
    specs = [optimizer_catalog.get("lowest")]
    with pytest.raises(ValueError, match="lowest selecionado exige"):
        optimizer_catalog.expand_evaluations(specs, lowest_variants=[])


def test_load_yaml_unknown_key(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nfoo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Chaves desconhecidas"):
        exp.load_yaml_spec(path)


def test_resolve_experiment_yaml_and_cli_override(tmp_path: Path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        yaml.dump({
            "name": "my_exp",
            "objectives": [3],
            "scenarios": ["simple"],
            "num_runs": 10,
            "no_rl": True,
            "lowest": ["cost=route"],
        }),
        encoding="utf-8",
    )
    resolved = exp.resolve_experiment(
        yaml_path=path,
        cli_overrides={"num_runs": 2},
    )
    assert resolved["name"] == "my_exp"
    assert resolved["num_runs"] == 2
    assert resolved["objectives"] == [3]
    assert len(resolved["_lowest_variants"]) == 1
    assert resolved["_lowest_variants"][0].cost_function == "route"


def test_resolve_experiment_requires_name():
    with pytest.raises(ValueError, match="name"):
        exp.resolve_experiment(cli_overrides={"objectives": [1]})


def test_write_run_json(tmp_path: Path):
    spec = exp.resolve_experiment(
        cli_overrides={
            "name": "unit_run",
            "objectives": [3],
            "scenarios": ["simple"],
            "no_rl": True,
            "num_runs": 1,
        },
    )
    path = exp.write_run_json(spec, runs_root=tmp_path)
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["spec"]["name"] == "unit_run"
    assert "package_version" in data
    assert "git_commit" in data
    assert "started_at" in data
    assert data["finished_at"] is None
    assert data["duration_seconds"] is None


def test_finalize_run_json(tmp_path: Path):
    from datetime import datetime, timezone

    spec = exp.resolve_experiment(
        cli_overrides={
            "name": "timed_run",
            "objectives": [3],
            "scenarios": ["simple"],
            "no_rl": True,
            "num_runs": 1,
        },
    )
    exp.write_run_json(spec, runs_root=tmp_path)
    started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    path = exp.finalize_run_json(
        "timed_run",
        runs_root=tmp_path,
        started_at=started,
        duration_seconds=12.34567,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["finished_at"] is not None
    assert data["duration_seconds"] == 12.3457
    assert data["spec"]["name"] == "timed_run"


def test_agent_complete_and_layout(tmp_path: Path):
    name = "skip_test"
    out = exp.agent_dir(name, 3, "simple", "random", runs_root=tmp_path)
    out.mkdir(parents=True)
    assert not exp.agent_complete(out)

    np.savez_compressed(out / "metrics_data.npz", ep__rewards=np.array([1.0]))
    assert exp.agent_complete(out)

    resolved = exp.resolve_agent_dir(tmp_path / name, 3, "simple", "random")
    assert resolved == out


def test_legacy_layout_resolve(tmp_path: Path):
    legacy = tmp_path / "obj_3" / "simple_scenario" / "random"
    legacy.mkdir(parents=True)
    (legacy / "metrics_data.npz").write_bytes(b"")  # empty file fails complete
    # resolve_agent_dir checks is_file; empty npz still counts as present
    np.savez_compressed(legacy / "metrics_data.npz", ep__rewards=np.array([1.0]))
    resolved = exp.resolve_agent_dir(tmp_path, 3, "simple", "random")
    assert resolved == legacy


def test_summary_csv_columns(tmp_path: Path):
    rows = [
        exp.summary_row_from_aggregate(
            objective=3,
            scenario="simple",
            agent="random",
            aggregate={
                "rewards": {"avg": 1.0, "std_dev": 0.1, "n": 2},
                "delivery_time": {"avg": 10.0, "std_dev": 1.0, "n": 2},
                "distance": {"avg": 5.0, "std_dev": 0.5, "n": 2},
            },
            truncated=0,
        )
    ]
    path = exp.write_summary_csv(rows, name="sum_test", runs_root=tmp_path)
    text = path.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    for col in exp.SUMMARY_CSV_FIELDS:
        assert col in header.split(",")
