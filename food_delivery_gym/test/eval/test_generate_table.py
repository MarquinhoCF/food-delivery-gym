"""Distância na planilha continua comparável e marca episódios truncados."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_table import build_workbook


def _summary(*, distance: dict | None, truncated: int, num_runs: int = 20) -> dict:
    rewards = {
        "avg": -10.0,
        "std_dev": 1.0,
        "median": -10.0,
        "mode": -10.0,
        "n": num_runs,
    }
    delivery = {
        "avg": 10.0,
        "std_dev": 1.0,
        "median": 10.0,
        "mode": 10.0,
        "n": num_runs,
    }
    return {
        "aggregate": {
            "rewards": rewards,
            "delivery_time": delivery,
            "distance": distance,
        },
        "truncated": truncated,
        "num_runs": num_runs,
    }


def _write_agent(root: Path, name: str, summary: dict) -> None:
    agent_dir = root / "obj_3" / "simple" / name
    agent_dir.mkdir(parents=True)
    (agent_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


def _avg_by_header(ws) -> dict[str, object]:
    headers = {
        ws.cell(2, col).value: col
        for col in range(1, ws.max_column + 1)
    }
    return {name: ws.cell(3, col).value for name, col in headers.items()}


def test_distance_sheet_marks_truncated_episodes(tmp_path: Path):
    root = tmp_path / "run"
    _write_agent(
        root,
        "agent_full",
        _summary(
            distance={
                "avg": 100.0,
                "std_dev": 1.0,
                "median": 100.0,
                "mode": 100.0,
                "n": 20,
            },
            truncated=0,
        ),
    )
    _write_agent(
        root,
        "agent_partial",
        _summary(
            distance={
                "avg": 80.5,
                "std_dev": 2.0,
                "median": 80.0,
                "mode": 80.0,
                "n": 16,
            },
            truncated=4,
        ),
    )
    _write_agent(
        root,
        "agent_all_truncated",
        _summary(distance=None, truncated=20),
    )

    wb = build_workbook(str(root), [3], ["simple"], [
        "agent_full",
        "agent_partial",
        "agent_all_truncated",
    ])
    ws = wb["Distância Percorrida"]
    averages = _avg_by_header(ws)

    assert averages["agent_full"] == 100.0
    assert averages["agent_partial"] == "80.5000 (4/20 eps truncados)"
    assert averages["agent_all_truncated"] == "(20/20 eps truncados)"

    # Menor distância entre valores numéricos: o parcial ainda entra na comparação.
    partial_col = next(
        col
        for col in range(1, ws.max_column + 1)
        if ws.cell(2, col).value == "agent_partial"
    )
    assert ws.cell(3, partial_col).font.bold is True
    all_col = next(
        col
        for col in range(1, ws.max_column + 1)
        if ws.cell(2, col).value == "agent_all_truncated"
    )
    assert ws.cell(3, all_col).font.bold is False
