"""Boxplot de distância marca episódios truncados sem descartar os válidos."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from scripts.generate_boxplots import _plot_single_scenario_ax, build_color_map
import matplotlib.pyplot as plt


def _labels(ax) -> list[str]:
    return [tick.get_text() for tick in ax.get_xticklabels()]


def test_distance_boxplot_marks_truncated_episodes():
    agents = ["agent_full", "agent_partial", "agent_all_truncated"]
    data = {
        "agent_full": {
            "simple": {"distance": [100.0, 110.0, 90.0]},
        },
        "agent_partial": {
            "simple": {"distance": [80.0, None, 70.0, float("nan")] + [75.0] * 16},
        },
        "agent_all_truncated": {
            "simple": {"distance": [None] * 20},
        },
    }
    fig, ax = plt.subplots()
    has_data = _plot_single_scenario_ax(
        ax,
        "distance",
        data,
        agents,
        "simple",
        build_color_map(agents),
        showfliers=True,
        show_means=False,
        show_mean_values=False,
        annotate_n=False,
    )
    labels = _labels(ax)
    plt.close(fig)

    assert has_data is True
    assert labels[0] == "Agent Full"
    assert labels[1] == "Agent Partial\n(2/20 eps truncados)"
    assert labels[2] == "Agent All Truncated\n(20/20 eps truncados)"


def test_other_metrics_do_not_mark_truncation():
    agents = ["agent_partial"]
    data = {
        "agent_partial": {
            "simple": {"rewards": [1.0, None, 3.0]},
        },
    }
    fig, ax = plt.subplots()
    _plot_single_scenario_ax(
        ax,
        "rewards",
        data,
        agents,
        "simple",
        build_color_map(agents),
        showfliers=True,
        show_means=False,
        show_mean_values=False,
        annotate_n=False,
    )
    labels = _labels(ax)
    plt.close(fig)

    assert labels == ["Agent Partial"]
