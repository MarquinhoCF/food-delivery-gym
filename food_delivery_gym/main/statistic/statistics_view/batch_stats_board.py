from __future__ import annotations

from math import ceil
import os
from typing import TYPE_CHECKING

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from food_delivery_gym.main.statistic.statistics_view.board import Board
from food_delivery_gym.main.statistic.metrics.route_reordering_metric import RouteReorderingMetric
from food_delivery_gym.main.statistic.metrics.establishment_metrics import (
    EstablishmentOrdersFulfilledMetric,
    EstablishmentMaxOrdersInQueueMetric,
    EstablishmentActiveTimeMetric,
)
from food_delivery_gym.main.statistic.metrics.driver_metrics import (
    DriverTimeSpentOnDelivery,
    DriverOrdersDeliveredMetric,
    DriverTotalDistanceMetric,
    DriverIdleTimeMetric,
    DriverTimeWaitingForOrderMetric,
)

if TYPE_CHECKING:
    from food_delivery_gym.main.statistic.simulation_stats import SimulationStats


class BatchStatsBoard(Board):
    
    # Estrutura de saída (relativa a dir_path):
    #   <dir_path>/figs/
    #       mean_results_<sum_reward>_route_reordering.png
    #       mean_results_<sum_reward>_other_metrics.png

    def __init__(
        self,
        sim_stats: SimulationStats,
    ) -> None:
        super().__init__(metrics=[])   # métricas são montadas internamente em _build
        self.sim_stats          = sim_stats

    def view(self) -> None:
        """Exibe os gráficos agregados em janelas interativas."""
        fig_reordering, fig_agents = self._build_figures()
        fig_reordering.suptitle(
            "Route Reordering Metric – Aggregate", fontsize=18, fontweight="bold"
        )
        fig_agents.suptitle(
            "Driver & Establishment Metrics – Aggregate", fontsize=18, fontweight="bold"
        )
        plt.show()

    def save(self, dir_path: str) -> None:
        matplotlib.use("Agg")

        figs_dir = os.path.join(dir_path, "figs")
        os.makedirs(figs_dir, exist_ok=True)

        agregate = self.sim_stats.get_aggregated_sim()
        sum_reward_avg = agregate.get("rewards", {}).get("avg", 0.0)

        prefix               = f"mean_results_{sum_reward_avg}"
        fig_reordering, fig_agents = self._build_figures()

        fig_reordering.savefig(
            os.path.join(figs_dir, f"{prefix}_route_reordering.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig_reordering)

        fig_agents.savefig(
            os.path.join(figs_dir, f"{prefix}_other_metrics.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig_agents)

    # ──────────────────────────────────────────────────────────────────
    #  Helpers internos
    # ──────────────────────────────────────────────────────────────────

    def _fig_height(self, rows: int) -> float:
        first_ep = self.sim_stats.get_episode_sim(0)
        num_drivers = len(first_ep.get("driver", {}))
        num_establishments = len(first_ep.get("establishment", {}))

        extra = max(num_drivers, num_establishments) * 0.5
        return max(6, rows * 3 + extra)

    def _build_figures(self) -> tuple[Figure, Figure]:
        drivers_stats = self.sim_stats.get_drivers_computed_stats()
        est_stats     = self.sim_stats.get_establishments_computed_stats()

        def _agg_col(agents_stats: dict, metric_key: str) -> dict:
            return {
                aid: stats.get(metric_key)
                for aid, stats in agents_stats.items()
            }

        # ── Métricas ──────────────────────────────────────────────────
        reordering_metric = RouteReorderingMetric(aggregate_drivers=drivers_stats)

        other_metrics = [
            EstablishmentOrdersFulfilledMetric(
                aggregate_data=_agg_col(est_stats, "orders_fulfilled")),
            EstablishmentMaxOrdersInQueueMetric(
                aggregate_data=_agg_col(est_stats, "max_orders_in_queue")),
            EstablishmentActiveTimeMetric(
                aggregate_data=_agg_col(est_stats, "active_time")),
            DriverTimeSpentOnDelivery(
                aggregate_data=_agg_col(drivers_stats, "time_spent_on_delivery")),
            DriverOrdersDeliveredMetric(
                aggregate_data=_agg_col(drivers_stats, "orders_delivered")),
            DriverTotalDistanceMetric(
                aggregate_data=_agg_col(drivers_stats, "total_distance")),
            DriverIdleTimeMetric(
                aggregate_data=_agg_col(drivers_stats, "idle_time")),
            DriverTimeWaitingForOrderMetric(
                aggregate_data=_agg_col(drivers_stats, "time_waiting_for_order")),
        ]

        # ── Figura 1: reordenação ─────────────────────────────────────
        fig_reordering = plt.figure(figsize=(12, 4))
        reordering_metric.view(fig_reordering.add_subplot(1, 1, 1))

        # ── Figura 2: agentes ─────────────────────────────────────────
        num    = len(other_metrics)
        rows   = ceil(num / 2)
        fig_agents = plt.figure(figsize=(12, self._fig_height(rows)))
        gs         = fig_agents.add_gridspec(rows, 2, hspace=0.9)
        for j, metric in enumerate(other_metrics):
            row, col = divmod(j, 2)
            metric.view(fig_agents.add_subplot(gs[row, col]))

        return fig_reordering, fig_agents