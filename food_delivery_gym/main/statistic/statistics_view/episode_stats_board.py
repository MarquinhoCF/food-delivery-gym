from __future__ import annotations

from math import ceil
import os
from typing import TYPE_CHECKING

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from food_delivery_gym.main.statistic.statistics_view.board import Board
from food_delivery_gym.main.statistic.metrics.poisson_order_generation_metric import PoissonOrderGenerationMetric
from food_delivery_gym.main.statistic.metrics.order_flow_metric import OrderFlowMetric
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


class EpisodeStatsBoard(Board):
    
    # Estrutura de saída (relativa a dir_path):
    #   <dir_path>/figs/run_<episode_idx>_results_<sum_reward>/
    #       order_generation.png
    #       route_reordering.png
    #       driver_establishment_metrics.png

    def __init__(self, sim_stats: SimulationStats, episode_idx: int) -> None:
        super().__init__(metrics=[])   # métricas são montadas internamente em _build
        self.sim_stats          = sim_stats
        self.episode_idx        = episode_idx

    def view(self) -> None:
        """Exibe os gráficos do episódio em janelas interativas."""
        figs = self._build_figures()
        titles = [
            "Order Generation and Pipeline Metrics",
            "Route Reordering Metric",
            "Driver and Establishment Metrics",
        ]
        for fig, title in zip(figs, titles):
            fig.suptitle(title, fontsize=18, fontweight="bold")
        plt.show()

    def save(self, dir_path: str) -> None:
        matplotlib.use("Agg")

        run_dir = self._make_run_dir(dir_path)
        figs    = self._build_figures()
        names   = [
            "order_generation.png",
            "route_reordering.png",
            "driver_establishment_metrics.png",
        ]
        for fig, name in zip(figs, names):
            fig.savefig(os.path.join(run_dir, name), dpi=300, bbox_inches="tight")
            plt.close(fig)

    # ──────────────────────────────────────────────────────────────────
    #  Helpers internos
    # ──────────────────────────────────────────────────────────────────

    def _run_dir_name(self) -> str:
        ep = self.sim_stats.get_episode_sim(self.episode_idx)
        ep_reward = float(ep.get("reward", 0.0))
        return f"run_{self.episode_idx + 1}_results_{ep_reward}"

    def _make_run_dir(self, dir_path: str) -> str:
        run_dir = os.path.join(dir_path, "figs", self._run_dir_name())
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _build_figures(self) -> list[Figure]:
        ep           = self.sim_stats.get_episode_sim(self.episode_idx)
        events       = ep.get("events", [])
        ep_drivers   = ep.get("driver", {})
        ep_est       = ep.get("establishment", {})

        # ── Métricas de pipeline / geração de pedidos ─────────────────
        pipeline_metrics = [
            PoissonOrderGenerationMetric(episode_events=events),
            OrderFlowMetric(episode_events=events),
        ]

        # ── Métrica de reordenação ────────────────────────────────────
        reordering_metric = RouteReorderingMetric(episode_drivers=ep_drivers)

        # ── Métricas de agentes ───────────────────────────────────────
        other_metrics = [
            EstablishmentOrdersFulfilledMetric(
                episode_data={eid: v.get("orders_fulfilled") for eid, v in ep_est.items()}),
            EstablishmentMaxOrdersInQueueMetric(
                episode_data={eid: v.get("max_orders_in_queue") for eid, v in ep_est.items()}),
            EstablishmentActiveTimeMetric(
                episode_data={eid: v.get("active_time") for eid, v in ep_est.items()}),
            DriverTimeSpentOnDelivery(
                episode_data={did: v.get("time_spent_on_delivery") for did, v in ep_drivers.items()}),
            DriverOrdersDeliveredMetric(
                episode_data={did: v.get("orders_delivered") for did, v in ep_drivers.items()}),
            DriverTotalDistanceMetric(
                episode_data={did: v.get("total_distance") for did, v in ep_drivers.items()}),
            DriverIdleTimeMetric(
                episode_data={did: v.get("idle_time") for did, v in ep_drivers.items()}),
            DriverTimeWaitingForOrderMetric(
                episode_data={did: v.get("time_waiting_for_order") for did, v in ep_drivers.items()}),
        ]

        figs: list[Figure] = []

        # ── Figura 1: pipeline ────────────────────────────────────────
        if pipeline_metrics:
            fig1_h = len(pipeline_metrics) * 3
            fig1   = plt.figure(figsize=(12, fig1_h))
            gs1    = fig1.add_gridspec(len(pipeline_metrics), 1, hspace=0.9)
            for i, metric in enumerate(pipeline_metrics):
                metric.view(fig1.add_subplot(gs1[i, 0]))
            figs.append(fig1)

        # ── Figura 2: reordenação ─────────────────────────────────────
        fig2 = plt.figure(figsize=(12, 4))
        reordering_metric.view(fig2.add_subplot(1, 1, 1))
        figs.append(fig2)

        # ── Figura 3: agentes ─────────────────────────────────────────
        num_establishments = len(ep_est)
        num_drivers = len(ep_drivers)
        if other_metrics:
            num     = len(other_metrics)
            rows    = ceil(num / 2)
            extra_h = max(num_drivers, num_establishments) * 0.8
            fig3_h  = max(6, rows * 3 + extra_h)
            fig3    = plt.figure(figsize=(12, fig3_h))
            gs3     = fig3.add_gridspec(rows, 2, hspace=0.9)
            for j, metric in enumerate(other_metrics):
                row, col = divmod(j, 2)
                metric.view(fig3.add_subplot(gs3[row, col]))
            figs.append(fig3)

        return figs