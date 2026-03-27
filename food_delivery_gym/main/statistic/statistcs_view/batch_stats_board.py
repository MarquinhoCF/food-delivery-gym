from math import ceil
import os

import matplotlib
from matplotlib import pyplot as plt

from food_delivery_gym.main.statistic.simulation_stats import SimulationStats
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


class BatchStatsBoard:

    def __init__(
        self,
        sim_stats: SimulationStats,
        num_drivers: int,
        num_establishments: int,
        sum_reward: float | None = None,
        save_figs: bool = False,
        dir_path: str = "./",
    ):
        self.sim_stats          = sim_stats
        self.num_drivers        = num_drivers
        self.num_establishments = num_establishments
        self.sum_reward         = sum_reward
        self.save_figs          = save_figs
        self.dir_path           = dir_path

        if self.save_figs:
            self.figs_dir = os.path.join(dir_path, "figs")
            os.makedirs(self.figs_dir, exist_ok=True)

    def _prefix(self) -> str:
        return f"mean_results_{self.sum_reward}"

    def _fig_height(self, rows: int) -> float:
        extra = max(self.num_drivers, self.num_establishments) * 0.5
        return max(6, rows * 3 + extra)
    
    def view(self) -> None:
        if self.save_figs:
            matplotlib.use("Agg")

        # ── Computa estatísticas agregadas ────────────────────────────
        drivers_stats = self.sim_stats.get_drivers_computed_stats()
        est_stats     = self.sim_stats.get_establishments_computed_stats()

        # ── Extrai coluna de uma métrica por agente ───────────
        def _agg_col(agents_stats: dict, metric_key: str) -> dict:
            """Retorna {agent_id: stats_dict} filtrando pela chave da métrica."""
            return {
                aid: stats.get(metric_key)
                for aid, stats in agents_stats.items()
            }

        reordering_metric = RouteReorderingMetric(
            aggregate_drivers=drivers_stats
        )

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

        # ══════════════════════════════════════════════════════════════
        # FIGURA 1 — Reordenação de Rotas (Agregado)
        # ══════════════════════════════════════════════════════════════
        fig2 = plt.figure(figsize=(12, 4))
        ax2  = fig2.add_subplot(1, 1, 1)
        reordering_metric.view(ax2)

        if self.save_figs:
            fig2.savefig(
                os.path.join(self.figs_dir, self._prefix() + "_route_reordering.png"),
                dpi=300, bbox_inches="tight",
            )
            plt.close(fig2)
        else:
            fig2.suptitle("Route Reordering Metric – Aggregate", fontsize=18, fontweight="bold")
            fig2.show()

        # ══════════════════════════════════════════════════════════════
        # FIGURA 2 — Outras Métricas (Agregado)
        # ══════════════════════════════════════════════════════════════
        if other_metrics:
            num    = len(other_metrics)
            rows   = ceil(num / 2)
            fig3_h = self._fig_height(rows)

            fig3 = plt.figure(figsize=(12, fig3_h))
            gs3  = fig3.add_gridspec(rows, 2, hspace=0.9)

            for j, metric in enumerate(other_metrics):
                row, col = divmod(j, 2)
                ax = fig3.add_subplot(gs3[row, col])
                metric.view(ax)

            if self.save_figs:
                fig3.savefig(
                    os.path.join(self.figs_dir, self._prefix() + "_other_metrics.png"),
                    dpi=300, bbox_inches="tight",
                )
                plt.close(fig3)
            else:
                fig3.suptitle("Driver & Establishment Metrics – Aggregate",
                               fontsize=18, fontweight="bold")
                plt.show()