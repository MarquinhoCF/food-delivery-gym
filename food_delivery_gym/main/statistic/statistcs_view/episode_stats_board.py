"""
Board de estatísticas para um único episódio.

Constrói todas as métricas internamente a partir de SimulationStats,
sem nenhuma referência ao ambiente vivo.
"""
from math import ceil
import os

import matplotlib
from matplotlib import pyplot as plt

from food_delivery_gym.main.statistic.simulation_stats import SimulationStats
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


class EpisodeStatsBoard:
    
    image_counter: int = 0  # controla nomes únicos de diretórios

    def __init__(
        self,
        sim_stats: SimulationStats,
        episode_idx: int,
        num_drivers: int,
        num_establishments: int,
        sum_reward: float | None = None,
        save_figs: bool = False,
        dir_path: str = "./",
    ):
        self.sim_stats         = sim_stats
        self.episode_idx       = episode_idx
        self.num_drivers       = num_drivers
        self.num_establishments = num_establishments
        self.sum_reward        = sum_reward
        self.save_figs         = save_figs
        self.dir_path          = dir_path

        if self.save_figs:
            self.figs_dir = os.path.join(dir_path, "figs")
            os.makedirs(self.figs_dir, exist_ok=True)

    @classmethod
    def reset_image_counter(cls) -> None:
        cls.image_counter = 0

    def view(self) -> None:
        if self.save_figs:
            matplotlib.use("Agg")

        # Extrai dados do episódio (sem finalize obrigatório)
        ep = self.sim_stats.get_episode_sim(self.episode_idx)
        events       = ep.get("events", [])
        ep_drivers   = ep.get("driver", {})
        ep_est       = ep.get("establishment", {})

        # ── Monta dados por agente ──────────────────────────────────
        driver_episode = {
            did: {k: v for k, v in metrics.items()}
            for did, metrics in ep_drivers.items()
        }
        est_episode = {
            eid: {k: v for k, v in metrics.items()}
            for eid, metrics in ep_est.items()
        }

        # ── Instancia métricas com dados do episódio ─────────────────
        pipeline_metrics = [
            PoissonOrderGenerationMetric(episode_events=events),
            OrderFlowMetric(episode_events=events),
        ]

        reordering_metric = RouteReorderingMetric(episode_drivers=driver_episode)

        other_metrics = [
            EstablishmentOrdersFulfilledMetric(
                episode_data={eid: v.get("orders_fulfilled") for eid, v in est_episode.items()}),
            EstablishmentMaxOrdersInQueueMetric(
                episode_data={eid: v.get("max_orders_in_queue") for eid, v in est_episode.items()}),
            EstablishmentActiveTimeMetric(
                episode_data={eid: v.get("active_time") for eid, v in est_episode.items()}),
            DriverTimeSpentOnDelivery(
                episode_data={did: v.get("time_spent_on_delivery") for did, v in driver_episode.items()}),
            DriverOrdersDeliveredMetric(
                episode_data={did: v.get("orders_delivered") for did, v in driver_episode.items()}),
            DriverTotalDistanceMetric(
                episode_data={did: v.get("total_distance") for did, v in driver_episode.items()}),
            DriverIdleTimeMetric(
                episode_data={did: v.get("idle_time") for did, v in driver_episode.items()}),
            DriverTimeWaitingForOrderMetric(
                episode_data={did: v.get("time_waiting_for_order") for did, v in driver_episode.items()}),
        ]

        # ── Gera nome de diretório único ──────────────────────────────
        EpisodeStatsBoard.image_counter += 1
        dir_name = f"run_{EpisodeStatsBoard.image_counter}_results_{self.sum_reward}"
        if self.save_figs:
            run_dir = os.path.join(self.figs_dir, dir_name)
            os.makedirs(run_dir, exist_ok=True)

        # ══════════════════════════════════════════════════════════════
        # FIGURA 1 — Pipeline / Geração de Pedidos
        # ══════════════════════════════════════════════════════════════
        if pipeline_metrics:
            fig1_height = len(pipeline_metrics) * 3
            fig1 = plt.figure(figsize=(12, fig1_height))
            gs1  = fig1.add_gridspec(len(pipeline_metrics), 1, hspace=0.9)

            for i, metric in enumerate(pipeline_metrics):
                ax = fig1.add_subplot(gs1[i, 0])
                metric.view(ax)

            if self.save_figs:
                fig1.savefig(
                    os.path.join(self.figs_dir, dir_name, "order_generation.png"),
                    dpi=300, bbox_inches="tight",
                )
                plt.close(fig1)
            else:
                fig1.suptitle("Order Generation and Pipeline Metrics",
                               fontsize=18, fontweight="bold")
                fig1.show()

        # ══════════════════════════════════════════════════════════════
        # FIGURA 2 — Reordenação de Rotas
        # ══════════════════════════════════════════════════════════════
        fig2 = plt.figure(figsize=(12, 4))
        ax2  = fig2.add_subplot(1, 1, 1)
        reordering_metric.view(ax2)

        if self.save_figs:
            fig2.savefig(
                os.path.join(self.figs_dir, dir_name, "route_reordering.png"),
                dpi=300, bbox_inches="tight",
            )
            plt.close(fig2)
        else:
            fig2.suptitle("Route Reordering Metric", fontsize=18, fontweight="bold")
            fig2.show()

        # ══════════════════════════════════════════════════════════════
        # FIGURA 3 — Métricas de Driver e Estabelecimento
        # ══════════════════════════════════════════════════════════════
        if other_metrics:
            num      = len(other_metrics)
            rows     = ceil(num / 2)
            extra_h  = max(self.num_drivers, self.num_establishments) * 0.8
            fig3_h   = max(6, rows * 3 + extra_h)

            fig3 = plt.figure(figsize=(12, fig3_h))
            gs3  = fig3.add_gridspec(rows, 2, hspace=0.9)

            for j, metric in enumerate(other_metrics):
                row, col = divmod(j, 2)
                ax = fig3.add_subplot(gs3[row, col])
                metric.view(ax)

            if self.save_figs:
                fig3.savefig(
                    os.path.join(self.figs_dir, dir_name, "driver_establishment_metrics.png"),
                    dpi=300, bbox_inches="tight",
                )
                plt.close(fig3)
            else:
                fig3.suptitle("Driver and Establishment Metrics",
                               fontsize=18, fontweight="bold")
                plt.show()