from math import ceil
import os
from typing import List

import matplotlib
from matplotlib import pyplot as plt

from food_delivery_gym.main.statistic.board import Board
from food_delivery_gym.main.statistic.metric import Metric
from food_delivery_gym.main.statistic.route_reordering_metric import RouteReorderingMetric

class SummarizedTotalMeanDataBoard(Board):
    image_counter = 0

    def __init__(
            self,
            metrics: List[Metric],
            num_drivers: int,
            num_establishments: int,
            sum_reward: int,
            save_figs: bool = False,
            dir_path: str = "./"
        ):
        super().__init__(metrics)
        self.num_drivers = num_drivers
        self.num_establishments = num_establishments
        self.sum_reward = sum_reward
        self.save_figs = save_figs

        if self.save_figs:
            self.dir_path = dir_path
            self.figs_dir = os.path.join(self.dir_path, "figs")
            os.makedirs(self.figs_dir, exist_ok=True)

    def get_prefix_image_name(self) -> str:
        return f"mean_results_{self.sum_reward}"

    def _calculate_fig_height(self, rows: int) -> float:
        base_height = 3
        additional = max(self.num_drivers, self.num_establishments) * 0.5
        return rows * base_height + additional

    def view(self) -> None:
        if self.save_figs:
            matplotlib.use("Agg")

        # ============================
        # 1) Separar as métricas por classe
        # ============================
        reordering_metric = None
        other_metrics = []

        for metric in self.metrics:
            if isinstance(metric, RouteReorderingMetric):
                reordering_metric = metric
            else:
                other_metrics.append(metric)

        # ============================
        # FIGURA 1 — SOMENTE RouteReorderingMetric
        # ============================
        if reordering_metric is not None:
            fig1 = plt.figure(figsize=(12, 4))
            ax1 = fig1.add_subplot(1, 1, 1)
            reordering_metric.view(ax1)

            if self.save_figs:
                image_name = self.get_prefix_image_name() + "_route_reordering.png"
                fig1.savefig(os.path.join(self.figs_dir, image_name),
                             dpi=300, bbox_inches='tight')
                plt.close(fig1)
            else:
                fig1.suptitle("Route Reordering Metric", fontsize=18, fontweight='bold')
                plt.show()

        # ============================
        # FIGURA 2 — OUTRAS MÉTRICAS
        # ============================
        if other_metrics:
            num = len(other_metrics)
            rows = ceil(num / 2)
            fig_height = self._calculate_fig_height(rows)

            fig2 = plt.figure(figsize=(12, fig_height))
            gs = fig2.add_gridspec(rows, 2, hspace=0.9)

            for i, metric in enumerate(other_metrics):
                row, col = divmod(i, 2)
                ax = fig2.add_subplot(gs[row, col])
                metric.view(ax)

            if self.save_figs:
                image_name = self.get_prefix_image_name() + "_other_metrics.png"
                fig2.savefig(os.path.join(self.figs_dir, image_name),
                             dpi=300, bbox_inches='tight')
                plt.close(fig2)
            else:
                fig2.suptitle("Other Summary Metrics", fontsize=18, fontweight='bold')
                plt.show()