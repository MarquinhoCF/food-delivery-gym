from math import ceil
import os
from typing import List

import matplotlib
from matplotlib import pyplot as plt

from food_delivery_gym.main.statistic.board import Board
from food_delivery_gym.main.statistic.metric import Metric


class SummarizedDataBoard(Board):
    image_counter = 0  # Variável estática para controlar o nome das imagens

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
        
        if self.save_figs == True:
            self.dir_path = dir_path
            self.figs_dir = os.path.join(self.dir_path, "figs")
            # Criar a pasta 'figs' caso não exista
            os.makedirs(self.figs_dir, exist_ok=True)

    def get_next_dir_name(self) -> str:
        """Gera um nome de arquivo único para evitar sobrescrições."""
        SummarizedDataBoard.image_counter += 1
        dir_name = f"run_{self.image_counter}_results_{self.sum_reward}"
        if self.save_figs == True:
            dir = os.path.join(self.dir_path, "figs", dir_name)
            # Criar a pasta 'dir' caso não exista
            os.makedirs(dir, exist_ok=True)
        return dir_name
        
    @staticmethod
    def reset_image_counter() -> None:
        SummarizedDataBoard.image_counter = 0

    def _calculate_fig_height(self) -> float:
        """
        Calcula a altura do gráfico com base na quantidade de motoristas e restaurantes.
        """
        base_height = 3  # Altura base para cada linha
        additional_height = (max(self.num_drivers, self.num_establishments)) * 0.8  # Altura adicional por item
        num_rows = ceil((len(self.metrics) - 1) / 2) + 1  # Número de linhas baseado nas métricas
        return num_rows * base_height + additional_height

    def view(self) -> None:
        # Se save_figs for True, usar o backend Agg
        if self.save_figs:
            matplotlib.use("Agg")

        if not self.metrics:
            return
        
        dir_name = self.get_next_dir_name()

        # ==============================
        # AGRUPAMENTO POR CLASSE
        # ==============================
        # Figura 1 — métricas de pipeline
        fig1_classes = {
            "PoissonOrderGenerationMetric",
            "OrderFlowMetric",
            "OrderPipelineStatusMetric"
        }

        # Figura 2 — métrica de reordenação
        fig2_classes = {"RouteReorderingMetric"}

        # Demais métricas → figura 3
        metrics_fig1 = [m for m in self.metrics if type(m).__name__ in fig1_classes]
        metrics_fig2 = [m for m in self.metrics if type(m).__name__ in fig2_classes]
        metrics_fig3 = [
            m for m in self.metrics
            if type(m).__name__ not in fig1_classes | fig2_classes
        ]

        # ==============================
        # FIGURA 1 — PIPELINE METRICS
        # ==============================
        if metrics_fig1:
            fig1_height = len(metrics_fig1) * 3
            fig1 = plt.figure(figsize=(12, fig1_height))
            gs1 = fig1.add_gridspec(len(metrics_fig1), 1, hspace=0.9)

            for i, metric in enumerate(metrics_fig1):
                ax = fig1.add_subplot(gs1[i, 0])
                metric.view(ax)

            if self.save_figs:
                fig1.savefig(os.path.join(self.figs_dir, dir_name, "order_generation.png"), dpi=300, bbox_inches='tight')
                plt.close(fig1)
            else:
                fig1.suptitle("Order Generation and Pipeline Metrics", fontsize=18, fontweight='bold')
                fig1.show()

        # ==============================
        # FIGURA 2 — ROUTE REORDERING METRIC
        # ==============================
        if metrics_fig2:
            fig2 = plt.figure(figsize=(12, 4))
            ax2 = fig2.add_subplot(1, 1, 1)
            metrics_fig2[0].view(ax2)

            if self.save_figs:
                fig2.savefig(os.path.join(self.figs_dir, dir_name, "route_reordering.png"), dpi=300, bbox_inches='tight')
                plt.close(fig2)
            else:
                fig2.suptitle("Route Reordering Metric", fontsize=18, fontweight='bold')
                fig2.show()

        # ==============================
        # FIGURA 3 — DEMAIS MÉTRICAS
        # ==============================
        if metrics_fig3:
            num_remaining = len(metrics_fig3)
            rows_remaining = ceil(num_remaining / 2)
            fig3_height = max(6, rows_remaining * 3)
            fig3 = plt.figure(figsize=(12, fig3_height))
            gs3 = fig3.add_gridspec(rows_remaining, 2, hspace=0.9)

            for j, metric in enumerate(metrics_fig3):
                row = j // 2
                col = j % 2
                ax = fig3.add_subplot(gs3[row, col])
                metric.view(ax)

            if self.save_figs:
                fig3.savefig(os.path.join(self.figs_dir, dir_name, "driver_establishment_metrics.png"), dpi=300, bbox_inches='tight')
                plt.close(fig3)
            else:
                fig3.suptitle("Driver and Establishment Metrics", fontsize=18, fontweight='bold')
                plt.show()