from math import ceil
import os
from typing import List

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

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
            dir_path: str = "./", 
            use_tkinter: bool = False
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
        self.use_tkinter = use_tkinter

    def get_next_image_name(self) -> str:
        """Gera um nome de arquivo único para evitar sobrescrições."""
        SummarizedDataBoard.image_counter += 1
        return f"run_{self.image_counter}_results_{self.sum_reward}_fig.png"
        
    @staticmethod
    def reset_image_counter() -> None:
        SummarizedDataBoard.image_counter = 0

    def view(self) -> None:
        if self.use_tkinter:
            self._view_with_tkinter()
        else:
            self._view_with_matplotlib()

    def _calculate_fig_height(self) -> float:
        """
        Calcula a altura do gráfico com base na quantidade de motoristas e restaurantes.
        """
        base_height = 3  # Altura base para cada linha
        additional_height = (max(self.num_drivers, self.num_establishments)) * 0.8  # Altura adicional por item
        num_rows = ceil((len(self.metrics) - 1) / 2) + 1  # Número de linhas baseado nas métricas
        return num_rows * base_height + additional_height

    def _view_with_tkinter(self) -> None:
        # Configuração inicial do Tkinter
        root = tk.Tk()
        root.title("Summarized Data Board")

        # Frame principal com barra de rolagem
        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=1)

        canvas = tk.Canvas(frame)
        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        # Configurar barra de rolagem no canvas
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Criar a figura Matplotlib
        fig_height = self._calculate_fig_height()
        fig = plt.figure(figsize=(12, fig_height))
        gs = fig.add_gridspec(ceil((len(self.metrics) - 1) / 2) + 1, 2, hspace=0.9)

        # Primeiro gráfico destacado
        ax1 = fig.add_subplot(gs[0, :])
        self.metrics[0].view(ax1)

        # Primeiro gráfico destacado
        ax2 = fig.add_subplot(gs[1, :])
        self.metrics[1].view(ax2)

        # Primeiro gráfico destacado
        ax3 = fig.add_subplot(gs[2, :])
        self.metrics[2].view(ax3)

        # Gráficos restantes
        for i, metric in enumerate(self.metrics[3:], start=3):
            row = (i + 1) // 2
            col = (i - 1) % 2

            ax = fig.add_subplot(gs[row, col])
            metric.view(ax)

        # Integrar figura ao Tkinter
        canvas_figure = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas_figure.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Mostrar janela
        root.mainloop()

    def _view_with_matplotlib(self) -> None:
        # Se save_figs for True, usar o backend Agg
        if self.save_figs:
            matplotlib.use("Agg")

        if not self.metrics:
            return

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
                image_name = self.get_next_image_name()
                fig1.savefig(os.path.join(self.figs_dir, image_name), dpi=300, bbox_inches='tight')
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
                image_name = self.get_next_image_name()
                fig2.savefig(os.path.join(self.figs_dir, image_name), dpi=300, bbox_inches='tight')
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
                image_name = self.get_next_image_name()
                fig3.savefig(os.path.join(self.figs_dir, image_name), dpi=300, bbox_inches='tight')
                plt.close(fig3)
            else:
                fig3.suptitle("Driver and Establishment Metrics", fontsize=18, fontweight='bold')
                plt.show()
