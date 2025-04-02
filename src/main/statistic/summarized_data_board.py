from math import ceil
import os
from typing import List

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

from src.main.statistic.board import Board
from src.main.statistic.metric import Metric


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
            use_total_mean: bool = False, 
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
        self.use_total_mean = use_total_mean
        self.use_tkinter = use_tkinter

    def get_next_image_name(self) -> str:
        """Gera um nome de arquivo único para evitar sobrescrições."""
        if self.use_total_mean:
            return f"mean_results_{self.sum_reward}_fig.png"
        else:
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

        # Gráficos restantes
        for i, metric in enumerate(self.metrics[1:], start=1):
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

        # Criar a figura Matplotlib
        fig_height = self._calculate_fig_height()
        fig = plt.figure(figsize=(12, fig_height))
        gs = fig.add_gridspec(ceil((len(self.metrics) - 1) / 2) + 1, 2, hspace=0.9)

        if not self.use_total_mean:
            # Primeiro gráfico destacado
            ax1 = fig.add_subplot(gs[0, :])
            self.metrics[0].view(ax1)
            
            # Gráficos restantes
            for i, metric in enumerate(self.metrics[1:], start=1):
                row = (i + 1) // 2
                col = (i - 1) % 2

                ax = fig.add_subplot(gs[row, col])
                metric.view(ax)
        else:
            # Distribuir todos os gráficos uniformemente
            num_metrics = len(self.metrics)
            rows = (num_metrics + 1) // 2  # Calcula quantas linhas são necessárias

            for i, metric in enumerate(self.metrics):
                row = i // 2
                col = i % 2

                ax = fig.add_subplot(gs[row, col])
                metric.view(ax)
        
        if self.save_figs:
            # Gerar nome de arquivo único e salvar imagem
            image_name = self.get_next_image_name()
            if self.use_total_mean:
                fig.savefig(self.dir_path + image_name, dpi=300, bbox_inches='tight')
            else:
                fig.savefig(self.figs_dir + "/" + image_name, dpi=300, bbox_inches='tight')
        else:
            # Mostrar gráficos
            plt.show()
        
        matplotlib.pyplot.close()
