from math import ceil
from typing import List

from matplotlib import pyplot as plt

from food_delivery_gym.main.statistic.board import Board
from food_delivery_gym.main.statistic.metric import Metric


class CustomBoard(Board):

    def __init__(self, metrics: List[Metric]):
        super().__init__(metrics)

    def view(self) -> None:
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(ceil(len(self.metrics) / 2) + 1, 2)

        # Primeiro gráfico ocupa todas as colunas da primeira linha
        ax1 = fig.add_subplot(gs[0, :])
        self.metrics[0].view(ax1)

        # Outros gráficos ocupam a segunda linha
        ax2 = fig.add_subplot(gs[1, 0])
        self.metrics[1].view(ax2)

        ax3 = fig.add_subplot(gs[1, 1])
        self.metrics[2].view(ax3)

        ax4 = fig.add_subplot(gs[2, 0])
        self.metrics[3].view(ax4)

        ax5 = fig.add_subplot(gs[2, 1])
        self.metrics[4].view(ax5)

        ax6 = fig.add_subplot(gs[3, 0])
        self.metrics[5].view(ax6)

        # Ajustar o layout para evitar sobreposição
        plt.tight_layout()

        # Exibir a figura
        plt.show()

