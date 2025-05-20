from math import ceil
from typing import List

from matplotlib import pyplot as plt

from food_delivery_gym.main.statistic.board import Board
from food_delivery_gym.main.statistic.metric import Metric


class DefaultBoard(Board):

    def __init__(self, metrics: List[Metric]):
        super().__init__(metrics)

    def view(self) -> None:
        # Criar uma figura e uma grade de subplots
        fig, axs = plt.subplots(ceil(len(self.metrics) / 2), 2, figsize=(10, 8))

        for ax, metric in zip(axs.flatten(), self.metrics):
            metric.view(ax)

        # Ajustar o layout para evitar sobreposição
        plt.tight_layout()

        # Exibir a figura
        plt.show()
