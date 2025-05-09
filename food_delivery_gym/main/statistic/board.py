from abc import abstractmethod
from typing import List

from food_delivery_gym.main.statistic.metric import Metric


class Board:

    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    @abstractmethod
    def view(self) -> None:
        pass
