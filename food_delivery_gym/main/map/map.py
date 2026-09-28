from abc import ABC, abstractmethod
from typing import List

import numpy as np

from food_delivery_gym.main.base.types import Coordinate, Number
from food_delivery_gym.main.utils.rng_factory import RngFactory


class Map(ABC):

    def __init__(self, size, rng: np.random.Generator | None = None):
        self.size = size
        self.rng = rng if rng is not None else RngFactory().next()

    @abstractmethod
    def distance(self, coord1: Coordinate, coord2: Coordinate) -> Number:
        pass

    @abstractmethod
    def acc_distance(self, coordinates: List[Coordinate]) -> Number:
        pass

    @abstractmethod
    def estimated_time(self, coord1: Coordinate, coord2: Coordinate, rate: Number) -> Number:
        pass

    @abstractmethod
    def random_point(self) -> Coordinate:
        pass

    @abstractmethod
    def move(self, origin: Coordinate, destination: Coordinate, rate: Number) -> Coordinate:
        pass
