from abc import ABC, abstractmethod
from typing import List

from food_delivery_gym.main.base.types import Coordinate, Number
from food_delivery_gym.main.utils.random_manager import RandomManager

class Map(ABC):

    def __init__(self, size):
        self.size = size
        self.rng = RandomManager().get_random_instance()

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
