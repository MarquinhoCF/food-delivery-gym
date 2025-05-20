from abc import ABC, abstractmethod

from food_delivery_gym.main.base.types import Number
from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.map.map import Map
from food_delivery_gym.main.route.route_segment import RouteSegment


class CostFunction(ABC):

    @abstractmethod
    def cost(self, map: Map, driver: Driver, route_segment: RouteSegment) -> Number:
        pass
