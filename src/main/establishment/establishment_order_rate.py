from numbers import Number
import random

from simpy.core import SimTime
from src.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from src.main.establishment.catalog import Catalog
from src.main.establishment.establishment import Establishment


class EstablishmentOrderRate(Establishment):
    def __init__(
            self,
            environment: FoodDeliverySimpyEnv,
            coordinate,
            available: bool,
            catalog: Catalog,
            production_capacity,
            order_production_time_rate,
            operating_radius,
            percentage_allocation_driver: Number = 0.7,
            max_prepare_time: Number = 60,
            min_prepare_time: Number = 20,
            id: Number = None,
            use_estimate: bool = False,
    ):
        super().__init__(
            environment, 
            coordinate, 
            available, 
            catalog, 
            percentage_allocation_driver, 
            id, 
            production_capacity, 
            use_estimate
        )
        self.order_production_time_rate = order_production_time_rate
        self.operating_radius = operating_radius
        self.max_prepare_time = max_prepare_time
        self.min_prepare_time = min_prepare_time

    def time_estimate_to_prepare_order(self) -> SimTime:
        time_to_prepare = None
        while time_to_prepare is None or time_to_prepare < self.min_prepare_time or time_to_prepare > self.max_prepare_time:   # TODO: Pesquisar uma forma de não afetar a distribuição
            time_to_prepare = round(random.expovariate(1 / self.order_production_time_rate))
        return time_to_prepare
