from typing import Optional

from food_delivery_gym.main.base.types import Coordinate, Number
from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.driver.driver_status import DriverStatus
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.route.route import Route


class ReactiveDriver(Driver):
    def __init__(
            self,
            id: Number,
            environment: FoodDeliverySimpyEnv,
            coordinate: Coordinate,
            available: bool,
            max_distance: Number,
            status: Optional[DriverStatus] = DriverStatus.AVAILABLE,
            movement_rate: Optional[Number] = 5,
            reward_objective: Optional[Number] = 1,
            start_processes: bool = True,
    ):
        super().__init__(
            id=id,
            environment=environment,
            coordinate=coordinate,
            available=available,
            status=status,
            movement_rate=movement_rate,
            reward_objective=reward_objective,
            start_processes=False,
        )
        self.max_distance = max_distance
        if start_processes:
            self.process(self.search_order())

    def accept_route_condition(self, route: Route):
        default_condition = super().accept_route_condition(route)
        order = route.route_segments[0].order
        pickup_coordinate = self.environment.map.distance(self.coordinate, order.establishment.coordinate)
        return default_condition and pickup_coordinate <= self.max_distance

    def search_order(self):
        while True:
            if self.available and self.status is DriverStatus.AVAILABLE and self.environment.count_ready_orders() > 0:
                search_timeout = self.timeout(20)
                order_request = self.environment.ready_orders.get(self.accept_route_condition)
                search_result = yield self.environment.any_of([order_request, search_timeout])
                if order_request in search_result:
                    self.accept_route(order_request.value)
            yield self.timeout(1)
