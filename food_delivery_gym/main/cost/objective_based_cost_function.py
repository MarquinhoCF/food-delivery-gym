
from food_delivery_gym.main.base.types import Number
from food_delivery_gym.main.cost.cost_function import CostFunction
from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.map.map import Map
from food_delivery_gym.main.order.order_status import OrderStatus
from food_delivery_gym.main.route.route_segment import RouteSegment


class ObjectiveBasedCostFunction(CostFunction):
    def __init__(self, objective: int = 1):
        """
        :param objective: Define o critério de custo.
            1 - Considera apenas o delay.
            2 - Considera apenas a distância.
        """
        self.objective = objective
        self.MAX_PENALTY = float('inf')

    def penalty(self, route_segment: RouteSegment):
        if route_segment.is_pickup() and (
            route_segment.order.status <= OrderStatus.DRIVER_ACCEPTED
            or route_segment.order.status <= OrderStatus.READY_AND_DRIVER_ACCEPTED
            or route_segment.order.status <= OrderStatus.PREPARING_AND_DRIVER_ACCEPTED
        ):
            return 0
        if route_segment.is_delivery() and route_segment.order.status <= OrderStatus.PICKED_UP:
            return 0
        return self.MAX_PENALTY

    def delay(self, map: Map, driver: Driver, route_segment: RouteSegment):
        current_delay = 0

        if driver.current_route_segment is not None:
            current_delay = driver.estimate_total_busy_time()

        new_segment_delay = map.estimated_time(
            driver.coordinate,
            route_segment.coordinate,
            driver.movement_rate
        )
        
        return current_delay + new_segment_delay

    def distance(self, map: Map, driver: Driver, route_segment: RouteSegment):
        current_distance = 0

        if driver.current_route_segment is not None:
            current_distance = driver.calculate_total_distance_to_travel()

        new_segment_distance = map.distance(
            driver.coordinate,
            route_segment.coordinate
        )

        return current_distance + new_segment_distance

    def cost(self, map: Map, driver: Driver, route_segment: RouteSegment) -> Number:
        if self.objective == 1:
            return self.delay(map, driver, route_segment) + self.penalty(route_segment)
        elif self.objective == 2:
            return self.distance(map, driver, route_segment) + self.penalty(route_segment)
        else:
            raise ValueError("Objetivo inválido. Use 1 para delay ou 2 para distância.")
