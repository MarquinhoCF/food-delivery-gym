from food_delivery_gym.main.cost.cost_function import CostFunction
from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.map.map import Map
from food_delivery_gym.main.order.order_status import OrderStatus
from food_delivery_gym.main.route.route_segment import RouteSegment


class MarginalRouteCostFunction(CostFunction):
    def __init__(self, objective: int = 1):
        """
        :param objective: Define o critério de custo marginal.
            1 - Impacto no tempo (delay marginal).
            2 - Impacto na distância (distância marginal).
        """
        self.objective = objective
        self.MAX_PENALTY = float('inf')
    
    @classmethod
    def get_cost_objective(cls, objective: int) -> int:
        if objective in [1, 3, 5, 7, 9, 10, 11, 12, 13]:
            return 1  # baseado em tempo de entrega
        elif objective in [2, 4, 6, 8]:
            return 2  # baseado em custo de operação (distância)
        else:
            raise ValueError(f"Objetivo {objective} não reconhecido.")

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

    def marginal_delay(self, map: Map, driver: Driver, route_segment: RouteSegment) -> float:
        # Custo da nova rota a partir da última posição válida do motorista
        new_segment_delay = map.estimated_time(
            driver.get_last_valid_coordinate(),
            route_segment.coordinate,
            driver.movement_rate
        )

        return new_segment_delay

    def marginal_distance(self, map: Map, driver: Driver, route_segment: RouteSegment) -> float:
        # Distância incremental a partir da última posição válida do motorista
        new_segment_distance = map.distance(
            driver.get_last_valid_coordinate(),
            route_segment.coordinate
        )

        return new_segment_distance

    def cost(self, map: Map, driver: Driver, route_segment: RouteSegment) -> float:
        if self.objective == 1:
            return self.marginal_delay(map, driver, route_segment) + self.penalty(route_segment)
        elif self.objective == 2:
            return self.marginal_distance(map, driver, route_segment) + self.penalty(route_segment)
        else:
            raise ValueError("Objetivo inválido. Use 1 para delay marginal ou 2 para distância marginal.")