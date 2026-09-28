from food_delivery_gym.main.base.types import Number
from food_delivery_gym.main.cost.cost_function import CostFunction
from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.map.map import Map
from food_delivery_gym.main.route.route_segment import RouteSegment


class WeightedScoreCostFunction(CostFunction):
    """Score ponderado (CFA): distância, fila e velocidade."""

    label = "Score Ponderado"

    def __init__(
        self,
        theta_distance: float = 3.0,
        theta_queue: float = 5.0,
        theta_velocity: float = 1.0,
    ):
        self.theta_distance = theta_distance
        self.theta_queue = theta_queue
        self.theta_velocity = theta_velocity

    def cost(self, map: Map, driver: Driver, route_segment: RouteSegment) -> Number:
        distance = map.distance(
            driver.get_last_valid_coordinate(),
            route_segment.coordinate,
        )
        queue_size = driver.get_number_of_orders_in_list()
        velocity = driver.get_velocity()
        return (
            self.theta_distance * distance
            + self.theta_queue * queue_size
            - self.theta_velocity * velocity
        )
