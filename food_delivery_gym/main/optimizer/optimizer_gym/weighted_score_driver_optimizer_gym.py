from typing import List

from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.map.map import Map
from food_delivery_gym.main.optimizer.optimizer_gym.optmizer_gym import OptimizerGym
from food_delivery_gym.main.route.route import Route


class WeightedScoreDriverOptimizerGym(OptimizerGym):

    def __init__(
        self,
        environment: FoodDeliveryGymEnv,
        theta_distance: float = 3.0,
        theta_queue: float = 5.0,
        theta_velocity: float = 1.0,
    ):
        super().__init__(environment)
        self.theta_distance = theta_distance
        self.theta_queue = theta_queue
        self.theta_velocity = theta_velocity

    def get_title(self):
        return "Otimizador de Score Ponderado (CFA)"

    def score(self, map: Map, driver: Driver, route: Route) -> float:
        distance = map.distance(
            driver.get_last_valid_coordinate(),
            route.route_segments[0].coordinate,
        )
        queue_size = driver.get_number_of_orders_in_list()
        velocity = driver.get_velocity()

        return (
            self.theta_distance * distance
            + self.theta_queue * queue_size
            - self.theta_velocity * velocity
        )

    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        # drivers = list(filter(lambda driver: driver.current_route is None or
        # driver.current_route.size() <= 1, drivers))
        map = self.gym_env.simpy_env.map
        best_driver = min(drivers, key=lambda driver: self.score(map, driver, route))
        return drivers.index(best_driver)