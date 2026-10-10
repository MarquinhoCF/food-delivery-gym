from typing import List

from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.map.map import Map
from food_delivery_gym.main.optimizer.optimizer_gym.optmizer_gym import OptimizerGym
from food_delivery_gym.main.route.route import Route


class NearestDriverOptimizerGym(OptimizerGym):

    def compare_distance(self, map: Map, driver: Driver, route: Route):
        return map.distance(driver.get_last_valid_coordinate(), route.route_segments[0].coordinate)
    
    def get_title(self):
        return "Otimizador do Motorista Mais Próximo"

    def ranked_actions(self, obs: dict, drivers: List[Driver], route: Route, *, rng=None) -> list[int]:
        del obs, rng # obs e rng não são usados
        map_ = self.gym_env.simpy_env.map
        return sorted(
            range(len(drivers)),
            key=lambda i: (self.compare_distance(map_, drivers[i], route), i),
        )

    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        # drivers = list(filter(lambda driver: driver.current_route is None or
        # driver.current_route.size() <= 1, drivers))
        nearest_driver = min(drivers, key=lambda driver: self.compare_distance(self.gym_env.simpy_env.map, driver, route))
        return drivers.index(nearest_driver)
