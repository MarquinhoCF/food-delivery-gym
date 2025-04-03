from typing import List

from src.main.driver.driver import Driver
from src.main.map.map import Map
from src.main.optimizer.optimizer_gym.optmizer_gym import OptimizerGym
from src.main.route.route import Route


class NearestDriverOptimizerGym(OptimizerGym):

    def compare_distance(self, map: Map, driver: Driver, route: Route):
        return map.distance(driver.get_last_coordinate_from_routes_list(), route.route_segments[0].coordinate)
    
    def get_title(self):
        return "Otimizador do Motorista Mais Próximo"

    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        # drivers = list(filter(lambda driver: driver.current_route is None or
        # driver.current_route.size() <= 1, drivers))
        nearest_driver = min(drivers, key=lambda driver: self.compare_distance(self.gym_env.simpy_env.map, driver, route))
        return drivers.index(nearest_driver)
