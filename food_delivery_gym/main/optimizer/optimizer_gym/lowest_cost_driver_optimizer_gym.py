from typing import List

from food_delivery_gym.main.cost.cost_function import CostFunction
from food_delivery_gym.main.cost.route_cost_function import RouteCostFunction
from food_delivery_gym.main.cost.marginal_route_cost_function import MarginalRouteCostFunction
from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer.optimizer_gym.optmizer_gym import OptimizerGym
from food_delivery_gym.main.route.route import Route


class LowestCostDriverOptimizerGym(OptimizerGym):
    
    def __init__(self, environment: FoodDeliveryGymEnv, cost_function: CostFunction | None = None):
        super().__init__(environment)
        self.cost_function = cost_function

    def get_cost_for_driver(self, driver: Driver, route: Route):
        map = self.gym_env.simpy_env.map
        return self.cost_function.cost(map, driver, route.route_segments[0])
    
    def get_title(self):
        if isinstance(self.cost_function, RouteCostFunction):
            return "Otimizador do Motorista com Menor Custo de Rota"
        elif isinstance(self.cost_function, MarginalRouteCostFunction):
            return "Otimizador do Motorista com Menor Custo Marginal de Rota"
        return "Otimizador do Motorista com Menor Custo"

    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        # drivers = list(filter(lambda driver: driver.current_route is None or
        # driver.current_route.size() <= 1, drivers))
        selected_driver = min(drivers, key=lambda driver: self.get_cost_for_driver(driver, route))
        return drivers.index(selected_driver)
