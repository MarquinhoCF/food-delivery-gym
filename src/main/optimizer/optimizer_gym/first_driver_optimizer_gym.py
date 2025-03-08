from typing import List
from src.main.driver.driver import Driver
from src.main.optimizer.optimizer_gym.optmizer_gym import OptimizerGym
from src.main.route.route import Route


class FirstDriverOptimizerGym(OptimizerGym):
    
    def get_title(self):
        return "Otimizador do Primeiro Motorista"

    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        # drivers = list(filter(lambda driver: driver.current_route is None or
        # driver.current_route.size() <= 1, drivers))
        
        # Retornando o index do primeiro motorista
        return 0 if len(drivers) > 0 else None
