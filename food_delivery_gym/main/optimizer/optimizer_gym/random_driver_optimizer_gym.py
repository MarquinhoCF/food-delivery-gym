from typing import List

from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.optimizer.optimizer_gym.optmizer_gym import OptimizerGym
from food_delivery_gym.main.route.route import Route


class RandomDriverOptimizerGym(OptimizerGym):

    def get_title(self):
        return "Otimizador do Motorista Aleatório"

    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        return self.gym_env.action_space.sample()
