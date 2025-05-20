from abc import ABC, abstractmethod

from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv


class Metric(ABC):

    def __init__(self, environment: FoodDeliverySimpyEnv):
        self.environment = environment

    @abstractmethod
    def view(self, ax) -> None:
        pass
