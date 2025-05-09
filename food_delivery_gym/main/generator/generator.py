from abc import abstractmethod, ABC

from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.utils.random_manager import RandomManager

class Generator(ABC):

    def __init__(self):
        self.rng = RandomManager().get_random_instance()

    @abstractmethod
    def generate(self, env: FoodDeliverySimpyEnv): pass