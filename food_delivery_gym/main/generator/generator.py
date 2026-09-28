from abc import abstractmethod, ABC

import numpy as np

from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.utils.rng_factory import RngFactory


class Generator(ABC):

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else RngFactory().next()

    @abstractmethod
    def generate(self, env: FoodDeliverySimpyEnv): pass
