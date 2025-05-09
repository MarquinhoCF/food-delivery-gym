from abc import abstractmethod, ABC

from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.generator import Generator


class TimeShiftGenerator(Generator, ABC):

    def __init__(self, function=lambda time: 1, time_shift=1):
        super().__init__()
        self.function = function
        self.time_shift = time_shift

    @abstractmethod
    def run(self, env: FoodDeliverySimpyEnv): pass

    def range(self, env: FoodDeliverySimpyEnv):
        return range(round(self.function(env.now)))

    def generate(self, env: FoodDeliverySimpyEnv):
        while True:
            self.run(env)
            yield env.timeout(self.time_shift)
