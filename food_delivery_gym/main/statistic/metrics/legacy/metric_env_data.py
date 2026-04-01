from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.statistic.metrics.metric import Metric


class MetricEnvData(Metric):

    def __init__(self, environment: FoodDeliverySimpyEnv):
        self.environment = environment

    def view(self, ax) -> None:
        pass
