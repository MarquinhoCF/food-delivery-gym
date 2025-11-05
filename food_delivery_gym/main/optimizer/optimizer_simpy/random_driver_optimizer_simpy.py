from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.optimizer.optimizer_simpy.optimizer_simpy import OptimizerSimpy
from food_delivery_gym.main.route.route import Route
from food_delivery_gym.main.utils.random_manager import RandomManager


class RandomDriverOptimizerSimpy(OptimizerSimpy):
    def __init__(self):
        self.rng = RandomManager().get_random_instance()

    def select_driver(self, env: FoodDeliverySimpyEnv, route: Route):
        drivers = env.available_drivers(route)
        return self.rng.choice(drivers, size=None) if len(drivers) > 0 else None
