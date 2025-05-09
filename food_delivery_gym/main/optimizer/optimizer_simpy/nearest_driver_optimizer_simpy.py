from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.optimizer.optimizer_simpy.optimizer_simpy import OptimizerSimpy
from food_delivery_gym.main.route.route import Route


class NearestDriverOptimizerSimpy(OptimizerSimpy):

    def compare_distance(self, env: FoodDeliverySimpyEnv, driver: Driver, route: Route):
        return env.map.distance(driver.coordinate, route.route_segments[0].coordinate)

    def select_driver(self, env: FoodDeliverySimpyEnv, route: Route):
        drivers = env.available_drivers(route)
        # drivers = list(filter(lambda driver: driver.current_route is None or
        # driver.current_route.size() <= 1, drivers))
        nearest_driver = min(drivers, key=lambda driver: self.compare_distance(env, driver, route))
        return nearest_driver
