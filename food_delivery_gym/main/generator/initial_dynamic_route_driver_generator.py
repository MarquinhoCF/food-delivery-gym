from food_delivery_gym.main.driver.driver import DriverStatus
from food_delivery_gym.main.driver.dynamic_route_driver import DynamicRouteDriver
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.initial_generator import InitialGenerator


class InitialDynamicRouteDriverGenerator(InitialGenerator):
    def __init__(self, num_drivers, vel_drivers, max_delay_percentage, max_capacity, reward_objective):
        super().__init__()
        self.num_drivers = num_drivers
        self.vel_drivers = vel_drivers
        self.max_delay_percentage = max_delay_percentage
        self.max_capacity = max_capacity
        self.reward_objective = reward_objective

    def run(self, env: FoodDeliverySimpyEnv):
        drivers = [
            DynamicRouteDriver(
                id=i+1,
                environment=env,
                coordinate=env.map.random_point(),
                available=True,
                max_delay_percentage=self.max_delay_percentage,
                max_capacity=self.max_capacity,
                status=DriverStatus.AVAILABLE,
                movement_rate=self.rng.integers(self.vel_drivers[0], self.vel_drivers[1]+1),
                # Gerar uma cor aleatória RGB para cada motorista
                color=(self.rng.integers(0, 255+1), self.rng.integers(0, 255+1), self.rng.integers(0, 255+1)),
                reward_objective=self.reward_objective,
            ) for i in range(self.num_drivers)
        ]
        env.add_drivers(drivers)
