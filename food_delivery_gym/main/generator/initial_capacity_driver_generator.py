from food_delivery_gym.main.base.dimensions import Dimensions
from food_delivery_gym.main.driver.capacity import Capacity
from food_delivery_gym.main.driver.capacity_driver import CapacityDriver
from food_delivery_gym.main.driver.driver_status import DriverStatus
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.initial_generator import InitialGenerator


class InitialCapacityDriverGenerator(InitialGenerator):
    def __init__(self, num_drivers, vel_drivers, bag_capacity, reward_objective, rng=None):
        super().__init__(rng=rng)
        self.num_drivers = num_drivers
        self.vel_drivers = vel_drivers
        self.bag_capacity = bag_capacity
        self.reward_objective = reward_objective

    def run(self, env: FoodDeliverySimpyEnv):
        capacity = Capacity(Dimensions(*self.bag_capacity))
        drivers = [
            CapacityDriver(
                id=i + 1,
                environment=env,
                coordinate=env.map.random_point(),
                available=True,
                capacity=capacity,
                status=DriverStatus.AVAILABLE,
                movement_rate=self.rng.integers(self.vel_drivers[0], self.vel_drivers[1] + 1),
                color=(
                    self.rng.integers(0, 255 + 1),
                    self.rng.integers(0, 255 + 1),
                    self.rng.integers(0, 255 + 1),
                ),
                reward_objective=self.reward_objective,
            )
            for i in range(self.num_drivers)
        ]
        env.add_drivers(drivers)
