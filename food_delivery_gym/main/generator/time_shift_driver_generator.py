from food_delivery_gym.main.base.dimensions import Dimensions
from food_delivery_gym.main.driver.capacity import Capacity
from food_delivery_gym.main.driver.driver import Driver, DriverStatus
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.time_shift_generator import TimeShiftGenerator


class TimeShiftDriverGenerator(TimeShiftGenerator):
    def __init__(self, function, time_shift=1):
        super().__init__(function, time_shift)

    def run(self, env: FoodDeliverySimpyEnv):
        capacity = Capacity(Dimensions(10, 10, 10, 10))
        drivers = [
            Driver(
                environment=env,
                coordinate=env.map.random_point(),
                capacity=capacity,
                available=True,
                status=DriverStatus.AVAILABLE,
                movement_rate=self.rng.uniform(1, 30),
            ) for _ in self.range(env)
        ]
        env.add_drivers(drivers)
