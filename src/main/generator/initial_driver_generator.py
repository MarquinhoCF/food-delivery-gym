from src.main.base.dimensions import Dimensions
from src.main.driver.capacity import Capacity
from src.main.driver.driver import Driver, DriverStatus
from src.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from src.main.generator.initial_generator import InitialGenerator


class InitialDriverGenerator(InitialGenerator):
    def __init__(self, num_drivers, vel_drivers, reward_objective, desconsider_capacity=False):
        super().__init__()
        self.num_drivers = num_drivers
        self.vel_drivers = vel_drivers
        self.reward_objective = reward_objective
        self.desconsider_capacity = desconsider_capacity

    def run(self, env: FoodDeliverySimpyEnv):
        capacity = Capacity(Dimensions(100, 100, 100, 100))

        drivers = [
            Driver(
                id=i+1,
                environment=env,
                coordinate=env.map.random_point(),
                desconsider_capacity=self.desconsider_capacity,
                capacity=None if self.desconsider_capacity else capacity,
                available=True,
                status=DriverStatus.AVAILABLE,
                movement_rate=self.rng.uniform(self.vel_drivers[0], self.vel_drivers[1]),
                # Gerar uma cor aleatória RGB para cada motorista
                color=(self.rng.randint(0, 255), self.rng.randint(0, 255), self.rng.randint(0, 255)),
                reward_objective=self.reward_objective,
            ) for i in range(self.num_drivers)
        ]
        env.add_drivers(drivers)
