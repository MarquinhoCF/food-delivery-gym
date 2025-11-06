from food_delivery_gym.main.base.dimensions import Dimensions
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.initial_generator import InitialGenerator
from food_delivery_gym.main.order.item import Item
from food_delivery_gym.main.establishment.catalog import Catalog
from food_delivery_gym.main.establishment.establishment_order_rate import EstablishmentOrderRate


class InitialEstablishmentOrderRateGenerator(InitialGenerator):
    def __init__(self, num_establishments, prepare_time, operating_radius, production_capacity, percentage_allocation_driver, use_estimate: bool = False):
        super().__init__()
        self.num_establishments = num_establishments
        self.prepare_time = prepare_time
        self.operating_radius = operating_radius
        self.production_capacity = production_capacity
        self.percentage_allocation_driver = percentage_allocation_driver
        self.use_estimate = use_estimate

    def run(self, env: FoodDeliverySimpyEnv):
        dimension = Dimensions(1, 1, 1, 1)
        catalog = Catalog([Item(f"type_{i}", dimension, 4) for i in range(5)])
        establishment = [
            EstablishmentOrderRate(
                id=i+1,
                environment=env,
                coordinate=env.map.random_point(),
                available=True,
                catalog=catalog,
                production_capacity=self.rng.integers(self.production_capacity[0], self.production_capacity[1]+1),
                use_estimate=self.use_estimate,
                order_production_time_rate=self.rng.uniform(self.prepare_time[0], self.prepare_time[1]),
                percentage_allocation_driver=self.percentage_allocation_driver,
                max_prepare_time=self.prepare_time[1],
                min_prepare_time=self.prepare_time[0],
                operating_radius=self.rng.integers(self.operating_radius[0], self.operating_radius[1]+1),
            )
            for i in range(self.num_establishments)
        ]
        env.add_establishments(establishment)