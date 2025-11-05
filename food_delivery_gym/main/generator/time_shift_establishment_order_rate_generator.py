from food_delivery_gym.main.base.dimensions import Dimensions
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.time_shift_generator import TimeShiftGenerator
from food_delivery_gym.main.order.item import Item
from food_delivery_gym.main.establishment.catalog import Catalog
from food_delivery_gym.main.establishment.establishment_order_rate import EstablishmentOrderRate


class TimeShiftEstablishmentOrderRateGenerator(TimeShiftGenerator):
    def __init__(self, function, time_shift, use_estimate: bool = False):
        super().__init__(function, time_shift)
        self.use_estimate = use_estimate

    def run(self, env: FoodDeliverySimpyEnv):
        dimension = Dimensions(1, 1, 1, 1)
        catalog = Catalog([Item(f"type_{i}", dimension, 4) for i in range(5)])
        establishments = [
            EstablishmentOrderRate(
                environment=env,
                coordinate=env.map.random_point(),
                available=True,
                catalog=catalog,
                production_capacity=1,
                use_estimate=self.use_estimate,
                order_production_time_rate=self.rng.integers(1, 10+1),
                operating_radius=self.rng.integers(10, 30+1)
            )
            for _ in self.range(env)
        ]
        env.add_establishments(establishments)
