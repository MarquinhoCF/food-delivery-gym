from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.initial_generator import InitialGenerator
from food_delivery_gym.main.order.item import Item
from food_delivery_gym.main.establishment.catalog import Catalog
from food_delivery_gym.main.establishment.establishment import Establishment


class InitialEstablishmentGenerator(InitialGenerator):
    def __init__(self, num_establishments, use_estimate: bool = False, rng=None):
        super().__init__(rng=rng)
        self.num_establishments = num_establishments
        self.use_estimate = use_estimate

    def run(self, env: FoodDeliverySimpyEnv):
        # TODO: ao criar o catálogo, sortear preparation_time por item uma vez
        # (uniforme 8–20, mesma política de time_estimate_to_prepare_order) e usar
        # esse tempo estimado de chegada no accept em vez de sortear por pedido.
        catalog = Catalog([Item(f"type_{i}") for i in range(5)])
        establishments = [
            Establishment(
                id=i + 1,
                environment=env,
                coordinate=env.map.random_point(),
                available=True,
                catalog=catalog,
                use_estimate=self.use_estimate,
            )
            for i in range(self.num_establishments)
        ]
        env.add_establishments(establishments)
