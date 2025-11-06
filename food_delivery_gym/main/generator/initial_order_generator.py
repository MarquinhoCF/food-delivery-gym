from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.initial_generator import InitialGenerator
from food_delivery_gym.main.order.order import Order


class InitialOrderGenerator(InitialGenerator):
    def __init__(self, num_orders):
        self.num_orders = num_orders

    def run(self, env: FoodDeliverySimpyEnv):
        for _ in range(self.num_orders):
            establishment = self.rng.choice(env.state.establishments, size=None)
            customer = self.rng.choice(env.state.customers, size=None)

            items = self.rng.choice(establishment.catalog.items, size=2, replace=False).tolist()

            order = Order(customer, establishment, env.now, items)

            env.state.add_orders([order])

            customer.place_order(order, establishment)
