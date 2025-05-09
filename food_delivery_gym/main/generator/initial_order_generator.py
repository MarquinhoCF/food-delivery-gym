from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.initial_generator import InitialGenerator
from food_delivery_gym.main.order.order import Order


class InitialOrderGenerator(InitialGenerator):
    def __init__(self, num_orders):
        self.num_orders = num_orders

    def run(self, env: FoodDeliverySimpyEnv):
        for _ in range(self.num_orders):
            establishment = self.rng.choice(env.state.establishments)
            customer = self.rng.choice(env.state.customers)

            items = self.rng.sample(establishment.catalog.items, 2)

            order = Order(customer, establishment, env.now, items)

            env.state.add_orders([order])

            customer.place_order(order, establishment)
