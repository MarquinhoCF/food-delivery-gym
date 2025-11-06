from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.time_shift_generator import TimeShiftGenerator
from food_delivery_gym.main.order.order import Order


class TimeShiftOrderGenerator(TimeShiftGenerator):
    def __init__(self, function, time_shift=1):
        super().__init__(function, time_shift)

    def run(self, env: FoodDeliverySimpyEnv):

        orders = []

        for _ in self.range(env):
            customer = self.rng.choice(env.state.customers, size=None)
            establishment = self.rng.choice(env.state.establishments, size=None)
            items = self.rng.choice(establishment.catalog.items, size=2, replace=False).tolist()
            order = Order(customer, establishment, env.now, items)
            orders.append(order)
            customer.place_order(order, establishment)

        env.state.add_orders(orders)
