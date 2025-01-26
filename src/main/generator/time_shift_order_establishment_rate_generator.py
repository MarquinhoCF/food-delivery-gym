from src.main.base.geometry import point_in_gauss_circle
from src.main.customer.customer import Customer
from src.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from src.main.generator.time_shift_generator import TimeShiftGenerator
from src.main.order.order import Order


class TimeShiftOrderEstablishmentRateGenerator(TimeShiftGenerator):

    def __init__(self, function, time_shift=1, max_orders=None):
        super().__init__(function, time_shift)
        self.max_orders = max_orders
        self.current_order_id = 1

    def process_establishment(self, env: FoodDeliverySimpyEnv, establishment):

        if establishment.establishment_id:

            customer = Customer(
                environment=env,
                coordinate=point_in_gauss_circle(
                    establishment.coordinate,
                    establishment.operating_radius,
                    env.map.size,
                    self.rng
                ),
                available=True,
                single_order=True
            )

            items = self.rng.sample(establishment.catalog.items, 2)

            order = Order(
                customer, 
                establishment, 
                env.now, 
                items, 
                id=self.current_order_id
            )

            self.current_order_id += 1

            env.state.add_customers([customer])
            env.state.add_orders([order])

            customer.place_order(order, establishment)

    def run(self, env: FoodDeliverySimpyEnv):
        for _ in self.range(env):
            # Verificar se o número de pedidos foi atingido
            if self.max_orders and (env.state.get_length_orders() >= self.max_orders):
                # TODO: Logs
                # print(f'Número máximo de pedidos atingido: {self.max_orders}')
                return

            establishment = self.rng.choice(env.state.establishments)
            #establishment = env.state.establishments[0]
            self.process_establishment(env, establishment)
