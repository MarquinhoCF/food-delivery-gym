import numpy as np
from food_delivery_gym.main.base.geometry import point_in_gauss_circle
from food_delivery_gym.main.customer.customer import Customer
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.generator import Generator
from food_delivery_gym.main.order.order import Order


class PoissonOrderGenerator(Generator):
    """
    Gerador de pedidos baseado em Processo de Poisson Homogêneo.

    Parameters
    ----------
    total_orders : int
        Número total de pedidos a serem gerados.
    time_window : float
        Janela de tempo total para geração dos pedidos (em minutos).
    lambda_rate : float, optional
        Taxa média de chegada (pedidos por unidade de tempo).
        Se None, será calculada como total_orders / time_window.
    """

    def __init__(self, total_orders: int, time_window: float, lambda_rate: float = None):
        super().__init__()

        if total_orders <= 0:
            raise ValueError("total_orders deve ser maior que 0")
        if time_window <= 0:
            raise ValueError("time_window deve ser maior que 0")

        self.total_orders = total_orders
        self.time_window = time_window
        self.lambda_rate = lambda_rate or (total_orders / time_window)

        self.current_order_id = 1
        self.orders_generated = 0

    # Geração de chegadas (Poisson homogêneo)
    def generate_arrival_times(self) -> list:
        arrival_times = []
        current_time = 0

        while len(arrival_times) < self.total_orders and current_time < self.time_window:
            interarrival = self.rng.exponential(1.0 / self.lambda_rate)
            current_time += interarrival
            if current_time <= self.time_window:
                arrival_times.append(current_time)

        return arrival_times[:self.total_orders]

    # Lógica de criação dos pedidos
    def process_establishment(self, env: FoodDeliverySimpyEnv, establishment):
        customer = Customer(
            id=self.current_order_id,
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

        items = self.rng.choice(establishment.catalog.items, size=2, replace=False).tolist()

        order = Order(
            id=self.current_order_id,
            customer=customer,
            establishment=establishment,
            request_date=env.now,
            items=items,
        )

        self.current_order_id += 1
        self.orders_generated += 1

        env.state.add_customers([customer])
        env.state.add_orders([order])
        customer.place_order(order, establishment)

    def generate(self, env: FoodDeliverySimpyEnv):
        arrival_times = self.generate_arrival_times()

        for arrival_time in arrival_times:
            wait_time = arrival_time - env.now
            if wait_time > 0:
                yield env.timeout(wait_time)

            if self.orders_generated >= self.total_orders:
                break

            establishment = self.rng.choice(env.state.establishments, size=None)
            self.process_establishment(env, establishment)