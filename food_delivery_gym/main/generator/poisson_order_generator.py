from typing import Optional

from food_delivery_gym.main.actors.resume import ResumeCursor
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
    estimated_num_orders : int
        Número total de pedidos a serem gerados.
    time_window : float
        Janela de tempo total para geração dos pedidos (em minutos).
    lambda_rate : float, optional
        Taxa média de chegada (pedidos por unidade de tempo).
        Se None, será calculada como estimated_num_orders / time_window.
    """

    def __init__(self, estimated_num_orders: int, time_window: float, lambda_rate: float = None, rng=None):
        super().__init__(rng=rng)

        if estimated_num_orders <= 0:
            raise ValueError("estimated_num_orders deve ser maior que 0")
        if time_window <= 0:
            raise ValueError("time_window deve ser maior que 0")

        self.estimated_num_orders = estimated_num_orders
        self.time_window = time_window
        self.lambda_rate = lambda_rate or (estimated_num_orders / time_window)

        self.current_order_id = 1

        self.arrival_times = self.generate_arrival_times()

    def get_number_of_orders_generated(self) -> int:
        return len(self.arrival_times)

    # Geração de chegadas (Poisson homogêneo)
    def generate_arrival_times(self) -> list:
        return self.sample_arrivals_after(0)

    def sample_arrivals_after(self, now: float) -> list:
        """Amostra chegadas em (now, time_window] com o RNG atual do gerador."""
        arrival_times = []
        current_time = now

        while current_time < self.time_window:
            interarrival = self.rng.exponential(1.0 / self.lambda_rate)
            current_time += interarrival
            if current_time <= self.time_window:
                arrival_times.append(current_time)

        return arrival_times

    def replace_unrealized_arrivals(self, now: float) -> float | None:
        """
        Substitui chegadas ainda não criadas por uma amostra independente a partir de `now`.

        O prefixo já realizado permanece para o índice coincidir com `current_order_id`.
        Retorna a espera até a primeira chegada nova, ou None se não houver futuro.
        """
        pending_index = max(0, self.current_order_id - 1)
        future = self.sample_arrivals_after(now)
        self.arrival_times = list(self.arrival_times[:pending_index]) + future
        if not future:
            return None
        return float(future[0] - now)

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

        env.state.add_customers([customer])
        env.state.add_orders([order])
        customer.place_order(order, establishment)

    def generate(self, env: FoodDeliverySimpyEnv, *, resume: Optional[ResumeCursor] = None):
        r = resume or ResumeCursor()
        start_index = int(r.extras.get("arrival_index", 0))

        # Espera já iniciada (clone copy, ou retarget do próximo pedido): cria esse pedido
        # e só depois segue a lista. Sem remaining, arrival_index só posiciona o loop.
        if r.has_pending_remaining():
            yield env.timeout(r.delay(0))
            establishment = self.rng.choice(env.state.establishments, size=None)
            self.process_establishment(env, establishment)
            start_index += 1

        for arrival_time in self.arrival_times[start_index:]:
            wait_time = arrival_time - env.now
            if wait_time > 0:
                yield env.timeout(wait_time)

            establishment = self.rng.choice(env.state.establishments, size=None)
            self.process_establishment(env, establishment)
