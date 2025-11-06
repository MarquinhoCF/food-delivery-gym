import numpy as np
from food_delivery_gym.main.generator.poisson_order_generator import PoissonOrderGenerator


class NonHomogeneousPoissonOrderGenerator(PoissonOrderGenerator):
    """
    Gerador de pedidos baseado em Processo de Poisson Não-Homogêneo (com thinning).

    Parameters
    ----------
    total_orders : int
        Número total de pedidos a serem gerados.
    time_window : float
        Janela de tempo total para geração dos pedidos (em minutos).
    rate_function : callable
        Função que recebe o tempo e retorna a taxa de chegada naquele momento.
    max_rate : float, optional
        Taxa máxima do processo. Se None, é estimada automaticamente.
    """

    def __init__(self, total_orders: int, time_window: float,
                 rate_function: callable, max_rate: float = None):
        super().__init__(total_orders, time_window, lambda_rate=None)

        self.rate_function = rate_function
        if max_rate is None:
            time_samples = np.linspace(0, time_window, 1000)
            rates = [rate_function(t) for t in time_samples]
            self.max_rate = max(rates) * 1.1  # margem de segurança
        else:
            self.max_rate = max_rate

    def get_rate_function(self):
        return self.rate_function

    def generate_arrival_times(self) -> list:
        arrival_times = []
        current_time = 0

        while len(arrival_times) < self.total_orders and current_time < self.time_window:
            interarrival = self.rng.exponential(1.0 / self.max_rate)
            current_time += interarrival
            if current_time > self.time_window:
                break

            acceptance_prob = self.rate_function(current_time) / self.max_rate
            if self.rng.random() < acceptance_prob:
                arrival_times.append(current_time)

        return arrival_times[:self.total_orders]