from collections import defaultdict

from matplotlib.ticker import MultipleLocator

from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.events.event_type import EventType
from food_delivery_gym.main.statistic.metric import Metric


class OrderCurveMetric(Metric):
    def __init__(self, environment: FoodDeliverySimpyEnv):
        super().__init__(environment)

    def view(self, ax) -> None:
        events = filter(
            lambda event: event.event_type in [
                EventType.CUSTOMER_PLACED_ORDER,
                EventType.DRIVER_DELIVERED_ORDER,
                EventType.ESTABLISHMENT_FINISHED_ORDER,
            ],
            self.environment.events
        )

        # Agrupar e contar os dados por status
        status_counts = defaultdict(lambda: defaultdict(int))
        for item in events:
            status_counts[item.event_type][item.time] += 1

        # Preparar os dados para plotar
        status_series = {}
        for status, times in status_counts.items():
            sorted_times = sorted(times.items())
            times, counts = zip(*sorted_times)
            status_series[status] = (times, counts)

        # Plotar os dados como pontos individuais
        for status, (times, counts) in status_series.items():
            ax.scatter(
                times,
                counts,
                label=status.name.lower(),
                s=25
            )

        # Configurações dos eixos para números inteiros
        ax.yaxis.set_major_locator(MultipleLocator(1))  # Ticks no eixo Y a cada 1 unidade

        # Garantir que o eixo Y sempre comece em 0
        ax.set_ylim(bottom=0)

        # Configurações do gráfico
        ax.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of orders', fontsize=11, fontweight='bold')
        ax.set_title('Number of orders by state over time', fontsize=12, fontweight='bold', pad=15)
        ax.legend(title='Order States', fontsize=10, title_fontsize='11')
        ax.grid(False)


