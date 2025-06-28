from typing import List

from matplotlib.ticker import FuncFormatter
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.statistic.metric import Metric


class EstablishmentMaxOrdersInQueueMetric(Metric):
    def __init__(self, environment: FoodDeliverySimpyEnv, establishments_statistics=None):
        super().__init__(environment)
        self.establishments_statistics = establishments_statistics

    def view(self, ax) -> None:
        if self.establishments_statistics is not None:
            est_ids = list(self.establishments_statistics.keys())
            means = [self.establishments_statistics[e]['max_orders_in_queue']['mean'] for e in est_ids]
            medians = [self.establishments_statistics[e]['max_orders_in_queue']['median'] for e in est_ids]
            modes = [self.establishments_statistics[e]['max_orders_in_queue']['mode'] for e in est_ids]
            std_devs = [self.establishments_statistics[e]['max_orders_in_queue']['std_dev'] for e in est_ids]

            # Criando o gráfico
            ax.errorbar(est_ids, means, yerr=std_devs, fmt='o', label='Média', capsize=5)
            ax.plot(est_ids, medians, marker='s', linestyle='', label='Mediana')
            ax.plot(est_ids, modes, marker='^', linestyle='', label='Moda')

            # Adicionando títulos e legendas
            ax.set_xlabel('Estabelecimento')
            ax.set_ylabel('Máximo de Pedidos na Fila')
            ax.set_title('Estatísticas do Máximo de Pedidos na Fila por Estabelecimento')
            ax.legend()
            ax.grid(True)
        else:
            establishments = self.environment.state.establishments

            # Usa os valores pontuais da simulação atual
            ids = [establishment.establishment_id for establishment in establishments]
            max_orders_in_queue: List[int] = [int(establishment.max_orders_in_queue) for establishment in establishments]
            title = 'Max Orders in Queue per Establishment'
            # print("\nMáximo de Pedidos na Fila por Estabelecimento:")

            # # TODO: Logs
            # for est_id, max_queue in zip(ids, max_orders_in_queue):
            #     print(f"Estabelecimento {est_id}: {max_queue} pedidos na fila")

            ax.barh(ids, max_orders_in_queue, color='orange')
            ax.set_xlabel('Max Orders in Queue')
            ax.set_ylabel('Establishments')
            ax.set_title(title)

            ax.set_yticks(ids)
            ax.set_yticklabels([str(int(driver_id)) for driver_id in ids])

            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

