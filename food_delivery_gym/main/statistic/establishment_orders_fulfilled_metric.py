from typing import List

from matplotlib.ticker import FuncFormatter
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.statistic.metric import Metric


class EstablishmentOrdersFulfilledMetric(Metric):
    def __init__(self, environment: FoodDeliverySimpyEnv, establishments_statistics=None):
        super().__init__(environment)
        self.establishments_statistics = establishments_statistics

    def view(self, ax) -> None:

        if self.establishments_statistics is not None:
            est_ids = list(self.establishments_statistics.keys())
            means = [self.establishments_statistics[e]['orders_fulfilled']['mean'] for e in est_ids]
            medians = [self.establishments_statistics[e]['orders_fulfilled']['median'] for e in est_ids]
            modes = [self.establishments_statistics[e]['orders_fulfilled']['mode'] for e in est_ids]
            std_devs = [self.establishments_statistics[e]['orders_fulfilled']['std_dev'] for e in est_ids]

            # Criando o gráfico
            ax.errorbar(est_ids, means, yerr=std_devs, fmt='o', label='Média', capsize=5)
            ax.plot(est_ids, medians, marker='s', linestyle='', label='Mediana')
            ax.plot(est_ids, modes, marker='^', linestyle='', label='Moda')

            # Adicionando títulos e legendas
            ax.set_xlabel('Estabelecimento')
            ax.set_ylabel('Pedidos Atendidos')
            ax.set_title('Estatísticas dos Pedidos Atendidos por Estabelecimento')
            ax.legend()
            ax.grid(True)

        else:
            establishments = self.environment.state.establishments

            # Usa os valores pontuais da simulação atual
            ids = [establishment.establishment_id for establishment in establishments]
            orders_fulfilled = [int(establishment.orders_fulfilled) for establishment in establishments]
            title = 'Orders Fulfilled per Establishment'
            # print("\nPedidos Atendidos por Estabelecimento:")

            # # TODO: Logs
            # for est_id, count in zip(ids, orders_fulfilled):
            #     print(f"Estabelecimento {est_id}: {count} pedidos atendidos")

            ax.barh(ids, orders_fulfilled, color='skyblue')
            ax.set_xlabel('Orders Fulfilled')
            ax.set_ylabel('Establishments')
            ax.set_title(title)

            ax.set_yticks(ids)
            ax.set_yticklabels([str(int(driver_id)) for driver_id in ids])

            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

        