from typing import List

from matplotlib.ticker import FuncFormatter
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.statistic.metric import Metric


class DriverOrdersDeliveredMetric(Metric):
    def __init__(self, environment: FoodDeliverySimpyEnv, drivers_statistics=None):
        super().__init__(environment)
        self.drivers_statistics = drivers_statistics

    def view(self, ax) -> None:
        if self.drivers_statistics is not None:
            est_ids = list(self.drivers_statistics.keys())
            means = [self.drivers_statistics[e]['orders_delivered']['mean'] for e in est_ids]
            medians = [self.drivers_statistics[e]['orders_delivered']['median'] for e in est_ids]
            modes = [self.drivers_statistics[e]['orders_delivered']['mode'] for e in est_ids]
            std_devs = [self.drivers_statistics[e]['orders_delivered']['std_dev'] for e in est_ids]

            # Criando o gráfico
            ax.errorbar(est_ids, means, yerr=std_devs, fmt='o', label='Média', capsize=5)
            ax.plot(est_ids, medians, marker='s', linestyle='', label='Mediana')
            ax.plot(est_ids, modes, marker='^', linestyle='', label='Moda')

            # Adicionando títulos e legendas
            ax.set_xlabel('Motoristas')
            ax.set_ylabel('Pedidos Entregues')
            ax.set_title('Estatísticas dos Pedidos Entregues por Motorista')
            ax.legend()
            ax.grid(True)
            
        else:
            drivers = self.environment.state.drivers

            # Usa os valores pontuais da simulação atual
            ids = [driver.driver_id for driver in drivers]
            orders_delivered: List[int] = [int(driver.orders_delivered) for driver in drivers]
            title = 'Orders Delivered per Driver'
            # print("\nPedidos Entregues por Motorista:")

            # # TODO: Logs
            # for driver_id, count in zip(ids, orders_delivered):
            #     print(f"Motorista {driver_id}: {count} pedidos entregues")

            ax.barh(ids, orders_delivered, color='blue')
            ax.set_xlabel('Orders Delivered')
            ax.set_ylabel('Drivers')
            ax.set_title(title)

            ax.set_yticks(ids)
            ax.set_yticklabels([str(int(driver_id)) for driver_id in ids])

            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

        