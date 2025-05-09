from typing import List

from matplotlib.ticker import FuncFormatter
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.statistic.metric import Metric


class DriverTimeWaitingForOrderMetric(Metric):
    def __init__(self, environment: FoodDeliverySimpyEnv, drivers_statistics=None):
        super().__init__(environment)
        self.drivers_statistics = drivers_statistics

    def view(self, ax) -> None:
        if self.drivers_statistics is not None:
            est_ids = list(self.drivers_statistics.keys())
            means = [self.drivers_statistics[e]['time_waiting_for_order']['mean'] for e in est_ids]
            medians = [self.drivers_statistics[e]['time_waiting_for_order']['median'] for e in est_ids]
            modes = [self.drivers_statistics[e]['time_waiting_for_order']['mode'] for e in est_ids]
            std_devs = [self.drivers_statistics[e]['time_waiting_for_order']['std_dev'] for e in est_ids]

            # Criando o gráfico
            ax.errorbar(est_ids, means, yerr=std_devs, fmt='o', label='Média', capsize=5)
            ax.plot(est_ids, medians, marker='s', linestyle='', label='Mediana')
            ax.plot(est_ids, modes, marker='^', linestyle='', label='Moda')

            # Adicionando títulos e legendas
            ax.set_xlabel('Motoristas')
            ax.set_ylabel('Tempo Esperando pelo Pedido')
            ax.set_title('Estatísticas do Tempo Esperando pelo Pedido por Motorista')
            ax.legend()
            ax.grid(True)

        else:
            drivers = self.environment.state.drivers

            # Usa os valores pontuais da simulação atual
            ids = [driver.driver_id for driver in drivers]
            times_waiting_for_order: List[int] = [int(driver.time_waiting_for_order) for driver in drivers]
            title = 'Time waiting for order per Driver'
            print("\nTempo que cada motorista passou esperando pelo pedido no Estabelecimento:")

            for driver_id, time in zip(ids, times_waiting_for_order):
                print(f"Motorista {driver_id}: {time:.2f} minutos esperando pelo pedido")

            ax.barh(ids, times_waiting_for_order, color='blue')
            ax.set_xlabel('Time waiting for order')
            ax.set_ylabel('Drivers')
            ax.set_title(title)

            ax.set_yticks(ids)
            ax.set_yticklabels([str(int(driver_id)) for driver_id in ids])

            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
