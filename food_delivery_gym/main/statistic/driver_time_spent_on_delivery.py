from typing import List

from matplotlib.ticker import FuncFormatter
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.statistic.metric import Metric


class DriverTimeSpentOnDelivery(Metric):
    def __init__(self, environment: FoodDeliverySimpyEnv, drivers_statistics=None):
        super().__init__(environment)
        self.drivers_statistics = drivers_statistics

    def view(self, ax) -> None:
        if self.drivers_statistics is not None:
            est_ids = list(self.drivers_statistics.keys())
            means = [self.drivers_statistics[e]['time_spent_on_delivery']['mean'] for e in est_ids]
            medians = [self.drivers_statistics[e]['time_spent_on_delivery']['median'] for e in est_ids]
            modes = [self.drivers_statistics[e]['time_spent_on_delivery']['mode'] for e in est_ids]
            std_devs = [self.drivers_statistics[e]['time_spent_on_delivery']['std_dev'] for e in est_ids]

            # Criando o gráfico
            ax.errorbar(est_ids, means, yerr=std_devs, fmt='o', label='Média', capsize=5)
            ax.plot(est_ids, medians, marker='s', linestyle='', label='Mediana')
            ax.plot(est_ids, modes, marker='^', linestyle='', label='Moda')

            # Adicionando títulos e legendas
            ax.set_xlabel('Motoristas', fontsize=11, fontweight='bold')
            ax.set_ylabel('Tempo Gasto na Entrega', fontsize=11, fontweight='bold')
            ax.set_title('Estatísticas do Tempo Gasto na Entrega por Motorista', fontsize=12, fontweight='bold', pad=15)
            ax.legend()
            ax.grid(True)

        else:
            drivers = self.environment.state.drivers

            # Usa os valores pontuais da simulação atual
            ids = [driver.driver_id for driver in drivers]

            _, drivers_metrics = self.environment.get_statistics_data()

            times_spent_on_delivery: List[int] = [sum(drivers_metrics[driver.driver_id]["time_spent_on_delivery"]) for driver in drivers]
            title = 'Time spent on delivery per Driver'
            # print("\nTempo que cada motorista gastou entregando os pedidos:")

            # # TODO: Logs
            # for driver_id, time in zip(ids, times_spent_on_delivery):
            #     print(f"Motorista {driver_id}: {time:.2f} minutos totais gastos entregando pedidos")

            ax.barh(ids, times_spent_on_delivery, color='blue')
            ax.set_xlabel('Time spent on delivery', fontsize=11, fontweight='bold')
            ax.set_ylabel('Drivers', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold', pad=15)

            ax.set_yticks(ids)
            ax.set_yticklabels([str(int(driver_id)) for driver_id in ids])

            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
