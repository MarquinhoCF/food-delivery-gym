from typing import List

from matplotlib.ticker import FuncFormatter
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.statistic.metric import Metric


class EstablishmentActiveTimeMetric(Metric):
    def __init__(self, environment: FoodDeliverySimpyEnv, establishments_statistics=None):
        super().__init__(environment)
        self.establishments_statistics = establishments_statistics

    def view(self, ax) -> None:
        if self.establishments_statistics is not None:
            est_ids = list(self.establishments_statistics.keys())
            means = [self.establishments_statistics[e]['active_time']['mean'] for e in est_ids]
            medians = [self.establishments_statistics[e]['active_time']['median'] for e in est_ids]
            modes = [self.establishments_statistics[e]['active_time']['mode'] for e in est_ids]
            std_devs = [self.establishments_statistics[e]['active_time']['std_dev'] for e in est_ids]

            # Criando o gráfico
            ax.errorbar(est_ids, means, yerr=std_devs, fmt='o', label='Média', capsize=5)
            ax.plot(est_ids, medians, marker='s', linestyle='', label='Mediana')
            ax.plot(est_ids, modes, marker='^', linestyle='', label='Moda')

            # Adicionando títulos e legendas
            ax.set_xlabel('Estabelecimento')
            ax.set_ylabel('Tempo Ativo')
            ax.set_title('Estatísticas do Tempo Ativo por Estabelecimento')
            ax.legend()
            ax.grid(True)

        else:
            establishments = self.environment.state.establishments

            # Usa os valores pontuais da simulação atual
            ids = [establishment.establishment_id for establishment in establishments]
            active_times: List[int] = [int(establishment.active_time) for establishment in establishments]
            title = 'Active Time per Establishment'
            # print("\nTempo Ativo por Estabelecimento:")

            # # TODO: Logs
            # for est_id, active_time in zip(ids, active_times):
            #     print(f"Estabelecimento {est_id}: {active_time:.2f} minutos ativo")

            ax.barh(ids, active_times, color='purple')
            ax.set_xlabel('Active Time')
            ax.set_ylabel('Establishments')
            ax.set_title(title)

            ax.set_yticks(ids)
            ax.set_yticklabels([str(int(driver_id)) for driver_id in ids])

            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

