from collections import defaultdict

from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.statistic.metric import Metric


class DriverStatusMetric(Metric):

    def __init__(self, environment: FoodDeliverySimpyEnv):
        super().__init__(environment)

    def view(self, ax) -> None:
        print("DRIVERS")
        drivers_status_counts = defaultdict(int)
        for driver in self.environment.state.drivers:
            drivers_status_counts[driver.status.name.lower()] += 1

        for status, count in drivers_status_counts.items():
            print(f"{status} {count}")

        # Extrair os dados para listas separadas
        labels = list(drivers_status_counts.keys())
        sizes = list(drivers_status_counts.values())

        # Criar o gráfico de pizza
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85)

        # Garantir que o gráfico seja desenhado como um círculo
        ax.axis('equal')

        # Adicionar um título (opcional)
        ax.set_title('Drivers state')

        # Mostrar o gráfico
        # plt.show()
