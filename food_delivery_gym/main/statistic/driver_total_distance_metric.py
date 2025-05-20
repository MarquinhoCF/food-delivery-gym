from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.statistic.metric import Metric


class DriverTotalDistanceMetric(Metric):
    def __init__(self, environment: FoodDeliverySimpyEnv, drivers_statistics=None):
        super().__init__(environment)
        self.drivers_statistics = drivers_statistics

    def view(self, ax) -> None:
        if self.drivers_statistics is not None:
            est_ids = list(self.drivers_statistics.keys())
            means = [self.drivers_statistics[e]['total_distance']['mean'] for e in est_ids]
            medians = [self.drivers_statistics[e]['total_distance']['median'] for e in est_ids]
            modes = [self.drivers_statistics[e]['total_distance']['mode'] for e in est_ids]
            std_devs = [self.drivers_statistics[e]['total_distance']['std_dev'] for e in est_ids]

            # Criando o gráfico
            ax.errorbar(est_ids, means, yerr=std_devs, fmt='o', label='Média', capsize=5)
            ax.plot(est_ids, medians, marker='s', linestyle='', label='Mediana')
            ax.plot(est_ids, modes, marker='^', linestyle='', label='Moda')

            # Adicionando títulos e legendas
            ax.set_xlabel('Motoristas')
            ax.set_ylabel('Distância Total Percorrida')
            ax.set_title('Estatísticas da Distância Total Percorrida por Motorista')
            ax.legend()
            ax.grid(True)

        else:
            drivers = self.environment.state.drivers

            # Usa os valores pontuais da simulação atual
            ids = [driver.driver_id for driver in drivers]
            total_distances = [driver.total_distance for driver in drivers]
            title = 'Total Distance Traveled per Driver'
            print("\nDistância Total Percorrida por Motorista:")

            for driver_id, distance in zip(ids, total_distances):
                print(f"Motorista {driver_id}: {distance:.2f} percorridos")
            
            ax.barh(ids, total_distances, color='red')
            ax.set_xlabel('Total Distance Traveled')
            ax.set_ylabel('Drivers')
            ax.set_title(title)
            
            ax.set_yticks(ids)
            ax.set_yticklabels([str(int(driver_id)) for driver_id in ids])