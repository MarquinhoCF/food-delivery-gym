from collections import defaultdict

from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.statistic.metric import Metric


class OrderStatusMetric(Metric):

    def __init__(self, environment: FoodDeliverySimpyEnv):
        super().__init__(environment)

    def view(self, ax) -> None:
        print("ORDERS")
        order_status_counts = defaultdict(int)
        for order in self.environment.state.orders:
            order_status_counts[order.status.name.lower()] += 1

        for status, count in order_status_counts.items():
            print(f"{status} {count}")

        # Extrair os dados para listas separadas
        labels = list(order_status_counts.keys())
        sizes = list(order_status_counts.values())

        # Calcular o total dos valores
        total = sum(sizes)

        # Filtrar valores maiores que 5% e agrupar o restante como "Outros"
        filtered_labels = []
        filtered_sizes = []
        others_size = 0

        for label, size in zip(labels, sizes):
            if size / total > 0.05:
                filtered_labels.append(label)
                filtered_sizes.append(size)
            else:
                others_size += size

        # Adicionar a categoria "Outros" se houver valores agrupados
        if others_size > 0:
            filtered_labels.append('others')
            filtered_sizes.append(others_size)

        # Criar o gráfico de pizza
        ax.pie(filtered_sizes, labels=filtered_labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85)

        # Garantir que o gráfico seja desenhado como um círculo
        ax.axis('equal')

        # Adicionar um título (opcional)
        ax.set_title('Orders state')

        # Mostrar o gráfico
        # plt.show()
