from collections import defaultdict

import numpy as np

from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.events.event_type import EventType
from food_delivery_gym.main.statistic.metric import Metric


class DistanceMetric(Metric):

    def __init__(self, environment: FoodDeliverySimpyEnv, table=False):
        super().__init__(environment)
        self.table = table

    def view(self, ax) -> None:
        events = list(filter(lambda env: env.event_type in [
            EventType.DRIVER_PICKING_UP_ORDER,
            EventType.DRIVER_DELIVERING_ORDER,
            EventType.DRIVER_PICKED_UP_ORDER,
            EventType.DRIVER_DELIVERED_ORDER
        ], self.environment.events))

        # events = list(filter(
        #     lambda env: (hasattr(env, "distance") and env.distance > 0) or not hasattr(env, "distance"),
        #     events
        # ))

        # Dicionário para armazenar os tempos de cada evento por order_id
        driver_events = defaultdict(dict)

        # Preenchendo o dicionário com os tempos dos eventos
        for event in events:
            driver_id = event.driver_id
            order_id = event.order.order_id
            event_type = event.event_type
            distance = 0
            if hasattr(event, "distance"):
                distance = event.distance

            driver_events[(driver_id, order_id)][event_type] = distance

        # Calculando a diferença de tempo para cada order_id
        distances = {}
        for (driver_id, order_id), total_distances in driver_events.items():
            if EventType.DRIVER_PICKING_UP_ORDER in total_distances and EventType.DRIVER_PICKED_UP_ORDER in total_distances:
                if driver_id in distances:
                    distances[driver_id] += total_distances[EventType.DRIVER_PICKING_UP_ORDER]
                else:
                    distances[driver_id] = total_distances[EventType.DRIVER_PICKING_UP_ORDER]
            if EventType.DRIVER_DELIVERING_ORDER in total_distances and EventType.DRIVER_DELIVERED_ORDER in total_distances:
                if driver_id in distances:
                    distances[driver_id] += total_distances[EventType.DRIVER_DELIVERING_ORDER]
                else:
                    distances[driver_id] = total_distances[EventType.DRIVER_DELIVERING_ORDER]

        # # Mostrando os resultados
        # for driver_id, total_distance in distances.items():
        #     driver = next((d for d in self.environment.state.drivers if d.driver_id == driver_id))
        #     print(f"Driver ID: {driver_id} - Total Distance: {total_distance} {driver.total_distance} {total_distance > driver.total_distance}")

        distances = distances.values()

        min_distance = min(distances)
        p1 = np.percentile(list(distances), 1)
        p10 = np.percentile(list(distances), 10)
        p50 = np.percentile(list(distances), 50)
        p90 = np.percentile(list(distances), 90)
        p95 = np.percentile(list(distances), 95)
        p99 = np.percentile(list(distances), 99)
        max_distance = max(distances)

        print(f"Min distance: {min_distance}")
        print(f"01th percentile: {p1}")
        print(f"10th percentile: {p10}")
        print(f"50th percentile: {p50}")
        print(f"90th percentile: {p90}")
        print(f"95th percentile: {p95}")
        print(f"99th percentile: {p99}")
        print(f"Max distance: {max_distance}")

        labels = ['Min ', '01th', '10th', '50th', '90th', '95th', '99th', 'Max']
        values = [min_distance, p1, p10, p50, p90, p95, p99, max_distance]

        ax.set_title('Total distance traveled percentile')

        if self.table:
            ax.axis('tight')
            ax.axis('off')
            table_data = [[label, value] for label, value in zip(labels, values)]
            table = ax.table(cellText=table_data, colLabels=['Entities', 'Total'], cellLoc='center', loc='center')
            # table.auto_set_font_size(False)
            # table.set_fontsize(12)
            # table.scale(1.2, 1.2)
        else:
            ax.barh(labels, values, align='center')
            ax.set_xlabel('Percentile')
            ax.set_ylabel('Entities')
