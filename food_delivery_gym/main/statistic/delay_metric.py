from collections import defaultdict

import numpy as np

from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.events.event_type import EventType
from food_delivery_gym.main.statistic.metric import Metric


class DelayMetric(Metric):
    def __init__(self, environment: FoodDeliverySimpyEnv, table=False):
        super().__init__(environment)
        self.table = table

    def view(self, ax) -> None:
        events = list(filter(lambda env: env.event_type in [
            EventType.DRIVER_ACCEPTED_DELIVERY,
            EventType.DRIVER_DELIVERED_ORDER
        ], self.environment.events))

        # Dicionário para armazenar os tempos de cada evento por order_id
        order_events = defaultdict(dict)

        # Preenchendo o dicionário com os tempos dos eventos
        for event in events:
            order_id = event.order.order_id
            event_type = event.event_type
            time = event.time

            order_events[order_id][event_type] = time

        # Calculando a diferença de tempo para cada order_id
        time_differences = {}
        for order_id, times in order_events.items():
            if EventType.DRIVER_ACCEPTED_DELIVERY in times and EventType.DRIVER_DELIVERED_ORDER in times:
                time_differences[order_id] = times[EventType.DRIVER_DELIVERED_ORDER] - times[
                    EventType.DRIVER_ACCEPTED_DELIVERY]

        # # Mostrando os resultados
        # for order_id, time_difference in time_differences.items():
        #     print(f"Order ID: {order_id} - Time Difference: {time_difference}")

        times = time_differences.values()

        min_time = min(times)
        p1 = np.percentile(list(times), 1)
        p10 = np.percentile(list(times), 10)
        p50 = np.percentile(list(times), 50)
        p90 = np.percentile(list(times), 90)
        p95 = np.percentile(list(times), 95)
        p99 = np.percentile(list(times), 99)
        max_time = max(times)

        print(f"Min time: {min_time}")
        print(f"01th percentile: {p1}")
        print(f"10th percentile: {p10}")
        print(f"50th percentile: {p50}")
        print(f"90th percentile: {p90}")
        print(f"95th percentile: {p95}")
        print(f"99th percentile: {p99}")
        print(f"Max time: {max_time}")

        labels = ['Min', '01th', '10th', '50th', '90th', '95th', '99th', 'Max']
        values = [min_time, p1, p10, p50, p90, p95, p99, max_time]

        ax.set_title('Delay percentile')

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
