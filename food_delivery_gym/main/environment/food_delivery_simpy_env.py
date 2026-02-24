from collections import Counter, defaultdict
from statistics import mode
from typing import List, Optional, Union

import numpy as np
from simpy import Environment, Event
from simpy.core import SimTime

from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.delivery_env_state import DeliveryEnvState
from food_delivery_gym.main.map.map import Map
from food_delivery_gym.main.order.delivery_rejection import DeliveryRejection
from food_delivery_gym.main.view.food_delivery_view import FoodDeliveryView


class FoodDeliverySimpyEnv(Environment):

    # Armazena listas de valores para cálculos estatísticos
    establishment_metrics = defaultdict(lambda: defaultdict(list))
    driver_metrics = defaultdict(lambda: defaultdict(list))

    def __init__(self, map: Map, generators, optimizer, view: FoodDeliveryView = None):
        super().__init__()
        self.map = map
        self.generators = generators
        self.optimizer = optimizer
        self.view = view
        self.env_mode = EnvMode.TRAINING
        self.last_time_step = 0
        self._state = DeliveryEnvState()
        self.init()

        self.core_events: List[Event] = []

    def set_env_mode(self, mode: EnvMode):
        self.env_mode = mode

    def add_core_event(self, event):
        self.core_events.append(event)
    
    def dequeue_core_event(self):
        if self.core_events:
            return self.core_events.pop(0)
        else:
            return None
    
    def clear_core_events(self):
        self.core_events.clear()

    @property
    def events(self):
        return self._state.events

    @property
    def state(self):
        return self._state

    def add_customers(self, customers):
        self._state.add_customers(customers)

    def add_establishments(self, establishments):
        self._state.add_establishments(establishments)

    def add_drivers(self, drivers):
        self._state.add_drivers(drivers)

    def available_drivers(self, route):
        return [driver for driver in self._state.drivers if driver.check_availability(route)]
    
    def get_drivers(self):
        return self._state.drivers

    def add_ready_order(self, order, event):
        self._state.orders_awaiting_delivery.append(order)

    def get_ready_orders(self):
        read_orders = []
        while len(self._state.orders_awaiting_delivery) > 0:
            read_orders = self._state.orders_awaiting_delivery
            self._state.orders_awaiting_delivery = []
        return read_orders

    def count_ready_orders(self):
        return len(self._state.orders_awaiting_delivery)

    def add_rejected_delivery(self, order, delivery_rejection: DeliveryRejection, event):
        order.add_delivery_rejection(delivery_rejection)
        self._state.orders_awaiting_delivery.append(order)

    def get_rejected_deliveries(self):
        rejected_orders = []
        while len(self._state.rejected_deliveries) > 0:
            rejected_orders = self._state.rejected_deliveries
            self._state.rejected_deliveries = []
        return rejected_orders

    def add_event(self, event):
        self._state.add_event(event)

    def init(self):
        for generator in self.generators:
            self.process(generator.generate(self))

        if self.optimizer:
            self.process(self.optimizer.generate(self))

    def log_events(self):
        self._state.log_events()

    def run(self, until: Optional[Union[SimTime, Event]] = None, render_mode=None):
        if render_mode == "human" and self.view is not None:
            if not isinstance(until, Event):
                until = until if isinstance(until, int) else float(until)
                while self.now < until and not self.view.quited:
                    self.view.render(self)
                    super().run(until=self.now + 1)
                if self.view.quited:
                    self.view.quit()
        else:
            super().run(until=until)

        if self.view is not None and self.view.quited:
            self.view.quit()

    def step(self, render_mode=None):
        super().step()
        if render_mode == "human" and self.view is not None:
            self.view.render(self)
            if self.view.quited:
                self.view.quit()
        
        if self.env_mode != EnvMode.TRAINING and self.last_time_step < self.now:
            self.update_statistics_variables()
            self.last_time_step = self.now

    def render(self):
        if self.view is not None and not self.view.quited:
            self.view.render(self)

    def close(self):
        if self.view is not None and not self.view.quited:
            self.view.quit()

    def print_enviroment_state(self, options = None):
        print(f'time_step = {self.now}')
        self._state.print_state(options)

    def update_statistics_variables(self):
        for establishment in self._state.establishments:
            establishment.update_statistics_variables()
        
        for driver in self._state.drivers:
            driver.update_statistics_variables()
    
    def update_spent_drivers(self):
        for driver in self._state.drivers:
            driver.update_spent_time()

    def register_statistic_data(self):
        for establishment in self._state.establishments:
            establishment.register_statistic_data()

        for driver in self._state.drivers:
            driver.register_statistic_data()

    def get_statistics_data(self):
        return FoodDeliverySimpyEnv.establishment_metrics, FoodDeliverySimpyEnv.driver_metrics

    def reset_statistics(self):
        for establishment in self._state.establishments:
            establishment.reset_statistics()

        for driver in self._state.drivers:
            driver.reset_statistics()


    def compute_statistics(self):
        statistics = {"establishments": {}, "drivers": {}}

        # Establishments
        for id, metrics in FoodDeliverySimpyEnv.establishment_metrics.items():
            statistics["establishments"][id] = {
                key: self.calculate_stats_generic(values)
                for key, values in metrics.items()
            }

        # Drivers
        for id, metrics in FoodDeliverySimpyEnv.driver_metrics.items():
            statistics["drivers"][id] = {
                key: self.calculate_stats_generic(values)
                for key, values in metrics.items()
            }

        return statistics
    
    def calculate_stats_generic(self, values):
        """
        Calcula estatísticas de forma genérica:
        - Lista de números → calcula diretamente
        - Lista de dicts → calcula por campo interno
        - Qualquer outra coisa → ignora e retorna {}
        """

        if not values:
            return {}

        # Caso 1: lista de números (int/float)
        if all(isinstance(v, (int, float, np.number)) for v in values):

            def compute_mode(vs):
                if len(vs) == 1:
                    return vs[0]
                counter = Counter(vs)
                return counter.most_common(1)[0][0]

            return {
                "mean": float(np.mean(values)),
                "std_dev": float(np.std(values)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "mode": compute_mode(values),
            }

        # Caso 2: lista de dicts → achatar por chave
        if all(isinstance(v, dict) for v in values):
            grouped = {}

            for item in values:
                for key, val in item.items():
                    if isinstance(val, (int, float, np.number)):
                        grouped.setdefault(key, []).append(val)

            if not grouped:
                return {}

            return {k: self.calculate_stats_generic(vs) for k, vs in grouped.items()}

        # Caso 3: tipos misturados → ignora
        return {}
    
    def save_metrics_to_file(self, filename="metrics_data.npz"):
        np.savez_compressed(
            filename,
            establishment_metrics=dict(FoodDeliverySimpyEnv.establishment_metrics),
            driver_metrics=dict(FoodDeliverySimpyEnv.driver_metrics)
        )