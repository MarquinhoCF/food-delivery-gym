from collections import defaultdict
from statistics import mode
from typing import List, Optional, Union

import numpy as np
from simpy import Environment, Event
from simpy.core import SimTime

from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.delivery_env_state import DeliveryEnvState
from food_delivery_gym.main.events.event_type import EventType
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
        self.last_time_step = 0
        self._state = DeliveryEnvState()
        self.init()

        self.core_events: List[Event] = []

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

    def step(self, mode: EnvMode, render_mode=None):
        super().step()
        if render_mode == "human" and self.view is not None:
            self.view.render(self)
            if self.view.quited:
                self.view.quit()
        
        if mode == EnvMode.EVALUATING and self.last_time_step < self.now:
            self.update_statistcs_variables()
            #self.print_enviroment_state()
            self.last_time_step = self.now

    def render(self):
        if self.view is not None and not self.view.quited:
            self.view.render(self)

    def close(self):
        if self.view is not None and not self.view.quited:
            self.view.quit()

    def print_enviroment_state(self, options = None):
        if options is None:
            options = {
                "customers": False,
                "establishments": True,
                "drivers": True,
                "orders": False,
                "events": False,
                "orders_delivered": True
            }
        print(f'time_step = {self.now}')
        self._state.print_state(options)

    def update_statistcs_variables(self):
        for establishment in self._state.establishments:
            establishment.update_statistcs_variables()
        
        for driver in self._state.drivers:
            driver.update_statistcs_variables()
    
    def update_spent_drivers(self):
        for driver in self._state.drivers:
            driver.update_spent_time()

    def register_statistic_data(self):
        for establishment in self._state.establishments:
            id = establishment.establishment_id
            FoodDeliverySimpyEnv.establishment_metrics[id]["orders_fulfilled"].append(establishment.orders_fulfilled)
            FoodDeliverySimpyEnv.establishment_metrics[id]["idle_time"].append(establishment.idle_time)
            FoodDeliverySimpyEnv.establishment_metrics[id]["active_time"].append(establishment.active_time)
            FoodDeliverySimpyEnv.establishment_metrics[id]["max_orders_in_queue"].append(establishment.max_orders_in_queue)

        for driver in self._state.drivers:
            id = driver.driver_id
            FoodDeliverySimpyEnv.driver_metrics[id]["orders_delivered"].append(driver.orders_delivered)
            FoodDeliverySimpyEnv.driver_metrics[id]["idle_time"].append(driver.idle_time)
            FoodDeliverySimpyEnv.driver_metrics[id]["time_waiting_for_order"].append(driver.time_waiting_for_order)
            FoodDeliverySimpyEnv.driver_metrics[id]["total_distance"].append(driver.total_distance)

    def get_statistics_data(self):
        return FoodDeliverySimpyEnv.establishment_metrics, FoodDeliverySimpyEnv.driver_metrics

    def reset_statistics(self):
        for establishment in self._state.establishments:
            id = establishment.establishment_id
            FoodDeliverySimpyEnv.establishment_metrics[id]["orders_fulfilled"].clear()
            FoodDeliverySimpyEnv.establishment_metrics[id]["idle_time"].clear()
            FoodDeliverySimpyEnv.establishment_metrics[id]["active_time"].clear()
            FoodDeliverySimpyEnv.establishment_metrics[id]["max_orders_in_queue"].clear()

        for driver in self._state.drivers:
            id = driver.driver_id
            FoodDeliverySimpyEnv.driver_metrics[id]["orders_delivered"].clear()
            FoodDeliverySimpyEnv.driver_metrics[id]["idle_time"].clear()
            FoodDeliverySimpyEnv.driver_metrics[id]["time_waiting_for_order"].clear()
            FoodDeliverySimpyEnv.driver_metrics[id]["total_distance"].clear()


    def compute_statistics(self):
        """Calcula média, desvio padrão, mediana e moda para os dados coletados."""

        def calculate_stats(values):
            return {
                "mean": np.mean(values) if values else 0,
                "std_dev": np.std(values) if len(values) > 1 else 0,
                "median": np.median(values) if values else 0,
                "mode": mode(values) if values else 0
            }

        statistics = {
            "establishments": {},
            "drivers": {}
        }

        for id, metrics in FoodDeliverySimpyEnv.establishment_metrics.items():
            statistics["establishments"][id] = {key: calculate_stats(values) for key, values in metrics.items()}

        for id, metrics in FoodDeliverySimpyEnv.driver_metrics.items():
            statistics["drivers"][id] = {key: calculate_stats(values) for key, values in metrics.items()}

        return statistics
    
    def save_metrics_to_file(self, filename="metrics_data.npz"):
        np.savez_compressed(
            filename,
            establishment_metrics=dict(FoodDeliverySimpyEnv.establishment_metrics),
            driver_metrics=dict(FoodDeliverySimpyEnv.driver_metrics)
        )