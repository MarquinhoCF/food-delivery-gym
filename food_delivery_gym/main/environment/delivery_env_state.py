from typing import List

from food_delivery_gym.main.environment.state_log import format_environment_state
from food_delivery_gym.main.order.order import Order


class DeliveryEnvState:
    def __init__(self):
        self._customers = []
        self._establishments = []
        self._drivers = []
        self._orders: List[Order] = []

        # Orders ready for picking up
        self.orders_awaiting_delivery: List[Order] = []
        self.orders_delivered = 0

        self._last_checked_orders_delivered = 0
        self.recently_delivered_orders: List[Order] = []

        self.successfully_assigned_routes = 0

        self.events = []

    @property
    def customers(self) -> List:
        return self._customers

    @property
    def establishments(self) -> List:
        return self._establishments

    @property
    def drivers(self) -> List:
        return self._drivers

    @property
    def orders(self) -> List[Order]:
        return self._orders

    def add_customers(self, customer: List):
        self._customers += customer

    def add_establishments(self, establishments: List) -> None:
        self._establishments += establishments

    def add_drivers(self, drivers: List) -> None:
        self._drivers += drivers

    def add_orders(self, orders: List) -> None:
        self._orders += orders

    def get_length_orders(self) -> int:
        return len(self._orders)

    def increment_assigned_routes(self) -> None:
        self.successfully_assigned_routes += 1

    def add_order_delivered(self, order: Order) -> None:
        self.recently_delivered_orders.append(order)
        self.orders_delivered += 1
    
    def get_orders_delivered(self) -> int:
        return self.orders_delivered
    
    def get_num_orders_delivered_since_last_check(self) -> int:
        delivered_since_last_check = self.orders_delivered - self._last_checked_orders_delivered
        self._last_checked_orders_delivered = self.orders_delivered
        return delivered_since_last_check
    
    def get_and_clear_recently_delivered_orders(self) -> List[Order]:
        orders = self.recently_delivered_orders
        self.recently_delivered_orders = []
        return orders

    def add_event(self, event) -> None:
        self.events.append(event)

    def log_events(self) -> None:
        for event in self.events:
            print(event)

    def print_state(self, options=None, *, time_step=None, current_order=None):
        print(format_environment_state(
            self,
            options,
            time_step=time_step,
            current_order=current_order,
        ))
