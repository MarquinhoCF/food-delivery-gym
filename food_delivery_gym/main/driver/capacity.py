from numbers import Number
from typing import Iterable

from food_delivery_gym.main.base.dimensions import Dimensions
from food_delivery_gym.main.order.order import Order


class Capacity:
    def __init__(self, dimensions: Dimensions) -> None:
        self.dimensions = dimensions

    def fits(self, dimensions: Dimensions) -> bool:
        return self.dimensions > dimensions

    @property
    def value(self) -> Number:
        return self.dimensions.value


def sum_orders_dimensions(orders: Iterable[Order]) -> Dimensions:
    """
    Soma as dimensões dos itens dos pedidos únicos.
    Itens sem dimensions são ignorados.
    """
    total = Dimensions(0, 0, 0, 0)
    seen_order_ids: set = set()
    for order in orders:
        if order.order_id in seen_order_ids:
            continue
        seen_order_ids.add(order.order_id)
        for item in order.items:
            if item.dimensions is not None:
                total += item.dimensions
    return total


def route_required_capacity(route) -> Dimensions:
    """Capacidade requerida por uma rota (pedidos únicos, não por segmento)."""
    return sum_orders_dimensions(segment.order for segment in route.route_segments)
