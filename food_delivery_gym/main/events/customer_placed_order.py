from food_delivery_gym.main.events.event_type import EventType
from food_delivery_gym.main.events.order_event import OrderEvent


class CustomerPlacedOrder(OrderEvent):
    def __init__(self, order, customer_id, establishment_id, time):
        super().__init__(order, customer_id, establishment_id, time, EventType.CUSTOMER_PLACED_ORDER)

    def __str__(self):
        return (f"Customer {self.customer_id} placed an "
                f"order {self.order.order_id} to "
                f"establishment {self.establishment_id} in "
                f"time {self.time}")
