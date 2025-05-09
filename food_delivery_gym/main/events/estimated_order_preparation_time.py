from food_delivery_gym.main.events.event_type import EventType
from food_delivery_gym.main.events.order_event import OrderEvent


class EstimatedOrderPreparationTime(OrderEvent):
    def __init__(self, order, customer_id, establishment_id, estimated_time, time):
        super().__init__(order, customer_id, establishment_id, time, EventType.ESTIMATED_ORDER_PREPARATION_TIME)
        self.estimated_time = estimated_time

    def __str__(self):
        return (f"Order {self.order.order_id} placed by "
                f"customer {self.customer_id} for "
                f"establishment {self.establishment_id} has an "
                f"estimated preparation time of {self.estimated_time} and is expected to be "
                f"ready in time {self.estimated_time + self.time} this event was generated in "
                f"time {self.time}")
