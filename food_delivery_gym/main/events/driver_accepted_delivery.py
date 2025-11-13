from food_delivery_gym.main.events.event_type import EventType
from food_delivery_gym.main.events.order_event import OrderEvent


class DriverAcceptedDelivery(OrderEvent):
    def __init__(self, order, customer_id, establishment_id, driver_id, time):
        super().__init__(order, customer_id, establishment_id, time, EventType.DRIVER_ACCEPTED_DELIVERY)
        self.driver_id = driver_id

    def __str__(self):
        return (f"Driver {self.driver_id} accepted to deliver "
                f"order {self.order.order_id} from "
                f"establishment {self.establishment_id} and from "
                f"customer {self.customer_id} in "
                f"time {self.time}")
