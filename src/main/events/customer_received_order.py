from src.main.events.event_type import EventType
from src.main.events.order_event import OrderEvent


class CustomerReceivedOrder(OrderEvent):
    def __init__(self, order, customer_id, establishment_id, driver_id, time):
        super().__init__(order, customer_id, establishment_id, time, EventType.CUSTOMER_RECEIVED_ORDER)
        self.driver_id = driver_id

    def __str__(self):
        return (f"Customer {self.customer_id} picked up the "
                f"order {self.order.order_id} with "
                f"driver {self.driver_id} from "
                f"establishment {self.establishment_id} in "
                f"time {self.time}")
