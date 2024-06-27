from src.main.events.event_type import EventType
from src.main.events.order_event import OrderEvent


class DriverPickedUpOrder(OrderEvent):
    def __init__(self, order_id, customer_id, establishment_id, driver_id, time):
        super().__init__(order_id, customer_id, establishment_id, time, EventType.DRIVER_PICKED_UP_ORDER)
        self.driver_id = driver_id

    def __str__(self):
        return (f"Driver {self.driver_id} picked up "
                f"order {self.order_id} from "
                f"establishment {self.establishment_id} and "
                f"customer {self.customer_id} in "
                f"time {self.time}")
