from food_delivery_gym.main.events.event_type import EventType
from food_delivery_gym.main.events.order_event import OrderEvent


class OptmizerRejectedDelivery(OrderEvent):
    def __init__(self, order, customer_id, establishment_id, time):
        super().__init__(order, customer_id, establishment_id, time, EventType.OPTMIZER_REJECTED_DELIVERY)

    def __str__(self):
        return (f"Optmizer did not select a driver for "
                f"order {self.order.order_id} from "
                f"establishment {self.establishment_id} and from "
                f"customer {self.customer_id} in "
                f"time {self.time}")
