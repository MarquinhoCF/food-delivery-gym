from food_delivery_gym.main.events.event import Event
from food_delivery_gym.main.events.event_type import EventType


class TimeForAgentAllocateDriver(Event):
    def __init__(self, order, customer_id, establishment_id, time):
        super().__init__(time, EventType.TIME_FOR_AGENT_ALLOCATE_DRIVER)
        self.order = order
        self.customer_id = customer_id
        self.establishment_id = establishment_id

    def __str__(self):
        return (f"It is time for the agent to select the driver "
                f"for order {self.order.order_id} "
                f"from customer {self.customer_id} "
                f"that was made at establishment {self.establishment_id} "
                f"at time {self.time}")