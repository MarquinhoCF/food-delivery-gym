from food_delivery_gym.main.events.event_type import EventType
from food_delivery_gym.main.events.route_event import RouteEvent


class DriverRejectedRoute(RouteEvent):
    def __init__(self, driver_id, route_id, time):
        super().__init__(driver_id, route_id, time, EventType.DRIVER_REJECTED_ROUTE)

    def __str__(self):
        return (f"Driver {self.driver_id} rejected the "
                f"route {self.route_id} in "
                f"time {self.time}")
