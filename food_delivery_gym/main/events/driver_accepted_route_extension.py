from food_delivery_gym.main.events.event_type import EventType
from food_delivery_gym.main.events.route_event import RouteEvent


class DriverAcceptedRouteExtension(RouteEvent):
    def __init__(self, driver_id, route_id, old_distance, distance, time):
        super().__init__(driver_id, route_id, distance, time, EventType.DRIVER_ACCEPTED_EXTENSION_ROUTE)
        self.old_distance = old_distance

    def __str__(self):
        return (f"Driver {self.driver_id} accepted the "
                f"route extension {self.route_id} and the total "
                f"distance increased from {self.old_distance} to {self.distance} in "
                f"time {self.time}")
