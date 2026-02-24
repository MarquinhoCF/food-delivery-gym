from food_delivery_gym.main.events.event_type import EventType
from food_delivery_gym.main.events.route_event import RouteEvent


class DriverAcceptedRouteExtension(RouteEvent):
    def __init__(self, driver_id, route_id, new_route_id, time):
        super().__init__(driver_id, route_id, time, EventType.DRIVER_ACCEPTED_EXTENSION_ROUTE)
        self.new_route_id = new_route_id

    def __str__(self):
        return (f"Driver {self.driver_id} accepted "
                f"an extension of route {self.route_id} with {self.new_route_id} in"
                f"time {self.time}")
