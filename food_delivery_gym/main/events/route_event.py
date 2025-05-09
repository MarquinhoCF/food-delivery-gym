from food_delivery_gym.main.events.event import Event


class RouteEvent(Event):
    def __init__(self, driver_id, route_id, distance, time, event_type):
        super().__init__(time, event_type)
        self.driver_id = driver_id
        self.route_id = route_id
        self.distance = distance
