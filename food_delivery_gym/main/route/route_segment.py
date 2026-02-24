from food_delivery_gym.main.order.order import Order
from food_delivery_gym.main.route.route_segment_type import RouteSegmentType


class RouteSegment:
    _id_counter = 0  # Variável estática para o ID

    def __init__(self, route_segment_type: RouteSegmentType, order: Order):
        self.route_segment_id = self._generate_id()  # Gera um ID único para cada instância
        self.route_segment_type = route_segment_type
        self.order = order
        self.coordinate = self.init_coordinates()
        self.required_capacity = self.order.required_capacity
    
    @classmethod
    def _generate_id(cls):
        RouteSegment._id_counter += 1
        return RouteSegment._id_counter

    def init_coordinates(self):
        if self.is_pickup():
            return self.order.establishment.coordinate
        return self.order.customer.coordinate

    def is_pickup(self) -> bool:
        return self.route_segment_type == RouteSegmentType.PICKUP

    def is_delivery(self) -> bool:
        return self.route_segment_type == RouteSegmentType.DELIVERY
