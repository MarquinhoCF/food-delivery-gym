from food_delivery_gym.main.order.order import Order
from food_delivery_gym.main.route.route_segment import RouteSegment
from food_delivery_gym.main.route.route_segment_type import RouteSegmentType


class DeliveryRouteSegment(RouteSegment):
    def __init__(self, order: Order):
        super().__init__(RouteSegmentType.DELIVERY, order)
        order.set_delivery_segment(self.route_segment_id)
