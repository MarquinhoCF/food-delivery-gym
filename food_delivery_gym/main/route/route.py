from typing import List

from food_delivery_gym.main.base.dimensions import Dimensions
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.route.route_segment import RouteSegment


class Route:
    _id_counter = 0  # Variável estática para o ID

    def __init__(self, environment: FoodDeliverySimpyEnv, route_segments: List[RouteSegment]):
        self.route_id = Route._generate_id()
        self.environment = environment
        self.route_segments = route_segments
        self.required_capacity = self.calculate_required_capacity()
        self.distance = self.calculate_total_distance()

    @classmethod
    def _generate_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    def calculate_required_capacity(self):
        dimensions = Dimensions(0, 0, 0, 0)
        for route_segment in self.route_segments:
            dimensions += route_segment.required_capacity
        return dimensions

    def has_next(self):
        return len(self.route_segments) > 0

    def next(self):
        self.required_capacity = self.calculate_required_capacity()
        self.distance = self.calculate_total_distance()
        return self.route_segments.pop(0)
    
    def get_current_order(self):
        if self.has_next():
            return self.route_segments[0].order
        return None
    
    def insert_segment_before_first_segment_by_id(self, route_segment_id: int):
        segment = self.find_route_segment_by_id(route_segment_id)
        self.route_segments.remove(segment)
        self.route_segments.insert(0, segment)

    def find_route_segment_by_id(self, route_segment_id: int) -> int:
        for idx, segment in enumerate(self.route_segments):
            if segment.route_segment_id == route_segment_id:
                return segment
        raise ValueError("Segmento de rota não encontrado.")

    def calculate_total_distance(self):
        coordinates = [segment.coordinate for segment in self.route_segments]
        return self.environment.map.acc_distance(coordinates)

    def extend_route(self, other_route):
        self.route_segments += other_route.route_segments
        self.required_capacity = self.calculate_required_capacity()
        self.distance = self.calculate_total_distance()

    def size(self):
        return len(self.route_segments)
