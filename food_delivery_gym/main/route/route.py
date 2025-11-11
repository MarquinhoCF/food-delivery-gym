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

    def swap_route_segments_by_id(self, route_segment_a: int, route_segment_b: int):
        index_a, index_b = self._find_segments_by_ids(route_segment_a, route_segment_b)
        
        self.route_segments[index_a], self.route_segments[index_b] = (
            self.route_segments[index_b],
            self.route_segments[index_a],
        )

        self.required_capacity = self.calculate_required_capacity()
        self.distance = self.calculate_total_distance()

    def _find_segments_by_ids(self, route_segment_a, route_segment_b):
        route_segment_a_idx = route_segment_b_idx = None

        for idx, segment in enumerate(self.route_segments):
            seg_id = segment.route_segment_id
            if seg_id == route_segment_a:
                route_segment_a_idx = idx
                if route_segment_b_idx is not None:
                    break
            elif seg_id == route_segment_b:
                route_segment_b_idx = idx
                if route_segment_a_idx is not None:
                    break

        if route_segment_a_idx is None or route_segment_b_idx is None:
            raise ValueError("Segmentos de rota não encontrados.")

        return route_segment_a_idx, route_segment_b_idx

    def calculate_total_distance(self):
        coordinates = [segment.coordinate for segment in self.route_segments]
        return self.environment.map.acc_distance(coordinates)

    def extend_route(self, other_route):
        self.route_segments += other_route.route_segments
        self.required_capacity = self.calculate_required_capacity()
        self.distance = self.calculate_total_distance()

    def size(self):
        return len(self.route_segments)
