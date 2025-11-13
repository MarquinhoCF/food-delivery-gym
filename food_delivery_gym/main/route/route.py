from typing import List

from food_delivery_gym.main.base.dimensions import Dimensions
from food_delivery_gym.main.base.types import Coordinate, Number
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.route.route_segment import RouteSegment


class Route:
    _id_counter = 0  # Variável estática para o ID

    def __init__(self, environment: FoodDeliverySimpyEnv, route_segments: List[RouteSegment]):
        self.route_id = Route._generate_id()
        self.environment = environment
        self.route_segments = route_segments
        self.required_capacity = self.calculate_required_capacity()

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

    def extend_route(self, other_route):
        self.route_segments += other_route.route_segments
        self.required_capacity = self.calculate_required_capacity()

    def size(self):
        return len(self.route_segments)
    
    def get_time_to_complete_route(self, current_coordinate: Coordinate, movement_rate: Number) -> Number:
        if not self.route_segments:
            return 0
        
        total_time = 0
        for segment in self.route_segments:
            total_time += self.environment.map.estimated_time(current_coordinate, segment.coordinate, movement_rate)

            if segment.is_pickup():
                total_time += segment.order.estimated_time_between_accept_and_start_picking_up
                # Tempo de viagem para coleta já considerado

            elif segment.is_delivery():
                total_time += segment.order.estimated_time_between_picked_up_and_start_delivery
                # Tempo de viagem para entrega já considerado
                total_time += segment.order.estimated_time_to_costumer_receive_order

            current_coordinate = segment.coordinate
                
        return total_time
    
    def get_distance_to_complete_route(self, current_coordinate: Coordinate) -> Number:
        coordinates = [segment.coordinate for segment in self.route_segments]
        coordinates.insert(0, current_coordinate)
        return self.environment.map.acc_distance(coordinates)
