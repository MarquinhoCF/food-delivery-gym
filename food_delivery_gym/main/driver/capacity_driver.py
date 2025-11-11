from typing import Optional

from food_delivery_gym.main.base.dimensions import Dimensions
from food_delivery_gym.main.base.types import Coordinate, Number
from food_delivery_gym.main.driver.capacity import Capacity
from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.driver.driver_status import DriverStatus
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.events.driver_accepted_route import DriverAcceptedRoute
from food_delivery_gym.main.route.route import Route


class CapacityDriver(Driver):
    
    def __init__(
        self,
        id: Number,
        environment: FoodDeliverySimpyEnv,
        coordinate: Coordinate,
        available: bool,
        color: Optional[tuple[int, int, int]] = (255, 0, 0),
        capacity: Optional[Capacity] = None,
        status: Optional[DriverStatus] = DriverStatus.AVAILABLE,
        movement_rate: Optional[Number] = 5,
        reward_objective: Optional[Number] = 1,
    ):
        # Define capacidade padrão se não fornecida
        if capacity is None:
            capacity = Capacity(Dimensions(100, 100, 100, 100))
        
        super().__init__(
            id=id,
            environment=environment,
            coordinate=coordinate,
            available=available,
            color=color,
            status=status,
            movement_rate=movement_rate,
            reward_objective=reward_objective,
        )
    
    def receive_route_requests(self, route: Route) -> None:
        self.route_requests.append(route)

        order = route.get_current_order()

        order.driver_allocated(
            self.now,
            self.time_between_accept_and_start_picking_up(),
            self.environment.map.estimated_time(
                self.get_last_valid_coordinate(), 
                order.establishment.coordinate, 
                self.movement_rate
            ),
            self.time_between_picked_up_and_start_delivery(),
            self.environment.map.estimated_time(
                order.establishment.coordinate, 
                order.customer.coordinate, 
                self.movement_rate
            ),
            self.estimate_time_to_costumer_receive_order(order)
        )
        # Não incrementa assigned_routes aqui - apenas quando aceita a rota
    
    def accept_route(self, route: Route) -> None:
        self.orders_list.append(route.get_current_order())

        # Incrementa rotas corretamente atribuídas quando o motorista aceita
        self.environment.state.increment_assigned_routes()

        if self.current_route is None:
            self.current_route = route
            self.publish_event(DriverAcceptedRoute(
                driver_id=self.driver_id,
                route_id=self.current_route.route_id,
                distance=self.current_route.distance,
                time=self.now
            ))
            self.accept_route_segments(self.current_route.route_segments)
            self.process(self.sequential_processor())
        else:
            self.accepted_route_extension(route)
    
    def accept_route_condition(self, route: Route) -> bool:
        return self.fits(route) and self.available
    
    def check_availability(self, route: Route) -> bool:
        return self.fits(route) and self.available
    
    def fits(self, route: Route) -> bool:
        return self.capacity.fits(route.required_capacity)