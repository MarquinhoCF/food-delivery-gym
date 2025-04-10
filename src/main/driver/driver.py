from typing import Optional, List

from simpy.events import ProcessGenerator

from src.main.actors.map_actor import MapActor
from src.main.base.dimensions import Dimensions
from src.main.base.types import Coordinate, Number
from src.main.driver.capacity import Capacity
from src.main.driver.driver_status import DriverStatus
from src.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from src.main.events.driver_accepted_delivery import DriverAcceptedDelivery
from src.main.events.driver_accepted_route import DriverAcceptedRoute
from src.main.events.driver_accepted_route_extension import DriverAcceptedRouteExtension
from src.main.events.driver_arrived_delivery_location import DriverArrivedDeliveryLocation
from src.main.events.driver_delivered_order import DriverDeliveredOrder
from src.main.events.driver_delivering_order import DriverDeliveringOrder
from src.main.events.driver_picked_up_order import DriverPickedUpOrder
from src.main.events.driver_picking_up_order import DriverPickingUpOrder
from src.main.events.driver_rejected_delivery import DriverRejectedDelivery
from src.main.events.driver_rejected_route import DriverRejectedRoute
from src.main.order.driver_delivery_rejection import DriverDeliveryRejection
from src.main.order.order import Order
from src.main.order.order_status import OrderStatus
from src.main.route.route import Route
from src.main.route.route_segment import RouteSegment


class Driver(MapActor):
    def __init__(
            self,
            id: Number,
            environment: FoodDeliverySimpyEnv,
            coordinate: Coordinate,
            available: bool,
            color: Optional[tuple[int, int, int]] = (255, 0, 0), # Cor vermelha
            desconsider_capacity: bool = False,
            capacity: Optional[Capacity] = Capacity(Dimensions(100, 100, 100, 100)),
            status: Optional[DriverStatus] = DriverStatus.AVAILABLE,
            movement_rate: Optional[Number] = 5
    ):
        
        self.driver_id = id

        super().__init__(environment, coordinate, available)

        self.color = color

        self.desconsider_capacity = desconsider_capacity

        if desconsider_capacity:
            self.capacity = Capacity(Dimensions(float('inf'), float('inf'), float('inf'), float('inf')))
        else:
            self.capacity = capacity
        
        self.status = status
        self.movement_rate = movement_rate

        self.current_route: Optional[Route] = None
        self.current_route_segment: Optional[RouteSegment] = None
        self.route_requests: List[Route] = []

        # Variáveis para controle do tempo gasto para entrega dos pedidos
        self.orders_list: List[Order] = []
        self.last_time_check: Number = 0
        self.sum_penalty_for_time_spent: Number = 0

        self.last_future_coordinate: Coordinate = coordinate

        self.total_distance: Number = 0
        self.last_total_distance: Number = 0

        # Variáveis para estatísticas
        self.orders_delivered: Number = 0
        self.idle_time: Number = 0
        self.time_waiting_for_order: Number = 0

        self.process(self.process_route_requests())
        self.process(self.move())

    def fits(self, route: Route) -> bool:
        return self.capacity.fits(route.required_capacity)

    def receive_route_requests(self, route: Route) -> None:
        self.route_requests.append(route)
        if self.desconsider_capacity:
            self.environment.state.increment_assigned_routes()

    def process_route_requests(self) -> ProcessGenerator:
        while True:
            if self.route_requests:
                route = self.route_requests.pop(0)
                self.process_route_request(route)
                yield self.timeout(self.time_to_accept_or_reject_route())
            else:
                yield self.timeout(1)

    def process_route_request(self, route: Route) -> None:
        accept = self.accept_route_condition(route)
        self.accept_route(route) if accept else self.reject_route(route)

    def accept_route(self, route: Route) -> None:
        self.orders_list.append(route.order)

        if not self.desconsider_capacity:
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
            self.last_future_coordinate = self.current_route.order.customer.coordinate
            self.process(self.sequential_processor())
        else:
            self.accepted_route_extension(route)

    def accept_route_segments(self, route_segments: List[RouteSegment]) -> None:
        for route_segment in route_segments:
            self.accept_route_segment(route_segment)

    def accept_route_segment(self, route_segment: RouteSegment) -> None:
        self.publish_event(DriverAcceptedDelivery(
            driver_id=self.driver_id,
            order=route_segment.order,
            customer_id=route_segment.order.customer.customer_id,
            establishment_id=route_segment.order.establishment.establishment_id,
            distance=self.current_route.distance,
            time=self.now
        ))
        if (route_segment.order.status == OrderStatus.PREPARING):
            route_segment.order.update_status(OrderStatus.PREPARING_AND_DRIVER_ACCEPTED)
        elif (route_segment.order.status == OrderStatus.READY):
            route_segment.order.update_status(OrderStatus.READY_AND_DRIVER_ACCEPTED)
        else:
            route_segment.order.update_status(OrderStatus.DRIVER_ACCEPTED)

    def accepted_route_extension(self, route: Route) -> None:
        old_distance = self.current_route.distance
        self.current_route.extend_route(route)
        self.publish_event(DriverAcceptedRouteExtension(
            driver_id=self.driver_id,
            route_id=self.current_route.route_id,
            old_distance=old_distance,
            distance=self.current_route.distance,
            time=self.now
        ))
        self.accept_route_segments(route.route_segments)
        self.last_future_coordinate = route.order.customer.coordinate

    def sequential_processor(self) -> ProcessGenerator:
        # Faz o motorista esperar o pedido estar pronto
        if self.current_route_segment is not None and not self.current_route_segment.order.isReady:
            self.status = DriverStatus.PICKING_UP_WAITING
            yield self.timeout(1)
            self.process(self.sequential_processor())
        # Processa a rota
        elif self.current_route.has_next():
            route_segment = self.current_route.next()
            self.current_route_segment = route_segment
            if route_segment.is_pickup():
                yield self.timeout(self.time_between_accept_and_start_picking_up())
                self.process(self.picking_up(route_segment.order))
            if route_segment.is_delivery():
                # TODO: Logs
                # print(f"Driver {self.driver_id} retirou o pedido no estabelecimento {self.current_route.order.establishment.establishment_id} no tempo {self.now}")
                yield self.timeout(self.time_between_picked_up_and_start_delivery())
                self.process(self.delivering(route_segment.order))
        else:
            self.current_route = None
            self.current_route_segment = None

    def reject_route(self, route: Route) -> None:
        self.publish_event(DriverRejectedRoute(
            driver_id=self.driver_id,
            route_id=self.current_route.route_id,
            distance=self.current_route.distance,
            time=self.now
        ))
        self.reject_route_segments(route.route_segments)

    def reject_route_segments(self, route_segments: List[RouteSegment]) -> None:
        for route_segment in route_segments:
            self.reject_route_segment(route_segment)

    def reject_route_segment(self, route_segment: RouteSegment) -> None:
        event = DriverRejectedDelivery(
            driver_id=self.driver_id,
            order=route_segment.order,
            customer_id=route_segment.order.customer.customer_id,
            establishment_id=route_segment.order.establishment.establishment_id,
            time=self.now
        )
        self.publish_event(event)

        if (route_segment.order.status == OrderStatus.PREPARING):
            route_segment.order.update_status(OrderStatus.PREPARING_AND_DRIVER_REJECTED)
        elif (route_segment.order.status == OrderStatus.READY):
            route_segment.order.update_status(OrderStatus.READY_AND_DRIVER_REJECTED)
        else:
            route_segment.order.update_status(OrderStatus.DRIVER_REJECTED)
            
        rejection = DriverDeliveryRejection(self, self.now)
        self.environment.add_rejected_delivery(route_segment.order, rejection, event)

    def picking_up(self, order: Order) -> ProcessGenerator:
        self.status = DriverStatus.PICKING_UP

        if order.status == OrderStatus.PREPARING:
            order.update_status(OrderStatus.PREPARING_AND_PICKING_UP)
        elif order.status == OrderStatus.READY:
            order.update_status(OrderStatus.READY_AND_PICKING_UP)
        else:
            order.update_status(OrderStatus.PICKING_UP)

        self.publish_event(DriverPickingUpOrder(
            order=order,
            customer_id=order.customer.customer_id,
            establishment_id=order.establishment.establishment_id,
            driver_id=self.driver_id,
            distance=self.environment.map.distance(self.coordinate, order.establishment.coordinate),
            time=self.now
        ))
        yield self.timeout(self.time_to_picking_up_order(order))
        self.picked_up(order)

    def picked_up(self, order: Order) -> None:
        self.publish_event(DriverPickedUpOrder(
            order=order,
            customer_id=order.customer.customer_id,
            establishment_id=order.establishment.establishment_id,
            driver_id=self.driver_id,
            time=self.now
        ))
        self.coordinate = order.establishment.coordinate
        self.process(self.sequential_processor())

    def delivering(self, order: Order) -> ProcessGenerator:
        self.status = DriverStatus.DELIVERING
        order.update_status(OrderStatus.DELIVERING)
        self.publish_event(DriverDeliveringOrder(
            order=order,
            customer_id=order.customer.customer_id,
            establishment_id=order.establishment.establishment_id,
            driver_id=self.driver_id,
            distance=self.environment.map.distance(self.coordinate, order.customer.coordinate),
            time=self.now
        ))
        yield self.timeout(self.time_to_deliver_order(order))
        self.process(self.wait_customer_pick_up_order(order))

    def wait_customer_pick_up_order(self, order: Order) -> ProcessGenerator:
        self.status = DriverStatus.DELIVERING_WAITING
        order.update_status(OrderStatus.DRIVER_ARRIVED_DELIVERY_LOCATION)
        self.publish_event(DriverArrivedDeliveryLocation(
            order=order,
            customer_id=order.customer.customer_id,
            establishment_id=order.establishment.establishment_id,
            driver_id=self.driver_id,
            time=self.now
        ))
        yield self.process(order.customer.receive_order(order, self))
        self.delivered(order)
        
    def delivered(self, order: Order) -> None:
        self.coordinate = order.customer.coordinate
        self.publish_event(DriverDeliveredOrder(
            order=order,
            customer_id=order.customer.customer_id,
            establishment_id=order.establishment.establishment_id,
            driver_id=self.driver_id,
            time=self.now
        ))
        self.status = DriverStatus.AVAILABLE
        order.update_status(OrderStatus.DELIVERED)
        self.process(self.sequential_processor())
        self.orders_delivered += 1
        self.environment.state.increment_orders_delivered()
        
        # Remove o pedido da lista de pedidos do motorista e soma a penalidade do tempo gasto para entrega
        for i, o in enumerate(self.orders_list):
            if o.order_id == order.order_id:
                # Se o pedido foi alocado depois do último tempo de verificação, soma o tempo desde a alocação até agora
                # Agora se o pedido foi alocado antes do último tempo de verificação, soma o tempo desde o último tempo de verificação até agora
                start_time = max(order.time_that_driver_was_allocated, self.last_time_check)
                penalty = self.now - start_time

                if not order.is_already_caught():
                    penalty *= 5
                
                self.sum_penalty_for_time_spent += penalty

                del self.orders_list[i]
                break

        # TODO: Logs
        # print(f"Driver {self.driver_id} entregou o pedido ao cliente no tempo {self.now}")

    def move(self) -> ProcessGenerator:
        while True:
            if self.current_route_segment:
                old_coordinate = self.coordinate
                self.coordinate = self.environment.map.move(
                    origin=self.coordinate,
                    destination=self.current_route_segment.coordinate,
                    rate=self.movement_rate
                )
                self.total_distance += self.environment.map.distance(old_coordinate, self.coordinate)
            yield self.timeout(1)

    def is_active(self) -> bool:
        return self.current_route is not None or self.current_route_segment is not None or len(self.route_requests) > 0

    def accept_route_condition(self, route: Route) -> bool:
        return self.fits(route) and self.available

    def check_availability(self, route: Route) -> bool:
        return self.fits(route) and self.available

    def estimate_time_to_driver_receive_order(self) -> int:
        return self.rng.randrange(1, 5)

    def time_to_accept_or_reject_route(self) -> int:
        return self.rng.randrange(3, 10)

    def time_between_accept_and_start_picking_up(self) -> int:
        return self.rng.randrange(0, 3)

    def time_to_picking_up_order(self, order: Order):
        return self.environment.map.estimated_time(self.coordinate, order.establishment.coordinate, self.movement_rate)

    def time_between_picked_up_and_start_delivery(self) -> int:
        return self.rng.randrange(0, 3)

    def time_to_deliver_order(self, order: Order) -> int:
        establishment_coordinates = order.establishment.coordinate
        customer_coordinates = order.customer.coordinate
        return self.environment.map.estimated_time(establishment_coordinates, customer_coordinates, self.movement_rate)
    
    def estimate_time_to_costumer_receive_order(self, order: Order) -> int:
        return order.customer.time_to_receive_order()

    def estimate_total_busy_time(self) -> Number:
        total_busy_time = 0
        valid_coordinate = self.coordinate  # Posição atual do motorista

        def add_travel_time(from_coord, to_coord):
            return self.environment.map.estimated_time(from_coord, to_coord, self.movement_rate)

        # Se o motorista já está em uma rota, inclui o tempo restante dessa rota
        if self.current_route:
            current_order = self.current_route_segment.order if self.current_route_segment else None
            
            if current_order:
                # Se o segmento atual é de coleta, considera o tempo para pegar o pedido
                if self.current_route_segment.is_pickup():
                    total_busy_time += (
                        current_order.estimated_time_to_ready - self.now
                        if self.status == DriverStatus.PICKING_UP_WAITING
                        else add_travel_time(self.coordinate, current_order.establishment.coordinate)
                    )
                    total_busy_time += self.time_between_picked_up_and_start_delivery()
                    total_busy_time += add_travel_time(current_order.establishment.coordinate, current_order.customer.coordinate)
                
                # Se o segmento atual é de entrega, considera o tempo de entrega
                elif self.current_route_segment.is_delivery():
                    if self.status == DriverStatus.DELIVERING:
                        total_busy_time += add_travel_time(self.coordinate, current_order.customer.coordinate)

                if self.status != DriverStatus.AVAILABLE:
                    total_busy_time += self.estimate_time_to_costumer_receive_order(current_order)

                # Atualiza a posição do motorista para o local da entrega
                valid_coordinate = current_order.customer.coordinate
        
        # Considera o tempo para processar todas as rotas na fila de pedidos
        for route in self.route_requests:
            # Tempo para aceitar ou rejeitar a rota
            total_busy_time += self.time_to_accept_or_reject_route()
            
            # Percorre cada segmento da rota
            for route_segment in route.route_segments:
                order = route_segment.order
                
                # Se o segmento é de coleta, calcula o tempo para pegar o pedido
                if route_segment.is_pickup():
                    total_busy_time += self.time_between_accept_and_start_picking_up()
                    total_busy_time += add_travel_time(valid_coordinate, order.establishment.coordinate)
                    total_busy_time += self.time_between_picked_up_and_start_delivery()
                
                # Se o segmento é de entrega, calcula o tempo de entrega
                elif route_segment.is_delivery():
                    total_busy_time += add_travel_time(order.establishment.coordinate, order.customer.coordinate)
                    total_busy_time += self.estimate_time_to_costumer_receive_order(order)                
                    valid_coordinate = order.customer.coordinate

        return max(total_busy_time, 0)

    
    def calculate_total_distance(self) -> Number:
        total_distance = 0
        valid_coordinate = self.coordinate  # Posição atual do motorista

        # Se o motorista já está em uma rota, inclui o tempo restante dessa rota
        if self.current_route is not None:
            current_order = self.current_route_segment.order if self.current_route_segment else None

            if current_order:
                # Se o segmento atual é de coleta, considera o tempo para pegar o pedido
                if self.current_route_segment.is_pickup():
                    total_distance += self.environment.map.distance(
                        self.coordinate, current_order.establishment.coordinate
                    )
                    total_distance += self.environment.map.distance(
                        current_order.establishment.coordinate, current_order.customer.coordinate
                    )

                # Se o segmento atual é de entrega, considera o tempo de entrega
                if self.current_route_segment.is_delivery():
                    if self.status == DriverStatus.DELIVERING:
                        total_distance += self.environment.map.distance(
                            self.coordinate, current_order.customer.coordinate
                        )

            # Atualiza a posição do motorista para o local da entrega
            valid_coordinate = self.current_route_segment.order.customer.coordinate
        
        # Considera o tempo para processar todas as rotas na fila de pedidos
        for route in self.route_requests:

            # Percorre cada segmento da rota
            for route_segment in route.route_segments:
                order = route_segment.order

                # Se o segmento é de coleta, calcula o tempo para pegar o pedido
                if route_segment.is_pickup():
                    total_distance += self.environment.map.distance(
                        valid_coordinate, order.establishment.coordinate
                    )

                # Se o segmento é de entrega, calcula o tempo de entrega
                if route_segment.is_delivery():
                    total_distance += self.environment.map.distance(
                        order.establishment.coordinate, order.customer.coordinate
                    )

                # Atualiza a posição após cada entrega
                valid_coordinate = order.customer.coordinate

        return max(total_distance, 0)
    
    def estimate_time_to_complete_next_order(self, nextOrder: Order):
        #   Este método só é chamado pelo ambiente gymnasium no momento em que o ambiente simpy já avançou a ponto de ter um 
        # novo pedido, portanto ele só é chamado quando tem um novo pedido para ser atribuído ao motorista
        #   Quando todos os pedidos forem atribuídos chegará um momento em que o próximo pedido não existirá, nesse caso
        # o método deve retornar 0
        if nextOrder == None:
            return 0

        estimated_time = self.time_between_accept_and_start_picking_up()
        estimated_time += self.environment.map.estimated_time(self.last_future_coordinate, nextOrder.establishment.coordinate, self.movement_rate)
        estimated_time += self.time_between_picked_up_and_start_delivery()
        estimated_time += self.environment.map.estimated_time(nextOrder.establishment.coordinate, nextOrder.customer.coordinate, self.movement_rate)
        estimated_time += self.estimate_time_to_costumer_receive_order(nextOrder)
        
        return estimated_time
    
    def update_statistcs_variables(self):
        if not self.is_active():
            self.idle_time += 1
        
        if self.status == DriverStatus.PICKING_UP_WAITING:
            self.time_waiting_for_order += 1

    def get_and_update_distance_traveled(self):
        distance_traveled = self.total_distance - self.last_total_distance
        self.last_total_distance = self.total_distance
        return distance_traveled

    def get_penality_for_time_spent_for_delivery(self) -> Number:
        total_penalty = self.sum_penalty_for_time_spent

        for order in self.orders_list:
            # Se o pedido foi alocado depois do último tempo de verificação, soma o tempo desde a alocação até agora
            # Agora se o pedido foi alocado antes do último tempo de verificação, soma o tempo desde o último tempo de verificação até agora
            start_time = max(order.time_that_driver_was_allocated, self.last_time_check)
            penalty = self.now - start_time

            if not order.is_already_caught():
                penalty *= 5
            
            total_penalty += penalty
        
        self.last_time_check = self.now
        self.sum_penalty_for_time_spent = 0
        return total_penalty
    
    def get_penality_for_late_orders(self) -> Number:
        penalty = (self.now - self.last_time_check) * len(self.orders_list)
        return penalty
    
    def get_last_coordinate_from_routes_list(self) -> Coordinate:
        if self.route_requests:
            return self.route_requests[-1].order.customer.coordinate
        return self.coordinate