from dataclasses import dataclass
from typing import Dict, List, Optional
from food_delivery_gym.main.base.types import Coordinate, Number
from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.driver.driver_status import DriverStatus
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.events.driver_picked_up_order import DriverPickedUpOrder
from food_delivery_gym.main.order.order import Order

@dataclass
class TimeWindow:
    order_id: Number
    earliest_delivery: Number  # now + normal delivery time
    latest_delivery: Number    # now + latest delivery time
    collected: bool = False

class DynamicRouteDriver(Driver):
    def __init__(
        self,
        id: Number,
        environment: FoodDeliverySimpyEnv,
        coordinate: Coordinate,
        available: bool,
        tolerance_percentage: Optional[Number] = 0.5,
        max_capacity: Optional[int] = 2,
        color: Optional[tuple[int, int, int]] = (255, 0, 0),
        status: Optional[DriverStatus] = DriverStatus.AVAILABLE,
        movement_rate: Optional[Number] = 5,
        reward_objective: Optional[Number] = 1,
    ):
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
        
        self.tolerance_percentage = tolerance_percentage  # Limite de piora percentual aceitável
        self.max_capacity = max_capacity  # Capacidade máxima de pedidos que o motorista pode carregar
        self.current_load = 0  # Carga atual de pedidos

        # Dicionário para armazenar janelas de tempo dos pedidos coletados
        self.time_windows: Dict[Number, TimeWindow] = {}

    def picked_up(self, order: Order) -> None:
        self.publish_event(DriverPickedUpOrder(
            order=order,
            customer_id=order.customer.customer_id,
            establishment_id=order.establishment.establishment_id,
            driver_id=self.driver_id,
            time=self.now
        ))
        order.picked_up(self.now)

        self.current_load += 1
        
        self._calculate_and_store_time_window(order)
        
        if self.current_load < self.max_capacity and len(self.orders_list) > self.current_load:

            collected_orders = [order for order in self.orders_list if order.is_already_caught()]
            next_to_collect_orders = [order for order in self.orders_list if not order.is_already_caught()]
            
            if not next_to_collect_orders:
                return  # Não há pedidos para coletar
            
            next_order = next_to_collect_orders[0]
            
            if self._can_collect_next_respecting_windows(next_order, collected_orders):
               # Insere o segmento de rota do próximo pedido antes do segmento atual
               self.current_route.insert_segment_before_first_segment_by_id(next_order.pick_up_route_segment_id)
               
        self.process(self.sequential_processor())

    def _calculate_and_store_time_window(self, order: Order) -> None:
        normal_delivery_time = (
            order.estimated_time_between_picked_up_and_start_delivery + 
            order.estimated_delivery_travel_time
        )
        
        latest_delivery_time = normal_delivery_time * (1 + self.tolerance_percentage)
        
        time_window = TimeWindow(
            order_id=order.order_id,
            earliest_delivery=self.now + normal_delivery_time,
            latest_delivery=self.now + latest_delivery_time,
            collected=True
        )
        
        self.time_windows[order.order_id] = time_window

    def _can_collect_next_respecting_windows(self, next_order: Order, collected_orders: List[Order]) -> bool:
        # Tempo para coletar o próximo pedido
        collection_time = self._estimate_collection_time(next_order)
        
        # Posição atual (será o estabelecimento do próximo pedido após coleta)
        current_position = next_order.establishment.coordinate
        current_time = self.now + collection_time
        
        # Verifica cada pedido coletado
        for order in collected_orders:
            time_window = self.time_windows.get(order.order_id)
            
            # Calcula tempo de entrega deste pedido após coletar o próximo
            delivery_time = current_time + self._estimate_delivery_time_from(current_position, order)
            
            # Verifica se o tempo de entrega está dentro da janela
            if delivery_time > time_window.latest_delivery:
                return False
            
            # Atualiza posição e tempo para o próximo pedido
            current_position = order.customer.coordinate
            current_time = delivery_time
        
        return True
    
    def delivered(self, order: Order) -> None:
        super().delivered(order)

        # Remove a janela de tempo do pedido entregue
        if order.order_id in self.time_windows:
            del self.time_windows[order.order_id]

        self.current_load -= 1

        # Após entregar, avalia se deve coletar próximo pedido antes de continuar entregas
        if self.current_load > 0:
            collected_orders = [order for order in self.orders_list if order.is_already_caught()]
            next_to_collect_orders = [order for order in self.orders_list if not order.is_already_caught()]

            # Se não há pedidos coletados ou não há pedidos para coletar, não faz nada
            if not collected_orders or not next_to_collect_orders:
                return
            
            next_order = next_to_collect_orders[0]
            
            # Verifica se coletar o próximo respeita as janelas dos pedidos coletados
            if self._can_collect_next_respecting_windows(next_order, collected_orders):
                # Insere o segmento de rota do próximo pedido antes do segmento atual
               self.current_route.insert_segment_before_first_segment_by_id(next_order.pick_up_route_segment_id)

    def _estimate_collection_time(self, order: Order) -> Number:
        """Estima o tempo para coletar um pedido"""
        travel_time = self.environment.map.estimated_time(self.coordinate, order.establishment.coordinate, self.movement_rate)

        # Tempo de espera até o pedido estar pronto
        wait_time = max(0, order.estimated_ready_time - self.now - travel_time)
        
        return (order.estimated_time_between_accept_and_start_picking_up + travel_time + wait_time)

    def _estimate_delivery_time_from(self, from_position: Coordinate, order: Order) -> Number:
        """Estima o tempo para entregar um pedido a partir de uma posição"""
        return (
            order.estimated_time_between_picked_up_and_start_delivery +
            self.environment.map.estimated_time(
                from_position,
                order.customer.coordinate,
                self.movement_rate
            )
        )