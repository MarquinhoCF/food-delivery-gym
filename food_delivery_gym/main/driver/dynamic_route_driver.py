from typing import List
from food_delivery_gym.main.base.types import Number
from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.order.order import Order

class DynamicRouteDriver(Driver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_delay_percentage = 80  # Limite de piora percentual aceitável
        self.max_capacity = 2  # Capacidade máxima de pedidos que o motorista pode carregar
        self.current_load = 0  # Carga atual de pedidos

    def picked_up(self, order: Order) -> None:
        self.current_load += 1
        
        if self.current_load > 0 and len(self.orders_list) > 1:
            current_order = self.orders_list[0]
            next_order = self.orders_list[1]

            if self.should_collect_next_before_delivery(current_order, next_order):
                self.current_route.swap_route_segments_by_id(
                    current_order.delivery_route_segment_id, 
                    next_order.pick_up_route_segment_id
                )

        super().picked_up(order)
    
    def delivered(self, order: Order) -> None:
        super().delivered(order)
        self.current_load -= 1

    def should_collect_next_before_delivery(self, current_order: Order, next_order: Order) -> bool:
        if self.current_load >= self.max_capacity:
            return False  # Capacidade máxima atingida, não pode coletar mais pedidos
        
        normal_delivery_time = self._calculate_normal_delivery_time(current_order)

        if normal_delivery_time == 0:
            return False  # Evita divisão por zero se o tempo normal for zero
        
        detour_delivery_time = self._calculate_detour_delivery_time(current_order, next_order)
        
        # Calcula a piora percentual no tempo de entrega
        delay_increase = detour_delivery_time - normal_delivery_time
        delay_percentage = (delay_increase / normal_delivery_time) * 100
        
        # Retorna True se a piora estiver dentro do limite aceitável
        return delay_percentage <= self.max_delay_percentage
    
    def _calculate_normal_delivery_time(self, current_order: Order) -> Number:
        """
        Calcula o tempo para entregar o pedido atual seguindo a rota normal.
        
        Caminho: estabelecimento (posição atual) -> cliente
        """
        
        return current_order.estimated_time_between_picked_up_and_start_delivery + current_order.estimated_delivery_travel_time
    
    def _calculate_detour_delivery_time(self, current_order: Order, next_order: Order) -> Number:
        """
        Calcula o tempo para entregar o pedido atual fazendo desvio para coletar o próximo.

        Caminho: estabelecimento atual -> estabelecimento do próximo -> cliente atual
        """
        
        time = next_order.estimated_time_between_accept_and_start_picking_up + next_order.estimated_pickup_travel_time
        
        # Adiciona tempo de espera se o pedido não estiver pronto ao chegar no estabelecimento
        time += max(0, next_order.estimated_ready_time - self.now)

        time += current_order.estimated_time_between_picked_up_and_start_delivery + current_order.estimated_delivery_travel_time

        return time