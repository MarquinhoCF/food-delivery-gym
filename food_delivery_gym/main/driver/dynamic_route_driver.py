from dataclasses import dataclass
from typing import Dict, List, Optional
from food_delivery_gym.main.base.types import Coordinate, Number
from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.driver.driver_status import DriverStatus
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.events.driver_picked_up_order import DriverPickedUpOrder
from food_delivery_gym.main.order.order import Order
from food_delivery_gym.main.route.route_segment_type import RouteSegmentType

@dataclass
class TimeWindow:
    order_id: Number
    earliest_delivery: Number  # now + normal delivery time
    latest_delivery: Number    # now + latest delivery time
    collected: bool = False

@dataclass
class ReorderingEvent:
    """Registra um evento de reordenação de rota"""
    time: Number
    order_id: Number
    estimated_time_saved: Number  # Positivo = tempo economizado, Negativo = tempo perdido
    estimated_distance_saved: Number  # Positivo = distância economizada, Negativo = distância aumentada
    route_segment_type: RouteSegmentType
    
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

        # Estatísticas de reordenação
        self.reordering_events: List[ReorderingEvent] = []
        self.total_reorderings = 0
        self.successful_reorderings = 0  # Reordenações que economizaram tempo
        self.failed_reorderings = 0  # Reordenações que perderam tempo
        self.total_time_saved = 0
        self.total_time_lost = 0
        self.total_distance_saved = 0
        self.total_distance_increased = 0

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
                # Calcula o impacto estimado da reordenação
                if self.environment.env_mode != EnvMode.TRAINING:
                    time_impact, distance_impact = self._calculate_reordering_impact(next_order, collected_orders, RouteSegmentType.PICKUP)
                    self._record_reordering_event(next_order.order_id, time_impact, distance_impact, RouteSegmentType.PICKUP)

                # Insere o segmento de rota do próximo pedido antes do segmento atual
                self.current_route.move_segment_to_front_by_id(next_order.pick_up_route_segment_id)
               
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
            
            current_position = order.customer.coordinate
            current_time = delivery_time
        
        return True
    
    def _calculate_reordering_impact(self, next_order: Order, collected_orders: List[Order], route_segment_type: RouteSegmentType) -> tuple[Number, Number]:
        time_without_reordering = 0
        distance_without_reordering = 0
        position = self.coordinate
        
        for order in collected_orders:
            delivery_distance = self.environment.map.distance(position, order.customer.coordinate)
            distance_without_reordering += delivery_distance
            time_without_reordering += self._estimate_delivery_time_from(position, order)
            position = order.customer.coordinate
        
        collection_distance = self.environment.map.distance(position, next_order.establishment.coordinate)
        distance_without_reordering += collection_distance
        time_without_reordering += self._estimate_collection_time_from_position(next_order, position)
        
        time_with_reordering = self._estimate_collection_time(next_order)
        distance_with_reordering = self.environment.map.distance(self.coordinate, next_order.establishment.coordinate)
        position = next_order.establishment.coordinate
        
        for order in collected_orders:
            delivery_distance = self.environment.map.distance(position, order.customer.coordinate)
            distance_with_reordering += delivery_distance
            time_with_reordering += self._estimate_delivery_time_from(position, order)
            position = order.customer.coordinate
        
        # Tempo e distância economizados (positivo) ou perdidos (negativo)
        time_saved = time_without_reordering - time_with_reordering
        distance_saved = distance_without_reordering - distance_with_reordering
        
        return time_saved, distance_saved
    
    def _record_reordering_event(self, order_id: Number, time_impact: Number, distance_impact: Number, route_segment_type: RouteSegmentType) -> None:
        event = ReorderingEvent(
            time=self.now,
            order_id=order_id,
            estimated_time_saved=time_impact,
            estimated_distance_saved=distance_impact,
            route_segment_type=route_segment_type
        )
        
        self.reordering_events.append(event)
        self.total_reorderings += 1
        
        if time_impact > 0:
            self.successful_reorderings += 1
            self.total_time_saved += time_impact
        else:
            self.failed_reorderings += 1
            self.total_time_lost += abs(time_impact)

        if distance_impact > 0:
            self.total_distance_saved += distance_impact
        else:
            self.total_distance_increased += abs(distance_impact)
    
    def delivered(self, order: Order) -> None:
        super().delivered(order)

        # Remove a janela de tempo do pedido entregue
        if order.order_id in self.time_windows:
            del self.time_windows[order.order_id]

        self.current_load -= 1

        # Após entregar, avalia se deve coletar próximo pedido antes de continuar entregas
        # Essa verificação é necessária por conta do seguinte cenário:
        #       Pense num motorista que terminou de entregar um pedido e tem mais alguns em sua lista
        #   Se ele não está carregando nenhum pedido é claro que ele deve coletar o próximo pedido da lista. Dessa 
        #   forma, o processamento da heurística de reordenação é desnecessário, melhor simplesmente deixar o 
        #   simulador seguir o seu fluxo normalmente e coletar o próximo pedido.
        if self.current_load > 0:
            collected_orders = [order for order in self.orders_list if order.is_already_caught()]
            next_to_collect_orders = [order for order in self.orders_list if not order.is_already_caught()]

            # Se não há pedidos coletados ou não há pedidos para coletar, não faz nada
            if not collected_orders or not next_to_collect_orders:
                return
            
            next_order = next_to_collect_orders[0]
            
            # Verifica se coletar o próximo respeita as janelas dos pedidos coletados
            if self._can_collect_next_respecting_windows(next_order, collected_orders):
                # Calcula o impacto estimado da reordenação
                if self.environment.env_mode != EnvMode.TRAINING:
                    time_impact, distance_impact = self._calculate_reordering_impact(next_order, collected_orders, RouteSegmentType.PICKUP)
                    self._record_reordering_event(next_order.order_id, time_impact, distance_impact, RouteSegmentType.PICKUP)

                # Insere o segmento de rota do próximo pedido antes do segmento atual
                self.current_route.move_segment_to_front_by_id(next_order.pick_up_route_segment_id)

    def _estimate_collection_time(self, order: Order) -> Number:
        """Estima o tempo para coletar um pedido"""
        travel_time = self.environment.map.estimated_time(self.coordinate, order.establishment.coordinate, self.movement_rate)

        # Tempo de espera até o pedido estar pronto
        wait_time = max(0, order.estimated_ready_time - self.now - travel_time)
        
        return (order.estimated_time_between_accept_and_start_picking_up + travel_time + wait_time)
    
    def _estimate_collection_time_from_position(self, order: Order, position: Coordinate) -> Number:
        """Estima o tempo para coletar um pedido de uma posição específica"""
        travel_time = self.environment.map.estimated_time(
            position, 
            order.establishment.coordinate, 
            self.movement_rate
        )
        wait_time = max(0, order.estimated_ready_time - self.now - travel_time)
        return (order.estimated_time_between_accept_and_start_picking_up + 
                travel_time + wait_time)

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
    
    def get_reordering_statistics(self) -> Dict:
        """Retorna estatísticas sobre reordenação de rotas"""
        return {
            'total_reorderings': self.total_reorderings,
            'successful_reorderings': self.successful_reorderings,
            'failed_reorderings': self.failed_reorderings,
            'total_time_saved': self.total_time_saved,
            'total_time_lost': self.total_time_lost,
            'net_time_impact': self.total_time_saved - self.total_time_lost,
            'total_distance_saved': self.total_distance_saved,
            'total_distance_increased': self.total_distance_increased,
            'net_distance_impact': self.total_distance_saved - self.total_distance_increased,
            'success_rate': (self.successful_reorderings / self.total_reorderings * 100 if self.total_reorderings > 0 else 0),
            'events': self.reordering_events
        }
    
    def register_statistic_data(self):
        id = self.driver_id
        FoodDeliverySimpyEnv.driver_metrics[id]["orders_delivered"].append(self.orders_delivered)
        FoodDeliverySimpyEnv.driver_metrics[id]["time_spent_on_delivery"].append(self.get_time_spent_on_delivery())
        FoodDeliverySimpyEnv.driver_metrics[id]["idle_time"].append(self.idle_time)
        FoodDeliverySimpyEnv.driver_metrics[id]["time_waiting_for_order"].append(self.time_waiting_for_order)
        FoodDeliverySimpyEnv.driver_metrics[id]["total_distance"].append(self.total_distance)
        reordering = {
            'total_reorderings': self.total_reorderings,
            'net_time_impact': self.total_time_saved - self.total_time_lost,
            'net_distance_impact': self.total_distance_saved - self.total_distance_increased,
            'success_rate': (self.successful_reorderings / self.total_reorderings * 100 if self.total_reorderings > 0 else 0),
        }
        FoodDeliverySimpyEnv.driver_metrics[id]["reordering"].append(reordering)
    
    def reset_statistics(self):
        id = self.driver_id
        FoodDeliverySimpyEnv.driver_metrics[id]["orders_delivered"].clear()
        FoodDeliverySimpyEnv.driver_metrics[id]["time_spent_on_delivery"].clear()
        FoodDeliverySimpyEnv.driver_metrics[id]["idle_time"].clear()
        FoodDeliverySimpyEnv.driver_metrics[id]["time_waiting_for_order"].clear()
        FoodDeliverySimpyEnv.driver_metrics[id]["total_distance"].clear()
        FoodDeliverySimpyEnv.driver_metrics[id]["reordering"].clear()