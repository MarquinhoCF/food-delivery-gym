from typing import List

from simpy.core import SimTime
from simpy.events import ProcessGenerator

from food_delivery_gym.main.actors.map_actor import MapActor
from food_delivery_gym.main.base.types import Coordinate, Number
from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.establishment.cook import Cook
from food_delivery_gym.main.events.estimated_order_preparation_time import EstimatedOrderPreparationTime
from food_delivery_gym.main.events.establishment_accepted_order import EstablishmentAcceptedOrder
from food_delivery_gym.main.events.establishment_finished_order import EstablishmentFinishedOrder
from food_delivery_gym.main.events.establishment_preparing_order import EstablishmentPreparingOrder
from food_delivery_gym.main.events.establishment_rejected_order import EstablishmentRejectedOrder
from food_delivery_gym.main.events.time_for_agent_allocate_driver import TimeForAgentAllocateDriver
from food_delivery_gym.main.order.order import Order
from food_delivery_gym.main.order.order_status import OrderStatus
from food_delivery_gym.main.establishment.catalog import Catalog


class Establishment(MapActor):
    def __init__(
            self,
            id: Number,
            environment: FoodDeliverySimpyEnv,
            coordinate: Coordinate,
            available: bool,
            catalog: Catalog,
            percentage_allocation_driver: Number = 0.7,
            production_capacity: Number = 4,
            use_estimate: bool = False
    ) -> None:
        
        self.establishment_id = id
            
        super().__init__(environment, coordinate, available)
        self.catalog = catalog
        self.production_capacity = production_capacity
        self.percentage_allocation_driver = percentage_allocation_driver
        self.use_estimate = use_estimate
        self.orders_in_preparation: int = 0

        self.order_requests: List[Order] = []
        self.orders_rejected: List[Order] = []
        
        self.num_cooks = production_capacity
        # Cria uma lista de instâncias de Cook
        self.cooks: list[Cook] = [Cook(self.environment) for _ in range(self.num_cooks)]

        # Variáveis para estatísticas
        self.orders_fulfilled: Number = 0
        self.max_orders_in_queue: Number = 0
        self.idle_time: Number = 0
        self.active_time: Number = 0

        self.process(self.process_order_requests())
        self.process(self.process_accepted_orders())

    def receive_order_requests(self, orders: List[Order]) -> None:
        self.order_requests += orders

    def process_order_requests(self) -> ProcessGenerator:
        while True:
            while self.order_requests:
                order = self.order_requests.pop(0)
                self.process(self.process_order_request(order))
            yield self.timeout(self.time_to_process_order_requests())

    def process_order_request(self, order) -> ProcessGenerator:
        yield self.timeout(self.time_to_accept_or_reject_order(order))
        accept = self.condition_to_accept(order)
        self.accept_order(order) if accept else self.reject_order(order)

    def accept_order(self, order) -> None:
        event = EstablishmentAcceptedOrder(
            order=order,
            customer_id=order.customer.customer_id,
            establishment_id=self.establishment_id,
            time=self.now
        )
        self.publish_event(event)
        estimated_time = self.estimate_preparation_time(order)

        available_cook = self.get_available_cook()

        available_cook.update_overload_time(estimated_time)
        order.establishment_accepted(self.now, estimated_time, available_cook.get_overloaded_until())

        available_cook.add_order_to_list(order)

        total_orders_in_queue = 0
        for cook in self.cooks:
            total_orders_in_queue += cook.get_length_orders_accepted()

        if (total_orders_in_queue > self.max_orders_in_queue):
            self.max_orders_in_queue = available_cook.get_length_orders_accepted()

        # TODO: Logs
        # print('\n----> Novo pedido <----')
        # print(order)

    def calculate_mean_overload_time(self) -> SimTime:
        # É necessário verificar se tempo de ocupação é pelo menos o momento atual para evitar valores negativos
        self.update_overload_time_cooks()

        establishment_busy_time = 0
        for i in range(0, self.num_cooks):
            establishment_busy_time += self.cooks[i].get_overloaded_until() - self.now
        establishment_busy_time = establishment_busy_time/self.num_cooks

        return establishment_busy_time

    def estimate_preparation_time(self, order) -> SimTime:
        estimated_time = self.time_estimate_to_prepare_order()
        event = EstimatedOrderPreparationTime(
            order=order,
            customer_id=order.customer.customer_id,
            establishment_id=self.establishment_id,
            estimated_time=estimated_time,
            time=self.now
        )
        self.publish_event(event)
        if self.use_estimate:
            self.environment.add_ready_order(order, event)
        return estimated_time

    def reject_order(self, order) -> None:
        self.publish_event(EstablishmentRejectedOrder(
            order=order,
            customer_id=order.customer.customer_id,
            establishment_id=self.establishment_id,
            time=self.now
        ))
        order.update_status(OrderStatus.ESTABLISHMENT_REJECTED)
        self.orders_rejected.append(order)

    def process_accepted_orders(self) -> ProcessGenerator:
        while True:
            for cook in self.cooks:
                if cook.get_length_orders_accepted() > 0 and not cook.get_is_cooking():
                    order = cook.pop_order()
                    cook.update_overload_time(order.estimated_preparation_duration, True)

                    updated_estimated_time = None
                    if cook.get_length_orders_accepted() == 0:
                        updated_estimated_time = cook.get_overloaded_until()
                    else:
                        updated_estimated_time = self.now + order.estimated_preparation_duration
                    
                    cook.set_is_cooking(True)
                    self.orders_in_preparation += 1
                    order.preparation_started(self.now, updated_estimated_time)
                    self.process(self.prepare_order(cook, order))

                yield self.timeout(self.time_check_to_start_preparation())

    def prepare_order(self, cook, order) -> ProcessGenerator:
        self.publish_event(EstablishmentPreparingOrder(
            order=order,
            customer_id=order.customer.customer_id,
            establishment_id=self.establishment_id,
            time=self.now
        ))

        order.update_status(OrderStatus.PREPARING)
        time_to_prepare = self.time_to_prepare_order(order.estimated_preparation_duration)
        order.set_actual_preparation_duration(time_to_prepare)

        time_to_allocate_driver = round(time_to_prepare * self.percentage_allocation_driver)

        # Define o tempo restante para preparar o pedido após alocar o motorista
        if time_to_allocate_driver <= time_to_prepare:
            remaining_time_to_prepare = time_to_prepare - time_to_allocate_driver
            yield from self._handle_driver_allocation(order, time_to_allocate_driver)
            yield self.timeout(remaining_time_to_prepare)
        # Trata para o caso em que o tempo de alocação do motorista (baseado na estimativa) é maior que o tempo efetivo de preparação
        else:
            yield self.timeout(time_to_prepare)
            excess_allocation_time = time_to_allocate_driver - time_to_prepare
            yield from self._handle_driver_allocation(order, excess_allocation_time)

        self.finish_order(cook, order)

    def _handle_driver_allocation(self, order: Order, allocation_time: SimTime):
        # Gerencia a alocação do motorista.
        yield self.timeout(allocation_time)
        allocation_event = TimeForAgentAllocateDriver(
            order=order,
            customer_id=order.customer.customer_id,
            establishment_id=self.establishment_id,
            time=self.now
        )
        self.publish_event(allocation_event)
        self.environment.add_core_event(allocation_event)

    def finish_order(self, cook, order: Order) -> None:
        event = EstablishmentFinishedOrder(
            order=order,
            customer_id=order.customer.customer_id,
            establishment_id=self.establishment_id,
            time=self.now
        )
        self.publish_event(event)
        order.ready(self.now)

        cook.set_is_cooking(False)
        self.orders_in_preparation -= 1
        cook.set_current_order_duration(0)
        self.orders_fulfilled += 1

        # TODO: Logs
        # print(f"\nPedido pronto no estabelecimento {self.establishment_id}: ")
        # print(order)

        if not self.use_estimate:
            self.environment.add_ready_order(order, event)

    def update_overload_time_cooks(self) -> None:
        for i in range(0, self.num_cooks):
            self.cooks[i].update_overload_time()

    def get_available_cook(self):
        self.update_overload_time_cooks()
        available_cook_index = 0
        for i in range(1, self.num_cooks):
            if self.cooks[i].get_overloaded_until() < self.cooks[available_cook_index].get_overloaded_until():
                available_cook_index = i
        return self.cooks[available_cook_index]


    def is_empty(self) -> bool:
        return sum(cook.get_length_orders_accepted() for cook in self.cooks) == 0

    def is_within_capacity(self) -> bool:
        return self.orders_in_preparation < self.production_capacity

    def is_full(self) -> bool:
        return self.orders_in_preparation >= self.production_capacity
    
    def is_active(self) -> bool:
        return not self.is_empty() or self.orders_in_preparation > 0

    def time_to_process_order_requests(self) -> SimTime:
        return self.rng.randrange(1, 5)

    def time_to_accept_or_reject_order(self, order: Order) -> SimTime:
        return self.rng.randrange(1, 5)

    def time_check_to_start_preparation(self) -> SimTime:
        return self.rng.randrange(1, 5)

    def time_estimate_to_prepare_order(self) -> SimTime:
        return self.rng.randrange(8, 20)

    def time_to_prepare_order(self, estimated_time: SimTime) -> SimTime:
        # Não faz sentido o tempo de preparo ser menor que 1
        return max(1, estimated_time + self.rng.randrange(-5, 5))

    def condition_to_accept(self, order) -> bool:
        return self.available
    
    def update_statistcs_variables(self):
        if self.is_active():
            self.active_time += 1
        else:
            self.idle_time += 1

    def get_coordinate(self) -> Coordinate:
        return self.coordinate