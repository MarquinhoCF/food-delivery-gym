from typing import List, TYPE_CHECKING

from food_delivery_gym.main.base.dimensions import Dimensions
from food_delivery_gym.main.base.types import Number
from food_delivery_gym.main.order.delivery_rejection import DeliveryRejection
from food_delivery_gym.main.order.item import Item
from food_delivery_gym.main.order.order_status import OrderStatus

if TYPE_CHECKING:
    # Importações apenas para anotação de tipos (não executam em runtime)
    from food_delivery_gym.main.customer.customer import Customer
    from food_delivery_gym.main.establishment.establishment import Establishment


class Order:
    def __init__(
            self,
            id: Number,
            customer: "Customer",
            establishment: "Establishment",
            request_date: int,
            items: List[Item],
    ):
        self.order_id = id
        self.customer: "Customer" = customer
        self.establishment: "Establishment" = establishment

        self.pick_up_route_segment_id = None
        self.delivery_route_segment_id = None

        self.request_date = request_date # Momento em que o pedido foi criado
        self.delivery_rejections: List[DeliveryRejection] = []
        self.items = items
        self.status: OrderStatus = OrderStatus.CREATED
        self.isReady = False
        self.required_capacity = self.calculate_required_capacity()

        # Atributos de tempo para preparação (atualizados pelo estabelecimento)
        self.time_establishment_accepted_order = None # Momento em que o pedido foi aceito pelo estabelecimento
        self.estimated_preparation_duration = None # Estimativa de tempo para preparo
        self.actual_preparation_duration = None # Tempo real que levou para preparar
        self.time_preparation_started = None # Momento em que a preparação começou
        self.estimated_ready_time = None # Estimativa do momento que o pedido ficará pronto
        self.time_order_became_ready = None # Momento em que o pedido ficou pronto
        
        # Atributos de tempo para entrega (atualizados pelo motorista)
        self.time_that_driver_was_allocated = None # Momento em que o motorista foi alocado
        self.time_it_was_picked_up = None # Momento em que o pedido foi retirado

        self.estimated_time_between_accept_and_start_picking_up = None # Estimativa de tempo entre aceitar o pedido e começar a retirá-lo
        self.estimated_pickup_travel_time = None # Estimativa de tempo de viagem para retirada
        self.estimated_time_between_picked_up_and_start_delivery = None # Estimativa de tempo entre retirar o pedido e começar a entrega
        self.estimated_delivery_travel_time = None # Estimativa de tempo de viagem para entrega
        self.estimated_time_to_costumer_receive_order = None # Estimativa de tempo para o cliente receber o pedido

    def calculate_required_capacity(self):
        dimensions = Dimensions(0, 0, 0, 0)
        for item in self.items:
            dimensions += item.dimensions
        return dimensions

    def update_status(self, status: OrderStatus):
        self.status = status

    def set_pickup_segment(self, route_segment_id: Number):
        self.pick_up_route_segment_id = route_segment_id
    
    def set_delivery_segment(self, route_segment_id: Number):
        self.delivery_route_segment_id = route_segment_id

    def establishment_accepted(self, now, estimated_preparation_duration: int, estimated_ready_time: int):
        self.status = OrderStatus.ESTABLISHMENT_ACCEPTED
        self.time_it_was_accepted = now
        self.estimated_preparation_duration = estimated_preparation_duration
        self.estimated_ready_time = estimated_ready_time

    def preparation_started(self, now, updated_estimated_ready_time: int):
        self.time_preparation_started = now
        self.estimated_ready_time = updated_estimated_ready_time

    def set_actual_preparation_duration(self, actual_preparation_duration: int):
        self.actual_preparation_duration = actual_preparation_duration

    def driver_allocated(
            self, 
            now, 
            estimated_time_between_accept_and_start_picking_up: int, 
            estimated_pickup_travel_time: int, 
            estimated_time_between_picked_up_and_start_delivery: int, 
            estimated_delivery_travel_time: int,
            estimated_time_to_costumer_receive_order: int
        ):
        self.time_that_driver_was_allocated = now
        self.estimated_time_between_accept_and_start_picking_up = estimated_time_between_accept_and_start_picking_up
        self.estimated_pickup_travel_time = estimated_pickup_travel_time
        self.estimated_time_between_picked_up_and_start_delivery = estimated_time_between_picked_up_and_start_delivery
        self.estimated_delivery_travel_time = estimated_delivery_travel_time
        self.estimated_time_to_costumer_receive_order = estimated_time_to_costumer_receive_order
    
    def ready(self, now):
        self.status = OrderStatus.READY
        self.isReady = True
        self.time_order_became_ready = now

    def picked_up(self, now):
        self.status = OrderStatus.PICKED_UP
        self.time_it_was_picked_up = now

    def is_already_caught(self):
        return self.status in (
            OrderStatus.PICKED_UP,
            OrderStatus.DELIVERING,
            OrderStatus.DRIVER_ARRIVED_DELIVERY_LOCATION,
            OrderStatus.RECEIVED,
            OrderStatus.DELIVERED,
        )

    def add_delivery_rejection(self, delivery_rejection: DeliveryRejection):
        self.delivery_rejections.append(delivery_rejection)

    def get_establishment(self) -> "Establishment":
        return self.establishment

    def get_customer(self) -> "Customer":
        return self.customer

    def get_estimated_ready_time(self) -> int:
        return self.estimated_ready_time

    def __str__(self):
        return (
            f"ID do Pedido: {self.order_id}\n"
            f"Coordenadas do Customer : {self.customer.coordinate}\n"
            f"Restaurante: {self.establishment.establishment_id}\n"
            f"Status: {self.status.name}\n"
            f"Tempo em que o pedido foi aceito: {self.time_it_was_accepted}\n"
            f"Tempo estimado de preparação: {self.estimated_preparation_duration}\n"
            f"Tempo em que a preparação começou: {self.time_preparation_started}\n"
            f"Tempo real de preparação: {self.actual_preparation_duration}\n"
            f"Tempo estimado para ficar pronto: {self.estimated_ready_time}\n"
            f"Tempo em que ficou pronto: {self.time_order_became_ready}\n"
            f"Tempo em que o motorista foi alocado: {self.time_that_driver_was_allocated}\n"
        )
