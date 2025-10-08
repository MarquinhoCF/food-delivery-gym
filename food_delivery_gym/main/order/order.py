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
        self.request_date = request_date
        self.items = items
        self.status: OrderStatus = OrderStatus.CREATED
        self.time_it_was_accepted = 0
        self.estimated_time_to_prepare = 0
        self.real_time_to_prepare = None
        self.time_preparation_started = None
        self.estimated_time_to_ready = 0
        self.time_that_driver_was_allocated = None
        self.time_it_was_ready = None
        self.isReady = False
        self.time_it_was_picked_up = None
        self.required_capacity = self.calculate_required_capacity()
        self.delivery_rejections: List[DeliveryRejection] = []

    def calculate_required_capacity(self):
        dimensions = Dimensions(0, 0, 0, 0)
        for item in self.items:
            dimensions += item.dimensions
        return dimensions

    def update_status(self, status: OrderStatus):
        self.status = status

    def establishment_accepted(self, now, estimated_time_to_prepare: int, estimated_time_to_ready: int):
        self.status = OrderStatus.ESTABLISHMENT_ACCEPTED
        self.time_it_was_accepted = now
        self.estimated_time_to_prepare = estimated_time_to_prepare
        self.estimated_time_to_ready = estimated_time_to_ready

    def preparation_started(self, now, updated_estimate_time_to_ready: int):
        self.time_preparation_started = now
        self.estimated_time_to_ready = updated_estimate_time_to_ready

    def set_real_time_to_prepare(self, real_time_to_prepare: int):
        self.real_time_to_prepare = real_time_to_prepare

    def driver_allocated(self, now):
        self.time_that_driver_was_allocated = now
    
    def ready(self, now):
        self.status = OrderStatus.READY
        self.isReady = True
        self.time_it_was_ready = now

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

    def get_estimated_time_to_ready(self) -> int:
        return self.estimated_time_to_ready

    def __str__(self):
        return (
            f"ID do Pedido: {self.order_id}\n"
            f"Coordenadas do Customer : {self.customer.coordinate}\n"
            f"Restaurante: {self.establishment.establishment_id}\n"
            f"Status: {self.status.name}\n"
            f"Tempo em que o pedido foi aceito: {self.time_it_was_accepted}\n"
            f"Tempo estimado de preparação: {self.estimated_time_to_prepare}\n"
            f"Tempo em que a preparação começou: {self.time_preparation_started}\n"
            f"Tempo real de preparação: {self.real_time_to_prepare}\n"
            f"Tempo estimado para ficar pronto: {self.estimated_time_to_ready}\n"
            f"Tempo em que o motorista foi alocado: {self.time_that_driver_was_allocated}\n"
            f"Tempo em que ficou pronto: {self.time_it_was_ready}\n"
        )
