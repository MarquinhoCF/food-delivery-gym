from typing import List

from food_delivery_gym.main.order.order import Order


class DeliveryEnvState:
    def __init__(self):
        self._customers = []
        self._establishments = []
        self._drivers = []
        self._orders: List[Order] = []

        # Orders ready for picking up
        self.orders_awaiting_delivery: List[Order] = []
        self.orders_delivered = 0

        self.successfully_assigned_routes = 0

        self.events = []

    @property
    def customers(self) -> List:
        return self._customers

    @property
    def establishments(self) -> List:
        return self._establishments

    @property
    def drivers(self) -> List:
        return self._drivers

    @property
    def orders(self) -> List[Order]:
        return self._orders

    def add_customers(self, customer: List):
        self._customers += customer

    def add_establishments(self, establishments: List) -> None:
        self._establishments += establishments

    def add_drivers(self, drivers: List) -> None:
        self._drivers += drivers

    def add_orders(self, orders: List) -> None:
        self._orders += orders

    def get_length_orders(self) -> int:
        return len(self._orders)

    def increment_assigned_routes(self) -> None:
        self.successfully_assigned_routes += 1

    def increment_orders_delivered(self) -> None:
        self.orders_delivered += 1

    def add_event(self, event) -> None:
        self.events.append(event)

    def log_events(self) -> None:
        for event in self.events:
            print(event)

    def print_state(self, options=None):
        if options is None:
            options = {
                "customers": True,
                "establishments": True,
                "drivers": True,
                "orders": True,
                "events": True,
                "orders_delivered": True
            }

        print("=== Estado do DeliveryEnvState ===")

        if options.get("customers", False):
            print("Clientes:")
            for idx, customer in enumerate(self.customers, start=1):
                print(f"Cliente {idx}: {customer.__dict__}")

        if options.get("establishments", False):
            print("\nEstabelecimentos:")
            for _, establishment in enumerate(self.establishments):
                print(f"Estabelecimento {establishment.establishment_id}: Coordenadas = {establishment.coordinate}, Pedidos em preparação = {establishment.orders_in_preparation}, Lenght vetor de pedidos aceitos = {sum(cook.get_length_orders_accepted() for cook in establishment.cooks)}, Tempo ocupado = {establishment.calculate_mean_overload_time()}")

        if options.get("drivers", False):
            print("\nMotoristas:")
            for _, driver in enumerate(self.drivers):
                print(f"Motorista {driver.driver_id}: Coordenadas = {driver.coordinate}, Última coordenada futura = {driver.last_future_coordinate}, Status = {driver.status} ")
                if driver.current_route_segment and driver.current_route_segment.order:
                    print(f"             ID do Pedido Atual: {driver.current_route_segment.order.order_id}, Status do Pedido Atual: {driver.current_route_segment.order.status}, Estabelecimento do Pedido Atual = {driver.current_route_segment.order.establishment.establishment_id}")

        if options.get("orders", False):
            print("\nPedidos:")
            for idx, order in enumerate(self.orders, start=1):
                print(f"Pedido {idx}: {order.__dict__}")

        if options.get("events", False):
            print("\nEventos:")
            for idx, event in enumerate(self.events, start=1):
                print(f"Evento {idx}: {event}")

        if options.get("orders_delivered", False):
            print(f"\nTotal de pedidos entregues: {self.orders_delivered}")
