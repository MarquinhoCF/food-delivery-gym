from __future__ import annotations

from food_delivery_gym.main.view.food_delivery_view import FoodDeliveryView
from food_delivery_gym.main.view.websocket_server import WebSocketBroadcaster
from food_delivery_gym.main.driver.driver_status import DriverStatus
from food_delivery_gym.main.customer.custumer_status import CustumerStatus


class LeafletView(FoodDeliveryView):
    """
    Envia o estado da simulação via WebSocket para uma página Leaflet.
    """

    def __init__(self, host: str = "localhost", port: int = 8765, fps: int = 10):
        super().__init__(fps=fps)
        self._broadcaster = WebSocketBroadcaster(host, port)
        self._broadcaster.start()
        self._step = 0

    def render(self, environment) -> None:
        if self.quited:
            return

        osm_map = environment.map   # OSMnxMap

        drivers = []
        for d in environment.state.drivers:
            lat, lon = osm_map.to_latlon(d.coordinate)
            route_path = []
            if d.current_route_segment is not None:
                route_path = osm_map.get_path_latlon(
                    d.coordinate,
                    d.current_route_segment.coordinate,
                )
            drivers.append({
                "id": d.driver_id,
                "lat": lat,
                "lon": lon,
                "status": d.status.name,
                "color": f"#{d.color[0]:02x}{d.color[1]:02x}{d.color[2]:02x}",
                "route": route_path,
                "orders": len(d.orders_list),
            })

        establishments = []
        for e in environment.state.establishments:
            lat, lon = osm_map.to_latlon(e.coordinate)
            establishments.append({
                "id": e.establishment_id,
                "lat": lat,
                "lon": lon,
                "active": e.is_active(),
                "queue": e.orders_in_preparation,
            })

        customers = []
        for c in environment.state.customers:
            if c.status == CustumerStatus.WAITING_DELIVERY:
                lat, lon = osm_map.to_latlon(c.coordinate)
                customers.append({
                    "id": c.customer_id,
                    "lat": lat,
                    "lon": lon,
                })

        payload = {
            "type": "state",
            "time": float(environment.now),
            "delivered": environment.state.get_orders_delivered(),
            "drivers": drivers,
            "establishments": establishments,
            "customers": customers,
        }

        self._broadcaster.broadcast(payload)
        self._step += 1

    def quit(self):
        self.quited = True