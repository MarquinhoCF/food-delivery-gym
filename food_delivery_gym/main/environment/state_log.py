"""Snapshot legível do estado do ambiente para o log do terminal."""

from typing import Any, Iterable, List, Optional

_WIDTH = 76
_RECENT_EVENTS = 8


def format_environment_state(
    state,
    options: Optional[dict] = None,
    *,
    time_step=None,
    current_order=None,
) -> str:
    options = _resolve_options(options)
    lines: List[str] = []

    lines.append(_rule("═"))
    lines.append(_header(state, time_step))
    lines.append(_rule("─"))

    if options.get("current_order", True):
        lines.extend(_section("Pedido atual", _current_order_lines(current_order)))

    if options.get("establishments", False):
        lines.extend(_section("Estabelecimentos", _establishment_lines(state.establishments)))

    if options.get("drivers", False):
        lines.extend(_section("Motoristas", _driver_lines(state.drivers)))

    if options.get("orders_awaiting", True):
        awaiting = getattr(state, "orders_awaiting_delivery", [])
        lines.extend(_section("Aguardando entrega", _order_lines(awaiting, empty="nenhum")))

    if options.get("customers", False):
        lines.extend(_section("Clientes", _customer_lines(state.customers)))

    if options.get("orders", False):
        lines.extend(_section("Pedidos", _order_lines(state.orders, empty="nenhum")))

    if options.get("events", False):
        lines.extend(_section("Eventos", _event_lines(state.events)))

    if options.get("orders_delivered", False):
        lines.append(
            f"  entregues={state.orders_delivered}"
            f"   rotas_aceitas={getattr(state, 'successfully_assigned_routes', 0)}"
        )

    lines.append(_rule("═"))
    return "\n".join(lines)


def format_step_result(action, reward, sum_reward, info=None) -> str:
    reward_text = _fmt_num(reward, digits=2)
    lines = [
        f"  ação={action}"
        f"   recompensa={reward_text}"
        f"   acumulada={sum_reward:.2f}"
    ]
    info_line = _format_info(info)
    if info_line:
        lines.append(f"  info: {info_line}")
    return "\n".join(lines)


def _resolve_options(options: Optional[dict]) -> dict:
    defaults = {
        "customers": False,
        "establishments": True,
        "drivers": True,
        "orders": False,
        "events": False,
        "orders_delivered": True,
        "orders_awaiting": True,
        "current_order": True,
    }
    if options is None:
        return defaults
    resolved = dict(defaults)
    resolved.update(options)
    return resolved


def _header(state, time_step) -> str:
    time_text = "-" if time_step is None else _fmt_num(time_step, digits=1)
    awaiting = len(getattr(state, "orders_awaiting_delivery", []) or [])
    rejected = len(getattr(state, "rejected_deliveries", []) or [])
    return (
        f" ESTADO   t={time_text}"
        f"   pedidos={len(state.orders)}"
        f"   entregues={state.orders_delivered}"
        f"   aguardando={awaiting}"
        f"   recusados={rejected}"
    )


def _section(title: str, body: List[str]) -> List[str]:
    lines = [f" {title}"]
    lines.extend(body)
    lines.append(_rule("─"))
    return lines


def _current_order_lines(order) -> List[str]:
    if order is None:
        return ["  nenhum"]
    return _order_lines([order], detail=True)


def _establishment_lines(establishments) -> List[str]:
    if not establishments:
        return ["  nenhum"]

    lines = ["  id   posição         prep  fila  ocupação"]
    for establishment in establishments:
        queue = sum(cook.get_length_orders_accepted() for cook in establishment.cooks)
        lines.append(
            f"  {establishment.establishment_id:>2}   "
            f"{_fmt_coord(establishment.coordinate)}  "
            f"{establishment.orders_in_preparation:>4}  "
            f"{queue:>4}  "
            f"{_fmt_num(establishment.calculate_mean_overload_time()):>8}"
        )
    return lines


def _driver_lines(drivers) -> List[str]:
    if not drivers:
        return ["  nenhum"]

    lines = ["  id   status                          posição        destino       qtd"]
    for driver in drivers:
        status = _enum_name(driver.status)
        queue_size = len(getattr(driver, "orders_list", []) or [])
        pending = len(getattr(driver, "route_requests", []) or [])
        lines.append(
            f"  {driver.driver_id:>2}   "
            f"{status:<30}  "
            f"{_fmt_coord(driver.coordinate)}  "
            f"{_fmt_coord(driver.get_last_valid_coordinate())}  "
            f"{queue_size:>3}"
        )
        current = _current_segment_line(driver)
        if current:
            lines.append(f"       atual: {current}")
        queue = _orders_inline(getattr(driver, "orders_list", []))
        if queue:
            lines.append(f"       fila:  {queue}")
        if pending:
            lines.append(f"       requisições pendentes: {pending}")
    return lines


def _current_segment_line(driver) -> Optional[str]:
    segment = getattr(driver, "current_route_segment", None)
    if segment is None or getattr(segment, "order", None) is None:
        return None
    order = segment.order
    kind = _enum_name(getattr(segment, "route_segment_type", None)).lower()
    return (
        f"#{order.order_id} {_enum_name(order.status)}"
        f"  est={order.establishment.establishment_id}"
        f"  segmento={kind}"
    )


def _customer_lines(customers) -> List[str]:
    if not customers:
        return ["  nenhum"]

    lines = ["  id   status                  posição"]
    for customer in customers:
        lines.append(
            f"  {customer.customer_id:>2}   "
            f"{_enum_name(customer.status):<22}  "
            f"{_fmt_coord(customer.coordinate)}"
        )
    return lines


def _order_lines(orders: Iterable, empty: str = "nenhum", detail: bool = False) -> List[str]:
    orders = list(orders or [])
    if not orders:
        return [f"  {empty}"]

    lines = []
    for order in orders:
        customer_id = getattr(getattr(order, "customer", None), "customer_id", "-")
        customer_coord = _fmt_coord(getattr(getattr(order, "customer", None), "coordinate", None))
        establishment_id = getattr(getattr(order, "establishment", None), "establishment_id", "-")
        lines.append(
            f"  #{order.order_id:<4} {_enum_name(order.status):<28}  "
            f"est={establishment_id:<3} cliente={customer_id} {customer_coord}"
        )
        if detail:
            timing = _order_timing(order)
            if timing:
                lines.append(f"       {timing}")
    return lines


def _order_timing(order) -> str:
    parts = []
    accepted = getattr(order, "time_it_was_accepted", None)
    if accepted is None:
        accepted = getattr(order, "time_establishment_accepted_order", None)
    ready_at = getattr(order, "estimated_ready_time", None)
    allocated = getattr(order, "time_that_driver_was_allocated", None)
    if accepted is not None:
        parts.append(f"aceito={_fmt_num(accepted, digits=1)}")
    if ready_at is not None:
        parts.append(f"pronto_em={_fmt_num(ready_at, digits=1)}")
    if allocated is not None:
        parts.append(f"alocado={_fmt_num(allocated, digits=1)}")
    return "  ".join(parts)


def _event_lines(events) -> List[str]:
    events = list(events or [])
    if not events:
        return ["  nenhum"]

    shown = events[-_RECENT_EVENTS:]
    hidden = len(events) - len(shown)
    lines = []
    if hidden:
        lines.append(f"  ... {hidden} anteriores omitidos")
    for event in shown:
        lines.append(f"  {_format_event(event)}")
    return lines


def _format_event(event) -> str:
    event_type = _enum_name(getattr(event, "event_type", type(event).__name__))
    order = getattr(event, "order", None)
    order_id = getattr(order, "order_id", None)
    bits = [f"t={_fmt_num(getattr(event, 'time', None), digits=1):<6} {event_type}"]
    if order_id is not None:
        bits.append(f"pedido=#{order_id}")
    if hasattr(event, "establishment_id"):
        bits.append(f"est={event.establishment_id}")
    if hasattr(event, "customer_id"):
        bits.append(f"cliente={event.customer_id}")
    if hasattr(event, "driver_id"):
        bits.append(f"motorista={event.driver_id}")
    return "  ".join(bits)


def _orders_inline(orders) -> str:
    parts = []
    for order in orders or []:
        parts.append(f"#{order.order_id} {_enum_name(order.status)}")
    return ", ".join(parts)


def _format_info(info: Any) -> str:
    if not info:
        return ""
    if isinstance(info, dict):
        return ", ".join(f"{key}={_compact(value)}" for key, value in info.items())
    return _compact(info)


def _compact(value: Any) -> str:
    text = str(value).replace("\n", " ")
    if len(text) > 80:
        return text[:77] + "..."
    return text


def _enum_name(value) -> str:
    if value is None:
        return "-"
    return getattr(value, "name", str(value))


def _fmt_coord(coord) -> str:
    if coord is None:
        return f"{'-':^13}"
    try:
        return f"({float(coord[0]):>5.1f},{float(coord[1]):>5.1f})"
    except (TypeError, ValueError, IndexError):
        return str(coord)


def _fmt_num(value, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.{digits}f}"


def _rule(char: str) -> str:
    return char * _WIDTH
