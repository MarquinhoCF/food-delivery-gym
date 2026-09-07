from collections import deque

import pytest

from food_delivery_gym.main.base.dimensions import Dimensions
from food_delivery_gym.main.driver.capacity import Capacity
from food_delivery_gym.main.driver.driver_status import DriverStatus
from food_delivery_gym.main.order.order_status import OrderStatus
from food_delivery_gym.main.route.route_segment_type import RouteSegmentType
from food_delivery_gym.test.conftest import (
    bare_simpy_env,
    make_customer,
    make_driver,
    make_dynamic_driver,
    make_establishment,
    make_order,
    make_pickup_delivery_route,
    make_reactive_driver,
)


def test_dimensions_and_capacity_fits():
    large = Dimensions(10, 10, 10, 10)
    small = Dimensions(5, 5, 5, 5)
    equal = Dimensions(10, 10, 10, 10)
    assert large > small
    assert small < large
    assert large == equal
    assert (large + small).length == 15
    assert large.value == 10 * 10 * 10 * 10

    capacity = Capacity(large)
    assert capacity.fits(small) is True
    assert capacity.fits(equal) is False


def test_driver_is_active_and_queue_counts():
    env = bare_simpy_env()
    driver = make_driver(env)
    assert driver.is_active() is False
    assert driver.get_number_of_orders_in_list() == 0

    establishment = make_establishment(env)
    order = make_order(env, establishment)
    route = make_pickup_delivery_route(env, order)
    driver.route_requests.append(route)
    assert driver.is_active() is True
    assert driver.get_number_of_orders_in_list() == 1


def test_get_status_for_observation():
    env = bare_simpy_env()
    driver = make_driver(env, status=DriverStatus.AVAILABLE)
    establishment = make_establishment(env)
    assert driver.get_status_for_observation() == DriverStatus.AVAILABLE

    o1 = make_order(env, establishment, order_id=1)
    o2 = make_order(env, establishment, customer=make_customer(env, customer_id=2, coordinate=(8, 8)), order_id=2)
    driver.orders_list = [o1]
    assert driver.get_status_for_observation() == DriverStatus.AVAILABLE
    driver.orders_list = [o1, o2]
    assert driver.get_status_for_observation() == DriverStatus.PROCESSING_PREVIOUS_ORDERS


def test_get_last_valid_coordinate():
    env = bare_simpy_env()
    driver = make_driver(env, coordinate=(1, 1))
    assert driver.get_last_valid_coordinate() == (1, 1)

    establishment = make_establishment(env, coordinate=(2, 2))
    customer = make_customer(env, coordinate=(9, 9))
    order = make_order(env, establishment, customer)
    driver.orders_list = [order]
    assert driver.get_last_valid_coordinate() == (9, 9)


def test_calculate_order_penalty_objectives():
    env = bare_simpy_env()
    # Avança o relógio SimPy de forma controlada
    env.run(until=10)

    establishment = make_establishment(env)
    order = make_order(env, establishment)
    order.time_that_driver_was_allocated = 0

    driver_default = make_driver(env, reward_objective=3)
    assert driver_default._calculate_order_penalty(order, 4) == 6  # now(10) - start(4)

    driver_heavy = make_driver(env, driver_id=2, reward_objective=9)
    # Ainda não coletado → ×5
    assert driver_heavy._calculate_order_penalty(order, 4) == 30

    order.status = OrderStatus.PICKED_UP
    order.time_it_was_picked_up = 7
    # Antes do pickup ×5 + depois do pickup 1×
    assert driver_heavy._calculate_order_penalty(order, 4) == (7 - 4) * 5 + (10 - 7)


def test_get_penality_for_late_orders():
    env = bare_simpy_env()
    env.run(until=20)
    driver = make_driver(env)
    driver.last_time_check = 5
    establishment = make_establishment(env)
    driver.orders_list = [make_order(env, establishment, order_id=1), make_order(env, establishment, order_id=2)]
    assert driver.get_penality_for_late_orders() == (20 - 5) * 2


def test_busy_time_and_distance_empty_and_with_route():
    env = bare_simpy_env()
    driver = make_driver(env, coordinate=(0, 0), movement_rate=5)
    assert driver.estimate_total_busy_time() == 0
    assert driver.calculate_total_distance_to_travel() == 0

    establishment = make_establishment(env, coordinate=(3, 0))
    customer = make_customer(env, coordinate=(3, 4))
    order = make_order(env, establishment, customer)
    order.estimated_ready_time = 0
    order.estimated_time_between_accept_and_start_picking_up = 0
    order.estimated_time_between_picked_up_and_start_delivery = 0
    order.estimated_delivery_travel_time = 1
    order.estimated_time_to_costumer_receive_order = 2
    route = make_pickup_delivery_route(env, order)
    driver.route_requests = deque([route])

    expected_distance = env.map.distance((0, 0), (3, 0)) + env.map.distance((3, 0), (3, 4))
    assert driver.calculate_total_distance_to_travel() == expected_distance
    assert driver.estimate_total_busy_time() > 0


def test_get_and_update_distance_traveled():
    driver = make_driver()
    driver.total_distance = 15
    driver.last_total_distance = 10
    assert driver.get_and_update_distance_traveled() == 5
    assert driver.last_total_distance == 15
    assert driver.get_and_update_distance_traveled() == 0


def test_dynamic_route_time_window():
    env = bare_simpy_env()
    env.run(until=5)
    driver = make_dynamic_driver(env, tolerance_percentage=0.5)
    establishment = make_establishment(env)
    order = make_order(env, establishment)
    order.estimated_time_between_picked_up_and_start_delivery = 2
    order.estimated_delivery_travel_time = 8

    driver._calculate_and_store_time_window(order)
    window = driver.time_windows[order.order_id]
    assert window.earliest_delivery == 5 + 10
    assert window.latest_delivery == 5 + 10 * 1.5
    assert window.collected is True


def test_can_collect_next_respecting_windows():
    env = bare_simpy_env()
    driver = make_dynamic_driver(env, coordinate=(0, 0), movement_rate=5, tolerance_percentage=0.0)

    est_near = make_establishment(env, establishment_id=1, coordinate=(1, 0))
    cust_near = make_customer(env, customer_id=1, coordinate=(1, 1))
    collected = make_order(env, est_near, cust_near, order_id=1)
    collected.estimated_ready_time = 0
    collected.estimated_time_between_accept_and_start_picking_up = 0
    collected.estimated_time_between_picked_up_and_start_delivery = 0
    collected.estimated_delivery_travel_time = 1
    collected.status = OrderStatus.PICKED_UP
    driver._calculate_and_store_time_window(collected)
    # Janela apertada: latest = now + 1
    driver.time_windows[collected.order_id].latest_delivery = 1

    est_far = make_establishment(env, establishment_id=2, coordinate=(15, 15))
    next_order = make_order(env, est_far, make_customer(env, customer_id=2, coordinate=(16, 16)), order_id=2)
    next_order.estimated_ready_time = 0
    next_order.estimated_time_between_accept_and_start_picking_up = 0

    assert driver._can_collect_next_respecting_windows(next_order, [collected]) is False

    # Com janela larga, deve caber
    driver.time_windows[collected.order_id].latest_delivery = 10_000
    assert driver._can_collect_next_respecting_windows(next_order, [collected]) is True


def test_reordering_event_stats():
    driver = make_dynamic_driver()
    driver._record_reordering_event(1, time_impact=5, distance_impact=2, route_segment_type=RouteSegmentType.PICKUP)
    driver._record_reordering_event(2, time_impact=-3, distance_impact=-4, route_segment_type=RouteSegmentType.PICKUP)

    stats = driver.get_reordering_statistics()
    assert stats["total_reorderings"] == 2
    assert stats["successful_reorderings"] == 1
    assert stats["failed_reorderings"] == 1
    assert stats["total_time_saved"] == 5
    assert stats["total_time_lost"] == 3
    assert stats["success_rate"] == 50.0
    assert stats["net_time_impact"] == 2
    assert stats["net_distance_impact"] == -2


def test_reactive_driver_accept_route_condition_by_distance():
    env = bare_simpy_env()
    driver = make_reactive_driver(env, coordinate=(0, 0), max_distance=5, available=True)

    est_near = make_establishment(env, coordinate=(2, 0))
    order_near = make_order(env, est_near, make_customer(env, coordinate=(3, 0)), order_id=1)
    route_near = make_pickup_delivery_route(env, order_near)
    assert driver.accept_route_condition(route_near) is True

    est_far = make_establishment(env, establishment_id=2, coordinate=(10, 10))
    order_far = make_order(env, est_far, make_customer(env, customer_id=2, coordinate=(11, 11)), order_id=2)
    route_far = make_pickup_delivery_route(env, order_far)
    assert driver.accept_route_condition(route_far) is False

    driver.available = False
    assert driver.accept_route_condition(route_near) is False
