import numpy as np
import pytest

from food_delivery_gym.main.establishment.cook import Cook
from food_delivery_gym.main.order.order_status import OrderStatus
from food_delivery_gym.test.conftest import (
    bare_simpy_env,
    make_customer,
    make_establishment,
    make_establishment_order_rate,
    make_order,
)


def test_cook_fifo_queue():
    env = bare_simpy_env()
    cook = Cook(env)
    est = make_establishment(env)
    o1 = make_order(env, est, order_id=1)
    o2 = make_order(env, est, order_id=2)
    cook.add_order_to_list(o1)
    cook.add_order_to_list(o2)
    assert cook.get_length_orders_accepted() == 2
    assert cook.pop_order() is o1
    assert cook.pop_order() is o2
    assert cook.get_length_orders_accepted() == 0


def test_cook_update_overload_time_branches():
    env = bare_simpy_env()
    cook = Cook(env)

    # Primeiro pedido
    cook.update_overload_time(10)
    assert cook.current_order_duration == 10
    assert cook.order_list_duration == 0
    assert cook.overloaded_until == 10

    # Pedido subsequente na fila
    cook.update_overload_time(5)
    assert cook.order_list_duration == 5
    assert cook.overloaded_until == 15

    # after_establishment_accept_order com current_order_duration != 0 → max(overloaded_until, now)
    env.run(until=20)
    cook.update_overload_time(5, after_establishment_accept_order=True)
    assert cook.overloaded_until == 20

    # after_establishment_accept_order com current_order_duration == 0
    cook2 = Cook(env)
    cook2.order_list_duration = 8
    cook2.update_overload_time(4, after_establishment_accept_order=True)
    assert cook2.current_order_duration == 4
    assert cook2.order_list_duration == 4
    assert cook2.overloaded_until == env.now + 4 + 4

    # Sem estimativa → clamp
    cook2.overloaded_until = 0
    cook2.update_overload_time()
    assert cook2.overloaded_until == env.now


def test_establishment_capacity_flags_and_available_cook():
    env = bare_simpy_env()
    est = make_establishment(env, production_capacity=2)
    assert est.is_empty() is True
    assert est.is_within_capacity() is True
    assert est.is_full() is False
    assert est.is_active() is False

    est.orders_in_preparation = 2
    assert est.is_full() is True
    assert est.is_within_capacity() is False
    assert est.is_active() is True

    est.cooks[0].overloaded_until = 50
    est.cooks[1].overloaded_until = 10
    assert est.get_available_cook() is est.cooks[1]


def test_accept_and_reject_order_side_effects():
    env = bare_simpy_env()
    est = make_establishment(env, production_capacity=2)
    order_ok = make_order(env, est, order_id=1)
    est.accept_order(order_ok)
    assert order_ok.status == OrderStatus.ESTABLISHMENT_ACCEPTED
    assert sum(c.get_length_orders_accepted() for c in est.cooks) == 1

    order_bad = make_order(env, est, customer=make_customer(env, customer_id=2), order_id=2)
    est.reject_order(order_bad)
    assert order_bad.status == OrderStatus.ESTABLISHMENT_REJECTED
    assert order_bad in est.orders_rejected


def test_time_to_prepare_order_floor():
    env = bare_simpy_env(seed=99)
    est = make_establishment(env)
    # Força RNG previsível: integers(-5,5) pode ser -5
    est.rng = np.random.default_rng(0)
    for estimated in [1, 3, 10]:
        assert est.time_to_prepare_order(estimated) >= 1


def test_establishment_order_rate_beta_params():
    env = bare_simpy_env()
    min_t, max_t = 20, 60

    low = make_establishment_order_rate(
        env, order_production_time_rate=min_t, min_prepare_time=min_t, max_prepare_time=max_t
    )
    assert low.a == pytest.approx(1.0)
    assert low.b == pytest.approx(6.0)

    mid = make_establishment_order_rate(
        env,
        establishment_id=2,
        order_production_time_rate=(min_t + max_t) / 2,
        min_prepare_time=min_t,
        max_prepare_time=max_t,
    )
    assert mid.a == pytest.approx(1 + 5 * 0.5)
    assert mid.b == pytest.approx(7 - mid.a)

    high = make_establishment_order_rate(
        env,
        establishment_id=3,
        order_production_time_rate=max_t,
        min_prepare_time=min_t,
        max_prepare_time=max_t,
    )
    assert high.a == pytest.approx(6.0)
    assert high.b == pytest.approx(1.0)

    assert low.operating_radius == 10
    assert low.min_prepare_time == min_t
    assert low.max_prepare_time == max_t


def test_establishment_order_rate_estimates_bounds_and_seed():
    env_a = bare_simpy_env(seed=7)
    env_b = bare_simpy_env(seed=7)
    # Consumir o mesmo número de RNGs do map antes dos atores seria frágil;
    # fixamos o rng do estabelecimento diretamente.
    est_a = make_establishment_order_rate(env_a, order_production_time_rate=40)
    est_b = make_establishment_order_rate(env_b, order_production_time_rate=40)
    est_a.rng = np.random.default_rng(123)
    est_b.rng = np.random.default_rng(123)

    samples_a = [est_a.time_estimate_to_prepare_order() for _ in range(30)]
    samples_b = [est_b.time_estimate_to_prepare_order() for _ in range(30)]
    assert samples_a == samples_b
    assert all(est_a.min_prepare_time <= s <= est_a.max_prepare_time for s in samples_a)


def test_establishment_order_rate_higher_rate_biases_mean_upward():
    env = bare_simpy_env()
    low = make_establishment_order_rate(env, establishment_id=1, order_production_time_rate=20)
    high = make_establishment_order_rate(env, establishment_id=2, order_production_time_rate=60)
    low.rng = np.random.default_rng(0)
    high.rng = np.random.default_rng(0)
    n = 200
    mean_low = sum(low.time_estimate_to_prepare_order() for _ in range(n)) / n
    mean_high = sum(high.time_estimate_to_prepare_order() for _ in range(n)) / n
    assert mean_high > mean_low
