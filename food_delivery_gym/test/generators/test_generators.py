import numpy as np
import pytest

from food_delivery_gym.main.generator.initial_dynamic_route_driver_generator import InitialDynamicRouteDriverGenerator
from food_delivery_gym.main.generator.initial_establishment_order_rate_generator import (
    InitialEstablishmentOrderRateGenerator,
)
from food_delivery_gym.main.generator.non_homogeneous_poisson_order_generator import (
    NonHomogeneousPoissonOrderGenerator,
)
from food_delivery_gym.main.generator.poisson_order_generator import PoissonOrderGenerator
from food_delivery_gym.main.utils.rng_factory import RngFactory
from food_delivery_gym.test.conftest import bare_simpy_env


def test_poisson_validation_and_defaults():
    with pytest.raises(ValueError):
        PoissonOrderGenerator(0, 10, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        PoissonOrderGenerator(5, 0, rng=np.random.default_rng(0))

    gen = PoissonOrderGenerator(20, 40, rng=np.random.default_rng(0))
    assert gen.lambda_rate == pytest.approx(0.5)


def test_poisson_arrival_times_monotonic_seeded():
    gen_a = PoissonOrderGenerator(30, 50, rng=np.random.default_rng(7))
    gen_b = PoissonOrderGenerator(30, 50, rng=np.random.default_rng(7))
    times_a = gen_a.arrival_times
    times_b = gen_b.arrival_times

    assert times_a == times_b
    assert gen_a.get_number_of_orders_generated() == len(times_a)
    assert all(0 < t <= 50 for t in times_a)
    assert times_a == sorted(times_a)
    assert len(set(times_a)) == len(times_a)


def test_nhpp_constant_rate_and_auto_max_rate():
    rate = 0.4
    gen = NonHomogeneousPoissonOrderGenerator(
        estimated_num_orders=20,
        time_window=40,
        rate_function=lambda t: rate,
        rng=np.random.default_rng(3),
    )
    assert gen.max_rate == pytest.approx(rate * 1.1)
    times = gen.arrival_times
    assert all(0 < t <= 40 for t in times)
    assert times == sorted(times)

    # Clamp: rate > max_rate não explode
    gen_clamp = NonHomogeneousPoissonOrderGenerator(
        estimated_num_orders=10,
        time_window=20,
        rate_function=lambda t: 10.0,
        max_rate=1.0,
        rng=np.random.default_rng(1),
    )
    assert all(0 < t <= 20 for t in gen_clamp.arrival_times)


def test_initial_dynamic_route_driver_generator_counts_and_ranges():
    env = bare_simpy_env(seed=11)
    gen = InitialDynamicRouteDriverGenerator(
        num_drivers=4,
        vel_drivers=[3, 5],
        tolerance_percentage=0.5,
        max_capacity=2,
        reward_objective=3,
        rng=RngFactory(seed=11).next(),
    )
    gen.run(env)
    drivers = env.state.drivers
    assert len(drivers) == 4
    assert [d.driver_id for d in drivers] == [1, 2, 3, 4]
    for driver in drivers:
        assert 3 <= driver.movement_rate <= 5
        assert driver.tolerance_percentage == 0.5
        assert driver.max_capacity == 2


def test_initial_establishment_order_rate_generator_counts_and_ranges():
    env = bare_simpy_env(seed=22)
    gen = InitialEstablishmentOrderRateGenerator(
        num_establishments=3,
        prepare_time=[8, 15],
        operating_radius=[5, 10],
        production_capacity=[2, 2],
        percentage_allocation_driver=0.7,
        rng=RngFactory(seed=22).next(),
    )
    gen.run(env)
    establishments = env.state.establishments
    assert len(establishments) == 3
    for est in establishments:
        assert est.use_estimate is True
        assert 8 <= est.order_production_time_rate <= 15
        assert 5 <= est.operating_radius <= 10
        assert est.production_capacity == 2
        assert est.min_prepare_time == 8
        assert est.max_prepare_time == 15
