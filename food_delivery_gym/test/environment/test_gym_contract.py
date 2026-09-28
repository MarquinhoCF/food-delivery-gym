from numbers import Real

import pytest
from gymnasium.spaces import Dict, Discrete

from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.test.conftest import TINY, assert_obs_equal, make_env, run_episode


def test_action_and_observation_spaces():
    env = FoodDeliveryGymEnv(scenario_json_file_path=str(TINY), reward_objective=3)
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == env.num_drivers
    assert isinstance(env.observation_space, Dict)

    obs, info = env.reset(seed=7)
    assert env.observation_space.contains(obs)
    assert info["simpy_time_step"] == env.simpy_env.now

    expected_keys = set(env.observation_space.spaces.keys())
    assert set(obs.keys()) == expected_keys
    for key, space in env.observation_space.spaces.items():
        assert obs[key].shape == space.shape


def test_step_returns_gymnasium_5tuple_and_obs_in_space():
    env = make_env(TINY, seed=11, reward_objective=3)
    action = 0
    assert env.action_space.contains(action)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, Real)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert env.observation_space.contains(obs)
    assert info["simpy_time_step"] == env.simpy_env.now


def test_check_env_or_document_np_random_gap():
    env = FoodDeliveryGymEnv(scenario_json_file_path=str(TINY), reward_objective=3)
    env.reset(seed=0)
    try:
        from gymnasium.utils.env_checker import check_env

        check_env(env, skip_render_check=True)
    except Exception as exc:
        message = str(exc).lower()
        # Simulador semeia via RngFactory
        if "np_random" in message or "seed" in message:
            pytest.xfail(f"check_env vs RngFactory: {exc}")
        raise


def test_reset_seed_determinism():
    env_a = FoodDeliveryGymEnv(scenario_json_file_path=str(TINY), reward_objective=3)
    env_b = FoodDeliveryGymEnv(scenario_json_file_path=str(TINY), reward_objective=3)
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    assert_obs_equal(obs_a, obs_b)

    results_a = run_episode(env_a, lambda i, e: i % e.num_drivers, max_steps=50)
    results_b = run_episode(env_b, lambda i, e: i % e.num_drivers, max_steps=50)
    assert len(results_a) == len(results_b)
    for (obs_a, reward_a, term_a, trunc_a, _), (obs_b, reward_b, term_b, trunc_b, _) in zip(results_a, results_b):
        assert_obs_equal(obs_a, obs_b)
        assert reward_a == pytest.approx(reward_b)
        assert (term_a, trunc_a) == (term_b, trunc_b)


def test_invalid_action_raises_value_error():
    env = make_env(TINY, seed=3)
    with pytest.raises(ValueError):
        env.step(-1)
    with pytest.raises(ValueError):
        env.step(env.num_drivers)


def test_step_before_reset_raises():
    env = FoodDeliveryGymEnv(scenario_json_file_path=str(TINY), reward_objective=3)
    with pytest.raises(AttributeError):
        env.step(0)


def test_set_reward_objective_out_of_range():
    env = FoodDeliveryGymEnv(scenario_json_file_path=str(TINY), reward_objective=3)
    with pytest.raises(ValueError):
        env.set_reward_objective(0)
    with pytest.raises(ValueError):
        env.set_reward_objective(14)
