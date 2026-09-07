import food_delivery_gym  # noqa: F401 — registra FoodDelivery-*-v1
import gymnasium as gym
import pytest

from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.test.conftest import STRESS, TINY, make_env


def _mod_policy(step_idx, env):
    return step_idx % env.num_drivers


@pytest.mark.parametrize("seed", [1, 21])
@pytest.mark.parametrize("scenario", [TINY, STRESS], ids=["tiny", "stress"])
def test_episode_invariants(seed, scenario):
    env = make_env(scenario, seed=seed, reward_objective=3)
    assert env.current_order is not None

    prev_now = env.simpy_env.now
    prev_delivered = env.simpy_env.state.get_orders_delivered()
    last_terminated = False
    last_truncated = False

    for step_idx in range(400):
        assert env.current_order is not None
        action = _mod_policy(step_idx, env)
        assert 0 <= action < env.num_drivers

        _, _, terminated, truncated, _ = env.step(action)

        now = env.simpy_env.now
        delivered = env.simpy_env.state.get_orders_delivered()
        assert now >= prev_now
        assert delivered >= prev_delivered
        assert delivered <= env.orders_generated
        prev_now = now
        prev_delivered = delivered

        if terminated or truncated:
            last_terminated = terminated
            last_truncated = truncated
            break
    else:
        raise AssertionError("episódio não terminou")

    if last_terminated:
        assert env.simpy_env.state.get_orders_delivered() >= env.orders_generated
    if last_truncated:
        assert env.simpy_env.now >= env.max_time_step - 1
    assert env.current_order is None


def test_gym_make_registered_env_one_step():
    from numbers import Real

    FoodDeliveryGymEnv.SCENARIO = None
    env = gym.make("FoodDelivery-simple-obj1-v1")
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == set(env.observation_space.spaces.keys())
    assert "simpy_time_step" in info
    action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    assert set(obs.keys()) == set(env.observation_space.spaces.keys())
    assert isinstance(reward, Real)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    env.close()
