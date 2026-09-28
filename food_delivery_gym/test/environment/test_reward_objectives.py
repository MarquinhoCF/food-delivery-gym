import pytest

from food_delivery_gym.test.conftest import TINY, make_env, run_episode


def _mod_policy(step_idx, env):
    return step_idx % env.num_drivers


PER_STEP_NON_POSITIVE = {1, 2, 3, 4, 9, 12, 13}
TERMINAL_ONLY = {5, 6, 7, 8, 10}


@pytest.mark.parametrize("objective", sorted(PER_STEP_NON_POSITIVE))
def test_per_step_rewards_non_positive(objective):
    env = make_env(TINY, seed=5, reward_objective=objective)
    transitions = run_episode(env, _mod_policy)
    for _, reward, _, _, _ in transitions:
        assert reward <= 0


@pytest.mark.parametrize("objective", sorted(TERMINAL_ONLY))
def test_terminal_only_rewards_zero_until_done(objective):
    env = make_env(TINY, seed=5, reward_objective=objective)
    transitions = run_episode(env, _mod_policy)
    assert transitions
    for _, reward, terminated, truncated, _ in transitions[:-1]:
        assert reward == 0
        assert not terminated and not truncated
    _, last_reward, terminated, truncated, _ = transitions[-1]
    assert terminated or truncated
    assert last_reward <= 0


def test_objective_11_non_negative_and_sums_to_delivered():
    env = make_env(TINY, seed=5, reward_objective=11)
    transitions = run_episode(env, _mod_policy)
    rewards = [reward for _, reward, _, _, _ in transitions]
    assert all(r >= 0 for r in rewards)
    assert sum(rewards) == env.simpy_env.state.get_orders_delivered()
