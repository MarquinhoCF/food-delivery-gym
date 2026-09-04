import pytest

from food_delivery_gym.main.optimizer.optimizer_gym.first_driver_optimizer_gym import FirstDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.rollout_optimizer_gym import RolloutOptimizerGym

from tests.conftest import (
    STRESS,
    TINY,
    assert_envs_consistent,
    assert_obs_equal,
    make_env,
    structural_snapshot,
)


def _step_first_driver(env, n_steps: int):
    results = []
    for _ in range(n_steps):
        obs, reward, terminated, truncated, _ = env.step(0)
        results.append((obs, reward, terminated, truncated, env.simpy_env.now,
                        None if env.current_order is None else env.current_order.order_id))
        if terminated or truncated:
            break
    return results


def test_clone_structural_fidelity():
    env = make_env(TINY, seed=7)
    env.step(0)
    env.step(0)
    cloned = env.clone()

    assert structural_snapshot(env) == structural_snapshot(cloned)
    assert cloned.simpy_env is not env.simpy_env
    assert cloned.simpy_env.rng_factory is not env.simpy_env.rng_factory
    assert_obs_equal(env.get_observation(), cloned.get_observation())


def test_clone_same_rng_same_trajectory():
    env = make_env(TINY, seed=21)
    _step_first_driver(env, n_steps=2)

    cloned = env.clone()
    original_results = _step_first_driver(env, n_steps=6)
    cloned_results = _step_first_driver(cloned, n_steps=6)

    assert len(original_results) == len(cloned_results)
    for (obs_a, reward_a, term_a, trunc_a, now_a, order_a), (obs_b, reward_b, term_b, trunc_b, now_b, order_b) in zip(
        original_results, cloned_results
    ):
        assert_obs_equal(obs_a, obs_b)
        assert reward_a == pytest.approx(reward_b)
        assert (term_a, trunc_a, now_a, order_a) == (term_b, trunc_b, now_b, order_b)


def test_clone_rng_isolation():
    env = make_env(TINY, seed=33)
    _step_first_driver(env, n_steps=2)

    clone_run = env.clone()
    clone_control = env.clone()

    clone_run.step(0)

    orig_obs, orig_reward, orig_term, orig_trunc, _ = env.step(0)
    ctrl_obs, ctrl_reward, ctrl_term, ctrl_trunc, _ = clone_control.step(0)

    assert_obs_equal(orig_obs, ctrl_obs)
    assert orig_reward == pytest.approx(ctrl_reward)
    assert (orig_term, orig_trunc, env.simpy_env.now) == (ctrl_term, ctrl_trunc, clone_control.simpy_env.now)

    two_a = env.clone()
    two_b = env.clone()
    a_results = _step_first_driver(two_a, n_steps=4)
    b_results = _step_first_driver(two_b, n_steps=4)
    assert len(a_results) == len(b_results)
    for left, right in zip(a_results, b_results):
        assert_obs_equal(left[0], right[0])
        assert left[1:] == right[1:]


def test_rollout_select_driver_smoke():
    env = make_env(TINY, seed=9)
    optimizer = RolloutOptimizerGym(
        env,
        base_optimizer_cls=FirstDriverOptimizerGym,
        horizon=1,
    )
    obs = env.get_observation()
    order = env.get_current_order()
    action = optimizer.assign_driver_to_order(obs, order)
    assert action in range(env.num_drivers)

    env.step(action)
    assert env.simpy_env.now >= 0


def test_clone_stress_short_horizon():
    env = make_env(STRESS, seed=101)
    _step_first_driver(env, n_steps=3)
    cloned = env.clone()
    original_results = _step_first_driver(env, n_steps=8)
    cloned_results = _step_first_driver(cloned, n_steps=8)
    assert len(original_results) == len(cloned_results)
    for left, right in zip(original_results, cloned_results):
        assert_obs_equal(left[0], right[0])
        assert left[1] == pytest.approx(right[1])
        assert left[2:] == right[2:]


def _lockstep_until_done(env_a, env_b, action_fn, context: str, max_steps: int = 400):
    """Avança os dois ambientes com a mesma ação e o mesmo RNG copiado no clone."""
    for step_idx in range(max_steps):
        assert_envs_consistent(env_a, env_b, context=f"{context} antes do step {step_idx}")

        if env_a.current_order is None:
            assert env_b.current_order is None
            return step_idx

        action = action_fn(step_idx, env_a)
        assert action == action_fn(step_idx, env_b)

        obs_a, reward_a, term_a, trunc_a, _ = env_a.step(action)
        obs_b, reward_b, term_b, trunc_b, _ = env_b.step(action)

        assert_obs_equal(obs_a, obs_b)
        assert reward_a == pytest.approx(reward_b), f"{context} reward no step {step_idx}"
        assert (term_a, trunc_a) == (term_b, trunc_b), f"{context} flags no step {step_idx}"
        assert env_a.simpy_env.now == env_b.simpy_env.now
        order_a = None if env_a.current_order is None else env_a.current_order.order_id
        order_b = None if env_b.current_order is None else env_b.current_order.order_id
        assert order_a == order_b, f"{context} current_order no step {step_idx}"
        assert_envs_consistent(env_a, env_b, context=f"{context} depois do step {step_idx}")

        if term_a or trunc_a:
            return step_idx + 1

    raise AssertionError(f"{context}: episódio não terminou em {max_steps} steps")


@pytest.mark.parametrize("seed", [0, 1, 7, 21, 42, 99, 1234])
@pytest.mark.parametrize("scenario", [TINY, STRESS], ids=["tiny", "stress"])
def test_clone_lockstep_full_episode_forced_rng(seed, scenario):
    """Clona no início, força o mesmo RNG e a mesma política, e compara cada step até o fim."""
    env = make_env(scenario, seed=seed)
    cloned = env.clone()
    assert_envs_consistent(env, cloned, context=f"seed={seed} logo após clone")

    def action_fn(step_idx, current_env):
        return step_idx % current_env.num_drivers

    steps = _lockstep_until_done(env, cloned, action_fn, context=f"seed={seed}")
    assert steps >= 1


@pytest.mark.parametrize("seed", [3, 17, 88])
def test_clone_at_every_decision_then_lockstep_rest(seed):
    """Em cada ponto de decisão, clona o estado atual e reproduz o restante do episódio."""
    env = make_env(STRESS, seed=seed)
    max_decisions = 80

    for decision_idx in range(max_decisions):
        if env.current_order is None:
            break

        clone_a = env.clone()
        clone_b = env.clone()
        assert_envs_consistent(env, clone_a, context=f"seed={seed} decisão {decision_idx} clone_a")
        assert_envs_consistent(clone_a, clone_b, context=f"seed={seed} decisão {decision_idx} dois clones")

        def rest_action_fn(step_idx, current_env, offset=decision_idx):
            return (offset + step_idx) % current_env.num_drivers

        _lockstep_until_done(
            clone_a,
            clone_b,
            rest_action_fn,
            context=f"seed={seed} restante a partir da decisão {decision_idx}",
        )

        action = decision_idx % env.num_drivers
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    else:
        raise AssertionError("episódio não terminou no limite de decisões")


@pytest.mark.parametrize("seed", [5, 34])
def test_clone_of_clone_matches_source(seed):
    env = make_env(STRESS, seed=seed)
    _step_first_driver(env, n_steps=4)
    first = env.clone()
    second = first.clone()
    assert_envs_consistent(env, first, context="clone direto")
    assert_envs_consistent(env, second, context="clone do clone")
    assert_envs_consistent(first, second, context="dois níveis de clone")

    def action_fn(step_idx, current_env):
        return step_idx % current_env.num_drivers

    _lockstep_until_done(env, second, action_fn, context=f"seed={seed} clone-de-clone")
