import numpy as np
import pytest

from food_delivery_gym.main.optimizer import catalog
from food_delivery_gym.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import NearestDriverOptimizerGym
from food_delivery_gym.main.optimizer.optimizer_gym.rollout_optimizer_gym import RolloutOptimizerGym
from food_delivery_gym.main.optimizer.terminal_cost.features import FEATURE_NAMES, extract_features
from food_delivery_gym.main.optimizer.terminal_cost.linear_model import (
    LinearTerminalCostModel,
    discounted_returns,
    fit_linear_model,
    linear_model_path,
    save_linear_model,
)

from food_delivery_gym.test.conftest import TINY, make_env


def test_discounted_returns_small_sequence():
    returns = discounted_returns([1.0, 2.0, 3.0], alpha=0.5)
    assert returns == pytest.approx([1.0 + 0.5 * (2.0 + 0.5 * 3.0), 2.0 + 0.5 * 3.0, 3.0])


def test_discounted_returns_alpha_one_is_suffix_sum():
    rewards = [-1.0, 4.0, -2.0]
    assert discounted_returns(rewards, alpha=1.0) == pytest.approx([1.0, 2.0, -2.0])


def test_extract_features_stable_dimension():
    env = make_env(TINY, seed=7)
    first = extract_features(env)
    assert first.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(first))

    env.step(0)
    second = extract_features(env)
    assert second.shape == first.shape


def test_fit_linear_model_recovers_synthetic_target():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(500, len(FEATURE_NAMES)))
    true_weights = rng.normal(size=len(FEATURE_NAMES))
    target = 2.5 + features @ true_weights

    coef, mean, std = fit_linear_model(features, target)
    model = LinearTerminalCostModel(coef=coef, feature_mean=mean, feature_std=std, alpha=0.9)

    for row in features[:10]:
        expected = 2.5 + row @ true_weights
        assert model.predict(row) == pytest.approx(expected, abs=1e-6)


def test_base_variant_key_for_instance():
    cls, kwargs = catalog.constructor_args("nearest_driver")
    assert catalog.base_variant_key_for_instance(cls, kwargs) == "nearest_driver"

    cls, kwargs = catalog.constructor_args("lowest", objective=3, cost_function="weighted_score")
    assert catalog.base_variant_key_for_instance(cls, kwargs) == "lowest_weighted_score"


def _write_model(tmp_path, scenario, objective, base_key, alpha, bias):
    coef = np.zeros(len(FEATURE_NAMES) + 1)
    coef[0] = bias
    save_linear_model(
        linear_model_path(scenario, objective, base_key, tmp_path),
        coef,
        np.zeros(len(FEATURE_NAMES)),
        np.ones(len(FEATURE_NAMES)),
        alpha=alpha,
        scenario=scenario,
        objective=objective,
        base_key=base_key,
    )


def test_terminal_cost_uses_saved_model(tmp_path, monkeypatch):
    env = make_env(TINY, seed=5, reward_objective=3)
    scenario = type(env).SCENARIO_NAME
    _write_model(tmp_path, scenario, 3, "nearest_driver", alpha=0.9, bias=-42.0)

    monkeypatch.setattr(
        "food_delivery_gym.main.optimizer.terminal_cost.linear_model.DEFAULT_ROOT", tmp_path
    )
    optimizer = RolloutOptimizerGym(
        env, base_optimizer_cls=NearestDriverOptimizerGym, alpha=0.9, horizon=2
    )
    assert optimizer.terminal_cost_to_go(env) == pytest.approx(-42.0)


def test_terminal_cost_zero_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "food_delivery_gym.main.optimizer.terminal_cost.linear_model.DEFAULT_ROOT", tmp_path
    )
    env = make_env(TINY, seed=5, reward_objective=3)
    optimizer = RolloutOptimizerGym(
        env, base_optimizer_cls=NearestDriverOptimizerGym, alpha=0.9, horizon=2
    )
    assert optimizer.terminal_cost_to_go(env) == 0.0


def test_terminal_cost_zero_on_alpha_mismatch(tmp_path, monkeypatch):
    env = make_env(TINY, seed=5, reward_objective=3)
    scenario = type(env).SCENARIO_NAME
    _write_model(tmp_path, scenario, 3, "nearest_driver", alpha=0.5, bias=-42.0)

    monkeypatch.setattr(
        "food_delivery_gym.main.optimizer.terminal_cost.linear_model.DEFAULT_ROOT", tmp_path
    )
    optimizer = RolloutOptimizerGym(
        env, base_optimizer_cls=NearestDriverOptimizerGym, alpha=0.9, horizon=2
    )
    assert optimizer.terminal_cost_to_go(env) == 0.0
