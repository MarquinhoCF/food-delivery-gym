"""
Regressão linear do retorno restante da política de base (custo terminal).

O alvo é o retorno descontado G_t = r_t + alpha*r_{t+1} + ... calculado
sobre episódios da própria base. `terminal_cost_to_go` do rollout usa a
predição para compensar o truncamento no horizonte.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from food_delivery_gym.main.optimizer.terminal_cost.features import FEATURE_NAMES

DEFAULT_ROOT = Path("data/terminal_cost")
_STD_FLOOR = 1e-8


def discounted_returns(rewards, alpha: float) -> np.ndarray:
    """G_t = r_t + alpha*G_{t+1}, calculado de trás para frente."""
    rewards = np.asarray(rewards, dtype=np.float64)
    returns = np.empty_like(rewards)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + alpha * acc
        returns[i] = acc
    return returns


def linear_model_path(scenario: str, objective: int, base_key: str, root: Path | str | None = None) -> Path:
    root = DEFAULT_ROOT if root is None else root
    return Path(root) / scenario / f"obj_{objective}" / base_key / "linear_model.npz"


def samples_path(scenario: str, objective: int, base_key: str, root: Path | str | None = None) -> Path:
    root = DEFAULT_ROOT if root is None else root
    return Path(root) / scenario / f"obj_{objective}" / base_key / "samples.npz"


def fit_linear_model(features: np.ndarray, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mínimos quadrados sobre features padronizadas, com viés.

    Retorna (coef, feature_mean, feature_std); coef[0] é o viés e coef[1:]
    os pesos na escala padronizada.
    """
    features = np.asarray(features, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != returns.shape[0]:
        raise ValueError("features (n, d) e returns (n,) incompatíveis")

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < _STD_FLOOR, 1.0, std)

    standardized = (features - mean) / std
    design = np.hstack([np.ones((features.shape[0], 1)), standardized])
    coef, *_ = np.linalg.lstsq(design, returns, rcond=None)
    return coef, mean, std


def save_linear_model(
    path: Path,
    coef: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    *,
    alpha: float,
    scenario: str,
    objective: int,
    base_key: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        coef=coef,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_names=np.array(FEATURE_NAMES),
        alpha=np.float64(alpha),
        scenario=np.str_(scenario),
        objective=np.int64(objective),
        base_key=np.str_(base_key),
    )


@dataclass
class LinearTerminalCostModel:
    coef: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    alpha: float

    @classmethod
    def load(cls, path: Path | str) -> "LinearTerminalCostModel":
        with np.load(path) as data:
            return cls(
                coef=np.asarray(data["coef"], dtype=np.float64),
                feature_mean=np.asarray(data["feature_mean"], dtype=np.float64),
                feature_std=np.asarray(data["feature_std"], dtype=np.float64),
                alpha=float(data["alpha"]),
            )

    def predict(self, features: np.ndarray) -> float:
        standardized = (np.asarray(features, dtype=np.float64) - self.feature_mean) / self.feature_std
        return float(self.coef[0] + standardized @ self.coef[1:])
