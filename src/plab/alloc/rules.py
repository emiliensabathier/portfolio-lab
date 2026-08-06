"""Long-only, fully-invested allocation rules.

Every rule has the strategy signature ``(date, history) -> {ticker: weight}`` and receives
only the estimation window handed over by the engine.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from plab.returns import simple_returns
from plab.risk.covariance import ledoit_wolf_covariance

Estimator = Callable[[pd.DataFrame], pd.DataFrame]

SUM_TOLERANCE = 1e-6


class OptimizerError(Exception):
    """Raised when an allocation optimizer fails to converge.

    Never rescued by falling back to equal weight: that would publish equal-weight results
    under a minimum-variance label.
    """


def _as_dict(weights: np.ndarray, columns: pd.Index) -> dict[str, float]:
    clipped = np.clip(weights, 0.0, None)
    return dict(zip(columns, clipped / clipped.sum(), strict=True))


def _solve(objective, n_assets: int, label: str) -> np.ndarray:
    start = np.full(n_assets, 1.0 / n_assets)
    constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * n_assets
    result = minimize(
        objective, start, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise OptimizerError(f"{label} did not converge: {result.message}")
    return result.x


def equal_weight(date: pd.Timestamp, history: pd.DataFrame) -> dict[str, float]:
    """Equal split across every asset in the window."""
    return dict.fromkeys(history.columns, 1.0 / len(history.columns))


def fixed_weights(mapping: dict[str, float]):
    """Build a strategy that always returns ``mapping``. Used for benchmarks."""
    total = sum(mapping.values())
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise ValueError(f"fixed weights sum to {total}, expected 1.0")

    def strategy(date: pd.Timestamp, history: pd.DataFrame) -> dict[str, float]:
        return dict(mapping)

    return strategy


def min_variance(
    date: pd.Timestamp,
    history: pd.DataFrame,
    *,
    estimator: Estimator = ledoit_wolf_covariance,
) -> dict[str, float]:
    """Long-only portfolio with the lowest estimated variance."""
    cov = estimator(simple_returns(history)).to_numpy()

    def variance(weights: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    solution = _solve(variance, cov.shape[0], "min_variance")
    return _as_dict(solution, history.columns)
