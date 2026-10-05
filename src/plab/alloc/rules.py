"""Long-only, fully-invested allocation rules.

Every rule has the strategy signature ``(date, history) -> {ticker: weight}`` and receives
only the estimation window handed over by the engine.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from plab.backtest.engine import Strategy
from plab.cash import align as align_risk_free
from plab.returns import simple_returns
from plab.risk.covariance import ledoit_wolf_covariance
from plab.risk.metrics import NUMERICAL_ZERO, PERIODS_PER_YEAR

Estimator = Callable[[pd.DataFrame], pd.DataFrame]

SUM_TOLERANCE = 1e-6


class OptimizerError(Exception):
    """Raised when an allocation optimizer fails to converge.

    Never rescued by falling back to equal weight: that would publish equal-weight results
    under a minimum-variance label.
    """


def _as_dict(weights: np.ndarray, columns: pd.Index) -> dict[str, float]:
    """Clip SLSQP's tiny bound violations and renormalize to a fully-invested book.

    Renormalizing is only legitimate when something positive survives the clip. A solution
    that is entirely non-positive is not a portfolio to be rescued by division — dividing
    by its ~zero sum would manufacture arbitrary weights out of numerical dust.
    """
    clipped = np.clip(weights, 0.0, None)
    total = float(clipped.sum())
    if total <= NUMERICAL_ZERO:
        raise OptimizerError(
            "optimizer returned no positive weight; refusing to renormalize it into a portfolio"
        )
    return dict(zip(columns, clipped / total, strict=True))


def _solve(objective, n_assets: int, label: str) -> np.ndarray:
    start = np.full(n_assets, 1.0 / n_assets)
    constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * n_assets
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise OptimizerError(f"{label} did not converge: {result.message}")
    return result.x


def equal_weight(date: pd.Timestamp, history: pd.DataFrame) -> dict[str, float]:
    """Equal split across every asset in the window."""
    return dict.fromkeys(history.columns, 1.0 / len(history.columns))


def fixed_weights(mapping: dict[str, float]) -> Strategy:
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


def risk_parity(
    date: pd.Timestamp,
    history: pd.DataFrame,
    *,
    estimator: Estimator = ledoit_wolf_covariance,
) -> dict[str, float]:
    """Long-only portfolio where every asset contributes the same share of total risk."""
    cov = estimator(simple_returns(history)).to_numpy()
    n_assets = cov.shape[0]
    target = 1.0 / n_assets

    def dispersion(weights: np.ndarray) -> float:
        # Guards the division below. Unreachable with the default shrinkage estimator,
        # which is positive definite by construction, but a caller may pass
        # sample_covariance, which can be singular on a short window.
        portfolio_vol = float(np.sqrt(weights @ cov @ weights))
        if portfolio_vol <= NUMERICAL_ZERO:
            return 1e6
        contributions = weights * (cov @ weights) / portfolio_vol
        shares = contributions / contributions.sum()
        return float(np.sum((shares - target) ** 2))

    solution = _solve(dispersion, n_assets, "risk_parity")
    return _as_dict(solution, history.columns)


def max_sharpe(
    date: pd.Timestamp,
    history: pd.DataFrame,
    *,
    estimator: Estimator = ledoit_wolf_covariance,
    risk_free: pd.Series | None = None,
) -> dict[str, float]:
    """Long-only tangency portfolio, estimated on the trailing window it is handed.

    Expected returns are the window's sample means, so the rule is fitted on the past
    and held over the following month: the weights are out-of-sample, the estimates
    behind them are not forward-looking, and they are noisy by construction. It is
    reported alongside the other rules precisely so that sensitivity is visible.

    ``risk_free`` is the annualized bill series. When given, the ratio maximized is excess
    return over volatility — the Sharpe the report publishes — using the bill's average
    over the same window. It is cut at ``date`` here, because the series arrives whole.
    Without it the ratio is measured against zero, which ranks a calm asset earning less
    than cash above a volatile one that beats it.
    """
    window = simple_returns(history)
    cov = estimator(window).to_numpy()
    # Both halves of the ratio must annualize with the same constant. The estimator
    # annualizes the covariance with PERIODS_PER_YEAR, so hard-coding 252 here would let
    # numerator and denominator drift apart silently if that constant ever changed.
    expected = window.mean().to_numpy() * PERIODS_PER_YEAR
    if risk_free is not None:
        # Weights sum to one, so subtracting the window's average bill from every asset
        # subtracts it once from the portfolio: w @ (mu - rf) = w @ mu - rf.
        expected = expected - float(align_risk_free(risk_free.loc[:date], window.index).mean())

    def negative_sharpe(weights: np.ndarray) -> float:
        # Guards the division below. Unreachable with the default shrinkage estimator,
        # which is positive definite by construction, but a caller may pass
        # sample_covariance, which can be singular on a short window.
        volatility = float(np.sqrt(weights @ cov @ weights))
        if volatility <= NUMERICAL_ZERO:
            return 1e6
        return -float(weights @ expected) / volatility

    solution = _solve(negative_sharpe, cov.shape[0], "max_sharpe")
    return _as_dict(solution, history.columns)
