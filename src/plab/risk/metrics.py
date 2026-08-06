"""Headline performance and risk metrics.

Every function takes a series of periodic simple returns and returns a scalar. Nothing
here knows where the returns came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252

# Guard threshold for divisions. Never compare a computed float to 0.0 by equality:
# a constant return series carries ~1e-18 of floating-point noise rather than an exact
# zero, so `x == 0.0` silently misses it and the division returns ~1e16 instead of inf.
# Any real annualized volatility is ~1e-2, so this threshold cannot mask a live value.
NUMERICAL_ZERO = 1e-12


def _wealth(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate implied by the return series."""
    if returns.empty:
        return float("nan")
    total = float(_wealth(returns).iloc[-1])
    years = len(returns) / PERIODS_PER_YEAR
    return total ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of returns."""
    return float(returns.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized excess return over annualized volatility.

    Infinite when volatility is zero, which is the mathematically correct answer for a
    riskless series rather than a masked division by zero.

    ``risk_free`` is de-annualized linearly rather than geometrically, consistent with the
    simple-return convention used throughout. The difference is negligible at realistic
    rates and would only matter for a high-rate regime.
    """
    excess = returns - risk_free / PERIODS_PER_YEAR
    volatility = annualized_volatility(excess)
    annual_excess = float(excess.mean() * PERIODS_PER_YEAR)
    if volatility <= NUMERICAL_ZERO:
        return np.inf if annual_excess > 0 else -np.inf if annual_excess < 0 else 0.0
    return annual_excess / volatility


def sortino_ratio(returns: pd.Series, target: float = 0.0) -> float:
    """Annualized excess return over downside deviation below ``target``.

    The downside deviation follows Sortino and Price: the sum of squared shortfalls is
    divided by the TOTAL number of periods, not by the number of losing ones. Periods
    above target contribute zero to the sum but still count in the denominator. Dividing
    by the count of losing periods instead inflates the deviation whenever losses are a
    minority — by a factor of about 1.5 at a 43% loss frequency — and understates the
    ratio correspondingly.
    """
    excess = returns - target / PERIODS_PER_YEAR
    downside = excess[excess < 0.0]
    annual_excess = float(excess.mean() * PERIODS_PER_YEAR)
    if downside.empty:
        return np.inf if annual_excess > 0 else 0.0
    deviation = float(
        np.sqrt((downside**2).sum() / len(excess)) * np.sqrt(PERIODS_PER_YEAR)
    )
    if deviation <= NUMERICAL_ZERO:
        return np.inf if annual_excess > 0 else 0.0
    return annual_excess / deviation


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown at each point, as a non-positive fraction of the running peak."""
    wealth = _wealth(returns)
    return wealth / wealth.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss, as a negative number."""
    if returns.empty:
        return float("nan")
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR divided by the absolute maximum drawdown."""
    worst = abs(max_drawdown(returns))
    if worst <= NUMERICAL_ZERO:
        return np.inf
    return cagr(returns) / worst


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Sensitivity of ``returns`` to ``benchmark``, estimated on their common dates."""
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        raise ValueError("fewer than two overlapping observations")
    reference = aligned.iloc[:, 1]
    variance = float(reference.var(ddof=1))
    if variance <= NUMERICAL_ZERO:
        raise ValueError("benchmark has zero variance; beta is undefined")
    covariance = float(aligned.iloc[:, 0].cov(reference))
    return covariance / variance


def summary(returns: pd.Series) -> dict[str, float]:
    """All headline metrics in one mapping."""
    return {
        "cagr": cagr(returns),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
    }
