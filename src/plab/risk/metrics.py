"""Headline performance and risk metrics.

Every function takes a series of periodic simple returns and returns a scalar. Nothing
here knows where the returns came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

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


def _excess(returns: pd.Series, risk_free: float | pd.Series) -> pd.Series:
    """Returns net of the periodic riskless rate, aligned when that rate varies.

    A scalar rate is subtracted directly. A series is reindexed onto the return index
    first: a rate observed on a day the portfolio did not trade is irrelevant, and a
    return day with no rate quote would otherwise silently become NaN and drop out of
    the mean, quietly shortening the sample the ratio is computed over.
    """
    if isinstance(risk_free, pd.Series):
        aligned = risk_free.reindex(returns.index)
        if aligned.isna().any():
            raise ValueError(
                f"{int(aligned.isna().sum())} of {len(returns)} return dates have no "
                "risk-free quote; align the series before computing a ratio"
            )
        rate = aligned
    else:
        rate = risk_free
    return returns - rate / PERIODS_PER_YEAR


def sharpe_ratio(returns: pd.Series, risk_free: float | pd.Series = 0.0) -> float:
    """Annualized excess return over annualized volatility.

    Infinite when volatility is zero, which is the mathematically correct answer for a
    riskless series rather than a masked division by zero.

    ``risk_free`` is de-annualized linearly rather than geometrically, consistent with the
    simple-return convention used throughout. The difference is negligible at realistic
    rates and would only matter for a high-rate regime. It accepts a series as well as a
    scalar, because cash paid nothing for a decade and then paid five percent, and one
    number for the whole sample would misstate both halves.
    """
    excess = _excess(returns, risk_free)
    volatility = annualized_volatility(excess)
    annual_excess = float(excess.mean() * PERIODS_PER_YEAR)
    if volatility <= NUMERICAL_ZERO:
        return np.inf if annual_excess > 0 else -np.inf if annual_excess < 0 else 0.0
    return annual_excess / volatility


def sharpe_difference_test(
    returns: pd.Series, benchmark: pd.Series, risk_free: float | pd.Series = 0.0
) -> dict[str, float]:
    """Is the Sharpe of ``returns`` different from the benchmark's? Two-sided.

    Jobson and Korkie (1981) with the correction of Memmel (2003). On per-period excess
    returns with Sharpe ratios ``a`` and ``b`` and correlation ``rho`` over ``T`` periods,

        theta = [2 - 2 rho + (a^2 + b^2 - 2 a b rho^2) / 2] / T,   z = (a - b) / sqrt(theta)

    and the p-value is the two-sided normal tail of ``z``. The variance assumes returns
    that are independent and normal; daily portfolio returns are neither, which this
    test does not repair. The correlation term matters: two strategies that hold the same
    assets have highly correlated returns, and treating them as independent would make a
    real difference look like noise.

    Returns the annualized Sharpe difference, ``z`` and the p-value.
    """
    if not returns.index.equals(benchmark.index):
        raise ValueError("the two return series must cover the same dates")
    first, second = _excess(returns, risk_free), _excess(benchmark, risk_free)
    sr_first = float(first.mean() / first.std(ddof=1))
    sr_second = float(second.mean() / second.std(ddof=1))
    rho = float(np.corrcoef(first, second)[0, 1])
    theta = (
        2.0 - 2.0 * rho + 0.5 * (sr_first**2 + sr_second**2 - 2.0 * sr_first * sr_second * rho**2)
    ) / len(first)
    if theta <= NUMERICAL_ZERO:
        raise ValueError(
            "the Sharpe difference has degenerate variance; the two series are the same "
            "strategy up to scale"
        )
    z = (sr_first - sr_second) / np.sqrt(theta)
    return {
        "difference": (sr_first - sr_second) * np.sqrt(PERIODS_PER_YEAR),
        "z": float(z),
        "p_value": float(2.0 * stats.norm.sf(abs(z))),
    }


def sortino_ratio(returns: pd.Series, target: float | pd.Series = 0.0) -> float:
    """Annualized excess return over downside deviation below ``target``.

    The downside deviation follows Sortino and Price: the sum of squared shortfalls is
    divided by the TOTAL number of periods, not by the number of losing ones. Periods
    above target contribute zero to the sum but still count in the denominator. Dividing
    by the count of losing periods instead inflates the deviation whenever losses are a
    minority — by a factor of about 1.5 at a 43% loss frequency — and understates the
    ratio correspondingly.

    ``target`` is the minimum acceptable return. Passing the riskless rate makes the
    downside the shortfall against cash, which is the comparison the published figures use.
    """
    excess = _excess(returns, target)
    downside = excess[excess < 0.0]
    annual_excess = float(excess.mean() * PERIODS_PER_YEAR)
    if downside.empty:
        return np.inf if annual_excess > 0 else 0.0
    deviation = float(np.sqrt((downside**2).sum() / len(excess)) * np.sqrt(PERIODS_PER_YEAR))
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


def summary(returns: pd.Series, risk_free: float | pd.Series = 0.0) -> dict[str, float]:
    """All headline metrics in one mapping.

    ``risk_free`` feeds both ratios: it is the excess-return base for the Sharpe and the
    minimum acceptable return for the Sortino. Defaulting it to zero keeps the raw
    return-over-risk reading available, but the published report passes the realized
    Treasury bill series, because a Sharpe measured against zero over a window where cash
    paid five percent is not a risk-adjusted number.
    """
    return {
        "cagr": cagr(returns),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free),
        "sortino": sortino_ratio(returns, risk_free),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
    }
