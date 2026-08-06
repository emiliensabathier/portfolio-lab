"""Tail risk measures.

Sign convention: every function returns a positive number expressing the magnitude of the
loss. A 95% VaR of 0.02 means "a 2% loss is exceeded 5% of the time".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie strictly between 0 and 1, got {level}")


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    """Empirical quantile of the loss distribution."""
    _check_level(level)
    return float(-np.quantile(returns.to_numpy(), 1.0 - level))


def historical_es(returns: pd.Series, level: float = 0.95) -> float:
    """Mean loss conditional on breaching the historical VaR threshold."""
    _check_level(level)
    threshold = np.quantile(returns.to_numpy(), 1.0 - level)
    tail = returns[returns <= threshold]
    if tail.empty:
        raise ValueError("no observations beyond the VaR threshold")
    return float(-tail.mean())


def cornish_fisher_var(returns: pd.Series, level: float = 0.95) -> float:
    """VaR adjusted for skewness and excess kurtosis (Cornish-Fisher expansion).

    Reduces to the Gaussian VaR when the sample is normal, and widens the tail when the
    distribution is left-skewed or fat-tailed.
    """
    _check_level(level)
    values = returns.to_numpy()
    z = stats.norm.ppf(1.0 - level)
    s = float(stats.skew(values))
    k = float(stats.kurtosis(values))  # already excess kurtosis
    adjusted = (
        z
        + (z**2 - 1.0) * s / 6.0
        + (z**3 - 3.0 * z) * k / 24.0
        - (2.0 * z**3 - 5.0 * z) * s**2 / 36.0
    )
    return float(-(values.mean() + adjusted * values.std(ddof=1)))
