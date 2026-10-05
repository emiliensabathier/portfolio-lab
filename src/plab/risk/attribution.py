"""Attribution of portfolio return and portfolio risk to individual assets.

Risk contribution is what visually separates risk parity from equal weight: two portfolios
can hold the same weights and carry entirely different risk shares.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plab.risk.metrics import NUMERICAL_ZERO


def return_contribution(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series:
    """Total return contributed by each asset over the whole period.

    Raises when the returns do not cover every weighted date and ticker. Reindexing alone
    would fill the gap with NaN, and ``sum`` skips NaN by default — so a ticker held at a
    real weight but missing from the returns would be reported as contributing exactly
    0.0, indistinguishable from an asset that genuinely went nowhere.
    """
    aligned = asset_returns.reindex(weights.index).reindex(columns=weights.columns)
    if aligned.isna().to_numpy().any():
        raise ValueError(
            "asset returns do not cover every weighted date and ticker; refusing to sum "
            "an incomplete attribution"
        )
    return (weights * aligned).sum(axis=0)


def risk_contribution(weights: dict[str, float], covariance: pd.DataFrame) -> pd.Series:
    """Euler risk contributions. They sum to the portfolio volatility."""
    order = list(covariance.columns)
    w = np.array([weights.get(ticker, 0.0) for ticker in order])
    sigma = covariance.to_numpy()
    portfolio_vol = float(np.sqrt(w @ sigma @ w))
    if portfolio_vol <= NUMERICAL_ZERO:
        raise ValueError("portfolio volatility is zero; risk contributions are undefined")
    return pd.Series(w * (sigma @ w) / portfolio_vol, index=order)


def bloc_contribution(contributions: pd.Series, blocs: dict[str, str]) -> pd.Series:
    """Aggregate per-asset contributions into their declared blocs."""
    labels = pd.Series({ticker: blocs[ticker] for ticker in contributions.index})
    return contributions.groupby(labels).sum()
