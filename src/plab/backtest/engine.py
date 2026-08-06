"""Monthly-rebalanced portfolio backtest.

The engine is the single seam between data and strategies. It hands a strategy a copy of
the trailing estimation window ending at the decision date and nothing else, so a strategy
cannot see the future even if its author tries. Weights decided on date ``d`` are applied
from the following trading day.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from plab.returns import simple_returns

Strategy = Callable[[pd.Timestamp, pd.DataFrame], dict[str, float]]

WEIGHT_SUM_TOLERANCE = 1e-6


class StrategyError(Exception):
    """Raised when a strategy returns unusable weights or cannot be run."""


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest parameters."""

    estimation_months: int = 36
    cost_bps: float = 5.0
    no_trade_band: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a backtest. ``returns`` is always the net series."""

    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame  # targets, at rebalance dates
    held_weights: pd.DataFrame  # actually held, daily, aligned with ``returns``
    turnover: pd.Series
    costs: pd.Series


def _rebalance_dates(index: pd.DatetimeIndex, first_valid: pd.Timestamp) -> list[pd.Timestamp]:
    """Last trading day of each month, from ``first_valid`` onwards."""
    eligible = index[index >= first_valid]
    if eligible.empty:
        return []
    month_ends = pd.Series(eligible, index=eligible).groupby(eligible.to_period("M")).max()
    return list(month_ends)


def _validate(weights: dict[str, float], columns: pd.Index, date: pd.Timestamp) -> pd.Series:
    unknown = set(weights) - set(columns)
    if unknown:
        raise StrategyError(f"strategy returned unknown tickers {sorted(unknown)} at {date}")
    series = pd.Series(weights, dtype=float).reindex(columns).fillna(0.0)
    total = float(series.sum())
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise StrategyError(f"weights at {date} sum to {total:.6f}, expected 1.0")
    return series


def run_backtest(
    prices: pd.DataFrame, strategy: Strategy, config: BacktestConfig
) -> BacktestResult:
    """Run ``strategy`` over ``prices`` with monthly rebalancing."""
    prices = prices.sort_index()
    asset_returns = simple_returns(prices)

    first_valid = prices.index.min() + pd.DateOffset(months=config.estimation_months)
    dates = _rebalance_dates(prices.index, first_valid)
    if not dates:
        raise StrategyError(
            f"price history is shorter than the estimation window "
            f"({config.estimation_months} months)"
        )

    window = pd.DateOffset(months=config.estimation_months)
    target_rows: dict[pd.Timestamp, pd.Series] = {}
    for date in dates:
        history = prices.loc[date - window : date]
        weights = strategy(date, history.copy())
        target_rows[date] = _validate(weights, prices.columns, date)

    targets = pd.DataFrame(target_rows).T.sort_index()

    # Hold the weights decided at the previous rebalance, applied from the next day.
    held = targets.reindex(asset_returns.index, method="ffill")
    held = held.loc[held.notna().all(axis=1)]
    active = asset_returns.loc[held.index]

    gross = (held * active).sum(axis=1)
    gross.name = None

    zero = pd.Series(0.0, index=gross.index)
    return BacktestResult(
        returns=gross,
        gross_returns=gross,
        weights=targets,
        held_weights=held,
        turnover=pd.Series(0.0, index=targets.index),
        costs=zero,
    )
