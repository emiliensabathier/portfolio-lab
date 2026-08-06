"""Monthly-rebalanced portfolio backtest.

The engine is the single seam between data and strategies. It hands a strategy a copy of
the trailing estimation window ending at the decision date and nothing else, so a strategy
cannot see the future even if its author tries. Weights decided on date ``d`` are applied
from the following trading day.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from plab.returns import simple_returns
from plab.risk.metrics import NUMERICAL_ZERO

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
    """Turn a strategy's weight dict into a validated Series over ``columns``.

    A ticker the strategy omits is taken as an explicit zero. A ticker it names with a
    non-finite value is an error: ``abs(nan - 1.0) > tolerance`` is False, so a NaN weight
    would otherwise slip through the sum check and propagate into every later result.
    """
    unknown = set(weights) - set(columns)
    if unknown:
        raise StrategyError(f"strategy returned unknown tickers {sorted(unknown)} at {date}")
    provided = pd.Series(weights, dtype=float)
    if not np.isfinite(provided.to_numpy()).all():
        raise StrategyError(f"strategy returned non-finite weights at {date}")
    series = provided.reindex(columns).fillna(0.0)
    total = float(series.sum())
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise StrategyError(f"weights at {date} sum to {total:.6f}, expected 1.0")
    return series


def _drifted_weights(previous: pd.Series, period_returns: pd.Series) -> pd.Series:
    """Weights after one period of market drift, before any rebalancing."""
    grown = previous * (1.0 + period_returns)
    total = float(grown.sum())
    if abs(total) <= NUMERICAL_ZERO:
        raise StrategyError("portfolio value collapsed to zero")
    return grown / total


def run_backtest(
    prices: pd.DataFrame, strategy: Strategy, config: BacktestConfig
) -> BacktestResult:
    """Run ``strategy`` over ``prices`` with monthly rebalancing, net of costs."""
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
    rebalance_days = set(dates)
    targets: dict[pd.Timestamp, pd.Series] = {}
    turnover: dict[pd.Timestamp, float] = {}
    costs: dict[pd.Timestamp, float] = {}

    held = pd.Series(0.0, index=prices.columns)
    held_rows: dict[pd.Timestamp, pd.Series] = {}
    gross_rows: dict[pd.Timestamp, float] = {}
    net_rows: dict[pd.Timestamp, float] = {}
    started = False
    # Costs are incurred at the close of the rebalance day and charged to the next
    # period's return, including the initial purchase.
    pending = 0.0

    for day in asset_returns.index:
        if started:
            period = asset_returns.loc[day]
            held_rows[day] = held.copy()
            gross_rows[day] = float((held * period).sum())
            net_rows[day] = gross_rows[day] - pending
            pending = 0.0
            held = _drifted_weights(held, period)

        if day not in rebalance_days:
            continue

        history = prices.loc[day - window : day]
        target = _validate(strategy(day, history.copy()), prices.columns, day)

        traded = float((target - held).abs().sum()) if started else float(target.abs().sum())
        if started and traded < config.no_trade_band:
            traded = 0.0
            target = held

        charge = traded * config.cost_bps / 10_000.0
        targets[day] = target
        turnover[day] = traded
        costs[day] = charge
        pending += charge

        held = target
        started = True

    if pending and net_rows:
        # A rebalance on the final day has no following period; charge it to the last one
        # so that reported costs always reconcile with the net series.
        last = max(net_rows)
        net_rows[last] -= pending

    net = pd.Series(net_rows).sort_index()
    gross = pd.Series(gross_rows).sort_index()
    return BacktestResult(
        returns=net,
        gross_returns=gross,
        weights=pd.DataFrame(targets).T.sort_index(),
        held_weights=pd.DataFrame(held_rows).T.sort_index(),
        turnover=pd.Series(turnover).sort_index(),
        costs=pd.Series(costs).sort_index(),
    )
