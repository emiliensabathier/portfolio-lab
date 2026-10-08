"""Monthly-rebalanced portfolio backtest.

The engine is the single seam between data and strategies. It hands a strategy a copy of
the trailing estimation window ending at the decision date and nothing else, so a strategy
cannot see the future even if its author tries. Weights decided on date ``d`` are applied
from the following trading day.

Transaction costs: turnover at a rebalance is the two-way distance between the drifted
weights held going in and the newly decided target, ``sum(|w_target - w_drifted|)``. The
charge is ``turnover * cost_bps / 10_000``, incurred at the close of the rebalance day and
debited from the *next* period's return — including the initial purchase. ``returns`` is
therefore always net of costs; ``gross_returns`` is what the same strategy would have earned
free of any friction.

Market impact, optional: a book of ``aum`` dollars trading a fraction ``t`` of itself in an
asset pays, on top of the spread, the square-root law ``Y * sigma * sqrt(t * aum / ADV)`` on
that trade (Toth et al. 2011; Bouchaud et al. 2018, ch. 12), where ``sigma`` is the asset's trailing
daily volatility and ``ADV`` its average daily dollar volume. Volume is averaged over the
sessions *before* the rebalance day, never including it.
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
class MarketImpact:
    """Square-root impact for a book of ``aum`` dollars.

    ``dollar_volume`` is each asset's daily traded value, on the price calendar.
    """

    aum: float
    dollar_volume: pd.DataFrame
    coefficient: float = 1.0
    volume_days: int = 20
    volatility_days: int = 60

    def __post_init__(self) -> None:
        if not self.aum > 0:
            raise ValueError(f"aum must be positive, got {self.aum}")


def _impact_charge(
    traded: pd.Series,
    day: pd.Timestamp,
    asset_returns: pd.DataFrame,
    impact: MarketImpact,
) -> float:
    """Square-root impact on the per-asset trade ``traded`` decided at the close of ``day``."""
    volume = impact.dollar_volume.reindex(columns=traded.index)
    adv = volume.loc[volume.index < day].tail(impact.volume_days).mean()
    active = traded > 0
    if not np.isfinite(adv[active]).all() or (adv[active] <= 0).any():
        raise StrategyError(f"no usable dollar volume before {day.date()} for a traded asset")
    sigma = asset_returns.loc[:day].tail(impact.volatility_days).std()
    participation = traded[active] * impact.aum / adv[active]
    per_asset = traded[active] * impact.coefficient * sigma[active] * np.sqrt(participation)
    return float(per_asset.sum())


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
    prices: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig,
    impact: MarketImpact | None = None,
) -> BacktestResult:
    """Run ``strategy`` over ``prices`` with monthly rebalancing, net of costs."""
    prices = prices.sort_index()
    if impact is not None and set(prices.columns) - set(impact.dollar_volume.columns):
        missing = sorted(set(prices.columns) - set(impact.dollar_volume.columns))
        raise StrategyError(f"no dollar volume supplied for {missing}")
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

        per_asset = (target - held).abs() if started else target.abs()
        traded = float(per_asset.sum())
        if started and traded < config.no_trade_band:
            traded = 0.0
            target = held
            per_asset = per_asset * 0.0

        charge = traded * config.cost_bps / 10_000.0
        if impact is not None and traded > 0:
            charge += _impact_charge(per_asset, day, asset_returns, impact)
        targets[day] = target
        turnover[day] = traded
        costs[day] = charge
        pending += charge

        held = target
        started = True

    if not net_rows:
        # The history ends on the first rebalance date, so no period follows it and the
        # charge has nothing to attach to. Dropping it silently would leave costs.sum()
        # irreconcilable with gross - net, which is precisely the invariant this module
        # promises.
        raise StrategyError(
            "the price history ends on the first rebalance date, so no return period "
            "follows it and the transaction cost cannot be charged"
        )

    if pending:
        # A rebalance on the final day has no following period; charge it to the last one
        # so that reported costs always reconcile with the net series. `pending` is either
        # exactly 0.0 or a real accumulated charge, never floating-point noise, so plain
        # truthiness is safe here despite the package's usual NUMERICAL_ZERO convention.
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
