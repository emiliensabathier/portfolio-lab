import numpy as np
import pandas as pd
import pytest

from plab.backtest.engine import BacktestConfig, StrategyError, run_backtest


def _flat_prices(n_days: int = 500) -> pd.DataFrame:
    """Constant prices: every return is zero, so costs are the only moving part."""
    index = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(100.0, index=index, columns=["A", "B"])


def _alternating_strategy():
    state = {"flip": False}

    def strategy(date, history):
        state["flip"] = not state["flip"]
        return {"A": 1.0, "B": 0.0} if state["flip"] else {"A": 0.0, "B": 1.0}

    return strategy


def test_a_strategy_that_never_trades_pays_nothing() -> None:
    def constant(date, history):
        return {"A": 0.5, "B": 0.5}

    result = run_backtest(_flat_prices(), constant, BacktestConfig(estimation_months=6))

    assert result.costs.iloc[1:].sum() == pytest.approx(0.0)
    assert result.turnover.iloc[1:].sum() == pytest.approx(0.0)


def test_full_switch_costs_twice_the_spread() -> None:
    config = BacktestConfig(estimation_months=6, cost_bps=5.0)

    result = run_backtest(_flat_prices(), _alternating_strategy(), config)

    # Every rebalance after the first moves the whole book: turnover 2.0, cost 10 bps.
    assert result.turnover.iloc[1] == pytest.approx(2.0)
    assert result.costs.loc[result.costs.index[1]] == pytest.approx(2.0 * 5.0 / 10_000)


def test_net_returns_are_gross_minus_costs() -> None:
    config = BacktestConfig(estimation_months=6, cost_bps=5.0)

    result = run_backtest(_flat_prices(), _alternating_strategy(), config)

    difference = (result.gross_returns - result.returns).sum()
    assert difference == pytest.approx(result.costs.sum(), abs=1e-12)


def _drifting_prices(n_days: int = 500) -> pd.DataFrame:
    """A rises steadily while B stays flat, so held weights drift away from any target."""
    index = pd.bdate_range("2015-01-01", periods=n_days)
    rising = 100.0 * np.cumprod(np.full(n_days, 1.001))
    return pd.DataFrame({"A": rising, "B": np.full(n_days, 100.0)}, index=index)


def test_no_trade_band_suppresses_a_rebalance_smaller_than_the_band() -> None:
    # The fixture must genuinely drift. Under flat prices the held weights never leave the
    # target, so `traded` is already 0.0 before the band is consulted — a test built that
    # way passes even with the band logic deleted outright, which is what this replaces.
    prices = _drifting_prices()

    def constant(date, history):
        return {"A": 0.5, "B": 0.5}

    unbanded = run_backtest(
        prices, constant, BacktestConfig(estimation_months=6, no_trade_band=0.0)
    )
    banded = run_backtest(
        prices, constant, BacktestConfig(estimation_months=6, no_trade_band=0.5)
    )

    # Monthly drift is worth about 0.011 of turnover per rebalance: a 0.5 band swallows
    # every one, a zero band trades on every one.
    assert unbanded.turnover.iloc[1:].sum() == pytest.approx(0.1739, rel=0.01)
    assert banded.turnover.iloc[1:].sum() == pytest.approx(0.0)


def test_a_history_ending_on_the_first_rebalance_raises() -> None:
    # 152 business days from 2015-01-01 land exactly on 2015-07-31, the first rebalance
    # after a six-month estimation window. No period follows it, so the initial purchase
    # charge has nothing to attach to and must be refused rather than dropped.
    prices = _flat_prices(152)

    def constant(date, history):
        return {"A": 0.5, "B": 0.5}

    with pytest.raises(StrategyError, match="ends on the first rebalance date"):
        run_backtest(prices, constant, BacktestConfig(estimation_months=6))


def test_held_weights_are_daily_and_aligned_with_the_return_series() -> None:
    def constant(date, history):
        return {"A": 0.5, "B": 0.5}

    result = run_backtest(_flat_prices(), constant, BacktestConfig(estimation_months=6))

    pd.testing.assert_index_equal(result.held_weights.index, result.returns.index)
    assert result.held_weights.sum(axis=1).round(9).eq(1.0).all()


def test_costs_reduce_performance_on_a_real_looking_series() -> None:
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2015-01-01", periods=800)
    levels = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, (800, 2)), axis=0))
    prices = pd.DataFrame(levels, index=index, columns=["A", "B"])

    free = run_backtest(prices, _alternating_strategy(),
                        BacktestConfig(estimation_months=6, cost_bps=0.0))
    charged = run_backtest(prices, _alternating_strategy(),
                           BacktestConfig(estimation_months=6, cost_bps=25.0))

    assert charged.returns.sum() < free.returns.sum()
