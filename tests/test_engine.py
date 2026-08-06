import numpy as np
import pandas as pd
import pytest

from plab.backtest.engine import BacktestConfig, StrategyError, run_backtest


def _prices(n_days: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n_days)
    steps = rng.normal(0.0004, 0.01, (n_days, 2))
    levels = 100.0 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(levels, index=index, columns=["A", "B"])


def equal_weight(date, history):
    return dict.fromkeys(history.columns, 1.0 / len(history.columns))


def test_engine_never_shows_a_strategy_data_beyond_the_decision_date() -> None:
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def spy(date, history):
        seen.append((date, history.index.max()))
        return equal_weight(date, history)

    run_backtest(_prices(), spy, BacktestConfig(estimation_months=6))

    assert seen, "the strategy was never called"
    for decision_date, last_observation in seen:
        assert last_observation <= decision_date


def test_engine_passes_exactly_the_estimation_window() -> None:
    widths: list[int] = []

    def measure(date, history):
        widths.append(len(history))
        return equal_weight(date, history)

    run_backtest(_prices(), measure, BacktestConfig(estimation_months=6))

    # 6 months of business days, allowing for holidays.
    assert all(115 <= width <= 135 for width in widths), widths


def test_strategy_cannot_mutate_the_history_it_receives() -> None:
    def vandal(date, history):
        history.iloc[0, 0] = -999.0
        return equal_weight(date, history)

    prices = _prices()
    original = prices.copy()

    run_backtest(prices, vandal, BacktestConfig(estimation_months=6))

    pd.testing.assert_frame_equal(prices, original)


def test_backtest_starts_after_the_estimation_window() -> None:
    prices = _prices()

    result = run_backtest(prices, equal_weight, BacktestConfig(estimation_months=6))

    assert result.returns.index.min() > prices.index.min() + pd.DateOffset(months=6)
    assert result.returns.index.max() <= prices.index.max()


def test_buy_and_hold_of_a_single_asset_reproduces_that_asset_return() -> None:
    prices = _prices()

    def all_in_a(date, history):
        return {"A": 1.0, "B": 0.0}

    result = run_backtest(
        prices, all_in_a, BacktestConfig(estimation_months=6, cost_bps=0.0)
    )
    asset = prices["A"].pct_change().loc[result.returns.index]

    pd.testing.assert_series_equal(result.returns, asset, check_names=False, atol=1e-12)


def test_weights_that_do_not_sum_to_one_are_rejected() -> None:
    def broken(date, history):
        return {"A": 0.5, "B": 0.2}

    with pytest.raises(StrategyError, match="sum"):
        run_backtest(_prices(), broken, BacktestConfig(estimation_months=6))


def test_unknown_ticker_in_weights_is_rejected() -> None:
    def broken(date, history):
        return {"A": 0.5, "GHOST": 0.5}

    with pytest.raises(StrategyError, match="GHOST"):
        run_backtest(_prices(), broken, BacktestConfig(estimation_months=6))


def test_history_shorter_than_the_estimation_window_raises() -> None:
    with pytest.raises(StrategyError, match="estimation window"):
        run_backtest(_prices(n_days=20), equal_weight, BacktestConfig(estimation_months=36))
