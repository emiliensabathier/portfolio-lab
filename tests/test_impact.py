import numpy as np
import pandas as pd
import pytest

from plab.backtest.engine import BacktestConfig, MarketImpact, StrategyError, run_backtest


def _noisy_prices(n_days: int = 400, seed: int = 0) -> pd.DataFrame:
    """Two assets with known daily volatility, so the impact charge can be recomputed."""
    index = pd.bdate_range("2015-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, [0.01, 0.02], size=(n_days, 2))
    return pd.DataFrame(100.0 * np.cumprod(1 + returns, axis=0), index=index, columns=["A", "B"])


def _volume(prices: pd.DataFrame, dollars: float = 1e8) -> pd.DataFrame:
    return pd.DataFrame(dollars, index=prices.index, columns=prices.columns)


def _alternating():
    state = {"flip": False}

    def strategy(date, history):
        state["flip"] = not state["flip"]
        return {"A": 1.0, "B": 0.0} if state["flip"] else {"A": 0.0, "B": 1.0}

    return strategy


CONFIG = BacktestConfig(estimation_months=6, cost_bps=5.0)


def test_impact_follows_the_square_root_law_on_each_traded_asset() -> None:
    prices = _noisy_prices()
    impact = MarketImpact(aum=1e7, dollar_volume=_volume(prices), coefficient=1.0)

    result = run_backtest(prices, _alternating(), CONFIG, impact=impact)

    day = result.costs.index[1]
    returns = prices.pct_change()
    sigma = returns.loc[:day].tail(impact.volatility_days).std()
    # A full switch sells all of one asset and buys all of the other: one unit each way.
    participation = 1.0 * 1e7 / 1e8
    expected = 2 * 5.0 / 10_000 + float((sigma * np.sqrt(participation)).sum())
    assert result.costs.loc[day] == pytest.approx(expected, rel=1e-9)


def test_impact_grows_with_the_square_root_of_assets_under_management() -> None:
    prices = _noisy_prices()
    small = run_backtest(
        prices, _alternating(), CONFIG, impact=MarketImpact(1e7, _volume(prices))
    )
    large = run_backtest(
        prices, _alternating(), CONFIG, impact=MarketImpact(4e7, _volume(prices))
    )

    spread = 2 * 5.0 / 10_000
    ratio = (large.costs.iloc[1:] - spread) / (small.costs.iloc[1:] - spread)
    assert np.allclose(ratio, 2.0)


def test_without_impact_the_charge_is_the_linear_spread_alone() -> None:
    prices = _noisy_prices()

    result = run_backtest(prices, _alternating(), CONFIG)

    assert result.costs.iloc[1] == pytest.approx(2 * 5.0 / 10_000)


def test_net_returns_still_reconcile_with_costs_under_impact() -> None:
    prices = _noisy_prices()
    impact = MarketImpact(aum=1e9, dollar_volume=_volume(prices))

    result = run_backtest(prices, _alternating(), CONFIG, impact=impact)

    difference = (result.gross_returns - result.returns).sum()
    assert difference == pytest.approx(result.costs.sum(), abs=1e-12)


def test_volume_on_the_rebalance_day_itself_is_not_used() -> None:
    prices = _noisy_prices()
    volume = _volume(prices)
    baseline = run_backtest(prices, _alternating(), CONFIG, impact=MarketImpact(1e9, volume))
    day = baseline.costs.index[1]
    spiked = volume.copy()
    spiked.loc[day:] = 1e15

    result = run_backtest(prices, _alternating(), CONFIG, impact=MarketImpact(1e9, spiked))

    assert result.costs.loc[day] == pytest.approx(baseline.costs.loc[day])


def test_missing_volume_before_a_trade_is_an_error() -> None:
    prices = _noisy_prices()
    volume = _volume(prices)
    volume["B"] = np.nan

    with pytest.raises(StrategyError, match="dollar volume"):
        run_backtest(prices, _alternating(), CONFIG, impact=MarketImpact(1e9, volume))


def test_volume_must_cover_every_asset() -> None:
    prices = _noisy_prices()

    with pytest.raises(StrategyError, match="dollar volume"):
        run_backtest(
            prices, _alternating(), CONFIG, impact=MarketImpact(1e9, _volume(prices)[["A"]])
        )


def test_assets_under_management_must_be_positive() -> None:
    prices = _noisy_prices()

    with pytest.raises(ValueError, match="aum"):
        MarketImpact(aum=0.0, dollar_volume=_volume(prices))
