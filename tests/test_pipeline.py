"""Tests for the end-to-end pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plab.backtest.engine import BacktestConfig
from plab.pipeline import STRATEGIES, render, run
from plab.risk.stress import HISTORICAL_SCENARIOS
from plab.universe import ETF_CORE, tickers


def _prices(n_days: int = 4_200, seed: int = 0) -> pd.DataFrame:
    """Synthetic prices spanning every stress window.

    Starting in 2007 is deliberate. A fixture beginning in 2010 leaves all three
    historical scenarios outside the data, so the stress branch never runs under test —
    which is exactly how a report shipped claiming 2008 coverage it did not have.
    """
    rng = np.random.default_rng(seed)
    names = tickers(ETF_CORE)
    index = pd.bdate_range("2007-07-02", periods=n_days)
    steps = rng.normal(0.0003, 0.009, (n_days, len(names)))
    levels = 100.0 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(levels, index=index, columns=names)


@pytest.fixture(scope="module")
def output():
    """Run the pipeline once: it drives several hundred SLSQP optimizations."""
    return run(_prices(), BacktestConfig(estimation_months=12))


def test_every_declared_strategy_is_backtested(output) -> None:
    assert set(output.results) == set(STRATEGIES)


def test_the_sixty_forty_benchmark_only_holds_spy_and_agg(output) -> None:
    weights = output.results["60/40"].weights.iloc[-1]
    assert weights["SPY"] == pytest.approx(0.6)
    assert weights["AGG"] == pytest.approx(0.4)
    assert weights.drop(["SPY", "AGG"]).sum() == pytest.approx(0.0)


def test_risk_shares_are_produced_for_every_strategy_and_sum_to_one(output) -> None:
    for name, shares in output.risk_shares.items():
        assert shares.sum() == pytest.approx(1.0), name


def test_asset_class_attribution_covers_the_three_blocs(output) -> None:
    frame = output.bloc_attribution["equal_weight"]
    assert set(frame.index) == {"equity", "rates", "real"}
    assert frame["risk_share"].sum() == pytest.approx(1.0)


def test_every_strategy_is_stressed_through_all_three_scenarios(output) -> None:
    # Covering 2008 is the reason this section exists: the out-of-sample backtest starts
    # in 2010 and cannot reach the crisis. A scenario quietly missing here would leave the
    # report asserting coverage it does not have — which is precisely what happened once.
    for name, scenarios in output.stress.items():
        assert set(scenarios) == set(HISTORICAL_SCENARIOS), name
        assert scenarios["gfc_2008"]["observations"] > 0, name


def test_rendering_produces_a_complete_document(output) -> None:
    html = render(output, generated_on="2026-08-06")

    assert html.startswith("<!doctype html>")
    assert "min_variance" in html
    assert "Beta vs 60/40" in html
    assert "Asset class attribution" in html
    assert "gfc_2008" in html
