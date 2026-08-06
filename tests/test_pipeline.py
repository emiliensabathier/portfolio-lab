"""Tests for the end-to-end pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plab.backtest.engine import BacktestConfig
from plab.pipeline import STRATEGIES, render, run
from plab.universe import ETF_CORE, tickers


def _prices(n_days: int = 1_800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = tickers(ETF_CORE)
    index = pd.bdate_range("2010-01-01", periods=n_days)
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


def test_rendering_produces_a_complete_document(output) -> None:
    html = render(output, generated_on="2026-08-06")

    assert html.startswith("<!doctype html>")
    assert "min_variance" in html
    assert "Beta vs 60/40" in html
    assert "Asset class attribution" in html
