from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import plab.__main__ as main_module
from plab.universe import ETF_CORE, tickers


def _synthetic_prices() -> pd.DataFrame:
    """A cheap synthetic price frame with the real universe's ten tickers.

    Spans 2007-2022 so it overlaps every historical stress scenario (the pipeline
    replays them against the raw asset returns, not the out-of-sample backtest), while
    still being a handful of random-walk columns rather than the real ten-year download.
    """
    index = pd.bdate_range("2007-01-01", "2022-12-30")
    rng = np.random.default_rng(0)
    levels = 100.0 * np.exp(
        np.cumsum(rng.normal(0.0002, 0.01, (len(index), len(ETF_CORE))), axis=0)
    )
    return pd.DataFrame(levels, index=index, columns=tickers(ETF_CORE))


def _synthetic_risk_free(index: pd.DatetimeIndex) -> pd.Series:
    """A flat two percent bill on the same calendar as the synthetic prices.

    The level is arbitrary: the point is that the series covers the backtest window, so
    the Sharpe denominator is exercised without the test reaching for a live quote.
    """
    return pd.Series(0.02, index=index, name="risk_free")


def test_main_writes_an_html_report_to_the_requested_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # __main__ is the only documented way a reader runs the project, yet nothing exercised
    # it: it was 0% covered. load_prices and load_risk_free are both monkeypatched so the
    # test needs neither network access nor the real ten-year price history. Patching only
    # the prices left the bill series reaching for a live quote, which then failed to align
    # with the synthetic 2007-2022 window — a network test wearing an offline test's badge.
    prices = _synthetic_prices()
    monkeypatch.setattr(main_module, "load_prices", lambda *args, **kwargs: prices)
    monkeypatch.setattr(
        main_module, "load_risk_free", lambda *args, **kwargs: _synthetic_risk_free(prices.index)
    )
    output = tmp_path / "report.html"
    # A 180-month (15-year) estimation window leaves only ~12 monthly rebalances in the
    # last year of the 16-year synthetic history, keeping the SLSQP-backed strategies
    # (min_variance, risk_parity, max_sharpe) fast, while the full history still overlaps
    # every historical stress scenario the pipeline replays.
    monkeypatch.setattr(
        "sys.argv", ["plab", "--output", str(output), "--estimation-months", "180"]
    )

    main_module.main()

    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_main_passes_the_end_date_to_both_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --end exists so a run can stop short of a bad tail upstream. It is only useful if it
    # reaches the bill series too: cutting the prices alone would leave the two loaded over
    # different windows, which is exactly the misalignment the flag is meant to avoid.
    prices = _synthetic_prices()
    seen: dict[str, str | None] = {}

    def _prices(*args, **kwargs):
        seen["prices"] = kwargs.get("end")
        return prices

    def _risk_free(*args, **kwargs):
        seen["risk_free"] = kwargs.get("end")
        return _synthetic_risk_free(prices.index)

    monkeypatch.setattr(main_module, "load_prices", _prices)
    monkeypatch.setattr(main_module, "load_risk_free", _risk_free)
    monkeypatch.setattr(
        "sys.argv",
        [
            "plab",
            "--output",
            str(tmp_path / "report.html"),
            "--estimation-months",
            "180",
            "--end",
            "2022-12-30",
        ],
    )

    main_module.main()

    assert seen == {"prices": "2022-12-30", "risk_free": "2022-12-30"}
