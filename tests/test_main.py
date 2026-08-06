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


def test_main_writes_an_html_report_to_the_requested_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # __main__ is the only documented way a reader runs the project, yet nothing exercised
    # it: it was 0% covered. load_prices is monkeypatched so the test needs neither network
    # access nor the real ten-year price history, and a one-month estimation window keeps
    # the backtest (and its SLSQP optimizers) cheap.
    monkeypatch.setattr(
        main_module, "load_prices", lambda *args, **kwargs: _synthetic_prices()
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
