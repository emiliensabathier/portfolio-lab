import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from plab.data.loader import load_prices
from plab.errors import DataError


def _frame(dates: list[str], data: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="date"))


class RecordingFetcher:
    """Fake fetcher: returns canned prices and counts how often it is called."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls = 0

    def __call__(self, tickers, start, end):
        self.calls += 1
        # A real fetcher returns the series it has and simply omits the rest; it does
        # not raise on an unknown ticker. Selecting an absent column with .loc would
        # raise KeyError here and pre-empt the loader's own DataError guard.
        available = [ticker for ticker in tickers if ticker in self.frame.columns]
        return self.frame.loc[start:end, available]


def test_load_prices_returns_requested_tickers(tmp_path: Path) -> None:
    fetcher = RecordingFetcher(
        _frame(["2020-01-02", "2020-01-03"], {"SPY": [100.0, 101.0], "AGG": [50.0, 50.5]})
    )

    prices = load_prices(["SPY", "AGG"], "2020-01-01", "2020-01-31",
                         cache_dir=tmp_path, fetcher=fetcher)

    assert list(prices.columns) == ["SPY", "AGG"]
    assert prices.loc["2020-01-03", "SPY"] == 101.0


def test_second_call_uses_cache(tmp_path: Path) -> None:
    fetcher = RecordingFetcher(
        _frame(["2020-01-02", "2020-01-03"], {"SPY": [100.0, 101.0]})
    )

    load_prices(["SPY"], "2020-01-01", "2020-01-31", cache_dir=tmp_path, fetcher=fetcher)
    load_prices(["SPY"], "2020-01-01", "2020-01-31", cache_dir=tmp_path, fetcher=fetcher)

    assert fetcher.calls == 1


def test_refresh_bypasses_cache(tmp_path: Path) -> None:
    fetcher = RecordingFetcher(
        _frame(["2020-01-02", "2020-01-03"], {"SPY": [100.0, 101.0]})
    )

    load_prices(["SPY"], "2020-01-01", "2020-01-31", cache_dir=tmp_path, fetcher=fetcher)
    load_prices(["SPY"], "2020-01-01", "2020-01-31", cache_dir=tmp_path,
                refresh=True, fetcher=fetcher)

    assert fetcher.calls == 2


def test_missing_ticker_raises(tmp_path: Path) -> None:
    fetcher = RecordingFetcher(_frame(["2020-01-02"], {"SPY": [100.0]}))

    with pytest.raises(DataError, match="GHOST"):
        load_prices(["GHOST"], "2020-01-01", "2020-01-31",
                    cache_dir=tmp_path, fetcher=fetcher)


def test_gap_in_series_raises_instead_of_filling(tmp_path: Path) -> None:
    fetcher = RecordingFetcher(
        _frame(["2020-01-02", "2020-01-03"], {"SPY": [100.0, float("nan")]})
    )

    with pytest.raises(DataError, match="missing"):
        load_prices(["SPY"], "2020-01-01", "2020-01-31",
                    cache_dir=tmp_path, fetcher=fetcher)


def test_a_stale_cache_entry_triggers_a_refetch(tmp_path: Path) -> None:
    fetcher = RecordingFetcher(
        _frame(["2020-01-02", "2020-01-03"], {"SPY": [100.0, 101.0]})
    )
    load_prices(["SPY"], "2020-01-01", "2020-01-31", cache_dir=tmp_path, fetcher=fetcher)

    # Hand-write a sidecar whose fetched_at is well past the staleness threshold, as if
    # the cache had been written months ago and never invalidated.
    meta_path = tmp_path / "SPY.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fetched_at"] = (datetime.now(UTC) - timedelta(days=400)).isoformat()
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    load_prices(["SPY"], "2020-01-01", "2020-01-31", cache_dir=tmp_path, fetcher=fetcher)

    assert fetcher.calls == 2


def test_a_gappy_series_is_never_written_to_the_cache(tmp_path: Path) -> None:
    fetcher = RecordingFetcher(
        _frame(["2020-01-02", "2020-01-03"], {"SPY": [100.0, float("nan")]})
    )

    with pytest.raises(DataError):
        load_prices(["SPY"], "2020-01-01", "2020-01-31",
                    cache_dir=tmp_path, fetcher=fetcher)

    # Caching an incomplete series would make a transient outage permanent: the next
    # run would read the poisoned cache and fail identically, with no signal that
    # refresh=True is what unblocks it.
    assert list(tmp_path.glob("SPY.*")) == []


def test_the_gap_error_names_the_dates_and_the_remedy(tmp_path: Path) -> None:
    """The only remedy is a shorter window, and the caller cannot pick one without being
    told where the panel breaks. Upstream holes are real: Yahoo served July 2026 with a
    session missing for AGG and a different one missing for VNQ."""
    fetcher = RecordingFetcher(
        _frame(
            ["2020-01-02", "2020-01-03", "2020-01-06"],
            {"SPY": [100.0, float("nan"), 101.0]},
        )
    )

    with pytest.raises(DataError) as excinfo:
        load_prices(["SPY"], "2020-01-01", "2020-01-31",
                    cache_dir=tmp_path, fetcher=fetcher)

    message = str(excinfo.value)
    assert "2020-01-03" in message
    assert "--end" in message
