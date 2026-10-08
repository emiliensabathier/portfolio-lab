"""Adjusted price loading with an on-disk parquet cache.

The fetcher is injected so the whole module is testable without network access.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

from plab.errors import DataError

# A cache entry older than this is treated as a miss and refetched, even if its
# (start, end) key still matches. Without this, a run with end=None (the CLI's default)
# never changes its cache key, so a stale cache — one written before newer trading days
# were available upstream — would be reused indefinitely and the report header would
# silently print a stale "as of" date.
CACHE_MAX_AGE = timedelta(days=1)


class Fetcher(Protocol):
    """Retrieves adjusted close prices for the given tickers."""

    def __call__(self, tickers: list[str], start: str, end: str | None) -> pd.DataFrame: ...


def yfinance_fetcher(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Default fetcher. Returns adjusted close prices, one column per ticker.

    ``end`` is inclusive, as everywhere else in the package. Yahoo stops the day before
    the ``end`` it is given, so the date is moved forward one day before it is passed on.
    """
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=_exclusive(end),
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise DataError(f"yfinance returned no data for {tickers}")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    return close


def yfinance_dollar_volume(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Daily traded value in dollars, one column per ticker: unadjusted close times shares.

    Adjusted closes are the wrong price here: they scale past sessions by later dividends
    and so understate what was actually traded. A session with zero shares is returned as
    missing, so that a market-impact model treats it as unknown rather than as free.
    """
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=_exclusive(end),
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise DataError(f"yfinance returned no volume for {tickers}")
    shares = raw["Volume"].replace(0, float("nan"))
    value = raw["Close"] * shares
    value.index = pd.DatetimeIndex(value.index).tz_localize(None)
    return value[tickers]


def _exclusive(end: str | None) -> str | None:
    """Yahoo stops the day before the ``end`` it is given; the package's ``end`` is inclusive."""
    return None if end is None else (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()


def _cache_paths(cache_dir: Path, ticker: str) -> tuple[Path, Path]:
    return cache_dir / f"{ticker}.parquet", cache_dir / f"{ticker}.json"


def _read_cached(cache_dir: Path, ticker: str, start: str, end: str | None) -> pd.Series | None:
    data_path, meta_path = _cache_paths(cache_dir, ticker)
    if not (data_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("start") != start or meta.get("end") != end:
        return None
    fetched_at = meta.get("fetched_at")
    if fetched_at is None or datetime.now(UTC) - datetime.fromisoformat(fetched_at) > CACHE_MAX_AGE:
        return None
    return pd.read_parquet(data_path)[ticker]


def _write_cache(
    cache_dir: Path, ticker: str, series: pd.Series, start: str, end: str | None
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = _cache_paths(cache_dir, ticker)
    series.to_frame(ticker).to_parquet(data_path)
    meta_path.write_text(
        json.dumps(
            {"start": start, "end": end, "fetched_at": datetime.now(UTC).isoformat()},
            indent=2,
        ),
        encoding="utf-8",
    )


def load_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    *,
    cache_dir: Path,
    refresh: bool = False,
    fetcher: Fetcher | None = None,
) -> pd.DataFrame:
    """Load adjusted close prices for ``tickers`` between ``start`` and ``end``.

    Raises DataError when a ticker is unavailable or its series contains gaps. Data is
    never forward-filled: an invented price becomes an invented return.
    """
    fetch = fetcher if fetcher is not None else yfinance_fetcher
    cache_dir = Path(cache_dir)

    cached: dict[str, pd.Series] = {}
    to_fetch: list[str] = []
    for ticker in tickers:
        series = None if refresh else _read_cached(cache_dir, ticker, start, end)
        if series is None:
            to_fetch.append(ticker)
        else:
            cached[ticker] = series

    if to_fetch:
        fetched = fetch(to_fetch, start, end)
        for ticker in to_fetch:
            if ticker not in fetched.columns:
                raise DataError(f"no price series returned for {ticker}")
            series = fetched[ticker]
            # Validate before caching. Persisting a gappy series would make a transient
            # upstream outage permanent: every later run reads the poisoned cache and
            # fails identically, with no way for the caller to know refresh=True is needed.
            if series.isna().any():
                gaps = series.index[series.isna()]
                raise DataError(
                    f"missing observations for {ticker} on {len(gaps)} session(s), "
                    f"{gaps[0].date()} to {gaps[-1].date()}; refusing to cache or fill gaps. "
                    f"Pass --end before {gaps[0].date()} to run on the intact history."
                )
            _write_cache(cache_dir, ticker, series, start, end)
            cached[ticker] = series

    prices = pd.DataFrame({ticker: cached[ticker] for ticker in tickers})
    prices.index = pd.DatetimeIndex(prices.index, name="date")
    prices = prices.sort_index()

    if prices.empty:
        raise DataError(f"empty price frame for {tickers}")
    # Second gap check, and not a duplicate of the per-ticker one above: this one catches
    # NaN introduced by aligning tickers whose trading calendars differ, and gaps in
    # series that came from the cache rather than the fetcher.
    incomplete = prices.columns[prices.isna().any()].tolist()
    if incomplete:
        # The dates are named because the caller's only remedy is to ask for a shorter
        # window, and it cannot choose one without knowing where the panel breaks.
        gaps = prices.index[prices.isna().any(axis=1)]
        raise DataError(
            f"missing observations for {incomplete} on {len(gaps)} session(s), "
            f"{gaps[0].date()} to {gaps[-1].date()}; refusing to fill gaps. "
            f"Pass --end before {gaps[0].date()} to run on the intact history."
        )
    return prices
