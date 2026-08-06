"""Adjusted price loading with an on-disk parquet cache.

The fetcher is injected so the whole module is testable without network access.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from plab.errors import DataError


class Fetcher(Protocol):
    """Retrieves adjusted close prices for the given tickers."""

    def __call__(
        self, tickers: list[str], start: str, end: str | None
    ) -> pd.DataFrame: ...


def yfinance_fetcher(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Default fetcher. Returns adjusted close prices, one column per ticker."""
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column"
    )
    if raw.empty:
        raise DataError(f"yfinance returned no data for {tickers}")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    return close


def _cache_paths(cache_dir: Path, ticker: str) -> tuple[Path, Path]:
    return cache_dir / f"{ticker}.parquet", cache_dir / f"{ticker}.json"


def _read_cached(cache_dir: Path, ticker: str, start: str, end: str | None) -> pd.Series | None:
    data_path, meta_path = _cache_paths(cache_dir, ticker)
    if not (data_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("start") != start or meta.get("end") != end:
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
            _write_cache(cache_dir, ticker, series, start, end)
            cached[ticker] = series

    prices = pd.DataFrame({ticker: cached[ticker] for ticker in tickers})
    prices.index = pd.DatetimeIndex(prices.index, name="date")
    prices = prices.sort_index()

    if prices.empty:
        raise DataError(f"empty price frame for {tickers}")
    incomplete = prices.columns[prices.isna().any()].tolist()
    if incomplete:
        raise DataError(f"missing observations for {incomplete}; refusing to fill gaps")
    return prices
