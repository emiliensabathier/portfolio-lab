"""The riskless leg: what cash actually paid, day by day.

Every Sharpe and Sortino in this repository used to be measured against zero. Over a
window that includes 2022-2025, when three-month bills paid close to five percent, that
is not a risk-adjusted number — it flatters every strategy by roughly the whole level of
short rates, and it flatters the high-volatility ones most.

The series here is the three-month Treasury bill, quoted the way the market quotes it
and converted once, here, to the basis the ratios need.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from plab.data.loader import Fetcher, load_prices
from plab.errors import DataError

# The three-month bill on the secondary market, quoted as a discount rate in percent —
# the same quantity the CBOE ^IRX index carries, taken from FRED instead.
#
# ^IRX was the original source and had to go: Yahoo serves it with roughly one month of
# history, so a backtest starting in 2007 either fails outright or, worse, aligns against
# a window that silently starts last month. FRED serves DTB3 daily back to 1954, which is
# the whole point of a riskless leg that has to cover the sample.
CASH_TICKER = "DTB3"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

PERCENT = 100.0
BILL_DAYS = 91
DISCOUNT_BASIS = 360
YEAR_DAYS = 365

# A bill quote can be stale across a holiday the equity market did not observe. Carrying
# a rate forward one or two sessions is accrual, not invention: the bill kept paying. A
# longer gap is a data problem and is refused, in the same spirit as the price loader.
MAX_STALE_SESSIONS = 3


def bond_equivalent_yield(discount: pd.Series) -> pd.Series:
    """Convert a bank-discount quote to the bond-equivalent yield.

    Bills are quoted on a discount basis against face value over a 360-day year. A Sharpe
    denominator wants the yield an investor actually earns on the price paid, over a
    365-day year. At a 3.7% discount quote the two differ by about 9 bp — small next to a
    Sharpe of one, and free to remove, which is why it is removed rather than disclosed.
    """
    return YEAR_DAYS * discount / (DISCOUNT_BASIS - BILL_DAYS * discount)


def fred_fetcher(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Fetcher for FRED daily series, one series per call.

    FRED publishes a two-column CSV — ``observation_date`` and the series id — and marks a
    day with no observation as ``.`` rather than omitting it. Those rows are dropped, not
    filled: a holiday has no quote, and ``align`` already carries the last rate across a
    short gap, which is accrual rather than invention.
    """
    if len(tickers) != 1:
        raise DataError(f"the FRED fetcher takes one series at a time, got {tickers}")
    series_id = tickers[0]
    frame = pd.read_csv(FRED_CSV.format(series=series_id), index_col=0, parse_dates=[0])
    series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()
    series.index = pd.DatetimeIndex(series.index)
    # FRED serves the full history and ignores date parameters on this endpoint, so the
    # requested window is applied here rather than upstream.
    series = series.loc[start:end]
    if series.empty:
        raise DataError(f"FRED returned no observations for {series_id} over {start}..{end}")
    return series.to_frame(series_id)


def load_risk_free(
    start: str,
    end: str | None = None,
    *,
    cache_dir: Path,
    refresh: bool = False,
    fetcher: Fetcher | None = None,
) -> pd.Series:
    """Load the bill series as an annualized decimal bond-equivalent yield."""
    frame = load_prices(
        [CASH_TICKER],
        start,
        end,
        cache_dir=cache_dir,
        refresh=refresh,
        fetcher=fetcher if fetcher is not None else fred_fetcher,
    )
    return bond_equivalent_yield(frame[CASH_TICKER] / PERCENT).rename("risk_free")


def align(risk_free: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Put the bill series on the portfolio's own trading calendar.

    Refuses rather than filling when the gap is longer than a long weekend, and refuses
    outright when the requested window starts before the series does — extending a rate
    backwards would invent the one number the ratio is measured against.
    """
    aligned = risk_free.reindex(risk_free.index.union(index)).ffill(
        limit=MAX_STALE_SESSIONS
    ).reindex(index)
    missing = aligned.isna()
    if missing.any():
        first, last = index[missing][0], index[missing][-1]
        raise DataError(
            f"{int(missing.sum())} of {len(index)} portfolio dates have no bill quote "
            f"within {MAX_STALE_SESSIONS} sessions, from {first.date()} to {last.date()}"
        )
    return aligned.rename("risk_free")
