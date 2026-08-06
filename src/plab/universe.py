"""Investable universes.

Single source of truth for which assets exist and from when. Availability is a declared
rule, never inferred by filling absent prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Asset:
    """One investable instrument."""

    ticker: str
    name: str
    bloc: str  # "equity" | "rates" | "real"
    inception: str  # first reliable trading date, ISO format


ETF_CORE: tuple[Asset, ...] = (
    Asset("SPY", "US large cap equity", "equity", "1993-01-29"),
    Asset("IWM", "US small cap equity", "equity", "2000-05-26"),
    Asset("EFA", "Developed ex-US equity", "equity", "2001-08-17"),
    Asset("EEM", "Emerging market equity", "equity", "2003-04-14"),
    Asset("AGG", "US aggregate bonds", "rates", "2003-09-26"),
    Asset("TLT", "US long treasuries", "rates", "2002-07-30"),
    Asset("HYG", "US high yield credit", "rates", "2007-04-11"),
    Asset("GLD", "Gold", "real", "2004-11-18"),
    Asset("VNQ", "US REITs", "real", "2004-09-29"),
    Asset("DBC", "Broad commodities", "real", "2006-02-03"),
)

# HYG, listed 2007-04-11, is the binding constraint on the common history.
CORE_START = "2007-07-02"


def tickers(universe: tuple[Asset, ...]) -> list[str]:
    """Ticker symbols in declaration order."""
    return [asset.ticker for asset in universe]


def available_at(universe: tuple[Asset, ...], date: str | pd.Timestamp) -> list[str]:
    """Tickers already listed at ``date``."""
    stamp = pd.Timestamp(date)
    return [asset.ticker for asset in universe if pd.Timestamp(asset.inception) <= stamp]
