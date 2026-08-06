"""Return conventions.

Simple (arithmetic) returns everywhere: a portfolio return is linear in weighted simple
returns, not in log returns. Log returns are produced only on explicit request.
"""

from __future__ import annotations

import pandas as pd


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Period-over-period simple returns. The first observation is dropped."""
    return prices.pct_change().iloc[1:]
