"""Stress testing.

Historical scenarios replay realized returns over a fixed window; they need no estimation
and therefore cover periods the out-of-sample backtest cannot reach, notably 2008.
"""

from __future__ import annotations

import pandas as pd

from plab.risk.metrics import max_drawdown

# Peak-to-trough windows on S&P 500 closing prices, each endpoint verified against SPY
# rather than recalled: 2007-10-09 (110.87) to 2009-03-09 (49.68), 2020-02-19 (308.40) to
# 2020-03-23 (204.42), and 2022-01-03 to the 2022-10-12 closing low (339.38). October 13th
# 2022 printed a lower intraday level but closed higher, and the 14th closed at 340.40 —
# above the 12th — so the closing-price convention used throughout puts the trough on the
# 12th.
HISTORICAL_SCENARIOS: dict[str, tuple[str, str]] = {
    "gfc_2008": ("2007-10-09", "2009-03-09"),
    "covid_2020": ("2020-02-19", "2020-03-23"),
    "rates_2022": ("2022-01-03", "2022-10-12"),
}


def replay(returns: pd.Series, scenario: str) -> dict[str, float]:
    """Statistics of ``returns`` restricted to a historical scenario window."""
    start, end = HISTORICAL_SCENARIOS[scenario]
    window = returns.loc[start:end]
    if window.empty:
        raise ValueError(f"the series does not overlap scenario {scenario} ({start}..{end})")
    return {
        "total_return": float((1.0 + window).prod() - 1.0),
        "max_drawdown": float(max_drawdown(window)),
        "worst_day": float(window.min()),
        "observations": float(len(window)),
    }
