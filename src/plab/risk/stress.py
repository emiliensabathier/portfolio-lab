"""Stress testing.

Historical scenarios replay realized returns over a fixed window; they need no estimation
and therefore cover periods the out-of-sample backtest cannot reach, notably 2008.
"""

from __future__ import annotations

import pandas as pd

from plab.risk.metrics import max_drawdown

HISTORICAL_SCENARIOS: dict[str, tuple[str, str]] = {
    "gfc_2008": ("2007-10-09", "2009-03-09"),
    "covid_2020": ("2020-02-19", "2020-03-23"),
    "rates_2022": ("2022-01-03", "2022-10-14"),
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


def parametric_shock(weights: dict[str, float], shocks: dict[str, float]) -> float:
    """Instantaneous portfolio return under a set of per-asset shocks."""
    missing = sorted(set(weights) - set(shocks))
    if missing:
        raise ValueError(f"no shock specified for {missing}")
    return float(sum(weight * shocks[ticker] for ticker, weight in weights.items()))
