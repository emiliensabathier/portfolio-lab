"""Two robustness checks on the headline result, written to ``docs/extensions.json``.

- **Long history.** The core universe starts in mid-2007, so its out-of-sample window
  begins in 2010 and never sees 2008. Dropping HYG and DBC buys back two and a half years
  of history: the same rules, out of sample, through the crisis.
- **Capacity.** The headline charges a flat 5 bps. Here the same backtests also pay
  square-root market impact on a book of 100 million, 1 billion and 10 billion dollars,
  with liquidity taken from each ETF's own traded value.

Network access is needed (Yahoo for prices and volume, FRED for the bill).

    python scripts/extensions.py [--end 2026-08-14]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plab.backtest.engine import BacktestConfig
from plab.cash import load_risk_free
from plab.data.loader import load_prices, yfinance_dollar_volume
from plab.pipeline import BENCHMARK, capacity, run
from plab.risk.metrics import sharpe_difference_test, summary
from plab.universe import CORE_START, ETF_CORE, ETF_LONG, LONG_START, tickers

ROOT = Path(__file__).resolve().parents[1]
CONFIG = BacktestConfig(estimation_months=36, cost_bps=5.0)
SIZES = (1e8, 1e9, 1e10)
# From the first out-of-sample month to the market trough, so the window is the part of the
# crisis the long backtest actually reaches, not the scenario's full 2007-10-09 start.
CRISIS = ("2007-12-01", "2009-03-09")


def long_history(end: str, cache: Path) -> dict:
    prices = load_prices(tickers(ETF_LONG), LONG_START, end, cache_dir=cache)
    bill = load_risk_free(LONG_START, end, cache_dir=cache)
    output = run(prices, CONFIG, risk_free=bill)
    reference = output.results[BENCHMARK].returns
    strategies = {}
    for name, result in output.results.items():
        crisis = result.returns.loc[CRISIS[0] : CRISIS[1]]
        entry = {
            **summary(result.returns, output.risk_free),
            "turnover": float(result.turnover.iloc[1:].sum() / (len(result.returns) / 252)),
            "crisis_return": float((1 + crisis).prod() - 1),
            "crisis_drawdown": float(summary(crisis)["max_drawdown"]),
        }
        if name != BENCHMARK:
            entry["p_vs_benchmark"] = sharpe_difference_test(
                result.returns, reference, output.risk_free
            )["p_value"]
        strategies[name] = entry
    window = output.results[BENCHMARK].returns.index
    return {
        "universe": tickers(ETF_LONG),
        "out_of_sample": [window[0].date().isoformat(), window[-1].date().isoformat()],
        "crisis_window": [crisis.index[0].date().isoformat(), crisis.index[-1].date().isoformat()],
        "strategies": strategies,
    }


def capacity_table(end: str, cache: Path) -> dict:
    names = tickers(ETF_CORE)
    prices = load_prices(names, CORE_START, end, cache_dir=cache)
    bill = load_risk_free(CORE_START, end, cache_dir=cache)
    volume = yfinance_dollar_volume(names, CORE_START, end).reindex(prices.index)
    table = capacity(prices, CONFIG, volume, SIZES, risk_free=bill)
    adv = volume.tail(252).mean()
    return {
        "sizes": [0.0, *SIZES],
        "mean_daily_value_traded_last_year": {k: float(v) for k, v in adv.items()},
        "sharpe": {n: {str(s): float(v) for s, v in table["sharpe"].loc[n].items()}
                   for n in table.index},
        "cost_drag": {n: {str(s): float(v) for s, v in table["cost_drag"].loc[n].items()}
                      for n in table.index},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()
    cache = Path(args.cache_dir)

    payload = {
        "end": args.end,
        "long_history": long_history(args.end, cache),
        "capacity": capacity_table(args.end, cache),
    }
    destination = ROOT / "docs" / "extensions.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
