"""Regenerate the regression fixture from cached prices.

Run manually after a deliberate methodology change, never in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

from plab.backtest.engine import BacktestConfig
from plab.data.loader import load_prices
from plab.pipeline import run
from plab.risk.metrics import summary
from plab.universe import CORE_START, ETF_CORE, tickers

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
CONFIG = BacktestConfig(estimation_months=36, cost_bps=5.0)


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    prices = load_prices(
        tickers(ETF_CORE), CORE_START, "2024-12-31", cache_dir=Path("cache")
    )
    prices.to_csv(FIXTURES / "prices.csv")

    output = run(prices, CONFIG)
    expected = {
        name: summary(result.returns) for name, result in output.results.items()
    }
    (FIXTURES / "expected_metrics.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"fixture written for {len(expected)} strategies")


if __name__ == "__main__":
    main()
