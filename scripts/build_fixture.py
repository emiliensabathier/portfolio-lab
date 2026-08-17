"""Regenerate the regression fixture from cached prices.

Run manually after a deliberate methodology change, never in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

from plab.backtest.engine import BacktestConfig
from plab.cash import load_risk_free
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

    # The bill series is frozen alongside the prices, because the published Sharpe and
    # Sortino are measured against it. Freezing the prices alone would pin every figure
    # except the two the ratios actually depend on.
    risk_free = load_risk_free(CORE_START, "2024-12-31", cache_dir=Path("cache"))
    risk_free.to_csv(FIXTURES / "risk_free.csv")

    output = run(prices, CONFIG, risk_free=risk_free)
    expected = {
        name: summary(result.returns, output.risk_free)
        for name, result in output.results.items()
    }
    (FIXTURES / "expected_metrics.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Freeze the stress outcomes too. The headline metrics alone would not have caught the
    # regression that mattered most here: a report that displayed a stress table while
    # silently omitting the 2008 crisis it claimed to cover.
    stress = {
        name: {scenario: dict(outcome) for scenario, outcome in scenarios.items()}
        for name, scenarios in output.stress.items()
    }
    (FIXTURES / "expected_stress.json").write_text(
        json.dumps(stress, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"fixture written for {len(expected)} strategies")


if __name__ == "__main__":
    main()
