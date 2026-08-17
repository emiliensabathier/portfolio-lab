"""Command-line entry point: ``python -m plab``."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from plab.backtest.engine import BacktestConfig
from plab.cash import load_risk_free
from plab.data.loader import load_prices
from plab.pipeline import render, run
from plab.universe import CORE_START, ETF_CORE, tickers


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the portfolio report")
    parser.add_argument("--output", default="reports/portfolio.html")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--refresh", action="store_true", help="ignore the cached prices")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--estimation-months", type=int, default=36)
    args = parser.parse_args()

    prices = load_prices(
        tickers(ETF_CORE),
        CORE_START,
        cache_dir=Path(args.cache_dir),
        refresh=args.refresh,
    )
    risk_free = load_risk_free(
        CORE_START, cache_dir=Path(args.cache_dir), refresh=args.refresh
    )
    config = BacktestConfig(
        estimation_months=args.estimation_months, cost_bps=args.cost_bps
    )
    output = run(prices, config, risk_free=risk_free)
    html = render(output, generated_on=datetime.now(UTC).date().isoformat())

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    print(f"wrote {destination} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
