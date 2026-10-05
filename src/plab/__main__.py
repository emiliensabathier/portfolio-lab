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
    # An upstream tail can be incomplete for days after the fact: a missing close on a
    # recent session makes the loader refuse the whole run, correctly, and there is
    # otherwise no way to ask for the history that is intact. Applies to the bill series
    # as well as to the prices, so the two stay on the same window.
    parser.add_argument(
        "--end",
        default=None,
        help="last date to load, inclusive, ISO format (default: latest available)",
    )
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--estimation-months", type=int, default=36)
    args = parser.parse_args()

    prices = load_prices(
        tickers(ETF_CORE),
        CORE_START,
        end=args.end,
        cache_dir=Path(args.cache_dir),
        refresh=args.refresh,
    )
    risk_free = load_risk_free(
        CORE_START, end=args.end, cache_dir=Path(args.cache_dir), refresh=args.refresh
    )
    config = BacktestConfig(estimation_months=args.estimation_months, cost_bps=args.cost_bps)
    output = run(prices, config, risk_free=risk_free)
    html = render(output, generated_on=datetime.now(UTC).date().isoformat())

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    print(f"wrote {destination} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
