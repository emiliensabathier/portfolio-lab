"""Render the one chart the README carries, from the frozen fixture.

GitHub shows a committed HTML file as source, not as a page, so the report's charts are
invisible to anyone browsing the repository. This writes the single figure that makes the
repository's point as a raster the README can embed directly.

It runs against the frozen price fixture rather than a live pull, so the picture is stable:
a chart that silently redrew itself on every run would drift away from the caption under it.

    python scripts/build_readme_chart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plab.backtest.engine import BacktestConfig  # noqa: E402
from plab.pipeline import run  # noqa: E402
from plab.report.charts import wealth_figure  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
OUTPUT = ROOT / "docs" / "growth-of-capital.png"
CONFIG = BacktestConfig(estimation_months=36, cost_bps=5.0)
DPI = 130


def main() -> None:
    prices = pd.read_csv(FIXTURES / "prices.csv", index_col=0, parse_dates=True)
    risk_free = pd.read_csv(FIXTURES / "risk_free.csv", index_col=0, parse_dates=True).iloc[:, 0]
    output = run(prices, CONFIG, risk_free=risk_free)

    series = {name: result.returns for name, result in output.results.items()}
    figure = wealth_figure(series)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, format="png", dpi=DPI, bbox_inches="tight")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes) from the frozen fixture")


if __name__ == "__main__":  # pragma: no cover
    main()
