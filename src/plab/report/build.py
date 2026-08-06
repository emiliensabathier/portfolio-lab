"""Assembly of the final HTML report.

Computes nothing: every number displayed here is produced by the risk and backtest modules.
"""

from __future__ import annotations

import html as html_escape

import pandas as pd

from plab.backtest.engine import BacktestResult
from plab.report.charts import drawdown_chart, risk_share_chart, wealth_chart
from plab.risk.metrics import beta, summary
from plab.risk.tail import cornish_fisher_var, historical_es, historical_var

STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
       max-width: 980px; padding: 2rem 1.25rem; color: #16181d; line-height: 1.5; }
h1 { font-size: 1.9rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.25rem; margin-top: 2.5rem; border-bottom: 1px solid #e3e5ea;
     padding-bottom: 0.35rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 0.45rem 0.6rem; border-bottom: 1px solid #eceef2; }
th:first-child, td:first-child { text-align: left; }
thead th { border-bottom: 2px solid #c9ccd4; }
.note { color: #5b6070; font-size: 0.9rem; }
svg { max-width: 100%; height: auto; }
"""


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html_escape.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html_escape.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _num(value: float) -> str:
    return f"{value:.2f}"


def _performance_table(
    results: dict[str, BacktestResult], benchmark: str | None = None
) -> str:
    reference = results[benchmark].returns if benchmark in results else None
    rows = []
    for name, result in results.items():
        stats = summary(result.returns)
        row = [
            name,
            _pct(stats["cagr"]),
            _pct(stats["volatility"]),
            _num(stats["sharpe"]),
            _num(stats["sortino"]),
            _pct(stats["max_drawdown"]),
            _num(stats["calmar"]),
            _num(float(result.turnover.mean() * 12)),
        ]
        if reference is not None:
            row.append(_num(beta(result.returns, reference)))
        rows.append(row)

    headers = ["Strategy", "CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown",
               "Calmar", "Turnover p.a."]
    if reference is not None:
        headers.append(f"Beta vs {benchmark}")
    return _table(headers, rows)


def _attribution_table(attribution: dict[str, pd.DataFrame]) -> str:
    rows = []
    for name, frame in attribution.items():
        for bloc, values in frame.iterrows():
            rows.append(
                [name, str(bloc), _pct(values["return_contribution"]),
                 _pct(values["risk_share"])]
            )
    return _table(
        ["Strategy", "Asset class", "Return contribution", "Share of risk"], rows
    )


def _tail_table(results: dict[str, BacktestResult]) -> str:
    rows = []
    for name, result in results.items():
        rows.append(
            [
                name,
                _pct(historical_var(result.returns, 0.95)),
                _pct(historical_es(result.returns, 0.95)),
                _pct(historical_var(result.returns, 0.99)),
                _pct(cornish_fisher_var(result.returns, 0.99)),
            ]
        )
    return _table(
        ["Strategy", "VaR 95%", "ES 95%", "VaR 99%", "Cornish-Fisher VaR 99%"], rows
    )


def _stress_table(stress: dict[str, dict[str, dict[str, float]]]) -> str:
    rows = []
    for name, scenarios in stress.items():
        for scenario, outcome in scenarios.items():
            rows.append(
                [
                    name,
                    scenario,
                    _pct(outcome["total_return"]),
                    _pct(outcome["max_drawdown"]),
                    _pct(outcome["worst_day"]),
                ]
            )
    return _table(["Strategy", "Scenario", "Total return", "Max drawdown", "Worst day"], rows)


def build_report(
    results: dict[str, BacktestResult],
    risk_shares: dict[str, pd.Series],
    stress: dict[str, dict[str, dict[str, float]]],
    generated_on: str,
    *,
    benchmark: str | None = None,
    bloc_attribution: dict[str, pd.DataFrame] | None = None,
) -> str:
    """Render the whole report as one self-contained HTML document."""
    if not results:
        raise ValueError("the report needs at least one backtest result")
    if benchmark is not None and benchmark not in results:
        # Silently dropping the Beta column would turn a caller's typo, or a renamed
        # strategy, into a quietly less informative report rather than an error.
        raise ValueError(
            f"benchmark {benchmark!r} is not among the backtested strategies "
            f"{sorted(results)}"
        )

    series = {name: result.returns for name, result in results.items()}
    start = min(s.index.min() for s in series.values()).date()
    end = max(s.index.max() for s in series.values()).date()

    sections = [
        "<h1>Multi-asset portfolio construction</h1>",
        f'<p class="note">Out-of-sample {start} to {end}. All figures are '
        f"net of transaction costs. Generated on {html_escape.escape(generated_on)}.</p>",
        "<h2>Performance and risk</h2>",
        _performance_table(results, benchmark),
        "<h2>Growth of capital</h2>",
        wealth_chart(series),
        "<h2>Drawdowns</h2>",
        drawdown_chart(series),
        "<h2>Tail risk</h2>",
        _tail_table(results),
    ]
    if risk_shares:
        sections += ["<h2>Risk contribution</h2>", risk_share_chart(risk_shares)]
    if bloc_attribution:
        sections += [
            "<h2>Asset class attribution</h2>",
            '<p class="note">Where the return came from, and where the risk sits. The two '
            "rarely match: that gap is what separates risk parity from equal weight.</p>",
            _attribution_table(bloc_attribution),
        ]
    if stress:
        sections += [
            "<h2>Stress tests</h2>",
            '<p class="note">Counterfactuals: each strategy\'s current allocation applied '
            "to the realized returns of a historical crisis, held static for the window "
            "with no rebalancing and no transaction costs — unlike every other figure in "
            "this report, these are not net of costs. This lets the scenarios reach "
            "periods, notably 2008, that the out-of-sample backtest itself cannot.</p>",
            _stress_table(stress),
        ]

    body = "\n".join(sections)
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<title>Multi-asset portfolio construction</title>\n"
        f"<style>{STYLE}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )
