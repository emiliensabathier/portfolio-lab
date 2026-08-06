"""Chart rendering to inline SVG.

Charts are embedded directly in the HTML: the report must open offline, with no CDN and no
external asset of any kind.
"""

from __future__ import annotations

import io

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

FIGSIZE = (9.0, 4.0)
DPI = 110


def figure_to_svg(fig: Figure) -> str:
    """Serialize a figure as inline SVG markup, stripped of its XML preamble."""
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    markup = buffer.getvalue()
    return markup[markup.index("<svg") :]


def _new_figure(title: str, ylabel: str) -> tuple[Figure, object]:
    fig = Figure(figsize=FIGSIZE, dpi=DPI)
    axes = fig.add_subplot(111)
    axes.set_title(title)
    axes.set_ylabel(ylabel)
    axes.grid(True, alpha=0.25)
    return fig, axes


def wealth_chart(series_by_name: dict[str, pd.Series]) -> str:
    """Cumulative growth of one unit of capital, net of costs."""
    fig, axes = _new_figure("Growth of 1 unit (net of costs)", "Wealth")
    for name, returns in series_by_name.items():
        axes.plot((1.0 + returns).cumprod(), label=name, linewidth=1.2)
    axes.legend(loc="upper left", frameon=False)
    return figure_to_svg(fig)


def drawdown_chart(series_by_name: dict[str, pd.Series]) -> str:
    """Drawdown paths."""
    from plab.risk.metrics import drawdown_series

    fig, axes = _new_figure("Drawdown", "Loss from peak")
    for name, returns in series_by_name.items():
        axes.plot(drawdown_series(returns), label=name, linewidth=1.0)
    axes.legend(loc="lower left", frameon=False)
    return figure_to_svg(fig)


def risk_share_chart(shares_by_name: dict[str, pd.Series]) -> str:
    """Share of total portfolio risk carried by each asset, per strategy."""
    frame = pd.DataFrame(shares_by_name).fillna(0.0)
    fig, axes = _new_figure("Risk contribution share", "Share of portfolio volatility")
    bottom = pd.Series(0.0, index=frame.columns)
    for asset in frame.index:
        axes.bar(frame.columns, frame.loc[asset], bottom=bottom, label=asset)
        bottom = bottom + frame.loc[asset]
    axes.legend(loc="upper right", frameon=False, ncol=2, fontsize="small")
    return figure_to_svg(fig)
