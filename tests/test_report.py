import numpy as np
import pandas as pd
import pytest

from plab.backtest.engine import BacktestResult
from plab.report.build import build_report
from plab.report.charts import figure_to_svg, wealth_chart


def _result(seed: int = 0) -> BacktestResult:
    index = pd.bdate_range("2015-01-01", periods=300)
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.normal(0.0004, 0.01, 300), index=index)
    weights = pd.DataFrame(0.5, index=index[::20], columns=["A", "B"])
    return BacktestResult(
        returns=returns,
        gross_returns=returns,
        weights=weights,
        held_weights=pd.DataFrame(0.5, index=index, columns=["A", "B"]),
        turnover=pd.Series(0.1, index=index[::20]),
        costs=pd.Series(0.00005, index=index[::20]),
    )


def test_figure_to_svg_produces_inline_markup() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = Figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])

    svg = figure_to_svg(fig)

    assert svg.lstrip().startswith("<svg")
    assert "<?xml" not in svg


def test_wealth_chart_returns_svg_for_every_series() -> None:
    svg = wealth_chart({"strategy": _result().returns, "benchmark": _result(1).returns})

    assert svg.lstrip().startswith("<svg")


def test_report_is_a_single_document_with_no_external_resources() -> None:
    html = build_report(
        results={"min_variance": _result(), "60/40": _result(1)},
        risk_shares={"min_variance": pd.Series({"A": 0.5, "B": 0.5})},
        stress={"min_variance": {"covid_2020": {"total_return": -0.2, "max_drawdown": -0.25,
                                                "worst_day": -0.07, "observations": 25.0}}},
        generated_on="2026-08-06",
    )

    assert html.startswith("<!doctype html>")
    # SVG namespace declarations legitimately contain http:// URIs, so test for actual
    # resource references instead of the bare substring.
    for forbidden in ('src="http', "href=\"http", "<script", "<link", "@import", "cdn"):
        assert forbidden not in html, forbidden


def test_report_shows_every_strategy_and_its_headline_metrics() -> None:
    html = build_report(
        results={"min_variance": _result(), "60/40": _result(1)},
        risk_shares={"min_variance": pd.Series({"A": 0.5, "B": 0.5})},
        stress={},
        generated_on="2026-08-06",
    )

    assert "min_variance" in html
    assert "60/40" in html
    assert "Sharpe" in html
    assert "Max drawdown" in html


def test_report_states_that_returns_are_net_of_costs() -> None:
    html = build_report(
        results={"min_variance": _result()},
        risk_shares={},
        stress={},
        generated_on="2026-08-06",
    )

    assert "net of transaction costs" in html


def test_report_shows_beta_when_a_benchmark_is_named() -> None:
    html = build_report(
        results={"min_variance": _result(), "60/40": _result(1)},
        risk_shares={},
        stress={},
        generated_on="2026-08-06",
        benchmark="60/40",
    )

    assert "Beta" in html


def test_report_shows_asset_class_attribution_when_supplied() -> None:
    attribution = pd.DataFrame(
        {"return_contribution": [0.31, 0.04], "risk_share": [0.85, 0.15]},
        index=["equity", "rates"],
    )

    html = build_report(
        results={"min_variance": _result()},
        risk_shares={},
        stress={},
        generated_on="2026-08-06",
        bloc_attribution={"min_variance": attribution},
    )

    assert "Asset class attribution" in html
    assert "equity" in html


def test_report_rejects_an_empty_result_set() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_report(results={}, risk_shares={}, stress={}, generated_on="2026-08-06")
