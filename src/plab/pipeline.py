"""End-to-end pipeline: prices in, report out."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from plab.alloc.rules import equal_weight, fixed_weights, max_sharpe, min_variance, risk_parity
from plab.backtest.engine import BacktestConfig, BacktestResult, Strategy, run_backtest
from plab.report.build import build_report
from plab.returns import simple_returns
from plab.risk.attribution import bloc_contribution, return_contribution, risk_contribution
from plab.risk.covariance import ledoit_wolf_covariance
from plab.risk.stress import HISTORICAL_SCENARIOS, replay
from plab.universe import ETF_CORE

STRATEGIES: dict[str, Strategy] = {
    "equal_weight": equal_weight,
    "min_variance": min_variance,
    "risk_parity": risk_parity,
    "max_sharpe": max_sharpe,
    "60/40": fixed_weights({"SPY": 0.6, "AGG": 0.4}),
}

BENCHMARK = "60/40"
BLOCS: dict[str, str] = {asset.ticker: asset.bloc for asset in ETF_CORE}


@dataclass(frozen=True)
class PipelineOutput:
    """Everything the report needs."""

    results: dict[str, BacktestResult]
    risk_shares: dict[str, pd.Series]
    stress: dict[str, dict[str, dict[str, float]]]
    bloc_attribution: dict[str, pd.DataFrame]


def run(prices: pd.DataFrame, config: BacktestConfig) -> PipelineOutput:
    """Backtest every declared strategy and derive its attribution and stress outcomes."""
    results: dict[str, BacktestResult] = {}
    risk_shares: dict[str, pd.Series] = {}
    stress: dict[str, dict[str, dict[str, float]]] = {}
    bloc_attribution: dict[str, pd.DataFrame] = {}

    asset_returns = simple_returns(prices)
    covariance = ledoit_wolf_covariance(asset_returns.tail(config.estimation_months * 21))

    for name, strategy in STRATEGIES.items():
        result = run_backtest(prices, strategy, config)
        results[name] = result

        final = result.weights.iloc[-1].to_dict()
        contributions = risk_contribution(final, covariance)
        shares = contributions / contributions.sum()
        risk_shares[name] = shares

        returns_by_asset = return_contribution(result.held_weights, asset_returns)
        bloc_attribution[name] = pd.DataFrame(
            {
                "return_contribution": bloc_contribution(returns_by_asset, BLOCS),
                "risk_share": bloc_contribution(shares, BLOCS),
            }
        ).fillna(0.0)

        outcomes: dict[str, dict[str, float]] = {}
        for scenario in HISTORICAL_SCENARIOS:
            try:
                outcomes[scenario] = replay(result.returns, scenario)
            except ValueError:
                continue  # the scenario predates the out-of-sample window
        if outcomes:
            stress[name] = outcomes

    return PipelineOutput(
        results=results,
        risk_shares=risk_shares,
        stress=stress,
        bloc_attribution=bloc_attribution,
    )


def render(output: PipelineOutput, generated_on: str) -> str:
    """Render the pipeline output as HTML."""
    return build_report(
        results=output.results,
        risk_shares=output.risk_shares,
        stress=output.stress,
        generated_on=generated_on,
        benchmark=BENCHMARK,
        bloc_attribution=output.bloc_attribution,
    )
