"""End-to-end pipeline: prices in, report out."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import pandas as pd

from plab.alloc.rules import equal_weight, fixed_weights, max_sharpe, min_variance, risk_parity
from plab.backtest.engine import BacktestConfig, BacktestResult, Strategy, run_backtest
from plab.cash import align as align_risk_free
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


def strategies_for(risk_free: pd.Series | None) -> dict[str, Strategy]:
    """The declared strategies, with the bill handed to the one rule that optimizes on it.

    Maximum Sharpe maximizes the ratio the report publishes, so when the report measures
    Sharpe against the bill the rule must too. ``STRATEGIES`` itself is left untouched.
    """
    if risk_free is None:
        return dict(STRATEGIES)
    return {**STRATEGIES, "max_sharpe": partial(max_sharpe, risk_free=risk_free)}


@dataclass(frozen=True)
class PipelineOutput:
    """Everything the report needs."""

    results: dict[str, BacktestResult]
    risk_shares: dict[str, pd.Series]
    stress: dict[str, dict[str, dict[str, float]]]
    bloc_attribution: dict[str, pd.DataFrame]
    risk_free: pd.Series | None = None


def run(
    prices: pd.DataFrame,
    config: BacktestConfig,
    risk_free: pd.Series | None = None,
) -> PipelineOutput:
    """Backtest every declared strategy and derive its attribution and stress outcomes."""
    results: dict[str, BacktestResult] = {}
    risk_shares: dict[str, pd.Series] = {}
    stress: dict[str, dict[str, dict[str, float]]] = {}
    bloc_attribution: dict[str, pd.DataFrame] = {}

    asset_returns = simple_returns(prices)
    covariance = ledoit_wolf_covariance(asset_returns.tail(config.estimation_months * 21))
    aligned_risk_free: pd.Series | None = None

    for name, strategy in strategies_for(risk_free).items():
        result = run_backtest(prices, strategy, config)
        results[name] = result
        if risk_free is not None and aligned_risk_free is None:
            # Every strategy shares one out-of-sample window, so the bill series is put on
            # that calendar once. Aligning per strategy would repeat the work and invite
            # the two to drift apart.
            aligned_risk_free = align_risk_free(risk_free, result.returns.index)

        final = result.weights.iloc[-1]
        contributions = risk_contribution(final.to_dict(), covariance)
        shares = contributions / contributions.sum()
        risk_shares[name] = shares

        returns_by_asset = return_contribution(result.held_weights, asset_returns)
        bloc_attribution[name] = pd.DataFrame(
            {
                "return_contribution": bloc_contribution(returns_by_asset, BLOCS),
                "risk_share": bloc_contribution(shares, BLOCS),
            }
        ).fillna(0.0)

        # Stress scenarios apply the strategy's final weights to the raw asset returns of
        # each episode, rather than slicing its own out-of-sample series. That series
        # cannot reach 2008 — the first 36 months of history are spent estimating — so
        # slicing it would silently drop the very crisis this section exists to cover.
        # Every row is therefore a counterfactual on the current allocation: comparable
        # with the others, and labelled as such in the report. No scenario is skipped; a
        # window the price history cannot reach is an error, not something to swallow.
        static = (asset_returns * final).sum(axis=1)
        stress[name] = {scenario: replay(static, scenario) for scenario in HISTORICAL_SCENARIOS}

    return PipelineOutput(
        results=results,
        risk_shares=risk_shares,
        stress=stress,
        bloc_attribution=bloc_attribution,
        risk_free=aligned_risk_free,
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
        risk_free=output.risk_free,
    )
