import numpy as np
import pandas as pd
import pytest

from plab.risk.attribution import bloc_contribution, return_contribution, risk_contribution


def test_return_contributions_sum_to_the_portfolio_return() -> None:
    index = pd.bdate_range("2020-01-01", periods=4)
    weights = pd.DataFrame(0.5, index=index, columns=["A", "B"])
    returns = pd.DataFrame(
        {"A": [0.01, -0.02, 0.03, 0.00], "B": [0.00, 0.01, -0.01, 0.02]}, index=index
    )

    contributions = return_contribution(weights, returns)
    portfolio = (weights * returns).sum(axis=1).sum()

    assert contributions.sum() == pytest.approx(portfolio)


def test_return_contribution_is_zero_for_an_unheld_asset() -> None:
    index = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [1.0] * 3, "B": [0.0] * 3}, index=index)
    returns = pd.DataFrame({"A": [0.01] * 3, "B": [0.05] * 3}, index=index)

    assert return_contribution(weights, returns)["B"] == pytest.approx(0.0)


def test_return_contribution_raises_when_a_held_ticker_has_no_returns() -> None:
    # Without the guard this reports C as contributing exactly 0.0 — a held position
    # silently reported as flat, which is worse than an error because it looks like data.
    index = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.5] * 3, "C": [0.5] * 3}, index=index)
    returns = pd.DataFrame({"A": [0.01] * 3}, index=index)

    with pytest.raises(ValueError, match="do not cover"):
        return_contribution(weights, returns)


def test_return_contribution_raises_when_a_weighted_date_has_no_returns() -> None:
    # A missing date would be skipped by the default skipna, quietly shortening the period
    # the attribution claims to cover.
    index = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [1.0] * 3}, index=index)
    returns = pd.DataFrame({"A": [0.01] * 2}, index=index[:2])

    with pytest.raises(ValueError, match="do not cover"):
        return_contribution(weights, returns)


def test_risk_contributions_sum_to_portfolio_volatility() -> None:
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    weights = {"A": 0.6, "B": 0.4}

    contributions = risk_contribution(weights, cov)
    w = np.array([0.6, 0.4])
    portfolio_vol = float(np.sqrt(w @ cov.to_numpy() @ w))

    assert contributions.sum() == pytest.approx(portfolio_vol)


def test_the_volatile_asset_carries_more_risk_than_its_weight_suggests() -> None:
    cov = pd.DataFrame(
        [[0.01, 0.0], [0.0, 0.25]], index=["calm", "wild"], columns=["calm", "wild"]
    )

    contributions = risk_contribution({"calm": 0.5, "wild": 0.5}, cov)
    shares = contributions / contributions.sum()

    assert shares["wild"] > 0.9


def test_risk_contribution_raises_on_a_riskless_portfolio() -> None:
    # The guard this task exists to enforce: with zero portfolio volatility the Euler
    # decomposition divides by zero, so contributions are undefined rather than zero.
    covariance = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.0]], index=["A", "B"], columns=["A", "B"]
    )

    with pytest.raises(ValueError, match="undefined"):
        risk_contribution({"A": 0.5, "B": 0.5}, covariance)


def test_bloc_contribution_rejects_a_ticker_with_no_declared_bloc() -> None:
    contributions = pd.Series({"SPY": 0.03, "AGG": 0.005})

    with pytest.raises(KeyError):
        bloc_contribution(contributions, {"SPY": "equity"})


def test_bloc_contribution_aggregates_by_declared_bloc() -> None:
    contributions = pd.Series({"SPY": 0.03, "IWM": 0.01, "AGG": 0.005})
    blocs = {"SPY": "equity", "IWM": "equity", "AGG": "rates"}

    aggregated = bloc_contribution(contributions, blocs)

    assert aggregated["equity"] == pytest.approx(0.04)
    assert aggregated["rates"] == pytest.approx(0.005)
