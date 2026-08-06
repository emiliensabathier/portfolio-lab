import numpy as np
import pandas as pd
import pytest

from plab.returns import simple_returns
from plab.risk.metrics import (
    PERIODS_PER_YEAR,
    annualized_volatility,
    beta,
    cagr,
    calmar_ratio,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summary,
)


def _series(values: list[float]) -> pd.Series:
    index = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=index)


def test_simple_returns_matches_hand_computation() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 99.0]}, index=pd.bdate_range("2020-01-01", periods=3)
    )

    result = simple_returns(prices)["A"]

    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(0.10)
    assert result.iloc[1] == pytest.approx(-0.10)


def test_constant_return_series_has_zero_volatility_and_infinite_sharpe() -> None:
    constant = _series([0.001] * PERIODS_PER_YEAR)

    assert annualized_volatility(constant) == pytest.approx(0.0)
    assert np.isinf(sharpe_ratio(constant))


def test_cagr_of_a_doubling_over_one_year() -> None:
    daily = 2 ** (1 / PERIODS_PER_YEAR) - 1
    returns = _series([daily] * PERIODS_PER_YEAR)

    assert cagr(returns) == pytest.approx(1.0, rel=1e-6)


def test_max_drawdown_of_a_known_path() -> None:
    # 1.0 -> 1.25 -> 1.00 -> 1.50 : worst peak-to-trough is -20%
    returns = _series([0.25, -0.20, 0.50])

    assert max_drawdown(returns) == pytest.approx(-0.20)
    assert drawdown_series(returns).min() == pytest.approx(-0.20)


def test_sortino_ignores_upside_deviation() -> None:
    upside_only = _series([0.01] * 50)
    mixed = _series([0.01, -0.01] * 25)

    assert np.isinf(sortino_ratio(upside_only))
    assert np.isfinite(sortino_ratio(mixed))


def test_calmar_is_cagr_over_absolute_max_drawdown() -> None:
    returns = _series([0.25, -0.20, 0.50])

    expected = cagr(returns) / abs(max_drawdown(returns))
    assert calmar_ratio(returns) == pytest.approx(expected)


def test_beta_of_a_series_against_itself_is_one() -> None:
    rng = np.random.default_rng(4)
    series = _series(list(rng.normal(0.0, 0.01, 500)))

    assert beta(series, series) == pytest.approx(1.0)


def test_beta_of_a_doubled_series_is_two() -> None:
    rng = np.random.default_rng(5)
    benchmark = _series(list(rng.normal(0.0, 0.01, 500)))

    assert beta(benchmark * 2.0, benchmark) == pytest.approx(2.0)


def test_beta_of_an_uncorrelated_series_is_near_zero() -> None:
    rng = np.random.default_rng(6)
    benchmark = _series(list(rng.normal(0.0, 0.01, 5_000)))
    independent = _series(list(rng.normal(0.0, 0.01, 5_000)))

    assert abs(beta(independent, benchmark)) < 0.1


def test_beta_against_a_riskless_benchmark_raises() -> None:
    flat = _series([0.0] * 50)

    with pytest.raises(ValueError, match="zero variance"):
        beta(_series([0.01] * 50), flat)


def test_summary_exposes_every_headline_metric() -> None:
    returns = _series([0.01, -0.005] * 60)

    assert set(summary(returns)) == {
        "cagr",
        "volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
    }
