import numpy as np
import pandas as pd
import pytest
from scipy import stats

from plab.returns import simple_returns
from plab.risk.metrics import (
    PERIODS_PER_YEAR,
    annualized_volatility,
    beta,
    cagr,
    calmar_ratio,
    drawdown_series,
    max_drawdown,
    sharpe_difference_test,
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


def test_sortino_divides_shortfalls_by_the_total_number_of_periods() -> None:
    # 50 periods, 25 of them at -1%. The canonical downside deviation divides the sum of
    # squared shortfalls by the TOTAL period count, not by the count of losing periods.
    # Dividing by 25 instead of 50 would inflate the deviation by sqrt(2) and shrink the
    # ratio by the same factor, which this expected value pins down.
    returns = _series([0.02, -0.01] * 25)

    expected_deviation = np.sqrt(25 * 0.01**2 / 50) * np.sqrt(PERIODS_PER_YEAR)
    expected = (float(returns.mean()) * PERIODS_PER_YEAR) / expected_deviation

    assert sortino_ratio(returns) == pytest.approx(expected)


def test_calmar_is_cagr_over_absolute_max_drawdown() -> None:
    # Path 1.0 -> 1.25 -> 1.00 -> 1.50 over 3 periods: final wealth 1.5 and a worst
    # peak-to-trough of -20%, both read off the path rather than recomputed by the code
    # under test.
    returns = _series([0.25, -0.20, 0.50])

    expected = (1.5 ** (PERIODS_PER_YEAR / 3) - 1.0) / 0.20
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


def _pair(n: int, sr_a: float, sr_b: float, rho: float, seed: int) -> tuple[pd.Series, pd.Series]:
    """Two daily series with unit-free target per-period Sharpe ratios and correlation."""
    rng = np.random.default_rng(seed)
    shocks = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], n)
    index = pd.bdate_range("2010-01-01", periods=n)
    vol = 0.01
    return (
        pd.Series(vol * (sr_a + shocks[:, 0]), index=index),
        pd.Series(vol * (sr_b + shocks[:, 1]), index=index),
    )


def test_the_sharpe_test_follows_jobson_korkie_with_the_memmel_correction() -> None:
    a, b = _pair(750, 0.06, 0.02, 0.6, seed=11)

    result = sharpe_difference_test(a, b)

    # Memmel (2003): theta = [2 - 2 rho + (SR_a^2 + SR_b^2 - 2 SR_a SR_b rho^2) / 2] / T,
    # on per-period ratios; z = (SR_a - SR_b) / sqrt(theta).
    sr_a, sr_b = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1)
    rho = np.corrcoef(a, b)[0, 1]
    theta = (2 - 2 * rho + 0.5 * (sr_a**2 + sr_b**2 - 2 * sr_a * sr_b * rho**2)) / len(a)
    z = (sr_a - sr_b) / np.sqrt(theta)
    assert result["z"] == pytest.approx(z)
    assert result["p_value"] == pytest.approx(2 * (1 - stats.norm.cdf(abs(z))))
    assert result["difference"] == pytest.approx((sr_a - sr_b) * np.sqrt(PERIODS_PER_YEAR))


def test_the_sharpe_test_is_antisymmetric() -> None:
    a, b = _pair(500, 0.05, 0.01, 0.3, seed=12)

    forward, backward = sharpe_difference_test(a, b), sharpe_difference_test(b, a)

    assert forward["z"] == pytest.approx(-backward["z"])
    assert forward["p_value"] == pytest.approx(backward["p_value"])


def test_the_sharpe_test_rejects_at_its_nominal_rate_under_the_null() -> None:
    # Equal true Sharpe ratios, correlated series: a correctly sized test rejects at 5%
    # about 5% of the time. 1,000 draws put the binomial standard error at 0.7 points.
    draws, periods, rho, sharpe = 1_000, 500, 0.5, 0.04
    rng = np.random.default_rng(0)
    first = rng.standard_normal((draws, periods))
    second = rho * first + np.sqrt(1 - rho**2) * rng.standard_normal((draws, periods))
    index = pd.bdate_range("2010-01-01", periods=periods)
    rejections = sum(
        sharpe_difference_test(
            pd.Series(0.01 * (sharpe + a), index=index),
            pd.Series(0.01 * (sharpe + b), index=index),
        )["p_value"]
        < 0.05
        for a, b in zip(first, second, strict=True)
    )
    assert 0.03 <= rejections / draws <= 0.07


def test_the_sharpe_test_detects_a_large_difference() -> None:
    a, b = _pair(2_500, 0.10, 0.00, 0.5, seed=13)
    assert sharpe_difference_test(a, b)["p_value"] < 0.01


def test_the_sharpe_test_measures_both_legs_against_the_same_bill() -> None:
    a, b = _pair(500, 0.05, 0.01, 0.3, seed=14)
    rate = 0.03

    with_bill = sharpe_difference_test(a, b, rate)

    assert with_bill["difference"] == pytest.approx(sharpe_ratio(a, rate) - sharpe_ratio(b, rate))


def test_the_sharpe_test_refuses_a_series_compared_with_itself() -> None:
    a, _ = _pair(300, 0.05, 0.0, 0.0, seed=15)
    with pytest.raises(ValueError, match="degenerate"):
        sharpe_difference_test(a, a)


def test_the_sharpe_test_refuses_series_on_different_dates() -> None:
    a, b = _pair(300, 0.05, 0.0, 0.0, seed=16)
    with pytest.raises(ValueError, match="same dates"):
        sharpe_difference_test(a, b.iloc[1:])
