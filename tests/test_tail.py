import numpy as np
import pandas as pd
import pytest

from plab.risk.tail import cornish_fisher_var, historical_es, historical_var


def test_historical_var_is_the_empirical_quantile_as_a_positive_loss() -> None:
    # 100 observations from -1.00 to -0.01 plus nothing positive: the 95% VaR is the
    # 5th percentile of returns, i.e. a loss of 0.95.
    returns = pd.Series(np.linspace(-1.0, -0.01, 100))

    assert historical_var(returns, level=0.95) == pytest.approx(0.95, abs=0.02)


def test_expected_shortfall_is_worse_than_var() -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0, 0.01, 5_000))

    assert historical_es(returns, 0.95) > historical_var(returns, 0.95)


def test_cornish_fisher_matches_gaussian_var_on_a_normal_sample() -> None:
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0, 0.01, 100_000))

    gaussian = 1.6448536 * 0.01
    assert cornish_fisher_var(returns, 0.95) == pytest.approx(gaussian, rel=0.05)


def test_cornish_fisher_exceeds_gaussian_on_a_left_skewed_sample() -> None:
    rng = np.random.default_rng(2)
    body = rng.normal(0.0, 0.005, 9_500)
    tail = rng.normal(-0.05, 0.01, 500)
    returns = pd.Series(np.concatenate([body, tail]))

    gaussian = 1.6448536 * float(returns.std(ddof=1))
    assert cornish_fisher_var(returns, 0.95) > gaussian


def test_invalid_level_raises() -> None:
    with pytest.raises(ValueError):
        historical_var(pd.Series([0.01, -0.01]), level=1.5)
