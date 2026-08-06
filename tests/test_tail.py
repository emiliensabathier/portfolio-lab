import numpy as np
import pandas as pd
import pytest

from plab.risk.tail import cornish_fisher_var, historical_es, historical_var

# Left-skewed sample used for the Cornish-Fisher golden value below.
# Sample moments: skew -1.6991, excess kurtosis 2.3330.
SKEWED = [0.01, 0.02, -0.01, 0.005, 0.015, -0.02, 0.008, -0.06, 0.012, -0.005, 0.003, 0.02]


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


def test_var_and_expected_shortfall_on_a_hand_computed_sample() -> None:
    # Ten observations, level 90%: the 10th-percentile index is 0.1*(10-1) = 0.9, so the
    # threshold interpolates between the two worst values, -0.10 and -0.05, landing at
    # -0.055. Only -0.10 breaches it, so the expected shortfall is exactly 0.10.
    returns = pd.Series([-0.10, -0.05, -0.02, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])

    assert historical_var(returns, 0.90) == pytest.approx(0.055)
    assert historical_es(returns, 0.90) == pytest.approx(0.10)


def test_cornish_fisher_matches_a_reference_value_on_a_skewed_sample() -> None:
    # A golden value, deliberately not a re-typed copy of the expansion: re-typing the
    # polynomial in the test would only prove the test and the code were typed the same
    # way. A mistyped denominator, a dropped term or a flipped sign all move this number,
    # whereas the near-normal and directional tests above survive every one of them.
    returns = pd.Series(SKEWED)

    assert cornish_fisher_var(returns, 0.95) == pytest.approx(0.045446949, rel=1e-6)


def test_cornish_fisher_widens_the_tail_beyond_the_gaussian_on_that_sample() -> None:
    # Same sample: the Gaussian VaR is 0.036919, so the left skew widens the tail by 23%.
    # This pins the size of the correction, not merely its sign.
    returns = pd.Series(SKEWED)

    assert cornish_fisher_var(returns, 0.95) / 0.036918838 == pytest.approx(1.231, rel=1e-3)


@pytest.mark.parametrize("function", [historical_var, historical_es, cornish_fisher_var])
@pytest.mark.parametrize("level", [1.5, 0.0, 1.0, -0.2])
def test_every_function_rejects_an_invalid_level(function, level: float) -> None:
    with pytest.raises(ValueError):
        function(pd.Series([0.01, -0.01]), level=level)


def test_expected_shortfall_raises_when_no_observation_breaches_the_threshold() -> None:
    # A NaN threshold makes every comparison False, emptying the tail. The guard must
    # raise rather than return a NaN that would propagate silently into the report.
    returns = pd.Series([0.01, -0.01, float("nan")])

    with pytest.raises(ValueError, match="beyond the VaR threshold"):
        historical_es(returns, 0.95)
