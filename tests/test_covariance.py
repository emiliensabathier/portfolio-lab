import numpy as np
import pandas as pd
import pytest

from plab.risk.covariance import (
    SingularCovarianceError,
    ledoit_wolf_covariance,
    sample_covariance,
    shrinkage_intensity,
)
from plab.risk.metrics import PERIODS_PER_YEAR


def _returns(n_obs: int, n_assets: int, seed: int = 0) -> pd.DataFrame:
    """Correlated, heteroskedastic returns.

    Deliberately not iid with equal variances. The Ledoit-Wolf target is a scaled
    identity, so for an iid equal-variance population the target already *is* the truth:
    the dispersion term is then pure estimation noise of the same order as the noise
    term, and the shrinkage intensity stays around 0.67 no matter how many observations
    arrive. Real asset returns have unequal volatilities and non-zero correlation, and
    that is what makes the intensity decay towards zero as data accumulates.
    """
    rng = np.random.default_rng(seed)
    vols = np.linspace(0.005, 0.03, n_assets)
    correlation = np.full((n_assets, n_assets), 0.3)
    np.fill_diagonal(correlation, 1.0)
    covariance = np.outer(vols, vols) * correlation
    data = rng.multivariate_normal(np.zeros(n_assets), covariance, n_obs)
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(n_assets)])


def _iid_returns(n_obs: int, n_assets: int, seed: int = 0) -> pd.DataFrame:
    """Equal-variance, uncorrelated returns: the shrinkage target is already correct."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, (n_obs, n_assets))
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(n_assets)])


def test_sample_covariance_is_annualized_and_labelled() -> None:
    returns = _returns(1_000, 3)

    cov = sample_covariance(returns)

    assert list(cov.columns) == ["A0", "A1", "A2"]
    daily_var = float(returns["A0"].var(ddof=1))
    assert cov.loc["A0", "A0"] == pytest.approx(daily_var * PERIODS_PER_YEAR)


def test_shrinkage_intensity_is_a_valid_proportion() -> None:
    assert 0.0 <= shrinkage_intensity(_returns(100, 20)) <= 1.0


def test_shrinkage_is_stronger_when_observations_are_scarce() -> None:
    scarce = shrinkage_intensity(_returns(40, 20, seed=1))
    plentiful = shrinkage_intensity(_returns(4_000, 20, seed=1))

    assert scarce > plentiful


def test_shrunk_covariance_is_positive_definite_when_sample_is_rank_deficient() -> None:
    # 10 observations, 20 assets: the sample covariance cannot have full rank.
    returns = _returns(10, 20, seed=2)

    shrunk = ledoit_wolf_covariance(returns)

    assert np.all(np.linalg.eigvalsh(shrunk.to_numpy()) > 0)


def test_shrunk_covariance_approaches_the_sample_when_data_is_plentiful() -> None:
    returns = _returns(20_000, 3, seed=3)

    shrunk = ledoit_wolf_covariance(returns).to_numpy()
    sample = sample_covariance(returns).to_numpy()

    # Intensity falls to ~2.7e-4 here, so the estimator is essentially the sample one.
    assert shrinkage_intensity(returns) < 0.01
    assert np.allclose(shrunk, sample, atol=1e-4)


def test_shrinkage_does_not_vanish_when_the_target_is_already_the_truth() -> None:
    # Documents a property that reads like a bug and is not one. For an equal-variance,
    # uncorrelated population the scaled-identity target IS the true covariance, so the
    # dispersion term measures nothing but estimation noise — the same quantity the noise
    # term measures. Their ratio therefore stays around 0.67 no matter how much data
    # arrives, whereas it decays to ~2.7e-4 on the correlated population above.
    plentiful_but_degenerate = _iid_returns(20_000, 3, seed=3)

    assert shrinkage_intensity(plentiful_but_degenerate) == pytest.approx(0.675, abs=0.02)


def test_degenerate_input_raises_instead_of_returning_a_broken_matrix() -> None:
    constant = pd.DataFrame({"A": [0.0] * 50, "B": [0.0] * 50})

    with pytest.raises(SingularCovarianceError):
        ledoit_wolf_covariance(constant)


def test_shrinkage_saturates_when_the_sample_is_mostly_noise() -> None:
    # At T=8, N=2 the noise term is about ten times the dispersion term, so the
    # min(noise, dispersion) truncation binds and the intensity clips at exactly 1.0.
    # This is the only test that exercises that clip: swapping min for max, or reversing
    # the operands, passes every other test in this file.
    returns = _iid_returns(8, 2, seed=1)

    assert shrinkage_intensity(returns) == 1.0

    # Full shrinkage means the estimate IS the scaled-identity target: no off-diagonal
    # term survives and both variances are equal.
    shrunk = ledoit_wolf_covariance(returns).to_numpy()
    assert shrunk[0, 1] == 0.0
    assert shrunk[0, 0] == pytest.approx(shrunk[1, 1])


def test_a_sample_already_equal_to_the_target_raises() -> None:
    # These four rows are centred and give S = 0.5 * I exactly, so the dispersion term
    # is exactly zero and the shrinkage intensity is undefined rather than merely small.
    already_the_target = pd.DataFrame(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], columns=["A", "B"]
    )

    with pytest.raises(SingularCovarianceError, match="identity target"):
        ledoit_wolf_covariance(already_the_target)


def test_a_single_observation_raises() -> None:
    single = pd.DataFrame([[0.01, 0.02]], columns=["A", "B"])

    with pytest.raises(SingularCovarianceError, match="two observations"):
        ledoit_wolf_covariance(single)


def test_similarly_volatile_uncorrelated_assets_are_accepted() -> None:
    # Regression guard for a scale bug. Two assets at the same 1% daily volatility with
    # zero correlation give a mean variance of ~1e-4 and a dispersion of ~1.3e-13 — below
    # an absolute 1e-12 floor, even though the input is entirely ordinary. The dispersion
    # guard must scale with mu**2, or this legitimate portfolio is rejected outright.
    equal_vol = _iid_returns(4_000, 2, seed=3)

    intensity = shrinkage_intensity(equal_vol)

    assert 0.0 <= intensity <= 1.0
    assert ledoit_wolf_covariance(equal_vol).shape == (2, 2)
