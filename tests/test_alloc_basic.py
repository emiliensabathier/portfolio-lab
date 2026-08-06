import numpy as np
import pandas as pd
import pytest

from plab.alloc.rules import OptimizerError, equal_weight, fixed_weights, min_variance
from plab.returns import simple_returns
from plab.risk.covariance import sample_covariance


def _history(n_days: int, vols: list[float], correlation: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(vols)
    corr = np.full((n, n), correlation)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    steps = rng.multivariate_normal(np.zeros(n), cov, n_days)
    levels = 100.0 * np.exp(np.cumsum(steps, axis=0))
    index = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(levels, index=index, columns=[f"A{i}" for i in range(n)])


def test_equal_weight_splits_evenly_and_sums_to_one() -> None:
    weights = equal_weight(pd.Timestamp("2020-01-31"), _history(100, [0.01, 0.02], 0.0))

    assert weights == {"A0": 0.5, "A1": 0.5}


def test_fixed_weights_ignores_history_and_returns_the_mapping() -> None:
    strategy = fixed_weights({"A0": 0.6, "A1": 0.4})

    assert strategy(pd.Timestamp("2020-01-31"), _history(100, [0.01, 0.02], 0.0)) == {
        "A0": 0.6,
        "A1": 0.4,
    }


def test_fixed_weights_rejects_a_mapping_that_does_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum"):
        fixed_weights({"A0": 0.6, "A1": 0.1})


def test_min_variance_matches_the_two_asset_analytic_solution() -> None:
    history = _history(6_000, [0.01, 0.02], 0.3, seed=1)
    cov = sample_covariance(pd.DataFrame(history).pct_change().iloc[1:]).to_numpy()

    v0, v1, c = cov[0, 0], cov[1, 1], cov[0, 1]
    expected_w0 = (v1 - c) / (v0 + v1 - 2 * c)

    weights = min_variance(history.index[-1], history, estimator=sample_covariance)

    assert weights["A0"] == pytest.approx(expected_w0, abs=1e-3)


def test_min_variance_overweights_the_calmer_asset() -> None:
    history = _history(3_000, [0.005, 0.03], 0.0, seed=2)

    weights = min_variance(history.index[-1], history)

    assert weights["A0"] > weights["A1"]


def test_min_variance_is_long_only_and_fully_invested() -> None:
    # A contract check, not an optimality check: _as_dict guarantees both properties by
    # construction. The two tests below are what actually constrain the optimizer.
    history = _history(2_000, [0.01, 0.02, 0.015], 0.5, seed=3)

    weights = min_variance(history.index[-1], history)

    assert all(weight >= -1e-9 for weight in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)


def test_min_variance_achieves_lower_variance_than_equal_weight() -> None:
    # The genuine optimality property. It fails if the objective is wrong in any way that
    # still returns a feasible point, which the long-only and fully-invested assertions
    # above cannot detect.
    history = _history(3_000, [0.005, 0.03, 0.015], 0.2, seed=5)
    covariance = sample_covariance(simple_returns(history))

    def variance_of(weights: dict[str, float]) -> float:
        vector = np.array([weights[ticker] for ticker in covariance.columns])
        return float(vector @ covariance.to_numpy() @ vector)

    optimized = min_variance(history.index[-1], history, estimator=sample_covariance)

    assert variance_of(optimized) < variance_of(equal_weight(history.index[-1], history))


def test_a_non_convergent_optimizer_raises_instead_of_falling_back(monkeypatch) -> None:
    # The most important line in this module. A failed optimization must never be rescued
    # with equal weight: that would publish equal-weight results under a minimum-variance
    # label. Without this test, a future edit could reintroduce such a fallback and the
    # whole suite would stay green.
    class FailedResult:
        success = False
        message = "simulated non-convergence"
        x = np.array([0.5, 0.5])

    monkeypatch.setattr("plab.alloc.rules.minimize", lambda *args, **kwargs: FailedResult())

    with pytest.raises(OptimizerError, match="did not converge"):
        min_variance(pd.Timestamp("2020-01-01"), _history(200, [0.01, 0.02], 0.0))
