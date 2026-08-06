import numpy as np
import pandas as pd
import pytest

from plab.alloc.rules import max_sharpe, min_variance, risk_parity
from plab.returns import simple_returns
from plab.risk.covariance import sample_covariance


def _history(n_days: int, vols: list[float], correlation: float,
             drifts: list[float] | None = None, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(vols)
    corr = np.full((n, n), correlation)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    mean = np.array(drifts if drifts is not None else [0.0] * n)
    steps = rng.multivariate_normal(mean, cov, n_days)
    levels = 100.0 * np.exp(np.cumsum(steps, axis=0))
    index = pd.bdate_range("2010-01-01", periods=n_days)
    return pd.DataFrame(levels, index=index, columns=[f"A{i}" for i in range(n)])


def _risk_contributions(weights: dict[str, float], cov: pd.DataFrame) -> np.ndarray:
    w = np.array([weights[c] for c in cov.columns])
    sigma = cov.to_numpy()
    portfolio_vol = float(np.sqrt(w @ sigma @ w))
    return w * (sigma @ w) / portfolio_vol


def test_risk_parity_equalizes_risk_contributions() -> None:
    history = _history(3_000, [0.005, 0.02, 0.01], 0.2, seed=1)

    weights = risk_parity(history.index[-1], history, estimator=sample_covariance)
    cov = sample_covariance(simple_returns(history))
    contributions = _risk_contributions(weights, cov)

    assert contributions.max() - contributions.min() < 0.01 * contributions.mean() + 1e-6


def test_risk_parity_sits_between_equal_weight_and_min_variance_on_the_calm_asset() -> None:
    history = _history(3_000, [0.005, 0.03], 0.0, seed=2)

    parity = risk_parity(history.index[-1], history)["A0"]
    lowest_variance = min_variance(history.index[-1], history)["A0"]

    assert 0.5 < parity < lowest_variance


def test_max_sharpe_favours_the_asset_with_the_better_reward_to_risk() -> None:
    history = _history(4_000, [0.01, 0.01], 0.0, drifts=[0.0008, 0.0], seed=3)

    weights = max_sharpe(history.index[-1], history)

    assert weights["A0"] > weights["A1"]


def test_advanced_rules_are_long_only_and_fully_invested() -> None:
    history = _history(2_000, [0.01, 0.02, 0.015], 0.4, seed=4)

    for rule in (risk_parity, max_sharpe):
        weights = rule(history.index[-1], history)
        assert all(weight >= -1e-9 for weight in weights.values()), rule.__name__
        assert sum(weights.values()) == pytest.approx(1.0), rule.__name__
