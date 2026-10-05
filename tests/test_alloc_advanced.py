import numpy as np
import pandas as pd
import pytest

from plab.alloc.rules import max_sharpe, min_variance, risk_parity
from plab.returns import simple_returns
from plab.risk.covariance import sample_covariance


def _history(
    n_days: int,
    vols: list[float],
    correlation: float,
    drifts: list[float] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
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


def _flat_rate(history: pd.DataFrame, level: float) -> pd.Series:
    return pd.Series(level, index=history.index, name="risk_free")


def test_max_sharpe_ranks_assets_on_return_in_excess_of_the_bill() -> None:
    # A0 is calm and earns about 2.5% a year, A1 is four times as volatile and earns about
    # 15%. Against zero, A0 has the better reward to risk; against a 4% bill, A0 no longer
    # beats cash at all and the tangency portfolio must move to A1. Measuring against zero
    # would pick the wrong portfolio whenever bills pay anything.
    history = _history(4_000, [0.0025, 0.01], 0.0, drifts=[0.0001, 0.0006], seed=5)

    against_zero = max_sharpe(history.index[-1], history)
    against_cash = max_sharpe(history.index[-1], history, risk_free=_flat_rate(history, 0.04))

    assert against_zero["A0"] > 0.5
    assert against_cash["A1"] > against_zero["A1"] + 0.3


def test_a_zero_bill_reproduces_the_raw_ratio() -> None:
    history = _history(2_000, [0.01, 0.02, 0.015], 0.3, drifts=[0.0003, 0.0005, 0.0], seed=6)

    raw = max_sharpe(history.index[-1], history)
    zero = max_sharpe(history.index[-1], history, risk_free=_flat_rate(history, 0.0))

    for ticker, weight in raw.items():
        assert zero[ticker] == pytest.approx(weight, abs=1e-6)


def test_max_sharpe_never_reads_a_bill_rate_beyond_the_decision_date() -> None:
    # The bill series is passed whole, so the rule itself must cut it at the decision date.
    history = _history(2_000, [0.0025, 0.01], 0.0, drifts=[0.0001, 0.0006], seed=7)
    date = history.index[-1]
    rate = _flat_rate(history, 0.01)
    future = pd.Series(0.50, index=pd.bdate_range(date + pd.Timedelta(days=1), periods=100))

    honest = max_sharpe(date, history, risk_free=rate)
    tempted = max_sharpe(date, history, risk_free=pd.concat([rate, future]))

    for ticker, weight in honest.items():
        assert tempted[ticker] == pytest.approx(weight, abs=1e-9)


def test_a_bill_series_that_does_not_cover_the_window_is_refused() -> None:
    from plab.errors import DataError

    history = _history(500, [0.01, 0.02], 0.0, seed=8)
    late = _flat_rate(history, 0.02).iloc[-10:]

    with pytest.raises(DataError, match="no bill quote"):
        max_sharpe(history.index[-1], history, risk_free=late)
