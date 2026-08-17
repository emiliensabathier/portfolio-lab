import pandas as pd
import pytest

from plab.cash import align, bond_equivalent_yield
from plab.errors import DataError
from plab.risk.metrics import sharpe_ratio


def _sessions(count: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=count)


def test_the_discount_quote_is_converted_to_the_yield_actually_earned():
    """A bill quoted at 3.70% discount yields about 3.79% on the price paid.

    The discount is against face value over a 360-day year; the yield is against the price
    paid over 365. Worked by hand: 365 * 0.037 / (360 - 91 * 0.037) = 0.037868.
    """
    quoted = pd.Series([0.037], index=_sessions(1))
    assert float(bond_equivalent_yield(quoted).iloc[0]) == pytest.approx(0.0378680, abs=1e-7)


def test_the_conversion_raises_the_rate_it_never_lowers_it():
    quoted = pd.Series([0.0, 0.01, 0.05], index=_sessions(3))
    converted = bond_equivalent_yield(quoted)
    assert converted.iloc[0] == pytest.approx(0.0)
    assert (converted.iloc[1:] > quoted.iloc[1:]).all()


def test_a_rate_is_carried_across_a_holiday_the_equity_market_did_not_observe():
    """Carrying a bill rate forward one session is accrual, not an invented price."""
    index = _sessions(5)
    quoted = pd.Series([0.02] * 5, index=index)
    aligned = align(quoted.drop(index[2]), index)
    assert len(aligned) == 5
    assert aligned.loc[index[2]] == pytest.approx(0.02)


def test_a_long_gap_is_refused_rather_than_carried():
    index = _sessions(12)
    sparse = pd.Series([0.02, 0.02], index=[index[0], index[-1]])
    with pytest.raises(DataError, match="no bill quote"):
        align(sparse, index)


def test_a_window_starting_before_the_series_is_refused():
    """Extending a rate backwards would invent the number the ratio is measured against."""
    index = _sessions(6)
    late = pd.Series([0.02, 0.02], index=index[-2:])
    with pytest.raises(DataError, match="no bill quote"):
        align(late, index)


def test_the_error_names_the_window_it_could_not_cover():
    index = _sessions(10)
    late = pd.Series([0.02], index=[index[-1]])
    with pytest.raises(DataError) as excinfo:
        align(late, index)
    assert str(index[0].date()) in str(excinfo.value)


def test_a_constant_series_matches_the_scalar_it_is_made_of():
    """The two code paths through the excess-return helper must not disagree."""
    index = _sessions(300)
    returns = pd.Series([0.0004] * 300, index=index)
    rate = pd.Series([0.02] * 300, index=index)
    returns.iloc[::7] = -0.003
    assert sharpe_ratio(returns, rate) == pytest.approx(sharpe_ratio(returns, 0.02))


def test_a_higher_riskless_rate_lowers_the_sharpe():
    index = _sessions(300)
    returns = pd.Series([0.0004] * 300, index=index)
    returns.iloc[::7] = -0.003
    assert sharpe_ratio(returns, 0.05) < sharpe_ratio(returns, 0.0)


def test_a_return_date_with_no_rate_raises_rather_than_shortening_the_sample():
    """Silently dropping the day would compute the ratio over a window nobody chose."""
    index = _sessions(10)
    returns = pd.Series([0.001] * 10, index=index)
    rate = pd.Series([0.02] * 9, index=index[:9])
    with pytest.raises(ValueError, match="no risk-free quote"):
        sharpe_ratio(returns, rate)
