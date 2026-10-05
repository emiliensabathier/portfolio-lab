import io
from urllib.error import URLError

import pandas as pd
import pytest

import plab.cash as cash_module
from plab.cash import (
    CASH_TICKER,
    FRED_TIMEOUT_SECONDS,
    align,
    bond_equivalent_yield,
    fred_fetcher,
)
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


class _FakeUrlopen:
    """Stands in for ``urllib.request.urlopen``: serves canned bytes, records the call."""

    def __init__(self, body: str = "", error: Exception | None = None) -> None:
        self.body = body
        self.error = error
        self.timeout: float | None = None

    def __call__(self, url: str, timeout: float | None = None) -> io.BytesIO:
        self.timeout = timeout
        if self.error is not None:
            raise self.error
        return io.BytesIO(self.body.encode("utf-8"))


def _serve(monkeypatch, body: str = "", error: Exception | None = None) -> _FakeUrlopen:
    fake = _FakeUrlopen(body, error)
    monkeypatch.setattr(cash_module, "urlopen", fake)
    return fake


def test_a_day_fred_marks_as_missing_is_dropped_rather_than_filled(monkeypatch):
    """FRED writes '.' for a day with no observation. A dropped day is a day align can
    carry forward as accrual; a filled one would be a rate nobody quoted."""
    _serve(monkeypatch, "observation_date,DTB3\n2020-01-02,1.52\n2020-01-03,.\n2020-01-06,1.54\n")
    frame = fred_fetcher([CASH_TICKER], "2020-01-01", None)
    assert list(frame.index.date.astype(str)) == ["2020-01-02", "2020-01-06"]
    assert frame[CASH_TICKER].tolist() == [1.52, 1.54]


def test_the_requested_window_is_applied_because_fred_ignores_it(monkeypatch):
    """The CSV endpoint serves the whole history whatever dates are asked for, so a run
    given --end must be cut here or the bill would outrun the prices. ``end`` is
    inclusive, as it is for the price fetcher."""
    _serve(
        monkeypatch, "observation_date,DTB3\n2019-12-31,1.51\n2020-01-02,1.52\n2020-01-06,1.54\n"
    )
    frame = fred_fetcher([CASH_TICKER], "2020-01-01", "2020-01-02")
    assert frame[CASH_TICKER].tolist() == [1.52]


def test_a_window_with_no_observations_is_refused(monkeypatch):
    _serve(monkeypatch, "observation_date,DTB3\n2020-01-02,1.52\n")
    with pytest.raises(DataError, match="no observations"):
        fred_fetcher([CASH_TICKER], "2021-01-01", "2021-12-31")


def test_the_download_is_bounded_by_a_timeout(monkeypatch):
    """A hung connection must fail the run, not freeze it."""
    fake = _serve(monkeypatch, "observation_date,DTB3\n2020-01-02,1.52\n")
    fred_fetcher([CASH_TICKER], "2020-01-01", None)
    assert fake.timeout == FRED_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "error",
    [URLError("name resolution failed"), TimeoutError("timed out"), ConnectionResetError()],
)
def test_a_network_failure_is_raised_as_a_data_error_naming_the_series(monkeypatch, error):
    _serve(monkeypatch, error=error)
    with pytest.raises(DataError, match=CASH_TICKER):
        fred_fetcher([CASH_TICKER], "2020-01-01", None)


@pytest.mark.parametrize(
    "body",
    [
        "",
        "<html><body>Service unavailable</body></html>",
        "observation_date,DGS10\n2020-01-02,1.9\n",
    ],
)
def test_a_response_that_is_not_the_series_is_refused(monkeypatch, body):
    """An empty body, an HTML error page, or the wrong series must not be parsed into a
    rate: each is refused with the series named."""
    _serve(monkeypatch, body)
    with pytest.raises(DataError, match=CASH_TICKER):
        fred_fetcher([CASH_TICKER], "2020-01-01", None)


def test_a_series_with_no_numeric_value_at_all_is_refused(monkeypatch):
    _serve(monkeypatch, "observation_date,DTB3\n2020-01-02,.\n2020-01-03,.\n")
    with pytest.raises(DataError, match="no observations"):
        fred_fetcher([CASH_TICKER], "2020-01-01", None)


def test_the_fred_fetcher_refuses_a_batch():
    """One series per URL is what the endpoint offers; a silent partial fetch would be
    worse than the refusal."""
    with pytest.raises(DataError, match="one series at a time"):
        fred_fetcher([CASH_TICKER, "DGS10"], "2020-01-01", None)


@pytest.mark.network
def test_fred_serves_the_whole_backtest_window():
    """The bill has to cover the whole sample, from 2007, with no hole longer than a
    long weekend."""
    frame = fred_fetcher([CASH_TICKER], "2007-07-02", None)
    assert frame.index[0].year == 2007
    assert len(frame) > 4000
