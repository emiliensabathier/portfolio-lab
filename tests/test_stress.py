import numpy as np
import pandas as pd
import pytest

from plab.risk.stress import HISTORICAL_SCENARIOS, replay


def _returns(start: str, end: str, value: float = -0.001) -> pd.Series:
    index = pd.bdate_range(start, end)
    return pd.Series(value, index=index)


def test_scenarios_cover_the_three_documented_shocks() -> None:
    assert set(HISTORICAL_SCENARIOS) == {"gfc_2008", "covid_2020", "rates_2022"}


def test_replay_restricts_the_series_to_the_scenario_window() -> None:
    returns = _returns("2019-01-01", "2021-12-31")

    outcome = replay(returns, "covid_2020")

    # Exact, not approximate: both endpoints are business days and the outer series is a
    # bdate_range, so label slicing yields precisely len(bdate_range(start, end)) — 24
    # observations here. A tolerance would let an inclusive/exclusive boundary slip
    # through, which is the one thing this test exists to pin down.
    start, end = HISTORICAL_SCENARIOS["covid_2020"]
    assert outcome["observations"] == len(pd.bdate_range(start, end))
    assert outcome["total_return"] < 0.0


def test_replay_reports_the_worst_single_day() -> None:
    index = pd.bdate_range("2020-02-19", "2020-03-23")
    values = np.full(len(index), -0.001)
    values[5] = -0.09
    returns = pd.Series(values, index=index)

    assert replay(returns, "covid_2020")["worst_day"] == pytest.approx(-0.09)


def test_replay_reports_the_drawdown_local_to_the_window() -> None:
    # The max_drawdown key is consumed by the report in Task 12, so it needs an assertion
    # of its own. The fixture deliberately extends well beyond the scenario and hides a
    # DEEPER crash outside it: a series confined to the window would make `window` and
    # `returns` the same object, and passing the unwindowed series to max_drawdown — the
    # exact wiring bug this guards against — would still produce the expected number.
    returns = pd.Series(0.0, index=pd.bdate_range("2020-01-02", "2020-04-30"))
    returns.loc["2020-01-15"] = -0.50

    window = pd.bdate_range("2020-02-19", "2020-03-23")
    returns.loc[window[3]] = -0.25
    returns.loc[window[4]] = 0.10

    outcome = replay(returns, "covid_2020")

    # Inside the window the wealth path is 1.0 -> 0.75 -> 0.825, so the drawdown is -0.25.
    # Across the whole series it would be -0.625, which is what a wrong slice would report.
    assert outcome["max_drawdown"] == pytest.approx(-0.25)


def test_replay_raises_when_the_series_does_not_cover_the_window() -> None:
    with pytest.raises(ValueError, match="gfc_2008"):
        replay(_returns("2015-01-01", "2016-01-01"), "gfc_2008")


def test_unknown_scenario_raises() -> None:
    with pytest.raises(KeyError):
        replay(_returns("2007-01-01", "2010-01-01"), "not_a_scenario")
