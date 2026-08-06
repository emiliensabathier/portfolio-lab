import numpy as np
import pandas as pd
import pytest

from plab.risk.stress import HISTORICAL_SCENARIOS, parametric_shock, replay


def _returns(start: str, end: str, value: float = -0.001) -> pd.Series:
    index = pd.bdate_range(start, end)
    return pd.Series(value, index=index)


def test_scenarios_cover_the_three_documented_shocks() -> None:
    assert set(HISTORICAL_SCENARIOS) == {"gfc_2008", "covid_2020", "rates_2022"}


def test_replay_restricts_the_series_to_the_scenario_window() -> None:
    returns = _returns("2019-01-01", "2021-12-31")

    outcome = replay(returns, "covid_2020")

    start, end = HISTORICAL_SCENARIOS["covid_2020"]
    expected = len(pd.bdate_range(start, end))
    assert outcome["observations"] == pytest.approx(expected, abs=2)
    assert outcome["total_return"] < 0.0


def test_replay_reports_the_worst_single_day() -> None:
    index = pd.bdate_range("2020-02-19", "2020-03-23")
    values = np.full(len(index), -0.001)
    values[5] = -0.09
    returns = pd.Series(values, index=index)

    assert replay(returns, "covid_2020")["worst_day"] == pytest.approx(-0.09)


def test_replay_raises_when_the_series_does_not_cover_the_window() -> None:
    with pytest.raises(ValueError, match="gfc_2008"):
        replay(_returns("2015-01-01", "2016-01-01"), "gfc_2008")


def test_unknown_scenario_raises() -> None:
    with pytest.raises(KeyError):
        replay(_returns("2007-01-01", "2010-01-01"), "not_a_scenario")


def test_parametric_shock_is_the_weighted_sum_of_asset_shocks() -> None:
    loss = parametric_shock({"SPY": 0.6, "AGG": 0.4}, {"SPY": -0.20, "AGG": -0.05})

    assert loss == pytest.approx(0.6 * -0.20 + 0.4 * -0.05)


def test_parametric_shock_rejects_an_unshocked_asset() -> None:
    with pytest.raises(ValueError, match="GLD"):
        parametric_shock({"SPY": 0.5, "GLD": 0.5}, {"SPY": -0.20})
