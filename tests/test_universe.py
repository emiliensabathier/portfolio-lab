import pandas as pd

from plab.universe import CORE_START, ETF_CORE, ETF_LONG, LONG_START, tickers


def test_core_universe_has_ten_assets_across_three_blocs() -> None:
    assert len(ETF_CORE) == 10
    assert {asset.bloc for asset in ETF_CORE} == {"equity", "rates", "real"}


def test_every_core_asset_is_listed_before_core_start() -> None:
    start = pd.Timestamp(CORE_START)
    for asset in ETF_CORE:
        assert pd.Timestamp(asset.inception) <= start, asset.ticker


def test_tickers_preserves_declaration_order() -> None:
    assert tickers(ETF_CORE)[0] == "SPY"
    assert len(tickers(ETF_CORE)) == 10


def test_the_long_universe_drops_only_the_two_late_listings() -> None:
    assert set(tickers(ETF_CORE)) - set(tickers(ETF_LONG)) == {"HYG", "DBC"}
    assert set(ETF_LONG) <= set(ETF_CORE)
    assert {asset.bloc for asset in ETF_LONG} == {"equity", "rates", "real"}


def test_the_long_start_is_the_last_inception_in_the_long_universe() -> None:
    assert LONG_START == max(asset.inception for asset in ETF_LONG)


def test_three_years_of_estimation_from_the_long_start_leave_2008_out_of_sample() -> None:
    first_decision = pd.Timestamp(LONG_START) + pd.DateOffset(months=36)
    assert first_decision < pd.Timestamp("2008-09-15")  # Lehman
