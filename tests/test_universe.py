import pandas as pd

from plab.universe import CORE_START, ETF_CORE, tickers


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
