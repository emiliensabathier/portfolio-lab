import json
from pathlib import Path

import pandas as pd
import pytest

from plab.backtest.engine import BacktestConfig
from plab.pipeline import run
from plab.risk.metrics import summary

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = BacktestConfig(estimation_months=36, cost_bps=5.0)


@pytest.fixture(scope="module")
def frozen_prices() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "prices.csv", index_col=0, parse_dates=True)


@pytest.fixture(scope="module")
def expected() -> dict[str, dict[str, float]]:
    return json.loads((FIXTURES / "expected_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def expected_stress() -> dict[str, dict[str, dict[str, float]]]:
    return json.loads((FIXTURES / "expected_stress.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def output(frozen_prices: pd.DataFrame):
    """Run the pipeline once for the whole module."""
    return run(frozen_prices, CONFIG)


def test_the_fixture_covers_the_documented_window(frozen_prices: pd.DataFrame) -> None:
    # The first trading day on or after CORE_START, whichever the exchange calendar gives.
    assert frozen_prices.index.min() <= pd.Timestamp("2007-07-06")
    assert frozen_prices.index.min() >= pd.Timestamp("2007-07-02")
    assert len(frozen_prices.columns) == 10


def test_out_of_sample_period_starts_after_the_estimation_window(output) -> None:
    first = min(result.returns.index.min() for result in output.results.values())
    assert first >= pd.Timestamp("2010-07-01")


# Deterministic strategies must reproduce exactly. Optimizer-driven ones go through SLSQP,
# whose last digits depend on the BLAS build, so they get a looser but still meaningful
# tolerance: a real methodology change moves these metrics by percent, not by 1e-4.
EXACT = {"equal_weight", "60/40"}
TOLERANCE_EXACT = 1e-9
TOLERANCE_OPTIMIZED = 1e-4


def test_every_stress_outcome_matches_the_frozen_reference(
    output, expected_stress: dict[str, dict[str, dict[str, float]]]
) -> None:
    # The headline metrics alone would not have caught this project's worst defect: a
    # report that displayed a stress table while silently omitting the 2008 crisis it
    # claimed to cover. Pinning these numbers means a shrunken observation count, or a
    # scenario that quietly disappears, fails here rather than in a reader's hands.
    assert set(output.stress) == set(expected_stress)

    for name, scenarios in output.stress.items():
        tolerance = TOLERANCE_EXACT if name in EXACT else TOLERANCE_OPTIMIZED
        assert set(scenarios) == set(expected_stress[name]), name
        for scenario, outcome in scenarios.items():
            reference = expected_stress[name][scenario]
            for metric, value in outcome.items():
                assert value == pytest.approx(reference[metric], rel=tolerance), (
                    name,
                    scenario,
                    metric,
                )


def test_every_metric_matches_the_frozen_reference(
    output, expected: dict[str, dict[str, float]]
) -> None:
    for name, result in output.results.items():
        actual = summary(result.returns)
        tolerance = TOLERANCE_EXACT if name in EXACT else TOLERANCE_OPTIMIZED
        for metric, reference in expected[name].items():
            assert actual[metric] == pytest.approx(reference, rel=tolerance), (
                f"{name}.{metric}"
            )
