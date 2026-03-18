"""Tests del calculador de features."""

from __future__ import annotations

from features.calculator import FEATURE_COLUMNS, FeatureCalculator
from tests.helpers import make_synthetic_candles


def test_calculator_creates_all_expected_columns() -> None:
    df = make_synthetic_candles(220, seed=11)
    result = FeatureCalculator().compute(df)
    for column in FEATURE_COLUMNS:
        assert column in result.columns


def test_calculator_output_has_no_nan_values() -> None:
    df = make_synthetic_candles(220, seed=13)
    result = FeatureCalculator().compute(df)
    assert result[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_bollinger_pct_stays_between_zero_and_one() -> None:
    df = make_synthetic_candles(220, seed=17)
    result = FeatureCalculator().compute(df)
    assert ((result["bb_pct"] >= 0.0) & (result["bb_pct"] <= 1.0)).all()

