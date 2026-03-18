"""Tests de construccion de target."""

from __future__ import annotations

import pytest

from target.builder import TargetBuilder, TargetConfig, TargetType
from tests.helpers import make_synthetic_candles


def test_target_balance_stays_between_20_and_80_pct() -> None:
    df = make_synthetic_candles(240, seed=7)
    config = TargetConfig(
        target_type=TargetType.NET_RETURN_THRESHOLD,
        horizon=5,
        threshold_pct=0.04,
        fee_pct=0.005,
    )
    result = TargetBuilder(config).build(df)
    balance = float(result["target"].mean())
    assert 0.20 <= balance <= 0.80


def test_target_config_raises_if_fee_exceeds_threshold() -> None:
    with pytest.raises(ValueError):
        TargetConfig(
            target_type=TargetType.NET_RETURN_THRESHOLD,
            horizon=5,
            threshold_pct=0.05,
            fee_pct=0.03,
        )

