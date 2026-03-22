"""Tests del sizing dinamico fraccional."""

from __future__ import annotations

from execution.position_sizer import KellyFractionalSizer


def test_position_sizer_returns_base_notional_when_disabled() -> None:
    sizer = KellyFractionalSizer(
        enabled=False,
        base_notional_usd=25,
        min_notional_usd=25,
        max_notional_usd=50,
    )

    decision = sizer.size(
        prob_buy=0.9,
        buy_threshold=0.62,
        capital_total=10_000,
        capital_available=10_000,
    )

    assert decision.notional_usd == 25
    assert decision.used_dynamic is False
    assert decision.reason == "dynamic_sizing_disabled"


def test_position_sizer_scales_up_for_strong_signal() -> None:
    sizer = KellyFractionalSizer(
        enabled=True,
        base_notional_usd=25,
        min_notional_usd=25,
        max_notional_usd=50,
        max_capital_fraction=0.005,
        kelly_fraction=0.25,
    )

    decision = sizer.size(
        prob_buy=0.67,
        buy_threshold=0.62,
        capital_total=10_000,
        capital_available=10_000,
    )

    assert decision.notional_usd > 25
    assert decision.notional_usd <= 50
    assert decision.used_dynamic is True
    assert decision.raw_kelly_fraction > 0
    assert decision.applied_kelly_fraction > 0


def test_position_sizer_caps_to_available_cash() -> None:
    sizer = KellyFractionalSizer(
        enabled=True,
        base_notional_usd=25,
        min_notional_usd=25,
        max_notional_usd=50,
    )

    decision = sizer.size(
        prob_buy=0.8,
        buy_threshold=0.62,
        capital_total=10_000,
        capital_available=30,
    )

    assert decision.notional_usd == 30
