"""Tests del guardian de riesgo."""

from __future__ import annotations

from datetime import datetime, timezone

from risk.guardian import Portfolio, RiskGuardian


def test_guardian_blocks_daily_drawdown_limit() -> None:
    guardian = RiskGuardian()
    portfolio = Portfolio(capital_total=1000, capital_disponible=1000, drawdown_hoy=0.02, posiciones_abiertas=0)
    ok, reason = guardian.check({"risk_pct": 0.005, "prob_buy": 0.9, "spread_pct": 0.001}, "BTC-USDT", portfolio)
    assert ok is False
    assert "DRAWDOWN" in reason


def test_guardian_allows_signal_when_everything_is_ok() -> None:
    guardian = RiskGuardian()
    portfolio = Portfolio(capital_total=1000, capital_disponible=1000, drawdown_hoy=0.0, posiciones_abiertas=0)
    ok, reason = guardian.check({"risk_pct": 0.005, "prob_buy": 0.9, "spread_pct": 0.001}, "BTC-USDT", portfolio)
    assert ok is True
    assert reason == "OK"


def test_guardian_each_checkpoint_can_block_independently() -> None:
    event_time = datetime(2026, 3, 17, 14, 0, tzinfo=timezone.utc)
    guardian = RiskGuardian(macro_event_times_utc=[event_time.isoformat()])
    portfolio = Portfolio(capital_total=1000, capital_disponible=1000, drawdown_hoy=0.0, posiciones_abiertas=0)

    assert guardian.check({"risk_pct": 0.02, "prob_buy": 0.9, "spread_pct": 0.001}, "BTC-USDT", portfolio)[0] is False
    assert guardian.check({"risk_pct": 0.005, "prob_buy": 0.9, "spread_pct": 0.01}, "BTC-USDT", portfolio)[0] is False
    assert guardian.check({"risk_pct": 0.005, "prob_buy": 0.5, "spread_pct": 0.001}, "BTC-USDT", portfolio)[0] is False
    assert guardian.check(
        {
            "risk_pct": 0.005,
            "prob_buy": 0.9,
            "spread_pct": 0.001,
            "timestamp_ms": int(event_time.timestamp() * 1000),
        },
        "BTC-USDT",
        portfolio,
    )[0] is False


def test_guardian_allows_exit_long_even_with_entry_filters_violated() -> None:
    guardian = RiskGuardian()
    portfolio = Portfolio(capital_total=1000, capital_disponible=1000, drawdown_hoy=0.0, posiciones_abiertas=1)

    ok, reason = guardian.check(
        {"signal": "EXIT_LONG", "risk_pct": 0.05, "prob_buy": 0.95, "spread_pct": 0.5},
        "BTC-USDT",
        portfolio,
    )

    assert ok is True
    assert reason == "OK"


def test_guardian_enters_caution_mode_after_repeated_fast_losing_exits() -> None:
    guardian = RiskGuardian(
        min_signal_prob=0.62,
        chop_caution_exit_count=2,
        chop_lock_exit_count=5,
        chop_lock_stop_count=5,
        chop_probability_buffer=0.05,
        chop_window_minutes=120,
    )
    portfolio = Portfolio(capital_total=1000, capital_disponible=1000, drawdown_hoy=0.0, posiciones_abiertas=0)

    guardian.register_exit(
        "TAO-USD",
        reason="signal_exit_long",
        pnl_pct=-0.35,
        timestamp_ms=1_000,
        holding_minutes=4.0,
    )
    guardian.register_exit(
        "TAO-USD",
        reason="signal_exit_long",
        pnl_pct=-0.22,
        timestamp_ms=2_000,
        holding_minutes=6.0,
    )

    ok, reason = guardian.check(
        {"risk_pct": 0.005, "prob_buy": 0.64, "spread_pct": 0.001, "timestamp_ms": 3_000},
        "TAO-USD",
        portfolio,
    )

    assert ok is False
    assert "CHOP" in reason
    assert "prob_buy" in reason


def test_guardian_blocks_asset_temporarily_after_repeated_stop_losses() -> None:
    guardian = RiskGuardian(
        chop_caution_exit_count=5,
        chop_lock_exit_count=5,
        chop_lock_stop_count=2,
        chop_block_minutes=30,
        chop_window_minutes=120,
    )
    portfolio = Portfolio(capital_total=1000, capital_disponible=1000, drawdown_hoy=0.0, posiciones_abiertas=0)

    guardian.register_exit(
        "TAO-USD",
        reason="stop_loss_hit",
        pnl_pct=-1.1,
        timestamp_ms=1_000,
        holding_minutes=8.0,
    )
    guardian.register_exit(
        "TAO-USD",
        reason="stop_loss_hit",
        pnl_pct=-1.3,
        timestamp_ms=2_000,
        holding_minutes=9.0,
    )

    blocked, reason = guardian.check(
        {"risk_pct": 0.005, "prob_buy": 0.9, "spread_pct": 0.001, "timestamp_ms": 3_000},
        "TAO-USD",
        portfolio,
    )
    released, release_reason = guardian.check(
        {"risk_pct": 0.005, "prob_buy": 0.9, "spread_pct": 0.001, "timestamp_ms": (2_000 + 31 * 60_000)},
        "TAO-USD",
        portfolio,
    )

    assert blocked is False
    assert "CHOP" in reason
    assert released is True
    assert release_reason == "OK"


def test_guardian_entry_adjustment_scales_size_in_caution_mode() -> None:
    guardian = RiskGuardian(
        min_signal_prob=0.62,
        chop_caution_exit_count=1,
        chop_lock_exit_count=5,
        chop_lock_stop_count=5,
        chop_probability_buffer=0.04,
        chop_risk_scale=0.35,
        chop_notional_scale=0.6,
    )

    guardian.register_exit(
        "TAO-USD",
        reason="signal_exit_long",
        pnl_pct=-0.25,
        timestamp_ms=1_000,
        holding_minutes=5.0,
    )
    adjustment = guardian.entry_adjustment("TAO-USD", timestamp_ms=2_000)

    assert adjustment.cautious is True
    assert adjustment.blocked is False
    assert adjustment.min_signal_prob == 0.66
    assert adjustment.notional_scale == 0.6
