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


def test_guardian_accepts_sell_when_confidence_comes_from_one_minus_prob_buy() -> None:
    guardian = RiskGuardian()
    portfolio = Portfolio(capital_total=1000, capital_disponible=1000, drawdown_hoy=0.0, posiciones_abiertas=1)

    ok, reason = guardian.check(
        {"signal": "SELL", "risk_pct": 0.005, "prob_buy": 0.1, "spread_pct": 0.001},
        "BTC-USDT",
        portfolio,
    )

    assert ok is True
    assert reason == "OK"
