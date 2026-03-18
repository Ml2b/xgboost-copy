"""Tests del constructor de velas."""

from __future__ import annotations

from data.candle_builder import CandleBuilder


def test_two_trades_same_minute_produce_single_open_candle() -> None:
    builder = CandleBuilder()
    assert builder.add_trade("BTC-USDT", 100.0, 1.0, 60_000) is None
    assert builder.add_trade("BTC-USDT", 101.0, 2.0, 90_000) is None
    open_candle = builder._open_candles["BTC-USDT"]
    assert open_candle.trade_count == 2
    assert open_candle.volume == 3.0
    assert open_candle.high == 101.0


def test_new_minute_closes_previous_candle() -> None:
    builder = CandleBuilder()
    builder.add_trade("BTC-USDT", 100.0, 1.0, 60_000)
    closed = builder.add_trade("BTC-USDT", 102.0, 1.0, 120_000)
    assert closed is not None
    assert closed.is_closed is True
    assert closed.open == 100.0
    assert closed.close == 100.0


def test_force_close_returns_current_candle() -> None:
    builder = CandleBuilder()
    builder.add_trade("BTC-USDT", 100.0, 1.5, 60_000)
    closed = builder.force_close("BTC-USDT", 95_000)
    assert closed is not None
    assert closed.is_closed is True
    assert "BTC-USDT" not in builder._open_candles


def test_late_trade_does_not_rewind_open_candle() -> None:
    builder = CandleBuilder()
    builder.add_trade("BTC-USDT", 100.0, 1.0, 60_000)
    builder.add_trade("BTC-USDT", 102.0, 1.0, 120_000)

    assert builder.add_trade("BTC-USDT", 99.0, 0.5, 90_000) is None
    open_candle = builder._open_candles["BTC-USDT"]
    assert open_candle.open_time == 120_000
    assert open_candle.open == 102.0
