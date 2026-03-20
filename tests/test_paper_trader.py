"""Tests del paper trader controlado."""

from __future__ import annotations

import fakeredis.aioredis

from execution.position_exit import PositionExitPolicy
from paper.paper_trader import PaperTrader


def test_paper_trader_buys_and_opens_position() -> None:
    trader = PaperTrader(
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        initial_cash=20000,
        order_notional_usd=100,
        fee_pct=0.0,
        slippage_pct=0.0,
    )
    trader.handle_candle({"product_id": "BTC-USD", "close": "100"})
    event = trader.handle_signal(
        {
            "product_id": "BTC-USD",
            "signal": "BUY",
            "prob_buy": 0.9,
            "model_id": "m1",
            "registry_key": "btc_usdt",
            "actionable": "true",
        }
    )

    state = trader.current_state()
    assert event["decision"] == "paper_buy_filled"
    assert "BTC-USD" in trader.positions
    assert round(state.cash, 6) == 19900.0
    assert state.open_positions == 1


def test_paper_trader_sells_and_realizes_pnl() -> None:
    trader = PaperTrader(
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        initial_cash=20000,
        order_notional_usd=100,
        fee_pct=0.0,
        slippage_pct=0.0,
        exit_policy=PositionExitPolicy(stop_loss_pct=50.0, take_profit_pct=50.0, max_hold_minutes=10_000),
    )
    trader.handle_candle({"product_id": "BTC-USD", "close": "100"})
    trader.handle_signal(
        {
            "product_id": "BTC-USD",
            "signal": "BUY",
            "prob_buy": 0.9,
            "model_id": "m1",
            "registry_key": "btc_usdt",
            "actionable": "true",
        }
    )
    trader.handle_candle({"product_id": "BTC-USD", "close": "110"})
    event = trader.handle_signal(
        {
            "product_id": "BTC-USD",
            "signal": "EXIT_LONG",
            "prob_buy": 0.1,
            "model_id": "m1",
            "registry_key": "btc_usdt",
            "actionable": "true",
        }
    )

    state = trader.current_state()
    assert event["decision"] == "paper_exit_signal_filled"
    assert round(event["realized_pnl"], 6) == 10.0
    assert state.open_positions == 0
    assert round(state.cash, 6) == 20010.0


def test_paper_trader_ignores_non_actionable_signal() -> None:
    trader = PaperTrader(redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True))
    trader.handle_candle({"product_id": "SOL-USD", "close": "50"})
    event = trader.handle_signal(
        {
            "product_id": "SOL-USD",
            "signal": "BUY",
            "prob_buy": 0.95,
            "model_id": "m2",
            "registry_key": "sol_usdt",
            "actionable": "false",
            "reason": "no_promoted_model",
        }
    )

    assert event["decision"] == "ignored_non_actionable"
    assert event["reason"] == "no_promoted_model"


def test_paper_trader_blocks_buy_without_market_price() -> None:
    trader = PaperTrader(redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True))
    event = trader.handle_signal(
        {
            "product_id": "ETH-USD",
            "signal": "BUY",
            "prob_buy": 0.9,
            "model_id": "m3",
            "registry_key": "eth_usdt",
            "actionable": "true",
        }
    )

    assert event["decision"] == "blocked_no_market_price"


def test_paper_trader_applies_per_asset_slippage_override() -> None:
    trader = PaperTrader(
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        initial_cash=20000,
        order_notional_usd=100,
        fee_pct=0.0,
        slippage_pct=0.1,
        slippage_pct_by_base={"PENGU": 2.0},
    )
    trader.handle_candle({"product_id": "PENGU-USD", "close": "10"})
    event = trader.handle_signal(
        {
            "product_id": "PENGU-USD",
            "signal": "BUY",
            "prob_buy": 0.9,
            "model_id": "m4",
            "registry_key": "pengu_usd",
            "actionable": "true",
        }
    )

    assert event["decision"] == "paper_buy_filled"
    assert event["applied_slippage_pct"] == 2.0
    assert round(event["fill_price"], 6) == 10.2


def test_paper_trader_forces_exit_on_take_profit_rule() -> None:
    trader = PaperTrader(
        redis_client=fakeredis.aioredis.FakeRedis(decode_responses=True),
        initial_cash=20000,
        order_notional_usd=100,
        fee_pct=0.0,
        slippage_pct=0.0,
        exit_policy=PositionExitPolicy(stop_loss_pct=1.0, take_profit_pct=5.0, max_hold_minutes=120),
    )
    trader.handle_candle({"product_id": "BTC-USD", "close": "100", "close_time": "1000"})
    trader.handle_signal(
        {
            "product_id": "BTC-USD",
            "signal": "BUY",
            "prob_buy": 0.9,
            "model_id": "m1",
            "registry_key": "btc_usdt",
            "actionable": "true",
        }
    )
    event = trader.handle_candle({"product_id": "BTC-USD", "close": "106", "close_time": "61000"})

    assert event is not None
    assert event["decision"] == "paper_exit_rule_filled"
    assert event["reason"] == "take_profit_hit"
