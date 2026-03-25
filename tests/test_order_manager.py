"""Tests del ejecutor live separado."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import fakeredis.aioredis

from execution.order_manager import OrderManager
from execution.position_sizer import KellyFractionalSizer
from risk.guardian import RiskGuardian


class DummyCoinbaseClient:
    """Cliente Coinbase falso para validar decisiones sin tocar red."""

    def __init__(self, balances=None) -> None:
        self._balances = balances or {"USD": Decimal("10000"), "BTC": Decimal("0"), "SOL": Decimal("0")}
        self._bid = 99.0
        self._ask = 99.1
        self._mid = 99.05
        self._spread_pct = 0.001

    def get_account_balances(self):
        return dict(self._balances)

    def get_best_bid_ask(self, product_id: str):
        return {
            "product_id": product_id,
            "bid": self._bid,
            "ask": self._ask,
            "mid": self._mid,
            "spread_pct": self._spread_pct,
        }

    def get_best_bid_ask_public(self, product_id: str):
        return self.get_best_bid_ask(product_id)

    def get_product_snapshot(self, product_id: str):
        base, quote = product_id.split("-", 1)
        return type(
            "Snapshot",
            (),
            {
                "product_id": product_id,
                "base_asset": base,
                "quote_asset": quote,
                "base_min_size": Decimal("0.0001"),
            },
        )()

    def place_market_buy_quote(self, product_id: str, quote_size):
        return {"order_id": f"buy-{product_id}", "quote_size": str(quote_size)}

    def place_market_sell_base(self, product_id: str, base_size):
        return {"order_id": f"sell-{product_id}", "base_size": str(base_size)}

    def get_order(self, order_id: str):
        return {"order": {"order_id": order_id, "status": "FILLED"}}

    def set_market(self, *, bid: float, ask: float, spread_pct: float = 0.001) -> None:
        self._bid = bid
        self._ask = ask
        self._mid = (bid + ask) / 2.0
        self._spread_pct = spread_pct


def fixed_sizer(base_notional: float = 25.0) -> KellyFractionalSizer:
    return KellyFractionalSizer(
        enabled=False,
        base_notional_usd=base_notional,
        min_notional_usd=base_notional,
        max_notional_usd=base_notional,
    )


def test_order_manager_buy_uses_dry_run_fixed_notional() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        manager = OrderManager(
            redis_client=redis_client,
            coinbase_client=DummyCoinbaseClient(),
            execution_enabled=True,
            dry_run=True,
            order_notional_usd=25,
            allowed_bases=["BTC"],
            position_sizer=fixed_sizer(25),
        )
        event = await manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "BUY",
                "prob_buy": 0.9,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )

        assert event["decision"] == "accepted_dry_run"
        assert event["order_payload"]["quote_size"] == 25

    asyncio.run(run())


def test_order_manager_blocks_sell_without_inventory() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        manager = OrderManager(
            redis_client=redis_client,
            coinbase_client=DummyCoinbaseClient({"USD": Decimal("10000"), "BTC": Decimal("0")}),
            execution_enabled=True,
            dry_run=True,
            order_notional_usd=25,
            allowed_bases=["BTC"],
            position_sizer=fixed_sizer(25),
        )
        event = await manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "EXIT_LONG",
                "prob_buy": 0.1,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )

        assert event["decision"] == "ignored_no_position"

    asyncio.run(run())


def test_order_manager_ignores_sell_when_inventory_exists_but_is_not_managed() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        manager = OrderManager(
            redis_client=redis_client,
            coinbase_client=DummyCoinbaseClient({"USD": Decimal("10000"), "BTC": Decimal("0.5")}),
            execution_enabled=True,
            dry_run=True,
            order_notional_usd=25,
            allowed_bases=["BTC"],
            position_sizer=fixed_sizer(25),
        )
        event = await manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "EXIT_LONG",
                "prob_buy": 0.1,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )

        assert event["decision"] == "ignored_no_position"
        assert event["signal"] == "EXIT_LONG"

    asyncio.run(run())


def test_order_manager_can_close_dry_run_position_without_real_inventory() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        manager = OrderManager(
            redis_client=redis_client,
            coinbase_client=DummyCoinbaseClient({"USD": Decimal("10000"), "BTC": Decimal("0")}),
            execution_enabled=True,
            dry_run=True,
            order_notional_usd=25,
            allowed_bases=["BTC"],
            position_sizer=fixed_sizer(25),
        )
        await manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "BUY",
                "prob_buy": 0.9,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )
        event = await manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "EXIT_LONG",
                "prob_buy": 0.1,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )

        assert event["decision"] == "accepted_dry_run"
        assert event["signal"] == "EXIT_LONG"

    asyncio.run(run())


def test_order_manager_restores_managed_position_from_redis_state() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        first_manager = OrderManager(
            redis_client=redis_client,
            coinbase_client=DummyCoinbaseClient({"USD": Decimal("10000"), "BTC": Decimal("0")}),
            execution_enabled=True,
            dry_run=True,
            order_notional_usd=25,
            allowed_bases=["BTC"],
            position_sizer=fixed_sizer(25),
        )
        await first_manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "BUY",
                "prob_buy": 0.9,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )

        second_manager = OrderManager(
            redis_client=redis_client,
            coinbase_client=DummyCoinbaseClient({"USD": Decimal("10000"), "BTC": Decimal("0")}),
            execution_enabled=True,
            dry_run=True,
            order_notional_usd=25,
            allowed_bases=["BTC"],
            position_sizer=fixed_sizer(25),
        )
        event = await second_manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "EXIT_LONG",
                "prob_buy": 0.1,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )

        assert event["decision"] == "accepted_dry_run"
        assert event["signal"] == "EXIT_LONG"

    asyncio.run(run())


def test_order_manager_ignores_non_actionable_assets() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        manager = OrderManager(
            redis_client=redis_client,
            coinbase_client=DummyCoinbaseClient(),
            execution_enabled=True,
            dry_run=True,
            order_notional_usd=25,
            allowed_bases=["BTC"],
            position_sizer=fixed_sizer(25),
        )
        event = await manager.handle_signal(
            {
                "product_id": "SOL-USD",
                "signal": "BUY",
                "prob_buy": 0.9,
                "model_id": "m2",
                "registry_key": "sol_usdt",
                "actionable": "false",
                "reason": "no_promoted_model",
            }
        )

        assert event["decision"] == "ignored_non_actionable"
        assert event["reason"] == "no_promoted_model"

    asyncio.run(run())


def test_order_manager_blocks_rebuy_when_spot_inventory_exists() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        manager = OrderManager(
            redis_client=redis_client,
            coinbase_client=DummyCoinbaseClient({"USD": Decimal("10000"), "BTC": Decimal("0.25")}),
            execution_enabled=True,
            dry_run=True,
            order_notional_usd=25,
            allowed_bases=["BTC"],
        )
        event = await manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "BUY",
                "prob_buy": 0.9,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )

        assert event["decision"] == "blocked_existing_position"

    asyncio.run(run())


def test_order_manager_enforces_cooldown_per_asset() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        manager = OrderManager(
            redis_client=redis_client,
            coinbase_client=DummyCoinbaseClient(),
            execution_enabled=True,
            dry_run=True,
            order_notional_usd=25,
            cooldown_seconds=60,
            allowed_bases=["BTC"],
            position_sizer=fixed_sizer(25),
        )
        first = await manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "BUY",
                "prob_buy": 0.9,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )
        second = await manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "BUY",
                "prob_buy": 0.91,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )

        assert first["decision"] == "accepted_dry_run"
        assert second["decision"] == "cooldown_active"

    asyncio.run(run())


def test_order_manager_dynamic_sizer_scales_notional_for_strong_signal() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        manager = OrderManager(
            redis_client=redis_client,
            coinbase_client=DummyCoinbaseClient(),
            execution_enabled=True,
            dry_run=True,
            order_notional_usd=25,
            allowed_bases=["BTC"],
        )
        event = await manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "BUY",
                "prob_buy": 0.67,
                "buy_threshold": 0.62,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )

        assert event["decision"] == "accepted_dry_run"
        assert event["order_payload"]["quote_size"] > 25
        assert event["sizing_dynamic"] == "true"
        assert event["sizing_reason"] == "kelly_fractional_dynamic"

    asyncio.run(run())


def test_order_manager_blocks_reentry_after_stop_loss_cluster() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        guardian = RiskGuardian(
            chop_caution_exit_count=5,
            chop_lock_exit_count=5,
            chop_lock_stop_count=1,
            chop_block_minutes=30,
            chop_window_minutes=120,
        )
        coinbase_client = DummyCoinbaseClient()
        coinbase_client.set_market(bid=99.9, ask=100.0)
        manager = OrderManager(
            redis_client=redis_client,
            coinbase_client=coinbase_client,
            guardian=guardian,
            execution_enabled=True,
            dry_run=True,
            order_notional_usd=25,
            cooldown_seconds=0,
            allowed_bases=["BTC"],
            position_sizer=fixed_sizer(25),
        )

        first = await manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "BUY",
                "prob_buy": 0.9,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )
        coinbase_client.set_market(bid=98.5, ask=98.6)
        stopped = await manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "HOLD",
                "prob_buy": 0.1,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "false",
            }
        )
        coinbase_client.set_market(bid=99.9, ask=100.0)
        second = await manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "BUY",
                "prob_buy": 0.92,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )

        assert first["decision"] == "accepted_dry_run"
        assert stopped["decision"] == "accepted_dry_run"
        assert stopped["reason"] == "stop_loss_hit"
        assert second["decision"] == "blocked_risk"
        assert "CHOP" in second["reason"]

    asyncio.run(run())


def test_order_manager_scales_notional_when_asset_is_in_caution_mode() -> None:
    async def run() -> None:
        import time

        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        guardian = RiskGuardian(
            min_signal_prob=0.62,
            chop_caution_exit_count=1,
            chop_lock_exit_count=5,
            chop_lock_stop_count=5,
            chop_probability_buffer=0.02,
            chop_notional_scale=0.5,
        )
        now_ms = int(time.time() * 1000)
        guardian.register_exit(
            "BTC-USD",
            reason="signal_exit_long",
            pnl_pct=-0.3,
            timestamp_ms=now_ms,
            holding_minutes=4.0,
        )
        manager = OrderManager(
            redis_client=redis_client,
            coinbase_client=DummyCoinbaseClient(),
            guardian=guardian,
            execution_enabled=True,
            dry_run=True,
            order_notional_usd=25,
            cooldown_seconds=0,
            allowed_bases=["BTC"],
            position_sizer=fixed_sizer(25),
        )
        event = await manager.handle_signal(
            {
                "product_id": "BTC-USD",
                "signal": "BUY",
                "prob_buy": 0.9,
                "model_id": "m1",
                "registry_key": "btc_usdt",
                "actionable": "true",
            }
        )

        assert event["decision"] == "accepted_dry_run"
        assert event["order_payload"]["quote_size"] == 12.5
        assert "chop_caution" in event["sizing_reason"]

    asyncio.run(run())
