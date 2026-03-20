"""Tests del ejecutor live separado."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import fakeredis.aioredis

from execution.order_manager import OrderManager


class DummyCoinbaseClient:
    """Cliente Coinbase falso para validar decisiones sin tocar red."""

    def __init__(self, balances=None) -> None:
        self._balances = balances or {"USD": Decimal("10000"), "BTC": Decimal("0"), "SOL": Decimal("0")}

    def get_account_balances(self):
        return dict(self._balances)

    def get_best_bid_ask(self, product_id: str):
        return {"product_id": product_id, "bid": 99.0, "ask": 99.1, "mid": 99.05, "spread_pct": 0.001}

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

        assert event["decision"] == "blocked_no_inventory"

    asyncio.run(run())


def test_order_manager_accepts_sell_close_when_inventory_exists() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        manager = OrderManager(
            redis_client=redis_client,
            coinbase_client=DummyCoinbaseClient({"USD": Decimal("10000"), "BTC": Decimal("0.5")}),
            execution_enabled=True,
            dry_run=True,
            order_notional_usd=25,
            allowed_bases=["BTC"],
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
        assert event["order_payload"]["side"] == "SELL"
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
