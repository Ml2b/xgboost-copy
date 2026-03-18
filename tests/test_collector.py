"""Tests del collector Coinbase."""

from __future__ import annotations

import asyncio
import json

import fakeredis.aioredis

from config import settings
from data.collector import CollectorWithCandles


class DummyCoinbaseClient:
    """Cliente Coinbase minimo para pruebas de auth y reconciliacion."""

    def __init__(self, candles: list[dict[str, object]] | None = None) -> None:
        self.candles = candles or []

    def has_private_credentials(self) -> bool:
        return True

    def build_ws_jwt(self) -> str:
        return "jwt-token"

    def get_candles(
        self,
        product_id: str,
        start: int,
        end: int,
        granularity: str = "ONE_MINUTE",
        limit: int | None = None,
        prefer_private: bool = True,
    ) -> list[dict[str, object]]:
        return [candle for candle in self.candles if candle["product_id"] == product_id]


def test_collector_ignores_trade_id_jumps_when_message_sequence_is_contiguous() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        collector = CollectorWithCandles(
            redis_client=redis_client,
            products=["BTC-USD"],
            products_per_connection=1,
        )

        await collector._handle_message(
            {
                "channel": "market_trades",
                "sequence_num": 100,
                "timestamp": "2026-03-18T00:00:00Z",
                "events": [
                    {
                        "type": "snapshot",
                        "trades": [
                            {
                                "trade_id": "10",
                                "product_id": "BTC-USD",
                                "price": "100.0",
                                "size": "0.1",
                                "side": "BUY",
                                "time": "2026-03-18T00:00:00Z",
                            }
                        ],
                    }
                ],
            },
            connection_key="BTC-USD",
        )
        await collector._handle_message(
            {
                "channel": "market_trades",
                "sequence_num": 101,
                "timestamp": "2026-03-18T00:00:01Z",
                "events": [
                    {
                        "type": "update",
                        "trades": [
                            {
                                "trade_id": "5000",
                                "product_id": "BTC-USD",
                                "price": "101.0",
                                "size": "0.2",
                                "side": "SELL",
                                "time": "2026-03-18T00:00:01Z",
                            }
                        ],
                    }
                ],
            },
            connection_key="BTC-USD",
        )

        assert collector.stats.gaps == 0
        assert await redis_client.xlen(settings.STREAM_SYSTEM_ERRORS) == 0

    asyncio.run(run())


def test_collector_detects_gap_using_message_sequence_num() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        collector = CollectorWithCandles(
            redis_client=redis_client,
            products=["BTC-USD"],
            products_per_connection=1,
        )

        await collector._handle_message(
            {
                "channel": "market_trades",
                "sequence_num": 200,
                "timestamp": "2026-03-18T00:00:00Z",
                "events": [
                    {
                        "type": "snapshot",
                        "trades": [
                            {
                                "trade_id": "1",
                                "product_id": "BTC-USD",
                                "price": "100.0",
                                "size": "0.1",
                                "side": "BUY",
                                "time": "2026-03-18T00:00:00Z",
                            }
                        ],
                    }
                ],
            },
            connection_key="BTC-USD",
        )
        await collector._handle_message(
            {
                "channel": "market_trades",
                "sequence_num": 203,
                "timestamp": "2026-03-18T00:00:01Z",
                "events": [
                    {
                        "type": "update",
                        "trades": [
                            {
                                "trade_id": "2",
                                "product_id": "BTC-USD",
                                "price": "101.0",
                                "size": "0.2",
                                "side": "BUY",
                                "time": "2026-03-18T00:00:01Z",
                            }
                        ],
                    }
                ],
            },
            connection_key="BTC-USD",
        )

        assert collector.stats.gaps == 1
        errors = await redis_client.xrange(settings.STREAM_SYSTEM_ERRORS)
        assert len(errors) == 1
        payload = errors[0][1]
        assert payload["error_type"] == "collector.sequence_gap"
        details = json.loads(payload["payload"])
        assert details["product_id"] == "BTC-USD"
        assert details["last_sequence"] == 200
        assert details["sequence_num"] == 203
        assert details["gap_size"] == 2

    asyncio.run(run())


def test_collector_does_not_count_heartbeat_as_trade_gap() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        collector = CollectorWithCandles(
            redis_client=redis_client,
            products=["BTC-USD"],
            products_per_connection=1,
        )

        await collector._handle_message(
            {
                "channel": "market_trades",
                "sequence_num": 4,
                "timestamp": "2026-03-18T00:00:00Z",
                "events": [
                    {
                        "type": "update",
                        "trades": [
                            {
                                "trade_id": "1",
                                "product_id": "BTC-USD",
                                "price": "100.0",
                                "size": "0.1",
                                "side": "BUY",
                                "time": "2026-03-18T00:00:00Z",
                            }
                        ],
                    }
                ],
            },
            connection_key="BTC-USD",
        )
        await collector._handle_message(
            {
                "channel": "heartbeats",
                "sequence_num": 5,
                "events": [{"heartbeat_counter": 100}],
            },
            connection_key="BTC-USD",
        )
        await collector._handle_message(
            {
                "channel": "market_trades",
                "sequence_num": 6,
                "timestamp": "2026-03-18T00:00:01Z",
                "events": [
                    {
                        "type": "update",
                        "trades": [
                            {
                                "trade_id": "2",
                                "product_id": "BTC-USD",
                                "price": "101.0",
                                "size": "0.2",
                                "side": "SELL",
                                "time": "2026-03-18T00:00:01Z",
                            }
                        ],
                    }
                ],
            },
            connection_key="BTC-USD",
        )

        assert collector.stats.gaps == 0
        assert await redis_client.xlen(settings.STREAM_SYSTEM_ERRORS) == 0

    asyncio.run(run())


def test_collector_batches_products_across_connections() -> None:
    collector = CollectorWithCandles(
        redis_client=None,
        products=["BTC-USD", "ETH-USD", "SOL-USD"],
        products_per_connection=2,
    )

    assert collector._build_product_batches() == [
        ["BTC-USD", "ETH-USD"],
        ["SOL-USD"],
    ]


def test_collector_adds_jwt_to_market_data_subscriptions_when_auth_enabled() -> None:
    collector = CollectorWithCandles(
        redis_client=None,
        coinbase_client=DummyCoinbaseClient(),
        products=["BTC-USD"],
        enable_ws_auth=True,
    )

    subscribe = collector._build_subscription_message(settings.COINBASE_CHANNEL, ["BTC-USD"])
    heartbeat = collector._build_subscription_message(settings.COINBASE_HEARTBEATS_CHANNEL)

    assert subscribe["jwt"] == "jwt-token"
    assert subscribe["product_ids"] == ["BTC-USD"]
    assert heartbeat["jwt"] == "jwt-token"


def test_collector_reconciles_only_missing_closed_candles() -> None:
    async def run() -> None:
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        candles = [
            {
                "product_id": "BTC-USD",
                "open_time": 120_000,
                "close_time": 179_999,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1.0,
                "trade_count": 0,
                "is_closed": True,
            },
            {
                "product_id": "BTC-USD",
                "open_time": 180_000,
                "close_time": 239_999,
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "volume": 1.2,
                "trade_count": 0,
                "is_closed": True,
            },
            {
                "product_id": "BTC-USD",
                "open_time": 240_000,
                "close_time": 299_999,
                "open": 101.5,
                "high": 103.0,
                "low": 101.0,
                "close": 102.5,
                "volume": 1.4,
                "trade_count": 0,
                "is_closed": True,
            },
        ]
        collector = CollectorWithCandles(
            redis_client=redis_client,
            coinbase_client=DummyCoinbaseClient(candles=candles),
            products=["BTC-USD"],
            enable_candle_reconcile=True,
            reconcile_cooldown_seconds=1,
        )
        collector._last_published_candle_open_by_product["BTC-USD"] = 180_000

        await collector._reconcile_recent_candles(
            ["BTC-USD"],
            reason="test",
            now_ms=300_000,
        )

        published = await redis_client.xrange(settings.STREAM_MARKET_CANDLES_1M)
        assert len(published) == 1
        payload = published[0][1]
        assert int(payload["open_time"]) == 240_000
        assert collector.stats.reconciled_candles == 1
        assert collector.stats.candles_published == 1

    asyncio.run(run())
