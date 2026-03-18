"""Probe live multi-activo con feed publico de Coinbase y ejecucion en dry-run."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import fakeredis
import fakeredis.aioredis
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from data.collector import CollectorWithCandles
from exchange.coinbase_client import CoinbaseAdvancedTradeClient
from execution.order_manager import OrderManager
from main import FeatureEngine
from model.inference import InferenceEngine
from model.registry import MultiAssetModelRegistry


def make_synthetic_candles(n: int = 180, seed: int = 42) -> pd.DataFrame:
    """Genera velas sinteticas para precargar el buffer de features."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=float)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.12, size=n)) + 0.8 * np.sin(idx / 8.0)
    open_ = np.concatenate([[close[0]], close[:-1]]) + rng.normal(0.0, 0.04, size=n)
    spread = np.abs(rng.normal(0.18, 0.03, size=n)) + 0.05
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(700.0 + 90.0 * np.sin(idx / 7.0) + rng.normal(0.0, 30.0, size=n)) + 1.0
    open_time = 1_700_000_000_000 + (idx.astype(int) * 60_000)
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


async def main() -> None:
    """Valida auth privada, feed publico y ejecucion dry-run multi-activo."""
    seconds = int(os.getenv("LIVE_PROBE_SECONDS", "45"))
    registry = MultiAssetModelRegistry(root_dir=settings.MODEL_REGISTRY_ROOT)
    coinbase_client = CoinbaseAdvancedTradeClient()
    accounts = coinbase_client.validate_credentials()
    resolved = coinbase_client.resolve_products_for_bases(settings.OBSERVED_BASES)
    products = list(resolved.values())
    if not products:
        raise RuntimeError("No se pudieron resolver productos observables en Coinbase.")

    server = fakeredis.FakeServer()
    redis_async = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    collector = CollectorWithCandles(
        redis_client=redis_async,
        coinbase_client=coinbase_client,
        products=products,
    )
    feature_engine = FeatureEngine(redis_client=redis_async)
    inference = InferenceEngine(registry=registry, redis_client=redis_async)
    order_manager = OrderManager(
        redis_client=redis_async,
        coinbase_client=coinbase_client,
        execution_enabled=settings.EXECUTION_ENABLED,
        dry_run=True,
        order_notional_usd=settings.PILOT_ORDER_NOTIONAL_USD,
    )
    _prime_feature_buffers(feature_engine, products)

    stop_event = asyncio.Event()
    tasks = [
        asyncio.create_task(collector.start(stop_event)),
        asyncio.create_task(feature_engine.start(stop_event)),
        asyncio.create_task(inference.start(stop_event)),
        asyncio.create_task(order_manager.start(stop_event)),
    ]

    try:
        await asyncio.sleep(seconds)
    finally:
        stop_event.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    streams = [
        settings.STREAM_MARKET_TRADES_RAW,
        settings.STREAM_MARKET_CANDLES_1M,
        settings.STREAM_MARKET_FEATURES,
        settings.STREAM_INFERENCE_SIGNALS,
        settings.STREAM_EXECUTION_EVENTS,
        settings.STREAM_SYSTEM_HEALTH,
        settings.STREAM_SYSTEM_ERRORS,
    ]
    counts = {stream: await redis_async.xlen(stream) for stream in streams}
    sample_signals = await redis_async.xrange(settings.STREAM_INFERENCE_SIGNALS, count=5)
    sample_execution = await redis_async.xrange(settings.STREAM_EXECUTION_EVENTS, count=5)
    print(
        {
            "accounts_detected": len(accounts),
            "observed_products": products,
            "actionable_registry_keys": registry.get_actionable_registry_keys(),
            "counts": counts,
            "sample_signals": sample_signals,
            "sample_execution_events": sample_execution,
        }
    )
    await redis_async.flushall()
    await redis_async.aclose()


def _prime_feature_buffers(feature_engine: FeatureEngine, products: list[str]) -> None:
    """Precarga historia sintetica para acelerar la primera emision de features."""
    for offset, product_id in enumerate(products):
        history = make_synthetic_candles(settings.FEATURE_BUFFER_SIZE, seed=301 + offset)
        for row in history.tail(settings.FEATURE_BUFFER_SIZE).to_dict(orient="records"):
            feature_engine.buffers[product_id].append(
                {
                    "product_id": product_id,
                    "open_time": int(row["open_time"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )


if __name__ == "__main__":
    asyncio.run(main())
