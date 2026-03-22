"""Probe live con paper trading controlado sobre señales reales."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
import sys
from pathlib import Path

import fakeredis
import fakeredis.aioredis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from data.collector import CollectorWithCandles
from exchange.coinbase_client import CoinbaseAdvancedTradeClient
from main import FeatureEngine
from model.inference import InferenceEngine
from model.registry import MultiAssetModelRegistry
from paper.paper_trader import PaperTrader
from scripts.live_multi_asset_probe import _prime_feature_buffers


async def _ensure_group(redis_client: fakeredis.aioredis.FakeRedis, stream: str, group: str) -> None:
    """Crea un consumer group desde el inicio del stream para no perder mensajes de arranque."""
    try:
        await redis_client.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def main() -> None:
    """Ejecuta un probe controlado con feed publico + paper trader."""
    seconds = int(os.getenv("LIVE_PROBE_SECONDS", "90"))
    registry = MultiAssetModelRegistry(root_dir=settings.MODEL_REGISTRY_ROOT)
    coinbase_client = CoinbaseAdvancedTradeClient()
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
        products_per_connection=1,
    )
    feature_engine = FeatureEngine(redis_client=redis_async)
    inference = InferenceEngine(registry=registry, redis_client=redis_async)
    paper_trader = PaperTrader(redis_client=redis_async)
    _prime_feature_buffers(feature_engine, products)
    await _ensure_group(redis_async, settings.STREAM_MARKET_CANDLES_1M, "feature-engine")
    await _ensure_group(redis_async, settings.STREAM_MARKET_FEATURES, "inference-engine")
    await _ensure_group(redis_async, settings.STREAM_MARKET_CANDLES_1M, "paper-trader-candles")
    await _ensure_group(redis_async, settings.STREAM_INFERENCE_SIGNALS, "paper-trader-signals")

    stop_event = asyncio.Event()
    tasks = [
        asyncio.create_task(feature_engine.start(stop_event)),
        asyncio.create_task(inference.start(stop_event)),
        asyncio.create_task(paper_trader.start(stop_event)),
    ]
    await asyncio.sleep(0.25)
    tasks.append(asyncio.create_task(collector.start(stop_event)))

    try:
        await asyncio.sleep(seconds)
    finally:
        stop_event.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    counts = {
        stream: await redis_async.xlen(stream)
        for stream in [
            settings.STREAM_MARKET_TRADES_RAW,
            settings.STREAM_MARKET_CANDLES_1M,
            settings.STREAM_MARKET_FEATURES,
            settings.STREAM_INFERENCE_SIGNALS,
            settings.STREAM_PAPER_EXECUTION_EVENTS,
            settings.STREAM_SYSTEM_ERRORS,
        ]
    }
    paper_events = await redis_async.xrange(settings.STREAM_PAPER_EXECUTION_EVENTS, count=10)
    print(
        json.dumps(
            {
                "observed_products": products,
                "actionable_registry_keys": registry.get_actionable_registry_keys(),
                "counts": counts,
                "paper_events": paper_events,
                "paper_state": asdict(paper_trader.current_state()),
            },
            indent=2,
        )
    )
    await redis_async.flushall()
    await redis_async.aclose()


if __name__ == "__main__":
    asyncio.run(main())
