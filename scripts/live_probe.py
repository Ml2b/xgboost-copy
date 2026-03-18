"""Probe live con Coinbase real y Redis en memoria compartido."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
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
from features.selector import SelectorConfig
from main import FeatureEngine
from model.inference import InferenceEngine
from model.registry import ModelRegistry
from model.trainer import Trainer
from validation.walk_forward import WalkForwardConfig


def make_synthetic_candles(n: int = 1800, seed: int = 42) -> pd.DataFrame:
    """Genera velas sinteticas para bootstrap del modelo."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=float)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.18, size=n)) + 1.8 * np.sin(idx / 12.0) + (idx * 0.015)
    open_ = np.concatenate([[close[0]], close[:-1]]) + rng.normal(0.0, 0.05, size=n)
    spread = np.abs(rng.normal(0.22, 0.04, size=n)) + 0.05
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(900.0 + 120.0 * np.sin(idx / 9.0) + rng.normal(0.0, 40.0, size=n)) + 1.0
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
    """Entrena un modelo base y valida el flujo live con datos reales."""
    seconds = int(os.getenv("LIVE_PROBE_SECONDS", "45"))
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(base_dir=Path(temp_dir) / "models")
        trainer = Trainer(
            registry=registry,
            data_loader=lambda: make_synthetic_candles(1800, seed=31),
            walk_forward_config=WalkForwardConfig(n_splits=3, min_train_size=600, gap_periods=5, verbose=False),
            selector_config=SelectorConfig(verbose=False, max_features=10),
        )
        result = trainer._retrain_cycle()
        if result.status != "trained":
            raise RuntimeError(f"No se pudo bootstrapear el modelo: {result.status}")

        server = fakeredis.FakeServer()
        redis_async = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
        collector = CollectorWithCandles(
            redis_client=redis_async,
            coinbase_client=CoinbaseAdvancedTradeClient(),
        )
        feature_engine = FeatureEngine(redis_client=redis_async)
        inference = InferenceEngine(registry=registry, redis_client=redis_async)
        _prime_feature_buffers(feature_engine)

        stop_event = asyncio.Event()
        tasks = [
            asyncio.create_task(collector.start(stop_event)),
            asyncio.create_task(feature_engine.start(stop_event)),
            asyncio.create_task(inference.start(stop_event)),
        ]

        try:
            await asyncio.sleep(seconds)
        finally:
            stop_event.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        streams = [
            "market.trades.raw",
            "market.candles.1m",
            "market.features",
            "inference.signals",
            "system.health",
            "system.errors",
        ]
        counts = {stream: await redis_async.xlen(stream) for stream in streams}
        sample_signals = await redis_async.xrange("inference.signals", count=5)
        sample_errors = await redis_async.xrange("system.errors", count=5)
        print(
            {
                "train_status": result.status,
                "promoted": result.promoted,
                "counts": counts,
                "sample_signals": sample_signals,
                "sample_errors": sample_errors,
            }
        )
        await redis_async.flushall()
        await redis_async.aclose()


def _prime_feature_buffers(feature_engine: FeatureEngine) -> None:
    """Precarga historia para que la primera vela live produzca features."""
    for offset, product_id in enumerate(settings.PRODUCTS):
        history = make_synthetic_candles(settings.FEATURE_BUFFER_SIZE, seed=101 + offset)
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
