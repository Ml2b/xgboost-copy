"""Smoke test end-to-end con Redis simulado y trades sinteticos."""

from __future__ import annotations

import asyncio
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
from main import FeatureEngine
from model.inference import InferenceEngine
from model.registry import ModelRegistry
from model.trainer import Trainer
from validation.walk_forward import WalkForwardConfig


def make_synthetic_candles(n: int = 1800, seed: int = 42) -> pd.DataFrame:
    """Genera velas con estructura suficiente para smoke tests."""
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


def candle_to_trades(row: pd.Series, sequence_start: int) -> list[dict[str, object]]:
    """Expande una vela sintetica a cuatro trades para el collector."""
    base_ts = int(row["open_time"])
    size = float(row["volume"]) / 4.0
    prices = [float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])]
    offsets = [1_000, 15_000, 30_000, 50_000]
    trades: list[dict[str, object]] = []
    for idx, (price, offset) in enumerate(zip(prices, offsets, strict=True)):
        trades.append(
            {
                "product_id": "BTC-USDT",
                "trade_id": sequence_start + idx,
                "price": price,
                "size": size,
                "side": "BUY",
                "time": pd.to_datetime(base_ts + offset, unit="ms", utc=True).isoformat().replace("+00:00", "Z"),
            }
        )
    return trades


async def main() -> None:
    """Entrena, promueve y valida el pipeline en memoria."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / "models"
        registry = ModelRegistry(base_dir=model_dir)
        historical = make_synthetic_candles(1800, seed=31)
        trainer = Trainer(
            registry=registry,
            data_loader=lambda: historical.copy(),
            walk_forward_config=WalkForwardConfig(n_splits=3, min_train_size=600, gap_periods=5, verbose=False),
        )
        train_result = trainer._retrain_cycle()
        if train_result.status != "trained":
            raise RuntimeError(f"Entrenamiento no exitoso: {train_result.status}")

        server = fakeredis.FakeServer()
        async_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
        collector = CollectorWithCandles(redis_client=async_redis)
        feature_engine = FeatureEngine(redis_client=async_redis)
        inference = InferenceEngine(registry=registry, redis_client=async_redis)

        stop_event = asyncio.Event()
        feature_task = asyncio.create_task(feature_engine.start(stop_event))
        inference_task = asyncio.create_task(inference.start(stop_event))
        await asyncio.sleep(0.2)

        sample = historical.iloc[:120].copy()
        sequence = 1
        for _, row in sample.iterrows():
            await collector._handle_message({"events": [{"trades": candle_to_trades(row, sequence)}]})
            sequence += 4

        last_row = sample.iloc[-1]
        forced = collector.candle_builder.force_close("BTC-USDT", int(last_row["open_time"]) + 59_000)
        if forced is not None:
            await async_redis.xadd(
                settings.STREAM_MARKET_CANDLES_1M,
                {key: str(value) for key, value in forced.to_dict().items()},
            )

        deadline = asyncio.get_running_loop().time() + 10.0
        while asyncio.get_running_loop().time() < deadline:
            if await async_redis.xlen(settings.STREAM_INFERENCE_SIGNALS) > 0:
                break
            await asyncio.sleep(0.1)

        stop_event.set()
        for task in (feature_task, inference_task):
            task.cancel()
        await asyncio.gather(feature_task, inference_task, return_exceptions=True)

        candle_count = await async_redis.xlen(settings.STREAM_MARKET_CANDLES_1M)
        feature_count = await async_redis.xlen(settings.STREAM_MARKET_FEATURES)
        signal_count = await async_redis.xlen(settings.STREAM_INFERENCE_SIGNALS)
        signals = await async_redis.xrange(settings.STREAM_INFERENCE_SIGNALS, count=3)
        print(
            {
                "train_status": train_result.status,
                "promoted": train_result.promoted,
                "candles": candle_count,
                "features": feature_count,
                "signals": signal_count,
                "sample_signals": signals,
            }
        )

        await async_redis.flushall()
        await async_redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
