"""Orquestador principal del sistema de trading."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from collections import defaultdict, deque
from typing import Any

import pandas as pd
from loguru import logger

from config import settings
from data.collector import CollectorWithCandles
from exchange.coinbase_client import CoinbaseAdvancedTradeClient
from execution.order_manager import OrderManager
from features.calculator import FeatureCalculator
from model.inference import InferenceEngine
from model.registry import MultiAssetModelRegistry
from model.trainer import MultiAssetTrainerService
from paper.paper_trader import PaperTrader


class FeatureEngine:
    """Consume velas cerradas y publica el ultimo vector de features."""

    def __init__(self, redis_client: Any, calculator: FeatureCalculator | None = None) -> None:
        self.redis_client = redis_client
        self.calculator = calculator or FeatureCalculator()
        self.buffers: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=settings.FEATURE_BUFFER_SIZE)
        )

    async def start(self, stop_event: asyncio.Event | None = None) -> None:
        stop_event = stop_event or asyncio.Event()
        await self._ensure_group(settings.STREAM_MARKET_CANDLES_1M, "feature-engine")
        while not stop_event.is_set():
            messages = await self.redis_client.xreadgroup(
                groupname="feature-engine",
                consumername="feature-1",
                streams={settings.STREAM_MARKET_CANDLES_1M: ">"},
                count=25,
                block=1000,
            )
            for _, stream_messages in messages:
                for message_id, payload in stream_messages:
                    await self._process_candle(payload)
                    await self.redis_client.xack(settings.STREAM_MARKET_CANDLES_1M, "feature-engine", message_id)

    async def _process_candle(self, payload: dict[str, Any]) -> None:
        product_id = payload.get("product_id", "")
        normalized = {
            "product_id": product_id,
            "open_time": int(payload["open_time"]),
            "open": float(payload["open"]),
            "high": float(payload["high"]),
            "low": float(payload["low"]),
            "close": float(payload["close"]),
            "volume": float(payload["volume"]),
        }
        self.buffers[product_id].append(normalized)
        buffer = list(self.buffers[product_id])
        if len(buffer) < max(70, settings.FEATURE_BUFFER_SIZE // 2):
            return

        frame = pd.DataFrame(buffer)
        features_frame = self.calculator.compute(frame)
        latest = features_frame.iloc[-1].to_dict()
        latest["product_id"] = product_id
        await self.redis_client.xadd(
            settings.STREAM_MARKET_FEATURES,
            {key: str(value) for key, value in latest.items()},
        )

    async def _ensure_group(self, stream: str, group_name: str) -> None:
        try:
            await self.redis_client.xgroup_create(stream, group_name, id="$", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise


async def run_service(name: str, factory, stop_event: asyncio.Event) -> None:
    """Mantiene vivo un servicio aunque falle y lo reinicia."""
    while not stop_event.is_set():
        try:
            await factory()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Servicio {} fallo y se reiniciara", name)
            await asyncio.sleep(5)


async def async_main() -> None:
    """Crea clientes y arranca todos los servicios en paralelo."""
    import redis.asyncio as redis_async
    import redis as redis_sync

    logger.add("logs/trading_{time:YYYY-MM-DD}.log", rotation="1 day", retention=7)
    logger.info("python main.py")
    logger.info("python -m pytest tests/")
    logger.warning(
        "Modo de ejecucion activo: enabled={} dry_run={}",
        settings.EXECUTION_ENABLED,
        settings.EXECUTION_DRY_RUN,
    )

    redis_client_async = _build_async_redis_client(redis_async)
    redis_client_sync = _build_sync_redis_client(redis_sync)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop_event.set)

    coinbase_client = CoinbaseAdvancedTradeClient()
    try:
        accounts = coinbase_client.validate_credentials()
        logger.info("Coinbase autenticado. accounts_detected={}", len(accounts))
    except Exception as exc:
        logger.warning("No se pudo validar la autenticacion Coinbase: {}", exc)

    observed_products = await _resolve_observed_products(redis_client_async, coinbase_client)
    registry = MultiAssetModelRegistry(root_dir=settings.MODEL_REGISTRY_ROOT)
    collector = CollectorWithCandles(
        redis_client=redis_client_async,
        coinbase_client=coinbase_client,
        products=observed_products,
    )
    feature_engine = FeatureEngine(redis_client=redis_client_async)
    inference = InferenceEngine(registry=registry, redis_client=redis_client_async)
    trainer = MultiAssetTrainerService(
        registry_root=settings.MODEL_REGISTRY_ROOT,
        redis_client=redis_client_sync,
    )
    order_manager = OrderManager(
        redis_client=redis_client_async,
        coinbase_client=coinbase_client,
        execution_enabled=settings.EXECUTION_ENABLED,
        dry_run=settings.EXECUTION_DRY_RUN,
        order_notional_usd=settings.PILOT_ORDER_NOTIONAL_USD,
    )
    paper_trader = PaperTrader(redis_client=redis_client_async)

    services = [
        asyncio.create_task(run_service("collector", lambda: collector.start(stop_event), stop_event)),
        asyncio.create_task(run_service("feature_engine", lambda: feature_engine.start(stop_event), stop_event)),
        asyncio.create_task(run_service("inference", lambda: inference.start(stop_event), stop_event)),
        asyncio.create_task(run_service("trainer", lambda: trainer.start(stop_event), stop_event)),
        asyncio.create_task(run_service("order_manager", lambda: order_manager.start(stop_event), stop_event)),
    ]
    if settings.PAPER_TRADING_ENABLED:
        services.append(
            asyncio.create_task(run_service("paper_trader", lambda: paper_trader.start(stop_event), stop_event))
        )

    try:
        await stop_event.wait()
    finally:
        for task in services:
            task.cancel()
        await asyncio.gather(*services, return_exceptions=True)
        await redis_client_async.close()
        redis_client_sync.close()


async def _resolve_observed_products(
    redis_client_async: Any,
    coinbase_client: CoinbaseAdvancedTradeClient,
) -> list[str]:
    """Resuelve el universo observado respetando prioridad de quotes."""
    explicit_products = os.getenv("PRODUCTS_CSV", "").strip()
    if explicit_products:
        products = [product.strip() for product in explicit_products.split(",") if product.strip()]
        logger.info("Usando PRODUCTS_CSV explicito: {}", products)
        return products

    resolved = coinbase_client.resolve_products_for_bases(settings.OBSERVED_BASES)
    observed_products = list(resolved.values())
    unsupported_bases = [
        base
        for base in settings.OBSERVED_BASES
        if base.strip().upper() not in resolved
    ]
    logger.info("Productos observados resueltos: {}", observed_products)
    if unsupported_bases:
        await redis_client_async.xadd(
            settings.STREAM_SYSTEM_ERRORS,
            {
                "error_type": "startup.unsupported_products",
                "payload": json.dumps({"unsupported_bases": unsupported_bases}),
            },
        )
        logger.warning("Bases no soportadas en Coinbase: {}", unsupported_bases)

    return observed_products or settings.PRODUCTS


def main() -> None:
    """Entry point del orquestador."""
    asyncio.run(async_main())


def _build_async_redis_client(redis_async_module: Any) -> Any:
    """Construye el cliente async soportando REDIS_URL o host/port clasicos."""
    if settings.REDIS_URL:
        return redis_async_module.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
    return redis_async_module.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        username=settings.REDIS_USERNAME,
        password=settings.REDIS_PASSWORD,
        db=settings.REDIS_DB,
        decode_responses=True,
    )


def _build_sync_redis_client(redis_sync_module: Any) -> Any:
    """Construye el cliente sync soportando REDIS_URL o host/port clasicos."""
    if settings.REDIS_URL:
        return redis_sync_module.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
    return redis_sync_module.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        username=settings.REDIS_USERNAME,
        password=settings.REDIS_PASSWORD,
        db=settings.REDIS_DB,
        decode_responses=True,
    )


if __name__ == "__main__":
    main()
