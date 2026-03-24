"""Collector de trades Coinbase con publicacion en Redis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from typing import Any

from loguru import logger

from config import settings
from data.candle_builder import CandleBuilder
from exchange.coinbase_client import CoinbaseAdvancedTradeClient
from features.order_flow import OrderFlowAggregator


@dataclass(slots=True)
class CollectorStats:
    """Metricas operativas del collector."""

    trades_received: int = 0
    candles_published: int = 0
    duplicates: int = 0
    gaps: int = 0
    reconciled_candles: int = 0
    lag_ms_total: float = 0.0
    lag_samples: int = 0
    lag_ema_ms: float = 0.0
    of_published: int = 0
    of_none: int = 0
    of_errors: int = 0
    l2_messages: int = 0
    l2_errors: int = 0

    @property
    def avg_lag_ms(self) -> float:
        """EMA del lag reciente en milisegundos (alpha=0.001, ~700 muestras)."""
        return self.lag_ema_ms


class CollectorWithCandles:
    """Consume trades, deduplica, arma velas y publica en Redis."""

    def __init__(
        self,
        redis_client: Any,
        candle_builder: CandleBuilder | None = None,
        coinbase_client: CoinbaseAdvancedTradeClient | None = None,
        products: list[str] | None = None,
        websocket_url: str = settings.COINBASE_WS_PUBLIC_URL,
        products_per_connection: int = settings.COINBASE_WS_PRODUCTS_PER_CONNECTION,
        enable_ws_auth: bool = settings.COINBASE_WS_AUTH_ENABLED,
        enable_candle_reconcile: bool = settings.COINBASE_CANDLE_RECONCILE_ENABLED,
        reconcile_lookback_minutes: int = settings.COINBASE_CANDLE_RECONCILE_LOOKBACK_MINUTES,
        reconcile_cooldown_seconds: int = settings.COINBASE_CANDLE_RECONCILE_COOLDOWN_SECONDS,
    ) -> None:
        self.redis_client = redis_client
        self.candle_builder = candle_builder or CandleBuilder()
        self.coinbase_client = coinbase_client
        self.products = products or settings.PRODUCTS
        self.websocket_url = websocket_url
        self.products_per_connection = max(1, int(products_per_connection))
        self.enable_ws_auth = bool(enable_ws_auth)
        self.enable_candle_reconcile = bool(enable_candle_reconcile)
        self.reconcile_lookback_minutes = max(1, int(reconcile_lookback_minutes))
        self.reconcile_cooldown_seconds = max(1, int(reconcile_cooldown_seconds))
        self.stats = CollectorStats()
        self._seen_hashes: deque[str] = deque(maxlen=50_000)
        self._seen_lookup: set[str] = set()
        self._last_sequence_by_connection: dict[str, int] = {}
        self._last_heartbeat_counter_by_connection: dict[str, int] = {}
        self._products_by_connection_key: dict[str, list[str]] = {
            self._connection_key(batch): list(batch)
            for batch in self._build_product_batches()
        }
        self._last_published_candle_open_by_product: dict[str, int] = {}
        self._last_reconcile_started_at_by_product: dict[str, float] = {}
        self._last_runtime_log_at = 0.0
        self._order_flow_agg = OrderFlowAggregator()

    async def start(self, stop_event: asyncio.Event | None = None) -> None:
        """Arranca una o varias conexiones con reconexion exponencial."""
        stop_event = stop_event or asyncio.Event()
        logger.info(
            "Collector iniciado. products={} ws_auth={} reconcile={} batches={}",
            len(self.products),
            self._should_auth_ws(),
            self.enable_candle_reconcile,
            len(self._build_product_batches()),
        )
        health_task = asyncio.create_task(self._health_loop(stop_event))
        socket_tasks = [
            asyncio.create_task(self._run_socket_loop(batch, stop_event))
            for batch in self._build_product_batches()
        ]
        try:
            if socket_tasks:
                await asyncio.gather(*socket_tasks)
            else:
                await stop_event.wait()
        finally:
            for task in socket_tasks:
                task.cancel()
            health_task.cancel()
            await asyncio.gather(*socket_tasks, return_exceptions=True)
            await asyncio.gather(health_task, return_exceptions=True)
            for product in self.products:
                candle = self.candle_builder.force_close(product, int(datetime.now(tz=timezone.utc).timestamp() * 1000))
                if candle is not None:
                    await self._publish_closed_candle(candle.to_dict())

    async def _run_socket_loop(self, products: list[str], stop_event: asyncio.Event) -> None:
        """Mantiene una conexion viva para un subconjunto de productos."""
        import time as _time

        backoff = 2
        attempts = 0
        connection_key = self._connection_key(products)
        logger.info("Collector socket loop iniciado. connection_key={}", connection_key)
        while not stop_event.is_set():
            t0 = _time.monotonic()
            try:
                await self._run_socket(products, stop_event)
                backoff = 2
                attempts = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                elapsed = _time.monotonic() - t0
                if elapsed > 120:
                    # Conexion estable >2min: resetear contador
                    attempts = 0
                    backoff = 2
                attempts += 1
                logger.warning(
                    "Collector websocket reconectando. connection_key={} attempt={} elapsed={:.0f}s error={}",
                    connection_key,
                    attempts,
                    elapsed,
                    exc,
                )
                await self._publish_error(
                    "collector.websocket",
                    {
                        "error": str(exc),
                        "attempt": attempts,
                        "products": products,
                        "connection_key": connection_key,
                    },
                )
                await self._reconcile_recent_candles(
                    products,
                    reason="websocket_reconnect",
                )
                if attempts >= 20:
                    raise
                await asyncio.sleep(min(backoff, 60))
                backoff = min(backoff * 2, 60)

    async def _run_socket(self, products: list[str], stop_event: asyncio.Event) -> None:
        import websockets

        connection_key = self._connection_key(products)
        self._last_heartbeat_counter_by_connection.pop(connection_key, None)
        async with websockets.connect(self.websocket_url, ping_interval=20, ping_timeout=20, max_size=20 * 1024 * 1024) as websocket:
            logger.info(
                "Collector websocket conectado. connection_key={} products={} auth={}",
                connection_key,
                products,
                self._should_auth_ws(),
            )
            await websocket.send(json.dumps(self._build_subscription_message(settings.COINBASE_CHANNEL, products)))
            await websocket.send(json.dumps(self._build_subscription_message(settings.COINBASE_HEARTBEATS_CHANNEL)))
            if settings.LEVEL2_ENABLED:
                await websocket.send(json.dumps(self._build_subscription_message(settings.COINBASE_LEVEL2_SUBSCRIBE_CHANNEL, products)))

            while not stop_event.is_set():
                raw_message = await asyncio.wait_for(websocket.recv(), timeout=settings.WS_TIMEOUT_SECONDS)
                payload = json.loads(raw_message)
                await self._handle_message(payload, connection_key=connection_key)

    async def _handle_message(self, message: dict[str, Any], connection_key: str) -> None:
        await self._track_connection_health(message, connection_key)
        await self._track_connection_sequence(message, connection_key)
        try:
            self._handle_level2_message(message)
            if str(message.get("channel", "")).lower() == settings.COINBASE_LEVEL2_CHANNEL:
                self.stats.l2_messages += 1
        except Exception as exc:
            self.stats.l2_errors += 1
            logger.warning("Collector L2 processing error ignorado: {}", exc)
        for trade in self._extract_trades(message):
            if not self._is_valid_trade(trade):
                await self._publish_error("collector.invalid_payload", trade)
                continue

            dedup_hash = self._hash_trade(trade)
            if dedup_hash in self._seen_lookup:
                self.stats.duplicates += 1
                continue
            self._remember_hash(dedup_hash)

            product_id = str(trade["product_id"])
            price = float(trade["price"])
            size = float(trade["size"])
            ts_ms = int(trade["ts_ms"])
            now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            lag_ms = max(0, now_ms - ts_ms)
            self.stats.trades_received += 1
            self.stats.lag_ms_total += lag_ms
            self.stats.lag_samples += 1
            if self.stats.lag_samples == 1:
                self.stats.lag_ema_ms = lag_ms
            else:
                self.stats.lag_ema_ms = 0.001 * lag_ms + 0.999 * self.stats.lag_ema_ms

            await self._publish_stream(settings.STREAM_MARKET_TRADES_RAW, trade)
            # Agregar trade al aggregator ANTES de cerrar la vela para que
            # el último trade se incluya en las métricas de order flow.
            if settings.ORDER_FLOW_ENABLED:
                self._order_flow_agg.add_trade(
                    product_id,
                    price,
                    size,
                    str(trade.get("side", "")),
                    ts_ms,
                )
            candle = self.candle_builder.add_trade(product_id, price, size, ts_ms)
            if candle is not None:
                if settings.ORDER_FLOW_ENABLED:
                    try:
                        of_metrics = self._order_flow_agg.close_window(
                            product_id,
                            candle.open_time,
                            price_open=candle.open,
                            price_close=candle.close,
                        )
                        if of_metrics is not None:
                            await self._publish_stream(
                                settings.STREAM_MARKET_ORDER_FLOW_1M,
                                {k: str(v) for k, v in of_metrics.items()},
                            )
                            self.stats.of_published += 1
                        else:
                            self.stats.of_none += 1
                    except Exception as exc:
                        self.stats.of_errors += 1
                        logger.warning(
                            "Collector order_flow close_window error: product={} error={}",
                            product_id,
                            exc,
                        )
                await self._publish_closed_candle(candle.to_dict())

    def _handle_level2_message(self, message: dict[str, Any]) -> None:
        """Despacha snapshots y updates del canal level2 al BookDepthTracker."""
        if not settings.LEVEL2_ENABLED:
            return
        if str(message.get("channel", "")).lower() != settings.COINBASE_LEVEL2_CHANNEL:
            return
        for event in message.get("events", []):
            event_type = str(event.get("type", "")).lower()
            product_id = event.get("product_id") or message.get("product_id")
            if not product_id:
                continue
            product_id = str(product_id)
            updates = event.get("updates", [])
            if event_type == "snapshot":
                bids = [
                    (str(u.get("price_level", "0")), str(u.get("new_quantity", "0")))
                    for u in updates
                    if str(u.get("side", "")).lower() == "bid"
                    and u.get("price_level") is not None
                    and u.get("new_quantity") is not None
                ]
                asks = [
                    (str(u.get("price_level", "0")), str(u.get("new_quantity", "0")))
                    for u in updates
                    if str(u.get("side", "")).lower() == "offer"
                    and u.get("price_level") is not None
                    and u.get("new_quantity") is not None
                ]
                self._order_flow_agg.apply_book_snapshot(product_id, bids, asks)
            elif event_type == "update":
                changes = [
                    (str(u.get("side", "")), str(u.get("price_level", "0")), str(u.get("new_quantity", "0")))
                    for u in updates
                    if u.get("price_level") is not None
                    and u.get("new_quantity") is not None
                ]
                self._order_flow_agg.apply_book_update(product_id, changes)

    def _extract_trades(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        trades: list[dict[str, Any]] = []
        events = message.get("events", [])
        if isinstance(message.get("trades"), list):
            events = [{"trades": message["trades"]}]

        for event in events:
            event_type = str(event.get("type", "")).lower()
            message_sequence = CollectorWithCandles._safe_int(message.get("sequence_num"))
            for index, trade in enumerate(event.get("trades", [])):
                trade_id = trade.get("trade_id")
                if trade_id in (None, "") and message_sequence:
                    trade_id = f"{message_sequence}:{index}"
                ts_ms = self._parse_time_to_ms(trade.get("time") or trade.get("trade_time") or message.get("timestamp"))
                trades.append(
                    {
                        "product_id": trade.get("product_id") or message.get("product_id"),
                        "trade_id": trade_id,
                        "price": trade.get("price"),
                        "size": trade.get("size"),
                        "side": trade.get("side", ""),
                        "ts_ms": ts_ms,
                        "event_type": event_type,
                    }
                )
        return trades

    @staticmethod
    def _is_valid_trade(trade: dict[str, Any]) -> bool:
        try:
            return float(trade["price"]) > 0 and float(trade["size"]) > 0 and int(trade["ts_ms"]) > 0
        except Exception:
            return False

    @staticmethod
    def _hash_trade(trade: dict[str, Any]) -> str:
        payload = f"{trade.get('product_id')}|{trade.get('trade_id')}|{trade.get('price')}|{trade.get('size')}|{trade.get('ts_ms')}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _remember_hash(self, digest: str) -> None:
        if len(self._seen_hashes) == self._seen_hashes.maxlen:
            removed = self._seen_hashes.popleft()
            self._seen_lookup.discard(removed)
        self._seen_hashes.append(digest)
        self._seen_lookup.add(digest)

    async def _track_connection_health(self, message: dict[str, Any], connection_key: str) -> None:
        """Usa heartbeat_counter para detectar perdidas a nivel conexion."""
        if str(message.get("channel", "")).lower() != settings.COINBASE_HEARTBEATS_CHANNEL:
            return

        counters = [
            self._safe_int(event.get("heartbeat_counter"))
            for event in message.get("events", [])
            if self._safe_int(event.get("heartbeat_counter")) > 0
        ]
        if not counters:
            return

        current = counters[-1]
        last = self._last_heartbeat_counter_by_connection.get(connection_key)
        if last is not None and current > last + 1:
            logger.warning(
                "Collector heartbeat gap detectado. connection_key={} last={} current={} missed={}",
                connection_key,
                last,
                current,
                current - last - 1,
            )
            await self._publish_error(
                "collector.heartbeat_gap",
                {
                    "connection_key": connection_key,
                    "last_heartbeat_counter": last,
                    "heartbeat_counter": current,
                    "missed": current - last - 1,
                },
            )
            await self._reconcile_recent_candles(
                self._products_by_connection_key.get(connection_key, []),
                reason="heartbeat_gap",
            )
        self._last_heartbeat_counter_by_connection[connection_key] = current

    async def _track_connection_sequence(self, message: dict[str, Any], connection_key: str) -> None:
        """Detecta gaps reales usando la secuencia global de la conexion."""
        sequence_num = self._safe_int(message.get("sequence_num"))
        if sequence_num <= 0:
            return

        is_snapshot = any(str(event.get("type", "")).lower() == "snapshot" for event in message.get("events", []))
        last_sequence = self._last_sequence_by_connection.get(connection_key)
        if last_sequence is None or is_snapshot:
            self._last_sequence_by_connection[connection_key] = sequence_num
            return
        if sequence_num <= last_sequence:
            return
        if sequence_num > last_sequence + 1:
            self.stats.gaps += 1
            logger.warning(
                "Collector sequence gap detectado. connection_key={} channel={} last_sequence={} sequence_num={} gap_size={}",
                connection_key,
                str(message.get("channel", "")).lower(),
                last_sequence,
                sequence_num,
                sequence_num - last_sequence - 1,
            )
            payload: dict[str, Any] = {
                "connection_key": connection_key,
                "channel": str(message.get("channel", "")).lower(),
                "last_sequence": last_sequence,
                "sequence_num": sequence_num,
                "gap_size": sequence_num - last_sequence - 1,
            }
            product_ids = self._extract_product_ids(message)
            if len(product_ids) == 1:
                payload["product_id"] = product_ids[0]
            elif product_ids:
                payload["product_ids"] = product_ids
            await self._publish_error("collector.sequence_gap", payload)
            await self._reconcile_recent_candles(
                product_ids or self._products_by_connection_key.get(connection_key, []),
                reason="sequence_gap",
            )
        self._last_sequence_by_connection[connection_key] = sequence_num

    def _build_product_batches(self) -> list[list[str]]:
        """Divide productos en lotes pequenos para reducir contencion del feed."""
        iterator = iter(self.products)
        batches: list[list[str]] = []
        while batch := list(islice(iterator, self.products_per_connection)):
            batches.append(batch)
        return batches

    @staticmethod
    def _connection_key(products: list[str]) -> str:
        return ",".join(products)

    @staticmethod
    def _extract_product_ids(message: dict[str, Any]) -> list[str]:
        """Extrae los productos afectados por un mensaje market_trades."""
        product_ids: list[str] = []
        for event in message.get("events", []):
            for trade in event.get("trades", []):
                product_id = trade.get("product_id") or message.get("product_id")
                if product_id:
                    normalized = str(product_id)
                    if normalized not in product_ids:
                        product_ids.append(normalized)
        if not product_ids and message.get("product_id"):
            product_ids.append(str(message["product_id"]))
        return product_ids

    async def _health_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            await self._publish_stream(
                settings.STREAM_SYSTEM_HEALTH,
                {
                    "service": "collector",
                    "trades_received": self.stats.trades_received,
                    "candles_published": self.stats.candles_published,
                    "reconciled_candles": self.stats.reconciled_candles,
                    "duplicates": self.stats.duplicates,
                    "gaps": self.stats.gaps,
                    "avg_lag_ms": round(self.stats.avg_lag_ms, 3),
                    "of_published": self.stats.of_published,
                    "of_none": self.stats.of_none,
                    "of_errors": self.stats.of_errors,
                    "l2_messages": self.stats.l2_messages,
                    "l2_errors": self.stats.l2_errors,
                },
            )
            self._log_runtime_summary()
            await asyncio.sleep(settings.HEALTH_PUBLISH_INTERVAL)

    def _log_runtime_summary(self) -> None:
        """Emite un resumen corto y legible para Render."""
        now = time.time()
        if now - self._last_runtime_log_at < settings.RUNTIME_LOG_INTERVAL_SECONDS:
            return
        self._last_runtime_log_at = now
        logger.info(
            "Collector resumen: trades={} candles={} reconciled={} duplicates={} gaps={} avg_lag_ms={}",
            self.stats.trades_received,
            self.stats.candles_published,
            self.stats.reconciled_candles,
            self.stats.duplicates,
            self.stats.gaps,
            round(self.stats.avg_lag_ms, 3),
        )

    def _build_subscription_message(
        self,
        channel: str,
        products: list[str] | None = None,
    ) -> dict[str, Any]:
        """Construye el mensaje subscribe con JWT si hay auth habilitada."""
        payload: dict[str, Any] = {
            "type": "subscribe",
            "channel": channel,
        }
        if products:
            payload["product_ids"] = products
        if self._should_auth_ws():
            payload["jwt"] = self.coinbase_client.build_ws_jwt()
        return payload

    def _should_auth_ws(self) -> bool:
        """Indica si esta conexion de market data puede autenticarse."""
        return (
            self.enable_ws_auth
            and self.coinbase_client is not None
            and self.coinbase_client.has_private_credentials()
        )

    async def _publish_closed_candle(self, payload: dict[str, Any], reconciled: bool = False) -> None:
        """Publica velas cerradas evitando duplicados por bucket."""
        product_id = str(payload.get("product_id", "")).strip().upper()
        open_time = int(payload.get("open_time", 0))
        last_published = self._last_published_candle_open_by_product.get(product_id)
        if last_published is not None and open_time <= last_published:
            return

        self._last_published_candle_open_by_product[product_id] = open_time
        if reconciled:
            self.stats.reconciled_candles += 1
        self.stats.candles_published += 1
        await self._publish_stream(settings.STREAM_MARKET_CANDLES_1M, payload)

    async def _reconcile_recent_candles(
        self,
        products: list[str],
        reason: str,
        now_ms: int | None = None,
    ) -> None:
        """Rellena velas 1m faltantes via REST sin reemplazar velas ya emitidas."""
        if not self.enable_candle_reconcile or self.coinbase_client is None or not products:
            return

        current_ms = now_ms or int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        current_bucket_open = (current_ms // 60_000) * 60_000
        end_seconds = current_ms // 1000
        start_seconds = end_seconds - (self.reconcile_lookback_minutes * 60)
        prefer_private = self.coinbase_client.has_private_credentials()

        for product_id in products:
            last_started_at = self._last_reconcile_started_at_by_product.get(product_id, 0.0)
            started_at = datetime.now(tz=timezone.utc).timestamp()
            if started_at - last_started_at < self.reconcile_cooldown_seconds:
                continue
            self._last_reconcile_started_at_by_product[product_id] = started_at
            try:
                candles = await asyncio.to_thread(
                    self.coinbase_client.get_candles,
                    product_id=product_id,
                    start=start_seconds,
                    end=end_seconds,
                    granularity=settings.COINBASE_CANDLE_RECONCILE_GRANULARITY,
                    prefer_private=prefer_private,
                )
            except Exception as exc:
                logger.warning(
                    "Collector reconcile fallo. product_id={} reason={} error={}",
                    product_id,
                    reason,
                    exc,
                )
                await self._publish_error(
                    "collector.reconcile_failed",
                    {
                        "product_id": product_id,
                        "reason": reason,
                        "error": str(exc),
                    },
                )
                continue

            for candle in candles:
                if int(candle["open_time"]) >= current_bucket_open:
                    continue
                await self._publish_closed_candle(candle, reconciled=True)

    async def _publish_error(self, error_type: str, payload: dict[str, Any]) -> None:
        body = {"error_type": error_type, "payload": json.dumps(payload)}
        await self._publish_stream(settings.STREAM_SYSTEM_ERRORS, body)

    async def _publish_stream(self, stream: str, payload: dict[str, Any]) -> None:
        if self.redis_client is None:
            return
        mapping = {key: json.dumps(value) if isinstance(value, (dict, list)) else str(value) for key, value in payload.items()}
        await self.redis_client.xadd(stream, mapping, maxlen=10000, approximate=True)

    @staticmethod
    def _parse_time_to_ms(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).replace("Z", "+00:00")
        return int(datetime.fromisoformat(text).timestamp() * 1000)

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0
