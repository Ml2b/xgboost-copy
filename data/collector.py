"""Collector de trades Coinbase con publicacion en Redis."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from typing import Any

from config import settings
from data.candle_builder import CandleBuilder
from exchange.coinbase_client import CoinbaseAdvancedTradeClient


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

    @property
    def avg_lag_ms(self) -> float:
        """Promedio de lag medido en milisegundos."""
        if self.lag_samples == 0:
            return 0.0
        return self.lag_ms_total / self.lag_samples


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
        self._seen_hashes: deque[str] = deque(maxlen=500)
        self._seen_lookup: set[str] = set()
        self._last_sequence_by_connection: dict[str, int] = {}
        self._last_heartbeat_counter_by_connection: dict[str, int] = {}
        self._products_by_connection_key: dict[str, list[str]] = {
            self._connection_key(batch): list(batch)
            for batch in self._build_product_batches()
        }
        self._last_published_candle_open_by_product: dict[str, int] = {}
        self._last_reconcile_started_at_by_product: dict[str, float] = {}

    async def start(self, stop_event: asyncio.Event | None = None) -> None:
        """Arranca una o varias conexiones con reconexion exponencial."""
        stop_event = stop_event or asyncio.Event()
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
        backoff = 2
        attempts = 0
        connection_key = self._connection_key(products)
        while not stop_event.is_set():
            try:
                await self._run_socket(products, stop_event)
                backoff = 2
                attempts = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempts += 1
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
        async with websockets.connect(self.websocket_url, ping_interval=20, ping_timeout=20) as websocket:
            await websocket.send(json.dumps(self._build_subscription_message(settings.COINBASE_CHANNEL, products)))
            await websocket.send(json.dumps(self._build_subscription_message(settings.COINBASE_HEARTBEATS_CHANNEL)))

            while not stop_event.is_set():
                raw_message = await asyncio.wait_for(websocket.recv(), timeout=settings.WS_TIMEOUT_SECONDS)
                payload = json.loads(raw_message)
                await self._handle_message(payload, connection_key=connection_key)

    async def _handle_message(self, message: dict[str, Any], connection_key: str) -> None:
        await self._track_connection_health(message, connection_key)
        await self._track_connection_sequence(message, connection_key)
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
            self.stats.trades_received += 1
            self.stats.lag_ms_total += max(0, now_ms - ts_ms)
            self.stats.lag_samples += 1

            await self._publish_stream(settings.STREAM_MARKET_TRADES_RAW, trade)
            candle = self.candle_builder.add_trade(product_id, price, size, ts_ms)
            if candle is not None:
                await self._publish_closed_candle(candle.to_dict())

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
                },
            )
            await asyncio.sleep(settings.HEALTH_PUBLISH_INTERVAL)

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
                candles = self.coinbase_client.get_candles(
                    product_id=product_id,
                    start=start_seconds,
                    end=end_seconds,
                    granularity=settings.COINBASE_CANDLE_RECONCILE_GRANULARITY,
                    prefer_private=prefer_private,
                )
            except Exception as exc:
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
        await self.redis_client.xadd(stream, mapping)

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
