"""Orquestador principal del sistema de trading."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
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
        self.candles_consumed = 0
        self.features_published = 0
        self._last_runtime_log_at = 0.0

    async def start(self, stop_event: asyncio.Event | None = None) -> None:
        stop_event = stop_event or asyncio.Event()
        logger.info("FeatureEngine iniciado. buffer_size={}", settings.FEATURE_BUFFER_SIZE)
        await self._ensure_group(settings.STREAM_MARKET_CANDLES_1M, "feature-engine")
        while not stop_event.is_set():
            try:
                messages = await self.redis_client.xreadgroup(
                    groupname="feature-engine",
                    consumername="feature-1",
                    streams={settings.STREAM_MARKET_CANDLES_1M: ">"},
                    count=25,
                    block=1000,
                )
            except Exception as exc:
                if "NOGROUP" in str(exc):
                    logger.warning("FeatureEngine: NOGROUP detectado, recreando consumer group")
                    await self._ensure_group(settings.STREAM_MARKET_CANDLES_1M, "feature-engine")
                    continue
                raise
            for _, stream_messages in messages:
                for message_id, payload in stream_messages:
                    await self._process_candle(payload)
                    await self.redis_client.xack(settings.STREAM_MARKET_CANDLES_1M, "feature-engine", message_id)

    async def _process_candle(self, payload: dict[str, Any]) -> None:
        product_id = payload.get("product_id", "")
        self.candles_consumed += 1
        normalized = {
            "product_id": product_id,
            "open_time": int(payload["open_time"]),
            "open": float(payload["open"]),
            "high": float(payload["high"]),
            "low": float(payload["low"]),
            "close": float(payload["close"]),
            "volume": float(payload["volume"]),
            "trade_count": int(float(payload.get("trade_count", 0))),
        }
        self.buffers[product_id].append(normalized)
        buffer = list(self.buffers[product_id])
        if len(buffer) < max(70, settings.FEATURE_BUFFER_SIZE // 2):
            return

        frame = pd.DataFrame(buffer)
        features_frame = self.calculator.compute(frame)
        if features_frame.empty:
            return
        latest = features_frame.iloc[-1].to_dict()
        latest["product_id"] = product_id
        await self.redis_client.xadd(
            settings.STREAM_MARKET_FEATURES,
            {key: str(value) for key, value in latest.items()},
        )
        self.features_published += 1
        self._log_runtime_summary()

    def _log_runtime_summary(self) -> None:
        """Resume el estado del motor de features sin ruido por vela."""
        now = time.time()
        if now - self._last_runtime_log_at < settings.RUNTIME_LOG_INTERVAL_SECONDS:
            return
        self._last_runtime_log_at = now
        logger.info(
            "FeatureEngine resumen: candles={} features={} activos_buffer={}",
            self.candles_consumed,
            self.features_published,
            len(self.buffers),
        )

    async def _ensure_group(self, stream: str, group_name: str) -> None:
        try:
            await self.redis_client.xgroup_create(stream, group_name, id="$", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise


class DiagnosticsServer:
    """Servidor HTTP minimalista que expone /status con métricas del bot en tiempo real."""

    _HOURS_BACK = 4.0
    _MAX_ROWS = 20_000

    def __init__(self, redis_client: Any, port: int = 10_000) -> None:
        self.redis_client = redis_client
        self.port = port

    async def start(self, stop_event: asyncio.Event | None = None) -> None:
        stop_event = stop_event or asyncio.Event()
        server = await asyncio.start_server(self._handle, "0.0.0.0", self.port)
        logger.info("DiagnosticsServer escuchando en puerto {}", self.port)
        async with server:
            await stop_event.wait()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
            request_line = raw.decode(errors="replace").strip()
            # Consumir headers (ignorar)
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if line in (b"\r\n", b"\n", b""):
                    break
        except Exception:
            writer.close()
            return

        path = request_line.split(" ")[1] if " " in request_line else "/"
        if path in ("/", "/status", "/health"):
            body = await self._build_status_json()
            content_type = "application/json"
            status_line = "200 OK"
        else:
            body = json.dumps({"error": "not found"})
            content_type = "application/json"
            status_line = "404 Not Found"

        body_bytes = body.encode()
        response = (
            f"HTTP/1.1 {status_line}\r\n"
            f"Content-Type: {content_type}; charset=utf-8\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode() + body_bytes
        try:
            writer.write(response)
            await writer.drain()
        finally:
            writer.close()

    async def _build_status_json(self) -> str:
        cutoff_ms = int((time.time() - self._HOURS_BACK * 3600) * 1000)
        min_id = f"{cutoff_ms}-0"
        since_24h = time.time() - 86400

        sig_rows = await self._xrange(settings.STREAM_INFERENCE_SIGNALS, min_id)
        exe_rows = await self._xrange(settings.STREAM_EXECUTION_EVENTS, min_id)
        paper_rows = await self._xrange(settings.STREAM_PAPER_EXECUTION_EVENTS, min_id)
        candle_stats = await asyncio.to_thread(self._analyse_candles_redis, min_id)
        history_stats = await asyncio.to_thread(self._analyse_history_store)
        retrain_stats = await asyncio.to_thread(self._analyse_retrains, since_24h)

        return json.dumps({
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "window_hours": self._HOURS_BACK,
            "process": self._process_info(),
            "signals": self._analyse_signals(sig_rows),
            "execution": self._analyse_execution(exe_rows),
            "paper": self._analyse_paper(paper_rows),
            "candles_stream": candle_stats,
            "history_store": history_stats,
            "retrains_24h": retrain_stats,
        }, indent=2)

    @staticmethod
    def _process_info() -> dict:
        """RAM y env vars de riesgo activos en el proceso."""
        info: dict = {
            "execution_dry_run": settings.EXECUTION_DRY_RUN,
            "execution_enabled": settings.EXECUTION_ENABLED,
            "paper_initial_cash": settings.PAPER_INITIAL_CASH,
            "max_risk_per_trade": settings.MAX_RISK_PER_TRADE,
            "max_spread_pct": settings.MAX_SPREAD_PCT,
            "max_daily_drawdown": settings.MAX_DAILY_DRAWDOWN,
            "pilot_order_notional_usd": settings.PILOT_ORDER_NOTIONAL_USD,
            "position_stop_loss_pct": settings.POSITION_STOP_LOSS_PCT,
            "position_take_profit_pct": settings.POSITION_TAKE_PROFIT_PCT,
            "position_max_hold_minutes": settings.POSITION_MAX_HOLD_MINUTES,
            "min_model_auc": settings.MIN_MODEL_AUC,
            "min_model_sharpe_for_promotion": settings.MIN_MODEL_SHARPE_FOR_PROMOTION,
            "max_model_drawdown_for_promotion": settings.MAX_MODEL_DRAWDOWN_FOR_PROMOTION,
        }
        try:
            mem = Path("/proc/meminfo").read_text()
            def _kb(label: str) -> int:
                for line in mem.splitlines():
                    if line.startswith(label):
                        return int(line.split()[1])
                return 0
            total_mb = round(_kb("MemTotal:") / 1024)
            avail_mb = round(_kb("MemAvailable:") / 1024)
            info["ram_total_mb"] = total_mb
            info["ram_available_mb"] = avail_mb
            info["ram_used_mb"] = total_mb - avail_mb
        except Exception:
            pass
        return info

    def _analyse_candles_redis(self, min_id: str) -> dict:
        """Cuenta velas en el stream de Redis para la ventana activa."""
        try:
            import redis as redis_sync
            if settings.REDIS_URL:
                r = redis_sync.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=5)
            else:
                r = redis_sync.Redis(
                    host=settings.REDIS_HOST, port=settings.REDIS_PORT,
                    decode_responses=True, socket_connect_timeout=5,
                )
            entries = r.xrange(settings.STREAM_MARKET_CANDLES_1M, min=min_id, count=self._MAX_ROWS)
            by_asset: Counter = Counter()
            for _msg_id, payload in entries:
                by_asset[payload.get("product_id", "?")] += 1
            return {
                "total": len(entries),
                "by_asset": dict(by_asset.most_common()),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _analyse_history_store(self) -> dict:
        """Lee el SQLite del trainer y reporta filas por activo."""
        import sqlite3
        db_path = Path(settings.TRAINER_HISTORY_DB_PATH)
        if not db_path.exists():
            return {"error": f"db not found: {db_path}"}
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT base_asset, COUNT(*) as n, "
                    "MAX(open_time) as latest_ms "
                    "FROM candles GROUP BY base_asset ORDER BY n DESC"
                ).fetchall()
                total = sum(r[1] for r in rows)
                now_ms = int(time.time() * 1000)
                by_asset = {
                    r[0]: {
                        "rows": r[1],
                        "latest_ago_min": round((now_ms - r[2]) / 60000, 1) if r[2] else None,
                    }
                    for r in rows
                }
            return {"total_rows": total, "assets": by_asset}
        except Exception as exc:
            return {"error": str(exc)}

    @staticmethod
    def _analyse_retrains(since_ts: float) -> dict:
        """Escanea los registry.json del model registry para contar retrains desde ayer."""
        import json as _json
        registry_root = Path(settings.MODEL_REGISTRY_ROOT)
        if not registry_root.exists():
            return {"error": f"registry not found: {registry_root}"}
        since_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(since_ts))
        total_trained = 0
        total_promoted = 0
        by_asset: dict[str, dict] = {}
        try:
            for asset_dir in sorted(registry_root.iterdir()):
                if not asset_dir.is_dir():
                    continue
                registry_path = asset_dir / "registry.json"
                if not registry_path.exists():
                    continue
                records = _json.loads(registry_path.read_text(encoding="utf-8"))
                recent = [r for r in records if r.get("created_at", "") >= since_iso]
                promoted = [r for r in recent if r.get("promoted")]
                total_trained += len(recent)
                total_promoted += len(promoted)
                if recent:
                    last = recent[-1]
                    by_asset[asset_dir.name] = {
                        "trained": len(recent),
                        "promoted": len(promoted),
                        "last_auc": round(last.get("metrics", {}).get("auc_val", 0), 4),
                        "last_trained_at": last.get("created_at", "")[:19],
                    }
            return {
                "since_utc": since_iso,
                "total_retrain_cycles": total_trained,
                "total_promoted": total_promoted,
                "by_asset": by_asset,
            }
        except Exception as exc:
            return {"error": str(exc)}

    async def _xrange(self, stream: str, min_id: str) -> list[dict]:
        try:
            results = await self.redis_client.xrange(stream, min=min_id, count=self._MAX_ROWS)
            return [payload for _msg_id, payload in results]
        except Exception as exc:
            logger.warning("DiagnosticsServer no pudo leer {}: {}", stream, exc)
            return []

    @staticmethod
    def _analyse_signals(rows: list[dict]) -> dict:
        total = len(rows)
        by_signal: Counter = Counter()
        by_asset: Counter = Counter()
        actionable = 0
        non_actionable_reasons: Counter = Counter()
        probs: list[float] = []
        for row in rows:
            sig = row.get("signal", "?").upper()
            by_signal[sig] += 1
            by_asset[row.get("product_id", "?")] += 1
            act = str(row.get("actionable", "false")).lower() == "true"
            if act:
                actionable += 1
            else:
                reason = row.get("reason") or "—"
                non_actionable_reasons[reason] += 1
            with contextlib.suppress(Exception):
                probs.append(float(row["prob_buy"]))
        return {
            "total": total,
            "actionable": actionable,
            "avg_prob_buy": round(sum(probs) / len(probs), 4) if probs else None,
            "by_signal": dict(by_signal.most_common()),
            "top_assets": dict(by_asset.most_common(10)),
            "non_actionable_reasons": dict(non_actionable_reasons.most_common(5)),
        }

    @staticmethod
    def _analyse_execution(rows: list[dict]) -> dict:
        by_decision: Counter = Counter()
        by_reason: Counter = Counter()
        dry_runs = 0
        live_orders = 0
        for row in rows:
            decision = row.get("decision", "?")
            by_decision[decision] += 1
            if decision == "accepted_dry_run":
                dry_runs += 1
            elif decision == "sent_live":
                live_orders += 1
            reason = row.get("reason", "")
            if reason:
                by_reason[reason] += 1
        return {
            "total": len(rows),
            "dry_runs": dry_runs,
            "live_orders": live_orders,
            "by_decision": dict(by_decision.most_common()),
            "block_reasons": dict(by_reason.most_common(10)),
        }

    @staticmethod
    def _analyse_paper(rows: list[dict]) -> dict:
        buy_fills = 0
        sell_fills = 0
        realized_pnl = 0.0
        last_equity: float | None = None
        last_cash: float | None = None
        last_drawdown: float | None = None
        fills_by_asset: Counter = Counter()
        pnl_by_asset: dict[str, float] = defaultdict(float)
        holding_ms_list: list[float] = []
        by_decision: Counter = Counter()
        for row in rows:
            decision = row.get("decision", "?")
            by_decision[decision] += 1
            asset = row.get("product_id", "?")
            if decision == "paper_buy_filled":
                buy_fills += 1
                fills_by_asset[asset] += 1
                with contextlib.suppress(Exception):
                    last_equity = float(row["equity"])
                    last_cash = float(row["cash"])
                    last_drawdown = float(row["drawdown_pct"])
            elif decision in {"paper_exit_signal_filled", "paper_exit_rule_filled", "paper_sell_filled"}:
                sell_fills += 1
                fills_by_asset[asset] += 1
                with contextlib.suppress(Exception):
                    pnl = float(row.get("realized_pnl", 0))
                    realized_pnl += pnl
                    pnl_by_asset[asset] += pnl  # type: ignore[index]
                with contextlib.suppress(Exception):
                    last_equity = float(row["equity"])
                    last_cash = float(row["cash"])
                    last_drawdown = float(row["drawdown_pct"])
                with contextlib.suppress(Exception):
                    holding_ms_list.append(float(row["holding_ms"]))
        return {
            "total": len(rows),
            "buy_fills": buy_fills,
            "sell_fills": sell_fills,
            "realized_pnl_usd": round(realized_pnl, 4),
            "last_equity_usd": round(last_equity, 4) if last_equity is not None else None,
            "last_cash_usd": round(last_cash, 4) if last_cash is not None else None,
            "last_drawdown_pct": round(last_drawdown * 100, 4) if last_drawdown is not None else None,
            "avg_holding_min": round(sum(holding_ms_list) / len(holding_ms_list) / 60_000, 2) if holding_ms_list else None,
            "fills_by_asset": dict(fills_by_asset.most_common()),
            "pnl_by_asset_usd": {k: round(v, 4) for k, v in sorted(pnl_by_asset.items(), key=lambda x: -abs(x[1]))},
            "by_decision": dict(by_decision.most_common()),
        }


async def run_service(name: str, factory, stop_event: asyncio.Event) -> None:
    """Mantiene vivo un servicio aunque falle y lo reinicia."""
    logger.info("Servicio {} lanzado", name)
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

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        "logs/trading_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention=7,
        level=log_level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
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
    logger.info("Universo observado cargado. products={} ", observed_products)
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
    diagnostics_port = int(os.getenv("PORT", "10000"))
    diagnostics = DiagnosticsServer(redis_client=redis_client_async, port=diagnostics_port)

    service_names = ["collector", "feature_engine", "inference", "trainer", "order_manager", "diagnostics"]
    services = [
        asyncio.create_task(run_service("collector", lambda: collector.start(stop_event), stop_event)),
        asyncio.create_task(run_service("feature_engine", lambda: feature_engine.start(stop_event), stop_event)),
        asyncio.create_task(run_service("inference", lambda: inference.start(stop_event), stop_event)),
        asyncio.create_task(run_service("trainer", lambda: trainer.start(stop_event), stop_event)),
        asyncio.create_task(run_service("order_manager", lambda: order_manager.start(stop_event), stop_event)),
        asyncio.create_task(run_service("diagnostics", lambda: diagnostics.start(stop_event), stop_event)),
    ]
    if settings.PAPER_TRADING_ENABLED:
        services.append(
            asyncio.create_task(run_service("paper_trader", lambda: paper_trader.start(stop_event), stop_event))
        )
        service_names.append("paper_trader")
    logger.info("Servicios principales activos: {}", service_names)

    try:
        await stop_event.wait()
    finally:
        for task in services:
            task.cancel()
        await asyncio.gather(*services, return_exceptions=True)
        if hasattr(redis_client_async, "aclose"):
            await redis_client_async.aclose()
        else:
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
