"""Ejecuta una sesion larga de paper trading y exporta un reporte por activo."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import fakeredis
import fakeredis.aioredis
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
from paper.paper_trader import PaperTrader
from scripts.live_multi_asset_probe import _prime_feature_buffers


STREAM_EXPORTS: list[tuple[str, str]] = [
    (settings.STREAM_MARKET_TRADES_RAW, "market_trades_raw.csv"),
    (settings.STREAM_MARKET_CANDLES_1M, "market_candles_1m.csv"),
    (settings.STREAM_MARKET_FEATURES, "market_features.csv"),
    (settings.STREAM_INFERENCE_SIGNALS, "inference_signals.csv"),
    (settings.STREAM_EXECUTION_EVENTS, "execution_events.csv"),
    (settings.STREAM_PAPER_EXECUTION_EVENTS, "paper_execution_events.csv"),
    (settings.STREAM_SYSTEM_HEALTH, "system_health.csv"),
    (settings.STREAM_SYSTEM_ERRORS, "system_errors.csv"),
]


async def _ensure_group(redis_client: fakeredis.aioredis.FakeRedis, stream: str, group: str) -> None:
    """Crea consumer groups desde el inicio del stream para no perder mensajes de arranque."""
    try:
        await redis_client.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI."""
    parser = argparse.ArgumentParser(description="Corre una sesion larga de paper trading multi-activo.")
    parser.add_argument("--seconds", type=int, default=3600, help="Duracion de la sesion en segundos.")
    parser.add_argument("--cohort-name", default="custom", help="Nombre logico de la cohorte/reporting.")
    parser.add_argument(
        "--session-root",
        default="data/live_sessions",
        help="Directorio raiz donde se guarda la sesion.",
    )
    parser.add_argument(
        "--allow-all-observed",
        action="store_true",
        help="Permite paper/dry-run para todos los activos observados con modelo promovido, sin tocar settings.",
    )
    parser.add_argument(
        "--registry-root",
        default=settings.MODEL_REGISTRY_ROOT,
        help="Root del registry multi-activo a usar durante la sesion.",
    )
    parser.add_argument(
        "--observed-bases-csv",
        default="",
        help="Lista opcional de activos base observados separada por comas.",
    )
    parser.add_argument(
        "--allowed-bases-csv",
        default="",
        help="Lista opcional de activos base habilitados para paper/dry-run.",
    )
    parser.add_argument("--paper-initial-cash", type=float, default=settings.PAPER_INITIAL_CASH)
    parser.add_argument("--order-notional-usd", type=float, default=settings.PILOT_ORDER_NOTIONAL_USD)
    parser.add_argument("--paper-fee-pct", type=float, default=settings.PAPER_FEE_PCT)
    parser.add_argument("--paper-slippage-pct", type=float, default=settings.PAPER_SLIPPAGE_PCT)
    parser.add_argument(
        "--paper-slippage-overrides-csv",
        default="",
        help="Overrides de slippage por activo, por ejemplo PENGU=2.0,ZEC=0.25.",
    )
    parser.add_argument(
        "--buy-threshold-overrides-csv",
        default="",
        help="Overrides de threshold BUY por activo, por ejemplo XLM=0.64.",
    )
    parser.add_argument(
        "--sell-threshold-overrides-csv",
        default="",
        help="Overrides de threshold SELL por activo, por ejemplo ZEC=0.20.",
    )
    return parser


async def main() -> None:
    """Ejecuta la sesion y exporta los resultados."""
    args = build_parser().parse_args()
    session_ts = datetime.now(tz=timezone.utc).strftime("session_%Y%m%dT%H%M%SZ")
    session_dir = Path(args.session_root) / session_ts
    session_dir.mkdir(parents=True, exist_ok=True)

    coinbase_client = CoinbaseAdvancedTradeClient()
    observed_bases = _parse_csv_list(args.observed_bases_csv) or list(settings.OBSERVED_BASES)
    resolved = coinbase_client.resolve_products_for_bases(observed_bases)
    products = list(resolved.values())
    if not products:
        raise RuntimeError("No se pudieron resolver productos observables en Coinbase.")

    explicit_allowed_bases = _parse_csv_list(args.allowed_bases_csv)
    if explicit_allowed_bases:
        allowed_bases = explicit_allowed_bases
    elif args.allow_all_observed:
        allowed_bases = list(observed_bases)
    else:
        allowed_bases = list(settings.EXECUTION_ALLOWED_BASES)
    registry = MultiAssetModelRegistry(
        root_dir=args.registry_root,
        execution_allowed_bases=allowed_bases,
    )
    buy_threshold_overrides = _parse_key_float_csv(args.buy_threshold_overrides_csv)
    sell_threshold_overrides = _parse_key_float_csv(args.sell_threshold_overrides_csv)

    server = fakeredis.FakeServer()
    redis_async = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    collector = CollectorWithCandles(
        redis_client=redis_async,
        coinbase_client=coinbase_client,
        products=products,
        products_per_connection=settings.COINBASE_WS_PRODUCTS_PER_CONNECTION,
    )
    feature_engine = FeatureEngine(redis_client=redis_async)
    inference = InferenceEngine(
        registry=registry,
        redis_client=redis_async,
        buy_thresholds_by_base=buy_threshold_overrides,
        sell_thresholds_by_base=sell_threshold_overrides,
    )
    order_manager = OrderManager(
        redis_client=redis_async,
        coinbase_client=coinbase_client,
        execution_enabled=settings.EXECUTION_ENABLED,
        dry_run=True,
        order_notional_usd=args.order_notional_usd,
        allowed_bases=allowed_bases,
    )
    slippage_overrides = _parse_key_float_csv(args.paper_slippage_overrides_csv)
    paper_trader = PaperTrader(
        redis_client=redis_async,
        initial_cash=args.paper_initial_cash,
        order_notional_usd=args.order_notional_usd,
        fee_pct=args.paper_fee_pct,
        slippage_pct=args.paper_slippage_pct,
        slippage_pct_by_base=slippage_overrides,
    )
    _prime_feature_buffers(feature_engine, products)
    await _ensure_group(redis_async, settings.STREAM_MARKET_CANDLES_1M, "feature-engine")
    await _ensure_group(redis_async, settings.STREAM_MARKET_FEATURES, "inference-engine")
    await _ensure_group(redis_async, settings.STREAM_INFERENCE_SIGNALS, "order-manager")
    await _ensure_group(redis_async, settings.STREAM_MARKET_CANDLES_1M, "paper-trader-candles")
    await _ensure_group(redis_async, settings.STREAM_INFERENCE_SIGNALS, "paper-trader-signals")

    stop_event = asyncio.Event()
    tasks = [
        asyncio.create_task(feature_engine.start(stop_event)),
        asyncio.create_task(inference.start(stop_event)),
        asyncio.create_task(order_manager.start(stop_event)),
        asyncio.create_task(paper_trader.start(stop_event)),
    ]
    await asyncio.sleep(0.25)
    tasks.append(asyncio.create_task(collector.start(stop_event)))

    try:
        await asyncio.sleep(args.seconds)
    finally:
        stop_event.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    report = await _build_and_export_report(
        redis_async=redis_async,
        session_dir=session_dir,
        seconds=args.seconds,
        observed_products=products,
        registry=registry,
        order_manager=order_manager,
        paper_trader=paper_trader,
        allowed_bases=allowed_bases,
        observed_bases=observed_bases,
        registry_root=args.registry_root,
        slippage_overrides=slippage_overrides,
        buy_threshold_overrides=buy_threshold_overrides,
        sell_threshold_overrides=sell_threshold_overrides,
        cohort_name=args.cohort_name,
    )
    print(json.dumps(report, indent=2))
    await redis_async.flushall()
    await redis_async.aclose()


async def _build_and_export_report(
    redis_async: Any,
    session_dir: Path,
    seconds: int,
    observed_products: list[str],
    registry: MultiAssetModelRegistry,
    order_manager: OrderManager,
    paper_trader: PaperTrader,
    allowed_bases: list[str],
    observed_bases: list[str],
    registry_root: str,
    slippage_overrides: dict[str, float],
    buy_threshold_overrides: dict[str, float],
    sell_threshold_overrides: dict[str, float],
    cohort_name: str,
) -> dict[str, Any]:
    """Exporta CSVs y construye el resumen JSON de la sesion."""
    stream_frames: dict[str, pd.DataFrame] = {}
    counts: dict[str, int] = {}
    for stream_name, filename in STREAM_EXPORTS:
        frame = await _fetch_stream_as_frame(redis_async, stream_name)
        stream_frames[stream_name] = frame
        counts[stream_name] = int(len(frame))
        frame.to_csv(session_dir / filename, index=False)

    candles_frame = stream_frames[settings.STREAM_MARKET_CANDLES_1M]
    if not candles_frame.empty and "product_id" in candles_frame.columns:
        for product_id, product_frame in candles_frame.groupby("product_id"):
            safe_name = str(product_id).replace("-", "_").lower()
            product_frame.to_csv(session_dir / f"{safe_name}_candles.csv", index=False)

    signals_frame = stream_frames[settings.STREAM_INFERENCE_SIGNALS]
    execution_frame = stream_frames[settings.STREAM_EXECUTION_EVENTS]
    paper_frame = stream_frames[settings.STREAM_PAPER_EXECUTION_EVENTS]
    errors_frame = stream_frames[settings.STREAM_SYSTEM_ERRORS]

    signals_by_product = Counter(_counter_key(signals_frame, "product_id"))
    signal_breakdown = Counter(
        f"{row.get('product_id', '')}:{row.get('signal', '')}"
        for row in _rows(signals_frame)
    )
    execution_decisions = Counter(_counter_key(execution_frame, "decision"))
    execution_by_product = Counter(
        f"{row.get('product_id', '')}:{row.get('decision', '')}"
        for row in _rows(execution_frame)
    )
    paper_decisions = Counter(_counter_key(paper_frame, "decision"))
    paper_by_product = Counter(
        f"{row.get('product_id', '')}:{row.get('decision', '')}"
        for row in _rows(paper_frame)
    )
    error_types = Counter(_counter_key(errors_frame, "error_type"))

    pnl_by_product = _build_pnl_by_product(observed_products, paper_frame, paper_trader)
    top_pnl = sorted(pnl_by_product.values(), key=lambda item: item["total_pnl"], reverse=True)[:5]
    worst_pnl = sorted(pnl_by_product.values(), key=lambda item: item["total_pnl"])[:5]

    report = {
        "session_dir": str(session_dir.resolve()),
        "cohort_name": cohort_name,
        "seconds": seconds,
        "registry_root": str(Path(registry_root)),
        "session_observed_bases": observed_bases,
        "observed_products": observed_products,
        "coinbase_resolution": {
            product.split("-", 1)[0]: product for product in observed_products
        },
        "session_allowed_bases": allowed_bases,
        "paper_config": {
            "initial_cash": paper_trader.initial_cash,
            "order_notional_usd": paper_trader.order_notional_usd,
            "fee_pct": paper_trader.fee_pct,
            "slippage_pct": paper_trader.slippage_pct,
            "slippage_pct_by_base": slippage_overrides,
        },
        "inference_config": {
            "default_buy_threshold": settings.MIN_SIGNAL_PROB,
            "default_sell_threshold": 1.0 - settings.MIN_SIGNAL_PROB,
            "buy_threshold_overrides": buy_threshold_overrides,
            "sell_threshold_overrides": sell_threshold_overrides,
        },
        "actionable_registry_keys": registry.get_actionable_registry_keys(),
        "counts": counts,
        "signals_by_product": dict(signals_by_product),
        "signal_breakdown": dict(signal_breakdown),
        "execution_decisions": dict(execution_decisions),
        "execution_by_product": dict(execution_by_product),
        "paper_decisions": dict(paper_decisions),
        "paper_by_product": dict(paper_by_product),
        "error_types": dict(error_types),
        "paper_state": asdict(paper_trader.current_state()),
        "order_manager_stats": asdict(order_manager.stats),
        "pnl_by_product": pnl_by_product,
        "top_pnl_products": top_pnl,
        "worst_pnl_products": worst_pnl,
    }
    (session_dir / "session_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_pnl_by_product(
    observed_products: list[str],
    paper_frame: pd.DataFrame,
    paper_trader: PaperTrader,
) -> dict[str, dict[str, Any]]:
    """Consolida PnL realizado y no realizado por activo."""
    summary: dict[str, dict[str, Any]] = {
        product_id: {
            "product_id": product_id,
            "buy_fills": 0,
            "sell_fills": 0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "open_quantity": 0.0,
            "entry_price": 0.0,
            "last_price": 0.0,
        }
        for product_id in observed_products
    }

    for row in _rows(paper_frame):
        product_id = str(row.get("product_id", "")).strip().upper()
        if not product_id or product_id not in summary:
            continue
        decision = str(row.get("decision", ""))
        if decision == "paper_buy_filled":
            summary[product_id]["buy_fills"] += 1
        elif decision == "paper_sell_filled":
            summary[product_id]["sell_fills"] += 1
            summary[product_id]["realized_pnl"] += _safe_float(row.get("realized_pnl", 0.0))

    for product_id, position in paper_trader.positions.items():
        last_price = float(paper_trader.last_close_by_product.get(product_id, position.entry_price))
        unrealized = (last_price - position.entry_price) * position.quantity
        product_summary = summary.setdefault(
            product_id,
            {
                "product_id": product_id,
                "buy_fills": 0,
                "sell_fills": 0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "total_pnl": 0.0,
                "open_quantity": 0.0,
                "entry_price": 0.0,
                "last_price": 0.0,
            },
        )
        product_summary["unrealized_pnl"] = round(unrealized, 8)
        product_summary["open_quantity"] = round(position.quantity, 12)
        product_summary["entry_price"] = round(position.entry_price, 8)
        product_summary["last_price"] = round(last_price, 8)

    for product_id, values in summary.items():
        total = float(values["realized_pnl"]) + float(values["unrealized_pnl"])
        values["realized_pnl"] = round(float(values["realized_pnl"]), 8)
        values["total_pnl"] = round(total, 8)
    return summary


async def _fetch_stream_as_frame(redis_async: Any, stream_name: str) -> pd.DataFrame:
    """Carga todas las entradas de un stream en un DataFrame."""
    entries = await redis_async.xrange(stream_name)
    rows: list[dict[str, Any]] = []
    for message_id, payload in entries:
        normalized = {"stream_id": message_id}
        normalized.update(payload)
        rows.append(normalized)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _counter_key(frame: pd.DataFrame, column: str) -> list[str]:
    """Extrae una columna para contadores sin NaN."""
    if frame.empty or column not in frame.columns:
        return []
    return [str(value) for value in frame[column].tolist() if str(value)]


def _rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convierte un DataFrame en filas dict para resumir."""
    if frame.empty:
        return []
    return frame.fillna("").to_dict(orient="records")


def _safe_float(value: Any) -> float:
    """Convierte a float tolerando strings vacios."""
    try:
        return float(value)
    except Exception:
        return 0.0


def _parse_csv_list(raw: str) -> list[str]:
    """Normaliza una lista CSV simple."""
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _parse_key_float_csv(raw: str) -> dict[str, float]:
    """Parsea overrides tipo BASE=1.5,OTRA=0.2."""
    overrides: dict[str, float] = {}
    for chunk in raw.split(","):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip().upper()
        value = value.strip()
        if not key or not value:
            continue
        try:
            overrides[key] = float(value)
        except ValueError:
            continue
    return overrides


if __name__ == "__main__":
    asyncio.run(main())
