"""Script de bootstrap: descarga OHLCV + order flow historico desde Binance.

Ejemplos de uso:
    # Ultimos 30 dias para todos los pares del bot
    python scripts/bootstrap_order_flow.py --days 30

    # Rango especifico para BTC y ETH
    python scripts/bootstrap_order_flow.py --symbols BTCUSDT ETHUSDT --start 2024-01-01 --end 2024-12-31

    # Solo 7 dias rapido para probar
    python scripts/bootstrap_order_flow.py --days 7 --symbols BTCUSDT

Que descarga:
    - klines 1m (OHLCV) → tabla candles
    - aggTrades 1m agregados → 13 ORDER_FLOW_TRADE_COLUMNS en candle_order_flow
    - Los 32 columns de L2 (spread, book depth) = 0.0 (requieren collector live)

Tiempo estimado: ~2-4 segundos por dia/par (throttle gentil a Binance).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Asegurar que el root del proyecto este en el path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from config import settings
from data.downloader import DEFAULT_SYMBOLS, BinanceVisionDownloader
from data.history_store import CandleHistoryStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap de datos historicos desde data.binance.vision"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Numero de dias hacia atras desde hoy (default: 30)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Fecha inicio ISO (YYYY-MM-DD). Sobreescribe --days.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Fecha fin ISO (YYYY-MM-DD). Default: ayer.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Simbolos Binance USDT a descargar (default: todos los del bot)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(settings.TRAINER_HISTORY_DB_PATH),
        help="Ruta al SQLite de historia (default: settings.TRAINER_HISTORY_DB_PATH)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.4,
        help="Segundos entre requests (default: 0.4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    end_date = args.end or (date.today() - timedelta(days=1)).isoformat()
    if args.start:
        start_date = args.start
    else:
        end = date.fromisoformat(end_date)
        start_date = (end - timedelta(days=args.days - 1)).isoformat()

    logger.info(
        "Bootstrap iniciado. simbolos={} start={} end={} db={}",
        len(args.symbols),
        start_date,
        end_date,
        args.db,
    )

    store = CandleHistoryStore(db_path=args.db)
    downloader = BinanceVisionDownloader(
        store=store,
        request_sleep=args.sleep,
    )

    summary = downloader.download_range(
        symbols=args.symbols,
        start_date=start_date,
        end_date=end_date,
    )

    total_candles = sum(summary.values())
    successful = sum(1 for v in summary.values() if v > 0)
    logger.info(
        "Bootstrap completado. pares_con_datos={}/{} total_candles={}",
        successful,
        len(args.symbols),
        total_candles,
    )

    # Resumen por par
    for symbol, candles in sorted(summary.items(), key=lambda x: -x[1]):
        if candles > 0:
            logger.info("  {} → {} candles", symbol, candles)


if __name__ == "__main__":
    main()
