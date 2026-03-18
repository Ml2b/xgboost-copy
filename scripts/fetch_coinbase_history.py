"""Descarga velas historicas via CCXT y las guarda en CSV."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ccxt
import pandas as pd
from ccxt.base.errors import NetworkError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI."""
    parser = argparse.ArgumentParser(description="Descarga OHLCV historico desde un exchange CCXT.")
    parser.add_argument("--exchange", default="coinbase", help="Exchange CCXT, por ejemplo coinbase o bitstamp.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USD", "ETH/USD"],
        help="Simbolos a descargar.",
    )
    parser.add_argument("--timeframe", default="1m", help="Timeframe OHLCV.")
    parser.add_argument("--candles", type=int, default=2000, help="Cantidad total de velas por simbolo.")
    parser.add_argument("--batch-size", type=int, default=300, help="Tamano de cada llamada fetch_ohlcv.")
    parser.add_argument("--output-dir", default="data/historical", help="Directorio destino.")
    parser.add_argument("--max-retries", type=int, default=5, help="Reintentos por lote ante errores de red.")
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Backoff base para reintentos de red.",
    )
    return parser


def main() -> None:
    """Punto de entrada CLI."""
    args = build_parser().parse_args()
    exchange = getattr(ccxt, args.exchange)({"enableRateLimit": True})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in args.symbols:
        rows = fetch_history(
            exchange=exchange,
            symbol=symbol,
            timeframe=args.timeframe,
            candles=args.candles,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
        )
        frame = rows_to_frame(rows, symbol, timeframe=args.timeframe)
        filename = f"{symbol.replace('/', '_').lower()}_{args.timeframe}_{args.exchange}.csv"
        path = output_dir / filename
        frame.to_csv(path, index=False)
        print(f"{symbol}: {len(frame)} velas -> {path}")


def fetch_history(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    candles: int,
    batch_size: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[list[float]]:
    """Descarga velas en lotes hasta alcanzar el total pedido."""
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - (candles * timeframe_ms)
    rows: list[list[float]] = []

    while len(rows) < candles:
        limit = min(batch_size, candles - len(rows))
        batch: list[list[float]] | None = None
        for attempt in range(1, max_retries + 1):
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                break
            except NetworkError:
                if attempt == max_retries:
                    raise
                time.sleep(retry_backoff_seconds * attempt)
        if batch is None:
            break
        if not batch:
            break

        if rows:
            batch = [row for row in batch if row[0] > rows[-1][0]]
        if not batch:
            break

        rows.extend(batch)
        since = int(batch[-1][0]) + timeframe_ms
        time.sleep(max(exchange.rateLimit, 200) / 1000.0)

    return rows[-candles:]


def rows_to_frame(rows: list[list[float]], symbol: str, timeframe: str) -> pd.DataFrame:
    """Convierte OHLCV a DataFrame en el formato del proyecto."""
    frame = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    product_id = symbol.replace("/", "-")
    candle_ms = ccxt.Exchange.parse_timeframe(timeframe) * 1000
    frame["product_id"] = product_id
    frame["close_time"] = frame["open_time"] + candle_ms - 1
    frame["trade_count"] = 0
    return frame


if __name__ == "__main__":
    main()
