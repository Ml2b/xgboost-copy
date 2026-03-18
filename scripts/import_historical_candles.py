"""Importa velas historicas a Redis streams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import redis

from config import settings
from historical_utils import normalize_candles, read_candles_file


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI."""
    parser = argparse.ArgumentParser(description="Importa velas historicas al stream de Redis.")
    parser.add_argument("--path", required=True, help="Ruta al CSV o Parquet.")
    parser.add_argument("--product-id", default=None, help="Producto a usar si el archivo no lo trae.")
    parser.add_argument("--redis-host", default=settings.REDIS_HOST, help="Host Redis.")
    parser.add_argument("--redis-port", type=int, default=settings.REDIS_PORT, help="Puerto Redis.")
    parser.add_argument("--stream", default=settings.STREAM_MARKET_CANDLES_1M, help="Stream destino.")
    parser.add_argument("--clear-stream", action="store_true", help="Borra el stream antes de insertar.")
    parser.add_argument("--limit", type=int, default=0, help="Limita la cantidad de velas cargadas.")
    parser.add_argument("--dry-run", action="store_true", help="Solo valida y resume, sin escribir en Redis.")
    return parser


def main() -> None:
    """Punto de entrada CLI."""
    args = build_parser().parse_args()
    raw = read_candles_file(args.path)
    candles = normalize_candles(raw, product_id=args.product_id, candle_seconds=settings.CANDLE_SECONDS)
    if args.limit > 0:
        candles = candles.tail(args.limit).reset_index(drop=True)

    summary = {
        "rows": len(candles),
        "products": sorted(candles["product_id"].unique().tolist()),
        "start_open_time": int(candles["open_time"].min()) if len(candles) else None,
        "end_open_time": int(candles["open_time"].max()) if len(candles) else None,
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        print(candles.head(3).to_json(orient="records", indent=2))
        return

    client = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    if args.clear_stream:
        client.delete(args.stream)

    inserted = 0
    for row in candles.to_dict(orient="records"):
        client.xadd(args.stream, {key: str(value) for key, value in row.items()})
        inserted += 1

    summary["stream"] = args.stream
    summary["inserted"] = inserted
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
