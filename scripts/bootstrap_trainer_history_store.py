"""Backfill del historico del trainer desde los CSV ya descargados."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from data.history_store import CandleHistoryStore
from historical_utils import normalize_candles, read_candles_file


DEFAULT_SEARCH_DIRS: tuple[Path, ...] = (
    Path("data/historical_6m_binanceus"),
    Path("tests/legacy/data/historical_phase1_6m_kucoin"),
    Path("tests/legacy/data/historical_phase2_6m_coinbase"),
)


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI del bootstrap historico."""
    parser = argparse.ArgumentParser(
        description="Puebla el SQLite del trainer con el historico principal ya descargado."
    )
    parser.add_argument(
        "--db-path",
        default=settings.TRAINER_HISTORY_DB_PATH,
        help="Ruta del SQLite historico del trainer.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Tamano de chunk para insertar en SQLite.",
    )
    parser.add_argument(
        "--bases-csv",
        default=",".join(settings.STUDIED_UNIVERSE_BASES),
        help="Lista de activos base separados por coma.",
    )
    parser.add_argument(
        "--report-path",
        default="reports/trainer_history_bootstrap.json",
        help="Ruta donde guardar el reporte JSON.",
    )
    return parser


def bootstrap_trainer_history_store(
    db_path: str | Path = settings.TRAINER_HISTORY_DB_PATH,
    target_bases: list[str] | None = None,
    chunk_size: int = 5000,
    search_dirs: list[Path] | None = None,
) -> dict[str, object]:
    """Importa el universo principal al SQLite del trainer y devuelve un resumen."""
    store = CandleHistoryStore(db_path=db_path)
    bases = target_bases or list(settings.STUDIED_UNIVERSE_BASES)
    resolved_search_dirs = search_dirs or list(DEFAULT_SEARCH_DIRS)

    results: list[dict[str, object]] = []
    for base in bases:
        try:
            history_path = resolve_history_file(base, resolved_search_dirs)
        except FileNotFoundError:
            results.append(
                {
                    "base": base,
                    "status": "missing_file",
                    "file": None,
                    "source_rows": 0,
                    "persisted_rows": store.get_row_count_for_base(base),
                }
            )
            continue

        raw = read_candles_file(history_path)
        normalized = normalize_candles(raw, product_id=None, candle_seconds=settings.CANDLE_SECONDS)
        normalized = filter_frame_for_base(normalized, base)
        if normalized.empty:
            results.append(
                {
                    "base": base,
                    "status": "empty_after_filter",
                    "file": str(history_path),
                    "source_rows": 0,
                    "persisted_rows": store.get_row_count_for_base(base),
                }
            )
            continue

        inserted = store.upsert_frame(
            normalized,
            source_name=str(history_path),
            chunk_size=chunk_size,
        )
        persisted_rows = store.get_row_count_for_base(base)
        results.append(
            {
                "base": base,
                "status": "backfilled",
                "file": str(history_path),
                "product_ids": sorted(normalized["product_id"].astype(str).str.upper().unique().tolist()),
                "source_rows": int(len(normalized)),
                "inserted_rows": int(inserted),
                "persisted_rows": int(persisted_rows),
                "min_open_time": int(normalized["open_time"].min()),
                "max_open_time": int(normalized["open_time"].max()),
            }
        )

    return {
        "db_path": str(Path(db_path)),
        "asset_count": len(results),
        "backfilled_count": sum(1 for item in results if item["status"] == "backfilled"),
        "missing_count": sum(1 for item in results if item["status"] != "backfilled"),
        "results": results,
    }


def resolve_history_file(base: str, search_dirs: list[Path]) -> Path:
    """Busca el CSV o parquet historico mas relevante para un activo base."""
    normalized_base = base.strip().lower()
    patterns = (
        f"{normalized_base}_*.csv",
        f"{normalized_base}_*.parquet",
        f"{normalized_base}_*.pq",
    )
    for directory in search_dirs:
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"No se encontro historico para {base}")


def filter_frame_for_base(frame: pd.DataFrame, base: str) -> pd.DataFrame:
    """Asegura que el DataFrame solo contenga el activo esperado."""
    normalized = frame.copy()
    base_column = (
        normalized["product_id"]
        .astype(str)
        .str.upper()
        .str.replace("/", "-", regex=False)
        .str.split("-", n=1)
        .str[0]
    )
    return normalized[base_column == base.strip().upper()].reset_index(drop=True)


def main() -> None:
    """Punto de entrada CLI."""
    args = build_parser().parse_args()
    bases = [base.strip().upper() for base in args.bases_csv.split(",") if base.strip()]
    summary = bootstrap_trainer_history_store(
        db_path=args.db_path,
        target_bases=bases,
        chunk_size=args.chunk_size,
    )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
