"""Persistencia historica de velas por activo para reentrenamiento."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings


class CandleHistoryStore:
    """Guarda todas las velas cerradas por activo y sincroniza desde Redis."""

    def __init__(self, db_path: str | Path = settings.TRAINER_HISTORY_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def sync_from_redis_stream(
        self,
        redis_client: Any,
        stream_name: str = settings.STREAM_MARKET_CANDLES_1M,
        batch_size: int = settings.TRAINER_HISTORY_SYNC_BATCH,
    ) -> int:
        """Importa nuevas velas desde Redis sin duplicar por producto y open_time."""
        last_stream_id = self.get_last_stream_id(stream_name)
        imported = 0
        range_start = "-" if last_stream_id is None else f"({last_stream_id}"

        while True:
            entries = redis_client.xrange(stream_name, min=range_start, max="+", count=batch_size)
            if not entries:
                break

            candles: list[dict[str, object]] = []
            latest_stream_id = last_stream_id
            for message_id, payload in entries:
                product_id = str(payload.get("product_id", "")).strip().upper()
                if not product_id:
                    continue
                candles.append(
                    {
                        "product_id": product_id,
                        "open_time": int(payload.get("open_time")),
                        "close_time": int(payload.get("close_time", payload.get("open_time", 0))),
                        "open": float(payload.get("open")),
                        "high": float(payload.get("high")),
                        "low": float(payload.get("low")),
                        "close": float(payload.get("close")),
                        "volume": float(payload.get("volume", 0.0)),
                        "trade_count": int(payload.get("trade_count", 0)),
                    }
                )
                latest_stream_id = message_id

            if candles:
                imported += self.upsert_candles(candles)
            if latest_stream_id:
                self.set_last_stream_id(stream_name, str(latest_stream_id))
                range_start = f"({latest_stream_id}"
            if len(entries) < batch_size:
                break

        return imported

    def upsert_candles(self, candles: list[dict[str, object]]) -> int:
        """Inserta o actualiza velas historicas sin perder la ultima version cerrada."""
        if not candles:
            return 0

        with sqlite3.connect(self.db_path) as connection:
            self._upsert_candles_with_connection(connection, candles)
            connection.commit()
        return len(candles)

    def upsert_frame(
        self,
        frame: pd.DataFrame,
        source_name: str | None = None,
        chunk_size: int = 5000,
    ) -> int:
        """Inserta un DataFrame normalizado de velas y registra metadatos del backfill."""
        if frame.empty:
            if source_name:
                self._record_bootstrap_load(
                    source_name=source_name,
                    row_count=0,
                    product_count=0,
                    min_open_time=None,
                    max_open_time=None,
                )
            return 0

        required = [
            "product_id",
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
        ]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"El frame no contiene columnas requeridas: {missing}")

        normalized = frame[required].copy()
        normalized["product_id"] = normalized["product_id"].astype(str).str.strip().str.upper()
        normalized["open_time"] = pd.to_numeric(normalized["open_time"], errors="raise").astype(int)
        normalized["close_time"] = pd.to_numeric(normalized["close_time"], errors="raise").astype(int)
        for column in ["open", "high", "low", "close", "volume"]:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(float)
        normalized["trade_count"] = pd.to_numeric(
            normalized["trade_count"],
            errors="coerce",
        ).fillna(0).astype(int)

        inserted = 0
        with sqlite3.connect(self.db_path) as connection:
            for start in range(0, len(normalized), chunk_size):
                chunk = normalized.iloc[start : start + chunk_size]
                chunk_records = [
                    {
                        "product_id": row.product_id,
                        "open_time": row.open_time,
                        "close_time": row.close_time,
                        "open": row.open,
                        "high": row.high,
                        "low": row.low,
                        "close": row.close,
                        "volume": row.volume,
                        "trade_count": row.trade_count,
                    }
                    for row in chunk.itertuples(index=False)
                ]
                self._upsert_candles_with_connection(connection, chunk_records)
                inserted += len(chunk_records)

            if source_name:
                self._record_bootstrap_load_with_connection(
                    connection=connection,
                    source_name=source_name,
                    row_count=len(normalized),
                    product_count=int(normalized["product_id"].nunique()),
                    min_open_time=int(normalized["open_time"].min()),
                    max_open_time=int(normalized["open_time"].max()),
                )
            connection.commit()
        return inserted

    def load_candles_for_base(self, base_asset: str) -> pd.DataFrame | None:
        """Carga el historico completo de un activo, ordenado por tiempo."""
        query = """
            SELECT
                product_id,
                open_time,
                close_time,
                open,
                high,
                low,
                close,
                volume,
                trade_count
            FROM candles
            WHERE base_asset = ?
            ORDER BY open_time ASC
        """
        with sqlite3.connect(self.db_path) as connection:
            frame = pd.read_sql_query(query, connection, params=[base_asset.strip().upper()])
        if frame.empty:
            return None
        return frame

    def get_row_count_for_base(self, base_asset: str) -> int:
        """Devuelve el numero de velas persistidas para un activo base."""
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "SELECT COUNT(*) FROM candles WHERE base_asset = ?",
                [base_asset.strip().upper()],
            )
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    def get_last_stream_id(self, stream_name: str) -> str | None:
        """Retorna el ultimo stream_id sincronizado para un stream concreto."""
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "SELECT last_stream_id FROM sync_state WHERE stream_name = ?",
                [stream_name],
            )
            row = cursor.fetchone()
        return str(row[0]) if row else None

    def set_last_stream_id(self, stream_name: str, last_stream_id: str) -> None:
        """Persista el ultimo stream_id importado para continuar incrementalmente."""
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO sync_state (stream_name, last_stream_id)
                VALUES (?, ?)
                ON CONFLICT(stream_name) DO UPDATE SET last_stream_id=excluded.last_stream_id
                """,
                [stream_name, last_stream_id],
            )
            connection.commit()

    def _initialize_schema(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS candles (
                    product_id TEXT NOT NULL,
                    base_asset TEXT NOT NULL,
                    open_time INTEGER NOT NULL,
                    close_time INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    trade_count INTEGER NOT NULL,
                    PRIMARY KEY (product_id, open_time)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_candles_base_asset_open_time
                ON candles(base_asset, open_time DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_state (
                    stream_name TEXT PRIMARY KEY,
                    last_stream_id TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS bootstrap_loads (
                    source_name TEXT PRIMARY KEY,
                    loaded_at_utc TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    product_count INTEGER NOT NULL,
                    min_open_time INTEGER,
                    max_open_time INTEGER
                )
                """
            )
            connection.commit()

    def _upsert_candles_with_connection(
        self,
        connection: sqlite3.Connection,
        candles: list[dict[str, object]],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO candles (
                product_id,
                base_asset,
                open_time,
                close_time,
                open,
                high,
                low,
                close,
                volume,
                trade_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_id, open_time) DO UPDATE SET
                close_time=excluded.close_time,
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                trade_count=excluded.trade_count
            """,
            [
                (
                    str(candle["product_id"]).strip().upper(),
                    str(candle["product_id"]).split("-", 1)[0].upper(),
                    int(candle["open_time"]),
                    int(candle["close_time"]),
                    float(candle["open"]),
                    float(candle["high"]),
                    float(candle["low"]),
                    float(candle["close"]),
                    float(candle["volume"]),
                    int(candle["trade_count"]),
                )
                for candle in candles
            ],
        )

    def _record_bootstrap_load(
        self,
        source_name: str,
        row_count: int,
        product_count: int,
        min_open_time: int | None,
        max_open_time: int | None,
    ) -> None:
        with sqlite3.connect(self.db_path) as connection:
            self._record_bootstrap_load_with_connection(
                connection=connection,
                source_name=source_name,
                row_count=row_count,
                product_count=product_count,
                min_open_time=min_open_time,
                max_open_time=max_open_time,
            )
            connection.commit()

    def _record_bootstrap_load_with_connection(
        self,
        connection: sqlite3.Connection,
        source_name: str,
        row_count: int,
        product_count: int,
        min_open_time: int | None,
        max_open_time: int | None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO bootstrap_loads (
                source_name,
                loaded_at_utc,
                row_count,
                product_count,
                min_open_time,
                max_open_time
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_name) DO UPDATE SET
                loaded_at_utc=excluded.loaded_at_utc,
                row_count=excluded.row_count,
                product_count=excluded.product_count,
                min_open_time=excluded.min_open_time,
                max_open_time=excluded.max_open_time
            """,
            [
                source_name,
                datetime.now(timezone.utc).isoformat(),
                int(row_count),
                int(product_count),
                min_open_time,
                max_open_time,
            ],
        )
