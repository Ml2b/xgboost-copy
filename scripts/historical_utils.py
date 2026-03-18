"""Helpers para cargar y normalizar velas historicas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TIME_COLUMN_ALIASES: dict[str, list[str]] = {
    "open_time": ["open_time", "timestamp", "time", "date", "datetime"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c"],
    "volume": ["volume", "v"],
    "product_id": ["product_id", "symbol", "ticker"],
    "close_time": ["close_time"],
    "trade_count": ["trade_count", "trades"],
}


def read_candles_file(path: str | Path) -> pd.DataFrame:
    """Lee un archivo CSV o Parquet con velas."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    raise ValueError(f"Formato no soportado: {file_path.suffix}")


def normalize_candles(
    df: pd.DataFrame,
    product_id: str | None,
    candle_seconds: int,
) -> pd.DataFrame:
    """Normaliza nombres y tipos al contrato de velas del proyecto."""
    renamed = df.copy()
    rename_map: dict[str, str] = {}
    for target, aliases in TIME_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in renamed.columns:
                rename_map[alias] = target
                break
    renamed = renamed.rename(columns=rename_map)

    required = ["open_time", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in renamed.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    normalized = pd.DataFrame()
    normalized["open_time"] = to_epoch_ms(renamed["open_time"])
    normalized["open"] = pd.to_numeric(renamed["open"], errors="raise")
    normalized["high"] = pd.to_numeric(renamed["high"], errors="raise")
    normalized["low"] = pd.to_numeric(renamed["low"], errors="raise")
    normalized["close"] = pd.to_numeric(renamed["close"], errors="raise")
    normalized["volume"] = pd.to_numeric(renamed["volume"], errors="raise")

    if "product_id" in renamed.columns:
        normalized["product_id"] = renamed["product_id"].astype(str)
    elif product_id:
        normalized["product_id"] = product_id
    else:
        raise ValueError("Debes pasar --product-id si el archivo no trae product_id.")

    if product_id:
        normalized = normalized[normalized["product_id"] == product_id].copy()

    if "close_time" in renamed.columns:
        normalized["close_time"] = to_epoch_ms(renamed["close_time"])
    else:
        normalized["close_time"] = normalized["open_time"] + (candle_seconds * 1000) - 1

    if "trade_count" in renamed.columns:
        normalized["trade_count"] = pd.to_numeric(renamed["trade_count"], errors="coerce").fillna(0).astype(int)
    else:
        normalized["trade_count"] = 0

    normalized = normalized.dropna().sort_values(["product_id", "open_time"]).drop_duplicates(
        subset=["product_id", "open_time"],
        keep="last",
    )
    normalized["open_time"] = normalized["open_time"].astype(int)
    normalized["close_time"] = normalized["close_time"].astype(int)
    return normalized.reset_index(drop=True)


def to_epoch_ms(series: pd.Series) -> pd.Series:
    """Convierte timestamps comunes a milisegundos Unix."""
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        max_value = float(numeric.dropna().max()) if not numeric.dropna().empty else 0.0
        if max_value >= 1e17:
            return (numeric // 1_000_000).astype("Int64")
        if max_value >= 1e14:
            return (numeric // 1_000).astype("Int64")
        if max_value >= 1e11:
            return numeric.astype("Int64")
        return (numeric * 1000).round().astype("Int64")

    parsed = pd.to_datetime(series, utc=True, errors="raise")
    return (parsed.view("int64") // 1_000_000).astype("Int64")
