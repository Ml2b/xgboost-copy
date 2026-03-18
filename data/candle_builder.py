"""Construccion de velas de 1 minuto a partir de trades individuales."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from config import settings


@dataclass(slots=True)
class Candle:
    """Representa una vela cerrada o abierta."""

    product_id: str
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    is_closed: bool

    def to_dict(self) -> dict[str, object]:
        """Serializa la vela para Redis o logs."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Candle":
        """Reconstruye la vela desde un diccionario."""
        return cls(
            product_id=str(payload["product_id"]),
            open_time=int(payload["open_time"]),
            close_time=int(payload["close_time"]),
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=float(payload["volume"]),
            trade_count=int(payload["trade_count"]),
            is_closed=bool(payload["is_closed"]),
        )


class CandleBuilder:
    """Agrupa trades dentro del bucket temporal configurado."""

    def __init__(self, candle_seconds: int = settings.CANDLE_SECONDS) -> None:
        self.candle_seconds = candle_seconds
        self._bucket_ms = candle_seconds * 1000
        self._open_candles: dict[str, Candle] = {}

    def add_trade(
        self,
        product_id: str,
        price: float,
        size: float,
        ts_ms: int,
    ) -> Optional[Candle]:
        """Agrega un trade y retorna la vela cerrada si el minuto cambio."""
        bucket_open = self._bucket_open(ts_ms)
        candle = self._open_candles.get(product_id)

        if candle is None:
            self._open_candles[product_id] = self._new_candle(
                product_id=product_id,
                bucket_open=bucket_open,
                price=price,
                size=size,
            )
            return None

        if bucket_open == candle.open_time:
            candle.high = max(candle.high, price)
            candle.low = min(candle.low, price)
            candle.close = price
            candle.volume += size
            candle.trade_count += 1
            candle.close_time = ts_ms
            return None

        # Ignora trades tardios para no rebobinar la vela abierta actual.
        if bucket_open < candle.open_time:
            return None

        closed = Candle(
            product_id=candle.product_id,
            open_time=candle.open_time,
            close_time=candle.open_time + self._bucket_ms - 1,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            trade_count=candle.trade_count,
            is_closed=True,
        )
        self._open_candles[product_id] = self._new_candle(
            product_id=product_id,
            bucket_open=bucket_open,
            price=price,
            size=size,
        )
        return closed

    def force_close(self, product_id: str, ts_ms: int) -> Optional[Candle]:
        """Cierra la vela abierta de un producto al apagar el sistema."""
        candle = self._open_candles.pop(product_id, None)
        if candle is None:
            return None
        return Candle(
            product_id=candle.product_id,
            open_time=candle.open_time,
            close_time=max(ts_ms, candle.open_time),
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            trade_count=candle.trade_count,
            is_closed=True,
        )

    def _bucket_open(self, ts_ms: int) -> int:
        return (ts_ms // self._bucket_ms) * self._bucket_ms

    def _new_candle(
        self,
        product_id: str,
        bucket_open: int,
        price: float,
        size: float,
    ) -> Candle:
        return Candle(
            product_id=product_id,
            open_time=bucket_open,
            close_time=bucket_open,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=size,
            trade_count=1,
            is_closed=False,
        )
