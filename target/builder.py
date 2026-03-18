"""Construccion de labels de trading sin leakage."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from config import settings


class TargetType(str, Enum):
    """Tipos de target soportados por el sistema."""

    NET_RETURN_THRESHOLD = "net_return_threshold"
    TRIPLE_BARRIER = "triple_barrier"


@dataclass(slots=True)
class TargetConfig:
    """Configuracion explicita del target."""

    target_type: TargetType
    horizon: int
    threshold_pct: float
    fee_pct: float = settings.FEE_PCT
    stop_loss_pct: float | None = None
    net_threshold_pct: float = field(init=False)

    def __post_init__(self) -> None:
        self.net_threshold_pct = self.threshold_pct - (self.fee_pct * 2.0)
        if self.net_threshold_pct <= 0:
            raise ValueError(
                "El threshold neto debe ser positivo despues de fees. "
                f"threshold_pct={self.threshold_pct}, fee_pct={self.fee_pct}"
            )
        if self.stop_loss_pct is None:
            self.stop_loss_pct = self.threshold_pct


class TargetBuilder:
    """Agrega la columna target y target_return al DataFrame."""

    def __init__(self, config: TargetConfig) -> None:
        self.config = config

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construye el target sobre una serie OHLC ordenada por tiempo."""
        if "close" not in df.columns:
            raise ValueError("El DataFrame debe contener la columna 'close'.")

        result = df.copy().sort_values("open_time").reset_index(drop=True)
        if self.config.target_type == TargetType.NET_RETURN_THRESHOLD:
            target, target_return = self._build_net_return_threshold(result)
        elif self.config.target_type == TargetType.TRIPLE_BARRIER:
            target, target_return = self._build_triple_barrier(result)
        else:
            raise ValueError(f"Target no soportado: {self.config.target_type}")

        result["target"] = target
        result["target_return"] = target_return
        result = result.iloc[: len(result) - self.config.horizon].copy()
        result["target"] = result["target"].astype(int)

        positive_ratio = float(result["target"].mean()) if len(result) else 0.0
        print(
            "[TargetBuilder] balance de clases: "
            f"positivos={int(result['target'].sum())}, total={len(result)}, "
            f"ratio={positive_ratio:.2%}"
        )
        return result.reset_index(drop=True)

    def _build_net_return_threshold(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        close = df["close"].astype(float).to_numpy()
        horizon = self.config.horizon
        fee = self.config.fee_pct / 100.0
        threshold = self.config.net_threshold_pct / 100.0

        gross = np.roll(close, -horizon) / close - 1.0
        net = gross - (fee * 2.0)
        target = (net > threshold).astype(int)
        return target, net

    def _build_triple_barrier(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        close = df["close"].astype(float).to_numpy()
        high = df.get("high", df["close"]).astype(float).to_numpy()
        low = df.get("low", df["close"]).astype(float).to_numpy()
        horizon = self.config.horizon
        fee = self.config.fee_pct / 100.0
        take_profit_pct = self.config.threshold_pct / 100.0
        stop_loss_pct = float(self.config.stop_loss_pct or self.config.threshold_pct) / 100.0

        target = np.zeros(len(df), dtype=int)
        target_return = np.zeros(len(df), dtype=float)

        for idx in range(len(df) - horizon):
            entry = close[idx]
            tp_level = entry * (1.0 + take_profit_pct)
            sl_level = entry * (1.0 - stop_loss_pct)
            exit_return = (close[idx + horizon] / entry) - 1.0 - (fee * 2.0)

            for future_idx in range(idx + 1, idx + horizon + 1):
                if high[future_idx] >= tp_level:
                    exit_return = take_profit_pct - (fee * 2.0)
                    target[idx] = 1
                    break
                if low[future_idx] <= sl_level:
                    exit_return = -stop_loss_pct - (fee * 2.0)
                    target[idx] = 0
                    break
            target_return[idx] = exit_return

        return target, target_return

