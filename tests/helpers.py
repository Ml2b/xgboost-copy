"""Helpers comunes para construir datasets sinteticos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_candles(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Genera velas sinteticas con tendencia, ciclos y ruido."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=float)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.18, size=n)) + 1.8 * np.sin(idx / 12.0) + (idx * 0.015)
    open_ = np.concatenate([[close[0]], close[:-1]]) + rng.normal(0.0, 0.05, size=n)
    spread = np.abs(rng.normal(0.22, 0.04, size=n)) + 0.05
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(900.0 + 120.0 * np.sin(idx / 9.0) + rng.normal(0.0, 40.0, size=n)) + 1.0
    open_time = 1_700_000_000_000 + (idx.astype(int) * 60_000)
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

