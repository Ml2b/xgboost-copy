"""Calculo de features tecnicas reutilizable en training y en live."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.order_flow import (
    ORDER_FLOW_DERIVED_COLUMNS,
    ORDER_FLOW_FEATURE_COLUMNS,  # noqa: F401 — re-exported for callers
    ORDER_FLOW_RAW_COLUMNS,
    OrderFlowFeatureCalculator,
)


FEATURE_COLUMNS: list[str] = [
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_30",
    "ret_60",
    "signed_return_1_volume",
    "rsi_14",
    "rsi_divergence",
    "macd_hist",
    "macd_cross_up",
    "macd_cross_down",
    "atr_pct",
    "bb_pct",
    "bb_width",
    "realized_vol_5",
    "realized_vol_20",
    "price_vs_ema_fast",
    "price_vs_ema_slow",
    "price_vs_ema_trend",
    "ema_fast_above_slow",
    "ema_full_alignment",
    "vwap_20_distance",
    "position_in_range_20",
    "range_compression_20",
    "trend_slope_pct",
    "volume_ratio",
    "volume_spike",
    "volume_trend",
    "trade_intensity_20",
    "hl_ratio",
    "close_position_in_candle",
    "body_to_range_ratio",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]


@dataclass(slots=True)
class FeatureCalculator:
    """Calcula el set completo de features del proyecto."""

    rsi_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50
    volume_ma_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    trend_period: int = 20

    def compute(
        self,
        df: pd.DataFrame,
        df_order_flow: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Agrega todas las features y elimina filas sin historia suficiente."""
        required = {"open_time", "open", "high", "low", "close", "volume"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

        result = df.copy().sort_values("open_time").reset_index(drop=True)
        close = result["close"].astype(float).to_numpy()
        high = result["high"].astype(float).to_numpy()
        low = result["low"].astype(float).to_numpy()
        open_ = result["open"].astype(float).to_numpy()
        volume = result["volume"].astype(float).to_numpy()

        result["ret_1"] = self._pct_return(close, 1)
        result["ret_5"] = self._pct_return(close, 5)
        result["ret_15"] = self._pct_return(close, 15)
        result["ret_30"] = self._pct_return(close, 30)
        result["ret_60"] = self._pct_return(close, 60)
        result["signed_return_1_volume"] = result["ret_1"] * np.log1p(volume)

        rsi = self._rsi_wilder(close, self.rsi_period)
        result["rsi_14"] = rsi
        result["rsi_divergence"] = self._rsi_divergence(close, rsi)

        macd_line, signal_line, hist = self._macd(close)
        result["macd_hist"] = hist
        result["macd_cross_up"] = self._cross_up(macd_line, signal_line).astype(float)
        result["macd_cross_down"] = self._cross_down(macd_line, signal_line).astype(float)

        atr = self._atr(high, low, close, self.atr_period)
        result["atr_pct"] = np.where(close > 0, atr / close, np.nan)

        bb_upper, bb_mid, bb_lower = self._bollinger(close)
        band_range = bb_upper - bb_lower
        bb_pct = np.full(len(result), np.nan, dtype=float)
        np.divide(close - bb_lower, band_range, out=bb_pct, where=band_range > 0)
        result["bb_pct"] = np.clip(bb_pct, 0.0, 1.0)
        bb_width = np.full(len(result), np.nan, dtype=float)
        np.divide(band_range, bb_mid, out=bb_width, where=bb_mid != 0)
        result["bb_width"] = bb_width

        log_ret = np.full(len(result), np.nan, dtype=float)
        if len(result) > 1:
            log_ret[1:] = np.diff(np.log(close))
        result["realized_vol_5"] = (
            pd.Series(log_ret).rolling(5, min_periods=5).std().to_numpy()
        )
        result["realized_vol_20"] = (
            pd.Series(log_ret).rolling(20, min_periods=20).std().to_numpy()
        )

        ema_fast = self._ema(close, self.ema_fast)
        ema_slow = self._ema(close, self.ema_slow)
        ema_trend = self._ema(close, self.ema_trend)
        result["price_vs_ema_fast"] = np.where(ema_fast != 0, (close / ema_fast - 1.0), np.nan)
        result["price_vs_ema_slow"] = np.where(ema_slow != 0, (close / ema_slow - 1.0), np.nan)
        result["price_vs_ema_trend"] = np.where(ema_trend != 0, (close / ema_trend - 1.0), np.nan)
        result["ema_fast_above_slow"] = (ema_fast > ema_slow).astype(float)
        result["ema_full_alignment"] = ((ema_fast > ema_slow) & (ema_slow > ema_trend)).astype(float)

        rolling_quote = pd.Series(close * volume).rolling(20, min_periods=20).sum().to_numpy()
        rolling_volume_20 = pd.Series(volume).rolling(20, min_periods=20).sum().to_numpy()
        vwap_20 = np.full(len(result), np.nan, dtype=float)
        np.divide(rolling_quote, rolling_volume_20, out=vwap_20, where=rolling_volume_20 > 0)
        vwap_distance = np.full(len(result), np.nan, dtype=float)
        np.divide(close - vwap_20, vwap_20, out=vwap_distance, where=vwap_20 > 0)
        result["vwap_20_distance"] = vwap_distance

        high_20 = pd.Series(high).rolling(20, min_periods=20).max().to_numpy()
        low_20 = pd.Series(low).rolling(20, min_periods=20).min().to_numpy()
        range_20 = high_20 - low_20
        position_in_range = np.full(len(result), np.nan, dtype=float)
        np.divide(close - low_20, range_20, out=position_in_range, where=range_20 > 0)
        result["position_in_range_20"] = np.clip(position_in_range, 0.0, 1.0)
        avg_range_20 = pd.Series(high - low).rolling(20, min_periods=20).mean().to_numpy()
        range_compression = np.full(len(result), np.nan, dtype=float)
        np.divide(high - low, avg_range_20, out=range_compression, where=avg_range_20 > 0)
        result["range_compression_20"] = range_compression
        result["trend_slope_pct"] = self._rolling_slope(close, self.trend_period)

        vol_mean = pd.Series(volume).rolling(self.volume_ma_period, min_periods=self.volume_ma_period).mean().to_numpy()
        volume_ratio = np.full(len(result), np.nan, dtype=float)
        np.divide(volume, vol_mean, out=volume_ratio, where=vol_mean > 0)
        result["volume_ratio"] = volume_ratio
        result["volume_spike"] = (result["volume_ratio"] > 2.0).astype(float)
        result["volume_trend"] = self._rolling_slope(volume, 10)

        trade_count = self._resolve_trade_count(result, close, open_)
        rolling_trade_count = pd.Series(trade_count).rolling(20, min_periods=20).sum().to_numpy()
        trade_intensity = np.full(len(result), np.nan, dtype=float)
        np.divide(rolling_trade_count, rolling_volume_20, out=trade_intensity, where=rolling_volume_20 > 0)
        result["trade_intensity_20"] = trade_intensity

        candle_range = high - low
        body = np.abs(close - open_)
        hl_ratio = np.full(len(result), np.nan, dtype=float)
        np.divide(candle_range, close, out=hl_ratio, where=close > 0)
        result["hl_ratio"] = hl_ratio
        close_position = np.full(len(result), np.nan, dtype=float)
        np.divide(close - low, candle_range, out=close_position, where=candle_range > 0)
        result["close_position_in_candle"] = np.clip(close_position, 0.0, 1.0)
        body_ratio = np.full(len(result), np.nan, dtype=float)
        np.divide(body, candle_range, out=body_ratio, where=candle_range > 0)
        result["body_to_range_ratio"] = body_ratio

        ts = pd.to_datetime(result["open_time"], unit="ms", utc=True)
        hour_angle = (ts.dt.hour.to_numpy() / 24.0) * (2.0 * np.pi)
        dow_angle = (ts.dt.dayofweek.to_numpy() / 7.0) * (2.0 * np.pi)
        result["hour_sin"] = np.sin(hour_angle)
        result["hour_cos"] = np.cos(hour_angle)
        result["day_of_week_sin"] = np.sin(dow_angle)
        result["day_of_week_cos"] = np.cos(dow_angle)

        if df_order_flow is not None and not df_order_flow.empty:
            # Merge raw columns first so OHLCV+raw are available together
            # Excluir columnas ya presentes en result (e.g. trade_count de OHLCV)
            raw_cols = [
                c for c in ORDER_FLOW_RAW_COLUMNS
                if c in df_order_flow.columns and c not in result.columns
            ]
            if raw_cols:
                result = result.merge(
                    df_order_flow[["open_time"] + raw_cols],
                    on="open_time",
                    how="left",
                )
                for col in raw_cols:
                    result[col] = result[col].fillna(0.0)
            # Compute all derived features over the merged OHLCV+raw df
            of_calc = OrderFlowFeatureCalculator()
            result = of_calc.compute_all(result)
            for col in ORDER_FLOW_DERIVED_COLUMNS:
                if col in result.columns:
                    result[col] = result[col].fillna(0.0)

        result = result.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        return result

    @property
    def feature_columns(self) -> list[str]:
        """Lista canonica de features: 36 OHLCV + 73 order flow = 109 total."""
        return FEATURE_COLUMNS + ORDER_FLOW_FEATURE_COLUMNS

    @staticmethod
    def _pct_return(values: np.ndarray, lag: int) -> np.ndarray:
        result = np.full(values.shape[0], np.nan, dtype=float)
        if lag >= len(values):
            return result
        result[lag:] = values[lag:] / values[:-lag] - 1.0
        return result

    @staticmethod
    def _resolve_trade_count(
        frame: pd.DataFrame,
        close: np.ndarray,
        open_: np.ndarray,
    ) -> np.ndarray:
        """Usa trade_count real si existe; si no, genera un proxy estable."""
        proxy = np.where(close > 0, (np.abs(close - open_) / close) * 10_000.0, 1.0)
        if "trade_count" in frame.columns:
            trade_count = frame["trade_count"].astype(float).to_numpy()
            invalid_mask = ~np.isfinite(trade_count) | (trade_count <= 0)
            if np.any(invalid_mask):
                trade_count = trade_count.copy()
                trade_count[invalid_mask] = proxy[invalid_mask]
            return np.maximum(trade_count, 1.0)
        return np.maximum(proxy, 1.0)

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().to_numpy()

    @staticmethod
    def _rsi_wilder(values: np.ndarray, period: int) -> np.ndarray:
        deltas = np.diff(values, prepend=np.nan)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = pd.Series(gains).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = pd.Series(losses).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0).to_numpy()

    @staticmethod
    def _rsi_divergence(close: np.ndarray, rsi: np.ndarray, window: int = 5) -> np.ndarray:
        divergence = np.zeros_like(close, dtype=float)
        for idx in range(window, len(close)):
            price_prev = close[idx - window : idx]
            rsi_prev = rsi[idx - window : idx]
            if np.isnan(rsi_prev).any():
                continue
            if close[idx] < np.nanmin(price_prev) and rsi[idx] > np.nanmin(rsi_prev):
                divergence[idx] = 1.0
            elif close[idx] > np.nanmax(price_prev) and rsi[idx] < np.nanmax(rsi_prev):
                divergence[idx] = -1.0
        return divergence

    def _macd(self, close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ema_fast = self._ema(close, self.macd_fast)
        ema_slow = self._ema(close, self.macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(
            span=self.macd_signal,
            adjust=False,
            min_periods=self.macd_signal,
        ).mean().to_numpy()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def _cross_up(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        prev_a = np.roll(a, 1)
        prev_b = np.roll(b, 1)
        out = (a > b) & (prev_a <= prev_b)
        out[0] = False
        return out

    @staticmethod
    def _cross_down(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        prev_a = np.roll(a, 1)
        prev_b = np.roll(b, 1)
        out = (a < b) & (prev_a >= prev_b)
        out[0] = False
        return out

    @staticmethod
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        true_range = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ])
        return pd.Series(true_range).ewm(alpha=1 / period, adjust=False, min_periods=period).mean().to_numpy()

    def _bollinger(self, close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        series = pd.Series(close)
        mid = series.rolling(self.bb_period, min_periods=self.bb_period).mean()
        std = series.rolling(self.bb_period, min_periods=self.bb_period).std()
        upper = mid + (std * self.bb_std)
        lower = mid - (std * self.bb_std)
        return upper.to_numpy(), mid.to_numpy(), lower.to_numpy()

    @staticmethod
    def _rolling_slope(values: np.ndarray, period: int) -> np.ndarray:
        slopes = np.full(len(values), np.nan, dtype=float)
        x = np.arange(period, dtype=float)
        x_centered = x - x.mean()
        denom = np.sum(x_centered ** 2)
        for idx in range(period - 1, len(values)):
            window = values[idx - period + 1 : idx + 1]
            if np.isnan(window).any():
                continue
            y_centered = window - window.mean()
            slope = float(np.sum(x_centered * y_centered) / denom)
            base = float(window.mean())
            slopes[idx] = slope / base if base != 0 else 0.0
        return slopes
