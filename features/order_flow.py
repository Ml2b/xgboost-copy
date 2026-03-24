"""Order flow features - implementacion completa de los 14 grupos del estudio."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ===========================================================================
# Column group definitions — mantener separados para facilitar seleccion
# ===========================================================================

# Group 1: flujo ejecutado (raw desde trades)
ORDER_FLOW_TRADE_COLUMNS: list[str] = [
    "buy_aggressive_volume",
    "sell_aggressive_volume",
    "total_traded_volume",
    "trade_count",
    "buy_count",
    "sell_count",
    "avg_trade_size",
    "median_trade_size",
    "max_trade_size",
    "volume_delta",
    "delta_ratio",
    "buy_sell_ratio",
    "trade_vwap",
]

# Group 4: spread y calidad del mercado (desde L2 top of book)
ORDER_FLOW_SPREAD_COLUMNS: list[str] = [
    "spread",
    "relative_spread",
    "mid_price",
    "best_bid_size",
    "best_ask_size",
    "top_of_book_imbalance",
    "microprice",
    "microprice_shift",
]

# Group 3: libro de ordenes (desde L2 profundidad)
ORDER_FLOW_BOOK_DEPTH_COLUMNS: list[str] = [
    "bid_depth_top5",
    "bid_depth_top10",
    "bid_depth_top20",
    "ask_depth_top5",
    "ask_depth_top10",
    "ask_depth_top20",
    "depth_imbalance_top10",
    "depth_slope_bid",
    "depth_slope_ask",
    "cumul_depth_1bp",
    "cumul_depth_5bp",
    "cumul_depth_10bp",
]

# Groups 5-7: consumo, absorcion, reposicion (desde L2 + trades, calculado en collector)
ORDER_FLOW_BOOK_DYNAMICS_COLUMNS: list[str] = [
    "ask_consumption_rate",
    "bid_consumption_rate",
    "net_consumption",
    "price_move_per_unit_flow",
    "seller_absorption_estimate",
    "buyer_absorption_estimate",
    "absorption_score",
    "flow_to_price_divergence",
    "refill_rate_ask",
    "refill_rate_bid",
    "cancel_rate_ask",
    "cancel_rate_bid",
    # --- Cancellation/execution separation ---
    "cancelled_bid_vol",
    "cancelled_ask_vol",
    "bid_cancel_ratio",
    "ask_cancel_ratio",
    "phantom_bid_vol",
    "cancel_asymmetry",
]

# Todas las columnas raw (se almacenan en Redis stream y SQLite)
ORDER_FLOW_RAW_COLUMNS: list[str] = (
    ORDER_FLOW_TRADE_COLUMNS
    + ORDER_FLOW_SPREAD_COLUMNS
    + ORDER_FLOW_BOOK_DEPTH_COLUMNS
    + ORDER_FLOW_BOOK_DYNAMICS_COLUMNS
)

# Groups 2, 11: desequilibrio y eficiencia (derivados de trades + precio OHLCV)
ORDER_FLOW_TRADE_DERIVED: list[str] = [
    "cvd_5",
    "cvd_15",
    "delta_acceleration",
    "volume_delta_normalized",
    "trade_size_skew",
    "buy_pressure_pct",
    "flow_efficiency",
    "net_flow_efficiency",
]

# Group 8: reaccion del precio (adicionales no presentes en calculator.py)
ORDER_FLOW_PRICE_REACTION: list[str] = [
    "ret_3m",
    "break_of_local_high",
    "break_of_local_low",
    "distance_to_recent_high",
    "distance_to_recent_low",
]

# Group 10: multi-timeframe (desde OHLCV rolling)
ORDER_FLOW_MTF: list[str] = [
    "trend_5m",
    "trend_15m",
    "trend_1h",
    "vwap_distance_15m",
    "vwap_distance_1h",
    "vol_state_1h",
]

# Group 12: regimenes
ORDER_FLOW_REGIME: list[str] = [
    "trend_regime",
    "volatility_regime",
    "spread_regime",
    "activity_regime",
    "liquidity_regime",
]

# Group 13: operativas
ORDER_FLOW_OPERATIONAL: list[str] = [
    "current_spread_cost",
    "fee_impact",
    "execution_quality_estimate",
]

ORDER_FLOW_DERIVED_COLUMNS: list[str] = (
    ORDER_FLOW_TRADE_DERIVED
    + ORDER_FLOW_PRICE_REACTION
    + ORDER_FLOW_MTF
    + ORDER_FLOW_REGIME
    + ORDER_FLOW_OPERATIONAL
)

ORDER_FLOW_FEATURE_COLUMNS: list[str] = (
    ORDER_FLOW_RAW_COLUMNS + ORDER_FLOW_DERIVED_COLUMNS
)

# Constantes de regimen
_FEE_PCT_DEFAULT = 0.0005  # 0.05% fee por defecto


# ===========================================================================
# OrderFlowWindow — acumula trades de una ventana 1m
# ===========================================================================


@dataclass(slots=True)
class OrderFlowWindow:
    """Acumula trades de una ventana de 1 minuto para un producto."""

    open_time_ms: int
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_count: int = 0
    sell_count: int = 0
    _sizes: list = field(default_factory=list)
    _vwap_num: float = 0.0

    def add_trade(self, price: float, size: float, side: str) -> None:
        self._sizes.append(size)
        self._vwap_num += price * size
        if str(side).upper() in ("BUY", "BID"):
            self.buy_volume += size
            self.buy_count += 1
        else:
            self.sell_volume += size
            self.sell_count += 1

    def to_metrics(self) -> dict:
        total = self.buy_volume + self.sell_volume
        n = len(self._sizes)
        sizes = np.array(self._sizes, dtype=float) if self._sizes else np.array([0.0])
        avg_size = float(np.mean(sizes)) if n > 0 else 0.0
        med_size = float(np.median(sizes)) if n > 0 else 0.0
        max_size = float(np.max(sizes)) if n > 0 else 0.0
        delta = self.buy_volume - self.sell_volume
        delta_ratio = delta / total if total > 0 else 0.0
        if self.sell_volume > 0:
            bsr = self.buy_volume / self.sell_volume
        elif self.buy_volume > 0:
            bsr = 10.0
        else:
            bsr = 1.0
        bsr = float(np.clip(bsr, 0.0, 10.0))
        trade_vwap = self._vwap_num / total if total > 0 else 0.0
        return {
            "buy_aggressive_volume": self.buy_volume,
            "sell_aggressive_volume": self.sell_volume,
            "total_traded_volume": total,
            "trade_count": n,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "avg_trade_size": avg_size,
            "median_trade_size": med_size,
            "max_trade_size": max_size,
            "volume_delta": delta,
            "delta_ratio": delta_ratio,
            "buy_sell_ratio": bsr,
            "trade_vwap": trade_vwap,
        }


# ===========================================================================
# BookDepthTracker — rastrea L2 por producto y calcula metricas de libro
# ===========================================================================


class BookDepthTracker:
    """
    Mantiene el estado del libro L2 por producto.
    Recibe snapshots y updates del canal level2 de Coinbase.
    Al cierre de cada ventana calcula spread, profundidad, consumo,
    absorcion y tasa de reposicion.
    """

    def __init__(self) -> None:
        # bids/asks actuales: {price_float: size_float}
        self._bids: dict[str, dict[float, float]] = {}
        self._asks: dict[str, dict[float, float]] = {}
        # estado al inicio del periodo (para calcular refill/cancel)
        self._window_start_ask_depth10: dict[str, float] = {}
        self._window_start_bid_depth10: dict[str, float] = {}
        # acumuladores de cambios durante la ventana
        self._ask_added: dict[str, float] = {}   # liquidez que aparecio en asks
        self._bid_added: dict[str, float] = {}   # liquidez que aparecio en bids
        self._ask_removed: dict[str, float] = {} # liquidez que desaparecio en asks
        self._bid_removed: dict[str, float] = {} # liquidez que desaparecio en bids

    def apply_snapshot(
        self,
        product_id: str,
        bids: list[tuple[str, str]],
        asks: list[tuple[str, str]],
    ) -> None:
        """Reemplaza el estado completo del libro."""
        self._bids[product_id] = {float(p): float(s) for p, s in bids if float(s) > 0}
        self._asks[product_id] = {float(p): float(s) for p, s in asks if float(s) > 0}
        self._reset_window_accumulators(product_id)

    def apply_update(
        self,
        product_id: str,
        changes: list[tuple[str, str, str]],
    ) -> None:
        """
        Aplica cambios incrementales.
        changes: list de (side, price, new_size)
        """
        bids = self._bids.setdefault(product_id, {})
        asks = self._asks.setdefault(product_id, {})
        for side, price_str, size_str in changes:
            price = float(price_str)
            size = float(size_str)
            if side.upper() in ("BID", "BUY", "B"):
                old = bids.get(price, 0.0)
                if size == 0.0:
                    bids.pop(price, None)
                    removed = old
                    added = 0.0
                else:
                    bids[price] = size
                    added = max(0.0, size - old)
                    removed = max(0.0, old - size)
                self._bid_added[product_id] = self._bid_added.get(product_id, 0.0) + added
                self._bid_removed[product_id] = self._bid_removed.get(product_id, 0.0) + removed
            else:
                old = asks.get(price, 0.0)
                if size == 0.0:
                    asks.pop(price, None)
                    removed = old
                    added = 0.0
                else:
                    asks[price] = size
                    added = max(0.0, size - old)
                    removed = max(0.0, old - size)
                self._ask_added[product_id] = self._ask_added.get(product_id, 0.0) + added
                self._ask_removed[product_id] = self._ask_removed.get(product_id, 0.0) + removed

    def get_metrics_and_reset(
        self,
        product_id: str,
        buy_volume: float,
        sell_volume: float,
        price_change: float,
        volume_delta: float,
    ) -> dict:
        """
        Calcula todas las metricas del libro al cierre de la ventana y reinicia
        los acumuladores para la siguiente ventana.
        """
        bids = self._bids.get(product_id, {})
        asks = self._asks.get(product_id, {})
        metrics: dict = {}

        if not bids and not asks:
            self._reset_window_accumulators(product_id)
            return metrics

        # --- Top of book ---
        sorted_bids = sorted(bids.keys(), reverse=True)
        sorted_asks = sorted(asks.keys())
        best_bid = sorted_bids[0] if sorted_bids else 0.0
        best_ask = sorted_asks[0] if sorted_asks else 0.0
        best_bid_sz = bids.get(best_bid, 0.0)
        best_ask_sz = asks.get(best_ask, 0.0)

        mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
        spread = best_ask - best_bid if mid > 0 else 0.0
        rel_spread = spread / mid if mid > 0 else 0.0
        metrics["spread"] = spread
        metrics["relative_spread"] = rel_spread
        metrics["mid_price"] = mid
        metrics["best_bid_size"] = best_bid_sz
        metrics["best_ask_size"] = best_ask_sz

        top_imb = 0.0
        denom = best_bid_sz + best_ask_sz
        if denom > 0:
            top_imb = (best_bid_sz - best_ask_sz) / denom
        metrics["top_of_book_imbalance"] = top_imb

        micro = 0.0
        if denom > 0:
            micro = (best_ask * best_bid_sz + best_bid * best_ask_sz) / denom
        micro_shift = (micro - mid) / mid if mid > 0 else 0.0
        metrics["microprice"] = micro
        metrics["microprice_shift"] = micro_shift

        # --- Profundidad top N ---
        def depth_top(levels: list[float], book: dict, n: int) -> float:
            return sum(book.get(p, 0.0) for p in levels[:n])

        metrics["bid_depth_top5"] = depth_top(sorted_bids, bids, 5)
        metrics["bid_depth_top10"] = depth_top(sorted_bids, bids, 10)
        metrics["bid_depth_top20"] = depth_top(sorted_bids, bids, 20)
        metrics["ask_depth_top5"] = depth_top(sorted_asks, asks, 5)
        metrics["ask_depth_top10"] = depth_top(sorted_asks, asks, 10)
        metrics["ask_depth_top20"] = depth_top(sorted_asks, asks, 20)

        bid10 = metrics["bid_depth_top10"]
        ask10 = metrics["ask_depth_top10"]

        # --- WBI ponderado por 1/rank (Weight Book Imbalance) ---
        # Cada nivel recibe peso 1/rank, lo que da mas importancia a niveles
        # cercanos al mid sin necesitar volumen bruto.
        def wbi_by_rank(levels: list[float], book: dict, n: int) -> float:
            total = 0.0
            for rank, price in enumerate(levels[:n], start=1):
                total += book.get(price, 0.0) / rank
            return total

        wbi_bid = wbi_by_rank(sorted_bids, bids, 10)
        wbi_ask = wbi_by_rank(sorted_asks, asks, 10)
        wbi_total = wbi_bid + wbi_ask
        metrics["depth_imbalance_top10"] = (wbi_bid - wbi_ask) / wbi_total if wbi_total > 0 else 0.0

        # --- Pendiente de profundidad (depth slope) ---
        def depth_slope(levels: list[float], book: dict, n: int = 5) -> float:
            if len(levels) < 2:
                return 0.0
            pts = [book.get(levels[i], 0.0) for i in range(min(n, len(levels)))]
            if len(pts) < 2:
                return 0.0
            x = np.arange(len(pts), dtype=float)
            y = np.array(pts, dtype=float)
            denom = np.sum((x - x.mean()) ** 2)
            if denom == 0:
                return 0.0
            return float(np.sum((x - x.mean()) * (y - y.mean())) / denom)

        metrics["depth_slope_bid"] = depth_slope(sorted_bids, bids)
        metrics["depth_slope_ask"] = depth_slope(sorted_asks, asks)

        # --- Profundidad acumulada por distancia al mid ---
        def cumul_depth_bps(bps: float) -> float:
            if mid <= 0:
                return 0.0
            thresh = mid * bps / 10_000.0
            b = sum(v for p, v in bids.items() if mid - p <= thresh)
            a = sum(v for p, v in asks.items() if p - mid <= thresh)
            return b + a

        metrics["cumul_depth_1bp"] = cumul_depth_bps(1)
        metrics["cumul_depth_5bp"] = cumul_depth_bps(5)
        metrics["cumul_depth_10bp"] = cumul_depth_bps(10)

        # --- Tasas de consumo ---
        start_ask10 = self._window_start_ask_depth10.get(product_id, ask10)
        start_bid10 = self._window_start_bid_depth10.get(product_id, bid10)

        ask_cons = buy_volume / start_ask10 if start_ask10 > 0 else 0.0
        bid_cons = sell_volume / start_bid10 if start_bid10 > 0 else 0.0
        metrics["ask_consumption_rate"] = float(np.clip(ask_cons, 0.0, 1.0))
        metrics["bid_consumption_rate"] = float(np.clip(bid_cons, 0.0, 1.0))
        metrics["net_consumption"] = metrics["ask_consumption_rate"] - metrics["bid_consumption_rate"]

        total_vol = buy_volume + sell_volume
        net_flow = buy_volume - sell_volume
        pmpuf = price_change / net_flow if abs(net_flow) > 0 else 0.0
        metrics["price_move_per_unit_flow"] = float(np.clip(pmpuf, -1e3, 1e3))

        # --- Absorcion ---
        delta_thresh = 0.3   # delta_ratio que se considera significativo
        price_thresh = 0.0005  # 0.05% movimiento "chico" relativo a la presion

        seller_abs = 0.0
        if volume_delta > 0 and ask_cons > delta_thresh and abs(price_change) < price_thresh:
            seller_abs = float(np.clip(volume_delta / (total_vol + 1e-9), 0.0, 1.0))
        metrics["seller_absorption_estimate"] = seller_abs

        buyer_abs = 0.0
        if volume_delta < 0 and bid_cons > delta_thresh and abs(price_change) < price_thresh:
            buyer_abs = float(np.clip(-volume_delta / (total_vol + 1e-9), 0.0, 1.0))
        metrics["buyer_absorption_estimate"] = buyer_abs

        abs_score = float(np.clip(
            0.5 * (seller_abs + buyer_abs)
            + 0.3 * (1 - min(abs(metrics["net_consumption"]), 1.0))
            + 0.2 * (1 - min(abs(price_change) / (price_thresh * 10 + 1e-9), 1.0)),
            0.0, 1.0
        ))
        metrics["absorption_score"] = abs_score

        delta_ratio_val = volume_delta / total_vol if total_vol > 0 else 0.0
        flow_price_div = 0.0
        if abs(delta_ratio_val) > 0.3 and abs(price_change) < price_thresh:
            flow_price_div = abs(delta_ratio_val)
        metrics["flow_to_price_divergence"] = float(np.clip(flow_price_div, 0.0, 1.0))

        # --- Refill y cancel ---
        ask_added = self._ask_added.get(product_id, 0.0)
        bid_added = self._bid_added.get(product_id, 0.0)
        ask_removed = self._ask_removed.get(product_id, 0.0)
        bid_removed = self._bid_removed.get(product_id, 0.0)

        refill_ask = (ask_added - buy_volume) / start_ask10 if start_ask10 > 0 else 0.0
        refill_bid = (bid_added - sell_volume) / start_bid10 if start_bid10 > 0 else 0.0
        metrics["refill_rate_ask"] = float(np.clip(refill_ask, 0.0, 5.0))
        metrics["refill_rate_bid"] = float(np.clip(refill_bid, 0.0, 5.0))

        cancel_ask = max(0.0, ask_removed - buy_volume) / start_ask10 if start_ask10 > 0 else 0.0
        cancel_bid = max(0.0, bid_removed - sell_volume) / start_bid10 if start_bid10 > 0 else 0.0
        metrics["cancel_rate_ask"] = float(np.clip(cancel_ask, 0.0, 5.0))
        metrics["cancel_rate_bid"] = float(np.clip(cancel_bid, 0.0, 5.0))

        # --- Cancellation vs execution separation ---
        # cancelled_*_vol: liquidez que desaparecio SIN ser ejecutada por trades
        cancelled_bid = max(0.0, bid_removed - sell_volume)
        cancelled_ask = max(0.0, ask_removed - buy_volume)
        metrics["cancelled_bid_vol"] = cancelled_bid
        metrics["cancelled_ask_vol"] = cancelled_ask

        # cancel_ratio: fraccion de la liquidez removida que fue cancelacion
        total_bid_removed = bid_removed
        total_ask_removed = ask_removed
        metrics["bid_cancel_ratio"] = (
            cancelled_bid / total_bid_removed if total_bid_removed > 0 else 0.0
        )
        metrics["ask_cancel_ratio"] = (
            cancelled_ask / total_ask_removed if total_ask_removed > 0 else 0.0
        )

        # phantom_bid_vol: ordenes bid que aparecieron y se cancelaron
        # en la misma ventana (aparecieron como added, luego removidas sin ejecucion)
        # Proxy: min(bid_added, cancelled_bid) — la parte que entro y salio
        phantom_bid = min(bid_added, cancelled_bid)
        metrics["phantom_bid_vol"] = phantom_bid

        # cancel_asymmetry: sesgo de cancelacion bid vs ask [-1, 1]
        cancel_total = cancelled_bid + cancelled_ask
        metrics["cancel_asymmetry"] = (
            (cancelled_bid - cancelled_ask) / cancel_total if cancel_total > 0 else 0.0
        )

        # Guardar estado inicial para proxima ventana
        self._window_start_ask_depth10[product_id] = ask10
        self._window_start_bid_depth10[product_id] = bid10
        self._reset_window_accumulators(product_id)

        return metrics

    def _reset_window_accumulators(self, product_id: str) -> None:
        self._ask_added.pop(product_id, None)
        self._bid_added.pop(product_id, None)
        self._ask_removed.pop(product_id, None)
        self._bid_removed.pop(product_id, None)


# ===========================================================================
# OrderFlowAggregator — gestiona ventanas de trades + book tracker
# ===========================================================================


class OrderFlowAggregator:
    """
    Gestiona ventanas de order flow por producto.
    Integra datos de trades (market_trades) y libro (level2).
    """

    def __init__(self) -> None:
        self._windows: dict[str, OrderFlowWindow] = {}
        self._book: BookDepthTracker = BookDepthTracker()

    # ---- Trades ----

    def add_trade(
        self,
        product_id: str,
        price: float,
        size: float,
        side: str,
        ts_ms: int,
    ) -> None:
        if product_id not in self._windows:
            open_ms = (ts_ms // 60_000) * 60_000
            self._windows[product_id] = OrderFlowWindow(open_time_ms=open_ms)
        self._windows[product_id].add_trade(price, size, side)

    # ---- Level 2 ----

    def apply_book_snapshot(
        self,
        product_id: str,
        bids: list[tuple[str, str]],
        asks: list[tuple[str, str]],
    ) -> None:
        self._book.apply_snapshot(product_id, bids, asks)

    def apply_book_update(
        self,
        product_id: str,
        changes: list[tuple[str, str, str]],
    ) -> None:
        self._book.apply_update(product_id, changes)

    # ---- Cierre de ventana ----

    def close_window(
        self,
        product_id: str,
        open_time_ms: int,
        price_open: float = 0.0,
        price_close: float = 0.0,
    ) -> dict | None:
        """
        Cierra la ventana activa y retorna el dict de metricas raw completo.
        price_open / price_close son el open y close de la vela que acaba de cerrar.
        """
        window = self._windows.pop(product_id, None)
        if window is None or window.buy_count + window.sell_count == 0:
            return None

        metrics = window.to_metrics()
        price_change = (price_close - price_open) / price_open if price_open > 0 else 0.0

        # Metricas del libro (NaN si no hay L2 suscripto)
        book_metrics = self._book.get_metrics_and_reset(
            product_id,
            buy_volume=metrics["buy_aggressive_volume"],
            sell_volume=metrics["sell_aggressive_volume"],
            price_change=price_change,
            volume_delta=metrics["volume_delta"],
        )
        metrics.update(book_metrics)

        metrics["open_time"] = open_time_ms
        metrics["product_id"] = product_id
        return metrics


# ===========================================================================
# OrderFlowFeatureCalculator — calcula derivados sobre el df merged
# ===========================================================================


@dataclass(slots=True)
class OrderFlowFeatureCalculator:
    """
    Recibe un DataFrame que tiene columnas OHLCV + ORDER_FLOW_RAW_COLUMNS
    (después del merge en FeatureCalculator.compute) y calcula todos los
    ORDER_FLOW_DERIVED_COLUMNS.
    """

    cvd_window_short: int = 5
    cvd_window_long: int = 15
    delta_accel_window: int = 3
    local_window: int = 20
    fee_pct: float = _FEE_PCT_DEFAULT

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los ORDER_FLOW_DERIVED_COLUMNS sobre el df merged."""
        result = df.copy()

        if "volume_delta" in result.columns:
            result = self._compute_trade_derived(result)

        if "close" in result.columns:
            result = self._compute_price_reaction(result)
            result = self._compute_mtf(result)

        result = self._compute_regime(result)
        result = self._compute_operational(result)

        # Asegurar que todas las columnas existan (NaN si no se computaron)
        for col in ORDER_FLOW_DERIVED_COLUMNS:
            if col not in result.columns:
                result[col] = np.nan

        return result

    # Mantener compatibilidad con código existente
    def compute_from_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.compute_all(df)

    # ------------------------------------------------------------------
    # Groups 2 + 11: derivados de trades + precio
    # ------------------------------------------------------------------

    def _compute_trade_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = pd.Series(df["volume_delta"].astype(float))
        df["cvd_5"] = delta.rolling(self.cvd_window_short, min_periods=self.cvd_window_short).sum()
        df["cvd_15"] = delta.rolling(self.cvd_window_long, min_periods=self.cvd_window_long).sum()

        dr = pd.Series(df["delta_ratio"].astype(float))
        df["delta_acceleration"] = (
            dr - dr.shift(self.delta_accel_window)
        ).fillna(0.0).to_numpy()

        total = df["total_traded_volume"].to_numpy(dtype=float)
        vd = df["volume_delta"].to_numpy(dtype=float)
        norm = np.zeros(len(df), dtype=float)
        np.divide(vd, total, out=norm, where=total > 0)
        df["volume_delta_normalized"] = norm

        avg = df["avg_trade_size"].to_numpy(dtype=float)
        med = df["median_trade_size"].to_numpy(dtype=float)
        skew = np.ones(len(df), dtype=float)
        np.divide(avg, med, out=skew, where=med > 0)
        df["trade_size_skew"] = skew

        tc = df["trade_count"].to_numpy(dtype=float)
        bc = df["buy_count"].to_numpy(dtype=float)
        bpp = np.full(len(df), 0.5, dtype=float)
        np.divide(bc, tc, out=bpp, where=tc > 0)
        df["buy_pressure_pct"] = bpp

        # Group 11: eficiencia del flujo (within-candle)
        if "close" in df.columns and "open" in df.columns:
            pc = df["close"].to_numpy(dtype=float) - df["open"].to_numpy(dtype=float)
            fe = np.zeros(len(df), dtype=float)
            np.divide(pc, total, out=fe, where=total > 0)
            df["flow_efficiency"] = fe

            abs_delta = np.abs(vd)
            nfe = np.zeros(len(df), dtype=float)
            np.divide(pc, abs_delta, out=nfe, where=abs_delta > 1e-10)
            df["net_flow_efficiency"] = nfe
        else:
            df["flow_efficiency"] = np.nan
            df["net_flow_efficiency"] = np.nan

        return df

    # ------------------------------------------------------------------
    # Group 8: reaccion del precio (adiciones al calculator base)
    # ------------------------------------------------------------------

    def _compute_price_reaction(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float) if "high" in df.columns else close
        low = df["low"].to_numpy(dtype=float) if "low" in df.columns else close
        n = len(df)
        w = self.local_window

        # ret_3m: retorno de 3 velas (no esta en FEATURE_COLUMNS base)
        ret3 = np.full(n, np.nan)
        if n > 3:
            ret3[3:] = close[3:] / close[:-3] - 1.0
        df["ret_3m"] = ret3

        # break of local high/low
        high_w = pd.Series(high).rolling(w, min_periods=w).max().to_numpy()
        low_w = pd.Series(low).rolling(w, min_periods=w).min().to_numpy()

        brk_high = np.zeros(n, dtype=float)
        brk_low = np.zeros(n, dtype=float)
        valid = np.isfinite(high_w) & np.isfinite(low_w)
        # break_of_local_high: distancia porcentual al max reciente (positivo = ruptura)
        np.divide(close - high_w, high_w, out=brk_high, where=valid & (high_w > 0))
        np.divide(low_w - close, low_w, out=brk_low, where=valid & (low_w > 0))
        df["break_of_local_high"] = brk_high
        df["break_of_local_low"] = brk_low

        # distancias al high/low reciente
        dist_high = np.full(n, np.nan)
        dist_low = np.full(n, np.nan)
        np.divide(high_w - close, close, out=dist_high, where=valid & (close > 0))
        np.divide(close - low_w, close, out=dist_low, where=valid & (close > 0))
        df["distance_to_recent_high"] = dist_high
        df["distance_to_recent_low"] = dist_low

        return df

    # ------------------------------------------------------------------
    # Group 10: multi-timeframe
    # ------------------------------------------------------------------

    def _compute_mtf(self, df: pd.DataFrame) -> pd.DataFrame:
        close = pd.Series(df["close"].astype(float))
        n = len(df)

        # Tendencia como posicion del precio respecto a EMA de ventana
        def ema_trend(window: int) -> np.ndarray:
            ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
            trend = np.full(n, np.nan)
            valid = ema.notna().to_numpy()
            np.divide(
                (close.to_numpy() - ema.to_numpy()),
                ema.to_numpy(),
                out=trend,
                where=valid & (ema.to_numpy() > 0),
            )
            return trend

        df["trend_5m"] = ema_trend(5)
        df["trend_15m"] = ema_trend(15)
        df["trend_1h"] = ema_trend(60)

        # VWAP distance en ventanas mayores
        if "volume" in df.columns:
            vol = df["volume"].astype(float)
            for w, col in [(15, "vwap_distance_15m"), (60, "vwap_distance_1h")]:
                rq = (close * vol).rolling(w, min_periods=w).sum()
                rv = vol.rolling(w, min_periods=w).sum()
                vwap = np.full(n, np.nan)
                np.divide(rq.to_numpy(), rv.to_numpy(), out=vwap, where=rv.to_numpy() > 0)
                dist = np.full(n, np.nan)
                np.divide(
                    close.to_numpy() - vwap, vwap, out=dist,
                    where=np.isfinite(vwap) & (vwap > 0),
                )
                df[col] = dist
        else:
            df["vwap_distance_15m"] = np.nan
            df["vwap_distance_1h"] = np.nan

        # Estado de volatilidad en 1h: vol_5 / rolling_mean_vol_60
        if "realized_vol_5" in df.columns:
            rv5 = pd.Series(df["realized_vol_5"].astype(float))
            rv5_mean = rv5.rolling(60, min_periods=10).mean()
            vol_ratio = np.full(n, 1.0)
            np.divide(rv5.to_numpy(), rv5_mean.to_numpy(), out=vol_ratio,
                      where=rv5_mean.notna().to_numpy() & (rv5_mean.to_numpy() > 0))
            # 0=baja, 1=media, 2=alta
            df["vol_state_1h"] = np.where(vol_ratio < 0.7, 0.0,
                                          np.where(vol_ratio > 1.4, 2.0, 1.0))
        else:
            df["vol_state_1h"] = 1.0

        return df

    # ------------------------------------------------------------------
    # Group 12: regimenes
    # ------------------------------------------------------------------

    def _compute_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)

        # trend_regime: -1 bajista / 0 lateral / 1 alcista
        if "ema_full_alignment" in df.columns and "trend_slope_pct" in df.columns:
            alignment = df["ema_full_alignment"].to_numpy(dtype=float)
            slope = df["trend_slope_pct"].to_numpy(dtype=float)
            regime = np.where(alignment > 0.5, 1.0,
                              np.where(slope < -1e-4, -1.0, 0.0))
            df["trend_regime"] = regime
        elif "close" in df.columns:
            close = df["close"].to_numpy(dtype=float)
            ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().to_numpy()
            df["trend_regime"] = np.where(close > ema20, 1.0,
                                          np.where(close < ema20, -1.0, 0.0))
        else:
            df["trend_regime"] = 0.0

        # volatility_regime: 0 baja / 1 media / 2 alta
        if "realized_vol_5" in df.columns:
            rv = df["realized_vol_5"].to_numpy(dtype=float)
            rv_med = pd.Series(rv).rolling(100, min_periods=10).median().to_numpy()
            df["volatility_regime"] = np.where(rv < rv_med * 0.7, 0.0,
                                               np.where(rv > rv_med * 1.5, 2.0, 1.0))
        else:
            df["volatility_regime"] = 1.0

        # spread_regime: 0 normal / 1 ampliado (expanding para evitar look-ahead)
        if "relative_spread" in df.columns:
            rs = pd.Series(df["relative_spread"].to_numpy(dtype=float))
            threshold = rs.expanding(min_periods=10).quantile(0.75)
            threshold = threshold.fillna(0.001)
            df["spread_regime"] = (rs > threshold).astype(float)
        else:
            df["spread_regime"] = 0.0

        # activity_regime: 0 quieto / 1 normal / 2 activo
        if "trade_count" in df.columns:
            tc = df["trade_count"].to_numpy(dtype=float)
            tc_med = pd.Series(tc).rolling(60, min_periods=5).median().to_numpy()
            df["activity_regime"] = np.where(tc < tc_med * 0.5, 0.0,
                                             np.where(tc > tc_med * 1.8, 2.0, 1.0))
        else:
            df["activity_regime"] = 1.0

        # liquidity_regime: 0 profundo / 1 medio / 2 delgado
        if "depth_imbalance_top10" in df.columns and "bid_depth_top10" in df.columns:
            total_depth = (
                df["bid_depth_top10"].to_numpy(dtype=float)
                + df["ask_depth_top10"].to_numpy(dtype=float)
            )
            td_med = pd.Series(total_depth).rolling(60, min_periods=5).median().to_numpy()
            df["liquidity_regime"] = np.where(total_depth > td_med * 1.3, 0.0,
                                              np.where(total_depth < td_med * 0.6, 2.0, 1.0))
        elif "volume_ratio" in df.columns:
            vr = df["volume_ratio"].to_numpy(dtype=float)
            df["liquidity_regime"] = np.where(vr > 1.5, 0.0,
                                              np.where(vr < 0.5, 2.0, 1.0))
        else:
            df["liquidity_regime"] = 1.0

        return df

    # ------------------------------------------------------------------
    # Group 13: operativas
    # ------------------------------------------------------------------

    def _compute_operational(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)

        # current_spread_cost: coste del spread como fraccion del precio
        if "relative_spread" in df.columns:
            df["current_spread_cost"] = df["relative_spread"].astype(float) / 2.0
        elif "close" in df.columns and "spread" in df.columns:
            close = df["close"].to_numpy(dtype=float)
            spread = df["spread"].to_numpy(dtype=float)
            cost = np.zeros(n, dtype=float)
            np.divide(spread / 2.0, close, out=cost, where=close > 0)
            df["current_spread_cost"] = cost
        else:
            df["current_spread_cost"] = 0.0

        # fee_impact: constante (fee de maker en Coinbase)
        df["fee_impact"] = self.fee_pct

        # execution_quality_estimate: 1 = perfecta, 0 = mala
        spread_cost = df["current_spread_cost"].to_numpy(dtype=float)
        spread_regime = df.get("spread_regime", pd.Series(np.zeros(n))).to_numpy(dtype=float)
        liq_regime = df.get("liquidity_regime", pd.Series(np.ones(n))).to_numpy(dtype=float)
        quality = np.clip(
            1.0
            - spread_cost / (self.fee_pct + 1e-9) * 0.4
            - spread_regime * 0.3
            - liq_regime / 2.0 * 0.3,
            0.0, 1.0,
        )
        df["execution_quality_estimate"] = quality

        return df
