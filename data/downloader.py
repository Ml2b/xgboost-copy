"""Descargador historico desde data.binance.vision.

Descarga klines (OHLCV + taker) y aggTrades 1m para todos los pares y
construye los 45 ORDER_FLOW_RAW_COLUMNS completos:

  - 13 trade columns  : desde aggTrades reales (buy/sell vol, delta, vwap...)
  - 32 L2 columns     : proxies derivados de klines con taker_buy/sell

Los 32 proxies de L2 aproximan spread, book depth y dynamics sin snapshots
de libro (no disponibles gratis). Son mejores que 0.0 para el primer
entrenamiento; el collector live los sobreescribe con datos reales.

Uso:
    python scripts/bootstrap_order_flow.py --days 30
"""

from __future__ import annotations

import io
import time
import zipfile
from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
from loguru import logger

from data.history_store import CandleHistoryStore
from features.order_flow import (
    ORDER_FLOW_BOOK_DEPTH_COLUMNS,
    ORDER_FLOW_BOOK_DYNAMICS_COLUMNS,
    ORDER_FLOW_RAW_COLUMNS,
    ORDER_FLOW_SPREAD_COLUMNS,
    ORDER_FLOW_TRADE_COLUMNS,
)

_VISION_BASE = "https://data.binance.vision/data/spot/daily"

_KLINES_COLS: list[str] = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trade_count",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]

_AGG_TRADE_COLS: list[str] = [
    "agg_trade_id", "price", "qty", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker", "best_match",
]

_L2_PROXY_COLS: list[str] = (
    ORDER_FLOW_SPREAD_COLUMNS
    + ORDER_FLOW_BOOK_DEPTH_COLUMNS
    + ORDER_FLOW_BOOK_DYNAMICS_COLUMNS
)

# Pares monitoreados por el bot (Binance USDT → product_id del sistema)
DEFAULT_SYMBOLS: list[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT",
    "BCHUSDT", "UNIUSDT", "PEPEUSDT", "BONKUSDT", "TAOUSDT",
    "SUIUSDT", "HBARUSDT", "FETUSDT", "AAVEUSDT", "XLMUSDT",
    "ONDOUSDT", "AKTUSDT",
]


class BinanceVisionDownloader:
    """Descarga OHLCV + order flow historico y los persiste en SQLite."""

    def __init__(
        self,
        store: CandleHistoryStore | None = None,
        request_sleep: float = 0.4,
        session: requests.Session | None = None,
    ) -> None:
        self.store = store or CandleHistoryStore()
        self.request_sleep = request_sleep
        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": "xgboost-trader-bootstrap/1.0"}
        )

    def download_range(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, int]:
        """Descarga y persiste datos para varios simbolos en un rango."""
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        summary: dict[str, int] = {}
        for symbol in symbols:
            total = self._download_symbol(symbol, start, end)
            summary[symbol] = total
        return summary

    def _download_symbol(
        self, symbol: str, start: date, end: date
    ) -> int:
        product_id = symbol_to_product_id(symbol)
        total = 0
        for day in _iter_dates(start, end):
            total += self._download_day(symbol, product_id, day)
        logger.info(
            "Downloader completado. symbol={} product_id={} candles={}",
            symbol, product_id, total,
        )
        return total

    def _download_day(
        self, symbol: str, product_id: str, day: date
    ) -> int:
        date_str = day.isoformat()

        klines = self._fetch_klines(symbol, date_str)
        if klines is None or klines.empty:
            return 0

        # 1. Persistir OHLCV
        candle_records = _klines_to_candle_records(klines, product_id)
        if candle_records:
            self.store.upsert_candles(candle_records)

        # 2. Iniciar records de order flow con proxies L2 desde klines
        of_by_time: dict[int, dict[str, Any]] = {
            r["open_time"]: r
            for r in _klines_to_l2_proxy(klines, product_id)
        }

        # 3. Enriquecer con datos reales de trade desde aggTrades
        trades = self._fetch_agg_trades(symbol, date_str)
        if trades is not None and not trades.empty:
            trade_records = _trades_to_order_flow(trades, klines, product_id)
            for tr in trade_records:
                ot = tr["open_time"]
                if ot in of_by_time:
                    for col in ORDER_FLOW_TRADE_COLUMNS:
                        of_by_time[ot][col] = tr.get(col, 0.0)
                else:
                    of_by_time[ot] = tr

        # 4. Persistir los 45 RAW columns combinados
        if of_by_time:
            self.store.upsert_order_flow(list(of_by_time.values()))

        return len(candle_records)

    def _fetch_klines(
        self, symbol: str, date_str: str
    ) -> pd.DataFrame | None:
        url = (
            f"{_VISION_BASE}/klines/{symbol}/1m"
            f"/{symbol}-1m-{date_str}.zip"
        )
        return self._fetch_csv(url, _KLINES_COLS)

    def _fetch_agg_trades(
        self, symbol: str, date_str: str
    ) -> pd.DataFrame | None:
        url = (
            f"{_VISION_BASE}/aggTrades/{symbol}"
            f"/{symbol}-aggTrades-{date_str}.zip"
        )
        return self._fetch_csv(url, _AGG_TRADE_COLS)

    def _fetch_csv(
        self, url: str, columns: list[str]
    ) -> pd.DataFrame | None:
        time.sleep(self.request_sleep)
        try:
            resp = self._session.get(url, timeout=60)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    return pd.read_csv(
                        f, header=None, names=columns, low_memory=False
                    )
        except Exception as exc:
            logger.warning(
                "Downloader fetch error. url={} exc={}", url, exc
            )
            return None


# ---------------------------------------------------------------------------
# Conversor OHLCV
# ---------------------------------------------------------------------------

def _klines_to_candle_records(
    klines: pd.DataFrame, product_id: str
) -> list[dict[str, Any]]:
    """Klines (microsegundos) → candle records (milisegundos)."""
    records: list[dict[str, Any]] = []
    for row in klines.itertuples(index=False):
        try:
            records.append({
                "product_id": product_id,
                "open_time": int(row.open_time) // 1000,
                "close_time": int(row.close_time) // 1000,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
                "trade_count": int(row.trade_count),
            })
        except (ValueError, TypeError):
            continue
    return records


# ---------------------------------------------------------------------------
# Proxies L2 desde klines taker data
# ---------------------------------------------------------------------------

def _klines_to_l2_proxy(
    klines: pd.DataFrame, product_id: str
) -> list[dict[str, Any]]:
    """Deriva los 32 L2 columns como proxies desde klines con taker data.

    Formulas:
      spread         = buy_vwap - sell_vwap  (effective spread proxy)
      mid_price      = (high + low) / 2
      relative_spread= spread / mid_price
      best_bid_size  = sell_vol / trade_count  (avg sell hitting bid)
      best_ask_size  = buy_vol / trade_count   (avg buy hitting ask)
      top_of_book_imbalance = (sell_vol - buy_vol) / volume
      microprice     = (sell_vol*buy_vwap + buy_vol*sell_vwap) / volume
      microprice_shift = microprice - mid_price
      bid/ask_depth_topN = rolling N-period sum of sell/buy taker vol
      depth_imbalance_top10 = (bid10 - ask10) / (bid10 + ask10)
      depth_slope_bid/ask   = top5 / top20 ratio (tapering proxy)
      cumul_depth_Xbp       = rolling vol * bp_fraction
      consumption_rate      = taker_vol / depth_top10
      net_consumption       = ask_rate - bid_rate
      price_move_per_unit_flow = price_change / volume
      absorption_estimate   = 1 - |price_change| / expected_impact
      absorption_score      = mean(buyer + seller absorption)
      flow_to_price_divergence = flow_direction - price_direction
      refill_rate           = taker_vol / prev_candle_vol
      cancel_rate           = max(0, prev_vol - curr_vol) / prev_vol
    """
    df = klines.copy()

    # Conversion numerica
    num_cols = [
        "open", "high", "low", "close", "volume", "quote_volume",
        "trade_count", "taker_buy_base_vol", "taker_buy_quote_vol",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Timestamps: microsegundos → milisegundos
    open_time_ms = df["open_time"].astype(int) // 1000

    # Intermedios
    sell_base = (df["volume"] - df["taker_buy_base_vol"]).clip(lower=0.0)
    sell_quote = (df["quote_volume"] - df["taker_buy_quote_vol"]).clip(lower=0.0)
    buy_base = df["taker_buy_base_vol"].clip(lower=0.0)
    buy_quote = df["taker_buy_quote_vol"].clip(lower=0.0)

    safe = lambda s: s.replace(0.0, np.nan)  # noqa: E731

    buy_vwap = (buy_quote / safe(buy_base)).fillna(df["close"])
    sell_vwap = (sell_quote / safe(sell_base)).fillna(df["close"])
    mid = (df["high"] + df["low"]) / 2.0

    # ── GROUP 4: Spread / top of book ──────────────────────────────────────
    spread = (buy_vwap - sell_vwap).clip(lower=0.0)
    rel_spread = (spread / safe(mid)).fillna(0.0)
    tc = df["trade_count"].clip(lower=1.0)
    best_bid = sell_base / tc
    best_ask = buy_base / tc
    tob_imbalance = ((sell_base - buy_base) / safe(df["volume"])).fillna(0.0)
    microprice = (
        (sell_base * buy_vwap + buy_base * sell_vwap) / safe(df["volume"])
    ).fillna(mid)
    micro_shift = microprice - mid

    # ── GROUP 3: Book depth (rolling taker volume) ──────────────────────────
    bid5 = sell_base.rolling(5, min_periods=1).sum()
    bid10 = sell_base.rolling(10, min_periods=1).sum()
    bid20 = sell_base.rolling(20, min_periods=1).sum()
    ask5 = buy_base.rolling(5, min_periods=1).sum()
    ask10 = buy_base.rolling(10, min_periods=1).sum()
    ask20 = buy_base.rolling(20, min_periods=1).sum()

    depth_imb = ((bid10 - ask10) / safe(bid10 + ask10)).fillna(0.0)
    slope_bid = (bid5 / safe(bid20)).fillna(0.0).clip(0.0, 2.0)
    slope_ask = (ask5 / safe(ask20)).fillna(0.0).clip(0.0, 2.0)

    vol_roll3 = df["volume"].rolling(3, min_periods=1).sum()
    vol_roll5 = df["volume"].rolling(5, min_periods=1).sum()
    vol_roll10 = df["volume"].rolling(10, min_periods=1).sum()
    cumul_1bp = vol_roll3 * 0.05
    cumul_5bp = vol_roll5 * 0.20
    cumul_10bp = vol_roll10 * 0.40

    # ── GROUPS 5-7: Dynamics ────────────────────────────────────────────────
    ask_cons = (buy_base / safe(ask10)).fillna(0.0).clip(0.0, 1.0)
    bid_cons = (sell_base / safe(bid10)).fillna(0.0).clip(0.0, 1.0)
    net_cons = ask_cons - bid_cons

    price_chg = df["close"] - df["open"]
    pm_per_flow = (price_chg / safe(df["volume"])).fillna(0.0)

    # Absorcion: flujo agresivo que NO resulto en movimiento de precio
    # absorption ≈ 1 - |price_change| / (spread * taker_vol)
    expected_impact = (spread * df["volume"]).clip(lower=1e-12)
    abs_chg = price_chg.abs()
    seller_abs = (1.0 - abs_chg / safe(expected_impact)).clip(0.0, 1.0).fillna(0.0)
    buyer_abs = seller_abs.copy()  # simetrico en proxy sin datos de un lado
    abs_score = (seller_abs + buyer_abs) / 2.0

    # Divergencia flujo vs precio
    flow_dir = ((buy_base - sell_base) / safe(df["volume"])).fillna(0.0)
    price_dir = (price_chg / safe(spread.replace(0.0, np.nan))).clip(-1.0, 1.0).fillna(0.0)
    flow_div = (flow_dir - price_dir).clip(-2.0, 2.0)

    # Refill y cancel (cambio de volumen candle a candle)
    prev_vol = df["volume"].shift(1).fillna(df["volume"])
    prev_buy = buy_base.shift(1).fillna(buy_base)
    prev_sell = sell_base.shift(1).fillna(sell_base)
    refill_ask = (buy_base / safe(prev_vol)).fillna(0.0).clip(0.0, 2.0)
    refill_bid = (sell_base / safe(prev_vol)).fillna(0.0).clip(0.0, 2.0)
    cancel_ask = ((prev_buy - buy_base).clip(lower=0.0) / safe(prev_buy)).fillna(0.0)
    cancel_bid = ((prev_sell - sell_base).clip(lower=0.0) / safe(prev_sell)).fillna(0.0)

    # ── Ensamblar columnas en orden de _L2_PROXY_COLS ──────────────────────
    computed: dict[str, pd.Series] = {
        # ORDER_FLOW_SPREAD_COLUMNS
        "spread": spread,
        "relative_spread": rel_spread,
        "mid_price": mid,
        "best_bid_size": best_bid,
        "best_ask_size": best_ask,
        "top_of_book_imbalance": tob_imbalance,
        "microprice": microprice,
        "microprice_shift": micro_shift,
        # ORDER_FLOW_BOOK_DEPTH_COLUMNS
        "bid_depth_top5": bid5,
        "bid_depth_top10": bid10,
        "bid_depth_top20": bid20,
        "ask_depth_top5": ask5,
        "ask_depth_top10": ask10,
        "ask_depth_top20": ask20,
        "depth_imbalance_top10": depth_imb,
        "depth_slope_bid": slope_bid,
        "depth_slope_ask": slope_ask,
        "cumul_depth_1bp": cumul_1bp,
        "cumul_depth_5bp": cumul_5bp,
        "cumul_depth_10bp": cumul_10bp,
        # ORDER_FLOW_BOOK_DYNAMICS_COLUMNS
        "ask_consumption_rate": ask_cons,
        "bid_consumption_rate": bid_cons,
        "net_consumption": net_cons,
        "price_move_per_unit_flow": pm_per_flow,
        "seller_absorption_estimate": seller_abs,
        "buyer_absorption_estimate": buyer_abs,
        "absorption_score": abs_score,
        "flow_to_price_divergence": flow_div,
        "refill_rate_ask": refill_ask,
        "refill_rate_bid": refill_bid,
        "cancel_rate_ask": cancel_ask,
        "cancel_rate_bid": cancel_bid,
    }

    records: list[dict[str, Any]] = []
    for i in range(len(df)):
        record: dict[str, Any] = {
            "product_id": product_id,
            "open_time": int(open_time_ms.iloc[i]),
        }
        # Todos los RAW columns a 0 por defecto (trade cols se llenan despues)
        for col in ORDER_FLOW_RAW_COLUMNS:
            record[col] = 0.0
        # Sobreescribir con proxies L2 calculados
        for col, series in computed.items():
            val = series.iloc[i]
            record[col] = 0.0 if (not np.isfinite(val)) else float(val)
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Agregador de aggTrades → 13 trade columns
# ---------------------------------------------------------------------------

def _trades_to_order_flow(
    trades: pd.DataFrame,
    klines: pd.DataFrame,
    product_id: str,
) -> list[dict[str, Any]]:
    """Agrega aggTrades por minuto → 13 ORDER_FLOW_TRADE_COLUMNS reales."""
    t = trades.copy()
    t["price"] = pd.to_numeric(t["price"], errors="coerce")
    t["qty"] = pd.to_numeric(t["qty"], errors="coerce")
    t["transact_time"] = pd.to_numeric(t["transact_time"], errors="coerce")
    t["is_buyer_maker"] = (
        t["is_buyer_maker"].astype(str).str.lower() == "true"
    )
    t = t.dropna(subset=["price", "qty", "transact_time"])

    # transact_time en microsegundos → open_time en milisegundos
    t["open_time"] = (
        t["transact_time"].astype(int) // 60_000_000
    ) * 60_000

    # is_buyer_maker=True → vendedor es taker (sell agresivo)
    t["is_buy"] = ~t["is_buyer_maker"]
    t["buy_qty"] = t["qty"].where(t["is_buy"], 0.0)
    t["sell_qty"] = t["qty"].where(~t["is_buy"], 0.0)
    t["notional"] = t["price"] * t["qty"]

    g = t.groupby("open_time", sort=True)
    agg = g.agg(
        buy_aggressive_volume=("buy_qty", "sum"),
        sell_aggressive_volume=("sell_qty", "sum"),
        total_traded_volume=("qty", "sum"),
        trade_count=("qty", "count"),
        buy_count=("is_buy", "sum"),
        avg_trade_size=("qty", "mean"),
        median_trade_size=("qty", "median"),
        max_trade_size=("qty", "max"),
        notional_sum=("notional", "sum"),
    ).reset_index()

    agg["sell_count"] = agg["trade_count"] - agg["buy_count"].astype(int)

    safe_vol = agg["total_traded_volume"].replace(0.0, np.nan)
    safe_sell = agg["sell_aggressive_volume"].replace(0.0, np.nan)
    agg["volume_delta"] = (
        agg["buy_aggressive_volume"] - agg["sell_aggressive_volume"]
    )
    agg["delta_ratio"] = (agg["volume_delta"] / safe_vol).fillna(0.0)
    agg["buy_sell_ratio"] = (
        agg["buy_aggressive_volume"] / safe_sell
    ).fillna(0.0).clip(0.0, 10.0)
    agg["trade_vwap"] = (agg["notional_sum"] / safe_vol).fillna(0.0)

    # Solo minutos presentes en klines (klines open_time en microsegundos → ms)
    valid: set[int] = set(
        (klines["open_time"].astype(int) // 1000).tolist()
    )
    agg = agg[agg["open_time"].isin(valid)]

    records: list[dict[str, Any]] = []
    for row in agg.itertuples(index=False):
        record: dict[str, Any] = {
            "product_id": product_id,
            "open_time": int(row.open_time),
        }
        for col in ORDER_FLOW_TRADE_COLUMNS:
            record[col] = float(getattr(row, col, 0.0))
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def symbol_to_product_id(symbol: str) -> str:
    """BTCUSDT → BTC-USD (formato product_id del sistema)."""
    s = symbol.upper().strip()
    if s.endswith("USDT"):
        return f"{s[:-4]}-USD"
    if s.endswith("USD"):
        return f"{s[:-3]}-USD"
    return s


def _iter_dates(start: date, end: date) -> Iterator[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)
