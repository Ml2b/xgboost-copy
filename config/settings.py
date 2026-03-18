"""Settings globales del sistema de trading con XGBoost."""

from __future__ import annotations

import os
from typing import Final


def _merge_unique_bases(*groups: list[str]) -> list[str]:
    """Une listas de activos preservando orden y evitando duplicados."""
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for base in group:
            normalized = base.strip().upper()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return merged


# Redis
REDIS_HOST: Final[str] = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT: Final[int] = int(os.getenv("REDIS_PORT", "6379"))
STREAM_MARKET_TRADES_RAW: Final[str] = "market.trades.raw"
STREAM_MARKET_CANDLES_1M: Final[str] = "market.candles.1m"
STREAM_MARKET_FEATURES: Final[str] = "market.features"
STREAM_INFERENCE_SIGNALS: Final[str] = "inference.signals"
STREAM_EXECUTION_EVENTS: Final[str] = "execution.events"
STREAM_PAPER_EXECUTION_EVENTS: Final[str] = "paper.execution.events"
STREAM_SYSTEM_HEALTH: Final[str] = "system.health"
STREAM_SYSTEM_ERRORS: Final[str] = "system.errors"

# Mercado y velas
CORE_BASES: Final[list[str]] = [
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "ADA",
    "DOGE",
    "AVAX",
    "DOT",
    "LINK",
    "LTC",
    "BCH",
    "UNI",
]
PHASE_ONE_BASES: Final[list[str]] = [
    "PEPE",
    "BONK",
    "FARTCOIN",
    "TAO",
    "SUI",
    "HBAR",
    "FET",
    "AAVE",
]
PHASE_TWO_EXPERIMENTAL_BASES: Final[list[str]] = [
    "ZEC",
    "XLM",
    "ONDO",
    "PENGU",
    "AERO",
    "WLFI",
    "ZRO",
    "MON",
    "AKT",
]
PHASE_TWO_FOCUS_BASES: Final[list[str]] = [
    "XLM",
    "ONDO",
    "AKT",
    "PENGU",
]
PHASE_TWO_EXPERIMENT_ONLY_BASES: Final[list[str]] = [
    "ZEC",
]
STUDIED_UNIVERSE_BASES: Final[list[str]] = _merge_unique_bases(
    CORE_BASES,
    PHASE_ONE_BASES,
    PHASE_TWO_FOCUS_BASES,
)
STUDIED_UNIVERSE_EXPERIMENT_BASES: Final[list[str]] = _merge_unique_bases(
    PHASE_TWO_EXPERIMENT_ONLY_BASES,
)
PHASE_ONE_TRAINING_SYMBOLS: Final[list[str]] = [
    "PEPE/USDT",
    "BONK/USDT",
    "FARTCOIN/USDT",
    "TAO/USDT",
    "SUI/USDT",
    "HBAR/USDT",
    "FET/USDT",
    "AAVE/USDT",
]
PHASE_TWO_EXPERIMENTAL_TRAINING_SYMBOLS: Final[list[str]] = [
    "ZEC/USD",
    "XLM/USD",
    "ONDO/USD",
    "PENGU/USD",
    "AERO/USD",
    "WLFI/USD",
    "ZRO/USD",
    "MON/USD",
    "AKT/USD",
]
PHASE_TWO_FOCUS_TRAINING_SYMBOLS: Final[list[str]] = [
    "XLM/USD",
    "ONDO/USD",
    "AKT/USD",
    "PENGU/USD",
]
PRODUCTS: Final[list[str]] = [
    product.strip()
    for product in os.getenv("PRODUCTS_CSV", "BTC-USDT,ETH-USDT").split(",")
    if product.strip()
]
CANDLE_SECONDS: Final[int] = 60
OBSERVED_BASES: Final[list[str]] = [
    base.strip()
    for base in os.getenv(
        "OBSERVED_BASES_CSV",
        ",".join(CORE_BASES + PHASE_ONE_BASES),
    ).split(",")
    if base.strip()
]
COINBASE_QUOTE_PRIORITY: Final[list[str]] = [
    quote.strip()
    for quote in os.getenv("COINBASE_QUOTE_PRIORITY_CSV", "USD,USDC,USDT").split(",")
    if quote.strip()
]

# Riesgo
MAX_RISK_PER_TRADE: Final[float] = 0.01
MAX_DAILY_DRAWDOWN: Final[float] = 0.02
MAX_SPREAD_PCT: Final[float] = 0.0025

# XGBoost
XGB_PARAMS: Final[dict[str, object]] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}

# Target
TARGET_HORIZON: Final[int] = 15
TARGET_THRESHOLD_PCT: Final[float] = 0.20
FEE_PCT: Final[float] = 0.05

# Reentrenamiento y senales
RETRAIN_INTERVAL: Final[int] = 2 * 3600
MIN_MODEL_AUC: Final[float] = 0.53
MIN_SIGNAL_PROB: Final[float] = 0.62
MODEL_RELOAD_INTERVAL: Final[int] = 30
MODEL_REGISTRY_ROOT: Final[str] = os.getenv("MODEL_REGISTRY_ROOT", "models/multi_asset_live_v2")
STUDIED_UNIVERSE_PAPER_ROOT: Final[str] = os.getenv(
    "STUDIED_UNIVERSE_PAPER_ROOT",
    "models/multi_asset_paper_24",
)
TRAINER_HISTORY_DB_PATH: Final[str] = os.getenv(
    "TRAINER_HISTORY_DB_PATH",
    "data/trainer_candle_history.sqlite3",
)
TRAINER_HISTORY_SYNC_BATCH: Final[int] = int(os.getenv("TRAINER_HISTORY_SYNC_BATCH", "1000"))
TRAINER_RECENCY_WEIGHT_HALF_LIFE_CANDLES: Final[int] = int(
    os.getenv("TRAINER_RECENCY_WEIGHT_HALF_LIFE_CANDLES", "43200")
)

# Operacion
FEATURE_BUFFER_SIZE: Final[int] = 100
WS_TIMEOUT_SECONDS: Final[int] = 30
HEALTH_PUBLISH_INTERVAL: Final[int] = 10
COINBASE_HTTP_TIMEOUT_SECONDS: Final[int] = int(os.getenv("COINBASE_HTTP_TIMEOUT_SECONDS", "10"))
COINBASE_WS_PUBLIC_URL: Final[str] = "wss://advanced-trade-ws.coinbase.com"
COINBASE_CHANNEL: Final[str] = "market_trades"
COINBASE_HEARTBEATS_CHANNEL: Final[str] = "heartbeats"
COINBASE_WS_PRODUCTS_PER_CONNECTION: Final[int] = int(
    os.getenv("COINBASE_WS_PRODUCTS_PER_CONNECTION", "1")
)
COINBASE_WS_AUTH_ENABLED: Final[bool] = os.getenv("COINBASE_WS_AUTH_ENABLED", "true").lower() == "true"
COINBASE_CANDLE_RECONCILE_ENABLED: Final[bool] = (
    os.getenv("COINBASE_CANDLE_RECONCILE_ENABLED", "true").lower() == "true"
)
COINBASE_CANDLE_RECONCILE_LOOKBACK_MINUTES: Final[int] = int(
    os.getenv("COINBASE_CANDLE_RECONCILE_LOOKBACK_MINUTES", "3")
)
COINBASE_CANDLE_RECONCILE_COOLDOWN_SECONDS: Final[int] = int(
    os.getenv("COINBASE_CANDLE_RECONCILE_COOLDOWN_SECONDS", "20")
)
COINBASE_CANDLE_RECONCILE_GRANULARITY: Final[str] = os.getenv(
    "COINBASE_CANDLE_RECONCILE_GRANULARITY",
    "ONE_MINUTE",
)
COINBASE_CREDENTIALS_PATH: Final[str] = os.getenv(
    "COINBASE_CREDENTIALS_PATH",
    "/etc/secrets/coinbase_api_key.json",
)
EXECUTION_ENABLED: Final[bool] = os.getenv("EXECUTION_ENABLED", "true").lower() == "true"
EXECUTION_DRY_RUN: Final[bool] = os.getenv("EXECUTION_DRY_RUN", "true").lower() == "true"
PILOT_ORDER_NOTIONAL_USD: Final[float] = float(os.getenv("PILOT_ORDER_NOTIONAL_USD", "25"))
EXECUTION_COOLDOWN_SECONDS: Final[int] = int(os.getenv("EXECUTION_COOLDOWN_SECONDS", "60"))
EXECUTION_ALLOWED_BASES: Final[list[str]] = [
    base.strip()
    for base in os.getenv("EXECUTION_ALLOWED_BASES_CSV", "BTC,ETH,ADA,BCH,UNI").split(",")
    if base.strip()
]
PAPER_TRADING_ENABLED: Final[bool] = os.getenv("PAPER_TRADING_ENABLED", "false").lower() == "true"
PAPER_INITIAL_CASH: Final[float] = float(os.getenv("PAPER_INITIAL_CASH", "10000"))
PAPER_FEE_PCT: Final[float] = float(os.getenv("PAPER_FEE_PCT", str(FEE_PCT)))
PAPER_SLIPPAGE_PCT: Final[float] = float(os.getenv("PAPER_SLIPPAGE_PCT", "0.0"))

# Eventos macro. Vacia por defecto; si queda vacia, el guardian deja pasar.
MACRO_EVENT_TIMES_UTC: Final[list[str]] = []
