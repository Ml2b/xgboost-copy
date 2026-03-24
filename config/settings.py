"""Settings globales del sistema de trading con XGBoost."""

from __future__ import annotations

import os
from urllib.parse import unquote, urlparse
from typing import Final

# Evita warnings de joblib/loky en entornos donde no se detectan cores fisicos.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))


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
_redis_url = os.getenv("REDIS_URL", "").strip()
_parsed_redis_url = urlparse(_redis_url) if _redis_url else None
REDIS_URL: Final[str] = _redis_url
REDIS_HOST: Final[str] = (
    _parsed_redis_url.hostname
    if _parsed_redis_url and _parsed_redis_url.hostname
    else os.getenv("REDIS_HOST", "127.0.0.1")
)
REDIS_PORT: Final[int] = (
    _parsed_redis_url.port
    if _parsed_redis_url and _parsed_redis_url.port
    else int(os.getenv("REDIS_PORT", "6379"))
)
REDIS_USERNAME: Final[str | None] = (
    unquote(_parsed_redis_url.username)
    if _parsed_redis_url and _parsed_redis_url.username
    else (os.getenv("REDIS_USERNAME") or None)
)
REDIS_PASSWORD: Final[str | None] = (
    unquote(_parsed_redis_url.password)
    if _parsed_redis_url and _parsed_redis_url.password
    else (os.getenv("REDIS_PASSWORD") or None)
)
REDIS_DB: Final[int] = (
    int(_parsed_redis_url.path.lstrip("/"))
    if _parsed_redis_url and _parsed_redis_url.path and _parsed_redis_url.path.lstrip("/")
    else int(os.getenv("REDIS_DB", "0"))
)
STREAM_MARKET_TRADES_RAW: Final[str] = "market.trades.raw"
STREAM_MARKET_CANDLES_1M: Final[str] = "market.candles.1m"
STREAM_MARKET_FEATURES: Final[str] = "market.features"
STREAM_INFERENCE_SIGNALS: Final[str] = "inference.signals"
STREAM_EXECUTION_EVENTS: Final[str] = "execution.events"
STREAM_PAPER_EXECUTION_EVENTS: Final[str] = "paper.execution.events"
EXECUTION_STATE_KEY: Final[str] = "execution.managed_positions"
STREAM_SYSTEM_HEALTH: Final[str] = "system.health"
STREAM_SYSTEM_ERRORS: Final[str] = "system.errors"
STREAM_MARKET_ORDER_FLOW_1M: Final[str] = "market.orderflow.1m"
STREAM_MARKET_LEVEL2: Final[str] = "market.level2"
ORDER_FLOW_ENABLED: Final[bool] = (
    os.getenv("ORDER_FLOW_ENABLED", "true").lower() == "true"
)
LEVEL2_ENABLED: Final[bool] = (
    os.getenv("LEVEL2_ENABLED", "true").lower() == "true"
)
COINBASE_LEVEL2_CHANNEL: Final[str] = "level2"

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
MAX_RISK_PER_TRADE: Final[float] = float(os.getenv("MAX_RISK_PER_TRADE", "0.01"))
MAX_DAILY_DRAWDOWN: Final[float] = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.02"))
MAX_SPREAD_PCT: Final[float] = float(os.getenv("MAX_SPREAD_PCT", "0.0025"))
POSITION_STOP_LOSS_PCT: Final[float] = float(os.getenv("POSITION_STOP_LOSS_PCT", "1.0"))
POSITION_TAKE_PROFIT_PCT: Final[float] = float(os.getenv("POSITION_TAKE_PROFIT_PCT", "1.5"))
POSITION_MAX_HOLD_MINUTES: Final[int] = int(os.getenv("POSITION_MAX_HOLD_MINUTES", "60"))

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
LIGHTGBM_ENABLED: Final[bool] = os.getenv("LIGHTGBM_ENABLED", "true").lower() == "true"
LGBM_PARAMS: Final[dict[str, object]] = {
    "objective": "binary",
    "metric": "auc",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "random_state": 42,
    "verbosity": -1,
}

# Target
TARGET_HORIZON: Final[int] = 15
TARGET_THRESHOLD_PCT: Final[float] = 0.20
FEE_PCT: Final[float] = 0.05

# Reentrenamiento y senales
RETRAIN_INTERVAL: Final[int] = 2 * 3600
MIN_MODEL_AUC: Final[float] = 0.53
MIN_MODEL_SHARPE_FOR_PROMOTION: Final[float] = float(
    os.getenv("MIN_MODEL_SHARPE_FOR_PROMOTION", "0.1")
)
MAX_MODEL_DRAWDOWN_FOR_PROMOTION: Final[float] = float(
    os.getenv("MAX_MODEL_DRAWDOWN_FOR_PROMOTION", "0.20")
)
MIN_MODEL_AUC_IMPROVEMENT: Final[float] = float(
    os.getenv("MIN_MODEL_AUC_IMPROVEMENT", "0.005")
)
MIN_SIGNAL_PROB: Final[float] = 0.62
MODEL_RELOAD_INTERVAL: Final[int] = 30
SIGNAL_CONTRACT: Final[str] = os.getenv("SIGNAL_CONTRACT", "long_only_v2")
REGIME_GATE_ENABLED: Final[bool] = os.getenv("REGIME_GATE_ENABLED", "true").lower() == "true"
REGIME_VOL_EXTREME_MAX: Final[float] = float(os.getenv("REGIME_VOL_EXTREME_MAX", "0.006"))
REGIME_RANGE_COMPRESSION_MIN: Final[float] = float(
    os.getenv("REGIME_RANGE_COMPRESSION_MIN", "0.25")
)
REGIME_BB_WIDTH_MIN: Final[float] = float(os.getenv("REGIME_BB_WIDTH_MIN", "0.004"))
REGIME_HMM_ENABLED: Final[bool] = os.getenv("REGIME_HMM_ENABLED", "true").lower() == "true"
REGIME_HMM_STATES: Final[int] = int(os.getenv("REGIME_HMM_STATES", "3"))
REGIME_MODEL_MIN_ROWS: Final[int] = int(os.getenv("REGIME_MODEL_MIN_ROWS", "300"))
DRIFT_MONITOR_ENABLED: Final[bool] = os.getenv("DRIFT_MONITOR_ENABLED", "true").lower() == "true"
DRIFT_BLOCK_THRESHOLD_RATIO: Final[float] = float(
    os.getenv("DRIFT_BLOCK_THRESHOLD_RATIO", "0.30")
)
DRIFT_BOUND_MARGIN_PCT: Final[float] = float(
    os.getenv("DRIFT_BOUND_MARGIN_PCT", "0.10")
)
DRIFT_MIN_FEATURE_CHECKS: Final[int] = int(os.getenv("DRIFT_MIN_FEATURE_CHECKS", "8"))
THRESHOLD_GRID_SIZE: Final[int] = int(os.getenv("THRESHOLD_GRID_SIZE", "7"))
THRESHOLD_MIN_SUPPORT_PCT: Final[float] = float(
    os.getenv("THRESHOLD_MIN_SUPPORT_PCT", "0.03")
)
THRESHOLD_MIN_SUPPORT_ABS: Final[int] = int(os.getenv("THRESHOLD_MIN_SUPPORT_ABS", "30"))
THRESHOLD_MIN_BUY_PRECISION: Final[float] = float(
    os.getenv("THRESHOLD_MIN_BUY_PRECISION", "0.55")
)
PROMOTION_MIN_BUY_SUPPORT: Final[int] = int(os.getenv("PROMOTION_MIN_BUY_SUPPORT", "30"))
SELECTIVE_PROMOTION_MIN_AUC: Final[float] = float(os.getenv("SELECTIVE_PROMOTION_MIN_AUC", "0.55"))
SELECTIVE_PROMOTION_MIN_SHARPE: Final[float] = float(os.getenv("SELECTIVE_PROMOTION_MIN_SHARPE", "2.0"))
SELECTIVE_PROMOTION_MIN_PRECISION: Final[float] = float(
    os.getenv("SELECTIVE_PROMOTION_MIN_PRECISION", "0.65")
)
SELECTIVE_PROMOTION_MIN_SUPPORT: Final[int] = int(os.getenv("SELECTIVE_PROMOTION_MIN_SUPPORT", "20"))
SELECTIVE_PROMOTION_MIN_THRESHOLD: Final[float] = float(
    os.getenv("SELECTIVE_PROMOTION_MIN_THRESHOLD", "0.62")
)
SELECTIVE_PROMOTION_MAX_DRAWDOWN: Final[float] = float(
    os.getenv("SELECTIVE_PROMOTION_MAX_DRAWDOWN", "0.10")
)
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
RUNTIME_LOG_INTERVAL_SECONDS: Final[int] = int(os.getenv("RUNTIME_LOG_INTERVAL_SECONDS", "60"))
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
POSITION_SIZER_ENABLED: Final[bool] = os.getenv("POSITION_SIZER_ENABLED", "true").lower() == "true"
POSITION_SIZER_KELLY_FRACTION: Final[float] = float(
    os.getenv("POSITION_SIZER_KELLY_FRACTION", "0.10")
)
POSITION_SIZER_MAX_CAPITAL_FRACTION: Final[float] = float(
    os.getenv("POSITION_SIZER_MAX_CAPITAL_FRACTION", "0.005")
)
POSITION_SIZER_MIN_NOTIONAL_USD: Final[float] = float(
    os.getenv("POSITION_SIZER_MIN_NOTIONAL_USD", str(PILOT_ORDER_NOTIONAL_USD))
)
POSITION_SIZER_MAX_NOTIONAL_USD: Final[float] = float(
    os.getenv("POSITION_SIZER_MAX_NOTIONAL_USD", str(PILOT_ORDER_NOTIONAL_USD * 2))
)
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
