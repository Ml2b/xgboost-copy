"""Compara 6m vs 12m flat vs 12m weighted para multiples criptos."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from features.calculator import FeatureCalculator
from features.selector import SelectorConfig
from scripts.fetch_coinbase_history import fetch_history, rows_to_frame
from target.builder import TargetBuilder, TargetConfig, TargetType
from validation.walk_forward import WalkForwardConfig, WalkForwardValidator


DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "UNI/USDT",
]


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI del experimento."""
    parser = argparse.ArgumentParser(description="Compara ventanas historicas para modelos por cripto.")
    parser.add_argument("--exchange", default="binanceus", help="Exchange CCXT para descarga historica.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Simbolos a procesar.")
    parser.add_argument("--timeframe", default="1m", help="Timeframe OHLCV.")
    parser.add_argument("--candles-6m", type=int, default=259200, help="Cantidad de velas del baseline reciente.")
    parser.add_argument("--holdout-candles", type=int, default=43200, help="Tamano del holdout fijo final.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Tamano de lote fetch_ohlcv.")
    parser.add_argument("--output-dir", default="data/historical_12m_binanceus", help="Directorio de CSVs anuales.")
    parser.add_argument("--report-path", default="reports/history_window_comparison_12m.json", help="JSON final del experimento.")
    parser.add_argument("--n-splits", type=int, default=5, help="Cantidad de folds walk-forward.")
    parser.add_argument("--min-train-size", type=int, default=1000, help="Minimo de train por fold.")
    parser.add_argument("--max-features", type=int, default=25, help="Maximo de features finales.")
    parser.add_argument("--recent-weight", type=float, default=2.0, help="Peso relativo del semestre reciente.")
    parser.add_argument("--skip-download", action="store_true", help="Usa CSVs existentes si ya estan descargados.")
    return parser


def main() -> None:
    """Ejecuta la comparacion completa y guarda un reporte JSON."""
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    candles_12m = args.candles_6m * 2
    exchange = getattr(ccxt, args.exchange)({"enableRateLimit": True})
    started = time.time()
    results: list[dict[str, object]] = []

    for symbol in args.symbols:
        csv_path = output_dir / f"{symbol.replace('/', '_').lower()}_{args.timeframe}_{args.exchange}.csv"
        if not args.skip_download or not csv_path.exists():
            rows = fetch_history(
                exchange=exchange,
                symbol=symbol,
                timeframe=args.timeframe,
                candles=candles_12m,
                batch_size=args.batch_size,
            )
            frame = rows_to_frame(rows, symbol, timeframe=args.timeframe)
            frame.to_csv(csv_path, index=False)
            print(f"{symbol}: {len(frame)} velas anuales -> {csv_path}")

        annual_candles = pd.read_csv(csv_path)
        result = run_symbol_experiments(
            frame=annual_candles,
            symbol=symbol,
            candles_6m=args.candles_6m,
            holdout_candles=args.holdout_candles,
            n_splits=args.n_splits,
            min_train_size=args.min_train_size,
            max_features=args.max_features,
            recent_weight=args.recent_weight,
        )
        results.append(result)
        partial_report = {
            "exchange": args.exchange,
            "timeframe": args.timeframe,
            "candles_6m": args.candles_6m,
            "candles_12m": candles_12m,
            "holdout_candles": args.holdout_candles,
            "recent_weight": args.recent_weight,
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "duration_seconds": round(time.time() - started, 2),
            "symbols": results,
        }
        report_path.write_text(json.dumps(partial_report, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))

    report = {
        "exchange": args.exchange,
        "timeframe": args.timeframe,
        "candles_6m": args.candles_6m,
        "candles_12m": candles_12m,
        "holdout_candles": args.holdout_candles,
        "recent_weight": args.recent_weight,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "duration_seconds": round(time.time() - started, 2),
        "symbols": results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "symbols_completed": len(results)}, indent=2))


def run_symbol_experiments(
    frame: pd.DataFrame,
    symbol: str,
    candles_6m: int,
    holdout_candles: int,
    n_splits: int,
    min_train_size: int,
    max_features: int,
    recent_weight: float,
) -> dict[str, object]:
    """Corre las tres variantes sobre un mismo simbolo."""
    calculator = FeatureCalculator()
    target_builder = TargetBuilder(
        TargetConfig(
            target_type=TargetType.NET_RETURN_THRESHOLD,
            horizon=settings.TARGET_HORIZON,
            threshold_pct=settings.TARGET_THRESHOLD_PCT,
            fee_pct=settings.FEE_PCT,
        )
    )
    ordered = frame.sort_values("open_time").reset_index(drop=True)
    recent_frame = ordered.tail(candles_6m).reset_index(drop=True)

    annual_result = evaluate_window(
        strategy_name="12m_flat",
        candles=ordered,
        calculator=calculator,
        target_builder=target_builder,
        holdout_candles=holdout_candles,
        n_splits=n_splits,
        min_train_size=min_train_size,
        max_features=max_features,
        sample_weight=None,
    )
    annual_weighted = evaluate_window(
        strategy_name="12m_weighted",
        candles=ordered,
        calculator=calculator,
        target_builder=target_builder,
        holdout_candles=holdout_candles,
        n_splits=n_splits,
        min_train_size=min_train_size,
        max_features=max_features,
        sample_weight=build_recent_weight_vector(ordered, candles_6m, recent_weight),
    )
    recent_result = evaluate_window(
        strategy_name="6m_recent",
        candles=recent_frame,
        calculator=calculator,
        target_builder=target_builder,
        holdout_candles=holdout_candles,
        n_splits=n_splits,
        min_train_size=min_train_size,
        max_features=max_features,
        sample_weight=None,
    )

    strategies = [recent_result, annual_result, annual_weighted]
    valid_strategies = [item for item in strategies if item.get("status") == "ok"]
    best = max(valid_strategies, key=lambda item: item["holdout_auc"]) if valid_strategies else None
    return {
        "symbol": symbol,
        "product_id": symbol.replace("/", "-"),
        "source_rows": int(len(ordered)),
        "source_start": to_iso(ordered["open_time"].iloc[0]),
        "source_end": to_iso(ordered["open_time"].iloc[-1]),
        "zero_volume_share": float((ordered["volume"].astype(float) <= 0).mean()),
        "best_strategy": best["strategy"] if best else None,
        "best_holdout_auc": best["holdout_auc"] if best else None,
        "strategies": strategies,
    }


def evaluate_window(
    strategy_name: str,
    candles: pd.DataFrame,
    calculator: FeatureCalculator,
    target_builder: TargetBuilder,
    holdout_candles: int,
    n_splits: int,
    min_train_size: int,
    max_features: int,
    sample_weight: np.ndarray | None,
) -> dict[str, object]:
    """Entrena una variante y la evalua sobre holdout fijo."""
    features_df = calculator.compute(candles.copy())
    labeled = target_builder.build(features_df)
    if len(labeled) <= holdout_candles + settings.TARGET_HORIZON + min_train_size:
        return {
            "strategy": strategy_name,
            "status": "insufficient_data",
            "rows_total": int(len(labeled)),
            "required_min_rows": int(holdout_candles + settings.TARGET_HORIZON + min_train_size + 1),
            "weighting": summarize_weights(sample_weight),
        }

    weights_labeled = align_weights_to_labeled(candles, labeled, sample_weight)
    holdout_start_idx = len(labeled) - holdout_candles
    train_end_idx = holdout_start_idx - settings.TARGET_HORIZON
    train_df = labeled.iloc[:train_end_idx].reset_index(drop=True)
    holdout_df = labeled.iloc[holdout_start_idx:].reset_index(drop=True)
    train_weights = weights_labeled[:train_end_idx] if weights_labeled is not None else None

    feature_cols = [column for column in calculator.feature_columns if column in train_df.columns]
    validator = WalkForwardValidator(
        config=WalkForwardConfig(
            n_splits=n_splits,
            gap_periods=settings.TARGET_HORIZON,
            min_train_size=min_train_size,
            verbose=False,
        ),
        model_factory=model_factory,
        selector_config=SelectorConfig(max_features=max_features, verbose=False),
    )
    try:
        wf_result = validator.validate(
            train_df,
            target_col="target",
            feature_cols=feature_cols,
            sample_weight=train_weights,
        )
    except (ValueError, RuntimeError) as exc:
        return {
            "strategy": strategy_name,
            "status": "insufficient_data",
            "rows_total": int(len(labeled)),
            "rows_train": int(len(train_df)),
            "rows_holdout": int(len(holdout_df)),
            "reason": str(exc),
            "weighting": summarize_weights(train_weights),
        }
    final_model = fit_final_model(
        train_df=train_df,
        stable_features=wf_result.stable_features,
        sample_weight=train_weights,
    )
    probs = predict_proba(final_model, holdout_df[wf_result.stable_features])
    holdout_target = holdout_df["target"].to_numpy(dtype=int)
    holdout_auc = float(roc_auc_score(holdout_target, probs))
    holdout_logloss = float(log_loss(holdout_target, probs, labels=[0, 1]))
    holdout_returns = holdout_df["target_return"].to_numpy(dtype=float)
    holdout_volatility = float(np.std(np.diff(np.log(np.clip(holdout_df["close"].to_numpy(dtype=float), 1e-9, None)))))

    return {
        "strategy": strategy_name,
        "status": "ok",
        "rows_total": int(len(labeled)),
        "rows_train": int(len(train_df)),
        "rows_holdout": int(len(holdout_df)),
        "train_start": to_iso(train_df["open_time"].iloc[0]),
        "train_end": to_iso(train_df["open_time"].iloc[-1]),
        "holdout_start": to_iso(holdout_df["open_time"].iloc[0]),
        "holdout_end": to_iso(holdout_df["open_time"].iloc[-1]),
        "holdout_auc": holdout_auc,
        "holdout_logloss": holdout_logloss,
        "holdout_target_mean": float(np.mean(holdout_target)),
        "holdout_return_mean": float(np.mean(holdout_returns)),
        "holdout_return_std": float(np.std(holdout_returns)),
        "holdout_volatility": holdout_volatility,
        "wf_auc_mean": wf_result.auc_mean,
        "wf_auc_std": wf_result.auc_std,
        "wf_internal_test_auc": wf_result.auc_test_final,
        "stable_feature_count": len(wf_result.stable_features),
        "stable_features": wf_result.stable_features,
        "weighting": summarize_weights(train_weights),
    }


def build_recent_weight_vector(frame: pd.DataFrame, candles_6m: int, recent_weight: float) -> np.ndarray:
    """Crea pesos 1x/2x sobre el ano completo usando el semestre reciente."""
    weights = np.ones(len(frame), dtype=float)
    recent_start_idx = max(0, len(frame) - candles_6m)
    weights[recent_start_idx:] = float(recent_weight)
    return weights


def align_weights_to_labeled(
    raw_candles: pd.DataFrame,
    labeled: pd.DataFrame,
    sample_weight: np.ndarray | None,
) -> np.ndarray | None:
    """Alinea los pesos originales con el DataFrame ya podado por features/target."""
    if sample_weight is None:
        return None
    if len(sample_weight) != len(raw_candles):
        raise ValueError("sample_weight debe coincidir con las velas crudas del experimento.")
    weight_frame = raw_candles[["open_time"]].copy()
    weight_frame["sample_weight"] = sample_weight
    merged = labeled[["open_time"]].merge(weight_frame, on="open_time", how="left")
    if merged["sample_weight"].isna().any():
        raise ValueError("No se pudieron alinear todos los sample_weight con labeled.")
    return merged["sample_weight"].to_numpy(dtype=float)


def fit_final_model(
    train_df: pd.DataFrame,
    stable_features: list[str],
    sample_weight: np.ndarray | None,
) -> object:
    """Entrena el modelo final usando el mismo esquema que el validador."""
    split = max(1, int(len(train_df) * 0.85))
    X_train = train_df.iloc[:split][stable_features]
    y_train = train_df.iloc[:split]["target"].to_numpy(dtype=int)
    X_val = train_df.iloc[split:][stable_features]
    y_val = train_df.iloc[split:]["target"].to_numpy(dtype=int)
    w_train = sample_weight[:split] if sample_weight is not None else None
    w_val = sample_weight[split:] if sample_weight is not None else None
    if len(X_val) == 0 or len(np.unique(y_val)) < 2:
        X_val = X_train.iloc[-min(100, len(X_train)) :]
        y_val = y_train[-len(X_val) :]
        if w_train is not None:
            w_val = w_train[-len(X_val) :]

    model = model_factory()
    fit_kwargs: dict[str, object] = {}
    import inspect

    fit_sig = inspect.signature(model.fit)
    if "eval_set" in fit_sig.parameters:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
    if w_train is not None and "sample_weight" in fit_sig.parameters:
        fit_kwargs["sample_weight"] = w_train
    if w_val is not None and "sample_weight_eval_set" in fit_sig.parameters:
        fit_kwargs["sample_weight_eval_set"] = [w_val]
    if "verbose" in fit_sig.parameters:
        fit_kwargs["verbose"] = False
    if "early_stopping_rounds" in fit_sig.parameters:
        fit_kwargs["early_stopping_rounds"] = 30

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        fit_kwargs.pop("early_stopping_rounds", None)
        fit_kwargs.pop("sample_weight_eval_set", None)
        model.fit(X_train, y_train, **fit_kwargs)
    return model


def model_factory() -> object:
    """Replica la factoria principal del proyecto."""
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(**settings.XGB_PARAMS)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=42)


def predict_proba(model: object, frame: pd.DataFrame) -> np.ndarray:
    """Normaliza la salida probabilistica del modelo."""
    if hasattr(model, "predict_proba"):
        probs_array = np.asarray(model.predict_proba(frame), dtype=float)
        return probs_array[:, 1] if probs_array.ndim == 2 else probs_array
    preds = model.predict(frame)
    return np.asarray(preds, dtype=float)


def summarize_weights(sample_weight: np.ndarray | None) -> dict[str, float] | None:
    """Resume el esquema de pesos usado en la estrategia."""
    if sample_weight is None:
        return None
    return {
        "min": float(np.min(sample_weight)),
        "max": float(np.max(sample_weight)),
        "mean": float(np.mean(sample_weight)),
    }


def to_iso(timestamp_ms: int | float) -> str:
    """Convierte un timestamp ms a ISO UTC."""
    return pd.to_datetime(int(timestamp_ms), unit="ms", utc=True).isoformat()


if __name__ == "__main__":
    main()
