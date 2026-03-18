"""Diagnostica el sesgo direccional de un activo dentro de una sesion exportada."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import DMatrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from features.calculator import FeatureCalculator
from historical_utils import normalize_candles, read_candles_file
from model.inference import InferenceEngine
from model.registry import MultiAssetModelRegistry
from target.builder import TargetBuilder, TargetConfig, TargetType


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI."""
    parser = argparse.ArgumentParser(description="Depura el sesgo de señales de un activo.")
    parser.add_argument("--session-dir", required=True, help="Directorio exportado por run_paper_session.py.")
    parser.add_argument("--registry-root", required=True, help="Root del registry a usar.")
    parser.add_argument("--product-id", required=True, help="Producto a diagnosticar, por ejemplo ZEC-USD.")
    parser.add_argument(
        "--historical-path",
        default="",
        help="CSV/Parquet historico opcional para medir balance de target del activo.",
    )
    return parser


def main() -> None:
    """Ejecuta el diagnostico y emite un resumen JSON."""
    args = build_parser().parse_args()
    session_dir = Path(args.session_dir)
    registry = MultiAssetModelRegistry(root_dir=args.registry_root, execution_allowed_bases=[args.product_id.split("-", 1)[0]])
    artifact = registry.resolve_artifact(args.product_id)
    if artifact is None or not artifact.model_path:
        raise FileNotFoundError(f"No hay modelo resoluble para {args.product_id} en {args.registry_root}.")

    features_frame = pd.read_csv(session_dir / "market_features.csv")
    signals_frame = pd.read_csv(session_dir / "inference_signals.csv")
    asset_features = features_frame[features_frame["product_id"] == args.product_id].copy()
    asset_signals = signals_frame[signals_frame["product_id"] == args.product_id].copy()
    if asset_features.empty or asset_signals.empty:
        raise ValueError(f"No hay filas de features/senales para {args.product_id} en {session_dir}.")

    model = InferenceEngine._load_model(artifact.model_path)
    X = asset_features[artifact.feature_names].astype(float)
    predicted_prob_buy = _predict_proba(model, X)
    signal_prob_buy = asset_signals["prob_buy"].astype(float).to_numpy(dtype=float)

    contribution_summary = _summarize_contributions(model, X, artifact.feature_names)
    historical_target_summary = _summarize_historical_targets(args.historical_path, args.product_id)

    sell_threshold = 1.0 - settings.MIN_SIGNAL_PROB
    output = {
        "product_id": args.product_id,
        "registry_key": artifact.registry_key,
        "model_id": artifact.model_id,
        "session_dir": str(session_dir),
        "rows": int(len(asset_signals)),
        "signal_counts": asset_signals["signal"].value_counts().to_dict(),
        "sell_threshold_prob_buy": round(sell_threshold, 6),
        "buy_threshold_prob_buy": round(settings.MIN_SIGNAL_PROB, 6),
        "prob_buy_summary_from_signals": _series_summary(signal_prob_buy),
        "prob_buy_summary_recomputed": _series_summary(predicted_prob_buy),
        "signal_prob_alignment_max_abs_diff": round(float(np.max(np.abs(signal_prob_buy - predicted_prob_buy))), 8),
        "latest_prob_buy": [round(float(value), 6) for value in signal_prob_buy[-5:]],
        "top_negative_contributors": contribution_summary["top_negative_contributors"],
        "top_positive_contributors": contribution_summary["top_positive_contributors"],
        "feature_value_snapshot": _latest_feature_snapshot(asset_features, artifact.feature_names),
        "historical_target_summary": historical_target_summary,
        "diagnosis": _build_diagnosis(
            product_id=args.product_id,
            signal_counts=asset_signals["signal"].value_counts().to_dict(),
            prob_buy=signal_prob_buy,
            contribution_summary=contribution_summary,
            historical_target_summary=historical_target_summary,
        ),
    }
    print(json.dumps(output, indent=2))


def _summarize_contributions(
    model: object,
    frame: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, list[dict[str, float]]]:
    """Calcula contribuciones medias del modelo si es XGBoost."""
    if not hasattr(model, "get_booster"):
        return {"top_negative_contributors": [], "top_positive_contributors": []}

    booster = model.get_booster()
    dmatrix = DMatrix(frame, feature_names=feature_names)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    contrib_frame = pd.DataFrame(contribs, columns=feature_names + ["bias"])
    mean_contrib = contrib_frame.mean().sort_values()
    top_negative = [
        {"feature": str(feature), "mean_contribution": round(float(value), 6)}
        for feature, value in mean_contrib.head(6).items()
    ]
    top_positive = [
        {"feature": str(feature), "mean_contribution": round(float(value), 6)}
        for feature, value in mean_contrib.tail(6).items()
    ]
    return {
        "top_negative_contributors": top_negative,
        "top_positive_contributors": top_positive,
    }


def _latest_feature_snapshot(frame: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
    """Retorna un snapshot corto de las features mas recientes."""
    latest = frame.iloc[-1]
    snapshot_keys = feature_names[: min(10, len(feature_names))]
    return {key: round(float(latest[key]), 6) for key in snapshot_keys}


def _series_summary(values: np.ndarray) -> dict[str, float]:
    """Resume una serie numerica corta."""
    if values.size == 0:
        return {}
    return {
        "min": round(float(np.min(values)), 6),
        "p25": round(float(np.percentile(values, 25)), 6),
        "median": round(float(np.median(values)), 6),
        "mean": round(float(np.mean(values)), 6),
        "p75": round(float(np.percentile(values, 75)), 6),
        "max": round(float(np.max(values)), 6),
    }


def _predict_proba(model: object, frame: pd.DataFrame) -> np.ndarray:
    """Obtiene P(BUY) con el mismo contrato que inferencia."""
    if hasattr(model, "predict_proba"):
        probs_array = np.asarray(model.predict_proba(frame), dtype=float)
        return probs_array[:, 1] if probs_array.ndim == 2 else probs_array
    return np.asarray(model.predict(frame), dtype=float)


def _summarize_historical_targets(historical_path: str, product_id: str) -> dict[str, Any]:
    """Mide el balance historico del target si se pasa un archivo fuente."""
    if not historical_path:
        return {}

    raw = read_candles_file(historical_path)
    candles = normalize_candles(raw, product_id=product_id, candle_seconds=settings.CANDLE_SECONDS)
    features_df = FeatureCalculator().compute(candles)
    labeled_df = TargetBuilder(
        TargetConfig(
            target_type=TargetType.NET_RETURN_THRESHOLD,
            horizon=settings.TARGET_HORIZON,
            threshold_pct=settings.TARGET_THRESHOLD_PCT,
            fee_pct=settings.FEE_PCT,
        )
    ).build(features_df)
    if labeled_df.empty:
        return {"rows": 0}

    target_share = labeled_df["target"].mean()
    return {
        "rows": int(len(labeled_df)),
        "buy_share": round(float(target_share), 6),
        "sell_share": round(float(1.0 - target_share), 6),
    }


def _build_diagnosis(
    product_id: str,
    signal_counts: dict[str, int],
    prob_buy: np.ndarray,
    contribution_summary: dict[str, list[dict[str, float]]],
    historical_target_summary: dict[str, Any],
) -> dict[str, Any]:
    """Construye una lectura compacta del sesgo observado."""
    sell_count = int(signal_counts.get("SELL", 0))
    hold_count = int(signal_counts.get("HOLD", 0))
    buy_count = int(signal_counts.get("BUY", 0))
    total = max(sell_count + hold_count + buy_count, 1)
    strongly_bearish = float(np.mean(prob_buy < (1.0 - settings.MIN_SIGNAL_PROB))) if prob_buy.size else 0.0

    diagnosis = {
        "product_id": product_id,
        "sell_share_in_session": round(sell_count / total, 6),
        "buy_share_in_session": round(buy_count / total, 6),
        "strongly_bearish_prob_share": round(strongly_bearish, 6),
        "likely_threshold_issue": bool(prob_buy.size and np.quantile(prob_buy, 0.75) > (1.0 - settings.MIN_SIGNAL_PROB)),
        "likely_model_bias": bool(prob_buy.size and np.mean(prob_buy) < 0.30),
        "headline": "",
    }

    top_negative = contribution_summary.get("top_negative_contributors", [])
    negative_features = [item["feature"] for item in top_negative]
    historical_sell_share = float(historical_target_summary.get("sell_share", 0.0) or 0.0)
    if diagnosis["likely_model_bias"]:
        diagnosis["headline"] = (
            "El sesgo SELL parece venir del modelo y de las features recientes, no de un threshold tibio."
        )
    else:
        diagnosis["headline"] = "El sesgo SELL parece mas cercano a un problema de calibracion de umbral."

    diagnosis["negative_features"] = negative_features
    diagnosis["historical_sell_share"] = round(historical_sell_share, 6)
    return diagnosis


if __name__ == "__main__":
    main()
