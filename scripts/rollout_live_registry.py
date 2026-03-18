"""Clona el registry live y aplica estrategias historicas ganadoras por activo."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from features.calculator import FeatureCalculator
from features.selector import SelectorConfig
from model.registry import ModelMetrics, ModelRegistry
from model.trainer import Trainer
from scripts.compare_history_windows import (
    align_weights_to_labeled,
    build_recent_weight_vector,
    fit_final_model,
)
from target.builder import TargetBuilder, TargetConfig, TargetType
from validation.walk_forward import WalkForwardConfig, WalkForwardValidator


DEFAULT_SELECTIONS = ["BTC/USDT=12m_flat", "ETH/USDT=12m_weighted"]


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI del rollout live."""
    parser = argparse.ArgumentParser(description="Clona el registry live y aplica estrategias historicas.")
    parser.add_argument("--source-root", default="models/multi_asset_6m_binanceus", help="Root base actual.")
    parser.add_argument("--target-root", default="models/multi_asset_live_v2", help="Nuevo root live.")
    parser.add_argument("--data-dir", default="data/historical_12m_binanceus", help="Directorio con CSVs de 12 meses.")
    parser.add_argument("--exchange", default="binanceus", help="Sufijo del exchange usado en los CSVs.")
    parser.add_argument("--timeframe", default="1m", help="Timeframe usado en los CSVs.")
    parser.add_argument("--candles-6m", type=int, default=259200, help="Tamano del baseline reciente.")
    parser.add_argument("--holdout-candles", type=int, default=43200, help="Tamano del holdout fijo final.")
    parser.add_argument("--recent-weight", type=float, default=2.0, help="Peso relativo del semestre reciente.")
    parser.add_argument("--n-splits", type=int, default=5, help="Cantidad de folds walk-forward.")
    parser.add_argument("--min-train-size", type=int, default=1000, help="Minimo de train por fold.")
    parser.add_argument("--max-features", type=int, default=25, help="Maximo de features finales.")
    parser.add_argument(
        "--selection",
        action="append",
        default=None,
        help="Asignacion SYMBOL=STRATEGY. Repetible. Ej: BTC/USDT=12m_flat",
    )
    parser.add_argument(
        "--report-path",
        default="reports/live_registry_rollout_v2.json",
        help="Ruta del resumen JSON final.",
    )
    return parser


def main() -> None:
    """Punto de entrada del rollout."""
    args = build_parser().parse_args()
    selections = parse_selections(args.selection or DEFAULT_SELECTIONS)

    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    clone_registry_root(source_root, target_root)

    summaries: list[dict[str, object]] = []
    for symbol, strategy in selections.items():
        summary = train_selection(
            symbol=symbol,
            strategy=strategy,
            data_dir=Path(args.data_dir),
            exchange=args.exchange,
            timeframe=args.timeframe,
            target_root=target_root,
            candles_6m=args.candles_6m,
            holdout_candles=args.holdout_candles,
            recent_weight=args.recent_weight,
            n_splits=args.n_splits,
            min_train_size=args.min_train_size,
            max_features=args.max_features,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    report = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "selections": summaries,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "completed": len(summaries)}, indent=2))


def clone_registry_root(source_root: Path, target_root: Path) -> None:
    """Copia el root completo y reescribe rutas internas al nuevo destino."""
    if not source_root.exists():
        raise FileNotFoundError(f"No existe el root fuente: {source_root}")
    if target_root.exists():
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root)

    for asset_dir in (path for path in target_root.iterdir() if path.is_dir()):
        rewrite_registry_paths(asset_dir)


def rewrite_registry_paths(asset_dir: Path) -> None:
    """Reapunta registry.json y active_model_meta.json al directorio clonado."""
    registry_path = asset_dir / "registry.json"
    if registry_path.exists():
        records = json.loads(registry_path.read_text(encoding="utf-8"))
        for record in records:
            record["model_path"] = str(asset_dir / Path(str(record["model_path"])).name)
        registry_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    active_meta_path = asset_dir / "active_model_meta.json"
    if active_meta_path.exists():
        meta = json.loads(active_meta_path.read_text(encoding="utf-8"))
        meta["model_path"] = str(asset_dir / Path(str(meta["model_path"])).name)
        active_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def parse_selections(items: list[str]) -> dict[str, str]:
    """Parsea la lista SYMBOL=STRATEGY a un diccionario normalizado."""
    parsed: dict[str, str] = {}
    allowed = {"6m_recent", "12m_flat", "12m_weighted"}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Selection invalida: {item}")
        symbol, strategy = item.split("=", 1)
        symbol = symbol.strip().upper()
        strategy = strategy.strip()
        if strategy not in allowed:
            raise ValueError(f"Estrategia invalida para {symbol}: {strategy}")
        parsed[symbol] = strategy
    return parsed


def train_selection(
    symbol: str,
    strategy: str,
    data_dir: Path,
    exchange: str,
    timeframe: str,
    target_root: Path,
    candles_6m: int,
    holdout_candles: int,
    recent_weight: float,
    n_splits: int,
    min_train_size: int,
    max_features: int,
) -> dict[str, object]:
    """Entrena una seleccion concreta y la registra en el root live nuevo."""
    csv_path = data_dir / f"{symbol.replace('/', '_').lower()}_{timeframe}_{exchange}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el CSV esperado para {symbol}: {csv_path}")

    raw_candles = pd.read_csv(csv_path).sort_values("open_time").reset_index(drop=True)
    selected_candles, raw_weights = select_window(
        frame=raw_candles,
        strategy=strategy,
        candles_6m=candles_6m,
        recent_weight=recent_weight,
    )

    calculator = FeatureCalculator()
    target_builder = TargetBuilder(
        TargetConfig(
            target_type=TargetType.NET_RETURN_THRESHOLD,
            horizon=settings.TARGET_HORIZON,
            threshold_pct=settings.TARGET_THRESHOLD_PCT,
            fee_pct=settings.FEE_PCT,
        )
    )
    features_df = calculator.compute(selected_candles.copy())
    labeled = target_builder.build(features_df)
    if len(labeled) <= holdout_candles + settings.TARGET_HORIZON + min_train_size:
        raise ValueError(f"{symbol} no tiene filas efectivas suficientes para {strategy}.")

    weights_labeled = align_weights_to_labeled(selected_candles, labeled, raw_weights)
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
        model_factory=Trainer._model_factory,
        selector_config=SelectorConfig(max_features=max_features, verbose=False),
    )
    wf_result = validator.validate(
        train_df,
        target_col="target",
        feature_cols=feature_cols,
        sample_weight=train_weights,
    )
    model = fit_final_model(
        train_df=train_df,
        stable_features=wf_result.stable_features,
        sample_weight=train_weights,
    )

    probs = Trainer._predict_proba(model, holdout_df[wf_result.stable_features])
    metrics = build_model_metrics(holdout_df=holdout_df, probs=probs)
    registry_dir = target_root / symbol.replace("/", "_").lower()
    registry = ModelRegistry(base_dir=registry_dir)
    record = registry.register(
        model=model,
        metrics=metrics,
        fechas={
            "train_start": to_iso(train_df["open_time"].iloc[0]),
            "train_end": to_iso(train_df["open_time"].iloc[-1]),
            "val_start": to_iso(holdout_df["open_time"].iloc[0]),
            "val_end": to_iso(holdout_df["open_time"].iloc[-1]),
        },
        feature_names=wf_result.stable_features,
    )
    promoted = registry.try_promote(record)

    return {
        "symbol": symbol,
        "strategy": strategy,
        "csv_path": str(csv_path),
        "registry_dir": str(registry_dir),
        "status": "trained",
        "promoted": promoted,
        "model_id": record.model_id,
        "model_path": record.model_path,
        "auc": metrics.auc_val,
        "sharpe": metrics.sharpe,
        "precision_buy": metrics.precision_buy,
        "win_rate": metrics.win_rate,
        "max_drawdown": metrics.max_drawdown,
        "stable_feature_count": len(wf_result.stable_features),
        "stable_features": wf_result.stable_features,
    }


def select_window(
    frame: pd.DataFrame,
    strategy: str,
    candles_6m: int,
    recent_weight: float,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Devuelve la ventana cruda y los pesos a usar para una estrategia."""
    if strategy == "6m_recent":
        return frame.tail(candles_6m).reset_index(drop=True), None
    if strategy == "12m_flat":
        return frame.reset_index(drop=True), None
    if strategy == "12m_weighted":
        ordered = frame.reset_index(drop=True)
        return ordered, build_recent_weight_vector(ordered, candles_6m, recent_weight)
    raise ValueError(f"Estrategia no soportada: {strategy}")


def build_model_metrics(holdout_df: pd.DataFrame, probs: np.ndarray) -> ModelMetrics:
    """Calcula las metricas del registro usando el holdout fijo final."""
    target = holdout_df["target"].to_numpy(dtype=int)
    returns = holdout_df["target_return"].to_numpy(dtype=float)
    auc = float(roc_auc_score(target, probs))
    buy_preds = (probs >= settings.MIN_SIGNAL_PROB).astype(int)
    precision_buy = float(precision_score(target, buy_preds, zero_division=0))
    trade_returns = returns[buy_preds == 1]
    win_rate = float(np.mean(trade_returns > 0)) if len(trade_returns) else 0.0
    sharpe = Trainer._compute_sharpe(trade_returns)
    max_drawdown = Trainer._compute_max_drawdown(trade_returns)
    return ModelMetrics(
        auc_val=auc,
        sharpe=sharpe,
        precision_buy=precision_buy,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
    )


def to_iso(timestamp_ms: int | float) -> str:
    """Convierte un timestamp ms a ISO UTC."""
    return pd.to_datetime(int(timestamp_ms), unit="ms", utc=True).isoformat()


if __name__ == "__main__":
    main()
