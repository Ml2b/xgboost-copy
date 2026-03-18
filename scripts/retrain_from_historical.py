"""Lanza un reentrenamiento puntual desde un archivo historico."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from features.selector import SelectorConfig
from historical_utils import normalize_candles, read_candles_file
from model.registry import ModelRegistry
from model.trainer import Trainer
from validation.walk_forward import WalkForwardConfig


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI."""
    parser = argparse.ArgumentParser(description="Reentrena un modelo desde velas historicas.")
    parser.add_argument("--path", required=True, help="Ruta al CSV o Parquet.")
    parser.add_argument("--product-id", default=None, help="Producto a usar si el archivo no lo trae.")
    parser.add_argument("--registry-dir", default="models", help="Directorio donde guardar modelos.")
    parser.add_argument("--n-splits", type=int, default=5, help="Cantidad de folds walk-forward.")
    parser.add_argument("--min-train-size", type=int, default=1000, help="Minimo de filas de train por fold.")
    parser.add_argument("--gap-periods", type=int, default=settings.TARGET_HORIZON, help="Gap temporal entre train y val.")
    parser.add_argument("--max-features", type=int, default=25, help="Maximo de features finales.")
    parser.add_argument("--verbose", action="store_true", help="Muestra detalles del walk-forward.")
    return parser


def main() -> None:
    """Punto de entrada CLI."""
    args = build_parser().parse_args()
    raw = read_candles_file(args.path)
    candles = normalize_candles(raw, product_id=args.product_id, candle_seconds=settings.CANDLE_SECONDS)

    registry = ModelRegistry(base_dir=args.registry_dir)
    trainer = Trainer(
        registry=registry,
        data_loader=lambda: candles.copy(),
        walk_forward_config=WalkForwardConfig(
            n_splits=args.n_splits,
            gap_periods=args.gap_periods,
            min_train_size=args.min_train_size,
            verbose=args.verbose,
        ),
        selector_config=SelectorConfig(max_features=args.max_features, verbose=args.verbose),
    )

    result = trainer._retrain_cycle()
    output = {
        "status": result.status,
        "promoted": result.promoted,
        "auc": result.auc,
        "feature_names": result.feature_names or [],
        "used_continuation": result.used_continuation,
        "history_rows": result.history_rows,
        "model_id": result.record.model_id if result.record else None,
        "model_path": result.record.model_path if result.record else None,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
