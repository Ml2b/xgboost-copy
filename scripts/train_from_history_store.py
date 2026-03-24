"""Entrena un modelo por activo usando los datos del CandleHistoryStore (SQLite).

Flujo completo:
  1. bootstrap_order_flow.py  →  descarga OHLCV + 45 order flow columns
  2. train_from_history_store.py  →  lee SQLite, computa 109 features, entrena

Uso:
    # Entrenar todos los activos con datos en el SQLite
    python scripts/train_from_history_store.py

    # Solo BTC y ETH, guardar modelos en carpeta personalizada
    python scripts/train_from_history_store.py --bases BTC ETH --registry-root models/test

    # Con mas folds y minimo de datos mas exigente
    python scripts/train_from_history_store.py --n-splits 5 --min-train-size 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from config import settings
from data.history_store import CandleHistoryStore
from features.selector import SelectorConfig
from model.registry import ModelRegistry, MultiAssetModelRegistry
from model.trainer import Trainer
from validation.walk_forward import WalkForwardConfig
from features.order_flow import ORDER_FLOW_RAW_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena modelos desde CandleHistoryStore con order flow."
    )
    parser.add_argument(
        "--db",
        default=str(settings.TRAINER_HISTORY_DB_PATH),
        help="Ruta al SQLite generado por bootstrap_order_flow.py",
    )
    parser.add_argument(
        "--registry-root",
        default=str(settings.MODEL_REGISTRY_ROOT),
        help="Directorio raiz donde guardar los modelos entrenados",
    )
    parser.add_argument(
        "--bases",
        nargs="+",
        default=None,
        help="Activos base a entrenar (ej: BTC ETH SOL). "
             "Default: todos los que tengan datos en el SQLite.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Folds de walk-forward (default: 3)",
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=500,
        help="Minimo de candles de entrenamiento por fold (default: 500)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=25,
        help="Maximo de features seleccionadas (default: 25)",
    )
    parser.add_argument(
        "--gap-periods",
        type=int,
        default=settings.TARGET_HORIZON,
        help="Gap entre train y val para evitar leakage",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra detalles del walk-forward",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Ruta opcional donde guardar resumen JSON",
    )
    return parser


def discover_bases(store: CandleHistoryStore) -> list[str]:
    """Detecta todos los activos base con datos en el SQLite."""
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT base_asset FROM candles ORDER BY base_asset"
        ).fetchall()
    return [r[0] for r in rows]


def train_base(
    base_asset: str,
    store: CandleHistoryStore,
    registry_root: Path,
    wf_config: WalkForwardConfig,
    sel_config: SelectorConfig,
) -> dict:
    """Entrena un activo y devuelve el resumen del ciclo."""
    registry_key = f"{base_asset.lower()}_usd"
    registry_dir = registry_root / registry_key
    registry_dir.mkdir(parents=True, exist_ok=True)

    def load_candles():
        return store.load_candles_for_base(base_asset)

    def load_order_flow():
        df = store.get_candles_with_order_flow(base_asset)
        if df.empty:
            return None
        of_cols = ["open_time"] + [
            c for c in ORDER_FLOW_RAW_COLUMNS if c in df.columns
        ]
        return df[of_cols]

    trainer = Trainer(
        registry=ModelRegistry(base_dir=registry_dir),
        data_loader=load_candles,
        order_flow_loader=load_order_flow,
        walk_forward_config=wf_config,
        selector_config=sel_config,
    )

    t0 = time.perf_counter()
    result = trainer._retrain_cycle()
    elapsed = round(time.perf_counter() - t0, 1)

    return {
        "base_asset": base_asset,
        "status": result.status,
        "promoted": result.promoted,
        "auc": round(result.auc, 4),
        "history_rows": result.history_rows,
        "feature_count": len(result.feature_names or []),
        "features": result.feature_names or [],
        "used_continuation": result.used_continuation,
        "model_id": result.record.model_id if result.record else None,
        "reason": result.reason,
        "elapsed_s": elapsed,
    }


def main() -> None:
    args = build_parser().parse_args()

    store = CandleHistoryStore(db_path=args.db)
    registry_root = Path(args.registry_root)

    bases = args.bases or discover_bases(store)
    if not bases:
        logger.error("No hay activos en el SQLite. Ejecuta bootstrap_order_flow.py primero.")
        sys.exit(1)

    wf_config = WalkForwardConfig(
        n_splits=args.n_splits,
        gap_periods=args.gap_periods,
        min_train_size=args.min_train_size,
        verbose=args.verbose,
    )
    sel_config = SelectorConfig(
        max_features=args.max_features,
        verbose=args.verbose,
    )

    logger.info(
        "Entrenamiento iniciado. activos={} db={} registry={}",
        len(bases), args.db, args.registry_root,
    )

    summaries = []
    for base in bases:
        logger.info("Entrenando {}...", base)
        try:
            summary = train_base(base, store, registry_root, wf_config, sel_config)
            summaries.append(summary)
            status_icon = "✓" if summary["status"] == "trained" else "✗"
            logger.info(
                "{} {} — status={} auc={} features={} promoted={} ({:.1f}s)",
                status_icon, base,
                summary["status"],
                summary["auc"],
                summary["feature_count"],
                summary["promoted"],
                summary["elapsed_s"],
            )
        except Exception as exc:
            logger.error("Error entrenando {}. exc={}", base, exc)
            summaries.append({
                "base_asset": base,
                "status": "error",
                "reason": str(exc),
            })

    trained = sum(1 for s in summaries if s.get("status") == "trained")
    promoted = sum(1 for s in summaries if s.get("promoted"))

    final = {
        "total": len(bases),
        "trained": trained,
        "promoted": promoted,
        "skipped": len(bases) - trained,
        "results": summaries,
    }

    print(json.dumps(final, indent=2))

    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(final, indent=2), encoding="utf-8")
        logger.info("Resumen guardado en {}", args.summary)

    logger.info(
        "Entrenamiento completado. trained={}/{} promoted={}",
        trained, len(bases), promoted,
    )


if __name__ == "__main__":
    main()
