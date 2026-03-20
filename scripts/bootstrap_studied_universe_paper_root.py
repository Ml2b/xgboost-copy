"""Consolida los 24 activos estudiados en un root paper separado."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from model.registry import ModelRegistry, MultiAssetModelRegistry


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI."""
    parser = argparse.ArgumentParser(
        description="Consolida el universo paper de 24 activos sin tocar el live principal."
    )
    parser.add_argument(
        "--live-root",
        default="models/multi_asset_live_v2",
        help="Root fuente del universo principal (core + fase 1).",
    )
    parser.add_argument(
        "--phase2-root",
        default="tests/legacy/models/multi_asset_phase2_experimental",
        help="Root fuente de la cohorte phase2.",
    )
    parser.add_argument(
        "--target-root",
        default=settings.STUDIED_UNIVERSE_PAPER_ROOT,
        help="Root destino para el universo paper consolidado.",
    )
    return parser


def bootstrap_studied_universe_paper_root(
    target_root: str | Path,
    live_root: str | Path = "models/multi_asset_live_v2",
    phase2_root: str | Path = "tests/legacy/models/multi_asset_phase2_experimental",
) -> dict[str, Any]:
    """Copia y normaliza los 24 activos estudiados en un root paper auditable."""
    target_path = Path(target_root)
    live_root_path = Path(live_root)
    phase2_root_path = Path(phase2_root)
    temp_target_path = target_path.parent / f"{target_path.name}.tmp"
    if temp_target_path.exists():
        shutil.rmtree(temp_target_path)
    temp_target_path.mkdir(parents=True, exist_ok=True)

    copied_assets: list[dict[str, Any]] = []
    for base in settings.STUDIED_UNIVERSE_BASES:
        source_root = live_root_path if base in settings.CORE_BASES + settings.PHASE_ONE_BASES else phase2_root_path
        asset_summary = _copy_asset_into_paper_root(
            base_asset=base,
            source_root=source_root,
            target_root=temp_target_path,
            final_target_root=target_path,
        )
        copied_assets.append(asset_summary)

    copied_bases = {asset["base"] for asset in copied_assets}
    expected_bases = set(settings.STUDIED_UNIVERSE_BASES)
    if copied_bases != expected_bases:
        raise ValueError(
            f"El universo consolidado quedo incompleto. expected={sorted(expected_bases)} got={sorted(copied_bases)}"
        )

    manifest = {
        "phase": "studied_universe_paper_24",
        "target_root": str(target_path),
        "live_root": str(live_root_path),
        "phase2_root": str(phase2_root_path),
        "observed_bases": list(settings.STUDIED_UNIVERSE_BASES),
        "excluded_experiment_bases": list(settings.STUDIED_UNIVERSE_EXPERIMENT_BASES),
        "paper_activation_policy": "promoted_or_latest_candidate",
        "assets": copied_assets,
    }
    (temp_target_path / "studied_universe_assets.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if target_path.exists():
        shutil.rmtree(target_path)
    temp_target_path.rename(target_path)
    return manifest


def _copy_asset_into_paper_root(
    base_asset: str,
    source_root: Path,
    target_root: Path,
    final_target_root: Path,
) -> dict[str, Any]:
    """Copia un asset y activa su ultimo modelo en el root paper."""
    source_registry = MultiAssetModelRegistry(
        root_dir=source_root,
        execution_allowed_bases=settings.STUDIED_UNIVERSE_BASES,
    )
    artifact = source_registry.resolve_artifact(f"{base_asset}-USD")
    if artifact is None:
        raise FileNotFoundError(f"No se encontro metadata para {base_asset} dentro de {source_root}.")

    source_asset_dir = source_root / artifact.registry_key
    if not source_asset_dir.exists():
        raise FileNotFoundError(f"No existe el directorio fuente de {base_asset}: {source_asset_dir}")

    target_asset_dir = target_root / artifact.registry_key
    if target_asset_dir.exists():
        shutil.rmtree(target_asset_dir)
    shutil.copytree(source_asset_dir, target_asset_dir)

    final_asset_dir = final_target_root / artifact.registry_key
    normalized_records = _normalize_registry_paths(target_asset_dir, final_asset_dir)
    activation_source = _ensure_active_meta(target_asset_dir, final_asset_dir, normalized_records)
    return {
        "base": base_asset,
        "registry_key": artifact.registry_key,
        "source_root": str(source_root),
        "target_dir": str(final_asset_dir),
        "activation_source": activation_source,
        "source_reason": artifact.reason,
        "activation_auc": (
            round(float(normalized_records[-1].metrics.auc_val), 6) if normalized_records else 0.0
        ),
        "feature_count": len(normalized_records[-1].feature_names) if normalized_records else 0,
        "model_count": len(normalized_records),
    }


def _normalize_registry_paths(asset_dir: Path, final_asset_dir: Path) -> list[Any]:
    """Reescribe rutas absolutas para que el root paper sea autocontenido."""
    registry_path = asset_dir / "registry.json"
    records = ModelRegistry.load_records_from_path(registry_path)
    if not records:
        return []

    serialized_records: list[dict[str, Any]] = []
    for record in records:
        local_model_path = Path(record.model_path).name
        payload = {
            "model_id": record.model_id,
            "model_path": local_model_path,
            "created_at": record.created_at,
            "train_start": record.train_start,
            "train_end": record.train_end,
            "val_start": record.val_start,
            "val_end": record.val_end,
            "metrics": {
                "auc_val": record.metrics.auc_val,
                "sharpe": record.metrics.sharpe,
                "precision_buy": record.metrics.precision_buy,
                "win_rate": record.metrics.win_rate,
                "max_drawdown": record.metrics.max_drawdown,
            },
            "promoted": record.promoted,
            "feature_names": list(record.feature_names),
            "n_features": record.n_features,
        }
        serialized_records.append(payload)

    registry_path.write_text(json.dumps(serialized_records, indent=2), encoding="utf-8")
    return ModelRegistry.load_records_from_path(registry_path)


def _ensure_active_meta(asset_dir: Path, final_asset_dir: Path, records: list[Any]) -> str:
    """Garantiza un active_model_meta local incluso si solo habia candidato."""
    if not records:
        raise ValueError(f"{asset_dir} no tiene registros entrenados para activar en paper.")

    active_meta_path = asset_dir / "active_model_meta.json"
    activation_source = "promoted"

    if active_meta_path.exists():
        active_meta = json.loads(active_meta_path.read_text(encoding="utf-8"))
        active_meta["model_path"] = Path(str(active_meta.get("model_path", ""))).name
        active_meta_path.write_text(json.dumps(active_meta, indent=2), encoding="utf-8")
        return activation_source

    latest = records[-1]
    active_meta_path.write_text(
        json.dumps(
            {
                "model_id": latest.model_id,
                "model_path": Path(latest.model_path).name,
                "feature_names": list(latest.feature_names),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return "candidate_latest"


def main() -> None:
    """Entry point CLI."""
    args = build_parser().parse_args()
    manifest = bootstrap_studied_universe_paper_root(
        target_root=args.target_root,
        live_root=args.live_root,
        phase2_root=args.phase2_root,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
