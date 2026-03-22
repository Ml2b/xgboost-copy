"""Registro y promocion de modelos entrenados."""

from __future__ import annotations

import json
import os
import shutil
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import settings


def _looks_like_windows_absolute_path(raw_path: str) -> bool:
    """Detecta paths estilo Windows incluso en runtimes Linux."""
    return len(raw_path) >= 3 and raw_path[1] == ":" and raw_path[2] in ("\\", "/")


def _serialize_model_reference(base_dir: Path, model_path: Path) -> str:
    """Guarda rutas relativas al registry para hacer portable el artefacto."""
    try:
        return model_path.relative_to(base_dir).as_posix()
    except ValueError:
        return model_path.name


def resolve_model_artifact_path(base_dir: Path, stored_path: str | None) -> Path | None:
    """Resuelve rutas relativas y recupera artefactos serializados en otro SO."""
    raw_path = str(stored_path or "").strip()
    if not raw_path:
        return None

    normalized = raw_path.replace("\\", "/")
    native_path = Path(raw_path)
    if native_path.is_absolute() and native_path.exists():
        return native_path

    normalized_path = Path(normalized)
    if not _looks_like_windows_absolute_path(normalized):
        relative_candidate = base_dir / normalized_path
        if relative_candidate.exists():
            return relative_candidate

    basename = normalized_path.name
    if basename:
        basename_candidate = base_dir / basename
        if basename_candidate.exists():
            return basename_candidate

    if not _looks_like_windows_absolute_path(normalized):
        return base_dir / normalized_path
    if basename:
        return base_dir / basename
    return None


def compute_artifact_sha256(model_path: Path | None) -> str:
    """Calcula un fingerprint estable del artefacto para detectar desalineaciones."""
    if model_path is None or not model_path.exists():
        return ""
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class ModelMetrics:
    """Metricas usadas para registrar y promover modelos."""

    auc_val: float
    sharpe: float
    precision_buy: float
    win_rate: float
    max_drawdown: float
    buy_support: int = 0
    exit_support: int = 0
    buy_threshold: float = settings.MIN_SIGNAL_PROB
    exit_threshold: float = 1.0 - settings.MIN_SIGNAL_PROB
    entry_ready: bool = False


@dataclass(slots=True)
class ModelRecord:
    """Metadata persistida para cada modelo."""

    model_id: str
    model_path: str
    created_at: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    metrics: ModelMetrics
    promoted: bool
    feature_names: list[str]
    n_features: int


@dataclass(slots=True)
class ResolvedModelArtifact:
    """Metadata lista para consumir por inferencia multi-activo."""

    registry_key: str
    base_asset: str
    model_id: str
    model_path: str | None
    feature_names: list[str]
    actionable: bool
    reason: str
    fingerprint: str


class ModelRegistry:
    """Guarda el historial de modelos y evita regresiones."""

    def __init__(self, base_dir: str | Path = "models") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base_dir / "registry.json"
        self.active_meta_path = self.base_dir / "active_model_meta.json"
        if not self.registry_path.exists():
            self._write_registry([])

    def register(
        self,
        model: object,
        metrics: ModelMetrics,
        fechas: dict[str, str],
        feature_names: list[str],
    ) -> ModelRecord:
        """Persiste un nuevo modelo y su metadata."""
        model_id = datetime.now(tz=timezone.utc).strftime("model_%Y%m%dT%H%M%S%fZ")
        suffix = ".json" if hasattr(model, "save_model") else ".pkl"
        model_path = self.base_dir / f"{model_id}{suffix}"
        self._save_model(model, model_path)

        record = ModelRecord(
            model_id=model_id,
            model_path=_serialize_model_reference(self.base_dir, model_path),
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            train_start=fechas["train_start"],
            train_end=fechas["train_end"],
            val_start=fechas["val_start"],
            val_end=fechas["val_end"],
            metrics=metrics,
            promoted=False,
            feature_names=list(feature_names),
            n_features=len(feature_names),
        )

        records = self._load_records()
        records.append(record)
        self._write_registry(records)
        return record

    def try_promote(self, record: ModelRecord) -> bool:
        """Promueve si supera el minimo y mejora al activo."""
        if not self._passes_promotion_floor(record):
            return False

        active = self._get_active_record()
        if active is None:
            return self._promote(record)

        auc_improvement = record.metrics.auc_val - active.metrics.auc_val
        sharpe_ok = record.metrics.sharpe >= max(settings.MIN_MODEL_SHARPE_FOR_PROMOTION, active.metrics.sharpe)
        drawdown_ok = record.metrics.max_drawdown <= settings.MAX_MODEL_DRAWDOWN_FOR_PROMOTION
        if (
            auc_improvement >= settings.MIN_MODEL_AUC_IMPROVEMENT
            and sharpe_ok
            and drawdown_ok
        ):
            return self._promote(record)
        return False

    def get_active_model_path(self) -> Optional[str]:
        """Retorna la ruta del modelo activo, si existe."""
        active = self._get_active_record()
        return active.model_path if active else None

    def get_active_feature_names(self) -> Optional[list[str]]:
        """Retorna el orden de features del modelo activo."""
        active = self._get_active_record()
        return active.feature_names if active else None

    def has_active_model(self) -> bool:
        """Indica si ya existe un modelo activo."""
        return self._get_active_record() is not None

    def print_summary(self) -> None:
        """Imprime el historial de modelos."""
        for record in self._load_records():
            status = "ACTIVE" if record.promoted else "CANDIDATE"
            print(
                f"{record.model_id} {status} "
                f"auc={record.metrics.auc_val:.4f} sharpe={record.metrics.sharpe:.3f} "
                f"features={record.n_features}"
            )

    def _promote(self, record: ModelRecord) -> bool:
        source = resolve_model_artifact_path(self.base_dir, record.model_path)
        if source is None or not source.exists():
            return False

        records = self._load_records()
        updated_records: list[ModelRecord] = []
        for existing in records:
            updated_records.append(
                ModelRecord(
                    model_id=existing.model_id,
                    model_path=existing.model_path,
                    created_at=existing.created_at,
                    train_start=existing.train_start,
                    train_end=existing.train_end,
                    val_start=existing.val_start,
                    val_end=existing.val_end,
                    metrics=existing.metrics,
                    promoted=existing.model_id == record.model_id,
                    feature_names=existing.feature_names,
                    n_features=existing.n_features,
                )
            )
        self._write_registry(updated_records)
        active_path = self.base_dir / f"active_model{source.suffix}"
        shutil.copyfile(source, active_path)
        self._atomic_write_text(
            self.active_meta_path,
            json.dumps(
                {
                    "model_id": record.model_id,
                    "model_path": _serialize_model_reference(self.base_dir, active_path),
                    "feature_names": record.feature_names,
                    "signal_contract": settings.SIGNAL_CONTRACT,
                    "artifact_sha256": compute_artifact_sha256(active_path),
                },
                indent=2,
            ),
        )
        return True

    def _get_active_record(self) -> Optional[ModelRecord]:
        for record in reversed(self._load_records()):
            if record.promoted:
                if self.active_meta_path.exists():
                    meta = json.loads(self.active_meta_path.read_text(encoding="utf-8"))
                    resolved_path = resolve_model_artifact_path(
                        self.base_dir,
                        meta.get("model_path", record.model_path),
                    )
                    record = ModelRecord(
                        model_id=record.model_id,
                        model_path=str(resolved_path) if resolved_path is not None else record.model_path,
                        created_at=record.created_at,
                        train_start=record.train_start,
                        train_end=record.train_end,
                        val_start=record.val_start,
                        val_end=record.val_end,
                        metrics=record.metrics,
                        promoted=True,
                        feature_names=meta.get("feature_names", record.feature_names),
                        n_features=len(meta.get("feature_names", record.feature_names)),
                    )
                else:
                    resolved_path = resolve_model_artifact_path(self.base_dir, record.model_path)
                    record = ModelRecord(
                        model_id=record.model_id,
                        model_path=str(resolved_path) if resolved_path is not None else record.model_path,
                        created_at=record.created_at,
                        train_start=record.train_start,
                        train_end=record.train_end,
                        val_start=record.val_start,
                        val_end=record.val_end,
                        metrics=record.metrics,
                        promoted=True,
                        feature_names=record.feature_names,
                        n_features=record.n_features,
                    )
                return record
        return None

    def _passes_promotion_floor(self, record: ModelRecord) -> bool:
        """Evita promociones por AUC aislada con perfil de trading debil."""
        standard_floor = (
            record.metrics.auc_val >= settings.MIN_MODEL_AUC
            and record.metrics.sharpe >= settings.MIN_MODEL_SHARPE_FOR_PROMOTION
            and record.metrics.max_drawdown <= settings.MAX_MODEL_DRAWDOWN_FOR_PROMOTION
            and record.metrics.entry_ready
            and record.metrics.buy_support >= settings.PROMOTION_MIN_BUY_SUPPORT
        )
        # Track selectivo para modelos con threshold alto y pocas entradas, pero muy limpias.
        selective_floor = (
            record.metrics.buy_threshold >= settings.SELECTIVE_PROMOTION_MIN_THRESHOLD
            and record.metrics.auc_val >= settings.SELECTIVE_PROMOTION_MIN_AUC
            and record.metrics.sharpe >= settings.SELECTIVE_PROMOTION_MIN_SHARPE
            and record.metrics.precision_buy >= settings.SELECTIVE_PROMOTION_MIN_PRECISION
            and record.metrics.max_drawdown <= settings.SELECTIVE_PROMOTION_MAX_DRAWDOWN
            and record.metrics.buy_support >= settings.SELECTIVE_PROMOTION_MIN_SUPPORT
        )
        return standard_floor or selective_floor

    def _load_records(self) -> list[ModelRecord]:
        return self.load_records_from_path(self.registry_path)

    def _write_registry(self, records: list[ModelRecord]) -> None:
        serializable = []
        for record in records:
            payload = asdict(record)
            payload["metrics"] = asdict(record.metrics)
            serializable.append(payload)
        self._atomic_write_text(self.registry_path, json.dumps(serializable, indent=2))

    @staticmethod
    def _save_model(model: object, model_path: Path) -> None:
        if hasattr(model, "save_model"):
            model.save_model(str(model_path))
            return
        import joblib

        joblib.dump(model, model_path)

    @staticmethod
    def load_records_from_path(registry_path: str | Path) -> list[ModelRecord]:
        """Carga registros persistidos desde cualquier registry.json."""
        path = Path(registry_path)
        if not path.exists():
            return []

        raw_records = json.loads(path.read_text(encoding="utf-8"))
        records: list[ModelRecord] = []
        for item in raw_records:
            metrics = ModelMetrics(**item["metrics"])
            records.append(
                ModelRecord(
                    model_id=item["model_id"],
                    model_path=item["model_path"],
                    created_at=item["created_at"],
                    train_start=item["train_start"],
                    train_end=item["train_end"],
                    val_start=item["val_start"],
                    val_end=item["val_end"],
                    metrics=metrics,
                    promoted=bool(item["promoted"]),
                    feature_names=list(item["feature_names"]),
                    n_features=int(item["n_features"]),
                )
            )
        return records

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, path)


class MultiAssetModelRegistry:
    """Resuelve modelos por activo base y separa observacion de ejecucion."""

    def __init__(
        self,
        root_dir: str | Path = settings.MODEL_REGISTRY_ROOT,
        execution_allowed_bases: list[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.execution_allowed_bases = {
            base.strip().upper()
            for base in (execution_allowed_bases or settings.EXECUTION_ALLOWED_BASES)
            if base.strip()
        }
        self._artifacts_by_registry_key: dict[str, ResolvedModelArtifact] = {}
        self._registry_key_by_base: dict[str, str] = {}
        self.refresh()

    def refresh(self) -> None:
        """Reescanea el root para detectar promociones o modelos nuevos."""
        artifacts: dict[str, ResolvedModelArtifact] = {}
        aliases: dict[str, str] = {}
        if not self.root_dir.exists():
            self._artifacts_by_registry_key = {}
            self._registry_key_by_base = {}
            return

        for asset_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            artifact = self._build_artifact(asset_dir)
            if artifact is None:
                continue
            artifacts[artifact.registry_key] = artifact
            aliases[artifact.base_asset] = artifact.registry_key

        self._artifacts_by_registry_key = artifacts
        self._registry_key_by_base = aliases

    def resolve_artifact(self, product_id: str) -> ResolvedModelArtifact | None:
        """Retorna el artefacto asociado a un product_id live."""
        registry_key = self._resolve_registry_key(product_id)
        if registry_key is None:
            return None
        return self._artifacts_by_registry_key.get(registry_key)

    def get_registry_keys(self) -> list[str]:
        """Lista los assets conocidos por el registry root."""
        return sorted(self._artifacts_by_registry_key)

    def get_actionable_registry_keys(self) -> list[str]:
        """Lista los assets con modelo promovido y habilitados para ejecucion."""
        return sorted(
            artifact.registry_key
            for artifact in self._artifacts_by_registry_key.values()
            if artifact.actionable
        )

    def _build_artifact(self, asset_dir: Path) -> ResolvedModelArtifact | None:
        registry_key = asset_dir.name.lower()
        base_asset = registry_key.split("_", 1)[0].upper()
        active_meta_path = asset_dir / "active_model_meta.json"
        registry_path = asset_dir / "registry.json"

        if active_meta_path.exists():
            meta = self._safe_read_json(active_meta_path)
            resolved_path = resolve_model_artifact_path(asset_dir, str(meta.get("model_path", "")))
            feature_names = list(meta.get("feature_names", []))
            model_id = str(meta.get("model_id", ""))
            path_exists = resolved_path is not None and resolved_path.exists()
            meta_valid = bool(path_exists and feature_names and model_id)
            expected_sha256 = str(meta.get("artifact_sha256", "")).strip()
            hash_ok = True
            if path_exists and expected_sha256:
                hash_ok = compute_artifact_sha256(resolved_path) == expected_sha256
            fingerprint = self._fingerprint_for_paths(
                [active_meta_path] + ([resolved_path] if resolved_path is not None else [])
            )
            if not path_exists:
                reason = "broken_model_path"
            elif not hash_ok:
                reason = "artifact_hash_mismatch"
            elif not feature_names:
                reason = "invalid_active_meta"
            elif base_asset not in self.execution_allowed_bases:
                reason = "base_not_enabled"
            else:
                reason = "ok"
            return ResolvedModelArtifact(
                registry_key=registry_key,
                base_asset=base_asset,
                model_id=model_id,
                model_path=str(resolved_path) if resolved_path is not None else None,
                feature_names=feature_names,
                actionable=meta_valid and hash_ok and base_asset in self.execution_allowed_bases,
                reason=reason,
                fingerprint=fingerprint,
            )

        records = ModelRegistry.load_records_from_path(registry_path)
        if records:
            latest = records[-1]
            resolved_path = resolve_model_artifact_path(asset_dir, latest.model_path)
            fingerprint = self._fingerprint_for_paths(
                [registry_path] + ([resolved_path] if resolved_path is not None else [])
            )
            return ResolvedModelArtifact(
                registry_key=registry_key,
                base_asset=base_asset,
                model_id=latest.model_id,
                model_path=str(resolved_path) if resolved_path is not None and resolved_path.exists() else None,
                feature_names=list(latest.feature_names),
                actionable=False,
                reason="no_promoted_model",
                fingerprint=fingerprint,
            )

        return ResolvedModelArtifact(
            registry_key=registry_key,
            base_asset=base_asset,
            model_id="",
            model_path=None,
            feature_names=[],
            actionable=False,
            reason="no_model",
            fingerprint=self._fingerprint_for_paths([asset_dir]),
        )

    def _resolve_registry_key(self, product_id: str) -> str | None:
        normalized = product_id.strip().lower().replace("-", "_")
        if normalized in self._artifacts_by_registry_key:
            return normalized

        base_asset = product_id.strip().split("-", 1)[0].upper()
        return self._registry_key_by_base.get(base_asset)

    @staticmethod
    def _safe_read_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _fingerprint_for_paths(paths: list[Path]) -> str:
        parts: list[str] = []
        for path in paths:
            if not path.exists():
                parts.append(f"{path}:missing")
                continue
            stat = path.stat()
            parts.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
        return "|".join(parts)
