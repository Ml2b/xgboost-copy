"""Inferencia continua sobre features publicadas en Redis."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from model.registry import ModelRegistry, MultiAssetModelRegistry, ResolvedModelArtifact


@dataclass(slots=True)
class LoadedModelBundle:
    """Modelo cargado en memoria junto con su fingerprint actual."""

    artifact: ResolvedModelArtifact
    model: object


@dataclass(slots=True)
class RegimeDecision:
    """Resultado de la compuerta de regimen de mercado."""

    regime: str
    actionable: bool
    reason: str


class InferenceEngine:
    """Carga el modelo activo y emite senales BUY/EXIT_LONG/HOLD."""

    def __init__(
        self,
        registry: ModelRegistry | MultiAssetModelRegistry | Any,
        redis_client: Any | None = None,
        buy_thresholds_by_base: dict[str, float] | None = None,
        sell_thresholds_by_base: dict[str, float] | None = None,
        regime_gate_enabled: bool = settings.REGIME_GATE_ENABLED,
        regime_vol_extreme_max: float = settings.REGIME_VOL_EXTREME_MAX,
        regime_range_compression_min: float = settings.REGIME_RANGE_COMPRESSION_MIN,
        regime_bb_width_min: float = settings.REGIME_BB_WIDTH_MIN,
    ) -> None:
        self.registry = registry
        self.redis_client = redis_client
        self.model: object | None = None
        self.model_path: str | None = None
        self.feature_names: list[str] = []
        self.model_id: str | None = None
        self.registry_key: str | None = None
        self.total_inferences = 0
        self.total_signals = 0
        self.latency_total_ms = 0.0
        self.observation_events = 0
        self._loaded_models: dict[str, LoadedModelBundle] = {}
        self._last_runtime_log_at = 0.0
        self.buy_thresholds_by_base = {
            base.strip().upper(): float(value)
            for base, value in (buy_thresholds_by_base or {}).items()
            if base.strip()
        }
        self.sell_thresholds_by_base = {
            base.strip().upper(): float(value)
            for base, value in (sell_thresholds_by_base or {}).items()
            if base.strip()
        }
        self.regime_gate_enabled = bool(regime_gate_enabled)
        self.regime_vol_extreme_max = float(regime_vol_extreme_max)
        self.regime_range_compression_min = float(regime_range_compression_min)
        self.regime_bb_width_min = float(regime_bb_width_min)
        self.load_active_model()

    def load_active_model(self) -> None:
        """Carga el modelo promovido mas reciente si existe."""
        if hasattr(self.registry, "refresh") and hasattr(self.registry, "resolve_artifact"):
            self.registry.refresh()
            self._drop_stale_cached_models()
            return

        model_path = self.registry.get_active_model_path()
        feature_names = self.registry.get_active_feature_names()
        if not model_path or not feature_names or not Path(model_path).exists():
            self.model = None
            self.model_path = None
            self.feature_names = []
            self.registry_key = None
            return

        if model_path == self.model_path:
            return

        self.model = self._load_model(model_path)
        self.model_path = model_path
        self.feature_names = feature_names
        self.model_id = Path(model_path).stem
        self.registry_key = "legacy"

    async def start(self, stop_event: asyncio.Event | None = None) -> None:
        """Consume market.features y publica inference.signals."""
        stop_event = stop_event or asyncio.Event()
        logger.info("InferenceEngine iniciado. model_reload_interval={}s", settings.MODEL_RELOAD_INTERVAL)
        reload_task = asyncio.create_task(self._reload_loop(stop_event))
        try:
            if self.redis_client is None:
                await stop_event.wait()
                return
            await self._ensure_group(settings.STREAM_MARKET_FEATURES, "inference-engine")
            while not stop_event.is_set():
                try:
                    messages = await self.redis_client.xreadgroup(
                        groupname="inference-engine",
                        consumername="inference-1",
                        streams={settings.STREAM_MARKET_FEATURES: ">"},
                        count=25,
                        block=1000,
                    )
                except Exception as exc:
                    if "NOGROUP" in str(exc):
                        logger.warning("InferenceEngine: NOGROUP detectado, recreando consumer group")
                        await self._ensure_group(settings.STREAM_MARKET_FEATURES, "inference-engine")
                        continue
                    raise
                for _, stream_messages in messages:
                    for message_id, payload in stream_messages:
                        result = self.predict_signal(payload)
                        if result is not None:
                            await self.redis_client.xadd(
                                settings.STREAM_INFERENCE_SIGNALS,
                                {key: str(value) for key, value in result.items()},
                                maxlen=50000,
                                approximate=True,
                            )
                            self.total_signals += int(result["signal"] != "HOLD")
                        await self.redis_client.xack(settings.STREAM_MARKET_FEATURES, "inference-engine", message_id)
        finally:
            reload_task.cancel()

    async def _reload_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            self.load_active_model()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=settings.MODEL_RELOAD_INTERVAL)
            except asyncio.TimeoutError:
                continue

    def predict_signal(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Reconstruye el vector en orden y decide la senal."""
        product_id = str(payload.get("product_id", "")).strip()
        bundle = self._get_loaded_bundle(product_id)
        if bundle is None:
            self.observation_events += 1
            self._log_runtime_summary()
            return self._build_observation_result(
                product_id=product_id,
                model_id="",
                registry_key="",
                reason="no_model",
            )

        started = time.perf_counter()
        row = [float(payload.get(name, 0.0)) for name in bundle.artifact.feature_names]
        frame = pd.DataFrame([row], columns=bundle.artifact.feature_names)
        prob_buy = float(self._predict_proba(bundle.model, frame)[0])
        buy_threshold, sell_threshold = self._resolve_signal_thresholds(product_id)
        regime = self._evaluate_regime(payload)
        effective_actionable = bundle.artifact.actionable and regime.actionable

        signal = "HOLD"
        if prob_buy >= buy_threshold:
            signal = "BUY"
        elif prob_buy <= sell_threshold:
            signal = "EXIT_LONG"
        if signal != "HOLD" and not regime.actionable:
            signal = "HOLD"

        reason = bundle.artifact.reason
        if bundle.artifact.actionable and not regime.actionable:
            reason = regime.reason

        latency_ms = (time.perf_counter() - started) * 1000.0
        self.total_inferences += 1
        self.latency_total_ms += latency_ms
        self._log_runtime_summary()

        return {
            "product_id": product_id,
            "signal": signal,
            "prob_buy": round(prob_buy, 6),
            "model_id": bundle.artifact.model_id,
            "latency_ms": round(latency_ms, 3),
            "actionable": str(effective_actionable).lower(),
            "reason": reason,
            "registry_key": bundle.artifact.registry_key,
            "buy_threshold": round(buy_threshold, 6),
            "exit_threshold": round(sell_threshold, 6),
            "sell_threshold": round(sell_threshold, 6),
            "regime": regime.regime,
            "regime_actionable": str(regime.actionable).lower(),
            "signal_contract": settings.SIGNAL_CONTRACT,
        }

    def _predict_proba(self, model: object, frame: pd.DataFrame) -> np.ndarray:
        if model is None:
            raise RuntimeError("No hay modelo activo cargado.")
        if hasattr(model, "predict_proba"):
            probs_array = np.asarray(model.predict_proba(frame), dtype=float)
            return probs_array[:, 1] if probs_array.ndim == 2 else probs_array
        preds = model.predict(frame)
        return np.asarray(preds, dtype=float)

    @staticmethod
    def _load_model(model_path: str) -> object:
        path = Path(model_path)
        if path.suffix == ".json":
            try:
                from xgboost import XGBClassifier  # type: ignore

                model = XGBClassifier()
                model.load_model(str(path))
                return model
            except ImportError as exc:
                raise RuntimeError("No se pudo cargar el modelo XGBoost sin xgboost instalado.") from exc

        import joblib

        return joblib.load(path)

    async def _ensure_group(self, stream: str, group_name: str) -> None:
        try:
            await self.redis_client.xgroup_create(stream, group_name, id="$", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def _get_loaded_bundle(self, product_id: str) -> LoadedModelBundle | None:
        artifact = self._resolve_artifact(product_id)
        if artifact is None:
            return None

        if not artifact.model_path or not artifact.feature_names:
            return LoadedModelBundle(artifact=artifact, model=self._NoOpModel())

        cached = self._loaded_models.get(artifact.registry_key)
        if cached is not None and cached.artifact.fingerprint == artifact.fingerprint:
            return cached

        loaded = LoadedModelBundle(
            artifact=artifact,
            model=self._load_model(artifact.model_path),
        )
        self._loaded_models[artifact.registry_key] = loaded
        logger.info(
            "InferenceEngine cargo modelo. registry_key={} model_id={} actionable={} features={}",
            artifact.registry_key,
            artifact.model_id,
            artifact.actionable,
            len(artifact.feature_names),
        )
        if artifact.actionable:
            self.model = loaded.model
            self.model_path = artifact.model_path
            self.feature_names = list(artifact.feature_names)
            self.model_id = artifact.model_id
            self.registry_key = artifact.registry_key
        return loaded

    def _resolve_artifact(self, product_id: str) -> ResolvedModelArtifact | None:
        if hasattr(self.registry, "resolve_artifact"):
            artifact = self.registry.resolve_artifact(product_id)
            if artifact is not None:
                return artifact
            normalized = product_id.strip().split("-", 1)[0].upper() if product_id else ""
            return ResolvedModelArtifact(
                registry_key=normalized.lower(),
                base_asset=normalized,
                model_id="",
                model_path=None,
                feature_names=[],
                actionable=False,
                reason="no_model",
                fingerprint=f"missing:{normalized}",
            )

        model_path = self.registry.get_active_model_path()
        feature_names = self.registry.get_active_feature_names()
        if not model_path or not feature_names or not Path(model_path).exists():
            return None
        return ResolvedModelArtifact(
            registry_key="legacy",
            base_asset=product_id.split("-", 1)[0].upper() if product_id else "",
            model_id=Path(model_path).stem,
            model_path=model_path,
            feature_names=list(feature_names),
            actionable=True,
            reason="ok",
            fingerprint=f"{model_path}:{Path(model_path).stat().st_mtime_ns}",
        )

    def _drop_stale_cached_models(self) -> None:
        if not hasattr(self.registry, "get_registry_keys"):
            return
        valid_keys = set(self.registry.get_registry_keys())
        stale_keys = [key for key in self._loaded_models if key not in valid_keys]
        for key in stale_keys:
            self._loaded_models.pop(key, None)

    def _resolve_signal_thresholds(self, product_id: str) -> tuple[float, float]:
        base_asset = product_id.strip().split("-", 1)[0].upper() if product_id else ""
        buy_threshold = float(self.buy_thresholds_by_base.get(base_asset, settings.MIN_SIGNAL_PROB))
        sell_threshold = float(
            self.sell_thresholds_by_base.get(
                base_asset,
                1.0 - settings.MIN_SIGNAL_PROB,
            )
        )
        buy_threshold = min(max(buy_threshold, 0.0), 1.0)
        sell_threshold = min(max(sell_threshold, 0.0), 1.0)
        if sell_threshold > buy_threshold:
            sell_threshold = buy_threshold
        return buy_threshold, sell_threshold

    def _evaluate_regime(self, payload: dict[str, Any]) -> RegimeDecision:
        """Bloquea ejecucion en regimenes extremos definidos por heuristicas."""
        if not self.regime_gate_enabled:
            return RegimeDecision(regime="disabled", actionable=True, reason="regime_gate_disabled")

        realized_vol_5 = self._to_float(payload.get("realized_vol_5"))
        range_compression_20 = self._to_float(payload.get("range_compression_20"))
        bb_width = self._to_float(payload.get("bb_width"))
        if not np.isfinite(realized_vol_5) or not np.isfinite(range_compression_20) or not np.isfinite(bb_width):
            return RegimeDecision(regime="unknown", actionable=True, reason="regime_insufficient_features")

        if realized_vol_5 >= self.regime_vol_extreme_max:
            return RegimeDecision(
                regime="extreme_volatility",
                actionable=False,
                reason="regime_blocked_extreme_volatility",
            )

        if (
            range_compression_20 <= self.regime_range_compression_min
            and bb_width <= self.regime_bb_width_min
        ):
            return RegimeDecision(
                regime="compressed",
                actionable=False,
                reason="regime_blocked_compression",
            )

        return RegimeDecision(regime="normal", actionable=True, reason="ok")

    def _log_runtime_summary(self) -> None:
        """Resume inferencia para los logs de Render."""
        now = time.time()
        if now - self._last_runtime_log_at < settings.RUNTIME_LOG_INTERVAL_SECONDS:
            return
        self._last_runtime_log_at = now
        avg_latency = 0.0
        if self.total_inferences > 0:
            avg_latency = self.latency_total_ms / self.total_inferences
        logger.info(
            "InferenceEngine resumen: total={} non_hold={} observations={} avg_latency_ms={} cached_models={}",
            self.total_inferences,
            self.total_signals,
            self.observation_events,
            round(avg_latency, 3),
            len(self._loaded_models),
        )

    @staticmethod
    def _build_observation_result(
        product_id: str,
        model_id: str,
        registry_key: str,
        reason: str,
    ) -> dict[str, Any]:
        buy_threshold = round(settings.MIN_SIGNAL_PROB, 6)
        sell_threshold = round(1.0 - settings.MIN_SIGNAL_PROB, 6)
        return {
            "product_id": product_id,
            "signal": "HOLD",
            "prob_buy": 0.5,
            "model_id": model_id,
            "latency_ms": 0.0,
            "actionable": "false",
            "reason": reason,
            "registry_key": registry_key,
            "buy_threshold": buy_threshold,
            "exit_threshold": sell_threshold,
            "sell_threshold": sell_threshold,
            "regime": "no_model",
            "regime_actionable": "false",
            "signal_contract": settings.SIGNAL_CONTRACT,
        }

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    class _NoOpModel:
        """Modelo neutro para publicar observacion sin huecos silenciosos."""

        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            return np.asarray([[0.5, 0.5]] * len(frame), dtype=float)
