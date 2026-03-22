"""Bundle serializable con ensemble, calibracion y guardas de runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from config import settings


@dataclass(slots=True)
class DriftEvaluation:
    """Resumen del monitoreo de drift para una fila de inferencia."""

    status: str
    actionable: bool
    reason: str
    outlier_ratio: float
    outlier_count: int
    checked_features: int


@dataclass(slots=True)
class RegimeEvaluation:
    """Decision del detector de regimen."""

    regime: str
    actionable: bool
    reason: str
    backend: str


@dataclass(slots=True)
class ThresholdSelectionResult:
    """Thresholds elegidos con una heuristica long-only."""

    buy_threshold: float
    exit_threshold: float
    buy_score: float
    exit_score: float
    buy_support: int
    exit_support: int
    buy_precision: float = 0.0
    entry_ready: bool = False


@dataclass(slots=True)
class ProbabilityCalibrator:
    """Calibra probabilidades de los boosters con metodo automatico."""

    method: str = "identity"
    isotonic_model: IsotonicRegression | None = None
    logistic_model: LogisticRegression | None = None

    @classmethod
    def fit(cls, raw_probs: np.ndarray, y_true: np.ndarray) -> ProbabilityCalibrator:
        """Ajusta un calibrador sencillo evitando sobrecomplicar el pipeline."""
        probs = np.clip(np.asarray(raw_probs, dtype=float).reshape(-1), 1e-6, 1.0 - 1e-6)
        target = np.asarray(y_true, dtype=int).reshape(-1)
        if len(probs) == 0 or len(np.unique(target)) < 2:
            return cls(method="identity")

        unique_probs = np.unique(np.round(probs, 6))
        if len(unique_probs) >= max(12, min(50, len(probs) // 8)):
            isotonic = IsotonicRegression(out_of_bounds="clip")
            isotonic.fit(probs, target)
            calibrator = cls(method="isotonic", isotonic_model=isotonic)
            if not calibrator.is_degenerate(probs):
                return calibrator

        logits = np.log(probs / (1.0 - probs)).reshape(-1, 1)
        logistic = LogisticRegression(max_iter=500, solver="lbfgs")
        logistic.fit(logits, target)
        calibrator = cls(method="platt", logistic_model=logistic)
        if not calibrator.is_degenerate(probs):
            return calibrator
        return cls(method="identity")

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """Convierte probabilidades crudas en probabilidades mas confiables."""
        probs = np.clip(np.asarray(raw_probs, dtype=float).reshape(-1), 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic" and self.isotonic_model is not None:
            calibrated = np.asarray(self.isotonic_model.predict(probs), dtype=float)
            return np.clip(calibrated, 1e-6, 1.0 - 1e-6)
        if self.method == "platt" and self.logistic_model is not None:
            logits = np.log(probs / (1.0 - probs)).reshape(-1, 1)
            calibrated = np.asarray(self.logistic_model.predict_proba(logits), dtype=float)[:, 1]
            return np.clip(calibrated, 1e-6, 1.0 - 1e-6)
        return probs

    def is_degenerate(self, raw_probs: np.ndarray) -> bool:
        """Detecta calibraciones casi constantes que no agregan informacion."""
        calibrated = self.transform(raw_probs)
        return float(np.std(calibrated)) < 0.01


@dataclass(slots=True)
class FeatureDriftMonitor:
    """Guarda percentiles de entrenamiento para bloquear drift severo."""

    bounds_by_feature: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    block_ratio: float = settings.DRIFT_BLOCK_THRESHOLD_RATIO
    margin_pct: float = settings.DRIFT_BOUND_MARGIN_PCT

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> FeatureDriftMonitor:
        """Persistencia liviana de percentiles por feature."""
        selected = feature_names or list(frame.columns)
        bounds_by_feature: dict[str, tuple[float, float, float]] = {}
        for name in selected:
            if name not in frame.columns:
                continue
            values = pd.Series(frame[name], dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            low = float(values.quantile(0.05))
            median = float(values.quantile(0.50))
            high = float(values.quantile(0.95))
            bounds_by_feature[name] = (low, median, high)
        return cls(bounds_by_feature=bounds_by_feature)

    def evaluate_payload(
        self,
        payload: dict[str, Any],
        feature_names: list[str] | None = None,
    ) -> DriftEvaluation:
        """Marca outliers respecto al rango observado en entrenamiento."""
        if not settings.DRIFT_MONITOR_ENABLED or not self.bounds_by_feature:
            return DriftEvaluation(
                status="disabled",
                actionable=True,
                reason="drift_monitor_disabled",
                outlier_ratio=0.0,
                outlier_count=0,
                checked_features=0,
            )

        selected = feature_names or list(self.bounds_by_feature)
        checked = 0
        outliers = 0
        for feature_name in selected:
            if feature_name not in self.bounds_by_feature or feature_name not in payload:
                continue
            try:
                value = float(payload[feature_name])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            low, median, high = self.bounds_by_feature[feature_name]
            scale = max(abs(high - low), abs(median) * self.margin_pct, 1e-9)
            margin = scale * self.margin_pct
            checked += 1
            if value < (low - margin) or value > (high + margin):
                outliers += 1

        if checked < settings.DRIFT_MIN_FEATURE_CHECKS:
            return DriftEvaluation(
                status="insufficient_checks",
                actionable=True,
                reason="drift_monitor_insufficient_checks",
                outlier_ratio=0.0,
                outlier_count=outliers,
                checked_features=checked,
            )

        outlier_ratio = outliers / float(checked)
        if outlier_ratio >= self.block_ratio:
            return DriftEvaluation(
                status="out_of_distribution",
                actionable=False,
                reason="drift_blocked_out_of_distribution",
                outlier_ratio=outlier_ratio,
                outlier_count=outliers,
                checked_features=checked,
            )
        return DriftEvaluation(
            status="ok",
            actionable=True,
            reason="ok",
            outlier_ratio=outlier_ratio,
            outlier_count=outliers,
            checked_features=checked,
        )


@dataclass(slots=True)
class MarketRegimeModel:
    """Detector de regimen con HMM si esta disponible y fallback heuristico."""

    backend: str = "heuristic"
    feature_names: list[str] = field(
        default_factory=lambda: [
            "realized_vol_5",
            "bb_width",
            "range_compression_20",
            "volume_ratio",
            "trend_slope_pct",
        ]
    )
    state_labels: dict[int, str] = field(default_factory=dict)
    state_actionable: dict[int, bool] = field(default_factory=dict)
    mean_: list[float] = field(default_factory=list)
    scale_: list[float] = field(default_factory=list)
    model: object | None = None
    heuristic_rules: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> MarketRegimeModel:
        """Aprende un detector de regimen usando solo history pre-holdout."""
        feature_names = [
            name
            for name in [
                "realized_vol_5",
                "bb_width",
                "range_compression_20",
                "volume_ratio",
                "trend_slope_pct",
            ]
            if name in frame.columns
        ]
        prepared = (
            frame[feature_names]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .reset_index(drop=True)
        )
        if len(prepared) < settings.REGIME_MODEL_MIN_ROWS:
            return cls._fit_heuristic(prepared, feature_names)

        raw = prepared.to_numpy(dtype=float)
        mean_ = raw.mean(axis=0)
        scale_ = raw.std(axis=0)
        scale_[scale_ == 0.0] = 1.0
        scaled = (raw - mean_) / scale_

        if settings.REGIME_HMM_ENABLED:
            try:
                from hmmlearn.hmm import GaussianHMM  # type: ignore

                n_components = min(settings.REGIME_HMM_STATES, max(2, len(prepared) // 250))
                hmm = GaussianHMM(
                    n_components=n_components,
                    covariance_type="diag",
                    n_iter=200,
                    random_state=42,
                )
                hmm.fit(scaled)
                states = np.asarray(hmm.predict(scaled), dtype=int)
                labels, actionable = cls._classify_states(prepared, states)
                return cls(
                    backend="hmm",
                    feature_names=feature_names,
                    state_labels=labels,
                    state_actionable=actionable,
                    mean_=mean_.tolist(),
                    scale_=scale_.tolist(),
                    model=hmm,
                )
            except Exception:
                pass

        return cls._fit_heuristic(prepared, feature_names)

    @classmethod
    def _fit_heuristic(
        cls,
        prepared: pd.DataFrame,
        feature_names: list[str],
    ) -> MarketRegimeModel:
        """Fallback aprendido desde cuantiles del propio entrenamiento."""
        if prepared.empty:
            heuristic_rules = {
                "vol_extreme_threshold": settings.REGIME_VOL_EXTREME_MAX,
                "compression_low_threshold": settings.REGIME_RANGE_COMPRESSION_MIN,
                "bb_width_low_threshold": settings.REGIME_BB_WIDTH_MIN,
            }
        else:
            heuristic_rules = {
                "vol_extreme_threshold": float(prepared["realized_vol_5"].quantile(0.95))
                if "realized_vol_5" in prepared.columns
                else settings.REGIME_VOL_EXTREME_MAX,
                "compression_low_threshold": float(prepared["range_compression_20"].quantile(0.15))
                if "range_compression_20" in prepared.columns
                else settings.REGIME_RANGE_COMPRESSION_MIN,
                "bb_width_low_threshold": float(prepared["bb_width"].quantile(0.15))
                if "bb_width" in prepared.columns
                else settings.REGIME_BB_WIDTH_MIN,
            }
        return cls(
            backend="heuristic",
            feature_names=feature_names,
            heuristic_rules=heuristic_rules,
        )

    @staticmethod
    def _classify_states(
        prepared: pd.DataFrame,
        states: np.ndarray,
    ) -> tuple[dict[int, str], dict[int, bool]]:
        """Mapea estados ocultos a etiquetas operativas simples."""
        frame = prepared.copy()
        frame["state_id"] = states
        grouped = frame.groupby("state_id").mean(numeric_only=True)
        labels = {int(state_id): "normal" for state_id in grouped.index.tolist()}
        actionable = {int(state_id): True for state_id in grouped.index.tolist()}

        if grouped.empty:
            return labels, actionable

        if "realized_vol_5" in grouped.columns:
            extreme_state = int(grouped["realized_vol_5"].idxmax())
            labels[extreme_state] = "extreme_volatility"
            actionable[extreme_state] = False

        remaining = [state_id for state_id in grouped.index.tolist() if actionable[int(state_id)]]
        if remaining and "bb_width" in grouped.columns and "range_compression_20" in grouped.columns:
            compression_score = (
                grouped.loc[remaining, "bb_width"].rank(method="first")
                + grouped.loc[remaining, "range_compression_20"].rank(method="first")
            )
            compressed_state = int(compression_score.idxmin())
            labels[compressed_state] = "compressed"
            actionable[compressed_state] = False

        return labels, actionable

    def evaluate_payload(self, payload: dict[str, Any]) -> RegimeEvaluation:
        """Determina si el mercado actual es operable para el modelo."""
        if not settings.REGIME_GATE_ENABLED:
            return RegimeEvaluation(
                regime="disabled",
                actionable=True,
                reason="regime_gate_disabled",
                backend=self.backend,
            )

        row = []
        for feature_name in self.feature_names:
            try:
                value = float(payload.get(feature_name))
            except (TypeError, ValueError):
                value = float("nan")
            row.append(value)
        values = np.asarray(row, dtype=float)
        if np.isnan(values).any():
            return RegimeEvaluation(
                regime="unknown",
                actionable=True,
                reason="regime_insufficient_features",
                backend=self.backend,
            )

        if self.backend == "hmm" and self.model is not None and self.mean_ and self.scale_:
            mean_ = np.asarray(self.mean_, dtype=float)
            scale_ = np.asarray(self.scale_, dtype=float)
            scaled = ((values - mean_) / scale_).reshape(1, -1)
            state_id = int(self.model.predict(scaled)[0])
            regime = self.state_labels.get(state_id, "normal")
            actionable = self.state_actionable.get(state_id, True)
            reason = "ok" if actionable else f"regime_blocked_{regime}"
            return RegimeEvaluation(
                regime=regime,
                actionable=actionable,
                reason=reason,
                backend=self.backend,
            )

        vol_extreme_threshold = float(
            self.heuristic_rules.get("vol_extreme_threshold", settings.REGIME_VOL_EXTREME_MAX)
        )
        compression_low_threshold = float(
            self.heuristic_rules.get("compression_low_threshold", settings.REGIME_RANGE_COMPRESSION_MIN)
        )
        bb_width_low_threshold = float(
            self.heuristic_rules.get("bb_width_low_threshold", settings.REGIME_BB_WIDTH_MIN)
        )
        realized_vol_5 = self._safe_float(payload.get("realized_vol_5"))
        range_compression_20 = self._safe_float(payload.get("range_compression_20"))
        bb_width = self._safe_float(payload.get("bb_width"))
        if not np.isfinite(realized_vol_5) or not np.isfinite(range_compression_20) or not np.isfinite(bb_width):
            return RegimeEvaluation(
                regime="unknown",
                actionable=True,
                reason="regime_insufficient_features",
                backend=self.backend,
            )

        if realized_vol_5 >= vol_extreme_threshold:
            return RegimeEvaluation(
                regime="extreme_volatility",
                actionable=False,
                reason="regime_blocked_extreme_volatility",
                backend=self.backend,
            )
        if range_compression_20 <= compression_low_threshold and bb_width <= bb_width_low_threshold:
            return RegimeEvaluation(
                regime="compressed",
                actionable=False,
                reason="regime_blocked_compression",
                backend=self.backend,
            )
        return RegimeEvaluation(
            regime="normal",
            actionable=True,
            reason="ok",
            backend=self.backend,
        )

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")


class SignalThresholdOptimizer:
    """Optimiza thresholds por activo usando el tramo purgado de validacion."""

    @classmethod
    def optimize(
        cls,
        calibrated_probs: np.ndarray,
        target_returns: np.ndarray,
        y_true: np.ndarray,
    ) -> ThresholdSelectionResult:
        """Busca thresholds con heuristicas simples y soporte minimo."""
        probs = np.asarray(calibrated_probs, dtype=float).reshape(-1)
        returns = np.asarray(target_returns, dtype=float).reshape(-1)
        target = np.asarray(y_true, dtype=int).reshape(-1)
        if len(probs) == 0:
            default_buy = float(settings.MIN_SIGNAL_PROB)
            default_exit = float(max(0.05, 1.0 - settings.MIN_SIGNAL_PROB))
            return ThresholdSelectionResult(
                buy_threshold=default_buy,
                exit_threshold=default_exit,
                buy_score=0.0,
                exit_score=0.0,
                buy_support=0,
                exit_support=0,
                buy_precision=0.0,
                entry_ready=False,
            )
        min_support = max(
            int(settings.THRESHOLD_MIN_SUPPORT_ABS),
            int(len(probs) * settings.THRESHOLD_MIN_SUPPORT_PCT),
        )

        default_buy = float(settings.MIN_SIGNAL_PROB)
        default_exit = float(max(0.05, 1.0 - settings.MIN_SIGNAL_PROB))

        buy_candidates = np.unique(
            np.clip(
                np.concatenate(
                    [
                        np.linspace(0.45, 0.85, settings.THRESHOLD_GRID_SIZE),
                        np.quantile(probs, [0.45, 0.55, 0.65, 0.75, 0.85]),
                    ]
                ),
                0.45,
                0.90,
            )
        )
        exit_candidates = np.unique(
            np.clip(
                np.concatenate(
                    [
                        np.linspace(0.10, 0.45, settings.THRESHOLD_GRID_SIZE),
                        np.quantile(probs, [0.10, 0.20, 0.30, 0.40]),
                    ]
                ),
                0.05,
                0.49,
            )
        )

        best_buy_threshold = default_buy
        best_buy_score = float("-inf")
        best_buy_support = 0
        best_buy_precision = 0.0
        for threshold in buy_candidates.tolist():
            mask = probs >= threshold
            support = int(mask.sum())
            if support < min_support:
                continue
            selected_returns = returns[mask]
            mean_return = float(np.mean(selected_returns))
            std_return = float(np.std(selected_returns))
            sharpe = 0.0 if std_return == 0.0 else mean_return / std_return * np.sqrt(len(selected_returns))
            precision = float(np.mean(target[mask] == 1))
            if precision < settings.THRESHOLD_MIN_BUY_PRECISION:
                continue
            score = sharpe + precision + (mean_return * 100.0)
            if score > best_buy_score:
                best_buy_threshold = float(threshold)
                best_buy_score = score
                best_buy_support = support
                best_buy_precision = precision

        best_exit_threshold = default_exit
        best_exit_score = float("-inf")
        best_exit_support = 0
        for threshold in exit_candidates.tolist():
            mask = probs <= threshold
            support = int(mask.sum())
            if support < min_support:
                continue
            selected_returns = returns[mask]
            negative_capture = float(-np.mean(selected_returns))
            down_precision = float(np.mean(target[mask] == 0))
            score = down_precision + (negative_capture * 100.0)
            if score > best_exit_score:
                best_exit_threshold = float(threshold)
                best_exit_score = score
                best_exit_support = support

        if best_exit_threshold >= best_buy_threshold:
            best_exit_threshold = max(0.05, best_buy_threshold - 0.10)

        entry_ready = best_buy_support >= min_support and best_buy_precision >= settings.THRESHOLD_MIN_BUY_PRECISION
        if not entry_ready:
            best_buy_threshold = default_buy
            best_buy_score = 0.0
            fallback_mask = probs >= best_buy_threshold
            best_buy_support = int(fallback_mask.sum())
            best_buy_precision = (
                float(np.mean(target[fallback_mask] == 1))
                if best_buy_support > 0
                else 0.0
            )

        return ThresholdSelectionResult(
            buy_threshold=round(best_buy_threshold, 6),
            exit_threshold=round(best_exit_threshold, 6),
            buy_score=float(best_buy_score if np.isfinite(best_buy_score) else 0.0),
            exit_score=float(best_exit_score if np.isfinite(best_exit_score) else 0.0),
            buy_support=best_buy_support,
            exit_support=best_exit_support,
            buy_precision=round(best_buy_precision, 6),
            entry_ready=entry_ready,
        )


@dataclass(slots=True)
class SignalModelBundle:
    """Artefacto unico cargable por registry e inferencia."""

    primary_model: object
    secondary_model: object | None
    calibrator: ProbabilityCalibrator
    drift_monitor: FeatureDriftMonitor
    regime_model: MarketRegimeModel
    feature_names: list[str]
    buy_threshold: float
    exit_threshold: float
    primary_model_name: str = "xgboost"
    secondary_model_name: str | None = None
    signal_contract: str = settings.SIGNAL_CONTRACT
    calibration_method: str = "identity"
    threshold_metadata: dict[str, float | int] = field(default_factory=dict)
    bundle_version: str = "signal_bundle_v1"

    def predict_raw_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Promedia probabilidades de los boosters disponibles."""
        probabilities = [self._predict_single_model(self.primary_model, frame)]
        if self.secondary_model is not None:
            probabilities.append(self._predict_single_model(self.secondary_model, frame))
        stacked = np.vstack(probabilities)
        return np.mean(stacked, axis=0)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Entrega probabilidad calibrada en formato sklearn compatible."""
        calibrated = self.calibrator.transform(self.predict_raw_proba(frame))
        return np.column_stack([1.0 - calibrated, calibrated])

    def evaluate_drift_payload(self, payload: dict[str, Any]) -> DriftEvaluation:
        """Wrapper para inferencia."""
        return self.drift_monitor.evaluate_payload(payload, self.feature_names)

    def evaluate_regime_payload(self, payload: dict[str, Any]) -> RegimeEvaluation:
        """Wrapper para inferencia."""
        return self.regime_model.evaluate_payload(payload)

    @staticmethod
    def _predict_single_model(model: object, frame: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            probs_array = np.asarray(model.predict_proba(frame), dtype=float)
            return probs_array[:, 1] if probs_array.ndim == 2 else probs_array.reshape(-1)
        preds = np.asarray(model.predict(frame), dtype=float).reshape(-1)
        return np.clip(preds, 1e-6, 1.0 - 1e-6)


def build_lightgbm_classifier() -> object | None:
    """Crea el segundo booster solo si la libreria esta disponible."""
    if not settings.LIGHTGBM_ENABLED:
        return None
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        return LGBMClassifier(**settings.LGBM_PARAMS)
    except Exception:
        return None
