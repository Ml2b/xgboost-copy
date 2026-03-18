"""Seleccion de features dentro de cada fold temporal."""

from __future__ import annotations

import copy
import inspect
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(slots=True)
class SelectorConfig:
    """Parametros heuristicos del selector."""

    correlation_threshold: float = 0.85
    min_shap_importance_pct: float = 0.005
    min_permutation_importance: float = 0.0
    min_fold_stability: float = 0.6
    max_features: int = 25
    verbose: bool = True


@dataclass(slots=True)
class FoldSelectionResult:
    """Resultado de un fold de seleccion."""

    fold_id: int
    features_input: list[str]
    features_after_corr: list[str]
    features_final: list[str]
    shap_importance: dict[str, float]
    perm_importance: dict[str, float]
    correlation_removed: list[str]
    weak_removed: list[str]
    n_features_final: int


class FeatureSelector:
    """Aplica SHAP, permutation importance y filtro de correlacion."""

    def __init__(self, config: SelectorConfig | None = None) -> None:
        self.config = config or SelectorConfig()
        self.results_: list[FoldSelectionResult] = []

    def select(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        model: object,
        fold_id: int,
    ) -> FoldSelectionResult:
        """Selecciona features relevantes dentro del fold actual."""
        features_input = list(X_train.columns)
        shap_importance = self._compute_shap_importance(model, X_val)
        perm_importance = self._compute_permutation_importance(model, X_val, y_val)

        features_after_corr, correlation_removed = self._apply_correlation_filter(
            X_train,
            shap_importance,
        )

        total_shap = sum(shap_importance.values()) or 1.0
        weak_removed: list[str] = []
        features_final: list[str] = []
        for feature in features_after_corr:
            shap_share = shap_importance.get(feature, 0.0) / total_shap
            perm_score = perm_importance.get(feature, 0.0)
            is_weak_shap = shap_share < self.config.min_shap_importance_pct
            is_weak_perm = perm_score <= self.config.min_permutation_importance
            if is_weak_shap and is_weak_perm:
                weak_removed.append(feature)
            else:
                features_final.append(feature)

        if not features_final:
            ranked = sorted(features_after_corr, key=lambda name: shap_importance.get(name, 0.0), reverse=True)
            features_final = ranked[: max(1, min(self.config.max_features, len(ranked)))]
            weak_removed = [name for name in features_after_corr if name not in features_final]

        if self.config.max_features and len(features_final) > self.config.max_features:
            features_final = sorted(
                features_final,
                key=lambda name: shap_importance.get(name, 0.0),
                reverse=True,
            )[: self.config.max_features]

        result = FoldSelectionResult(
            fold_id=fold_id,
            features_input=features_input,
            features_after_corr=features_after_corr,
            features_final=features_final,
            shap_importance=shap_importance,
            perm_importance=perm_importance,
            correlation_removed=correlation_removed,
            weak_removed=weak_removed,
            n_features_final=len(features_final),
        )
        self.results_.append(result)

        if self.config.verbose:
            print(
                f"[FeatureSelector] fold={fold_id} input={len(features_input)} "
                f"corr={len(correlation_removed)} weak={len(weak_removed)} "
                f"final={len(features_final)}"
            )
        return result

    def get_stable_features(self) -> list[str]:
        """Retorna features presentes en al menos min_fold_stability de los folds."""
        if not self.results_:
            return []

        counts: dict[str, int] = {}
        mean_shap: dict[str, list[float]] = {}
        for result in self.results_:
            for feature in result.features_final:
                counts[feature] = counts.get(feature, 0) + 1
                mean_shap.setdefault(feature, []).append(result.shap_importance.get(feature, 0.0))

        min_appearances = max(1, int(np.ceil(self.config.min_fold_stability * len(self.results_))))
        stable = [
            feature
            for feature, count in counts.items()
            if count >= min_appearances
        ]
        stable.sort(key=lambda name: float(np.mean(mean_shap.get(name, [0.0]))), reverse=True)
        return stable[: self.config.max_features]

    def _compute_shap_importance(
        self,
        model: object,
        X_val: pd.DataFrame,
    ) -> dict[str, float]:
        """Calcula importancia SHAP; con colinealidad la importancia se reparte."""
        feature_names = list(X_val.columns)
        try:
            import shap  # type: ignore

            explainer = shap.TreeExplainer(
                model,
                data=X_val,
                model_output="probability",
                feature_perturbation="interventional",
            )
            sample = X_val.iloc[: min(500, len(X_val))]
            shap_values = explainer.shap_values(sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            importance = np.abs(np.asarray(shap_values)).mean(axis=0)
            return {name: float(value) for name, value in zip(feature_names, importance)}
        except Exception as exc:
            warnings.warn(
                "SHAP no disponible o fallo al calcularlo. "
                f"Se usa fallback de gain. detalle={exc}"
            )
            return self._fallback_gain_importance(model, feature_names)

    def _fallback_gain_importance(
        self,
        model: object,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Fallback con gain de XGBoost o importancias del modelo."""
        if hasattr(model, "get_booster"):
            scores = model.get_booster().get_score(importance_type="gain")
            total = sum(scores.values()) or 1.0
            return {name: float(scores.get(name, 0.0) / total) for name in feature_names}

        if hasattr(model, "feature_importances_"):
            values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
            total = float(values.sum()) or 1.0
            return {name: float(values[idx] / total) for idx, name in enumerate(feature_names)}

        uniform = 1.0 / max(len(feature_names), 1)
        return {name: uniform for name in feature_names}

    def _compute_permutation_importance(
        self,
        model: object,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        n_repeats: int = 5,
    ) -> dict[str, float]:
        """Mide caida de AUC al permutar cada feature."""
        base_probs = self._predict_proba(model, X_val)
        base_auc = roc_auc_score(y_val, base_probs)
        rng = np.random.default_rng(42)
        perm_scores: dict[str, float] = {}

        for column in X_val.columns:
            auc_drops: list[float] = []
            for _ in range(n_repeats):
                permuted = X_val.copy()
                permuted[column] = rng.permutation(permuted[column].to_numpy())
                perm_probs = self._predict_proba(model, permuted)
                perm_auc = roc_auc_score(y_val, perm_probs)
                auc_drops.append(base_auc - perm_auc)
            perm_scores[column] = float(np.mean(auc_drops))

        return perm_scores

    def _apply_correlation_filter(
        self,
        X_train: pd.DataFrame,
        shap_importance: dict[str, float],
    ) -> tuple[list[str], list[str]]:
        """Entre dos features correlacionadas elimina la de menor SHAP."""
        corr_matrix = X_train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_remove: set[str] = set()
        for column in upper.columns:
            high_corr_partners = upper.index[upper[column] > self.config.correlation_threshold]
            for partner in high_corr_partners.tolist():
                if column in to_remove or partner in to_remove:
                    continue
                keep = column
                drop = partner
                if shap_importance.get(partner, 0.0) > shap_importance.get(column, 0.0):
                    keep = partner
                    drop = column
                to_remove.add(drop)
                if self.config.verbose:
                    print(
                        "[FeatureSelector] correlacion alta: "
                        f"keep={keep} drop={drop} corr={upper.loc[partner, column]:.3f}"
                    )

        kept = [name for name in X_train.columns if name not in to_remove]
        return kept, sorted(to_remove)

    @staticmethod
    def _predict_proba(model: object, X: pd.DataFrame) -> np.ndarray:
        """Normaliza la salida de probabilidad entre modelos distintos."""
        if hasattr(model, "predict_proba"):
            probs_array = np.asarray(model.predict_proba(X), dtype=float)
            return probs_array[:, 1] if probs_array.ndim == 2 else probs_array

        if hasattr(model, "predict"):
            try:
                return np.asarray(model.predict(X), dtype=float)
            except Exception:
                pass

        if hasattr(model, "get_booster"):
            import xgboost as xgb  # type: ignore

            matrix = xgb.DMatrix(X, feature_names=list(X.columns))
            return np.asarray(model.predict(matrix), dtype=float)

        raise TypeError("El modelo no expone una API compatible de prediccion.")
