"""Walk-forward validation con reserva temprana del test final."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from features.selector import FeatureSelector, SelectorConfig


@dataclass(slots=True)
class WalkForwardConfig:
    """Configuracion del esquema walk-forward."""

    n_splits: int = 5
    test_size_pct: float = 0.15
    gap_periods: int = 15
    min_train_size: int = 1000
    verbose: bool = True


@dataclass(slots=True)
class FoldResult:
    """Metrica y limites temporales de un fold."""

    fold_id: int
    train_start_idx: int
    train_end_idx: int
    val_start_idx: int
    val_end_idx: int
    train_size: int
    val_size: int
    auc_val: float
    logloss_val: float
    selected_features: list[str]
    model: object = field(repr=False)


@dataclass(slots=True)
class WalkForwardResult:
    """Resultado agregado del walk-forward."""

    folds: list[FoldResult]
    stable_features: list[str]
    auc_mean: float
    auc_std: float
    auc_test_final: float
    test_start_idx: int
    test_indices: list[int]
    final_test_evaluations: int
    final_model: object = field(repr=False)

    def print_summary(self) -> None:
        """Imprime un resumen compacto del proceso."""
        print("Fold | Train | Val | AUC | Features")
        for fold in self.folds:
            print(
                f"{fold.fold_id:>4} | {fold.train_size:>5} | {fold.val_size:>3} | "
                f"{fold.auc_val:.4f} | {len(fold.selected_features)}"
            )
        print(
            f"auc_mean={self.auc_mean:.4f} auc_std={self.auc_std:.4f} "
            f"auc_test_final={self.auc_test_final:.4f}"
        )


class WalkForwardValidator:
    """Valida el modelo sin mezclar informacion futura con pasada."""

    def __init__(
        self,
        config: WalkForwardConfig,
        model_factory: Callable[[], object],
        selector_config: SelectorConfig | None = None,
    ) -> None:
        self.config = config
        self.model_factory = model_factory
        self.selector_config = selector_config or SelectorConfig(verbose=config.verbose)

    def validate(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> WalkForwardResult:
        """Ejecuta folds temporales y evalua el test final una sola vez."""
        ordered = df.copy().sort_values("open_time").reset_index(drop=True)
        weights = self._normalize_sample_weight(sample_weight, len(ordered))
        n_rows = len(ordered)
        test_size = max(1, int(n_rows * self.config.test_size_pct))
        test_start_idx = n_rows - test_size
        df_wf = ordered.iloc[:test_start_idx].reset_index(drop=True)
        df_test = ordered.iloc[test_start_idx:].reset_index(drop=True)
        weights_wf = weights[:test_start_idx] if weights is not None else None

        if len(df_wf) <= self.config.min_train_size + self.config.gap_periods:
            raise ValueError("No hay suficientes datos para armar los folds walk-forward.")

        if self.config.verbose:
            print(
                f"[WalkForward] total={n_rows} walk_forward={len(df_wf)} "
                f"test_final={len(df_test)}"
            )

        selector = FeatureSelector(copy_selector_config(self.selector_config))
        splits = self._create_splits(len(df_wf))
        fold_results: list[FoldResult] = []

        for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
            X_train = df_wf.iloc[train_idx][feature_cols]
            y_train = df_wf.iloc[train_idx][target_col].to_numpy(dtype=int)
            X_val = df_wf.iloc[val_idx][feature_cols]
            y_val = df_wf.iloc[val_idx][target_col].to_numpy(dtype=int)
            w_train = weights_wf[train_idx] if weights_wf is not None else None
            w_val = weights_wf[val_idx] if weights_wf is not None else None

            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue

            base_model = self._fit_model(
                self.model_factory(),
                X_train,
                y_train,
                X_val,
                y_val,
                sample_weight_train=w_train,
                sample_weight_val=w_val,
            )
            selection = selector.select(X_train, y_train, X_val, y_val, base_model, fold_id)

            X_train_sel = X_train[selection.features_final]
            X_val_sel = X_val[selection.features_final]
            fold_model = self._fit_model(
                self.model_factory(),
                X_train_sel,
                y_train,
                X_val_sel,
                y_val,
                sample_weight_train=w_train,
                sample_weight_val=w_val,
            )
            probs = self._predict_proba(fold_model, X_val_sel)
            auc = float(roc_auc_score(y_val, probs))
            ll = float(log_loss(y_val, probs, labels=[0, 1]))

            fold_results.append(
                FoldResult(
                    fold_id=fold_id,
                    train_start_idx=int(train_idx[0]),
                    train_end_idx=int(train_idx[-1]),
                    val_start_idx=int(val_idx[0]),
                    val_end_idx=int(val_idx[-1]),
                    train_size=len(train_idx),
                    val_size=len(val_idx),
                    auc_val=auc,
                    logloss_val=ll,
                    selected_features=selection.features_final,
                    model=fold_model,
                )
            )

        if not fold_results:
            raise RuntimeError("Ningun fold produjo un modelo evaluable.")

        stable_features = selector.get_stable_features()
        if not stable_features:
            best_fold = max(fold_results, key=lambda fold: fold.auc_val)
            stable_features = best_fold.selected_features

        final_model = self._fit_final_model(df_wf, stable_features, target_col, sample_weight=weights_wf)
        final_probs = self._predict_proba(final_model, df_test[stable_features])
        auc_test_final = float(roc_auc_score(df_test[target_col].to_numpy(dtype=int), final_probs))

        aucs = [fold.auc_val for fold in fold_results]
        return WalkForwardResult(
            folds=fold_results,
            stable_features=stable_features,
            auc_mean=float(np.mean(aucs)),
            auc_std=float(np.std(aucs)),
            auc_test_final=auc_test_final,
            test_start_idx=test_start_idx,
            test_indices=list(range(test_start_idx, n_rows)),
            final_test_evaluations=1,
            final_model=final_model,
        )

    def _create_splits(self, wf_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
        remaining = wf_rows - self.config.min_train_size - self.config.gap_periods
        if remaining <= self.config.n_splits:
            raise ValueError("No hay espacio suficiente para los folds solicitados.")

        val_size = max(1, remaining // self.config.n_splits)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in range(self.config.n_splits):
            train_end = self.config.min_train_size - 1 + (fold * val_size)
            val_start = train_end + self.config.gap_periods + 1
            val_end = min(val_start + val_size - 1, wf_rows - 1)
            if val_start >= wf_rows or val_end <= val_start:
                break
            train_idx = np.arange(0, train_end + 1)
            val_idx = np.arange(val_start, val_end + 1)
            splits.append((train_idx, val_idx))
        return splits

    def _fit_final_model(
        self,
        df_wf: pd.DataFrame,
        stable_features: list[str],
        target_col: str,
        sample_weight: np.ndarray | None = None,
    ) -> object:
        n_rows = len(df_wf)
        split = max(1, int(n_rows * 0.85))
        X_train = df_wf.iloc[:split][stable_features]
        y_train = df_wf.iloc[:split][target_col].to_numpy(dtype=int)
        X_val = df_wf.iloc[split:][stable_features]
        y_val = df_wf.iloc[split:][target_col].to_numpy(dtype=int)
        w_train = sample_weight[:split] if sample_weight is not None else None
        w_val = sample_weight[split:] if sample_weight is not None else None
        if len(X_val) == 0 or len(np.unique(y_val)) < 2:
            X_val = X_train.iloc[-min(100, len(X_train)) :]
            y_val = y_train[-len(X_val) :]
            if w_train is not None:
                w_val = w_train[-len(X_val) :]
        return self._fit_model(
            self.model_factory(),
            X_train,
            y_train,
            X_val,
            y_val,
            sample_weight_train=w_train,
            sample_weight_val=w_val,
        )

    @staticmethod
    def _fit_model(
        model: object,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        sample_weight_train: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> object:
        fit_sig = inspect.signature(model.fit)
        fit_kwargs: dict[str, object] = {}
        if "eval_set" in fit_sig.parameters:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        if sample_weight_train is not None and "sample_weight" in fit_sig.parameters:
            fit_kwargs["sample_weight"] = sample_weight_train
        if sample_weight_val is not None and "sample_weight_eval_set" in fit_sig.parameters:
            fit_kwargs["sample_weight_eval_set"] = [sample_weight_val]
        if "verbose" in fit_sig.parameters:
            fit_kwargs["verbose"] = False
        if "early_stopping_rounds" in fit_sig.parameters:
            fit_kwargs["early_stopping_rounds"] = 30

        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("early_stopping_rounds", None)
            model.fit(X_train, y_train, **fit_kwargs)
        return model

    @staticmethod
    def _predict_proba(model: object, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            probs_array = np.asarray(model.predict_proba(X), dtype=float)
            return probs_array[:, 1] if probs_array.ndim == 2 else probs_array
        preds = model.predict(X)
        return np.asarray(preds, dtype=float)

    @staticmethod
    def _normalize_sample_weight(
        sample_weight: np.ndarray | pd.Series | None,
        expected_length: int,
    ) -> np.ndarray | None:
        if sample_weight is None:
            return None
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(weights) != expected_length:
            raise ValueError("sample_weight debe tener la misma longitud que df.")
        return weights


def copy_selector_config(config: SelectorConfig) -> SelectorConfig:
    """Evita compartir estado mutable entre validadores."""
    return SelectorConfig(
        correlation_threshold=config.correlation_threshold,
        min_shap_importance_pct=config.min_shap_importance_pct,
        min_permutation_importance=config.min_permutation_importance,
        min_fold_stability=config.min_fold_stability,
        max_features=config.max_features,
        verbose=config.verbose,
    )
