"""Loop periodico de reentrenamiento del modelo."""

from __future__ import annotations

import asyncio
import gc
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import precision_score, roc_auc_score

from config import settings
from data.history_store import CandleHistoryStore
from features.calculator import FeatureCalculator
from features.order_flow import ORDER_FLOW_RAW_COLUMNS
from features.selector import SelectorConfig
from model.registry import ModelMetrics, ModelRecord, ModelRegistry, MultiAssetModelRegistry
from model.signal_bundle import (
    FeatureDriftMonitor,
    MarketRegimeModel,
    ProbabilityCalibrator,
    SignalModelBundle,
    SignalThresholdOptimizer,
    build_lightgbm_classifier,
)
from target.builder import TargetBuilder, TargetConfig, TargetType
from validation.walk_forward import (
    WalkForwardConfig,
    WalkForwardValidator,
    build_purged_train_validation_bounds,
)


@dataclass(slots=True)
class RetrainCycleResult:
    """Resumen de un ciclo de entrenamiento."""

    status: str
    record: ModelRecord | None = None
    promoted: bool = False
    auc: float = 0.0
    feature_names: list[str] | None = None
    used_continuation: bool = False
    history_rows: int = 0
    reason: str = ""


class Trainer:
    """Carga historico, valida, entrena y registra nuevos modelos."""

    def __init__(
        self,
        registry: ModelRegistry,
        redis_client: object | None = None,
        data_loader: Callable[[], pd.DataFrame] | None = None,
        order_flow_loader: Callable[[], pd.DataFrame | None] | None = None,
        calculator: FeatureCalculator | None = None,
        target_builder: TargetBuilder | None = None,
        walk_forward_config: WalkForwardConfig | None = None,
        selector_config: SelectorConfig | None = None,
        retrain_interval: int = settings.RETRAIN_INTERVAL,
        recency_weight_half_life_candles: int = settings.TRAINER_RECENCY_WEIGHT_HALF_LIFE_CANDLES,
    ) -> None:
        self.registry = registry
        self.redis_client = redis_client
        self.data_loader = data_loader
        self.order_flow_loader = order_flow_loader
        self.calculator = calculator or FeatureCalculator()
        self.target_builder = target_builder or TargetBuilder(
            TargetConfig(
                target_type=TargetType.NET_RETURN_THRESHOLD,
                horizon=settings.TARGET_HORIZON,
                threshold_pct=settings.TARGET_THRESHOLD_PCT,
                fee_pct=settings.FEE_PCT,
            )
        )
        self.walk_forward_config = walk_forward_config or WalkForwardConfig(verbose=False)
        self.selector_config = selector_config or SelectorConfig(verbose=False)
        self.retrain_interval = retrain_interval
        self.recency_weight_half_life_candles = max(1, int(recency_weight_half_life_candles))

    async def start(self, stop_event: asyncio.Event | None = None) -> None:
        """Loop asyncrono que reentrena cada RETRAIN_INTERVAL."""
        stop_event = stop_event or asyncio.Event()
        logger.info(
            "Trainer unitario iniciado. registry_dir={} retrain_interval_s={}",
            self.registry.base_dir,
            self.retrain_interval,
        )
        while not stop_event.is_set():
            await asyncio.to_thread(self._retrain_cycle)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.retrain_interval)
            except asyncio.TimeoutError:
                continue

    def _retrain_cycle(self) -> RetrainCycleResult:
        """Ejecuta el pipeline completo de reentrenamiento."""
        candles = self._load_candles()
        wf = self.walk_forward_config
        # Umbral real: walk-forward necesita min_train + gap + splits filas,
        # y reserva test_size_pct, asi que el total minimo es mayor.
        _wf_min = wf.min_train_size + wf.gap_periods + wf.n_splits + 1
        _min_rows = int(_wf_min / (1.0 - wf.test_size_pct)) + 1
        if candles is None or len(candles) < _min_rows:
            return RetrainCycleResult(status="insufficient_data", history_rows=0 if candles is None else len(candles))

        df_order_flow = self._load_order_flow()
        features_df = self.calculator.compute(candles, df_order_flow=df_order_flow)
        labeled_df = self.target_builder.build(features_df)
        if len(labeled_df) < _min_rows:
            return RetrainCycleResult(status="insufficient_data", history_rows=len(labeled_df))

        feature_cols = [
            column
            for column in self.calculator.feature_columns
            if column in labeled_df.columns
        ]
        sample_weight = self._build_recency_sample_weight(labeled_df)

        validator = WalkForwardValidator(
            config=self.walk_forward_config,
            model_factory=self._model_factory,
            selector_config=self.selector_config,
        )
        try:
            wf_result = validator.validate(
                labeled_df,
                target_col="target",
                feature_cols=feature_cols,
                sample_weight=sample_weight,
            )
        except ValueError as exc:
            logger.warning(
                "Trainer ciclo degradado a insufficient_data. registry_dir={} history_rows={} reason={}",
                self.registry.base_dir,
                len(labeled_df),
                exc,
            )
            return RetrainCycleResult(
                status="insufficient_data",
                history_rows=len(labeled_df),
                reason=str(exc),
            )
        except RuntimeError as exc:
            if "evaluable" not in str(exc).lower():
                raise
            logger.warning(
                "Trainer ciclo degradado a no_evaluable. registry_dir={} history_rows={} reason={}",
                self.registry.base_dir,
                len(labeled_df),
                exc,
            )
            return RetrainCycleResult(
                status="no_evaluable",
                history_rows=len(labeled_df),
                reason=str(exc),
            )
        stable_features = wf_result.stable_features
        if not stable_features:
            return RetrainCycleResult(
                status="no_stable_features",
                history_rows=len(labeled_df),
                reason="walk-forward eliminó todas las features",
            )
        df_wf = labeled_df.iloc[: wf_result.test_start_idx].reset_index(drop=True)
        df_test = labeled_df.iloc[wf_result.test_start_idx :].reset_index(drop=True)
        sample_weight_wf = sample_weight[: wf_result.test_start_idx] if sample_weight is not None else None
        final_model, used_continuation = self._fit_final_model(
            df_wf,
            stable_features,
            sample_weight=sample_weight_wf,
        )
        metrics = self._compute_holdout_metrics(df_test, stable_features, final_model)

        timestamps = pd.to_datetime(labeled_df["open_time"], unit="ms", utc=True)
        holdout_start_idx = max(1, wf_result.test_start_idx)
        record = self.registry.register(
            model=final_model,
            metrics=metrics,
            fechas={
                "train_start": timestamps.iloc[0].isoformat(),
                "train_end": timestamps.iloc[holdout_start_idx - 1].isoformat(),
                "val_start": timestamps.iloc[holdout_start_idx].isoformat() if holdout_start_idx < len(timestamps) else timestamps.iloc[-1].isoformat(),
                "val_end": timestamps.iloc[-1].isoformat(),
            },
            feature_names=stable_features,
        )
        promoted = self.registry.try_promote(record)
        return RetrainCycleResult(
            status="trained",
            record=record,
            promoted=promoted,
            auc=metrics.auc_val,
            feature_names=stable_features,
            used_continuation=used_continuation,
            history_rows=len(candles),
        )

    def _load_candles(self) -> pd.DataFrame | None:
        if self.data_loader is not None:
            return self.data_loader()
        if self.redis_client is None:
            return None

        entries = self.redis_client.xrange(settings.STREAM_MARKET_CANDLES_1M, count=5000)
        records: list[dict[str, object]] = []
        for _, payload in entries:
            records.append(
                {
                    "product_id": payload.get("product_id"),
                    "open_time": int(payload.get("open_time")),
                    "close_time": int(payload.get("close_time")),
                    "open": float(payload.get("open")),
                    "high": float(payload.get("high")),
                    "low": float(payload.get("low")),
                    "close": float(payload.get("close")),
                    "volume": float(payload.get("volume")),
                    "trade_count": int(payload.get("trade_count")),
                }
            )
        if not records:
            return None
        return pd.DataFrame.from_records(records)

    def _load_order_flow(self) -> pd.DataFrame | None:
        if self.order_flow_loader is not None:
            return self.order_flow_loader()
        if self.redis_client is None:
            return None
        entries = self.redis_client.xrange(
            "market.orderflow.1m", count=5000
        )
        records: list[dict[str, object]] = []
        for _, payload in entries:
            row: dict[str, object] = {
                "open_time": int(payload.get("open_time", 0))
            }
            for col in ORDER_FLOW_RAW_COLUMNS:
                row[col] = float(payload.get(col, 0.0))
            records.append(row)
        if not records:
            return None
        return pd.DataFrame.from_records(records)

    def _fit_final_model(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        sample_weight: np.ndarray | None = None,
    ) -> tuple[object, bool]:
        train_end, val_start = build_purged_train_validation_bounds(
            len(df),
            self.walk_forward_config.gap_periods,
        )
        X_train = df.iloc[:train_end][feature_names]
        y_train = df.iloc[:train_end]["target"].to_numpy(dtype=int)
        X_val = df.iloc[val_start:][feature_names]
        y_val = df.iloc[val_start:]["target"].to_numpy(dtype=int)
        w_train = sample_weight[:train_end] if sample_weight is not None else None
        w_val = sample_weight[val_start:] if sample_weight is not None else None
        if len(X_val) == 0:
            X_val = X_train.iloc[-min(100, len(X_train)) :]
            y_val = y_train[-len(X_val) :]
            if w_train is not None:
                w_val = w_train[-len(X_val) :]

        active_model_path = self.registry.get_active_model_path()
        active_feature_names = self.registry.get_active_feature_names()
        can_continue = active_feature_names == feature_names
        used_continuation = False
        primary_xgb_model = None
        if (
            active_model_path
            and Path(active_model_path).exists()
            and can_continue
            and Path(active_model_path).suffix == ".json"
        ):
            try:
                primary_xgb_model = self._fit_model_instance(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    sample_weight_train=w_train,
                    sample_weight_val=w_val,
                    xgb_model=active_model_path,
                )
                used_continuation = True
            except Exception:
                primary_xgb_model = None
                used_continuation = False

        if primary_xgb_model is None:
            primary_xgb_model = self._fit_model_instance(
                X_train,
                y_train,
                X_val,
                y_val,
                sample_weight_train=w_train,
                sample_weight_val=w_val,
            )
        secondary_model = self._fit_lightgbm_model(
            X_train,
            y_train,
            X_val,
            y_val,
            sample_weight_train=w_train,
            sample_weight_val=w_val,
        )
        raw_val_probs = self._blend_validation_probs(primary_xgb_model, secondary_model, X_val)
        calibrator = ProbabilityCalibrator.fit(raw_val_probs, y_val)
        calibrated_val_probs = calibrator.transform(raw_val_probs)
        target_returns = (
            df.iloc[val_start:]["target_return"].to_numpy(dtype=float)
            if "target_return" in df.columns
            else np.zeros(len(X_val), dtype=float)
        )
        threshold_result = SignalThresholdOptimizer.optimize(
            calibrated_val_probs,
            target_returns,
            y_val,
        )
        bundle = SignalModelBundle(
            primary_model=primary_xgb_model,
            secondary_model=secondary_model,
            calibrator=calibrator,
            drift_monitor=FeatureDriftMonitor.fit(df[feature_names], feature_names=feature_names),
            regime_model=MarketRegimeModel.fit(df),
            feature_names=list(feature_names),
            buy_threshold=threshold_result.buy_threshold,
            exit_threshold=threshold_result.exit_threshold,
            primary_model_name="xgboost",
            secondary_model_name="lightgbm" if secondary_model is not None else None,
            calibration_method=calibrator.method,
            threshold_metadata={
                "buy_score": round(threshold_result.buy_score, 6),
                "exit_score": round(threshold_result.exit_score, 6),
                "buy_support": threshold_result.buy_support,
                "exit_support": threshold_result.exit_support,
                "buy_precision": threshold_result.buy_precision,
                "entry_ready": threshold_result.entry_ready,
            },
        )
        return bundle, used_continuation

    def _compute_holdout_metrics(self, df: pd.DataFrame, feature_names: list[str], model: object) -> ModelMetrics:
        X_val = df[feature_names]
        y_val = df["target"].to_numpy(dtype=int)
        if len(X_val) == 0:
            raise ValueError("El holdout final no puede estar vacio.")

        probs = self._predict_proba(model, X_val)
        try:
            auc = float(roc_auc_score(y_val, probs))
        except ValueError:
            auc = 0.5
        buy_threshold = float(getattr(model, "buy_threshold", settings.MIN_SIGNAL_PROB))
        buy_preds = (probs >= buy_threshold).astype(int)
        precision_buy = float(precision_score(y_val, buy_preds, zero_division=0))
        threshold_metadata = getattr(model, "threshold_metadata", {})
        buy_support = int(threshold_metadata.get("buy_support", int(buy_preds.sum())))
        exit_support = int(threshold_metadata.get("exit_support", 0))
        entry_ready = bool(
            threshold_metadata.get(
                "entry_ready",
                buy_support >= settings.PROMOTION_MIN_BUY_SUPPORT,
            )
        )
        exit_threshold = float(
            getattr(model, "exit_threshold", max(0.05, 1.0 - settings.MIN_SIGNAL_PROB))
        )

        returns = df["target_return"].to_numpy(dtype=float)
        trade_returns = returns[buy_preds == 1]
        win_rate = float(np.mean(trade_returns > 0)) if len(trade_returns) else 0.0
        sharpe = self._compute_sharpe(trade_returns)
        max_drawdown = self._compute_max_drawdown(trade_returns)
        return ModelMetrics(
            auc_val=auc,
            sharpe=sharpe,
            precision_buy=precision_buy,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            buy_support=buy_support,
            exit_support=exit_support,
            buy_threshold=buy_threshold,
            exit_threshold=exit_threshold,
            entry_ready=entry_ready,
        )

    @staticmethod
    def _compute_sharpe(trade_returns: np.ndarray) -> float:
        if len(trade_returns) < 2:
            return 0.0
        std = float(np.std(trade_returns))
        if std == 0:
            return 0.0
        return float(np.mean(trade_returns) / std * np.sqrt(len(trade_returns)))

    @staticmethod
    def _compute_max_drawdown(trade_returns: np.ndarray) -> float:
        if len(trade_returns) == 0:
            return 0.0
        cumulative = np.cumsum(trade_returns)
        peak = np.maximum.accumulate(cumulative)
        return float(np.max(peak - cumulative))

    @staticmethod
    def _predict_proba(model: object, X: pd.DataFrame) -> np.ndarray:
        probs_array = np.asarray(model.predict_proba(X), dtype=float)
        return probs_array[:, 1] if probs_array.ndim == 2 else probs_array

    def _fit_model_instance(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        sample_weight_train: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
        xgb_model: str | None = None,
    ) -> object:
        """Entrena un modelo final, con continuation opcional y fallback controlado."""
        model = self._model_factory()
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
        if xgb_model and "xgb_model" in fit_sig.parameters:
            fit_kwargs["xgb_model"] = xgb_model

        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("early_stopping_rounds", None)
            model.fit(X_train, y_train, **fit_kwargs)
        return model

    def _build_recency_sample_weight(self, df: pd.DataFrame) -> np.ndarray:
        """Asigna mas peso a las velas recientes sin descartar la historia previa."""
        length = len(df)
        if length == 0:
            return np.array([], dtype=float)
        age = np.arange(length - 1, -1, -1, dtype=float)
        weights = np.power(0.5, age / float(self.recency_weight_half_life_candles))
        return np.clip(weights, 1e-3, 1.0).astype(float)

    @staticmethod
    def _model_factory() -> object:
        try:
            from xgboost import XGBClassifier  # type: ignore

            return XGBClassifier(**settings.XGB_PARAMS)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=42)

    def _fit_lightgbm_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        sample_weight_train: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> object | None:
        """Entrena LightGBM si esta disponible; si no, se omite sin romper el flujo."""
        model = build_lightgbm_classifier()
        if model is None:
            return None
        fit_sig = inspect.signature(model.fit)
        fit_kwargs: dict[str, object] = {}
        if "eval_set" in fit_sig.parameters:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        if sample_weight_train is not None and "sample_weight" in fit_sig.parameters:
            fit_kwargs["sample_weight"] = sample_weight_train
        if sample_weight_val is not None and "eval_sample_weight" in fit_sig.parameters:
            fit_kwargs["eval_sample_weight"] = [sample_weight_val]
        if "callbacks" in fit_sig.parameters:
            try:
                from lightgbm import early_stopping, log_evaluation  # type: ignore

                fit_kwargs["callbacks"] = [
                    early_stopping(30, verbose=False),
                    log_evaluation(period=0),
                ]
            except Exception:
                pass
        if "verbose" in fit_sig.parameters:
            fit_kwargs["verbose"] = -1
        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("verbose", None)
            fit_kwargs.pop("callbacks", None)
            model.fit(X_train, y_train, **fit_kwargs)
        return model

    def _blend_validation_probs(
        self,
        primary_model: object,
        secondary_model: object | None,
        X_val: pd.DataFrame,
    ) -> np.ndarray:
        """Promedia boosters disponibles antes de calibrar probabilidades."""
        probabilities = [self._predict_proba(primary_model, X_val)]
        if secondary_model is not None:
            probabilities.append(self._predict_proba(secondary_model, X_val))
        stacked = np.vstack(probabilities)
        return np.mean(stacked, axis=0)


class MultiAssetTrainerService:
    """Reentrena un modelo por activo base sin mezclar series entre si."""

    def __init__(
        self,
        registry_root: str | Path = settings.MODEL_REGISTRY_ROOT,
        redis_client: object | None = None,
        retrain_interval: int = settings.RETRAIN_INTERVAL,
        walk_forward_config: WalkForwardConfig | None = None,
        selector_config: SelectorConfig | None = None,
        history_store: CandleHistoryStore | None = None,
        observed_bases: list[str] | None = None,
    ) -> None:
        self.registry_root = Path(registry_root)
        self.redis_client = redis_client
        self.retrain_interval = retrain_interval
        self.walk_forward_config = walk_forward_config or WalkForwardConfig(
            n_splits=3,
            gap_periods=10,
            min_train_size=500,
            verbose=False,
        )
        self.selector_config = selector_config or SelectorConfig(verbose=False)
        self.multi_registry = MultiAssetModelRegistry(root_dir=self.registry_root)
        self.history_store = history_store or CandleHistoryStore()
        self.observed_bases: list[str] = [
            b.strip().upper()
            for b in (observed_bases or settings.OBSERVED_BASES)
            if b.strip()
        ]

    async def start(self, stop_event: asyncio.Event | None = None) -> None:
        """Loop asyncrono que recorre los assets y dispara su trainer dedicado."""
        stop_event = stop_event or asyncio.Event()
        logger.info(
            "MultiAssetTrainerService iniciado. registry_root={} retrain_interval_s={} sync_interval_s={}",
            self.registry_root,
            self.retrain_interval,
            settings.TRAINER_HISTORY_SYNC_INTERVAL,
        )
        await asyncio.gather(
            self._sync_loop(stop_event),
            self._retrain_loop(stop_event),
        )

    async def _sync_loop(self, stop_event: asyncio.Event) -> None:
        """Sincroniza velas y order flow de Redis → SQLite cada TRAINER_HISTORY_SYNC_INTERVAL segundos."""
        # Sync inmediato al arrancar para no perder datos acumulados en Redis
        try:
            await asyncio.to_thread(self._sync_history)
            logger.info("Sync inicial completado al arrancar _sync_loop")
        except Exception as exc:
            logger.warning("Sync inicial fallido (continua): {}", exc)
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=settings.TRAINER_HISTORY_SYNC_INTERVAL)
            except asyncio.TimeoutError:
                pass
            try:
                await asyncio.to_thread(self._sync_history)
            except Exception as exc:
                logger.warning("Trainer sync_loop error (continua): {}", exc)

    async def _retrain_loop(self, stop_event: asyncio.Event) -> None:
        """Reentrena todos los activos cada retrain_interval segundos."""
        while not stop_event.is_set():
            try:
                await asyncio.to_thread(self._retrain_all_assets)
                self.multi_registry.refresh()
            except Exception as exc:
                logger.warning("Trainer retrain_loop error (continua): {}", exc)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.retrain_interval)
            except asyncio.TimeoutError:
                continue

    def _retrain_all_assets(self) -> None:
        cycle_started = time.perf_counter()
        self._sync_history()
        total_assets = 0
        trained_assets = 0
        promoted_assets = 0
        logger.info(
            "Trainer ciclo iniciado. assets={} registry_root={}",
            len(self.multi_registry.get_registry_keys()),
            self.registry_root,
        )
        for artifact in self._iter_registry_artifacts():
            total_assets += 1
            asset_started = time.perf_counter()
            logger.info(
                "Trainer asset iniciado. base_asset={} registry_key={}",
                artifact.base_asset,
                artifact.registry_key,
            )
            trainer = Trainer(
                registry=ModelRegistry(base_dir=self.registry_root / artifact.registry_key),
                data_loader=lambda base_asset=artifact.base_asset: self._load_candles_for_base(base_asset),
                order_flow_loader=lambda base_asset=artifact.base_asset: self._load_order_flow_for_base(base_asset),
                walk_forward_config=self.walk_forward_config,
                selector_config=self.selector_config,
                retrain_interval=self.retrain_interval,
            )
            result = trainer._retrain_cycle()
            del trainer
            gc.collect()
            if result.status == "trained":
                trained_assets += 1
            if result.promoted:
                promoted_assets += 1
            logger.info(
                "Trainer asset finalizado. base_asset={} status={} promoted={} auc={} continuation={} history_rows={} reason={} duration_s={}",
                artifact.base_asset,
                result.status,
                result.promoted,
                round(result.auc, 4),
                result.used_continuation,
                result.history_rows,
                result.reason,
                round(time.perf_counter() - asset_started, 2),
            )
        logger.info(
            "Trainer ciclo finalizado. assets={} trained={} promoted={} duration_s={}",
            total_assets,
            trained_assets,
            promoted_assets,
            round(time.perf_counter() - cycle_started, 2),
        )

    def _load_candles_for_base(self, base_asset: str) -> pd.DataFrame | None:
        return self.history_store.load_candles_for_base(base_asset)

    def _load_order_flow_for_base(
        self, base_asset: str
    ) -> pd.DataFrame | None:
        df = self.history_store.get_candles_with_order_flow(base_asset)
        if df.empty:
            return None
        of_cols = ["open_time"] + [
            c for c in ORDER_FLOW_RAW_COLUMNS if c in df.columns
        ]
        return df[of_cols]

    def _sync_history(self) -> None:
        """Persiste velas y order flow nuevos antes del reentrenamiento."""
        if self.redis_client is None:
            return
        synced_rows = self.history_store.sync_from_redis_stream(
            self.redis_client
        )
        logger.info(
            "Trainer sync candles completado. inserted_rows={}", synced_rows
        )
        synced_of = self.history_store.sync_order_flow_from_redis_stream(
            self.redis_client
        )
        logger.info(
            "Trainer sync order_flow completado. inserted_rows={}", synced_of
        )

    def _iter_registry_artifacts(self):
        """Itera todos los assets: los que ya tienen registry + los observados nuevos."""
        from model.registry import ResolvedModelArtifact

        seen_bases: set[str] = set()

        # Assets con registry existente
        for registry_key in self.multi_registry.get_registry_keys():
            artifact = self.multi_registry.resolve_artifact(
                registry_key.replace("_", "-").upper()
            )
            if artifact is not None:
                seen_bases.add(artifact.base_asset.upper())
                yield artifact

        # Assets observados sin registry aun — crear directorio y arrancar desde cero
        for base in self.observed_bases:
            base_upper = base.strip().upper()
            if not base_upper or base_upper in seen_bases:
                continue
            registry_key = f"{base_upper.lower()}_usd"
            asset_dir = self.registry_root / registry_key
            asset_dir.mkdir(parents=True, exist_ok=True)
            seen_bases.add(base_upper)
            yield ResolvedModelArtifact(
                registry_key=registry_key,
                base_asset=base_upper,
                model_id="",
                model_path=None,
                feature_names=[],
                actionable=False,
                reason="new_asset",
                fingerprint=f"new:{base_upper}",
            )
