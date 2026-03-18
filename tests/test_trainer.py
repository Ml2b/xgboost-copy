"""Tests del trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.selector import SelectorConfig
from model.registry import ModelMetrics
from model.registry import ModelRegistry
from model.trainer import Trainer
from tests.helpers import make_synthetic_candles
from validation.walk_forward import WalkForwardConfig


def test_trainer_cycle_completes_with_synthetic_data(tmp_path) -> None:
    registry = ModelRegistry(base_dir=tmp_path / "models")
    trainer = Trainer(
        registry=registry,
        data_loader=lambda: make_synthetic_candles(1800, seed=23),
        walk_forward_config=WalkForwardConfig(n_splits=3, min_train_size=600, gap_periods=5, verbose=False),
        selector_config=SelectorConfig(verbose=False, max_features=10),
    )

    result = trainer._retrain_cycle()
    assert result.status == "trained"
    assert result.record is not None


def test_trainer_cycle_handles_insufficient_data_without_error(tmp_path) -> None:
    registry = ModelRegistry(base_dir=tmp_path / "models")
    trainer = Trainer(
        registry=registry,
        data_loader=lambda: make_synthetic_candles(500, seed=29),
        walk_forward_config=WalkForwardConfig(n_splits=3, min_train_size=700, gap_periods=5, verbose=False),
    )

    result = trainer._retrain_cycle()
    assert result.status == "insufficient_data"
    assert result.record is None


def test_trainer_recency_weights_favor_recent_rows(tmp_path) -> None:
    registry = ModelRegistry(base_dir=tmp_path / "models")
    trainer = Trainer(
        registry=registry,
        data_loader=lambda: make_synthetic_candles(120, seed=31),
        recency_weight_half_life_candles=20,
    )

    weights = trainer._build_recency_sample_weight(make_synthetic_candles(120, seed=31))

    assert len(weights) == 120
    assert float(weights[-1]) == 1.0
    assert float(weights[0]) < float(weights[-1])
    assert np.all(np.diff(weights) >= 0)


class _PersistedDummyModel:
    def __init__(self, content: str) -> None:
        self.content = content

    def save_model(self, path: str) -> None:
        Path(path).write_text(self.content, encoding="utf-8")


class _ContinuationSensitiveModel:
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_set=None,
        sample_weight=None,
        sample_weight_eval_set=None,
        verbose: bool | None = None,
        early_stopping_rounds: int | None = None,
        xgb_model: str | None = None,
    ) -> "_ContinuationSensitiveModel":
        if xgb_model:
            raise RuntimeError("continuation_not_supported")
        self.was_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.full(len(X), 0.4), np.full(len(X), 0.6)])


def test_trainer_falls_back_to_cold_start_when_continuation_fails(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(base_dir=tmp_path / "models")
    record = registry.register(
        model=_PersistedDummyModel("old"),
        metrics=ModelMetrics(auc_val=0.60, sharpe=0.3, precision_buy=0.6, win_rate=0.55, max_drawdown=0.1),
        fechas={"train_start": "a", "train_end": "b", "val_start": "c", "val_end": "d"},
        feature_names=["f1", "f2"],
    )
    assert registry.try_promote(record) is True

    trainer = Trainer(registry=registry)
    monkeypatch.setattr(trainer, "_model_factory", lambda: _ContinuationSensitiveModel())

    df = pd.DataFrame(
        {
            "open_time": np.arange(100, dtype=int),
            "f1": np.linspace(0.0, 1.0, 100),
            "f2": np.linspace(1.0, 2.0, 100),
            "target": np.array(([0, 1] * 50), dtype=int),
        }
    )
    weights = np.ones(len(df), dtype=float)

    model, used_continuation = trainer._fit_final_model(df, ["f1", "f2"], sample_weight=weights)

    assert used_continuation is False
    assert isinstance(model, _ContinuationSensitiveModel)
