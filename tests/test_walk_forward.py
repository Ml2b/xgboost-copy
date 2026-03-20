"""Tests del validador walk-forward."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from features.calculator import FeatureCalculator
from target.builder import TargetBuilder, TargetConfig, TargetType
from tests.helpers import make_synthetic_candles
from validation.walk_forward import WalkForwardConfig, WalkForwardValidator, build_purged_train_validation_bounds


def test_walk_forward_computes_final_test_once_and_keeps_it_separate() -> None:
    candles = make_synthetic_candles(2200, seed=19)
    features = FeatureCalculator().compute(candles)
    labeled = TargetBuilder(
        TargetConfig(
            target_type=TargetType.NET_RETURN_THRESHOLD,
            horizon=5,
            threshold_pct=0.04,
            fee_pct=0.005,
        )
    ).build(features)

    validator = WalkForwardValidator(
        config=WalkForwardConfig(n_splits=3, min_train_size=700, gap_periods=5, verbose=False),
        model_factory=lambda: RandomForestClassifier(n_estimators=40, random_state=42),
    )
    result = validator.validate(labeled, target_col="target", feature_cols=FeatureCalculator().feature_columns)

    assert result.final_test_evaluations == 1
    test_index_set = set(result.test_indices)
    for fold in result.folds:
        fold_range = set(range(fold.train_start_idx, fold.train_end_idx + 1)) | set(range(fold.val_start_idx, fold.val_end_idx + 1))
        assert test_index_set.isdisjoint(fold_range)


def test_walk_forward_accepts_sample_weight_without_breaking_contract() -> None:
    candles = make_synthetic_candles(2200, seed=23)
    features = FeatureCalculator().compute(candles)
    labeled = TargetBuilder(
        TargetConfig(
            target_type=TargetType.NET_RETURN_THRESHOLD,
            horizon=5,
            threshold_pct=0.04,
            fee_pct=0.005,
        )
    ).build(features)
    weights = [1.0] * len(labeled)
    for idx in range(len(weights) // 2, len(weights)):
        weights[idx] = 2.0

    validator = WalkForwardValidator(
        config=WalkForwardConfig(n_splits=3, min_train_size=700, gap_periods=5, verbose=False),
        model_factory=lambda: RandomForestClassifier(n_estimators=40, random_state=42),
    )
    result = validator.validate(
        labeled,
        target_col="target",
        feature_cols=FeatureCalculator().feature_columns,
        sample_weight=weights,
    )

    assert result.final_test_evaluations == 1
    assert result.auc_test_final >= 0.0


def test_final_purged_split_reserves_gap_between_train_and_validation() -> None:
    train_end, val_start = build_purged_train_validation_bounds(1000, 15)
    assert train_end == 835
    assert val_start == 850
