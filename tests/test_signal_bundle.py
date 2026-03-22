"""Tests del bundle serializable con calibracion y guardas de runtime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.signal_bundle import (
    FeatureDriftMonitor,
    MarketRegimeModel,
    ProbabilityCalibrator,
    SignalModelBundle,
    SignalThresholdOptimizer,
)


class DummyProbModel:
    """Modelo determinista con salida compatible tipo sklearn."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def predict_proba(self, frame: pd.DataFrame):
        base = np.clip(frame.iloc[:, 0].to_numpy(dtype=float) * self.scale, 0.0, 1.0)
        return np.column_stack([1.0 - base, base])


def test_probability_calibrator_outputs_valid_probabilities() -> None:
    raw = np.asarray([0.1, 0.2, 0.7, 0.8, 0.9], dtype=float)
    y_true = np.asarray([0, 0, 1, 1, 1], dtype=int)

    calibrator = ProbabilityCalibrator.fit(raw, y_true)
    calibrated = calibrator.transform(raw)

    assert calibrator.method in {"identity", "isotonic", "platt"}
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)


def test_threshold_optimizer_returns_consistent_ordering() -> None:
    probs = np.asarray([0.12, 0.18, 0.25, 0.61, 0.68, 0.74, 0.81, 0.88] * 6, dtype=float)
    returns = np.asarray([-0.02, -0.01, -0.004, 0.002, 0.007, 0.011, 0.018, 0.021] * 6, dtype=float)
    y_true = np.asarray([0, 0, 0, 1, 1, 1, 1, 1] * 6, dtype=int)

    result = SignalThresholdOptimizer.optimize(probs, returns, y_true)

    assert 0.0 <= result.exit_threshold < result.buy_threshold <= 1.0
    assert result.buy_support > 0
    assert result.entry_ready is True


def test_threshold_optimizer_marks_entry_not_ready_without_buy_support() -> None:
    probs = np.asarray([0.20, 0.25, 0.30, 0.35, 0.40] * 20, dtype=float)
    returns = np.asarray([0.001, -0.002, 0.0005, -0.001, 0.0001] * 20, dtype=float)
    y_true = np.asarray([0, 1, 0, 1, 0] * 20, dtype=int)

    result = SignalThresholdOptimizer.optimize(probs, returns, y_true)

    assert result.buy_support == 0
    assert result.entry_ready is False


def test_threshold_optimizer_preserves_real_support_when_entry_not_ready() -> None:
    probs = np.asarray([0.30, 0.40, 0.50, 0.64, 0.68] * 4, dtype=float)
    returns = np.asarray([-0.003, -0.002, 0.001, 0.005, 0.006] * 4, dtype=float)
    y_true = np.asarray([0, 0, 0, 1, 1] * 4, dtype=int)

    result = SignalThresholdOptimizer.optimize(probs, returns, y_true)

    assert result.entry_ready is False
    assert result.buy_threshold == 0.62
    assert result.buy_support == 8
    assert result.buy_precision == 1.0


def test_feature_drift_monitor_blocks_large_outlier_ratio() -> None:
    frame = pd.DataFrame(
        {
            "f1": np.linspace(0.0, 1.0, 100),
            "f2": np.linspace(0.1, 1.1, 100),
            "f3": np.linspace(-1.0, 1.0, 100),
            "f4": np.linspace(10.0, 20.0, 100),
            "f5": np.linspace(100.0, 200.0, 100),
            "f6": np.linspace(0.01, 0.02, 100),
            "f7": np.linspace(5.0, 7.0, 100),
            "f8": np.linspace(50.0, 70.0, 100),
        }
    )
    monitor = FeatureDriftMonitor.fit(frame)

    evaluation = monitor.evaluate_payload(
        {
            "f1": 99.0,
            "f2": 98.0,
            "f3": 97.0,
            "f4": 96.0,
            "f5": 95.0,
            "f6": 0.015,
            "f7": 6.0,
            "f8": 60.0,
        }
    )

    assert evaluation.actionable is False
    assert evaluation.status == "out_of_distribution"
    assert evaluation.outlier_ratio > 0.3


def test_market_regime_model_heuristic_blocks_extreme_volatility() -> None:
    frame = pd.DataFrame(
        {
            "realized_vol_5": np.linspace(0.001, 0.010, 120),
            "bb_width": np.linspace(0.002, 0.020, 120),
            "range_compression_20": np.linspace(0.1, 1.0, 120),
            "volume_ratio": np.linspace(0.5, 2.0, 120),
            "trend_slope_pct": np.linspace(-0.01, 0.01, 120),
        }
    )
    regime_model = MarketRegimeModel.fit(frame)

    evaluation = regime_model.evaluate_payload(
        {
            "realized_vol_5": 0.05,
            "bb_width": 0.03,
            "range_compression_20": 0.8,
            "volume_ratio": 1.0,
            "trend_slope_pct": 0.001,
        }
    )

    assert evaluation.actionable is False
    assert evaluation.regime == "extreme_volatility"


def test_signal_model_bundle_blends_models_and_exposes_runtime_helpers() -> None:
    frame = pd.DataFrame({"f1": [0.8], "f2": [0.2]})
    bundle = SignalModelBundle(
        primary_model=DummyProbModel(scale=1.0),
        secondary_model=DummyProbModel(scale=0.5),
        calibrator=ProbabilityCalibrator(method="identity"),
        drift_monitor=FeatureDriftMonitor.fit(pd.DataFrame({"f1": np.linspace(0.0, 1.0, 50), "f2": np.linspace(0.0, 1.0, 50)})),
        regime_model=MarketRegimeModel.fit(
            pd.DataFrame(
                {
                    "realized_vol_5": np.linspace(0.001, 0.010, 120),
                    "bb_width": np.linspace(0.002, 0.020, 120),
                    "range_compression_20": np.linspace(0.1, 1.0, 120),
                    "volume_ratio": np.linspace(0.5, 2.0, 120),
                    "trend_slope_pct": np.linspace(-0.01, 0.01, 120),
                }
            )
        ),
        feature_names=["f1", "f2"],
        buy_threshold=0.7,
        exit_threshold=0.2,
        primary_model_name="xgboost",
        secondary_model_name="lightgbm",
    )

    probs = bundle.predict_proba(frame)
    drift = bundle.evaluate_drift_payload({"f1": 0.5, "f2": 0.5})
    regime = bundle.evaluate_regime_payload(
        {
            "realized_vol_5": 0.002,
            "bb_width": 0.01,
            "range_compression_20": 0.7,
            "volume_ratio": 1.1,
            "trend_slope_pct": 0.001,
        }
    )

    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0
    assert drift.actionable is True
    assert regime.reason in {"ok", "regime_gate_disabled"}
