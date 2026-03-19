"""Tests del motor de inferencia."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from model.inference import InferenceEngine
from model.registry import MultiAssetModelRegistry


class DummyRegistry:
    """Registry minimo para tests de inferencia."""

    def __init__(self, model_path: str, feature_names: list[str]) -> None:
        self._model_path = model_path
        self._feature_names = feature_names

    def get_active_model_path(self) -> str:
        return self._model_path

    def get_active_feature_names(self) -> list[str]:
        return self._feature_names


class DummyProbModel:
    """Modelo determinista con predict_proba simple."""

    def predict_proba(self, frame):
        base = float(frame.iloc[0, 0])
        prob = max(0.0, min(1.0, base))
        return [[1.0 - prob, prob]]


def test_inference_returns_buy_sell_or_hold(tmp_path) -> None:
    model_path = tmp_path / "active.pkl"
    joblib.dump(DummyProbModel(), model_path)
    engine = InferenceEngine(DummyRegistry(str(model_path), ["f1", "f2"]))  # type: ignore[arg-type]

    assert engine.predict_signal({"f1": 0.9, "f2": 0.0})["signal"] == "BUY"
    assert engine.predict_signal({"f1": 0.1, "f2": 0.0})["signal"] == "SELL"
    assert engine.predict_signal({"f1": 0.5, "f2": 0.0})["signal"] == "HOLD"


def test_inference_latency_stays_under_100ms_on_cpu(tmp_path) -> None:
    model_path = tmp_path / "active.pkl"
    joblib.dump(DummyProbModel(), model_path)
    engine = InferenceEngine(DummyRegistry(str(model_path), ["f1", "f2"]))  # type: ignore[arg-type]

    result = engine.predict_signal({"f1": 0.8, "f2": 0.2})
    assert result is not None
    assert result["latency_ms"] < 100.0


def test_inference_routes_each_product_to_its_own_model(tmp_path) -> None:
    root = tmp_path / "models"
    _write_asset(root, "btc_usdt", ["f1"], True)
    _write_asset(root, "sol_usdt", ["g1"], False)

    joblib.dump(DummyProbModel(), root / "btc_usdt" / "model_btc_usdt.pkl")
    joblib.dump(DummyProbModel(), root / "sol_usdt" / "model_sol_usdt.pkl")

    registry = MultiAssetModelRegistry(root_dir=root, execution_allowed_bases=["BTC"])
    engine = InferenceEngine(registry)

    btc_result = engine.predict_signal({"product_id": "BTC-USD", "f1": 0.9})
    sol_result = engine.predict_signal({"product_id": "SOL-USD", "g1": 0.9})

    assert btc_result is not None
    assert btc_result["signal"] == "BUY"
    assert btc_result["actionable"] == "true"
    assert btc_result["registry_key"] == "btc_usdt"

    assert sol_result is not None
    assert sol_result["signal"] == "BUY"
    assert sol_result["actionable"] == "false"
    assert sol_result["reason"] == "no_promoted_model"
    assert sol_result["registry_key"] == "sol_usdt"


def test_inference_accepts_per_asset_threshold_overrides(tmp_path) -> None:
    root = tmp_path / "models"
    _write_asset(root, "zec_usd", ["f1"], True)
    joblib.dump(DummyProbModel(), root / "zec_usd" / "model_zec_usd.pkl")

    registry = MultiAssetModelRegistry(root_dir=root, execution_allowed_bases=["ZEC"])
    engine = InferenceEngine(
        registry,
        buy_thresholds_by_base={"ZEC": 0.62},
        sell_thresholds_by_base={"ZEC": 0.20},
    )

    sell_result = engine.predict_signal({"product_id": "ZEC-USD", "f1": 0.18})
    hold_result = engine.predict_signal({"product_id": "ZEC-USD", "f1": 0.24})

    assert sell_result is not None
    assert sell_result["signal"] == "SELL"
    assert sell_result["sell_threshold"] == 0.2

    assert hold_result is not None
    assert hold_result["signal"] == "HOLD"
    assert hold_result["buy_threshold"] == 0.62


def test_inference_blocks_signal_when_regime_is_extreme(tmp_path) -> None:
    model_path = tmp_path / "active.pkl"
    joblib.dump(DummyProbModel(), model_path)
    engine = InferenceEngine(  # type: ignore[arg-type]
        DummyRegistry(str(model_path), ["f1"]),
        regime_gate_enabled=True,
        regime_vol_extreme_max=0.001,
        regime_range_compression_min=0.2,
        regime_bb_width_min=0.003,
    )

    result = engine.predict_signal(
        {
            "product_id": "BTC-USD",
            "f1": 0.95,
            "realized_vol_5": 0.002,
            "range_compression_20": 0.8,
            "bb_width": 0.02,
        }
    )
    assert result is not None
    assert result["signal"] == "HOLD"
    assert result["actionable"] == "false"
    assert result["reason"] == "regime_blocked_extreme_volatility"
    assert result["regime"] == "extreme_volatility"
    assert result["regime_actionable"] == "false"


def test_inference_keeps_signal_when_regime_gate_disabled(tmp_path) -> None:
    model_path = tmp_path / "active.pkl"
    joblib.dump(DummyProbModel(), model_path)
    engine = InferenceEngine(  # type: ignore[arg-type]
        DummyRegistry(str(model_path), ["f1"]),
        regime_gate_enabled=False,
    )

    result = engine.predict_signal(
        {
            "product_id": "ETH-USD",
            "f1": 0.9,
            "realized_vol_5": 0.01,
            "range_compression_20": 0.05,
            "bb_width": 0.001,
        }
    )
    assert result is not None
    assert result["signal"] == "BUY"
    assert result["actionable"] == "true"
    assert result["regime"] == "disabled"
    assert result["regime_actionable"] == "true"


def _write_asset(root: Path, registry_key: str, feature_names: list[str], promoted: bool) -> None:
    asset_dir = root / registry_key
    asset_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"model_{registry_key}.pkl"
    registry_payload = [
        {
            "model_id": f"model_{registry_key}",
            "model_path": str(asset_dir / model_name),
            "created_at": "2026-03-18T00:00:00+00:00",
            "train_start": "a",
            "train_end": "b",
            "val_start": "c",
            "val_end": "d",
            "metrics": {
                "auc_val": 0.61,
                "sharpe": 0.2,
                "precision_buy": 0.6,
                "win_rate": 0.55,
                "max_drawdown": 0.1,
            },
            "promoted": promoted,
            "feature_names": feature_names,
            "n_features": len(feature_names),
        }
    ]
    (asset_dir / "registry.json").write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")
    if promoted:
        (asset_dir / "active_model_meta.json").write_text(
            json.dumps(
                {
                    "model_id": f"model_{registry_key}",
                    "model_path": str(asset_dir / model_name),
                    "feature_names": feature_names,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
