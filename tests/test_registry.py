"""Tests del registro de modelos."""

from __future__ import annotations

import json
from pathlib import Path

from model.registry import ModelMetrics, ModelRegistry, MultiAssetModelRegistry


class DummyModel:
    """Modelo minimo para simular save_model."""

    def __init__(self, content: str) -> None:
        self.content = content

    def save_model(self, path: str) -> None:
        Path(path).write_text(self.content, encoding="utf-8")


def test_first_model_is_promoted_if_it_passes_minimum(tmp_path) -> None:
    registry = ModelRegistry(base_dir=tmp_path)
    record = registry.register(
        model=DummyModel("first"),
        metrics=ModelMetrics(auc_val=0.60, sharpe=0.3, precision_buy=0.6, win_rate=0.55, max_drawdown=0.1),
        fechas={"train_start": "a", "train_end": "b", "val_start": "c", "val_end": "d"},
        feature_names=["f1", "f2"],
    )
    assert registry.try_promote(record) is True
    assert registry.has_active_model() is True


def test_first_model_is_not_promoted_with_negative_sharpe(tmp_path) -> None:
    registry = ModelRegistry(base_dir=tmp_path)
    record = registry.register(
        model=DummyModel("first"),
        metrics=ModelMetrics(auc_val=0.60, sharpe=-0.1, precision_buy=0.6, win_rate=0.55, max_drawdown=0.1),
        fechas={"train_start": "a", "train_end": "b", "val_start": "c", "val_end": "d"},
        feature_names=["f1", "f2"],
    )
    assert registry.try_promote(record) is False
    assert registry.has_active_model() is False


def test_worse_model_does_not_replace_active_one(tmp_path) -> None:
    registry = ModelRegistry(base_dir=tmp_path)
    first = registry.register(
        model=DummyModel("first"),
        metrics=ModelMetrics(auc_val=0.60, sharpe=0.3, precision_buy=0.6, win_rate=0.55, max_drawdown=0.1),
        fechas={"train_start": "a", "train_end": "b", "val_start": "c", "val_end": "d"},
        feature_names=["f1", "f2"],
    )
    assert registry.try_promote(first) is True

    second = registry.register(
        model=DummyModel("second"),
        metrics=ModelMetrics(auc_val=0.58, sharpe=0.35, precision_buy=0.61, win_rate=0.56, max_drawdown=0.1),
        fechas={"train_start": "a", "train_end": "b", "val_start": "c", "val_end": "d"},
        feature_names=["f3"],
    )
    assert registry.try_promote(second) is False

    meta = json.loads((tmp_path / "active_model_meta.json").read_text(encoding="utf-8"))
    assert meta["feature_names"] == ["f1", "f2"]


def test_multi_asset_registry_resolves_by_base_and_observation_state(tmp_path) -> None:
    btc_dir = tmp_path / "btc_usdt"
    btc_dir.mkdir()
    (btc_dir / "model_btc.json").write_text("{}", encoding="utf-8")
    (btc_dir / "active_model_meta.json").write_text(
        json.dumps(
            {
                "model_id": "model_btc",
                "model_path": str(btc_dir / "model_btc.json"),
                "feature_names": ["f1", "f2"],
            }
        ),
        encoding="utf-8",
    )

    sol_dir = tmp_path / "sol_usdt"
    sol_dir.mkdir()
    (sol_dir / "model_sol.json").write_text("{}", encoding="utf-8")
    (sol_dir / "registry.json").write_text(
        json.dumps(
            [
                {
                    "model_id": "model_sol",
                    "model_path": str(sol_dir / "model_sol.json"),
                    "created_at": "2026-03-18T00:00:00+00:00",
                    "train_start": "a",
                    "train_end": "b",
                    "val_start": "c",
                    "val_end": "d",
                    "metrics": {
                        "auc_val": 0.49,
                        "sharpe": 0.0,
                        "precision_buy": 0.5,
                        "win_rate": 0.5,
                        "max_drawdown": 0.1,
                    },
                    "promoted": False,
                    "feature_names": ["g1"],
                    "n_features": 1,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    registry = MultiAssetModelRegistry(root_dir=tmp_path, execution_allowed_bases=["BTC"])

    btc = registry.resolve_artifact("BTC-USD")
    sol = registry.resolve_artifact("SOL-USD")

    assert btc is not None
    assert btc.registry_key == "btc_usdt"
    assert btc.actionable is True
    assert btc.reason == "ok"

    assert sol is not None
    assert sol.registry_key == "sol_usdt"
    assert sol.actionable is False
    assert sol.reason == "no_promoted_model"


def test_multi_asset_registry_recovers_windows_style_active_path(tmp_path) -> None:
    btc_dir = tmp_path / "btc_usdt"
    btc_dir.mkdir()
    (btc_dir / "model_btc.json").write_text("{}", encoding="utf-8")
    (btc_dir / "active_model_meta.json").write_text(
        json.dumps(
            {
                "model_id": "model_btc",
                "model_path": "C:\\\\models\\\\trained\\\\model_btc.json",
                "feature_names": ["f1", "f2"],
            }
        ),
        encoding="utf-8",
    )

    registry = MultiAssetModelRegistry(root_dir=tmp_path, execution_allowed_bases=["BTC"])
    artifact = registry.resolve_artifact("BTC-USD")

    assert artifact is not None
    assert artifact.actionable is True
    assert artifact.reason == "ok"
    assert artifact.model_path is not None
    assert Path(artifact.model_path).name == "model_btc.json"


def test_multi_asset_registry_tracks_placeholder_assets_without_model(tmp_path) -> None:
    pepe_dir = tmp_path / "pepe_usdt"
    pepe_dir.mkdir()
    (pepe_dir / "registry.json").write_text("[]\n", encoding="utf-8")

    registry = MultiAssetModelRegistry(root_dir=tmp_path, execution_allowed_bases=["PEPE"])
    pepe = registry.resolve_artifact("PEPE-USD")

    assert pepe is not None
    assert pepe.registry_key == "pepe_usdt"
    assert pepe.actionable is False
    assert pepe.reason == "no_model"
