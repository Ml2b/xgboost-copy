"""Tests del root paper consolidado para el universo estudiado."""

from __future__ import annotations

import json
from pathlib import Path

from config import settings
from scripts.bootstrap_studied_universe_paper_root import bootstrap_studied_universe_paper_root


def test_studied_universe_keeps_zec_as_experiment() -> None:
    assert "ZEC" not in settings.STUDIED_UNIVERSE_BASES
    assert "ZEC" in settings.STUDIED_UNIVERSE_EXPERIMENT_BASES


def test_bootstrap_studied_universe_copies_promoted_and_candidate(tmp_path, monkeypatch) -> None:
    live_root = tmp_path / "live"
    phase2_root = tmp_path / "phase2"
    target_root = tmp_path / "paper"
    live_root.mkdir()
    phase2_root.mkdir()

    _make_promoted_asset(live_root, "btc_usdt", "btc_model.json", ["f1", "f2"])
    _make_candidate_asset(phase2_root, "xlm_usd", "xlm_model.json", ["g1"])

    monkeypatch.setattr(settings, "STUDIED_UNIVERSE_BASES", ["BTC", "XLM"])
    monkeypatch.setattr(settings, "STUDIED_UNIVERSE_EXPERIMENT_BASES", ["ZEC"])
    monkeypatch.setattr(settings, "CORE_BASES", ["BTC"])
    monkeypatch.setattr(settings, "PHASE_ONE_BASES", [])

    manifest = bootstrap_studied_universe_paper_root(
        target_root=target_root,
        live_root=live_root,
        phase2_root=phase2_root,
    )

    assert [asset["base"] for asset in manifest["assets"]] == ["BTC", "XLM"]
    assert manifest["excluded_experiment_bases"] == ["ZEC"]

    btc_meta = json.loads((target_root / "btc_usdt" / "active_model_meta.json").read_text(encoding="utf-8"))
    xlm_meta = json.loads((target_root / "xlm_usd" / "active_model_meta.json").read_text(encoding="utf-8"))

    assert Path(btc_meta["model_path"]).name == "btc_model.json"
    assert Path(xlm_meta["model_path"]).name == "xlm_model.json"
    assert (target_root / "xlm_usd" / "xlm_model.json").exists()
    assert manifest["assets"][0]["activation_source"] == "promoted"
    assert manifest["assets"][1]["activation_source"] == "candidate_latest"


def _make_promoted_asset(root: Path, registry_key: str, model_name: str, features: list[str]) -> None:
    asset_dir = root / registry_key
    asset_dir.mkdir()
    model_path = asset_dir / model_name
    model_path.write_text("{}", encoding="utf-8")
    (asset_dir / "registry.json").write_text(
        json.dumps(
            [
                {
                    "model_id": model_name.removesuffix(".json"),
                    "model_path": str(model_path),
                    "created_at": "2026-03-18T00:00:00+00:00",
                    "train_start": "a",
                    "train_end": "b",
                    "val_start": "c",
                    "val_end": "d",
                    "metrics": {
                        "auc_val": 0.58,
                        "sharpe": 0.3,
                        "precision_buy": 0.6,
                        "win_rate": 0.55,
                        "max_drawdown": 0.1,
                    },
                    "promoted": True,
                    "feature_names": features,
                    "n_features": len(features),
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (asset_dir / "active_model_meta.json").write_text(
        json.dumps(
            {
                "model_id": model_name.removesuffix(".json"),
                "model_path": str(model_path),
                "feature_names": features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _make_candidate_asset(root: Path, registry_key: str, model_name: str, features: list[str]) -> None:
    asset_dir = root / registry_key
    asset_dir.mkdir()
    model_path = asset_dir / model_name
    model_path.write_text("{}", encoding="utf-8")
    (asset_dir / "registry.json").write_text(
        json.dumps(
            [
                {
                    "model_id": model_name.removesuffix(".json"),
                    "model_path": str(model_path),
                    "created_at": "2026-03-18T00:00:00+00:00",
                    "train_start": "a",
                    "train_end": "b",
                    "val_start": "c",
                    "val_end": "d",
                    "metrics": {
                        "auc_val": 0.54,
                        "sharpe": 0.1,
                        "precision_buy": 0.55,
                        "win_rate": 0.51,
                        "max_drawdown": 0.2,
                    },
                    "promoted": False,
                    "feature_names": features,
                    "n_features": len(features),
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
