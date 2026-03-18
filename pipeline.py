"""Pipeline offline para evaluar y entrenar sobre historico."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import settings
from features.calculator import FeatureCalculator
from features.selector import SelectorConfig
from model.registry import ModelRegistry
from model.trainer import Trainer
from target.builder import TargetBuilder, TargetConfig, TargetType
from validation.walk_forward import WalkForwardConfig


@dataclass(slots=True)
class PipelineConfig:
    """Configuracion del pipeline offline."""

    target_horizon: int = settings.TARGET_HORIZON
    target_threshold_pct: float = settings.TARGET_THRESHOLD_PCT
    fee_pct: float = settings.FEE_PCT
    registry_dir: str = "models"


class FeaturePipeline:
    """Ejecuta el entrenamiento offline completo sobre un DataFrame historico."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.registry = ModelRegistry(base_dir=self.config.registry_dir)

    def run(self, df_candles: pd.DataFrame):
        """Corre un ciclo completo de entrenamiento y validacion."""
        trainer = Trainer(
            registry=self.registry,
            data_loader=lambda: df_candles.copy(),
            calculator=FeatureCalculator(),
            target_builder=TargetBuilder(
                TargetConfig(
                    target_type=TargetType.NET_RETURN_THRESHOLD,
                    horizon=self.config.target_horizon,
                    threshold_pct=self.config.target_threshold_pct,
                    fee_pct=self.config.fee_pct,
                )
            ),
            walk_forward_config=WalkForwardConfig(verbose=False),
            selector_config=SelectorConfig(verbose=False),
        )
        return trainer._retrain_cycle()

