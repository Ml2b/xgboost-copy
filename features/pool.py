"""Pool candidato de features tecnicas del sistema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Describe una feature candidata y su hipotesis."""

    name: str
    category: str
    hypothesis: str
    risk_note: str = ""
    candidate: bool = True


FEATURE_POOL: list[FeatureSpec] = [
    FeatureSpec("ret_1", "returns", "Momentum inmediato de una vela."),
    FeatureSpec("ret_5", "returns", "Momentum de muy corto plazo."),
    FeatureSpec("ret_15", "returns", "Contexto del mismo horizonte del target."),
    FeatureSpec("ret_30", "returns", "Direccion reciente de media hora."),
    FeatureSpec("ret_60", "returns", "Contexto horario."),
    FeatureSpec("rsi_14", "momentum", "Sobrecompra y sobreventa suavizadas."),
    FeatureSpec("rsi_divergence", "momentum", "Agotamiento de precio vs RSI."),
    FeatureSpec("macd_hist", "momentum", "Aceleracion y desaceleracion del impulso."),
    FeatureSpec("macd_cross_up", "momentum", "Cambio alcista de momentum."),
    FeatureSpec("macd_cross_down", "momentum", "Cambio bajista de momentum."),
    FeatureSpec("atr_pct", "volatility", "Rango esperado normalizado por precio."),
    FeatureSpec("bb_pct", "volatility", "Posicion del precio en Bollinger."),
    FeatureSpec("bb_width", "volatility", "Compresion o expansion de bandas."),
    FeatureSpec("realized_vol_20", "volatility", "Volatilidad realizada reciente."),
    FeatureSpec("price_vs_ema_fast", "trend", "Distancia al EMA rapido."),
    FeatureSpec("price_vs_ema_slow", "trend", "Distancia al EMA lento."),
    FeatureSpec("price_vs_ema_trend", "trend", "Distancia al EMA tendencial."),
    FeatureSpec("ema_fast_above_slow", "trend", "Tendencia corta binaria."),
    FeatureSpec("ema_full_alignment", "trend", "Alineacion completa de EMAs."),
    FeatureSpec("position_in_range_20", "structure", "Ubicacion relativa en el rango."),
    FeatureSpec("trend_slope_pct", "trend", "Pendiente lineal normalizada."),
    FeatureSpec("volume_ratio", "volume", "Volumen relativo a su media."),
    FeatureSpec("volume_spike", "volume", "Evento de volumen extremo."),
    FeatureSpec("volume_trend", "volume", "Pendiente de volumen."),
    FeatureSpec("hl_ratio", "microstructure", "Rango high-low relativo."),
    FeatureSpec("close_position_in_candle", "microstructure", "Cierre dentro de la vela."),
    FeatureSpec("body_to_range_ratio", "microstructure", "Cuerpo relativo al rango."),
    FeatureSpec("hour_sin", "temporal", "Hora del dia codificada sin saltos."),
    FeatureSpec("hour_cos", "temporal", "Complemento ciclico de la hora."),
    FeatureSpec("day_of_week_sin", "temporal", "Patron semanal ciclico."),
    FeatureSpec("day_of_week_cos", "temporal", "Complemento ciclico semanal."),
]


def candidate_feature_names() -> list[str]:
    """Retorna la lista de features marcadas como candidatas."""
    return [spec.name for spec in FEATURE_POOL if spec.candidate]

