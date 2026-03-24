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
    FeatureSpec("signed_return_1_volume", "returns", "Retorno inmediato ponderado por flujo de volumen."),
    FeatureSpec("rsi_14", "momentum", "Sobrecompra y sobreventa suavizadas."),
    FeatureSpec("rsi_divergence", "momentum", "Agotamiento de precio vs RSI."),
    FeatureSpec("macd_hist", "momentum", "Aceleracion y desaceleracion del impulso."),
    FeatureSpec("macd_cross_up", "momentum", "Cambio alcista de momentum."),
    FeatureSpec("macd_cross_down", "momentum", "Cambio bajista de momentum."),
    FeatureSpec("atr_pct", "volatility", "Rango esperado normalizado por precio."),
    FeatureSpec("bb_pct", "volatility", "Posicion del precio en Bollinger."),
    FeatureSpec("bb_width", "volatility", "Compresion o expansion de bandas."),
    FeatureSpec("realized_vol_5", "volatility", "Volatilidad de reaccion rapida a micro-shocks."),
    FeatureSpec("realized_vol_20", "volatility", "Volatilidad realizada reciente."),
    FeatureSpec("price_vs_ema_fast", "trend", "Distancia al EMA rapido."),
    FeatureSpec("price_vs_ema_slow", "trend", "Distancia al EMA lento."),
    FeatureSpec("price_vs_ema_trend", "trend", "Distancia al EMA tendencial."),
    FeatureSpec("ema_fast_above_slow", "trend", "Tendencia corta binaria."),
    FeatureSpec("ema_full_alignment", "trend", "Alineacion completa de EMAs."),
    FeatureSpec("vwap_20_distance", "structure", "Desviacion del cierre respecto al VWAP rolling."),
    FeatureSpec("position_in_range_20", "structure", "Ubicacion relativa en el rango."),
    FeatureSpec("range_compression_20", "structure", "Compresion del rango actual frente a su media."),
    FeatureSpec("trend_slope_pct", "trend", "Pendiente lineal normalizada."),
    FeatureSpec("volume_ratio", "volume", "Volumen relativo a su media."),
    FeatureSpec("volume_spike", "volume", "Evento de volumen extremo."),
    FeatureSpec("volume_trend", "volume", "Pendiente de volumen."),
    FeatureSpec("trade_intensity_20", "volume", "Intensidad de ejecucion en ventana corta."),
    FeatureSpec("hl_ratio", "microstructure", "Rango high-low relativo."),
    FeatureSpec("close_position_in_candle", "microstructure", "Cierre dentro de la vela."),
    FeatureSpec("body_to_range_ratio", "microstructure", "Cuerpo relativo al rango."),
    FeatureSpec("hour_sin", "temporal", "Hora del dia codificada sin saltos."),
    FeatureSpec("hour_cos", "temporal", "Complemento ciclico de la hora."),
    FeatureSpec("day_of_week_sin", "temporal", "Patron semanal ciclico."),
    FeatureSpec("day_of_week_cos", "temporal", "Complemento ciclico semanal."),
    # Group 1: flujo ejecutado (raw trades)
    FeatureSpec("buy_aggressive_volume", "order_flow", "Volumen comprador agresivo en la ventana."),
    FeatureSpec("sell_aggressive_volume", "order_flow", "Volumen vendedor agresivo en la ventana."),
    FeatureSpec("total_traded_volume", "order_flow", "Volumen total ejecutado en la ventana."),
    FeatureSpec("trade_count", "order_flow", "Cantidad de trades ejecutados."),
    FeatureSpec("buy_count", "order_flow", "Numero de trades compradores."),
    FeatureSpec("sell_count", "order_flow", "Numero de trades vendedores."),
    FeatureSpec("avg_trade_size", "order_flow", "Tamano medio por trade."),
    FeatureSpec("median_trade_size", "order_flow", "Mediana robusta del tamano por trade."),
    FeatureSpec("max_trade_size", "order_flow", "Print maximo detectado en la ventana."),
    FeatureSpec("volume_delta", "order_flow", "Delta neto buy minus sell agresivo."),
    FeatureSpec("delta_ratio", "order_flow", "Delta normalizado por volumen total."),
    FeatureSpec("buy_sell_ratio", "order_flow", "Ratio buy/sell agresivo, clip [0,10]."),
    FeatureSpec("trade_vwap", "order_flow", "VWAP de ticks ejecutados en la ventana."),
    # Group 4: spread y calidad del mercado (L2 top of book)
    FeatureSpec("spread", "order_flow", "Spread absoluto best_ask - best_bid."),
    FeatureSpec("relative_spread", "order_flow", "Spread relativo normalizado por mid price."),
    FeatureSpec("mid_price", "order_flow", "Precio medio del libro en el cierre."),
    FeatureSpec("best_bid_size", "order_flow", "Tamano en el mejor bid."),
    FeatureSpec("best_ask_size", "order_flow", "Tamano en el mejor ask."),
    FeatureSpec("top_of_book_imbalance", "order_flow", "Desequilibrio bid/ask en top of book."),
    FeatureSpec("microprice", "order_flow", "Precio ponderado por tamano de best bid/ask."),
    FeatureSpec("microprice_shift", "order_flow", "Desviacion del microprecio respecto al mid."),
    # Group 3: libro de ordenes (L2 profundidad)
    FeatureSpec("bid_depth_top5", "order_flow", "Volumen acumulado en los 5 mejores bids."),
    FeatureSpec("bid_depth_top10", "order_flow", "Volumen acumulado en los 10 mejores bids."),
    FeatureSpec("bid_depth_top20", "order_flow", "Volumen acumulado en los 20 mejores bids."),
    FeatureSpec("ask_depth_top5", "order_flow", "Volumen acumulado en los 5 mejores asks."),
    FeatureSpec("ask_depth_top10", "order_flow", "Volumen acumulado en los 10 mejores asks."),
    FeatureSpec("ask_depth_top20", "order_flow", "Volumen acumulado en los 20 mejores asks."),
    FeatureSpec("depth_imbalance_top10", "order_flow", "Desequilibrio bid/ask en top 10 niveles."),
    FeatureSpec("depth_slope_bid", "order_flow", "Pendiente de la curva de profundidad bid."),
    FeatureSpec("depth_slope_ask", "order_flow", "Pendiente de la curva de profundidad ask."),
    FeatureSpec("cumul_depth_1bp", "order_flow", "Liquidez acumulada a 1bp del mid."),
    FeatureSpec("cumul_depth_5bp", "order_flow", "Liquidez acumulada a 5bp del mid."),
    FeatureSpec("cumul_depth_10bp", "order_flow", "Liquidez acumulada a 10bp del mid."),
    # Groups 5-7: consumo, absorcion, reposicion (L2 + trades)
    FeatureSpec("ask_consumption_rate", "order_flow", "Fraccion del ask top10 consumida por compras."),
    FeatureSpec("bid_consumption_rate", "order_flow", "Fraccion del bid top10 consumida por ventas."),
    FeatureSpec("net_consumption", "order_flow", "Consumo neto ask minus bid."),
    FeatureSpec("price_move_per_unit_flow", "order_flow", "Movimiento de precio por unidad de flujo neto."),
    FeatureSpec("seller_absorption_estimate", "order_flow", "Estimacion de absorcion vendedora."),
    FeatureSpec("buyer_absorption_estimate", "order_flow", "Estimacion de absorcion compradora."),
    FeatureSpec("absorption_score", "order_flow", "Score compuesto de absorcion del libro."),
    FeatureSpec("flow_to_price_divergence", "order_flow", "Divergencia entre flujo y movimiento de precio."),
    FeatureSpec("refill_rate_ask", "order_flow", "Tasa de reposicion del lado ask."),
    FeatureSpec("refill_rate_bid", "order_flow", "Tasa de reposicion del lado bid."),
    FeatureSpec("cancel_rate_ask", "order_flow", "Tasa de cancelacion del lado ask."),
    FeatureSpec("cancel_rate_bid", "order_flow", "Tasa de cancelacion del lado bid."),
    # Groups 2+11: derivados de trades + precio
    FeatureSpec("cvd_5", "order_flow", "CVD acumulado en 5 velas."),
    FeatureSpec("cvd_15", "order_flow", "CVD acumulado en 15 velas."),
    FeatureSpec("delta_acceleration", "order_flow", "Aceleracion del delta vs velas previas."),
    FeatureSpec("volume_delta_normalized", "order_flow", "Delta normalizado por volumen total."),
    FeatureSpec("trade_size_skew", "order_flow", "Asimetria avg/median del tamano de trade."),
    FeatureSpec("buy_pressure_pct", "order_flow", "Porcentaje de trades compradores."),
    FeatureSpec("flow_efficiency", "order_flow", "Movimiento de precio por unidad de volumen."),
    FeatureSpec("net_flow_efficiency", "order_flow", "Movimiento de precio por unidad de delta neto."),
    # Group 8: reaccion del precio
    FeatureSpec("ret_3m", "order_flow", "Retorno de 3 velas (aprox 3 minutos)."),
    FeatureSpec("break_of_local_high", "order_flow", "Distancia porcentual al maximo local reciente."),
    FeatureSpec("break_of_local_low", "order_flow", "Distancia porcentual al minimo local reciente."),
    FeatureSpec("distance_to_recent_high", "order_flow", "Distancia al maximo en ventana local."),
    FeatureSpec("distance_to_recent_low", "order_flow", "Distancia al minimo en ventana local."),
    # Group 10: multi-timeframe
    FeatureSpec("trend_5m", "order_flow", "Tendencia precio vs EMA 5 velas."),
    FeatureSpec("trend_15m", "order_flow", "Tendencia precio vs EMA 15 velas."),
    FeatureSpec("trend_1h", "order_flow", "Tendencia precio vs EMA 60 velas."),
    FeatureSpec("vwap_distance_15m", "order_flow", "Distancia al VWAP rolling 15 velas."),
    FeatureSpec("vwap_distance_1h", "order_flow", "Distancia al VWAP rolling 60 velas."),
    FeatureSpec("vol_state_1h", "order_flow", "Estado de volatilidad en ventana horaria (0/1/2)."),
    # Group 12: regimenes
    FeatureSpec("trend_regime", "order_flow", "Regimen de tendencia (-1 bajista/0 lateral/1 alcista)."),
    FeatureSpec("volatility_regime", "order_flow", "Regimen de volatilidad (0 baja/1 media/2 alta)."),
    FeatureSpec("spread_regime", "order_flow", "Regimen de spread (0 normal/1 ampliado)."),
    FeatureSpec("activity_regime", "order_flow", "Regimen de actividad (0 quieto/1 normal/2 activo)."),
    FeatureSpec("liquidity_regime", "order_flow", "Regimen de liquidez (0 profundo/1 medio/2 delgado)."),
    # Group 13: operativas
    FeatureSpec("current_spread_cost", "order_flow", "Coste del spread como fraccion del precio."),
    FeatureSpec("fee_impact", "order_flow", "Impacto del fee en la operacion."),
    FeatureSpec("execution_quality_estimate", "order_flow", "Estimacion de calidad de ejecucion [0,1]."),
]


def candidate_feature_names() -> list[str]:
    """Retorna la lista de features marcadas como candidatas."""
    return [spec.name for spec in FEATURE_POOL if spec.candidate]
