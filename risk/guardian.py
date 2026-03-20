"""Checkpoints de riesgo antes de aceptar una senal."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from config import settings


@dataclass(slots=True)
class Portfolio:
    """Estado minimo del portfolio para checks de riesgo."""

    capital_total: float
    capital_disponible: float
    drawdown_hoy: float
    posiciones_abiertas: int


class RiskGuardian:
    """Valida una senal contra limites operativos y de riesgo."""

    def __init__(
        self,
        max_risk_per_trade: float = settings.MAX_RISK_PER_TRADE,
        max_daily_drawdown: float = settings.MAX_DAILY_DRAWDOWN,
        max_spread_pct: float = settings.MAX_SPREAD_PCT,
        min_signal_prob: float = settings.MIN_SIGNAL_PROB,
        macro_event_times_utc: list[str] | None = None,
    ) -> None:
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.max_spread_pct = max_spread_pct
        self.min_signal_prob = min_signal_prob
        self.macro_event_times_utc = macro_event_times_utc or settings.MACRO_EVENT_TIMES_UTC

    def check(self, signal: dict[str, float | str], product_id: str, portfolio: Portfolio) -> tuple[bool, str]:
        """Retorna si la senal se puede ejecutar."""
        risk_pct = float(signal.get("risk_pct", 0.0))
        prob_buy = float(signal.get("prob_buy", 0.0))
        spread_pct = float(signal.get("spread_pct", 0.0))
        signal_ts_ms = int(signal.get("timestamp_ms", 0) or 0)
        signal_name = str(signal.get("signal", "BUY")).upper()
        is_exit_signal = signal_name in {"SELL", "EXIT_LONG"}

        # Los cierres protectivos no deben quedar bloqueados por la misma
        # logica de entrada. Si ya existe posicion, priorizamos salir.
        if is_exit_signal:
            return True, "OK"

        confidence = prob_buy
        if risk_pct > self.max_risk_per_trade:
            return False, "CAPITAL: riesgo por trade excedido"
        if portfolio.drawdown_hoy >= self.max_daily_drawdown:
            return False, "DRAWDOWN DIARIO: limite alcanzado"
        if spread_pct > self.max_spread_pct:
            return False, f"SPREAD: spread alto para {product_id}"
        if confidence < self.min_signal_prob:
            return False, "CALIDAD SENAL: probabilidad insuficiente"
        if self._is_in_macro_window(signal_ts_ms):
            return False, "EVENTO MACRO: ventana bloqueada"
        return True, "OK"

    def _is_in_macro_window(self, timestamp_ms: int) -> bool:
        if not self.macro_event_times_utc or timestamp_ms <= 0:
            return False

        signal_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        half_window = timedelta(minutes=30)
        for event_iso in self.macro_event_times_utc:
            event_time = datetime.fromisoformat(event_iso.replace("Z", "+00:00"))
            if abs(signal_time - event_time) <= half_window:
                return True
        return False
