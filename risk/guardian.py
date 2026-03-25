"""Checkpoints de riesgo antes de aceptar una senal."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Deque

from config import settings


@dataclass(slots=True)
class Portfolio:
    """Estado minimo del portfolio para checks de riesgo."""

    capital_total: float
    capital_disponible: float
    drawdown_hoy: float
    posiciones_abiertas: int


@dataclass(slots=True)
class AdverseExit:
    """Salida reciente que sugiere mercado serruchado para un activo."""

    timestamp_ms: int
    reason: str
    pnl_pct: float
    holding_minutes: float


@dataclass(slots=True)
class ChopAdjustment:
    """Ajuste tactico de entradas cuando un activo entra en serrucho."""

    cautious: bool
    blocked: bool
    min_signal_prob: float
    max_risk_pct: float
    notional_scale: float
    adverse_exit_count: int
    stop_loss_count: int
    blocked_until_ms: int = 0


class RiskGuardian:
    """Valida una senal contra limites operativos y de riesgo."""

    def __init__(
        self,
        max_risk_per_trade: float = settings.MAX_RISK_PER_TRADE,
        max_daily_drawdown: float = settings.MAX_DAILY_DRAWDOWN,
        max_spread_pct: float = settings.MAX_SPREAD_PCT,
        min_signal_prob: float = settings.MIN_SIGNAL_PROB,
        macro_event_times_utc: list[str] | None = None,
        chop_guard_enabled: bool = settings.CHOP_GUARD_ENABLED,
        chop_window_minutes: int = settings.CHOP_GUARD_WINDOW_MINUTES,
        chop_fast_exit_minutes: int = settings.CHOP_GUARD_FAST_EXIT_MINUTES,
        chop_adverse_pnl_pct: float = settings.CHOP_GUARD_ADVERSE_PNL_PCT,
        chop_caution_exit_count: int = settings.CHOP_GUARD_CAUTION_EXIT_COUNT,
        chop_lock_exit_count: int = settings.CHOP_GUARD_LOCK_EXIT_COUNT,
        chop_lock_stop_count: int = settings.CHOP_GUARD_LOCK_STOP_COUNT,
        chop_probability_buffer: float = settings.CHOP_GUARD_PROBABILITY_BUFFER,
        chop_risk_scale: float = settings.CHOP_GUARD_RISK_SCALE,
        chop_notional_scale: float = settings.CHOP_GUARD_NOTIONAL_SCALE,
        chop_block_minutes: int = settings.CHOP_GUARD_BLOCK_MINUTES,
    ) -> None:
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.max_spread_pct = max_spread_pct
        self.min_signal_prob = min_signal_prob
        self.macro_event_times_utc = macro_event_times_utc or settings.MACRO_EVENT_TIMES_UTC
        self.chop_guard_enabled = bool(chop_guard_enabled)
        self.chop_window_minutes = max(1, int(chop_window_minutes))
        self.chop_fast_exit_minutes = max(1, int(chop_fast_exit_minutes))
        self.chop_adverse_pnl_pct = min(0.0, float(chop_adverse_pnl_pct))
        self.chop_caution_exit_count = max(1, int(chop_caution_exit_count))
        self.chop_lock_exit_count = max(self.chop_caution_exit_count, int(chop_lock_exit_count))
        self.chop_lock_stop_count = max(1, int(chop_lock_stop_count))
        self.chop_probability_buffer = max(0.0, float(chop_probability_buffer))
        self.chop_risk_scale = min(1.0, max(0.0, float(chop_risk_scale)))
        self.chop_notional_scale = min(1.0, max(0.0, float(chop_notional_scale)))
        self.chop_block_minutes = max(1, int(chop_block_minutes))
        self._recent_adverse_exits_by_asset: dict[str, Deque[AdverseExit]] = {}
        self._blocked_until_by_asset: dict[str, int] = {}

    def check(self, signal: dict[str, float | str], product_id: str, portfolio: Portfolio) -> tuple[bool, str]:
        """Retorna si la senal se puede ejecutar."""
        risk_pct = float(signal.get("risk_pct", 0.0))
        prob_buy = float(signal.get("prob_buy", 0.0))
        spread_pct = float(signal.get("spread_pct", 0.0))
        signal_ts_ms = int(signal.get("timestamp_ms", 0) or 0)
        if signal_ts_ms <= 0:
            signal_ts_ms = int(time.time() * 1000)
        signal_name = str(signal.get("signal", "BUY")).upper()
        is_exit_signal = signal_name in {"SELL", "EXIT_LONG"}

        # Los cierres protectivos no deben quedar bloqueados por la misma
        # logica de entrada. Si ya existe posicion, priorizamos salir.
        if is_exit_signal:
            return True, "OK"

        if risk_pct > self.max_risk_per_trade:
            return False, "CAPITAL: riesgo por trade excedido"
        if portfolio.drawdown_hoy >= self.max_daily_drawdown:
            return False, "DRAWDOWN DIARIO: limite alcanzado"
        if spread_pct > self.max_spread_pct:
            return False, f"SPREAD: spread alto para {product_id}"
        chop_adjustment = self.entry_adjustment(product_id, timestamp_ms=signal_ts_ms)
        if chop_adjustment.blocked:
            remaining_minutes = max(1, int((chop_adjustment.blocked_until_ms - signal_ts_ms) / 60_000))
            return (
                False,
                "CHOP: activo en enfriamiento "
                f"({chop_adjustment.stop_loss_count} stops, {chop_adjustment.adverse_exit_count} salidas adversas). Reintentar en {remaining_minutes}m",
            )
        if prob_buy < chop_adjustment.min_signal_prob:
            return (
                False,
                "CHOP: activo serruchado, exigir "
                f"prob_buy>={chop_adjustment.min_signal_prob:.2f} mientras se normaliza",
            )
        if risk_pct > chop_adjustment.max_risk_pct:
            return (
                False,
                "CHOP: activo serruchado, riesgo por trade excede "
                f"modo cautela ({chop_adjustment.max_risk_pct:.4f})",
            )
        if prob_buy < self.min_signal_prob:
            return False, "CALIDAD SENAL: probabilidad insuficiente"
        if self._is_in_macro_window(signal_ts_ms):
            return False, "EVENTO MACRO: ventana bloqueada"
        return True, "OK"

    def register_exit(
        self,
        product_id: str,
        *,
        reason: str,
        pnl_pct: float,
        timestamp_ms: int,
        holding_minutes: float,
    ) -> None:
        """Registra salidas adversas para endurecer entradas en patron serruchado."""
        if not self.chop_guard_enabled:
            return

        normalized_product = str(product_id).strip().upper()
        if not normalized_product:
            return

        exit_timestamp_ms = int(timestamp_ms or 0)
        if exit_timestamp_ms <= 0:
            exit_timestamp_ms = int(time.time() * 1000)
        self._prune_exit_history(normalized_product, exit_timestamp_ms)

        if not self._is_adverse_exit(reason=reason, pnl_pct=pnl_pct, holding_minutes=holding_minutes):
            return

        history = self._recent_adverse_exits_by_asset.setdefault(normalized_product, deque())
        history.append(
            AdverseExit(
                timestamp_ms=exit_timestamp_ms,
                reason=str(reason).strip(),
                pnl_pct=float(pnl_pct),
                holding_minutes=max(0.0, float(holding_minutes)),
            )
        )
        self._prune_exit_history(normalized_product, exit_timestamp_ms)

        stop_loss_count = sum(1 for exit_row in history if exit_row.reason == "stop_loss_hit")
        if len(history) >= self.chop_lock_exit_count or stop_loss_count >= self.chop_lock_stop_count:
            self._blocked_until_by_asset[normalized_product] = max(
                self._blocked_until_by_asset.get(normalized_product, 0),
                exit_timestamp_ms + (self.chop_block_minutes * 60_000),
            )

    def entry_adjustment(self, product_id: str, *, timestamp_ms: int) -> ChopAdjustment:
        """Devuelve el perfil de cautela vigente para nuevas entradas."""
        if not self.chop_guard_enabled:
            return ChopAdjustment(
                cautious=False,
                blocked=False,
                min_signal_prob=self.min_signal_prob,
                max_risk_pct=self.max_risk_per_trade,
                notional_scale=1.0,
                adverse_exit_count=0,
                stop_loss_count=0,
            )

        normalized_product = str(product_id).strip().upper()
        if not normalized_product:
            return ChopAdjustment(
                cautious=False,
                blocked=False,
                min_signal_prob=self.min_signal_prob,
                max_risk_pct=self.max_risk_per_trade,
                notional_scale=1.0,
                adverse_exit_count=0,
                stop_loss_count=0,
            )

        self._prune_exit_history(normalized_product, timestamp_ms)
        blocked_until_ms = self._blocked_until_by_asset.get(normalized_product, 0)
        history = self._recent_adverse_exits_by_asset.get(normalized_product, deque())
        stop_loss_count = sum(1 for exit_row in history if exit_row.reason == "stop_loss_hit")
        cautious = len(history) >= self.chop_caution_exit_count
        return ChopAdjustment(
            cautious=cautious,
            blocked=blocked_until_ms > timestamp_ms,
            min_signal_prob=min(0.999, self.min_signal_prob + self.chop_probability_buffer)
            if cautious
            else self.min_signal_prob,
            max_risk_pct=self.max_risk_per_trade * self.chop_risk_scale if cautious else self.max_risk_per_trade,
            notional_scale=self.chop_notional_scale if cautious else 1.0,
            adverse_exit_count=len(history),
            stop_loss_count=stop_loss_count,
            blocked_until_ms=blocked_until_ms,
        )

    def _is_adverse_exit(self, *, reason: str, pnl_pct: float, holding_minutes: float) -> bool:
        normalized_reason = str(reason).strip().lower()
        if normalized_reason == "stop_loss_hit":
            return True
        if pnl_pct > self.chop_adverse_pnl_pct:
            return False
        return normalized_reason in {"signal_exit_long", "max_hold_hit"} and holding_minutes <= self.chop_fast_exit_minutes

    def _prune_exit_history(self, product_id: str, timestamp_ms: int) -> None:
        history = self._recent_adverse_exits_by_asset.get(product_id)
        if history is None:
            return

        min_timestamp_ms = timestamp_ms - (self.chop_window_minutes * 60_000)
        while history and history[0].timestamp_ms < min_timestamp_ms:
            history.popleft()
        if history:
            return

        self._recent_adverse_exits_by_asset.pop(product_id, None)
        if self._blocked_until_by_asset.get(product_id, 0) <= timestamp_ms:
            self._blocked_until_by_asset.pop(product_id, None)

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
