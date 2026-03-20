"""Politica simple de salida para posiciones long."""

from __future__ import annotations

from dataclasses import dataclass

from config import settings


@dataclass(slots=True)
class ExitDecision:
    """Resultado de evaluar una posicion contra reglas de salida."""

    should_exit: bool
    reason: str = ""
    pnl_pct: float = 0.0
    holding_minutes: float = 0.0


class PositionExitPolicy:
    """Aplica stop loss, take profit y tiempo maximo de tenencia."""

    def __init__(
        self,
        stop_loss_pct: float = settings.POSITION_STOP_LOSS_PCT,
        take_profit_pct: float = settings.POSITION_TAKE_PROFIT_PCT,
        max_hold_minutes: int = settings.POSITION_MAX_HOLD_MINUTES,
    ) -> None:
        self.stop_loss_pct = max(0.0, float(stop_loss_pct))
        self.take_profit_pct = max(0.0, float(take_profit_pct))
        self.max_hold_minutes = max(0, int(max_hold_minutes))

    def evaluate(
        self,
        *,
        entry_price: float,
        opened_at_ms: int,
        current_price: float,
        now_ms: int,
    ) -> ExitDecision:
        """Evalua si corresponde forzar salida de una posicion abierta."""
        if entry_price <= 0 or current_price <= 0 or now_ms <= 0 or opened_at_ms <= 0:
            return ExitDecision(should_exit=False)

        pnl_pct = ((current_price / entry_price) - 1.0) * 100.0
        holding_minutes = max(0.0, (now_ms - opened_at_ms) / 60_000.0)

        if self.stop_loss_pct > 0.0 and pnl_pct <= -self.stop_loss_pct:
            return ExitDecision(
                should_exit=True,
                reason="stop_loss_hit",
                pnl_pct=pnl_pct,
                holding_minutes=holding_minutes,
            )
        if self.take_profit_pct > 0.0 and pnl_pct >= self.take_profit_pct:
            return ExitDecision(
                should_exit=True,
                reason="take_profit_hit",
                pnl_pct=pnl_pct,
                holding_minutes=holding_minutes,
            )
        if self.max_hold_minutes > 0 and holding_minutes >= self.max_hold_minutes:
            return ExitDecision(
                should_exit=True,
                reason="max_hold_hit",
                pnl_pct=pnl_pct,
                holding_minutes=holding_minutes,
            )
        return ExitDecision(
            should_exit=False,
            pnl_pct=pnl_pct,
            holding_minutes=holding_minutes,
        )
