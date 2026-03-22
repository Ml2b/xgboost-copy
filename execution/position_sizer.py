"""Sizing dinamico de posiciones para senales long-only."""

from __future__ import annotations

from dataclasses import dataclass

from config import settings


@dataclass(slots=True)
class PositionSizingDecision:
    """Resultado auditable del calculo de sizing."""

    notional_usd: float
    risk_pct: float
    raw_kelly_fraction: float
    applied_kelly_fraction: float
    confidence_scale: float
    reward_risk_ratio: float
    used_dynamic: bool
    reason: str


class KellyFractionalSizer:
    """Convierte una probabilidad calibrada en notional conservador."""

    def __init__(
        self,
        *,
        enabled: bool = settings.POSITION_SIZER_ENABLED,
        base_notional_usd: float = settings.PILOT_ORDER_NOTIONAL_USD,
        min_notional_usd: float = settings.POSITION_SIZER_MIN_NOTIONAL_USD,
        max_notional_usd: float = settings.POSITION_SIZER_MAX_NOTIONAL_USD,
        max_capital_fraction: float = settings.POSITION_SIZER_MAX_CAPITAL_FRACTION,
        kelly_fraction: float = settings.POSITION_SIZER_KELLY_FRACTION,
        stop_loss_pct: float = settings.POSITION_STOP_LOSS_PCT,
        take_profit_pct: float = settings.POSITION_TAKE_PROFIT_PCT,
        max_risk_per_trade: float = settings.MAX_RISK_PER_TRADE,
    ) -> None:
        self.enabled = bool(enabled)
        self.base_notional_usd = max(0.0, float(base_notional_usd))
        self.min_notional_usd = max(0.0, float(min_notional_usd))
        self.max_notional_usd = max(self.base_notional_usd, float(max_notional_usd))
        self.max_capital_fraction = max(0.0, float(max_capital_fraction))
        self.kelly_fraction = max(0.0, float(kelly_fraction))
        self.stop_loss_pct = max(0.01, float(stop_loss_pct))
        self.take_profit_pct = max(0.01, float(take_profit_pct))
        self.max_risk_per_trade = max(0.0, float(max_risk_per_trade))

    def size(
        self,
        *,
        prob_buy: float,
        buy_threshold: float,
        capital_total: float,
        capital_available: float,
    ) -> PositionSizingDecision:
        """Calcula el notional sugerido sin exceder limites de capital y riesgo."""
        capital_total = max(0.0, float(capital_total))
        capital_available = max(0.0, float(capital_available))
        fallback_notional = self._cap_to_available(
            self.base_notional_usd,
            capital_available,
        )
        reward_risk_ratio = self._reward_risk_ratio()
        stop_distance = self.stop_loss_pct / 100.0
        base_risk_pct = self._risk_pct(fallback_notional, capital_total, stop_distance)

        if not self.enabled:
            return PositionSizingDecision(
                notional_usd=round(fallback_notional, 6),
                risk_pct=round(base_risk_pct, 8),
                raw_kelly_fraction=0.0,
                applied_kelly_fraction=0.0,
                confidence_scale=0.0,
                reward_risk_ratio=round(reward_risk_ratio, 6),
                used_dynamic=False,
                reason="dynamic_sizing_disabled",
            )

        p_hat = min(max(float(prob_buy), 0.0), 0.999999)
        threshold = min(max(float(buy_threshold), 0.0), 0.999999)
        q_hat = 1.0 - p_hat
        raw_kelly = max(0.0, ((p_hat * reward_risk_ratio) - q_hat) / max(reward_risk_ratio, 1e-9))
        confidence_scale = 0.0
        if threshold < 1.0:
            confidence_scale = max(0.0, min(1.0, (p_hat - threshold) / max(1e-9, 1.0 - threshold)))
        applied_kelly = raw_kelly * self.kelly_fraction * confidence_scale

        capital_target = capital_total * min(self.max_capital_fraction, applied_kelly)
        risk_target = capital_total * self.max_risk_per_trade / max(stop_distance, 1e-9)
        desired_notional = max(self.base_notional_usd, self.min_notional_usd, capital_target)
        capped_notional = min(
            desired_notional,
            self.max_notional_usd,
            risk_target if risk_target > 0 else desired_notional,
        )
        final_notional = self._cap_to_available(capped_notional, capital_available)
        used_dynamic = final_notional > (self.base_notional_usd + 1e-9)

        return PositionSizingDecision(
            notional_usd=round(final_notional, 6),
            risk_pct=round(self._risk_pct(final_notional, capital_total, stop_distance), 8),
            raw_kelly_fraction=round(raw_kelly, 8),
            applied_kelly_fraction=round(applied_kelly, 8),
            confidence_scale=round(confidence_scale, 8),
            reward_risk_ratio=round(reward_risk_ratio, 6),
            used_dynamic=used_dynamic,
            reason="kelly_fractional_dynamic" if used_dynamic else "pilot_floor_notional",
        )

    def _cap_to_available(self, target_notional: float, capital_available: float) -> float:
        if capital_available <= 0:
            return 0.0
        return max(0.0, min(float(target_notional), capital_available))

    def _reward_risk_ratio(self) -> float:
        stop_distance = max(self.stop_loss_pct / 100.0, 1e-9)
        take_profit_distance = max(self.take_profit_pct / 100.0, stop_distance)
        return float(take_profit_distance / stop_distance)

    @staticmethod
    def _risk_pct(notional_usd: float, capital_total: float, stop_distance: float) -> float:
        if capital_total <= 0 or notional_usd <= 0:
            return 0.0
        return float((notional_usd * stop_distance) / capital_total)
