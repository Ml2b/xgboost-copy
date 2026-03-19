"""Servicio de paper trading controlado sobre inference.signals."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from config import settings
from risk.guardian import Portfolio, RiskGuardian


@dataclass(slots=True)
class PaperPosition:
    """Posicion spot paper abierta por activo."""

    product_id: str
    quantity: float
    entry_price: float
    entry_notional: float
    entry_fee: float
    opened_at_ms: int
    model_id: str


@dataclass(slots=True)
class PaperPortfolioState:
    """Estado consolidado del portfolio paper."""

    cash: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    drawdown_pct: float
    open_positions: int


class PaperTrader:
    """Consume señales y simula ejecuciones spot sin tocar dinero real."""

    def __init__(
        self,
        redis_client: Any,
        guardian: RiskGuardian | None = None,
        initial_cash: float = settings.PAPER_INITIAL_CASH,
        order_notional_usd: float = settings.PILOT_ORDER_NOTIONAL_USD,
        fee_pct: float = settings.PAPER_FEE_PCT,
        slippage_pct: float = settings.PAPER_SLIPPAGE_PCT,
        slippage_pct_by_base: dict[str, float] | None = None,
    ) -> None:
        self.redis_client = redis_client
        self.guardian = guardian or RiskGuardian()
        self.initial_cash = float(initial_cash)
        self.order_notional_usd = float(order_notional_usd)
        self.fee_pct = float(fee_pct)
        self.slippage_pct = float(slippage_pct)
        self.slippage_pct_by_base = {
            base.strip().upper(): float(value)
            for base, value in (slippage_pct_by_base or {}).items()
            if base.strip()
        }
        self.cash = float(initial_cash)
        self.realized_pnl = 0.0
        self.peak_equity = float(initial_cash)
        self.positions: dict[str, PaperPosition] = {}
        self.last_close_by_product: dict[str, float] = {}
        self.processed_signals = 0
        self.buy_fills = 0
        self.sell_fills = 0
        self._last_runtime_log_at = 0.0

    async def start(self, stop_event: asyncio.Event | None = None) -> None:
        """Arranca consumidores de velas y señales en paralelo."""
        stop_event = stop_event or asyncio.Event()
        logger.info(
            "PaperTrader iniciado. initial_cash={} order_notional_usd={} fee_pct={} slippage_pct={}",
            self.initial_cash,
            self.order_notional_usd,
            self.fee_pct,
            self.slippage_pct,
        )
        candle_task = asyncio.create_task(self._consume_candles(stop_event))
        signal_task = asyncio.create_task(self._consume_signals(stop_event))
        try:
            await asyncio.gather(candle_task, signal_task)
        finally:
            candle_task.cancel()
            signal_task.cancel()
            await asyncio.gather(candle_task, signal_task, return_exceptions=True)

    async def _consume_candles(self, stop_event: asyncio.Event) -> None:
        await self._ensure_group(settings.STREAM_MARKET_CANDLES_1M, "paper-trader-candles")
        while not stop_event.is_set():
            messages = await self.redis_client.xreadgroup(
                groupname="paper-trader-candles",
                consumername="paper-candles-1",
                streams={settings.STREAM_MARKET_CANDLES_1M: ">"},
                count=50,
                block=1000,
            )
            for _, stream_messages in messages:
                for message_id, payload in stream_messages:
                    self.handle_candle(payload)
                    await self.redis_client.xack(settings.STREAM_MARKET_CANDLES_1M, "paper-trader-candles", message_id)

    async def _consume_signals(self, stop_event: asyncio.Event) -> None:
        await self._ensure_group(settings.STREAM_INFERENCE_SIGNALS, "paper-trader-signals")
        while not stop_event.is_set():
            messages = await self.redis_client.xreadgroup(
                groupname="paper-trader-signals",
                consumername="paper-signals-1",
                streams={settings.STREAM_INFERENCE_SIGNALS: ">"},
                count=50,
                block=1000,
            )
            for _, stream_messages in messages:
                for message_id, payload in stream_messages:
                    event = self.handle_signal(payload)
                    await self._publish_event(event)
                    await self.redis_client.xack(settings.STREAM_INFERENCE_SIGNALS, "paper-trader-signals", message_id)
                    self._log_runtime_summary()

    def handle_candle(self, payload: dict[str, Any]) -> None:
        """Actualiza el ultimo close visto por producto."""
        product_id = str(payload.get("product_id", "")).strip().upper()
        if not product_id:
            return
        try:
            self.last_close_by_product[product_id] = float(payload["close"])
        except Exception:
            return

    def handle_signal(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Procesa una señal paper y retorna el evento auditable."""
        started = time.perf_counter()
        self.processed_signals += 1
        product_id = str(payload.get("product_id", "")).strip().upper()
        signal = str(payload.get("signal", "HOLD")).upper()
        prob_buy = float(payload.get("prob_buy", 0.5))
        actionable = self._to_bool(payload.get("actionable", False))
        model_id = str(payload.get("model_id", ""))
        registry_key = str(payload.get("registry_key", ""))
        timestamp_ms = int(time.time() * 1000)

        if signal == "HOLD":
            return self._base_event(
                product_id=product_id,
                signal=signal,
                decision="ignored_hold",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=actionable,
                latency_ms=self._elapsed_ms(started),
            )

        if not actionable:
            return self._base_event(
                product_id=product_id,
                signal=signal,
                decision="ignored_non_actionable",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                reason=str(payload.get("reason", "non_actionable")),
                latency_ms=self._elapsed_ms(started),
            )

        price = self.last_close_by_product.get(product_id)
        if price is None or price <= 0:
            return self._base_event(
                product_id=product_id,
                signal=signal,
                decision="blocked_no_market_price",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                latency_ms=self._elapsed_ms(started),
            )

        if signal == "BUY":
            return self._handle_buy(
                product_id=product_id,
                price=price,
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                timestamp_ms=timestamp_ms,
                started=started,
            )
        return self._handle_sell(
            product_id=product_id,
            price=price,
            prob_buy=prob_buy,
            model_id=model_id,
            registry_key=registry_key,
            timestamp_ms=timestamp_ms,
            started=started,
        )

    def _handle_buy(
        self,
        product_id: str,
        price: float,
        prob_buy: float,
        model_id: str,
        registry_key: str,
        timestamp_ms: int,
        started: float,
    ) -> dict[str, Any]:
        if product_id in self.positions:
            return self._base_event(
                product_id=product_id,
                signal="BUY",
                decision="blocked_existing_position",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                latency_ms=self._elapsed_ms(started),
            )

        equity = self.current_state().equity
        portfolio = Portfolio(
            capital_total=equity,
            capital_disponible=self.cash,
            drawdown_hoy=self.current_state().drawdown_pct,
            posiciones_abiertas=len(self.positions),
        )
        risk_pct = self.order_notional_usd / max(equity, 1e-9)
        allowed, reason = self.guardian.check(
            {
                "signal": "BUY",
                "risk_pct": risk_pct,
                "prob_buy": prob_buy,
                "spread_pct": 0.0,
                "timestamp_ms": timestamp_ms,
            },
            product_id,
            portfolio,
        )
        if not allowed:
            return self._base_event(
                product_id=product_id,
                signal="BUY",
                decision="blocked_risk",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                reason=reason,
                latency_ms=self._elapsed_ms(started),
            )

        fee_rate = self.fee_pct / 100.0
        applied_slippage_pct = self._slippage_for_product(product_id)
        execution_price = price * (1.0 + applied_slippage_pct / 100.0)
        fee = self.order_notional_usd * fee_rate
        total_cost = self.order_notional_usd + fee
        if total_cost > self.cash:
            return self._base_event(
                product_id=product_id,
                signal="BUY",
                decision="blocked_no_cash",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                reason="Cash insuficiente para abrir posicion paper",
                latency_ms=self._elapsed_ms(started),
            )

        quantity = self.order_notional_usd / execution_price
        self.cash -= total_cost
        self.positions[product_id] = PaperPosition(
            product_id=product_id,
            quantity=quantity,
            entry_price=execution_price,
            entry_notional=self.order_notional_usd,
            entry_fee=fee,
            opened_at_ms=timestamp_ms,
            model_id=model_id,
        )
        self.buy_fills += 1
        state = self.current_state()
        logger.info(
            "PaperTrader BUY filled. product_id={} qty={} fill_price={} cash={} equity={}",
            product_id,
            round(quantity, 12),
            round(execution_price, 8),
            round(state.cash, 8),
            round(state.equity, 8),
        )
        return self._base_event(
            product_id=product_id,
            signal="BUY",
            decision="paper_buy_filled",
            prob_buy=prob_buy,
            model_id=model_id,
            registry_key=registry_key,
            actionable=True,
            fill_price=round(execution_price, 8),
            quantity=round(quantity, 12),
            fee=round(fee, 8),
            applied_slippage_pct=round(applied_slippage_pct, 6),
            cash=round(state.cash, 8),
            equity=round(state.equity, 8),
            drawdown_pct=round(state.drawdown_pct, 8),
            latency_ms=self._elapsed_ms(started),
        )

    def _handle_sell(
        self,
        product_id: str,
        price: float,
        prob_buy: float,
        model_id: str,
        registry_key: str,
        timestamp_ms: int,
        started: float,
    ) -> dict[str, Any]:
        position = self.positions.get(product_id)
        if position is None:
            return self._base_event(
                product_id=product_id,
                signal="SELL",
                decision="blocked_no_position",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                latency_ms=self._elapsed_ms(started),
            )

        fee_rate = self.fee_pct / 100.0
        applied_slippage_pct = self._slippage_for_product(product_id)
        execution_price = price * (1.0 - applied_slippage_pct / 100.0)
        gross_notional = position.quantity * execution_price
        fee = gross_notional * fee_rate
        proceeds = gross_notional - fee
        cost_basis = position.entry_notional + position.entry_fee
        realized_pnl = proceeds - cost_basis

        self.cash += proceeds
        self.realized_pnl += realized_pnl
        self.positions.pop(product_id, None)
        self.sell_fills += 1
        state = self.current_state()
        logger.info(
            "PaperTrader SELL filled. product_id={} qty={} fill_price={} realized_pnl={} cash={} equity={}",
            product_id,
            round(position.quantity, 12),
            round(execution_price, 8),
            round(realized_pnl, 8),
            round(state.cash, 8),
            round(state.equity, 8),
        )
        return self._base_event(
            product_id=product_id,
            signal="SELL",
            decision="paper_sell_filled",
            prob_buy=prob_buy,
            model_id=model_id,
            registry_key=registry_key,
            actionable=True,
            fill_price=round(execution_price, 8),
            quantity=round(position.quantity, 12),
            fee=round(fee, 8),
            applied_slippage_pct=round(applied_slippage_pct, 6),
            realized_pnl=round(realized_pnl, 8),
            holding_ms=timestamp_ms - position.opened_at_ms,
            cash=round(state.cash, 8),
            equity=round(state.equity, 8),
            drawdown_pct=round(state.drawdown_pct, 8),
            latency_ms=self._elapsed_ms(started),
        )

    def current_state(self) -> PaperPortfolioState:
        """Calcula equity y drawdown marcando a mercado las posiciones abiertas."""
        unrealized = 0.0
        for product_id, position in self.positions.items():
            last_price = self.last_close_by_product.get(product_id, position.entry_price)
            unrealized += (last_price - position.entry_price) * position.quantity
        equity = self.cash + sum(
            position.quantity * self.last_close_by_product.get(product_id, position.entry_price)
            for product_id, position in self.positions.items()
        )
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = 0.0 if self.peak_equity <= 0 else max(0.0, (self.peak_equity - equity) / self.peak_equity)
        return PaperPortfolioState(
            cash=self.cash,
            equity=equity,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized,
            drawdown_pct=drawdown,
            open_positions=len(self.positions),
        )

    async def _publish_event(self, payload: dict[str, Any]) -> None:
        await self.redis_client.xadd(
            settings.STREAM_PAPER_EXECUTION_EVENTS,
            {key: str(value) for key, value in payload.items()},
        )

    async def _ensure_group(self, stream: str, group_name: str) -> None:
        try:
            await self.redis_client.xgroup_create(stream, group_name, id="$", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def _log_runtime_summary(self) -> None:
        """Resume el paper trading para observacion remota."""
        now = time.time()
        if now - self._last_runtime_log_at < settings.RUNTIME_LOG_INTERVAL_SECONDS:
            return
        self._last_runtime_log_at = now
        state = self.current_state()
        logger.info(
            "PaperTrader resumen: signals={} buys={} sells={} open_positions={} cash={} equity={} realized_pnl={} unrealized_pnl={}",
            self.processed_signals,
            self.buy_fills,
            self.sell_fills,
            state.open_positions,
            round(state.cash, 8),
            round(state.equity, 8),
            round(state.realized_pnl, 8),
            round(state.unrealized_pnl, 8),
        )

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() == "true"

    def _slippage_for_product(self, product_id: str) -> float:
        base_asset = product_id.strip().split("-", 1)[0].upper()
        return float(self.slippage_pct_by_base.get(base_asset, self.slippage_pct))

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000.0, 3)

    @staticmethod
    def _base_event(
        product_id: str,
        signal: str,
        decision: str,
        prob_buy: float,
        model_id: str,
        registry_key: str,
        actionable: bool,
        latency_ms: float,
        reason: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        payload = {
            "product_id": product_id,
            "signal": signal,
            "decision": decision,
            "prob_buy": round(prob_buy, 6),
            "model_id": model_id,
            "registry_key": registry_key,
            "actionable": str(actionable).lower(),
            "reason": reason,
            "latency_ms": latency_ms,
        }
        payload.update(extra)
        return payload
