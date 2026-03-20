"""Consumidor de senales que aplica riesgo y ejecuta en Coinbase."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from loguru import logger

from config import settings
from exchange.coinbase_client import CoinbaseAdvancedTradeClient
from execution.position_exit import PositionExitPolicy
from risk.guardian import Portfolio, RiskGuardian


@dataclass(slots=True)
class ExecutionStats:
    """Metricas basicas del ejecutor."""

    processed: int = 0
    dry_runs: int = 0
    live_orders: int = 0
    rejected: int = 0


@dataclass(slots=True)
class ManagedPosition:
    """Posicion abierta por este proceso para soportar salidas long-only."""

    product_id: str
    entry_price: float
    quantity: float
    opened_at_ms: int
    model_id: str


class OrderManager:
    """Convierte senales elegibles en decisiones de ejecucion auditables."""

    def __init__(
        self,
        redis_client: Any,
        coinbase_client: CoinbaseAdvancedTradeClient,
        guardian: RiskGuardian | None = None,
        execution_enabled: bool = settings.EXECUTION_ENABLED,
        dry_run: bool = settings.EXECUTION_DRY_RUN,
        order_notional_usd: float = settings.PILOT_ORDER_NOTIONAL_USD,
        cooldown_seconds: int = settings.EXECUTION_COOLDOWN_SECONDS,
        allowed_bases: list[str] | None = None,
        drawdown_today: float = 0.0,
        exit_policy: PositionExitPolicy | None = None,
    ) -> None:
        self.redis_client = redis_client
        self.coinbase_client = coinbase_client
        self.guardian = guardian or RiskGuardian()
        self.execution_enabled = execution_enabled
        self.dry_run = dry_run
        self.order_notional_usd = float(order_notional_usd)
        self.cooldown_seconds = int(cooldown_seconds)
        self.allowed_bases = {
            base.strip().upper()
            for base in (allowed_bases or settings.EXECUTION_ALLOWED_BASES)
            if base.strip()
        }
        self.drawdown_today = drawdown_today
        self.exit_policy = exit_policy or PositionExitPolicy()
        self.stats = ExecutionStats()
        self._cooldown_until_by_asset: dict[str, float] = {}
        self._asset_locks: dict[str, asyncio.Lock] = {}
        self._managed_positions: dict[str, ManagedPosition] = {}
        self._state_loaded = False
        self._last_runtime_log_at = 0.0

    async def start(self, stop_event: asyncio.Event | None = None) -> None:
        """Consume inference.signals y publica execution.events."""
        stop_event = stop_event or asyncio.Event()
        logger.info(
            "OrderManager iniciado. execution_enabled={} dry_run={} allowed_bases={}",
            self.execution_enabled,
            self.dry_run,
            sorted(self.allowed_bases),
        )
        await self._ensure_state_loaded()
        await self._ensure_group(settings.STREAM_INFERENCE_SIGNALS, "order-manager")
        while not stop_event.is_set():
            messages = await self.redis_client.xreadgroup(
                groupname="order-manager",
                consumername="order-manager-1",
                streams={settings.STREAM_INFERENCE_SIGNALS: ">"},
                count=25,
                block=1000,
            )
            for _, stream_messages in messages:
                for message_id, payload in stream_messages:
                    event = await self.handle_signal(payload)
                    await self._publish_event(event)
                    await self.redis_client.xack(settings.STREAM_INFERENCE_SIGNALS, "order-manager", message_id)
                    self._log_runtime_summary()

    async def handle_signal(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Evalua una senal y retorna el evento de ejecucion resultante."""
        await self._ensure_state_loaded()
        product_id = str(payload.get("product_id", "")).strip().upper()
        lock = self._asset_locks.setdefault(product_id or "UNKNOWN", asyncio.Lock())
        async with lock:
            return await self._handle_signal_locked(payload)

    async def _handle_signal_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Version serializada por activo para evitar dobles ordenes."""
        started = time.perf_counter()
        self.stats.processed += 1
        product_id = str(payload.get("product_id", "")).strip().upper()
        base_asset = product_id.split("-", 1)[0] if product_id else ""
        signal = str(payload.get("signal", "HOLD")).upper()
        if signal == "SELL":
            signal = "EXIT_LONG"
        prob_buy = float(payload.get("prob_buy", 0.5))
        actionable = self._to_bool(payload.get("actionable", False))
        registry_key = str(payload.get("registry_key", ""))
        model_id = str(payload.get("model_id", ""))
        managed_position = self._managed_positions.get(product_id)

        if signal == "HOLD" and managed_position is None:
            return self._base_event(
                product_id=product_id,
                signal=signal,
                decision="ignored_hold",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=actionable,
                dry_run=self.dry_run,
                latency_ms=self._elapsed_ms(started),
            )

        if signal == "BUY" and self._is_in_cooldown(product_id):
            return self._base_event(
                product_id=product_id,
                signal=signal,
                decision="cooldown_active",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                reason=f"Cooldown activo de {self.cooldown_seconds}s",
                dry_run=self.dry_run,
                latency_ms=self._elapsed_ms(started),
            )

        if signal == "BUY" and not actionable:
            return self._base_event(
                product_id=product_id,
                signal=signal,
                decision="ignored_non_actionable",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                reason=str(payload.get("reason", "non_actionable")),
                dry_run=self.dry_run,
                latency_ms=self._elapsed_ms(started),
            )

        if signal == "BUY" and base_asset not in self.allowed_bases:
            return self._base_event(
                product_id=product_id,
                signal=signal,
                decision="ignored_base_not_enabled",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                dry_run=self.dry_run,
                latency_ms=self._elapsed_ms(started),
            )

        try:
            if self.dry_run:
                balances: dict = {}
                best_bid_ask = self.coinbase_client.get_best_bid_ask(product_id, prefer_private=False)
            else:
                balances = self.coinbase_client.get_account_balances()
                best_bid_ask = self.coinbase_client.get_best_bid_ask(product_id)
            product = self.coinbase_client.get_product_snapshot(product_id)
        except Exception as exc:
            self.stats.rejected += 1
            return self._base_event(
                product_id=product_id,
                signal=signal,
                decision="rejected_exchange",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=actionable,
                reason=str(exc),
                dry_run=self.dry_run,
                latency_ms=self._elapsed_ms(started),
            )

        portfolio = self._build_portfolio(product.quote_asset, balances)
        current_timestamp_ms = int(time.time() * 1000)
        current_exit_price = float(best_bid_ask.get("bid") or best_bid_ask.get("mid") or 0.0)

        if managed_position is not None:
            auto_exit = self.exit_policy.evaluate(
                entry_price=managed_position.entry_price,
                opened_at_ms=managed_position.opened_at_ms,
                current_price=current_exit_price,
                now_ms=current_timestamp_ms,
            )
            if auto_exit.should_exit:
                return await self._handle_sell(
                    product_id=product_id,
                    prob_buy=prob_buy,
                    model_id=model_id,
                    registry_key=registry_key,
                    actionable=True,
                    base_asset=product.base_asset,
                    base_min_size=product.base_min_size,
                    balances=balances,
                    latency_started=started,
                    signal_label="EXIT_LONG",
                    exit_reason=auto_exit.reason,
                    managed_position=managed_position,
                )

        if signal == "HOLD":
            return self._base_event(
                product_id=product_id,
                signal=signal,
                decision="ignored_hold",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                reason="position_open_waiting_exit" if managed_position is not None else "",
                dry_run=self.dry_run,
                latency_ms=self._elapsed_ms(started),
            )

        if signal == "BUY":
            risk_payload = {
                "signal": signal,
                "risk_pct": self._risk_pct(portfolio.capital_total),
                "prob_buy": prob_buy,
                "spread_pct": best_bid_ask["spread_pct"],
                "timestamp_ms": current_timestamp_ms,
            }
            accepted, reason = self.guardian.check(risk_payload, product_id, portfolio)
            if not accepted:
                self.stats.rejected += 1
                return self._base_event(
                    product_id=product_id,
                    signal=signal,
                    decision="blocked_risk",
                    prob_buy=prob_buy,
                    model_id=model_id,
                    registry_key=registry_key,
                    actionable=False,
                    reason=reason,
                    spread_pct=best_bid_ask["spread_pct"],
                    dry_run=self.dry_run,
                    latency_ms=self._elapsed_ms(started),
                )
            return await self._handle_buy(
                product_id=product_id,
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=actionable,
                quote_asset=product.quote_asset,
                base_asset=product.base_asset,
                base_min_size=product.base_min_size,
                balances=balances,
                latency_started=started,
                reference_price=float(best_bid_ask.get("ask") or best_bid_ask.get("mid") or 0.0),
            )

        return await self._handle_sell(
            product_id=product_id,
            prob_buy=prob_buy,
            model_id=model_id,
            registry_key=registry_key,
            actionable=actionable,
            base_asset=product.base_asset,
            base_min_size=product.base_min_size,
            balances=balances,
            latency_started=started,
            signal_label=signal,
            exit_reason=str(payload.get("reason", "signal_exit_long")),
            managed_position=managed_position,
        )

    async def _handle_buy(
        self,
        product_id: str,
        prob_buy: float,
        model_id: str,
        registry_key: str,
        actionable: bool,
        quote_asset: str,
        base_asset: str,
        base_min_size: Decimal,
        balances: dict[str, Decimal],
        latency_started: float,
        reference_price: float,
    ) -> dict[str, Any]:
        quote_balance = balances.get(quote_asset.upper(), Decimal("0"))
        base_balance = balances.get(base_asset.upper(), Decimal("0"))
        if product_id in self._managed_positions or base_balance >= base_min_size:
            self.stats.rejected += 1
            return self._base_event(
                product_id=product_id,
                signal="BUY",
                decision="blocked_existing_position",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                reason=f"Ya existe inventario spot de {base_asset}",
                dry_run=self.dry_run,
                latency_ms=self._elapsed_ms(latency_started),
            )

        if not self.dry_run and quote_balance < Decimal(str(self.order_notional_usd)):
            self.stats.rejected += 1
            return self._base_event(
                product_id=product_id,
                signal="BUY",
                decision="blocked_no_quote_balance",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                reason=f"Saldo insuficiente en {quote_asset}",
                dry_run=self.dry_run,
                latency_ms=self._elapsed_ms(latency_started),
            )

        if self.dry_run or not self.execution_enabled:
            self.stats.dry_runs += 1
            self._mark_cooldown(product_id)
            synthetic_quantity = 0.0 if reference_price <= 0 else self.order_notional_usd / reference_price
            self._managed_positions[product_id] = ManagedPosition(
                product_id=product_id,
                entry_price=reference_price,
                quantity=synthetic_quantity,
                opened_at_ms=int(time.time() * 1000),
                model_id=model_id,
            )
            await self._persist_managed_positions()
            logger.info(
                "OrderManager dry-run BUY aceptado. product_id={} notional_usd={} model_id={}",
                product_id,
                self.order_notional_usd,
                model_id,
            )
            return self._base_event(
                product_id=product_id,
                signal="BUY",
                decision="accepted_dry_run",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=actionable,
                order_payload={"side": "BUY", "quote_size": self.order_notional_usd},
                dry_run=self.dry_run,
                latency_ms=self._elapsed_ms(latency_started),
            )

        try:
            response = self.coinbase_client.place_market_buy_quote(product_id, self.order_notional_usd)
            self.stats.live_orders += 1
            synthetic_quantity = 0.0 if reference_price <= 0 else self.order_notional_usd / reference_price
            self._managed_positions[product_id] = ManagedPosition(
                product_id=product_id,
                entry_price=reference_price,
                quantity=synthetic_quantity,
                opened_at_ms=int(time.time() * 1000),
                model_id=model_id,
            )
            await self._persist_managed_positions()
            logger.info(
                "OrderManager envio BUY live. product_id={} notional_usd={} model_id={}",
                product_id,
                self.order_notional_usd,
                model_id,
            )
            event = await self._live_order_event(
                product_id=product_id,
                signal="BUY",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=actionable,
                response=response,
                latency_started=latency_started,
            )
            self._mark_cooldown(product_id)
            return event
        except Exception as exc:
            self.stats.rejected += 1
            return self._base_event(
                product_id=product_id,
                signal="BUY",
                decision="rejected_exchange",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                reason=str(exc),
                dry_run=self.dry_run,
                latency_ms=self._elapsed_ms(latency_started),
            )

    async def _handle_sell(
        self,
        product_id: str,
        prob_buy: float,
        model_id: str,
        registry_key: str,
        actionable: bool,
        base_asset: str,
        base_min_size: Decimal,
        balances: dict[str, Decimal],
        latency_started: float,
        signal_label: str,
        exit_reason: str,
        managed_position: ManagedPosition | None,
    ) -> dict[str, Any]:
        available_base = balances.get(base_asset.upper(), Decimal("0"))
        managed_quantity = Decimal(str(managed_position.quantity)) if managed_position is not None else Decimal("0")
        effective_base = available_base if available_base >= base_min_size else managed_quantity
        if effective_base < base_min_size:
            self._managed_positions.pop(product_id, None)
            await self._persist_managed_positions()
            self.stats.rejected += 1
            return self._base_event(
                product_id=product_id,
                signal=signal_label,
                decision="blocked_no_inventory",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                reason=f"Sin inventario spot de {base_asset}",
                dry_run=self.dry_run,
                latency_ms=self._elapsed_ms(latency_started),
            )

        if self.dry_run or not self.execution_enabled:
            self.stats.dry_runs += 1
            self._mark_cooldown(product_id)
            self._managed_positions.pop(product_id, None)
            await self._persist_managed_positions()
            logger.info(
                "OrderManager dry-run EXIT aceptado. product_id={} base_size={} model_id={} reason={}",
                product_id,
                float(effective_base),
                model_id,
                exit_reason,
            )
            return self._base_event(
                product_id=product_id,
                signal=signal_label,
                decision="accepted_dry_run",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=actionable,
                reason=exit_reason,
                order_payload={"side": "SELL", "base_size": float(effective_base)},
                dry_run=self.dry_run,
                latency_ms=self._elapsed_ms(latency_started),
            )

        try:
            response = self.coinbase_client.place_market_sell_base(product_id, effective_base)
            self.stats.live_orders += 1
            self._managed_positions.pop(product_id, None)
            await self._persist_managed_positions()
            logger.info(
                "OrderManager envio EXIT live. product_id={} base_size={} model_id={} reason={}",
                product_id,
                float(effective_base),
                model_id,
                exit_reason,
            )
            event = await self._live_order_event(
                product_id=product_id,
                signal=signal_label,
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=actionable,
                response=response,
                latency_started=latency_started,
            )
            if exit_reason:
                event["reason"] = exit_reason
            self._mark_cooldown(product_id)
            return event
        except Exception as exc:
            self.stats.rejected += 1
            return self._base_event(
                product_id=product_id,
                signal=signal_label,
                decision="rejected_exchange",
                prob_buy=prob_buy,
                model_id=model_id,
                registry_key=registry_key,
                actionable=False,
                reason=str(exc),
                dry_run=self.dry_run,
                latency_ms=self._elapsed_ms(latency_started),
            )

    async def _live_order_event(
        self,
        product_id: str,
        signal: str,
        prob_buy: float,
        model_id: str,
        registry_key: str,
        actionable: bool,
        response: dict[str, Any],
        latency_started: float,
    ) -> dict[str, Any]:
        order_id = self._extract_order_id(response)
        event = self._base_event(
            product_id=product_id,
            signal=signal,
            decision="sent_live",
            prob_buy=prob_buy,
            model_id=model_id,
            registry_key=registry_key,
            actionable=actionable,
            order_id=order_id,
            exchange_response=response,
            dry_run=self.dry_run,
            latency_ms=self._elapsed_ms(latency_started),
        )
        if not order_id:
            return event

        try:
            order = self.coinbase_client.get_order(order_id)
        except Exception:
            return event

        status = str(
            order.get("order", {}).get("status")
            or order.get("status")
            or ""
        ).upper()
        if status:
            event["order_status"] = status
        if status == "FILLED":
            event["decision"] = "filled"
        return event

    def _build_portfolio(self, quote_asset: str, balances: dict[str, Decimal]) -> Portfolio:
        if self.dry_run:
            capital_total = settings.PAPER_INITIAL_CASH
            capital_disponible = capital_total
            posiciones_abiertas = len(self._managed_positions)
        else:
            capital_total = sum(
                float(balances.get(currency, Decimal("0")))
                for currency in ("USD", "USDC", "USDT")
            )
            capital_total = max(capital_total, self.order_notional_usd)
            capital_disponible = float(balances.get(quote_asset.upper(), Decimal("0")))
            posiciones_abiertas = sum(
                1
                for base_asset, balance in balances.items()
                if base_asset in self.allowed_bases and balance > 0
            )
        return Portfolio(
            capital_total=capital_total,
            capital_disponible=capital_disponible,
            drawdown_hoy=self.drawdown_today,
            posiciones_abiertas=posiciones_abiertas,
        )

    def _risk_pct(self, capital_total: float) -> float:
        if self.dry_run:
            return 0.0
        if capital_total <= 0:
            return 1.0
        stop_distance = settings.POSITION_STOP_LOSS_PCT / 100.0
        return float((self.order_notional_usd * stop_distance) / capital_total)

    def _is_in_cooldown(self, product_id: str) -> bool:
        return self._cooldown_until_by_asset.get(product_id, 0.0) > time.time()

    def _mark_cooldown(self, product_id: str) -> None:
        self._cooldown_until_by_asset[product_id] = time.time() + self.cooldown_seconds

    def _log_runtime_summary(self) -> None:
        """Resume el estado del ejecutor para inspeccion remota."""
        now = time.time()
        if now - self._last_runtime_log_at < settings.RUNTIME_LOG_INTERVAL_SECONDS:
            return
        self._last_runtime_log_at = now
        logger.info(
            "OrderManager resumen: processed={} dry_runs={} live_orders={} rejected={} cooldown_assets={} managed_positions={}",
            self.stats.processed,
            self.stats.dry_runs,
            self.stats.live_orders,
            self.stats.rejected,
            len(self._cooldown_until_by_asset),
            len(self._managed_positions),
        )

    async def _publish_event(self, payload: dict[str, Any]) -> None:
        mapping = {
            key: json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            for key, value in payload.items()
        }
        await self.redis_client.xadd(settings.STREAM_EXECUTION_EVENTS, mapping, maxlen=50000, approximate=True)

    async def _ensure_group(self, stream: str, group_name: str) -> None:
        try:
            await self.redis_client.xgroup_create(stream, group_name, id="$", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def _ensure_state_loaded(self) -> None:
        """Restaura posiciones abiertas del proceso tras reinicios controlados."""
        if self._state_loaded:
            return
        self._state_loaded = True
        if self.redis_client is None:
            return
        try:
            raw_positions = await self.redis_client.hgetall(settings.EXECUTION_STATE_KEY)
        except Exception:
            logger.exception("OrderManager no pudo cargar estado persistido")
            return
        for product_id, raw_payload in raw_positions.items():
            try:
                payload = json.loads(raw_payload)
                self._managed_positions[product_id] = ManagedPosition(
                    product_id=str(payload["product_id"]),
                    entry_price=float(payload["entry_price"]),
                    quantity=float(payload["quantity"]),
                    opened_at_ms=int(payload["opened_at_ms"]),
                    model_id=str(payload.get("model_id", "")),
                )
            except Exception:
                logger.warning("OrderManager ignoro estado invalido de {}", product_id)
        if self._managed_positions:
            logger.info(
                "OrderManager restauro posiciones persistidas: {}",
                sorted(self._managed_positions),
            )

    async def _persist_managed_positions(self) -> None:
        """Persiste el estado minimo para sobrevivir a reinicios de Render."""
        if self.redis_client is None:
            return
        mapping = {
            product_id: json.dumps(
                {
                    "product_id": position.product_id,
                    "entry_price": position.entry_price,
                    "quantity": position.quantity,
                    "opened_at_ms": position.opened_at_ms,
                    "model_id": position.model_id,
                }
            )
            for product_id, position in self._managed_positions.items()
        }
        await self.redis_client.delete(settings.EXECUTION_STATE_KEY)
        if mapping:
            await self.redis_client.hset(settings.EXECUTION_STATE_KEY, mapping=mapping)

    @staticmethod
    def _extract_order_id(payload: dict[str, Any]) -> str:
        if "order_id" in payload:
            return str(payload["order_id"])
        for key in ("success_response", "order", "response", "result"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                extracted = OrderManager._extract_order_id(nested)
                if extracted:
                    return extracted
        return ""

    @staticmethod
    def _to_bool(value: Any) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

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
        dry_run: bool,
        latency_ms: float,
        reason: str = "",
        spread_pct: float | None = None,
        order_id: str = "",
        order_payload: dict[str, Any] | None = None,
        exchange_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "product_id": product_id,
            "signal": signal,
            "decision": decision,
            "prob_buy": round(prob_buy, 6),
            "model_id": model_id,
            "registry_key": registry_key,
            "actionable": str(actionable).lower(),
            "reason": reason,
            "dry_run": str(dry_run).lower(),
            "latency_ms": latency_ms,
        }
        if spread_pct is not None:
            payload["spread_pct"] = round(spread_pct, 6)
        if order_id:
            payload["order_id"] = order_id
        if order_payload is not None:
            payload["order_payload"] = order_payload
        if exchange_response is not None:
            payload["exchange_response"] = exchange_response
        return payload
