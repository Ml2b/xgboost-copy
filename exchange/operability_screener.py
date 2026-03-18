"""Screening de operabilidad para activos spot de Coinbase."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from config import settings
from exchange.coinbase_client import CoinbaseAdvancedTradeClient


DEFAULT_EXCLUDED_BASES: tuple[str, ...] = (
    "USDC",
    "USDT",
    "PYUSD",
    "DAI",
    "USDS",
    "FDUSD",
    "EURC",
    "EUR",
    "GBP",
    "USD",
    "GYEN",
)


@dataclass(slots=True)
class OperabilityConfig:
    """Parametros heurísticos del screener de activos."""

    quote_priority: tuple[str, ...] = tuple(settings.COINBASE_QUOTE_PRIORITY)
    min_quote_volume_24h: float = 1_000_000.0
    max_spread_pct: float = 0.0025
    min_listing_age_days: int = 45
    candle_lookback_minutes: int = 300
    min_candle_coverage_pct: float = 0.90
    max_candidates_for_candles: int = 80
    excluded_bases: tuple[str, ...] = DEFAULT_EXCLUDED_BASES


@dataclass(slots=True)
class AssetScreeningResult:
    """Resultado detallado por activo base."""

    base_asset: str
    base_name: str
    product_id: str
    quote_asset: str
    quote_volume_24h: float
    spread_pct: float | None
    listing_age_days: int
    candle_coverage_pct: float
    daily_realized_vol_pct: float
    liquidity_score: float
    microstructure_score: float
    stability_score: float
    maturity_score: float
    score: float
    eligible: bool
    tier: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serializa el resultado para JSON/CSV/Markdown."""
        payload = asdict(self)
        payload["quote_volume_24h"] = round(self.quote_volume_24h, 2)
        payload["spread_pct"] = None if self.spread_pct is None else round(self.spread_pct, 6)
        payload["candle_coverage_pct"] = round(self.candle_coverage_pct, 4)
        payload["daily_realized_vol_pct"] = round(self.daily_realized_vol_pct, 4)
        payload["liquidity_score"] = round(self.liquidity_score, 4)
        payload["microstructure_score"] = round(self.microstructure_score, 4)
        payload["stability_score"] = round(self.stability_score, 4)
        payload["maturity_score"] = round(self.maturity_score, 4)
        payload["score"] = round(self.score, 2)
        return payload


@dataclass(slots=True)
class OperabilityScreenReport:
    """Reporte agregado del universo analizado."""

    generated_at: str
    total_products_seen: int
    total_base_assets_seen: int
    candidate_base_assets: int
    config: dict[str, Any]
    results: list[AssetScreeningResult]

    def to_dict(self) -> dict[str, Any]:
        """Serializa el reporte completo."""
        return {
            "generated_at": self.generated_at,
            "total_products_seen": self.total_products_seen,
            "total_base_assets_seen": self.total_base_assets_seen,
            "candidate_base_assets": self.candidate_base_assets,
            "config": self.config,
            "summary": {
                "train_now": sum(1 for result in self.results if result.tier == "train_now"),
                "observe": sum(1 for result in self.results if result.tier == "observe"),
                "skip": sum(1 for result in self.results if result.tier == "skip"),
            },
            "results": [result.to_dict() for result in self.results],
        }


class CoinbaseOperabilityScreener:
    """Evalua que activos spot de Coinbase valen la pena para entrenamiento por activo."""

    def __init__(
        self,
        client: CoinbaseAdvancedTradeClient,
        config: OperabilityConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or OperabilityConfig()

    def screen_assets(self, now: datetime | None = None) -> OperabilityScreenReport:
        """Construye el ranking priorizado de activos operables."""
        reference_time = now or datetime.now(timezone.utc)
        raw_products = self.client.list_public_products(product_type="SPOT", get_all_products=True)
        candidate_products = self._select_preferred_products(raw_products)
        spreads = self.client.get_best_bid_ask_many(
            [product["product_id"] for product in candidate_products],
            prefer_private=self.client.has_private_credentials(),
        )

        candle_targets = {
            product["product_id"]
            for product in sorted(
                candidate_products,
                key=lambda item: item["quote_volume_24h"],
                reverse=True,
            )[: self.config.max_candidates_for_candles]
        }

        metrics: list[dict[str, Any]] = []
        for product in candidate_products:
            product_id = product["product_id"]
            spread_snapshot = spreads.get(product_id)
            coverage_pct = 0.0
            realized_vol_pct = 0.0
            if product_id in candle_targets:
                coverage_pct, realized_vol_pct = self._measure_recent_candles(
                    product_id=product_id,
                    lookback_minutes=self.config.candle_lookback_minutes,
                    now=reference_time,
                )

            metrics.append(
                {
                    **product,
                    "spread_pct": None if spread_snapshot is None else float(spread_snapshot["spread_pct"]),
                    "candle_coverage_pct": coverage_pct,
                    "daily_realized_vol_pct": realized_vol_pct,
                }
            )

        liquidity_values = [math.log1p(metric["quote_volume_24h"]) for metric in metrics if metric["quote_volume_24h"] > 0]
        spread_values = [metric["spread_pct"] for metric in metrics if metric["spread_pct"] is not None]

        results: list[AssetScreeningResult] = []
        for metric in metrics:
            liquidity_score = self._percentile_rank(
                liquidity_values,
                math.log1p(metric["quote_volume_24h"]) if metric["quote_volume_24h"] > 0 else 0.0,
            )
            microstructure_score = 0.0
            if metric["spread_pct"] is not None and spread_values:
                microstructure_score = 1.0 - self._percentile_rank(spread_values, float(metric["spread_pct"]))
            stability_score = max(0.0, min(metric["candle_coverage_pct"], 1.0))
            maturity_score = max(0.0, min(metric["listing_age_days"] / 180.0, 1.0))
            score = 100.0 * (
                (0.40 * liquidity_score)
                + (0.25 * microstructure_score)
                + (0.20 * stability_score)
                + (0.15 * maturity_score)
            )

            reasons = self._build_reasons(metric)
            eligible = not reasons
            tier = self._resolve_tier(eligible=eligible, score=score, quote_volume_24h=metric["quote_volume_24h"])
            results.append(
                AssetScreeningResult(
                    base_asset=metric["base_asset"],
                    base_name=metric["base_name"],
                    product_id=metric["product_id"],
                    quote_asset=metric["quote_asset"],
                    quote_volume_24h=metric["quote_volume_24h"],
                    spread_pct=metric["spread_pct"],
                    listing_age_days=metric["listing_age_days"],
                    candle_coverage_pct=metric["candle_coverage_pct"],
                    daily_realized_vol_pct=metric["daily_realized_vol_pct"],
                    liquidity_score=liquidity_score,
                    microstructure_score=microstructure_score,
                    stability_score=stability_score,
                    maturity_score=maturity_score,
                    score=score,
                    eligible=eligible,
                    tier=tier,
                    reasons=["ok"] if eligible else reasons,
                )
            )

        results.sort(
            key=lambda item: (item.tier == "train_now", item.tier == "observe", item.score, item.quote_volume_24h),
            reverse=True,
        )

        total_bases_seen = len(
            {
                str(product.get("base_currency_id") or product.get("base_display_symbol") or "").upper()
                for product in raw_products
                if str(product.get("base_currency_id") or product.get("base_display_symbol") or "").strip()
            }
        )
        return OperabilityScreenReport(
            generated_at=reference_time.isoformat(),
            total_products_seen=len(raw_products),
            total_base_assets_seen=total_bases_seen,
            candidate_base_assets=len(candidate_products),
            config=asdict(self.config),
            results=results,
        )

    def _select_preferred_products(self, raw_products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}
        for product in raw_products:
            if str(product.get("product_type", "")).upper() != "SPOT":
                continue

            base_asset = str(
                product.get("base_currency_id")
                or product.get("base_display_symbol")
                or ""
            ).upper()
            quote_asset = str(
                product.get("quote_currency_id")
                or product.get("quote_display_symbol")
                or ""
            ).upper()
            if not base_asset or quote_asset not in self.config.quote_priority:
                continue
            if base_asset in self.config.excluded_bases:
                continue

            normalized = {
                "product_id": str(product.get("product_id", "")).upper(),
                "base_asset": base_asset,
                "base_name": str(product.get("base_name", base_asset)),
                "quote_asset": quote_asset,
                "status": str(product.get("status", "")),
                "trading_disabled": bool(product.get("trading_disabled", False)),
                "is_disabled": bool(product.get("is_disabled", False)),
                "cancel_only": bool(product.get("cancel_only", False)),
                "limit_only": bool(product.get("limit_only", False)),
                "post_only": bool(product.get("post_only", False)),
                "view_only": bool(product.get("view_only", False)),
                "auction_mode": bool(product.get("auction_mode", False)),
                "quote_volume_24h": self._extract_quote_volume(product),
                "listing_age_days": self._extract_listing_age_days(product),
            }
            if not normalized["product_id"]:
                continue

            current = candidates.get(base_asset)
            if current is None or self._should_replace_product(normalized, current):
                candidates[base_asset] = normalized

        return list(candidates.values())

    def _should_replace_product(self, candidate: dict[str, Any], current: dict[str, Any]) -> bool:
        candidate_rank = self.config.quote_priority.index(candidate["quote_asset"])
        current_rank = self.config.quote_priority.index(current["quote_asset"])
        if candidate_rank != current_rank:
            return candidate_rank < current_rank
        return float(candidate["quote_volume_24h"]) > float(current["quote_volume_24h"])

    def _measure_recent_candles(
        self,
        product_id: str,
        lookback_minutes: int,
        now: datetime,
    ) -> tuple[float, float]:
        # Tomamos solo velas ya cerradas para que la cobertura no se distorsione.
        end_seconds = int(now.timestamp()) - 60
        start_seconds = end_seconds - (lookback_minutes * 60)
        try:
            candles = self.client.get_candles(
                product_id=product_id,
                start=start_seconds,
                end=end_seconds,
                granularity="ONE_MINUTE",
                limit=min(lookback_minutes, 350),
                prefer_private=False,
            )
        except Exception:
            return 0.0, 0.0

        if not candles:
            return 0.0, 0.0

        expected = min(lookback_minutes, 350)
        unique_minutes = {int(candle["open_time"]) for candle in candles}
        coverage_pct = min(len(unique_minutes) / expected, 1.0) if expected > 0 else 0.0

        closes = [float(candle["close"]) for candle in candles if float(candle["close"]) > 0]
        if len(closes) < 2:
            return coverage_pct, 0.0

        log_returns: list[float] = []
        previous_close = closes[0]
        for close in closes[1:]:
            if previous_close <= 0 or close <= 0:
                previous_close = close
                continue
            log_returns.append(math.log(close / previous_close))
            previous_close = close

        if len(log_returns) < 2:
            return coverage_pct, 0.0

        mean_return = sum(log_returns) / len(log_returns)
        variance = sum((value - mean_return) ** 2 for value in log_returns) / len(log_returns)
        realized_vol_pct = (math.sqrt(variance) * math.sqrt(1440.0)) * 100.0
        return coverage_pct, realized_vol_pct

    def _build_reasons(self, metric: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        if metric["status"].lower() != "online":
            reasons.append("status_not_online")
        if metric["trading_disabled"] or metric["is_disabled"]:
            reasons.append("trading_disabled")
        if metric["cancel_only"] or metric["limit_only"] or metric["post_only"] or metric["view_only"]:
            reasons.append("trading_flags_restrictive")
        if metric["auction_mode"]:
            reasons.append("auction_mode")
        if float(metric["quote_volume_24h"]) < self.config.min_quote_volume_24h:
            reasons.append("low_quote_volume_24h")
        spread_pct = metric["spread_pct"]
        if spread_pct is None:
            reasons.append("missing_spread")
        elif float(spread_pct) > self.config.max_spread_pct:
            reasons.append("wide_spread")
        if int(metric["listing_age_days"]) < self.config.min_listing_age_days:
            reasons.append("recent_listing")
        if float(metric["candle_coverage_pct"]) < self.config.min_candle_coverage_pct:
            reasons.append("low_candle_coverage")
        return reasons

    @staticmethod
    def _resolve_tier(eligible: bool, score: float, quote_volume_24h: float) -> str:
        if eligible and score >= 70.0:
            return "train_now"
        if eligible or (score >= 50.0 and quote_volume_24h >= 500_000.0):
            return "observe"
        return "skip"

    @staticmethod
    def _extract_quote_volume(product: dict[str, Any]) -> float:
        approximate_quote_volume = str(product.get("approximate_quote_24h_volume", "")).strip()
        if approximate_quote_volume:
            return float(approximate_quote_volume)
        price = float(product.get("price") or 0.0)
        volume_24h = float(product.get("volume_24h") or 0.0)
        return price * volume_24h

    @staticmethod
    def _extract_listing_age_days(product: dict[str, Any]) -> int:
        raw_new_at = str(product.get("new_at", "")).strip()
        if not raw_new_at:
            return 0
        try:
            listed_at = datetime.fromisoformat(raw_new_at.replace("Z", "+00:00"))
        except ValueError:
            return 0
        delta = datetime.now(timezone.utc) - listed_at.astimezone(timezone.utc)
        return max(int(delta.total_seconds() // 86_400), 0)

    @staticmethod
    def _percentile_rank(values: list[float], value: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return 1.0

        less_than = sum(1 for item in values if item < value)
        equal_to = sum(1 for item in values if item == value)
        return (less_than + (0.5 * equal_to)) / len(values)
