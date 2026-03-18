"""Tests del screener de operabilidad Coinbase."""

from __future__ import annotations

from datetime import datetime, timezone

from exchange.operability_screener import CoinbaseOperabilityScreener, OperabilityConfig


class DummyScreenerClient:
    """Cliente minimo para probar el screener sin red."""

    def __init__(
        self,
        products: list[dict[str, object]],
        spreads: dict[str, dict[str, float]],
        candles: dict[str, list[dict[str, object]]],
    ) -> None:
        self.products = products
        self.spreads = spreads
        self.candles = candles

    def list_public_products(self, **_: object) -> list[dict[str, object]]:
        return self.products

    def get_best_bid_ask_many(self, product_ids: list[str], prefer_private: bool = True) -> dict[str, dict[str, float]]:
        _ = prefer_private
        return {product_id: self.spreads[product_id] for product_id in product_ids if product_id in self.spreads}

    def get_candles(
        self,
        product_id: str,
        start: int,
        end: int,
        granularity: str = "ONE_MINUTE",
        limit: int | None = None,
        prefer_private: bool = True,
    ) -> list[dict[str, object]]:
        _ = (start, end, granularity, limit, prefer_private)
        return self.candles.get(product_id, [])

    def has_private_credentials(self) -> bool:
        return False


def make_product(
    product_id: str,
    quote_volume_24h: float,
    new_at: str = "2025-01-01T00:00:00Z",
    status: str = "online",
    trading_disabled: bool = False,
) -> dict[str, object]:
    base_asset, quote_asset = product_id.split("-", 1)
    return {
        "product_id": product_id,
        "product_type": "SPOT",
        "base_currency_id": base_asset,
        "base_name": base_asset,
        "quote_currency_id": quote_asset,
        "status": status,
        "trading_disabled": trading_disabled,
        "is_disabled": False,
        "cancel_only": False,
        "limit_only": False,
        "post_only": False,
        "view_only": False,
        "auction_mode": False,
        "approximate_quote_24h_volume": str(quote_volume_24h),
        "new_at": new_at,
    }


def make_candles(count: int, start_open_time_ms: int = 1_700_000_000_000) -> list[dict[str, object]]:
    candles: list[dict[str, object]] = []
    price = 100.0
    for index in range(count):
        open_time = start_open_time_ms + (index * 60_000)
        close_price = price * 1.001
        candles.append(
            {
                "open_time": open_time,
                "close_time": open_time + 59_999,
                "open": price,
                "high": close_price,
                "low": price,
                "close": close_price,
                "volume": 10.0,
            }
        )
        price = close_price
    return candles


def test_screener_prefers_quote_priority_and_excludes_stables() -> None:
    client = DummyScreenerClient(
        products=[
            make_product("BTC-USDC", 7_000_000),
            make_product("BTC-USD", 6_000_000),
            make_product("USDC-USD", 50_000_000),
        ],
        spreads={
            "BTC-USD": {"product_id": "BTC-USD", "bid": 100.0, "ask": 100.05, "mid": 100.025, "spread_pct": 0.0005},
            "BTC-USDC": {"product_id": "BTC-USDC", "bid": 100.0, "ask": 100.04, "mid": 100.02, "spread_pct": 0.0004},
        },
        candles={"BTC-USD": make_candles(300), "BTC-USDC": make_candles(300)},
    )

    report = CoinbaseOperabilityScreener(client=client).screen_assets(now=datetime(2026, 3, 18, tzinfo=timezone.utc))

    assert [result.base_asset for result in report.results] == ["BTC"]
    assert report.results[0].product_id == "BTC-USD"


def test_screener_marks_train_now_when_asset_passes_filters() -> None:
    client = DummyScreenerClient(
        products=[make_product("SUI-USD", 8_000_000)],
        spreads={
            "SUI-USD": {"product_id": "SUI-USD", "bid": 2.0, "ask": 2.002, "mid": 2.001, "spread_pct": 0.001},
        },
        candles={"SUI-USD": make_candles(300)},
    )
    config = OperabilityConfig(
        min_quote_volume_24h=1_000_000,
        max_spread_pct=0.0025,
        min_listing_age_days=30,
        candle_lookback_minutes=300,
        min_candle_coverage_pct=0.9,
    )

    report = CoinbaseOperabilityScreener(client=client, config=config).screen_assets(
        now=datetime(2026, 3, 18, tzinfo=timezone.utc)
    )

    result = report.results[0]
    assert result.eligible is True
    assert result.tier == "train_now"
    assert result.reasons == ["ok"]


def test_screener_marks_observe_when_spread_and_coverage_fail() -> None:
    client = DummyScreenerClient(
        products=[make_product("BONK-USD", 3_000_000)],
        spreads={
            "BONK-USD": {"product_id": "BONK-USD", "bid": 0.10, "ask": 0.104, "mid": 0.102, "spread_pct": 0.0392156863},
        },
        candles={"BONK-USD": make_candles(120)},
    )
    config = OperabilityConfig(
        min_quote_volume_24h=1_000_000,
        max_spread_pct=0.0025,
        min_listing_age_days=30,
        candle_lookback_minutes=300,
        min_candle_coverage_pct=0.9,
    )

    report = CoinbaseOperabilityScreener(client=client, config=config).screen_assets(
        now=datetime(2026, 3, 18, tzinfo=timezone.utc)
    )

    result = report.results[0]
    assert result.eligible is False
    assert result.tier == "observe"
    assert "wide_spread" in result.reasons
    assert "low_candle_coverage" in result.reasons
