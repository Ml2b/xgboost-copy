"""Tests para el bootstrap de Render."""

from __future__ import annotations

from pathlib import Path

from data.history_store import CandleHistoryStore
from scripts.render_start import parse_product_ids_from_env, seed_recent_history_for_product


class DummyCoinbaseClient:
    """Cliente fake que devuelve velas 1m sinteticas hacia atras."""

    def __init__(self) -> None:
        self.calls = 0

    def get_candles(
        self,
        product_id: str,
        start: int,
        end: int,
        granularity: str = "ONE_MINUTE",
        limit: int | None = None,
        prefer_private: bool = False,
    ) -> list[dict[str, object]]:
        self.calls += 1
        candles: list[dict[str, object]] = []
        current = int(end) - 60
        target = int(start)
        max_rows = int(limit or 350)

        while current >= target and len(candles) < max_rows:
            open_time = current * 1000
            candles.append(
                {
                    "product_id": product_id,
                    "open_time": open_time,
                    "close_time": open_time + 59_999,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1.0,
                    "trade_count": 0,
                    "is_closed": True,
                }
            )
            current -= 60
        candles.sort(key=lambda item: int(item["open_time"]))
        return candles


def test_parse_product_ids_from_env_deduplicates_and_normalizes() -> None:
    product_ids = parse_product_ids_from_env(" btc-usd, ETH-USD,btc-usd, , ada-usd ")
    assert product_ids == ["BTC-USD", "ETH-USD", "ADA-USD"]


def test_seed_recent_history_for_product_backfills_until_target_rows(tmp_path: Path) -> None:
    store = CandleHistoryStore(db_path=tmp_path / "trainer_history.sqlite3")
    client = DummyCoinbaseClient()

    row_count = seed_recent_history_for_product(
        store=store,
        client=client,
        product_id="BTC-USD",
        target_rows=500,
        chunk_limit=200,
        max_requests=3,
    )

    assert row_count >= 500
    assert store.get_row_count_for_base("BTC") >= 500
    assert client.calls >= 3
