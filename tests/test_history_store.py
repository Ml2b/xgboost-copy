"""Tests del almacenamiento historico de velas para el trainer."""

from __future__ import annotations

import sqlite3

import fakeredis
import pandas as pd

from data.history_store import CandleHistoryStore


def test_history_store_syncs_incrementally_and_dedupes(tmp_path) -> None:
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    store = CandleHistoryStore(db_path=tmp_path / "history.sqlite3")

    redis_client.xadd(
        "market.candles.1m",
        {
            "product_id": "BTC-USD",
            "open_time": "1000",
            "close_time": "1060",
            "open": "10",
            "high": "11",
            "low": "9",
            "close": "10.5",
            "volume": "100",
            "trade_count": "5",
        },
        id="1-0",
    )
    redis_client.xadd(
        "market.candles.1m",
        {
            "product_id": "BTC-USD",
            "open_time": "1060",
            "close_time": "1120",
            "open": "10.5",
            "high": "11.5",
            "low": "10.2",
            "close": "11.0",
            "volume": "110",
            "trade_count": "4",
        },
        id="2-0",
    )

    imported_first = store.sync_from_redis_stream(redis_client, batch_size=1)
    frame_first = store.load_candles_for_base("BTC")

    assert imported_first == 2
    assert frame_first is not None
    assert len(frame_first) == 2
    assert store.get_last_stream_id("market.candles.1m") == "2-0"

    redis_client.xadd(
        "market.candles.1m",
        {
            "product_id": "BTC-USD",
            "open_time": "1060",
            "close_time": "1120",
            "open": "10.5",
            "high": "11.7",
            "low": "10.2",
            "close": "11.2",
            "volume": "120",
            "trade_count": "6",
        },
        id="3-0",
    )
    redis_client.xadd(
        "market.candles.1m",
        {
            "product_id": "ETH-USD",
            "open_time": "1000",
            "close_time": "1060",
            "open": "20",
            "high": "21",
            "low": "19.5",
            "close": "20.4",
            "volume": "90",
            "trade_count": "3",
        },
        id="4-0",
    )

    imported_second = store.sync_from_redis_stream(redis_client, batch_size=10)
    btc_frame = store.load_candles_for_base("BTC")
    eth_frame = store.load_candles_for_base("ETH")

    assert imported_second == 2
    assert btc_frame is not None
    assert len(btc_frame) == 2
    assert float(btc_frame.iloc[-1]["close"]) == 11.2
    assert eth_frame is not None
    assert len(eth_frame) == 1
    assert store.get_last_stream_id("market.candles.1m") == "4-0"


def test_history_store_backfills_from_frame_and_records_metadata(tmp_path) -> None:
    store = CandleHistoryStore(db_path=tmp_path / "history.sqlite3")
    frame = pd.DataFrame(
        [
            {
                "product_id": "BTC-USD",
                "open_time": 1000,
                "close_time": 1059,
                "open": 10.0,
                "high": 11.0,
                "low": 9.5,
                "close": 10.7,
                "volume": 100.0,
                "trade_count": 5,
            },
            {
                "product_id": "BTC-USD",
                "open_time": 1060,
                "close_time": 1119,
                "open": 10.7,
                "high": 10.9,
                "low": 10.1,
                "close": 10.3,
                "volume": 95.0,
                "trade_count": 4,
            },
        ]
    )

    inserted = store.upsert_frame(frame, source_name="btc.csv", chunk_size=1)
    persisted = store.load_candles_for_base("BTC")

    assert inserted == 2
    assert persisted is not None
    assert len(persisted) == 2
    assert store.get_row_count_for_base("BTC") == 2

    with sqlite3.connect(tmp_path / "history.sqlite3") as connection:
        row = connection.execute(
            "SELECT source_name, row_count, product_count, min_open_time, max_open_time FROM bootstrap_loads"
        ).fetchone()

    assert row == ("btc.csv", 2, 1, 1000, 1060)
