"""Tests del cliente privado de Coinbase."""

from __future__ import annotations

import json

from exchange.coinbase_client import CoinbaseAdvancedTradeClient


class DummyPublicClient:
    """Cliente publico minimo para tests sin red."""

    def get_public_product(self, product_id: str):
        return {
            "product_id": product_id,
            "base_currency_id": product_id.split("-", 1)[0],
            "quote_currency_id": product_id.split("-", 1)[1],
            "base_increment": "0.00000001",
            "quote_increment": "0.01",
            "base_min_size": "0.0001",
            "quote_min_size": "1",
            "trading_disabled": False,
            "is_disabled": False,
            "status": "online",
        }


class DummyProductsClient:
    """Cliente publico minimo para listar productos."""

    def get_public_products(self, **_: object):
        return {
            "products": [
                {"product_id": "BTC-USD", "status": "online"},
                {"product_id": "ETH-USD", "status": "online"},
            ]
        }


class DummyOrderBookClient:
    """Cliente minimo para best bid/ask sin credenciales privadas."""

    def get_best_bid_ask(self, product_ids):
        return {
            "pricebooks": [
                {
                    "product_id": product_id,
                    "bids": [{"price": "100"}],
                    "asks": [{"price": "101"}],
                }
                for product_id in product_ids
            ]
        }


class DummyCandleClient:
    """Cliente minimo para tests de normalizacion de velas."""

    def get_public_candles(self, product_id: str, start: str, end: str, granularity: str, limit=None):
        return {
            "candles": [
                {
                    "start": "1700000060",
                    "low": "99",
                    "high": "102",
                    "open": "100",
                    "close": "101",
                    "volume": "5",
                },
                {
                    "start": "1700000000",
                    "low": "98",
                    "high": "101",
                    "open": "99",
                    "close": "100",
                    "volume": "4",
                },
            ]
        }


def test_coinbase_client_loads_key_file_and_preserves_newlines(tmp_path) -> None:
    key_path = tmp_path / "coinbase.json"
    key_path.write_text(
        json.dumps(
            {
                "name": "organizations/test/apiKeys/test-key",
                "privateKey": "-----BEGIN PRIVATE KEY-----\\nline-1\\nline-2\\n-----END PRIVATE KEY-----\\n",
            }
        ),
        encoding="utf-8",
    )

    client = CoinbaseAdvancedTradeClient(credentials_path=key_path, public_client=DummyPublicClient())

    assert client.private_client is not None
    assert "\nline-1\n" in client.private_client.api_secret
    assert "\\n" not in client.private_client.api_secret


def test_coinbase_client_builds_ws_jwt_from_key_file(tmp_path, monkeypatch) -> None:
    key_path = tmp_path / "coinbase.json"
    key_path.write_text(
        json.dumps(
            {
                "name": "organizations/test/apiKeys/test-key",
                "privateKey": "-----BEGIN PRIVATE KEY-----\\nline-1\\nline-2\\n-----END PRIVATE KEY-----\\n",
            }
        ),
        encoding="utf-8",
    )

    calls: dict[str, str] = {}

    def fake_build_ws_jwt(key_var: str, secret_var: str) -> str:
        calls["key_var"] = key_var
        calls["secret_var"] = secret_var
        return "jwt-token"

    monkeypatch.setattr("coinbase.jwt_generator.build_ws_jwt", fake_build_ws_jwt)

    client = CoinbaseAdvancedTradeClient(credentials_path=key_path, public_client=DummyPublicClient())

    assert client.build_ws_jwt() == "jwt-token"
    assert calls["key_var"] == "organizations/test/apiKeys/test-key"
    assert "\nline-1\n" in calls["secret_var"]
    assert "\\n" not in calls["secret_var"]


def test_coinbase_client_resolves_products_by_quote_priority(tmp_path) -> None:
    key_path = tmp_path / "coinbase.json"
    key_path.write_text(
        json.dumps({"name": "organizations/test/apiKeys/test-key", "privateKey": "secret"}),
        encoding="utf-8",
    )

    client = CoinbaseAdvancedTradeClient(
        credentials_path=key_path,
        public_client=DummyPublicClient(),
        private_client=object(),
    )

    resolved = client.resolve_products_for_bases(["BTC", "ETH"], quote_priority=["USD", "USDT"])

    assert resolved == {"BTC": "BTC-USD", "ETH": "ETH-USD"}


def test_coinbase_client_lists_public_products(tmp_path) -> None:
    key_path = tmp_path / "coinbase.json"
    key_path.write_text(
        json.dumps({"name": "organizations/test/apiKeys/test-key", "privateKey": "secret"}),
        encoding="utf-8",
    )

    client = CoinbaseAdvancedTradeClient(
        credentials_path=key_path,
        public_client=DummyProductsClient(),
        private_client=None,
    )

    products = client.list_public_products(limit=2, get_all_products=False)

    assert [product["product_id"] for product in products] == ["BTC-USD", "ETH-USD"]


def test_coinbase_client_gets_best_bid_ask_without_private_client(tmp_path) -> None:
    key_path = tmp_path / "coinbase.json"
    key_path.write_text(
        json.dumps({"name": "organizations/test/apiKeys/test-key", "privateKey": "secret"}),
        encoding="utf-8",
    )

    client = CoinbaseAdvancedTradeClient(
        credentials_path=key_path,
        public_client=DummyOrderBookClient(),
        private_client=None,
    )

    snapshot = client.get_best_bid_ask("BTC-USD", prefer_private=False)

    assert snapshot["product_id"] == "BTC-USD"
    assert snapshot["mid"] == 100.5
    assert round(snapshot["spread_pct"], 6) == round((101 - 100) / 100.5, 6)


def test_coinbase_client_normalizes_public_candles_in_ascending_order(tmp_path) -> None:
    key_path = tmp_path / "coinbase.json"
    key_path.write_text(
        json.dumps({"name": "organizations/test/apiKeys/test-key", "privateKey": "secret"}),
        encoding="utf-8",
    )

    client = CoinbaseAdvancedTradeClient(
        credentials_path=key_path,
        public_client=DummyCandleClient(),
        private_client=None,
    )

    candles = client.get_candles(
        product_id="BTC-USD",
        start=1_700_000_000,
        end=1_700_000_120,
        granularity="ONE_MINUTE",
        prefer_private=False,
    )

    assert [candle["open_time"] for candle in candles] == [1_700_000_000_000, 1_700_000_060_000]
    assert candles[0]["close_time"] == 1_700_000_059_999
    assert candles[1]["close"] == 101.0
