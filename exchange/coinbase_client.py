"""Cliente Coinbase Advanced Trade para autenticacion privada y utilidades spot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any
from uuid import uuid4

from config import settings


@dataclass(slots=True)
class CoinbaseProductSnapshot:
    """Metadata minima del producto para sizing y validaciones."""

    product_id: str
    base_asset: str
    quote_asset: str
    base_increment: Decimal
    quote_increment: Decimal
    base_min_size: Decimal
    quote_min_size: Decimal
    trading_disabled: bool
    is_disabled: bool
    status: str


class CoinbaseAdvancedTradeClient:
    """Envuelve el SDK oficial para operaciones spot controladas."""

    def __init__(
        self,
        credentials_path: str | Path | None = None,
        timeout: int = settings.COINBASE_HTTP_TIMEOUT_SECONDS,
        public_client: Any | None = None,
        private_client: Any | None = None,
    ) -> None:
        self.credentials_path = Path(credentials_path or settings.COINBASE_CREDENTIALS_PATH)
        self.timeout = timeout
        self.public_client = public_client or self._build_public_client(timeout)
        self.private_client = private_client or self._build_private_client()
        self._credentials_payload = self._load_credentials_payload()

    def validate_credentials(self) -> list[dict[str, Any]]:
        """Valida que la clave permita acceder a cuentas privadas."""
        return self.list_accounts(limit=5)

    def has_private_credentials(self) -> bool:
        """Indica si hay credenciales CDP disponibles para auth privada."""
        return bool(self._credentials_payload.get("name") and self._credentials_payload.get("privateKey"))

    def build_ws_jwt(self) -> str:
        """Construye un JWT corto para suscripciones WebSocket autenticadas."""
        if not self.has_private_credentials():
            raise FileNotFoundError(
                f"No se encontro una clave Coinbase valida en {self.credentials_path}."
            )

        from coinbase import jwt_generator  # type: ignore

        return str(
            jwt_generator.build_ws_jwt(
                key_var=str(self._credentials_payload["name"]),
                secret_var=str(self._credentials_payload["privateKey"]),
            )
        )

    def resolve_products_for_bases(
        self,
        bases: list[str],
        quote_priority: list[str] | None = None,
    ) -> dict[str, str]:
        """Resuelve el product_id spot preferido por cada activo base."""
        resolved: dict[str, str] = {}
        for base in bases:
            normalized_base = base.strip().upper()
            for quote in (quote_priority or settings.COINBASE_QUOTE_PRIORITY):
                candidate = f"{normalized_base}-{quote.strip().upper()}"
                try:
                    snapshot = self.get_product_snapshot(candidate)
                except Exception:
                    continue
                if snapshot.trading_disabled or snapshot.is_disabled:
                    continue
                if snapshot.status.lower() != "online":
                    continue
                resolved[normalized_base] = snapshot.product_id
                break
        return resolved

    def list_public_products(
        self,
        limit: int | None = None,
        product_type: str = "SPOT",
        get_all_products: bool = True,
        product_venue: str | None = "CBE",
    ) -> list[dict[str, Any]]:
        """Lista productos publicos en formato plano para screening y discovery."""
        kwargs: dict[str, Any] = {
            "product_type": product_type,
            "get_all_products": get_all_products,
        }
        if limit is not None:
            kwargs["limit"] = limit
        if product_venue:
            kwargs["product_venue"] = product_venue

        response = self.public_client.get_public_products(**kwargs)
        payload = self._to_plain(response)
        return list(payload.get("products", []))

    def list_accounts(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Retorna cuentas disponibles en formato plano."""
        client = self._require_private_client()
        response = client.get_accounts(limit=limit)
        payload = self._to_plain(response)
        return list(payload.get("accounts", []))

    def get_account_balances(self) -> dict[str, Decimal]:
        """Consolida balances disponibles por moneda."""
        balances: dict[str, Decimal] = {}
        for account in self.list_accounts():
            available_balance = account.get("available_balance", {}) or {}
            currency = str(
                account.get("currency")
                or available_balance.get("currency")
                or account.get("currency_code")
                or ""
            ).upper()
            if not currency:
                continue
            available_value = self._to_decimal(
                available_balance.get("value")
                or available_balance.get("amount")
                or account.get("available")
                or 0
            )
            if available_value <= 0:
                continue
            balances[currency] = balances.get(currency, Decimal("0")) + available_value
        return balances

    def get_best_bid_ask_many(
        self,
        product_ids: list[str],
        prefer_private: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Retorna mejor bid/ask actual para multiples productos."""
        if not product_ids:
            return {}

        client = self.private_client if prefer_private and self.private_client is not None else self.public_client
        results: dict[str, dict[str, float]] = {}
        normalized_product_ids = [product_id.strip().upper() for product_id in product_ids if product_id.strip()]
        for start_index in range(0, len(normalized_product_ids), 50):
            batch_ids = normalized_product_ids[start_index : start_index + 50]
            response = client.get_best_bid_ask(product_ids=batch_ids)
            payload = self._to_plain(response)
            pricebooks = list(payload.get("pricebooks", []))
            for pricebook in pricebooks:
                product_id = str(pricebook.get("product_id", "")).upper()
                bids = list(pricebook.get("bids", []))
                asks = list(pricebook.get("asks", []))
                if not product_id or not bids or not asks:
                    continue

                bid = self._to_decimal(bids[0].get("price", 0))
                ask = self._to_decimal(asks[0].get("price", 0))
                if bid <= 0 or ask <= 0:
                    continue

                mid = (bid + ask) / Decimal("2")
                spread_pct = (ask - bid) / mid if mid > 0 else Decimal("0")
                results[product_id] = {
                    "product_id": product_id,
                    "bid": float(bid),
                    "ask": float(ask),
                    "mid": float(mid),
                    "spread_pct": float(spread_pct),
                }
        return results

    def get_best_bid_ask(
        self,
        product_id: str,
        prefer_private: bool = True,
    ) -> dict[str, float]:
        """Retorna mejor bid/ask y spread actual del libro."""
        results = self.get_best_bid_ask_many([product_id], prefer_private=prefer_private)
        normalized_product_id = product_id.strip().upper()
        if normalized_product_id not in results:
            raise RuntimeError(f"No hay libro disponible para {normalized_product_id}.")
        return results[normalized_product_id]

    def get_best_bid_ask_public(self, product_id: str) -> dict[str, float]:
        """Estima bid/ask del ultimo trade publico. Sin auth, para dry_run."""
        response = self.public_client.get_public_market_trades(
            product_id=product_id.strip().upper(),
            limit=1,
        )
        payload = self._to_plain(response)
        trades = list(payload.get("trades", []))
        if not trades:
            raise RuntimeError(f"Sin trades publicos para {product_id}.")
        price = float(self._to_decimal(trades[0].get("price", 0)))
        if price <= 0:
            raise RuntimeError(f"Precio invalido en trades publicos para {product_id}.")
        spread = price * 0.001
        normalized = product_id.strip().upper()
        return {
            "product_id": normalized,
            "bid": round(price - spread / 2, 8),
            "ask": round(price + spread / 2, 8),
            "mid": price,
            "spread_pct": 0.001,
        }

    def get_product_snapshot(self, product_id: str) -> CoinbaseProductSnapshot:
        """Carga metadata publica del producto spot."""
        response = self.public_client.get_public_product(product_id)
        payload = self._to_plain(response)
        return CoinbaseProductSnapshot(
            product_id=str(payload.get("product_id", product_id)),
            base_asset=str(payload.get("base_currency_id") or payload.get("base_display_symbol") or product_id.split("-", 1)[0]).upper(),
            quote_asset=str(payload.get("quote_currency_id") or payload.get("quote_display_symbol") or product_id.split("-", 1)[1]).upper(),
            base_increment=self._to_decimal(payload.get("base_increment", "0.00000001")),
            quote_increment=self._to_decimal(payload.get("quote_increment", "0.01")),
            base_min_size=self._to_decimal(payload.get("base_min_size", "0")),
            quote_min_size=self._to_decimal(payload.get("quote_min_size", "0")),
            trading_disabled=bool(payload.get("trading_disabled", False)),
            is_disabled=bool(payload.get("is_disabled", False)),
            status=str(payload.get("status", "")),
        )

    def get_candles(
        self,
        product_id: str,
        start: int,
        end: int,
        granularity: str = "ONE_MINUTE",
        limit: int | None = None,
        prefer_private: bool = True,
    ) -> list[dict[str, Any]]:
        """Retorna velas normalizadas y ordenadas ascendentemente."""
        client = self.private_client if prefer_private and self.private_client is not None else self.public_client
        getter_name = "get_candles" if client is self.private_client else "get_public_candles"
        response = getattr(client, getter_name)(
            product_id=product_id,
            start=str(int(start)),
            end=str(int(end)),
            granularity=granularity,
            limit=limit,
        )
        payload = self._to_plain(response)
        raw_candles = list(payload.get("candles", []))
        interval_ms = self._granularity_to_ms(granularity)
        normalized: list[dict[str, Any]] = []
        for candle in raw_candles:
            open_time_ms = int(candle["start"]) * 1000
            normalized.append(
                {
                    "product_id": product_id,
                    "open_time": open_time_ms,
                    "close_time": open_time_ms + interval_ms - 1,
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle["volume"]),
                    "trade_count": 0,
                    "is_closed": True,
                }
            )
        normalized.sort(key=lambda item: int(item["open_time"]))
        return normalized

    def place_market_buy_quote(self, product_id: str, quote_size: float | Decimal) -> dict[str, Any]:
        """Envia una compra market usando notional en quote currency."""
        client = self._require_private_client()
        product = self.get_product_snapshot(product_id)
        normalized_quote_size = self._normalize_size(
            self._to_decimal(quote_size),
            increment=product.quote_increment,
            minimum=product.quote_min_size,
        )
        response = client.market_order_buy(
            client_order_id=self._new_client_order_id("buy"),
            product_id=product.product_id,
            quote_size=self._format_decimal(normalized_quote_size),
        )
        return self._to_plain(response)

    def place_market_sell_base(self, product_id: str, base_size: float | Decimal) -> dict[str, Any]:
        """Envia una venta market usando size del activo base."""
        client = self._require_private_client()
        product = self.get_product_snapshot(product_id)
        normalized_base_size = self._normalize_size(
            self._to_decimal(base_size),
            increment=product.base_increment,
            minimum=product.base_min_size,
        )
        response = client.market_order_sell(
            client_order_id=self._new_client_order_id("sell"),
            product_id=product.product_id,
            base_size=self._format_decimal(normalized_base_size),
        )
        return self._to_plain(response)

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Consulta el estado actual de una orden."""
        client = self._require_private_client()
        return self._to_plain(client.get_order(order_id))

    @staticmethod
    def _build_public_client(timeout: int) -> Any:
        from coinbase.rest import RESTClient  # type: ignore

        return RESTClient(timeout=timeout)

    def _build_private_client(self) -> Any:
        if not self.credentials_path.exists():
            return None

        from coinbase.rest import RESTClient  # type: ignore

        return RESTClient(key_file=str(self.credentials_path), timeout=self.timeout)

    def _require_private_client(self) -> Any:
        if self.private_client is None:
            raise FileNotFoundError(
                f"No se encontro el archivo de credenciales Coinbase en {self.credentials_path}."
            )
        return self.private_client

    @staticmethod
    def _new_client_order_id(prefix: str) -> str:
        return f"{prefix}-{uuid4().hex[:24]}"

    @classmethod
    def _normalize_size(
        cls,
        value: Decimal,
        increment: Decimal,
        minimum: Decimal,
    ) -> Decimal:
        normalized = cls._round_down(value, increment)
        if normalized < minimum:
            raise ValueError(f"Size {normalized} por debajo del minimo {minimum}.")
        return normalized

    @staticmethod
    def _round_down(value: Decimal, increment: Decimal) -> Decimal:
        if increment <= 0:
            return value
        quantized = value.quantize(increment, rounding=ROUND_DOWN)
        if quantized <= 0:
            return Decimal("0")
        return quantized

    @staticmethod
    def _to_decimal(value: Any) -> Decimal:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    @staticmethod
    def _format_decimal(value: Decimal) -> str:
        return format(value.normalize(), "f")

    def _load_credentials_payload(self) -> dict[str, str]:
        if not self.credentials_path.exists():
            return {}
        payload = json.loads(self.credentials_path.read_text(encoding="utf-8"))
        private_key = str(payload.get("privateKey", ""))
        return {
            "name": str(payload.get("name", "")),
            "privateKey": private_key.replace("\\n", "\n"),
        }

    @staticmethod
    def _granularity_to_ms(granularity: str) -> int:
        mapping = {
            "ONE_MINUTE": 60_000,
            "FIVE_MINUTE": 5 * 60_000,
            "FIFTEEN_MINUTE": 15 * 60_000,
            "THIRTY_MINUTE": 30 * 60_000,
            "ONE_HOUR": 60 * 60_000,
            "TWO_HOUR": 2 * 60 * 60_000,
            "SIX_HOUR": 6 * 60 * 60_000,
            "ONE_DAY": 24 * 60 * 60_000,
        }
        return mapping.get(granularity.upper(), 60_000)

    @classmethod
    def _to_plain(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, list):
            return [cls._to_plain(item) for item in value]
        if isinstance(value, dict):
            return {str(key): cls._to_plain(item) for key, item in value.items()}
        if hasattr(value, "to_dict"):
            return cls._to_plain(value.to_dict())
        if hasattr(value, "__dict__"):
            return {
                str(key): cls._to_plain(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        try:
            return {str(key): cls._to_plain(item) for key, item in dict(value).items()}
        except Exception:
            return value
