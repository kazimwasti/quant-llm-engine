"""Async TopstepX / ProjectX REST client.

Maps an approved ``engine.ExecutionPayload`` onto:

    POST https://api.topstepx.com/api/Auth/loginKey
    POST https://api.topstepx.com/api/Account/search
    POST https://api.topstepx.com/api/Contract/search
    POST https://api.topstepx.com/api/Order/place

Credentials are read from the environment (never hard-coded):

    TOPSTEPX_USERNAME
    TOPSTEPX_API_KEY
    TOPSTEPX_ACCOUNT_ID   (optional; otherwise first tradable account)
    TOPSTEPX_BASE_URL     (optional; default https://api.topstepx.com)
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

from .engine import ExecutionPayload

DEFAULT_BASE_URL = "https://api.topstepx.com"
LOGIN_PATH = "/api/Auth/loginKey"
VALIDATE_PATH = "/api/Auth/validate"
ACCOUNT_SEARCH_PATH = "/api/Account/search"
CONTRACT_SEARCH_PATH = "/api/Contract/search"
CONTRACT_AVAILABLE_PATH = "/api/Contract/available"
ORDER_PLACE_PATH = "/api/Order/place"

BRACKET_MODE_ERROR = "Brackets cannot be used with Position Brackets"


class OrderType(IntEnum):
    LIMIT = 1
    MARKET = 2
    STOP = 4
    TRAILING_STOP = 5
    JOIN_BID = 6
    JOIN_ASK = 7


class OrderSide(IntEnum):
    BID = 0  # buy
    ASK = 1  # sell


class ProjectXError(RuntimeError):
    def __init__(self, message: str, *, error_code: Optional[int] = None, body: Optional[dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.body = body or {}


@dataclass(frozen=True)
class ProjectXAccount:
    id: int
    name: str
    balance: Optional[float] = None
    can_trade: bool = False
    is_visible: bool = True
    simulated: Optional[bool] = None

    @classmethod
    def from_api(cls, raw: dict) -> "ProjectXAccount":
        return cls(
            id=int(raw["id"]),
            name=str(raw.get("name") or ""),
            balance=_maybe_float(raw.get("balance")),
            can_trade=bool(raw.get("canTrade", False)),
            is_visible=bool(raw.get("isVisible", True)),
            simulated=raw.get("simulated"),
        )


@dataclass(frozen=True)
class ProjectXContract:
    id: str
    name: str
    description: str = ""
    tick_size: Optional[float] = None
    tick_value: Optional[float] = None
    active_contract: bool = False
    symbol_id: str = ""

    @classmethod
    def from_api(cls, raw: dict) -> "ProjectXContract":
        return cls(
            id=str(raw["id"]),
            name=str(raw.get("name") or ""),
            description=str(raw.get("description") or ""),
            tick_size=_maybe_float(raw.get("tickSize")),
            tick_value=_maybe_float(raw.get("tickValue")),
            active_contract=bool(raw.get("activeContract", False)),
            symbol_id=str(raw.get("symbolId") or ""),
        )


@dataclass
class OrderResult:
    submitted: bool
    order_id: Optional[int]
    account_id: Optional[int]
    contract_id: Optional[str]
    request: dict
    response: dict
    reason: str
    stop_order_id: Optional[int] = None
    extras: dict = field(default_factory=dict)


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _payload_side(payload: ExecutionPayload) -> OrderSide:
    action = (payload.action or "").upper()
    side = (payload.side or "").lower()
    if action == "BUY" or side == "long":
        return OrderSide.BID
    if action == "SELL" or side == "short":
        return OrderSide.ASK
    raise ValueError(f"cannot map payload action={payload.action!r} side={payload.side!r} to an order side")


def _contract_size(payload: ExecutionPayload) -> int:
    quantity = int(payload.quantity)
    if quantity < 1:
        raise ValueError(f"payload quantity {payload.quantity} is below 1 contract")
    return quantity


def stop_ticks(payload: ExecutionPayload, tick_size: Optional[float]) -> Optional[int]:
    """Convert ATR stop distance into whole ticks for a stop-loss bracket."""
    if tick_size is None or tick_size <= 0:
        return None
    distance = float(payload.stop_distance or 0.0)
    if distance <= 0:
        return None
    ticks = int(round(distance / tick_size))
    return ticks if ticks >= 1 else None


class TopstepXClient:
    """Async client for the TopstepX ProjectX Gateway REST API."""

    def __init__(
        self,
        *,
        user_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        account_id: Optional[int] = None,
        live_contracts: bool = True,
        timeout: float = 20.0,
        attach_stop_bracket: bool = True,
    ):
        self.user_name = user_name or os.getenv("TOPSTEPX_USERNAME", "")
        self.api_key = api_key or os.getenv("TOPSTEPX_API_KEY", "")
        self.base_url = (base_url or os.getenv("TOPSTEPX_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        env_account = os.getenv("TOPSTEPX_ACCOUNT_ID")
        self.preferred_account_id = account_id or (int(env_account) if env_account else None)
        self.live_contracts = live_contracts
        self.attach_stop_bracket = attach_stop_bracket
        self._token: Optional[str] = None
        self._lock = asyncio.Lock()
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("TopstepX execution requires httpx; install requirements.txt") from exc
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={"accept": "application/json", "Content-Type": "application/json"},
        )

    @classmethod
    def from_env(cls, **kwargs) -> "TopstepXClient":
        return cls(**kwargs)

    async def __aenter__(self) -> "TopstepXClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def login(self, force: bool = False) -> str:
        """POST /api/Auth/loginKey and cache the session JWT."""
        async with self._lock:
            if self._token and not force:
                return self._token
            if not self.user_name or not self.api_key:
                raise ProjectXError(
                    "missing TopstepX credentials; set TOPSTEPX_USERNAME and TOPSTEPX_API_KEY"
                )
            body = await self._post(
                LOGIN_PATH,
                {"userName": self.user_name, "apiKey": self.api_key},
                authenticated=False,
            )
            token = body.get("token")
            if not token:
                raise ProjectXError("loginKey did not return a token", body=body)
            self._token = str(token)
            return self._token

    async def validate_session(self) -> bool:
        await self.login()
        body = await self._post(VALIDATE_PATH, {})
        return bool(body.get("success", True))

    async def search_accounts(self, *, only_active: bool = True) -> list[ProjectXAccount]:
        await self.login()
        body = await self._post(ACCOUNT_SEARCH_PATH, {"onlyActiveAccounts": only_active})
        return [ProjectXAccount.from_api(row) for row in body.get("accounts") or []]

    async def resolve_account(
        self,
        account_id: Optional[int] = None,
        account_name: Optional[str] = None,
    ) -> ProjectXAccount:
        """Pick a tradable account: explicit id, name match, then first canTrade."""
        accounts = await self.search_accounts(only_active=True)
        if not accounts:
            raise ProjectXError("no active TopstepX accounts returned by Account/search")

        target_id = account_id if account_id is not None else self.preferred_account_id
        if target_id is not None:
            for account in accounts:
                if account.id == int(target_id):
                    if not account.can_trade:
                        raise ProjectXError(f"account {account.id} ({account.name}) has canTrade=false")
                    return account
            raise ProjectXError(f"account id {target_id} not found among active accounts")

        if account_name:
            needle = account_name.lower()
            matches = [a for a in accounts if needle in a.name.lower()]
            if not matches:
                raise ProjectXError(f"no active account name matching {account_name!r}")
            chosen = next((a for a in matches if a.can_trade), matches[0])
            if not chosen.can_trade:
                raise ProjectXError(f"account {chosen.id} ({chosen.name}) has canTrade=false")
            return chosen

        tradable = [a for a in accounts if a.can_trade]
        if not tradable:
            raise ProjectXError("no active account with canTrade=true")
        return tradable[0]

    async def search_contracts(self, search_text: str) -> list[ProjectXContract]:
        await self.login()
        body = await self._post(
            CONTRACT_SEARCH_PATH,
            {"searchText": search_text, "live": self.live_contracts},
        )
        return [ProjectXContract.from_api(row) for row in body.get("contracts") or []]

    async def resolve_contract(self, symbol_or_id: str) -> ProjectXContract:
        """Resolve a CON.* id, root symbol (MNQ), or contract name to the active listing."""
        query = symbol_or_id.strip()
        if not query:
            raise ValueError("symbol_or_id is empty")

        if query.upper().startswith("CON."):
            matches = await self.search_contracts(query)
            exact = next((c for c in matches if c.id == query), None)
            if exact:
                return exact
            # Fall through to available list in case search truncated.
            available = await self._available_contracts()
            exact = next((c for c in available if c.id == query), None)
            if exact:
                return exact
            raise ProjectXError(f"contract id {query!r} not found")

        matches = await self.search_contracts(query)
        return self._pick_contract(query, matches)

    async def submit_payload(
        self,
        payload: ExecutionPayload,
        *,
        contract: str,
        account_id: Optional[int] = None,
        account_name: Optional[str] = None,
        order_type: OrderType = OrderType.MARKET,
        attach_stop_bracket: Optional[bool] = None,
    ) -> OrderResult:
        """Translate an approved ExecutionPayload into POST /api/Order/place.

        Unapproved / HOLD payloads are returned without calling the API.
        """
        if not payload.approved or payload.action.upper() == "HOLD":
            return OrderResult(
                submitted=False,
                order_id=None,
                account_id=None,
                contract_id=None,
                request={},
                response={},
                reason=payload.reason or "payload not approved for live submission",
            )

        account = await self.resolve_account(account_id=account_id, account_name=account_name)
        resolved = await self.resolve_contract(contract)
        side = _payload_side(payload)
        size = _contract_size(payload)
        use_bracket = self.attach_stop_bracket if attach_stop_bracket is None else attach_stop_bracket
        ticks = stop_ticks(payload, resolved.tick_size) if use_bracket else None

        request: dict[str, Any] = {
            "accountId": account.id,
            "contractId": resolved.id,
            "type": int(order_type),
            "side": int(side),
            "size": size,
            "limitPrice": float(payload.entry) if order_type == OrderType.LIMIT and payload.entry is not None else None,
            "stopPrice": None,
            "trailPrice": None,
            "customTag": f"jev-{uuid.uuid4().hex[:12]}",
        }
        if ticks is not None:
            request["stopLossBracket"] = {"ticks": ticks, "type": int(OrderType.STOP)}

        try:
            body = await self._post(ORDER_PLACE_PATH, request)
        except ProjectXError as exc:
            message = str(exc)
            if ticks is not None and BRACKET_MODE_ERROR in message:
                request.pop("stopLossBracket", None)
                body = await self._post(ORDER_PLACE_PATH, request)
                stop_id = await self._place_protective_stop(
                    account.id, resolved.id, payload, side, size
                )
                return OrderResult(
                    submitted=True,
                    order_id=body.get("orderId"),
                    account_id=account.id,
                    contract_id=resolved.id,
                    request=request,
                    response=body,
                    reason="entry placed; stop sent as a separate order (Position Brackets mode)",
                    stop_order_id=stop_id,
                )
            raise

        return OrderResult(
            submitted=True,
            order_id=body.get("orderId"),
            account_id=account.id,
            contract_id=resolved.id,
            request=request,
            response=body,
            reason="accepted",
        )

    async def _place_protective_stop(
        self,
        account_id: int,
        contract_id: str,
        payload: ExecutionPayload,
        entry_side: OrderSide,
        size: int,
    ) -> Optional[int]:
        if payload.stop_price is None:
            return None
        stop_side = OrderSide.ASK if entry_side == OrderSide.BID else OrderSide.BID
        body = await self._post(
            ORDER_PLACE_PATH,
            {
                "accountId": account_id,
                "contractId": contract_id,
                "type": int(OrderType.STOP),
                "side": int(stop_side),
                "size": size,
                "limitPrice": None,
                "stopPrice": float(payload.stop_price),
                "trailPrice": None,
                "customTag": f"jev-stop-{uuid.uuid4().hex[:10]}",
            },
        )
        return body.get("orderId")

    async def _available_contracts(self) -> list[ProjectXContract]:
        await self.login()
        body = await self._post(CONTRACT_AVAILABLE_PATH, {"live": self.live_contracts})
        return [ProjectXContract.from_api(row) for row in body.get("contracts") or []]

    @staticmethod
    def _pick_contract(query: str, matches: list[ProjectXContract]) -> ProjectXContract:
        if not matches:
            raise ProjectXError(f"no contracts matched {query!r}")
        q = query.lower()
        ranked = sorted(
            matches,
            key=lambda c: (
                0 if c.active_contract else 1,
                0 if c.name.lower() == q else 1,
                0 if c.symbol_id.lower().endswith("." + q) or c.symbol_id.lower() == q else 1,
                0 if q in c.name.lower() else 1,
                c.name,
            ),
        )
        return ranked[0]

    async def _post(self, path: str, payload: dict, *, authenticated: bool = True) -> dict:
        headers = {}
        if authenticated:
            token = await self.login()
            headers["Authorization"] = f"Bearer {token}"

        response = await self._http.post(path, json=payload, headers=headers)
        if response.status_code == 401 and authenticated:
            await self.login(force=True)
            headers["Authorization"] = f"Bearer {self._token}"
            response = await self._http.post(path, json=payload, headers=headers)

        try:
            body = response.json()
        except ValueError as exc:
            raise ProjectXError(
                f"{path} returned non-JSON status={response.status_code}: {response.text[:300]}"
            ) from exc

        if response.status_code >= 400:
            raise ProjectXError(
                body.get("errorMessage") or f"{path} HTTP {response.status_code}",
                error_code=body.get("errorCode") if isinstance(body, dict) else None,
                body=body if isinstance(body, dict) else {},
            )

        if isinstance(body, dict) and body.get("success") is False:
            raise ProjectXError(
                body.get("errorMessage") or f"{path} failed",
                error_code=body.get("errorCode"),
                body=body,
            )
        return body if isinstance(body, dict) else {"value": body}
