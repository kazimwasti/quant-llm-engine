"""Fail-closed controls for shadow and paper-trading operational tests.

This module deliberately does not know a broker's flatten or reconnect API.
Instead it protects the decision immediately before that API is called.  The
broker integration supplies the actual reconnect / cancel / flatten actions
and records their outcome.  Keeping this boundary explicit makes the safety
logic deterministic and testable without credentials or network access.
"""

from __future__ import annotations

import json
import os
from hashlib import sha256
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock

from .engine import ExecutionPayload
from .manager import PortfolioAllocation, PortfolioCapacityGuard
from .risk import PropFirmRiskManager


@dataclass(frozen=True)
class OrderAuthorization:
    """Result returned before a payload can be sent to a broker."""

    permitted: bool
    client_order_id: str
    reason: str
    requires_flatten: bool = False


@dataclass
class _JournalState:
    kill_switch_active: bool = False
    session_halt_reason: str = ""
    reserved_order_ids: list[str] = field(default_factory=list)


class SafetyJournal:
    """Small durable journal for the state that must survive a process restart.

    Reserved identifiers include requests whose broker outcome is unknown.  A
    restart must reconcile those requests with the broker before it attempts a
    new order; releasing them automatically risks a duplicate order.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._claims_dir = self.path.with_suffix(self.path.suffix + ".claims")
        self._lock = RLock()
        self._state = self._read()

    @property
    def kill_switch_active(self) -> bool:
        with self._lock:
            return self._state.kill_switch_active

    @property
    def session_halt_reason(self) -> str:
        with self._lock:
            return self._state.session_halt_reason

    def reserve(self, client_order_id: str) -> bool:
        with self._lock:
            if client_order_id in self._state.reserved_order_ids:
                return False
            if not self._claim(client_order_id):
                return False
            self._state.reserved_order_ids.append(client_order_id)
            try:
                self._write()
            except Exception:
                # Retain the claim if durability fails: an uncertain order ID
                # must not become reusable merely because the disk is unhealthy.
                self._state.reserved_order_ids.remove(client_order_id)
                raise
            return True

    def release_known_rejection(self, client_order_id: str) -> None:
        """Release an ID only when the broker definitively rejected it."""
        with self._lock:
            if client_order_id not in self._state.reserved_order_ids:
                return
            self._state.reserved_order_ids.remove(client_order_id)
            self._write()
            self._claim_path(client_order_id).unlink(missing_ok=True)

    def engage_kill_switch(self) -> None:
        with self._lock:
            self._state.kill_switch_active = True
            self._write()

    def clear_kill_switch(self) -> None:
        with self._lock:
            self._state.kill_switch_active = False
            self._write()

    def record_session_halt(self, reason: str) -> None:
        with self._lock:
            self._state.session_halt_reason = reason
            self._write()

    def clear_session_halt(self) -> None:
        with self._lock:
            self._state.session_halt_reason = ""
            self._write()

    def _read(self) -> _JournalState:
        if not self.path.exists():
            return _JournalState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return _JournalState(
                kill_switch_active=bool(raw.get("kill_switch_active", False)),
                session_halt_reason=str(raw.get("session_halt_reason", "")),
                reserved_order_ids=[str(value) for value in raw.get("reserved_order_ids", [])],
            )
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"cannot read safety journal {self.path}") from exc

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self._state), sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)

    def _claim(self, client_order_id: str) -> bool:
        """Create a cross-process exclusive reservation before an API call."""
        self._claims_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self._claim_path(client_order_id), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(client_order_id)
            stream.flush()
            os.fsync(stream.fileno())
        return True

    def _claim_path(self, client_order_id: str) -> Path:
        digest = sha256(client_order_id.encode("utf-8")).hexdigest()
        return self._claims_dir / digest


class PaperTradingSafetyController:
    """Authorize paper orders only after the operational safety checks pass.

    A new process begins disconnected.  Its owner must mark the broker session
    healthy after authentication and state reconciliation, which prevents a
    restart from replaying an order before account state is known.
    """

    def __init__(
        self,
        risk: PropFirmRiskManager,
        journal_path: str | Path,
        *,
        paper_trading: bool = True,
        capacity_guard: PortfolioCapacityGuard | None = None,
    ) -> None:
        self.risk = risk
        self.paper_trading = paper_trading
        self.journal = SafetyJournal(journal_path)
        self.connection_healthy = False
        self.capacity_guard = capacity_guard

    def mark_connection_healthy(self) -> None:
        """Call only after broker authentication and position reconciliation."""
        self.connection_healthy = True

    def mark_connection_lost(self) -> None:
        """Fail closed on an API/WebSocket disconnect or an uncertain timeout."""
        self.connection_healthy = False

    def engage_kill_switch(self) -> OrderAuthorization:
        """Persist the lock; the caller must then cancel/flatten at the broker."""
        self.journal.engage_kill_switch()
        return OrderAuthorization(
            permitted=False,
            client_order_id="",
            reason="kill switch active; cancel open orders and flatten positions",
            requires_flatten=True,
        )

    def clear_kill_switch(self) -> None:
        """Explicit operator action after a successful broker-state review."""
        self.journal.clear_kill_switch()

    def clear_session_halt(self) -> None:
        """Clear a prior-session risk lock after reset_day and reconciliation."""
        if self.risk.circuit_breaker_active():
            raise RuntimeError("cannot clear a session halt while the risk manager is locked")
        self.journal.clear_session_halt()

    def authorize(
        self,
        payload: ExecutionPayload,
        client_order_id: str,
        *,
        symbol: str | None = None,
        sector: str | None = None,
        correlations: dict[str, float] | None = None,
    ) -> OrderAuthorization:
        """Reserve a unique order ID when an approved paper order may be sent."""
        client_order_id = client_order_id.strip()
        if not client_order_id:
            return OrderAuthorization(False, client_order_id, "client_order_id is required")
        if not self.paper_trading:
            return OrderAuthorization(False, client_order_id, "paper trading mode is required")
        if self.journal.kill_switch_active:
            return OrderAuthorization(
                False,
                client_order_id,
                "kill switch active; cancel open orders and flatten positions",
                requires_flatten=True,
            )
        if self.journal.session_halt_reason:
            return OrderAuthorization(False, client_order_id, self.journal.session_halt_reason)
        if not self.connection_healthy:
            return OrderAuthorization(False, client_order_id, "broker connection is not healthy")
        if self.risk.circuit_breaker_active():
            reason = self.risk.halt_reason or "session risk lock active"
            self.journal.record_session_halt(reason)
            return OrderAuthorization(False, client_order_id, reason)
        if not payload.approved or payload.action.upper() == "HOLD":
            return OrderAuthorization(False, client_order_id, payload.reason or "payload is not approved")
        if self.capacity_guard is not None:
            if not symbol or payload.side is None:
                return OrderAuthorization(
                    False, client_order_id, "symbol and side are required for manager capacity checks"
                )
            permitted, reason = self.capacity_guard.reserve(
                PortfolioAllocation(
                    order_id=client_order_id,
                    symbol=symbol.upper(),
                    side=payload.side,
                    risk_dollars=payload.risk_dollars,
                    sector=sector,
                ),
                self.risk.equity,
                correlations,
            )
            if not permitted:
                return OrderAuthorization(False, client_order_id, reason)
        if not self.journal.reserve(client_order_id):
            if self.capacity_guard is not None:
                self.capacity_guard.release(client_order_id)
            return OrderAuthorization(
                False,
                client_order_id,
                "duplicate or unreconciled client_order_id",
            )
        return OrderAuthorization(True, client_order_id, "authorized")

    def record_definitive_rejection(self, client_order_id: str) -> None:
        """Permit a retry only after the broker conclusively rejected the request."""
        order_id = client_order_id.strip()
        self.journal.release_known_rejection(order_id)
        if self.capacity_guard is not None:
            self.capacity_guard.release(order_id)

    def release_capacity(self, client_order_id: str) -> None:
        """Release a manager allocation after a reconciled close or cancellation."""
        if self.capacity_guard is not None:
            self.capacity_guard.release(client_order_id.strip())
