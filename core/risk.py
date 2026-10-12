"""Universal prop-firm risk manager.

Applies to TopstepX, Apex, and other evaluation / live prop platforms.

Hard rules (always on, cannot be loosened):
- 0.5% of current equity risked per trade
- 2.0% of session starting equity daily-loss circuit breaker

A platform may impose a *tighter* daily loss cap; it cannot raise the 2.0%
breaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


PER_TRADE_RISK_PCT = 0.005
DAILY_LOSS_LIMIT_PCT = 0.02


class Platform(str, Enum):
    TOPSTEPX = "topstepx"
    APEX = "apex"
    GENERIC = "generic"


class CircuitBreakerTripped(RuntimeError):
    """Raised when a new order would violate the daily-loss breaker."""


@dataclass(frozen=True)
class PlatformProfile:
    """Optional extra constraints layered under the universal 2.0% breaker."""

    name: Platform
    daily_loss_usd: Optional[float] = None
    trailing_drawdown_usd: Optional[float] = None
    allow_overnight: bool = False


PLATFORM_PROFILES: dict[Platform, PlatformProfile] = {
    Platform.TOPSTEPX: PlatformProfile(
        name=Platform.TOPSTEPX,
        allow_overnight=False,
    ),
    Platform.APEX: PlatformProfile(
        name=Platform.APEX,
        allow_overnight=False,
    ),
    Platform.GENERIC: PlatformProfile(
        name=Platform.GENERIC,
        allow_overnight=True,
    ),
}


@dataclass
class PositionSize:
    quantity: float
    risk_dollars: float
    stop_distance: float
    dollar_risk_per_unit: float
    blocked: bool
    reason: str = ""


@dataclass
class PropFirmRiskManager:
    """Session risk state for one funded / evaluation account."""

    starting_equity: float
    platform: Platform = Platform.GENERIC
    per_trade_risk_pct: float = PER_TRADE_RISK_PCT
    daily_loss_limit_pct: float = DAILY_LOSS_LIMIT_PCT
    daily_loss_limit_usd: Optional[float] = 500.0
    max_consecutive_stopouts: int = 2
    point_value: float = 1.0
    min_quantity: float = 0.0
    quantity_step: float = 1.0
    daily_profit_target: Optional[float] = None
    baseline_confidence: float = 0.65
    max_target_confidence: float = 0.80
    confidence_escalation_progress: float = 0.75
    target_risk_floor_multiplier: float = 0.25
    session_progress: float = 0.0
    equity: float = field(init=False)
    session_start_equity: float = field(init=False)
    peak_equity: float = field(init=False)
    realized_daily_pnl: float = field(default=0.0, init=False)
    open_risk_dollars: float = field(default=0.0, init=False)
    halted: bool = field(default=False, init=False)
    halt_reason: str = field(default="", init=False)
    consecutive_stopouts: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.starting_equity <= 0:
            raise ValueError("starting_equity must be > 0")
        if not (0 < self.per_trade_risk_pct <= DAILY_LOSS_LIMIT_PCT):
            raise ValueError(
                "per_trade_risk_pct must be > 0 and <= 2.0% hard daily cap"
            )
        self.daily_loss_limit_pct = min(self.daily_loss_limit_pct, DAILY_LOSS_LIMIT_PCT)
        if self.daily_loss_limit_pct <= 0:
            raise ValueError("daily_loss_limit_pct must be > 0")
        if self.daily_loss_limit_usd is not None and self.daily_loss_limit_usd <= 0:
            raise ValueError("daily_loss_limit_usd must be > 0")
        if self.max_consecutive_stopouts < 1:
            raise ValueError("max_consecutive_stopouts must be >= 1")
        if self.point_value <= 0:
            raise ValueError("point_value must be > 0")
        if self.quantity_step <= 0:
            raise ValueError("quantity_step must be > 0")
        if self.daily_profit_target is not None and self.daily_profit_target <= 0:
            raise ValueError("daily_profit_target must be > 0 when configured")
        if not 0 <= self.baseline_confidence <= self.max_target_confidence <= 1:
            raise ValueError("confidence thresholds must satisfy 0 <= baseline <= max <= 1")
        if not 0 < self.confidence_escalation_progress <= 1:
            raise ValueError("confidence_escalation_progress must be in (0, 1]")
        if not 0 < self.target_risk_floor_multiplier <= 1:
            raise ValueError("target_risk_floor_multiplier must be in (0, 1]")
        self.equity = float(self.starting_equity)
        self.session_start_equity = float(self.starting_equity)
        self.peak_equity = float(self.starting_equity)

    @property
    def profile(self) -> PlatformProfile:
        return PLATFORM_PROFILES.get(self.platform, PLATFORM_PROFILES[Platform.GENERIC])

    @property
    def daily_pnl(self) -> float:
        return self.equity - self.session_start_equity

    @property
    def daily_loss_budget(self) -> float:
        pct_budget = self.session_start_equity * self.daily_loss_limit_pct
        if self.daily_loss_limit_usd is not None:
            pct_budget = min(pct_budget, self.daily_loss_limit_usd)
        usd_cap = self.profile.daily_loss_usd
        if usd_cap is None:
            return pct_budget
        return min(pct_budget, usd_cap)

    @property
    def remaining_daily_budget(self) -> float:
        unused = self.daily_loss_budget + min(self.daily_pnl, 0.0) - self.open_risk_dollars
        return max(0.0, unused)

    def reset_day(self, equity: Optional[float] = None) -> None:
        if equity is not None:
            if equity <= 0:
                raise ValueError("equity must be > 0")
            self.equity = float(equity)
        self.session_start_equity = self.equity
        self.peak_equity = max(self.peak_equity, self.equity)
        self.realized_daily_pnl = 0.0
        self.open_risk_dollars = 0.0
        self.halted = False
        self.halt_reason = ""
        self.session_progress = 0.0
        self.consecutive_stopouts = 0

    def record_trade_outcome(self, pnl: float, *, stop_out: bool) -> None:
        """Track stop-out streaks and immediately trip the session lock when due."""
        if stop_out:
            self.consecutive_stopouts += 1
        else:
            self.consecutive_stopouts = 0
        if self.consecutive_stopouts >= self.max_consecutive_stopouts:
            self.halted = True
            self.halt_reason = (
                f"session locked after {self.consecutive_stopouts} consecutive stop-outs"
            )
        self._trip_if_breached()

    def required_confidence(self, session_progress: Optional[float] = None) -> float:
        """Confidence gate rises near a daily target, then relaxes if behind pace."""
        if self.daily_profit_target is None:
            return self.baseline_confidence
        elapsed = self.session_progress if session_progress is None else float(session_progress)
        if not 0 <= elapsed <= 1:
            raise ValueError("session_progress must be between 0 and 1")
        progress = max(0.0, self.daily_pnl) / self.daily_profit_target
        if progress <= self.confidence_escalation_progress:
            required = self.baseline_confidence
        else:
            fraction = min(
                1.0,
                (progress - self.confidence_escalation_progress)
                / (1.0 - self.confidence_escalation_progress),
            )
            required = self.baseline_confidence + fraction * (
                self.max_target_confidence - self.baseline_confidence
            )
        if elapsed > 0.5 and progress < elapsed:
            decay = min(1.0, (elapsed - 0.5) / 0.5)
            required = self.baseline_confidence + (required - self.baseline_confidence) * (1.0 - decay)
        return min(self.max_target_confidence, max(self.baseline_confidence, required))

    def target_risk_multiplier(self) -> float:
        """Scale per-trade dollars-at-risk down as realized/unrealized target nears."""
        if self.daily_profit_target is None:
            return 1.0
        progress = max(0.0, self.daily_pnl) / self.daily_profit_target
        if progress <= self.confidence_escalation_progress:
            return 1.0
        fraction = min(
            1.0,
            (progress - self.confidence_escalation_progress)
            / (1.0 - self.confidence_escalation_progress),
        )
        return 1.0 - fraction * (1.0 - self.target_risk_floor_multiplier)

    def mark_equity(self, equity: float) -> None:
        if equity < 0:
            raise ValueError("equity cannot be negative")
        self.equity = float(equity)
        self.peak_equity = max(self.peak_equity, self.equity)
        self._trip_if_breached()

    def register_pnl(self, pnl: float) -> None:
        self.equity += float(pnl)
        self.realized_daily_pnl += float(pnl)
        self.peak_equity = max(self.peak_equity, self.equity)
        self._trip_if_breached()

    def release_open_risk(self, dollars: float) -> None:
        self.open_risk_dollars = max(0.0, self.open_risk_dollars - float(dollars))

    def circuit_breaker_active(self) -> bool:
        self._trip_if_breached()
        return self.halted

    def can_trade(self, additional_risk: float = 0.0) -> tuple[bool, str]:
        self._trip_if_breached()
        if self.halted:
            return False, self.halt_reason
        if additional_risk < 0:
            return False, "additional_risk cannot be negative"
        if additional_risk > self.remaining_daily_budget + 1e-9:
            return (
                False,
                "order risk would breach the 2.0% daily-loss circuit breaker",
            )
        return True, ""

    def size_position(
        self,
        stop_distance: float,
        *,
        equity: Optional[float] = None,
        point_value: Optional[float] = None,
        cost_per_contract: float = 0.0,
    ) -> PositionSize:
        if stop_distance <= 0:
            return PositionSize(
                quantity=0.0,
                risk_dollars=0.0,
                stop_distance=stop_distance,
                dollar_risk_per_unit=0.0,
                blocked=True,
                reason="stop_distance must be > 0",
            )

        mark = float(self.equity if equity is None else equity)
        if mark <= 0:
            return PositionSize(
                quantity=0.0,
                risk_dollars=0.0,
                stop_distance=stop_distance,
                dollar_risk_per_unit=0.0,
                blocked=True,
                reason="equity must be > 0",
            )

        pv = float(self.point_value if point_value is None else point_value)
        if pv <= 0:
            return PositionSize(
                quantity=0.0,
                risk_dollars=0.0,
                stop_distance=stop_distance,
                dollar_risk_per_unit=0.0,
                blocked=True,
                reason="point_value must be > 0",
            )
        if cost_per_contract < 0:
            return PositionSize(
                quantity=0.0,
                risk_dollars=0.0,
                stop_distance=stop_distance,
                dollar_risk_per_unit=0.0,
                blocked=True,
                reason="cost_per_contract cannot be negative",
            )

        risk_dollars = (
            mark
            * self.per_trade_risk_pct
            * self.target_risk_multiplier()
        )
        allowed, reason = self.can_trade(additional_risk=risk_dollars)
        if not allowed:
            return PositionSize(
                quantity=0.0,
                risk_dollars=risk_dollars,
                stop_distance=stop_distance,
                dollar_risk_per_unit=stop_distance * pv + cost_per_contract,
                blocked=True,
                reason=reason,
            )

        # Include round-trip commissions/slippage supplied by the caller so
        # sized quantity stays within the nominal risk budget after friction.
        dollar_risk_per_unit = stop_distance * pv + cost_per_contract
        raw_qty = risk_dollars / dollar_risk_per_unit
        quantity = self._round_quantity(raw_qty)

        if quantity <= 0 or quantity < self.min_quantity:
            return PositionSize(
                quantity=0.0,
                risk_dollars=risk_dollars,
                stop_distance=stop_distance,
                dollar_risk_per_unit=dollar_risk_per_unit,
                blocked=True,
                reason="sized quantity is below the tradable minimum",
            )

        actual_risk = quantity * dollar_risk_per_unit
        allowed, reason = self.can_trade(additional_risk=actual_risk)
        if not allowed:
            return PositionSize(
                quantity=0.0,
                risk_dollars=actual_risk,
                stop_distance=stop_distance,
                dollar_risk_per_unit=dollar_risk_per_unit,
                blocked=True,
                reason=reason,
            )

        return PositionSize(
            quantity=quantity,
            risk_dollars=actual_risk,
            stop_distance=stop_distance,
            dollar_risk_per_unit=dollar_risk_per_unit,
            blocked=False,
        )

    def reserve_open_risk(self, size: PositionSize) -> None:
        if size.blocked or size.quantity <= 0:
            raise ValueError("cannot reserve risk for a blocked / empty size")
        allowed, reason = self.can_trade(additional_risk=size.risk_dollars)
        if not allowed:
            raise CircuitBreakerTripped(reason)
        self.open_risk_dollars += size.risk_dollars

    def _round_quantity(self, raw: float) -> float:
        steps = int(raw // self.quantity_step)
        return steps * self.quantity_step

    def _trip_if_breached(self) -> None:
        loss = -min(self.daily_pnl, 0.0)
        if loss + 1e-9 >= self.daily_loss_budget:
            self.halted = True
            self.halt_reason = (
                f"{self.platform.value} daily-loss circuit breaker tripped: "
                f"loss {loss:.2f} reached the {self.daily_loss_budget:.2f} cap "
                f"(max {self.daily_loss_limit_pct * 100:.1f}% / configured USD limit)"
            )
            return

        dd_cap = self.profile.trailing_drawdown_usd
        if dd_cap is not None:
            drawdown = self.peak_equity - self.equity
            if drawdown + 1e-9 >= dd_cap:
                self.halted = True
                self.halt_reason = (
                    f"{self.platform.value} trailing drawdown {drawdown:.2f} "
                    f"reached cap {dd_cap:.2f}"
                )


class RiskManager(PropFirmRiskManager):
    """Bridge alias so backtest.py can initialize RiskManager with account_balance."""
    def __init__(self, account_balance=10000, **kwargs):
        super().__init__(starting_equity=float(account_balance), **kwargs)

    def check_risk(self, position_size, current_drawdown):
        return not self.circuit_breaker_active()


class ExecutionPayload:
    def __init__(self, symbol, action, quantity, price):
        self.symbol = symbol
        self.action = action
        self.quantity = quantity
        self.price = price
