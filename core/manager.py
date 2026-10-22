"""Manager layer: the deterministic screen pipeline plus the policy sources above it.

Pipeline (per rebalance, e.g. every 5-minute bar close):

    500 symbols
      |  Tier 1 gates: bars, dollar volume, vol floor, CUSUM freeze     (math)
      |  Tier 2 forecast: E[X], sigma via GARCH + cheap features        (math)
      |  Cost gate: E[X] must clear round-trip cost by a multiple       (math)
      |  Z-rank + greedy correlation-penalized top-K                    (math)
      v
    shortlist (default 5) -> LLM sub-agents deep-dive -> core/risk.py sizing

The only LLM-shaped slot is ``PolicySource.propose``: it sees a numeric
context summary and returns ``PolicyWeights`` over normalized Tier-1
features -- numbers, never orders (Rule 1). Every source that proposes
weights is scored by ``BanditHarness`` against realized rewards, so an LLM
manager earns trust with outcomes, not rhetoric (Rule 2).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from threading import RLock
from typing import Mapping, Optional, Protocol

import pandas as pd

from .forecast import Forecaster, MomentumVolForecaster, PolicyWeights
from .risk import PropFirmRiskManager
from .screener import (
    CostModel,
    GreedySelection,
    correlation_matrix,
    cusum_alarm,
    dollar_volume,
    effective_bets,
    greedy_select,
    latest_features,
)


# ---------------------------------------------------------------------------
# Configuration and results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScreenerConfig:
    """All thresholds for one screen run. Numbers in, pass/fail out."""

    # Stage-0 gates
    min_bars: int = 250
    min_dollar_volume: float = 1_000_000.0
    dollar_volume_window: int = 20
    min_atr_pct: float = 1e-5          # per-bar ATR floor as fraction of price
    block_on_cusum: bool = True        # freeze symbols mid regime-break

    # Features
    trend_window: int = 50
    zscore_window: int = 40
    shock_window: int = 50
    shock_k: float = 3.0
    vol_window: int = 250
    cusum_k: float = 1.5   # in units of local sigma (see screener.cusum_alarm)
    cusum_h: float = 6.0

    # Forecast / cost / selection
    horizon_bars: int = 12
    edge_cost_multiple: float = 2.0    # E[X] >= multiple * round-trip cost
    cost: CostModel = field(default_factory=CostModel)
    top_k: int = 5
    corr_window: int = 250
    corr_penalty_lambda: float = 0.5
    max_pairwise_abs_corr: float = 0.85


@dataclass
class SymbolReport:
    """Everything the screen decided about one symbol, and why."""

    symbol: str
    bars: int = 0
    dollar_volume: float = 0.0
    passed_gates: bool = False
    gate_failures: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    expected_return: Optional[float] = None
    sigma: Optional[float] = None
    round_trip_cost: Optional[float] = None
    z_score: Optional[float] = None
    selected: bool = False
    blocked_by_correlation: bool = False
    forecast_method: str = ""


@dataclass
class ScreenResult:
    """Outcome of one screen run over the whole universe."""

    reports: dict[str, SymbolReport]
    shortlist: list[str]
    blocked: list[str]
    effective_bets: float
    correlation: pd.DataFrame
    policy: PolicyWeights
    context: dict[str, float]
    topology: Optional["TradeTopology"] = None
    # Attribution token for delayed realized-P&L feedback. Keep screen output
    # self-describing so a later screen cannot steal an earlier reward.
    policy_source: Optional[str] = None
    screen_id: Optional[int] = None
    manager_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Screen pipeline
# ---------------------------------------------------------------------------

def _gate_symbol(df: pd.DataFrame, config: ScreenerConfig) -> SymbolReport:
    report = SymbolReport(symbol="", bars=len(df))
    if len(df) < config.min_bars:
        report.gate_failures.append(f"insufficient_bars(<{config.min_bars})")
    if "volume" in df.columns:
        adv = float(dollar_volume(df["close"], df["volume"], config.dollar_volume_window).iloc[-1])
        report.dollar_volume = adv
        if not adv == adv or adv < config.min_dollar_volume:  # NaN or too thin
            report.gate_failures.append("dollar_volume")
    return report


def build_context(universe: Mapping[str, pd.DataFrame], config: ScreenerConfig) -> dict[str, float]:
    """Numeric summary handed to policy sources (and later, the manager LLM)."""
    alarms = 0
    vol_pcts: list[float] = []
    for df in universe.values():
        returns = df["close"].astype(float).pct_change().tail(config.trend_window)
        if bool(cusum_alarm(returns, config.cusum_k, config.cusum_h).iloc[-1]):
            alarms += 1
        feats = latest_features(
            df,
            trend_window=config.trend_window,
            zscore_window=config.zscore_window,
            shock_window=config.shock_window,
            shock_k=config.shock_k,
            vol_window=config.vol_window,
        )
        if feats["vol_percentile"] == feats["vol_percentile"]:
            vol_pcts.append(feats["vol_percentile"])
    n = max(len(universe), 1)
    return {
        "n_universe": float(len(universe)),
        "pct_cusum_alarms": alarms / n,
        "avg_vol_percentile": sum(vol_pcts) / max(len(vol_pcts), 1),
    }


def run_screen(
    universe: Mapping[str, pd.DataFrame],
    config: Optional[ScreenerConfig] = None,
    forecaster: Optional[Forecaster] = None,
    policy: Optional[PolicyWeights] = None,
    context: Optional[dict[str, float]] = None,
) -> ScreenResult:
    """Run gates -> forecasts -> cost gate -> Z-rank -> correlation-penalized selection."""
    config = config or ScreenerConfig()
    forecaster = forecaster or MomentumVolForecaster()
    policy = (policy or PolicyWeights.equal()).normalized()
    context = context or build_context(universe, config)

    reports: dict[str, SymbolReport] = {}
    candidates: dict[str, SymbolReport] = {}

    for symbol, df in universe.items():
        report = _gate_symbol(df, config)
        report.symbol = symbol
        reports[symbol] = report
        if report.gate_failures:
            continue

        feats = latest_features(
            df,
            trend_window=config.trend_window,
            zscore_window=config.zscore_window,
            shock_window=config.shock_window,
            shock_k=config.shock_k,
            vol_window=config.vol_window,
        )
        report.features = feats
        if feats["atr_pct"] != feats["atr_pct"] or feats["atr_pct"] < config.min_atr_pct:
            report.gate_failures.append("atr_floor")
            continue
        if config.block_on_cusum and feats["cusum_alarm"] > 0:
            report.gate_failures.append("cusum_freeze")
            continue

        candidates[symbol] = report

    # Stage-2 forecasts and cost gate on survivors only.
    z_scores: dict[str, float] = {}
    sigmas: dict[str, float] = {}
    for symbol, report in candidates.items():
        forecast = forecaster.forecast(symbol, universe[symbol], config.horizon_bars, policy)
        report.expected_return = forecast.expected_return
        report.sigma = forecast.sigma
        report.forecast_method = forecast.method
        sigma_bar = forecast.sigma / (config.horizon_bars**0.5)  # per-bar sigma
        cost = config.cost.round_trip_cost(sigma_bar)
        report.round_trip_cost = cost
        if forecast.sigma <= 1e-12 or forecast.expected_return != forecast.expected_return:
            report.gate_failures.append("degenerate_forecast")
            continue
        if forecast.expected_return < config.edge_cost_multiple * cost:
            report.gate_failures.append("edge_below_cost")
            continue
        report.z_score = (forecast.expected_return - cost) / forecast.sigma
        z_scores[symbol] = report.z_score
        sigmas[symbol] = forecast.sigma

    closes_by_symbol = {
        symbol: universe[symbol]["close"].astype(float) for symbol in z_scores
    }
    corr = correlation_matrix(closes_by_symbol, window=config.corr_window)
    selection: GreedySelection = greedy_select(
        z_scores,
        corr,
        top_k=config.top_k,
        penalty_lambda=config.corr_penalty_lambda,
        max_pairwise_abs_corr=config.max_pairwise_abs_corr,
    )
    for symbol in selection.selected:
        reports[symbol].selected = True
    for symbol in selection.blocked:
        reports[symbol].blocked_by_correlation = True

    shortlist_vols = {symbol: sigmas[symbol] for symbol in selection.selected}
    n_eff = effective_bets(
        {symbol: 1.0 for symbol in selection.selected}, shortlist_vols, corr
    )

    return ScreenResult(
        reports=reports,
        shortlist=selection.selected,
        blocked=selection.blocked,
        effective_bets=n_eff,
        correlation=corr,
        policy=policy,
        context=context,
    )


# ---------------------------------------------------------------------------
# Policy sources and the bandit that referees them
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScreenContext:
    """What a policy source is allowed to know. Numbers only."""

    n_universe: int
    pct_cusum_alarms: float
    avg_vol_percentile: float
    target_enabled: bool = False
    target_progress: float = 0.0
    session_progress: float = 0.0
    remaining_loss_budget_fraction: float = 1.0


@dataclass(frozen=True)
class PortfolioContext:
    """Numeric account state a manager may use without receiving order authority.

    ``target_progress`` is the fraction of the configured daily target already
    earned.  It is deliberately not a "catch-up" signal: being behind late in
    a session must never make the mandate more aggressive.
    """

    target_enabled: bool = False
    target_progress: float = 0.0
    session_progress: float = 0.0
    remaining_loss_budget_fraction: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "target_progress",
            "session_progress",
            "remaining_loss_budget_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    @classmethod
    def from_risk(cls, risk: PropFirmRiskManager) -> "PortfolioContext":
        target = risk.daily_profit_target
        progress = 0.0 if target is None else min(1.0, max(0.0, risk.daily_pnl / target))
        budget = risk.daily_loss_budget
        remaining = 0.0 if budget <= 0 else min(1.0, risk.remaining_daily_budget / budget)
        return cls(
            target_enabled=target is not None,
            target_progress=progress,
            session_progress=risk.session_progress,
            remaining_loss_budget_fraction=remaining,
        )


@dataclass(frozen=True)
class TradeTopology:
    """Bounded trading mandate attached to every manager-approved symbol.

    The limits are intentionally broad enough to leave entry timing and thesis
    formation to a specialist sub-agent.  They are portfolio instructions, not
    orders.  The risk manager remains the final hard limit.

    ``max_decision_entropy`` is a *maximum* normalized entropy over the three
    BUY/SELL/HOLD probabilities.  Lower entropy means a more decisive model;
    requiring a minimum entropy would have the opposite of the intended safety
    effect.
    """

    risk_multiplier: float = 1.0
    max_holding_bars: int = 12
    min_confidence: float = 0.65
    max_decision_entropy: float = 0.85
    envelope_distance_multiplier: float = 1.0
    style_preference: str = "confirmed_structure"
    confirmation_band: float = 0.07
    max_open_positions: int = 5
    max_gross_risk_fraction: float = 0.015
    max_net_directional_risk_fraction: float = 0.010
    max_sector_positions: int = 2
    max_correlated_positions: int = 2
    correlation_capacity_threshold: float = 0.80
    max_data_age_seconds: float = 90.0
    max_spread_bps: float = 15.0
    max_impact_bps: float = 12.0
    max_order_volume_fraction: float = 0.01
    reentry_cooldown_bars: int = 6
    rationale: str = "normal market conditions"

    def __post_init__(self) -> None:
        if not 0.25 <= self.risk_multiplier <= 1.20:
            raise ValueError("risk_multiplier must be between 0.25 and 1.20")
        if not 1 <= self.max_holding_bars <= 48:
            raise ValueError("max_holding_bars must be between 1 and 48")
        if not 0.50 <= self.min_confidence <= 0.95:
            raise ValueError("min_confidence must be between 0.50 and 0.95")
        if not 0.25 <= self.max_decision_entropy <= 1.0:
            raise ValueError("max_decision_entropy must be between 0.25 and 1")
        if not 0.60 <= self.envelope_distance_multiplier <= 1.40:
            raise ValueError("envelope_distance_multiplier must be between 0.60 and 1.40")
        if self.style_preference not in {"confirmed_structure", "defensive_continuation"}:
            raise ValueError("style_preference is not recognized")
        if not 0.0 <= self.confirmation_band <= 0.20:
            raise ValueError("confirmation_band must be between 0 and 0.20")
        if self.max_open_positions < 1 or self.max_sector_positions < 1:
            raise ValueError("position capacity limits must be positive")
        if self.max_correlated_positions < 0:
            raise ValueError("max_correlated_positions must be non-negative")
        if not 0 < self.max_gross_risk_fraction <= 0.05:
            raise ValueError("max_gross_risk_fraction must be in (0, 0.05]")
        if not 0 < self.max_net_directional_risk_fraction <= self.max_gross_risk_fraction:
            raise ValueError("net directional risk must be positive and no larger than gross risk")
        if not 0 < self.correlation_capacity_threshold <= 1:
            raise ValueError("correlation_capacity_threshold must be in (0, 1]")
        if self.max_data_age_seconds <= 0 or self.max_spread_bps < 0 or self.max_impact_bps < 0:
            raise ValueError("execution-quality thresholds must be non-negative")
        if not 0 < self.max_order_volume_fraction <= 0.10:
            raise ValueError("max_order_volume_fraction must be in (0, 0.10]")
        if not 0 <= self.reentry_cooldown_bars <= 48:
            raise ValueError("reentry_cooldown_bars must be between 0 and 48")


def derive_trade_topology(context: ScreenContext) -> TradeTopology:
    """Convert cheap market/account measurements into a conservative mandate.

    A modest 1.15x risk budget is allowed only early in a calm session, with
    most of the loss budget intact, and only when a configured target is still
    far away.  The policy never "chases" a target late in the session.
    """

    turbulent = context.pct_cusum_alarms >= 0.30 or context.avg_vol_percentile >= 0.80
    near_target = context.target_enabled and context.target_progress >= 0.75
    calm_early_far_from_target = (
        context.target_enabled
        and context.target_progress < 0.25
        and context.session_progress <= 0.50
        and context.remaining_loss_budget_fraction >= 0.70
        and not turbulent
    )
    if turbulent:
        return TradeTopology(
            risk_multiplier=0.75,
            max_holding_bars=6,
            min_confidence=0.70,
            max_decision_entropy=0.72,
            envelope_distance_multiplier=0.70,
            style_preference="defensive_continuation",
            rationale="turbulent regime: shorten exposure and require clearer decisions",
        )
    if near_target:
        return TradeTopology(
            risk_multiplier=0.50,
            max_holding_bars=8,
            min_confidence=0.72,
            max_decision_entropy=0.75,
            envelope_distance_multiplier=0.75,
            style_preference="defensive_continuation",
            rationale="near daily target: protect gains rather than extend risk",
        )
    if calm_early_far_from_target:
        return TradeTopology(
            risk_multiplier=1.15,
            max_holding_bars=12,
            min_confidence=0.65,
            max_decision_entropy=0.85,
            envelope_distance_multiplier=1.15,
            rationale="calm early session with ample budget and distant target",
        )
    return TradeTopology()


@dataclass(frozen=True)
class PortfolioAllocation:
    """A manager reservation for one open or pending directional position."""

    order_id: str
    symbol: str
    side: str
    risk_dollars: float
    sector: Optional[str] = None

    def __post_init__(self) -> None:
        if self.side not in {"long", "short"}:
            raise ValueError("side must be long or short")
        if not self.order_id or not self.symbol or self.risk_dollars <= 0:
            raise ValueError("order_id, symbol, and positive risk_dollars are required")


class PortfolioCapacityGuard:
    """In-memory, constant-small-work portfolio gate for manager allocations.

    It only examines existing allocations (normally a handful) and correlations
    already computed by the screen; it never launches model inference.  The
    broker-facing layer must release a reservation on a definitive rejection or
    close.
    """

    def __init__(self, topology: TradeTopology) -> None:
        self.topology = topology
        self._allocations: dict[str, PortfolioAllocation] = {}
        self._lock = RLock()

    def check(
        self,
        allocation: PortfolioAllocation,
        account_equity: float,
        correlations: Optional[Mapping[str, float]] = None,
    ) -> tuple[bool, str]:
        if account_equity <= 0:
            return False, "account_equity must be positive"
        with self._lock:
            if allocation.order_id in self._allocations:
                return False, "duplicate portfolio allocation"
            open_allocations = list(self._allocations.values())
            if len(open_allocations) >= self.topology.max_open_positions:
                return False, "manager maximum open-position capacity reached"
            gross = sum(item.risk_dollars for item in open_allocations) + allocation.risk_dollars
            if gross > account_equity * self.topology.max_gross_risk_fraction + 1e-9:
                return False, "manager gross-risk capacity reached"
            signed = sum(
                item.risk_dollars if item.side == "long" else -item.risk_dollars
                for item in open_allocations
            )
            signed += allocation.risk_dollars if allocation.side == "long" else -allocation.risk_dollars
            if abs(signed) > account_equity * self.topology.max_net_directional_risk_fraction + 1e-9:
                return False, "manager net directional-risk capacity reached"
            if allocation.sector is not None:
                sector_count = sum(item.sector == allocation.sector for item in open_allocations)
                if sector_count >= self.topology.max_sector_positions:
                    return False, "manager sector capacity reached"
            correlations = correlations or {}
            correlated_count = sum(
                abs(float(correlations.get(item.symbol, 0.0)))
                >= self.topology.correlation_capacity_threshold
                for item in open_allocations
            )
            # Count the candidate itself: a cluster of ``max`` must not grow to
            # ``max + 1``.  max(1, ...) keeps a zero limit meaning "no cluster
            # beyond the candidate alone" instead of "block everything".
            if correlated_count >= max(1, self.topology.max_correlated_positions):
                return False, "manager correlated-position capacity reached"
            return True, ""

    def reserve(
        self,
        allocation: PortfolioAllocation,
        account_equity: float,
        correlations: Optional[Mapping[str, float]] = None,
    ) -> tuple[bool, str]:
        with self._lock:
            allowed, reason = self.check(allocation, account_equity, correlations)
            if allowed:
                self._allocations[allocation.order_id] = allocation
            return allowed, reason

    def release(self, order_id: str) -> None:
        with self._lock:
            self._allocations.pop(order_id, None)

    def allocations(self) -> tuple[PortfolioAllocation, ...]:
        with self._lock:
            return tuple(self._allocations.values())


class ReentryCooldown:
    """Turnover budget: a stop-out delays re-entry, then stops blocking.

    After a stop-out, entries in the same symbol -- or in a symbol correlated
    with it at or beyond ``correlation_capacity_threshold`` -- are blocked for
    ``reentry_cooldown_bars`` bars.  The cooldown then expires on its own, so a
    genuinely strong later setup is only delayed, never forbidden.  Checks are
    dict lookups over the small stopped-symbol set; no model inference,
    feature computation, or extra market-data requests.
    """

    def __init__(self, topology: TradeTopology) -> None:
        self.topology = topology
        self._stopout_bars: dict[str, int] = {}
        self._lock = RLock()

    def record_stopout(self, symbol: str, bar: int) -> None:
        symbol = symbol.upper()
        with self._lock:
            current = self._stopout_bars.get(symbol)
            if current is None or bar >= current:
                self._stopout_bars[symbol] = bar

    def blocked(
        self,
        symbol: str,
        bar: int,
        correlations: Optional[Mapping[str, float]] = None,
    ) -> tuple[bool, str]:
        symbol = symbol.upper()
        with self._lock:
            own = self._stopout_bars.get(symbol)
            if own is not None and bar - own < self.topology.reentry_cooldown_bars:
                remaining = self.topology.reentry_cooldown_bars - (bar - own)
                return True, f"re-entry cooldown for {symbol}: {remaining} bars remaining after stop-out"
            correlations = correlations or {}
            for stopped, stop_bar in self._stopout_bars.items():
                if stopped == symbol or bar - stop_bar >= self.topology.reentry_cooldown_bars:
                    continue
                if abs(float(correlations.get(stopped, 0.0))) >= self.topology.correlation_capacity_threshold:
                    remaining = self.topology.reentry_cooldown_bars - (bar - stop_bar)
                    return True, (
                        f"re-entry cooldown for correlated {stopped}: "
                        f"{remaining} bars remaining after stop-out"
                    )
            return False, ""

    def clear(self, symbol: str) -> None:
        with self._lock:
            self._stopout_bars.pop(symbol.upper(), None)


class PolicySource(Protocol):
    """Proposes feature weights for the screen. This is the manager-LLM slot.

    An implementation formats the numeric ``ScreenContext`` into its prompt,
    parses the reply into four non-negative weights, and clamps anything
    invalid -- the LLM emits numbers, never orders. Plug the Qwen adapter in
    here exactly like ``LLMPolicySource`` below.
    """

    name: str

    def propose(self, context: ScreenContext) -> PolicyWeights:
        ...


@dataclass(frozen=True)
class StaticPolicySource:
    """Fixed weights; the deterministic baseline every LLM source must beat."""

    name: str = "static"
    weights: PolicyWeights = field(default_factory=PolicyWeights)

    def propose(self, context: ScreenContext) -> PolicyWeights:
        return self.weights


@dataclass(frozen=True)
class HeuristicPolicySource:
    """Deterministic regime tilt: after widespread regime breaks, favor
    volatility-regime and shock features over stale trend, since pre-break
    trend statistics no longer describe the tape."""

    name: str = "heuristic"
    base: PolicyWeights = field(default_factory=PolicyWeights)
    alarm_threshold: float = 0.5

    def propose(self, context: ScreenContext) -> PolicyWeights:
        if context.pct_cusum_alarms > self.alarm_threshold:
            return PolicyWeights(trend=0.5, stretch=0.5, shock=0.5, vol_regime=1.5)
        return self.base


class LLMPolicySource:
    """Integration point for the fine-tuned manager LLM (offline stub).

    Subclass and implement ``_weights_from_context``: run the Qwen adapter on
    ``build_manager_prompt`` and parse the reply with
    ``parse_manager_completion`` (both below, shared with
    ``scripts/generate_manager_dataset.py`` and ``scripts/train_manager_lora.py``
    so train-time and inference-time formats cannot drift). The pipeline
    enforces normalization, so a malformed reply degrades to a re-normalized
    vector instead of a broken screen.
    """

    def __init__(self, name: str = "llm") -> None:
        self.name = name

    def propose(self, context: ScreenContext) -> PolicyWeights:
        raise NotImplementedError(
            "wire the manager LLM here: build_manager_prompt -> adapter -> parse"
        )


# ---------------------------------------------------------------------------
# Manager LLM text contract (train-time and inference-time must match)
# ---------------------------------------------------------------------------

MANAGER_PROMPT_TEMPLATE = (
    "You are the portfolio manager of a deterministic screening pipeline.\n"
    "Given the market context JSON, propose non-negative PolicyWeights as a\n"
    "JSON object with keys trend, stretch, shock, vol_regime.\n"
    "Context: {context_json}\n"
    "Respond with only the JSON object.\n"
)


def build_manager_prompt(context: Mapping[str, float]) -> str:
    """Serialize a numeric context dict into the manager prompt."""
    ordered = {key: round(float(context[key]), 6) for key in sorted(context)}
    return MANAGER_PROMPT_TEMPLATE.format(context_json=json.dumps(ordered))


def build_manager_completion(weights: PolicyWeights) -> str:
    """Serialize normalized weights into the manager completion target."""
    normalized = weights.normalized()
    return json.dumps(
        {key: round(value, 4) for key, value in normalized.as_dict().items()}
    )


def parse_manager_completion(text: str) -> Optional[PolicyWeights]:
    """Parse an LLM reply into PolicyWeights; None if unparseable.

    Accepts the first JSON object found in the text, clamps negatives to
    zero, and returns None when no object parses or every weight is zero.
    The caller decides the fallback (equal weights is the sane default).
    """
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    payload = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    start = -1
                    continue
                try:
                    weights = PolicyWeights(
                        trend=max(0.0, float(payload.get("trend", 0.0))),
                        stretch=max(0.0, float(payload.get("stretch", 0.0))),
                        shock=max(0.0, float(payload.get("shock", 0.0))),
                        vol_regime=max(0.0, float(payload.get("vol_regime", 0.0))),
                    )
                except (TypeError, ValueError):
                    return None
                total = (
                    weights.trend + weights.stretch + weights.shock + weights.vol_regime
                )
                if total <= 0:
                    return None
                return weights
    return None


class BanditHarness:
    """Epsilon-greedy referee over policy sources.

    Each source that proposes weights gets scored on realized reward (for
    example next-session PnL of the shortlist it produced). The harness
    starts exploring uniformly and shifts toward whatever actually works.
    Deterministic given its seed.
    """

    def __init__(
        self,
        epsilon: float = 0.15,
        decay: float = 0.995,
        min_epsilon: float = 0.02,
        seed: int = 0,
    ) -> None:
        if not 0 <= epsilon <= 1 or not 0 <= min_epsilon <= epsilon:
            raise ValueError("require 0 <= min_epsilon <= epsilon <= 1")
        if not 0 < decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self._rng = random.Random(seed)
        self._mean: dict[str, float] = {}
        self._count: dict[str, int] = {}

    def select(self, names: list[str]) -> str:
        if not names:
            raise ValueError("names must not be empty")
        if len(names) == 1:
            return names[0]
        if self._rng.random() < self.epsilon:
            return self._rng.choice(names)
        return max(names, key=lambda n: (self._mean.get(n, 0.0), self._count.get(n, 0)))

    def record(self, name: str, reward: float) -> None:
        count = self._count.get(name, 0)
        mean = self._mean.get(name, 0.0)
        self._count[name] = count + 1
        self._mean[name] = mean + (reward - mean) / (count + 1)
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

    def stats(self) -> dict[str, dict[str, float]]:
        return {
            name: {"mean": self._mean[name], "count": float(self._count[name])}
            for name in self._mean
        }


@dataclass
class ManagedScreen:
    """One managed screening loop: sources propose, bandit picks, screen runs.

    Usage in a live/backtest loop::

        managed = ManagedScreen(sources={"static": src_a, "llm": src_b})
        result, chosen = managed.screen(universe)
        ... execute, observe realized pnl ...
        managed.feedback(pnl)   # scores whichever source proposed today
    """

    sources: Mapping[str, PolicySource]
    bandit: BanditHarness = field(default_factory=lambda: BanditHarness(seed=0))
    config: Optional[ScreenerConfig] = None
    forecaster: Optional[Forecaster] = None
    _last_source: Optional[str] = field(default=None, init=False)
    _screen_sequence: int = field(default=0, init=False)
    _scored_screen_ids: set[int] = field(default_factory=set, init=False)

    def screen(
        self,
        universe: Mapping[str, pd.DataFrame],
        portfolio: Optional[PortfolioContext] = None,
    ) -> tuple[ScreenResult, str]:
        """Screen a universe and attach one bounded mandate to its shortlist.

        Callers with a live ``PropFirmRiskManager`` should pass
        ``PortfolioContext.from_risk(risk)``.  Omitting it preserves a neutral
        mandate for research-only screens.
        """
        raw_context = build_context(universe, self.config or ScreenerConfig())
        portfolio = portfolio or PortfolioContext()
        raw_context.update(
            {
                "target_enabled": float(portfolio.target_enabled),
                "target_progress": portfolio.target_progress,
                "session_progress": portfolio.session_progress,
                "remaining_loss_budget_fraction": portfolio.remaining_loss_budget_fraction,
            }
        )
        context = ScreenContext(
            n_universe=int(raw_context["n_universe"]),
            pct_cusum_alarms=raw_context["pct_cusum_alarms"],
            avg_vol_percentile=raw_context["avg_vol_percentile"],
            target_enabled=portfolio.target_enabled,
            target_progress=portfolio.target_progress,
            session_progress=portfolio.session_progress,
            remaining_loss_budget_fraction=portfolio.remaining_loss_budget_fraction,
        )
        names = list(self.sources)
        chosen = self.bandit.select(names)
        self._last_source = chosen
        self._screen_sequence += 1
        policy = self.sources[chosen].propose(context)
        result = run_screen(
            universe, config=self.config, forecaster=self.forecaster, policy=policy,
            context=raw_context,
        )
        result.topology = derive_trade_topology(context)
        result.policy_source = chosen
        result.screen_id = self._screen_sequence
        result.manager_id = id(self)
        return result, chosen

    def feedback(self, reward: float, *, screen: Optional[ScreenResult] = None) -> None:
        """Score realized reward, preferably against the screen that earned it.

        Passing ``screen`` makes asynchronous/multi-symbol feedback safe: the
        reward is routed to that screen's policy and duplicate settlement is
        rejected. The no-argument attribution form remains for older callers.
        """
        if not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError("reward must be a finite number")
        if screen is None:
            source = self._last_source
            screen_id = None
        else:
            source, screen_id = screen.policy_source, screen.screen_id
            if (
                source not in self.sources
                or screen_id is None
                or screen.manager_id != id(self)
            ):
                raise ValueError("screen is missing valid manager attribution")
            if screen_id in self._scored_screen_ids:
                raise ValueError(f"screen {screen_id} has already been scored")
        if source is None:
            raise RuntimeError("no screen has run yet; nothing to score")
        self.bandit.record(source, float(reward))
        if screen_id is not None:
            self._scored_screen_ids.add(screen_id)

    def feedback_from_trades(
        self,
        screen: ScreenResult,
        trades,
        *,
        starting_equity: float,
    ) -> float:
        """Score a screen from its closed trades' net P&L as account return.

        ``trades`` should contain only trades produced by this screen's
        shortlist and settled before the next feedback batch. No-trade batches
        are not scored, avoiding a zero-reward bias for policies with slower
        trade horizons.
        """
        if not math.isfinite(starting_equity) or starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        allowed = {symbol.upper() for symbol in screen.shortlist}
        pnl = 0.0
        count = 0
        for trade in trades:
            symbol = str(trade.get("symbol", "")).upper()
            if symbol not in allowed:
                raise ValueError(f"trade symbol {symbol!r} is outside the screen shortlist")
            value = float(trade["pnl"])
            if not math.isfinite(value):
                raise ValueError("trade P&L must be finite")
            pnl += value
            count += 1
        if count == 0:
            raise ValueError("cannot score a screen without realized trades")
        reward = pnl / float(starting_equity)
        self.feedback(reward, screen=screen)
        return reward
