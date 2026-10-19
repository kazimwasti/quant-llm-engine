"""Jev-style decision engine: indicators -> discrete logits -> risk-sized payload.

Pipeline
--------
1. Ingest the latest bar from ``strategy.py`` (200 EMA, Stochastic, Wilder ATR).
2. Format a constrained decision prompt (no open-ended generation).
3. Read next-token / continuation scores only for ``BUY``, ``SELL``, ``HOLD``,
   then softmax over that closed set (zero-hallucination action mapping).
4. Route BUY/SELL into ``PropFirmRiskManager.size_position`` and the 2.0%
   daily-loss circuit breaker before emitting an execution payload.

When no causal LM is attached, a deterministic feature-logit fallback produces
the same three-way distribution from the structured prompt vector so the rest
of the pipeline stays testable without a trained transformer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional, Protocol, Sequence

import pandas as pd

from .manager import TradeTopology
from .risk import PositionSize, PropFirmRiskManager
from .strategy import (
    DEFAULT_ATR_STOP_MULTIPLIER,
    DEFAULT_ATR_TARGET_MULTIPLIER,
    add_indicators,
    generate_signals,
    latest_snapshot,
)

ACTIONS = ("BUY", "SELL", "HOLD")
ACTION_TO_SIDE = {"BUY": "long", "SELL": "short"}


class TokenLogitBackend(Protocol):
    """One-shot scorer: prompt prefix -> raw scores for each action string."""

    def score_actions(self, prompt: str, actions: Sequence[str]) -> Mapping[str, float]:
        ...


EncodeFn = Callable[[str], list[int]]


@dataclass(frozen=True)
class MarketState:
    close: float
    ema_200: float
    stoch_k: float
    stoch_d: float
    atr_14: float
    long_stop: float
    short_stop: float
    signal: int = 0
    session_allowed: bool = True
    volatility_active: bool = True
    expected_drift: float = 0.0
    session_progress: float = 0.0
    htf_ema: float = math.nan
    long_structure_confirmed: bool = False
    short_structure_confirmed: bool = False
    bullish_rejection: bool = False
    bearish_rejection: bool = False

    @property
    def ema_status(self) -> str:
        if math.isnan(self.close) or math.isnan(self.ema_200):
            return "UNKNOWN"
        if self.close > self.ema_200:
            return "ABOVE"
        if self.close < self.ema_200:
            return "BELOW"
        return "EQUAL"

    @property
    def stoch_regime(self) -> str:
        if math.isnan(self.stoch_k) or math.isnan(self.stoch_d):
            return "UNKNOWN"
        if self.stoch_k > self.stoch_d:
            return "K_ABOVE_D"
        if self.stoch_k < self.stoch_d:
            return "K_BELOW_D"
        return "K_EQUAL_D"

    @property
    def rule_signal(self) -> str:
        if self.signal > 0:
            return "LONG"
        if self.signal < 0:
            return "SHORT"
        return "FLAT"

    def is_complete(self) -> bool:
        values = (
            self.close,
            self.ema_200,
            self.stoch_k,
            self.stoch_d,
            self.atr_14,
            self.long_stop,
            self.short_stop,
            self.expected_drift,
            self.htf_ema,
        )
        return all(math.isfinite(v) for v in values)


@dataclass(frozen=True)
class Decision:
    action: str
    probabilities: dict[str, float]
    logits: dict[str, float]
    confidence: float
    prompt: str
    source: str
    market: MarketState


@dataclass(frozen=True)
class ExecutionPayload:
    approved: bool
    action: str
    model_action: str
    quantity: float
    side: Optional[str]
    entry: Optional[float]
    stop_price: Optional[float]
    stop_distance: float
    risk_dollars: float
    probabilities: dict[str, float]
    confidence: float
    circuit_breaker: bool
    reason: str
    prompt: str
    source: str
    size: Optional[PositionSize] = None
    max_holding_bars: Optional[int] = None
    thesis: Optional["TradeThesis"] = None


@dataclass(frozen=True)
class TradePreflight:
    """Cheap quote/data checks supplied by the market-data and execution adapter."""

    data_age_seconds: float
    spread_bps: float
    estimated_impact_bps: float
    order_volume_fraction: float
    data_complete: bool = True
    trading_halted: bool = False
    corporate_action_pending: bool = False

    def __post_init__(self) -> None:
        for name in ("data_age_seconds", "spread_bps", "estimated_impact_bps", "order_volume_fraction"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    def failure_reason(self, topology: TradeTopology) -> str:
        if not self.data_complete:
            return "market-data preflight is incomplete"
        if self.trading_halted:
            return "market-data preflight reports a trading halt"
        if self.corporate_action_pending:
            return "market-data preflight reports a pending corporate action"
        if self.data_age_seconds > topology.max_data_age_seconds:
            return "market-data preflight is stale"
        if self.spread_bps > topology.max_spread_bps:
            return "estimated spread exceeds manager limit"
        if self.estimated_impact_bps > topology.max_impact_bps:
            return "estimated impact exceeds manager limit"
        if self.order_volume_fraction > topology.max_order_volume_fraction:
            return "estimated order participation exceeds manager limit"
        return ""


@dataclass(frozen=True)
class TradeThesis:
    """Audit record assembled from already-computed decision state; never generated text."""

    style: str
    entry_basis: str
    invalidation_price: float
    review_after_bars: int


def ingest_market_state(df: pd.DataFrame) -> MarketState:
    """Compute strategy columns and return the latest bar as a MarketState."""
    snapshot = latest_snapshot(generate_signals(add_indicators(df)))
    if snapshot is None:
        raise ValueError("OHLC frame is empty")
    return MarketState(
        close=float(snapshot["close"]),
        ema_200=float(snapshot.get("ema_200", math.nan)),
        stoch_k=float(snapshot.get("stoch_k", math.nan)),
        stoch_d=float(snapshot.get("stoch_d", math.nan)),
        atr_14=float(snapshot.get("atr_14", math.nan)),
        long_stop=float(snapshot.get("long_stop", math.nan)),
        short_stop=float(snapshot.get("short_stop", math.nan)),
        signal=int(snapshot.get("signal", 0) or 0),
        session_allowed=bool(snapshot.get("session_allowed", False)),
        volatility_active=bool(snapshot.get("volatility_active", False)),
        expected_drift=float(snapshot.get("ema_drift", 0.0) or 0.0),
        session_progress=float(snapshot.get("session_progress", 0.0) or 0.0),
        htf_ema=float(snapshot.get("htf_ema", math.nan)),
        long_structure_confirmed=bool(snapshot.get("long_structure_confirmed", False)),
        short_structure_confirmed=bool(snapshot.get("short_structure_confirmed", False)),
        bullish_rejection=bool(snapshot.get("bullish_rejection", False)),
        bearish_rejection=bool(snapshot.get("bearish_rejection", False)),
    )


def format_decision_prompt(state: MarketState, topology: Optional[TradeTopology] = None) -> str:
    """Constrained strategy prompt. Next token is expected to be BUY|SELL|HOLD."""
    mandate = ""
    if topology is not None:
        mandate = (
            "\nManager mandate (portfolio limits, not a directional order):\n"
            f"- Risk budget multiplier: {topology.risk_multiplier:.2f}x base risk.\n"
            f"- Maximum holding time: {topology.max_holding_bars} bars.\n"
            f"- Minimum directional confidence: {topology.min_confidence:.2f}.\n"
            f"- Maximum normalized decision entropy: {topology.max_decision_entropy:.2f}.\n"
            f"- Re-evaluation envelope multiplier: {topology.envelope_distance_multiplier:.2f}x.\n"
            f"- Preferred style: {topology.style_preference}.\n"
            f"- Context: {topology.rationale}\n"
        )
    return (
        "Strategy objective and hard decision rules:\n"
        "- Only consider setups with a 1.0x ATR stop and a minimum 2.2x ATR profit target (at least 1:2.2 reward-to-risk).\n"
        "- A directional entry requires the matching rule signal, confirmed structure, and higher-timeframe trend alignment.\n"
        "- Structural confirmation can include a directional rejection wick or a qualifying EMA/ATR pullback.\n"
        "- Session and volatility regimes must both be active.\n"
        "- If any requirement is missing, the regime is choppy/low-conviction, or evidence is ambiguous, choose HOLD.\n"
        "- These rules are constraints, not a promise of any win rate or return.\n"
        + mandate
        + "\n"
        "State:\n"
        f"close={state.close:.6f}\n"
        f"ema_200={state.ema_200:.6f}\n"
        f"ema_status={state.ema_status}\n"
        f"htf_ema={state.htf_ema:.6f}\n"
        f"htf_alignment={('ABOVE' if state.close > state.htf_ema else 'BELOW' if state.close < state.htf_ema else 'EQUAL') if math.isfinite(state.htf_ema) else 'UNKNOWN'}\n"
        f"stoch_k={state.stoch_k:.4f}\n"
        f"stoch_d={state.stoch_d:.4f}\n"
        f"stoch_regime={state.stoch_regime}\n"
        f"atr_14={state.atr_14:.6f}\n"
        f"ema_drift_per_bar={state.expected_drift:.6f}\n"
        f"session_progress={state.session_progress:.4f}\n"
        f"session_allowed={int(state.session_allowed)}\n"
        f"volatility_active={int(state.volatility_active)}\n"
        f"long_structure_confirmed={int(state.long_structure_confirmed)}\n"
        f"short_structure_confirmed={int(state.short_structure_confirmed)}\n"
        f"bullish_rejection={int(state.bullish_rejection)}\n"
        f"bearish_rejection={int(state.bearish_rejection)}\n"
        f"long_stop={state.long_stop:.6f}\n"
        f"short_stop={state.short_stop:.6f}\n"
        f"rule_signal={state.rule_signal}\n"
        "\n"
        "Options:\n"
        "[BUY] Go long. Price is above the 200 EMA with upward stochastic momentum.\n"
        "[SELL] Go short. Price is below the 200 EMA with downward stochastic momentum.\n"
        "[HOLD] Stay flat. Mixed, incomplete, or insufficient edge.\n"
        "\n"
        "Select one option. Answer only with BUY, SELL, or HOLD.\n"
        "Action:"
    )


def softmax(logits: Mapping[str, float]) -> dict[str, float]:
    if not logits:
        raise ValueError("logits must be non-empty")
    peak = max(logits.values())
    exps = {key: math.exp(value - peak) for key, value in logits.items()}
    total = sum(exps.values())
    if total <= 0:
        uniform = 1.0 / len(logits)
        return {key: uniform for key in logits}
    return {key: value / total for key, value in exps.items()}


def normalized_decision_entropy(probabilities: Mapping[str, float]) -> float:
    """Shannon entropy of the closed action distribution, scaled to [0, 1]."""
    if len(probabilities) < 2:
        return 0.0
    total = sum(max(0.0, float(value)) for value in probabilities.values())
    if total <= 0:
        return 1.0
    entropy = 0.0
    for value in probabilities.values():
        probability = max(0.0, float(value)) / total
        if probability > 0:
            entropy -= probability * math.log(probability)
    return entropy / math.log(len(probabilities))


def feature_action_logits(state: MarketState) -> dict[str, float]:
    """Closed-set logits from the structured prompt vector (LM-free fallback)."""
    if not state.is_complete() or state.atr_14 <= 0:
        return {"BUY": 0.0, "SELL": 0.0, "HOLD": 3.0}

    trend = (state.close - state.ema_200) / state.atr_14
    momentum = (state.stoch_k - state.stoch_d) / 100.0
    rule = float(state.signal)

    buy = 0.85 * trend + 2.4 * momentum + 1.25 * max(rule, 0.0)
    sell = 0.85 * (-trend) + 2.4 * (-momentum) + 1.25 * max(-rule, 0.0)
    # HOLD wins when trend/momentum disagree or both are small.
    disagreement = abs(trend) * (1.0 if trend * momentum < 0 else 0.35)
    hold = 0.55 - 0.35 * abs(trend) - 0.80 * abs(momentum) + disagreement
    if not state.long_structure_confirmed:
        buy = -math.inf
    if not state.short_structure_confirmed:
        sell = -math.inf
    if not state.long_structure_confirmed and not state.short_structure_confirmed:
        hold = max(hold, 1.0)
    return {"BUY": buy, "SELL": sell, "HOLD": hold}


class FeatureLogitBackend:
    """Jev-shaped 3-way scorer that does not call a language model."""

    def __init__(self, state: MarketState):
        self.state = state

    def score_actions(self, prompt: str, actions: Sequence[str]) -> Mapping[str, float]:
        del prompt
        base = feature_action_logits(self.state)
        return {action: float(base.get(action, 0.0)) for action in actions}


class TorchCausalLMBackend:
    """Extract continuation log-probs from a causal LM (no text generation).

    ``model`` must return either a tensor of shape (batch, seq, vocab) or an
    object with a ``.logits`` attribute (Hugging Face ``CausalLMOutput``).
    """

    def __init__(self, model, encode: EncodeFn, pad_token_id: int = 0):
        self.model = model
        self.encode = encode
        self.pad_token_id = pad_token_id

    def score_actions(self, prompt: str, actions: Sequence[str]) -> Mapping[str, float]:
        import torch

        prompt_ids = self.encode(prompt)
        if not prompt_ids:
            raise ValueError("tokenizer produced an empty prompt")

        scores: dict[str, float] = {}
        self.model.eval()
        with torch.no_grad():
            for action in actions:
                action_ids = self.encode(action)
                if not action_ids:
                    raise ValueError(f"tokenizer produced no tokens for {action!r}")
                token_ids = prompt_ids + action_ids
                x = torch.tensor([token_ids], dtype=torch.long)
                logits = self._forward_logits(x)
                log_probs = torch.log_softmax(logits[0], dim=-1)
                # Teacher-force each action token at the position that predicts it.
                offset = len(prompt_ids) - 1
                total = 0.0
                for step, token_id in enumerate(action_ids):
                    total += float(log_probs[offset + step, token_id])
                scores[action] = total
        return scores

    def _forward_logits(self, input_ids):
        output = self.model(input_ids)
        logits = getattr(output, "logits", output)
        if logits.dim() != 3:
            raise ValueError(
                f"expected logits of shape (batch, seq, vocab), got {tuple(logits.shape)}"
            )
        return logits


class TorchSequenceClassifierBackend:
    """Closed BUY/SELL/HOLD scorer backed by a PEFT sequence classifier."""

    ACTION_LABELS = {"BUY": 0, "SELL": 1, "HOLD": 2}

    def __init__(self, model, tokenizer, max_length: int = 1024):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def score_actions(self, prompt: str, actions: Sequence[str]) -> Mapping[str, float]:
        import torch

        self.model.eval()
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        device = self.model.get_input_embeddings().weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
        return {action: float(logits[self.ACTION_LABELS[action]].item()) for action in actions}


def extract_decision(
    state: MarketState,
    backend: Optional[TokenLogitBackend] = None,
    min_confidence: float = 0.65,
    topology: Optional[TradeTopology] = None,
) -> Decision:
    """Softmax over BUY/SELL/HOLD scores only; never parse generated text."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    prompt = format_decision_prompt(state, topology)
    source = "feature_fallback"
    if backend is None:
        backend = FeatureLogitBackend(state)
    else:
        source = "lm"

    raw = dict(backend.score_actions(prompt, ACTIONS))
    missing = [action for action in ACTIONS if action not in raw]
    if missing:
        raise KeyError(f"backend omitted action scores: {missing}")

    logits = {action: float(raw[action]) for action in ACTIONS}
    probabilities = softmax(logits)
    action = max(ACTIONS, key=lambda name: probabilities[name])
    confidence = probabilities[action]
    if action in ("BUY", "SELL") and probabilities[action] < min_confidence:
        action = "HOLD"
    if (
        action in ("BUY", "SELL")
        and topology is not None
        and normalized_decision_entropy(probabilities) > topology.max_decision_entropy
    ):
        action = "HOLD"
        source = f"{source}+entropy_blocked"
    if not state.session_allowed or not state.volatility_active:
        action = "HOLD"
        source = f"{source}+regime_blocked"
    if (action == "BUY" and state.close <= state.htf_ema) or (
        action == "SELL" and state.close >= state.htf_ema
    ):
        action = "HOLD"
        source = f"{source}+htf_blocked"
    if (action == "BUY" and (state.signal != 1 or not state.long_structure_confirmed)) or (
        action == "SELL" and (state.signal != -1 or not state.short_structure_confirmed)
    ):
        action = "HOLD"
        source = f"{source}+structure_or_signal_blocked"
    if not state.is_complete():
        action = "HOLD"
        source = f"{source}+incomplete_state"
    return Decision(
        action=action,
        probabilities=probabilities,
        logits=logits,
        confidence=confidence,
        prompt=prompt,
        source=source,
        market=state,
    )


def confirm_marginal_decision(
    primary: Decision,
    *,
    backend: Optional[TokenLogitBackend],
    min_confidence: float,
    topology: Optional[TradeTopology],
) -> Decision:
    """Ask an optional second scorer only for marginal directional decisions.

    This preserves the Jev one-shot path for clear decisions.  When configured,
    the second scorer runs only inside the manager's narrow confirmation band;
    disagreement fails closed to HOLD.
    """
    if backend is None or primary.action not in {"BUY", "SELL"} or topology is None:
        return primary
    if primary.confidence > min_confidence + topology.confirmation_band:
        return primary
    secondary = extract_decision(
        primary.market,
        backend=backend,
        min_confidence=min_confidence,
        topology=topology,
    )
    if secondary.action == primary.action:
        return replace(primary, source=f"{primary.source}+secondary_confirmed")
    return replace(primary, action="HOLD", source=f"{primary.source}+model_disagreement_blocked")


def build_trade_thesis(
    decision: Decision, stop_price: float, topology: Optional[TradeTopology]
) -> TradeThesis:
    """Create a concise, deterministic entry audit record without token generation."""
    style = topology.style_preference if topology is not None else "confirmed_structure"
    review_after = topology.max_holding_bars if topology is not None else 12
    basis = (
        "confirmed bullish structure, signal, and higher-timeframe alignment"
        if decision.action == "BUY"
        else "confirmed bearish structure, signal, and higher-timeframe alignment"
    )
    return TradeThesis(
        style=style,
        entry_basis=basis,
        invalidation_price=stop_price,
        review_after_bars=review_after,
    )


class DiscreteDeviationTrigger:
    """Re-arm a multi-bar expected path and emit only on envelope deviation.

    The first observation is always a decision checkpoint. Thereafter model
    re-evaluation is requested when price leaves the path's ATR envelope or
    when the discrete horizon completes. Call ``observe`` on every bar, even
    when no model decision is due.
    """

    def __init__(
        self,
        horizon: int = 12,
        envelope_atr_multiplier: float = 1.0,
    ) -> None:
        if horizon < 1 or envelope_atr_multiplier <= 0:
            raise ValueError("horizon and envelope_atr_multiplier must be positive")
        self.horizon = horizon
        self.envelope_atr_multiplier = envelope_atr_multiplier
        self.anchor_close = math.nan
        self.anchor_drift = 0.0
        self.checkpoint = 0
        self.last_deviation_atr = 0.0

    def _arm(self, state: MarketState) -> None:
        drift = state.expected_drift if math.isfinite(state.expected_drift) else 0.0
        self.anchor_close = state.close
        self.anchor_drift = drift
        self.checkpoint = 0

    def observe(self, state: MarketState, manager_multiplier: float = 1.0) -> bool:
        """Return whether the predicted path needs a new decision.

        ``manager_multiplier`` is the manager's third, cheap term in the
        envelope equation.  It multiplies the agent's existing volatility and
        confidence-derived envelope rather than replacing it: values below one
        re-evaluate sooner; values above one conserve inference calls.
        """
        if not 0.60 <= manager_multiplier <= 1.40:
            raise ValueError("manager_multiplier must be between 0.60 and 1.40")
        if not state.is_complete():
            return False
        if not math.isfinite(self.anchor_close):
            self._arm(state)
            return True
        self.checkpoint += 1
        step = min(self.checkpoint, self.horizon)
        expected = self.anchor_close + self.anchor_drift * step
        width = (
            state.atr_14
            * self.envelope_atr_multiplier
            * math.sqrt(step)
            * manager_multiplier
        )
        self.last_deviation_atr = abs(state.close - expected) / max(state.atr_14, 1e-12)
        deviated = abs(state.close - expected) > width
        expired = self.checkpoint >= self.horizon
        if deviated or expired:
            self._arm(state)
            return True
        return False


def _stop_for_action(state: MarketState, action: str) -> tuple[float, float]:
    if action == "BUY":
        stop_price = state.long_stop
        distance = state.close - stop_price
    elif action == "SELL":
        stop_price = state.short_stop
        distance = stop_price - state.close
    else:
        return math.nan, 0.0
    if not math.isfinite(distance) or distance <= 0:
        fallback = DEFAULT_ATR_STOP_MULTIPLIER * state.atr_14
        distance = fallback if math.isfinite(fallback) and fallback > 0 else 0.0
        if action == "BUY":
            stop_price = state.close - distance
        else:
            stop_price = state.close + distance
    return stop_price, distance


def build_execution_payload(
    decision: Decision,
    risk: PropFirmRiskManager,
    *,
    point_value: Optional[float] = None,
    cost_per_contract: float = 0.0,
    manager_risk_multiplier: float = 1.0,
    max_holding_bars: Optional[int] = None,
    max_decision_entropy: Optional[float] = None,
    topology: Optional[TradeTopology] = None,
    preflight: Optional[TradePreflight] = None,
) -> ExecutionPayload:
    """Size BUY/SELL through the prop-firm manager; HOLD never opens risk."""
    breaker = risk.circuit_breaker_active()
    allowed, allow_reason = risk.can_trade(additional_risk=0.0)

    if decision.action == "HOLD":
        reason = "model selected HOLD"
        if not decision.market.is_complete():
            reason = "incomplete market state; forced HOLD"
        return ExecutionPayload(
            approved=False,
            action="HOLD",
            model_action=decision.action,
            quantity=0.0,
            side=None,
            entry=decision.market.close,
            stop_price=None,
            stop_distance=0.0,
            risk_dollars=0.0,
            probabilities=decision.probabilities,
            confidence=decision.confidence,
            circuit_breaker=breaker,
            reason=reason,
            prompt=decision.prompt,
            source=decision.source,
            max_holding_bars=max_holding_bars,
        )

    if topology is not None:
        if preflight is None:
            quality_reason = "manager topology requires a market-data preflight"
        else:
            quality_reason = preflight.failure_reason(topology)
        if quality_reason:
            return ExecutionPayload(
                approved=False,
                action="HOLD",
                model_action=decision.action,
                quantity=0.0,
                side=ACTION_TO_SIDE[decision.action],
                entry=decision.market.close,
                stop_price=None,
                stop_distance=0.0,
                risk_dollars=0.0,
                probabilities=decision.probabilities,
                confidence=decision.confidence,
                circuit_breaker=breaker,
                reason=quality_reason,
                prompt=decision.prompt,
                source=f"{decision.source}+preflight_blocked",
                max_holding_bars=max_holding_bars,
            )

    if (
        max_decision_entropy is not None
        and normalized_decision_entropy(decision.probabilities) > max_decision_entropy
    ):
        return ExecutionPayload(
            approved=False,
            action="HOLD",
            model_action=decision.action,
            quantity=0.0,
            side=ACTION_TO_SIDE[decision.action],
            entry=decision.market.close,
            stop_price=None,
            stop_distance=0.0,
            risk_dollars=0.0,
            probabilities=decision.probabilities,
            confidence=decision.confidence,
            circuit_breaker=breaker,
            reason="manager entropy limit blocked directional trade",
            prompt=decision.prompt,
            source=f"{decision.source}+entropy_blocked",
            max_holding_bars=max_holding_bars,
        )

    if breaker or not allowed:
        return ExecutionPayload(
            approved=False,
            action="HOLD",
            model_action=decision.action,
            quantity=0.0,
            side=ACTION_TO_SIDE[decision.action],
            entry=decision.market.close,
            stop_price=None,
            stop_distance=0.0,
            risk_dollars=0.0,
            probabilities=decision.probabilities,
            confidence=decision.confidence,
            circuit_breaker=True,
            reason=allow_reason or risk.halt_reason or "daily-loss circuit breaker",
            prompt=decision.prompt,
            source=decision.source,
            max_holding_bars=max_holding_bars,
        )

    stop_price, stop_distance = _stop_for_action(decision.market, decision.action)
    size = risk.size_position(
        stop_distance,
        point_value=point_value,
        cost_per_contract=cost_per_contract,
        manager_risk_multiplier=manager_risk_multiplier,
    )
    if size.blocked or size.quantity <= 0:
        return ExecutionPayload(
            approved=False,
            action="HOLD",
            model_action=decision.action,
            quantity=0.0,
            side=ACTION_TO_SIDE[decision.action],
            entry=decision.market.close,
            stop_price=stop_price,
            stop_distance=stop_distance,
            risk_dollars=size.risk_dollars,
            probabilities=decision.probabilities,
            confidence=decision.confidence,
            circuit_breaker=risk.circuit_breaker_active(),
            reason=size.reason or "position sizing blocked",
            prompt=decision.prompt,
            source=decision.source,
            size=size,
            max_holding_bars=max_holding_bars,
        )

    return ExecutionPayload(
        approved=True,
        action=decision.action,
        model_action=decision.action,
        quantity=size.quantity,
        side=ACTION_TO_SIDE[decision.action],
        entry=decision.market.close,
        stop_price=stop_price,
        stop_distance=size.stop_distance,
        risk_dollars=size.risk_dollars,
        probabilities=decision.probabilities,
        confidence=decision.confidence,
        circuit_breaker=False,
        reason="validated",
        prompt=decision.prompt,
        source=decision.source,
        size=size,
        max_holding_bars=max_holding_bars,
        thesis=build_trade_thesis(decision, stop_price, topology),
    )


@dataclass
class TradingEngine:
    """Wires strategy indicators, Jev logit readout, and prop-firm risk."""

    risk: PropFirmRiskManager
    backend: Optional[TokenLogitBackend] = None
    min_confidence: float = 0.65
    auto_reserve: bool = False
    topology: Optional[TradeTopology] = None
    confirmation_backend: Optional[TokenLogitBackend] = None
    deviation_trigger: DiscreteDeviationTrigger = field(default_factory=DiscreteDeviationTrigger)

    def step(self, df: pd.DataFrame, preflight: Optional[TradePreflight] = None) -> ExecutionPayload:
        state = ingest_market_state(df)
        self.risk.session_progress = state.session_progress
        if self.risk.circuit_breaker_active():
            return ExecutionPayload(
                approved=False,
                action="HOLD",
                model_action="HOLD",
                quantity=0.0,
                side=None,
                entry=state.close,
                stop_price=None,
                stop_distance=0.0,
                risk_dollars=0.0,
                probabilities={"BUY": 0.0, "SELL": 0.0, "HOLD": 1.0},
                confidence=1.0,
                circuit_breaker=True,
                reason=self.risk.halt_reason or "session risk lock active",
                prompt="",
                source="risk_circuit_breaker",
            )
        if not self.deviation_trigger.observe(
            state,
            self.topology.envelope_distance_multiplier if self.topology is not None else 1.0,
        ):
            return ExecutionPayload(
                approved=False,
                action="HOLD",
                model_action="HOLD",
                quantity=0.0,
                side=None,
                entry=state.close,
                stop_price=None,
                stop_distance=0.0,
                risk_dollars=0.0,
                probabilities={"BUY": 0.0, "SELL": 0.0, "HOLD": 1.0},
                confidence=1.0,
                circuit_breaker=self.risk.circuit_breaker_active(),
                reason="expected path remains inside ATR deviation envelope",
                prompt="",
                source="deviation_trigger_skipped",
            )
        backend = self.backend
        if backend is None:
            backend = FeatureLogitBackend(state)
        required_confidence = max(
            self.min_confidence,
            self.risk.required_confidence(state.session_progress),
            self.topology.min_confidence if self.topology is not None else 0.0,
        )
        decision = extract_decision(
            state,
            backend=backend,
            min_confidence=required_confidence,
            topology=self.topology,
        )
        decision = confirm_marginal_decision(
            decision,
            backend=self.confirmation_backend,
            min_confidence=required_confidence,
            topology=self.topology,
        )
        payload = build_execution_payload(
            decision,
            self.risk,
            manager_risk_multiplier=(
                self.topology.risk_multiplier if self.topology is not None else 1.0
            ),
            max_holding_bars=(
                self.topology.max_holding_bars if self.topology is not None else None
            ),
            max_decision_entropy=(
                self.topology.max_decision_entropy if self.topology is not None else None
            ),
            topology=self.topology,
            preflight=preflight,
        )
        if self.auto_reserve and payload.approved and payload.size is not None:
            self.risk.reserve_open_risk(payload.size)
        return payload
