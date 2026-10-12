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
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Protocol, Sequence

import pandas as pd

from risk import PositionSize, PropFirmRiskManager
from strategy import (
    DEFAULT_ATR_STOP_MULTIPLIER,
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
    )


def format_decision_prompt(state: MarketState) -> str:
    """Discrete Jev-style prompt. Next token is expected to be BUY|SELL|HOLD."""
    return (
        "State:\n"
        f"close={state.close:.6f}\n"
        f"ema_200={state.ema_200:.6f}\n"
        f"ema_status={state.ema_status}\n"
        f"stoch_k={state.stoch_k:.4f}\n"
        f"stoch_d={state.stoch_d:.4f}\n"
        f"stoch_regime={state.stoch_regime}\n"
        f"atr_14={state.atr_14:.6f}\n"
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


def extract_decision(
    state: MarketState,
    backend: Optional[TokenLogitBackend] = None,
    min_confidence: float = 0.0,
) -> Decision:
    """Softmax over BUY/SELL/HOLD scores only; never parse generated text."""
    prompt = format_decision_prompt(state)
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
    if confidence < min_confidence:
        action = "HOLD"
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
        )

    stop_price, stop_distance = _stop_for_action(decision.market, decision.action)
    size = risk.size_position(stop_distance, point_value=point_value)
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
    )


@dataclass
class TradingEngine:
    """Wires strategy indicators, Jev logit readout, and prop-firm risk."""

    risk: PropFirmRiskManager
    backend: Optional[TokenLogitBackend] = None
    min_confidence: float = 0.0
    auto_reserve: bool = False

    def step(self, df: pd.DataFrame) -> ExecutionPayload:
        state = ingest_market_state(df)
        backend = self.backend
        if backend is None:
            backend = FeatureLogitBackend(state)
        decision = extract_decision(
            state,
            backend=backend,
            min_confidence=self.min_confidence,
        )
        payload = build_execution_payload(decision, self.risk)
        if self.auto_reserve and payload.approved and payload.size is not None:
            self.risk.reserve_open_risk(payload.size)
        return payload
