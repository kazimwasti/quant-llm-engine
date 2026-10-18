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
import random
from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol

import pandas as pd

from .forecast import Forecaster, MomentumVolForecaster, PolicyWeights
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
        feats = latest_features(df)
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

        feats = latest_features(df)
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

    def screen(self, universe: Mapping[str, pd.DataFrame]) -> tuple[ScreenResult, str]:
        raw_context = build_context(universe, self.config or ScreenerConfig())
        context = ScreenContext(
            n_universe=int(raw_context["n_universe"]),
            pct_cusum_alarms=raw_context["pct_cusum_alarms"],
            avg_vol_percentile=raw_context["avg_vol_percentile"],
        )
        names = list(self.sources)
        chosen = self.bandit.select(names)
        self._last_source = chosen
        policy = self.sources[chosen].propose(context)
        result = run_screen(
            universe, config=self.config, forecaster=self.forecaster, policy=policy,
            context=raw_context,
        )
        return result, chosen

    def feedback(self, reward: float) -> None:
        if self._last_source is None:
            raise RuntimeError("no screen has run yet; nothing to score")
        self.bandit.record(self._last_source, reward)
