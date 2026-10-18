"""Generate the manager-LLM training set via retrospective grid search.

For each historical window this script runs the real screening pipeline
math (gates, cost gate, Z-rank, greedy selection) under every PolicyWeights
combination on a fixed grid, scores each combination by the cost-adjusted
forward return of its shortlist, and emits (context -> teacher-weights)
pairs:

    reward(theta) = sum_{s in shortlist(theta)} z_s * forward_return_s

where z already nets out round-trip cost. The teacher target is the average
of the top-3 positive-reward combinations (smoothing against noise-chasing);
windows where nothing both trades and makes money teach the equal-weights
default ("no edge found" is the honest majority class, kept at reduced
weight so SFT does not collapse to it).

Universes
---------
- Real windows walked through --input (default data/5m.csv), single symbol.
- Synthetic universes (scripts.run_screen.synthetic_universe) across many
  seeds to give the context distribution breadth (multi-symbol correlation,
  regime breaks, liquidity failures).

Output JSONL matches the manager prompt/completion contract in
``core.manager`` (build_manager_prompt / build_manager_completion); the
fine-tuning script is ``scripts/train_manager_lora.py``.

Run anywhere (CPU is fine):
    python scripts/generate_manager_dataset.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.forecast import PolicyWeights, fit_garch11  # noqa: E402
from core.manager import (  # noqa: E402
    ScreenerConfig,
    build_manager_completion,
    build_manager_prompt,
)
from core.screener import (  # noqa: E402
    correlation_matrix,
    cusum_alarm,
    greedy_select,
    latest_features,
)
from scripts.run_screen import synthetic_universe  # noqa: E402

GRID_VALUES = (0.25, 0.5, 1.0, 2.0)
TEACHER_TOP_K = 3
EQUAL_WEIGHTS = PolicyWeights.equal()
# Keep every Nth no-edge row. Equal-weights windows are the honest majority
# class, but at full weight they dominate SFT and the model collapses to a
# constant output; 2:1 keeps the default strong while preserving signal.
DEFAULT_KEEP_EVERY = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/5m.csv")
    parser.add_argument("--output", default="data/manager_train.jsonl")
    parser.add_argument("--history", type=int, default=700, help="bars of history per window")
    parser.add_argument("--stride", type=int, default=12, help="bars between real windows")
    parser.add_argument("--horizon", type=int, default=12, help="forward bars for the reward")
    parser.add_argument("--synthetic-universes", type=int, default=80)
    parser.add_argument("--synthetic-symbols", type=int, default=12)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--edge-cost-multiple", type=float, default=1.0)
    parser.add_argument("--garch-max-iter", type=int, default=150,
                        help="Nelder-Mead iterations per GARCH fit; warm starts make 150 plenty")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def normalized_features(feats: dict[str, float]) -> tuple[float, float, float, float]:
    """Clip features to [-1, 1] with the same scales as MomentumVolForecaster."""

    def clip1(value: float, scale: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return max(-1.0, min(1.0, value / scale))

    return (
        clip1(feats["trend_tstat"], 5.0),
        clip1(feats["stretch_z"], 3.0),
        feats["volume_shock"] * 2.0 - 1.0,
        feats["vol_percentile"] * 2.0 - 1.0,
    )


def sigma_horizon(
    returns: pd.Series,
    horizon: int,
    max_iter: int = 300,
    warm_start: tuple[float, float, float] | None = None,
) -> tuple[float, str, tuple[float, float, float] | None]:
    """Per-horizon sigma from GARCH(1,1), falling back to realized vol."""
    clean = returns.dropna().tail(1500)
    try:
        fit = fit_garch11(clean.tolist(), max_iter=max_iter, warm_start=warm_start)
        return math.sqrt(fit.variance_path(horizon)[-1]), "garch11", fit.unconstrained
    except ValueError:
        if len(clean) >= 2:
            return float(np.std(clean.to_numpy())) * math.sqrt(horizon), "realized", None
        return float("nan"), "none", None


class WindowState:
    """Everything per window that does not depend on the policy weights."""

    def __init__(
        self,
        universe: dict[str, pd.DataFrame],
        config: ScreenerConfig,
        horizon: int,
        source: str,
        window_end_bar: int,
        precomputed_features: dict[str, dict[str, float]] | None = None,
        garch_warm: dict[str, tuple[float, float, float]] | None = None,
        garch_max_iter: int = 300,
    ):
        self.config = config
        self.horizon = horizon
        self.source = source
        self.window_end_bar = window_end_bar
        self.n_universe = len(universe)
        self.garch_params: dict[str, tuple[float, float, float]] = {}

        self.normalized: dict[str, tuple[float, float, float, float]] = {}
        self.sigmas: dict[str, float] = {}
        self.candidates: list[str] = []
        self.forward_returns: dict[str, float] = {}

        for symbol, df in universe.items():
            feats = (precomputed_features or {}).get(symbol) or latest_features(df)
            if feats["atr_pct"] != feats["atr_pct"] or feats["atr_pct"] < config.min_atr_pct:
                continue
            if config.block_on_cusum and feats["cusum_alarm"] > 0:
                continue
            self.normalized[symbol] = normalized_features(feats)
            sigma_h, _method, params = sigma_horizon(
                df["close"].astype(float).pct_change(),
                horizon,
                max_iter=garch_max_iter,
                warm_start=(garch_warm or {}).get(symbol),
            )
            if params is not None:
                self.garch_params[symbol] = params
            self.sigmas[symbol] = sigma_h
            closes = df["close"].astype(float).to_numpy()
            if len(closes) > horizon and closes[-1 - horizon] > 0:
                self.forward_returns[symbol] = closes[-1] / closes[-1 - horizon] - 1.0

        self.candidates = [
            s for s in self.normalized
            if math.isfinite(self.sigmas.get(s, float("nan")))
            and s in self.forward_returns
        ]
        closes = {s: df["close"] for s, df in universe.items() if s in self.candidates}
        self.corr = correlation_matrix(closes, window=config.corr_window)

    def evaluate(self, weights: PolicyWeights) -> tuple[float, int]:
        """Reward and shortlist size for one grid point."""
        policy = weights.normalized()
        config = self.config
        z_scores: dict[str, float] = {}
        for symbol in self.candidates:
            n_trend, n_stretch, n_shock, n_regime = self.normalized[symbol]
            composite = (
                policy.trend * n_trend
                + policy.stretch * n_stretch
                + policy.shock * n_shock
                + policy.vol_regime * n_regime
            )
            sigma_h = self.sigmas[symbol]
            expected = composite * sigma_h
            sigma_bar = sigma_h / (self.horizon**0.5)
            cost = config.cost.round_trip_cost(sigma_bar)
            if expected < config.edge_cost_multiple * cost:
                continue
            z_scores[symbol] = (expected - cost) / sigma_h
        if not z_scores:
            return 0.0, 0
        selection = greedy_select(
            z_scores,
            self.corr,
            top_k=config.top_k,
            penalty_lambda=config.corr_penalty_lambda,
            max_pairwise_abs_corr=config.max_pairwise_abs_corr,
        )
        reward = sum(z_scores[s] * self.forward_returns[s] for s in selection.selected)
        return reward, len(selection.selected)


def context_for(alarm_fraction: float, avg_vol_pct: float, n_universe: int) -> dict[str, float]:
    return {
        "n_universe": float(n_universe),
        "pct_cusum_alarms": round(alarm_fraction, 6),
        "avg_vol_percentile": round(avg_vol_pct, 6),
    }


def teacher_weights(
    state: WindowState, grid: list[PolicyWeights]
) -> tuple[PolicyWeights, float, int]:
    scored = []
    for weights in grid:
        reward, size = state.evaluate(weights)
        if size > 0:
            scored.append((reward, weights))
    positive = [pair for pair in scored if pair[0] > 0]
    if not positive:
        # No grid point both traded and made money forward: the honest target
        # is the equal-weights default, not the least-bad loser.
        return EQUAL_WEIGHTS, 0.0, 0
    positive.sort(key=lambda pair: -pair[0])
    top = positive[:TEACHER_TOP_K]
    best_reward = top[0][0]
    trend = sum(w.trend for _, w in top) / len(top)
    stretch = sum(w.stretch for _, w in top) / len(top)
    shock = sum(w.shock for _, w in top) / len(top)
    vol_regime = sum(w.vol_regime for _, w in top) / len(top)
    return (
        PolicyWeights(trend=trend, stretch=stretch, shock=shock, vol_regime=vol_regime),
        best_reward,
        len(scored),
    )


def real_windows(
    df: pd.DataFrame, history: int, stride: int, horizon: int, vol_window: int
) -> list[tuple[dict[str, pd.DataFrame], int]]:
    windows = []
    warmup = max(vol_window + 50, history)
    for end in range(warmup, len(df) - horizon, stride):
        windows.append(({"MNQ": df.iloc[end - history : end].reset_index(drop=True)}, end))
    return windows


def window_context(
    frames: dict[str, pd.DataFrame],
    features: dict[str, dict[str, float]] | None = None,
) -> tuple[float, float]:
    """Cusum alarm fraction and mean vol percentile across the universe."""
    alarms = 0
    vol_pcts: list[float] = []
    for symbol, frame in frames.items():
        returns = frame["close"].astype(float).pct_change().tail(50)
        if bool(cusum_alarm(returns).iloc[-1]):
            alarms += 1
        feats = (features or {}).get(symbol) or latest_features(frame)
        vol_pcts.append(
            feats["vol_percentile"] if math.isfinite(feats["vol_percentile"]) else 0.5
        )
    return alarms / max(len(frames), 1), float(np.mean(vol_pcts)) if vol_pcts else 0.5


def main() -> int:
    args = parse_args()
    if args.history < 300:
        raise ValueError("--history must be >= 300 for the Tier-1 windows to be valid")
    if args.horizon < 1 or args.stride < 1:
        raise ValueError("--horizon and --stride must be >= 1")

    config = ScreenerConfig(edge_cost_multiple=args.edge_cost_multiple, top_k=args.top_k)
    grid = [
        PolicyWeights(trend=a, stretch=b, shock=c, vol_regime=d)
        for a, b, c, d in itertools.product(GRID_VALUES, repeat=4)
    ]

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    jobs: list[tuple[WindowState, str, int, dict[str, float]]] = []

    if input_path.is_file():
        df = pd.read_csv(input_path)
        garch_cache: dict[str, tuple[float, float, float]] = {}
        n_real = 0
        for window, end in real_windows(
            df, args.history, args.stride, args.horizon, config.vol_window
        ):
            frame = window["MNQ"]
            feats = latest_features(frame)
            state = WindowState(
                window, config, args.horizon, "real_mnq", end - 1,
                precomputed_features={"MNQ": feats},
                garch_warm=garch_cache,
                garch_max_iter=args.garch_max_iter,
            )
            garch_cache.update(state.garch_params)
            if not state.candidates:
                continue
            alarm_fraction, avg_vol_pct = window_context(window, features={"MNQ": feats})
            jobs.append(
                (state, "real_mnq", end - 1, context_for(alarm_fraction, avg_vol_pct, state.n_universe))
            )
            n_real += 1
        print(f"real windows: {n_real} (from {input_path})")
    else:
        print(f"input CSV not found ({input_path}); generating synthetic-only data")

    n_synth = 0
    for index in range(args.synthetic_universes):
        seed = args.seed + index
        frames = synthetic_universe(
            args.synthetic_symbols, n_bars=args.history + args.horizon, seed=seed
        )
        history = {
            s: frame.iloc[: args.history].reset_index(drop=True)
            for s, frame in frames.items()
        }
        features = {s: latest_features(f) for s, f in history.items()}
        state = WindowState(
            history, config, args.horizon, "synthetic", seed,
            precomputed_features=features,
            garch_max_iter=args.garch_max_iter,
        )
        if not state.candidates:
            continue
        alarm_fraction, avg_vol_pct = window_context(history, features=features)
        jobs.append(
            (state, "synthetic", seed, context_for(alarm_fraction, avg_vol_pct, state.n_universe))
        )
        n_synth += 1
    print(f"synthetic windows: {n_synth}")

    output_rows: list[dict] = []
    for state, source, end_bar, context in jobs:
        weights, reward, n_scored = teacher_weights(state, grid)
        output_rows.append(
            {
                "prompt": build_manager_prompt(context),
                "completion": build_manager_completion(weights),
                "meta": {
                    "source": source,
                    "window_end_bar": int(end_bar),
                    "forward_horizon": int(args.horizon),
                    "best_reward": round(float(reward), 8),
                    "n_scored_grid_points": int(n_scored),
                    "grid_size": len(grid),
                },
            }
        )

    kept: list[dict] = []
    default_seen = 0
    for row in output_rows:
        is_default = row["meta"]["n_scored_grid_points"] == 0
        if not is_default or default_seen % DEFAULT_KEEP_EVERY == 0:
            kept.append(row)
        if is_default:
            default_seen += 1

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")

    scored_count = sum(1 for r in kept if r["meta"]["n_scored_grid_points"] > 0)
    print(
        f"wrote {len(kept)} rows to {output_path} "
        f"(scored={scored_count}, default={len(kept)-scored_count}; "
        f"dropped {len(output_rows)-len(kept)} excess default rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
