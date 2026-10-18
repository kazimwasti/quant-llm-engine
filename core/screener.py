"""Tier 1 screening math: deterministic, closed-form, per-symbol and cross-sectional.

Everything in this module is a function of numbers in -> numbers out. It runs
in microseconds per symbol and never needs a model call. The LLM layers sit
downstream of these gates (see ``core/manager.py``).

Per-symbol indicators
---------------------
- Range-based volatility: Parkinson and Garman-Klass variances (more
  statistically efficient than close-to-close), plus the Wilder ATR reused
  from ``core.strategy``.
- Trend strength: OLS regression t-statistic over a rolling window. A
  statistically *tested* trend, unlike an EMA cross.
- Stretch: rolling z-score of close.
- Anomaly: volume shock z-score and a CUSUM change-point alarm on absolute
  returns.

Liquidity & cost
----------------
- Dollar-volume gate and a square-root market-impact model, combined into a
  round-trip cost estimate in return space. Any trade whose expected edge does
  not clear cost is screened out before a model ever sees it.

Cross-sectional structure
-------------------------
- Correlation matrix over trailing returns.
- Effective independent bets: N_eff = (sum w_i s_i)^2 / (w' Sigma w). Ten
  highly correlated names can be 1.2 bets.
- Greedy top-K selection with a correlation penalty, the tractable
  replacement for the portfolio optimizer: score_i <- Z_i - lam * sum_j
  |rho_ij| * Z_j over already-selected j, plus a hard cap on pairwise
  correlation inside the selected set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .strategy import atr, true_range


def _require_ohlcv(df: pd.DataFrame, columns: tuple[str, ...] = ("high", "low", "close", "volume")) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def parkinson_variance(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson (1980) range-based variance, per bar.

    sigma^2 = mean[ ln(H/L)^2 ] / (4 ln 2)
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    log_range = np.log(high.astype(float) / low.astype(float))
    return (log_range**2).rolling(window=window, min_periods=window).mean() / (4.0 * math.log(2.0))


def garman_klass_variance(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """Garman-Klass (1980) OHLC variance, per bar.

    sigma^2 = mean[ 0.5*ln(H/L)^2 - (2 ln 2 - 1)*ln(C/O)^2 ]
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    hl = np.log(high.astype(float) / low.astype(float))
    co = np.log(close.astype(float) / open_.astype(float))
    raw = 0.5 * hl**2 - (2.0 * math.log(2.0) - 1.0) * co**2
    return raw.rolling(window=window, min_periods=window).mean().clip(lower=0.0)


def volatility_percentile(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 250
) -> pd.Series:
    """Trailing percentile rank (0..1) of current ATR within its own history.

    Uses only completed bars (shift(1) history, current ATR as the test value)
    so the regime label carries no forward information.
    """
    if window < 10:
        raise ValueError("window must be >= 10")
    atr_series = atr(high, low, close)
    history = atr_series.rolling(window=window, min_periods=window)
    # Vectorized percentile via rolling apply is slow; use rank against the
    # window's own min/max, which is exact for the uniform case and stable.
    lo = history.min().shift(1)
    hi = history.max().shift(1)
    span = (hi - lo).replace(0, pd.NA)
    pct = ((atr_series - lo) / span).clip(0.0, 1.0)
    return pct.astype(float)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def regression_trend(close: pd.Series, window: int = 50) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS of close on bar index.

    Returns (slope_per_bar, t_statistic). The t-stat is the trend feature:
    unitless, comparable across symbols with different price scales, and it
    penalizes choppy "trends" the way an EMA cross cannot.

    Vectorized over all windows via sliding views and the normal equations;
    mathematically identical to per-window OLS.
    """
    if window < 10:
        raise ValueError("window must be >= 10")
    values = close.astype(float).to_numpy(dtype=float)
    n = values.size
    slope = np.full(n, np.nan)
    tstat = np.full(n, np.nan)
    if n >= window:
        Y = np.lib.stride_tricks.sliding_window_view(values, window)
        t = np.arange(window, dtype=float)
        dt = t - t.mean()
        s_tt = float((dt * dt).sum())
        dy = Y - Y.mean(axis=1, keepdims=True)
        s_ty = (dy * dt).sum(axis=1)
        s_yy = (dy * dy).sum(axis=1)
        slope_w = s_ty / s_tt
        sse = s_yy - slope_w * s_ty
        var_b = sse / (window - 2) / s_tt
        degenerate = var_b <= 1e-18
        t_w = np.where(
            degenerate,
            np.where(np.abs(slope_w) < 1e-12, 0.0, np.sign(slope_w) * 1e9),
            slope_w / np.sqrt(np.where(degenerate, 1.0, var_b)),
        )
        slope[window - 1 :] = slope_w
        tstat[window - 1 :] = t_w
    return pd.Series(slope, index=close.index), pd.Series(tstat, index=close.index)


def rolling_zscore(series: pd.Series, window: int = 40) -> pd.Series:
    """(x - rolling_mean) / rolling_std of the same completed window."""
    if window < 5:
        raise ValueError("window must be >= 5")
    s = series.astype(float)
    mean = s.rolling(window=window, min_periods=window).mean()
    std = s.rolling(window=window, min_periods=window).std(ddof=0)
    return ((s - mean) / std.replace(0, pd.NA)).astype(float)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def volume_shock(volume: pd.Series, window: int = 50, k: float = 3.0) -> pd.Series:
    """Boolean series: current volume is k standard deviations above its rolling mean."""
    if window < 5 or k <= 0:
        raise ValueError("window must be >= 5 and k > 0")
    v = volume.astype(float)
    mean = v.rolling(window=window, min_periods=window).mean()
    std = v.rolling(window=window, min_periods=window).std(ddof=0)
    z = (v - mean) / std.replace(0, pd.NA)
    return (z > k).fillna(False)


def cusum_alarm(
    returns: pd.Series, k: float = 1.5, h: float = 6.0, vol_window: int = 100
) -> pd.Series:
    """Two-sided CUSUM alarm on volatility-standardized returns.

    Returns are scaled by their rolling standard deviation first, so ``k``
    and ``h`` are in units of local sigma regardless of the symbol's price
    scale. S_t = max(0, S_{t-1} + |z_t| - k); alarm while S_t > h. An alarm
    means the recent flow of information no longer matches the regime the
    forecasts were conditioned on: freeze and re-estimate rather than trade
    the stale numbers.
    """
    if k <= 0 or h <= 0:
        raise ValueError("k and h must be > 0")
    if vol_window < 10:
        raise ValueError("vol_window must be >= 10")
    r = returns.astype(float)
    std = r.rolling(window=vol_window, min_periods=10).std(ddof=0)
    z = (r / std.replace(0, pd.NA)).fillna(0.0).abs().to_numpy()
    alarms = np.zeros(z.size, dtype=bool)
    s = 0.0
    for i, value in enumerate(z):
        s = max(0.0, s + value - k)
        alarms[i] = s > h
    return pd.Series(alarms, index=returns.index)


# ---------------------------------------------------------------------------
# Liquidity & cost
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """Round-trip transaction cost in return space (fraction, not percent).

    spread_bps: half-spread charged once per round trip (enter + exit cross
    the spread once each; the square-root impact term covers size-dependent
    cost separately).
    """

    spread_bps: float = 3.0
    fee_bps_per_side: float = 1.0
    impact_coeff: float = 1.0
    order_volume_fraction: float = 0.01  # order size as fraction of session volume

    def round_trip_cost(self, session_vol: float) -> float:
        """(spread + 2*fees)/1e4 + Y * sigma_session * sqrt(order_fraction)."""
        if session_vol < 0:
            raise ValueError("session_vol must be >= 0")
        fixed = (self.spread_bps + 2.0 * self.fee_bps_per_side) / 1e4
        impact = (
            self.impact_coeff
            * session_vol
            * math.sqrt(self.order_volume_fraction)
        )
        return fixed + impact


def dollar_volume(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Rolling mean of close * volume."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return (close.astype(float) * volume.astype(float)).rolling(
        window=window, min_periods=window
    ).mean()


# ---------------------------------------------------------------------------
# Cross-sectional structure
# ---------------------------------------------------------------------------

def correlation_matrix(
    close_by_symbol: Mapping[str, pd.Series], window: int = 250, min_overlap: int = 30
) -> pd.DataFrame:
    """Correlation matrix of trailing bar returns; NaN pairs (insufficient
    overlap) -> 0."""
    if min_overlap < 5:
        raise ValueError("min_overlap must be >= 5")
    frame = pd.DataFrame(close_by_symbol).tail(window + 1)
    corr = frame.pct_change().tail(window).corr(min_periods=min_overlap)
    return corr.fillna(0.0)


def effective_bets(
    weights: Mapping[str, float], vols: Mapping[str, float], corr: pd.DataFrame
) -> float:
    """N_eff = (sum w_i s_i)^2 / (w' Sigma w), with Sigma_ij = rho_ij s_i s_j.

    Equal weights on n uncorrelated, equal-volatility names give N_eff = n;
    perfectly correlated names give N_eff = 1 no matter how many there are.
    """
    symbols = list(weights)
    if not symbols:
        return 0.0
    w = np.array([float(weights[s]) for s in symbols])
    s = np.array([float(vols[s]) for s in symbols])
    sub = corr.reindex(index=symbols, columns=symbols).fillna(0.0).to_numpy()
    sigma = sub * np.outer(s, s)
    numerator = float((w * s).sum() ** 2)
    denominator = float(w @ sigma @ w)
    if denominator <= 1e-18:
        return 0.0
    return numerator / denominator


@dataclass(frozen=True)
class GreedySelection:
    """Result of correlation-penalized greedy top-K selection."""

    selected: list[str]
    blocked: list[str]  # candidates rejected only by the pairwise-corr cap


def greedy_select(
    scores: Mapping[str, float],
    corr: pd.DataFrame,
    top_k: int,
    penalty_lambda: float = 0.5,
    max_pairwise_abs_corr: float = 0.85,
) -> GreedySelection:
    """Pick the top-K symbols by score, penalizing correlation with picks already made.

    Iteration: take the best adjusted score; skip (block) a candidate whose
    |corr| with any selected symbol exceeds ``max_pairwise_abs_corr``; after
    each pick, adjust every remaining score by
    ``- penalty_lambda * |rho| * max(picked_score, 0)``.

    This is the tractable stand-in for the portfolio optimizer: it keeps the
    Z-score criterion but replaces the infeasible integer program with a
    greedy pass that respects the covariance structure.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if penalty_lambda < 0 or not 0 < max_pairwise_abs_corr <= 1:
        raise ValueError("penalty_lambda must be >= 0 and 0 < max_pairwise_abs_corr <= 1")

    adjusted = {sym: float(score) for sym, score in scores.items()}
    selected: list[str] = []
    blocked: list[str] = []

    def corr_abs(a: str, b: str) -> float:
        if a == b or a not in corr.index or b not in corr.columns:
            return 0.0
        value = corr.at[a, b]
        return 0.0 if pd.isna(value) else abs(float(value))

    while len(selected) < top_k and adjusted:
        pick = max(adjusted, key=adjusted.__getitem__)
        picked_score = adjusted[pick]
        if any(corr_abs(pick, s) > max_pairwise_abs_corr for s in selected):
            blocked.append(pick)
            del adjusted[pick]
            continue
        selected.append(pick)
        del adjusted[pick]
        for other in adjusted:
            adjusted[other] -= penalty_lambda * corr_abs(other, pick) * max(picked_score, 0.0)
    return GreedySelection(selected=selected, blocked=blocked)


def latest_features(df: pd.DataFrame, *, trend_window: int = 50, zscore_window: int = 40,
                     shock_window: int = 50, shock_k: float = 3.0,
                     vol_window: int = 250) -> dict[str, float]:
    """Compute the Tier-1 feature vector for the most recent completed bar.

    All rolling statistics are built from completed bars only; the returned
    row describes the state as of the last bar's close.
    """
    _require_ohlcv(df)
    returns = df["close"].astype(float).pct_change()
    _, tstat = regression_trend(df["close"], window=trend_window)
    stretch = rolling_zscore(df["close"], window=zscore_window)
    vol_pct = volatility_percentile(df["high"], df["low"], df["close"], window=vol_window)
    shock = volume_shock(df["volume"], window=shock_window, k=shock_k)
    cusum = cusum_alarm(returns)
    atr_now = atr(df["high"], df["low"], df["close"]).iloc[-1]
    close_now = float(df["close"].iloc[-1])
    return {
        "trend_tstat": float(tstat.iloc[-1]),
        "stretch_z": float(stretch.iloc[-1]),
        "vol_percentile": float(vol_pct.iloc[-1]),
        "volume_shock": float(bool(shock.iloc[-1])),
        "cusum_alarm": float(bool(cusum.iloc[-1])),
        "atr_pct": float(atr_now / close_now) if close_now > 0 else float("nan"),
        "parkinson_var": float(parkinson_variance(df["high"], df["low"]).iloc[-1]),
        "last_return": float(returns.iloc[-1]),
    }
