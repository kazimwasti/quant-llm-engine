"""Trend / momentum strategy indicators and ATR-based dynamic stops.

Indicators
----------
- 200-period EMA of close
- Stochastic oscillator: %K = 14, %D = 3
- 14-period ATR (Wilder) used as a dynamic stop distance
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

EMA_PERIOD = 200
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
ATR_PERIOD = 14
DEFAULT_ATR_STOP_MULTIPLIER = 1.5


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def ema(series: pd.Series, period: int = EMA_PERIOD) -> pd.Series:
    """Exponential moving average (span = period, adjust=False)."""
    if period < 1:
        raise ValueError("period must be >= 1")
    return series.astype(float).ewm(span=period, adjust=False).mean()


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = STOCH_K_PERIOD,
    d_period: int = STOCH_D_PERIOD,
) -> tuple[pd.Series, pd.Series]:
    """Fast stochastic %K and %D.

    %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    %D = SMA(%K, d_period)
    """
    if k_period < 1 or d_period < 1:
        raise ValueError("k_period and d_period must be >= 1")

    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    denom = (highest_high - lowest_low).replace(0, pd.NA)
    percent_k = 100.0 * (close - lowest_low) / denom
    percent_k = percent_k.astype(float).fillna(0.0).clip(0.0, 100.0)
    percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()
    return percent_k, percent_d


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: max(high-low, |high-prev_close|, |low-prev_close|)."""
    high = high.astype(float)
    low = low.astype(float)
    prev_close = close.astype(float).shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = ranges.max(axis=1)
    # First bar has no previous close; fall back to high-low.
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """Wilder ATR: EWM of true range with alpha = 1/period."""
    if period < 1:
        raise ValueError("period must be >= 1")
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def dynamic_stop_loss(
    close: pd.Series,
    atr_series: pd.Series,
    side: str,
    multiplier: float = DEFAULT_ATR_STOP_MULTIPLIER,
) -> pd.Series:
    """ATR stop: long = close - m*ATR, short = close + m*ATR."""
    if multiplier <= 0:
        raise ValueError("multiplier must be > 0")
    side = side.lower()
    distance = multiplier * atr_series.astype(float)
    close = close.astype(float)
    if side == "long":
        return close - distance
    if side == "short":
        return close + distance
    raise ValueError("side must be 'long' or 'short'")


def add_indicators(
    df: pd.DataFrame,
    ema_period: int = EMA_PERIOD,
    k_period: int = STOCH_K_PERIOD,
    d_period: int = STOCH_D_PERIOD,
    atr_period: int = ATR_PERIOD,
    atr_stop_multiplier: float = DEFAULT_ATR_STOP_MULTIPLIER,
) -> pd.DataFrame:
    """Return a copy of `df` with strategy columns attached.

    Added columns
    -------------
    ema_200, stoch_k, stoch_d, atr_14, long_stop, short_stop
    """
    _require_columns(df, ("high", "low", "close"))
    out = df.copy()
    out["ema_200"] = ema(out["close"], period=ema_period)
    out["stoch_k"], out["stoch_d"] = stochastic(
        out["high"],
        out["low"],
        out["close"],
        k_period=k_period,
        d_period=d_period,
    )
    out["atr_14"] = atr(out["high"], out["low"], out["close"], period=atr_period)
    out["long_stop"] = dynamic_stop_loss(
        out["close"], out["atr_14"], side="long", multiplier=atr_stop_multiplier
    )
    out["short_stop"] = dynamic_stop_loss(
        out["close"], out["atr_14"], side="short", multiplier=atr_stop_multiplier
    )
    return out


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Attach discrete trade signals using EMA trend + stochastic cross.

    signal:  1 = long, -1 = short, 0 = flat
    Long:  close > 200 EMA and %K crosses above %D
    Short: close < 200 EMA and %K crosses below %D
    """
    required = ("close", "ema_200", "stoch_k", "stoch_d")
    _require_columns(df, required)
    out = df.copy()
    k_prev = out["stoch_k"].shift(1)
    d_prev = out["stoch_d"].shift(1)
    cross_up = (k_prev <= d_prev) & (out["stoch_k"] > out["stoch_d"])
    cross_down = (k_prev >= d_prev) & (out["stoch_k"] < out["stoch_d"])
    long_signal = (out["close"] > out["ema_200"]) & cross_up
    short_signal = (out["close"] < out["ema_200"]) & cross_down
    out["signal"] = 0
    out.loc[long_signal, "signal"] = 1
    out.loc[short_signal, "signal"] = -1
    return out


def latest_snapshot(df: pd.DataFrame) -> Optional[dict]:
    """Last-bar indicator snapshot, or None if the frame is empty."""
    if df.empty:
        return None
    row = df.iloc[-1]
    keys = [
        "close",
        "ema_200",
        "stoch_k",
        "stoch_d",
        "atr_14",
        "long_stop",
        "short_stop",
        "signal",
    ]
    return {key: row[key] for key in keys if key in df.columns}
