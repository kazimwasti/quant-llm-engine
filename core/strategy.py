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
DEFAULT_ATR_STOP_MULTIPLIER = 1.0
DEFAULT_ATR_TARGET_MULTIPLIER = 2.2
HTF_EMA_PERIOD_5M = 240
ATR_REGIME_WINDOW = 50
ATR_LOWER_PERCENTILE = 0.30
ATR_UPPER_PERCENTILE = 0.80
MARKET_TIMEZONE = "America/New_York"
NO_TRADE_WINDOWS = ((9 * 60 + 30, 9 * 60 + 45), (12 * 60, 14 * 60))


def discrete_expected_checkpoints(
    close: float, drift_per_bar: float, horizon: int = 12
) -> list[float]:
    """Build the expected price checkpoint path for the next ``horizon`` bars."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return [float(close) + float(drift_per_bar) * step for step in range(1, horizon + 1)]


def atr_envelope_widths(
    atr_value: float, horizon: int = 12, multiplier: float = 1.0
) -> list[float]:
    """Return square-root-of-time ATR deviation widths for each checkpoint."""
    if atr_value <= 0 or multiplier <= 0 or horizon < 1:
        raise ValueError("atr_value, multiplier, and horizon must be positive")
    return [float(atr_value) * multiplier * (step**0.5) for step in range(1, horizon + 1)]


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
    atr_target_multiplier: float = DEFAULT_ATR_TARGET_MULTIPLIER,
    atr_regime_window: int = ATR_REGIME_WINDOW,
    market_timezone: str = MARKET_TIMEZONE,
) -> pd.DataFrame:
    """Return a copy of `df` with strategy columns attached.

    Added columns
    -------------
    ema_200, stoch_k, stoch_d, atr_14, long_stop, short_stop
    """
    _require_columns(df, ("high", "low", "close"))
    out = df.copy()
    out["ema_200"] = ema(out["close"], period=ema_period)
    out["ema_drift"] = out["ema_200"].diff(3) / 3.0
    out["htf_ema"] = out["close"].astype(float).ewm(
        span=HTF_EMA_PERIOD_5M,
        adjust=False,
        min_periods=HTF_EMA_PERIOD_5M,
    ).mean()
    out["stoch_k"], out["stoch_d"] = stochastic(
        out["high"],
        out["low"],
        out["close"],
        k_period=k_period,
        d_period=d_period,
    )
    out["atr_14"] = atr(out["high"], out["low"], out["close"], period=atr_period)
    if atr_regime_window < 2:
        raise ValueError("atr_regime_window must be >= 2")
    atr_history = out["atr_14"].rolling(
        window=atr_regime_window, min_periods=atr_regime_window
    )
    out["atr_median"] = atr_history.median().shift(1)
    out["atr_p30"] = atr_history.quantile(ATR_LOWER_PERCENTILE).shift(1)
    out["atr_p80"] = atr_history.quantile(ATR_UPPER_PERCENTILE).shift(1)
    out["volatility_active"] = out["atr_14"].between(
        out["atr_p30"], out["atr_p80"], inclusive="both"
    )
    out["volatility_active"] = out["volatility_active"].fillna(False)

    # Structural confirmation: test a pullback to the EMA / one-ATR price
    # envelope, or a directional rejection wick at least 0.3 ATR long.
    bar_open = out["open"].astype(float) if "open" in out.columns else out["close"].astype(float)
    bar_high = out["high"].astype(float)
    bar_low = out["low"].astype(float)
    bar_close = out["close"].astype(float)
    lower_shadow = pd.concat([bar_open, bar_close], axis=1).min(axis=1) - bar_low
    upper_shadow = bar_high - pd.concat([bar_open, bar_close], axis=1).max(axis=1)
    out["bullish_rejection"] = (bar_close > bar_open) & (lower_shadow >= 0.3 * out["atr_14"])
    out["bearish_rejection"] = (bar_close < bar_open) & (upper_shadow >= 0.3 * out["atr_14"])
    out["long_structure_confirmed"] = (
        (bar_low <= out["ema_200"])
        | (bar_low <= bar_close - out["atr_14"])
        | out["bullish_rejection"]
    )
    out["short_structure_confirmed"] = (
        (bar_high >= out["ema_200"])
        | (bar_high >= bar_close + out["atr_14"])
        | out["bearish_rejection"]
    )

    timestamps = None
    if "timestamp" in out.columns:
        timestamps = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    elif isinstance(out.index, pd.DatetimeIndex):
        timestamps = pd.Series(pd.to_datetime(out.index, utc=True), index=out.index)
    if timestamps is None:
        # No clock data means timing cannot be validated; block safely.
        out["session_allowed"] = False
        out["session_progress"] = 0.0
    else:
        local = timestamps.dt.tz_convert(market_timezone)
        minute_of_day = local.dt.hour * 60 + local.dt.minute
        session_minutes = ((minute_of_day - 18 * 60) % (24 * 60)).clip(upper=23 * 60)
        out["session_progress"] = session_minutes / float(23 * 60)
        allowed = pd.Series(local.notna(), index=out.index)
        for start_minute, end_minute in NO_TRADE_WINDOWS:
            allowed &= ~minute_of_day.between(start_minute, end_minute, inclusive="left")
        out["session_allowed"] = allowed.fillna(False)
    out["trade_allowed"] = out["session_allowed"] & out["volatility_active"]
    out["long_stop"] = dynamic_stop_loss(
        out["close"], out["atr_14"], side="long", multiplier=atr_stop_multiplier
    )
    out["short_stop"] = dynamic_stop_loss(
        out["close"], out["atr_14"], side="short", multiplier=atr_stop_multiplier
    )
    if atr_target_multiplier <= 0:
        raise ValueError("atr_target_multiplier must be > 0")
    out["long_target"] = out["close"] + atr_target_multiplier * out["atr_14"]
    out["short_target"] = out["close"] - atr_target_multiplier * out["atr_14"]
    out["htf_long_allowed"] = out["close"] > out["htf_ema"]
    out["htf_short_allowed"] = out["close"] < out["htf_ema"]
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
    if "htf_ema" not in out.columns:
        out["htf_ema"] = out["close"].astype(float).ewm(
            span=HTF_EMA_PERIOD_5M, adjust=False, min_periods=HTF_EMA_PERIOD_5M
        ).mean()
    k_prev = out["stoch_k"].shift(1)
    d_prev = out["stoch_d"].shift(1)
    cross_up = (k_prev <= d_prev) & (out["stoch_k"] > out["stoch_d"])
    cross_down = (k_prev >= d_prev) & (out["stoch_k"] < out["stoch_d"])
    gates = out.get("trade_allowed", pd.Series(True, index=out.index)).astype(bool)
    long_structure = out.get("long_structure_confirmed", pd.Series(False, index=out.index)).astype(bool)
    short_structure = out.get("short_structure_confirmed", pd.Series(False, index=out.index)).astype(bool)
    long_signal = gates & long_structure & (out["close"] > out["ema_200"]) & (out["close"] > out["htf_ema"]) & cross_up
    short_signal = gates & short_structure & (out["close"] < out["ema_200"]) & (out["close"] < out["htf_ema"]) & cross_down
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
        "htf_ema",
        "long_target",
        "short_target",
        "htf_long_allowed",
        "htf_short_allowed",
        "ema_drift",
        "signal",
        "atr_median",
        "volatility_active",
        "atr_p30",
        "atr_p80",
        "bullish_rejection",
        "bearish_rejection",
        "long_structure_confirmed",
        "short_structure_confirmed",
        "session_allowed",
        "trade_allowed",
        "session_progress",
    ]
    return {key: row[key] for key in keys if key in df.columns}

def evaluate_signal(row, history=None):
    """Return -1/0/1 only when the precomputed regime gates permit trading."""
    del history
    if not bool(row.get("trade_allowed", False)):
        return 0
    return int(row.get("signal", 0) or 0)
