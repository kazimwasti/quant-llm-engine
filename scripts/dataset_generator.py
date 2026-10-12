"""Label intraday OHLC history into Jev-style BUY/SELL/HOLD JSONL pairs.

For each bar after indicator warmup, candidate trades use a 1.0x ATR stop and
2.2x ATR target. Forward bars are scanned over a 24-bar horizon by default,
starting at the next bar so the signal close is known. The best positive
directional outcome becomes the label; otherwise the label is HOLD. Session,
volatility, and higher-timeframe trend gates are carried into the labels.

Each JSONL record reuses ``engine.format_decision_prompt`` so fine-tuning pairs
match the live logit readout prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.engine import ACTIONS, MarketState, format_decision_prompt
from core.strategy import (
    DEFAULT_ATR_STOP_MULTIPLIER,
    DEFAULT_ATR_TARGET_MULTIPLIER,
    add_indicators,
    generate_signals,
)

OHLC_ALIASES = {
    "open": ("open", "o", "Open", "OPEN"),
    "high": ("high", "h", "High", "HIGH"),
    "low": ("low", "l", "Low", "LOW"),
    "close": ("close", "c", "Close", "CLOSE", "price"),
    "timestamp": (
        "timestamp",
        "datetime",
        "date",
        "time",
        "Date",
        "Datetime",
        "Timestamp",
    ),
}

DEFAULT_BAR_MINUTES = 5
DEFAULT_HORIZON_BARS = 24  # two hours of 5-minute bars


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _pick_column(columns: list[str], aliases: tuple[str, ...]) -> Optional[str]:
    lowered = {name.lower(): name for name in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def load_ohlc(csv_path: str | Path) -> pd.DataFrame:
    """Load OHLC CSV and normalize column names to open/high/low/close."""
    path = _project_path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV is empty: {path}")

    rename: dict[str, str] = {}
    for canonical, aliases in OHLC_ALIASES.items():
        matched = _pick_column(list(frame.columns), aliases)
        if matched is not None and matched != canonical:
            rename[matched] = canonical
    frame = frame.rename(columns=rename)

    missing = [name for name in ("high", "low", "close") if name not in frame.columns]
    if missing:
        raise ValueError(
            f"CSV missing required OHLC columns {missing}. Found: {list(frame.columns)}"
        )

    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
        frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
        frame = frame.set_index("timestamp")
    else:
        frame.index = pd.RangeIndex(len(frame), name="timestamp")

    for col in ("open", "high", "low", "close"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["high", "low", "close"])
    if frame.empty:
        raise ValueError("no valid OHLC rows after parsing")
    return frame


def load_hourly_ohlc(csv_path: str | Path) -> pd.DataFrame:
    """Backward-compatible alias for :func:`load_ohlc`."""
    return load_ohlc(csv_path)


def _market_state_from_row(row: pd.Series) -> MarketState:
    signal = row["signal"] if "signal" in row.index else 0
    return MarketState(
        close=float(row["close"]),
        ema_200=float(row["ema_200"]),
        stoch_k=float(row["stoch_k"]),
        stoch_d=float(row["stoch_d"]),
        atr_14=float(row["atr_14"]),
        long_stop=float(row["long_stop"]),
        short_stop=float(row["short_stop"]),
        signal=int(signal or 0),
        session_allowed=bool(row.get("session_allowed", False)),
        volatility_active=bool(row.get("volatility_active", False)),
        expected_drift=float(row.get("ema_drift", 0.0) or 0.0),
        session_progress=float(row.get("session_progress", 0.0) or 0.0),
        htf_ema=float(row.get("htf_ema", float("nan"))),
        long_structure_confirmed=bool(row.get("long_structure_confirmed", False)),
        short_structure_confirmed=bool(row.get("short_structure_confirmed", False)),
    )


def _first_barrier(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    start: int,
    horizon: int,
    entry: float,
    atr_value: float,
    stop_atr_multiplier: float,
    target_atr_multiplier: float,
) -> tuple[str, dict[str, Any]]:
    """Simulate both asymmetric candidate trades and label the better outcome."""
    end = min(start + horizon, len(highs))
    stop_distance = stop_atr_multiplier * atr_value
    target_distance = target_atr_multiplier * atr_value
    outcomes: dict[str, tuple[float, str, Optional[int]]] = {}
    for side in ("BUY", "SELL"):
        sign = 1.0 if side == "BUY" else -1.0
        stop = entry - sign * stop_distance
        target = entry + sign * target_distance
        result = float(closes.iloc[end - 1]) - entry if end > start else 0.0
        reason = "timeout"
        bars_to_outcome = None
        for offset, idx in enumerate(range(start, end), start=1):
            hit_stop = float(lows.iloc[idx]) <= stop if side == "BUY" else float(highs.iloc[idx]) >= stop
            hit_target = float(highs.iloc[idx]) >= target if side == "BUY" else float(lows.iloc[idx]) <= target
            if hit_stop and hit_target:
                result = -sign * stop_distance
                reason = "same_bar_stop_first"
                bars_to_outcome = offset
                break
            if hit_stop:
                result = -sign * stop_distance
                reason = "stop"
                bars_to_outcome = offset
                break
            if hit_target:
                result = sign * target_distance
                reason = "target"
                bars_to_outcome = offset
                break
        outcomes[side] = (result * sign, reason, bars_to_outcome)

    buy_r, buy_reason, buy_bars = outcomes["BUY"]
    sell_r, sell_reason, sell_bars = outcomes["SELL"]
    if buy_r > 0 and buy_r > sell_r:
        return "BUY", {"reason": f"buy_{buy_reason}", "bars_to_outcome": buy_bars,
                       "buy_return_points": buy_r, "sell_return_points": sell_r}
    if sell_r > 0 and sell_r > buy_r:
        return "SELL", {"reason": f"sell_{sell_reason}", "bars_to_outcome": sell_bars,
                        "buy_return_points": buy_r, "sell_return_points": sell_r}
    return "HOLD", {"reason": "no_positive_unambiguous_edge", "bars_to_outcome": None,
                    "buy_return_points": buy_r, "sell_return_points": sell_r}


def iter_labeled_examples(
    df: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON_BARS,
    atr_multiplier: Optional[float] = None,
    stop_atr_multiplier: float = DEFAULT_ATR_STOP_MULTIPLIER,
    target_atr_multiplier: float = DEFAULT_ATR_TARGET_MULTIPLIER,
    bar_minutes: int = DEFAULT_BAR_MINUTES,
) -> Iterator[dict[str, Any]]:
    """Yield Jev training records from an indicator-enriched OHLC frame."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if atr_multiplier is not None:
        stop_atr_multiplier = atr_multiplier
    if stop_atr_multiplier <= 0 or target_atr_multiplier <= 0:
        raise ValueError("stop and target ATR multipliers must be > 0")

    required = ("high", "low", "close", "ema_200", "htf_ema", "stoch_k", "stoch_d", "atr_14")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"frame missing columns {missing}; run add_indicators() first")

    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    n = len(df)

    for i in range(n - horizon):
        row = df.iloc[i]
        atr = float(row["atr_14"])
        close = float(row["close"])
        ema_200 = float(row["ema_200"])
        if not all(map(pd.notna, (atr, close, ema_200, row["stoch_k"], row["stoch_d"]))):
            continue
        if atr <= 0:
            continue

        action, outcome = _first_barrier(
            highs, lows, df["close"].astype(float), i + 1, horizon,
            close, atr, stop_atr_multiplier, target_atr_multiplier,
        )

        state = _market_state_from_row(row)
        if not state.session_allowed or not state.volatility_active:
            action = "HOLD"
            outcome = {"reason": "regime_filter", "bars_to_outcome": 0}
        elif (
            (action == "BUY" and close <= state.htf_ema)
            or (action == "SELL" and close >= state.htf_ema)
        ):
            action = "HOLD"
            outcome = {**outcome, "reason": "htf_trend_filter"}
        if not state.is_complete():
            continue

        prompt = format_decision_prompt(state)
        ts = df.index[i]
        timestamp = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        yield {
            "prompt": prompt,
            "completion": f" {action}",
            "action": action,
            "options": list(ACTIONS),
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": action},
            ],
            "meta": {
                "timestamp": timestamp,
                "bar_index": int(i),
                "entry": close,
                "atr_14": atr,
                "atr_multiplier": stop_atr_multiplier,
                "stop_atr_multiplier": stop_atr_multiplier,
                "target_atr_multiplier": target_atr_multiplier,
                "long_stop": close - stop_atr_multiplier * atr,
                "long_target": close + target_atr_multiplier * atr,
                "short_stop": close + stop_atr_multiplier * atr,
                "short_target": close - target_atr_multiplier * atr,
                "horizon": horizon,
                "bar_minutes": bar_minutes,
                "horizon_minutes": horizon * bar_minutes,
                **outcome,
            },
        }


def generate_jsonl(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    horizon: int = DEFAULT_HORIZON_BARS,
    atr_multiplier: Optional[float] = None,
    stop_atr_multiplier: float = DEFAULT_ATR_STOP_MULTIPLIER,
    target_atr_multiplier: float = DEFAULT_ATR_TARGET_MULTIPLIER,
    bar_minutes: int = DEFAULT_BAR_MINUTES,
) -> dict[str, int]:
    """Read OHLC CSV, label at the selected bar interval, and write JSONL."""
    if bar_minutes < 1:
        raise ValueError("bar_minutes must be >= 1")
    if atr_multiplier is not None:
        stop_atr_multiplier = atr_multiplier
    raw = load_ohlc(csv_path)
    if isinstance(raw.index, pd.DatetimeIndex) and len(raw) > 1:
        intervals = raw.index.to_series().diff().dropna().dt.total_seconds().div(60)
        inferred_minutes = float(intervals.median())
        if abs(inferred_minutes - bar_minutes) > max(0.01, bar_minutes * 0.01):
            raise ValueError(
                f"CSV median bar interval is {inferred_minutes:g} minutes, but "
                f"--bar-minutes is {bar_minutes}; pass the actual interval explicitly"
            )
    featured = generate_signals(add_indicators(
        raw,
        atr_stop_multiplier=stop_atr_multiplier,
        atr_target_multiplier=target_atr_multiplier,
    ))
    output = _project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in iter_labeled_examples(
            featured,
            horizon=horizon,
            stop_atr_multiplier=stop_atr_multiplier,
            target_atr_multiplier=target_atr_multiplier,
            bar_minutes=bar_minutes,
        ):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[record["action"]] += 1
            written += 1

    summary = {"total": written, **{action: counts[action] for action in ACTIONS}}
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Jev-style JSONL fine-tune pairs from intraday OHLC history "
            "using asymmetric 1.0x ATR stop / 2.2x ATR target barriers."
        )
    )
    parser.add_argument("--input", "-i", default="data/5m.csv", help="Path to OHLC CSV (default: data/5m.csv)")
    parser.add_argument(
        "--bar-minutes", type=int, default=DEFAULT_BAR_MINUTES,
        help="Expected CSV interval in minutes (default: 5; rejects mismatched timestamped data)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/jev_train_5m.jsonl",
        help="Output JSONL path (default: data/jev_train_5m.jsonl)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON_BARS,
        help="Forward bars for 1.0x ATR stop / 2.2x ATR target barriers (default: 24; 2 hours at 5m)",
    )
    parser.add_argument(
        "--stop-atr-multiplier", "--atr-multiplier",
        dest="stop_atr_multiplier",
        type=float,
        default=DEFAULT_ATR_STOP_MULTIPLIER,
        help="Stop distance in ATR units (default: 1.0; --atr-multiplier retained as alias)",
    )
    parser.add_argument("--target-atr-multiplier", type=float,
                        default=DEFAULT_ATR_TARGET_MULTIPLIER,
                        help="Profit target distance in ATR units (default: 2.2)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = generate_jsonl(
        args.input,
        args.output,
        horizon=args.horizon,
        stop_atr_multiplier=args.stop_atr_multiplier,
        target_atr_multiplier=args.target_atr_multiplier,
        bar_minutes=args.bar_minutes,
    )
    print(
        f"Wrote {summary['total']} pairs to {args.output} "
        f"(BUY={summary['BUY']}, SELL={summary['SELL']}, HOLD={summary['HOLD']})"
    )


if __name__ == "__main__":
    main()
