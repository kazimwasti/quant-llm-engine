"""Label 1-hour OHLC history into Jev-style BUY/SELL/HOLD JSONL pairs.

For each bar after indicator warmup, a 1.5x ATR envelope is placed around close:

    upper = close + 1.5 * ATR   (long target / short stop)
    lower = close - 1.5 * ATR   (long stop   / short target)

Forward bars are scanned (starting at the next hour so the signal close is known).
The first barrier touched is the ideal action; same-bar both-sides or no touch
inside the horizon is HOLD.

Each JSONL record reuses ``engine.format_decision_prompt`` so fine-tuning pairs
match the live logit readout prefix.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

from engine import ACTIONS, MarketState, format_decision_prompt
from strategy import DEFAULT_ATR_STOP_MULTIPLIER, add_indicators, generate_signals

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

DEFAULT_HORIZON_BARS = 24  # 24 hours of 1-hour bars


def _pick_column(columns: list[str], aliases: tuple[str, ...]) -> Optional[str]:
    lowered = {name.lower(): name for name in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def load_hourly_ohlc(csv_path: str | Path) -> pd.DataFrame:
    """Load a 1-hour OHLC CSV and normalize column names to open/high/low/close."""
    path = Path(csv_path)
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
    )


def _first_barrier(
    highs: pd.Series,
    lows: pd.Series,
    start: int,
    horizon: int,
    upper: float,
    lower: float,
) -> tuple[str, dict[str, Any]]:
    """Return BUY/SELL/HOLD from the first 1.5x ATR barrier touch after `start`."""
    end = min(start + horizon, len(highs))
    for offset, idx in enumerate(range(start, end), start=1):
        hit_upper = float(highs.iloc[idx]) >= upper
        hit_lower = float(lows.iloc[idx]) <= lower
        if hit_upper and hit_lower:
            return "HOLD", {
                "reason": "same_bar_both_barriers",
                "bars_to_outcome": offset,
            }
        if hit_upper:
            return "BUY", {
                "reason": "long_target_before_stop",
                "bars_to_outcome": offset,
            }
        if hit_lower:
            return "SELL", {
                "reason": "short_target_before_stop",
                "bars_to_outcome": offset,
            }
    return "HOLD", {
        "reason": "timeout",
        "bars_to_outcome": None,
    }


def iter_labeled_examples(
    df: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON_BARS,
    atr_multiplier: float = DEFAULT_ATR_STOP_MULTIPLIER,
) -> Iterator[dict[str, Any]]:
    """Yield Jev training records from an indicator-enriched OHLC frame."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be > 0")

    required = ("high", "low", "close", "ema_200", "stoch_k", "stoch_d", "atr_14")
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

        distance = atr_multiplier * atr
        upper = close + distance
        lower = close - distance
        action, outcome = _first_barrier(
            highs, lows, start=i + 1, horizon=horizon, upper=upper, lower=lower
        )

        state = _market_state_from_row(row)
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
                "atr_multiplier": atr_multiplier,
                "upper": upper,
                "lower": lower,
                "horizon": horizon,
                **outcome,
            },
        }


def generate_jsonl(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    horizon: int = DEFAULT_HORIZON_BARS,
    atr_multiplier: float = DEFAULT_ATR_STOP_MULTIPLIER,
) -> dict[str, int]:
    """Read 1-hour OHLC CSV, label with 1.5x ATR barriers, write JSONL."""
    raw = load_hourly_ohlc(csv_path)
    featured = generate_signals(add_indicators(raw, atr_stop_multiplier=atr_multiplier))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in iter_labeled_examples(
            featured, horizon=horizon, atr_multiplier=atr_multiplier
        ):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[record["action"]] += 1
            written += 1

    summary = {"total": written, **{action: counts[action] for action in ACTIONS}}
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Jev-style JSONL fine-tune pairs from 1-hour OHLC history "
            "using 1.5x ATR stop/target first-touch labels."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to 1-hour OHLC CSV")
    parser.add_argument(
        "--output",
        "-o",
        default="jev_train.jsonl",
        help="Output JSONL path (default: jev_train.jsonl)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON_BARS,
        help="Forward bars to wait for a 1.5x ATR touch (default: 24)",
    )
    parser.add_argument(
        "--atr-multiplier",
        type=float,
        default=DEFAULT_ATR_STOP_MULTIPLIER,
        help="Stop/target distance in ATR units (default: 1.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = generate_jsonl(
        args.input,
        args.output,
        horizon=args.horizon,
        atr_multiplier=args.atr_multiplier,
    )
    print(
        f"Wrote {summary['total']} pairs to {args.output} "
        f"(BUY={summary['BUY']}, SELL={summary['SELL']}, HOLD={summary['HOLD']})"
    )


if __name__ == "__main__":
    main()
