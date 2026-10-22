"""Label intraday OHLC history into Jev-style BUY/SELL/HOLD JSONL pairs.

For each bar after indicator warmup, candidate trades use a 1.0x ATR stop and
2.2x ATR target. Forward bars are scanned over a 24-bar horizon by default,
starting at the next bar so the signal close is known. The best positive
directional outcome becomes the label; otherwise the label is HOLD. Session,
volatility, and higher-timeframe trend gates are carried into the labels.

Every record's ``meta`` carries ``candidate`` (the rule side that fired, if
any) and ``realized_r`` (the candidate's outcome in stop-distance multiples:
+2.2 for a target, -1.0 for a stop, mark-to-market at expiry for a timeout).
``scripts/train_lora.py`` weights its loss by ``realized_r`` so trades that
paid more teach more, while background no-setup HOLD rows are down-weighted.

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
    align_higher_timeframe_ema,
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
DEFAULT_MAX_DIRECTIONAL_FRACTION = 0.40


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

    missing = [name for name in ("open", "high", "low", "close") if name not in frame.columns]
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

    frame = frame.dropna(subset=["open", "high", "low", "close"])
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
        bullish_rejection=bool(row.get("bullish_rejection", False)),
        bearish_rejection=bool(row.get("bearish_rejection", False)),
    )


def _first_barrier(
    opens: pd.Series,
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    start: int,
    horizon: int,
    atr_value: float,
    stop_atr_multiplier: float,
    target_atr_multiplier: float,
) -> dict[str, dict[str, Any]]:
    """Return first barrier outcomes, resolving same-bar touches stop-first.

    ``realized_r`` expresses each side's outcome in stop-distance multiples:
    ``+target/stop`` for a target hit, ``-1.0`` for a stop, and the signed
    mark-to-market move at expiry for a timeout.
    """
    end = min(start + horizon, len(highs))
    if start >= end:
        return {}
    # Backtests signal on a completed bar and enter at the next bar's open.
    entry = float(opens.iloc[start])
    stop_distance = stop_atr_multiplier * atr_value
    target_distance = target_atr_multiplier * atr_value
    outcomes: dict[str, dict[str, Any]] = {}
    for side in ("BUY", "SELL"):
        sign = 1.0 if side == "BUY" else -1.0
        stop = entry - sign * stop_distance
        target = entry + sign * target_distance
        reason = "timeout"
        bars_to_outcome = None
        exit_price = float(closes.iloc[end - 1])
        for offset, idx in enumerate(range(start, end), start=1):
            hit_stop = float(lows.iloc[idx]) <= stop if side == "BUY" else float(highs.iloc[idx]) >= stop
            hit_target = float(highs.iloc[idx]) >= target if side == "BUY" else float(lows.iloc[idx]) <= target
            if hit_stop and hit_target:
                reason = "same_bar_stop_first"
                bars_to_outcome = offset
                exit_price = stop
                break
            if hit_stop:
                reason = "stop"
                bars_to_outcome = offset
                exit_price = stop
                break
            if hit_target:
                reason = "target"
                bars_to_outcome = offset
                exit_price = target
                break
        outcomes[side] = {
            "reason": reason,
            "bars_to_outcome": bars_to_outcome,
            "realized_r": sign * (exit_price - entry) / stop_distance,
        }
    return outcomes


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

    required = ("open", "high", "low", "close", "ema_200", "htf_ema", "stoch_k", "stoch_d", "atr_14")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"frame missing columns {missing}; run add_indicators() first")

    opens = df["open"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    closes = df["close"].astype(float)
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

        outcomes = _first_barrier(
            opens, highs, lows, closes, i + 1, horizon,
            atr, stop_atr_multiplier, target_atr_multiplier,
        )

        state = _market_state_from_row(row)
        action = "HOLD"
        outcome: dict[str, Any] = {"reason": "no_qualifying_directional_target", "bars_to_outcome": None}
        candidate = "BUY" if state.signal == 1 else "SELL" if state.signal == -1 else None
        realized_r: Optional[float] = None
        if not state.session_allowed or not state.volatility_active:
            outcome["reason"] = "regime_filter"
        elif candidate is None:
            outcome["reason"] = "no_directional_rule_signal"
        elif candidate == "BUY" and not state.long_structure_confirmed:
            outcome["reason"] = "long_structure_not_confirmed"
        elif candidate == "SELL" and not state.short_structure_confirmed:
            outcome["reason"] = "short_structure_not_confirmed"
        elif candidate == "BUY" and close <= state.htf_ema:
            outcome["reason"] = "htf_trend_filter"
        elif candidate == "SELL" and close >= state.htf_ema:
            outcome["reason"] = "htf_trend_filter"
        elif outcomes.get(candidate, {}).get("reason") == "target":
            action = candidate
            realized_r = outcomes[candidate]["realized_r"]
            outcome = {**outcomes[candidate], "reason": f"{candidate.lower()}_target_before_stop"}
        elif candidate in outcomes:
            realized_r = outcomes[candidate]["realized_r"]
            outcome = {**outcomes[candidate], "reason": f"{candidate.lower()}_{outcomes[candidate]['reason']}"}
        if not state.is_complete():
            continue

        entry = float(opens.iloc[i + 1])
        prompt = format_decision_prompt(state)
        ts = row["timestamp"] if "timestamp" in row.index else df.index[i]
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
                "entry": entry,
                "atr_14": atr,
                "atr_multiplier": stop_atr_multiplier,
                "stop_atr_multiplier": stop_atr_multiplier,
                "target_atr_multiplier": target_atr_multiplier,
                "long_stop": entry - stop_atr_multiplier * atr,
                "long_target": entry + target_atr_multiplier * atr,
                "short_stop": entry + stop_atr_multiplier * atr,
                "short_target": entry - target_atr_multiplier * atr,
                "horizon": horizon,
                "bar_minutes": bar_minutes,
                "horizon_minutes": horizon * bar_minutes,
                "candidate": candidate,
                "realized_r": realized_r,
                **outcome,
            },
        }


def generate_jsonl(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    hourly_input: str | Path,
    horizon: int = DEFAULT_HORIZON_BARS,
    atr_multiplier: Optional[float] = None,
    stop_atr_multiplier: float = DEFAULT_ATR_STOP_MULTIPLIER,
    target_atr_multiplier: float = DEFAULT_ATR_TARGET_MULTIPLIER,
    bar_minutes: int = DEFAULT_BAR_MINUTES,
    max_directional_fraction: float = DEFAULT_MAX_DIRECTIONAL_FRACTION,
) -> dict[str, int]:
    """Read OHLC CSV, label at the selected bar interval, and write JSONL."""
    records = _records_for_inputs(
        csv_path,
        hourly_input,
        horizon=horizon,
        atr_multiplier=atr_multiplier,
        stop_atr_multiplier=stop_atr_multiplier,
        target_atr_multiplier=target_atr_multiplier,
        bar_minutes=bar_minutes,
        max_directional_fraction=max_directional_fraction,
    )
    return _write_records(records, output_path)


def _records_for_inputs(
    csv_path: str | Path,
    hourly_input: str | Path,
    *,
    horizon: int,
    atr_multiplier: Optional[float],
    stop_atr_multiplier: float,
    target_atr_multiplier: float,
    bar_minutes: int,
    max_directional_fraction: float,
    apply_cap: bool = True,
    expected_symbol: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Generate one symbol's records without writing an intermediate file."""
    if bar_minutes < 1:
        raise ValueError("bar_minutes must be >= 1")
    if not 0 < max_directional_fraction < 1:
        raise ValueError("max_directional_fraction must be between 0 and 1")
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
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("Timestamped 5-minute data is required for HTF-constrained labels")
    hourly = load_hourly_ohlc(hourly_input)
    if not isinstance(hourly.index, pd.DatetimeIndex):
        raise ValueError("Timestamped hourly data is required for HTF-constrained labels")
    if "symbol" not in raw.columns or "symbol" not in hourly.columns:
        raise ValueError("Both datasets must contain a symbol column so instrument matching can be verified")
    five_symbols = set(raw["symbol"].dropna().astype(str).str.upper())
    hourly_symbols = set(hourly["symbol"].dropna().astype(str).str.upper())
    if len(five_symbols) != 1 or len(hourly_symbols) != 1 or five_symbols != hourly_symbols:
        raise ValueError("5-minute and hourly inputs must use the same instrument")
    if expected_symbol and five_symbols != {expected_symbol.upper()}:
        raise ValueError(
            f"CSV symbol {sorted(five_symbols)} does not match filename stem {expected_symbol.upper()}"
        )
    first_intraday = raw.index.min()
    warmup_bars = int((hourly.index < first_intraday).sum())
    if warmup_bars < 200:
        raise ValueError(f"Hourly input has only {warmup_bars} completed warmup bars before the 5m window; need at least 200")
    hourly_intervals = hourly.index.to_series().diff().dropna().dt.total_seconds().div(60)
    if not hourly_intervals.empty and abs(float(hourly_intervals.median()) - 60.0) > 0.6:
        raise ValueError(f"Hourly input median bar interval is {hourly_intervals.median():g} minutes; expected 60")

    intraday = raw.reset_index().rename(columns={raw.index.name or "index": "timestamp"})
    hourly_bars = hourly.reset_index().rename(columns={hourly.index.name or "index": "timestamp"})
    htf_ema = align_higher_timeframe_ema(intraday, hourly_bars)
    if not htf_ema.notna().any():
        raise ValueError("No completed hourly EMA aligns with the 5-minute input")
    featured = generate_signals(add_indicators(
        intraday,
        atr_stop_multiplier=stop_atr_multiplier,
        atr_target_multiplier=target_atr_multiplier,
        higher_timeframe_ema=htf_ema,
    ))
    records = list(iter_labeled_examples(
        featured,
        horizon=horizon,
        stop_atr_multiplier=stop_atr_multiplier,
        target_atr_multiplier=target_atr_multiplier,
        bar_minutes=bar_minutes,
    ))
    return _cap_directional_fraction(records, max_directional_fraction) if apply_cap else records


def _write_records(records: list[dict[str, Any]], output_path: str | Path) -> dict[str, int]:
    output = _project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[record["action"]] += 1

    summary = {"total": len(records), **{action: counts[action] for action in ACTIONS}}
    return summary


def generate_universe_jsonl(
    intraday_dir: str | Path,
    hourly_dir: str | Path,
    output_path: str | Path,
    **options,
) -> dict[str, int]:
    """Build one consolidated multi-symbol dataset from paired CSV folders.

    CSVs are paired by filename stem (for example ``AAPL.csv`` in each
    directory). Every record receives ``meta.symbol`` and the directional
    prevalence cap is applied once across the combined universe.
    """
    intraday_root = _project_path(intraday_dir)
    hourly_root = _project_path(hourly_dir)
    if not intraday_root.is_dir() or not hourly_root.is_dir():
        raise NotADirectoryError("multi-symbol inputs must both be directories")
    intraday_files = {path.stem.upper(): path for path in intraday_root.glob("*.csv")}
    hourly_files = {path.stem.upper(): path for path in hourly_root.glob("*.csv")}
    if not intraday_files:
        raise ValueError(f"no CSV files found in {intraday_root}")
    missing_hourly = sorted(set(intraday_files) - set(hourly_files))
    missing_intraday = sorted(set(hourly_files) - set(intraday_files))
    if missing_hourly or missing_intraday:
        raise ValueError(
            f"input folders must have matching CSV stems; missing hourly={missing_hourly}, "
            f"missing intraday={missing_intraday}"
        )

    records: list[dict[str, Any]] = []
    for symbol in sorted(intraday_files):
        symbol_records = _records_for_inputs(
            intraday_files[symbol], hourly_files[symbol],
            horizon=options.get("horizon", DEFAULT_HORIZON_BARS),
            atr_multiplier=options.get("atr_multiplier"),
            stop_atr_multiplier=options.get("stop_atr_multiplier", DEFAULT_ATR_STOP_MULTIPLIER),
            target_atr_multiplier=options.get("target_atr_multiplier", DEFAULT_ATR_TARGET_MULTIPLIER),
            bar_minutes=options.get("bar_minutes", DEFAULT_BAR_MINUTES),
            max_directional_fraction=options.get("max_directional_fraction", DEFAULT_MAX_DIRECTIONAL_FRACTION),
            apply_cap=False,
            expected_symbol=symbol,
        )
        for record in symbol_records:
            record["meta"]["symbol"] = symbol
        records.extend(symbol_records)
    records = _cap_directional_fraction(
        records, options.get("max_directional_fraction", DEFAULT_MAX_DIRECTIONAL_FRACTION)
    )
    records.sort(key=lambda row: (row["meta"].get("timestamp", ""), row["meta"].get("symbol", "")))
    summary = _write_records(records, output_path)
    summary["symbols"] = len(intraday_files)
    return summary


def _evenly_spaced(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Deterministically retain examples across the full chronological span."""
    if count >= len(items):
        return items
    if count <= 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    return [items[i] for i in indices]


def _cap_directional_fraction(
    records: list[dict[str, Any]], max_fraction: float
) -> list[dict[str, Any]]:
    """Keep all HOLD rows and cap BUY+SELL prevalence, preserving chronology."""
    holds = [row for row in records if row["action"] == "HOLD"]
    buys = [row for row in records if row["action"] == "BUY"]
    sells = [row for row in records if row["action"] == "SELL"]
    directional = len(buys) + len(sells)
    if directional == 0:
        return records
    if not holds:
        raise ValueError("Cannot retain a HOLD-majority training set: no HOLD examples were generated")
    maximum = max(1, int(len(holds) * max_fraction / (1.0 - max_fraction)))
    if directional <= maximum:
        return records
    buy_keep = min(len(buys), round(maximum * len(buys) / directional))
    sell_keep = min(len(sells), maximum - buy_keep)
    remaining = maximum - buy_keep - sell_keep
    if remaining and buy_keep < len(buys):
        extra = min(remaining, len(buys) - buy_keep)
        buy_keep += extra
        remaining -= extra
    if remaining and sell_keep < len(sells):
        sell_keep += min(remaining, len(sells) - sell_keep)
    selected = holds + _evenly_spaced(buys, buy_keep) + _evenly_spaced(sells, sell_keep)
    return sorted(selected, key=lambda row: row["meta"]["bar_index"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Jev-style JSONL fine-tune pairs from intraday OHLC history "
            "using asymmetric 1.0x ATR stop / 2.2x ATR target barriers."
        )
    )
    parser.add_argument("--input", "-i", default="data/5m.csv", help="Path to OHLC CSV (default: data/5m.csv)")
    parser.add_argument("--hourly-input",
                        help="Matching timestamped hourly OHLC CSV with at least 200 warmup bars")
    parser.add_argument("--input-dir", help="Directory of per-symbol intraday CSVs; pair by filename with --hourly-dir")
    parser.add_argument("--hourly-dir", help="Directory of matching per-symbol hourly CSVs")
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
    parser.add_argument("--max-directional-fraction", type=float,
                        default=DEFAULT_MAX_DIRECTIONAL_FRACTION,
                        help="Cap combined BUY/SELL rows at this fraction; all HOLD rows are retained (default: 0.40)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if bool(args.input_dir) != bool(args.hourly_dir):
        raise SystemExit("--input-dir and --hourly-dir must be provided together")
    options = dict(
        horizon=args.horizon,
        stop_atr_multiplier=args.stop_atr_multiplier,
        target_atr_multiplier=args.target_atr_multiplier,
        bar_minutes=args.bar_minutes,
        max_directional_fraction=args.max_directional_fraction,
    )
    if args.input_dir:
        summary = generate_universe_jsonl(args.input_dir, args.hourly_dir, args.output, **options)
    else:
        if not args.hourly_input:
            raise SystemExit("--hourly-input is required when --input-dir is not used")
        summary = generate_jsonl(args.input, args.output, hourly_input=args.hourly_input, **options)
    print(
        f"Wrote {summary['total']} pairs from {summary.get('symbols', 1)} symbol(s) to {args.output} "
        f"(BUY={summary['BUY']}, SELL={summary['SELL']}, HOLD={summary['HOLD']})"
    )


if __name__ == "__main__":
    main()
