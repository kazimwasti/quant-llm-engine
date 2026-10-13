"""Download a matching 5-minute/hourly futures dataset for cloud backtests.

The 5-minute source is limited by Yahoo Finance. Hourly data includes a long
warmup so the completed-hour EMA is established before the 5-minute window.
All timestamps are exported as UTC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalize(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if bars is None or bars.empty:
        raise RuntimeError(f"Yahoo Finance returned no bars for {symbol}")
    if isinstance(bars.columns, pd.MultiIndex):
        for level in range(bars.columns.nlevels):
            values = set(map(str, bars.columns.get_level_values(level)))
            if {"Open", "High", "Low", "Close"}.issubset(values):
                bars.columns = bars.columns.get_level_values(level)
                break
    bars.columns = [str(c).strip().lower().replace(" ", "_") for c in bars.columns]
    required = ["open", "high", "low", "close"]
    if not set(required).issubset(bars.columns):
        raise RuntimeError(f"Yahoo response lacks OHLC columns: {list(bars.columns)}")
    result = bars[[c for c in (*required, "volume") if c in bars.columns]].copy().reset_index()
    time_col = next((c for c in ("datetime", "date", "timestamp") if c in result), result.columns[0])
    result = result.rename(columns={time_col: "timestamp"})
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    result = result.dropna(subset=["timestamp", *required]).sort_values("timestamp")
    result = result.drop_duplicates("timestamp", keep="last")
    result["symbol"] = symbol
    return result.reset_index(drop=True)


def download_pair(symbol: str = "MNQ=F", test_days: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get recent 5m bars plus matching hourly history with >=200 warmup bars."""
    if test_days < 1 or test_days > 59:
        raise ValueError("test_days must be between 1 and 59 (Yahoo 5m history limit)")
    five = _normalize(yf.download(
        symbol, period=f"{test_days}d", interval="5m", auto_adjust=False,
        prepost=True, progress=False, threads=False,
    ), symbol)
    first, last = five.timestamp.min(), five.timestamp.max()
    warmup_start = first - pd.Timedelta(days=180)
    hourly = _normalize(yf.download(
        symbol, start=warmup_start.strftime("%Y-%m-%d"),
        end=(last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1h", auto_adjust=False, prepost=True, progress=False, threads=False,
    ), symbol)
    hourly = hourly[hourly.timestamp <= last].reset_index(drop=True)
    preceding = int((hourly.timestamp < first).sum())
    if preceding < 200:
        raise RuntimeError(
            f"Only {preceding} hourly bars precede the 5m window; need 200. "
            "Try a longer hourly lookback or another data vendor."
        )
    return five, hourly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=("MNQ=F", "NQ=F"), default="MNQ=F")
    parser.add_argument("--test-days", type=int, default=50)
    parser.add_argument("--five-minute-output", default="data/5m_cloud.csv")
    parser.add_argument("--hourly-output", default="data/hourly_cloud.csv")
    args = parser.parse_args()
    five, hourly = download_pair(args.symbol, args.test_days)
    for frame, name in ((five, args.five_minute_output), (hourly, args.hourly_output)):
        path = Path(name)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"Saved {len(frame):,} rows to {path}: {frame.timestamp.min()} through {frame.timestamp.max()}")
    print(f"Hourly warmup before 5m window: {(hourly.timestamp < five.timestamp.min()).sum():,} bars")


if __name__ == "__main__":
    main()
