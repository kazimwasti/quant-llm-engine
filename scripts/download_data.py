"""Download Yahoo Finance 5-minute OHLC bars for NQ or MNQ futures.

Yahoo limits intraday history. For interval=5m, the practical maximum is about
60 days, so the default requests the recent month. MNQ is tried first; if Yahoo
has no intraday history for it, NQ is tried as the Nasdaq futures alternative.

Examples:
    python scripts/download_data.py
    python scripts/download_data.py --symbol NQ=F --period 30d
    python scripts/download_data.py --symbol MNQ=F --output data/mnq_5m.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def download_5m(symbol: str = "MNQ=F", period: str = "30d") -> pd.DataFrame:
    """Fetch and normalize 5-minute OHLC data from Yahoo Finance."""
    bars = yf.download(
        tickers=symbol,
        period=period,
        interval="5m",
        auto_adjust=False,
        prepost=True,
        progress=False,
        threads=False,
    )
    if bars is None or bars.empty:
        raise RuntimeError(f"Yahoo Finance returned no 5-minute bars for {symbol!r} ({period}).")

    if isinstance(bars.columns, pd.MultiIndex):
        # yfinance may return (field, ticker) or (ticker, field), depending on version.
        for level in range(bars.columns.nlevels):
            values = set(map(str, bars.columns.get_level_values(level)))
            if {"Open", "High", "Low", "Close"}.issubset(values):
                bars.columns = bars.columns.get_level_values(level)
                break

    bars.columns = [str(column).strip().lower().replace(" ", "_") for column in bars.columns]
    required = ["open", "high", "low", "close"]
    missing = [name for name in required if name not in bars.columns]
    if missing:
        raise RuntimeError(f"Yahoo response is missing columns {missing}: {list(bars.columns)}")

    result = bars.loc[:, [name for name in ("open", "high", "low", "close", "volume") if name in bars.columns]].copy()
    result = result.reset_index()
    time_column = next((name for name in ("datetime", "date", "timestamp") if name in result.columns), None)
    if time_column is None:
        time_column = result.columns[0]
    result = result.rename(columns={time_column: "timestamp"})
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    result = result.dropna(subset=["timestamp", *required]).sort_values("timestamp")
    result = result.drop_duplicates(subset="timestamp", keep="last")
    if result.empty:
        raise RuntimeError("Yahoo response contained no valid OHLC rows")
    result["symbol"] = symbol
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=("MNQ=F", "NQ=F"), default="MNQ=F",
                        help="Yahoo continuous futures ticker (default: MNQ=F)")
    parser.add_argument("--period", default="30d", help="Yahoo lookback period (5m history is limited; default: 30d)")
    parser.add_argument("--output", default="data/5m.csv", help="Output CSV path (default: data/5m.csv)")
    args = parser.parse_args()

    try:
        data = download_5m(args.symbol, args.period)
    except RuntimeError:
        alternate = "NQ=F" if args.symbol == "MNQ=F" else "MNQ=F"
        print(f"No bars for {args.symbol}; trying futures alternative {alternate}.")
        data = download_5m(alternate, args.period)
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    print(
        f"Saved {len(data):,} 5-minute {data['symbol'].iloc[0]} bars to {output} "
        f"({data['timestamp'].iloc[0]} through {data['timestamp'].iloc[-1]})"
    )


if __name__ == "__main__":
    main()
