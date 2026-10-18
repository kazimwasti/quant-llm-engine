from pathlib import Path
import sys
import pandas as pd

ROOT = Path("/kaggle/input")
REQUIRED = {"open", "high", "low", "close", "volume"}
ALIASES = {
    "datetime": "timestamp", "date": "timestamp", "time": "timestamp",
    "ticker": "symbol", "instrument": "symbol",
}

def audit(path: Path):
    try:
        df = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    except Exception as exc:
        print(f"SKIP {path}: could not read ({exc})")
        return
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns and v not in df.columns})
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        print(f"SKIP {path}: missing OHLCV columns {missing}; found={list(df.columns)}")
        return
    time_col = "timestamp" if "timestamp" in df.columns else None
    if time_col is None:
        print(f"CHECK {path}: OHLCV present but no timestamp column.")
        return
    for col in REQUIRED:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    bad = ts.isna()
    invalid = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"] > df[["open", "close", "high"]].min(axis=1)) |
        (df["volume"] < 0)
    )
    work = df.assign(_ts=ts)
    duplicate = int(work.duplicated(["_ts"] + (["symbol"] if "symbol" in work else [] )).sum())
    print(f"FILE {path}")
    print(f"  rows={len(df):,} symbols={work['symbol'].nunique() if 'symbol' in work else 'unknown'} "
          f"utc_range={ts.min()} .. {ts.max()}")
    print(f"  bad_timestamps={int(bad.sum())} invalid_ohlcv={int(invalid.sum())} duplicate_time_keys={duplicate}")
    if "symbol" in work:
        for symbol, g in work.groupby("symbol", dropna=False):
            g = g.sort_values("_ts")
            gaps = g["_ts"].diff().dropna()
            print(f"  {symbol}: rows={len(g):,} range={g['_ts'].min()} .. {g['_ts'].max()} "
                  f"median_gap={gaps.median() if len(gaps) else 'n/a'}")

files = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".parquet", ".pq"}]
print(f"Kaggle input root exists={ROOT.exists()} | candidate files={len(files)}")
if not files:
    print("No CSV/Parquet market files found. Attach a Kaggle Dataset containing the vendor files.")
for file in files:
    audit(file)
