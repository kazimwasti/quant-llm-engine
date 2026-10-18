"""Run the Tier 1/2 screen over a demo universe and print the funnel.

Examples
--------
    python scripts/run_screen.py                        # data/*.csv + synthetic universe
    python scripts/run_screen.py --synthetic-only --synthetic 24
    python scripts/run_screen.py --top-k 3 --min-dollar-volume 5000000

Loads any OHLCV CSVs in --data-dir (e.g. data/5m.csv) as symbols, adds a
seeded synthetic universe with correlated/illiquid/flat/regime-broken names
so every gate has something to catch, then prints gate outcomes, forecasts,
Z-ranks, and the correlation-penalized shortlist.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.forecast import MomentumVolForecaster  # noqa: E402
from core.manager import ScreenerConfig, run_screen  # noqa: E402

OHLCV = ("open", "high", "low", "close", "volume")


def load_csv_universe(data_dir: Path) -> dict[str, pd.DataFrame]:
    universe: dict[str, pd.DataFrame] = {}
    if not data_dir.exists():
        return universe
    for path in sorted(data_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not all(col in frame.columns for col in OHLCV):
            continue
        universe[path.stem] = frame
    return universe


def synthetic_universe(n_symbols: int, n_bars: int = 700, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Seeded demo universe: correlated names plus deliberate gate-bait."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n_bars)

    specs: list[tuple[str, float, float, float, float]] = []  # (name, beta, drift, vol, volume)
    if n_symbols >= 1:
        specs.append(("SYN_ILLIQ", 0.0, 1e-5, 1e-3, 300.0))          # fails dollar volume
    if n_symbols >= 2:
        specs.append(("SYN_FLAT", 0.0, 0.0, 2e-7, 5e4))              # fails ATR floor
    if n_symbols >= 3:
        specs.append(("SYN_REGIMEBREAK", 0.2, 8e-5, 6e-4, 5e4))      # blows up at the end
    for i in range(3, n_symbols):
        specs.append(
            (
                f"SYN_{i:02d}",
                float(rng.uniform(0.3, 0.9)),
                float(rng.uniform(-2e-4, 2e-4)),
                float(rng.uniform(6e-4, 2e-3)),
                float(5e4),
            )
        )

    universe: dict[str, pd.DataFrame] = {}
    for name, beta, drift, vol, base_volume in specs:
        rets = drift + vol * rng.standard_normal(n_bars) + beta * vol * common
        if name == "SYN_REGIMEBREAK":
            rets[-40:] += rng.standard_normal(40) * 8e-3  # variance shock -> CUSUM
        close = 100.0 * np.exp(np.cumsum(rets))
        open_ = np.empty_like(close)
        open_[0] = 100.0
        open_[1:] = close[:-1]
        wick = np.abs(rng.standard_normal(n_bars)) * vol * close
        high = np.maximum(open_, close) + wick
        low = np.minimum(open_, close) - wick
        volume = base_volume * np.exp(rng.standard_normal(n_bars) * 0.3)
        universe[name] = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=n_bars, freq="5min"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    return universe


def print_result(result, config: ScreenerConfig) -> None:
    reports = result.reports
    print(f"\nuniverse: {len(reports)} symbols | policy: {result.policy.as_dict()}")
    print(f"context : {result.context}")

    funnel: dict[str, int] = {}
    for report in reports.values():
        for failure in report.gate_failures or ["passed"]:
            funnel[failure] = funnel.get(failure, 0) + 1
    print("\ngate funnel:")
    for reason, count in sorted(funnel.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<28} {count}")

    scored = [r for r in reports.values() if r.z_score is not None]
    if scored:
        print("\nscored candidates (Z = (E - cost) / sigma):")
        header = f"  {'symbol':<18}{'E[h]':>9}{'sigma':>9}{'cost':>9}{'Z':>7}  flags"
        print(header)
        for report in sorted(scored, key=lambda r: -r.z_score):
            flags = "SELECTED" if report.selected else ("corr-blocked" if report.blocked_by_correlation else "")
            print(
                f"  {report.symbol:<18}{report.expected_return:>9.5f}"
                f"{report.sigma:>9.5f}{report.round_trip_cost:>9.5f}"
                f"{report.z_score:>7.2f}  {flags} [{report.forecast_method}]"
            )
    else:
        print("\nno symbol cleared the cost gate")

    print(f"\nshortlist       : {result.shortlist}")
    print(f"corr-blocked    : {result.blocked}")
    print(f"effective bets  : {result.effective_bets:.2f} "
          f"(of {len(result.shortlist)} selected)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="directory with OHLCV CSVs")
    parser.add_argument("--synthetic", type=int, default=12, help="synthetic universe size")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--bars", type=int, default=700)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-dollar-volume", type=float, default=1_000_000.0)
    parser.add_argument("--min-bars", type=int, default=250)
    args = parser.parse_args()

    universe: dict[str, pd.DataFrame] = {}
    if not args.synthetic_only:
        loaded = load_csv_universe(Path(args.data_dir))
        print(f"loaded {len(loaded)} CSV symbol(s) from {args.data_dir}: {sorted(loaded)}")
        universe.update(loaded)
    if args.synthetic > 0:
        universe.update(synthetic_universe(args.synthetic, n_bars=args.bars, seed=args.seed))

    config = ScreenerConfig(
        min_bars=args.min_bars,
        min_dollar_volume=args.min_dollar_volume,
        top_k=args.top_k,
    )
    result = run_screen(universe, config=config, forecaster=MomentumVolForecaster())
    print_result(result, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
