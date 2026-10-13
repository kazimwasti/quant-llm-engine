"""Summarize paired backtest trade/equity CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def summarize(name: str, prefix: Path, initial_balance: float) -> dict:
    trades_path = Path(f"{prefix}_trades.csv")
    equity_path = Path(f"{prefix}_equity.csv")
    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    equity = pd.read_csv(equity_path) if equity_path.exists() else pd.DataFrame()
    if not trades_path.exists() or not equity_path.exists():
        raise FileNotFoundError(f"Expected both {trades_path} and {equity_path}")
    pnl = pd.to_numeric(trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    gains, losses = pnl[pnl > 0].sum(), abs(pnl[pnl < 0].sum())
    curve = pd.to_numeric(equity.get("equity", pd.Series(dtype=float)), errors="coerce").dropna()
    peak = curve.cummax().clip(lower=initial_balance)
    max_dd = float((peak - curve).max()) if not curve.empty else 0.0
    max_loss_streak = streak = 0
    for value in pnl:
        streak = streak + 1 if value < 0 else 0
        max_loss_streak = max(max_loss_streak, streak)
    return {
        "run": name,
        "trades": int(len(pnl)),
        "net_pnl": float(pnl.sum()),
        "max_drawdown": max_dd,
        "profit_factor": float(gains / losses) if losses else (float("inf") if gains else 0.0),
        "max_loss_streak": max_loss_streak,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PREFIX",
                        help="Repeat for each output prefix, e.g. baseline=results/rule_baseline")
    parser.add_argument("--initial-balance", type=float, default=50_000.0)
    parser.add_argument("--output", default="results/comparison.csv")
    args = parser.parse_args()
    rows = []
    for item in args.run:
        if "=" not in item:
            parser.error(f"invalid --run {item!r}; expected NAME=PREFIX")
        name, raw_prefix = item.split("=", 1)
        prefix = Path(raw_prefix)
        if not prefix.is_absolute():
            prefix = PROJECT_ROOT / prefix
        rows.append(summarize(name, prefix, args.initial_balance))
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved comparison to {output}")


if __name__ == "__main__":
    main()
