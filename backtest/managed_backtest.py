"""Out-of-sample screen -> multi-symbol backtest -> automatic policy feedback."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

from core.manager import ManagedScreen, PortfolioContext, ScreenerConfig
from .backtest import Simulator


def run_managed_backtest(
    managed: ManagedScreen,
    universe: Mapping[str, pd.DataFrame],
    *,
    history_bars: int,
    initial_balance: float = 50_000.0,
    higher_timeframes: Optional[Mapping[str, pd.DataFrame]] = None,
    portfolio: Optional[PortfolioContext] = None,
    simulator_options: Optional[dict] = None,
    output_dir: Optional[str | Path] = "data/managed_backtest",
) -> dict:
    """Screen only the historical prefix, backtest its shortlist, then learn.

    Each symbol needs aligned OHLCV rows with history before the out-of-sample
    segment. The simulator uses the prefix for indicator warmup but will not
    place or evaluate trades until ``history_bars``. After every selected
    symbol finishes, their realized net P&L is aggregated and automatically
    attributed to the policy source that produced this screen.

    The screen runs once per call, so callers can invoke this on successive
    walk-forward folds while retaining the same ``ManagedScreen`` and bandit.
    No-trade folds intentionally do not update the bandit.
    """
    if not universe:
        raise ValueError("universe must contain at least one symbol")
    if not isinstance(history_bars, int) or history_bars < 1:
        raise ValueError("history_bars must be a positive integer")
    if not math.isfinite(initial_balance) or initial_balance <= 0:
        raise ValueError("initial_balance must be positive")
    config = managed.config or ScreenerConfig()
    if history_bars < config.min_bars:
        raise ValueError(f"history_bars must be at least the screener minimum ({config.min_bars})")

    lengths = {symbol: len(frame) for symbol, frame in universe.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError("all symbols must have the same number of aligned bars")
    if any(length <= history_bars for length in lengths.values()):
        short = sorted(symbol for symbol, length in lengths.items() if length <= history_bars)
        raise ValueError(f"each symbol needs more than {history_bars} bars; too short: {short}")

    # The screener's cross-symbol calculations use row alignment. Verify the
    # full time axis rather than only the split point to avoid comparing
    # different market intervals across tickers.
    timestamp_presence = {"timestamp" in frame.columns for frame in universe.values()}
    if len(timestamp_presence) != 1:
        raise ValueError("either every symbol must have timestamps or none may have them")
    if True in timestamp_presence:
        reference = pd.to_datetime(next(iter(universe.values()))["timestamp"], utc=True)
        for symbol, frame in universe.items():
            timestamps = pd.to_datetime(frame["timestamp"], utc=True)
            if timestamps.isna().any() or not timestamps.reset_index(drop=True).equals(
                reference.reset_index(drop=True)
            ):
                raise ValueError(f"timestamps for {symbol} are not aligned with the universe")

    history = {symbol: frame.iloc[:history_bars].copy() for symbol, frame in universe.items()}
    result, chosen_source = managed.screen(history, portfolio)
    selected = [symbol for symbol in result.shortlist if symbol in universe]
    if not selected:
        return {
            "policy_source": chosen_source,
            "screen": result,
            "shortlist": [],
            "symbol_summaries": {},
            "trades": [],
            "net_pnl": 0.0,
            "feedback_reward": None,
            "feedback_updated": False,
            "bandit_stats": managed.bandit.stats(),
        }

    options = dict(simulator_options or {})
    forbidden = {"initial_balance", "symbol", "topology"}.intersection(options)
    if forbidden:
        raise ValueError(f"simulator_options cannot override managed fields: {sorted(forbidden)}")
    per_symbol_balance = float(initial_balance) / len(selected)
    output_root = Path(output_dir) if output_dir is not None else None
    summaries: dict[str, dict] = {}
    all_trades: list[dict] = []

    for symbol in selected:
        simulator = Simulator(
            per_symbol_balance,
            symbol=symbol,
            topology=result.topology,
            **options,
        )
        output_prefix = "managed_backtest"
        if output_root is not None:
            safe_symbol = re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol)
            output_prefix = str(output_root / safe_symbol)
        summary = simulator.run_backtest(
            universe[symbol],
            higher_timeframe=(higher_timeframes or {}).get(symbol),
            output_prefix=output_prefix,
            trade_start_index=history_bars,
        )
        summaries[symbol] = summary
        all_trades.extend(simulator.trades)

    reward = None
    if all_trades:
        reward = managed.feedback_from_trades(
            result,
            all_trades,
            starting_equity=float(initial_balance),
        )

    return {
        "policy_source": chosen_source,
        "screen": result,
        "shortlist": selected,
        "symbol_summaries": summaries,
        "trades": all_trades,
        "net_pnl": sum(float(trade["pnl"]) for trade in all_trades),
        "feedback_reward": reward,
        "feedback_updated": reward is not None,
        "bandit_stats": managed.bandit.stats(),
    }
