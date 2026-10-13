"""Deterministic historical simulator for intraday Jev strategy bars.

Input is OHLC CSV (default expected interval: 5 minutes). Decisions are made
from completed bars and fill at the next bar's open. OHLC-only intra-bar
movement uses an explicit conservative ambiguity policy.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.engine import (
    DiscreteDeviationTrigger,
    FeatureLogitBackend,
    MarketState,
    TorchSequenceClassifierBackend,
    build_execution_payload,
    extract_decision,
)
from core.risk import Platform, PropFirmRiskManager
from core.strategy import (
    DEFAULT_ATR_TARGET_MULTIPLIER,
    add_indicators,
    align_higher_timeframe_ema,
    generate_signals,
)

logger = logging.getLogger("Backtester")


class Simulator:
    def __init__(
        self,
        initial_balance: float = 50_000.0,
        *,
        symbol: str = "MNQ",
        point_value: float = 2.0,
        platform: Platform = Platform.GENERIC,
        slippage_points: float = 0.0,
        commission_per_contract: float = 2.50,
        min_confidence: float = 0.65,
        ambiguous_stop_probability: float = 0.75,
        ambiguous_fill_policy: str = "conservative",
        random_seed: int = 1729,
        backend=None,
        daily_profit_target: float | None = 1_000.0,
        deviation_horizon: int = 12,
        deviation_envelope_atr: float = 1.0,
        tick_size: float = 0.25,
    ) -> None:
        if initial_balance <= 0 or point_value <= 0:
            raise ValueError("initial_balance and point_value must be positive")
        if slippage_points < 0 or commission_per_contract < 0:
            raise ValueError("slippage and commission cannot be negative")
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if not 0 <= min_confidence <= 1 or not 0 <= ambiguous_stop_probability <= 1:
            raise ValueError("confidence and ambiguous-stop probability must be between 0 and 1")
        if ambiguous_fill_policy not in {"conservative", "probabilistic"}:
            raise ValueError("ambiguous_fill_policy must be conservative or probabilistic")
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.symbol = symbol.upper()
        self.point_value = float(point_value)
        self.slippage_points = float(slippage_points)
        self.commission_per_contract = float(commission_per_contract)
        self.tick_size = float(tick_size)
        self.min_confidence = float(min_confidence)
        self.ambiguous_stop_probability = float(ambiguous_stop_probability)
        self.ambiguous_fill_policy = ambiguous_fill_policy
        self.rng = random.Random(random_seed)
        self.ambiguous_bars = 0
        self.ambiguous_stop_fills = 0
        self.backend = backend
        self.daily_profit_target = daily_profit_target
        self.deviation_trigger = DiscreteDeviationTrigger(
            horizon=deviation_horizon,
            envelope_atr_multiplier=deviation_envelope_atr,
        )
        self.evaluations = 0
        self.skipped_evaluations = 0
        self.risk_manager = PropFirmRiskManager(
            starting_equity=self.balance,
            platform=platform,
            point_value=self.point_value,
            daily_profit_target=daily_profit_target,
            baseline_confidence=self.min_confidence,
        )
        self.position: dict | None = None
        self.pending_order: dict | None = None
        self.cancelled_pending_orders = 0
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []

    def run_backtest(
        self,
        source: pd.DataFrame,
        *,
        higher_timeframe: pd.DataFrame | None = None,
        output_prefix: str = "backtest",
    ) -> dict:
        required = {"open", "high", "low", "close"}
        missing = required.difference(source.columns)
        if missing:
            raise ValueError(f"CSV is missing OHLC columns: {sorted(missing)}")
        if source.empty:
            raise ValueError("OHLC input has no rows")

        intraday = source.reset_index(drop=True)
        higher_timeframe_ema = None
        if higher_timeframe is not None:
            higher_timeframe_ema = align_higher_timeframe_ema(
                intraday, higher_timeframe.reset_index(drop=True)
            )
            if not higher_timeframe_ema.notna().any():
                raise ValueError(
                    "no completed higher-timeframe EMA aligns with the intraday data; "
                    "provide matching dates and enough hourly warmup history"
                )
        df = generate_signals(
            add_indicators(intraday, higher_timeframe_ema=higher_timeframe_ema)
        )
        if "timestamp" in intraday.columns:
            timestamps = pd.to_datetime(intraday["timestamp"], utc=True, errors="coerce")
        elif isinstance(source.index, pd.DatetimeIndex):
            timestamps = pd.Series(source.index, index=source.index).reset_index(drop=True)
        else:
            timestamps = pd.Series(range(len(df)))

        previous_session = None
        timestamp_values = tuple(timestamps)
        for i, row in enumerate(df.itertuples(index=False)):
            ts = timestamp_values[i]
            local_ts = ts.tz_convert("America/New_York") if pd.notna(ts) and getattr(ts, "tzinfo", None) else ts
            session = (local_ts - pd.Timedelta(hours=18)).date() if pd.notna(local_ts) and hasattr(local_ts, "date") else None
            if previous_session is not None and session is not None and session != previous_session:
                self._cancel_pending_order("session_end")
                if self.position is not None:
                    self._close(float(row.open), "session_end", ts)
                self.risk_manager.reset_day(self.balance)
            if session is not None:
                previous_session = session
            local_minute = local_ts.hour * 60 + local_ts.minute if pd.notna(local_ts) and hasattr(local_ts, "hour") else 0
            session_elapsed = min(1.0, ((local_minute - 18 * 60) % (24 * 60)) / (23 * 60))
            self.risk_manager.session_progress = session_elapsed
            self._activate_pending_order(row, ts)

            if pd.isna(row.ema_200) or pd.isna(row.atr_14) or row.atr_14 <= 0:
                continue

            state = MarketState(
                close=float(row.close), ema_200=float(row.ema_200),
                stoch_k=float(row.stoch_k), stoch_d=float(row.stoch_d),
                atr_14=float(row.atr_14), long_stop=float(row.long_stop),
                short_stop=float(row.short_stop), signal=int(row.signal),
                session_allowed=bool(row.session_allowed),
                volatility_active=bool(row.volatility_active),
                expected_drift=float(row.ema_drift),
                session_progress=float(row.session_progress),
                htf_ema=float(row.htf_ema),
                long_structure_confirmed=bool(row.long_structure_confirmed),
                short_structure_confirmed=bool(row.short_structure_confirmed),
            )
            decision_due = self.deviation_trigger.observe(state)

            self._manage_position(row, ts)
            equity = self.balance
            if self.position:
                direction = 1 if self.position["action"] == "BUY" else -1
                equity += (float(row.close) - self.position["entry"]) * direction * self.position["quantity"] * self.point_value
            self.risk_manager.mark_equity(max(equity, 0.0))
            if self.risk_manager.circuit_breaker_active() and self.position is not None:
                self._close(float(row.close), "circuit_breaker", ts)
                equity = self.balance
            self.equity_curve.append({"timestamp": ts, "balance": self.balance, "equity": equity,
                                      "position": self.position["action"] if self.position else "FLAT"})

            if (
                self.position is None
                and self.pending_order is None
                and not self.risk_manager.circuit_breaker_active()
            ):
                if not decision_due:
                    self.skipped_evaluations += 1
                    continue
                self.evaluations += 1
                decision = extract_decision(
                    state,
                    self.backend or FeatureLogitBackend(state),
                    min_confidence=max(
                        self.min_confidence,
                        self.risk_manager.required_confidence(session_elapsed),
                    ),
                )
                friction_per_contract = (2 * self.slippage_points * self.point_value
                                         + 2 * self.commission_per_contract)
                payload = build_execution_payload(
                    decision,
                    self.risk_manager,
                    point_value=self.point_value,
                    cost_per_contract=friction_per_contract,
                )
                if payload.approved and payload.size is not None:
                    self.risk_manager.reserve_open_risk(payload.size)
                    self.pending_order = {
                        "action": payload.action, "quantity": int(payload.quantity),
                        "risk_dollars": payload.risk_dollars,
                        "stop_distance": payload.stop_distance,
                        "target_distance": DEFAULT_ATR_TARGET_MULTIPLIER * float(row.atr_14),
                        "entry_atr": float(row.atr_14),
                        "signal_time": ts,
                    }

        if self.position is not None:
            last = df.iloc[-1]
            self._close(float(last["close"]), "end_of_data", timestamp_values[-1])
        self._cancel_pending_order("end_of_data")
        return self._write_results(output_prefix)

    def _activate_pending_order(self, row, timestamp) -> None:
        """Fill an approved completed-bar signal on the following bar's open."""
        pending = self.pending_order
        if pending is None:
            return
        fill_entry = float(row.open) + (
            self.slippage_points if pending["action"] == "BUY" else -self.slippage_points
        )
        is_long = pending["action"] == "BUY"
        stop_distance = pending["stop_distance"]
        target_distance = pending["target_distance"]
        self.position = {
            "action": pending["action"],
            "quantity": pending["quantity"],
            "entry": fill_entry,
            "stop": fill_entry + (-stop_distance if is_long else stop_distance),
            "risk_dollars": pending["risk_dollars"],
            "target": fill_entry + (target_distance if is_long else -target_distance),
            "entry_atr": pending["entry_atr"],
            "breakeven_active": False,
            "entry_time": timestamp,
            "signal_time": pending["signal_time"],
        }
        self.pending_order = None

    def _cancel_pending_order(self, reason: str) -> None:
        """Release risk for a signal that never received a next-bar fill."""
        if self.pending_order is None:
            return
        self.risk_manager.release_open_risk(self.pending_order["risk_dollars"])
        self.pending_order = None
        self.cancelled_pending_orders += 1
        logger.info("Cancelled pending order: %s", reason)

    def _manage_position(self, row, timestamp) -> None:
        if self.position is None:
            return
        p = self.position
        o, h, l, c = (float(getattr(row, name)) for name in ("open", "high", "low", "close"))
        is_long = p["action"] == "BUY"
        if (is_long and o <= p["stop"]) or (not is_long and o >= p["stop"]):
            self._close(o, "breakeven" if p["breakeven_active"] else "stop", timestamp)
            return
        if (is_long and o >= p["target"]) or (not is_long and o <= p["target"]):
            self._close(o, "target", timestamp)
            return

        # OHLC bars omit true tick order. Simulate the common nearer-extreme
        # path and apply the ratchet immediately when the +1 ATR level trades.
        path = [o, h, l, c] if abs(o - h) <= abs(o - l) else [o, l, h, c]
        trigger = p["entry"] + (p["entry_atr"] if is_long else -p["entry_atr"])
        # A favorable opening gap already crossed the activation threshold.
        if not p["breakeven_active"] and (
            (is_long and o >= trigger) or (not is_long and o <= trigger)
        ):
            p["stop"] = (
                max(p["stop"], p["entry"] + self.tick_size)
                if is_long
                else min(p["stop"], p["entry"] - self.tick_size)
            )
            p["breakeven_active"] = True
            p["breakeven_time"] = timestamp
        stop_touched = l <= p["stop"] if is_long else h >= p["stop"]
        target_touched = h >= p["target"] if is_long else l <= p["target"]
        if stop_touched and target_touched:
            self._resolve_ambiguous_bar(p, timestamp)
            return
        for start, end in zip(path, path[1:]):
            if is_long:
                if end > start:
                    if not p["breakeven_active"] and start < trigger <= end:
                        p["stop"] = max(p["stop"], p["entry"] + self.tick_size)
                        p["breakeven_active"] = True
                        p["breakeven_time"] = timestamp
                    if start < p["target"] <= end:
                        self._close(p["target"], "target", timestamp)
                        return
                elif end < start and end <= p["stop"] < start:
                    reason = "breakeven" if p["breakeven_active"] else "stop"
                    self._close(p["stop"], reason, timestamp)
                    return
            else:
                if end < start:
                    if not p["breakeven_active"] and end <= trigger < start:
                        p["stop"] = min(p["stop"], p["entry"] - self.tick_size)
                        p["breakeven_active"] = True
                        p["breakeven_time"] = timestamp
                    if end <= p["target"] < start:
                        self._close(p["target"], "target", timestamp)
                        return
                elif end > start and start < p["stop"] <= end:
                    reason = "breakeven" if p["breakeven_active"] else "stop"
                    self._close(p["stop"], reason, timestamp)
                    return

    def _resolve_ambiguous_bar(self, position: dict, timestamp) -> None:
        """Resolve both-barrier OHLC bars without assuming favorable tick order."""
        self.ambiguous_bars += 1
        stop_fill = (
            True
            if self.ambiguous_fill_policy == "conservative"
            else self.rng.random() < self.ambiguous_stop_probability
        )
        self.ambiguous_stop_fills += int(stop_fill)
        if position["action"] == "BUY":
            price = position["stop"] if stop_fill else position["target"]
        else:
            price = position["stop"] if stop_fill else position["target"]
        self._close(price, "ambiguous_stop" if stop_fill else "ambiguous_target", timestamp)

    def _close(self, raw_exit: float, reason: str, timestamp) -> None:
        p = self.position
        if p is None:
            return
        direction = 1 if p["action"] == "BUY" else -1
        exit_price = raw_exit - self.slippage_points * direction
        gross = (exit_price - p["entry"]) * direction * p["quantity"] * self.point_value
        fees = 2.0 * self.commission_per_contract * p["quantity"]
        pnl = gross - fees
        self.balance += pnl
        self.risk_manager.release_open_risk(p["risk_dollars"])
        self.risk_manager.mark_equity(max(self.balance, 0.0))
        self.risk_manager.record_trade_outcome(
            pnl,
            stop_out=reason in {"stop", "ambiguous_stop"} and pnl < 0,
        )
        self.trades.append({"entry_time": p["entry_time"], "exit_time": timestamp,
                            "symbol": self.symbol, "side": p["action"], "entry": p["entry"],
                            "exit": exit_price, "quantity": p["quantity"], "pnl": pnl,
                            "fees": fees, "reason": reason,
                            "breakeven_active": p["breakeven_active"],
                            "breakeven_time": p.get("breakeven_time"),
                            "signal_time": p.get("signal_time"),
                            "balance": self.balance})
        self.position = None

    def _write_results(self, prefix: str) -> dict:
        trades = pd.DataFrame(self.trades)
        equity = pd.DataFrame(self.equity_curve)
        output_base = Path(prefix)
        if not output_base.is_absolute():
            output_base = PROJECT_ROOT / output_base
        output_base.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(f"{output_base}_trades.csv", index=False)
        equity.to_csv(f"{output_base}_equity.csv", index=False)
        pnl = trades["pnl"] if not trades.empty else pd.Series(dtype=float)
        winners, losers = pnl[pnl > 0], pnl[pnl < 0]
        summary = {
            "trades": int(len(pnl)),
            "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
            "average_trade_return": float(pnl.mean()) if len(pnl) else 0.0,
            "average_trade_return_pct": (
                float(pnl.mean() / self.initial_balance * 100.0) if len(pnl) else 0.0
            ),
            "net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
            "profit_factor": float(winners.sum() / abs(losers.sum())) if len(losers) and losers.sum() else None,
            "ending_balance": self.balance,
            "daily_breaker_active": self.risk_manager.circuit_breaker_active(),
            "ambiguous_bars": self.ambiguous_bars,
            "ambiguous_stop_fills": self.ambiguous_stop_fills,
            "ambiguous_stop_probability": self.ambiguous_stop_probability,
            "ambiguous_fill_policy": self.ambiguous_fill_policy,
            "min_confidence": self.min_confidence,
            "backend": "qwen_lora" if self.backend is not None else "feature_fallback",
            "model_evaluations": self.evaluations,
            "skipped_evaluations": self.skipped_evaluations,
            "daily_profit_target": self.risk_manager.daily_profit_target,
            "required_confidence_now": self.risk_manager.required_confidence(),
            "consecutive_stopouts": self.risk_manager.consecutive_stopouts,
            "breakeven_exits": int((trades.get("reason", pd.Series(dtype=str)) == "breakeven").sum()),
            "cancelled_pending_orders": self.cancelled_pending_orders,
        }
        if not equity.empty:
            peak = equity["equity"].cummax()
            summary["max_drawdown"] = float((peak - equity["equity"]).max())
        print(summary)
        logger.info("Wrote %s_trades.csv and %s_equity.csv", prefix, prefix)
        return summary


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def _validate_interval(df: pd.DataFrame, expected_minutes: int) -> None:
    if expected_minutes < 1:
        raise ValueError("bar interval must be >= 1 minute")
    if "timestamp" not in df.columns or len(df) < 2:
        return
    gaps = df["timestamp"].diff().dropna().dt.total_seconds().div(60)
    median = float(gaps.median())
    if abs(median - expected_minutes) > max(0.01, expected_minutes * 0.01):
        raise ValueError(
            f"CSV median bar interval is {median:g} minutes; expected {expected_minutes}"
        )


def _load_lora_backend(adapter_dir: str) -> TorchSequenceClassifierBackend:
    """Load the PEFT sequence-classifier adapter produced by train_lora.py."""
    try:
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers, peft, and torch to load a Qwen adapter") from exc

    config = PeftConfig.from_pretrained(adapter_dir)
    base_id = config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_id,
        num_labels=3,
        label2id={"BUY": 0, "SELL": 1, "HOLD": 2},
        id2label={0: "BUY", 1: "SELL", 2: "HOLD"},
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return TorchSequenceClassifierBackend(model, tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Jev strategy on intraday OHLC bars")
    parser.add_argument("--input", default="data/5m.csv", help="OHLC CSV (default: data/5m.csv)")
    parser.add_argument(
        "--hourly-input",
        help=(
            "Completed hourly OHLC CSV for the higher-timeframe gate. "
            "Values are aligned only after their hourly bar has closed."
        ),
    )
    parser.add_argument("--balance", type=float, default=50_000.0)
    parser.add_argument("--symbol", choices=("MNQ", "NQ"), help="Contract sizing symbol (inferred from CSV if omitted)")
    parser.add_argument("--point-value", type=float, help="Dollars per index point (defaults: MNQ=2, NQ=20)")
    parser.add_argument("--slippage-points", type=float, default=0.25,
                        help="Adverse slippage per side in index points (default: one tick)")
    parser.add_argument("--commission-per-contract", type=float, default=2.50,
                        help="Commission and exchange fees per contract per side; broker dependent")
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--daily-target", type=float, default=1_000.0,
                        help="Daily profit target; 0 disables target confidence scaling")
    parser.add_argument("--deviation-horizon", type=int, default=12)
    parser.add_argument("--deviation-envelope-atr", type=float, default=1.0)
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--adapter-dir", help="Fine-tuned PEFT Qwen adapter directory (default: models/qwen_adapter)")
    backend_group.add_argument("--feature-fallback", action="store_true",
                               help="Use the handcrafted scorer for diagnostics, not Qwen validation")
    parser.add_argument("--ambiguous-stop-probability", type=float, default=0.75,
                        help="Probability an ambiguous both-barrier bar fills at stop (probabilistic policy only)")
    parser.add_argument(
        "--ambiguous-fill-policy",
        choices=("conservative", "probabilistic"),
        default="conservative",
        help="Both-barrier OHLC fill policy; conservative assumes the stop fills first",
    )
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--platform", choices=[p.value for p in Platform], default=Platform.GENERIC.value)
    parser.add_argument("--output-prefix", default="data/backtest")
    parser.add_argument("--bar-minutes", type=int, default=5,
                        help="Expected timestamp interval (default: 5; set to 60 for hourly data)")
    args = parser.parse_args()
    path = Path(args.input)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        parser.error(f"input file does not exist: {path}")
    data = _read_csv(path)
    hourly_data = None
    if args.hourly_input:
        hourly_path = Path(args.hourly_input)
        if not hourly_path.is_absolute():
            hourly_path = PROJECT_ROOT / hourly_path
        if not hourly_path.is_file():
            parser.error(f"hourly input file does not exist: {hourly_path}")
        hourly_data = _read_csv(hourly_path)
        try:
            _validate_interval(hourly_data, 60)
        except ValueError as exc:
            parser.error(str(exc))
    csv_symbol = str(data["symbol"].iloc[0]).upper() if "symbol" in data.columns else "MNQ=F"
    inferred_symbol = "NQ" if csv_symbol.startswith("NQ") and not csv_symbol.startswith("MNQ") else "MNQ"
    symbol = args.symbol or inferred_symbol
    point_values = {"MNQ": 2.0, "NQ": 20.0}
    point_value = args.point_value if args.point_value is not None else point_values[symbol]
    if args.daily_target < 0:
        parser.error("--daily-target must be >= 0")
    daily_target = args.daily_target or None
    adapter_dir = args.adapter_dir or "models/qwen_adapter"
    if args.feature_fallback:
        backend = None
    elif not (Path(adapter_dir) if Path(adapter_dir).is_absolute() else PROJECT_ROOT / adapter_dir).is_dir():
        parser.error(f"adapter directory not found: {adapter_dir}; train it or pass --feature-fallback")
    else:
        backend = _load_lora_backend(str(Path(adapter_dir) if Path(adapter_dir).is_absolute() else PROJECT_ROOT / adapter_dir))
    sim = Simulator(args.balance, symbol=symbol, point_value=point_value,
                    platform=Platform(args.platform), slippage_points=args.slippage_points,
                    commission_per_contract=args.commission_per_contract,
                    min_confidence=args.min_confidence,
                    ambiguous_stop_probability=args.ambiguous_stop_probability,
                    ambiguous_fill_policy=args.ambiguous_fill_policy,
                    random_seed=args.random_seed,
                    backend=backend,
                    daily_profit_target=daily_target,
                    deviation_horizon=args.deviation_horizon,
                    deviation_envelope_atr=args.deviation_envelope_atr)
    _validate_interval(data, args.bar_minutes)
    sim.run_backtest(data, higher_timeframe=hourly_data, output_prefix=args.output_prefix)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
