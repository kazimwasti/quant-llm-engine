import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np

from strategy import add_indicators, evaluate_signal
from risk import RiskManager, ExecutionPayload
from engine import FeatureLogitBackend, TradingEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Backtester")


class Simulator:
    def __init__(self, initial_balance: float = 50000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.risk_manager = RiskManager(account_balance=initial_balance)
        self.engine = TradingEngine(backend=FeatureLogitBackend())
        
        # Open position state
        self.position = None  # None, 'BUY', or 'SELL'
        self.entry_price = 0.0
        self.position_units = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
        # Trade & Performance Tracking
        self.trades = []
        self.equity_curve = []

    def run_backtest(self, df: pd.DataFrame):
        logger.info(f"Starting backtest on {len(df)} rows. Initial Balance: ${self.initial_balance:,.2f}")
        
        # 1. Compute Indicators
        df = add_indicators(df)

        for idx, row in df.iterrows():
            timestamp = row.get("timestamp", idx)
            close = row["close"]
            high = row["high"]
            low = row["low"]

            # Skip rows where indicators haven't warmed up
            if pd.isna(row.get("ema_200")) or pd.isna(row.get("atr_14")):
                continue

            # Update Equity Tracking
            current_equity = self.balance
            if self.position == "BUY":
                current_equity += (close - self.entry_price) * self.position_units
            elif self.position == "SELL":
                current_equity += (self.entry_price - close) * self.position_units

            if current_equity > self.peak_balance:
                self.peak_balance = current_equity

            self.equity_curve.append({
                "timestamp": timestamp,
                "balance": self.balance,
                "equity": current_equity,
                "position": self.position or "FLAT"
            })

            # 2. Manage Open Positions (Check Stops & Take Profits)
            if self.position == "BUY":
                if low <= self.stop_loss:
                    self._close_position(self.stop_loss, "Stop Loss", timestamp)
                elif high >= self.take_profit:
                    self._close_position(self.take_profit, "Take Profit", timestamp)
            elif self.position == "SELL":
                if high >= self.stop_loss:
                    self._close_position(self.stop_loss, "Stop Loss", timestamp)
                elif low <= self.take_profit:
                    self._close_position(self.take_profit, "Take Profit", timestamp)

            # 3. Evaluate New Signals if Flat
            if self.position is None:
                # Get raw technical signal
                tech_signal = evaluate_signal(row)
                
                # Run through Jev Engine & Risk Gate
                action, confidence, payload = self.engine.evaluate_and_gate(
                    row=row,
                    risk_manager=self.risk_manager,
                    contract_value=20.0  # e.g., $20 per point for NQ
                )

                if payload.approved and payload.action in ["BUY", "SELL"]:
                    self._open_position(payload, close, timestamp)

        # Print Performance Summary
        self._generate_summary()

    def _open_position(self, payload: ExecutionPayload, current_price: float, timestamp):
        self.position = payload.action
        self.entry_price = current_price
        self.position_units = payload.quantity
        self.stop_loss = payload.stop_price
        
        # Set 1.5x ATR Take Profit matching risk
        atr_distance = abs(current_price - payload.stop_price)
        if self.position == "BUY":
            self.take_profit = current_price + atr_distance
        else:
            self.take_profit = current_price - atr_distance

        logger.debug(f"[{timestamp}] Opened {self.position} @ {current_price:.2f} | Units: {payload.quantity} | SL: {self.stop_loss:.2f} | TP: {self.take_profit:.2f}")

    def _close_position(self, exit_price: float, reason: str, timestamp):
        if self.position == "BUY":
            pnl = (exit_price - self.entry_price) * self.position_units
        else:
            pnl = (self.entry_price - exit_price) * self.position_units

        self.balance += pnl
        self.risk_manager.record_trade_pnl(pnl)

        self.trades.append({
            "timestamp": timestamp,
            "side": self.position,
            "entry": self.entry_price,
            "exit": exit_price,
            "units": self.position_units,
            "pnl": pnl,
            "reason": reason,
            "ending_balance": self.balance
        })

        logger.debug(f"[{timestamp}] Closed {self.position} @ {exit_price:.2f} ({reason}) | PnL: ${pnl:,.2f} | Balance: ${self.balance:,.2f}")
        self.position = None

    def _generate_summary(self):
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        if trades_df.empty:
            logger.warning("No trades were executed during the backtest.")
            return

        # Save to CSV
        trades_df.to_csv("trades_log.csv", index=False)
        equity_df.to_csv("equity_curve.csv", index=False)

        total_trades = len(trades_df)
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]
        
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = self.balance - self.initial_balance
        gross_profit = wins["pnl"].sum() if not wins.empty else 0
        gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

        equity_df["drawdown"] = (equity_df["peak"] if "peak" in equity_df else equity_df["equity"].cummax()) - equity_df["equity"]
        max_drawdown = equity_df["drawdown"].max()

        print("\n" + "="*50)
        print("          BACKTEST PERFORMANCE SUMMARY          ")
        print("="*50)
        print(f"Initial Balance:     ${self.initial_balance:,.2f}")
        print(f"Final Balance:       ${self.balance:,.2f}")
        print(f"Total Net Profit:    ${total_pnl:,.2f} ({(total_pnl/self.initial_balance)*100:.2f}%)")
        print(f"Total Trades:        {total_trades}")
        print(f"Win Rate:            {win_rate:.2f}%")
        print(f"Profit Factor:       {profit_factor:.2f}" if not np.isnan(profit_factor) else "Profit Factor:       N/A")
        print(f"Max Drawdown:        ${max_drawdown:,.2f}")
        print("="*50)
        print("Logs saved to trades_log.csv and equity_curve.csv\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Jev Strategy Backtest")
    parser.add_argument("--input", type=str, default="hourly.csv", help="Path to historical OHLC CSV")
    parser.add_argument("--balance", type=float, default=50000.0, help="Starting account balance")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file '{args.input}' not found. Please provide a valid CSV file.")
        sys.exit(1)

    df_data = pd.read_csv(args.input)
    sim = Simulator(initial_balance=args.balance)
    sim.run_backtest(df_data)