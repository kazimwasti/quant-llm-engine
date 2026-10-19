"""Regression tests for leakage and adverse-fill protections."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import pandas as pd

from backtest.backtest import Simulator
from core.manager import TradeTopology
from core.strategy import align_higher_timeframe_ema


class BacktestIntegrityTests(unittest.TestCase):
    def test_completed_hourly_values_are_not_available_before_hour_close(self) -> None:
        hourly = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01 07:00", periods=4, freq="h", tz="UTC"),
                "close": [100.0, 110.0, 1_000.0, 1_010.0],
            }
        )
        intraday = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-01-01 09:55+00:00", "2026-01-01 10:00+00:00"]
                )
            }
        )

        aligned = align_higher_timeframe_ema(intraday, hourly, ema_period=2)

        # At 09:55, the 09:00-10:00 close of 1,000 is still unknown.
        self.assertAlmostEqual(aligned.iloc[0], 106.6666666667)
        # It becomes available exactly when that hourly bar completes.
        self.assertAlmostEqual(aligned.iloc[1], 702.2222222222)

    def test_conservative_policy_charges_stop_on_an_ambiguous_bar(self) -> None:
        simulator = Simulator(
            initial_balance=1_000.0,
            point_value=1.0,
            commission_per_contract=0.0,
            ambiguous_fill_policy="conservative",
        )
        simulator.position = {
            "action": "BUY",
            "quantity": 1,
            "entry": 100.0,
            "stop": 90.0,
            "target": 110.0,
            "risk_dollars": 10.0,
            "entry_atr": 10.0,
            "breakeven_active": False,
            "entry_time": pd.Timestamp("2026-01-01", tz="UTC"),
        }

        simulator._manage_position(
            SimpleNamespace(open=100.0, high=111.0, low=89.0, close=100.0),
            pd.Timestamp("2026-01-01 00:05", tz="UTC"),
        )

        self.assertEqual(simulator.ambiguous_bars, 1)
        self.assertEqual(simulator.ambiguous_stop_fills, 1)
        self.assertEqual(simulator.trades[-1]["reason"], "ambiguous_stop")
        self.assertEqual(simulator.balance, 990.0)

    def test_completed_bar_signal_fills_at_next_open_with_adverse_slippage(self) -> None:
        simulator = Simulator(
            initial_balance=1_000.0,
            point_value=1.0,
            commission_per_contract=0.0,
            slippage_points=0.25,
        )
        signal_time = pd.Timestamp("2026-01-01 00:00", tz="UTC")
        simulator.pending_order = {
            "action": "BUY",
            "quantity": 1,
            "risk_dollars": 10.0,
            "stop_distance": 10.0,
            "target_distance": 22.0,
            "entry_atr": 10.0,
            "signal_time": signal_time,
        }

        fill_time = pd.Timestamp("2026-01-01 00:05", tz="UTC")
        simulator._activate_pending_order(
            SimpleNamespace(open=105.0, high=106.0, low=104.0, close=105.0), fill_time
        )

        self.assertIsNone(simulator.pending_order)
        self.assertEqual(simulator.position["entry"], 105.25)
        self.assertEqual(simulator.position["stop"], 95.25)
        self.assertEqual(simulator.position["target"], 127.25)
        self.assertEqual(simulator.position["entry_time"], fill_time)
        self.assertEqual(simulator.position["signal_time"], signal_time)

    def test_manager_holding_limit_forces_flat_at_next_bar_open(self) -> None:
        simulator = Simulator(
            initial_balance=1_000.0,
            point_value=1.0,
            commission_per_contract=0.0,
            topology=TradeTopology(max_holding_bars=1),
        )
        simulator.position = {
            "action": "BUY",
            "quantity": 1,
            "entry": 100.0,
            "stop": 90.0,
            "target": 110.0,
            "risk_dollars": 10.0,
            "entry_atr": 10.0,
            "breakeven_active": False,
            "entry_time": pd.Timestamp("2026-01-01", tz="UTC"),
            "max_holding_bars": 1,
            "bars_held": 1,
        }
        simulator._manage_position(
            SimpleNamespace(open=101.0, high=102.0, low=100.5, close=101.0),
            pd.Timestamp("2026-01-01 00:10", tz="UTC"),
        )
        self.assertIsNone(simulator.position)
        self.assertEqual(simulator.trades[-1]["reason"], "manager_time_limit")
        self.assertEqual(simulator.trades[-1]["exit"], 101.0)


if __name__ == "__main__":
    unittest.main()
