"""Tests for the profit levers: R-weighted training loss, barrier outcomes
in the labeler meta, and confidence-scaled position sizing."""

from __future__ import annotations

import unittest

import pandas as pd

from core.engine import Decision, MarketState, build_execution_payload
from core.risk import PropFirmRiskManager
from scripts.dataset_generator import _first_barrier, iter_labeled_examples
from scripts.train_lora import example_weight


def _barrier_series(lows, highs, closes, entry: float = 100.0):
    opens = pd.Series([entry] * len(closes), dtype=float)
    return (
        opens,
        pd.Series(highs, dtype=float),
        pd.Series(lows, dtype=float),
        pd.Series(closes, dtype=float),
    )


class ExampleWeightTests(unittest.TestCase):
    def test_directional_rows_scale_with_realized_r(self) -> None:
        self.assertAlmostEqual(example_weight("BUY", {"realized_r": 2.2}), 3.2)
        self.assertAlmostEqual(example_weight("SELL", {"realized_r": 1.0}), 2.0)
        self.assertAlmostEqual(
            example_weight("BUY", {"realized_r": 2.2}, r_weight_scale=0.0), 1.0
        )
        # A negative outcome on a directional row keeps the base weight; the
        # label itself already encodes the loss as a HOLD downstream of rules.
        self.assertAlmostEqual(example_weight("BUY", {"realized_r": -1.0}), 1.0)

    def test_hold_rows_split_by_candidate(self) -> None:
        # Gated candidates (stop-outs, timeouts) are the rules' false
        # positives and carry the selectivity signal: full weight.
        self.assertAlmostEqual(example_weight("HOLD", {"realized_r": -1.0}), 1.0)
        self.assertAlmostEqual(example_weight("HOLD", {"realized_r": 0.3}), 1.0)
        # Background no-setup rows are the 99% majority: down-weighted.
        self.assertAlmostEqual(example_weight("HOLD", {"realized_r": None}), 0.25)
        self.assertAlmostEqual(
            example_weight("HOLD", {"realized_r": None}, hold_weight=1.0), 1.0
        )

    def test_legacy_rows_without_realized_r_fall_back(self) -> None:
        self.assertAlmostEqual(example_weight("BUY", {}), 1.0)
        self.assertAlmostEqual(example_weight("HOLD", {}), 0.25)
        self.assertAlmostEqual(example_weight("BUY", "not-a-dict"), 1.0)


class FirstBarrierTests(unittest.TestCase):
    def test_target_stop_and_timeout_outcomes(self) -> None:
        # Entry 100 with ATR 1.0: stops at 99/101, targets at 102.2/97.8.
        opens, highs, lows, closes = _barrier_series(
            lows=[99.5, 99.4, 99.6],
            highs=[101.0, 102.3, 101.0],
            closes=[100.5, 100.8, 101.0],
        )
        outcomes = _first_barrier(
            opens, highs, lows, closes, start=0, horizon=3,
            atr_value=1.0, stop_atr_multiplier=1.0, target_atr_multiplier=2.2,
        )
        self.assertEqual(outcomes["BUY"]["reason"], "target")
        self.assertAlmostEqual(outcomes["BUY"]["realized_r"], 2.2)
        # The short side's stop (101) trades before its target (97.8).
        self.assertEqual(outcomes["SELL"]["reason"], "stop")
        self.assertAlmostEqual(outcomes["SELL"]["realized_r"], -1.0)

    def test_timeout_marks_to_market_at_expiry(self) -> None:
        opens, highs, lows, closes = _barrier_series(
            lows=[99.9, 99.9, 99.9],
            highs=[100.4, 100.4, 100.4],
            closes=[100.1, 100.2, 100.5],
        )
        outcomes = _first_barrier(
            opens, highs, lows, closes, start=0, horizon=3,
            atr_value=1.0, stop_atr_multiplier=1.0, target_atr_multiplier=2.2,
        )
        self.assertEqual(outcomes["BUY"]["reason"], "timeout")
        self.assertAlmostEqual(outcomes["BUY"]["realized_r"], 0.5)
        self.assertEqual(outcomes["SELL"]["reason"], "timeout")
        self.assertAlmostEqual(outcomes["SELL"]["realized_r"], -0.5)

    def test_same_bar_touch_resolves_stop_first(self) -> None:
        opens, highs, lows, closes = _barrier_series(
            lows=[98.0], highs=[103.0], closes=[100.0],
        )
        outcomes = _first_barrier(
            opens, highs, lows, closes, start=0, horizon=1,
            atr_value=1.0, stop_atr_multiplier=1.0, target_atr_multiplier=2.2,
        )
        self.assertEqual(outcomes["BUY"]["reason"], "same_bar_stop_first")
        self.assertAlmostEqual(outcomes["BUY"]["realized_r"], -1.0)


class LabeledMetaTests(unittest.TestCase):
    def _featured_frame(self) -> pd.DataFrame:
        n = 40
        return pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [103.0] * n,  # breaches the 102.2 BUY target at once
                "low": [99.5] * n,
                "close": [100.5] * n,
                "ema_200": [90.0] * n,
                "htf_ema": [90.0] * n,
                "stoch_k": [80.0] * n,
                "stoch_d": [70.0] * n,
                "atr_14": [1.0] * n,
                "long_stop": [99.0] * n,
                "short_stop": [101.0] * n,
                "signal": [1] * n,
                "session_allowed": [True] * n,
                "volatility_active": [True] * n,
                "long_structure_confirmed": [True] * n,
                "short_structure_confirmed": [False] * n,
                "bullish_rejection": [True] * n,
                "bearish_rejection": [False] * n,
            }
        )

    def test_meta_carries_candidate_and_realized_r(self) -> None:
        records = list(iter_labeled_examples(self._featured_frame()))
        self.assertTrue(records)
        for record in records:
            meta = record["meta"]
            self.assertIn("candidate", meta)
            self.assertIn("realized_r", meta)
            self.assertEqual(meta["candidate"], "BUY")
        directional = [record for record in records if record["action"] == "BUY"]
        self.assertTrue(directional)
        for record in directional:
            self.assertEqual(record["meta"]["reason"], "buy_target_before_stop")
            self.assertAlmostEqual(record["meta"]["realized_r"], 2.2)


class ConfidenceSizingTests(unittest.TestCase):
    def test_scale_mapping_is_linear_and_clamped(self) -> None:
        risk = PropFirmRiskManager(starting_equity=10_000.0)
        self.assertEqual(risk.confidence_risk_scale(None), 1.0)
        self.assertAlmostEqual(risk.confidence_risk_scale(0.65), 0.25)
        self.assertAlmostEqual(risk.confidence_risk_scale(1.0), 1.0)
        self.assertAlmostEqual(risk.confidence_risk_scale(0.825), 0.625)
        self.assertAlmostEqual(risk.confidence_risk_scale(0.10), 0.25)  # clamped

    def test_sizing_never_scales_up_and_is_monotonic(self) -> None:
        risk = PropFirmRiskManager(starting_equity=10_000.0, point_value=1.0)
        legacy = risk.size_position(2.0)
        full = risk.size_position(2.0, confidence=1.0)
        mid = risk.size_position(2.0, confidence=0.825)
        floor = risk.size_position(2.0, confidence=0.65)
        below = risk.size_position(2.0, confidence=0.30)
        self.assertAlmostEqual(legacy.risk_dollars, 50.0)
        self.assertAlmostEqual(full.risk_dollars, 50.0)
        # Quantity rounding: 12.5 raw contracts floor to 6 -> 12.0 dollars.
        self.assertAlmostEqual(floor.risk_dollars, 12.0)
        self.assertAlmostEqual(below.risk_dollars, 12.0)  # clamped to the floor
        self.assertGreater(full.risk_dollars, mid.risk_dollars)
        self.assertGreaterEqual(mid.risk_dollars, floor.risk_dollars)

    def test_engine_payload_sizes_with_confidence(self) -> None:
        market = MarketState(
            close=100.0, ema_200=90.0, stoch_k=80.0, stoch_d=70.0,
            atr_14=2.0, long_stop=98.0, short_stop=102.0, htf_ema=90.0,
            signal=1, long_structure_confirmed=True,
        )
        risk = PropFirmRiskManager(starting_equity=10_000.0, point_value=1.0)

        def decision(confidence: float, buy_probability: float) -> Decision:
            return Decision(
                action="BUY",
                probabilities={
                    "BUY": buy_probability,
                    "SELL": 0.05,
                    "HOLD": max(0.0, 0.95 - buy_probability),
                },
                logits={"BUY": 3.0, "SELL": 0.0, "HOLD": 0.0},
                confidence=confidence, prompt="p", source="t", market=market,
            )

        certain = build_execution_payload(
            decision(1.0, 1.0), risk, manager_risk_multiplier=1.20
        )
        likely = build_execution_payload(
            decision(0.90, 0.90), risk, manager_risk_multiplier=1.20
        )
        self.assertTrue(certain.approved and likely.approved)
        self.assertGreater(certain.risk_dollars, likely.risk_dollars)


if __name__ == "__main__":
    unittest.main()
