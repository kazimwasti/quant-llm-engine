"""Offline acceptance tests for the Phase F5 operational safety gate."""

from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.engine import Decision, ExecutionPayload, MarketState
from core.manager import PortfolioCapacityGuard, ReentryCooldown, TradeTopology
from core.operations import PaperTradingSafetyController
from core.risk import PropFirmRiskManager


def approved_payload() -> ExecutionPayload:
    market = MarketState(
        close=20_000.0,
        ema_200=19_900.0,
        stoch_k=70.0,
        stoch_d=60.0,
        atr_14=20.0,
        long_stop=19_980.0,
        short_stop=20_020.0,
        htf_ema=19_800.0,
        long_structure_confirmed=True,
    )
    decision = Decision(
        action="BUY",
        probabilities={"BUY": 0.9, "SELL": 0.05, "HOLD": 0.05},
        logits={"BUY": 3.0, "SELL": 0.0, "HOLD": 0.0},
        confidence=0.9,
        prompt="test",
        source="test",
        market=market,
    )
    return ExecutionPayload(
        approved=True,
        action="BUY",
        model_action=decision.action,
        quantity=1.0,
        side="long",
        entry=market.close,
        stop_price=market.long_stop,
        stop_distance=20.0,
        risk_dollars=40.0,
        probabilities=decision.probabilities,
        confidence=decision.confidence,
        circuit_breaker=False,
        reason="validated",
        prompt=decision.prompt,
        source=decision.source,
    )


class OperationalReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(
            prefix="quant-operations-", dir=Path(__file__).resolve().parent
        )
        self.journal = Path(self.tmp.name) / "safety-state.json"
        self.risk = PropFirmRiskManager(starting_equity=50_000.0, daily_loss_limit_usd=500.0)
        self.payload = approved_payload()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def controller(self, **kwargs) -> PaperTradingSafetyController:
        controller = PaperTradingSafetyController(self.risk, self.journal, **kwargs)
        controller.mark_connection_healthy()
        return controller

    def test_disconnect_blocks_orders_until_explicit_reconnect(self) -> None:
        controller = self.controller()
        controller.mark_connection_lost()
        self.assertFalse(controller.authorize(self.payload, "disconnect-1").permitted)
        controller.mark_connection_healthy()
        self.assertTrue(controller.authorize(self.payload, "disconnect-1").permitted)

    def test_restart_does_not_replay_a_reserved_order(self) -> None:
        first = self.controller()
        self.assertTrue(first.authorize(self.payload, "restart-1").permitted)

        restarted = self.controller()
        replay = restarted.authorize(self.payload, "restart-1")
        self.assertFalse(replay.permitted)
        self.assertIn("duplicate", replay.reason)

    def test_concurrent_controllers_cannot_reserve_the_same_order(self) -> None:
        first, second = self.controller(), self.controller()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda controller: controller.authorize(self.payload, "concurrent-1"),
                    (first, second),
                )
            )
        self.assertEqual(sum(result.permitted for result in results), 1)

    def test_kill_switch_requires_flatten_and_survives_restart(self) -> None:
        controller = self.controller()
        stopped = controller.engage_kill_switch()
        self.assertTrue(stopped.requires_flatten)

        restarted = self.controller()
        blocked = restarted.authorize(self.payload, "kill-1")
        self.assertFalse(blocked.permitted)
        self.assertTrue(blocked.requires_flatten)

    def test_daily_loss_cap_blocks_new_order(self) -> None:
        controller = self.controller()
        self.risk.mark_equity(49_500.0)
        blocked = controller.authorize(self.payload, "daily-loss-1")
        self.assertFalse(blocked.permitted)
        self.assertIn("daily-loss", blocked.reason)

        restarted_risk = PropFirmRiskManager(
            starting_equity=50_000.0, daily_loss_limit_usd=500.0
        )
        restarted = PaperTradingSafetyController(restarted_risk, self.journal)
        restarted.mark_connection_healthy()
        self.assertFalse(restarted.authorize(self.payload, "daily-loss-2").permitted)

    def test_loss_streak_blocks_new_order(self) -> None:
        controller = self.controller()
        self.risk.record_trade_outcome(-50.0, stop_out=True)
        self.risk.record_trade_outcome(-50.0, stop_out=True)
        blocked = controller.authorize(self.payload, "streak-1")
        self.assertFalse(blocked.permitted)
        self.assertIn("consecutive", blocked.reason)

    def test_paper_mode_is_mandatory(self) -> None:
        controller = self.controller(paper_trading=False)
        blocked = controller.authorize(self.payload, "live-1")
        self.assertFalse(blocked.permitted)
        self.assertIn("paper", blocked.reason)

    def test_definitive_rejection_allows_a_retry(self) -> None:
        controller = self.controller()
        self.assertTrue(controller.authorize(self.payload, "retry-1").permitted)
        controller.record_definitive_rejection("retry-1")
        self.assertTrue(controller.authorize(self.payload, "retry-1").permitted)

    def test_manager_capacity_and_cooldown_gate_authorization(self) -> None:
        topology = TradeTopology()
        controller = self.controller(
            capacity_guard=PortfolioCapacityGuard(topology),
            cooldown=ReentryCooldown(topology),
        )
        # Manager gates fail closed without the context they need.
        no_symbol = controller.authorize(self.payload, "ctx-1")
        self.assertFalse(no_symbol.permitted)
        no_bar = controller.authorize(self.payload, "ctx-2", symbol="MNQ")
        self.assertFalse(no_bar.permitted)
        self.assertIn("cooldown", no_bar.reason)

        # A normal authorization passes both gates and reserves capacity.
        self.assertTrue(controller.authorize(self.payload, "ctx-3", symbol="MNQ", bar=0).permitted)

        # A stop-out starts the symbol cooldown; inside the window orders wait.
        controller.record_stopout("MNQ", bar=2)
        blocked = controller.authorize(self.payload, "ctx-4", symbol="MNQ", bar=5)
        self.assertFalse(blocked.permitted)
        self.assertIn("cooldown", blocked.reason)

        # Correlated symbols inherit the cooldown until it expires.
        correlated = controller.authorize(
            self.payload, "ctx-5", symbol="ES", bar=5, correlations={"MNQ": 0.95}
        )
        self.assertFalse(correlated.permitted)
        self.assertIn("cooldown", correlated.reason)
        expired = controller.authorize(
            self.payload, "ctx-6", symbol="ES", bar=8, correlations={"MNQ": 0.95}
        )
        self.assertTrue(expired.permitted)


if __name__ == "__main__":
    unittest.main()
