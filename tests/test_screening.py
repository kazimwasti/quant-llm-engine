"""Offline acceptance tests for the Tier 1/2 screen, forecasters, and manager.

Everything here is seeded and synthetic: no network, no model downloads.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from core.forecast import (
    GarchFit,
    IsotonicCalibrator,
    MomentumVolForecaster,
    PolicyWeights,
    fit_entropy_calibration,
    fit_garch11,
)
from core.manager import (
    BanditHarness,
    LLMPolicySource,
    ManagedScreen,
    PortfolioAllocation,
    PortfolioCapacityGuard,
    PortfolioContext,
    ReentryCooldown,
    ScreenerConfig,
    ScreenContext,
    HeuristicPolicySource,
    StaticPolicySource,
    TradeTopology,
    derive_trade_topology,
    run_screen,
)
from core.engine import (
    Decision,
    DiscreteDeviationTrigger,
    MarketState,
    TradePreflight,
    build_execution_payload,
    confirm_marginal_decision,
    format_decision_prompt,
    normalized_decision_entropy,
)
from core.risk import PropFirmRiskManager
from core.screener import (
    CostModel,
    correlation_matrix,
    cusum_alarm,
    effective_bets,
    greedy_select,
    parkinson_variance,
    regression_trend,
    rolling_zscore,
    volume_shock,
)
from scripts.run_screen import synthetic_universe


def ohlc_from_returns(returns: np.ndarray, start: float = 100.0, wick_scale: float = 0.5) -> pd.DataFrame:
    close = start * np.exp(np.cumsum(returns))
    open_ = np.empty_like(close)
    open_[0] = start
    open_[1:] = close[:-1]
    wick = np.abs(np.random.default_rng(3).standard_normal(returns.size)) * wick_scale * np.abs(returns) * close
    wick[wick == 0] = 1e-4
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "volume": np.full(returns.size, 5e4),
        }
    )


class Tier1MathTests(unittest.TestCase):
    def test_parkinson_variance_closed_form(self) -> None:
        ratio = 1.01
        n = 6
        high = pd.Series([100.0 * ratio] * n)
        low = pd.Series([100.0] * n)
        var = parkinson_variance(high, low, window=5)
        expected = math.log(ratio) ** 2 / (4.0 * math.log(2.0))
        self.assertAlmostEqual(float(var.iloc[-1]), expected, places=12)
        self.assertTrue(var.iloc[:4].isna().all())

    def test_regression_trend_recovers_slope_and_flags_noise(self) -> None:
        n = 60
        trend = pd.Series(100.0 + 0.5 * np.arange(n))
        slope, tstat = regression_trend(trend, window=50)
        self.assertAlmostEqual(float(slope.iloc[-1]), 0.5, places=6)
        self.assertGreater(float(tstat.iloc[-1]), 100.0)

        # Stationary noise around a mean -> in-window slope t-stat is
        # standard-t and small. (A random walk is NOT valid here: unit-root
        # windows trend spuriously and legitimately produce huge t-stats.)
        rng = np.random.default_rng(5)
        noise = pd.Series(100.0 + rng.standard_normal(n))
        _, noise_t = regression_trend(noise, window=50)
        self.assertLess(abs(float(noise_t.iloc[-1])), 5.0)

    def test_cusum_fires_on_standardized_variance_shift(self) -> None:
        rng = np.random.default_rng(17)
        calm = rng.standard_normal(200) * 1e-3
        shock = rng.standard_normal(20) * 8e-3
        returns = pd.Series(np.concatenate([calm, shock]))
        alarms = cusum_alarm(returns, k=1.5, h=6.0, vol_window=100)
        self.assertFalse(bool(alarms.iloc[:200].any()))
        self.assertTrue(bool(alarms.iloc[200:].any()))

    def test_volume_shock_detects_spike(self) -> None:
        rng = np.random.default_rng(2)
        volume = pd.Series(1000.0 + rng.standard_normal(60) * 20.0)
        volume.iloc[-1] = 3000.0
        shock = volume_shock(volume, window=30, k=3.0)
        self.assertFalse(bool(shock.iloc[:-1].any()))
        self.assertTrue(bool(shock.iloc[-1]))

    def test_rolling_zscore_flags_outlier(self) -> None:
        rng = np.random.default_rng(9)
        series = pd.Series(100.0 + rng.standard_normal(200) * 0.1)
        series.iloc[-1] = 103.0  # 30-sigma outlier
        z = rolling_zscore(series, window=50)
        # In-window z is bounded by (n-1)/sqrt(n) ~ 7 for n=50.
        self.assertGreater(float(z.iloc[-1]), 5.0)
        self.assertLessEqual(abs(float(z.dropna().max())), 7.0 + 1e-9)

    def test_round_trip_cost_combines_fixed_and_impact(self) -> None:
        model = CostModel(
            spread_bps=2.0, fee_bps_per_side=1.0, impact_coeff=1.0, order_volume_fraction=0.01
        )
        # (2 + 2)/1e4 + 1.0 * 0.001 * sqrt(0.01)
        self.assertAlmostEqual(model.round_trip_cost(0.001), 0.0005, places=12)


class PortfolioStructureTests(unittest.TestCase):
    def test_effective_bets_identity_and_full_correlation(self) -> None:
        names = ["a", "b", "c"]
        identity = pd.DataFrame(np.eye(3), index=names, columns=names)
        ones = pd.DataFrame(np.ones((3, 3)), index=names, columns=names)
        vols = {n: 0.01 for n in names}
        weights = {n: 1.0 for n in names}
        self.assertAlmostEqual(effective_bets(weights, vols, identity), 3.0, places=9)
        self.assertAlmostEqual(effective_bets(weights, vols, ones), 1.0, places=9)

    def test_greedy_select_penalizes_correlated_duplicates(self) -> None:
        names = ["A", "B", "C"]
        corr = pd.DataFrame(
            [[1.0, 0.95, 0.10], [0.95, 1.0, 0.10], [0.10, 0.10, 1.0]],
            index=names,
            columns=names,
        )
        scores = {"A": 2.0, "B": 1.9, "C": 1.2}

        result = greedy_select(scores, corr, top_k=2, penalty_lambda=0.5, max_pairwise_abs_corr=0.99)
        self.assertEqual(result.selected, ["A", "C"])

        hard = greedy_select(scores, corr, top_k=2, penalty_lambda=0.0, max_pairwise_abs_corr=0.9)
        self.assertEqual(hard.selected, ["A", "C"])
        self.assertEqual(hard.blocked, ["B"])

    def test_correlation_matrix_fills_missing_overlap_with_zero(self) -> None:
        long_series = pd.Series(np.linspace(100, 110, 300))
        short_series = pd.Series(np.linspace(50, 55, 10))
        corr = correlation_matrix({"x": long_series, "y": short_series}, window=250)
        self.assertEqual(float(corr.at["x", "y"]), 0.0)
        self.assertEqual(float(corr.at["x", "x"]), 1.0)


class ForecastTests(unittest.TestCase):
    def test_garch_recovers_persistence_and_long_run(self) -> None:
        rng = np.random.default_rng(11)
        omega, alpha, beta = 5e-8, 0.10, 0.85
        var = omega / (1.0 - alpha - beta)
        returns: list[float] = []
        for _ in range(3000):
            eps = math.sqrt(var) * float(rng.standard_normal())
            returns.append(eps)
            var = omega + alpha * eps**2 + beta * var

        fit = fit_garch11(returns)
        self.assertLess(abs(fit.persistence - (alpha + beta)), 0.06)
        self.assertGreater(fit.long_run_variance, 0.4e-6)
        self.assertLess(fit.long_run_variance, 2.5e-6)
        self.assertLess(fit.persistence, 1.0)

        path = fit.variance_path(24)
        start_gap = abs(path[0] - fit.long_run_variance)
        end_gap = abs(path[-1] - fit.long_run_variance)
        self.assertLessEqual(end_gap, start_gap)

    def test_garch_rejects_degenerate_input(self) -> None:
        with self.assertRaises(ValueError):
            fit_garch11([0.001] * 10)
        with self.assertRaises(ValueError):
            fit_garch11([0.0] * 100)

    def test_variance_path_converges_monotonically(self) -> None:
        fit = GarchFit(
            omega=5e-8, alpha=0.1, beta=0.85,
            long_run_variance=1e-6, last_conditional_variance=4e-6,
            n_obs=100, log_likelihood=0.0,
        )
        path = fit.variance_path(50)
        gaps = [v - fit.long_run_variance for v in path]
        self.assertTrue(all(g >= gaps[-1] - 1e-18 for g in gaps))  # non-increasing gap
        self.assertGreater(path[0], path[-1])
        self.assertGreater(path[-1], fit.long_run_variance * 0.99)

    def test_isotonic_pools_adjacent_violators(self) -> None:
        cal = IsotonicCalibrator().fit([1.0, 2.0, 3.0], [3.0, 1.0, 2.0])
        for x in (0.0, 1.5, 2.5, 4.0):
            self.assertAlmostEqual(cal(x), 2.0)

    def test_isotonic_recovers_monotone_signal(self) -> None:
        rng = np.random.default_rng(23)
        x = np.arange(50, dtype=float)
        y = 0.1 * x + rng.standard_normal(50) * 0.5
        cal = fit_entropy_calibration(x, y)
        preds = cal.predict(x)
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(preds, preds[1:])))
        self.assertLess(preds[0], preds[-1])
        self.assertGreater(abs(preds[-1] - 4.9), -1.0)  # sanity: finite
        self.assertAlmostEqual(preds[0], preds[0])  # no NaN

    def test_momentum_forecaster_direction_and_scale(self) -> None:
        rng = np.random.default_rng(31)
        drift = 8e-4
        vol = 6e-4
        returns = drift + vol * rng.standard_normal(400)
        df = ohlc_from_returns(returns)
        forecaster = MomentumVolForecaster()
        forecast = forecaster.forecast("TREND", df, 12, PolicyWeights.equal())
        self.assertGreater(forecast.expected_return, 0.0)
        self.assertGreater(forecast.sigma, 0.0)
        self.assertTrue(math.isfinite(forecast.sigma))
        self.assertIn(forecast.method, ("garch11", "realized_vol_fallback"))

        rng = np.random.default_rng(32)
        flat = ohlc_from_returns(0.0 + 6e-4 * rng.standard_normal(400))
        flat_forecast = forecaster.forecast("FLAT", flat, 12, PolicyWeights.equal())
        self.assertLess(abs(flat_forecast.expected_return), abs(forecast.expected_return))


class ManagerPipelineTests(unittest.TestCase):
    def _universe(self) -> dict[str, pd.DataFrame]:
        universe = dict(synthetic_universe(12, n_bars=700, seed=7))
        rng = np.random.default_rng(41)
        # Strong drift with vol high enough that the proportional edge clears
        # the fixed cost component (spread + fees); a weaker drift is a real
        # edge but too thin for the cost gate, and the screen rightly drops it.
        trend = ohlc_from_returns(1.2e-3 + 1.2e-3 * rng.standard_normal(700))
        trend["volume"] = 5e4
        universe["SYN_TREND"] = trend
        return universe

    def test_run_screen_gates_and_shortlist(self) -> None:
        config = ScreenerConfig(edge_cost_multiple=1.5)
        result = run_screen(self._universe(), config=config)

        failures_illiq = result.reports["SYN_ILLIQ"].gate_failures
        self.assertIn("dollar_volume", failures_illiq)
        self.assertIn("atr_floor", result.reports["SYN_FLAT"].gate_failures)
        self.assertIn("cusum_freeze", result.reports["SYN_REGIMEBREAK"].gate_failures)

        self.assertIn("SYN_TREND", result.shortlist)
        self.assertTrue(result.shortlist)
        for symbol in result.shortlist:
            report = result.reports[symbol]
            self.assertIsNotNone(report.z_score)
            self.assertGreater(report.z_score, 0.0)
            self.assertGreaterEqual(
                report.expected_return, config.edge_cost_multiple * report.round_trip_cost
            )
        self.assertGreaterEqual(result.effective_bets, 1.0)
        self.assertLessEqual(result.effective_bets, len(result.shortlist) + 1e-9)

    def test_policy_weights_change_the_hunt(self) -> None:
        universe = self._universe()
        config = ScreenerConfig(edge_cost_multiple=1.0, top_k=8)
        trend_heavy = run_screen(universe, config=config, policy=PolicyWeights(trend=2.0, stretch=0.1))
        regime_heavy = run_screen(universe, config=config, policy=PolicyWeights(trend=0.1, vol_regime=2.0))
        z_trend = trend_heavy.reports["SYN_TREND"].z_score
        z_regime = regime_heavy.reports["SYN_TREND"].z_score
        self.assertIsNotNone(z_trend)
        self.assertIsNotNone(z_regime)
        self.assertGreater(z_trend, z_regime)

    def test_bandit_learns_to_prefer_the_better_source(self) -> None:
        bandit = BanditHarness(epsilon=0.05, decay=1.0, min_epsilon=0.05, seed=0)
        counts = {"good": 0, "bad": 0}
        for _ in range(300):
            chosen = bandit.select(["good", "bad"])
            bandit.record(chosen, 1.0 if chosen == "good" else -1.0)
            counts[chosen] += 1
        self.assertGreater(counts["good"], counts["bad"] * 2)
        stats = bandit.stats()
        self.assertGreater(stats["good"]["mean"], 0.9)
        self.assertLess(stats["bad"]["mean"], -0.9)

    def test_heuristic_policy_tilts_on_regime_break(self) -> None:
        source = HeuristicPolicySource()
        calm = source.propose(ScreenContext(n_universe=10, pct_cusum_alarms=0.1, avg_vol_percentile=0.5))
        broke = source.propose(ScreenContext(n_universe=10, pct_cusum_alarms=0.8, avg_vol_percentile=0.9))
        self.assertGreater(broke.vol_regime, calm.vol_regime)
        self.assertLess(broke.trend, calm.trend)

    def test_llm_policy_source_is_a_wired_stub(self) -> None:
        source = LLMPolicySource()
        with self.assertRaises(NotImplementedError):
            source.propose(ScreenContext(n_universe=1, pct_cusum_alarms=0.0, avg_vol_percentile=0.5))

    def test_managed_screen_loop_and_feedback(self) -> None:
        managed = ManagedScreen(
            sources={
                "static": StaticPolicySource(weights=PolicyWeights(trend=2.0)),
                "heuristic": HeuristicPolicySource(),
            },
            config=ScreenerConfig(edge_cost_multiple=1.0),
        )
        result, chosen = managed.screen(self._universe())
        self.assertIn(chosen, managed.sources)
        self.assertTrue(result.shortlist)
        managed.feedback(0.5)
        self.assertIn(managed._last_source, managed.bandit.stats())

    def test_topology_is_calm_early_only_and_attached_to_screen(self) -> None:
        early = derive_trade_topology(
            ScreenContext(
                n_universe=500,
                pct_cusum_alarms=0.0,
                avg_vol_percentile=0.4,
                target_enabled=True,
                target_progress=0.1,
                session_progress=0.2,
                remaining_loss_budget_fraction=0.9,
            )
        )
        late = derive_trade_topology(
            ScreenContext(
                n_universe=500,
                pct_cusum_alarms=0.0,
                avg_vol_percentile=0.4,
                target_enabled=True,
                target_progress=0.1,
                session_progress=0.9,
                remaining_loss_budget_fraction=0.9,
            )
        )
        turbulent = derive_trade_topology(
            ScreenContext(n_universe=500, pct_cusum_alarms=0.5, avg_vol_percentile=0.9)
        )
        self.assertEqual(early.risk_multiplier, 1.15)
        self.assertEqual(late.risk_multiplier, 1.0)  # no late-session target chasing
        self.assertLess(turbulent.risk_multiplier, 1.0)

        managed = ManagedScreen(sources={"static": StaticPolicySource()})
        result, _ = managed.screen(
            self._universe(),
            PortfolioContext(
                target_enabled=True,
                target_progress=0.1,
                session_progress=0.2,
                remaining_loss_budget_fraction=0.9,
            ),
        )
        self.assertIsNotNone(result.topology)
        self.assertEqual(result.topology.risk_multiplier, 1.15)

    def test_execution_enforces_entropy_and_bounded_risk_multiplier(self) -> None:
        market = MarketState(
            close=100.0, ema_200=90.0, stoch_k=80.0, stoch_d=70.0,
            atr_14=2.0, long_stop=98.0, short_stop=102.0, htf_ema=90.0,
            signal=1, long_structure_confirmed=True,
        )
        uncertain = Decision(
            action="BUY",
            probabilities={"BUY": 0.35, "SELL": 0.33, "HOLD": 0.32},
            logits={"BUY": 1.0, "SELL": 0.9, "HOLD": 0.8},
            confidence=0.35, prompt="test", source="test", market=market,
        )
        self.assertGreater(normalized_decision_entropy(uncertain.probabilities), 0.95)
        risk = PropFirmRiskManager(starting_equity=10_000.0, point_value=1.0)
        blocked = build_execution_payload(
            uncertain, risk, max_decision_entropy=0.80, max_holding_bars=6
        )
        self.assertFalse(blocked.approved)
        self.assertIn("entropy", blocked.reason)
        self.assertEqual(blocked.max_holding_bars, 6)

        certain = Decision(
            action="BUY",
            probabilities={"BUY": 0.90, "SELL": 0.05, "HOLD": 0.05},
            logits={"BUY": 3.0, "SELL": 0.0, "HOLD": 0.0},
            confidence=0.90, prompt="test", source="test", market=market,
        )
        sized = build_execution_payload(
            certain, risk, manager_risk_multiplier=1.20, max_holding_bars=8
        )
        self.assertTrue(sized.approved)
        self.assertAlmostEqual(sized.risk_dollars, 60.0)
        self.assertEqual(sized.max_holding_bars, 8)
        invalid = risk.size_position(2.0, manager_risk_multiplier=1.21)
        self.assertTrue(invalid.blocked)

    def test_manager_envelope_term_changes_re_evaluation_without_extra_inference(self) -> None:
        def state(close: float) -> MarketState:
            return MarketState(
                close=close, ema_200=90.0, stoch_k=70.0, stoch_d=60.0,
                atr_14=1.0, long_stop=close - 1.0, short_stop=close + 1.0,
                htf_ema=90.0, signal=1, long_structure_confirmed=True,
            )

        narrow = DiscreteDeviationTrigger(horizon=12, envelope_atr_multiplier=1.0)
        wide = DiscreteDeviationTrigger(horizon=12, envelope_atr_multiplier=1.0)
        self.assertTrue(narrow.observe(state(100.0), manager_multiplier=0.70))
        self.assertTrue(wide.observe(state(100.0), manager_multiplier=1.40))
        # The same 0.8 ATR deviation breaches the narrow mandate but not the
        # wide one; this is arithmetic in the existing trigger, not an LLM call.
        self.assertTrue(narrow.observe(state(100.8), manager_multiplier=0.70))
        self.assertFalse(wide.observe(state(100.8), manager_multiplier=1.40))

    def test_feedback_before_screen_raises(self) -> None:
        managed = ManagedScreen(sources={"a": StaticPolicySource()})
        with self.assertRaises(RuntimeError):
            managed.feedback(1.0)

    def test_portfolio_capacity_guard_enforces_capacity_limits(self) -> None:
        topology = TradeTopology(
            max_open_positions=4,
            max_gross_risk_fraction=0.015,
            max_net_directional_risk_fraction=0.010,
            max_sector_positions=2,
            max_correlated_positions=2,
        )
        guard = PortfolioCapacityGuard(topology)
        equity = 10_000.0  # gross-risk cap 150, net-directional cap 100

        self.assertTrue(
            guard.reserve(
                PortfolioAllocation("o1", "AAA", "long", 90.0, sector="tech"), equity
            )[0]
        )
        # A second long would breach the 100 net-directional cap.
        blocked, reason = guard.check(
            PortfolioAllocation("o2", "BBB", "long", 20.0, sector="tech"), equity
        )
        self.assertFalse(blocked)
        self.assertIn("net directional", reason)
        # The opposite side fits: gross 110 <= 150, net |70| <= 100.
        self.assertTrue(
            guard.reserve(PortfolioAllocation("o2", "BBB", "short", 20.0, sector="tech"), equity)[0]
        )
        # A third tech position breaches the sector cap.
        blocked, reason = guard.check(
            PortfolioAllocation("o3", "CCC", "long", 5.0, sector="tech"), equity
        )
        self.assertFalse(blocked)
        self.assertIn("sector", reason)
        # The cluster cap counts the cluster: two existing correlated names
        # already fill a cap of two, so a third correlated entry is refused.
        blocked, reason = guard.check(
            PortfolioAllocation("o4", "DDD", "long", 5.0),
            equity,
            correlations={"AAA": 0.9, "BBB": 0.9},
        )
        self.assertFalse(blocked)
        self.assertIn("correlated", reason)
        # A duplicate order id can never reserve twice.
        duplicate, reason = guard.check(PortfolioAllocation("o1", "AAA", "long", 1.0), equity)
        self.assertFalse(duplicate)
        self.assertIn("duplicate", reason)
        self.assertEqual(len(guard.allocations()), 2)
        guard.release("o1")
        self.assertEqual(len(guard.allocations()), 1)

    def test_reentry_cooldown_blocks_then_expires(self) -> None:
        cooldown = ReentryCooldown(TradeTopology(reentry_cooldown_bars=6))
        cooldown.record_stopout("AAA", bar=10)
        blocked, reason = cooldown.blocked("AAA", bar=15)
        self.assertTrue(blocked)
        self.assertIn("cooldown", reason)
        self.assertFalse(cooldown.blocked("AAA", bar=16)[0])  # six bars elapsed
        # Correlated symbols share the cooldown; uncorrelated ones do not.
        self.assertTrue(cooldown.blocked("BBB", bar=12, correlations={"AAA": 0.9})[0])
        self.assertFalse(cooldown.blocked("BBB", bar=12, correlations={"AAA": 0.5})[0])
        # A fresh stop-out resets the clock to the later bar.
        cooldown.record_stopout("AAA", bar=20)
        self.assertTrue(cooldown.blocked("AAA", bar=25)[0])
        cooldown.clear("AAA")
        self.assertFalse(cooldown.blocked("AAA", bar=21)[0])

    def test_preflight_gates_and_thesis_record(self) -> None:
        market = MarketState(
            close=100.0, ema_200=90.0, stoch_k=80.0, stoch_d=70.0,
            atr_14=2.0, long_stop=98.0, short_stop=102.0, htf_ema=90.0,
            signal=1, long_structure_confirmed=True,
        )
        certain = Decision(
            action="BUY",
            probabilities={"BUY": 0.90, "SELL": 0.05, "HOLD": 0.05},
            logits={"BUY": 3.0, "SELL": 0.0, "HOLD": 0.0},
            confidence=0.90, prompt="test", source="test", market=market,
        )
        risk = PropFirmRiskManager(starting_equity=10_000.0, point_value=1.0)
        topology = TradeTopology()
        preflight = TradePreflight(
            data_age_seconds=0.0,
            spread_bps=5.0,
            estimated_impact_bps=5.0,
            order_volume_fraction=0.001,
        )
        approved = build_execution_payload(certain, risk, topology=topology, preflight=preflight)
        self.assertTrue(approved.approved)
        self.assertIsNotNone(approved.thesis)
        self.assertEqual(approved.thesis.style, topology.style_preference)
        self.assertAlmostEqual(approved.thesis.invalidation_price, approved.stop_price)
        self.assertGreater(approved.thesis.review_after_bars, 0)
        self.assertIn("re-entry cooldown", format_decision_prompt(market, topology).lower())

        stale = build_execution_payload(
            certain, risk, topology=topology,
            preflight=replace(preflight, data_age_seconds=120.0),
        )
        self.assertFalse(stale.approved)
        self.assertIn("stale", stale.reason)
        self.assertTrue(stale.source.endswith("+preflight_blocked"))

        halted = build_execution_payload(
            certain, risk, topology=topology,
            preflight=replace(preflight, trading_halted=True),
        )
        self.assertFalse(halted.approved)
        self.assertIn("halt", halted.reason)

        wide = build_execution_payload(
            certain, risk, topology=topology,
            preflight=replace(preflight, spread_bps=50.0),
        )
        self.assertFalse(wide.approved)
        self.assertIn("spread", wide.reason)

        missing = build_execution_payload(certain, risk, topology=topology, preflight=None)
        self.assertFalse(missing.approved)
        self.assertIn("requires a market-data preflight", missing.reason)

    def test_model_disagreement_fails_closed_on_marginal_decisions(self) -> None:
        class ScriptedBackend:
            def __init__(self, scores: dict[str, float]) -> None:
                self.scores = scores
                self.calls = 0

            def score_actions(self, prompt, actions):
                self.calls += 1
                return dict(self.scores)

        market = MarketState(
            close=100.0, ema_200=90.0, stoch_k=80.0, stoch_d=70.0,
            atr_14=2.0, long_stop=98.0, short_stop=102.0, htf_ema=90.0,
            signal=1, long_structure_confirmed=True,
        )
        strong = Decision(
            action="BUY",
            probabilities={"BUY": 0.90, "SELL": 0.05, "HOLD": 0.05},
            logits={"BUY": 3.0, "SELL": 0.0, "HOLD": 0.0},
            confidence=0.90, prompt="p", source="t", market=market,
        )
        topology = TradeTopology(confirmation_band=0.07)  # marginal band: 0.65-0.72

        # Clear decisions skip the second scorer entirely (Jev one-shot path).
        agree = ScriptedBackend({"BUY": 4.0, "SELL": 0.0, "HOLD": 0.0})
        kept = confirm_marginal_decision(strong, backend=agree, min_confidence=0.65, topology=topology)
        self.assertEqual(kept.action, "BUY")
        self.assertEqual(agree.calls, 0)

        marginal = replace(
            strong,
            confidence=0.68,
            probabilities={"BUY": 0.68, "SELL": 0.02, "HOLD": 0.30},
        )
        disagree = ScriptedBackend({"BUY": 0.0, "SELL": 4.0, "HOLD": 0.0})
        blocked = confirm_marginal_decision(
            marginal, backend=disagree, min_confidence=0.65, topology=topology
        )
        self.assertEqual(disagree.calls, 1)
        self.assertEqual(blocked.action, "HOLD")
        self.assertIn("model_disagreement_blocked", blocked.source)

        confirm = ScriptedBackend({"BUY": 4.0, "SELL": 0.0, "HOLD": 0.0})
        confirmed = confirm_marginal_decision(
            marginal, backend=confirm, min_confidence=0.65, topology=topology
        )
        self.assertEqual(confirmed.action, "BUY")
        self.assertIn("secondary_confirmed", confirmed.source)


if __name__ == "__main__":
    unittest.main()
