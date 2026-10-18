"""Tier 2 numeric models: volatility forecasting, calibration, cheap edge scores.

These run without the LLM. The GARCH forecaster produces the sigma that the
Z-rank consumes; the isotonic calibrator maps a raw confidence signal (for
example decision entropy from the re-eval envelope) onto realized error so
that uncertainty estimates are calibrated before anything trusts them.

Components
----------
- ``fit_garch11``: GARCH(1,1) via Gaussian QMLE. The optimizer is a compact
  Nelder-Mead implemented here so the package keeps its zero-dependency
  footprint beyond pandas/numpy.
- ``GarchFit.variance_path``: h-step ahead conditional variance, converging
  to the long-run variance at rate (alpha + beta).
- ``IsotonicCalibrator``: pool-adjacent-violators monotone regression. Used
  to turn uncalibrated model confidence into a defensible sigma input.
- ``MomentumVolForecaster``: the cheap E[X] generator for stage-1 screening.
  Deterministic arithmetic on Tier-1 features, steered by ``PolicyWeights``
  so the manager layer (LLM or bandit) can modulate what the screen hunts
  without touching any formula.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from .screener import latest_features


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GarchFit:
    """Fitted GARCH(1,1) with h-step variance forecasting."""

    omega: float
    alpha: float
    beta: float
    long_run_variance: float
    last_conditional_variance: float
    n_obs: int
    log_likelihood: float
    unconstrained: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    def variance_path(self, horizon: int) -> list[float]:
        """Conditional variance forecasts for bars T+1 .. T+horizon.

        v_1 = omega + alpha * eps_T^2 + beta * sigma_T^2, then
        v_h = v_long + (alpha + beta)^(h-1) * (v_1 - v_long).
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        phi = self.persistence
        v1 = self.last_conditional_variance
        return [
            self.long_run_variance + (phi ** step) * (v1 - self.long_run_variance)
            for step in range(horizon)
        ]

    def volatility_path(self, horizon: int) -> list[float]:
        return [math.sqrt(v) for v in self.variance_path(horizon)]


def _nelder_mead(
    objective,
    x0: Sequence[float],
    step: float = 0.5,
    max_iter: int = 300,
    x_tol: float = 1e-8,
    f_tol: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Compact Nelder-Mead minimizer over unconstrained coordinates."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float)
        point[i] += step if point[i] == 0.0 else 0.35 * abs(point[i])
        simplex.append(point)
    values = [objective(x) for x in simplex]

    for _ in range(max_iter):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        if (
            max(abs(simplex[i] - simplex[0]).max() for i in range(1, n + 1)) < x_tol
            and abs(values[-1] - values[0]) < f_tol
        ):
            break

        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1])
        f_reflected = objective(reflected)

        if values[0] <= f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
            continue
        if f_reflected < values[0]:
            expanded = centroid + 2.0 * (reflected - centroid)
            f_expanded = objective(expanded)
            if f_expanded < f_reflected:
                simplex[-1], values[-1] = expanded, f_expanded
            else:
                simplex[-1], values[-1] = reflected, f_reflected
            continue

        if f_reflected < values[-1]:
            contracted = centroid + 0.5 * (reflected - centroid)
        else:
            contracted = centroid + 0.5 * (simplex[-1] - centroid)
        f_contracted = objective(contracted)
        if f_contracted < min(values[-1], f_reflected):
            simplex[-1], values[-1] = contracted, f_contracted
            continue
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
            values[i] = objective(simplex[i])

    best = int(np.argmin(values))
    return simplex[best], values[best]


def _unconstrain(p: np.ndarray) -> tuple[float, float, float]:
    """Map unconstrained coordinates to (omega, alpha, beta) with alpha+beta<1."""
    omega = math.exp(float(p[0]))
    a = math.exp(float(p[1]))
    b = math.exp(float(p[2]))
    z = 1.0 + a + b
    return omega, a / z, b / z


def _garch_nll(params: np.ndarray, eps2: np.ndarray, var_init: float) -> float:
    omega, alpha, beta = _unconstrain(params)
    total = 0.0
    var = max(var_init, omega)
    for e2 in eps2:
        # Evaluate eps_t under var_t (built from information through t-1),
        # then update the recursion for t+1.
        if var <= 1e-18 or not math.isfinite(var):
            return 1e300
        total += math.log(var) + e2 / var
        var = omega + alpha * e2 + beta * var
    return total


def fit_garch11(
    returns: Sequence[float],
    max_iter: int = 300,
    warm_start: Optional[Sequence[float]] = None,
) -> GarchFit:
    """Fit GARCH(1,1) by Gaussian QMLE on per-bar returns.

    ``warm_start`` accepts the ``unconstrained`` vector of a previous fit —
    for overlapping windows the optimum barely moves, so starting there cuts
    iterations sharply. Raises ValueError for degenerate input (too few
    observations or zero variance); callers fall back to realized vol there.
    """
    eps = np.asarray(returns, dtype=float)
    eps = eps[np.isfinite(eps)]
    if eps.size < 30:
        raise ValueError("GARCH fit needs at least 30 returns")
    sample_var = float(np.var(eps))
    if sample_var <= 1e-18:
        raise ValueError("returns have zero variance; nothing to model")

    # Init at persistence 0.90, alpha/beta split 0.08/0.82, omega pinned to
    # make the long-run variance match the sample variance.
    long_run = sample_var
    alpha0, beta0 = 0.08, 0.82
    omega0 = long_run * (1.0 - alpha0 - beta0)
    if warm_start is not None and len(warm_start) == 3:
        p0 = np.array(warm_start, dtype=float)
    else:
        p0 = np.array([math.log(omega0), math.log(0.8), math.log(8.2)])

    eps2 = eps**2
    best, nll = _nelder_mead(lambda p: _garch_nll(p, eps2, sample_var), p0, max_iter=max_iter)
    omega, alpha, beta = _unconstrain(best)

    var = sample_var
    for e2 in eps2:
        var = omega + alpha * e2 + beta * var
    # After consuming every observation this is the T+1 variance forecast.
    last_var = var

    return GarchFit(
        omega=omega,
        alpha=alpha,
        beta=beta,
        long_run_variance=omega / max(1.0 - alpha - beta, 1e-12),
        last_conditional_variance=last_var,
        n_obs=int(eps.size),
        log_likelihood=-nll,
        unconstrained=(float(best[0]), float(best[1]), float(best[2])),
    )


# ---------------------------------------------------------------------------
# Isotonic calibration (PAVA)
# ---------------------------------------------------------------------------

class IsotonicCalibrator:
    """Monotone non-decreasing fit of y on x via pool-adjacent-violators.

    Used to map a raw confidence/entropy signal onto realized outcomes:
    bucket historical calls by entropy, check whether higher entropy really
    meant larger error, and reuse the fitted step function as a defensible
    sigma estimate. An uncalibrated signal is worse than no signal once it
    feeds a Z-rank, so this gate runs before anything else trusts it.
    """

    def __init__(self) -> None:
        self._breakpoints: list[float] = []
        self._values: list[float] = []
        self._fitted = False

    def fit(self, x: Sequence[float], y: Sequence[float]) -> "IsotonicCalibrator":
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) < 2:
            raise ValueError("need at least two points")
        pairs = sorted((float(xi), float(yi)) for xi, yi in zip(x, y))

        block_x: list[float] = []  # max x in block
        block_sum: list[float] = []
        block_n: list[int] = []

        def push(xi: float, yi: float) -> None:
            block_x.append(xi)
            block_sum.append(yi)
            block_n.append(1)
            while len(block_x) >= 2 and _block_value(-2) > _block_value(-1):
                block_sum[-2] += block_sum[-1]
                block_n[-2] += block_n[-1]
                block_x.pop()
                block_sum.pop()
                block_n.pop()

        def _block_value(index: int) -> float:
            return block_sum[index] / block_n[index]

        for xi, yi in pairs:
            push(xi, yi)

        self._breakpoints = block_x
        self._values = [s / n for s, n in zip(block_sum, block_n)]
        self._fitted = True
        return self

    def __call__(self, x: float) -> float:
        if not self._fitted:
            raise RuntimeError("calibrator must be fit before use")
        x = float(x)
        if x <= self._breakpoints[0]:
            return self._values[0]
        if x >= self._breakpoints[-1]:
            return self._values[-1]
        index = bisect.bisect_left(self._breakpoints, x)
        return self._values[index]

    def predict(self, xs: Sequence[float]) -> list[float]:
        return [self(x) for x in xs]


def fit_entropy_calibration(
    entropy: Sequence[float], realized_error: Sequence[float]
) -> IsotonicCalibrator:
    """Calibrate decision entropy against realized absolute forecast error.

    Returns a monotone map entropy -> expected error. If the fitted curve is
    flat (error does not increase with entropy), the signal is uninformative
    and should not be trusted as a variance input.
    """
    return IsotonicCalibrator().fit(entropy, realized_error)


# ---------------------------------------------------------------------------
# Cheap E[X] forecaster
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyWeights:
    """Weights over normalized Tier-1 features; the manager layer proposes these.

    The forecaster multiplies each normalized feature (clipped to [-1, 1]) by
    its weight and scales the composite by horizon sigma, so a weight vector
    changes *what the screen hunts* without changing any formula.
    """

    trend: float = 1.0
    stretch: float = 0.5
    shock: float = 0.1
    vol_regime: float = 0.25

    @staticmethod
    def equal() -> "PolicyWeights":
        return PolicyWeights()

    def normalized(self) -> "PolicyWeights":
        total = self.trend + self.stretch + self.shock + self.vol_regime
        if total <= 0:
            raise ValueError("policy weights must not all be zero")
        return PolicyWeights(
            trend=self.trend / total,
            stretch=self.stretch / total,
            shock=self.shock / total,
            vol_regime=self.vol_regime / total,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "trend": self.trend,
            "stretch": self.stretch,
            "shock": self.shock,
            "vol_regime": self.vol_regime,
        }


@dataclass(frozen=True)
class Forecast:
    """Stage-1 expected edge and uncertainty for one symbol, in return space."""

    symbol: str
    expected_return: float
    sigma: float
    method: str
    features: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)


class Forecaster(Protocol):
    """Anything that turns bars + policy into (E[X], sigma)."""

    def forecast(
        self, symbol: str, df: pd.DataFrame, horizon_bars: int, policy: PolicyWeights
    ) -> Forecast:
        ...


@dataclass(frozen=True)
class MomentumVolForecaster:
    """Deterministic E[X] from Tier-1 features; sigma from GARCH(1,1).

    E = composite(features, policy) * sigma_horizon, which makes the Z-rank
    reduce to composite - cost/sigma: a high score only survives when the
    edge clears transaction cost in units of uncertainty. A LightGBM ranker
    can slot in behind the same ``Forecaster`` protocol later without any
    change to the screening pipeline.
    """

    max_returns: int = 1500
    feature_window: int = 250

    def forecast(
        self, symbol: str, df: pd.DataFrame, horizon_bars: int, policy: PolicyWeights
    ) -> Forecast:
        if horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        weights = policy.normalized()
        feats = latest_features(df)

        def clip1(value: float, scale: float) -> float:
            if not math.isfinite(value):
                return 0.0
            return max(-1.0, min(1.0, value / scale))

        n_trend = clip1(feats["trend_tstat"], 5.0)
        n_stretch = clip1(feats["stretch_z"], 3.0)
        n_shock = feats["volume_shock"] * 2.0 - 1.0
        n_regime = feats["vol_percentile"] * 2.0 - 1.0

        composite = (
            weights.trend * n_trend
            + weights.stretch * n_stretch
            + weights.shock * n_shock
            + weights.vol_regime * n_regime
        )

        returns = df["close"].astype(float).pct_change().dropna().tail(self.max_returns)
        diagnostics: dict[str, float] = {}
        try:
            fit = fit_garch11(returns.tolist())
            var_h = fit.variance_path(horizon_bars)[-1]
            sigma_h = math.sqrt(var_h)
            diagnostics = {
                "garch_omega": fit.omega,
                "garch_alpha": fit.alpha,
                "garch_beta": fit.beta,
                "garch_persistence": fit.persistence,
            }
            method = "garch11"
        except ValueError:
            realized = float(np.std(returns.to_numpy())) if len(returns) >= 2 else float("nan")
            sigma_h = realized * math.sqrt(horizon_bars)
            method = "realized_vol_fallback"

        expected_return = composite * sigma_h
        return Forecast(
            symbol=symbol,
            expected_return=float(expected_return),
            sigma=float(sigma_h),
            method=method,
            features=feats,
            diagnostics=diagnostics,
        )
