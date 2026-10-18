# 🚀 Quant LLM Engine (`quant-llm-engine`)

An institutional-grade algorithmic trading and backtesting engine driven by fine-tuned Large Language Models (Qwen LoRA adapters) combined with strict risk management controls and multi-timeframe structural gates.

---

## 📌 Project Overview

**Quant LLM Engine** bridges rule-based quantitative strategy filters with generative AI decision engines. Rather than relying solely on traditional technical indicators, the framework formats intraday price action, indicator confluence, and high-timeframe (HTF) trend states into dynamic structured prompts evaluated by fine-tuned Qwen 2.5 LoRA adapters.

The system features robust risk execution parameters engineered specifically to meet prop firm rules (e.g., TopstepX) and institutional execution standards.

---

## 🛠️ Key Architectural Features

- **Asymmetric Risk-to-Reward (1 : 2.2 R:R):** Executes trades with tight stop losses ($1.0\times\text{ATR}$) and high target payouts ($2.2\times\text{ATR}$) to maximize statistical edge.
- **Multi-Timeframe Trend Gate:** Requires 1-hour trend alignment before permitting 5-minute intraday execution triggers.
- **Dynamic Rejection & Pullback Entry Filter:** Eliminates low-probability breakout entries by requiring localized candle shadow rejections or structural support/resistance sweeps.
- **Trailing Breakeven Ratchet:** Locks in risk-free position management once price achieves target profit thresholds.
- **Session Circuit Breaker:** Hard daily drawdown cap ($-\$500$) and loss-streak triggers to protect portfolio capital during adverse market regimes.
- **Hybrid Execution Engine:** Features a full rule-based feature fallback mode alongside LLM-driven inference via QLoRA fine-tuning.

---

## 📂 Repository Structure

```text
quant-llm-engine/
├── core/                       # Core execution, risk, and strategy modules
│   ├── engine.py               # Main trade evaluation loop & decision state
│   ├── risk.py                 # Circuit breakers, position sizing, & trailing stops
│   ├── strategy.py             # Feature engineering & fallback scoring logic
│   └── execution.py            # Order routing & trade fill simulation
├── data/                       # Market historical data & generated datasets
│   ├── 5m.csv                  # 5-minute price action candles
│   ├── hourly.csv              # High-timeframe trend context
│   └── jev_train_5m.jsonl      # Formatted prompt dataset for LoRA fine-tuning
├── models/                     # Tokenizers, corpus builders, & fine-tuned adapters
│   ├── bpe.py / corpus.py      # Custom tokenization utilities
│   └── qwen_adapter/           # Trained QLoRA adapter weights (ignored in git)
├── scripts/                    # Utility, data, and training execution scripts
│   ├── dataset_generator.py   # Synthesizes JSONL instruction datasets
│   ├── download_data.py        # Historical data fetching pipelines
│   ├── run_screen.py           # Tier 1/2 screen demo CLI (see Screening pipeline)
│   └── train_lora.py           # PEFT / QLoRA 4-bit fine-tuning script
└── tests/                      # Offline operational-safety acceptance tests
```

## Screening pipeline (`core/screener.py`, `core/forecast.py`, `core/manager.py`)

A deterministic three-tier funnel that decides *which* symbols deserve LLM
inference, so sub-agent models never see a name that cheap math has not
already cleared:

1. **Tier 1 (closed-form math)** — liquidity/dollar-volume gates, Parkinson
   and Garman-Klass volatility, regression-t-statistic trend strength,
   rolling z-score stretch, volume shocks, and a scale-invariant CUSUM
   regime-break alarm (a firing alarm freezes a symbol until its forecasts
   are re-estimated).
2. **Tier 2 (numeric models)** — GARCH(1,1) volatility forecasting (QMLE via
   a built-in Nelder-Mead, no scipy dependency), a cheap deterministic
   E[X] forecaster over Tier-1 features, and isotonic calibration for
   mapping decision entropy onto realized error before anything trusts it
   as a variance input.
3. **Cost gate + Z-rank selection** — expected edge must clear the
   round-trip cost (spread + fees + square-root impact) by a configurable
   multiple; survivors are ranked by Z = (E − cost)/σ and selected greedily
   with a correlation penalty and pairwise-correlation cap, then scored by
   effective independent bets (N_eff) so ten correlated picks are not
   mistaken for ten bets.

The LLM-shaped slot is `PolicySource.propose` in `core/manager.py`: the
manager model receives a numeric context summary and returns four
`PolicyWeights` — numbers, never orders. Every policy source is scored by an
epsilon-greedy `BanditHarness` against realized rewards, so an LLM manager
must beat the deterministic baselines on outcomes to keep influence. Wire a
fine-tuned adapter in by subclassing `LLMPolicySource`.

Run the offline screen demo (no network, seeded synthetic universe):

```powershell
python scripts/run_screen.py --synthetic-only
python -m unittest tests.test_screening -v
```

## Operational test readiness

`core.operations.PaperTradingSafetyController` is the broker-facing safety
boundary for the Phase F operational tests. It starts disconnected, permits
paper trading only, persists order IDs and session safety locks across restarts,
and requires a manual reset after account reconciliation.

Run the offline operational suite before a paper-trading deployment:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Copy `.env.example` to `.env` only on the paper-trading host and keep real
credentials out of Git. See `OPERATIONS_TEST_PLAN.md` for the practice-account
restart, disconnect, flatten, and risk-limit checklist.

## Conservative cloud backtests

Use matching, completed 5-minute and hourly histories from the same instrument
and date range. The hourly EMA is made available only after that hourly bar
closes, decisions fill at the next 5-minute open with adverse slippage, and a
bar that touches both stop and target is charged to the stop by default.

```powershell
python backtest/backtest.py --feature-fallback `
  --input data/5m.csv --hourly-input data/hourly.csv `
  --slippage-points 0.25 --commission-per-contract 2.50 `
  --ambiguous-fill-policy conservative --output-prefix results/cloud-baseline
```

Use `--ambiguous-fill-policy probabilistic` only for sensitivity analysis, not
as the primary risk estimate. Compare configurations on chronological,
untouched periods; do not select parameters from the same period used to report
performance.

### Generate and train the Qwen classifier

Build the JSONL from matching 5-minute/hourly files. The generator requires
200 hourly warmup bars and caps directional labels at 40% of the output while
retaining every HOLD row:

```powershell
python scripts/dataset_generator.py --input data/5m_cloud.csv --hourly-input data/hourly_cloud.csv --output data/jev_train_5m.jsonl
python scripts/train_lora.py --data data/jev_train_5m.jsonl --output-dir models/qwen_adapter
```

The generator labels BUY/SELL only when a matching strategy signal and regime
gates are active and the 2.2x ATR target is reached before the 1.0x ATR stop;
ambiguous same-bar touches resolve stop-first. QLoRA evaluation uses the final
chronological segment with a purge at least as large as the label horizon, and
logs validation cross-entropy plus per-class precision/recall/support. These
are research diagnostics, not guarantees of live or out-of-sample performance.

### Prepare and compare a matched cloud run

The checked-in `data/5m.csv` and `data/hourly.csv` are not a matched pair.
Create fresh, disposable inputs instead (Yahoo Finance currently limits the
5-minute interval to recent history; hourly history includes 180 calendar days
of warmup):

```powershell
python scripts/prepare_cloud_data.py --symbol MNQ=F --test-days 50
```

Review the provider, symbol, timestamps, and date coverage before using the
files. The downloader requires at least 200 hourly bars before the first 5m
bar and writes timestamps in UTC. Provider data can be revised and is not a
substitute for exchange-quality historical data.

Run baseline and adapter with identical inputs, costs, and fill policy. The
adapter command requires a locally available `models/qwen_adapter`; no adapter
weights are included in this repository:

```powershell
python backtest/backtest.py --feature-fallback --input data/5m_cloud.csv --hourly-input data/hourly_cloud.csv --slippage-points 0.25 --commission-per-contract 2.50 --ambiguous-fill-policy conservative --output-prefix results/rule_baseline
python backtest/backtest.py --adapter-dir models/qwen_adapter --input data/5m_cloud.csv --hourly-input data/hourly_cloud.csv --slippage-points 0.25 --commission-per-contract 2.50 --ambiguous-fill-policy conservative --output-prefix results/qwen_adapter
python scripts/compare_backtests.py --run baseline=results/rule_baseline --run adapter=results/qwen_adapter --output results/comparison.csv
```

The comparison reports net P&L, drawdown, profit factor, trade count, win rate,
and longest losing streak. It summarizes outputs; it does not establish an
out-of-sample result. Freeze a chronological test window before comparing,
keep it untouched during tuning, and rerun both models at higher slippage to
measure cost sensitivity. The current adapter is loaded directly by PEFT; this
repo does not implement a vLLM endpoint client. Do not start vLLM for this
backtester.

Do not upload the existing CSVs as a pair, and do not upload credentials or
unreviewed data to a cloud volume. This repository does not provision a
Runpod Pod, private remote, network volume, or account credentials. Create
those in your own account after reviewing the inputs and expected spend.
