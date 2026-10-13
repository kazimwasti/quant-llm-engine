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
│   └── train_lora.py           # PEFT / QLoRA 4-bit fine-tuning script
└── tests/                      # Offline operational-safety acceptance tests
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
