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
