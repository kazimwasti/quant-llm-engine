# Quant futures Kaggle data kit

Checked against public product and rule pages in October 2026. This pack is a research starter, not a price-history dataset and not trading advice.

## What it contains

- `instrument_universe.csv`: products shown on both firms' published approved-product lists, plus XRP futures marked as not currently listed by either firm.
- `prop_rule_sources.csv`: the specific rule pages to check and the operational constraints they describe.
- `data_requirements.csv`: the bar, contract, cost, and calendar fields needed for a defensible multi-market backtest.
- `data_source_catalog.csv`: primary/licensed sources to investigate for actual historical files.
- `strategy_metrics.json`: exact target definitions and cautions.
- `audit_kaggle_inputs.py`: scans attached Kaggle inputs for OHLCV schema problems, invalid bars, duplicates, gaps, and date coverage.

It does **not** contain historical candles. Those are vendor data and must be obtained under the applicable license. Attaching this kit alone will not increase the number of backtest trades.

## Instrument coverage

The first comparison universe is the intersection of the current public Topstep and Lucid product lists. It includes equity index, FX, energy, metals, and agricultural futures. XRP/Micro XRP are CME products, but they do not appear on the firms' current published approved-product lists. Treat XRP as research-only unless the firm's live platform/product list and your specific account rules say otherwise.

The product matrix is not a guarantee that every product is enabled for every plan, account phase, platform, or day. Verify the exact account before trading. Contract multipliers, tick values, trading hours, margin, and fees belong in an instrument reference table and must be current.

## Why the last run was long-only

The 44-trade diagnostic came from the `feature_fallback` backend and all its trades were BUY. That is a signal-generation/configuration issue, not something that position sizing or more fine-tuning fixes. The model's historical label set was also imbalanced (367 BUY, 182 SELL, 5,983 HOLD). Before fitting another adapter, audit the rule that emits short signals, confirm the feature conditions are symmetric, count raw long/short candidates by symbol and session, and test the short rule independently. Risk sizing scales outcomes; it cannot create a missing short edge.

## Data acquisition plan

Start with the common-product universe rather than every listed instrument at once:

1. Obtain at least 3 years, preferably 5+ years, of 5-minute OHLCV per product, with individual contract identifiers and a documented continuous-contract roll map. More years are useful only if contract rolls, sessions, and data quality are handled correctly.
2. Include 1-minute bars if affordable; they improve intrabar stop/target ordering. Tick/quote data is better for execution realism but is a separate, larger cost.
3. Include exchange calendars, daylight-saving-aware timestamps, contract expiries/roll dates, and per-firm fee/commission schedules.
4. Use a development period for feature/model/threshold selection, rolling walk-forward folds for iteration, and a final untouched period used once.
5. Add a vendor's licensed files to a Kaggle Dataset, attach that dataset to the notebook, and run `audit_kaggle_inputs.py`. Keep vendor terms and redistribution restrictions in mind.

Recommended sources are listed in `data_source_catalog.csv`. CME DataMine is the first-party source. Databento and Barchart are commercial alternatives. Free public data can be useful for code smoke tests, but should not be assumed to provide complete, accurate, years-long intraday futures history.

## Prop-firm compliance design

Do not encode a single generic “100k funded account” rule. A simulator needs a profile for each firm, plan, phase (evaluation/sim funded/live), platform, account size, and payout state. Enforce each profile before order submission and on every equity update:

- product allowlist and max contracts / scaling limits
- trailing or end-of-day loss limit, including whether unrealized P&L counts
- daily loss limit, reset time and timezone
- session cutoff and mandatory flat time
- news/event restrictions, if the plan has them
- payout/consistency rules
- whether automation/API/copier use is permitted for that exact phase/platform
- fees, slippage, and any exchange/broker restrictions

For example, Topstep's API article says ProjectX API automation is for simulated Combine/XFA accounts under normal rules, not Live Funded Accounts; it also describes restrictions on remote order routing. Lucid's public materials allow automated strategies subject to its rules, but time cutoffs and account details vary. Re-check linked rules before using any live or funded account. This kit does not connect to a broker or place trades.

## How to read the targets

At exactly +2.2R per winner and -1R per loser, 70% wins implies gross expectancy 1.24R and gross profit factor 5.13. These are identities, not forecasts. Fees, slippage, stop gaps, time exits, partial exits, and breakeven exits change realized outcomes. The 44-trade diagnostic had 31.82% wins, 0.174R expectancy, PF 1.388, and all long trades; it does not support those targets. Its short duration also makes the annualized Sharpe and drawdown estimates unreliable.

Measure performance from realized net P&L after costs. Report win rate, actual average winner/loser, EV in R, profit factor, daily net-return Sharpe, maximum drawdown, trade count, long/short breakdown, and uncertainty intervals. Keep risk per trade/account-rule compliance separate from signal quality: lowering risk changes dollar drawdown, but it does not improve win rate, EV in R, or PF.
