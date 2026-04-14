# Strategy 1: OBI + Microprice Timing vs TWAP

## Overview

This project implements a limit-order-book timing strategy, with strict trading constraint:
- trade exactly `1 share per minute` per stock
- benchmark against a standard `TWAP` that trades at the first tick of each minute

The strategy uses top-5 limit order book data and order-event information to decide **when within each minute** to buy or sell. It does not change trade size. It only changes timing.

## Files

- [data_preprocessing.py]: load real LOB data and compute features
- [strategy.py]: timing logic, execution rules, and weight optimizer
- [backtest.py]: performance metrics, summary tables, and plots
- [main.py]: end-to-end pipeline
- [output/optimise_run]: saved summary tables and plots from an optimized run

## Data And Preprocessing

The code reads real training data from files named:
- `AMZN_5levels_train.csv`
- `GOOG_5levels_train.csv`
- `INTC_5levels_train.csv`
- `MSFT_5levels_train.csv`

Raw column names are preserved exactly.

Main preprocessing steps:
- parse `Time` using `%H:%M:%S.%f`
- create `minute` buckets for the 1-share-per-minute constraint
- create `row_in_minute`
- compute `log_mid` and `log_ret`
- remove trading halts if present
- drop invalid quotes where best ask is not above best bid

## Feature Set

The strategy uses a mix of quote, depth, order-flow, and short-horizon momentum features.

### Core microstructure features

- `Spread`: raw bid-ask spread from the dataset
- `MidPrice`: raw midpoint from the dataset
- `spread_bps`: spread scaled by midpoint
- `microprice`: size-weighted price using best bid/ask
- `micro_minus_mid`: microprice minus midpoint
- `micro_norm`: normalized microprice deviation

### Depth and imbalance features

- `bid_depth`: total bid size across 5 levels
- `ask_depth`: total ask size across 5 levels
- `depth_ratio`: bid depth divided by ask depth
- `obi`: 5-level order book imbalance
- `obi_l1`: level-1 imbalance only
- `obi_l2`: imbalance using levels 1 and 2

### Shape features

- `bid_slope`: slope of bid prices across levels
- `ask_slope`: slope of ask prices across levels

### Momentum and volatility features

- `momentum_5s`
- `momentum_15s`
- `momentum_30s`
- `realised_vol_30`

These are row-based differences, used as short-horizon proxies for intraminute price pressure.

### Order-flow features

- `exec_flag`: visible or hidden execution
- `cancel_flag`: partial or full cancellation
- `buy_exec`
- `sell_exec`
- `buy_pressure`: rolling count of buy executions
- `sell_pressure`: rolling count of sell executions
- `order_flow_imbalance`

### Sell-specific features

To avoid treating sells as a mirror image of buys, the current version adds:
- `bid_momentum_5s`: change in best bid over 5 rows
- `bid_momentum_norm_5s`: best-bid change normalized by spread
- `micro_trend_5s`: change in `micro_norm`
- `obi_change_5s`: change in `obi`

These are meant to detect whether upward pressure is still building or starting to fade.

## Scoring Logic

### Buy score

The buy side uses a composite pressure score:

`score = w_obi * obi + w_micro * micro_norm + w_momentum * signed_log_momentum + w_ofi * order_flow_imbalance`

Interpretation:
- positive score means upward pressure
- negative score means downward pressure

For buys, downward pressure is desirable because the strategy wants to buy cheaper, so:
- lower score is better for buy timing

### Sell score

The sell side uses a separate score, not just the negative of the buy score.

The sell score rewards:
- positive `obi`
- positive `micro_norm`
- positive `order_flow_imbalance`
- positive normalized best-bid momentum
- signs of pressure fading after strength, using `micro_trend_5s` and `obi_change_5s`

The sell score penalizes:
- overextended upward chasing through a `sell_chase_penalty`

This is designed to prefer selling when the book is still strong, but not at the most overextended point.

## Execution Logic

The strategy always trades exactly one share per complete minute.

### Benchmark

TWAP benchmark:
- buy at the first tick of the minute using `AskPrice_1`
- sell at the first tick of the minute using `BidPrice_1`

### Strategy execution

For each minute:
1. compute the relevant side-specific score at every tick
2. optionally neutralize score on extreme-momentum ticks using a volatility filter
3. define a percentile threshold inside that minute
4. choose the best candidate before the deadline window
5. if needed, force a trade near the end of the minute

### Buy execution rule

- identify the most negative score region in the minute
- prefer ticks with no concurrent execution
- if no early candidate exists, use the best available tick in the deadline window

### Sell execution rule

- identify the most positive score region in the minute
- apply a stricter structural filter before accepting a sell candidate

The current structural filter prefers sell ticks where:
- `micro_norm` is not negative
- `obi` is not negative
- `micro_trend_5s <= 0`, meaning pressure is no longer accelerating upward
- normalized best-bid momentum is not in the most overextended range inside that minute

If no such candidate exists:
- relax to any signal tick
- if still nothing qualifies, execute in the deadline window

### Fill assumptions

- buy fills at `AskPrice_1`
- sell fills at `BidPrice_1`

This is a simple market-order fill model.

## Weight Optimizer

The optimizer lives in [strategy.py](/Users/f./Downloads/algo_trading/strategy.py).

It performs a coarse grid search over:
- `w_obi`
- `w_micro`
- `w_momentum`
- `w_ofi`

### Buy-side objective

For buys, the optimizer selects weights that maximize average realized improvement over TWAP in basis points.

Buy improvement per trade:
- `TWAP price - strategy price`

Higher is better.

### Sell-side objective

For sells, the optimizer uses a separate training objective.

It still simulates one trade per minute, but scores candidate weights using:
- future best-bid advantage over a fixed horizon, plus
- a smaller anchor to realized sell-vs-TWAP improvement

Intuition:
- a good sell should happen before the best bid weakens
- future bid behavior is a better sell-side training target than a symmetric midpoint-based target

The sell optimizer therefore tries to find weights that favor:
- strong but not overextended bid conditions
- earlier liquidation before bid decay

## Performance Measurement

Backtesting is handled in [backtest.py](/Users/f./Downloads/algo_trading/backtest.py).

Main metrics:
- `n_trades`
- `avg_strategy_price`
- `avg_twap_price`
- `improvement_per_share`
- `improvement_bps`
- `% Better`
- `% Forced`
- `total_improvement`
- `std_per_trade`
- `sharpe_improvement`

Interpretation:
- for buys, lower execution price is better
- for sells, higher execution price is better

## Running The Pipeline

Run the full pipeline with:

```bash
python main.py --data_dir . --out_dir output/optimise_run
```

Run with weight optimization:

```bash
python main.py --data_dir . --optimise --out_dir output/optimise_run
```

This will:
- load the real CSVs
- compute features
- split train/test chronologically
- run the strategy
- compare against TWAP
- save summary tables and labelled plots

## Current Limitations

- momentum features are row-based, not exact event-time windows
- fill model assumes immediate execution at best quote
- no transaction fees or queue-position model
- optimizer is coarse grid search, so it is slow and not globally optimal
- sell-side tuning improved the framework, but sell performance is still less stable than buy performance

## Summary

This strategy is a timing layer on top of a fixed-volume schedule. It does not try to predict long-horizon returns. Instead, it uses short-horizon limit-order-book signals to improve intraminute execution relative to TWAP.

Current version highlights:
- real-data preprocessing
- OBI + microprice + momentum + OFI timing
- separate sell-side scoring
- separate sell-side optimizer objective based on future best-bid behavior
- exactly one share per minute per ticker
