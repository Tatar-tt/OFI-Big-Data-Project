# Pre-registration: Order Flow Imbalance at Scale (DES431, Group 11)

Frozen on 12 Nov 2026, before any model was fit on real data. Changes after this commit must be listed under "Deviations" with the reason.

## Data
- Binance USDⓈ-M perpetual futures, `bookTicker` + `aggTrades` from data.binance.vision
- Symbols: LINKUSDT, DOGEUSDT
- Train: 2024-01 (UTC). Test: 2024-02 (UTC). The test month is evaluated once.
- Scope was cut from 4 symbols × 10 months to 2 symbols × 2 months because of the $50 AWS Learner Lab budget and the timeline.

## Features (1-second bars)
- OFI per bookTicker update following Cont, Kukanov & Stoikov (2014), summed per second.
- `ofi_ws` = sum over the trailing w seconds, w ∈ {1, 5, 10, 30}.
- Target `fwd_ret_ws` = mid(t + w) / mid(t) − 1, in bps.
- Volatility regime: 300-second realized volatility of 1-second returns, split into terciles using train-month thresholds only.

## The 16 specifications
| Dimension | Values |
|---|---|
| OFI window = horizon | 1 s, 5 s, 10 s, 30 s |
| Model | Ridge regression on `ofi_ws` (predicts `fwd_ret_ws`); Logistic regression on `ofi_ws` (predicts sign of `fwd_ret_ws`) |
| Cost | taker: fee 5 bps per side; maker: fee 2 bps per side. Both add the full quoted spread per round trip |

`spec_id` = `w{window}_{model}_{cost}`, e.g. `w5_ridge_taker`. Each spec is run separately per symbol (32 results).

## Trading rule
- Sample every w seconds (non-overlapping) so returns do not overlap.
- Ridge: trade in the direction of the prediction when |prediction| > round-trip cost.
- Logistic: trade when |p − 0.5| > 0.05, direction = sign(p − 0.5).
- Round-trip cost (bps) = 2 × fee + `spread_bps`.
- PnL in bps on constant notional; final equity = 1 + ΣPnL / 10,000.

## Model selection inside train
- Purged 5-fold time-series CV on 2024-01 with a 60-second embargo.
- Ridge alpha chosen from {0.1, 1, 10} by CV R². Logistic C from {0.1, 1, 10} by CV log-loss. Nothing else is tuned.

## Metrics reported for every spec
Sharpe (hourly net PnL, annualized with √8760), Deflated Sharpe, max drawdown, win rate, final equity, OOS R², hit rate, number of trades.

## Controls
- Sanity check: contemporaneous regression of `ret_1s` on `ofi_1s` must give a positive slope.
- Null: shuffle `ofi_*` within each symbol-day and rerun all 16 specs.
- Overfitting: a separate grid of 300–500 specifications on train-month hourly PnL; CSCV with S = 16 gives PBO. Deflated Sharpe uses the number of trials in that grid.

## Decision rule
OFI "predicts net of costs" for a symbol only if the taker-cost spec has positive OOS Sharpe, Deflated Sharpe > 0.95, and the shuffled-OFI null does not reach the same Sharpe. A negative result is reported as such.

## Deviations
(none yet)
