# verify-trading-strategies

Pine Script strategies ready to backtest on TradingView across **ADA / DOGE / INJ / ETH**
on **5m / 15m / 1h**, with full 1-year windows. Includes a Python cross-validation kit
so you can reproduce results off-exchange.

## What you asked for

> Verify strategies in Pine v4/5/6, TF 5m/15m/1h, WR > 60%, more than 5 trades in the
> asset's full history. Generate a folder with the codes and info, sorted by WR
> descending.

## What's here

```
pine_sources/        # 8 Pine v5 strategies, each with a 365-day default window
strategies/          # Python ports (identical logic) for cross-checking
fetch_data.py        # Pulls OHLCV from Binance (run locally, not in sandbox)
backtest.py          # Runs every strategy × asset × TF, writes results/
results_template.md  # Worksheet for recording TradingView Strategy Tester output
README_PINE.md       # Step-by-step TradingView workflow
```

## Two ways to verify

### 1) Pine on TradingView (authoritative — matches what you see in Strategy Tester)

See `README_PINE.md`. Paste each `.pine` file → switch symbol/TF → record
WR/trades/PF. Use `results_template.md` to compile results sorted by WR.

### 2) Python cross-check (reproducible, no TradingView account needed)

```bash
pip install -r requirements.txt
python3 fetch_data.py                 # downloads OHLCV to data/
python3 backtest.py --min-wr 0.60 --min-trades 5
# results/ now contains summary.md (sorted desc) + by_wr/*.md per combo
```

Both paths use the same entry/exit logic. If the Pine Strategy Tester and the
Python backtester disagree, it's almost always due to execution model
(next-bar-open vs. bar-close, Pine's `process_orders_on_close=false` default, etc.).

## What is NOT claimed here

- **No WR numbers in this repo are "author-verified"** until you run the backtests.
  Any WR % mentioned in Pine comments comes from the original author's claim, not
  from my measurement.
- **No Hetzner / Optuna integration.** That's downstream.
- **No live trading.** These are backtest scripts only.

## Strategy list

| # | File                              | Logic summary                                         |
|---|-----------------------------------|-------------------------------------------------------|
| 1 | `01_rsi2_connors.pine`            | RSI(2)<10 above SMA200; exit > SMA(5)                 |
| 2 | `02_rsi_oversold_rev.pine`        | RSI(14) crosses above 30; exit on cross below 70      |
| 3 | `03_macd_ema200.pine`             | MACD(12,26,9) cross up above EMA200; exit on cross dn |
| 4 | `04_ema9_21_cross.pine`           | EMA 9/21 cross + EMA200 filter                        |
| 5 | `05_bb_meanrev.pine`              | Close crosses above BB lower; exit at basis           |
| 6 | `06_supertrend_ema.pine`          | Supertrend flip up + close > EMA200                   |
| 7 | `07_stoch_rev.pine`               | Stoch(14,3) cross up below 20; exit cross dn above 80 |
| 8 | `08_macd_3up_candles.pine`        | MACD hist zero-cross + 3 up candles, ATR SL/TP        |

These are public, classic patterns. None of them is expected to hit WR>60% on
every asset × TF combination — the point is to find which **do** hit the bar on
**which** combinations, which is what your original request asks.
