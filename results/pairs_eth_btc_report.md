# Pairs trading: ETH vs BTC (1h, cointegration)

## Method
- Inner-join ETHUSD and BTCUSD on hourly timestamp.
- Hedge ratio via OLS: `log(ETH) = alpha + beta * log(BTC)` (in-sample, full history).
- Spread = `log(ETH) - beta * log(BTC)`.
- Z-score: rolling 100-bar mean/std of spread.
- LONG ETH when `z[t-1] < -2.0` AND `z[t-1] > z[t-2]` (turning up).
- EXIT when `|z[t-1]| < 0.5` OR after 200-bar timeout.
- Short side intentionally NOT taken (long-only framework).
- Execution: next-bar open, fees 10bps/side, slippage 5bps/side.

## Fit
- Overlapping bars: **39841** (2018-08-02 10:00:00+00:00 -> 2023-02-17 10:00:00+00:00)
- alpha = -6.947353
- beta  = 1.386946
- Raw entry signals: 1417
- Raw exit signals:  8714

## Backtest result (single run)

| Metric | Value |
|--------|-------|
| Trades | 170 |
| Wins / Losses | 90 / 80 |
| Win rate (WR) | 52.94% |
| Profit factor (PF) | 0.859 |
| Total return | -70.57% |
| Max drawdown (DD) | -72.78% |
| Avg win | 4.495% |
| Avg loss | -5.886% |
| Exits by reason | signal:165,timeout:5 |
| Period | 2018-08-02 10:00:00+00:00 -> 2023-02-17 10:00:00+00:00 |

## Caveats
- Hedge ratio fitted in-sample on the full overlap (look-ahead in the
  beta only, not in the z-score which is rolling). For a deployable
  variant, refit beta on a rolling window or a walk-forward train slice.
- Long-only: half of the cointegration edge (short ETH when z>>0) is
  discarded. WR and trade count reflect long leg only.
- Not registered in `grail_loop.py` SPACES — this is stand-alone.
