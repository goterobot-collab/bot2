# Backtest Summary — qualifying combos

- Filter: WR > 60% AND trades > 5
- Fees: 0.100%/side, slippage: 0.050%/side
- Execution: next-bar open, long-only, one position at a time
- Qualifying combos: **5** / 39
- ⚠ 5 of 5 qualifying rows have negative total return.

| # | Strategy | Asset | TF | Trades | WR % | PF | Total Ret % | Max DD % | Exits |
|---|----------|-------|----|--------|------|-----|-------------|----------|-------|
| 1 | `rsi_oversold_rev` | ETHUSD | 1h | 158 | 64.56 | 0.955 | -70.98 | -72.71 | signal:158 |
| 2 | `rsi_oversold_rev` | DOGEUSD | 1h | 49 | 63.27 | 0.956 | -26.51 | -55.74 | signal:49 |
| 3 | `bb_meanrev` | ETHUSD | 1h | 747 | 62.12 | 0.74 | -92.28 | -92.39 | eod:1,signal:746 |
| 4 | `stoch_rev` | ETHUSD | 1h | 296 | 61.15 | 0.949 | -71.4 | -76.92 | eod:1,signal:295 |
| 5 | `rsi2_regime_atr` | DOGEUSD | 1h | 195 | 61.03 | 0.871 | -21.93 | -48.03 | signal:155,sl:40 |
