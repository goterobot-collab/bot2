# Backtest Summary — qualifying combos

- Filter: WR > 60% AND trades > 5
- Fees: 0.100%/side, slippage: 0.050%/side
- Execution: next-bar open, long-only, one position at a time
- Qualifying combos: **4** / 24
- ⚠ 4 of 4 qualifying rows have **negative total return** — high WR with asymmetric R:R (big losses, small wins). WR alone is not an edge.

| # | Strategy | Asset | TF | Trades | WR % | PF | Total Ret % | Max DD % |
|---|----------|-------|----|--------|------|-----|-------------|----------|
| 1 | `rsi_oversold_rev` | ETHUSD | 1h | 158 | 64.56 | 0.955 | -70.98 | -72.71 |
| 2 | `rsi_oversold_rev` | DOGEUSD | 1h | 49 | 63.27 | 0.956 | -26.51 | -55.74 |
| 3 | `bb_meanrev` | ETHUSD | 1h | 747 | 62.12 | 0.74 | -92.28 | -92.39 |
| 4 | `stoch_rev` | ETHUSD | 1h | 296 | 61.15 | 0.949 | -71.4 | -76.92 |
