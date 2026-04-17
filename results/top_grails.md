# 🏆 TOP GRAILS (deduplicated & ranked)

- Source: `results/grails_loop.jsonl` (17967 total evaluations)
- Unique **(strategy, asset, params, exit)** combos: **13**
- Unique **strategy configs** (asset-agnostic): **13**
- Family breakdown (asset-agnostic configs): bb_trend_rejoin=6, zscore_revert=5, rsi2_regime=2
- Score = `ret × PF × trade-penalty × DD-penalty` (rewards profit, consistency, low DD)
- Filter: WR>60% AND PnL>0 AND maxDD>-30% AND PF>1 AND trades>5

| Rank | Score | Strategy | Asset | TF | Trades | WR % | PF | Ret % | DD % | Params | Exit |
|------|-------|----------|-------|----|--------|------|-----|-------|------|--------|------|
| 1 | 62.14 | `bb_trend_rejoin` | ETHUSD | 1h | 149 | 69.8 | 1.5 | 52.04 | -9.84 | `{"bb_len":24,"bb_mult":2.0,"trend":75}` | `{"sl_atr":2.0,"tp_atr":3.0,"timeout":168}` |
| 2 | 53.26 | `bb_trend_rejoin` | ETHUSD | 1h | 65 | 72.31 | 1.978 | 86.08 | -10.27 | `{"bb_len":80,"bb_mult":2.0,"trend":300}` | `{"sl_atr":3.5,"timeout":168}` |
| 3 | 51.02 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.439 | 44.99 | -9.11 | `{"bb_len":24,"bb_mult":2.0,"trend":75}` | `{"sl_atr":2.5,"tp_atr":3.0,"trail_atr":5.0,"timeout":48}` |
| 4 | 47.00 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.496 | 55.47 | -15.95 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.5,"tp_atr":5.0,"trail_atr":4.0,"timeout":24}` |
| 5 | 39.31 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.413 | 48.03 | -16.0 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.0,"tp_atr":3.0,"trail_atr":4.0,"timeout":24}` |
| 6 | 34.51 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.383 | 44.31 | -16.74 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":3.0,"tp_atr":4.0,"timeout":72}` |
| 7 | 33.84 | `bb_trend_rejoin` | ETHUSD | 1h | 127 | 70.87 | 1.38 | 43.3 | -17.34 | `{"bb_len":30,"bb_mult":2.0,"trend":100}` | `{"sl_atr":2.5,"tp_atr":5.0,"trail_atr":5.0,"timeout":96}` |
| 8 | 32.29 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.373 | 41.52 | -18.05 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.5,"trail_atr":4.0,"timeout":72}` |
| 9 | 18.97 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.251 | 26.47 | -16.56 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.5,"trail_atr":3.0}` |
| 10 | 17.47 | `rsi2_regime` | ETHUSD | 1h | 52 | 78.85 | 1.831 | 23.38 | -8.35 | `{"rsi_len":5,"rsi_buy":15,"sma_trend":100,"slope_bars":15,"exit_sma":8}` | `{"sl_atr":2.0,"trail_atr":4.0,"timeout":120}` |
| 11 | 15.62 | `bb_trend_rejoin` | DOGEUSD | 1h | 66 | 68.18 | 1.577 | 26.68 | -14.04 | `{"bb_len":24,"bb_mult":1.5,"trend":100}` | `{"sl_atr":2.5,"tp_atr":8.0,"trail_atr":4.0,"timeout":96}` |
| 12 | 10.77 | `rsi2_regime` | ETHUSD | 1h | 72 | 69.44 | 1.404 | 23.12 | -19.08 | `{"rsi_len":4,"rsi_buy":12,"sma_trend":100,"slope_bars":20,"exit_sma":13}` | `{"sl_atr":4.0,"tp_atr":10.0,"trail_atr":5.0,"timeout":96}` |
| 13 | 1.60 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 64.24 | 1.046 | 2.22 | -17.04 | `{"bb_len":24,"bb_mult":2.25,"trend":125}` | `{"sl_atr":2.5,"tp_atr":4.0,"trail_atr":3.0,"timeout":96}` |
