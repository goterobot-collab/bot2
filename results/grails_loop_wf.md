# 🏆 GRAIL loop — found configs

- Filter: WR > 60% AND PnL > 0 AND maxDD > -30.0% AND PF > 1.0
- Trades > 5 (inherited from backtest.py)

| # | Strategy | Asset | TF | Trades | WR % | PF | Ret % | DD % | Params | Exit cfg |
|---|----------|-------|----|--------|------|-----|-------|------|--------|----------|
| 1 | `rsi2_regime` | ETHUSD | 1h | 72 | 69.44 | 1.404 | 23.12 | -19.08 | `{"rsi_len":4,"rsi_buy":12,"sma_trend":100,"slope_bars":20,"exit_sma":13}` | `{"sl_atr":4.0,"tp_atr":10.0,"trail_atr":5.0,"timeout":96}` |
| 1 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.413 | 48.03 | -16.0 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.0,"tp_atr":3.0,"trail_atr":4.0,"timeout":24}` |
| 1 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 64.24 | 1.046 | 2.22 | -17.04 | `{"bb_len":24,"bb_mult":2.25,"trend":125}` | `{"sl_atr":2.5,"tp_atr":4.0,"trail_atr":3.0,"timeout":96}` |
| 2 | `bb_trend_rejoin` | ETHUSD | 1h | 127 | 70.87 | 1.38 | 43.3 | -17.34 | `{"bb_len":30,"bb_mult":2.0,"trend":100}` | `{"sl_atr":2.5,"tp_atr":5.0,"trail_atr":5.0,"timeout":96}` |
| 3 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.373 | 41.52 | -18.05 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.5,"trail_atr":4.0,"timeout":72}` |
| 2 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.383 | 44.31 | -16.74 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":3.0,"tp_atr":4.0,"timeout":72}` |
| 3 | `bb_trend_rejoin` | DOGEUSD | 1h | 66 | 68.18 | 1.577 | 26.68 | -14.04 | `{"bb_len":24,"bb_mult":1.5,"trend":100}` | `{"sl_atr":2.5,"tp_atr":8.0,"trail_atr":4.0,"timeout":96}` |
| 1 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.439 | 44.99 | -9.11 | `{"bb_len":24,"bb_mult":2.0,"trend":75}` | `{"sl_atr":2.5,"tp_atr":3.0,"trail_atr":5.0,"timeout":48}` |
| 2 | `bb_trend_rejoin` | ETHUSD | 1h | 149 | 69.8 | 1.5 | 52.04 | -9.84 | `{"bb_len":24,"bb_mult":2.0,"trend":75}` | `{"sl_atr":2.0,"tp_atr":3.0,"timeout":168}` |
| 1 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.496 | 55.47 | -15.95 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.5,"tp_atr":5.0,"trail_atr":4.0,"timeout":24}` |
| 1 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.251 | 26.47 | -16.56 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.5,"trail_atr":3.0}` |
| 3 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 63.55 | 1.128 | 16.68 | -17.64 | `{"bb_len":30,"bb_mult":1.5,"trend":75}` | `{"sl_atr":2.0,"tp_atr":2.5,"trail_atr":3.5,"timeout":18}` |
