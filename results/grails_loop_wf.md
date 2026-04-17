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
| 2 | `rsi2_regime` | ETHUSD | 1h | 210 | 65.24 | 1.184 | 36.6 | -25.14 | `{"rsi_len":3,"rsi_buy":10,"sma_trend":125,"slope_bars":20,"exit_sma":21}` | `{"sl_atr":3.0,"tp_atr":10.0,"timeout":168}` |
| 4 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.338 | 37.54 | -18.14 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.5,"timeout":72}` |
| 2 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 69.06 | 1.21 | 18.97 | -12.63 | `{"bb_len":30,"bb_mult":1.75,"trend":75}` | `{"sl_atr":2.5,"tp_atr":3.5,"trail_atr":4.0,"timeout":18}` |
| 3 | `zscore_revert` | ETHUSD | 1h | 137 | 62.77 | 1.484 | 56.84 | -11.09 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":1.5,"tp_atr":3.0,"trail_atr":4.0,"timeout":24}` |
| 4 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 67.69 | 1.297 | 39.49 | -11.77 | `{"bb_len":24,"bb_mult":2.0,"trend":100}` | `{"sl_atr":2.0,"tp_atr":8.0,"trail_atr":2.5,"timeout":168}` |
| 2 | `rsi2_regime` | ETHUSD | 1h | 192 | 68.75 | 1.155 | 17.09 | -18.76 | `{"rsi_len":5,"rsi_buy":25,"sma_trend":75,"slope_bars":80,"exit_sma":8}` | `{"sl_atr":4.0,"tp_atr":10.0,"timeout":36}` |
| 5 | `bb_trend_rejoin` | ETHUSD | 1h | 166 | 63.25 | 1.12 | 9.26 | -22.8 | `{"bb_len":8,"bb_mult":2.25,"trend":75}` | `{"sl_atr":2.0,"tp_atr":3.0,"trail_atr":5.0,"timeout":48}` |
| 1 | `rsi2_regime` | ETHUSD | 1h | 81 | 69.14 | 1.191 | 11.51 | -15.12 | `{"rsi_len":3,"rsi_buy":8,"sma_trend":125,"slope_bars":100,"exit_sma":13}` | `{"sl_atr":3.0,"tp_atr":10.0,"trail_atr":4.0,"timeout":168}` |
| 4 | `bb_trend_rejoin` | ETHUSD | 1h | 129 | 67.44 | 1.25 | 26.38 | -16.56 | `{"bb_len":30,"bb_mult":2.0,"trend":100}` | `{"sl_atr":2.5,"trail_atr":3.0,"timeout":36}` |
| 4 | `bb_trend_rejoin` | ETHUSD | 1h | 148 | 64.19 | 1.303 | 33.76 | -9.74 | `{"bb_len":24,"bb_mult":2.25,"trend":100}` | `{"sl_atr":1.5,"tp_atr":8.0,"trail_atr":4.0,"timeout":120}` |
| 5 | `bb_trend_rejoin` | ETHUSD | 1h | 95 | 68.42 | 1.324 | 37.11 | -17.06 | `{"bb_len":60,"bb_mult":2.0,"trend":300}` | `{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":5.0,"timeout":96}` |
| 6 | `bb_trend_rejoin` | ETHUSD | 1h | 142 | 66.2 | 1.233 | 22.69 | -11.21 | `{"bb_len":24,"bb_mult":2.25,"trend":100}` | `{"sl_atr":2.5,"trail_atr":2.5,"timeout":24}` |
| 3 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.24 | 25.93 | -20.64 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.0}` |
