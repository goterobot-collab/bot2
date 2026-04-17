# 🏆 TOP GRAILS (deduplicated & ranked)

- Source: `results/grails_loop.jsonl` (101649 total evaluations)
- Unique **(strategy, asset, params, exit)** combos: **60**
- Unique **strategy configs** (asset-agnostic): **60**
- Family breakdown (asset-agnostic configs): bb_trend_rejoin=27, rsi2_regime=18, zscore_revert=15
- Score = `ret × PF × trade-penalty × DD-penalty` (rewards profit, consistency, low DD)
- Filter: WR>60% AND PnL>0 AND maxDD>-30% AND PF>1 AND trades>5

| Rank | Score | Strategy | Asset | TF | Trades | WR % | PF | Ret % | DD % | Params | Exit |
|------|-------|----------|-------|----|--------|------|-----|-------|------|--------|------|
| 1 | 74.03 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 69.74 | 1.436 | 65.76 | -11.77 | `{"bb_len":24,"bb_mult":2.0,"trend":100}` | `{"sl_atr":2.0,"tp_atr":5.0,"trail_atr":5.0,"timeout":72}` |
| 2 | 62.14 | `bb_trend_rejoin` | ETHUSD | 1h | 149 | 69.8 | 1.5 | 52.04 | -9.84 | `{"bb_len":24,"bb_mult":2.0,"trend":75}` | `{"sl_atr":2.0,"tp_atr":3.0,"timeout":168}` |
| 3 | 61.83 | `rsi2_regime` | ETHUSD | 1h | 297 | 71.72 | 1.341 | 76.85 | -23.43 | `{"rsi_len":2,"rsi_buy":7,"sma_trend":75,"slope_bars":100,"exit_sma":13}` | `{"sl_atr":4.0,"tp_atr":10.0,"timeout":36}` |
| 4 | 53.63 | `rsi2_regime` | ETHUSD | 1h | 147 | 68.71 | 1.38 | 61.65 | -17.85 | `{"rsi_len":4,"rsi_buy":15,"sma_trend":100,"slope_bars":20,"exit_sma":21}` | `{"sl_atr":3.0,"tp_atr":7.0,"timeout":168}` |
| 5 | 53.26 | `bb_trend_rejoin` | ETHUSD | 1h | 65 | 72.31 | 1.978 | 86.08 | -10.27 | `{"bb_len":80,"bb_mult":2.0,"trend":300}` | `{"sl_atr":3.5,"timeout":168}` |
| 6 | 51.02 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.439 | 44.99 | -9.11 | `{"bb_len":24,"bb_mult":2.0,"trend":75}` | `{"sl_atr":2.5,"tp_atr":3.0,"trail_atr":5.0,"timeout":48}` |
| 7 | 50.48 | `zscore_revert` | ETHUSD | 1h | 137 | 62.77 | 1.484 | 56.84 | -11.09 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":1.5,"tp_atr":3.0,"trail_atr":4.0,"timeout":24}` |
| 8 | 47.55 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.507 | 56.02 | -16.71 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":3.0,"tp_atr":5.0,"trail_atr":4.0,"timeout":24}` |
| 9 | 47.00 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.496 | 55.47 | -15.95 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.5,"tp_atr":5.0,"trail_atr":4.0,"timeout":24}` |
| 10 | 44.92 | `zscore_revert` | ETHUSD | 1h | 137 | 63.5 | 1.447 | 51.88 | -11.09 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":1.5,"tp_atr":2.0,"trail_atr":4.0,"timeout":72}` |
| 11 | 42.55 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.445 | 51.99 | -17.34 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.5,"tp_atr":4.0,"timeout":48}` |
| 12 | 40.16 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 67.69 | 1.297 | 39.49 | -11.77 | `{"bb_len":24,"bb_mult":2.0,"trend":100}` | `{"sl_atr":2.0,"tp_atr":8.0,"trail_atr":2.5,"timeout":168}` |
| 13 | 39.31 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.413 | 48.03 | -16.0 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.0,"tp_atr":3.0,"trail_atr":4.0,"timeout":24}` |
| 14 | 38.77 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.41 | 41.05 | -9.11 | `{"bb_len":24,"bb_mult":2.0,"trend":75}` | `{"sl_atr":2.5,"tp_atr":8.0,"trail_atr":4.0,"timeout":24}` |
| 15 | 34.84 | `bb_trend_rejoin` | ETHUSD | 1h | 148 | 64.19 | 1.303 | 33.76 | -9.74 | `{"bb_len":24,"bb_mult":2.25,"trend":100}` | `{"sl_atr":1.5,"tp_atr":8.0,"trail_atr":4.0,"timeout":120}` |
| 16 | 34.51 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.383 | 44.31 | -16.74 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":3.0,"tp_atr":4.0,"timeout":72}` |
| 17 | 33.84 | `bb_trend_rejoin` | ETHUSD | 1h | 127 | 70.87 | 1.38 | 43.3 | -17.34 | `{"bb_len":30,"bb_mult":2.0,"trend":100}` | `{"sl_atr":2.5,"tp_atr":5.0,"trail_atr":5.0,"timeout":96}` |
| 18 | 33.21 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.38 | 42.49 | -18.05 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.5,"tp_atr":5.0,"trail_atr":4.0,"timeout":48}` |
| 19 | 32.29 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.373 | 41.52 | -18.05 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.5,"trail_atr":4.0,"timeout":72}` |
| 20 | 31.67 | `rsi2_regime` | ETHUSD | 1h | 52 | 75.0 | 1.938 | 40.05 | -9.04 | `{"rsi_len":5,"rsi_buy":15,"sma_trend":100,"slope_bars":30,"exit_sma":13}` | `{"sl_atr":2.0,"tp_atr":4.0,"trail_atr":5.0,"timeout":48}` |
| 21 | 30.94 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 63.74 | 1.595 | 42.99 | -10.09 | `{"bb_len":30,"bb_mult":2.25,"trend":100}` | `{"sl_atr":1.5,"tp_atr":4.0,"trail_atr":3.5,"timeout":120}` |
| 22 | 30.05 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.334 | 38.89 | -15.88 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.0,"tp_atr":3.0,"trail_atr":4.0,"timeout":120}` |
| 23 | 29.98 | `bb_trend_rejoin` | ETHUSD | 1h | 66 | 68.18 | 1.573 | 51.35 | -18.39 | `{"bb_len":80,"bb_mult":2.0,"trend":300}` | `{"sl_atr":3.5,"tp_atr":5.0,"trail_atr":4.0,"timeout":48}` |
| 24 | 28.45 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.338 | 37.54 | -18.14 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.5,"timeout":72}` |
| 25 | 23.62 | `bb_trend_rejoin` | ETHUSD | 1h | 183 | 71.04 | 1.244 | 33.95 | -20.18 | `{"bb_len":30,"bb_mult":1.75,"trend":100}` | `{"sl_atr":2.5,"tp_atr":6.0,"timeout":120}` |
| 26 | 23.40 | `rsi2_regime` | ETHUSD | 1h | 61 | 67.21 | 1.592 | 41.38 | -13.48 | `{"rsi_len":4,"rsi_buy":10,"sma_trend":150,"slope_bars":50,"exit_sma":21}` | `{"sl_atr":5.0,"tp_atr":10.0,"timeout":48}` |
| 27 | 22.92 | `rsi2_regime` | ETHUSD | 1h | 158 | 70.89 | 1.277 | 26.96 | -13.29 | `{"rsi_len":2,"rsi_buy":3,"sma_trend":100,"slope_bars":5,"exit_sma":8}` | `{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":5.0}` |
| 28 | 22.58 | `bb_trend_rejoin` | ETHUSD | 1h | 183 | 71.04 | 1.198 | 25.28 | -19.53 | `{"bb_len":30,"bb_mult":1.75,"trend":100}` | `{"sl_atr":2.5,"tp_atr":3.5,"trail_atr":4.0}` |
| 29 | 22.38 | `bb_trend_rejoin` | DOGEUSD | 1h | 66 | 68.18 | 1.719 | 35.07 | -14.04 | `{"bb_len":24,"bb_mult":1.5,"trend":100}` | `{"sl_atr":2.5,"timeout":24}` |
| 30 | 22.16 | `bb_trend_rejoin` | ETHUSD | 1h | 90 | 66.67 | 1.441 | 34.32 | -15.8 | `{"bb_len":30,"bb_mult":2.25,"trend":100}` | `{"sl_atr":3.0,"tp_atr":4.0,"trail_atr":3.0,"timeout":36}` |
| 31 | 22.10 | `rsi2_regime` | ETHUSD | 1h | 210 | 65.24 | 1.184 | 36.6 | -25.14 | `{"rsi_len":3,"rsi_buy":10,"sma_trend":125,"slope_bars":20,"exit_sma":21}` | `{"sl_atr":3.0,"tp_atr":10.0,"timeout":168}` |
| 32 | 22.01 | `bb_trend_rejoin` | ETHUSD | 1h | 140 | 68.57 | 1.275 | 28.39 | -14.84 | `{"bb_len":24,"bb_mult":2.25,"trend":100}` | `{"sl_atr":3.5,"tp_atr":8.0,"trail_atr":3.0,"timeout":36}` |
| 33 | 19.76 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.304 | 34.88 | -20.64 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.0,"tp_atr":4.0,"timeout":120}` |
| 34 | 19.38 | `bb_trend_rejoin` | ETHUSD | 1h | 95 | 68.42 | 1.324 | 37.11 | -17.06 | `{"bb_len":60,"bb_mult":2.0,"trend":300}` | `{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":5.0,"timeout":96}` |
| 35 | 18.97 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.251 | 26.47 | -16.56 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.5,"trail_atr":3.0}` |
| 36 | 18.89 | `bb_trend_rejoin` | ETHUSD | 1h | 129 | 67.44 | 1.25 | 26.38 | -16.56 | `{"bb_len":30,"bb_mult":2.0,"trend":100}` | `{"sl_atr":2.5,"trail_atr":3.0,"timeout":36}` |
| 37 | 17.61 | `zscore_revert` | ETHUSD | 1h | 66 | 68.18 | 1.513 | 49.19 | -22.34 | `{"z_len":80,"z_enter":2.0,"trend":300}` | `{"sl_atr":3.0}` |
| 38 | 17.51 | `bb_trend_rejoin` | ETHUSD | 1h | 66 | 68.18 | 1.521 | 48.66 | -22.34 | `{"bb_len":80,"bb_mult":2.0,"trend":300}` | `{"sl_atr":3.0,"tp_atr":8.0,"trail_atr":5.0,"timeout":120}` |
| 39 | 17.47 | `rsi2_regime` | ETHUSD | 1h | 52 | 78.85 | 1.831 | 23.38 | -8.35 | `{"rsi_len":5,"rsi_buy":15,"sma_trend":100,"slope_bars":15,"exit_sma":8}` | `{"sl_atr":2.0,"trail_atr":4.0,"timeout":120}` |
| 40 | 17.19 | `bb_trend_rejoin` | ETHUSD | 1h | 142 | 66.2 | 1.233 | 22.69 | -11.21 | `{"bb_len":24,"bb_mult":2.25,"trend":100}` | `{"sl_atr":2.5,"trail_atr":2.5,"timeout":24}` |
| 41 | 16.80 | `bb_trend_rejoin` | ETHUSD | 1h | 130 | 63.08 | 1.236 | 23.6 | -14.79 | `{"bb_len":30,"bb_mult":2.0,"trend":100}` | `{"sl_atr":3.0,"tp_atr":8.0,"trail_atr":2.5,"timeout":120}` |
| 42 | 15.87 | `rsi2_regime` | ETHUSD | 1h | 313 | 63.9 | 1.164 | 22.72 | -20.85 | `{"rsi_len":2,"rsi_buy":7,"sma_trend":75,"slope_bars":100,"exit_sma":5}` | `{"sl_atr":4.0,"trail_atr":3.5,"timeout":12}` |
| 43 | 15.75 | `rsi2_regime` | ETHUSD | 1h | 88 | 67.05 | 1.293 | 27.59 | -16.25 | `{"rsi_len":3,"rsi_buy":7,"sma_trend":125,"slope_bars":15,"exit_sma":21}` | `{"sl_atr":4.0,"tp_atr":5.0,"timeout":72}` |
| 44 | 15.62 | `bb_trend_rejoin` | DOGEUSD | 1h | 66 | 68.18 | 1.577 | 26.68 | -14.04 | `{"bb_len":24,"bb_mult":1.5,"trend":100}` | `{"sl_atr":2.5,"tp_atr":8.0,"trail_atr":4.0,"timeout":96}` |
| 45 | 15.29 | `rsi2_regime` | ETHUSD | 1h | 192 | 68.75 | 1.155 | 17.09 | -18.76 | `{"rsi_len":5,"rsi_buy":25,"sma_trend":75,"slope_bars":80,"exit_sma":8}` | `{"sl_atr":4.0,"tp_atr":10.0,"timeout":36}` |
| 46 | 15.05 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 63.55 | 1.128 | 16.68 | -17.64 | `{"bb_len":30,"bb_mult":1.5,"trend":75}` | `{"sl_atr":2.0,"tp_atr":2.5,"trail_atr":3.5,"timeout":18}` |
| 47 | 14.93 | `rsi2_regime` | ETHUSD | 1h | 53 | 67.92 | 1.899 | 23.86 | -11.19 | `{"rsi_len":3,"rsi_buy":7,"sma_trend":75,"slope_bars":5,"exit_sma":8}` | `{"sl_atr":2.5,"tp_atr":4.0,"trail_atr":4.0,"timeout":120}` |
| 48 | 13.97 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.24 | 25.93 | -20.64 | `{"z_len":30,"z_enter":2.0,"trend":100}` | `{"sl_atr":2.0}` |
| 49 | 13.88 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 69.06 | 1.21 | 18.97 | -12.63 | `{"bb_len":30,"bb_mult":1.75,"trend":75}` | `{"sl_atr":2.5,"tp_atr":3.5,"trail_atr":4.0,"timeout":18}` |
| 50 | 13.37 | `rsi2_regime` | ETHUSD | 1h | 69 | 68.12 | 1.526 | 21.66 | -9.23 | `{"rsi_len":3,"rsi_buy":7,"sma_trend":100,"slope_bars":30,"exit_sma":8}` | `{"sl_atr":4.0,"tp_atr":5.0,"trail_atr":5.0,"timeout":18}` |
| 51 | 12.27 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 67.27 | 1.136 | 15.7 | -14.55 | `{"bb_len":24,"bb_mult":2.25,"trend":125}` | `{"sl_atr":2.5,"tp_atr":5.0,"timeout":24}` |
| 52 | 10.77 | `rsi2_regime` | ETHUSD | 1h | 72 | 69.44 | 1.404 | 23.12 | -19.08 | `{"rsi_len":4,"rsi_buy":12,"sma_trend":100,"slope_bars":20,"exit_sma":13}` | `{"sl_atr":4.0,"tp_atr":10.0,"trail_atr":5.0,"timeout":96}` |
| 53 | 8.67 | `rsi2_regime` | ETHUSD | 1h | 90 | 67.78 | 1.195 | 16.19 | -17.99 | `{"rsi_len":3,"rsi_buy":8,"sma_trend":125,"slope_bars":80,"exit_sma":21}` | `{"sl_atr":3.0,"trail_atr":4.0,"timeout":168}` |
| 54 | 8.57 | `rsi2_regime` | ETHUSD | 1h | 72 | 70.83 | 1.232 | 17.81 | -16.81 | `{"rsi_len":5,"rsi_buy":15,"sma_trend":125,"slope_bars":10,"exit_sma":21}` | `{"sl_atr":5.0,"tp_atr":7.0,"trail_atr":4.0,"timeout":120}` |
| 55 | 5.38 | `bb_trend_rejoin` | ETHUSD | 1h | 166 | 63.25 | 1.12 | 9.26 | -22.8 | `{"bb_len":8,"bb_mult":2.25,"trend":75}` | `{"sl_atr":2.0,"tp_atr":3.0,"trail_atr":5.0,"timeout":48}` |
| 56 | 5.30 | `rsi2_regime` | ETHUSD | 1h | 119 | 67.23 | 1.135 | 8.64 | -17.52 | `{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":5,"exit_sma":8}` | `{"sl_atr":3.0,"tp_atr":4.0,"timeout":18}` |
| 57 | 4.93 | `rsi2_regime` | ETHUSD | 1h | 86 | 67.44 | 1.183 | 9.57 | -18.63 | `{"rsi_len":3,"rsi_buy":5,"sma_trend":250,"slope_bars":15,"exit_sma":8}` | `{"sl_atr":5.0,"trail_atr":5.0}` |
| 58 | 4.88 | `rsi2_regime` | ETHUSD | 1h | 81 | 69.14 | 1.191 | 11.51 | -15.12 | `{"rsi_len":3,"rsi_buy":8,"sma_trend":125,"slope_bars":100,"exit_sma":13}` | `{"sl_atr":3.0,"tp_atr":10.0,"trail_atr":4.0,"timeout":168}` |
| 59 | 1.80 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 63.55 | 1.044 | 2.88 | -24.61 | `{"bb_len":30,"bb_mult":1.5,"trend":75}` | `{"sl_atr":2.0,"tp_atr":3.5,"trail_atr":3.5,"timeout":96}` |
| 60 | 1.60 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 64.24 | 1.046 | 2.22 | -17.04 | `{"bb_len":24,"bb_mult":2.25,"trend":125}` | `{"sl_atr":2.5,"tp_atr":4.0,"trail_atr":3.0,"timeout":96}` |
