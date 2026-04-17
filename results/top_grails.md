# 🏆 TOP GRAILS (deduplicated & ranked)

- Source: `results/grails_loop.jsonl` (5831 total evaluations)
- Unique grail configs: **49**
- Score = `ret × PF × trade-penalty × DD-penalty` (rewards profit, consistency, low DD)
- Filter: WR>60% AND PnL>0 AND maxDD>-30% AND PF>1 AND trades>5

| Rank | Score | Strategy | Asset | TF | Trades | WR % | PF | Ret % | DD % | Params | Exit |
|------|-------|----------|-------|----|--------|------|-----|-------|------|--------|------|
| 1 | 57.59 | `bb_trend_rejoin` | DOGEUSD | 1h | 23 | 82.61 | 3.02 | 19.07 | -3.35 | `{"bb_len":20,"bb_mult":2.5,"trend":150}` | `{"sl_atr":2.0,"timeout":72}` |
| 2 | 36.54 | `bb_trend_rejoin` | DOGEUSD | 1h | 27 | 74.07 | 2.278 | 16.04 | -4.69 | `{"bb_len":10,"bb_mult":2.5,"trend":200}` | `{"sl_atr":3.0,"tp_atr":4.0,"trail_atr":3.0,"timeout":48}` |
| 3 | 30.21 | `bb_trend_rejoin` | ETHUSD | 1h | 12 | 66.67 | 3.058 | 9.88 | -1.6 | `{"bb_len":50,"bb_mult":2.5,"trend":100}` | `{"sl_atr":1.5,"tp_atr":3.0}` |
| 4 | 27.58 | `bb_trend_rejoin` | ETHUSD | 1h | 132 | 64.39 | 1.314 | 34.98 | -21.38 | `{"bb_len":30,"bb_mult":2.0,"trend":100}` | `{"sl_atr":2.0,"tp_atr":3.0,"trail_atr":3.0,"timeout":24}` |
| 5 | 20.00 | `bb_trend_rejoin` | ETHUSD | 1h | 257 | 65.76 | 1.16 | 28.73 | -16.74 | `{"bb_len":20,"bb_mult":2.0,"trend":100}` | `{"sl_atr":2.0,"tp_atr":3.0,"timeout":24}` |
| 6 | 18.48 | `bb_trend_rejoin` | DOGEUSD | 1h | 27 | 74.07 | 1.674 | 11.04 | -5.3 | `{"bb_len":10,"bb_mult":2.5,"trend":200}` | `{"sl_atr":2.0,"trail_atr":3.0}` |
| 7 | 18.48 | `bb_trend_rejoin` | DOGEUSD | 1h | 27 | 74.07 | 1.674 | 11.04 | -5.3 | `{"bb_len":10,"bb_mult":2.5,"trend":200}` | `{"sl_atr":2.0,"trail_atr":3.0,"timeout":24}` |
| 8 | 18.07 | `rsi2_regime` | ADAUSD | 1h | 18 | 77.78 | 1.84 | 9.82 | -3.81 | `{"rsi_len":3,"rsi_buy":5,"sma_trend":300,"slope_bars":20,"exit_sma":13}` | `{"sl_atr":2.5,"tp_atr":7.0}` |
| 9 | 17.61 | `bb_trend_rejoin` | DOGEUSD | 1h | 27 | 74.07 | 1.61 | 10.94 | -4.6 | `{"bb_len":10,"bb_mult":2.5,"trend":200}` | `{"sl_atr":1.5,"tp_atr":3.0,"timeout":48}` |
| 10 | 17.61 | `bb_trend_rejoin` | DOGEUSD | 1h | 27 | 74.07 | 1.61 | 10.94 | -4.6 | `{"bb_len":10,"bb_mult":2.5,"trend":200}` | `{"sl_atr":1.5,"tp_atr":3.0}` |
| 11 | 17.61 | `bb_trend_rejoin` | DOGEUSD | 1h | 27 | 74.07 | 1.61 | 10.94 | -4.6 | `{"bb_len":10,"bb_mult":2.5,"trend":200}` | `{"sl_atr":1.5,"tp_atr":3.0,"timeout":72}` |
| 12 | 17.41 | `rsi2_regime` | ETHUSD | 1h | 138 | 66.67 | 1.258 | 23.06 | -28.22 | `{"rsi_len":4,"rsi_buy":15,"sma_trend":100,"slope_bars":50,"exit_sma":8}` | `{"sl_atr":4.0,"tp_atr":7.0,"trail_atr":4.0,"timeout":36}` |
| 13 | 13.77 | `rsi2_regime` | ETHUSD | 1h | 16 | 62.5 | 2.135 | 6.45 | -4.28 | `{"rsi_len":4,"rsi_buy":8,"sma_trend":100,"slope_bars":30,"exit_sma":5}` | `{"sl_atr":1.5,"tp_atr":2.0,"trail_atr":4.0,"timeout":72}` |
| 14 | 13.02 | `rsi2_regime` | DOGEUSD | 1h | 18 | 72.22 | 1.982 | 6.57 | -4.59 | `{"rsi_len":4,"rsi_buy":12,"sma_trend":100,"slope_bars":80,"exit_sma":5}` | `{"sl_atr":2.0,"tp_atr":4.0,"timeout":72}` |
| 15 | 12.40 | `ema_cross_trend` | ADAUSD | 1h | 42 | 61.9 | 1.228 | 10.1 | -9.45 | `{"fast":34,"slow":21,"trend":150}` | `{"sl_atr":3.0,"tp_atr":8.0,"trail_atr":4.0}` |
| 16 | 11.57 | `rsi2_regime` | ADAUSD | 1h | 20 | 65.0 | 1.443 | 8.02 | -7.09 | `{"rsi_len":4,"rsi_buy":10,"sma_trend":150,"slope_bars":10,"exit_sma":13}` | `{"sl_atr":1.5,"tp_atr":4.0,"trail_atr":4.0}` |
| 17 | 11.36 | `ema_cross_trend` | ADAUSD | 1h | 26 | 61.54 | 1.429 | 7.95 | -10.13 | `{"fast":34,"slow":21,"trend":100}` | `{"sl_atr":3.0,"tp_atr":12.0,"trail_atr":2.0}` |
| 18 | 11.36 | `rsi2_regime` | ETHUSD | 1h | 27 | 62.96 | 1.469 | 7.73 | -7.47 | `{"rsi_len":3,"rsi_buy":5,"sma_trend":100,"slope_bars":80,"exit_sma":8}` | `{"sl_atr":1.0,"tp_atr":5.0,"trail_atr":4.0,"timeout":48}` |
| 19 | 11.04 | `bb_trend_rejoin` | ETHUSD | 1h | 245 | 66.53 | 1.114 | 16.52 | -19.35 | `{"bb_len":30,"bb_mult":1.5,"trend":100}` | `{"sl_atr":3.0,"tp_atr":4.0,"trail_atr":3.0}` |
| 20 | 9.44 | `bb_trend_rejoin` | ETHUSD | 1h | 57 | 63.16 | 1.21 | 13.0 | -17.14 | `{"bb_len":50,"bb_mult":2.5,"trend":250}` | `{"sl_atr":3.0,"tp_atr":4.0,"timeout":72}` |
| 21 | 9.12 | `bb_trend_rejoin` | DOGEUSD | 1h | 13 | 69.23 | 1.603 | 5.69 | -6.51 | `{"bb_len":20,"bb_mult":2.5,"trend":100}` | `{"sl_atr":1.5,"timeout":24}` |
| 22 | 6.62 | `bb_trend_rejoin` | ETHUSD | 1h | 267 | 60.3 | 1.074 | 10.27 | -21.46 | `{"bb_len":20,"bb_mult":2.0,"trend":100}` | `{"sl_atr":1.5,"tp_atr":3.0,"timeout":24}` |
| 23 | 6.21 | `bb_trend_rejoin` | ADAUSD | 1h | 21 | 80.95 | 1.338 | 4.64 | -6.18 | `{"bb_len":20,"bb_mult":2.5,"trend":150}` | `{"sl_atr":3.0,"tp_atr":4.0}` |
| 24 | 6.02 | `rsi2_regime` | ADAUSD | 1h | 18 | 77.78 | 1.462 | 4.12 | -6.4 | `{"rsi_len":3,"rsi_buy":5,"sma_trend":250,"slope_bars":30,"exit_sma":5}` | `{"sl_atr":1.5,"tp_atr":2.0,"trail_atr":2.0,"timeout":24}` |
| 25 | 5.72 | `rsi2_regime` | ADAUSD | 1h | 14 | 78.57 | 1.577 | 3.63 | -3.59 | `{"rsi_len":3,"rsi_buy":5,"sma_trend":200,"slope_bars":20,"exit_sma":8}` | `{"sl_atr":2.0,"tp_atr":4.0,"trail_atr":4.0,"timeout":72}` |
| 26 | 5.72 | `rsi2_regime` | ADAUSD | 1h | 14 | 78.57 | 1.577 | 3.63 | -3.59 | `{"rsi_len":3,"rsi_buy":5,"sma_trend":200,"slope_bars":20,"exit_sma":8}` | `{"sl_atr":2.0,"tp_atr":7.0,"timeout":36}` |
| 27 | 5.45 | `rsi2_regime` | ADAUSD | 1h | 13 | 76.92 | 1.514 | 3.6 | -4.91 | `{"rsi_len":4,"rsi_buy":8,"sma_trend":250,"slope_bars":30,"exit_sma":8}` | `{"sl_atr":1.5,"tp_atr":5.0,"trail_atr":4.0,"timeout":48}` |
| 28 | 4.60 | `bb_trend_rejoin` | DOGEUSD | 1h | 23 | 65.22 | 1.286 | 3.58 | -6.51 | `{"bb_len":10,"bb_mult":2.5,"trend":150}` | `{"sl_atr":3.0,"tp_atr":4.0,"trail_atr":2.0,"timeout":72}` |
| 29 | 4.59 | `rsi2_regime` | ETHUSD | 1h | 16 | 68.75 | 1.557 | 2.95 | -3.9 | `{"rsi_len":4,"rsi_buy":8,"sma_trend":100,"slope_bars":10,"exit_sma":3}` | `{"sl_atr":1.5,"tp_atr":2.0,"trail_atr":2.0,"timeout":120}` |
| 30 | 4.40 | `rsi2_regime` | ADAUSD | 1h | 8 | 75.0 | 1.414 | 3.11 | -3.55 | `{"rsi_len":4,"rsi_buy":8,"sma_trend":150,"slope_bars":20,"exit_sma":13}` | `{"sl_atr":2.5,"trail_atr":3.0}` |
| 31 | 4.10 | `rsi2_regime` | ADAUSD | 1h | 19 | 63.16 | 1.277 | 3.21 | -7.59 | `{"rsi_len":3,"rsi_buy":5,"sma_trend":250,"slope_bars":80,"exit_sma":8}` | `{"sl_atr":1.0,"tp_atr":7.0,"trail_atr":4.0,"timeout":36}` |
| 32 | 3.36 | `rsi2_regime` | ETHUSD | 1h | 71 | 66.2 | 1.126 | 4.98 | -16.68 | `{"rsi_len":3,"rsi_buy":5,"sma_trend":200,"slope_bars":80,"exit_sma":8}` | `{"sl_atr":3.0,"tp_atr":7.0,"timeout":120}` |
| 33 | 2.98 | `rsi2_regime` | ETHUSD | 1h | 12 | 66.67 | 1.417 | 2.1 | -3.35 | `{"rsi_len":4,"rsi_buy":5,"sma_trend":200,"slope_bars":30,"exit_sma":5}` | `{"sl_atr":4.0,"trail_atr":3.0,"timeout":48}` |
| 34 | 2.63 | `rsi2_regime` | DOGEUSD | 1h | 129 | 61.24 | 1.066 | 4.11 | -24.31 | `{"rsi_len":2,"rsi_buy":10,"sma_trend":100,"slope_bars":80,"exit_sma":8}` | `{"sl_atr":1.5,"tp_atr":4.0,"timeout":36}` |
| 35 | 2.60 | `bb_trend_rejoin` | DOGEUSD | 1h | 13 | 69.23 | 1.242 | 2.09 | -6.51 | `{"bb_len":20,"bb_mult":2.5,"trend":100}` | `{"sl_atr":1.5,"trail_atr":3.0,"timeout":72}` |
| 36 | 2.33 | `bb_trend_rejoin` | DOGEUSD | 1h | 13 | 61.54 | 1.205 | 1.93 | -8.15 | `{"bb_len":20,"bb_mult":2.5,"trend":100}` | `{"sl_atr":1.5,"trail_atr":2.0,"timeout":48}` |
| 37 | 2.33 | `bb_trend_rejoin` | DOGEUSD | 1h | 13 | 61.54 | 1.205 | 1.93 | -8.15 | `{"bb_len":20,"bb_mult":2.5,"trend":100}` | `{"sl_atr":1.5,"trail_atr":2.0,"timeout":72}` |
| 38 | 2.30 | `rsi2_regime` | DOGEUSD | 1h | 24 | 66.67 | 1.15 | 2.0 | -14.23 | `{"rsi_len":4,"rsi_buy":12,"sma_trend":150,"slope_bars":50,"exit_sma":5}` | `{"sl_atr":4.0,"timeout":48}` |
| 39 | 2.18 | `rsi2_regime` | ADAUSD | 1h | 18 | 72.22 | 1.213 | 1.8 | -5.33 | `{"rsi_len":3,"rsi_buy":5,"sma_trend":300,"slope_bars":10,"exit_sma":5}` | `{"sl_atr":2.0,"tp_atr":4.0,"trail_atr":4.0,"timeout":72}` |
| 40 | 1.68 | `ema_cross_trend` | ADAUSD | 1h | 26 | 65.38 | 1.086 | 1.55 | -11.03 | `{"fast":34,"slow":21,"trend":100}` | `{"tp_atr":12.0,"trail_atr":5.0}` |
| 41 | 1.57 | `rsi2_regime` | ADAUSD | 1h | 22 | 68.18 | 1.111 | 1.41 | -8.66 | `{"rsi_len":4,"rsi_buy":10,"sma_trend":250,"slope_bars":50,"exit_sma":8}` | `{"sl_atr":3.0,"trail_atr":4.0,"timeout":24}` |
| 42 | 1.14 | `bb_trend_rejoin` | DOGEUSD | 1h | 84 | 60.71 | 1.065 | 1.78 | -27.04 | `{"bb_len":10,"bb_mult":2.0,"trend":100}` | `{"sl_atr":3.0,"tp_atr":4.0,"timeout":48}` |
| 43 | 1.12 | `rsi2_regime` | ETHUSD | 1h | 75 | 62.67 | 1.064 | 1.75 | -15.16 | `{"rsi_len":3,"rsi_buy":5,"sma_trend":200,"slope_bars":20,"exit_sma":13}` | `{"sl_atr":2.5,"tp_atr":3.0,"trail_atr":4.0}` |
| 44 | 1.09 | `rsi2_regime` | ETHUSD | 1h | 72 | 66.67 | 1.066 | 1.71 | -18.52 | `{"rsi_len":4,"rsi_buy":12,"sma_trend":100,"slope_bars":30,"exit_sma":8}` | `{"sl_atr":4.0,"tp_atr":5.0,"trail_atr":4.0,"timeout":24}` |
| 45 | 1.02 | `bb_trend_rejoin` | ADAUSD | 1h | 26 | 69.23 | 1.065 | 0.96 | -13.77 | `{"bb_len":30,"bb_mult":2.0,"trend":100}` | `{"sl_atr":2.0,"tp_atr":3.0}` |
| 46 | 0.85 | `rsi2_regime` | ETHUSD | 1h | 6 | 66.67 | 1.163 | 0.98 | -6.05 | `{"rsi_len":4,"rsi_buy":5,"sma_trend":150,"slope_bars":20,"exit_sma":13}` | `{"sl_atr":2.5,"tp_atr":3.0,"trail_atr":3.0,"timeout":36}` |
| 47 | 0.64 | `rsi2_regime` | DOGEUSD | 1h | 18 | 66.67 | 1.092 | 0.59 | -4.33 | `{"rsi_len":4,"rsi_buy":12,"sma_trend":100,"slope_bars":80,"exit_sma":3}` | `{"sl_atr":4.0,"timeout":48}` |
| 48 | 0.33 | `bb_trend_rejoin` | ADAUSD | 1h | 16 | 81.25 | 1.057 | 0.31 | -7.59 | `{"bb_len":20,"bb_mult":2.5,"trend":100}` | `{"sl_atr":3.0,"tp_atr":4.0,"timeout":72}` |
| 49 | 0.12 | `rsi2_regime` | DOGEUSD | 1h | 14 | 64.29 | 1.041 | 0.12 | -9.69 | `{"rsi_len":4,"rsi_buy":10,"sma_trend":100,"slope_bars":50,"exit_sma":8}` | `{"sl_atr":2.0,"trail_atr":3.0,"timeout":24}` |
