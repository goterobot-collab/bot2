# Grails on 4h/1d (relaxed DD<=40%)

- Total WF-passing grails on 4h/1d: 74
- Filter: WR>60, PF>1.1, DD>-40, trades>=30, both halves WR>55 ret>0
- Purpose: trend families need more DD tolerance than mean-rev on 1h

| Rank | Strategy | Asset | TF | WR | PF | Ret% | DD% | Trades | Params | Exit |
|------|----------|-------|----|-----|----|------|-----|--------|--------|------|
| 1 | `bb_trend_rejoin` | ETHUSD | 1d | 83.9 | 10.67 | 162.8 | -6.3 | 31 | `{"bb_len":10,"bb_mult":1.2,"trend":75}` | `{"sl_atr":3.0,"tp_atr":5.0,"trail_atr":5.0,"timeout":72}` |
| 2 | `keltner_squeeze` | BTCUSD | 1d | 60.6 | 5.71 | 1138.3 | -12.0 | 33 | `{"len":14,"bb_mult":1.5,"kc_mult":1.5,"trend":150}` | `{"sl_atr":3.0,"tp_atr":6.0,"timeout":24}` |
| 3 | `keltner_squeeze` | BTCUSD | 1d | 63.2 | 5.37 | 3580.3 | -27.0 | 38 | `{"len":20,"bb_mult":1.5,"kc_mult":2.0,"trend":50}` | `{"sl_atr":3.0,"timeout":48}` |
| 4 | `keltner_squeeze` | BTCUSD | 1d | 63.2 | 4.79 | 1417.7 | -36.2 | 38 | `{"len":14,"bb_mult":2.5,"kc_mult":2.0,"trend":100}` | `{"sl_atr":3.0,"tp_atr":6.0,"timeout":72}` |
| 5 | `keltner_squeeze` | BTCUSD | 1d | 71.0 | 4.41 | 1114.6 | -24.7 | 31 | `{"len":20,"bb_mult":2.0,"kc_mult":2.5,"trend":300}` | `{"sl_atr":3.0,"tp_atr":6.0,"timeout":48}` |
| 6 | `keltner_squeeze` | BTCUSD | 1d | 62.9 | 4.37 | 1183.1 | -16.5 | 35 | `{"len":20,"bb_mult":2.0,"kc_mult":2.5,"trend":200}` | `{"sl_atr":1.5,"tp_atr":6.0}` |
| 7 | `keltner_squeeze` | BTCUSD | 1d | 69.7 | 4.34 | 1120.3 | -24.7 | 33 | `{"len":20,"bb_mult":2.0,"kc_mult":2.5,"trend":150}` | `{"sl_atr":3.0,"tp_atr":6.0,"timeout":24}` |
| 8 | `psar_trend` | BTCUSD | 1d | 68.8 | 3.90 | 1726.4 | -28.5 | 32 | `{"hi_len":30,"exit_len":15,"trend":200,"slope_bars":20}` | `{"tp_atr":8.0,"trail_atr":5.0}` |
| 9 | `keltner_squeeze` | BTCUSD | 1d | 62.2 | 3.82 | 575.4 | -16.4 | 37 | `{"len":14,"bb_mult":2.5,"kc_mult":2.5,"trend":300}` | `{"sl_atr":2.0,"tp_atr":3.0,"timeout":48}` |
| 10 | `psar_trend` | BTCUSD | 1d | 62.2 | 3.53 | 1992.2 | -31.3 | 45 | `{"hi_len":14,"exit_len":5,"trend":200,"slope_bars":20}` | `{"tp_atr":8.0,"trail_atr":5.0}` |
| 11 | `psar_trend` | BTCUSD | 1d | 63.2 | 3.32 | 934.6 | -33.8 | 38 | `{"hi_len":55,"exit_len":5,"trend":200,"slope_bars":10}` | `{"tp_atr":8.0,"trail_atr":5.0}` |
| 12 | `tsmom_voltarget` | BTCUSD | 1d | 62.3 | 2.98 | 5927.9 | -34.2 | 106 | `{"mom_lb":36,"sma_trend":100,"vol_len":14,"vol_win":400,"vol_q":0.8}` | `{"tp_atr":3.0,"timeout":24}` |
| 13 | `supertrend_atr` | BTCUSD | 1d | 63.3 | 2.96 | 455.6 | -33.0 | 30 | `{"st_atr":7,"st_mult":2.5,"ema_trend":50}` | `{"sl_atr":3.0,"tp_atr":4.0,"trail_atr":4.0}` |
| 14 | `keltner_squeeze` | BTCUSD | 1d | 60.5 | 2.88 | 398.2 | -34.2 | 38 | `{"len":14,"bb_mult":2.5,"kc_mult":2.0,"trend":100}` | `{"sl_atr":2.0,"trail_atr":3.0}` |
| 15 | `psar_trend` | BTCUSD | 1d | 64.9 | 2.83 | 986.1 | -40.0 | 37 | `{"hi_len":30,"exit_len":10,"trend":200,"slope_bars":20}` | `{"tp_atr":8.0,"trail_atr":5.0}` |
| 16 | `psar_trend` | BTCUSD | 1d | 62.1 | 2.83 | 2220.1 | -31.2 | 58 | `{"hi_len":14,"exit_len":5,"trend":100,"slope_bars":20}` | `{"tp_atr":5.0,"trail_atr":5.0}` |
| 17 | `psar_trend` | BTCUSD | 1d | 64.9 | 2.74 | 545.1 | -28.9 | 37 | `{"hi_len":30,"exit_len":15,"trend":200,"slope_bars":10}` | `{"trail_atr":4.0}` |
| 18 | `bb_trend_rejoin` | ETHUSD | 4h | 73.3 | 2.53 | 41.8 | -7.0 | 30 | `{"bb_len":20,"bb_mult":2.0,"trend":50}` | `{"sl_atr":1.25,"tp_atr":3.5,"trail_atr":2.5,"timeout":12}` |
| 19 | `keltner_squeeze` | BTCUSD | 1d | 63.2 | 2.50 | 231.1 | -18.0 | 38 | `{"len":14,"bb_mult":2.5,"kc_mult":2.5,"trend":300}` | `{"sl_atr":1.5,"tp_atr":2.0,"trail_atr":4.0,"timeout":24}` |
| 20 | `adx_pullback` | ETHUSD | 1d | 62.5 | 2.48 | 332.1 | -21.5 | 40 | `{"adx_len":10,"adx_min":30,"ema_len":34,"lookback":12}` | `{"sl_atr":1.5,"tp_atr":2.0,"timeout":24}` |
| 21 | `bb_trend_rejoin` | ETHUSD | 4h | 77.4 | 2.48 | 119.6 | -26.4 | 53 | `{"bb_len":20,"bb_mult":2.0,"trend":100}` | `{"sl_atr":2.0,"tp_atr":6.0,"trail_atr":5.0,"timeout":48}` |
| 22 | `psar_trend` | BTCUSD | 1d | 65.2 | 2.48 | 1660.6 | -33.2 | 69 | `{"hi_len":20,"exit_len":5,"trend":200,"slope_bars":10}` | `{"tp_atr":3.0,"trail_atr":5.0}` |
| 23 | `psar_trend` | BTCUSD | 1d | 62.5 | 2.47 | 926.3 | -37.9 | 48 | `{"hi_len":14,"exit_len":5,"trend":100,"slope_bars":20}` | `{"sl_atr":3.0,"tp_atr":8.0,"trail_atr":5.0}` |
| 24 | `zscore_revert` | BTCUSD | 1d | 74.2 | 2.44 | 136.9 | -21.5 | 31 | `{"z_len":14,"z_enter":2.0,"trend":150}` | `{"sl_atr":1.5,"tp_atr":5.0,"timeout":24}` |
| 25 | `psar_trend` | BTCUSD | 1d | 65.6 | 2.36 | 296.1 | -27.6 | 32 | `{"hi_len":55,"exit_len":15,"trend":300,"slope_bars":10}` | `{"trail_atr":4.0}` |
| 26 | `psar_trend` | BTCUSD | 1d | 63.6 | 2.33 | 630.6 | -39.0 | 44 | `{"hi_len":20,"exit_len":15,"trend":100,"slope_bars":10}` | `{"tp_atr":8.0,"trail_atr":4.0}` |
| 27 | `psar_trend` | ETHUSD | 1d | 67.5 | 2.27 | 635.3 | -38.8 | 40 | `{"hi_len":10,"exit_len":10,"trend":150,"slope_bars":10}` | `{"tp_atr":3.0,"trail_atr":5.0}` |
| 28 | `psar_trend` | BTCUSD | 1d | 62.5 | 2.22 | 530.3 | -37.9 | 48 | `{"hi_len":30,"exit_len":15,"trend":150,"slope_bars":10}` | `{"sl_atr":3.0,"tp_atr":5.0,"trail_atr":4.0}` |
| 29 | `psar_trend` | BTCUSD | 1d | 61.2 | 2.20 | 905.7 | -39.5 | 49 | `{"hi_len":20,"exit_len":20,"trend":200,"slope_bars":20}` | `{"tp_atr":5.0,"trail_atr":4.0}` |
| 30 | `zscore_revert` | BTCUSD | 1d | 77.4 | 2.19 | 106.8 | -21.6 | 31 | `{"z_len":14,"z_enter":2.0,"trend":150}` | `{"sl_atr":2.5,"timeout":48}` |
| 31 | `keltner_squeeze` | BTCUSD | 4h | 60.3 | 2.08 | 165.5 | -17.3 | 78 | `{"len":50,"bb_mult":1.5,"kc_mult":2.0,"trend":200}` | `{"sl_atr":2.0,"tp_atr":3.0,"trail_atr":4.0,"timeout":48}` |
| 32 | `zscore_revert` | BTCUSD | 1d | 74.2 | 2.04 | 93.0 | -21.6 | 31 | `{"z_len":14,"z_enter":2.0,"trend":200}` | `{"sl_atr":2.5,"tp_atr":3.0,"trail_atr":4.0,"timeout":24}` |
| 33 | `keltner_squeeze` | BTCUSD | 1d | 61.0 | 2.02 | 136.6 | -23.2 | 41 | `{"len":20,"bb_mult":2.0,"kc_mult":2.5,"trend":100}` | `{"sl_atr":3.0,"trail_atr":2.0,"timeout":72}` |
| 34 | `keltner_squeeze` | BTCUSD | 1d | 65.7 | 2.01 | 97.0 | -19.1 | 35 | `{"len":20,"bb_mult":1.5,"kc_mult":2.0,"trend":200}` | `{"sl_atr":3.0,"trail_atr":2.0,"timeout":24}` |
| 35 | `keltner_squeeze` | BTCUSD | 4h | 61.3 | 1.97 | 116.1 | -14.5 | 62 | `{"len":50,"bb_mult":2.0,"kc_mult":2.5,"trend":300}` | `{"sl_atr":2.0,"tp_atr":3.0}` |
| 36 | `bb_trend_rejoin` | ETHUSD | 4h | 79.2 | 1.94 | 71.5 | -18.6 | 48 | `{"bb_len":24,"bb_mult":2.25,"trend":175}` | `{"sl_atr":3.0,"tp_atr":6.0,"trail_atr":4.0,"timeout":96}` |
| 37 | `zscore_revert` | BTCUSD | 1d | 77.8 | 1.92 | 88.2 | -22.3 | 36 | `{"z_len":10,"z_enter":2.0,"trend":150}` | `{"sl_atr":1.5,"tp_atr":4.0,"trail_atr":4.0,"timeout":72}` |
| 38 | `zscore_revert` | ETHUSD | 4h | 62.5 | 1.91 | 79.0 | -19.3 | 56 | `{"z_len":20,"z_enter":2.0,"trend":100}` | `{"sl_atr":1.0,"tp_atr":4.0,"trail_atr":4.0,"timeout":72}` |
| 39 | `keltner_squeeze` | BTCUSD | 4h | 60.5 | 1.89 | 91.5 | -12.0 | 76 | `{"len":50,"bb_mult":2.0,"kc_mult":2.5,"trend":150}` | `{"sl_atr":1.5,"tp_atr":2.0,"trail_atr":4.0,"timeout":72}` |
| 40 | `bb_trend_rejoin` | BTCUSD | 1d | 76.0 | 1.86 | 103.8 | -31.6 | 50 | `{"bb_len":10,"bb_mult":1.5,"trend":75}` | `{"sl_atr":2.5,"tp_atr":5.0,"timeout":24}` |
