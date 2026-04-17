# Monte Carlo permutation test (Aronson EBTA)

- N permutations: **500** per grail
- Asset/TF: **ETHUSD 1h** (real OHLCV, 39841 bars)
- Procedure: shuffle entry-bar positions (preserve total entry count); rebuild exit signal deterministically from price; re-simulate.
- p-value = (1 + #{null_ret >= observed_ret}) / (1 + N)
- Significant if p < 0.01

| Rank | Strategy | Trades | Entries | Obs Ret % | Null Mean % | Null 95th % | Null Max % | p-value | Significant |
|------|----------|--------|---------|-----------|-------------|-------------|------------|---------|-------------|
| 1 | `rsi2_regime` | 158 | 218 | 65.44 | -21.00 | 16.39 | 60.26 | 0.0020 | YES |
| 2 | `rsi2_regime` | 103 | 171 | 45.41 | -20.09 | 21.64 | 71.18 | 0.0140 | no |
| 3 | `rsi2_regime` | 84 | 122 | 44.57 | -12.93 | 18.14 | 48.21 | 0.0060 | YES |
| 4 | `rsi2_regime` | 297 | 473 | 181.98 | -47.67 | -3.75 | 69.35 | 0.0020 | YES |
| 5 | `rsi2_regime` | 119 | 174 | 36.75 | -14.98 | 23.41 | 82.61 | 0.0220 | no |

## Configurations tested

- **#1** params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":100,"slope_bars":5,"exit_sma":8}` exit=`{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":5.0}` -> p=0.0020
- **#2** params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":125,"slope_bars":80,"exit_sma":13}` exit=`{"sl_atr":5.0,"tp_atr":7.0,"timeout":48}` -> p=0.0140
- **#3** params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":80,"exit_sma":8}` exit=`{"sl_atr":5.0,"timeout":96}` -> p=0.0060
- **#4** params=`{"rsi_len":2,"rsi_buy":7,"sma_trend":75,"slope_bars":100,"exit_sma":13}` exit=`{"sl_atr":4.0,"tp_atr":10.0,"timeout":36}` -> p=0.0020
- **#5** params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":5,"exit_sma":8}` exit=`{"sl_atr":3.0,"tp_atr":4.0,"timeout":18}` -> p=0.0220
