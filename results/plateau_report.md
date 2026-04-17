# Parameter-plateau stability report

- Input: 115 unique WF grails
- Jitter: +/-20% in 10% steps on every numeric param and exit-cfg value
- Plateau threshold: >= 75% of neighbors still pass WF filter

| Rank | Strategy | Asset | TF | Trades | WR | PF | Ret% | DD% | Neighbors | Passed | Frac | Verdict |
|------|----------|-------|----|--------|----|----|------|-----|-----------|--------|------|---------|
| 1 | `rsi2_regime` | ETHUSD | 1h | 158 | 70.89 | 1.277 | 26.96 | -13.29 | 22 | 22 | 1.00 | PLATEAU |
| 2 | `rsi2_regime` | ETHUSD | 1h | 103 | 66.02 | 1.277 | 23.66 | -16.79 | 22 | 22 | 1.00 | PLATEAU |
| 3 | `rsi2_regime` | ETHUSD | 1h | 84 | 72.62 | 1.558 | 26.54 | -10.79 | 21 | 19 | 0.90 | PLATEAU |
| 4 | `rsi2_regime` | ETHUSD | 1h | 297 | 71.72 | 1.341 | 76.85 | -23.43 | 23 | 20 | 0.87 | PLATEAU |
| 5 | `rsi2_regime` | ETHUSD | 1h | 119 | 67.23 | 1.135 | 8.64 | -17.52 | 23 | 19 | 0.83 | PLATEAU |
| 6 | `bb_trend_rejoin` | ETHUSD | 1h | 127 | 70.87 | 1.38 | 43.3 | -17.34 | 23 | 18 | 0.78 | PLATEAU |
| 7 | `bb_trend_rejoin` | ETHUSD | 1h | 183 | 71.04 | 1.244 | 33.95 | -20.18 | 23 | 18 | 0.78 | PLATEAU |
| 8 | `rsi2_regime` | ETHUSD | 1h | 146 | 69.18 | 1.272 | 23.89 | -15.26 | 23 | 18 | 0.78 | PLATEAU |
| 9 | `rsi2_regime` | ETHUSD | 1h | 116 | 71.55 | 1.251 | 20.81 | -24.4 | 23 | 18 | 0.78 | PLATEAU |
| 10 | `rsi2_regime` | ETHUSD | 1h | 147 | 68.71 | 1.38 | 61.65 | -17.85 | 22 | 17 | 0.77 | PLATEAU |
| 11 | `rsi2_regime` | ETHUSD | 1h | 288 | 68.06 | 1.219 | 52.92 | -22.63 | 21 | 16 | 0.76 | PLATEAU |
| 12 | `rsi2_regime` | ETHUSD | 1h | 72 | 69.44 | 1.404 | 23.12 | -19.08 | 21 | 16 | 0.76 | PLATEAU |
| 13 | `rsi2_regime` | ETHUSD | 1h | 135 | 69.63 | 1.177 | 13.78 | -16.87 | 24 | 18 | 0.75 | PLATEAU |
| 14 | `rsi2_regime` | ETHUSD | 1h | 210 | 70.0 | 1.271 | 35.25 | -17.7 | 21 | 15 | 0.71 | spike |
| 15 | `bb_trend_rejoin` | ETHUSD | 1h | 125 | 72.0 | 1.392 | 43.72 | -18.62 | 24 | 16 | 0.67 | spike |
| 16 | `bb_trend_rejoin` | ETHUSD | 1h | 183 | 71.04 | 1.198 | 25.28 | -19.53 | 24 | 16 | 0.67 | spike |
| 17 | `bb_trend_rejoin` | ETHUSD | 1h | 196 | 64.8 | 1.093 | 11.21 | -20.56 | 24 | 16 | 0.67 | spike |
| 18 | `rsi2_regime` | ETHUSD | 1h | 140 | 68.57 | 1.3 | 41.55 | -27.25 | 23 | 15 | 0.65 | spike |
| 19 | `rsi2_regime` | ETHUSD | 1h | 113 | 69.03 | 1.313 | 34.83 | -12.67 | 19 | 12 | 0.63 | spike |
| 20 | `rsi2_regime` | ETHUSD | 1h | 97 | 70.1 | 1.198 | 14.59 | -17.52 | 23 | 14 | 0.61 | spike |
| 21 | `bb_trend_rejoin` | ETHUSD | 1h | 149 | 69.8 | 1.5 | 52.04 | -9.84 | 24 | 14 | 0.58 | spike |
| 22 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 67.88 | 1.198 | 25.19 | -14.36 | 24 | 14 | 0.58 | spike |
| 23 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 71.59 | 1.249 | 16.05 | -14.38 | 24 | 14 | 0.58 | spike |
| 24 | `bb_trend_rejoin` | ETHUSD | 1h | 308 | 65.58 | 1.086 | 12.46 | -26.78 | 24 | 14 | 0.58 | spike |
| 25 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 69.74 | 1.436 | 65.76 | -11.77 | 23 | 13 | 0.57 | spike |
| 26 | `bb_trend_rejoin` | ETHUSD | 1h | 125 | 72.0 | 1.335 | 36.25 | -20.49 | 23 | 13 | 0.57 | spike |
| 27 | `rsi2_regime` | ETHUSD | 1h | 192 | 68.75 | 1.155 | 17.09 | -18.76 | 23 | 13 | 0.57 | spike |
| 28 | `rsi2_regime` | ETHUSD | 1h | 256 | 67.58 | 1.099 | 15.89 | -26.87 | 23 | 13 | 0.57 | spike |
| 29 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.439 | 44.99 | -9.11 | 24 | 13 | 0.54 | spike |
| 30 | `rsi2_regime` | ETHUSD | 1h | 312 | 63.78 | 1.194 | 27.3 | -20.27 | 24 | 13 | 0.54 | spike |
| 31 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 69.06 | 1.21 | 18.97 | -12.63 | 24 | 13 | 0.54 | spike |
| 32 | `rsi2_regime` | ETHUSD | 1h | 238 | 66.81 | 1.111 | 15.83 | -24.77 | 24 | 13 | 0.54 | spike |
| 33 | `rsi2_regime` | ETHUSD | 1h | 84 | 73.81 | 1.326 | 17.69 | -13.08 | 21 | 11 | 0.52 | spike |
| 34 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 67.27 | 1.136 | 15.7 | -14.55 | 23 | 12 | 0.52 | spike |
| 35 | `bb_trend_rejoin` | ETHUSD | 1h | 143 | 72.03 | 1.619 | 68.11 | -8.9 | 24 | 12 | 0.50 | spike |
| 36 | `bb_trend_rejoin` | ETHUSD | 1h | 129 | 67.44 | 1.25 | 26.38 | -16.56 | 24 | 12 | 0.50 | spike |
| 37 | `rsi2_regime` | ETHUSD | 1h | 313 | 63.9 | 1.164 | 22.72 | -20.85 | 24 | 12 | 0.50 | spike |
| 38 | `bb_trend_rejoin` | ETHUSD | 1h | 166 | 63.25 | 1.12 | 9.26 | -22.8 | 24 | 12 | 0.50 | spike |
| 39 | `rsi2_regime` | ETHUSD | 1h | 72 | 70.83 | 1.232 | 17.81 | -16.81 | 23 | 11 | 0.48 | spike |
| 40 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.496 | 55.47 | -15.95 | 19 | 9 | 0.47 | spike |
| 41 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.459 | 54.25 | -17.34 | 19 | 9 | 0.47 | spike |
| 42 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.445 | 51.99 | -17.34 | 19 | 9 | 0.47 | spike |
| 43 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.383 | 44.31 | -16.74 | 19 | 9 | 0.47 | spike |
| 44 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.38 | 42.49 | -18.05 | 19 | 9 | 0.47 | spike |
| 45 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.38 | 42.49 | -18.05 | 19 | 9 | 0.47 | spike |
| 46 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.38 | 42.44 | -18.14 | 19 | 9 | 0.47 | spike |
| 47 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.373 | 41.52 | -18.05 | 19 | 9 | 0.47 | spike |
| 48 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.338 | 37.54 | -18.14 | 19 | 9 | 0.47 | spike |
| 49 | `rsi2_regime` | ETHUSD | 1h | 81 | 69.14 | 1.191 | 11.51 | -15.12 | 24 | 11 | 0.46 | spike |
| 50 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 64.24 | 1.046 | 2.22 | -17.04 | 24 | 11 | 0.46 | spike |
| 51 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.41 | 41.05 | -9.11 | 23 | 10 | 0.43 | spike |
| 52 | `bb_trend_rejoin` | ETHUSD | 1h | 131 | 64.89 | 1.275 | 29.4 | -16.1 | 23 | 10 | 0.43 | spike |
| 53 | `rsi2_regime` | ETHUSD | 1h | 53 | 75.47 | 1.752 | 19.94 | -11.18 | 23 | 10 | 0.43 | spike |
| 54 | `rsi2_regime` | ETHUSD | 1h | 185 | 67.57 | 1.104 | 11.45 | -18.82 | 23 | 10 | 0.43 | spike |
| 55 | `bb_trend_rejoin` | ETHUSD | 1h | 90 | 66.67 | 1.441 | 34.32 | -15.8 | 24 | 10 | 0.42 | spike |
| 56 | `rsi2_regime` | ETHUSD | 1h | 52 | 73.08 | 1.533 | 26.51 | -12.38 | 24 | 10 | 0.42 | spike |
| 57 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 67.69 | 1.297 | 39.49 | -11.77 | 23 | 9 | 0.39 | spike |
| 58 | `bb_trend_rejoin` | ETHUSD | 1h | 65 | 72.31 | 1.978 | 86.08 | -10.27 | 24 | 9 | 0.38 | spike |
| 59 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.507 | 56.02 | -16.71 | 16 | 6 | 0.38 | spike |
| 60 | `bb_trend_rejoin` | ETHUSD | 1h | 66 | 68.18 | 1.521 | 48.66 | -22.34 | 24 | 9 | 0.38 | spike |
| 61 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.413 | 48.03 | -16.0 | 16 | 6 | 0.38 | spike |
| 62 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.334 | 38.89 | -15.88 | 16 | 6 | 0.38 | spike |
| 63 | `rsi2_regime` | ETHUSD | 1h | 65 | 69.23 | 1.673 | 37.35 | -9.19 | 24 | 9 | 0.38 | spike |
| 64 | `rsi2_regime` | ETHUSD | 1h | 210 | 65.24 | 1.184 | 36.6 | -25.14 | 24 | 9 | 0.38 | spike |
| 65 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.304 | 34.88 | -20.64 | 16 | 6 | 0.38 | spike |
| 66 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.24 | 25.93 | -20.64 | 16 | 6 | 0.38 | spike |
| 67 | `bb_trend_rejoin` | ETHUSD | 1h | 174 | 67.82 | 1.115 | 11.55 | -22.5 | 24 | 9 | 0.38 | spike |
| 68 | `bb_trend_rejoin` | ETHUSD | 1h | 174 | 67.82 | 1.079 | 5.87 | -25.15 | 24 | 9 | 0.38 | spike |
| 69 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.622 | 42.71 | -11.83 | 23 | 8 | 0.35 | spike |
| 70 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.513 | 36.25 | -12.36 | 23 | 8 | 0.35 | spike |
| 71 | `rsi2_regime` | ETHUSD | 1h | 225 | 69.33 | 1.16 | 30.0 | -29.7 | 23 | 8 | 0.35 | spike |
| 72 | `rsi2_regime` | ETHUSD | 1h | 88 | 67.05 | 1.293 | 27.59 | -16.25 | 23 | 8 | 0.35 | spike |
| 73 | `rsi2_regime` | ETHUSD | 1h | 69 | 68.12 | 1.526 | 21.66 | -9.23 | 23 | 8 | 0.35 | spike |
| 74 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 63.74 | 1.595 | 42.99 | -10.09 | 24 | 8 | 0.33 | spike |
| 75 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 71.94 | 1.4 | 42.32 | -12.99 | 24 | 8 | 0.33 | spike |
| 76 | `bb_trend_rejoin` | ETHUSD | 1h | 95 | 68.42 | 1.324 | 37.11 | -17.06 | 24 | 8 | 0.33 | spike |
| 77 | `bb_trend_rejoin` | ETHUSD | 1h | 127 | 67.72 | 1.333 | 34.75 | -13.51 | 24 | 8 | 0.33 | spike |
| 78 | `bb_trend_rejoin` | ETHUSD | 1h | 140 | 68.57 | 1.275 | 28.39 | -14.84 | 24 | 8 | 0.33 | spike |
| 79 | `bb_trend_rejoin` | ETHUSD | 1h | 140 | 68.57 | 1.275 | 28.39 | -14.84 | 24 | 8 | 0.33 | spike |
| 80 | `rsi2_regime` | ETHUSD | 1h | 52 | 78.85 | 1.831 | 23.38 | -8.35 | 21 | 7 | 0.33 | spike |
| 81 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 63.55 | 1.128 | 16.68 | -17.64 | 24 | 8 | 0.33 | spike |
| 82 | `bb_trend_rejoin` | ETHUSD | 1h | 276 | 67.39 | 1.075 | 9.93 | -23.52 | 24 | 8 | 0.33 | spike |
| 83 | `rsi2_regime` | ETHUSD | 1h | 61 | 67.21 | 1.592 | 41.38 | -13.48 | 22 | 7 | 0.32 | spike |
| 84 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.336 | 36.49 | -14.43 | 19 | 6 | 0.32 | spike |
| 85 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.259 | 28.16 | -19.33 | 19 | 6 | 0.32 | spike |
| 86 | `zscore_revert` | ETHUSD | 1h | 132 | 64.39 | 1.261 | 27.64 | -21.2 | 19 | 6 | 0.32 | spike |
| 87 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.251 | 26.47 | -16.56 | 19 | 6 | 0.32 | spike |
| 88 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.251 | 26.47 | -16.56 | 19 | 6 | 0.32 | spike |
| 89 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 68.75 | 1.534 | 22.76 | -13.89 | 23 | 7 | 0.30 | spike |
| 90 | `bb_trend_rejoin` | ETHUSD | 1h | 78 | 71.79 | 1.129 | 4.54 | -13.02 | 23 | 7 | 0.30 | spike |
| 91 | `bb_trend_rejoin` | ETHUSD | 1h | 66 | 68.18 | 1.573 | 51.35 | -18.39 | 24 | 7 | 0.29 | spike |
| 92 | `rsi2_regime` | ETHUSD | 1h | 60 | 68.33 | 1.348 | 24.64 | -13.61 | 24 | 7 | 0.29 | spike |
| 93 | `rsi2_regime` | ETHUSD | 1h | 90 | 67.78 | 1.195 | 16.19 | -17.99 | 24 | 7 | 0.29 | spike |
| 94 | `bb_trend_rejoin` | ETHUSD | 1h | 174 | 67.82 | 1.089 | 7.7 | -22.5 | 24 | 7 | 0.29 | spike |
| 95 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 63.55 | 1.044 | 2.88 | -24.61 | 24 | 7 | 0.29 | spike |
| 96 | `rsi2_regime` | ETHUSD | 1h | 53 | 67.92 | 1.899 | 23.86 | -11.19 | 23 | 6 | 0.26 | spike |
| 97 | `bb_trend_rejoin` | ETHUSD | 1h | 130 | 63.08 | 1.241 | 24.15 | -15.22 | 24 | 6 | 0.25 | spike |
| 98 | `bb_trend_rejoin` | ETHUSD | 1h | 130 | 63.08 | 1.236 | 23.6 | -14.79 | 24 | 6 | 0.25 | spike |
| 99 | `rsi2_regime` | ETHUSD | 1h | 52 | 75.0 | 1.938 | 40.05 | -9.04 | 21 | 5 | 0.24 | spike |
| 100 | `bb_trend_rejoin` | DOGEUSD | 1h | 66 | 68.18 | 1.719 | 35.07 | -14.04 | 23 | 5 | 0.22 | spike |
| 101 | `bb_trend_rejoin` | ETHUSD | 1h | 148 | 64.19 | 1.303 | 33.76 | -9.74 | 23 | 5 | 0.22 | spike |
| 102 | `bb_trend_rejoin` | DOGEUSD | 1h | 66 | 68.18 | 1.577 | 26.68 | -14.04 | 23 | 5 | 0.22 | spike |
| 103 | `bb_trend_rejoin` | ETHUSD | 1h | 130 | 61.54 | 1.251 | 24.9 | -13.96 | 23 | 5 | 0.22 | spike |
| 104 | `zscore_revert` | ETHUSD | 1h | 66 | 68.18 | 1.513 | 49.19 | -22.34 | 19 | 4 | 0.21 | spike |
| 105 | `rsi2_regime` | ETHUSD | 1h | 86 | 67.44 | 1.183 | 9.57 | -18.63 | 19 | 4 | 0.21 | spike |
| 106 | `rsi2_regime` | ETHUSD | 1h | 60 | 68.33 | 1.337 | 23.98 | -14.65 | 24 | 5 | 0.21 | spike |
| 107 | `bb_trend_rejoin` | ETHUSD | 1h | 96 | 67.71 | 1.134 | 11.2 | -28.85 | 24 | 5 | 0.21 | spike |
| 108 | `zscore_revert` | ETHUSD | 1h | 137 | 62.77 | 1.484 | 56.84 | -11.09 | 16 | 3 | 0.19 | spike |
| 109 | `zscore_revert` | ETHUSD | 1h | 137 | 63.5 | 1.447 | 51.88 | -11.09 | 16 | 3 | 0.19 | spike |
| 110 | `zscore_revert` | ETHUSD | 1h | 137 | 63.5 | 1.447 | 51.88 | -11.09 | 16 | 3 | 0.19 | spike |
| 111 | `bb_trend_rejoin` | ETHUSD | 1h | 92 | 61.96 | 1.404 | 26.13 | -10.74 | 23 | 4 | 0.17 | spike |
| 112 | `bb_trend_rejoin` | ETHUSD | 1h | 142 | 66.2 | 1.233 | 22.69 | -11.21 | 23 | 4 | 0.17 | spike |
| 113 | `rsi2_regime` | ETHUSD | 1h | 54 | 64.81 | 1.463 | 7.7 | -5.89 | 24 | 4 | 0.17 | spike |
| 114 | `bb_trend_rejoin` | ADAUSD | 1h | 58 | 67.24 | 1.064 | 1.53 | -15.49 | 24 | 2 | 0.08 | spike |
| 115 | `rsi2_regime` | ETHUSD | 1h | 50 | 66.0 | 1.229 | 6.42 | -12.23 | 24 | 0 | 0.00 | spike |

## Survivors (plateau)

- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":100,"slope_bars":5,"exit_sma":8}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":5.0}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":125,"slope_bars":80,"exit_sma":13}` | exit=`{"sl_atr":5.0,"tp_atr":7.0,"timeout":48}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":80,"exit_sma":8}` | exit=`{"sl_atr":5.0,"timeout":96}` | frac=0.90
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":7,"sma_trend":75,"slope_bars":100,"exit_sma":13}` | exit=`{"sl_atr":4.0,"tp_atr":10.0,"timeout":36}` | frac=0.87
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":5,"exit_sma":8}` | exit=`{"sl_atr":3.0,"tp_atr":4.0,"timeout":18}` | frac=0.83
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":2.0,"trend":100}` | exit=`{"sl_atr":2.5,"tp_atr":5.0,"trail_atr":5.0,"timeout":96}` | frac=0.78
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":1.75,"trend":100}` | exit=`{"sl_atr":2.5,"tp_atr":6.0,"timeout":120}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":10,"sma_trend":75,"slope_bars":15,"exit_sma":8}` | exit=`{"sl_atr":5.0,"tp_atr":6.0,"timeout":96}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":100,"slope_bars":100,"exit_sma":13}` | exit=`{"sl_atr":5.0,"tp_atr":6.0,"trail_atr":4.0,"timeout":120}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":100,"slope_bars":20,"exit_sma":21}` | exit=`{"sl_atr":3.0,"tp_atr":7.0,"timeout":168}` | frac=0.77
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":20,"sma_trend":100,"slope_bars":5,"exit_sma":13}` | exit=`{"sl_atr":4.0,"tp_atr":7.0,"timeout":24}` | frac=0.76
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":12,"sma_trend":100,"slope_bars":20,"exit_sma":13}` | exit=`{"sl_atr":4.0,"tp_atr":10.0,"trail_atr":5.0,"timeout":96}` | frac=0.76
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":125,"slope_bars":100,"exit_sma":8}` | exit=`{"sl_atr":3.5,"tp_atr":10.0,"trail_atr":3.5,"timeout":48}` | frac=0.75
