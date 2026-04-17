# Parameter-plateau stability report

- Input: 1095 unique WF grails
- Jitter: +/-20% in 10% steps on every numeric param and exit-cfg value
- Plateau threshold: >= 75% of neighbors still pass WF filter

| Rank | Strategy | Asset | TF | Trades | WR | PF | Ret% | DD% | Neighbors | Passed | Frac | Verdict |
|------|----------|-------|----|--------|----|----|------|-----|-----------|--------|------|---------|
| 1 | `rsi2_regime` | ETHUSD | 1h | 259 | 71.04 | 1.291 | 76.41 | -28.01 | 23 | 23 | 1.00 | PLATEAU |
| 2 | `rsi2_regime` | ETHUSD | 1h | 215 | 72.09 | 1.357 | 75.68 | -27.82 | 19 | 19 | 1.00 | PLATEAU |
| 3 | `rsi2_regime` | ETHUSD | 1h | 128 | 69.53 | 1.678 | 74.26 | -17.87 | 21 | 21 | 1.00 | PLATEAU |
| 4 | `rsi2_regime` | ETHUSD | 1h | 149 | 73.15 | 1.515 | 70.0 | -15.65 | 22 | 22 | 1.00 | PLATEAU |
| 5 | `rsi2_regime` | ETHUSD | 1h | 101 | 70.3 | 1.618 | 62.02 | -13.27 | 22 | 22 | 1.00 | PLATEAU |
| 6 | `rsi2_regime` | ETHUSD | 1h | 139 | 69.78 | 1.439 | 61.29 | -16.36 | 23 | 23 | 1.00 | PLATEAU |
| 7 | `rsi2_regime` | ETHUSD | 1h | 88 | 70.45 | 1.6 | 55.44 | -14.25 | 23 | 23 | 1.00 | PLATEAU |
| 8 | `rsi2_regime` | ETHUSD | 1h | 128 | 69.53 | 1.504 | 55.33 | -16.91 | 19 | 19 | 1.00 | PLATEAU |
| 9 | `rsi2_regime` | ETHUSD | 1h | 254 | 73.62 | 1.258 | 51.74 | -18.15 | 21 | 21 | 1.00 | PLATEAU |
| 10 | `rsi2_regime` | ETHUSD | 1h | 232 | 70.26 | 1.286 | 50.87 | -24.89 | 23 | 23 | 1.00 | PLATEAU |
| 11 | `rsi2_regime` | ETHUSD | 1h | 129 | 68.99 | 1.381 | 50.31 | -20.81 | 23 | 23 | 1.00 | PLATEAU |
| 12 | `rsi2_regime` | ETHUSD | 1h | 128 | 69.53 | 1.445 | 49.72 | -19.97 | 22 | 22 | 1.00 | PLATEAU |
| 13 | `rsi2_regime` | ETHUSD | 1h | 181 | 67.4 | 1.307 | 48.91 | -16.55 | 24 | 24 | 1.00 | PLATEAU |
| 14 | `rsi2_regime` | ETHUSD | 1h | 66 | 74.24 | 1.8 | 46.83 | -11.92 | 22 | 22 | 1.00 | PLATEAU |
| 15 | `rsi2_regime` | ETHUSD | 1h | 79 | 72.15 | 1.584 | 46.36 | -11.46 | 24 | 24 | 1.00 | PLATEAU |
| 16 | `rsi2_regime` | ETHUSD | 1h | 77 | 66.23 | 1.593 | 38.1 | -12.53 | 23 | 23 | 1.00 | PLATEAU |
| 17 | `rsi2_regime` | ETHUSD | 1h | 71 | 73.24 | 1.534 | 37.77 | -12.82 | 24 | 24 | 1.00 | PLATEAU |
| 18 | `rsi2_regime` | ETHUSD | 1h | 239 | 65.69 | 1.224 | 33.68 | -23.44 | 23 | 23 | 1.00 | PLATEAU |
| 19 | `rsi2_regime` | ETHUSD | 1h | 147 | 68.71 | 1.265 | 28.73 | -22.89 | 16 | 16 | 1.00 | PLATEAU |
| 20 | `rsi2_regime` | ETHUSD | 1h | 260 | 68.08 | 1.167 | 28.54 | -24.3 | 19 | 19 | 1.00 | PLATEAU |
| 21 | `rsi2_regime` | ETHUSD | 1h | 158 | 70.89 | 1.277 | 26.96 | -13.29 | 22 | 22 | 1.00 | PLATEAU |
| 22 | `rsi2_regime` | ETHUSD | 1h | 66 | 71.21 | 1.504 | 26.31 | -11.82 | 23 | 23 | 1.00 | PLATEAU |
| 23 | `rsi2_regime` | ETHUSD | 1h | 103 | 66.02 | 1.277 | 23.66 | -16.79 | 22 | 22 | 1.00 | PLATEAU |
| 24 | `rsi2_regime` | ETHUSD | 1h | 118 | 66.1 | 1.203 | 23.4 | -16.06 | 24 | 24 | 1.00 | PLATEAU |
| 25 | `rsi2_regime` | ETHUSD | 1h | 154 | 68.18 | 1.196 | 21.86 | -19.03 | 24 | 24 | 1.00 | PLATEAU |
| 26 | `rsi2_regime` | ETHUSD | 1h | 157 | 70.06 | 1.194 | 21.66 | -17.3 | 22 | 22 | 1.00 | PLATEAU |
| 27 | `rsi2_regime` | ETHUSD | 1h | 116 | 72.41 | 1.224 | 18.69 | -23.96 | 23 | 23 | 1.00 | PLATEAU |
| 28 | `rsi2_regime` | ETHUSD | 1h | 116 | 74.14 | 1.225 | 14.99 | -11.53 | 23 | 23 | 1.00 | PLATEAU |
| 29 | `rsi2_regime` | ETHUSD | 1h | 157 | 70.06 | 1.113 | 10.02 | -19.12 | 19 | 19 | 1.00 | PLATEAU |
| 30 | `rsi2_regime` | ETHUSD | 1h | 128 | 68.75 | 1.474 | 61.38 | -18.16 | 23 | 22 | 0.96 | PLATEAU |
| 31 | `rsi2_regime` | ETHUSD | 1h | 98 | 74.49 | 1.6 | 34.56 | -10.97 | 23 | 22 | 0.96 | PLATEAU |
| 32 | `rsi2_regime` | ETHUSD | 1h | 102 | 74.51 | 1.497 | 48.48 | -13.41 | 21 | 20 | 0.95 | PLATEAU |
| 33 | `rsi2_regime` | ETHUSD | 1h | 559 | 65.3 | 1.095 | 27.65 | -28.42 | 21 | 20 | 0.95 | PLATEAU |
| 34 | `rsi2_regime` | ETHUSD | 1h | 123 | 71.54 | 1.411 | 38.86 | -17.02 | 19 | 18 | 0.95 | PLATEAU |
| 35 | `rsi2_regime` | ETHUSD | 1h | 148 | 69.59 | 1.104 | 9.31 | -21.14 | 24 | 22 | 0.92 | PLATEAU |
| 36 | `rsi2_regime` | ETHUSD | 1h | 278 | 67.99 | 1.285 | 69.32 | -25.19 | 23 | 21 | 0.91 | PLATEAU |
| 37 | `rsi2_regime` | ETHUSD | 1h | 277 | 67.87 | 1.263 | 62.23 | -24.86 | 23 | 21 | 0.91 | PLATEAU |
| 38 | `rsi2_regime` | ETHUSD | 1h | 102 | 74.51 | 1.608 | 58.22 | -12.09 | 23 | 21 | 0.91 | PLATEAU |
| 39 | `rsi2_regime` | ETHUSD | 1h | 140 | 67.86 | 1.333 | 48.66 | -20.81 | 23 | 21 | 0.91 | PLATEAU |
| 40 | `rsi2_regime` | ETHUSD | 1h | 228 | 67.98 | 1.24 | 46.06 | -29.34 | 23 | 21 | 0.91 | PLATEAU |
| 41 | `rsi2_regime` | ETHUSD | 1h | 117 | 71.79 | 1.354 | 40.43 | -17.52 | 23 | 21 | 0.91 | PLATEAU |
| 42 | `rsi2_regime` | ETHUSD | 1h | 120 | 70.83 | 1.271 | 28.36 | -15.03 | 23 | 21 | 0.91 | PLATEAU |
| 43 | `rsi2_regime` | ETHUSD | 1h | 115 | 65.22 | 1.086 | 6.18 | -17.63 | 23 | 21 | 0.91 | PLATEAU |
| 44 | `rsi2_regime` | ETHUSD | 1h | 220 | 67.73 | 1.277 | 47.9 | -20.07 | 22 | 20 | 0.91 | PLATEAU |
| 45 | `rsi2_regime` | ETHUSD | 1h | 154 | 68.83 | 1.193 | 23.86 | -16.72 | 22 | 20 | 0.91 | PLATEAU |
| 46 | `rsi2_regime` | ETHUSD | 1h | 97 | 77.32 | 1.912 | 79.03 | -13.41 | 21 | 19 | 0.90 | PLATEAU |
| 47 | `rsi2_regime` | ETHUSD | 1h | 127 | 69.29 | 1.25 | 28.51 | -16.11 | 21 | 19 | 0.90 | PLATEAU |
| 48 | `rsi2_regime` | ETHUSD | 1h | 117 | 74.36 | 1.409 | 27.65 | -17.36 | 21 | 19 | 0.90 | PLATEAU |
| 49 | `rsi2_regime` | ETHUSD | 1h | 84 | 72.62 | 1.558 | 26.54 | -10.79 | 21 | 19 | 0.90 | PLATEAU |
| 50 | `rsi2_regime` | ETHUSD | 1h | 148 | 68.92 | 1.184 | 19.4 | -18.26 | 21 | 19 | 0.90 | PLATEAU |
| 51 | `rsi2_regime` | ETHUSD | 1h | 107 | 72.9 | 1.877 | 93.94 | -14.84 | 19 | 17 | 0.89 | PLATEAU |
| 52 | `rsi2_regime` | ETHUSD | 1h | 301 | 66.78 | 1.288 | 57.76 | -20.0 | 19 | 17 | 0.89 | PLATEAU |
| 53 | `rsi2_regime` | ETHUSD | 1h | 112 | 73.21 | 1.645 | 68.59 | -16.55 | 16 | 14 | 0.88 | PLATEAU |
| 54 | `rsi2_regime` | ETHUSD | 1h | 142 | 71.13 | 1.389 | 52.15 | -14.22 | 24 | 21 | 0.88 | PLATEAU |
| 55 | `rsi2_regime` | ETHUSD | 1h | 157 | 70.7 | 1.194 | 20.71 | -21.44 | 16 | 14 | 0.88 | PLATEAU |
| 56 | `rsi2_regime` | ETHUSD | 1h | 297 | 71.72 | 1.341 | 76.85 | -23.43 | 23 | 20 | 0.87 | PLATEAU |
| 57 | `rsi2_regime` | ETHUSD | 1h | 219 | 70.78 | 1.321 | 69.23 | -26.7 | 23 | 20 | 0.87 | PLATEAU |
| 58 | `rsi2_regime` | ETHUSD | 1h | 117 | 70.09 | 1.69 | 63.14 | -21.46 | 23 | 20 | 0.87 | PLATEAU |
| 59 | `rsi2_regime` | ETHUSD | 1h | 95 | 76.84 | 1.687 | 35.5 | -8.86 | 23 | 20 | 0.87 | PLATEAU |
| 60 | `rsi2_regime` | ETHUSD | 1h | 94 | 71.28 | 1.418 | 26.82 | -10.37 | 23 | 20 | 0.87 | PLATEAU |
| 61 | `rsi2_regime` | ETHUSD | 1h | 112 | 70.54 | 1.232 | 23.63 | -15.75 | 22 | 19 | 0.86 | PLATEAU |
| 62 | `rsi2_regime` | ETHUSD | 1h | 148 | 71.62 | 1.275 | 30.37 | -16.92 | 24 | 20 | 0.83 | PLATEAU |
| 63 | `rsi2_regime` | ETHUSD | 1h | 124 | 70.97 | 1.263 | 23.85 | -15.07 | 24 | 20 | 0.83 | PLATEAU |
| 64 | `rsi2_regime` | ETHUSD | 1h | 63 | 73.02 | 1.443 | 16.92 | -8.16 | 24 | 20 | 0.83 | PLATEAU |
| 65 | `rsi2_regime` | ETHUSD | 1h | 167 | 63.47 | 1.142 | 14.39 | -15.98 | 24 | 20 | 0.83 | PLATEAU |
| 66 | `rsi2_regime` | ETHUSD | 1h | 300 | 68.67 | 1.057 | 4.17 | -23.19 | 24 | 20 | 0.83 | PLATEAU |
| 67 | `rsi2_regime` | ETHUSD | 1h | 174 | 75.29 | 1.694 | 127.02 | -17.09 | 23 | 19 | 0.83 | PLATEAU |
| 68 | `rsi2_regime` | ETHUSD | 1h | 139 | 67.63 | 1.526 | 62.18 | -22.44 | 23 | 19 | 0.83 | PLATEAU |
| 69 | `rsi2_regime` | ETHUSD | 1h | 96 | 67.71 | 1.583 | 60.87 | -18.27 | 23 | 19 | 0.83 | PLATEAU |
| 70 | `rsi2_regime` | ETHUSD | 1h | 114 | 69.3 | 1.625 | 58.37 | -18.31 | 23 | 19 | 0.83 | PLATEAU |
| 71 | `rsi2_regime` | ETHUSD | 1h | 345 | 66.96 | 1.205 | 46.83 | -18.84 | 23 | 19 | 0.83 | PLATEAU |
| 72 | `rsi2_regime` | ETHUSD | 1h | 113 | 72.57 | 1.373 | 41.01 | -17.52 | 23 | 19 | 0.83 | PLATEAU |
| 73 | `rsi2_regime` | ETHUSD | 1h | 236 | 69.07 | 1.154 | 29.09 | -17.83 | 23 | 19 | 0.83 | PLATEAU |
| 74 | `rsi2_regime` | ETHUSD | 1h | 230 | 66.96 | 1.195 | 27.2 | -20.86 | 23 | 19 | 0.83 | PLATEAU |
| 75 | `rsi2_regime` | ETHUSD | 1h | 314 | 66.56 | 1.143 | 25.23 | -20.56 | 23 | 19 | 0.83 | PLATEAU |
| 76 | `rsi2_regime` | ETHUSD | 1h | 80 | 66.25 | 1.491 | 23.83 | -14.49 | 23 | 19 | 0.83 | PLATEAU |
| 77 | `rsi2_regime` | ETHUSD | 1h | 127 | 73.23 | 1.297 | 22.32 | -11.7 | 23 | 19 | 0.83 | PLATEAU |
| 78 | `rsi2_regime` | ETHUSD | 1h | 119 | 67.23 | 1.135 | 8.64 | -17.52 | 23 | 19 | 0.83 | PLATEAU |
| 79 | `rsi2_regime` | ETHUSD | 1h | 314 | 64.33 | 1.094 | 12.37 | -21.61 | 22 | 18 | 0.82 | PLATEAU |
| 80 | `rsi2_regime` | ETHUSD | 1h | 248 | 66.13 | 1.212 | 53.24 | -29.58 | 21 | 17 | 0.81 | PLATEAU |
| 81 | `rsi2_regime` | ETHUSD | 1h | 175 | 70.86 | 1.304 | 52.32 | -16.19 | 21 | 17 | 0.81 | PLATEAU |
| 82 | `rsi2_regime` | ETHUSD | 1h | 115 | 71.3 | 1.511 | 35.51 | -12.95 | 21 | 17 | 0.81 | PLATEAU |
| 83 | `rsi2_regime` | ETHUSD | 1h | 150 | 72.67 | 1.64 | 62.89 | -16.16 | 24 | 19 | 0.79 | PLATEAU |
| 84 | `bb_trend_rejoin` | ETHUSD | 1h | 127 | 70.87 | 1.438 | 52.27 | -17.34 | 24 | 19 | 0.79 | PLATEAU |
| 85 | `bb_trend_rejoin` | ETHUSD | 1h | 127 | 70.87 | 1.388 | 44.79 | -17.34 | 24 | 19 | 0.79 | PLATEAU |
| 86 | `bb_trend_rejoin` | ETHUSD | 1h | 183 | 71.58 | 1.295 | 42.92 | -18.52 | 24 | 19 | 0.79 | PLATEAU |
| 87 | `bb_trend_rejoin` | ETHUSD | 1h | 131 | 66.41 | 1.334 | 38.89 | -15.88 | 24 | 19 | 0.79 | PLATEAU |
| 88 | `rsi2_regime` | ETHUSD | 1h | 100 | 67.0 | 1.396 | 34.52 | -20.74 | 24 | 19 | 0.79 | PLATEAU |
| 89 | `rsi2_regime` | ETHUSD | 1h | 124 | 72.58 | 1.275 | 30.95 | -19.09 | 24 | 19 | 0.79 | PLATEAU |
| 90 | `bb_trend_rejoin` | ETHUSD | 1h | 183 | 71.04 | 1.206 | 27.48 | -20.18 | 24 | 19 | 0.79 | PLATEAU |
| 91 | `rsi2_regime` | ETHUSD | 1h | 271 | 65.68 | 1.121 | 17.9 | -19.92 | 24 | 19 | 0.79 | PLATEAU |
| 92 | `rsi2_regime` | ETHUSD | 1h | 351 | 70.37 | 1.094 | 15.48 | -25.94 | 24 | 19 | 0.79 | PLATEAU |
| 93 | `rsi2_regime` | ETHUSD | 1h | 213 | 69.48 | 1.297 | 58.68 | -16.79 | 23 | 18 | 0.78 | PLATEAU |
| 94 | `rsi2_regime` | ETHUSD | 1h | 213 | 69.01 | 1.272 | 50.95 | -16.79 | 23 | 18 | 0.78 | PLATEAU |
| 95 | `rsi2_regime` | ETHUSD | 1h | 94 | 70.21 | 1.552 | 43.36 | -17.44 | 23 | 18 | 0.78 | PLATEAU |
| 96 | `bb_trend_rejoin` | ETHUSD | 1h | 127 | 70.87 | 1.38 | 43.3 | -17.34 | 23 | 18 | 0.78 | PLATEAU |
| 97 | `rsi2_regime` | ETHUSD | 1h | 165 | 66.67 | 1.265 | 36.95 | -15.08 | 23 | 18 | 0.78 | PLATEAU |
| 98 | `bb_trend_rejoin` | ETHUSD | 1h | 183 | 71.04 | 1.244 | 33.95 | -20.18 | 23 | 18 | 0.78 | PLATEAU |
| 99 | `rsi2_regime` | ETHUSD | 1h | 72 | 69.44 | 1.55 | 31.88 | -16.67 | 23 | 18 | 0.78 | PLATEAU |
| 100 | `bb_trend_rejoin` | ETHUSD | 1h | 183 | 71.04 | 1.227 | 30.63 | -21.76 | 23 | 18 | 0.78 | PLATEAU |
| 101 | `rsi2_regime` | ETHUSD | 1h | 72 | 69.44 | 1.395 | 28.37 | -17.07 | 23 | 18 | 0.78 | PLATEAU |
| 102 | `bb_trend_rejoin` | ETHUSD | 1h | 196 | 66.84 | 1.186 | 27.97 | -17.71 | 23 | 18 | 0.78 | PLATEAU |
| 103 | `bb_trend_rejoin` | ETHUSD | 1h | 183 | 71.04 | 1.197 | 25.91 | -22.3 | 23 | 18 | 0.78 | PLATEAU |
| 104 | `rsi2_regime` | ETHUSD | 1h | 146 | 69.18 | 1.272 | 23.89 | -15.26 | 23 | 18 | 0.78 | PLATEAU |
| 105 | `rsi2_regime` | ETHUSD | 1h | 116 | 71.55 | 1.251 | 20.81 | -24.4 | 23 | 18 | 0.78 | PLATEAU |
| 106 | `rsi2_regime` | ETHUSD | 1h | 171 | 65.5 | 1.159 | 19.85 | -22.16 | 23 | 18 | 0.78 | PLATEAU |
| 107 | `bb_trend_rejoin` | ETHUSD | 1h | 196 | 66.33 | 1.137 | 18.63 | -17.66 | 23 | 18 | 0.78 | PLATEAU |
| 108 | `rsi2_regime` | ETHUSD | 1h | 66 | 72.73 | 1.418 | 18.27 | -9.79 | 23 | 18 | 0.78 | PLATEAU |
| 109 | `bb_trend_rejoin` | ETHUSD | 1h | 196 | 65.82 | 1.115 | 15.04 | -17.66 | 23 | 18 | 0.78 | PLATEAU |
| 110 | `rsi2_regime` | ETHUSD | 1h | 100 | 70.0 | 1.207 | 14.1 | -12.7 | 23 | 18 | 0.78 | PLATEAU |
| 111 | `rsi2_regime` | ETHUSD | 1h | 91 | 70.33 | 1.221 | 12.51 | -20.0 | 23 | 18 | 0.78 | PLATEAU |
| 112 | `rsi2_regime` | ETHUSD | 1h | 72 | 69.44 | 1.221 | 8.52 | -13.35 | 23 | 18 | 0.78 | PLATEAU |
| 113 | `rsi2_regime` | ETHUSD | 1h | 72 | 69.44 | 1.221 | 8.52 | -13.35 | 23 | 18 | 0.78 | PLATEAU |
| 114 | `rsi2_regime` | ETHUSD | 1h | 147 | 68.71 | 1.38 | 61.65 | -17.85 | 22 | 17 | 0.77 | PLATEAU |
| 115 | `rsi2_regime` | ETHUSD | 1h | 126 | 71.43 | 1.478 | 38.35 | -11.53 | 22 | 17 | 0.77 | PLATEAU |
| 116 | `rsi2_regime` | ETHUSD | 1h | 126 | 71.43 | 1.473 | 38.03 | -11.64 | 22 | 17 | 0.77 | PLATEAU |
| 117 | `rsi2_regime` | ETHUSD | 1h | 105 | 71.43 | 1.65 | 70.72 | -16.16 | 21 | 16 | 0.76 | PLATEAU |
| 118 | `rsi2_regime` | ETHUSD | 1h | 288 | 68.06 | 1.219 | 52.92 | -22.63 | 21 | 16 | 0.76 | PLATEAU |
| 119 | `rsi2_regime` | ETHUSD | 1h | 103 | 66.99 | 1.361 | 37.16 | -12.0 | 21 | 16 | 0.76 | PLATEAU |
| 120 | `rsi2_regime` | ETHUSD | 1h | 72 | 69.44 | 1.404 | 23.12 | -19.08 | 21 | 16 | 0.76 | PLATEAU |
| 121 | `rsi2_regime` | ETHUSD | 1h | 72 | 68.06 | 1.162 | 5.93 | -15.92 | 21 | 16 | 0.76 | PLATEAU |
| 122 | `rsi2_regime` | ETHUSD | 1h | 357 | 70.59 | 1.287 | 84.62 | -26.41 | 24 | 18 | 0.75 | PLATEAU |
| 123 | `rsi2_regime` | ETHUSD | 1h | 79 | 72.15 | 1.59 | 46.1 | -12.66 | 24 | 18 | 0.75 | PLATEAU |
| 124 | `rsi2_regime` | ETHUSD | 1h | 135 | 69.63 | 1.177 | 13.78 | -16.87 | 24 | 18 | 0.75 | PLATEAU |
| 125 | `rsi2_regime` | ETHUSD | 1h | 152 | 69.74 | 1.09 | 7.47 | -16.96 | 24 | 18 | 0.75 | PLATEAU |
| 126 | `rsi2_regime` | ETHUSD | 1h | 145 | 68.97 | 1.398 | 59.2 | -26.09 | 23 | 17 | 0.74 | spike |
| 127 | `rsi2_regime` | ETHUSD | 1h | 233 | 68.24 | 1.164 | 32.61 | -23.77 | 23 | 17 | 0.74 | spike |
| 128 | `rsi2_regime` | ETHUSD | 1h | 77 | 64.94 | 1.427 | 27.58 | -10.51 | 23 | 17 | 0.74 | spike |
| 129 | `rsi2_regime` | ETHUSD | 1h | 234 | 68.8 | 1.167 | 27.11 | -20.62 | 23 | 17 | 0.74 | spike |
| 130 | `rsi2_regime` | ETHUSD | 1h | 142 | 68.31 | 1.272 | 23.49 | -17.14 | 23 | 17 | 0.74 | spike |
| 131 | `rsi2_regime` | ETHUSD | 1h | 166 | 70.48 | 1.18 | 23.13 | -16.07 | 23 | 17 | 0.74 | spike |
| 132 | `rsi2_regime` | ETHUSD | 1h | 125 | 68.8 | 1.2 | 22.68 | -12.71 | 23 | 17 | 0.74 | spike |
| 133 | `rsi2_regime` | ETHUSD | 1h | 129 | 71.32 | 1.264 | 20.57 | -12.73 | 23 | 17 | 0.74 | spike |
| 134 | `rsi2_regime` | ETHUSD | 1h | 123 | 69.11 | 1.197 | 15.2 | -17.13 | 19 | 14 | 0.74 | spike |
| 135 | `rsi2_regime` | ETHUSD | 1h | 73 | 69.86 | 1.327 | 12.43 | -13.74 | 19 | 14 | 0.74 | spike |
| 136 | `rsi2_regime` | ETHUSD | 1h | 130 | 70.77 | 1.177 | 12.41 | -12.02 | 19 | 14 | 0.74 | spike |
| 137 | `rsi2_regime` | ETHUSD | 1h | 127 | 68.5 | 1.146 | 10.88 | -17.13 | 19 | 14 | 0.74 | spike |
| 138 | `rsi2_regime` | ETHUSD | 1h | 154 | 70.13 | 1.294 | 36.89 | -20.91 | 22 | 16 | 0.73 | spike |
| 139 | `bb_trend_rejoin` | ETHUSD | 1h | 196 | 66.33 | 1.173 | 24.88 | -19.33 | 22 | 16 | 0.73 | spike |
| 140 | `bb_trend_rejoin` | ETHUSD | 1h | 196 | 65.82 | 1.136 | 18.37 | -19.33 | 22 | 16 | 0.73 | spike |
| 141 | `rsi2_regime` | ETHUSD | 1h | 484 | 64.26 | 1.16 | 42.53 | -25.95 | 21 | 15 | 0.71 | spike |
| 142 | `rsi2_regime` | ETHUSD | 1h | 210 | 70.0 | 1.271 | 35.25 | -17.7 | 21 | 15 | 0.71 | spike |
| 143 | `rsi2_regime` | ETHUSD | 1h | 67 | 71.64 | 1.541 | 21.75 | -13.16 | 21 | 15 | 0.71 | spike |
| 144 | `rsi2_regime` | ETHUSD | 1h | 198 | 70.2 | 1.29 | 48.52 | -15.43 | 24 | 17 | 0.71 | spike |
| 145 | `bb_trend_rejoin` | ETHUSD | 1h | 126 | 72.22 | 1.404 | 46.74 | -16.74 | 24 | 17 | 0.71 | spike |
| 146 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 72.66 | 1.348 | 33.69 | -12.42 | 24 | 17 | 0.71 | spike |
| 147 | `rsi2_regime` | ETHUSD | 1h | 145 | 71.03 | 1.288 | 25.3 | -17.25 | 24 | 17 | 0.71 | spike |
| 148 | `rsi2_regime` | ETHUSD | 1h | 65 | 72.31 | 1.294 | 12.3 | -8.03 | 24 | 17 | 0.71 | spike |
| 149 | `rsi2_regime` | ETHUSD | 1h | 137 | 69.34 | 1.072 | 4.61 | -21.92 | 24 | 17 | 0.71 | spike |
| 150 | `rsi2_regime` | ETHUSD | 1h | 239 | 70.71 | 1.425 | 112.38 | -18.56 | 23 | 16 | 0.70 | spike |
| 151 | `rsi2_regime` | ETHUSD | 1h | 79 | 77.22 | 2.044 | 64.41 | -13.41 | 23 | 16 | 0.70 | spike |
| 152 | `rsi2_regime` | ETHUSD | 1h | 336 | 72.92 | 1.213 | 53.12 | -21.22 | 23 | 16 | 0.70 | spike |
| 153 | `bb_trend_rejoin` | ETHUSD | 1h | 127 | 70.87 | 1.424 | 48.82 | -17.34 | 23 | 16 | 0.70 | spike |
| 154 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 71.22 | 1.334 | 31.8 | -15.12 | 23 | 16 | 0.70 | spike |
| 155 | `rsi2_regime` | ETHUSD | 1h | 229 | 67.25 | 1.158 | 29.74 | -29.39 | 23 | 16 | 0.70 | spike |
| 156 | `rsi2_regime` | ETHUSD | 1h | 140 | 66.43 | 1.244 | 28.8 | -25.06 | 23 | 16 | 0.70 | spike |
| 157 | `rsi2_regime` | ETHUSD | 1h | 137 | 68.61 | 1.29 | 25.04 | -20.83 | 23 | 16 | 0.70 | spike |
| 158 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 71.94 | 1.242 | 22.15 | -17.08 | 23 | 16 | 0.70 | spike |
| 159 | `rsi2_regime` | ETHUSD | 1h | 134 | 70.15 | 1.266 | 20.81 | -11.7 | 23 | 16 | 0.70 | spike |
| 160 | `rsi2_regime` | ETHUSD | 1h | 137 | 67.15 | 1.221 | 18.46 | -20.36 | 23 | 16 | 0.70 | spike |
| 161 | `bb_trend_rejoin` | ETHUSD | 1h | 212 | 64.62 | 1.098 | 11.31 | -18.17 | 23 | 16 | 0.70 | spike |
| 162 | `rsi2_regime` | ETHUSD | 1h | 106 | 73.58 | 1.142 | 8.77 | -14.33 | 23 | 16 | 0.70 | spike |
| 163 | `rsi2_regime` | ETHUSD | 1h | 219 | 66.67 | 1.092 | 8.59 | -22.46 | 23 | 16 | 0.70 | spike |
| 164 | `zscore_revert` | ETHUSD | 1h | 254 | 64.96 | 1.153 | 26.69 | -16.81 | 16 | 11 | 0.69 | spike |
| 165 | `zscore_revert` | ETHUSD | 1h | 254 | 64.96 | 1.153 | 26.69 | -16.81 | 16 | 11 | 0.69 | spike |
| 166 | `zscore_revert` | ETHUSD | 1h | 254 | 64.96 | 1.153 | 26.69 | -16.81 | 16 | 11 | 0.69 | spike |
| 167 | `rsi2_regime` | ETHUSD | 1h | 145 | 69.66 | 1.431 | 64.62 | -20.63 | 22 | 15 | 0.68 | spike |
| 168 | `rsi2_regime` | ETHUSD | 1h | 224 | 73.66 | 1.575 | 144.5 | -16.34 | 21 | 14 | 0.67 | spike |
| 169 | `rsi2_regime` | ETHUSD | 1h | 70 | 77.14 | 2.025 | 58.42 | -9.11 | 24 | 16 | 0.67 | spike |
| 170 | `rsi2_regime` | ETHUSD | 1h | 151 | 68.87 | 1.373 | 50.19 | -15.19 | 24 | 16 | 0.67 | spike |
| 171 | `rsi2_regime` | ETHUSD | 1h | 365 | 66.85 | 1.2 | 48.3 | -20.75 | 24 | 16 | 0.67 | spike |
| 172 | `bb_trend_rejoin` | ETHUSD | 1h | 125 | 72.0 | 1.416 | 47.38 | -18.62 | 24 | 16 | 0.67 | spike |
| 173 | `bb_trend_rejoin` | ETHUSD | 1h | 125 | 72.0 | 1.392 | 43.72 | -18.62 | 24 | 16 | 0.67 | spike |
| 174 | `rsi2_regime` | ETHUSD | 1h | 72 | 72.22 | 1.845 | 36.44 | -14.48 | 24 | 16 | 0.67 | spike |
| 175 | `rsi2_regime` | ETHUSD | 1h | 111 | 66.67 | 1.304 | 35.11 | -16.82 | 24 | 16 | 0.67 | spike |
| 176 | `bb_trend_rejoin` | ETHUSD | 1h | 128 | 68.75 | 1.275 | 29.61 | -17.34 | 24 | 16 | 0.67 | spike |
| 177 | `rsi2_regime` | ETHUSD | 1h | 184 | 69.02 | 1.201 | 29.07 | -17.86 | 24 | 16 | 0.67 | spike |
| 178 | `bb_trend_rejoin` | ETHUSD | 1h | 128 | 68.75 | 1.255 | 26.91 | -17.58 | 24 | 16 | 0.67 | spike |
| 179 | `bb_trend_rejoin` | ETHUSD | 1h | 183 | 71.04 | 1.198 | 25.28 | -19.53 | 24 | 16 | 0.67 | spike |
| 180 | `rsi2_regime` | ETHUSD | 1h | 64 | 68.75 | 1.476 | 25.07 | -16.26 | 21 | 14 | 0.67 | spike |
| 181 | `bb_trend_rejoin` | ETHUSD | 1h | 196 | 64.8 | 1.093 | 11.21 | -20.56 | 24 | 16 | 0.67 | spike |
| 182 | `rsi2_regime` | ETHUSD | 1h | 204 | 67.16 | 1.089 | 8.51 | -24.65 | 21 | 14 | 0.67 | spike |
| 183 | `bb_trend_rejoin` | ETHUSD | 1h | 138 | 71.74 | 1.097 | 7.13 | -19.03 | 24 | 16 | 0.67 | spike |
| 184 | `rsi2_regime` | ETHUSD | 1h | 174 | 71.26 | 1.426 | 75.73 | -18.31 | 23 | 15 | 0.65 | spike |
| 185 | `rsi2_regime` | ETHUSD | 1h | 53 | 77.36 | 2.362 | 67.94 | -9.02 | 23 | 15 | 0.65 | spike |
| 186 | `rsi2_regime` | ETHUSD | 1h | 52 | 78.85 | 2.571 | 59.98 | -8.41 | 23 | 15 | 0.65 | spike |
| 187 | `rsi2_regime` | ETHUSD | 1h | 289 | 68.17 | 1.168 | 42.58 | -29.67 | 23 | 15 | 0.65 | spike |
| 188 | `rsi2_regime` | ETHUSD | 1h | 676 | 63.31 | 1.103 | 42.43 | -25.49 | 23 | 15 | 0.65 | spike |
| 189 | `rsi2_regime` | ETHUSD | 1h | 140 | 68.57 | 1.3 | 41.55 | -27.25 | 23 | 15 | 0.65 | spike |
| 190 | `rsi2_regime` | ETHUSD | 1h | 247 | 68.02 | 1.167 | 36.24 | -25.64 | 23 | 15 | 0.65 | spike |
| 191 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 71.22 | 1.296 | 27.64 | -15.71 | 23 | 15 | 0.65 | spike |
| 192 | `rsi2_regime` | ETHUSD | 1h | 62 | 70.97 | 1.531 | 26.59 | -11.56 | 23 | 15 | 0.65 | spike |
| 193 | `rsi2_regime` | ETHUSD | 1h | 55 | 69.09 | 1.292 | 13.04 | -11.87 | 23 | 15 | 0.65 | spike |
| 194 | `rsi2_regime` | ETHUSD | 1h | 162 | 67.9 | 1.16 | 11.86 | -23.21 | 23 | 15 | 0.65 | spike |
| 195 | `rsi2_regime` | ETHUSD | 1h | 72 | 68.06 | 1.154 | 5.78 | -16.43 | 23 | 15 | 0.65 | spike |
| 196 | `rsi2_regime` | ETHUSD | 1h | 72 | 68.06 | 1.154 | 5.78 | -16.43 | 23 | 15 | 0.65 | spike |
| 197 | `rsi2_regime` | ETHUSD | 1h | 148 | 69.59 | 1.512 | 79.81 | -17.34 | 22 | 14 | 0.64 | spike |
| 198 | `rsi2_regime` | ETHUSD | 1h | 254 | 69.29 | 1.128 | 22.82 | -23.57 | 22 | 14 | 0.64 | spike |
| 199 | `rsi2_regime` | ETHUSD | 1h | 113 | 69.03 | 1.313 | 34.83 | -12.67 | 19 | 12 | 0.63 | spike |
| 200 | `zscore_revert` | ETHUSD | 1h | 250 | 67.2 | 1.148 | 23.95 | -18.2 | 19 | 12 | 0.63 | spike |
| 201 | `zscore_revert` | ETHUSD | 1h | 250 | 67.2 | 1.132 | 20.5 | -18.2 | 19 | 12 | 0.63 | spike |
| 202 | `zscore_revert` | ETHUSD | 1h | 250 | 66.8 | 1.13 | 19.97 | -18.2 | 19 | 12 | 0.63 | spike |
| 203 | `zscore_revert` | ETHUSD | 1h | 250 | 66.8 | 1.13 | 19.97 | -18.2 | 19 | 12 | 0.63 | spike |
| 204 | `zscore_revert` | ETHUSD | 1h | 250 | 66.8 | 1.13 | 19.97 | -18.2 | 19 | 12 | 0.63 | spike |
| 205 | `zscore_revert` | ETHUSD | 1h | 250 | 66.8 | 1.13 | 19.97 | -18.2 | 19 | 12 | 0.63 | spike |
| 206 | `rsi2_regime` | ETHUSD | 1h | 300 | 63.33 | 1.107 | 19.2 | -16.59 | 19 | 12 | 0.63 | spike |
| 207 | `rsi2_regime` | ETHUSD | 1h | 123 | 69.92 | 1.159 | 13.38 | -18.25 | 19 | 12 | 0.63 | spike |
| 208 | `bb_trend_rejoin` | ETHUSD | 1h | 182 | 72.53 | 1.322 | 46.87 | -18.75 | 24 | 15 | 0.62 | spike |
| 209 | `rsi2_regime` | ETHUSD | 1h | 165 | 67.88 | 1.294 | 46.81 | -18.23 | 24 | 15 | 0.62 | spike |
| 210 | `bb_trend_rejoin` | ETHUSD | 1h | 126 | 72.22 | 1.386 | 42.52 | -17.72 | 24 | 15 | 0.62 | spike |
| 211 | `bb_trend_rejoin` | ETHUSD | 1h | 126 | 72.22 | 1.383 | 42.09 | -17.72 | 24 | 15 | 0.62 | spike |
| 212 | `bb_trend_rejoin` | ETHUSD | 1h | 182 | 72.53 | 1.235 | 32.94 | -20.8 | 24 | 15 | 0.62 | spike |
| 213 | `rsi2_regime` | ETHUSD | 1h | 150 | 68.0 | 1.212 | 31.72 | -22.17 | 24 | 15 | 0.62 | spike |
| 214 | `rsi2_regime` | DOGEUSD | 1h | 77 | 66.23 | 1.428 | 29.67 | -15.96 | 24 | 15 | 0.62 | spike |
| 215 | `bb_trend_rejoin` | ETHUSD | 1h | 131 | 65.65 | 1.242 | 25.72 | -20.55 | 24 | 15 | 0.62 | spike |
| 216 | `bb_trend_rejoin` | ETHUSD | 1h | 131 | 66.41 | 1.236 | 24.38 | -19.24 | 24 | 15 | 0.62 | spike |
| 217 | `bb_trend_rejoin` | ETHUSD | 1h | 131 | 65.65 | 1.213 | 21.83 | -20.75 | 24 | 15 | 0.62 | spike |
| 218 | `bb_trend_rejoin` | ETHUSD | 1h | 138 | 71.74 | 1.123 | 10.14 | -18.48 | 24 | 15 | 0.62 | spike |
| 219 | `bb_trend_rejoin` | ETHUSD | 1h | 138 | 71.74 | 1.112 | 8.85 | -18.48 | 24 | 15 | 0.62 | spike |
| 220 | `rsi2_regime` | ETHUSD | 1h | 172 | 72.67 | 1.519 | 77.97 | -14.04 | 21 | 13 | 0.62 | spike |
| 221 | `rsi2_regime` | ETHUSD | 1h | 250 | 67.6 | 1.173 | 32.59 | -22.71 | 21 | 13 | 0.62 | spike |
| 222 | `rsi2_regime` | ETHUSD | 1h | 90 | 73.33 | 1.37 | 20.92 | -11.69 | 21 | 13 | 0.62 | spike |
| 223 | `rsi2_regime` | ETHUSD | 1h | 278 | 70.5 | 1.295 | 86.66 | -27.57 | 23 | 14 | 0.61 | spike |
| 224 | `rsi2_regime` | ETHUSD | 1h | 52 | 78.85 | 2.469 | 55.28 | -8.41 | 23 | 14 | 0.61 | spike |
| 225 | `rsi2_regime` | ETHUSD | 1h | 147 | 71.43 | 1.355 | 47.18 | -13.51 | 23 | 14 | 0.61 | spike |
| 226 | `rsi2_regime` | ETHUSD | 1h | 91 | 67.03 | 1.42 | 39.69 | -17.53 | 23 | 14 | 0.61 | spike |
| 227 | `rsi2_regime` | ETHUSD | 1h | 186 | 70.97 | 1.197 | 30.66 | -17.87 | 23 | 14 | 0.61 | spike |
| 228 | `rsi2_regime` | ETHUSD | 1h | 72 | 69.44 | 1.425 | 30.35 | -18.16 | 23 | 14 | 0.61 | spike |
| 229 | `bb_trend_rejoin` | ETHUSD | 1h | 183 | 71.58 | 1.21 | 26.58 | -20.87 | 23 | 14 | 0.61 | spike |
| 230 | `bb_trend_rejoin` | ETHUSD | 1h | 183 | 71.04 | 1.187 | 23.48 | -20.87 | 23 | 14 | 0.61 | spike |
| 231 | `bb_trend_rejoin` | ETHUSD | 1h | 183 | 71.04 | 1.185 | 23.11 | -20.87 | 23 | 14 | 0.61 | spike |
| 232 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 71.22 | 1.213 | 19.62 | -16.51 | 23 | 14 | 0.61 | spike |
| 233 | `rsi2_regime` | ETHUSD | 1h | 97 | 70.1 | 1.198 | 14.59 | -17.52 | 23 | 14 | 0.61 | spike |
| 234 | `bb_trend_rejoin` | ETHUSD | 1h | 138 | 72.46 | 1.129 | 10.43 | -16.8 | 23 | 14 | 0.61 | spike |
| 235 | `bb_trend_rejoin` | ETHUSD | 1h | 138 | 72.46 | 1.118 | 9.24 | -16.8 | 23 | 14 | 0.61 | spike |
| 236 | `bb_trend_rejoin` | ETHUSD | 1h | 143 | 72.03 | 1.657 | 73.6 | -8.9 | 24 | 14 | 0.58 | spike |
| 237 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 70.26 | 1.488 | 73.31 | -11.77 | 24 | 14 | 0.58 | spike |
| 238 | `bb_trend_rejoin` | ETHUSD | 1h | 143 | 72.03 | 1.653 | 73.12 | -8.9 | 24 | 14 | 0.58 | spike |
| 239 | `bb_trend_rejoin` | ETHUSD | 1h | 143 | 72.03 | 1.629 | 69.65 | -8.9 | 24 | 14 | 0.58 | spike |
| 240 | `bb_trend_rejoin` | ETHUSD | 1h | 143 | 71.33 | 1.574 | 64.27 | -8.9 | 24 | 14 | 0.58 | spike |
| 241 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 69.74 | 1.428 | 62.04 | -11.77 | 24 | 14 | 0.58 | spike |
| 242 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 69.23 | 1.406 | 59.02 | -11.77 | 24 | 14 | 0.58 | spike |
| 243 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 75.0 | 1.7 | 56.6 | -19.27 | 24 | 14 | 0.58 | spike |
| 244 | `bb_trend_rejoin` | ETHUSD | 1h | 190 | 71.05 | 1.382 | 55.39 | -12.94 | 24 | 14 | 0.58 | spike |
| 245 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 68.21 | 1.382 | 54.51 | -11.77 | 24 | 14 | 0.58 | spike |
| 246 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 68.21 | 1.382 | 54.51 | -11.77 | 24 | 14 | 0.58 | spike |
| 247 | `bb_trend_rejoin` | ETHUSD | 1h | 149 | 69.8 | 1.5 | 52.04 | -9.84 | 24 | 14 | 0.58 | spike |
| 248 | `bb_trend_rejoin` | ETHUSD | 1h | 143 | 68.53 | 1.457 | 49.5 | -11.57 | 24 | 14 | 0.58 | spike |
| 249 | `bb_trend_rejoin` | ETHUSD | 1h | 182 | 72.53 | 1.332 | 48.7 | -19.23 | 24 | 14 | 0.58 | spike |
| 250 | `rsi2_regime` | ETHUSD | 1h | 56 | 71.43 | 2.321 | 48.58 | -12.06 | 24 | 14 | 0.58 | spike |
| 251 | `bb_trend_rejoin` | ETHUSD | 1h | 149 | 69.8 | 1.464 | 47.28 | -9.84 | 24 | 14 | 0.58 | spike |
| 252 | `bb_trend_rejoin` | ETHUSD | 1h | 149 | 69.8 | 1.464 | 47.28 | -9.84 | 24 | 14 | 0.58 | spike |
| 253 | `bb_trend_rejoin` | ETHUSD | 1h | 125 | 72.8 | 1.437 | 46.97 | -18.74 | 24 | 14 | 0.58 | spike |
| 254 | `bb_trend_rejoin` | ETHUSD | 1h | 90 | 67.78 | 1.562 | 45.76 | -12.28 | 24 | 14 | 0.58 | spike |
| 255 | `bb_trend_rejoin` | ETHUSD | 1h | 125 | 72.0 | 1.4 | 43.37 | -18.74 | 24 | 14 | 0.58 | spike |
| 256 | `bb_trend_rejoin` | ETHUSD | 1h | 90 | 71.11 | 1.529 | 41.47 | -16.09 | 24 | 14 | 0.58 | spike |
| 257 | `bb_trend_rejoin` | ETHUSD | 1h | 90 | 67.78 | 1.485 | 36.49 | -12.71 | 24 | 14 | 0.58 | spike |
| 258 | `rsi2_regime` | ETHUSD | 1h | 71 | 70.42 | 1.405 | 32.78 | -17.11 | 24 | 14 | 0.58 | spike |
| 259 | `rsi2_regime` | ETHUSD | 1h | 90 | 70.0 | 1.41 | 31.09 | -15.18 | 24 | 14 | 0.58 | spike |
| 260 | `bb_trend_rejoin` | ETHUSD | 1h | 126 | 69.05 | 1.298 | 30.54 | -18.36 | 24 | 14 | 0.58 | spike |
| 261 | `bb_trend_rejoin` | ETHUSD | 1h | 126 | 69.05 | 1.298 | 30.54 | -18.36 | 24 | 14 | 0.58 | spike |
| 262 | `bb_trend_rejoin` | ETHUSD | 1h | 127 | 69.29 | 1.282 | 30.12 | -15.71 | 24 | 14 | 0.58 | spike |
| 263 | `rsi2_regime` | ETHUSD | 1h | 89 | 70.79 | 1.388 | 29.52 | -15.21 | 24 | 14 | 0.58 | spike |
| 264 | `rsi2_regime` | ETHUSD | 1h | 159 | 69.18 | 1.391 | 29.32 | -13.66 | 24 | 14 | 0.58 | spike |
| 265 | `rsi2_regime` | ETHUSD | 1h | 70 | 70.0 | 1.466 | 27.45 | -16.5 | 24 | 14 | 0.58 | spike |
| 266 | `bb_trend_rejoin` | ETHUSD | 1h | 196 | 65.31 | 1.175 | 25.83 | -17.41 | 24 | 14 | 0.58 | spike |
| 267 | `rsi2_regime` | ETHUSD | 1h | 242 | 66.53 | 1.149 | 25.37 | -23.92 | 24 | 14 | 0.58 | spike |
| 268 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 67.88 | 1.198 | 25.19 | -14.36 | 24 | 14 | 0.58 | spike |
| 269 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 67.88 | 1.198 | 25.19 | -14.36 | 24 | 14 | 0.58 | spike |
| 270 | `bb_trend_rejoin` | ETHUSD | 1h | 127 | 69.29 | 1.245 | 25.14 | -16.77 | 24 | 14 | 0.58 | spike |
| 271 | `rsi2_regime` | ETHUSD | 1h | 284 | 66.55 | 1.13 | 25.11 | -24.93 | 24 | 14 | 0.58 | spike |
| 272 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 67.88 | 1.189 | 23.83 | -14.36 | 24 | 14 | 0.58 | spike |
| 273 | `rsi2_regime` | ETHUSD | 1h | 175 | 68.57 | 1.179 | 21.96 | -16.3 | 24 | 14 | 0.58 | spike |
| 274 | `bb_trend_rejoin` | ETHUSD | 1h | 212 | 65.09 | 1.157 | 21.13 | -14.83 | 24 | 14 | 0.58 | spike |
| 275 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 71.59 | 1.317 | 19.67 | -14.18 | 24 | 14 | 0.58 | spike |
| 276 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 71.59 | 1.312 | 19.31 | -14.18 | 24 | 14 | 0.58 | spike |
| 277 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 71.59 | 1.291 | 19.22 | -14.38 | 24 | 14 | 0.58 | spike |
| 278 | `rsi2_regime` | ETHUSD | 1h | 149 | 68.46 | 1.168 | 17.05 | -16.21 | 24 | 14 | 0.58 | spike |
| 279 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 71.59 | 1.26 | 16.72 | -14.38 | 24 | 14 | 0.58 | spike |
| 280 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 71.59 | 1.249 | 16.05 | -14.38 | 24 | 14 | 0.58 | spike |
| 281 | `rsi2_regime` | ETHUSD | 1h | 159 | 68.55 | 1.203 | 14.75 | -22.3 | 24 | 14 | 0.58 | spike |
| 282 | `bb_trend_rejoin` | ETHUSD | 1h | 161 | 67.7 | 1.12 | 12.52 | -18.71 | 24 | 14 | 0.58 | spike |
| 283 | `bb_trend_rejoin` | ETHUSD | 1h | 308 | 65.58 | 1.086 | 12.46 | -26.78 | 24 | 14 | 0.58 | spike |
| 284 | `bb_trend_rejoin` | ETHUSD | 1h | 162 | 67.9 | 1.114 | 11.67 | -17.01 | 24 | 14 | 0.58 | spike |
| 285 | `bb_trend_rejoin` | ETHUSD | 1h | 204 | 70.1 | 1.075 | 5.56 | -23.06 | 24 | 14 | 0.58 | spike |
| 286 | `bb_trend_rejoin` | ETHUSD | 1h | 78 | 73.08 | 1.078 | 2.28 | -16.56 | 24 | 14 | 0.58 | spike |
| 287 | `rsi2_regime` | ETHUSD | 1h | 209 | 66.03 | 1.193 | 36.66 | -19.74 | 21 | 12 | 0.57 | spike |
| 288 | `rsi2_regime` | ETHUSD | 1h | 271 | 65.31 | 1.139 | 27.89 | -26.44 | 21 | 12 | 0.57 | spike |
| 289 | `rsi2_regime` | ETHUSD | 1h | 606 | 61.39 | 1.078 | 23.39 | -23.73 | 21 | 12 | 0.57 | spike |
| 290 | `rsi2_regime` | ETHUSD | 1h | 118 | 66.95 | 1.211 | 23.19 | -14.78 | 21 | 12 | 0.57 | spike |
| 291 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 70.26 | 1.489 | 73.56 | -11.77 | 23 | 13 | 0.57 | spike |
| 292 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 69.74 | 1.444 | 66.53 | -11.77 | 23 | 13 | 0.57 | spike |
| 293 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 69.74 | 1.436 | 65.76 | -11.77 | 23 | 13 | 0.57 | spike |
| 294 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 69.74 | 1.435 | 65.6 | -11.77 | 23 | 13 | 0.57 | spike |
| 295 | `rsi2_regime` | ETHUSD | 1h | 239 | 69.46 | 1.28 | 64.2 | -26.63 | 23 | 13 | 0.57 | spike |
| 296 | `bb_trend_rejoin` | ETHUSD | 1h | 143 | 71.33 | 1.571 | 63.91 | -8.9 | 23 | 13 | 0.57 | spike |
| 297 | `rsi2_regime` | ETHUSD | 1h | 53 | 77.36 | 2.732 | 63.07 | -9.77 | 23 | 13 | 0.57 | spike |
| 298 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 69.74 | 1.426 | 63.07 | -11.77 | 23 | 13 | 0.57 | spike |
| 299 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 69.74 | 1.426 | 63.07 | -11.77 | 23 | 13 | 0.57 | spike |
| 300 | `rsi2_regime` | ETHUSD | 1h | 88 | 68.18 | 1.641 | 60.36 | -16.26 | 23 | 13 | 0.57 | spike |
| 301 | `rsi2_regime` | ETHUSD | 1h | 81 | 76.54 | 1.851 | 60.23 | -15.24 | 23 | 13 | 0.57 | spike |
| 302 | `bb_trend_rejoin` | ETHUSD | 1h | 90 | 68.89 | 1.736 | 59.74 | -12.47 | 23 | 13 | 0.57 | spike |
| 303 | `bb_trend_rejoin` | ETHUSD | 1h | 89 | 73.03 | 1.748 | 59.36 | -16.09 | 23 | 13 | 0.57 | spike |
| 304 | `rsi2_regime` | ETHUSD | 1h | 52 | 76.92 | 2.137 | 56.92 | -10.21 | 23 | 13 | 0.57 | spike |
| 305 | `bb_trend_rejoin` | ETHUSD | 1h | 89 | 73.03 | 1.714 | 55.96 | -16.09 | 23 | 13 | 0.57 | spike |
| 306 | `bb_trend_rejoin` | ETHUSD | 1h | 89 | 73.03 | 1.637 | 50.54 | -16.09 | 23 | 13 | 0.57 | spike |
| 307 | `rsi2_regime` | ETHUSD | 1h | 52 | 73.08 | 1.882 | 45.73 | -13.23 | 23 | 13 | 0.57 | spike |
| 308 | `bb_trend_rejoin` | ETHUSD | 1h | 149 | 69.8 | 1.454 | 45.09 | -9.84 | 23 | 13 | 0.57 | spike |
| 309 | `bb_trend_rejoin` | ETHUSD | 1h | 149 | 69.8 | 1.454 | 45.09 | -9.84 | 23 | 13 | 0.57 | spike |
| 310 | `bb_trend_rejoin` | ETHUSD | 1h | 90 | 67.78 | 1.539 | 43.23 | -12.72 | 23 | 13 | 0.57 | spike |
| 311 | `rsi2_regime` | ETHUSD | 1h | 331 | 66.47 | 1.19 | 40.83 | -22.42 | 23 | 13 | 0.57 | spike |
| 312 | `bb_trend_rejoin` | ETHUSD | 1h | 125 | 72.0 | 1.335 | 36.25 | -20.49 | 23 | 13 | 0.57 | spike |
| 313 | `bb_trend_rejoin` | ETHUSD | 1h | 125 | 72.0 | 1.335 | 36.25 | -20.49 | 23 | 13 | 0.57 | spike |
| 314 | `bb_trend_rejoin` | ETHUSD | 1h | 125 | 72.0 | 1.333 | 35.83 | -20.49 | 23 | 13 | 0.57 | spike |
| 315 | `rsi2_regime` | ETHUSD | 1h | 173 | 71.1 | 1.237 | 32.79 | -21.31 | 23 | 13 | 0.57 | spike |
| 316 | `rsi2_regime` | ETHUSD | 1h | 52 | 78.85 | 1.994 | 26.62 | -9.91 | 23 | 13 | 0.57 | spike |
| 317 | `rsi2_regime` | ETHUSD | 1h | 67 | 68.66 | 1.361 | 17.44 | -13.72 | 23 | 13 | 0.57 | spike |
| 318 | `rsi2_regime` | ETHUSD | 1h | 192 | 68.75 | 1.155 | 17.09 | -18.76 | 23 | 13 | 0.57 | spike |
| 319 | `rsi2_regime` | ETHUSD | 1h | 256 | 67.58 | 1.099 | 15.89 | -26.87 | 23 | 13 | 0.57 | spike |
| 320 | `rsi2_regime` | ADAUSD | 1h | 75 | 70.67 | 1.165 | 8.41 | -12.45 | 23 | 13 | 0.57 | spike |
| 321 | `rsi2_regime` | ETHUSD | 1h | 100 | 68.0 | 1.102 | 5.82 | -10.59 | 23 | 13 | 0.57 | spike |
| 322 | `rsi2_regime` | ADAUSD | 1h | 78 | 69.23 | 1.117 | 5.71 | -12.45 | 23 | 13 | 0.57 | spike |
| 323 | `bb_trend_rejoin` | ETHUSD | 1h | 143 | 72.03 | 1.632 | 69.94 | -8.9 | 22 | 12 | 0.55 | spike |
| 324 | `bb_trend_rejoin` | ETHUSD | 1h | 143 | 72.03 | 1.632 | 69.94 | -8.9 | 22 | 12 | 0.55 | spike |
| 325 | `rsi2_regime` | ETHUSD | 1h | 60 | 71.67 | 1.894 | 61.39 | -8.88 | 22 | 12 | 0.55 | spike |
| 326 | `rsi2_regime` | ETHUSD | 1h | 150 | 68.0 | 1.38 | 60.95 | -17.09 | 22 | 12 | 0.55 | spike |
| 327 | `rsi2_regime` | ETHUSD | 1h | 146 | 69.18 | 1.284 | 33.78 | -20.48 | 22 | 12 | 0.55 | spike |
| 328 | `rsi2_regime` | ETHUSD | 1h | 150 | 67.33 | 1.202 | 24.37 | -19.43 | 22 | 12 | 0.55 | spike |
| 329 | `rsi2_regime` | ETHUSD | 1h | 287 | 67.25 | 1.12 | 21.72 | -20.88 | 22 | 12 | 0.55 | spike |
| 330 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 73.86 | 1.856 | 67.55 | -16.12 | 24 | 13 | 0.54 | spike |
| 331 | `bb_trend_rejoin` | ETHUSD | 1h | 141 | 72.34 | 1.498 | 56.0 | -10.59 | 24 | 13 | 0.54 | spike |
| 332 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 68.21 | 1.381 | 55.54 | -11.77 | 24 | 13 | 0.54 | spike |
| 333 | `bb_trend_rejoin` | ETHUSD | 1h | 149 | 69.8 | 1.522 | 53.98 | -9.84 | 24 | 13 | 0.54 | spike |
| 334 | `bb_trend_rejoin` | ETHUSD | 1h | 213 | 68.54 | 1.287 | 49.18 | -17.28 | 24 | 13 | 0.54 | spike |
| 335 | `bb_trend_rejoin` | ETHUSD | 1h | 90 | 67.78 | 1.617 | 45.65 | -11.13 | 24 | 13 | 0.54 | spike |
| 336 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.445 | 45.54 | -9.11 | 24 | 13 | 0.54 | spike |
| 337 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.439 | 44.99 | -9.11 | 24 | 13 | 0.54 | spike |
| 338 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 71.43 | 1.448 | 44.56 | -9.11 | 24 | 13 | 0.54 | spike |
| 339 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.405 | 41.73 | -9.11 | 24 | 13 | 0.54 | spike |
| 340 | `bb_trend_rejoin` | ETHUSD | 1h | 268 | 67.91 | 1.179 | 35.14 | -21.33 | 24 | 13 | 0.54 | spike |
| 341 | `bb_trend_rejoin` | ETHUSD | 1h | 167 | 66.47 | 1.238 | 32.18 | -15.43 | 24 | 13 | 0.54 | spike |
| 342 | `bb_trend_rejoin` | ETHUSD | 1h | 167 | 66.47 | 1.238 | 32.18 | -15.43 | 24 | 13 | 0.54 | spike |
| 343 | `bb_trend_rejoin` | ETHUSD | 1h | 167 | 66.47 | 1.238 | 32.18 | -15.43 | 24 | 13 | 0.54 | spike |
| 344 | `bb_trend_rejoin` | ETHUSD | 1h | 167 | 66.47 | 1.226 | 30.16 | -15.43 | 24 | 13 | 0.54 | spike |
| 345 | `bb_trend_rejoin` | ETHUSD | 1h | 180 | 72.22 | 1.222 | 30.16 | -23.09 | 24 | 13 | 0.54 | spike |
| 346 | `bb_trend_rejoin` | ETHUSD | 1h | 167 | 66.47 | 1.219 | 28.92 | -15.43 | 24 | 13 | 0.54 | spike |
| 347 | `rsi2_regime` | ETHUSD | 1h | 158 | 70.89 | 1.293 | 27.82 | -17.9 | 24 | 13 | 0.54 | spike |
| 348 | `rsi2_regime` | ETHUSD | 1h | 312 | 63.78 | 1.194 | 27.3 | -20.27 | 24 | 13 | 0.54 | spike |
| 349 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 69.06 | 1.21 | 18.97 | -12.63 | 24 | 13 | 0.54 | spike |
| 350 | `rsi2_regime` | ETHUSD | 1h | 238 | 66.81 | 1.111 | 15.83 | -24.77 | 24 | 13 | 0.54 | spike |
| 351 | `bb_trend_rejoin` | ETHUSD | 1h | 167 | 63.47 | 1.122 | 13.9 | -20.03 | 24 | 13 | 0.54 | spike |
| 352 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 72.66 | 1.148 | 12.37 | -17.12 | 24 | 13 | 0.54 | spike |
| 353 | `bb_trend_rejoin` | ETHUSD | 1h | 166 | 63.25 | 1.116 | 8.8 | -24.0 | 24 | 13 | 0.54 | spike |
| 354 | `bb_trend_rejoin` | ETHUSD | 1h | 162 | 67.9 | 1.061 | 3.58 | -17.01 | 24 | 13 | 0.54 | spike |
| 355 | `zscore_revert` | ETHUSD | 1h | 245 | 66.53 | 1.144 | 22.32 | -19.35 | 19 | 10 | 0.53 | spike |
| 356 | `zscore_revert` | ETHUSD | 1h | 245 | 66.53 | 1.126 | 18.92 | -19.35 | 19 | 10 | 0.53 | spike |
| 357 | `rsi2_regime` | ETHUSD | 1h | 194 | 68.04 | 1.357 | 70.94 | -21.05 | 21 | 11 | 0.52 | spike |
| 358 | `rsi2_regime` | ETHUSD | 1h | 79 | 72.15 | 2.15 | 58.58 | -10.18 | 21 | 11 | 0.52 | spike |
| 359 | `rsi2_regime` | ETHUSD | 1h | 147 | 65.31 | 1.336 | 50.31 | -18.68 | 21 | 11 | 0.52 | spike |
| 360 | `rsi2_regime` | ETHUSD | 1h | 497 | 64.59 | 1.174 | 44.55 | -29.82 | 21 | 11 | 0.52 | spike |
| 361 | `rsi2_regime` | ETHUSD | 1h | 52 | 75.0 | 1.633 | 35.61 | -15.52 | 21 | 11 | 0.52 | spike |
| 362 | `rsi2_regime` | ETHUSD | 1h | 67 | 68.66 | 1.488 | 31.63 | -17.46 | 21 | 11 | 0.52 | spike |
| 363 | `rsi2_regime` | ETHUSD | 1h | 97 | 67.01 | 1.324 | 31.0 | -12.0 | 21 | 11 | 0.52 | spike |
| 364 | `rsi2_regime` | ETHUSD | 1h | 118 | 66.95 | 1.188 | 19.68 | -14.78 | 21 | 11 | 0.52 | spike |
| 365 | `rsi2_regime` | ETHUSD | 1h | 84 | 73.81 | 1.326 | 17.69 | -13.08 | 21 | 11 | 0.52 | spike |
| 366 | `rsi2_regime` | ETHUSD | 1h | 110 | 63.64 | 1.166 | 13.8 | -19.35 | 21 | 11 | 0.52 | spike |
| 367 | `bb_trend_rejoin` | ETHUSD | 1h | 89 | 73.03 | 1.988 | 74.86 | -13.98 | 23 | 12 | 0.52 | spike |
| 368 | `rsi2_regime` | ETHUSD | 1h | 128 | 67.97 | 1.465 | 59.03 | -18.16 | 23 | 12 | 0.52 | spike |
| 369 | `rsi2_regime` | ETHUSD | 1h | 97 | 68.04 | 1.54 | 54.8 | -16.63 | 23 | 12 | 0.52 | spike |
| 370 | `bb_trend_rejoin` | ETHUSD | 1h | 141 | 72.34 | 1.472 | 52.36 | -10.59 | 23 | 12 | 0.52 | spike |
| 371 | `rsi2_regime` | ETHUSD | 1h | 64 | 68.75 | 1.574 | 35.23 | -15.46 | 23 | 12 | 0.52 | spike |
| 372 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.329 | 32.34 | -10.23 | 23 | 12 | 0.52 | spike |
| 373 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.329 | 32.34 | -10.23 | 23 | 12 | 0.52 | spike |
| 374 | `rsi2_regime` | ETHUSD | 1h | 67 | 68.66 | 1.547 | 30.52 | -9.21 | 23 | 12 | 0.52 | spike |
| 375 | `rsi2_regime` | ETHUSD | 1h | 91 | 70.33 | 1.397 | 30.36 | -26.39 | 23 | 12 | 0.52 | spike |
| 376 | `bb_trend_rejoin` | ETHUSD | 1h | 268 | 67.54 | 1.16 | 28.96 | -21.33 | 23 | 12 | 0.52 | spike |
| 377 | `bb_trend_rejoin` | ETHUSD | 1h | 167 | 66.47 | 1.208 | 27.13 | -15.43 | 23 | 12 | 0.52 | spike |
| 378 | `bb_trend_rejoin` | ETHUSD | 1h | 167 | 66.47 | 1.206 | 26.68 | -15.43 | 23 | 12 | 0.52 | spike |
| 379 | `bb_trend_rejoin` | ETHUSD | 1h | 268 | 67.54 | 1.145 | 25.56 | -21.33 | 23 | 12 | 0.52 | spike |
| 380 | `bb_trend_rejoin` | ETHUSD | 1h | 268 | 67.16 | 1.134 | 23.17 | -21.33 | 23 | 12 | 0.52 | spike |
| 381 | `bb_trend_rejoin` | ETHUSD | 1h | 167 | 65.87 | 1.178 | 22.66 | -15.43 | 23 | 12 | 0.52 | spike |
| 382 | `rsi2_regime` | ETHUSD | 1h | 52 | 76.92 | 1.816 | 22.52 | -11.14 | 23 | 12 | 0.52 | spike |
| 383 | `rsi2_regime` | ETHUSD | 1h | 96 | 76.04 | 1.4 | 17.21 | -9.61 | 23 | 12 | 0.52 | spike |
| 384 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 67.88 | 1.145 | 16.64 | -14.36 | 23 | 12 | 0.52 | spike |
| 385 | `rsi2_regime` | ETHUSD | 1h | 113 | 69.91 | 1.168 | 16.41 | -17.52 | 23 | 12 | 0.52 | spike |
| 386 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 67.27 | 1.136 | 15.7 | -14.55 | 23 | 12 | 0.52 | spike |
| 387 | `bb_trend_rejoin` | ETHUSD | 1h | 138 | 72.46 | 1.126 | 10.28 | -17.91 | 23 | 12 | 0.52 | spike |
| 388 | `rsi2_regime` | ETHUSD | 1h | 73 | 67.12 | 1.139 | 5.17 | -15.77 | 23 | 12 | 0.52 | spike |
| 389 | `rsi2_regime` | ETHUSD | 1h | 239 | 66.11 | 1.062 | 4.95 | -21.4 | 23 | 12 | 0.52 | spike |
| 390 | `rsi2_regime` | ETHUSD | 1h | 443 | 64.56 | 1.053 | 4.62 | -24.64 | 23 | 12 | 0.52 | spike |
| 391 | `bb_trend_rejoin` | ETHUSD | 1h | 143 | 72.03 | 1.632 | 69.94 | -8.9 | 22 | 11 | 0.50 | spike |
| 392 | `bb_trend_rejoin` | ETHUSD | 1h | 143 | 72.03 | 1.619 | 68.11 | -8.9 | 24 | 12 | 0.50 | spike |
| 393 | `bb_trend_rejoin` | ETHUSD | 1h | 87 | 73.56 | 1.641 | 51.78 | -21.49 | 24 | 12 | 0.50 | spike |
| 394 | `rsi2_regime` | ETHUSD | 1h | 271 | 70.11 | 1.207 | 45.31 | -29.09 | 22 | 11 | 0.50 | spike |
| 395 | `bb_trend_rejoin` | ETHUSD | 1h | 87 | 73.56 | 1.544 | 43.06 | -21.49 | 24 | 12 | 0.50 | spike |
| 396 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 64.84 | 1.495 | 38.41 | -12.28 | 24 | 12 | 0.50 | spike |
| 397 | `rsi2_regime` | ETHUSD | 1h | 63 | 68.25 | 1.615 | 37.88 | -12.76 | 24 | 12 | 0.50 | spike |
| 398 | `zscore_revert` | ETHUSD | 1h | 254 | 64.96 | 1.199 | 36.86 | -16.81 | 16 | 8 | 0.50 | spike |
| 399 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 67.03 | 1.471 | 36.59 | -16.06 | 24 | 12 | 0.50 | spike |
| 400 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 67.03 | 1.471 | 36.59 | -16.06 | 24 | 12 | 0.50 | spike |
| 401 | `bb_trend_rejoin` | ETHUSD | 1h | 128 | 67.97 | 1.336 | 36.26 | -15.76 | 24 | 12 | 0.50 | spike |
| 402 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.48 | 34.28 | -10.64 | 24 | 12 | 0.50 | spike |
| 403 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.48 | 34.28 | -10.64 | 24 | 12 | 0.50 | spike |
| 404 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 68.18 | 1.572 | 33.65 | -11.0 | 24 | 12 | 0.50 | spike |
| 405 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 71.59 | 1.488 | 31.69 | -13.74 | 24 | 12 | 0.50 | spike |
| 406 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 71.59 | 1.5 | 31.05 | -12.04 | 24 | 12 | 0.50 | spike |
| 407 | `rsi2_regime` | ETHUSD | 1h | 110 | 70.0 | 1.346 | 30.91 | -15.15 | 22 | 11 | 0.50 | spike |
| 408 | `bb_trend_rejoin` | ETHUSD | 1h | 128 | 67.97 | 1.289 | 30.71 | -15.39 | 24 | 12 | 0.50 | spike |
| 409 | `rsi2_regime` | ETHUSD | 1h | 154 | 66.23 | 1.199 | 30.24 | -27.74 | 24 | 12 | 0.50 | spike |
| 410 | `bb_trend_rejoin` | ETHUSD | 1h | 128 | 67.97 | 1.282 | 29.69 | -15.39 | 24 | 12 | 0.50 | spike |
| 411 | `bb_trend_rejoin` | ETHUSD | 1h | 128 | 67.97 | 1.279 | 29.59 | -16.11 | 24 | 12 | 0.50 | spike |
| 412 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 72.73 | 1.489 | 28.98 | -11.55 | 24 | 12 | 0.50 | spike |
| 413 | `bb_trend_rejoin` | ETHUSD | 1h | 128 | 67.97 | 1.272 | 28.57 | -16.11 | 24 | 12 | 0.50 | spike |
| 414 | `bb_trend_rejoin` | ETHUSD | 1h | 129 | 67.44 | 1.258 | 27.47 | -16.56 | 24 | 12 | 0.50 | spike |
| 415 | `rsi2_regime` | ETHUSD | 1h | 168 | 69.64 | 1.335 | 27.47 | -15.55 | 24 | 12 | 0.50 | spike |
| 416 | `rsi2_regime` | ETHUSD | 1h | 245 | 68.57 | 1.14 | 27.45 | -20.02 | 24 | 12 | 0.50 | spike |
| 417 | `bb_trend_rejoin` | ETHUSD | 1h | 129 | 67.44 | 1.25 | 26.38 | -16.56 | 24 | 12 | 0.50 | spike |
| 418 | `rsi2_regime` | ETHUSD | 1h | 296 | 66.55 | 1.134 | 25.7 | -19.2 | 24 | 12 | 0.50 | spike |
| 419 | `rsi2_regime` | ETHUSD | 1h | 313 | 63.9 | 1.165 | 22.83 | -20.78 | 24 | 12 | 0.50 | spike |
| 420 | `rsi2_regime` | ETHUSD | 1h | 313 | 63.9 | 1.164 | 22.72 | -20.85 | 24 | 12 | 0.50 | spike |
| 421 | `bb_trend_rejoin` | ETHUSD | 1h | 250 | 67.2 | 1.132 | 20.5 | -18.2 | 24 | 12 | 0.50 | spike |
| 422 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 71.94 | 1.201 | 18.12 | -15.51 | 24 | 12 | 0.50 | spike |
| 423 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 67.88 | 1.153 | 17.93 | -14.36 | 24 | 12 | 0.50 | spike |
| 424 | `bb_trend_rejoin` | ETHUSD | 1h | 77 | 74.03 | 1.472 | 16.3 | -7.41 | 24 | 12 | 0.50 | spike |
| 425 | `bb_trend_rejoin` | ETHUSD | 1h | 166 | 63.25 | 1.12 | 9.26 | -22.8 | 24 | 12 | 0.50 | spike |
| 426 | `rsi2_regime` | ETHUSD | 1h | 82 | 65.85 | 1.148 | 7.3 | -14.44 | 24 | 12 | 0.50 | spike |
| 427 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 64.24 | 1.076 | 6.61 | -17.04 | 24 | 12 | 0.50 | spike |
| 428 | `rsi2_regime` | ETHUSD | 1h | 52 | 75.0 | 2.322 | 62.62 | -8.33 | 23 | 11 | 0.48 | spike |
| 429 | `bb_trend_rejoin` | ETHUSD | 1h | 219 | 67.58 | 1.3 | 53.37 | -15.93 | 23 | 11 | 0.48 | spike |
| 430 | `bb_trend_rejoin` | ETHUSD | 1h | 219 | 67.58 | 1.3 | 53.37 | -15.93 | 23 | 11 | 0.48 | spike |
| 431 | `bb_trend_rejoin` | ETHUSD | 1h | 141 | 71.63 | 1.438 | 48.89 | -10.45 | 23 | 11 | 0.48 | spike |
| 432 | `rsi2_regime` | ETHUSD | 1h | 53 | 79.25 | 2.191 | 47.34 | -10.9 | 23 | 11 | 0.48 | spike |
| 433 | `rsi2_regime` | ETHUSD | 1h | 90 | 71.11 | 1.567 | 41.5 | -11.26 | 23 | 11 | 0.48 | spike |
| 434 | `rsi2_regime` | ETHUSD | 1h | 95 | 72.63 | 1.482 | 40.56 | -16.49 | 23 | 11 | 0.48 | spike |
| 435 | `rsi2_regime` | ETHUSD | 1h | 82 | 75.61 | 1.6 | 37.45 | -15.45 | 23 | 11 | 0.48 | spike |
| 436 | `rsi2_regime` | DOGEUSD | 1h | 67 | 67.16 | 1.538 | 36.77 | -20.59 | 23 | 11 | 0.48 | spike |
| 437 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.367 | 36.01 | -9.11 | 23 | 11 | 0.48 | spike |
| 438 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 71.59 | 1.561 | 35.21 | -13.7 | 23 | 11 | 0.48 | spike |
| 439 | `rsi2_regime` | ETHUSD | 1h | 174 | 69.54 | 1.239 | 34.49 | -22.07 | 23 | 11 | 0.48 | spike |
| 440 | `rsi2_regime` | ETHUSD | 1h | 71 | 69.01 | 1.35 | 28.32 | -23.17 | 23 | 11 | 0.48 | spike |
| 441 | `bb_trend_rejoin` | ETHUSD | 1h | 167 | 66.47 | 1.205 | 26.56 | -15.43 | 23 | 11 | 0.48 | spike |
| 442 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 70.45 | 1.364 | 23.51 | -13.09 | 23 | 11 | 0.48 | spike |
| 443 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 70.45 | 1.342 | 22.62 | -13.09 | 23 | 11 | 0.48 | spike |
| 444 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 70.45 | 1.338 | 22.04 | -13.09 | 23 | 11 | 0.48 | spike |
| 445 | `rsi2_regime` | ETHUSD | 1h | 52 | 75.0 | 1.679 | 19.85 | -10.78 | 23 | 11 | 0.48 | spike |
| 446 | `rsi2_regime` | ETHUSD | 1h | 58 | 70.69 | 1.556 | 19.43 | -13.96 | 23 | 11 | 0.48 | spike |
| 447 | `rsi2_regime` | ETHUSD | 1h | 111 | 68.47 | 1.261 | 17.99 | -9.44 | 23 | 11 | 0.48 | spike |
| 448 | `rsi2_regime` | ETHUSD | 1h | 72 | 70.83 | 1.232 | 17.81 | -16.81 | 23 | 11 | 0.48 | spike |
| 449 | `rsi2_regime` | ETHUSD | 1h | 72 | 73.61 | 1.252 | 15.64 | -20.15 | 23 | 11 | 0.48 | spike |
| 450 | `rsi2_regime` | ETHUSD | 1h | 197 | 67.01 | 1.155 | 13.92 | -20.02 | 23 | 11 | 0.48 | spike |
| 451 | `rsi2_regime` | ETHUSD | 1h | 143 | 71.33 | 1.172 | 10.91 | -12.51 | 23 | 11 | 0.48 | spike |
| 452 | `bb_trend_rejoin` | ETHUSD | 1h | 77 | 72.73 | 1.271 | 10.04 | -8.71 | 23 | 11 | 0.48 | spike |
| 453 | `rsi2_regime` | ETHUSD | 1h | 52 | 78.85 | 2.178 | 46.03 | -10.9 | 21 | 10 | 0.48 | spike |
| 454 | `rsi2_regime` | ETHUSD | 1h | 52 | 76.92 | 1.938 | 41.54 | -11.11 | 21 | 10 | 0.48 | spike |
| 455 | `rsi2_regime` | ETHUSD | 1h | 98 | 67.35 | 1.179 | 13.57 | -14.95 | 21 | 10 | 0.48 | spike |
| 456 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.588 | 69.56 | -15.22 | 19 | 9 | 0.47 | spike |
| 457 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.578 | 68.05 | -15.22 | 19 | 9 | 0.47 | spike |
| 458 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.557 | 64.61 | -15.22 | 19 | 9 | 0.47 | spike |
| 459 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.553 | 63.56 | -15.16 | 19 | 9 | 0.47 | spike |
| 460 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.547 | 61.88 | -15.16 | 19 | 9 | 0.47 | spike |
| 461 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.535 | 61.31 | -15.22 | 19 | 9 | 0.47 | spike |
| 462 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.469 | 55.77 | -17.34 | 19 | 9 | 0.47 | spike |
| 463 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.496 | 55.47 | -15.95 | 19 | 9 | 0.47 | spike |
| 464 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.494 | 55.3 | -16.05 | 19 | 9 | 0.47 | spike |
| 465 | `rsi2_regime` | ETHUSD | 1h | 293 | 69.97 | 1.247 | 54.29 | -21.09 | 19 | 9 | 0.47 | spike |
| 466 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.459 | 54.25 | -17.34 | 19 | 9 | 0.47 | spike |
| 467 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.459 | 54.25 | -17.34 | 19 | 9 | 0.47 | spike |
| 468 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.459 | 54.25 | -17.34 | 19 | 9 | 0.47 | spike |
| 469 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.459 | 54.25 | -17.34 | 19 | 9 | 0.47 | spike |
| 470 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.445 | 51.99 | -17.34 | 19 | 9 | 0.47 | spike |
| 471 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.434 | 50.11 | -16.74 | 19 | 9 | 0.47 | spike |
| 472 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.434 | 49.42 | -16.2 | 19 | 9 | 0.47 | spike |
| 473 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.434 | 49.42 | -16.2 | 19 | 9 | 0.47 | spike |
| 474 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.434 | 49.42 | -16.2 | 19 | 9 | 0.47 | spike |
| 475 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.434 | 49.42 | -16.2 | 19 | 9 | 0.47 | spike |
| 476 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.424 | 48.82 | -17.34 | 19 | 9 | 0.47 | spike |
| 477 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.408 | 47.75 | -17.34 | 19 | 9 | 0.47 | spike |
| 478 | `rsi2_regime` | ETHUSD | 1h | 181 | 71.82 | 1.28 | 46.63 | -23.83 | 19 | 9 | 0.47 | spike |
| 479 | `rsi2_regime` | ETHUSD | 1h | 150 | 69.33 | 1.476 | 46.24 | -17.4 | 19 | 9 | 0.47 | spike |
| 480 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.383 | 44.31 | -16.74 | 19 | 9 | 0.47 | spike |
| 481 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.383 | 44.31 | -16.74 | 19 | 9 | 0.47 | spike |
| 482 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.383 | 44.31 | -16.74 | 19 | 9 | 0.47 | spike |
| 483 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.373 | 42.51 | -17.34 | 19 | 9 | 0.47 | spike |
| 484 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.38 | 42.49 | -18.05 | 19 | 9 | 0.47 | spike |
| 485 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.38 | 42.49 | -18.05 | 19 | 9 | 0.47 | spike |
| 486 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.38 | 42.44 | -18.14 | 19 | 9 | 0.47 | spike |
| 487 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.376 | 41.95 | -18.05 | 19 | 9 | 0.47 | spike |
| 488 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.376 | 41.95 | -18.05 | 19 | 9 | 0.47 | spike |
| 489 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.37 | 41.69 | -16.37 | 19 | 9 | 0.47 | spike |
| 490 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.373 | 41.52 | -18.05 | 19 | 9 | 0.47 | spike |
| 491 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.373 | 41.52 | -18.05 | 19 | 9 | 0.47 | spike |
| 492 | `zscore_revert` | ETHUSD | 1h | 127 | 70.87 | 1.338 | 37.54 | -18.14 | 19 | 9 | 0.47 | spike |
| 493 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.338 | 37.32 | -16.37 | 19 | 9 | 0.47 | spike |
| 494 | `rsi2_regime` | ETHUSD | 1h | 121 | 70.25 | 1.278 | 32.08 | -17.18 | 19 | 9 | 0.47 | spike |
| 495 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.268 | 29.7 | -20.64 | 19 | 9 | 0.47 | spike |
| 496 | `bb_trend_rejoin` | ETHUSD | 1h | 66 | 69.7 | 1.675 | 68.13 | -22.34 | 24 | 11 | 0.46 | spike |
| 497 | `bb_trend_rejoin` | ETHUSD | 1h | 87 | 73.56 | 1.862 | 63.49 | -16.89 | 24 | 11 | 0.46 | spike |
| 498 | `rsi2_regime` | ETHUSD | 1h | 52 | 75.0 | 2.562 | 58.04 | -8.8 | 24 | 11 | 0.46 | spike |
| 499 | `bb_trend_rejoin` | ETHUSD | 1h | 66 | 68.18 | 1.568 | 56.2 | -22.34 | 24 | 11 | 0.46 | spike |
| 500 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.528 | 52.26 | -11.11 | 24 | 11 | 0.46 | spike |
| 501 | `bb_trend_rejoin` | ETHUSD | 1h | 126 | 69.84 | 1.469 | 50.13 | -13.19 | 24 | 11 | 0.46 | spike |
| 502 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.476 | 46.34 | -11.11 | 24 | 11 | 0.46 | spike |
| 503 | `rsi2_regime` | ETHUSD | 1h | 65 | 69.23 | 1.92 | 44.28 | -10.52 | 24 | 11 | 0.46 | spike |
| 504 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.45 | 43.18 | -11.11 | 24 | 11 | 0.46 | spike |
| 505 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 71.59 | 1.731 | 42.51 | -11.06 | 24 | 11 | 0.46 | spike |
| 506 | `bb_trend_rejoin` | ETHUSD | 1h | 219 | 67.12 | 1.249 | 41.84 | -15.93 | 24 | 11 | 0.46 | spike |
| 507 | `bb_trend_rejoin` | ETHUSD | 1h | 219 | 67.12 | 1.249 | 41.84 | -15.93 | 24 | 11 | 0.46 | spike |
| 508 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 71.94 | 1.374 | 40.46 | -12.26 | 24 | 11 | 0.46 | spike |
| 509 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 71.94 | 1.37 | 39.96 | -12.26 | 24 | 11 | 0.46 | spike |
| 510 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 71.94 | 1.369 | 39.82 | -12.26 | 24 | 11 | 0.46 | spike |
| 511 | `rsi2_regime` | ETHUSD | 1h | 58 | 70.69 | 1.544 | 38.25 | -11.44 | 24 | 11 | 0.46 | spike |
| 512 | `rsi2_regime` | ETHUSD | 1h | 72 | 75.0 | 1.638 | 37.9 | -10.41 | 24 | 11 | 0.46 | spike |
| 513 | `bb_trend_rejoin` | ETHUSD | 1h | 141 | 68.79 | 1.347 | 37.73 | -11.26 | 24 | 11 | 0.46 | spike |
| 514 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 71.94 | 1.352 | 37.5 | -12.26 | 24 | 11 | 0.46 | spike |
| 515 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.07 | 1.387 | 37.03 | -11.11 | 24 | 11 | 0.46 | spike |
| 516 | `rsi2_regime` | ETHUSD | 1h | 72 | 70.83 | 1.394 | 31.67 | -15.05 | 24 | 11 | 0.46 | spike |
| 517 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 64.84 | 1.421 | 31.48 | -14.11 | 24 | 11 | 0.46 | spike |
| 518 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 64.84 | 1.42 | 31.39 | -14.11 | 24 | 11 | 0.46 | spike |
| 519 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 67.76 | 1.193 | 29.76 | -17.18 | 24 | 11 | 0.46 | spike |
| 520 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 67.76 | 1.193 | 29.76 | -17.18 | 24 | 11 | 0.46 | spike |
| 521 | `rsi2_regime` | ETHUSD | 1h | 67 | 67.16 | 1.513 | 28.68 | -9.21 | 24 | 11 | 0.46 | spike |
| 522 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 67.76 | 1.184 | 28.21 | -17.18 | 24 | 11 | 0.46 | spike |
| 523 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 68.03 | 1.289 | 26.95 | -9.91 | 24 | 11 | 0.46 | spike |
| 524 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 69.32 | 1.381 | 22.35 | -10.83 | 24 | 11 | 0.46 | spike |
| 525 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 68.18 | 1.337 | 21.87 | -13.74 | 24 | 11 | 0.46 | spike |
| 526 | `rsi2_regime` | ETHUSD | 1h | 208 | 68.75 | 1.167 | 20.5 | -17.47 | 24 | 11 | 0.46 | spike |
| 527 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 68.18 | 1.335 | 19.84 | -10.83 | 24 | 11 | 0.46 | spike |
| 528 | `rsi2_regime` | ETHUSD | 1h | 73 | 64.38 | 1.314 | 18.95 | -9.21 | 24 | 11 | 0.46 | spike |
| 529 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 71.94 | 1.179 | 15.71 | -15.98 | 24 | 11 | 0.46 | spike |
| 530 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 68.18 | 1.207 | 12.6 | -12.51 | 24 | 11 | 0.46 | spike |
| 531 | `bb_trend_rejoin` | ETHUSD | 1h | 110 | 64.55 | 1.131 | 11.55 | -28.53 | 24 | 11 | 0.46 | spike |
| 532 | `rsi2_regime` | ETHUSD | 1h | 81 | 69.14 | 1.191 | 11.51 | -15.12 | 24 | 11 | 0.46 | spike |
| 533 | `rsi2_regime` | ETHUSD | 1h | 60 | 70.0 | 1.212 | 11.19 | -18.74 | 24 | 11 | 0.46 | spike |
| 534 | `bb_trend_rejoin` | ETHUSD | 1h | 161 | 67.7 | 1.109 | 10.84 | -18.71 | 24 | 11 | 0.46 | spike |
| 535 | `bb_trend_rejoin` | ETHUSD | 1h | 110 | 64.55 | 1.125 | 10.74 | -29.16 | 24 | 11 | 0.46 | spike |
| 536 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 68.18 | 1.163 | 9.38 | -12.3 | 24 | 11 | 0.46 | spike |
| 537 | `rsi2_regime` | ETHUSD | 1h | 80 | 65.0 | 1.157 | 7.88 | -11.88 | 24 | 11 | 0.46 | spike |
| 538 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 64.24 | 1.046 | 2.22 | -17.04 | 24 | 11 | 0.46 | spike |
| 539 | `rsi2_regime` | ETHUSD | 1h | 71 | 71.83 | 1.467 | 36.24 | -15.2 | 22 | 10 | 0.45 | spike |
| 540 | `rsi2_regime` | ETHUSD | 1h | 80 | 68.75 | 1.869 | 71.43 | -12.98 | 23 | 10 | 0.43 | spike |
| 541 | `rsi2_regime` | ETHUSD | 1h | 77 | 72.73 | 1.948 | 55.64 | -15.66 | 23 | 10 | 0.43 | spike |
| 542 | `bb_trend_rejoin` | ETHUSD | 1h | 137 | 63.5 | 1.447 | 51.88 | -11.09 | 23 | 10 | 0.43 | spike |
| 543 | `bb_trend_rejoin` | ETHUSD | 1h | 149 | 69.8 | 1.485 | 49.23 | -9.84 | 23 | 10 | 0.43 | spike |
| 544 | `bb_trend_rejoin` | ETHUSD | 1h | 219 | 67.58 | 1.273 | 47.02 | -15.93 | 23 | 10 | 0.43 | spike |
| 545 | `bb_trend_rejoin` | ETHUSD | 1h | 213 | 68.54 | 1.271 | 45.53 | -17.28 | 23 | 10 | 0.43 | spike |
| 546 | `bb_trend_rejoin` | ETHUSD | 1h | 213 | 68.54 | 1.263 | 43.74 | -17.28 | 23 | 10 | 0.43 | spike |
| 547 | `bb_trend_rejoin` | ETHUSD | 1h | 213 | 68.54 | 1.263 | 43.74 | -17.28 | 23 | 10 | 0.43 | spike |
| 548 | `rsi2_regime` | ETHUSD | 1h | 52 | 73.08 | 1.903 | 42.17 | -12.94 | 23 | 10 | 0.43 | spike |
| 549 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.41 | 41.05 | -9.11 | 23 | 10 | 0.43 | spike |
| 550 | `rsi2_regime` | ETHUSD | 1h | 73 | 69.86 | 1.587 | 40.52 | -12.92 | 23 | 10 | 0.43 | spike |
| 551 | `rsi2_regime` | ETHUSD | 1h | 225 | 68.89 | 1.198 | 40.37 | -18.43 | 23 | 10 | 0.43 | spike |
| 552 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 68.18 | 1.602 | 36.48 | -12.3 | 23 | 10 | 0.43 | spike |
| 553 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.449 | 32.67 | -9.95 | 23 | 10 | 0.43 | spike |
| 554 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.449 | 32.67 | -9.95 | 23 | 10 | 0.43 | spike |
| 555 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.449 | 32.67 | -9.95 | 23 | 10 | 0.43 | spike |
| 556 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.449 | 32.67 | -9.95 | 23 | 10 | 0.43 | spike |
| 557 | `bb_trend_rejoin` | ETHUSD | 1h | 131 | 64.89 | 1.275 | 29.4 | -16.1 | 23 | 10 | 0.43 | spike |
| 558 | `rsi2_regime` | ETHUSD | 1h | 70 | 71.43 | 1.671 | 27.81 | -12.89 | 23 | 10 | 0.43 | spike |
| 559 | `rsi2_regime` | ETHUSD | 1h | 121 | 72.73 | 1.287 | 27.36 | -21.97 | 23 | 10 | 0.43 | spike |
| 560 | `rsi2_regime` | ETHUSD | 1h | 52 | 76.92 | 1.773 | 20.3 | -11.18 | 23 | 10 | 0.43 | spike |
| 561 | `rsi2_regime` | ETHUSD | 1h | 53 | 75.47 | 1.752 | 19.94 | -11.18 | 23 | 10 | 0.43 | spike |
| 562 | `rsi2_regime` | ETHUSD | 1h | 52 | 71.15 | 1.625 | 17.82 | -12.56 | 23 | 10 | 0.43 | spike |
| 563 | `bb_trend_rejoin` | ETHUSD | 1h | 165 | 67.27 | 1.136 | 15.58 | -14.55 | 23 | 10 | 0.43 | spike |
| 564 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 72.66 | 1.166 | 14.26 | -15.06 | 23 | 10 | 0.43 | spike |
| 565 | `rsi2_regime` | ETHUSD | 1h | 52 | 75.0 | 1.486 | 14.0 | -14.77 | 23 | 10 | 0.43 | spike |
| 566 | `rsi2_regime` | ETHUSD | 1h | 185 | 67.57 | 1.104 | 11.45 | -18.82 | 23 | 10 | 0.43 | spike |
| 567 | `bb_trend_rejoin` | ETHUSD | 1h | 110 | 64.55 | 1.127 | 10.89 | -28.99 | 23 | 10 | 0.43 | spike |
| 568 | `bb_trend_rejoin` | ETHUSD | 1h | 110 | 64.55 | 1.116 | 9.68 | -28.99 | 23 | 10 | 0.43 | spike |
| 569 | `rsi2_regime` | ETHUSD | 1h | 303 | 68.98 | 1.059 | 6.45 | -28.93 | 23 | 10 | 0.43 | spike |
| 570 | `bb_trend_rejoin` | ETHUSD | 1h | 110 | 64.55 | 1.087 | 6.17 | -28.99 | 23 | 10 | 0.43 | spike |
| 571 | `bb_trend_rejoin` | ETHUSD | 1h | 110 | 64.55 | 1.087 | 6.17 | -28.99 | 23 | 10 | 0.43 | spike |
| 572 | `rsi2_regime` | ETHUSD | 1h | 115 | 69.57 | 1.478 | 44.77 | -27.02 | 21 | 9 | 0.43 | spike |
| 573 | `rsi2_regime` | ETHUSD | 1h | 52 | 78.85 | 2.077 | 43.47 | -11.3 | 21 | 9 | 0.43 | spike |
| 574 | `rsi2_regime` | ETHUSD | 1h | 52 | 76.92 | 1.883 | 38.63 | -13.59 | 21 | 9 | 0.43 | spike |
| 575 | `rsi2_regime` | ETHUSD | 1h | 370 | 64.86 | 1.152 | 24.48 | -26.97 | 21 | 9 | 0.43 | spike |
| 576 | `rsi2_regime` | ETHUSD | 1h | 53 | 71.7 | 1.696 | 18.08 | -13.88 | 21 | 9 | 0.43 | spike |
| 577 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.75 | 1.528 | 52.26 | -11.11 | 24 | 10 | 0.42 | spike |
| 578 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 67.69 | 1.362 | 50.9 | -11.49 | 24 | 10 | 0.42 | spike |
| 579 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 67.69 | 1.362 | 50.9 | -11.49 | 24 | 10 | 0.42 | spike |
| 580 | `bb_trend_rejoin` | ETHUSD | 1h | 107 | 71.03 | 1.62 | 45.36 | -7.72 | 24 | 10 | 0.42 | spike |
| 581 | `bb_trend_rejoin` | ETHUSD | 1h | 107 | 71.03 | 1.62 | 45.36 | -7.72 | 24 | 10 | 0.42 | spike |
| 582 | `bb_trend_rejoin` | ETHUSD | 1h | 107 | 71.03 | 1.62 | 45.36 | -7.72 | 24 | 10 | 0.42 | spike |
| 583 | `rsi2_regime` | ETHUSD | 1h | 77 | 68.83 | 1.494 | 41.22 | -12.17 | 24 | 10 | 0.42 | spike |
| 584 | `bb_trend_rejoin` | ETHUSD | 1h | 140 | 72.14 | 1.338 | 36.44 | -12.26 | 24 | 10 | 0.42 | spike |
| 585 | `bb_trend_rejoin` | ETHUSD | 1h | 219 | 65.75 | 1.214 | 34.77 | -17.33 | 24 | 10 | 0.42 | spike |
| 586 | `rsi2_regime` | ETHUSD | 1h | 60 | 68.33 | 1.467 | 34.57 | -13.66 | 24 | 10 | 0.42 | spike |
| 587 | `bb_trend_rejoin` | ETHUSD | 1h | 90 | 66.67 | 1.442 | 34.41 | -15.8 | 24 | 10 | 0.42 | spike |
| 588 | `bb_trend_rejoin` | ETHUSD | 1h | 90 | 66.67 | 1.441 | 34.32 | -15.8 | 24 | 10 | 0.42 | spike |
| 589 | `bb_trend_rejoin` | ETHUSD | 1h | 90 | 66.67 | 1.417 | 32.71 | -16.1 | 24 | 10 | 0.42 | spike |
| 590 | `rsi2_regime` | ETHUSD | 1h | 64 | 67.19 | 1.509 | 31.94 | -15.2 | 24 | 10 | 0.42 | spike |
| 591 | `bb_trend_rejoin` | ETHUSD | 1h | 208 | 68.27 | 1.199 | 29.61 | -20.36 | 24 | 10 | 0.42 | spike |
| 592 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.401 | 29.17 | -12.2 | 24 | 10 | 0.42 | spike |
| 593 | `bb_trend_rejoin` | ETHUSD | 1h | 188 | 68.09 | 1.205 | 29.09 | -19.58 | 24 | 10 | 0.42 | spike |
| 594 | `rsi2_regime` | ETHUSD | 1h | 52 | 73.08 | 1.533 | 26.51 | -12.38 | 24 | 10 | 0.42 | spike |
| 595 | `rsi2_regime` | ETHUSD | 1h | 175 | 69.71 | 1.179 | 19.52 | -17.57 | 24 | 10 | 0.42 | spike |
| 596 | `rsi2_regime` | ETHUSD | 1h | 135 | 67.41 | 1.229 | 14.72 | -18.51 | 24 | 10 | 0.42 | spike |
| 597 | `rsi2_regime` | ETHUSD | 1h | 210 | 68.57 | 1.094 | 13.24 | -19.32 | 24 | 10 | 0.42 | spike |
| 598 | `rsi2_regime` | ETHUSD | 1h | 58 | 74.14 | 1.282 | 12.97 | -15.72 | 24 | 10 | 0.42 | spike |
| 599 | `bb_trend_rejoin` | ETHUSD | 1h | 174 | 68.39 | 1.119 | 12.21 | -21.54 | 24 | 10 | 0.42 | spike |
| 600 | `bb_trend_rejoin` | ETHUSD | 1h | 208 | 64.9 | 1.098 | 11.26 | -24.58 | 24 | 10 | 0.42 | spike |
| 601 | `bb_trend_rejoin` | ETHUSD | 1h | 174 | 68.39 | 1.11 | 11.03 | -22.81 | 24 | 10 | 0.42 | spike |
| 602 | `bb_trend_rejoin` | ETHUSD | 1h | 166 | 64.46 | 1.067 | 3.05 | -25.94 | 24 | 10 | 0.42 | spike |
| 603 | `bb_trend_rejoin` | ETHUSD | 1h | 78 | 73.08 | 1.091 | 2.84 | -15.16 | 24 | 10 | 0.42 | spike |
| 604 | `rsi2_regime` | ETHUSD | 1h | 125 | 64.8 | 1.046 | 1.69 | -17.44 | 22 | 9 | 0.41 | spike |
| 605 | `bb_trend_rejoin` | ETHUSD | 1h | 87 | 73.56 | 1.719 | 57.31 | -20.56 | 23 | 9 | 0.39 | spike |
| 606 | `rsi2_regime` | ETHUSD | 1h | 52 | 78.85 | 2.4 | 51.5 | -10.65 | 23 | 9 | 0.39 | spike |
| 607 | `rsi2_regime` | ETHUSD | 1h | 52 | 76.92 | 2.37 | 50.17 | -6.68 | 23 | 9 | 0.39 | spike |
| 608 | `bb_trend_rejoin` | ETHUSD | 1h | 107 | 71.03 | 1.571 | 40.99 | -7.72 | 23 | 9 | 0.39 | spike |
| 609 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 67.69 | 1.297 | 39.49 | -11.77 | 23 | 9 | 0.39 | spike |
| 610 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 67.69 | 1.297 | 39.49 | -11.77 | 23 | 9 | 0.39 | spike |
| 611 | `rsi2_regime` | ETHUSD | 1h | 50 | 72.0 | 1.522 | 35.24 | -13.8 | 23 | 9 | 0.39 | spike |
| 612 | `rsi2_regime` | DOGEUSD | 1h | 65 | 67.69 | 1.48 | 29.01 | -9.7 | 23 | 9 | 0.39 | spike |
| 613 | `rsi2_regime` | ETHUSD | 1h | 64 | 71.88 | 1.477 | 27.39 | -12.61 | 23 | 9 | 0.39 | spike |
| 614 | `bb_trend_rejoin` | ETHUSD | 1h | 89 | 65.17 | 1.388 | 26.07 | -17.11 | 23 | 9 | 0.39 | spike |
| 615 | `rsi2_regime` | ETHUSD | 1h | 50 | 72.0 | 1.469 | 25.0 | -13.15 | 23 | 9 | 0.39 | spike |
| 616 | `bb_trend_rejoin` | ETHUSD | 1h | 89 | 66.29 | 1.353 | 24.25 | -17.11 | 23 | 9 | 0.39 | spike |
| 617 | `bb_trend_rejoin` | ETHUSD | 1h | 201 | 70.65 | 1.2 | 23.83 | -17.79 | 23 | 9 | 0.39 | spike |
| 618 | `bb_trend_rejoin` | ETHUSD | 1h | 208 | 65.38 | 1.169 | 23.61 | -24.58 | 23 | 9 | 0.39 | spike |
| 619 | `bb_trend_rejoin` | ETHUSD | 1h | 167 | 65.87 | 1.153 | 18.71 | -15.43 | 23 | 9 | 0.39 | spike |
| 620 | `rsi2_regime` | ETHUSD | 1h | 52 | 76.92 | 1.616 | 18.35 | -12.54 | 23 | 9 | 0.39 | spike |
| 621 | `rsi2_regime` | ETHUSD | 1h | 79 | 68.35 | 1.393 | 14.96 | -14.62 | 23 | 9 | 0.39 | spike |
| 622 | `rsi2_regime` | ETHUSD | 1h | 52 | 76.92 | 1.445 | 13.25 | -15.96 | 23 | 9 | 0.39 | spike |
| 623 | `rsi2_regime` | ETHUSD | 1h | 81 | 64.2 | 1.178 | 10.23 | -16.63 | 23 | 9 | 0.39 | spike |
| 624 | `rsi2_regime` | ADAUSD | 1h | 60 | 71.67 | 1.22 | 10.23 | -12.45 | 23 | 9 | 0.39 | spike |
| 625 | `bb_trend_rejoin` | ETHUSD | 1h | 110 | 60.91 | 1.098 | 6.27 | -12.27 | 23 | 9 | 0.39 | spike |
| 626 | `bb_trend_rejoin` | ETHUSD | 1h | 110 | 60.91 | 1.098 | 6.27 | -12.27 | 23 | 9 | 0.39 | spike |
| 627 | `rsi2_regime` | ETHUSD | 1h | 53 | 62.26 | 1.777 | 33.78 | -10.56 | 21 | 8 | 0.38 | spike |
| 628 | `rsi2_regime` | ETHUSD | 1h | 53 | 71.7 | 1.648 | 17.61 | -12.05 | 21 | 8 | 0.38 | spike |
| 629 | `rsi2_regime` | ETHUSD | 1h | 52 | 71.15 | 1.629 | 17.03 | -12.05 | 21 | 8 | 0.38 | spike |
| 630 | `rsi2_regime` | ETHUSD | 1h | 53 | 71.7 | 1.622 | 16.84 | -12.05 | 21 | 8 | 0.38 | spike |
| 631 | `rsi2_regime` | ETHUSD | 1h | 239 | 65.27 | 1.07 | 8.63 | -20.93 | 21 | 8 | 0.38 | spike |
| 632 | `rsi2_regime` | ETHUSD | 1h | 239 | 65.27 | 1.055 | 4.73 | -20.93 | 21 | 8 | 0.38 | spike |
| 633 | `bb_trend_rejoin` | ETHUSD | 1h | 65 | 72.31 | 1.978 | 86.08 | -10.27 | 24 | 9 | 0.38 | spike |
| 634 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.507 | 56.02 | -16.71 | 16 | 6 | 0.38 | spike |
| 635 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.507 | 56.02 | -16.71 | 16 | 6 | 0.38 | spike |
| 636 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 71.43 | 1.558 | 52.37 | -11.76 | 24 | 9 | 0.38 | spike |
| 637 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.421 | 49.36 | -16.0 | 16 | 6 | 0.38 | spike |
| 638 | `bb_trend_rejoin` | ETHUSD | 1h | 66 | 68.18 | 1.521 | 48.66 | -22.34 | 24 | 9 | 0.38 | spike |
| 639 | `bb_trend_rejoin` | ETHUSD | 1h | 66 | 68.18 | 1.521 | 48.66 | -22.34 | 24 | 9 | 0.38 | spike |
| 640 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.413 | 48.03 | -16.0 | 16 | 6 | 0.38 | spike |
| 641 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 71.43 | 1.502 | 47.71 | -12.45 | 24 | 9 | 0.38 | spike |
| 642 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.396 | 45.39 | -16.37 | 16 | 6 | 0.38 | spike |
| 643 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.401 | 45.33 | -16.74 | 16 | 6 | 0.38 | spike |
| 644 | `rsi2_regime` | ETHUSD | 1h | 74 | 70.27 | 1.71 | 44.44 | -14.08 | 24 | 9 | 0.38 | spike |
| 645 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.39 | 43.05 | -17.72 | 16 | 6 | 0.38 | spike |
| 646 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.386 | 42.52 | -17.72 | 16 | 6 | 0.38 | spike |
| 647 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.375 | 42.48 | -16.37 | 16 | 6 | 0.38 | spike |
| 648 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.383 | 42.09 | -17.72 | 16 | 6 | 0.38 | spike |
| 649 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.383 | 42.09 | -17.72 | 16 | 6 | 0.38 | spike |
| 650 | `bb_trend_rejoin` | ETHUSD | 1h | 107 | 71.03 | 1.571 | 40.99 | -7.72 | 24 | 9 | 0.38 | spike |
| 651 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.344 | 40.25 | -15.81 | 16 | 6 | 0.38 | spike |
| 652 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 70.07 | 1.43 | 40.22 | -11.18 | 24 | 9 | 0.38 | spike |
| 653 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 64.84 | 1.531 | 39.66 | -12.68 | 24 | 9 | 0.38 | spike |
| 654 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.349 | 39.18 | -16.74 | 16 | 6 | 0.38 | spike |
| 655 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.349 | 39.18 | -16.74 | 16 | 6 | 0.38 | spike |
| 656 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.334 | 38.89 | -15.88 | 16 | 6 | 0.38 | spike |
| 657 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.334 | 38.89 | -15.88 | 16 | 6 | 0.38 | spike |
| 658 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.334 | 38.89 | -15.88 | 16 | 6 | 0.38 | spike |
| 659 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.334 | 38.89 | -15.88 | 16 | 6 | 0.38 | spike |
| 660 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.346 | 38.76 | -16.74 | 16 | 6 | 0.38 | spike |
| 661 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.324 | 37.87 | -17.25 | 16 | 6 | 0.38 | spike |
| 662 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.324 | 37.87 | -17.25 | 16 | 6 | 0.38 | spike |
| 663 | `rsi2_regime` | ETHUSD | 1h | 65 | 69.23 | 1.673 | 37.35 | -9.19 | 24 | 9 | 0.38 | spike |
| 664 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.525 | 37.32 | -10.97 | 24 | 9 | 0.38 | spike |
| 665 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.338 | 37.32 | -16.37 | 16 | 6 | 0.38 | spike |
| 666 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.337 | 37.16 | -16.37 | 16 | 6 | 0.38 | spike |
| 667 | `rsi2_regime` | ETHUSD | 1h | 210 | 65.24 | 1.184 | 36.6 | -25.14 | 24 | 9 | 0.38 | spike |
| 668 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.508 | 36.01 | -10.64 | 24 | 9 | 0.38 | spike |
| 669 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.508 | 36.01 | -10.64 | 24 | 9 | 0.38 | spike |
| 670 | `rsi2_regime` | ETHUSD | 1h | 52 | 67.31 | 2.025 | 35.7 | -8.8 | 24 | 9 | 0.38 | spike |
| 671 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.304 | 34.88 | -20.64 | 16 | 6 | 0.38 | spike |
| 672 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.304 | 34.88 | -20.64 | 16 | 6 | 0.38 | spike |
| 673 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.304 | 34.88 | -20.64 | 16 | 6 | 0.38 | spike |
| 674 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.317 | 34.73 | -18.35 | 16 | 6 | 0.38 | spike |
| 675 | `zscore_revert` | ETHUSD | 1h | 126 | 72.22 | 1.315 | 34.33 | -18.35 | 16 | 6 | 0.38 | spike |
| 676 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.48 | 34.28 | -10.64 | 24 | 9 | 0.38 | spike |
| 677 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.48 | 34.28 | -10.64 | 24 | 9 | 0.38 | spike |
| 678 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 68.75 | 1.735 | 33.07 | -11.37 | 24 | 9 | 0.38 | spike |
| 679 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.449 | 32.67 | -9.95 | 24 | 9 | 0.38 | spike |
| 680 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 68.75 | 1.68 | 31.12 | -11.84 | 24 | 9 | 0.38 | spike |
| 681 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 68.75 | 1.68 | 31.12 | -11.84 | 24 | 9 | 0.38 | spike |
| 682 | `bb_trend_rejoin` | ETHUSD | 1h | 182 | 69.78 | 1.224 | 29.39 | -20.85 | 24 | 9 | 0.38 | spike |
| 683 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.401 | 29.17 | -12.2 | 24 | 9 | 0.38 | spike |
| 684 | `bb_trend_rejoin` | ETHUSD | 1h | 140 | 68.57 | 1.275 | 28.33 | -14.67 | 24 | 9 | 0.38 | spike |
| 685 | `bb_trend_rejoin` | ETHUSD | 1h | 208 | 68.27 | 1.192 | 28.26 | -20.36 | 24 | 9 | 0.38 | spike |
| 686 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 68.03 | 1.294 | 27.15 | -12.4 | 24 | 9 | 0.38 | spike |
| 687 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 68.03 | 1.294 | 27.15 | -12.4 | 24 | 9 | 0.38 | spike |
| 688 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.24 | 25.93 | -20.64 | 16 | 6 | 0.38 | spike |
| 689 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.24 | 25.93 | -20.64 | 16 | 6 | 0.38 | spike |
| 690 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.237 | 25.55 | -20.64 | 16 | 6 | 0.38 | spike |
| 691 | `rsi2_regime` | ETHUSD | 1h | 89 | 67.42 | 1.272 | 25.03 | -15.69 | 24 | 9 | 0.38 | spike |
| 692 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.333 | 24.51 | -12.2 | 24 | 9 | 0.38 | spike |
| 693 | `bb_trend_rejoin` | ETHUSD | 1h | 187 | 67.91 | 1.178 | 23.58 | -20.23 | 24 | 9 | 0.38 | spike |
| 694 | `bb_trend_rejoin` | ETHUSD | 1h | 187 | 67.91 | 1.178 | 23.58 | -20.23 | 24 | 9 | 0.38 | spike |
| 695 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 68.18 | 1.372 | 23.22 | -11.98 | 24 | 9 | 0.38 | spike |
| 696 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 68.75 | 1.518 | 22.46 | -11.37 | 24 | 9 | 0.38 | spike |
| 697 | `bb_trend_rejoin` | ETHUSD | 1h | 200 | 68.0 | 1.178 | 21.81 | -15.58 | 24 | 9 | 0.38 | spike |
| 698 | `rsi2_regime` | ETHUSD | 1h | 69 | 68.12 | 1.411 | 16.67 | -10.74 | 24 | 9 | 0.38 | spike |
| 699 | `rsi2_regime` | ETHUSD | 1h | 69 | 68.12 | 1.411 | 16.67 | -10.74 | 24 | 9 | 0.38 | spike |
| 700 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 66.98 | 1.206 | 14.2 | -12.2 | 24 | 9 | 0.38 | spike |
| 701 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 68.18 | 1.222 | 13.25 | -13.09 | 24 | 9 | 0.38 | spike |
| 702 | `bb_trend_rejoin` | ETHUSD | 1h | 174 | 67.82 | 1.115 | 11.55 | -22.5 | 24 | 9 | 0.38 | spike |
| 703 | `bb_trend_rejoin` | ETHUSD | 1h | 174 | 67.82 | 1.115 | 11.55 | -22.5 | 24 | 9 | 0.38 | spike |
| 704 | `rsi2_regime` | ETHUSD | 1h | 52 | 71.15 | 1.39 | 10.78 | -9.29 | 24 | 9 | 0.38 | spike |
| 705 | `bb_trend_rejoin` | ETHUSD | 1h | 174 | 68.39 | 1.106 | 10.08 | -22.29 | 24 | 9 | 0.38 | spike |
| 706 | `bb_trend_rejoin` | ETHUSD | 1h | 141 | 67.38 | 1.121 | 9.31 | -24.23 | 24 | 9 | 0.38 | spike |
| 707 | `bb_trend_rejoin` | ETHUSD | 1h | 78 | 73.08 | 1.241 | 9.1 | -14.37 | 24 | 9 | 0.38 | spike |
| 708 | `bb_trend_rejoin` | ETHUSD | 1h | 141 | 67.38 | 1.111 | 8.27 | -24.23 | 24 | 9 | 0.38 | spike |
| 709 | `rsi2_regime` | ADAUSD | 1h | 78 | 73.08 | 1.136 | 7.88 | -21.99 | 24 | 9 | 0.38 | spike |
| 710 | `bb_trend_rejoin` | ETHUSD | 1h | 174 | 67.82 | 1.079 | 5.87 | -25.15 | 24 | 9 | 0.38 | spike |
| 711 | `bb_trend_rejoin` | ETHUSD | 1h | 206 | 66.5 | 1.069 | 5.87 | -22.89 | 24 | 9 | 0.38 | spike |
| 712 | `rsi2_regime` | ETHUSD | 1h | 95 | 66.32 | 1.101 | 5.55 | -14.06 | 24 | 9 | 0.38 | spike |
| 713 | `rsi2_regime` | ETHUSD | 1h | 320 | 65.62 | 1.045 | 1.62 | -29.69 | 24 | 9 | 0.38 | spike |
| 714 | `rsi2_regime` | ETHUSD | 1h | 166 | 71.08 | 1.119 | 11.88 | -21.8 | 22 | 8 | 0.36 | spike |
| 715 | `rsi2_regime` | ETHUSD | 1h | 52 | 71.15 | 2.091 | 56.6 | -7.38 | 23 | 8 | 0.35 | spike |
| 716 | `rsi2_regime` | ETHUSD | 1h | 52 | 71.15 | 2.027 | 52.52 | -7.38 | 23 | 8 | 0.35 | spike |
| 717 | `bb_trend_rejoin` | ETHUSD | 1h | 137 | 62.77 | 1.44 | 50.12 | -13.35 | 23 | 8 | 0.35 | spike |
| 718 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.622 | 42.71 | -11.83 | 23 | 8 | 0.35 | spike |
| 719 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.622 | 42.71 | -11.83 | 23 | 8 | 0.35 | spike |
| 720 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.622 | 42.71 | -11.83 | 23 | 8 | 0.35 | spike |
| 721 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.622 | 42.71 | -11.83 | 23 | 8 | 0.35 | spike |
| 722 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.622 | 42.71 | -11.83 | 23 | 8 | 0.35 | spike |
| 723 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 71.43 | 1.435 | 41.7 | -12.41 | 23 | 8 | 0.35 | spike |
| 724 | `bb_trend_rejoin` | ETHUSD | 1h | 147 | 71.43 | 1.435 | 41.7 | -12.41 | 23 | 8 | 0.35 | spike |
| 725 | `rsi2_regime` | ETHUSD | 1h | 104 | 63.46 | 1.433 | 41.22 | -10.06 | 23 | 8 | 0.35 | spike |
| 726 | `rsi2_regime` | ETHUSD | 1h | 65 | 64.62 | 1.553 | 39.91 | -13.49 | 23 | 8 | 0.35 | spike |
| 727 | `rsi2_regime` | ETHUSD | 1h | 128 | 67.19 | 1.323 | 39.74 | -22.16 | 23 | 8 | 0.35 | spike |
| 728 | `rsi2_regime` | ETHUSD | 1h | 248 | 66.94 | 1.198 | 39.35 | -24.62 | 23 | 8 | 0.35 | spike |
| 729 | `rsi2_regime` | ETHUSD | 1h | 52 | 67.31 | 2.045 | 36.32 | -11.33 | 23 | 8 | 0.35 | spike |
| 730 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.513 | 36.25 | -12.36 | 23 | 8 | 0.35 | spike |
| 731 | `rsi2_regime` | ETHUSD | 1h | 53 | 73.58 | 1.72 | 34.44 | -15.4 | 23 | 8 | 0.35 | spike |
| 732 | `rsi2_regime` | ETHUSD | 1h | 52 | 69.23 | 1.934 | 32.23 | -7.07 | 23 | 8 | 0.35 | spike |
| 733 | `rsi2_regime` | ETHUSD | 1h | 50 | 76.0 | 2.166 | 30.33 | -9.92 | 23 | 8 | 0.35 | spike |
| 734 | `rsi2_regime` | ETHUSD | 1h | 225 | 69.33 | 1.16 | 30.0 | -29.7 | 23 | 8 | 0.35 | spike |
| 735 | `rsi2_regime` | ETHUSD | 1h | 88 | 67.05 | 1.293 | 27.59 | -16.25 | 23 | 8 | 0.35 | spike |
| 736 | `rsi2_regime` | ETHUSD | 1h | 53 | 69.81 | 1.525 | 27.08 | -18.11 | 23 | 8 | 0.35 | spike |
| 737 | `rsi2_regime` | ETHUSD | 1h | 72 | 69.44 | 1.335 | 26.91 | -23.17 | 23 | 8 | 0.35 | spike |
| 738 | `rsi2_regime` | DOGEUSD | 1h | 61 | 65.57 | 1.447 | 26.51 | -15.68 | 23 | 8 | 0.35 | spike |
| 739 | `rsi2_regime` | ETHUSD | 1h | 72 | 69.44 | 1.312 | 25.07 | -23.17 | 23 | 8 | 0.35 | spike |
| 740 | `rsi2_regime` | ETHUSD | 1h | 69 | 68.12 | 1.526 | 21.66 | -9.23 | 23 | 8 | 0.35 | spike |
| 741 | `rsi2_regime` | ETHUSD | 1h | 52 | 73.08 | 1.435 | 20.27 | -20.48 | 23 | 8 | 0.35 | spike |
| 742 | `rsi2_regime` | ETHUSD | 1h | 74 | 68.92 | 1.42 | 19.25 | -10.49 | 23 | 8 | 0.35 | spike |
| 743 | `rsi2_regime` | ETHUSD | 1h | 53 | 67.92 | 1.444 | 19.15 | -17.16 | 23 | 8 | 0.35 | spike |
| 744 | `rsi2_regime` | ETHUSD | 1h | 83 | 73.49 | 1.449 | 18.36 | -11.07 | 23 | 8 | 0.35 | spike |
| 745 | `bb_trend_rejoin` | ETHUSD | 1h | 177 | 67.8 | 1.126 | 14.12 | -25.4 | 23 | 8 | 0.35 | spike |
| 746 | `rsi2_regime` | ETHUSD | 1h | 99 | 67.68 | 1.191 | 11.14 | -16.14 | 23 | 8 | 0.35 | spike |
| 747 | `rsi2_regime` | ETHUSD | 1h | 87 | 63.22 | 1.162 | 11.07 | -20.91 | 23 | 8 | 0.35 | spike |
| 748 | `rsi2_regime` | ETHUSD | 1h | 71 | 73.24 | 1.189 | 10.76 | -20.15 | 23 | 8 | 0.35 | spike |
| 749 | `rsi2_regime` | ETHUSD | 1h | 178 | 63.48 | 1.131 | 10.57 | -14.72 | 23 | 8 | 0.35 | spike |
| 750 | `rsi2_regime` | ETHUSD | 1h | 180 | 62.22 | 1.103 | 7.9 | -16.58 | 23 | 8 | 0.35 | spike |
| 751 | `rsi2_regime` | ADAUSD | 1h | 59 | 71.19 | 1.181 | 7.66 | -14.64 | 23 | 8 | 0.35 | spike |
| 752 | `rsi2_regime` | ADAUSD | 1h | 122 | 60.66 | 1.072 | 4.04 | -16.0 | 23 | 8 | 0.35 | spike |
| 753 | `bb_trend_rejoin` | ETHUSD | 1h | 65 | 70.77 | 1.896 | 83.65 | -10.27 | 24 | 8 | 0.33 | spike |
| 754 | `rsi2_regime` | ETHUSD | 1h | 52 | 78.85 | 2.485 | 55.43 | -10.71 | 21 | 7 | 0.33 | spike |
| 755 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 63.74 | 1.649 | 48.73 | -9.63 | 24 | 8 | 0.33 | spike |
| 756 | `bb_trend_rejoin` | ETHUSD | 1h | 90 | 66.67 | 1.556 | 43.98 | -15.24 | 24 | 8 | 0.33 | spike |
| 757 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 63.74 | 1.595 | 42.99 | -10.09 | 24 | 8 | 0.33 | spike |
| 758 | `bb_trend_rejoin` | ETHUSD | 1h | 95 | 69.47 | 1.363 | 42.82 | -17.06 | 24 | 8 | 0.33 | spike |
| 759 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 71.94 | 1.4 | 42.32 | -12.99 | 24 | 8 | 0.33 | spike |
| 760 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 63.74 | 1.568 | 40.57 | -10.09 | 24 | 8 | 0.33 | spike |
| 761 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 61.54 | 1.538 | 38.31 | -11.56 | 24 | 8 | 0.33 | spike |
| 762 | `bb_trend_rejoin` | ETHUSD | 1h | 95 | 68.42 | 1.324 | 37.11 | -17.06 | 24 | 8 | 0.33 | spike |
| 763 | `bb_trend_rejoin` | ETHUSD | 1h | 107 | 70.09 | 1.497 | 36.51 | -7.72 | 24 | 8 | 0.33 | spike |
| 764 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 61.54 | 1.517 | 36.48 | -12.95 | 24 | 8 | 0.33 | spike |
| 765 | `bb_trend_rejoin` | ETHUSD | 1h | 127 | 67.72 | 1.333 | 34.75 | -13.51 | 24 | 8 | 0.33 | spike |
| 766 | `rsi2_regime` | DOGEUSD | 1h | 57 | 68.42 | 1.487 | 30.71 | -18.19 | 24 | 8 | 0.33 | spike |
| 767 | `rsi2_regime` | ETHUSD | 1h | 72 | 72.22 | 1.545 | 30.65 | -10.9 | 24 | 8 | 0.33 | spike |
| 768 | `rsi2_regime` | ETHUSD | 1h | 52 | 69.23 | 2.285 | 29.2 | -8.09 | 24 | 8 | 0.33 | spike |
| 769 | `bb_trend_rejoin` | ETHUSD | 1h | 140 | 68.57 | 1.275 | 28.39 | -14.84 | 24 | 8 | 0.33 | spike |
| 770 | `bb_trend_rejoin` | ETHUSD | 1h | 140 | 68.57 | 1.275 | 28.39 | -14.84 | 24 | 8 | 0.33 | spike |
| 771 | `bb_trend_rejoin` | ETHUSD | 1h | 140 | 68.57 | 1.275 | 28.39 | -14.84 | 24 | 8 | 0.33 | spike |
| 772 | `rsi2_regime` | ETHUSD | 1h | 53 | 73.58 | 1.57 | 27.1 | -13.87 | 24 | 8 | 0.33 | spike |
| 773 | `rsi2_regime` | ETHUSD | 1h | 183 | 66.12 | 1.174 | 26.54 | -20.56 | 21 | 7 | 0.33 | spike |
| 774 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 68.75 | 1.543 | 24.98 | -13.78 | 24 | 8 | 0.33 | spike |
| 775 | `rsi2_regime` | ETHUSD | 1h | 52 | 76.92 | 1.853 | 24.76 | -7.38 | 21 | 7 | 0.33 | spike |
| 776 | `rsi2_regime` | ETHUSD | 1h | 72 | 72.22 | 1.416 | 24.19 | -14.15 | 24 | 8 | 0.33 | spike |
| 777 | `rsi2_regime` | ETHUSD | 1h | 52 | 78.85 | 1.831 | 23.38 | -8.35 | 21 | 7 | 0.33 | spike |
| 778 | `rsi2_regime` | DOGEUSD | 1h | 73 | 64.38 | 1.318 | 22.96 | -13.43 | 24 | 8 | 0.33 | spike |
| 779 | `rsi2_regime` | ETHUSD | 1h | 52 | 71.15 | 1.554 | 21.22 | -11.58 | 24 | 8 | 0.33 | spike |
| 780 | `rsi2_regime` | ETHUSD | 1h | 72 | 72.22 | 1.354 | 20.89 | -12.53 | 24 | 8 | 0.33 | spike |
| 781 | `bb_trend_rejoin` | ETHUSD | 1h | 132 | 62.88 | 1.198 | 20.02 | -21.2 | 24 | 8 | 0.33 | spike |
| 782 | `rsi2_regime` | ETHUSD | 1h | 53 | 67.92 | 1.489 | 19.75 | -11.19 | 24 | 8 | 0.33 | spike |
| 783 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 68.75 | 1.4 | 17.81 | -13.78 | 24 | 8 | 0.33 | spike |
| 784 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 63.55 | 1.128 | 16.68 | -17.64 | 24 | 8 | 0.33 | spike |
| 785 | `rsi2_regime` | ETHUSD | 1h | 66 | 66.67 | 1.331 | 16.31 | -8.72 | 24 | 8 | 0.33 | spike |
| 786 | `rsi2_regime` | ETHUSD | 1h | 129 | 66.67 | 1.258 | 15.93 | -10.96 | 24 | 8 | 0.33 | spike |
| 787 | `bb_trend_rejoin` | ETHUSD | 1h | 78 | 73.08 | 1.357 | 13.12 | -7.07 | 24 | 8 | 0.33 | spike |
| 788 | `bb_trend_rejoin` | ETHUSD | 1h | 78 | 73.08 | 1.347 | 12.69 | -7.07 | 24 | 8 | 0.33 | spike |
| 789 | `bb_trend_rejoin` | ETHUSD | 1h | 78 | 71.79 | 1.305 | 11.4 | -12.15 | 24 | 8 | 0.33 | spike |
| 790 | `bb_trend_rejoin` | ETHUSD | 1h | 78 | 74.36 | 1.286 | 10.83 | -9.68 | 24 | 8 | 0.33 | spike |
| 791 | `bb_trend_rejoin` | ETHUSD | 1h | 174 | 67.82 | 1.106 | 10.12 | -21.38 | 24 | 8 | 0.33 | spike |
| 792 | `bb_trend_rejoin` | ETHUSD | 1h | 276 | 67.39 | 1.075 | 9.93 | -23.52 | 24 | 8 | 0.33 | spike |
| 793 | `bb_trend_rejoin` | ETHUSD | 1h | 276 | 67.39 | 1.075 | 9.93 | -23.52 | 24 | 8 | 0.33 | spike |
| 794 | `bb_trend_rejoin` | ETHUSD | 1h | 174 | 67.82 | 1.075 | 5.68 | -21.38 | 24 | 8 | 0.33 | spike |
| 795 | `bb_trend_rejoin` | ETHUSD | 1h | 174 | 67.82 | 1.075 | 5.68 | -21.38 | 24 | 8 | 0.33 | spike |
| 796 | `bb_trend_rejoin` | ETHUSD | 1h | 65 | 72.31 | 1.792 | 70.9 | -9.84 | 22 | 7 | 0.32 | spike |
| 797 | `rsi2_regime` | ETHUSD | 1h | 61 | 67.21 | 1.592 | 41.38 | -13.48 | 22 | 7 | 0.32 | spike |
| 798 | `rsi2_regime` | ETHUSD | 1h | 90 | 66.67 | 1.533 | 36.51 | -9.72 | 22 | 7 | 0.32 | spike |
| 799 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 71.7 | 1.513 | 36.25 | -12.36 | 22 | 7 | 0.32 | spike |
| 800 | `rsi2_regime` | ETHUSD | 1h | 68 | 67.65 | 1.475 | 28.12 | -9.19 | 22 | 7 | 0.32 | spike |
| 801 | `rsi2_regime` | ETHUSD | 1h | 152 | 65.79 | 1.128 | 10.91 | -23.38 | 22 | 7 | 0.32 | spike |
| 802 | `bb_trend_rejoin` | ETHUSD | 1h | 141 | 67.38 | 1.13 | 10.08 | -17.76 | 22 | 7 | 0.32 | spike |
| 803 | `rsi2_regime` | ETHUSD | 1h | 149 | 66.44 | 1.446 | 73.42 | -22.04 | 19 | 6 | 0.32 | spike |
| 804 | `zscore_revert` | ETHUSD | 1h | 137 | 62.77 | 1.407 | 45.49 | -13.35 | 19 | 6 | 0.32 | spike |
| 805 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.394 | 44.72 | -14.43 | 19 | 6 | 0.32 | spike |
| 806 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.336 | 36.49 | -14.43 | 19 | 6 | 0.32 | spike |
| 807 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.336 | 36.49 | -14.43 | 19 | 6 | 0.32 | spike |
| 808 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.336 | 36.49 | -14.43 | 19 | 6 | 0.32 | spike |
| 809 | `zscore_revert` | ETHUSD | 1h | 128 | 67.97 | 1.337 | 36.39 | -14.52 | 19 | 6 | 0.32 | spike |
| 810 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.315 | 35.36 | -16.56 | 19 | 6 | 0.32 | spike |
| 811 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.315 | 35.36 | -16.56 | 19 | 6 | 0.32 | spike |
| 812 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.315 | 35.36 | -16.56 | 19 | 6 | 0.32 | spike |
| 813 | `zscore_revert` | ETHUSD | 1h | 128 | 67.97 | 1.285 | 30.4 | -15.31 | 19 | 6 | 0.32 | spike |
| 814 | `zscore_revert` | ETHUSD | 1h | 128 | 67.97 | 1.285 | 30.4 | -15.31 | 19 | 6 | 0.32 | spike |
| 815 | `zscore_revert` | ETHUSD | 1h | 128 | 67.97 | 1.285 | 30.4 | -15.31 | 19 | 6 | 0.32 | spike |
| 816 | `zscore_revert` | ETHUSD | 1h | 128 | 67.97 | 1.279 | 29.54 | -15.31 | 19 | 6 | 0.32 | spike |
| 817 | `zscore_revert` | ETHUSD | 1h | 128 | 67.97 | 1.279 | 29.54 | -15.31 | 19 | 6 | 0.32 | spike |
| 818 | `zscore_revert` | ETHUSD | 1h | 128 | 67.97 | 1.272 | 28.53 | -15.31 | 19 | 6 | 0.32 | spike |
| 819 | `zscore_revert` | ETHUSD | 1h | 128 | 67.97 | 1.272 | 28.53 | -15.31 | 19 | 6 | 0.32 | spike |
| 820 | `zscore_revert` | ETHUSD | 1h | 128 | 67.97 | 1.272 | 28.53 | -15.31 | 19 | 6 | 0.32 | spike |
| 821 | `zscore_revert` | ETHUSD | 1h | 128 | 67.97 | 1.272 | 28.53 | -15.31 | 19 | 6 | 0.32 | spike |
| 822 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.264 | 28.32 | -16.56 | 19 | 6 | 0.32 | spike |
| 823 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.259 | 28.16 | -19.33 | 19 | 6 | 0.32 | spike |
| 824 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.259 | 28.16 | -19.33 | 19 | 6 | 0.32 | spike |
| 825 | `zscore_revert` | ETHUSD | 1h | 132 | 64.39 | 1.261 | 27.64 | -21.2 | 19 | 6 | 0.32 | spike |
| 826 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.251 | 26.47 | -16.56 | 19 | 6 | 0.32 | spike |
| 827 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.251 | 26.47 | -16.56 | 19 | 6 | 0.32 | spike |
| 828 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.251 | 26.47 | -16.56 | 19 | 6 | 0.32 | spike |
| 829 | `zscore_revert` | ETHUSD | 1h | 129 | 67.44 | 1.251 | 26.47 | -16.56 | 19 | 6 | 0.32 | spike |
| 830 | `rsi2_regime` | ETHUSD | 1h | 139 | 66.19 | 1.111 | 9.73 | -17.23 | 19 | 6 | 0.32 | spike |
| 831 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 63.74 | 1.645 | 48.69 | -11.53 | 23 | 7 | 0.30 | spike |
| 832 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 63.74 | 1.601 | 44.27 | -11.05 | 23 | 7 | 0.30 | spike |
| 833 | `bb_trend_rejoin` | ETHUSD | 1h | 91 | 63.74 | 1.58 | 42.79 | -11.53 | 23 | 7 | 0.30 | spike |
| 834 | `bb_trend_rejoin` | ETHUSD | 1h | 148 | 64.19 | 1.338 | 38.7 | -9.74 | 23 | 7 | 0.30 | spike |
| 835 | `bb_trend_rejoin` | ETHUSD | 1h | 148 | 64.19 | 1.337 | 38.63 | -9.43 | 23 | 7 | 0.30 | spike |
| 836 | `bb_trend_rejoin` | ETHUSD | 1h | 148 | 64.19 | 1.335 | 38.21 | -9.74 | 23 | 7 | 0.30 | spike |
| 837 | `rsi2_regime` | ETHUSD | 1h | 71 | 70.42 | 1.7 | 37.34 | -13.52 | 23 | 7 | 0.30 | spike |
| 838 | `bb_trend_rejoin` | DOGEUSD | 1h | 63 | 68.25 | 1.788 | 36.14 | -16.43 | 23 | 7 | 0.30 | spike |
| 839 | `rsi2_regime` | ETHUSD | 1h | 112 | 66.96 | 1.418 | 28.23 | -7.66 | 23 | 7 | 0.30 | spike |
| 840 | `rsi2_regime` | DOGEUSD | 1h | 105 | 69.52 | 1.311 | 24.06 | -12.22 | 23 | 7 | 0.30 | spike |
| 841 | `rsi2_regime` | DOGEUSD | 1h | 74 | 66.22 | 1.324 | 23.99 | -19.52 | 23 | 7 | 0.30 | spike |
| 842 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 68.75 | 1.534 | 22.76 | -13.89 | 23 | 7 | 0.30 | spike |
| 843 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 68.75 | 1.534 | 22.76 | -13.89 | 23 | 7 | 0.30 | spike |
| 844 | `rsi2_regime` | ETHUSD | 1h | 72 | 72.22 | 1.352 | 22.57 | -19.06 | 23 | 7 | 0.30 | spike |
| 845 | `bb_trend_rejoin` | ETHUSD | 1h | 89 | 64.04 | 1.331 | 22.29 | -17.11 | 23 | 7 | 0.30 | spike |
| 846 | `rsi2_regime` | ETHUSD | 1h | 122 | 66.39 | 1.194 | 14.82 | -9.23 | 23 | 7 | 0.30 | spike |
| 847 | `bb_trend_rejoin` | ETHUSD | 1h | 177 | 67.8 | 1.111 | 11.7 | -25.53 | 23 | 7 | 0.30 | spike |
| 848 | `bb_trend_rejoin` | ETHUSD | 1h | 78 | 71.79 | 1.305 | 11.63 | -13.04 | 23 | 7 | 0.30 | spike |
| 849 | `rsi2_regime` | ETHUSD | 1h | 58 | 70.69 | 1.234 | 11.03 | -12.61 | 23 | 7 | 0.30 | spike |
| 850 | `rsi2_regime` | ETHUSD | 1h | 80 | 62.5 | 1.236 | 10.34 | -12.13 | 23 | 7 | 0.30 | spike |
| 851 | `rsi2_regime` | ADAUSD | 1h | 70 | 70.0 | 1.131 | 6.89 | -17.72 | 23 | 7 | 0.30 | spike |
| 852 | `rsi2_regime` | ETHUSD | 1h | 53 | 69.81 | 1.244 | 6.85 | -14.96 | 23 | 7 | 0.30 | spike |
| 853 | `bb_trend_rejoin` | ETHUSD | 1h | 212 | 64.62 | 1.065 | 6.14 | -22.97 | 23 | 7 | 0.30 | spike |
| 854 | `bb_trend_rejoin` | ETHUSD | 1h | 78 | 71.79 | 1.129 | 4.54 | -13.02 | 23 | 7 | 0.30 | spike |
| 855 | `bb_trend_rejoin` | ETHUSD | 1h | 126 | 69.84 | 1.485 | 51.79 | -13.19 | 24 | 7 | 0.29 | spike |
| 856 | `bb_trend_rejoin` | ETHUSD | 1h | 66 | 68.18 | 1.573 | 51.35 | -18.39 | 24 | 7 | 0.29 | spike |
| 857 | `bb_trend_rejoin` | ETHUSD | 1h | 195 | 67.18 | 1.351 | 50.31 | -11.49 | 24 | 7 | 0.29 | spike |
| 858 | `bb_trend_rejoin` | DOGEUSD | 1h | 63 | 68.25 | 1.803 | 37.98 | -14.33 | 24 | 7 | 0.29 | spike |
| 859 | `bb_trend_rejoin` | DOGEUSD | 1h | 63 | 68.25 | 1.803 | 37.98 | -14.33 | 24 | 7 | 0.29 | spike |
| 860 | `bb_trend_rejoin` | DOGEUSD | 1h | 63 | 68.25 | 1.803 | 37.98 | -14.33 | 24 | 7 | 0.29 | spike |
| 861 | `rsi2_regime` | ETHUSD | 1h | 178 | 66.85 | 1.246 | 34.81 | -20.19 | 24 | 7 | 0.29 | spike |
| 862 | `bb_trend_rejoin` | DOGEUSD | 1h | 63 | 68.25 | 1.734 | 34.51 | -14.33 | 24 | 7 | 0.29 | spike |
| 863 | `rsi2_regime` | ETHUSD | 1h | 56 | 69.64 | 1.653 | 34.17 | -14.24 | 24 | 7 | 0.29 | spike |
| 864 | `bb_trend_rejoin` | ETHUSD | 1h | 142 | 66.9 | 1.303 | 30.4 | -10.73 | 24 | 7 | 0.29 | spike |
| 865 | `bb_trend_rejoin` | ETHUSD | 1h | 148 | 62.16 | 1.267 | 29.55 | -14.11 | 24 | 7 | 0.29 | spike |
| 866 | `rsi2_regime` | ETHUSD | 1h | 93 | 61.29 | 1.335 | 28.57 | -15.18 | 24 | 7 | 0.29 | spike |
| 867 | `bb_trend_rejoin` | ETHUSD | 1h | 142 | 66.9 | 1.275 | 27.23 | -11.96 | 24 | 7 | 0.29 | spike |
| 868 | `bb_trend_rejoin` | ETHUSD | 1h | 107 | 67.29 | 1.355 | 25.12 | -10.48 | 24 | 7 | 0.29 | spike |
| 869 | `rsi2_regime` | ETHUSD | 1h | 60 | 68.33 | 1.348 | 24.64 | -13.61 | 24 | 7 | 0.29 | spike |
| 870 | `bb_trend_rejoin` | ETHUSD | 1h | 148 | 62.16 | 1.228 | 24.28 | -14.11 | 24 | 7 | 0.29 | spike |
| 871 | `rsi2_regime` | ETHUSD | 1h | 64 | 70.31 | 1.607 | 23.93 | -14.18 | 24 | 7 | 0.29 | spike |
| 872 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 66.98 | 1.236 | 16.44 | -10.32 | 24 | 7 | 0.29 | spike |
| 873 | `bb_trend_rejoin` | ETHUSD | 1h | 106 | 66.98 | 1.236 | 16.44 | -10.32 | 24 | 7 | 0.29 | spike |
| 874 | `rsi2_regime` | ETHUSD | 1h | 90 | 67.78 | 1.195 | 16.19 | -17.99 | 24 | 7 | 0.29 | spike |
| 875 | `bb_trend_rejoin` | ETHUSD | 1h | 174 | 67.82 | 1.089 | 7.7 | -22.5 | 24 | 7 | 0.29 | spike |
| 876 | `bb_trend_rejoin` | ETHUSD | 1h | 209 | 65.55 | 1.063 | 5.29 | -21.48 | 24 | 7 | 0.29 | spike |
| 877 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 63.55 | 1.044 | 2.88 | -24.61 | 24 | 7 | 0.29 | spike |
| 878 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 63.55 | 1.044 | 2.88 | -24.61 | 24 | 7 | 0.29 | spike |
| 879 | `zscore_revert` | ETHUSD | 1h | 66 | 69.7 | 1.675 | 68.13 | -22.34 | 21 | 6 | 0.29 | spike |
| 880 | `zscore_revert` | ETHUSD | 1h | 66 | 69.7 | 1.675 | 68.13 | -22.34 | 21 | 6 | 0.29 | spike |
| 881 | `zscore_revert` | ETHUSD | 1h | 66 | 69.7 | 1.675 | 68.13 | -22.34 | 21 | 6 | 0.29 | spike |
| 882 | `rsi2_regime` | ETHUSD | 1h | 52 | 71.15 | 2.139 | 60.75 | -7.84 | 21 | 6 | 0.29 | spike |
| 883 | `rsi2_regime` | ETHUSD | 1h | 150 | 68.0 | 1.323 | 44.25 | -14.07 | 21 | 6 | 0.29 | spike |
| 884 | `rsi2_regime` | ETHUSD | 1h | 53 | 73.58 | 1.569 | 25.55 | -18.28 | 21 | 6 | 0.29 | spike |
| 885 | `rsi2_regime` | ETHUSD | 1h | 81 | 66.67 | 1.3 | 18.57 | -14.9 | 21 | 6 | 0.29 | spike |
| 886 | `rsi2_regime` | ETHUSD | 1h | 54 | 66.67 | 1.302 | 10.36 | -12.69 | 21 | 6 | 0.29 | spike |
| 887 | `rsi2_regime` | ETHUSD | 1h | 84 | 61.9 | 1.116 | 4.86 | -9.34 | 22 | 6 | 0.27 | spike |
| 888 | `bb_trend_rejoin` | ETHUSD | 1h | 65 | 70.77 | 1.665 | 61.1 | -9.84 | 23 | 6 | 0.26 | spike |
| 889 | `bb_trend_rejoin` | ETHUSD | 1h | 141 | 70.92 | 1.407 | 45.2 | -9.76 | 23 | 6 | 0.26 | spike |
| 890 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 67.19 | 2.04 | 39.73 | -11.95 | 23 | 6 | 0.26 | spike |
| 891 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 67.19 | 1.899 | 33.87 | -11.95 | 23 | 6 | 0.26 | spike |
| 892 | `rsi2_regime` | ETHUSD | 1h | 329 | 63.83 | 1.174 | 33.72 | -11.21 | 23 | 6 | 0.26 | spike |
| 893 | `bb_trend_rejoin` | ETHUSD | 1h | 107 | 69.16 | 1.384 | 28.69 | -7.72 | 23 | 6 | 0.26 | spike |
| 894 | `rsi2_regime` | ETHUSD | 1h | 53 | 67.92 | 1.899 | 23.86 | -11.19 | 23 | 6 | 0.26 | spike |
| 895 | `bb_trend_rejoin` | ETHUSD | 1h | 150 | 66.67 | 1.239 | 21.84 | -9.84 | 23 | 6 | 0.26 | spike |
| 896 | `rsi2_regime` | ETHUSD | 1h | 72 | 63.89 | 1.287 | 20.08 | -18.99 | 23 | 6 | 0.26 | spike |
| 897 | `rsi2_regime` | ETHUSD | 1h | 80 | 65.0 | 1.212 | 13.34 | -15.85 | 23 | 6 | 0.26 | spike |
| 898 | `bb_trend_rejoin` | ETHUSD | 1h | 83 | 63.86 | 1.23 | 10.45 | -13.94 | 23 | 6 | 0.26 | spike |
| 899 | `bb_trend_rejoin` | ETHUSD | 1h | 141 | 66.67 | 1.1 | 6.77 | -17.68 | 23 | 6 | 0.26 | spike |
| 900 | `rsi2_regime` | DOGEUSD | 1h | 76 | 71.05 | 1.973 | 85.98 | -13.71 | 24 | 6 | 0.25 | spike |
| 901 | `rsi2_regime` | ETHUSD | 1h | 104 | 62.5 | 1.555 | 65.01 | -13.31 | 24 | 6 | 0.25 | spike |
| 902 | `rsi2_regime` | ETHUSD | 1h | 71 | 69.01 | 1.675 | 43.44 | -11.75 | 24 | 6 | 0.25 | spike |
| 903 | `rsi2_regime` | DOGEUSD | 1h | 51 | 74.51 | 2.023 | 41.17 | -10.62 | 24 | 6 | 0.25 | spike |
| 904 | `bb_trend_rejoin` | DOGEUSD | 1h | 65 | 63.08 | 1.872 | 33.69 | -9.04 | 24 | 6 | 0.25 | spike |
| 905 | `bb_trend_rejoin` | DOGEUSD | 1h | 63 | 68.25 | 1.772 | 32.64 | -12.19 | 24 | 6 | 0.25 | spike |
| 906 | `bb_trend_rejoin` | ETHUSD | 1h | 107 | 69.16 | 1.384 | 28.69 | -7.72 | 24 | 6 | 0.25 | spike |
| 907 | `bb_trend_rejoin` | ETHUSD | 1h | 142 | 66.9 | 1.262 | 25.66 | -11.89 | 24 | 6 | 0.25 | spike |
| 908 | `bb_trend_rejoin` | ETHUSD | 1h | 142 | 66.9 | 1.262 | 25.66 | -11.89 | 24 | 6 | 0.25 | spike |
| 909 | `bb_trend_rejoin` | ETHUSD | 1h | 142 | 66.9 | 1.262 | 25.66 | -11.89 | 24 | 6 | 0.25 | spike |
| 910 | `bb_trend_rejoin` | ETHUSD | 1h | 130 | 63.08 | 1.241 | 24.15 | -15.22 | 24 | 6 | 0.25 | spike |
| 911 | `bb_trend_rejoin` | ETHUSD | 1h | 130 | 63.08 | 1.236 | 23.6 | -14.79 | 24 | 6 | 0.25 | spike |
| 912 | `bb_trend_rejoin` | ETHUSD | 1h | 130 | 63.08 | 1.236 | 23.6 | -14.79 | 24 | 6 | 0.25 | spike |
| 913 | `bb_trend_rejoin` | ETHUSD | 1h | 130 | 63.08 | 1.236 | 23.6 | -14.79 | 24 | 6 | 0.25 | spike |
| 914 | `rsi2_regime` | ETHUSD | 1h | 57 | 68.42 | 1.328 | 22.03 | -14.65 | 24 | 6 | 0.25 | spike |
| 915 | `bb_trend_rejoin` | ETHUSD | 1h | 78 | 73.08 | 1.339 | 12.35 | -7.07 | 24 | 6 | 0.25 | spike |
| 916 | `rsi2_regime` | ETHUSD | 1h | 57 | 66.67 | 1.124 | 5.3 | -11.79 | 24 | 6 | 0.25 | spike |
| 917 | `rsi2_regime` | ETHUSD | 1h | 52 | 75.0 | 1.938 | 40.05 | -9.04 | 21 | 5 | 0.24 | spike |
| 918 | `rsi2_regime` | ETHUSD | 1h | 104 | 62.5 | 1.533 | 61.1 | -12.7 | 23 | 5 | 0.22 | spike |
| 919 | `bb_trend_rejoin` | DOGEUSD | 1h | 66 | 68.18 | 1.719 | 35.07 | -14.04 | 23 | 5 | 0.22 | spike |
| 920 | `bb_trend_rejoin` | ETHUSD | 1h | 148 | 64.19 | 1.303 | 33.76 | -9.74 | 23 | 5 | 0.22 | spike |
| 921 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 67.19 | 1.629 | 27.38 | -15.28 | 23 | 5 | 0.22 | spike |
| 922 | `bb_trend_rejoin` | DOGEUSD | 1h | 66 | 68.18 | 1.577 | 26.68 | -14.04 | 23 | 5 | 0.22 | spike |
| 923 | `bb_trend_rejoin` | DOGEUSD | 1h | 66 | 68.18 | 1.577 | 26.68 | -14.04 | 23 | 5 | 0.22 | spike |
| 924 | `bb_trend_rejoin` | ETHUSD | 1h | 130 | 61.54 | 1.251 | 24.9 | -13.96 | 23 | 5 | 0.22 | spike |
| 925 | `bb_trend_rejoin` | ETHUSD | 1h | 130 | 63.08 | 1.24 | 24.01 | -14.84 | 23 | 5 | 0.22 | spike |
| 926 | `bb_trend_rejoin` | ETHUSD | 1h | 130 | 63.08 | 1.237 | 23.61 | -14.84 | 23 | 5 | 0.22 | spike |
| 927 | `bb_trend_rejoin` | ETHUSD | 1h | 130 | 63.08 | 1.235 | 23.51 | -15.66 | 23 | 5 | 0.22 | spike |
| 928 | `bb_trend_rejoin` | DOGEUSD | 1h | 66 | 68.18 | 1.393 | 17.47 | -14.04 | 23 | 5 | 0.22 | spike |
| 929 | `bb_trend_rejoin` | DOGEUSD | 1h | 66 | 68.18 | 1.393 | 17.47 | -14.04 | 23 | 5 | 0.22 | spike |
| 930 | `bb_trend_rejoin` | DOGEUSD | 1h | 66 | 68.18 | 1.393 | 17.47 | -14.04 | 23 | 5 | 0.22 | spike |
| 931 | `rsi2_regime` | ETHUSD | 1h | 57 | 66.67 | 1.125 | 5.36 | -11.79 | 23 | 5 | 0.22 | spike |
| 932 | `rsi2_regime` | ETHUSD | 1h | 50 | 66.0 | 1.203 | 3.42 | -7.08 | 23 | 5 | 0.22 | spike |
| 933 | `rsi2_regime` | ETHUSD | 1h | 80 | 67.5 | 1.086 | 3.28 | -15.58 | 23 | 5 | 0.22 | spike |
| 934 | `zscore_revert` | ETHUSD | 1h | 66 | 68.18 | 1.568 | 56.2 | -22.34 | 19 | 4 | 0.21 | spike |
| 935 | `zscore_revert` | ETHUSD | 1h | 66 | 68.18 | 1.568 | 56.2 | -22.34 | 19 | 4 | 0.21 | spike |
| 936 | `zscore_revert` | ETHUSD | 1h | 66 | 68.18 | 1.568 | 56.2 | -22.34 | 19 | 4 | 0.21 | spike |
| 937 | `zscore_revert` | ETHUSD | 1h | 66 | 68.18 | 1.513 | 49.19 | -22.34 | 19 | 4 | 0.21 | spike |
| 938 | `zscore_revert` | ETHUSD | 1h | 66 | 68.18 | 1.513 | 49.19 | -22.34 | 19 | 4 | 0.21 | spike |
| 939 | `zscore_revert` | ETHUSD | 1h | 66 | 68.18 | 1.513 | 49.19 | -22.34 | 19 | 4 | 0.21 | spike |
| 940 | `rsi2_regime` | ETHUSD | 1h | 86 | 67.44 | 1.183 | 9.57 | -18.63 | 19 | 4 | 0.21 | spike |
| 941 | `rsi2_regime` | ETHUSD | 1h | 186 | 66.67 | 1.052 | 2.84 | -22.12 | 19 | 4 | 0.21 | spike |
| 942 | `bb_trend_rejoin` | ETHUSD | 1h | 66 | 69.7 | 1.689 | 61.76 | -16.57 | 24 | 5 | 0.21 | spike |
| 943 | `bb_trend_rejoin` | ETHUSD | 1h | 67 | 67.16 | 1.573 | 54.66 | -21.42 | 24 | 5 | 0.21 | spike |
| 944 | `bb_trend_rejoin` | ETHUSD | 1h | 67 | 67.16 | 1.483 | 43.72 | -21.42 | 24 | 5 | 0.21 | spike |
| 945 | `bb_trend_rejoin` | ETHUSD | 1h | 143 | 67.13 | 1.345 | 37.93 | -12.77 | 24 | 5 | 0.21 | spike |
| 946 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 68.75 | 1.652 | 29.01 | -13.09 | 24 | 5 | 0.21 | spike |
| 947 | `rsi2_regime` | DOGEUSD | 1h | 53 | 73.58 | 1.47 | 24.15 | -19.39 | 24 | 5 | 0.21 | spike |
| 948 | `rsi2_regime` | ETHUSD | 1h | 60 | 68.33 | 1.337 | 23.98 | -14.65 | 24 | 5 | 0.21 | spike |
| 949 | `bb_trend_rejoin` | DOGEUSD | 1h | 64 | 68.75 | 1.54 | 23.59 | -13.09 | 24 | 5 | 0.21 | spike |
| 950 | `rsi2_regime` | ETHUSD | 1h | 57 | 70.18 | 1.381 | 20.29 | -13.39 | 24 | 5 | 0.21 | spike |
| 951 | `rsi2_regime` | ETHUSD | 1h | 85 | 67.06 | 1.268 | 18.78 | -11.27 | 24 | 5 | 0.21 | spike |
| 952 | `bb_trend_rejoin` | DOGEUSD | 1h | 65 | 64.62 | 1.345 | 14.91 | -10.74 | 24 | 5 | 0.21 | spike |
| 953 | `bb_trend_rejoin` | DOGEUSD | 1h | 70 | 62.86 | 1.292 | 14.89 | -15.88 | 24 | 5 | 0.21 | spike |
| 954 | `bb_trend_rejoin` | DOGEUSD | 1h | 65 | 64.62 | 1.331 | 14.34 | -11.17 | 24 | 5 | 0.21 | spike |
| 955 | `bb_trend_rejoin` | DOGEUSD | 1h | 65 | 64.62 | 1.309 | 13.44 | -10.25 | 24 | 5 | 0.21 | spike |
| 956 | `bb_trend_rejoin` | DOGEUSD | 1h | 65 | 64.62 | 1.309 | 13.44 | -10.25 | 24 | 5 | 0.21 | spike |
| 957 | `bb_trend_rejoin` | ETHUSD | 1h | 96 | 67.71 | 1.134 | 11.2 | -28.85 | 24 | 5 | 0.21 | spike |
| 958 | `bb_trend_rejoin` | ETHUSD | 1h | 96 | 67.71 | 1.134 | 11.2 | -28.85 | 24 | 5 | 0.21 | spike |
| 959 | `rsi2_regime` | ETHUSD | 1h | 67 | 65.67 | 1.211 | 8.65 | -7.98 | 24 | 5 | 0.21 | spike |
| 960 | `bb_trend_rejoin` | ETHUSD | 1h | 110 | 60.91 | 1.11 | 7.38 | -10.85 | 24 | 5 | 0.21 | spike |
| 961 | `rsi2_regime` | ETHUSD | 1h | 72 | 65.28 | 1.086 | 3.52 | -16.36 | 24 | 5 | 0.21 | spike |
| 962 | `rsi2_regime` | ETHUSD | 1h | 50 | 66.0 | 1.203 | 3.42 | -7.08 | 24 | 5 | 0.21 | spike |
| 963 | `rsi2_regime` | ETHUSD | 1h | 50 | 66.0 | 1.129 | 2.13 | -8.24 | 24 | 5 | 0.21 | spike |
| 964 | `rsi2_regime` | ETHUSD | 1h | 50 | 66.0 | 1.087 | 1.32 | -8.24 | 24 | 5 | 0.21 | spike |
| 965 | `zscore_revert` | ETHUSD | 1h | 67 | 67.16 | 1.529 | 51.05 | -22.34 | 21 | 4 | 0.19 | spike |
| 966 | `zscore_revert` | ETHUSD | 1h | 67 | 67.16 | 1.529 | 51.05 | -22.34 | 21 | 4 | 0.19 | spike |
| 967 | `zscore_revert` | ETHUSD | 1h | 67 | 67.16 | 1.529 | 51.05 | -22.34 | 21 | 4 | 0.19 | spike |
| 968 | `rsi2_regime` | ETHUSD | 1h | 80 | 65.0 | 1.185 | 11.1 | -16.79 | 21 | 4 | 0.19 | spike |
| 969 | `rsi2_regime` | ETHUSD | 1h | 53 | 71.7 | 1.314 | 9.81 | -13.5 | 21 | 4 | 0.19 | spike |
| 970 | `rsi2_regime` | ETHUSD | 1h | 90 | 65.56 | 1.151 | 8.23 | -23.4 | 21 | 4 | 0.19 | spike |
| 971 | `rsi2_regime` | ETHUSD | 1h | 88 | 61.36 | 1.152 | 6.47 | -15.39 | 21 | 4 | 0.19 | spike |
| 972 | `rsi2_regime` | ETHUSD | 1h | 89 | 62.92 | 1.098 | 5.3 | -20.91 | 21 | 4 | 0.19 | spike |
| 973 | `rsi2_regime` | DOGEUSD | 1h | 55 | 63.64 | 1.101 | 4.22 | -19.77 | 21 | 4 | 0.19 | spike |
| 974 | `zscore_revert` | ETHUSD | 1h | 137 | 62.77 | 1.484 | 56.84 | -11.09 | 16 | 3 | 0.19 | spike |
| 975 | `zscore_revert` | ETHUSD | 1h | 137 | 63.5 | 1.447 | 51.88 | -11.09 | 16 | 3 | 0.19 | spike |
| 976 | `zscore_revert` | ETHUSD | 1h | 137 | 63.5 | 1.447 | 51.88 | -11.09 | 16 | 3 | 0.19 | spike |
| 977 | `zscore_revert` | ETHUSD | 1h | 137 | 63.5 | 1.447 | 51.88 | -11.09 | 16 | 3 | 0.19 | spike |
| 978 | `zscore_revert` | ETHUSD | 1h | 137 | 63.5 | 1.447 | 51.88 | -11.09 | 16 | 3 | 0.19 | spike |
| 979 | `zscore_revert` | ETHUSD | 1h | 137 | 63.5 | 1.447 | 51.88 | -11.09 | 16 | 3 | 0.19 | spike |
| 980 | `zscore_revert` | ETHUSD | 1h | 137 | 63.5 | 1.447 | 51.88 | -11.09 | 16 | 3 | 0.19 | spike |
| 981 | `zscore_revert` | ETHUSD | 1h | 137 | 62.77 | 1.446 | 50.96 | -13.35 | 16 | 3 | 0.19 | spike |
| 982 | `zscore_revert` | ETHUSD | 1h | 137 | 62.77 | 1.441 | 50.55 | -11.09 | 16 | 3 | 0.19 | spike |
| 983 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.304 | 34.36 | -19.33 | 16 | 3 | 0.19 | spike |
| 984 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.304 | 34.36 | -19.33 | 16 | 3 | 0.19 | spike |
| 985 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.304 | 34.36 | -19.33 | 16 | 3 | 0.19 | spike |
| 986 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.304 | 34.36 | -19.33 | 16 | 3 | 0.19 | spike |
| 987 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.262 | 28.64 | -19.33 | 16 | 3 | 0.19 | spike |
| 988 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.259 | 28.16 | -19.33 | 16 | 3 | 0.19 | spike |
| 989 | `zscore_revert` | ETHUSD | 1h | 131 | 66.41 | 1.259 | 28.16 | -19.33 | 16 | 3 | 0.19 | spike |
| 990 | `zscore_revert` | ETHUSD | 1h | 132 | 64.39 | 1.261 | 27.64 | -21.2 | 16 | 3 | 0.19 | spike |
| 991 | `zscore_revert` | ETHUSD | 1h | 132 | 64.39 | 1.261 | 27.64 | -21.2 | 16 | 3 | 0.19 | spike |
| 992 | `bb_trend_rejoin` | DOGEUSD | 1h | 70 | 62.86 | 1.467 | 25.72 | -15.88 | 22 | 4 | 0.18 | spike |
| 993 | `rsi2_regime` | ETHUSD | 1h | 85 | 67.06 | 1.34 | 25.03 | -11.27 | 22 | 4 | 0.18 | spike |
| 994 | `rsi2_regime` | ADAUSD | 1h | 55 | 69.09 | 1.328 | 13.03 | -9.24 | 22 | 4 | 0.18 | spike |
| 995 | `bb_trend_rejoin` | ADAUSD | 1h | 58 | 67.24 | 1.221 | 6.98 | -13.18 | 22 | 4 | 0.18 | spike |
| 996 | `bb_trend_rejoin` | ETHUSD | 1h | 92 | 63.04 | 1.447 | 29.97 | -10.23 | 23 | 4 | 0.17 | spike |
| 997 | `bb_trend_rejoin` | ETHUSD | 1h | 144 | 66.67 | 1.278 | 27.76 | -13.05 | 23 | 4 | 0.17 | spike |
| 998 | `bb_trend_rejoin` | ETHUSD | 1h | 220 | 64.09 | 1.174 | 27.24 | -18.05 | 23 | 4 | 0.17 | spike |
| 999 | `bb_trend_rejoin` | DOGEUSD | 1h | 66 | 66.67 | 1.575 | 26.89 | -14.04 | 23 | 4 | 0.17 | spike |
| 1000 | `bb_trend_rejoin` | ETHUSD | 1h | 92 | 61.96 | 1.404 | 26.13 | -10.74 | 23 | 4 | 0.17 | spike |
| 1001 | `bb_trend_rejoin` | ETHUSD | 1h | 92 | 61.96 | 1.404 | 26.13 | -10.74 | 23 | 4 | 0.17 | spike |
| 1002 | `bb_trend_rejoin` | ETHUSD | 1h | 92 | 60.87 | 1.355 | 23.29 | -10.74 | 23 | 4 | 0.17 | spike |
| 1003 | `rsi2_regime` | ETHUSD | 1h | 53 | 66.04 | 1.584 | 22.87 | -11.8 | 23 | 4 | 0.17 | spike |
| 1004 | `bb_trend_rejoin` | ETHUSD | 1h | 142 | 66.2 | 1.233 | 22.69 | -11.21 | 23 | 4 | 0.17 | spike |
| 1005 | `rsi2_regime` | ETHUSD | 1h | 57 | 68.42 | 1.396 | 21.81 | -12.35 | 23 | 4 | 0.17 | spike |
| 1006 | `bb_trend_rejoin` | ETHUSD | 1h | 92 | 63.04 | 1.34 | 21.76 | -10.23 | 23 | 4 | 0.17 | spike |
| 1007 | `rsi2_regime` | ETHUSD | 1h | 52 | 71.15 | 1.364 | 16.74 | -21.78 | 23 | 4 | 0.17 | spike |
| 1008 | `bb_trend_rejoin` | DOGEUSD | 1h | 70 | 62.86 | 1.207 | 10.05 | -15.88 | 23 | 4 | 0.17 | spike |
| 1009 | `bb_trend_rejoin` | ETHUSD | 1h | 110 | 60.91 | 1.094 | 5.93 | -12.23 | 23 | 4 | 0.17 | spike |
| 1010 | `bb_trend_rejoin` | ETHUSD | 1h | 110 | 60.91 | 1.094 | 5.93 | -12.23 | 23 | 4 | 0.17 | spike |
| 1011 | `rsi2_regime` | ETHUSD | 1h | 50 | 66.0 | 1.188 | 3.21 | -6.62 | 23 | 4 | 0.17 | spike |
| 1012 | `rsi2_regime` | ETHUSD | 1h | 50 | 66.0 | 1.188 | 3.21 | -6.62 | 23 | 4 | 0.17 | spike |
| 1013 | `rsi2_regime` | ETHUSD | 1h | 50 | 66.0 | 1.188 | 3.21 | -6.62 | 23 | 4 | 0.17 | spike |
| 1014 | `bb_trend_rejoin` | ETHUSD | 1h | 54 | 70.37 | 1.657 | 54.06 | -20.23 | 24 | 4 | 0.17 | spike |
| 1015 | `bb_trend_rejoin` | ETHUSD | 1h | 54 | 70.37 | 1.579 | 45.92 | -20.23 | 24 | 4 | 0.17 | spike |
| 1016 | `rsi2_regime` | DOGEUSD | 1h | 68 | 69.12 | 1.623 | 44.46 | -19.97 | 24 | 4 | 0.17 | spike |
| 1017 | `rsi2_regime` | DOGEUSD | 1h | 65 | 63.08 | 1.478 | 37.59 | -18.75 | 24 | 4 | 0.17 | spike |
| 1018 | `rsi2_regime` | ETHUSD | 1h | 51 | 72.55 | 1.639 | 29.87 | -10.6 | 24 | 4 | 0.17 | spike |
| 1019 | `rsi2_regime` | ETHUSD | 1h | 90 | 71.11 | 1.4 | 28.61 | -21.17 | 24 | 4 | 0.17 | spike |
| 1020 | `rsi2_regime` | ETHUSD | 1h | 69 | 68.12 | 1.371 | 24.85 | -16.99 | 24 | 4 | 0.17 | spike |
| 1021 | `rsi2_regime` | ETHUSD | 1h | 50 | 74.0 | 1.441 | 16.37 | -13.55 | 24 | 4 | 0.17 | spike |
| 1022 | `rsi2_regime` | ETHUSD | 1h | 54 | 64.81 | 1.463 | 7.7 | -5.89 | 24 | 4 | 0.17 | spike |
| 1023 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 63.08 | 1.07 | 6.78 | -21.56 | 24 | 4 | 0.17 | spike |
| 1024 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 63.08 | 1.067 | 6.3 | -22.64 | 24 | 4 | 0.17 | spike |
| 1025 | `rsi2_regime` | ETHUSD | 1h | 53 | 64.15 | 1.268 | 4.75 | -7.45 | 24 | 4 | 0.17 | spike |
| 1026 | `bb_trend_rejoin` | ETHUSD | 1h | 214 | 63.08 | 1.053 | 4.15 | -24.35 | 24 | 4 | 0.17 | spike |
| 1027 | `rsi2_regime` | ETHUSD | 1h | 114 | 69.3 | 1.311 | 22.6 | -9.7 | 19 | 3 | 0.16 | spike |
| 1028 | `rsi2_regime` | DOGEUSD | 1h | 86 | 63.95 | 1.178 | 11.16 | -19.48 | 21 | 3 | 0.14 | spike |
| 1029 | `bb_trend_rejoin` | ADAUSD | 1h | 58 | 67.24 | 1.221 | 6.98 | -13.18 | 22 | 3 | 0.14 | spike |
| 1030 | `rsi2_regime` | ETHUSD | 1h | 53 | 64.15 | 1.506 | 22.24 | -14.7 | 23 | 3 | 0.13 | spike |
| 1031 | `rsi2_regime` | ETHUSD | 1h | 50 | 76.0 | 1.614 | 18.26 | -11.9 | 23 | 3 | 0.13 | spike |
| 1032 | `rsi2_regime` | DOGEUSD | 1h | 54 | 68.52 | 1.25 | 13.67 | -21.26 | 23 | 3 | 0.13 | spike |
| 1033 | `bb_trend_rejoin` | DOGEUSD | 1h | 70 | 62.86 | 1.261 | 12.96 | -15.88 | 23 | 3 | 0.13 | spike |
| 1034 | `bb_trend_rejoin` | DOGEUSD | 1h | 70 | 62.86 | 1.229 | 11.15 | -15.88 | 23 | 3 | 0.13 | spike |
| 1035 | `rsi2_regime` | ETHUSD | 1h | 54 | 64.81 | 1.355 | 6.26 | -5.11 | 23 | 3 | 0.13 | spike |
| 1036 | `rsi2_regime` | ETHUSD | 1h | 54 | 66.67 | 1.226 | 5.16 | -7.14 | 23 | 3 | 0.13 | spike |
| 1037 | `bb_trend_rejoin` | ADAUSD | 1h | 58 | 67.24 | 1.09 | 2.42 | -17.67 | 23 | 3 | 0.13 | spike |
| 1038 | `rsi2_regime` | ETHUSD | 1h | 79 | 62.03 | 1.066 | 1.99 | -9.07 | 23 | 3 | 0.13 | spike |
| 1039 | `bb_trend_rejoin` | ETHUSD | 1h | 54 | 70.37 | 1.577 | 45.78 | -20.23 | 24 | 3 | 0.12 | spike |
| 1040 | `bb_trend_rejoin` | ETHUSD | 1h | 139 | 70.5 | 1.399 | 41.94 | -10.93 | 24 | 3 | 0.12 | spike |
| 1041 | `rsi2_regime` | ETHUSD | 1h | 52 | 69.23 | 2.011 | 25.95 | -7.62 | 24 | 3 | 0.12 | spike |
| 1042 | `rsi2_regime` | ETHUSD | 1h | 52 | 69.23 | 1.969 | 25.3 | -8.1 | 24 | 3 | 0.12 | spike |
| 1043 | `rsi2_regime` | ETHUSD | 1h | 52 | 67.31 | 1.615 | 24.69 | -10.62 | 24 | 3 | 0.12 | spike |
| 1044 | `rsi2_regime` | ETHUSD | 1h | 52 | 67.31 | 1.875 | 23.92 | -11.19 | 24 | 3 | 0.12 | spike |
| 1045 | `rsi2_regime` | ETHUSD | 1h | 52 | 67.31 | 1.505 | 21.08 | -12.55 | 24 | 3 | 0.12 | spike |
| 1046 | `rsi2_regime` | ETHUSD | 1h | 52 | 71.15 | 1.588 | 13.59 | -10.46 | 24 | 3 | 0.12 | spike |
| 1047 | `bb_trend_rejoin` | DOGEUSD | 1h | 68 | 64.71 | 1.218 | 10.03 | -14.46 | 24 | 3 | 0.12 | spike |
| 1048 | `rsi2_regime` | ETHUSD | 1h | 165 | 63.64 | 1.126 | 9.67 | -11.19 | 24 | 3 | 0.12 | spike |
| 1049 | `bb_trend_rejoin` | ETHUSD | 1h | 110 | 60.91 | 1.094 | 5.93 | -12.23 | 24 | 3 | 0.12 | spike |
| 1050 | `rsi2_regime` | ETHUSD | 1h | 53 | 64.15 | 1.14 | 2.5 | -7.45 | 24 | 3 | 0.12 | spike |
| 1051 | `zscore_revert` | ETHUSD | 1h | 67 | 65.67 | 1.462 | 42.56 | -22.34 | 19 | 2 | 0.11 | spike |
| 1052 | `zscore_revert` | ETHUSD | 1h | 67 | 65.67 | 1.462 | 42.56 | -22.34 | 19 | 2 | 0.11 | spike |
| 1053 | `zscore_revert` | ETHUSD | 1h | 67 | 65.67 | 1.41 | 36.62 | -22.34 | 19 | 2 | 0.11 | spike |
| 1054 | `zscore_revert` | ETHUSD | 1h | 67 | 65.67 | 1.41 | 36.62 | -22.34 | 19 | 2 | 0.11 | spike |
| 1055 | `rsi2_regime` | DOGEUSD | 1h | 57 | 68.42 | 1.511 | 27.52 | -12.9 | 21 | 2 | 0.10 | spike |
| 1056 | `rsi2_regime` | ETHUSD | 1h | 53 | 64.15 | 1.45 | 22.88 | -16.55 | 21 | 2 | 0.10 | spike |
| 1057 | `rsi2_regime` | ETHUSD | 1h | 53 | 64.15 | 1.329 | 17.05 | -20.51 | 21 | 2 | 0.10 | spike |
| 1058 | `rsi2_regime` | ETHUSD | 1h | 82 | 62.2 | 1.182 | 11.45 | -15.67 | 21 | 2 | 0.10 | spike |
| 1059 | `rsi2_regime` | ETHUSD | 1h | 60 | 65.0 | 1.555 | 41.03 | -15.69 | 22 | 2 | 0.09 | spike |
| 1060 | `rsi2_regime` | ETHUSD | 1h | 88 | 61.36 | 1.265 | 11.43 | -12.39 | 22 | 2 | 0.09 | spike |
| 1061 | `rsi2_regime` | ETHUSD | 1h | 53 | 66.04 | 1.924 | 32.77 | -8.21 | 23 | 2 | 0.09 | spike |
| 1062 | `bb_trend_rejoin` | ETHUSD | 1h | 110 | 60.91 | 1.094 | 5.93 | -12.23 | 23 | 2 | 0.09 | spike |
| 1063 | `rsi2_regime` | DOGEUSD | 1h | 58 | 65.52 | 1.098 | 4.01 | -22.42 | 23 | 2 | 0.09 | spike |
| 1064 | `rsi2_regime` | ETHUSD | 1h | 52 | 71.15 | 1.747 | 34.31 | -9.28 | 24 | 2 | 0.08 | spike |
| 1065 | `bb_trend_rejoin` | ETHUSD | 1h | 54 | 66.67 | 1.382 | 28.44 | -27.1 | 24 | 2 | 0.08 | spike |
| 1066 | `bb_trend_rejoin` | ETHUSD | 1h | 54 | 66.67 | 1.372 | 27.49 | -27.1 | 24 | 2 | 0.08 | spike |
| 1067 | `rsi2_regime` | ETHUSD | 1h | 94 | 70.21 | 1.281 | 16.62 | -15.28 | 24 | 2 | 0.08 | spike |
| 1068 | `rsi2_regime` | ETHUSD | 1h | 65 | 69.23 | 1.292 | 15.45 | -14.01 | 24 | 2 | 0.08 | spike |
| 1069 | `rsi2_regime` | ETHUSD | 1h | 51 | 70.59 | 1.55 | 12.62 | -11.04 | 24 | 2 | 0.08 | spike |
| 1070 | `rsi2_regime` | ETHUSD | 1h | 52 | 65.38 | 1.408 | 12.07 | -14.76 | 24 | 2 | 0.08 | spike |
| 1071 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 63.64 | 1.141 | 7.89 | -13.92 | 24 | 2 | 0.08 | spike |
| 1072 | `bb_trend_rejoin` | ETHUSD | 1h | 88 | 63.64 | 1.12 | 6.42 | -14.27 | 24 | 2 | 0.08 | spike |
| 1073 | `bb_trend_rejoin` | ADAUSD | 1h | 58 | 67.24 | 1.064 | 1.53 | -15.49 | 24 | 2 | 0.08 | spike |
| 1074 | `bb_trend_rejoin` | ADAUSD | 1h | 58 | 67.24 | 1.064 | 1.53 | -15.49 | 24 | 2 | 0.08 | spike |
| 1075 | `bb_trend_rejoin` | ADAUSD | 1h | 58 | 67.24 | 1.064 | 1.53 | -15.49 | 24 | 2 | 0.08 | spike |
| 1076 | `rsi2_regime` | ETHUSD | 1h | 50 | 74.0 | 1.734 | 29.89 | -10.81 | 21 | 1 | 0.05 | spike |
| 1077 | `rsi2_regime` | DOGEUSD | 1h | 83 | 61.45 | 1.364 | 24.27 | -9.44 | 23 | 1 | 0.04 | spike |
| 1078 | `rsi2_regime` | ETHUSD | 1h | 72 | 68.06 | 1.271 | 16.04 | -18.16 | 23 | 1 | 0.04 | spike |
| 1079 | `rsi2_regime` | DOGEUSD | 1h | 54 | 61.11 | 1.297 | 15.92 | -14.44 | 23 | 1 | 0.04 | spike |
| 1080 | `rsi2_regime` | ETHUSD | 1h | 52 | 69.23 | 1.578 | 14.5 | -12.07 | 23 | 1 | 0.04 | spike |
| 1081 | `rsi2_regime` | DOGEUSD | 1h | 78 | 64.1 | 1.206 | 12.67 | -16.91 | 23 | 1 | 0.04 | spike |
| 1082 | `rsi2_regime` | ETHUSD | 1h | 72 | 68.06 | 1.174 | 9.73 | -18.25 | 23 | 1 | 0.04 | spike |
| 1083 | `rsi2_regime` | ETHUSD | 1h | 54 | 62.96 | 1.067 | 1.62 | -12.85 | 23 | 1 | 0.04 | spike |
| 1084 | `rsi2_regime` | ETHUSD | 1h | 52 | 63.46 | 1.92 | 30.9 | -4.71 | 24 | 0 | 0.00 | spike |
| 1085 | `rsi2_regime` | ETHUSD | 1h | 53 | 62.26 | 1.765 | 28.02 | -7.99 | 23 | 0 | 0.00 | spike |
| 1086 | `rsi2_regime` | ETHUSD | 1h | 50 | 64.0 | 1.4 | 10.12 | -9.21 | 19 | 0 | 0.00 | spike |
| 1087 | `bb_trend_rejoin` | ETHUSD | 1h | 51 | 62.75 | 1.171 | 9.34 | -24.75 | 24 | 0 | 0.00 | spike |
| 1088 | `bb_trend_rejoin` | ETHUSD | 1h | 51 | 62.75 | 1.171 | 9.34 | -24.75 | 24 | 0 | 0.00 | spike |
| 1089 | `bb_trend_rejoin` | ETHUSD | 1h | 51 | 62.75 | 1.171 | 9.34 | -24.75 | 24 | 0 | 0.00 | spike |
| 1090 | `rsi2_regime` | ETHUSD | 1h | 50 | 66.0 | 1.258 | 7.31 | -13.13 | 24 | 0 | 0.00 | spike |
| 1091 | `rsi2_regime` | ETHUSD | 1h | 50 | 66.0 | 1.229 | 6.42 | -12.23 | 24 | 0 | 0.00 | spike |
| 1092 | `rsi2_regime` | ETHUSD | 1h | 50 | 66.0 | 1.185 | 5.2 | -14.78 | 24 | 0 | 0.00 | spike |
| 1093 | `rsi2_regime` | ETHUSD | 1h | 50 | 66.0 | 1.165 | 4.43 | -13.05 | 24 | 0 | 0.00 | spike |
| 1094 | `rsi2_regime` | ETHUSD | 1h | 50 | 64.0 | 1.11 | 2.17 | -6.29 | 22 | 0 | 0.00 | spike |
| 1095 | `rsi2_regime` | ETHUSD | 1h | 50 | 64.0 | 1.11 | 2.17 | -6.29 | 23 | 0 | 0.00 | spike |

## Survivors (plateau)

- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":20,"sma_trend":100,"slope_bars":50,"exit_sma":21}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":5.0,"timeout":168}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":20,"sma_trend":100,"slope_bars":100,"exit_sma":21}` | exit=`{"sl_atr":5.0,"tp_atr":10.0,"timeout":48}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":100,"slope_bars":65,"exit_sma":13}` | exit=`{"sl_atr":4.0,"tp_atr":7.0,"timeout":120}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":18,"sma_trend":75,"slope_bars":65,"exit_sma":21}` | exit=`{"sl_atr":5.0,"tp_atr":7.0,"timeout":168}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":8,"sma_trend":125,"slope_bars":40,"exit_sma":21}` | exit=`{"sl_atr":5.0,"tp_atr":7.0,"timeout":168}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":100,"slope_bars":40,"exit_sma":21}` | exit=`{"sl_atr":5.0,"tp_atr":6.0,"trail_atr":5.0}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":100,"slope_bars":65,"exit_sma":21}` | exit=`{"sl_atr":5.0,"tp_atr":7.0,"trail_atr":5.0,"timeout":72}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":100,"slope_bars":65,"exit_sma":13}` | exit=`{"sl_atr":4.0,"tp_atr":5.0,"timeout":24}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":15,"sma_trend":75,"slope_bars":80,"exit_sma":21}` | exit=`{"sl_atr":5.0,"tp_atr":10.0,"timeout":72}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":5,"sma_trend":100,"slope_bars":80,"exit_sma":13}` | exit=`{"sl_atr":3.5,"trail_atr":5.0,"timeout":96}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":100,"slope_bars":65,"exit_sma":21}` | exit=`{"sl_atr":3.0,"tp_atr":7.0,"trail_atr":5.0,"timeout":48}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":100,"slope_bars":65,"exit_sma":13}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"timeout":36}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":125,"slope_bars":5,"exit_sma":13}` | exit=`{"sl_atr":5.0,"tp_atr":6.0}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":75,"slope_bars":40,"exit_sma":21}` | exit=`{"sl_atr":5.0,"trail_atr":5.0}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":75,"slope_bars":10,"exit_sma":21}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":5.0,"timeout":36}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":75,"slope_bars":20,"exit_sma":13}` | exit=`{"sl_atr":4.0,"timeout":96}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":75,"slope_bars":30,"exit_sma":21}` | exit=`{"sl_atr":3.5,"timeout":120}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":5,"sma_trend":100,"slope_bars":80,"exit_sma":8}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":5.0,"timeout":24}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":75,"slope_bars":10,"exit_sma":21}` | exit=`{"sl_atr":5.0,"tp_atr":7.0,"timeout":48}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":5,"sma_trend":75,"slope_bars":30,"exit_sma":21}` | exit=`{"sl_atr":5.0,"tp_atr":7.0}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":100,"slope_bars":5,"exit_sma":8}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":5.0}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":75,"slope_bars":40,"exit_sma":13}` | exit=`{"sl_atr":5.0,"tp_atr":7.0,"trail_atr":5.0,"timeout":96}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":125,"slope_bars":80,"exit_sma":13}` | exit=`{"sl_atr":5.0,"tp_atr":7.0,"timeout":48}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":125,"slope_bars":40,"exit_sma":21}` | exit=`{"sl_atr":5.0,"tp_atr":6.0}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":150,"slope_bars":80,"exit_sma":13}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"timeout":18}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":100,"slope_bars":5,"exit_sma":13}` | exit=`{"sl_atr":3.5,"trail_atr":5.0}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":100,"slope_bars":100,"exit_sma":13}` | exit=`{"sl_atr":4.0,"tp_atr":6.0,"timeout":168}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":75,"slope_bars":50,"exit_sma":8}` | exit=`{"sl_atr":5.0,"tp_atr":6.0,"timeout":24}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":100,"slope_bars":5,"exit_sma":13}` | exit=`{"sl_atr":4.0,"trail_atr":5.0,"timeout":96}` | frac=1.00
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":10,"sma_trend":100,"slope_bars":100,"exit_sma":21}` | exit=`{"sl_atr":3.0,"tp_atr":6.0,"timeout":36}` | frac=0.96
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":50,"exit_sma":8}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":5.0,"timeout":168}` | frac=0.96
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":75,"slope_bars":50,"exit_sma":21}` | exit=`{"sl_atr":4.0,"tp_atr":7.0,"trail_atr":5.0,"timeout":36}` | frac=0.95
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":10,"sma_trend":75,"slope_bars":50,"exit_sma":8}` | exit=`{"sl_atr":2.0,"tp_atr":2.5,"trail_atr":5.0,"timeout":36}` | frac=0.95
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":75,"slope_bars":40,"exit_sma":21}` | exit=`{"sl_atr":4.0,"timeout":24}` | frac=0.95
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":125,"slope_bars":65,"exit_sma":21}` | exit=`{"sl_atr":3.0,"trail_atr":5.0,"timeout":48}` | frac=0.92
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":5,"sma_trend":100,"slope_bars":40,"exit_sma":21}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":5.0,"timeout":24}` | frac=0.91
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":5,"sma_trend":100,"slope_bars":40,"exit_sma":21}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":5.0,"timeout":96}` | frac=0.91
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":75,"slope_bars":50,"exit_sma":21}` | exit=`{"sl_atr":3.5,"tp_atr":10.0,"trail_atr":5.0,"timeout":24}` | frac=0.91
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":100,"slope_bars":40,"exit_sma":21}` | exit=`{"sl_atr":3.0,"tp_atr":7.0,"timeout":96}` | frac=0.91
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":20,"sma_trend":100,"slope_bars":80,"exit_sma":13}` | exit=`{"sl_atr":3.5,"timeout":36}` | frac=0.91
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":5,"exit_sma":21}` | exit=`{"sl_atr":3.0,"tp_atr":10.0,"timeout":168}` | frac=0.91
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":10,"sma_trend":75,"slope_bars":50,"exit_sma":21}` | exit=`{"sl_atr":3.5,"tp_atr":6.0,"trail_atr":5.0,"timeout":168}` | frac=0.91
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":125,"slope_bars":50,"exit_sma":21}` | exit=`{"sl_atr":4.0,"tp_atr":7.0,"timeout":36}` | frac=0.91
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":18,"sma_trend":100,"slope_bars":10,"exit_sma":13}` | exit=`{"sl_atr":5.0,"tp_atr":10.0,"timeout":48}` | frac=0.91
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":100,"slope_bars":20,"exit_sma":21}` | exit=`{"sl_atr":3.0,"tp_atr":4.0,"trail_atr":5.0,"timeout":36}` | frac=0.91
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":50,"exit_sma":21}` | exit=`{"sl_atr":4.0,"tp_atr":7.0,"timeout":48}` | frac=0.90
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":10,"sma_trend":100,"slope_bars":100,"exit_sma":21}` | exit=`{"sl_atr":5.0,"tp_atr":7.0,"trail_atr":5.0,"timeout":96}` | frac=0.90
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":100,"slope_bars":100,"exit_sma":8}` | exit=`{"sl_atr":4.0,"tp_atr":5.0,"timeout":72}` | frac=0.90
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":80,"exit_sma":8}` | exit=`{"sl_atr":5.0,"timeout":96}` | frac=0.90
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":10,"sma_trend":100,"slope_bars":65,"exit_sma":13}` | exit=`{"sl_atr":4.0,"trail_atr":5.0,"timeout":48}` | frac=0.90
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":100,"slope_bars":100,"exit_sma":21}` | exit=`{"sl_atr":5.0,"timeout":48}` | frac=0.89
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":20,"sma_trend":100,"slope_bars":10,"exit_sma":8}` | exit=`{"sl_atr":5.0,"timeout":48}` | frac=0.89
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":75,"slope_bars":30,"exit_sma":21}` | exit=`{"sl_atr":5.0,"tp_atr":10.0,"timeout":120}` | frac=0.88
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":125,"slope_bars":80,"exit_sma":21}` | exit=`{"sl_atr":3.0,"tp_atr":4.0,"timeout":24}` | frac=0.88
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":100,"slope_bars":5,"exit_sma":13}` | exit=`{"sl_atr":5.0,"tp_atr":10.0,"trail_atr":5.0,"timeout":48}` | frac=0.88
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":7,"sma_trend":75,"slope_bars":100,"exit_sma":13}` | exit=`{"sl_atr":4.0,"tp_atr":10.0,"timeout":36}` | frac=0.87
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":20,"sma_trend":100,"slope_bars":100,"exit_sma":21}` | exit=`{"sl_atr":3.5,"tp_atr":7.0,"trail_atr":4.0,"timeout":96}` | frac=0.87
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":100,"slope_bars":80,"exit_sma":13}` | exit=`{"sl_atr":3.5,"tp_atr":7.0,"trail_atr":5.0}` | frac=0.87
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":75,"slope_bars":100,"exit_sma":8}` | exit=`{"sl_atr":5.0,"tp_atr":6.0,"trail_atr":4.0,"timeout":12}` | frac=0.87
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":75,"slope_bars":100,"exit_sma":13}` | exit=`{"sl_atr":3.5,"tp_atr":7.0,"timeout":12}` | frac=0.87
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":75,"slope_bars":30,"exit_sma":21}` | exit=`{"sl_atr":3.0,"trail_atr":5.0,"timeout":24}` | frac=0.86
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":125,"slope_bars":65,"exit_sma":13}` | exit=`{"sl_atr":3.5,"tp_atr":10.0,"trail_atr":5.0,"timeout":24}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":18,"sma_trend":75,"slope_bars":100,"exit_sma":13}` | exit=`{"sl_atr":3.5,"tp_atr":6.0,"timeout":18}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":75,"slope_bars":65,"exit_sma":8}` | exit=`{"sl_atr":4.0,"tp_atr":6.0,"timeout":36}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":150,"slope_bars":80,"exit_sma":8}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"timeout":18}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":5,"sma_trend":125,"slope_bars":50,"exit_sma":21}` | exit=`{"sl_atr":4.0,"tp_atr":6.0,"trail_atr":5.0}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":20,"sma_trend":75,"slope_bars":100,"exit_sma":21}` | exit=`{"sl_atr":4.0,"tp_atr":6.0,"trail_atr":4.0,"timeout":24}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":100,"slope_bars":40,"exit_sma":13}` | exit=`{"sl_atr":3.5,"tp_atr":7.0}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":100,"slope_bars":40,"exit_sma":21}` | exit=`{"sl_atr":4.0,"timeout":18}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":10,"exit_sma":13}` | exit=`{"sl_atr":3.5,"tp_atr":10.0,"timeout":24}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":7,"sma_trend":75,"slope_bars":50,"exit_sma":21}` | exit=`{"sl_atr":4.0,"tp_atr":7.0,"timeout":72}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":15,"exit_sma":21}` | exit=`{"sl_atr":3.0,"tp_atr":6.0,"trail_atr":5.0,"timeout":36}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":12,"sma_trend":125,"slope_bars":65,"exit_sma":21}` | exit=`{"sl_atr":4.0,"tp_atr":5.0,"trail_atr":5.0,"timeout":36}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":5,"sma_trend":100,"slope_bars":100,"exit_sma":8}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":4.0,"timeout":72}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":7,"sma_trend":75,"slope_bars":80,"exit_sma":21}` | exit=`{"sl_atr":4.0,"tp_atr":7.0,"timeout":96}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":75,"slope_bars":5,"exit_sma":8}` | exit=`{"sl_atr":5.0,"trail_atr":4.0,"timeout":12}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":10,"sma_trend":75,"slope_bars":40,"exit_sma":8}` | exit=`{"sl_atr":3.5,"tp_atr":10.0}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":5,"exit_sma":8}` | exit=`{"sl_atr":3.0,"tp_atr":4.0,"timeout":18}` | frac=0.83
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":5,"sma_trend":125,"slope_bars":40,"exit_sma":8}` | exit=`{"sl_atr":5.0,"tp_atr":7.0,"timeout":24}` | frac=0.82
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":20,"sma_trend":100,"slope_bars":65,"exit_sma":21}` | exit=`{"sl_atr":2.5,"tp_atr":3.0,"trail_atr":5.0,"timeout":168}` | frac=0.81
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":20,"sma_trend":75,"slope_bars":100,"exit_sma":21}` | exit=`{"sl_atr":2.5,"tp_atr":10.0,"timeout":48}` | frac=0.81
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":15,"exit_sma":8}` | exit=`{"sl_atr":4.0,"tp_atr":7.0,"timeout":96}` | frac=0.81
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":125,"slope_bars":65,"exit_sma":8}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"timeout":48}` | frac=0.79
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":2.0,"trend":100}` | exit=`{"sl_atr":2.5,"tp_atr":3.0,"trail_atr":5.0,"timeout":120}` | frac=0.79
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":2.0,"trend":100}` | exit=`{"sl_atr":2.5,"tp_atr":3.5,"timeout":72}` | frac=0.79
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":1.75,"trend":100}` | exit=`{"sl_atr":2.5,"tp_atr":3.0}` | frac=0.79
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":2.0,"trend":100}` | exit=`{"sl_atr":2.0,"tp_atr":3.0,"trail_atr":4.0}` | frac=0.79
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":100,"slope_bars":10,"exit_sma":13}` | exit=`{"sl_atr":3.5,"tp_atr":5.0,"trail_atr":5.0,"timeout":168}` | frac=0.79
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":18,"sma_trend":75,"slope_bars":100,"exit_sma":21}` | exit=`{"sl_atr":3.0,"tp_atr":5.0,"trail_atr":4.0,"timeout":168}` | frac=0.79
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":1.75,"trend":100}` | exit=`{"sl_atr":2.5,"tp_atr":3.5,"timeout":120}` | frac=0.79
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":5,"sma_trend":125,"slope_bars":100,"exit_sma":8}` | exit=`{"sl_atr":3.0,"tp_atr":4.0,"timeout":12}` | frac=0.79
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":18,"sma_trend":75,"slope_bars":80,"exit_sma":21}` | exit=`{"sl_atr":4.0,"tp_atr":6.0,"trail_atr":5.0,"timeout":24}` | frac=0.79
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":12,"sma_trend":75,"slope_bars":15,"exit_sma":21}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"timeout":96}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":12,"sma_trend":75,"slope_bars":15,"exit_sma":21}` | exit=`{"sl_atr":3.5,"tp_atr":10.0,"trail_atr":5.0,"timeout":24}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":8,"sma_trend":100,"slope_bars":5,"exit_sma":13}` | exit=`{"sl_atr":3.5,"tp_atr":10.0,"trail_atr":5.0,"timeout":120}` | frac=0.78
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":2.0,"trend":100}` | exit=`{"sl_atr":2.5,"tp_atr":5.0,"trail_atr":5.0,"timeout":96}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":150,"slope_bars":50,"exit_sma":13}` | exit=`{"sl_atr":4.0,"timeout":48}` | frac=0.78
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":1.75,"trend":100}` | exit=`{"sl_atr":2.5,"tp_atr":6.0,"timeout":120}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":12,"sma_trend":100,"slope_bars":20,"exit_sma":13}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"trail_atr":5.0,"timeout":168}` | frac=0.78
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":1.75,"trend":100}` | exit=`{"sl_atr":2.5,"tp_atr":8.0,"trail_atr":5.0}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":12,"sma_trend":100,"slope_bars":30,"exit_sma":21}` | exit=`{"sl_atr":3.5,"tp_atr":5.0,"trail_atr":5.0,"timeout":36}` | frac=0.78
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":1.75,"trend":100}` | exit=`{"sl_atr":2.0,"tp_atr":2.5,"timeout":36}` | frac=0.78
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":1.75,"trend":100}` | exit=`{"sl_atr":2.5}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":10,"sma_trend":75,"slope_bars":15,"exit_sma":8}` | exit=`{"sl_atr":5.0,"tp_atr":6.0,"timeout":96}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":100,"slope_bars":100,"exit_sma":13}` | exit=`{"sl_atr":5.0,"tp_atr":6.0,"trail_atr":4.0,"timeout":120}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":175,"slope_bars":65,"exit_sma":13}` | exit=`{"sl_atr":4.0,"tp_atr":7.0,"timeout":72}` | frac=0.78
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":1.75,"trend":100}` | exit=`{"sl_atr":2.0,"tp_atr":4.0,"timeout":36}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":75,"slope_bars":50,"exit_sma":8}` | exit=`{"sl_atr":2.5,"tp_atr":4.0,"trail_atr":5.0,"timeout":12}` | frac=0.78
- **bb_trend_rejoin** ETHUSD 1h | params=`{"bb_len":30,"bb_mult":1.75,"trend":100}` | exit=`{"sl_atr":2.0,"tp_atr":4.0,"trail_atr":5.0}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":75,"slope_bars":80,"exit_sma":21}` | exit=`{"sl_atr":3.5,"timeout":18}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":3,"rsi_buy":8,"sma_trend":100,"slope_bars":15,"exit_sma":8}` | exit=`{"sl_atr":3.0,"tp_atr":6.0,"timeout":18}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":12,"sma_trend":100,"slope_bars":20,"exit_sma":8}` | exit=`{"sl_atr":3.5,"tp_atr":10.0,"trail_atr":5.0,"timeout":120}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":12,"sma_trend":100,"slope_bars":15,"exit_sma":8}` | exit=`{"sl_atr":3.5,"trail_atr":5.0,"timeout":18}` | frac=0.78
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":100,"slope_bars":20,"exit_sma":21}` | exit=`{"sl_atr":3.0,"tp_atr":7.0,"timeout":168}` | frac=0.77
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":75,"slope_bars":10,"exit_sma":8}` | exit=`{"sl_atr":3.5,"timeout":36}` | frac=0.77
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":75,"slope_bars":10,"exit_sma":8}` | exit=`{"sl_atr":3.5,"tp_atr":10.0,"timeout":12}` | frac=0.77
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":15,"sma_trend":100,"slope_bars":100,"exit_sma":21}` | exit=`{"sl_atr":4.0,"tp_atr":10.0,"timeout":120}` | frac=0.76
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":20,"sma_trend":100,"slope_bars":5,"exit_sma":13}` | exit=`{"sl_atr":4.0,"tp_atr":7.0,"timeout":24}` | frac=0.76
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":20,"sma_trend":75,"slope_bars":40,"exit_sma":21}` | exit=`{"sl_atr":2.0,"tp_atr":2.5,"trail_atr":5.0,"timeout":24}` | frac=0.76
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":12,"sma_trend":100,"slope_bars":20,"exit_sma":13}` | exit=`{"sl_atr":4.0,"tp_atr":10.0,"trail_atr":5.0,"timeout":96}` | frac=0.76
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":4,"rsi_buy":12,"sma_trend":100,"slope_bars":30,"exit_sma":8}` | exit=`{"sl_atr":4.0,"tp_atr":7.0,"trail_atr":5.0,"timeout":120}` | frac=0.76
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":7,"sma_trend":75,"slope_bars":40,"exit_sma":13}` | exit=`{"sl_atr":3.5,"tp_atr":4.0,"timeout":48}` | frac=0.75
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":5,"rsi_buy":18,"sma_trend":75,"slope_bars":5,"exit_sma":21}` | exit=`{"sl_atr":3.0,"tp_atr":4.0,"trail_atr":4.0,"timeout":36}` | frac=0.75
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":125,"slope_bars":100,"exit_sma":8}` | exit=`{"sl_atr":3.5,"tp_atr":10.0,"trail_atr":3.5,"timeout":48}` | frac=0.75
- **rsi2_regime** ETHUSD 1h | params=`{"rsi_len":2,"rsi_buy":3,"sma_trend":150,"slope_bars":80,"exit_sma":21}` | exit=`{"sl_atr":3.5,"trail_atr":5.0,"timeout":72}` | frac=0.75
