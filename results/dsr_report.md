# Deflated Sharpe Ratio Report

Bailey & López de Prado (2014). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

- Source: `/home/user/bot2/results/grails_loop_wf.jsonl`
- Unique trials (N): **1,308,965**
- E[max SR] under H0: **4.9208**
- Min trades filter: 30
- Trials passing DSR > 0.95: **0** (0.00%)

## Heuristic SR proxy

Bernoulli payoff with win-rate w and PF p: b = p(1-w)/w, mu = wb-(1-w), sigma = sqrt(w(1-w))(b+1). Trial SR = (mu/sigma)*sqrt(trades). Skew/kurt from same model.

## Top 20 by DSR

| strategy | asset | tf | trades | wr | pf | ret | dd | sr | dsr | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bb_trend_rejoin | ETHUSD | 1h | 32 | 90.6200 | 8.5360 | 130.3900 | -4.5700 | 7.2816 | 0.8809 | 0 |
| bb_trend_rejoin | ETHUSD | 1h | 32 | 90.6200 | 8.2590 | 123.4100 | -4.5700 | 7.1224 | 0.8691 | 0 |
| bb_trend_rejoin | ETHUSD | 1h | 32 | 87.5000 | 6.9780 | 124.3800 | -5.1900 | 6.4008 | 0.8408 | 0 |
| bb_trend_rejoin | DOGEUSD | 1h | 32 | 65.6200 | 5.2600 | 55.2500 | -4.2500 | 4.6442 | 0.2711 | 0 |
| bb_trend_rejoin | DOGEUSD | 1h | 32 | 71.8800 | 4.4480 | 52.3200 | -2.8900 | 4.4523 | 0.2050 | 0 |
| bb_trend_rejoin | ETHUSD | 1h | 36 | 80.5600 | 3.1290 | 98.7200 | -8.4400 | 3.5754 | 0.0172 | 0 |
| rsi2_regime | ETHUSD | 1h | 43 | 76.7400 | 2.9960 | 72.3500 | -7.2500 | 3.7765 | 0.0144 | 0 |
| bb_trend_rejoin | ETHUSD | 1h | 32 | 81.2500 | 3.1810 | 72.7100 | -14.7400 | 3.4178 | 0.0126 | 0 |
| bb_trend_rejoin | DOGEUSD | 1h | 32 | 71.8800 | 3.5510 | 39.9000 | -3.7000 | 3.7778 | 0.0125 | 0 |
| rsi2_regime | ETHUSD | 1h | 53 | 77.3600 | 2.7320 | 63.0700 | -9.7700 | 3.7906 | 0.0096 | 0 |
| bb_trend_rejoin | ETHUSD | 1h | 32 | 84.3800 | 3.0630 | 71.7100 | -14.7400 | 3.2042 | 0.0089 | 0 |
| rsi2_regime | ETHUSD | 1h | 49 | 77.5500 | 2.7750 | 54.2300 | -10.7700 | 3.7071 | 0.0074 | 0 |
| bb_trend_rejoin | ETHUSD | 1h | 32 | 81.2500 | 2.9900 | 64.4400 | -14.7400 | 3.1999 | 0.0036 | 0 |
| bb_trend_rejoin | ETHUSD | 1h | 32 | 81.2500 | 2.9830 | 68.6100 | -14.6500 | 3.1917 | 0.0034 | 0 |
| bb_trend_rejoin | DOGEUSD | 1h | 32 | 68.7500 | 3.5090 | 42.7900 | -4.1800 | 3.6875 | 0.0029 | 0 |
| bb_trend_rejoin | ETHUSD | 1h | 32 | 78.1200 | 3.0430 | 43.6300 | -6.1500 | 3.3020 | 0.0027 | 0 |
| rsi2_regime | ETHUSD | 1h | 43 | 76.7400 | 2.8070 | 71.7200 | -8.6000 | 3.5247 | 0.0026 | 0 |
| bb_trend_rejoin | ETHUSD | 1h | 32 | 84.3800 | 2.8780 | 68.1200 | -17.5900 | 2.9821 | 0.0024 | 0 |
| bb_trend_rejoin | ETHUSD | 1h | 37 | 75.6800 | 2.9640 | 91.5900 | -7.8000 | 3.4685 | 0.0023 | 0 |
| adx_pullback | ADAUSD | 1h | 31 | 3.2300 | 0.0570 | -41.2200 | -39.7700 | -10.6135 | 0.0017 | 0 |

## Top 20 - parameter detail

- **bb_trend_rejoin / ETHUSD 1h** — DSR=0.8809 SR=7.282 trades=32 PF=8.536 ret=130.39%
  - params: `{"bb_len": 100, "bb_mult": 2.25, "trend": 300}`
  - exit:   `{"sl_atr": 3.5, "timeout": 168, "tp_atr": 4.0, "trail_atr": null}`
- **bb_trend_rejoin / ETHUSD 1h** — DSR=0.8691 SR=7.122 trades=32 PF=8.259 ret=123.41%
  - params: `{"bb_len": 100, "bb_mult": 2.25, "trend": 300}`
  - exit:   `{"sl_atr": 3.5, "timeout": null, "tp_atr": 8.0, "trail_atr": 5.0}`
- **bb_trend_rejoin / ETHUSD 1h** — DSR=0.8408 SR=6.401 trades=32 PF=6.978 ret=124.38%
  - params: `{"bb_len": 100, "bb_mult": 2.25, "trend": 300}`
  - exit:   `{"sl_atr": 4.0, "timeout": 72, "tp_atr": 6.0, "trail_atr": null}`
- **bb_trend_rejoin / DOGEUSD 1h** — DSR=0.2711 SR=4.644 trades=32 PF=5.260 ret=55.25%
  - params: `{"bb_len": 24, "bb_mult": 2.0, "trend": 100}`
  - exit:   `{"sl_atr": 4.0, "timeout": 18, "tp_atr": null, "trail_atr": 5.0}`
- **bb_trend_rejoin / DOGEUSD 1h** — DSR=0.2050 SR=4.452 trades=32 PF=4.448 ret=52.32%
  - params: `{"bb_len": 24, "bb_mult": 2.0, "trend": 100}`
  - exit:   `{"sl_atr": 2.5, "timeout": 12, "tp_atr": 3.5, "trail_atr": null}`
- **bb_trend_rejoin / ETHUSD 1h** — DSR=0.0172 SR=3.575 trades=36 PF=3.129 ret=98.72%
  - params: `{"bb_len": 80, "bb_mult": 2.75, "trend": 300}`
  - exit:   `{"sl_atr": 3.5, "timeout": 96, "tp_atr": 6.0, "trail_atr": 5.0}`
- **rsi2_regime / ETHUSD 1h** — DSR=0.0144 SR=3.776 trades=43 PF=2.996 ret=72.35%
  - params: `{"exit_sma": 21, "rsi_buy": 10, "rsi_len": 4, "slope_bars": 5, "sma_trend": 100}`
  - exit:   `{"sl_atr": 3.0, "timeout": 18, "tp_atr": 5.0, "trail_atr": 3.5}`
- **bb_trend_rejoin / ETHUSD 1h** — DSR=0.0126 SR=3.418 trades=32 PF=3.181 ret=72.71%
  - params: `{"bb_len": 100, "bb_mult": 2.25, "trend": 300}`
  - exit:   `{"sl_atr": 3.5, "timeout": 36, "tp_atr": 4.0, "trail_atr": 3.5}`
- **bb_trend_rejoin / DOGEUSD 1h** — DSR=0.0125 SR=3.778 trades=32 PF=3.551 ret=39.90%
  - params: `{"bb_len": 24, "bb_mult": 2.0, "trend": 100}`
  - exit:   `{"sl_atr": 3.5, "timeout": 12, "tp_atr": 4.0, "trail_atr": 3.5}`
- **rsi2_regime / ETHUSD 1h** — DSR=0.0096 SR=3.791 trades=53 PF=2.732 ret=63.07%
  - params: `{"exit_sma": 13, "rsi_buy": 10, "rsi_len": 4, "slope_bars": 65, "sma_trend": 125}`
  - exit:   `{"sl_atr": 5.0, "timeout": 18, "tp_atr": null, "trail_atr": null}`
- **bb_trend_rejoin / ETHUSD 1h** — DSR=0.0089 SR=3.204 trades=32 PF=3.063 ret=71.71%
  - params: `{"bb_len": 100, "bb_mult": 2.25, "trend": 300}`
  - exit:   `{"sl_atr": 3.5, "timeout": 168, "tp_atr": null, "trail_atr": 3.5}`
- **rsi2_regime / ETHUSD 1h** — DSR=0.0074 SR=3.707 trades=49 PF=2.775 ret=54.23%
  - params: `{"exit_sma": 13, "rsi_buy": 15, "rsi_len": 5, "slope_bars": 50, "sma_trend": 100}`
  - exit:   `{"sl_atr": 5.0, "timeout": 12, "tp_atr": 10.0, "trail_atr": null}`
- **bb_trend_rejoin / ETHUSD 1h** — DSR=0.0036 SR=3.200 trades=32 PF=2.990 ret=64.44%
  - params: `{"bb_len": 100, "bb_mult": 2.25, "trend": 300}`
  - exit:   `{"sl_atr": 3.5, "timeout": 36, "tp_atr": 5.0, "trail_atr": 3.5}`
- **bb_trend_rejoin / ETHUSD 1h** — DSR=0.0034 SR=3.192 trades=32 PF=2.983 ret=68.61%
  - params: `{"bb_len": 100, "bb_mult": 2.25, "trend": 300}`
  - exit:   `{"sl_atr": 4.0, "timeout": 72, "tp_atr": 5.0, "trail_atr": 3.5}`
- **bb_trend_rejoin / DOGEUSD 1h** — DSR=0.0029 SR=3.687 trades=32 PF=3.509 ret=42.79%
  - params: `{"bb_len": 24, "bb_mult": 2.0, "trend": 100}`
  - exit:   `{"sl_atr": 2.0, "timeout": 12, "tp_atr": 3.5, "trail_atr": 5.0}`
- **bb_trend_rejoin / ETHUSD 1h** — DSR=0.0027 SR=3.302 trades=32 PF=3.043 ret=43.63%
  - params: `{"bb_len": 100, "bb_mult": 2.25, "trend": 300}`
  - exit:   `{"sl_atr": 3.5, "timeout": 18, "tp_atr": 5.0, "trail_atr": 5.0}`
- **rsi2_regime / ETHUSD 1h** — DSR=0.0026 SR=3.525 trades=43 PF=2.807 ret=71.72%
  - params: `{"exit_sma": 21, "rsi_buy": 10, "rsi_len": 4, "slope_bars": 5, "sma_trend": 100}`
  - exit:   `{"sl_atr": 3.5, "timeout": 18, "tp_atr": 4.0, "trail_atr": 3.5}`
- **bb_trend_rejoin / ETHUSD 1h** — DSR=0.0024 SR=2.982 trades=32 PF=2.878 ret=68.12%
  - params: `{"bb_len": 100, "bb_mult": 2.25, "trend": 300}`
  - exit:   `{"sl_atr": 4.0, "timeout": 120, "tp_atr": 5.0, "trail_atr": 4.0}`
- **bb_trend_rejoin / ETHUSD 1h** — DSR=0.0023 SR=3.469 trades=37 PF=2.964 ret=91.59%
  - params: `{"bb_len": 80, "bb_mult": 2.75, "trend": 300}`
  - exit:   `{"sl_atr": 3.0, "timeout": null, "tp_atr": 4.0, "trail_atr": null}`
- **adx_pullback / ADAUSD 1h** — DSR=0.0017 SR=-10.614 trades=31 PF=0.057 ret=-41.22%
  - params: `{"adx_len": 14, "adx_min": 40, "ema_len": 20, "lookback": 5}`
  - exit:   `{"sl_atr": 1.5, "timeout": null, "tp_atr": null, "trail_atr": 3.0}`
