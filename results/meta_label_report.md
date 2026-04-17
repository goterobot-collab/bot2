# Triple-Barrier + Meta-Labeling Report

- Primary: `rsi2_regime` ETHUSD 1h
- Params: `{"rsi_len": 2, "rsi_buy": 3, "sma_trend": 100, "slope_bars": 5, "exit_sma": 8}`
- Exit cfg: `{"sl_atr": 3.5, "tp_atr": 4.0, "trail_atr": 5.0, "timeout": null}`
- Fees/slip: 0.100% / 0.050% per side
- Train/test split: first 70% / last 30% of trades (chronological)
- Classifier: **LightGBM**
- Meta-label keep threshold: P(profit) > 0.6

## Triple-barrier outcomes (full sample)

- Profit barrier hit (+1): **4**
- Stop / trail barrier hit (-1): **16**
- Timeout / signal exit (0): **138**

## Comparison

| Set | Trades | WR % | PF | Total Ret % | Max DD % | Avg Win % | Avg Loss % |
|---|---|---|---|---|---|---|---|
| Primary (full sample) | 158 | 70.89 | 1.277 | 26.96 | -13.29 | 1.116 | -2.128 |
| Baseline (test 30%, no filter) | 48 | 66.67 | 1.029 | -0.00 | -12.19 | 1.056 | -2.054 |
| Meta-filtered (test 30%, p>0.6) | 31 | 77.42 | 1.918 | 9.41 | -6.72 | 0.810 | -1.448 |

- Trades filtered out: **17** of 48 (35.4%)
- DeltaWR = **+10.75 pp**, DeltaPF = **+0.889**, DeltaDD = **+5.48 pp** (positive = shallower DD)

## Classifier calibration on holdout (5 buckets of P(profit))

| Bucket | n | avg P(profit) | actual hit rate |
|---|---|---|---|
| (0.015199999999999998, 0.0693] | 10 | 0.037 | 0.300 |
| (0.0693, 0.832] | 9 | 0.385 | 0.667 |
| (0.832, 0.971] | 10 | 0.909 | 0.700 |
| (0.971, 0.996] | 9 | 0.989 | 0.778 |
| (0.996, 1.0] | 10 | 0.998 | 0.900 |

## Verdict

**MODEST IMPROVEMENT**: meta-filter raises WR and PF without materially worsening DD on the holdout.

## Notes

- Meta-label binary target = `1 if trade.ret_pct > 0 else 0` (after fees), applied AFTER triple-barrier exit. The triple-barrier outcome is reported separately for context; the AFML formulation collapses TP -> 1 and SL/TO -> 0, but here a vertical-barrier exit can still be net-profitable, so we use realised PnL sign as the target.
- Features use only data observable at the entry bar (no look-ahead).
- Train/test is chronological (no shuffling) - leakage-safe.
- A single train/test split on a few dozen test trades is high-variance; bigger sample (more assets / longer history) is needed for a definitive call.
