# Probability of Backtest Overfitting (PBO)

Method: simplified 2-fold CSCV (Bailey, Borwein, Lopez de Prado, Zhu 2014)
using h1_ret / h2_ret as the two walk-forward periods.

- Unique trials: **1,303,077**
- **PBO = 0.000**

_Healthy: in-sample winners tend to remain above-median out of sample._

## Fold detail

| Fold | Train | Test | Best train return | Test return | Test rel-rank | Logit | Overfit? |
|------|-------|------|------------------:|------------:|--------------:|------:|:--------:|
| A | h1 | h2 | 558.85 | 182.28 | 0.9999 | 8.898 | no |
| B | h2 | h1 | 761.82 | 140.07 | 0.9999 | 9.011 | no |

Note: with only two walk-forward halves the CSCV combinatorial set has size 2,
so PBO is quantized to {0.0, 0.5, 1.0}. Treat as a coarse sanity check.
