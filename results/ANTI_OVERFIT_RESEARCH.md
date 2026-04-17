# Anti-Overfit Research Brief

Compiled from agent research (López de Prado, Harvey & Liu, Bailey, Pardo, White).

## Diagnosis of v1 grail loop

- N=16,600 "grails" found out of ~857k iterations
- Minimum trades was 5; WR>60% filter — no Deflated Sharpe, no OOS holdout
- Expected max Sharpe under null with N=16,000 ≈ √(2·ln(16000)) = **4.37**
- User confirmed #1 grail collapsed in TV Strategy Tester:
  WR 90.62% → 62.58%, PF 8.5 → 0.75, +129% → -59.43%
- PBO (Probability of Backtest Overfitting) likely >0.8 — selection barely better than random

## 8 anti-overfit techniques (priority order)

### 1. Walk-Forward Analysis — **v2 implemented**
- **Current v2:** 50/50 split; config must pass grail filter on full + both halves
- **Pardo's rule:** IS:OOS 3:1 to 4:1, 8-12 windows, re-optimize every 1/4 IS; OOS efficiency > 0.5
- **TODO:** upgrade to rolling WFA with 8-12 windows

### 2. Deflated Sharpe Ratio (DSR) — **not implemented**
- Deflates observed SR by expected max SR under null given N, skew, kurtosis
- **Rule:** DSR p-value < 0.05 (i.e., DSR > 0.95)
- Implementation: `mlfinlab.backtest_statistics.deflated_sharpe_ratio()`
- **Sources:** Bailey & LdP 2014 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

### 3. Monte Carlo Permutation — **not implemented**
- Shuffle signals (not returns) ≥1,000 times, build null distribution
- **Rule:** p < 0.01
- **Sources:** Aronson *Evidence-Based Technical Analysis*; Masters Monte Carlo methods

### 4. Locked holdout — **partially: v2 has 50/50 not 60/20/20**
- Reserve final 20% untouched; test ONCE
- **TODO:** split data 40/40/20 instead of 50/50

### 5. Parameter-plateau stability — **not implemented**
- Jitter each param ±20%, require ≥70-80% of neighbors to remain profitable
- Rejects spike-overfits
- **TODO:** add `tools/plateau_test.py`

### 6. PBO / CSCV (López de Prado) — **not implemented**
- Split perf matrix into S=16 submatrices; C(16,8) train/test combos; measure IS-best vs OOS median
- **Rule:** PBO < 0.5 (ideally < 0.2)
- Implementation: `mlfinlab` has it
- **Sources:** Bailey, Borwein, LdP, Zhu 2014

### 7. Harvey-Liu Bonferroni haircut — **not implemented**
- For N=16,000 trials: Bonferroni t-threshold ≈ 4.66
- Haircut SR = (SR - SR_adjusted) / SR; reject if SR_haircut < 0.5
- **Sources:** Harvey & Liu 2015 https://faculty.fuqua.duke.edu/~charvey/Research/Published_Papers/P116_Evaluating_trading_strategies.pdf

### 8. Minimum Track Record (MinTRL) — **not implemented**
- MinTRL = 1 + [1 - γ₃·SR + (γ₄-1)/4·SR²] · (Z_α / (SR - SR*))²
- For target SR=1 at 95% conf: ~2-3y daily (~500-750 obs)
- **Sources:** Bailey & LdP 2012

## Recommended tooling stack

| Layer | Library | Why |
|-------|---------|-----|
| Overfit metrics (PBO/DSR) | `mlfinlab` | Canonical LdP implementations |
| WF cross-validation | `skfolio` | `CombinatorialPurgedCV` |
| Parameter search | `optuna` | + `MedianPruner` for early stop |
| Backtest engine | `pybroker` or `vectorbt` | Native WF |

## Priority upgrades for this repo

1. **NOW:** v2 walk-forward already running (50/50 split, trades>50, min 20/half)
2. **NEXT:** parameter-plateau test — wrap each surviving grail, jitter params ±20%, require ≥75% neighbors still passing. Kills isolated-spike overfits.
3. **THEN:** compute PBO on the v2 jsonl pool. If >0.5, whole v2 loop is still contaminated.
4. **FINAL:** locked 20% OOS holdout — only touched after all other filters pass.

---

## Crypto strategy archetypes — evidence-rated

**Reality check:** peer-reviewed crypto research shows edge comes from payoff
ASYMMETRY, not hit rate. Real 1h studies report WR 45–58%. Strategies
claiming >60% WR are usually (a) curve-fit, (b) mean-rev with fat-tail
blow-ups, or (c) bull-only subsample.

### Rating A (strongest, peer-reviewed / mechanical edge)

| # | Archetype | Typical WR | Notes |
|---|-----------|-----------|-------|
| 1 | Cross-sectional momentum (top decile / bottom decile, weekly) | 52-55% | Liu-Tsyvinski-Wu 2022 *RFS* |
| 2 | Time-series momentum + vol-targeting (BTC/ETH) | 48-55% | Bianchi-Babiak 2022; Moskowitz-style |
| 3 | Funding-rate carry / perp-spot basis (BTC/ETH only) | **>70%** | Soska 2021; mechanical edge |
| 4 | Market-making (Avellaneda-Stoikov) | >65% | Needs sub-ms infra, not 1h-bar |

### Rating B (moderate)

| # | Archetype | Typical WR | Notes |
|---|-----------|-----------|-------|
| 5 | Cointegrated pairs (ETH-BTC, SOL-ETH) | 60-65% | Regime-break risk (LUNA, FTX) |
| 6 | Regime-switching trend (HMM + breakout) | mixed | Ardia 2019; edge in Sharpe not WR |
| 7 | LightGBM on residual returns (G-Research Kaggle winners) | <55% | Tiny per-trade edge, huge volume |
| 8 | Z-score mean-rev on BTC-residualized returns | 58-64% | Fat left-tail on regime shift |
| 9 | Triple-Barrier + Meta-Labeling (López de Prado) | 52% → 60%+ | Framework, not strategy |

### Rating C (weak / no independent evidence)

| # | Archetype | Why rated C |
|---|-----------|-------------|
| 10 | **BB + EMA-trend filter on 1h** | No peer-reviewed OOS evidence. Chan & Lento 2021 found BB variants on BTC daily lost edge after costs. Matches my v1 grails collapse. |

## Reorientation

My v1 and v2 loops both focus on indicator-based entry/exit on single-asset
1h bars — which the literature says is archetype #10 (weak C). The
evidence-backed edges (TSMOM, funding carry, pairs) are NOT in my search
space.

**Next-version loop should include:**
- TSMOM with vol targeting
- Meta-labeling wrapper around existing primary signals
- Pairs/cross-asset trades (requires multi-asset data)
- Funding-rate carry (requires funding-rate feed, not in current data)

## Sources (agent research output)

- Harvey & Liu 2015 — https://faculty.fuqua.duke.edu/~charvey/Research/Published_Papers/P116_Evaluating_trading_strategies.pdf
- Bailey & LdP DSR — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Bailey/Borwein/LdP/Zhu PBO — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Liu-Tsyvinski-Wu 2022 — https://academic.oup.com/rfs/article/35/11/4667/6427708
- Funding-rate carry — https://arxiv.org/abs/2108.12508
- Pardo *Evaluation and Optimization of Trading Strategies*
- LdP *Advances in Financial Machine Learning*
- mlfinlab — https://github.com/hudson-and-thames/mlfinlab
- skfolio — https://github.com/skfolio/skfolio
- PyBroker — https://github.com/edtechre/pybroker
- vectorbt — https://github.com/polakowo/vectorbt
