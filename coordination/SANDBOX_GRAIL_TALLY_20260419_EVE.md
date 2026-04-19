# SANDBOX GRAIL TALLY — 2026-04-19 19:20 UTC (evening update)

**Autor:** sandbox (/home/user/bot2) — Claude Opus 4.7
**Rama:** `claude/verify-trading-strategies-Fnf0P`
**Último commit:** `996e548`

## Total encontrados

| Métrica | Valor | Δ vs 13:45 |
|---|---|---|
| **Pool total post-dedup** | **644 grails** | +78 |
| **V8 FINAL MASTER** | **56 grails** (A=0, B=19, C=37) | +10 |

## Por wave (HUNTER multi-stage)

| Wave | TFs | Progreso | Grails brutos |
|------|-----|----------|---------------|
| **H1_1h_4h_1d** (batch 3566–3569) | 1h/4h/1d | 100% | 102 |
| **H1_15m_5m** | 15m/5m | 400/840 (47.6%) | 20 (en curso) |
| **H2_1h_4h_1d** (batch 3594/3598) | 1h/4h/1d | 100% | 294 |
| **H2_5m_15m** | 5m/15m | 1400/4158 (33.7%) | 4 (en curso) |
| **H3_1h_4h_1d** (batch 3601–3603) | 1h/4h/1d | 100% | 114 |
| **H3_5m_15m** | 5m/15m | 100% | 7 |
| **H4_1h_4h_1d** (batch 3607, familias crypto) | 1h/4h/1d | 100% | 26 |
| **H4_5m_15m** | 5m/15m | 100% | 0 |
| **H5_1h_4h_1d** (batch 3604–3606, nuevas familias) ✨ | 1h/4h/1d | **100%** | **70** |
| **H5_5m_15m** ✨ | 5m/15m | iniciando | 0 |
| **Total bruto** | — | — | **~637** |

## H5 wave (recién completada — fractal/Ehlers/regression)

Nuevas 15 estrategias testeadas, 70 grails en 1h/4h/1d:

**Familias batch 3604 — fractal/chaos:**
- `TV_Hurst_Regime_Switch`
- `TV_Detrended_Fluctuation` (top performer en 1d)
- `TV_FractalDimension_Break`
- `TV_ZigZag_Swing`
- `TV_Bill_Williams_Fractal`

**Familias batch 3605 — Ehlers DSP:**
- `TV_Ehlers_MESA_Cross`
- `TV_Ehlers_Sine_Wave`
- `TV_Ehlers_Fisher_Transform`
- `TV_Ehlers_Super_Smoother`
- `TV_Ehlers_Trend_Mode`

**Familias batch 3606 — regresión/ML:**
- `TV_Linear_Regression_Channel`
- `TV_Kernel_Ridge_Score`
- `TV_OLS_Residual_Fade`
- `TV_AR1_Forecast`
- `TV_Information_Coefficient`

## Mejoras del hunter aplicadas hoy (commit `8c496d2`)

**#1 Early-abort low-signal combos:**
- Si primeros 5 trials dan 0 trades → aborta los 15 restantes
- Ahorro ~30% wall time en combos infértiles

**#4 Warmstart desde progress.json:**
- Carga 465 params conocidos buenos de h1-h5 previos
- Inyecta como trials 0-2 antes de random sampling
- Verificado: Session_Vol_Regime/PYTH/1d hits WR=73.5% PF=6.45 en trial 0

Backward compatible. Sin warmstart disponible → comportamiento original.

## Tiers V8 actuales

- **Tier A** (plateau + MC p<0.05): **0**
- **Tier B** (plateau OR MC p<0.05): **19**
- **Tier C** (n≥30 + WR≥65% + PF≥1.3): **37**
- **Total V8**: **56**

## Estado del hunter ahora

- **18 workers vivos** (h1_15m_5m, h2_5m_15m, h5_5m_15m × 6 c/u)
- Watcher PID 6397 (esperando movimiento o ETA 40min)
- Commit+push automático cada ciclo (~45min)

## Pendiente / cola

- H1_15m_5m terminar (~440 tasks restantes, ETA ~8h)
- H2_5m_15m terminar (~2758 tasks restantes, ETA ~10h)
- H5_5m_15m recién lanzado (~945 tasks, ETA ~6h)
- Después: Monte-Carlo ampliado + PBO + DSR + holdout OOS sobre V8
- Posible h6 wave: buscar batches 3608+ o crear nuevas familias
