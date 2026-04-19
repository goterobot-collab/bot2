# SANDBOX GRAIL TALLY — 2026-04-19 13:45 UTC

**Autor:** sandbox (/home/user/bot2) — Claude
**Rama:** `claude/verify-trading-strategies-Fnf0P`
**Último commit:** `5237aca`

## Total encontrados

| Métrica | Valor |
|---|---|
| **Pool total post-dedup** | **566 grails** |
| **V8 FINAL MASTER** | **46 grails** (A=0, B=17, C=29) |

## Por wave (HUNTER multi-stage)

| Wave | TFs | Progreso | Grails brutos |
|------|-----|----------|---------------|
| **H1_1h_4h_1d** (batch 3566–3569) | 1h/4h/1d | 100% | **102** |
| **H1_15m_5m** | 15m/5m | 400/840 (47.6%) | 20 (en curso) |
| **H2_1h_4h_1d** (batch 3594/3598) | 1h/4h/1d | 100% | **294** |
| **H2_5m_15m** | 5m/15m | 1400/4158 (33.7%) | 4 (en curso) |
| **H3_1h_4h_1d** (batch 3601–3603) | 1h/4h/1d | 100% | **114** |
| **H3_5m_15m** | 5m/15m | 100% | 7 |
| **H4_1h_4h_1d** (batch 3607, familias nuevas) | 1h/4h/1d | 100% | **26** |
| **H4_5m_15m** | 5m/15m | 100% | 0 |
| **Total bruto (pre-dedup)** | — | — | **~567** |

## Tiers V8 (filtro anti-overfit sandbox)

- **Tier A** (plateau + MC p<0.05): **0**
- **Tier B** (plateau OR MC p<0.05): **17**
- **Tier C** (n≥30 + WR≥65% + PF≥1.3): **29**
- **Total V8**: **46**

## Batch 3607 (HUNTER4 nuevo — 3 familias crypto-specific)

Creado hoy, inyectó 26 grails en H4_1h_4h_1d:

- `TV_Funding_Rate_Fade` — perp-funding proxy (return-acceleration jerk z-score)
- `TV_Session_Vol_Regime` — ATR-quantile regime switch (high=trend / low=revert)
- `TV_OrderFlow_Momentum_Proxy` — close-location × signed-volume exhaustion fade

**Top H4 grails con n estadísticamente significativo:**

| Strategy | Sym | TF | WR | n | PF | total % |
|---|---|---|---|---|---|---|
| TV_Session_Vol_Regime | PYTH | 1d | 73.5% | 34 | 6.45 | 253% |
| TV_OrderFlow_Momentum_Proxy | DYDX | 1d | 77.3% | 22 | 2.64 | 48% |
| TV_Funding_Rate_Fade | DYDX | 4h | 71.4% | 14 | 24.50 | 39% |
| TV_Funding_Rate_Fade | JUP | 1d | 76.9% | 13 | 23.33 | 111% |
| TV_Funding_Rate_Fade | PYTH | 4h | 83.3% | 12 | 5.18 | 62% |

## Estado del hunter ahora

- **13 procesos hunter vivos** (H1_15m_5m + H2_5m_15m)
- Watcher en background refresca V8 cada ~40 min
- Commit+push automático cada ciclo

## Pendiente / cola

- H1_15m_5m terminar (~400 tasks restantes, ETA ~10h)
- H2_5m_15m terminar (~2758 tasks restantes, ETA ~12h)
- Después: Monte-Carlo ampliado + PBO + DSR + holdout OOS sobre V8
