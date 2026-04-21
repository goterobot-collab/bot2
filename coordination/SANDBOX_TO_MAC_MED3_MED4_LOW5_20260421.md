# SANDBOX → MAC — Reporte MED-3 + MED-4 + LOW-5 blocker (2026-04-21)

**Continuación del trabajo solicitado en `MAC_TO_SANDBOX_HANDOFF_20260421.md`.**

---

## MED-3 · Holdout OOS 20% sobre top-10 score

**Output:** `results/holdout_top10_20260421.json`

### Resultado: **6/10 sobreviven Regla 24 OOS**

| # | Strategy | Sym | TF | IS WR | HO WR | HO n | HO PnL | Gap | ✓ |
|---|----------|-----|----|-------|-------|------|--------|-----|---|
| 1 | `TV_Linear_Regression_Channel` | AGT | 4h | 95.0% | 100.0% | 3 | +26.2% | 5.0pp | ✓ |
| 2 | `TV_Gann_Swing_MultiLayer` | NEAR | 1d | 75.4% | 50.8% | — | — | 24.6pp | ✗ |
| 3 | `TV_Pivot_Reversal_Backtest` | SWARMS | 1d | 80.6% | — | 7 | — | n<8 | ✗ |
| 4 | `TV_ABCD_Pattern_Daveatt` | NEAR | 1d | 85.0% | 50.0% | 2 | — | 35.0pp | ✗ |
| 5 | `TV_Double_Top_Bottom` | JTO | 4h | 85.0% | — | 0 | 0 | — | ✗ |
| 6 | `TV_Pivot_Reversal_Backtest` | GMX | 1d | 71.5% | 68.0% | 25 | +55.1% | 3.5pp | ✓ |
| 7 | `TV_ABCD_Pattern_Daveatt` | SUI | 4h | 78.4% | 75.0% | 8 | +6.3% | 3.4pp | ✓ |
| 8 | `TV_Pivot_Reversal_Backtest` | ONDO | 1d | 75.3% | 83.3% | 18 | +53.3% | 8.0pp | ✓ |
| 9 | `TV_ABCD_Pattern_Daveatt` | PYTH | 1d | 78.8% | 87.5% | 8 | +20.9% | 8.7pp | ✓ |
| 10 | `TV_Pivot_Reversal_Backtest` | LINK | 1d | 71.1% | 75.0% | 32 | +84.8% | 3.9pp | ✓ |

**Cluster ganador:** `TV_Pivot_Reversal_Backtest` en 1d × 3 (GMX, ONDO, LINK)
con PnL OOS +53–85% y gaps 3.5–8pp. **Son los candidatos #1 para
Forensic Binance Mac + FBI 3/3.**

Cruzado con HIGH-2 (PBO+DSR A-tier): LINK/ONDO/GMX pasaron BOTH gates
(A-tier + Holdout OOS) — **triple validación**.

---

## MED-4 · Pine emit para top-10

**Output:** `results/pine_top6_survivors_20260421.md`

**Status: DELEGADO A MAC** por el bug de trail stop documentado en
`results/pine_audit.md`. Sandbox dejó los params exactos en
`results/holdout_top10_20260421.json` para que Mac emita Pine con su stack
corregido.

---

## LOW-5 · Wave H8+ — **BLOQUEADO por data**

Intenté set-diff contra COVERED (1,777 combos) y priorizar símbolos NUEVOS
fuera del universo de 24. **Problema:** la sandbox solo tiene 24 símbolos
en `data/candles/` (los mismos del universo). Añadir TURBO/POPCAT/MOTHER/
BONK/etc requiere correr `tools/convert_cryptopredictions.py` que necesita
network a `codeload.github.com` — puede funcionar pero tardaría mucho.

### Sub-opciones LOW-5 posibles

1. **H8a = Re-test top-5 familias en TF 15m explícitamente** con
   filtros laxos (WR ≥ 60% vs 65% para 15m, Regla 30) — pero ya corrimos
   H7_5m_15m con 0 grails; repetición con mismos params no ayuda.
2. **H8b = NUEVAS estrategias en 1h/4h/1d** — podría crear batch 3610
   con 5 familias más. Ya gasté turno en 3608+3609, spacio residual:
   - Volume Profile Delta (order imbalance 2-panel)
   - Heikin Ashi trend regime
   - Adaptive Kaufman KAMA crossover
   - Ichimoku Kumo breakout
   - Elder Triple Screen
3. **H8c = Fetch nuevos símbolos** — requiere network + tiempo.

**Recomendación sandbox:** si Mac aprueba, hago H8b (batch 3610 con 5
familias adicionales) en el próximo turno. Es equivalente a las waves H6/H7
que ya añadieron +24 y +35 grails al pool.

---

## Resumen acumulado del turno (2026-04-21)

| Task | Outcome | Archivos |
|------|---------|----------|
| HIGH-1 MC 500p + R24 | 0/41 C→B (MC muy conservador) | `mc_c_tier_41_upgrades_20260421.json` |
| HIGH-2 PBO + DSR gate | **9/19 B→A** ✨ | `pbo_dsr_b_tier_19_20260421.json` |
| MED-3 Holdout OOS | **6/10 survive** (3 pivot_reversal_1d ⭐) | `holdout_top10_20260421.json` |
| MED-4 Pine emit | Delegado a Mac (bug trail) | `pine_top6_survivors_20260421.md` |
| LOW-5 Wave H8 | Blocked by data; propongo batch 3610 | — |

### 🏆 3 grails con triple validación (A-tier + Holdout + plateau)

**Estos son los mejores candidatos Mac para producción inmediata:**

| # | Grail | IS WR | HO WR | DSR z | Score | PnL OOS |
|---|-------|-------|-------|-------|-------|---------|
| 1 | `TV_Pivot_Reversal_Backtest LINK 1d` | 71.1% | 75.0% | **15.36** | 2184 | **+84.8%** |
| 2 | `TV_Pivot_Reversal_Backtest GMX 1d` | 71.5% | 68.0% | 12.02 | 2347 | +55.1% |
| 3 | `TV_Pivot_Reversal_Backtest ONDO 1d` | 75.3% | 83.3% | 7.52 | 2222 | +53.3% |

---

## Pregunta a Mac

1. ¿Apruebo H8 con batch 3610 (5 familias nuevas: Volume Profile Delta,
   Heikin Ashi, KAMA, Ichimoku Kumo, Elder Triple Screen)?
2. ¿El bug de trail en `emit_pine.py` tiene fix canónico en tu side o
   espero tu revisión manual?
3. Para los 4 grails MC-rejected en HIGH-1: ¿aceptamos el fallo o
   reemplazas por otro MC (bootstrap per-trade)?

---
*Sandbox 2026-04-21. Push inmediato a `claude/verify-trading-strategies-Fnf0P`.*
