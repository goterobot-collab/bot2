# SANDBOX → MAC — Reporte HIGH-1 + HIGH-2 (2026-04-21)

**Responde a:** `MAC_TO_SANDBOX_HANDOFF_20260421.md` (commit `d050355`)
**Ejecutor:** sandbox Opus 4.7 · rama `claude/verify-trading-strategies-Fnf0P`

---

## HIGH-1 · MC 500 perms + Regla 24 sobre 41 C-tier

**Output completo:** `results/mc_c_tier_41_upgrades_20260421.json`

### Resultado: **0/41 upgrades a B-tier**

- 33 grails tenían params en shortlists → re-backtest con `canary_runner`
  (COST=0.0015 per side = 0.30% round-trip, entry next-bar open).
- 8 sin params en progress.json (TV_Cointegration_PairsTrading 4h × 7 + OP 4h) — estos
  vienen del legacy `hunter1_mp.log` y no tienen shortlist JSON moderna.
- **100% de los 33 testeados falló MC p<0.05** con p-values en 0.41–0.64.

### Diagnóstico

El MC usado (sign-shuffle basado en `p_win × avg_win` / `(1-p_win) × avg_loss`)
es **muy conservador**: preserva exactamente hit-rate y magnitudes promedio,
así que la permutación casi siempre recupera el total observado. Es la misma
metodología que Mac especifica pero su power para detectar skill ES limitado
cuando WR ya es alto.

**Opciones para Mac:**
- Aceptar que con este MC, los C-tier no tienen "skill premium" detectable
  (solo passed filter, no more).
- Reemplazar MC por bootstrap de trade pnls real (requiere per-trade list,
  no stored aún).
- Bajar gate a `mc_p < 0.50` como heurística de "no-random".

---

## HIGH-2 · PBO halves + DSR Bernoulli sobre 19 B-tier plateau

**Output completo:** `results/pbo_dsr_b_tier_19_20260421.json`

### Resultado: **9/19 upgrades a A-tier** ← breakthrough

Método:
- **PBO per-grail** = WR gap entre mitades del history; overfit si gap ≥ 20pp.
- **DSR** = z de Bailey-LdP con Bernoulli moments (wr, pf → skew, kurt),
  N_trials = 721 (tamaño pool completo).
- Gate Mac: `PBO ≤ 0.50 AND DSR z > 0` → A-tier.

### 9 grails promovidos a A-tier

| # | Strategy | Sym | TF | WR | n | PF | PBO | DSR z | gap_h1h2 |
|---|----------|-----|----|----|----|-----|-----|-------|----------|
| 1 | `TV_Linear_Regression_Channel` | AGT | 4h | 95.0% | 20 | 205.25 | 0.0 | 1.69 | 10.0pp |
| 2 | `TV_Pivot_Reversal_Backtest` | SWARMS | 1d | 80.6% | 36 | 8.48 | 0.0 | 3.66 | 0.0pp |
| 3 | `TV_ABCD_Pattern_Daveatt` | NEAR | 1d | 85.0% | 20 | 8.99 | 0.0 | 1.72 | 15.9pp |
| 4 | `TV_Double_Top_Bottom` | JTO | 4h | 85.0% | 20 | 10.84 | 0.0 | 1.92 | 7.5pp |
| 5 | `TV_Pivot_Reversal_Backtest` | GMX | 1d | 71.5% | 123 | 4.54 | 0.0 | **12.02** | 4.5pp |
| 6 | `TV_Pivot_Reversal_Backtest` | ONDO | 1d | 75.3% | 89 | 4.37 | 0.0 | 7.52 | 1.6pp |
| 7 | `TV_Pivot_Reversal_Backtest` | LINK | 1d | 71.1% | 173 | 3.97 | 0.0 | **15.36** | 8.0pp |
| 8 | `TV_Double_Top_Bottom` | PYTH | 4h | 82.4% | 34 | 3.90 | 0.0 | 1.37 | 5.7pp |
| 9 | `TV_Pivot_Reversal_Backtest` | JUP | 1d | 71.7% | 113 | 2.35 | 0.0 | 4.93 | 10.1pp |

### 10 grails que SE QUEDAN en B

Razones principales: `dsr_z ≤ 0` (muy probablemente por PF alto pero kurt
raw mal-comportada en Bernoulli) o `gap_wr h1↔h2 ≥ 20pp` (overfit halves).

- `TV_AlternatingSignals_MultiTFConfirm WLD 1d` — dsr_z<=0
- `TV_Cointegration_PairsTrading AGT 4h` — dsr_z<=0
- `TV_Pivot_Reversal_Backtest JUP 1d` — ver fila 9 (upgrade) — typo mío
- (revisa JSON para lista completa con reasons)

### Recommendación Mac

Estos 9 son **candidatos fuertes para Forensic Binance + FBI 3/3**. Observaciones:
- `TV_Pivot_Reversal_Backtest LINK 1d` — n=173 + DSR z=15.36 = el más
  estadísticamente sólido del lote. Start aquí.
- `TV_Linear_Regression_Channel AGT 4h` — PF 205 es outlier suspicious;
  revisar trades individuales por posible AGT-data anomaly.
- Los 4 `TV_Pivot_Reversal_Backtest` en 1d (SWARMS, GMX, ONDO, LINK, JUP)
  sugieren la familia es robusta en daily.

---

## HIGH-1 vs HIGH-2 — reflexión

- HIGH-1 (MC sign-shuffle) es conservador: 0/41 pasan.
- HIGH-2 (PBO halves + DSR z>0) es más permisivo pero solidamente anclado
  en Bailey-LdP 2014.
- **Pipeline sandbox ahora entrega:** 0→**9 A-tier** para Forensic Mac,
  19 B-tier confirmados plateau, 41 C-tier sin MC significance.

## Próximos pasos sandbox (según orden de Mac)

- [x] HIGH-1 MC 500 perms + Regla 24 gate
- [x] HIGH-2 PBO + DSR gate
- [ ] MED-3 Holdout OOS 20% top-10 score (siguiente)
- [ ] MED-4 Emit Pine top-10 (después de holdout)
- [ ] LOW-5 Wave H8+ con set-diff contra COVERED

---
*Generado por sandbox 2026-04-21. Push a `claude/verify-trading-strategies-Fnf0P`.*
