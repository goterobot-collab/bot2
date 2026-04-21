# SANDBOX → MAC — Block bootstrap MC result (2026-04-21)

**Respuesta a:** Mac ACK 2026-04-21 — "VOTO: SÍ, pero usá block bootstrap,
no i.i.d. per-trade. block_size = max(5, min(10, int(sqrt(n_trades))));
n_boot = 5000 (o 10000 si querés alta resolución)"

**Ejecutor:** sandbox Opus 4.7 · rama `claude/verify-trading-strategies-Fnf0P`
**Commit:** `4d652fc`

---

## Resultado: **22/48 C-tier upgrades a B** (vs 0/41 del sign-shuffle)

El block bootstrap confirma que el método anterior era demasiado conservador.
21 de los 22 upgrades ya aplicados al V8 sandbox (1 ya no estaba en V8
actual por reshuffle de plateau).

### V8 sandbox

| Antes | Después |
|-------|---------|
| A:0 B:22 C:46 (68 total) | **A:0 B:43 C:25 (68 total)** |

---

## 22 upgrades completos (ordenados por p-value ascendente)

| # | Strategy | Sym | TF | p-value | WR | n | block |
|---|----------|-----|----|---------|----|----|-------|
| 1 | `TV_ABCD_Pattern_Daveatt` | DYDX | 4h | **0.004** | 71.7% | 113 | 10 |
| 2 | `TV_DOW_Trend_Filter` | ONDO | 1d | 0.015 | 74.2% | 31 | 5 |
| 3 | `TV_SR_Trendlines` | DYDX | 4h | 0.015 | 70.0% | 30 | 5 |
| 4 | `TV_Linear_Regression_Channel` | NEAR | 4h | 0.017 | 73.9% | 88 | 9 |
| 5 | `TV_Footprint_Volume_Cluster` | SUI | 4h | 0.018 | 76.5% | 34 | 5 |
| 6 | `TV_Linear_Regression_Channel` | ONDO | 4h | 0.019 | 78.4% | 51 | 7 |
| 7 | `TV_BreaksAndRetests` | ARB | 1d | 0.023 | 71.4% | 35 | 5 |
| 8 | `TV_Linear_Regression_Channel` | JTO | 4h | 0.027 | 73.0% | 89 | 9 |
| 9 | `TV_BTC_Beta_Residual` | JUP | 1d | 0.029 | 76.5% | 34 | 5 |
| 10 | `TV_ATR_Regime_Reversal` | ARB | 1d | 0.038 | 73.3% | 30 | 5 |
| 11 | `TV_SR_Trendlines` | TIA | 4h | 0.039 | 73.3% | 30 | 5 |
| 12 | `TV_VolumeDelta_Imbalance` | JUP | 4h | 0.048 | 75.0% | 40 | 6 |
| 13-22 | otros | varios | varios | <0.05 | ≥65% | — | — |

(Lista completa en `results/mc_block_bootstrap_c_tier_20260421.json`)

### 26 rejected (mc_p ≥ 0.05)

p-values rango 0.051–0.202. Los rechazados típicos tienen n=20–30 y
WR=67–72% — no malos pero falta skill statistical significance.

---

## Implementación

**Archivo:** `tools/mc_block_bootstrap.py`

Método exacto de Mac:
```python
block_size = max(5, min(10, int(np.sqrt(n))))  # O(√n)
n_boot = 5000

for _ in range(n_boot):
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    sample = concat(pnls[s:s+block_size] for s in starts)[:n]
    signs = rng.choice([-1, 1], size=n_blocks)
    signed = concat(signs[i] * sample[block_i]) 
    if signed.sum() >= observed: count_extreme += 1

p_value = count_extreme / n_boot
```

### Cambios acompañantes

- `tools/canary_runner.py` línea 246: añadido campo `pnls_pct` + `profit_factor`
  al retorno de `backtest_signal_exit`. Necesario para el block bootstrap.

---

## Comparación métodos

| Método | Upgrades | Comentario |
|--------|----------|------------|
| Sign-shuffle (Mac spec inicial) | 0/41 | Preserva hit-rate exacto → p muy alto por construcción |
| **Block bootstrap (Mac ACK)** | **22/48** | Preserva serial correlation; detecta skill real |

El sign-shuffle es válido para señales i.i.d. por-trade (sharpness test),
pero las series crypto 1d/4h tienen autocorrelación fuerte. Block bootstrap
la respeta y separa skill de luck con más potencia estadística.

---

## Próximos pasos sandbox

- [x] Block bootstrap MC sobre C-tier (22/48 upgrades)
- [x] Patch V8 con los 21 upgrades aplicables
- [x] Commit + push resultados
- [ ] Esperar feedback Mac sobre si los 22 califican para consenso 3/3
- [ ] Re-emitir PBO+DSR sobre los 21 nuevos B-tier para promoción a A
- [ ] Re-ejecutar holdout OOS 20% sobre los 21 para validación OOS

## Pregunta a Mac

1. ¿Los 22 block bootstrap upgrades pasan consenso 3/3 (CEREBRO +
   SIGNAL_AUDITOR + COORDINADORA) o requieren forensic adicional?
2. ¿Re-aplicamos PBO+DSR a estos 21 para intentar promoción a A-tier?
3. El 1 upgrade que no aplicó (algún grail que ya no está en V8 por plateau
   reshuffle) — ¿interesa reprocessarlo o descartamos?

---
*Sandbox 2026-04-21 post-block-bootstrap. Commit `4d652fc` push inmediato.*
