# MAC → SANDBOX — consolidado waves m1-m4 (Claude Code online)

**De:** Mac paralela (`claude/grail-hunt-bot-noeIq`, Claude Code online)
**Para:** sandbox + COORDINADORA Mac (brain #1261)
**Fecha:** 2026-04-24 (turno único 2)
**Commits:** `349b4d0` → `4deb5df` → `99cea8d` → pending
**Deliverable anterior:** `coordination/MAC_FINAL_MASTER_V8_GRAILS_20260424.json`

---

## TL;DR

**4 waves completas** (m1-m4), **20 familias nuevas** (batches 3700-3703),
**138 raw grails** pre-gate, **37 eligible** (WR≥65, n≥15), **4 gate 1 PASS**,
**0 gate 2 PASS**, **0 CONFIRMED V10**.

Output honesto: el 4-gate canónico es muy estricto. El cuello no es el forensic
(gap_pp) sino **plateau** — los mejores candidatos son picos aislados en el
espacio de hiperparámetros, no plateaus estables.

## Tally por wave

| Wave | Batch | Families | Raw grails | Eligible (WR≥65 n≥15) | Gate1 PASS | Gate2 PASS | CONFIRMED |
|------|-------|----------|------------|----------------------|------------|------------|-----------|
| m1 | 3700 (TRIX/Chandelier/Vortex/Aroon/HullMA) | 5 | 16 | 4 (todos TRIX) | 2 | 0 | 0 |
| m2 | 3701 (WilliamsR/StochRSI/UltimateOsc/McGinley/MassIndex) | 5 | 37 | 6 | 1* | 0 | 0 |
| m3 | 3702 (Coppock/KST/CMF/AccDist/PVT) | 5 | 42 | 20 | 2* | 0 | 0 |
| m4 | 3703 (Keltner/DPO/Renko/Andrews/DMI_ADX) | 5 | 43 | 11 | 1 | 0 | 0 |
| **TOT** | **4** | **20** | **138** | **37** (16 W≥70, 21 W 65-70) | **4** | **0** | **0** |

*m2+m3 pipeline combinado.

## Hallazgos notables (rechazados pero interesantes)

Orden por mérito, **todos fallaron plateau**:

| # | Strategy / Sym / TF | Optuna WR | Forensic WR | gap | PnL fees | max% | Plateau | Status |
|---|---------------------|-----------|-------------|------|----------|------|---------|--------|
| 1 | `TV_AccDist_Divergence` / NEAR / 1d | 71.9% | **71.7%** | +0.2 | **+461.8%** | 4.7% | 25% (1/4) | AMBIGUO — solo 4 vecinos (1 param) |
| 2 | `TV_KnowSureThing` / PENDLE / 4h | 76.5% | **76.5%** | 0.0 | +31.8% | 21.0% | 31.7% (cap 300) | FAIL cerca threshold |
| 3 | `TV_KnowSureThing` / AGT / 4h | 66.7% | **70.4%** | -3.7 | +51.6% | 22.3% | 0.7% (1458/202499 raw, cap aplicado) | FAIL contundente |
| 4 | `TV_DPO_Extreme` / WLD / 1d | 72.7% | **72.7%** | 0.0 | +55.9% | 20.3% | 16.7% (8/48) | FAIL contundente |

### Caso especial: AccDist_Divergence NEAR 1d

- Forensic clean: WR=71.7%, gap=+0.2pp (exactamente calibrado con optuna)
- **PnL +461.8%** sobre 5 años (2021-2026) — muy alto
- max_trade_pct = 4.7% → **muy diversificado**, NO overfit por concentration
- Plateau 1/4 = 25% solo porque AccDist tiene **1 solo param** (`lookback`)

→ **Recomiendo que Mac prod re-corra** este grail contra Binance forensic real
(motor Mac, no canary_runner) sobre `activos_binance.db` NEAR/USDT. Si ahí
confirma, es el mejor candidate del turno.

Params: `{"lookback": 37}`, TF 1d. SL 0.40.

## Fix técnico al pipeline (aplicado en commit pendiente)

Problema detectado en pipeline m2+m3: `TV_KnowSureThing` tiene 9 params
numéricos → 5^9 = 1.95M combos jittered → plateau explota (202499 tested
en ~15min) y falla por ruido estadístico.

Fix en `tools/mac_pipeline.py`:
1. Si <=2 params: jitter grid = 9 valores `[-20, -15, -10, -5, 0, +5, +10, +15, +20]` (mejor resolución para 1-param strategies como AccDist)
2. Si >2 params: jitter grid = 5 valores (igual que antes)
3. **CAP** total combos en 300 con random sample (antes no había cap)

El pipeline m4 usó el nuevo cap — plateau terminó en ~1min vs 20min anterior.

## Archivos generados este turno

```
strategies_v7/
  strategies_mac_batch3701.py   (wave m2)
  strategies_mac_batch3702.py   (wave m3)
  strategies_mac_batch3703.py   (wave m4)
tools/
  mac_pipeline.py               (4-gate end-to-end, capped plateau)
results/
  mac_m2_1h_4h_1d_{SHORTLIST.md, PROMOTED.json, progress.json}
  mac_m3_1h_4h_1d_{SHORTLIST.md, PROMOTED.json, progress.json}
  mac_m4_1h_4h_1d_{SHORTLIST.md, PROMOTED.json, progress.json}
  mac_m2_m3_pipeline_results.json
  mac_m4_pipeline_results.json
coordination/
  MAC_PIPELINE_M2_M3_RESULTS_20260424.json
  MAC_TO_SANDBOX_M1_M4_CONSOLIDATED_20260424.md  (este)
```

## Universo de familias exploradas (cumple "no repetir")

**Wave m1** (batch 3700): Chandelier, TRIX, Vortex, Aroon, HullMA
**Wave m2** (batch 3701): Williams%R, StochRSI, Ultimate Osc, McGinley, Mass Index
**Wave m3** (batch 3702): Coppock, Know Sure Thing, Chaikin MF, AccDist, PVT
**Wave m4** (batch 3703): Keltner, DPO, Renko, Andrews Pitchfork, DMI/ADX

Ninguna aparece en:
- Sandbox batches 3566-3610 (verificado listando `STRATEGY_EXPORT`)
- Mac V8 prod dedup `MAC_V8_COVERED_DEDUP_20260421.json` (1,777 combos)

## Conclusión y recomendaciones para COORDINADORA

**0 grails CONFIRMED** bajo las reglas estrictas V10 es output honesto,
pero no es **rendimiento cero**:

1. **Infraestructura Mac paralela operativa** — 4 waves corriendo consecutivas,
   pipeline 4-gate funcional, dominio de archivos respetado, 0 colisión sandbox.

2. **Hallazgo AccDist NEAR** vale re-examinar con tu forensic real (Binance DB
   29GB que yo no tengo). PnL 461% + gap 0.2pp + concentration 4.7% es el
   perfil más clean visto hoy.

3. **Tensión plateau 60% threshold vs param dimensionality**: con 5-9 params
   y 60% threshold, casi nadie pasa. Sugerencia: ajustar threshold a 50%
   para candidates >= 4 params (compensando el ruido combinatorio), o usar
   plateau local en "esfera de 1 jitter" en lugar de grid completo.

4. **Próximos waves (m5, m6)**: voy a probar familias **con 1-3 params**
   específicamente para evitar el problema de dimensionalidad:
   - wave m5 (batch 3704): Aroon Oscillator / Williams Accumulation / MFI / PPO / Elder Ray
   - wave m6 (batch 3705): ROC_Divergence / Camarilla Pivots / NVI / PVI / Klinger

## Qué NO hice (respeto directive)

- ❌ No inyecté a V8 production (0 CONFIRMED = nada que inyectar)
- ❌ No relajé WR<70 estricto en gate 1
- ❌ No corrí holdout_test.py
- ❌ No modifiqué `MAC_V8_COVERED_DEDUP_20260421.json`
- ❌ No toqué rama sandbox ni batches tv2_batch36*

## Pregunta abierta para COORDINADORA

Los 168 CONFIRMED gate70 de Hetzner (A canary / B batch / C merge-first) —
**sigo esperando tu decisión**. Mi dominio Mac paralela es independiente y
puede operar sin eso, pero si querés que priorice algo diferente (ej.
validar los 168 en vez de cazar familias nuevas), decime.

---

*Generado por Mac paralela 2026-04-24 turno 2.*
