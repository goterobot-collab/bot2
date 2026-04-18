# SANDBOX — Cola de trabajo 2 días (2026-04-18 → 2026-04-19)

**From**: Mac Claude (COORDINADORA)
**Date**: 2026-04-18
**Objetivo**: 2 días de trabajo paralelo, sin overlap con Hetzner ni Mac.

---

## ❌ NO TOCAR (claims de otros nodos)

| Nodo | Claim | Batches/combos |
|------|-------|----------------|
| Hetzner | `mac_survivors_20260417` | 3,579 combos — RUNNING |
| Hetzner | `lote2_momentum_combo` | 350+ combos — RUNNING |
| Hetzner | `top50_divbounce_vwap` | DONE — 5 grails pendientes review Mac |
| Mac | `orchestration_only` | — (no compute) |
| Sandbox | `novel_families_hunt` | seed 160000-160012 — tu pool previo |

**Todo lo de abajo es sandbox-exclusivo.**

---

## 📅 DÍA 1 (2026-04-18, ~12h compute)

### Bloque A — Terminar HUNTER1 + HUNTER2 (pipeline actual)
- **Estado**: en curso (batch 3566/4 running)
- **Scope**: 119 strats × 21 symbols × 5 TFs = ~12,495 combos
- **Gate**: WR≥70%, n≥dynamic_min, PF≥1.2, R24 gap≤10pp
- **PROMOTED logic**: ≥1 combo pasa → estrategia entera a `sandbox_h1_PROMOTED.json` / `sandbox_h2_PROMOTED.json`
- **Tiempo est.**: 6-8h con 4 workers multiprocessing
- **Output**:
  - `results/sandbox_h1_SHORTLIST.md` + `sandbox_h1_PROMOTED.json`
  - `results/sandbox_h2_SHORTLIST.md` + `sandbox_h2_PROMOTED.json`
  - Append CONFIRMED a `coordination/mac_inbox.jsonl`

### Bloque B — Anti-overfit sobre PROMOTED grails
- **Input**: grails que pasen R24 de Bloque A + tus 125 plateau survivors previos
- **Pipeline**: plateau + Monte Carlo bootstrap + PBO + DSR + walk-forward
- **Gate final**: PBO≤0.2, DSR>0, MC CI 95% no cruza WR=50%, walk-forward últimos 30% no vistos
- **Tiempo est.**: 2-3h
- **Output**: `results/sandbox_antioverfit_day1.md` — cuáles sobreviven los 6 filtros

---

## 📅 DÍA 2 (2026-04-19, ~12h compute)

### Bloque C — Rol B HUNTER (estrategias nuevas)
**Objetivo**: convertir 15 estrategias nuevas a Python pickle-safe, empujar como 3 batches de 5 strats.

**Batches asignados (IDs libres)**:
| Batch | Familia | Fuentes sugeridas |
|-------|---------|-------------------|
| `strategies_tv2_batch3601.py` | **Order flow** (5 strats) | CVD divergence, delta imbalance, VPIN, Kyle lambda, footprint absorption |
| `strategies_tv2_batch3602.py` | **Wyckoff** (5 strats) | Spring, Upthrust, Phase A/B/C/D, Composite Man accumulation, distribution climax |
| `strategies_tv2_batch3603.py` | **Seasonality** (5 strats) | Time-of-day × day-of-week, Asia/EU/US session breakouts, weekend gap fade, monthly expiry bias, funding arb |

**Reglas conversión (SANDBOX_HANDOFF.md Rol B)**:
- Pickle-safe: `def` nombrado, NUNCA lambdas
- Sin look-ahead
- BB ddof=0, RSI/ATR Wilder's RMA, VWAP reset diario
- Header con fuente (autor + URL + referencia) — NUNCA TV_INV_* inventadas
- Canonical check: `len(mod.STRATEGY_EXPORT) == 5`
- Registrar en `coordination/BATCH_REGISTRY.md` con status `NOT_TESTED`

**Tiempo est.**: 3-4h investigación + conversión

### Bloque D — Test de los nuevos batches
- **Scope**: 15 strats × 21 symbols × 5 TFs = ~1,575 combos
- **Mismo pipeline que Bloque A**
- **Tiempo est.**: 1-2h
- **Output**: `results/sandbox_new_batches_SHORTLIST.md` + `_PROMOTED.json`

### Bloque E — Re-validación already_tested (Paso 5 opcional)
- **Input**: 38 grails de `coordination/already_tested_grails.json`
- **Pipeline**: anti-overfit 6 pasos (plateau + MC + PBO + DSR + walk-forward + re-forensic con fees actualizados)
- **Objetivo**: detectar overfit retroactivo en grails ya inyectados a V8
- **Tiempo est.**: 3-4h
- **Output**: `results/sandbox_already_tested_antioverfit.md` — cuáles sobreviven, cuáles caen (Mac usa esto para re-evaluar V8)

---

## 🎯 PRIORIZACIÓN SI SE CUELGA EL TIEMPO

Si te queda corto:
1. **Bloque A** (obligatorio — es lo que ya arrancaste)
2. **Bloque C** (alta prioridad — nuevas strats = potencial edge)
3. **Bloque B** (medium — anti-overfit sobre lo que ya tenés)
4. **Bloque D** (medium — test de las nuevas)
5. **Bloque E** (baja — es re-validación retroactiva, no bloquea producción)

## 📤 COMMITS INTERMEDIOS OBLIGATORIOS

Al terminar cada Bloque:
```bash
git add results/ coordination/mac_inbox.jsonl coordination/BATCH_REGISTRY.md strategies_v7/
git commit -m "Sandbox Día X Bloque Y: <resumen>"
git push origin claude/verify-trading-strategies-Fnf0P
```

Esto permite a Mac/Hetzner ver progreso sin esperar 2 días.

## 📊 MÉTRICAS A REPORTAR AL FINAL DE DÍA 2

```
{
  "total_strats_tested": <N>,
  "total_combos": <N>,
  "grails_found_R24_confirmed": <N>,
  "promoted_strategies": [...],
  "new_batches_created": ["3601", "3602", "3603"],
  "already_tested_survivors": <N>/38,
  "already_tested_failures": [...]
}
```

Commit este resumen como `results/sandbox_2day_final_report.md`.

---

## ❓ DECISIONES QUE ESCALÁS A COORDINADORA (no decidir solo)

- Si un grail tiene WR≥85% pero PBO>0.3 (overfit borderline): pedir decisión
- Si encontrás bug en un batch existente (como el `fn`/`gen` de 3560-3565): flagear, no re-escribir solo
- Si una fuente de Pine Script es ambigua (autor desconocido, sin backtest): skip y anotar, no inventar

---

— Mac Claude COORDINADORA 2026-04-18
