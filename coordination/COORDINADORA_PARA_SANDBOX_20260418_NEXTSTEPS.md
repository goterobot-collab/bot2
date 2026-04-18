# COORDINADORA → Sandbox — Next steps

**Date**: 2026-04-18T12:00Z
**From**: Mac COORDINADORA
**Re**: HUNTER1 1200/2100 done + 102 grails (57% progreso)

---

## 1. SÍ, commitea la lista completa

Commitea al repo:
- `results/sandbox_h1_grails_full_102.json` (lista completa ordenada por PF, con n_trades y WR de cada uno)
- `results/sandbox_h1_grails_reliable_n20plus.json` (subset confiable n≥20 — por ahora 4)

Así Mac COORDINADORA puede auditar sin esperar a que termines.

## 2. CONTINUÁ con 15m + 5m (restante 43% = 900 tasks)

- Esos TFs son críticos (R30 — 15m obligatorio, 5m captura micro-tendencias en altcoins volátiles).
- Esperable: más grails pero también más ruido (n<10 en 5m va a ser brutal).
- Aplicá el mismo dynamic_min_trades (WR100%→n≥2, WR90%→n≥4, WR80%→n≥8, WR70%→n≥8).

## 3. Observación sobre los "102 grails"

Coincidimos: top 57 con n<10 son **ruido estadístico** — los filtrará Monte Carlo en Bloque B.

**Los 4 confiables que ya validaron (n≥20)** son los únicos que Mac mira hoy:
1. TV_Cointegration_PairsTrading × SFP 4h (n=493, WR 72%, PF 1.31) ⭐
2. TV_VolumeProfile_Reversion × SFP 1d (n=234, WR 82%, PF 1.36) ⭐
3. TV_Cointegration_PairsTrading × SUI 4h (n=157, WR 74.5%)
4. TV_VolumeProfile_Reversion × TIA 1d (n=111, WR 77.5%)

**SFP aparece 2 veces** → ese símbolo tiene edge real para estas familias. Marcar SFP como **priority asset** para universo expandido post-promotion.

## 4. PLAN AL TERMINAR HUNTER1 (orden automático, sin preguntar)

Cuando HUNTER1 llegue a 2100/2100, **arrancá inmediatamente** con la cola del `SANDBOX_2DAY_QUEUE.md`:

### Paso A (siguiente inmediato) — HUNTER2 (99 strats)
- Batches 3594 (50 strats) + 3598 (49 strats) = 99 HUNTER2 strats × 21 symbols × 5 TFs
- Expectativa: ~10,395 combos. Con 4 workers ~3-4h.
- Output: `results/sandbox_h2_SHORTLIST.md` + `sandbox_h2_PROMOTED.json`

### Paso B — Anti-overfit sobre los 102 H1 grails + los que salgan de H2
- Pipeline 6 filtros: plateau + Monte Carlo bootstrap (1000 iter) + PBO + DSR + walk-forward 70/30 + re-forensic con fees 0.30%
- Gate final: PBO≤0.2, DSR>0, MC CI95 no cruza WR=50%, WF últimos 30% positivo
- **Expectativa realista**: 4 confiables + quizás 5-10 que sobrevivan del top 57 = ~10-15 estrategias saldrán del pipeline completo
- Output: `results/sandbox_antioverfit_day1.md`

### Paso C — Rol Hunter (3 batches nuevos)
Si queda tiempo del Día 1, empezá con:
- `strategies_tv2_batch3601.py` (Order flow — CVD divergence, delta imbalance, VPIN, Kyle lambda, footprint absorption)
- `strategies_tv2_batch3602.py` (Wyckoff — Spring, Upthrust, Phase A/B/C/D, Composite Man, distribution climax)
- `strategies_tv2_batch3603.py` (Seasonality — ToD×DoW, Asia/EU/US session breakouts, weekend gap fade, monthly expiry, funding arb)

Reglas conversión (Rol B de SANDBOX_HANDOFF.md):
- Pickle-safe: `def` nombrado, NUNCA lambdas
- STRATEGY_EXPORT canónico `{name: {"gen": fn, "space": fn}}` (NO `fn`/`gen_long`+`gen_short`)
- Sin look-ahead, BB ddof=0, RSI/ATR Wilder's RMA, VWAP daily reset
- Header con fuente real (autor + URL + ref) — NUNCA TV_INV_* inventadas
- Verificar con `importlib.import_module` + `len(mod.STRATEGY_EXPORT) == 5` (NO grep — ver memoria R26)

### Paso D (Día 2) — Testear los 3 batches nuevos
- 15 strats × 21 symbols × 5 TFs = ~1,575 combos
- Mismo pipeline H1/H2

### Paso E (Día 2, prioridad baja) — Re-validación already_tested
- 38 grails en `coordination/already_tested_grails.json`
- Pipeline anti-overfit 6 pasos
- Output: `results/sandbox_already_tested_antioverfit.md`

---

## 5. COMMITS OBLIGATORIOS

Al terminar cada paso:
```bash
git add results/ coordination/mac_inbox.jsonl coordination/BATCH_REGISTRY.md strategies_v7/
git commit -m "Sandbox Paso <X>: <resumen> (<N> grails, <M> survivors anti-overfit)"
git push origin HEAD
```

Así Mac ve progreso sin esperar 48h.

## 6. Escalás a COORDINADORA si

- Un grail n≥50 tiene WR≥85% pero PBO>0.3 → pedir decisión (borderline overfit)
- Encontrás bug en un batch existente → flagear, no re-escribir solo
- Pine script fuente es ambigua (autor desconocido, sin backtest) → skip + anotar, no inventar

---

**Resumen de la orden de Sabrina** (2026-04-18T12:00): *"empezá con 5 y 15m, y armá el plan para cuando termines — arrancar con las otras tareas pendientes"*

Mac COORDINADORA confirma: autonomía total para ejecutar A→B→C→D→E en orden. No pidas permiso entre pasos. Solo escalar los casos del punto 6.

— Mac COORDINADORA 2026-04-18T12:00Z
