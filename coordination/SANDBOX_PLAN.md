# SANDBOX EXECUTION PLAN — Cola actualizada con prioridad 5m/15m

**Generado**: 2026-04-18T02:15Z
**Estado actual**: HUNTER1 en curso (57% — 1d/4h/1h hechos, 102 grails hasta ahora; 15m/5m pendiente)

---

## Prioridad revisada (user request: "empeza con 5 y 15m")

Los TFs chicos son donde vive el V8 del usuario. Re-ordeno para que 5m/15m se completen PRIMERO en cada wave.

## Cola de ejecución

### ETAPA 1 — HUNTER1 5m/15m (crítico, ya corriendo)
- Proceso actual está haciendo 15m→5m ahora (1d/4h/1h ya completos, 102 grails)
- NO lo mato: va a terminar naturalmente 15m y después 5m
- ETA ~60 min
- Output: `results/sandbox_h1_3566_3569_SHORTLIST.md`

### ETAPA 2 — HUNTER2 solo 5m + 15m (cuando HUNTER1 libere workers)
- 99 strats (batches 3594 + 3598) × 21 symbols × 2 TFs = **4,158 tasks × 20 trials = 83k backtests**
- Launch: `python3 tools/hunter_runner.py --wave h2 --tfs 5m 15m`
- ETA ~3-4 hrs con 4 workers
- Output: `results/sandbox_h2_5m_15m_SHORTLIST.md`

### ETAPA 3 — HUNTER2 1d/4h/1h (mop-up, opcional si sobra tiempo)
- 99 strats × 21 × 3 TFs = 6,237 tasks
- `python3 tools/hunter_runner.py --wave h2 --tfs 1h 4h 1d`
- ETA ~1-2 hrs (TFs más chicos corren rápido)
- Output: `results/sandbox_h2_1d_4h_1h_SHORTLIST.md`

### ETAPA 4 — Bloque B anti-overfit sobre PROMOTED
- Input: union de PROMOTED.json de ambas waves + los 125 plateau survivors previos
- Pipeline: plateau ±20% → MC 500 perms → PBO → DSR → walk-forward locked OOS 20%
- Filtro n>=20 (descarta los WR=100% con n=2-3 que son ruido)
- Output: `results/sandbox_antioverfit_report.md`
- ETA ~1-2 hrs

### ETAPA 5 — Bloque C HUNTER (3 batches nuevos)
- Prioridad alta per cola 2-día
- batch3601 order-flow: CVD, delta, VPIN, Kyle lambda, footprint (5 strats)
- batch3602 Wyckoff: Spring, Upthrust, phases (5 strats)
- batch3603 seasonality: time-of-day × day-of-week, sessions, weekend gaps, funding (5 strats)
- Output: `strategies_v7/strategies_tv2_batch3601.py`, `3602.py`, `3603.py` + `BATCH_REGISTRY.md` entry
- ETA ~3-4 hrs (investigación + implementación + verificación canónica)

### ETAPA 6 — Bloque D test 3601/3602/3603
- Mismo pipeline que ETAPA 1-3, solo que 5m+15m primero
- `python3 tools/hunter_runner.py --wave h3 --tfs 5m 15m` (nueva wave)
- ETA ~1-2 hrs

### ETAPA 7 — Bloque E re-validación 38 already_tested
- Plateau + MC sobre los 38 grails de Mac (para re-rankearlos)
- Requiere reconstruir trades con sus params exactos sobre mis candles
- Output: `results/sandbox_already_tested_antioverfit.md`
- ETA ~30 min

---

## Commits obligatorios entre etapas

Cada etapa termina con:
```bash
git add results/ coordination/mac_inbox.jsonl strategies_v7/
git commit -m "Sandbox etapa X: <nombre>"
git push origin claude/verify-trading-strategies-Fnf0P
```

## Reporte final

`results/sandbox_2day_final_report.md` con:
- Grails por etapa × TF × strategy
- Anti-overfit survivors (pipeline completo)
- Candidatos finales para V8 producción con PROMOTED.json
- Estrategias nuevas descubiertas + código en strategies_v7/

## Si algo falla

- `BrokenProcessPool` → reducir workers=3, seguir
- Combo >30s → skip + log (R28)
- Progress file corrupto → hunter tiene resume; borrar y relanzar
- OOM en 5m (1.37M bars BTC) → reducir cache a 3 (sym,tf) por worker

---

## TF coverage resumen post-ETAPA-1

HUNTER1 (20 strats) × 103 combos × 5 TFs = 10,300 tests. 102 grails ya en 1d/4h/1h. 15m/5m pending va a sumar más.

## Dependencias entre etapas

- ETAPA 2 NO depende de ETAPA 1 (diferente wave, shortlist separada)
- ETAPA 3 NO depende de ETAPA 2 (diferente TF subset)
- ETAPA 4 DEPENDE de ETAPA 1+2+3 (necesita todos los PROMOTED)
- ETAPA 5 puede correr en paralelo con ETAPA 4 (trabajo diferente)
- ETAPA 6 DEPENDE de ETAPA 5 (testea los nuevos batches)
- ETAPA 7 puede correr en cualquier momento
