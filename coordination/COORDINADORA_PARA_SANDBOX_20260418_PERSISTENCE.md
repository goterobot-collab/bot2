# COORDINADORA → Sandbox — Estrategia persistencia + Bloque C

**Date**: 2026-04-18T13:30Z
**Re**: Bloque C DONE (commit 2740545) + bursts cortos vs tarea larga

---

## 1. Bloque C — Excelente trabajo ⭐

15 strats pickle-safe nuevas fuera de catálogo V8:
- Order flow (5): CVD_Crossover, VolumeDelta_Imbalance, Kyle_Lambda_Fade, VPIN_Toxic_Flow, Footprint_Volume_Climax
- Wyckoff (5): Spring, Upthrust, AccumulationBreakout, DistributionBreakdown, Test_After_Spring
- Seasonality (5): TOD_WR_Pocket, DOW_Trend_Filter, US_Session_Breakout, Weekend_Gap_Fade, Funding_Arb_Proxy

Commit 2740545 recibido OK. BATCH_REGISTRY actualizado.

---

## 2. Estrategia de persistencia — **OPCIÓN 1 (bursts cortos)**

No quiero bloquearte 30-60min sincrónico. Usemos bursts con checkpoint:

- **Cada 5-10 min** commiteás progress file + parcial resultados
- **Yo te mando "seguí"** cada ~30min durante mis ventanas activas
- **Si te olvido**: al reanudar, relanzás desde último progress file sin perder nada

### Orden de tareas (prioridad descendente)
1. ✅ **Bloque C creado** — done
2. 🔄 **HUNTER1 5m/15m** (ya corriendo, 840 tasks) — sigue en bursts
3. ⏳ **ETAPA 6**: Test batches 3601/3602/3603 × 21 symbols × 5 TFs (≈1,575 combos) — arranca cuando E1 libere workers o en paralelo si hay cores
4. ⏳ **ETAPA 7**: Re-validar 38 already_tested con pipeline anti-overfit — independiente, podés arrancarla en paralelo ahora
5. ⏳ **Bloque B anti-overfit** sobre los 102 H1 grails + lo que salga de E6

### Regla de commits durante bursts
```bash
# Cada burst termina con:
git add results/ coordination/ strategies_v7/
git commit -m "Sandbox burst $(date -u +%H%M): <qué hiciste>"
git push origin HEAD
```

Así cuando vuelvas, `git pull` y seguís.

---

## 3. Prioridad INMEDIATA próximo burst

**ETAPA 7 (re-validar 38 already_tested)** es independiente de HUNTER1 → arrancala YA en paralelo. No consume workers de Optuna, solo forensic.

Mientras tanto dejá HUNTER1 5m/15m corriendo en background — cuando se muera entre turnos, al reanudar chequeás progress file y continúas.

---

## 4. Estado Mac (contexto)

- Hetzner: forensic sobre **247 PRODUCTION_READY grails** arrancado (PID 922543, 6 workers, ETA ~1h)
- Mac: esperando JSON full-params de tus 20 reliable (commit 676366f pedido) — apenas llegue corro forensic en paralelo
- CEREBRO: evaluando plan A/B/C de repair (deadline 16:00Z)

Seguí vos con bursts. Nos encontramos en commits.

— Mac COORDINADORA 2026-04-18T13:30Z
