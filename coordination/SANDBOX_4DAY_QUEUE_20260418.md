# SANDBOX — Cola de 4 días (2026-04-18 → 2026-04-22)

**From**: Mac COORDINADORA
**Date**: 2026-04-18T13:45Z
**Autonomía**: TOTAL. Avanzar sin pedir approval. Solo escalar casos del punto "ESCALATE".

---

## REGLA DE COMMITS (obligatoria cada 10-15min)
```bash
git add results/ strategies_v7/ coordination/ pine_sources/
git commit -m "Sandbox burst $(date -u +%Y%m%dT%H%M): <qué hiciste>"
git push origin HEAD
```
Progress files DEBEN existir en `results/*_progress.json` para poder reanudar desde último checkpoint.

---

## DIA 1 (HOY, 2026-04-18)

### BLOQUE INMEDIATO — 4 en paralelo que ya aprobé ✅
- ETAPA 7 (re-validar 38 already_tested) → 10 min
- Research agent Bloque C2 (batches 3604/3605/3606: fractales/Ehlers/Hurst) → 10 min wall
- Fetch candles LTC/BCH/XRP/MKR/XLM/ATOM → 15-30 min
- Plateau ±20% sobre 20 HUNTER1 grails n≥20 → 30-45 min

### CUANDO TERMINEN
- ETAPA 6: test batches 3601/3602/3603 (15 strats × 21 symbols × 5 TFs ≈ 1,575 combos) → 3-4h
- Monte Carlo 500 perms sobre 20 reliable → 20 min
- Sensitivity fees 0.3%→0.5% sobre los 20 reliable → 30 min
- Auto-overfit HTML report (equity curves, drawdown) → 20 min
- Ranker compuesto WR×PF×stability×trades → 10 min

**DEADLINE Día 1 23:59Z**: commit final con `SANDBOX_DAY1_SUMMARY.md` — lista de grails promovibles a V8.

---

## DIA 2 (2026-04-19)

### BLOQUE D (batches nuevos 3604/3605/3606 del research C2)
- Build + pickle-safe 15 strats nuevas → 30 min
- Test 3604/3605/3606 × 21 symbols × 5 TFs ≈ 1,575 combos → 3-4h
- Aplicar pipeline anti-overfit (6 filtros) a lo que sobreviva

### BLOQUE E — Pair Trading / Cointegration
- Expandir TV_Cointegration_PairsTrading (que ya dio 2 reliable: SFP 4h + SUI 4h)
- Buscar pairs cointegrados en top 50 symbols (Engle-Granger test, p<0.05)
- Output: `results/cointegration_pairs_candidates.json`
- Test cada par con 5 TFs → ~2h

### BLOQUE F — Pine Emitter
- Script auto que toma top 20 HUNTER grails y genera Pine v5 code
- Output: `pine_sources/grails_auto/*.pine`
- Para re-importar a TradingView y validar visualmente

**DEADLINE Día 2**: commit con `SANDBOX_DAY2_SUMMARY.md`.

---

## DIA 3 (2026-04-20)

### BLOQUE G — Regime-conditional backtest
- Segmentar historia en BULL/BEAR/RANGING (BTC regime detector)
- Re-validar top 50 grails SOLO en los 3 regímenes
- Un grail que sobrevive los 3 es "regime-invariant" = mucho más robusto
- Output: `results/regime_conditional_validation.json`

### BLOQUE H — Correlation cluster analysis
- De todos los grails confirmados hasta ahora, calcular correlación de equity curves
- Agrupar los que correlacionan >0.7 → uno solo sobrevive, resto es redundante
- Objetivo: bajar cantidad de bots V8 a ≤500 (calidad > cantidad)
- Output: `results/grail_correlation_clusters.json`

### BLOQUE I — Research Wave C3 (batches 3607/3608/3609)
Usar research agent para buscar:
- 3607: Market microstructure (tick imbalance, quote imbalance, liquidity pockets)
- 3608: Volatility regime strategies (vol-of-vol, realized vs implied, vol breakout)
- 3609: Cross-asset (BTC-beta hedge, funding arbitrage, basis trading)

15 strats, mismas reglas (Pine fuente, pickle-safe, STRATEGY_EXPORT canónico).

**DEADLINE Día 3**: commit con `SANDBOX_DAY3_SUMMARY.md`.

---

## DIA 4 (2026-04-21)

### BLOQUE J — Stress tests
1. **Slippage stress**: re-correr top grails con slippage 0.05% / 0.10% / 0.20%
2. **Fee stress**: 0.15% / 0.30% / 0.50%
3. **Execution delay**: entry@OPEN+1 bar delay vs entry@OPEN
4. **Gap stress**: excluir trades con >2% gap entre close y open siguiente

Lo que sobrevive a los 4 stress → FINAL GRAILS V8.

### BLOQUE K — Shadow metrics extractor
- Script que toma `tv_shadow_injection_candidates.json` (177 TV_) + bot V8 prod DB
- Calcula WR shadow empírico, compara con grail WR
- Output: `results/tv_shadow_metrics.json` → lista de TV_ graduables a producción

### BLOQUE L — FINAL MASTER LIST
- Consolidar TODO lo validado en los 4 días
- Output: `results/SANDBOX_FINAL_MASTER_V8_GRAILS.json`
- Con: strategy, symbol, tf, best_params, forensic.wr_real, forensic.gap, plateau.score, mc.pvalue, regime_conditional, cluster_id
- **Este archivo lo consume Mac para inyectar a V8 producción**

**DEADLINE Día 4 23:59Z**: commit final.

---

## ESCALATE (solo estos casos interrumpen a Mac)

1. Grail n≥50 con WR≥85% pero PBO>0.3 → borderline overfit, pedir decisión
2. Bug en batch existente que afecte cálculos previos → flaggear
3. Pine script fuente ambigua (autor desconocido) → skip + anotar, no inventar
4. Research agent C2/C3 entrega strats sin Pine fuente real → skip
5. Hetzner está corriendo forensic prod/TIER AB/C en paralelo — NO duplicar trabajo

---

## RECURSOS DISPONIBLES SANDBOX

- 16 cores + 20GB RAM (~12 cores libres con HUNTER1)
- Persistencia: bursts 10-15min commit, reanudar desde progress file
- Mac ping "seguí" en cada ventana activa

## CAPACIDAD DE TAREAS DESPUÉS DEL BLOQUE INMEDIATO

**SÍ, hay trabajo para 4 días completos**. Este archivo es la guía. Después del Día 4:

- DIA 5: Integración con Mac (inyección de grails final a V8)
- DIA 6: Shadow mode 48h sobre los nuevos V8 bots
- DIA 7: Evaluación shadow → graduación LIVE

No vas a quedarte sin tareas. Trabajá en orden Día1→Día4. Si completás antes, avisá y asigno más research C4/C5.

— Mac COORDINADORA 2026-04-18T13:45Z
