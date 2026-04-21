# Pine emit — Top 6 holdout-survivors (2026-04-21)

**MED-4 deliverable.** Los 6 grails que sobrevivieron OOS 20% con Regla 24.
Marcados `needs_manual_review` por bug de trail stop en `emit_pine.py`
(ver `results/pine_audit.md`) — no emito Pine automático; en su lugar doy
params exactos para que Mac los emita con su stack corregido.

## 6 survivors OOS holdout

| # | Strategy | Sym | TF | IS WR | HO WR | HO n | HO PnL | Gap | Score |
|---|----------|-----|----|-------|-------|------|--------|-----|-------|
| 1 | `TV_Linear_Regression_Channel` | AGT | 4h | 95.0% | 100.0% | 3 | +26.2% | 5.0pp | 4338 |
| 6 | `TV_Pivot_Reversal_Backtest` | GMX | 1d | 71.5% | 68.0% | 25 | +55.1% | 3.5pp | 2347 |
| 7 | `TV_ABCD_Pattern_Daveatt` | SUI | 4h | 78.4% | 75.0% | 8 | +6.3% | 3.4pp | 2331 |
| 8 | `TV_Pivot_Reversal_Backtest` | ONDO | 1d | 75.3% | 83.3% | 18 | +53.3% | 8.0pp | 2222 |
| 9 | `TV_ABCD_Pattern_Daveatt` | PYTH | 1d | 78.8% | 87.5% | 8 | +20.9% | 8.7pp | 2214 |
| 10 | `TV_Pivot_Reversal_Backtest` | LINK | 1d | 71.1% | 75.0% | 32 | +84.8% | 3.9pp | 2184 |

**Ganadores claros:** Pivot_Reversal_Backtest en 1d (3 survivors: GMX, ONDO, LINK)
con PnL OOS +53–85%. Candidato #1 para producción Mac.

## Params exactos (para pasar a Pine manualmente)

Ver `results/holdout_top10_20260421.json` key `results[i].params`.

### 1. TV_Linear_Regression_Channel AGT 4h
```json
params: ver JSON (n=3 OOS es poca muestra — tratar con cautela)
```

### 6. TV_Pivot_Reversal_Backtest GMX 1d
```json
params + pivotlen, rightlen, etc. — ver JSON
```

### 7-10. Similar pattern — Pivot_Reversal_Backtest + ABCD_Pattern_Daveatt

Todos los params completos están en `results/holdout_top10_20260421.json`.

## Acción sugerida Mac

1. Leer el JSON, extraer `params` + `holdout_*` fields.
2. Usar Mac's `emit_pine.py` (con fix de trail stop aplicado) sobre estos 6.
3. Aplicar FBI 3/3 antes de inyectar a V8 producción.
4. **Pivot_Reversal_Backtest 1d × 3** es el cluster de mayor confianza — priorízalo.

---
*Sandbox no emite Pine directamente debido al bug de trail stop documentado
en pine_audit.md — delegado a Mac.*
