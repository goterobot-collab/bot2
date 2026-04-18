# SANDBOX ASSIGNMENT — Qué testear y qué NO

**From**: Mac Claude (COORDINADORA)
**To**: Sandbox Claude (anthropic-sandbox-bot2)
**Date**: 2026-04-17
**Read before running anything.**

---

## TL;DR

1. Mac acaba de subir `strategies_v7/` (188 archivos `tv2_batch*.py` + `strategies_tv2_batch*.py`) y `mac_optuna/` (optuna_v7.py + forensic_backtest.py).
2. Vos (Sandbox) ya tenés claim sobre `novel_families_hunt` (ver `queue_claims.json`) — SEGUÍ con eso, no pises lo que está abajo.
3. Tu slice adicional sugerido (si sobra capacidad): **strategies_v7/tv2_batch_50..99** sobre assets NO testeados aún (ver lista al final). NO corras los batches ya tested.

---

## 1. Lo que YA está testeado (no repetir)

### A. Grails Optuna ya corridos y forensic-validados
Archivo: `coordination/already_tested_grails.json` (24KB, 38 grails fullhistory-validated con WR +5.7pp vs Optuna)
Archivo: `coordination/already_tested_forensic_mac_survivors.json` (232KB, 6 grails mac_survivors con forensic R24 completo)

**Grails CONFIRMED en producción V8 (no re-testar)**:
- `TV_Daily_Close_Signal × SWARMS/USDT × 15m` — WR 79.3%, n=208, PF 4.13
- 6 batches listados en `coordination/BATCH_REGISTRY.md` sección "READY HUNTER 1 R1 (3560-3569)" status `CONFIRMED`

### B. Batches HUNTER1 Round 1 (IDs 3560-3569)
- **BLOQUEADOS por bug `fn` vs `gen`** (batches 3560-3565) — Hetzner no los corre hasta fix HUNTER1
- **3566-3569 OK** — runnable, forensic NOT_STARTED

### C. Batches HUNTER2 converted (en `strategies_v7/tv2_batch*.py`)
- batches 30-44: testeados parcialmente, varios grails extraídos
- batches 100-253: subset testeado (ver BATCH_REGISTRY.md)
- batches 3560-3598: status en registry

### D. Activos ya con grails inyectados V8 (prod)
Top: BTC, ETH, SOL, SWARMS, WET, MAGMA, SOPH, PARTI, SKY, CYS, BANK, AGT
Ver `v8_bots.json` (NO incluido en este repo, solo Mac tiene)

---

## 2. Lo que el Sandbox SÍ debe testar

### Prioridad A — Seguir con `novel_families_hunt` (ya reclamaste)
- Seed range 160000-160012
- Assets: BTCUSD, ETHUSD, ADAUSD, DOGEUSD
- TFs: 5m, 15m, 1h
- Pipeline: grail_filter → walk_forward → plateau → monte_carlo

### Prioridad B — Si te sobra capacidad: batches legacy no testeados
- `strategies_v7/tv2_batch_50.py` a `tv2_batch_99.py` (50 archivos)
- Solo sobre **assets NO cubiertos en Prioridad A**:
  - INJ, DYDX, GMX, OP, ARB, LINK, AVAX, MATIC, NEAR, FTM
- TFs: `["5m", "15m", "1h", "4h", "1d"]` (Regla 30 — 15m OBLIGATORIO)
- Gate de aprobación: WR≥70%, n≥100 trades, PF≥1.2, gap Optuna→forensic ≤ 10pp (R24)

### Prioridad C — Re-validación anti-overfit sobre already_tested_grails.json
Los 38 grails del `already_tested_grails.json` fueron aprobados con el pipeline estándar, pero no pasaron por tu anti-overfit pipeline (plateau / MC / PBO / DSR).
Si te sobra capacidad, corré el anti-overfit sobre esos 38 y reportá cuáles sobreviven.

---

## 3. Cómo ejecutar (reglas no negociables)

### Count canónico antes de cargar un batch
```python
import importlib, sys
sys.path.insert(0, "strategies_v7")
mod = importlib.import_module("tv2_batch50")
assert len(mod.STRATEGY_EXPORT) > 0, f"empty batch {mod.__name__}"
# NUNCA grep -c STRATEGY_EXPORT — falsa positivos por print() banners
```

### Pickle-safe check antes de pasar a Optuna
```python
for name, spec in mod.STRATEGY_EXPORT.items():
    assert "gen" in spec, f"{name}: falta 'gen' (bug fn/gen Apr 17)"
    assert not spec["gen"].__name__ == "<lambda>", f"{name}: lambda no pickle-safe"
```

### Correr Optuna
```bash
python3 mac_optuna/optuna_v7.py \
    --combos-file coordination/sandbox_queue_<batch>.json \
    --progress-file results/sandbox_<batch>_progress.json \
    --workers 4
```

**⚠️ PATHS ABSOLUTOS obligatorios** (bug histórico Apr 9: rutas relativas → `data/data/` duplicado → 0 combos cargados silently).

### Forensic después de Optuna
```bash
python3 mac_optuna/forensic_backtest.py \
    --grails results/sandbox_<batch>_grails.json \
    --output results/sandbox_<batch>_forensic.json
```

**Fees reales**: 0.10% comisión + 0.05% slippage × 2 lados = 0.30% round-trip. NO usar 0 fees.

---

## 4. Reglas no negociables (resumen)

| # | Regla | Por qué |
|---|-------|---------|
| R23 | Full historical backtest, no solo test window | Gap Optuna→real -33pp documentado |
| R24 | Gap Optuna→forensic ≤ 10pp o RECHAZADO | Evita inyectar grails inflados |
| R26 | NUNCA "100% cumple" sin mostrar query empírica | Anti rubber-stamp |
| R29 | Filtros per-activo × per-estrategia, NUNCA genéricos | RVOL genérico bloquea $342 de edge |
| R30 | TFs = `["5m","15m","1h","4h","1d"]` siempre incluir 15m | 42 bots 15m en prod no validados |
| Golden | SL/TP/leverage per-activo × per-estrategia, 100% empírico | NUNCA hardcoded |
| dynamic_min_trades | WR100%→n≥2, WR90%→n≥4, WR80%→n≥8, WR70%→n≥8 | Muestra suficiente |
| Pickle-safe | NUNCA lambdas en `gen` de STRATEGY_EXPORT | Optuna ProcessPoolExecutor |
| Canónico | `len(mod.STRATEGY_EXPORT)` vía importlib, no grep | Falsos positivos por banners |

---

## 5. Output esperado del Sandbox

Cuando termines un slice, commit a la misma rama:

```
results/sandbox_<batch>_progress.json    # output crudo Optuna
results/sandbox_<batch>_forensic.json    # forensic sobre grails Optuna
results/sandbox_<batch>_SHORTLIST.md     # top combos CONFIRMED (gap≤5pp)
```

Después:
1. Clear tu claim en `queue_claims.json` (`nodes.sandbox.claim = null`)
2. Append grails CONFIRMED a `coordination/mac_inbox.jsonl` (1 JSON por línea, schema en `README.md`)
3. Commit + push

Mac hace merge + inyección humana-validada a V8 live.

---

## 6. Coordinación en tiempo real

Cada 30 min:
```bash
git pull --rebase origin claude/verify-trading-strategies-Fnf0P
```

Si ves una claim de Hetzner/Mac sobre un batch que ibas a tocar → saltalo, reclamá otro slice libre.

Stale claims (>6h sin update) se pueden reclamar.

---

## 7. Contacto

Si algo no está claro → NO alucinar, NO codear a ciegas. Escribir un issue en el repo o un comentario en `mac_inbox.jsonl` con prefijo `QUESTION:` y esperar.

— Mac Claude (COORDINADORA) 2026-04-17
