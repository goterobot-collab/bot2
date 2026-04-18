# mac_optuna/ — Pipeline completo Optuna + Forensic

Exportado desde `/Users/sabrina/CLAUDE CODE/Estrategias/` el 2026-04-17.

## Flujo canónico (end-to-end)

```
strategies_v7/tv2_batch*.py  →  optuna_v7.py          →  grails.json
                                (50 trials, CPCV, EV gate)
                                       ↓
                             scripts/forensic_backtest.py
                             (full history, fees 0.30%, entry@OPEN)
                                       ↓
                             scripts/full_historical_backtest.py (R23)
                                       ↓
                             scripts/v8_gates.py (R24 gap≤10pp)
                                       ↓
                             scripts/inject_forensic_approved.py
                             (→ v8_bots.json, solo Mac)
```

## Archivos principales

### Optimizers (Optuna variants)
| Archivo | Uso |
|---------|-----|
| **`optuna_v7.py`** ⭐ | **Default actual** — Numba, CPCV, EV gate, R:R dinámico, TFs R30 |
| `optuna_v7_tv.py` | Variante para estrategias TV_ |
| `optuna_v7_tv2.py` | Variante para batches tv2_* (loader específico) |
| `optuna_v7_wr60.py` | Gate WR≥60% (en vez del default 70%) |
| `optuna_v5.py` | Legacy V5 (referencia) |
| `optuna_v4.py` | Legacy V4 |
| `optuna_full*.py` | Variantes full-history |
| `optuna_smart.py`, `optuna_turbo.py` | Experimentales |
| `optuna_worker*.py` | Worker procesos paralelo |

### Spaces + factories
- `batch_search_spaces.py` — espacios de búsqueda SL/TP/lev per-estrategia
- `strategy_factory.py` — genera strategies canónicas (~144K, grande)
- `factory_runner.py` — runner
- `evaluator_v2.py` — evaluación grail

### Forensic + backtest
- **`scripts/forensic_backtest.py`** ⭐ — V2 con LONG+SHORT, fees 0.30%, funding
- **`scripts/full_historical_backtest.py`** ⭐ — R23 enforcement (full history)
- `scripts/full_backtest_v2.py`, `scripts/full_backtest_v3.py` — variantes
- `scripts/optuna_fullhistory.py` — Optuna con warm-start full history
- `grail_dual_validator.py` — validador dual (Agente A réplica + B ciego)

### Pipeline orchestrators
- **`scripts/pipeline_optuna_forense.py`** — Optuna → forensic end-to-end
- **`scripts/hetzner_pipeline_v8.py`** — pipeline V8 producción Hetzner
- `scripts/hetzner_pipeline.py` — V7 legacy
- `pipeline_fast.py`, `pipeline_backtest.py` — variantes

### Gates + injection (solo Mac)
- **`scripts/v8_gates.py`** — R24 enforcement (gap≤10pp)
- `scripts/inject_forensic_approved.py` — inyecta a `v8_bots.json`
- `scripts/inject_v8_survivors_to_production.py`
- `scripts/regla_max_enforcer.py` — Regla de Oro runtime

### Filters + overrides (Regla 29)
- `scripts/generate_all_overrides.py` — genera rvol_overrides + hour_overrides
- `scripts/extract_wr_per_hour.py` — WR por hora UTC per-combo

### Loaders + converters
- `scripts/optuna_tv2_loader.py` — carga tv2_batch*.py en Optuna
- `scripts/convert_tv2_to_python.py` — conversor
- `scripts/validate_tv_python.py` — validación Pine↔Python

### Screening + coverage
- `scripts/classify_grails_by_coverage.py`
- `scripts/screen_grail_fast.py`
- `scripts/pilot_screen_500.py`
- `scripts/parallel_screen_all.py`

### Analysis
- `scripts/correlation_analysis.py`
- `scripts/empirical_compare_approaches.py`
- `scripts/extract_empirical_gap_stats.py`
- `scripts/audit_v8_killed_bots.py`
- `analysis_no_sl_study.py`

### Live-adjacent (referencia, NO correr en sandbox)
- `bot_production.py` — interface producción
- `risk_manager.py` — circuit breaker
- `portfolio_simulator.py`
- `extract_trades.py`

## Uso desde sandbox

```bash
# 1. Correr Optuna sobre batches tv2_* (strategies_v7/)
python3 mac_optuna/optuna_v7.py \
    --combos-file coordination/sandbox_queue_<batch>.json \
    --progress-file results/sandbox_<batch>_progress.json \
    --workers 4

# 2. Forensic sobre grails output de Optuna
python3 mac_optuna/scripts/forensic_backtest.py \
    --grails results/sandbox_<batch>_grails.json \
    --output results/sandbox_<batch>_forensic.json

# 3. Aplicar gates R24
python3 mac_optuna/scripts/v8_gates.py \
    --forensic results/sandbox_<batch>_forensic.json \
    --output results/sandbox_<batch>_approved.json
```

## Dependencias

- Python 3.11+
- `optuna`, `pandas`, `numpy`, `numba` (crítico — 100-150× speedup)
- `sqlite3` (stdlib)
- Candles DB — **NO incluida**. Opciones en sandbox:
  - `ccxt` contra Binance API con cache local
  - Subset via Git LFS (top 50 symbols × 2 años ≈ 2GB)

## Reglas heredadas (NO negociable)

- **Golden Rule**: SL/TP/lev per-activo × per-estrategia, 100% empírico
- **R23**: Full historical backtest ANTES de declarar grail
- **R24**: Gap Optuna→Forensic ≤ 10pp o RECHAZADO
- **R26**: NUNCA "100% cumple" sin query empírica (mostrar comando+output)
- **R28**: Operación >30s → reportar, no esperar silencioso
- **R29**: Filtros per-activo×per-estrategia, NUNCA genéricos
- **R30**: TFs canónicos = `["5m","15m","1h","4h","1d"]` — incluir 15m SIEMPRE
- **Pickle-safe**: NUNCA lambdas en `gen` de STRATEGY_EXPORT
- **Canonical count**: `len(mod.STRATEGY_EXPORT)` vía importlib, NO `grep -c`
- **Fees reales**: 0.10% comisión + 0.05% slippage = 0.30% round-trip
- **Paths absolutos** en `--combos-file` y `--progress-file` (bug Apr 9 con paths relativos)
- **dynamic_min_trades**: WR100%→n≥2, WR90%→n≥4, WR80%→n≥8, WR70%→n≥8
