# BATCH_REGISTRY — HUNTER Pipeline Tracking

**Owner**: COORDINADORA (escritura) · todas las sesiones (lectura)
**Canonical path**: `/Users/sabrina/CLAUDE CODE/BOT V7/BATCH_REGISTRY.md`
**Creado**: 2026-04-17 11:00 UTC
**Última actualización**: 2026-04-17 11:00 UTC

---

## 📐 Convenciones (diseñadas con CEREBRO para trazabilidad end-to-end)

### `forensic_status` — estados granulares

| Estado | Significado | Next action |
|--------|-------------|-------------|
| `NOT_STARTED` | Batch existe, Optuna no corrió | Esperar auth Sabrina |
| `OPTUNA_RUNNING` | Corriendo en Hetzner | Vigilar progress file |
| `GRAILS_OUT` | Optuna dio grails, listo para forensic | Handoff a CEREBRO |
| `FORENSIC_RUNNING` | CEREBRO corriendo `forensic_backtest.py` | Esperar reporte |
| `CONFIRMED` | Gap Optuna→Forensic ≤ 5pp (Regla 24) | Producción directa |
| `WARNING` | Gap 5-10pp (Regla 24) | Shadow mode obligatorio |
| `INFLATED` | Gap > 10pp (Regla 24) | **RECHAZADO** |
| `REJECTED` | PnL neto ≤ 0 o no pasa gates base | **RECHAZADO** |

### `origin`

| Valor | Significado | Forensic extra |
|-------|-------------|----------------|
| `H1_R1` | HUNTER1 Round 1 (GitHub/Kaggle/ArXiv/Freqtrade/Jesse) | Standard |
| `H1_R3` | HUNTER1 Round 3 (volatility/ML/stat-arb advanced) | Standard |
| `H2_CONVERTED` | HUNTER2 Pine → HUNTER1 Python | **EXTRA-CAREFUL** — Pine→Python bugs (ATR param, BB ddof, RSI Wilder, VWAP daily reset) |

### `tfs_r30` (Regla 30 — aprobada Sabrina 2026-04-15)

- `R30_OK` = Optuna corre con `TFS = ["5m","15m","1h","4h","1d"]` (incluye 15m)
- `R30_FAIL` = falta 15m — **bloquea launch hasta corrección**

**Estado actual verificado empíricamente (2026-04-17 11:10 UTC)**:
```
$ grep -n "^USE_TFS" Estrategias/optuna_v7.py
158:USE_TFS = ['5m', '15m', '1h', '4h', '1d']
```
→ **R30_OK** ✅ — Optuna ya está compliant, no bloquea launch.

### `auth_sabrina`

- ⏳ = pendiente
- ✅ = autorizado explícitamente por Sabrina
- ❌ = denegado

---

## 🟢 READY para Optuna — HUNTER 1 Round 1 (3560-3569)

**50 estrategias verificadas empíricamente** (2026-04-17, Regla 26 — re-verificado con `len(mod.STRATEGY_EXPORT)`):
- ✅ Archivos existen en `/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/`
- ✅ STRATEGY_EXPORT count = 50 (via `len(mod.STRATEGY_EXPORT)` import directo — canónico)
- ✅ 0 lambdas en campo `gen` (pickle-safe confirmado — Regla Apr 9)
- ✅ 10/10 batches con fuente citada en header (sin TV_INV_* inventadas — Regla Apr 14)

> **CORRECCIÓN 2026-04-17 canary Paso 1.5**: Conteo inicial "55" por grep fue falso positivo (banner `print("✅ Batch X — 5 strategies")` en el módulo contaba como match adicional). El conteo canónico vía `importlib` confirma **5 strategies per batch × 10 batches = 50**.

| Batch | N | Origin | Source (short) | TFs R30 | Pickle-safe | Loader Compat | Forensic | Auth |
|-------|---|--------|---------------|---------|-------------|---------------|----------|------|
| 3560 | 5 | H1_R1 | Freqtrade community (high-WR) | R30_OK ✅ | ✅ | ❌ fn/gen bug | **BLOCKED_INCOMPAT** | ⏳ |
| 3561 | 5 | H1_R1 | QuantConnect + Freqtrade + Jesse | R30_OK ✅ | ✅ | ❌ fn/gen bug | **BLOCKED_INCOMPAT** | ⏳ |
| 3562 | 5 | H1_R1 | GitHub (CombinedBinHAndCluc, berlinguyinca) | R30_OK ✅ | ✅ | ❌ fn/gen bug | **BLOCKED_INCOMPAT** | ⏳ |
| 3563 | 5 | H1_R1 | Research agents + Jesse DualThrust | R30_OK ✅ | ✅ | ❌ fn/gen bug | **BLOCKED_INCOMPAT** | ⏳ |
| 3564 | 5 | H1_R1 | OBV divergence + Wilder ADX+DI | R30_OK ✅ | ✅ | ❌ fn/gen bug | **BLOCKED_INCOMPAT** | ⏳ |
| 3565 | 5 | H1_R1 | TradingView established | R30_OK ✅ | ✅ | ❌ fn/gen bug | **BLOCKED_INCOMPAT** | ⏳ |
| 3566 | 5 | H1_R1 | Medium Systematic Crypto + Bitsgap 2025 | R30_OK ✅ | ✅ | ✅ `gen` | NOT_STARTED | ⏳ |
| 3567 | 5 | H1_R1 | edgetrader + Johansen mean-reversion | R30_OK ✅ | ✅ | ✅ `gen` | NOT_STARTED | ⏳ |
| 3568 | 5 | H1_R1 | TradingView premium + Investopedia | R30_OK ✅ | ✅ | ✅ `gen` | NOT_STARTED | ⏳ |
| 3569 | 5 | H1_R1 | Smart Money Concepts + YouTube educators | R30_OK ✅ | ✅ | ✅ `gen` | NOT_STARTED | ⏳ |
| **TOTAL** | **50** | | | **10/10 ✅** | **10/10 ✅** | **4/10 ✅** (20 strats runnable) | **6/10 BLOCKED** | **0/10 ⏳** |

### 🔴 BUG 2026-04-17 — batches 3560-3565 usan `fn` incompatible con Optuna V7

**Causa raíz**: `optuna_v7.py:_load_strategies_in_worker()` solo reconoce `gen` o `gen_long`+`gen_short`; batches 3560-3565 usan `fn` → silent drop (30 strats no ejecutables).
**Detectado**: Canary R30 Paso 2 — log reportó 50 combos en vez de 125 (5×5×5 esperado).
**Acciones**:
1. BUZON enviado: `COORDINADORA_PARA_HUNTER1_20260417_BUG_FN_VS_GEN.md` (fix opción A recomendada: rename `fn` → `gen`)
2. Agent Bus BROADCAST msg#583 → todas las sesiones (HIGH priority)
3. Canary R30 **BLOCKED** hasta fix HUNTER1
4. Gate pre-Optuna extendido — verificación canónica `all 'gen' in STRATEGY_EXPORT.values()` ahora obligatoria (nuevo item #10 del checklist)

**Estrategia de unblock**:
- ⏳ Esperar respuesta HUNTER1 en `HUNTER1_PARA_COORDINADORA_20260417_FIX_FN_VS_GEN.md`
- Al recibir fix confirmado: re-scan, re-validar `loader_compat` → `✅ gen`, cambiar forensic_status a `NOT_STARTED`, relanzar Canary R30 con 5 strats NUEVAS (no viejas de 2450/2773)

### 📋 Checklist pre-Optuna (R24 + R30 + pickle-safe)

- [x] Archivos Python existen (10/10)
- [x] STRATEGY_EXPORT válido (55/55)
- [x] Pickle-safe: sin lambdas en `gen` (10/10 — verificado con grep)
- [x] Fuente citada en header (10/10 — verificado con head -20)
- [x] `optuna_v7.py` tiene `USE_TFS = ['5m','15m','1h','4h','1d']` (línea 158) — **R30 compliant ✅**
- [ ] Paths absolutos en comandos Optuna (no `data/data/` — error histórico Apr 9) — verificar al lanzar
- [ ] Progress file único per-batch (no reutilizar nombres) — verificar al lanzar
- [ ] Hetzner workers online (`ssh hetzner 'ps aux | grep optuna'`) — verificar al lanzar
- [ ] **Autorización explícita Sabrina** ("lanza Optuna") ⏳ **ÚNICO BLOQUEO RESTANTE**

---

## 🟡 IN PROGRESS — HUNTER 1 Round 3 (3570-3574)

**Target**: 25 estrategias en 5 batches (ETA ~4h/batch ≈ 20h total)
**Tema declarado** (HUNTER1 broadcast): volatility + ML + stat-arb

| Batch | Target N | Origin | Tema | Status | ETA |
|-------|----------|--------|------|--------|-----|
| 3570 | 5 | H1_R3 | por definir HUNTER1 | NO_FILE_YET | ~4h |
| 3571 | 5 | H1_R3 | por definir HUNTER1 | NO_FILE_YET | ~8h |
| 3572 | 5 | H1_R3 | por definir HUNTER1 | NO_FILE_YET | ~12h |
| 3573 | 5 | H1_R3 | por definir HUNTER1 | NO_FILE_YET | ~16h |
| 3574 | 5 | H1_R3 | por definir HUNTER1 | NO_FILE_YET | ~20h |

**Nota**: COORDINADORA actualiza esta tabla al recibir BUZON `HUNTER1_PARA_COORDINADORA_*_BATCH357X.md` confirmando batch listo.

---

## 🔵 RESERVED — HUNTER 2 Pine Converted (3580+)

**Workflow**: HUNTER2 descubre Pine → HUNTER1 convierte a Python → aquí

| Batch | Pine source (HUNTER2) | Converter (HUNTER1) | Status |
|-------|----------------------|---------------------|--------|
| 3580 | pine_batch_1 (HUNTER2 en progreso ETA ~6h) | — | WAITING_H2 |

**Regla CEREBRO — forensic extra-careful para H2_CONVERTED**:
- Validar ATR parametrización (Wilder's RMA vs SMA)
- Validar BB usa `ddof=0`
- Validar RSI usa `ewm(alpha=1/n, adjust=False)` (Wilder's RMA)
- Validar VWAP resetea daily por sesión
- Gap tolerance más estricto (CONFIRMED ≤ 3pp para H2_CONVERTED)

---

## 🌙 Overnight Hetzner runs — forensic completos

### mac_survivors_20260417 (overnight Apr 16→17)

**Origin**: Hetzner watchdog v2 (Apr 17 11:54 → 14:30). 6 grails candidatos generados.
**Forensic evaluation**: CEREBRO (2026-04-17 ~14:45, reporte en `CEREBRO_PARA_COORDINADORA_20260417_FORENSIC_MAC_SURVIVORS.md`)

| Grail | Opt WR | Real WR | Gap | n_trades | PF | Verdict | R24 |
|-------|--------|---------|-----|----------|-----|---------|-----|
| TV_Daily_Close_Signal × SWARMS 15m | 80.6% | 79.3% | -1.3pp ✅ | **208** ✅ | 4.13 | **APPROVED** | **CONFIRMED** |
| TV_ROC_Extreme × PARTI 4h | 76.3% | 73.0% | -3.3pp ✅ | 37 ⚠️ | 1.21 | APPROVED | WARN_n |
| TV_Trend_Exhaustion × SKY 1h | 78.3% | 72.2% | -6.1pp ⚠️ | 36 ⚠️ | 1.89 | APPROVED | WARN_gap+n |
| TV_Parallel_Channel × CYS 15m | 74.3% | 64.4% | -9.9pp ⚠️ | 59 | 2.57 | **BLOCKED** | FAIL_gate_wr |
| TV_Crab_Pattern × BANK 4h | 70.6% | 63.0% | -7.6pp ⚠️ | 46 | 1.13 | **BLOCKED** | FAIL_gate_wr+pf |
| TV_VWAP_MR_Validated × BANK 1h | 93.8% | — | — | 16 ❌ | — | **REJECT_pre** | n<100 |

**forensic_status**: `COMPLETE` (1 CONFIRMED, 2 WARN, 3 REJECTED)
**Gap promedio 5 testeados**: -5.6pp (valida pipeline Optuna V7 reducido overfitting vs v6 donde veíamos +33pp AGT)

**Acciones aprobadas COORDINADORA (2026-04-17 ~15:00)**:
1. **SWARMS 15m** → V8 shadow mode 20 trades (Regla 10) — ESTRATEGIAS inyecta + SIGNAL_AUDITOR precheck RVOL override R29
2. **PARTI 4h + SKY 1h** → shadow-only hasta n=100
3. **CYS 15m + BANK Crab 4h + BANK VWAP 1h** → REJECT definitivo
4. **R30 validado empíricamente** — SWARMS 15m = 208 trades reales = primera evidencia forensic end-to-end de 15m funcionando post-R30

**Ratio productividad observado**: 1 CONFIRMED / 6 candidatos = **17% Optuna→R24-full**. Proyección: 13,800 combos queue → ~50-100 grails Optuna → ~8-17 CONFIRMED.

---

## 📊 Historia de estados

| Fecha UTC | Batch | De | A | Motivo | Quién |
|-----------|-------|----|----|--------|-------|
| 2026-04-17 11:00 | 3560-3569 | — | NOT_STARTED | Registry inicializado, strats verificados empíricamente (pickle-safe + fuentes documentadas) | COORDINADORA |
| 2026-04-17 (canary P1.5) | 3560-3569 | — | — | Conteo corregido 55→50 (Regla 26): grep contaba banner `print("✅ Batch X")` como STRATEGY_EXPORT match. Canónico vía `len(mod.STRATEGY_EXPORT)` = 5/batch × 10 = 50. | COORDINADORA |
| 2026-04-17 ~14:45 (canary P2) | 3560-3565 | NOT_STARTED | BLOCKED_INCOMPAT | BUG DETECTADO: batches usan `fn` en vez de `gen`; `_load_strategies_in_worker` hace silent drop (30 strats no ejecutables). Canary mostró 50 combos de 125 esperados — las 2 que pasaban eran versiones viejas (2450/2773). BUZON enviado a HUNTER1 fix opción A. | COORDINADORA |
| 2026-04-17 ~14:45 | mac_survivors_20260417 | FORENSIC_RUNNING | COMPLETE | CEREBRO forensic: 1 CONFIRMED (SWARMS 15m), 2 WARN, 3 REJECTED. Gap avg -5.6pp. R30 validado con 208 trades reales en 15m. | CEREBRO → COORDINADORA |
| 2026-04-17 ~15:00 | mac_survivors/SWARMS 15m | CONFIRMED | AUTH_SHADOW | SWARMS 15m (208 trades, PF 4.13) autorizada inject V8 shadow mode 20 trades. ESTRATEGIAS ejecuta + SIGNAL_AUDITOR precheck RVOL R29. | COORDINADORA |

---

## 🚨 Regla crítica para COORDINADORA

**NO lanzar Optuna hasta que Sabrina escriba literalmente "lanza Optuna" / "autorizado" / similar explícito.**

Si llega autorización:
1. Verificar `optuna_v7.py` → `TFS` incluye `"15m"` (Regla 30)
2. Verificar workers Hetzner online (`ssh hetzner 'ps aux | grep optuna'`)
3. Lanzar con **paths ABSOLUTOS** (`/Users/sabrina/CLAUDE CODE/...`) y progress files **únicos por batch** (`batch_3560_progress.json`, no `progress.json`)
4. Cambiar `forensic` a `OPTUNA_RUNNING` + fecha/hora en esta tabla
5. Notificar Agent Bus `SIGNAL` a CEREBRO (`prepare forensic`)
6. Registrar en `DECISION_LOG.md`

---

## 📞 Cross-reference

- **Broadcast arquitectura**: `BOT V7/buzon/BROADCAST_ARQUITECTURA_HUNTER_A_TODAS_SESIONES.md`
- **HUNTER1→COORDINADORA**: `BOT V7/buzon/HUNTER1_PARA_COORDINADORA_HUNTER_ARCHITECTURE.md`
- **Memoria one-pager**: `/Users/sabrina/CLAUDE CODE/HUNTER_MEMORIA_PARA_GUARDAR.md`
- **Confirmación COORDINADORA**: `BOT V7/buzon/COORDINADORA_CONFIRMACION_HUNTER_ARQUITECTURA_20260417.md`
- **Reglas 24 (forensic) y 30 (15m)**: `~/.claude/CLAUDE.md`
- **Regla pickle-safe** (no lambdas en `gen`): `feedback/` Apr 9 2026

---

**Mantenedor**: COORDINADORA. Cualquier otra sesión que quiera cambiar el estado de un batch → BUZON a COORDINADORA, no editar directamente.
