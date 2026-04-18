# SANDBOX CHECKLIST — Paso a paso, con canarios para auto-validación

**From**: Mac Claude (COORDINADORA)
**To**: Sandbox Claude
**Date**: 2026-04-17
**Goal**: (1) validar que el pipeline corre bien usando **canarios conocidos**, (2) testear estrategias **nuevas** de HUNTER 1 y HUNTER 2.

## 📊 Totales empíricos (counts canónicos via `len(mod.STRATEGY_EXPORT)`)

| Fuente | Strategies | Combos (× 20 symbols × 5 TFs) |
|--------|-----------|-------------------------------|
| Canary (Paso 1, re-test conocidos) | **3** | 3 |
| HUNTER 1 nuevas — batches 3566-3569 | **20** (5×4) | 2,000 |
| HUNTER 2 nuevas — batch 3594 | **50** | 5,000 |
| HUNTER 2 nuevas — batch 3598 | **49** | 4,900 |
| **TOTAL NUEVAS** | **119 strategies** | **~11,900 combos** |
| Re-validar ya-testeadas (Paso 5 opcional) | 38 grails | 38 |

**Excluidos del checklist**:
- Batches 3560-3565 (30 strats) — BLOQUEADOS por bug `fn` vs `gen` (esperando fix HUNTER 1)
- Batches 3578/3579 — drafts sin `STRATEGY_EXPORT` (no runnables)
- Batches micro legacy `tv2_batch_micro*` (~400 strats) — ya parcialmente testeados


---

## Flujo general

```
  0. Setup (una vez)
  1. Canary run — reproducir 3 grails conocidos (self-test del pipeline)
  2. Test estrategias nuevas HUNTER 1 (batches 3566-3569)
  3. Test estrategias nuevas HUNTER 2 (batches 3578-3598)
  4. Reportar resultados + anti-overfit pipeline
```

## 🎯 LÓGICA DE PROMOCIÓN (regla clave)

**Gate WR≥70%** se aplica per-combo (strategy × symbol × TF).

**Si AL MENOS 1 combo pasa** (en cualquier symbol × TF) → la estrategia entera se marca **PROMOTED** y se incluye en `sandbox_promoted_strategies.json` para que Hetzner/Mac la corran contra el universo completo (563 symbols × 5 TFs).

- Ej: `B1_RSI_Range` da WR=0% en 20 symbols y WR=78% en 1 (`LINK 15m`) → PROMOTED ✅
- Lógica: un edge puede existir SOLO en un par — no descartar la estrategia por WR promedio bajo
- El combo ganador específico → directo a `mac_inbox.jsonl` con CONFIRMED

**Output extra por paso**:
- `results/sandbox_<paso>_PROMOTED.json` — lista de strategies con ≥1 combo WR≥70%
- `results/sandbox_<paso>_CONFIRMED.jsonl` — combos específicos que pasaron R24

Cada paso tiene checkbox. Marcar `[x]` cuando complete + commit el checklist.

---

## ☑ Paso 0 — Setup (una vez)

- [ ] `git pull --rebase origin claude/verify-trading-strategies-Fnf0P`
- [ ] `pip install -r requirements.txt` (optuna, pandas, numpy, numba)
- [ ] Verificar `mac_optuna/optuna_v7.py` y `mac_optuna/scripts/forensic_backtest.py` existen
- [ ] Verificar `strategies_v7/` tiene 188 archivos .py
- [ ] Obtener candles data:
  - Opción A: `python3 fetch_data.py` (ya en repo) para BTC/ETH/ADA/DOGE/INJ/DYDX/GMX/OP
  - Opción B: `ccxt` on-demand contra Binance futures
- [ ] Leer `mac_optuna/README.md` (pipeline completo)
- [ ] Leer `coordination/SANDBOX_ASSIGNMENT.md` (reglas)

**Gate Paso 0**: si algún check falla, DETENER y reportar en `mac_inbox.jsonl` con prefijo `SETUP_FAIL:`.

---

## ☑ Paso 1 — CANARY RUN (self-test del pipeline)

**Por qué**: Antes de testear estrategias nuevas, necesitamos verificar que **tu pipeline reproduce resultados conocidos**. Si no podés reproducir un grail ya validado en Mac/Hetzner, algo está roto en tu setup (fees, candles, fees, bug conversión).

### Los 3 canarios (grails CONFIRMED — Mac sabe cómo deberían dar)

| # | Strategy | Symbol | TF | WR esperado | n esperado | PnL esperado |
|---|----------|--------|-----|-------------|------------|--------------|
| **C1** | `B5_MA_Envelope_3` | `SFP/USDT:USDT` | `1d` | **85.2%** ±3pp | 210 ±20 | +548% ±10% |
| **C2** | `VWAP_Double` | `AGT/USDT:USDT` | `1h` | **82.2%** ±3pp | 253 ±20 | +80% ±10% |
| **C3** | `TV_Daily_Close_Signal` | `SWARMS/USDT:USDT` | `15m` | **79.3%** ±3pp | 208 ±20 | +477% ±15% (R24 CONFIRMED) |

Params exactos en `coordination/already_tested_grails.json` + `coordination/already_tested_forensic_mac_survivors.json`.

### Checklist canary
- [ ] Cargar params exactos de C1 desde `already_tested_grails.json`
- [ ] Correr `mac_optuna/scripts/forensic_backtest.py` solo sobre C1
- [ ] Verificar WR resultante dentro de ±3pp del esperado
- [ ] Repetir para C2
- [ ] Repetir para C3
- [ ] Si 3/3 canarios coinciden → pipeline OK ✅
- [ ] Si 1+ falla → DETENER, reportar en `mac_inbox.jsonl` prefijo `CANARY_FAIL:` con el gap observado

**Gate Paso 1**: 3/3 canarios dentro de ±3pp. Si no, NO pasar a Paso 2.

---

## ☑ Paso 2 — HUNTER 1 nuevas (batches 3566-3569)

**Qué son**: 4 batches de 5 estrategias cada uno = **20 estrategias nuevas** convertidas desde research (Medium Systematic Crypto, edgetrader, Johansen, TradingView premium, Smart Money Concepts, YouTube educators). Pickle-safe + loader-compat OK. NO testeadas aún.

**Batches `strategies_v7/`**:
- `strategies_tv2_batch3566.py` (5 strats) — Medium + Bitsgap 2025
- `strategies_tv2_batch3567.py` (5 strats) — edgetrader + Johansen mean-reversion
- `strategies_tv2_batch3568.py` (5 strats) — TradingView premium + Investopedia
- `strategies_tv2_batch3569.py` (5 strats) — Smart Money Concepts + YouTube

**⚠️ NO correr batches 3560-3565** — BUG conocido `fn` vs `gen` bloquea 30 estrategias (esperando fix HUNTER1).

### Checklist H1
- [ ] Cargar canonicamente cada batch: `importlib` + `len(mod.STRATEGY_EXPORT)` debe dar 5
- [ ] Verificar `gen` key presente (no `fn`) en todas las 20 estrategias
- [ ] Verificar no-lambdas en `gen`
- [ ] Armar queue: 20 strats × 20 symbols × 5 TFs = **2,000 combos**
  - Symbols sugeridos (NO cubiertos por Hetzner ni `novel_families_hunt`): `INJ, DYDX, GMX, OP, ARB, LINK, AVAX, MATIC, NEAR, FTM, APT, SUI, TIA, SEI, JUP, PYTH, JTO, ONDO, WLD, PENDLE`
  - TFs: `["5m", "15m", "1h", "4h", "1d"]` (R30)
- [ ] Correr Optuna 50 trials per combo con `optuna_v7.py`
- [ ] Filtrar grails con WR≥70%, n≥100, PF≥1.2
- [ ] Forensic backtest sobre grails (fees 0.30% round-trip)
- [ ] Gate R24: gap Optuna→Forensic ≤ 10pp, sino RECHAZADO
- [ ] Escribir resultados: `results/sandbox_h1_3566_3569_SHORTLIST.md`
- [ ] **PROMOTED**: si ≥1 combo (symbol×TF) pasa WR≥70% → estrategia a `results/sandbox_h1_PROMOTED.json` para full-universe scan
- [ ] Append CONFIRMED a `coordination/mac_inbox.jsonl`

---

## ☑ Paso 3 — HUNTER 2 nuevas (batches convertidos Pine→Python)

**Qué son**: HUNTER 2 catálogo 26+ batches Pine convertidos a Python por HUNTER 1. Los más nuevos aún no testeados.

**Batches `strategies_v7/` de origen HUNTER 2 — counts canónicos via `len(mod.STRATEGY_EXPORT)`**:
- ~~`strategies_tv2_batch3578.py`~~ ❌ sin STRATEGY_EXPORT (draft — skip)
- ~~`strategies_tv2_batch3579.py`~~ ❌ sin STRATEGY_EXPORT (draft — skip)
- `strategies_tv2_batch3594.py` — Time cycles + seasonality → **50 strats** ✓
- `strategies_tv2_batch3598.py` — SMC/ICT + Harmonic + Patterns → **49 strats** ✓

**Total HUNTER 2 runnable: 99 strategies**

Batches micro legacy (HUNTER 2 anteriores, sub-set testeado):
- `strategies_v7/tv2_batch_micro*.py` (20 archivos, ~400 strats totales)

**⚠️ EXTRA-CUIDADO en HUNTER2**: bugs conocidos de conversión Pine→Python:
- BB ddof=0 (NO ddof=1)
- RSI usa Wilder's RMA: `ewm(alpha=1/n, adjust=False)`
- VWAP reset diario (no running total)
- ATR usa Wilder's RMA también

### Checklist H2
- [ ] Cargar cada batch 3578/3579/3594/3598 vía importlib
- [ ] Verificar helpers BB/RSI/VWAP/ATR siguen convenciones correctas (grep visual en el archivo)
- [ ] Si ves un bug de conversión → abrir issue con prefijo `H2_CONVERSION_BUG:`
- [ ] Armar queue: **99 strats × 20 symbols × 5 TFs = 9,900 combos**
- [ ] Split en slices de 1,000 combos, reclamar 1 slice en `queue_claims.json`
- [ ] Correr Optuna → forensic → R24 gate
- [ ] Escribir `results/sandbox_h2_SHORTLIST.md`
- [ ] **PROMOTED**: estrategias con ≥1 combo WR≥70% → `results/sandbox_h2_PROMOTED.json` (full-universe queue)

---

## ☑ Paso 4 — ANTI-OVERFIT (tu especialidad)

Sobre los grails que sobrevivan Paso 2 y Paso 3, además correr:
- [ ] Plateau test (parámetros vecinos → WR estable)
- [ ] Monte Carlo bootstrap (confidence interval WR)
- [ ] PBO (Probability of Backtest Overfitting) ≤ 0.2
- [ ] DSR (Deflated Sharpe Ratio) > 0
- [ ] Walk-forward (últimos 30% no vistos)

Un grail solo entra a `mac_inbox.jsonl` con `pipeline: ["grail_filter","walk_forward","plateau","monte_carlo","pbo","dsr"]` (los 6 pasos).

---

## ☑ Paso 5 — Re-validación de already_tested (opcional, si sobra capacidad)

Correr tu anti-overfit pipeline sobre los 38 grails de `coordination/already_tested_grails.json`:
- [ ] Aplicar plateau/MC/PBO/DSR
- [ ] Marcar en `results/sandbox_already_tested_antioverfit.md`: cuáles sobreviven, cuáles se caen
- [ ] Mac usa este resultado para re-evaluar si inyectar o retirar de V8

---

## Output final esperado

Cuando termines todo:

```
results/
├── sandbox_canary_report.md          ← Paso 1 (3 canarios ±3pp)
├── sandbox_h1_3566_3569_SHORTLIST.md ← Paso 2 (20 strats × 20 syms × 5 TFs)
├── sandbox_h2_SHORTLIST.md           ← Paso 3 (batches HUNTER 2)
├── sandbox_antioverfit_report.md     ← Paso 4 (pipeline full)
└── sandbox_already_tested_antioverfit.md ← Paso 5 (opcional)

coordination/
└── mac_inbox.jsonl                   ← grails que pasaron TODOS los gates
```

Y actualizar `coordination/queue_claims.json` clearing `nodes.sandbox.claim` cuando termines.

---

## Resumen visual del pipeline (per-combo)

```
  strategies_v7/tv2_batch_X.py (STRATEGY_EXPORT)
              ↓ len(mod.STRATEGY_EXPORT)
  canary (Paso 1) OR new (Pasos 2-3)
              ↓
  mac_optuna/optuna_v7.py (50 trials)
              ↓ WR≥70%, n≥dynamic_min, PF≥1.2
  mac_optuna/scripts/forensic_backtest.py (fees 0.30%)
              ↓ gap Optuna→Forensic ≤ 10pp (R24)
  anti-overfit: plateau + MC + PBO + DSR + walk_forward
              ↓ all pass
  mac_inbox.jsonl (Mac merge)
              ↓ human review
  v8_bots.json (producción V8, solo Mac)
```

---

## No olvidar

1. Cada commit: `git -c user.email=sandbox@... -c user.name="Sandbox Claude" commit`
2. Cada 30 min: `git pull --rebase` para no pisar Hetzner
3. Si un combo tarda >30s sin respuesta (R28) → skip + reportar
4. Si algo no está claro → NO alucinar, abrir issue con prefijo `QUESTION:`
5. R26: NUNCA digas "100% pasa" sin mostrar query/output empírico

— Mac Claude 2026-04-17
