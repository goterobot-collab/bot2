# SANDBOX HANDOFF — Instrucciones completas

**From**: Mac Claude (COORDINADORA)
**To**: Claude Code Sandbox
**Date**: 2026-04-17
**Repo**: `git@github.com:goterobot-collab/bot2.git`
**Branch**: `claude/verify-trading-strategies-Fnf0P`

---

## 🎯 TUS 2 ROLES

### Rol A — TESTER (pipeline de validación)
Correr Optuna + Forensic sobre estrategias ya convertidas y reportar grails.

### Rol B — HUNTER (descubrimiento de estrategias nuevas)
Buscar estrategias nuevas en fuentes públicas, convertirlas a Python pickle-safe, y empujarlas al repo para que Mac/Hetzner/Sandbox las corran.

**Priorizá Rol A primero** (canary + HUNTER1/HUNTER2 nuevas). Cuando termines o si te aburrís esperando Optuna → Rol B en paralelo.

---

## 📥 SETUP (una vez)

```bash
cd ~
git clone git@github.com:goterobot-collab/bot2.git bot2-collab
cd bot2-collab
git checkout claude/verify-trading-strategies-Fnf0P
pip install optuna pandas numpy numba
```

Leer en orden:
1. `coordination/SANDBOX_ASSIGNMENT.md` — reglas no negociables
2. `coordination/SANDBOX_CHECKLIST.md` — pasos 0-5
3. `mac_optuna/README.md` — pipeline canónico
4. `coordination/BATCH_REGISTRY.md` — qué ya está testeado (no repetir)

---

## 📊 DATOS (candles ya en el repo)

Sin acceso Binance API (403). Usá los CSVs pusheados:

```python
import pandas as pd
df = pd.read_csv("data/candles/SFP_5m.csv.gz", compression="gzip")
# columns: ts (ms), open, high, low, close, volume
```

**21 símbolos disponibles** (5m + 1h nativos): SFP, AGT, SWARMS, INJ, DYDX, GMX, OP, ARB, LINK, AVAX, NEAR, APT, SUI, TIA, SEI, JUP, PYTH, JTO, ONDO, WLD, PENDLE.

**15m / 4h / 1d**: resamplear desde 5m/1h con pandas.

**NO disponibles**: MATIC, FTM (sin datos en DB Mac). Skipealos.

---

## 🟢 ROL A — TESTER

### Paso 1: CANARY (self-test obligatorio)

Reproducir 3 grails CONFIRMED antes de testear nada nuevo:

| # | Strategy | Symbol | TF | WR esperado ±3pp |
|---|----------|--------|-----|-------------------|
| C1 | B5_MA_Envelope_3 | SFP | 1d | 85.2% |
| C2 | VWAP_Double | AGT | 1h | 82.2% |
| C3 | TV_Daily_Close_Signal | SWARMS | 15m | 79.3% |

Params exactos en `coordination/already_tested_grails.json`.

Si 3/3 coinciden ±3pp → pipeline OK. Si falla → STOP, reportar en `mac_inbox.jsonl` con prefijo `CANARY_FAIL:`.

### Paso 2: HUNTER1 nuevas (batches 3566-3569 = 20 strats)
### Paso 3: HUNTER2 nuevas (batches 3594 + 3598 = 99 strats)

Para ambos pasos:
```bash
python3 mac_optuna/optuna_v7.py \
  --combos-file /abs/path/to/sandbox_queue.json \
  --progress-file /abs/path/to/results/progress.json \
  --workers 4

python3 mac_optuna/scripts/forensic_backtest.py \
  --grails results/grails.json \
  --output results/forensic.json
```

**Queue**: 119 strats × 21 symbols × 5 TFs = ~12,495 combos.

### Paso 4: ANTI-OVERFIT
Sobre grails supervivientes: plateau + Monte Carlo + PBO + DSR + walk-forward.

### Output esperado
```
results/
├── sandbox_canary_report.md
├── sandbox_h1_SHORTLIST.md
├── sandbox_h1_PROMOTED.json        ← estrategias con ≥1 combo WR≥70%
├── sandbox_h2_SHORTLIST.md
├── sandbox_h2_PROMOTED.json
└── sandbox_antioverfit_report.md

coordination/mac_inbox.jsonl         ← append CONFIRMED (gap ≤10pp)
```

---

## 🎯 LÓGICA DE PROMOCIÓN (regla clave)

- Gate WR≥70%, n≥dynamic_min, PF≥1.2 **per-combo** (strat × symbol × TF)
- Gate R24: gap Optuna→Forensic ≤10pp → CONFIRMED
- **PROMOTION**: si **≥1 combo** de una estrategia pasa WR≥70% → la estrategia entera va a `PROMOTED.json` para que Hetzner/Mac la corran contra 563 symbols full universe
- Lógica: un edge puede existir solo en 1 par (ej: `B1_RSI_Range × LINK × 15m` = 78%, resto = 0%) → la estrategia sirve

---

## 🔍 ROL B — HUNTER (estrategias nuevas)

**Objetivo**: buscar estrategias de trading públicas, convertirlas a Python pickle-safe, empujar al repo como nuevo batch `strategies_tv2_batchXXXX.py`.

### Fuentes permitidas (prioridad)
1. **TradingView Pine Script** (v4/v5/v6 ONLY — v1/v2/v3 REPINTAN, prohibidas)
2. **GitHub repos**: backtrader, freqtrade, jesse-ai, vectorbt, hummingbot strategies
3. **Kaggle notebooks** de crypto trading (filtrar por WR empírico reportado)
4. **ArXiv/SSRN papers** quant con pseudocódigo claro
5. **Blogs técnicos**: QuantInsti, Medium Systematic Crypto, edgetrader, Investopedia Algo Trading
6. **YouTube educators** con Pine published + backtest evidence
7. **Smart Money Concepts / ICT** (order blocks, FVG, liquidity sweeps)

### Fuentes PROHIBIDAS
- Pine v1/v2/v3 (repintan)
- Estrategias sin source/backtest público (alucinación)
- "TV_INV_*" (inventadas sin fuente) — ver `feedback_tv_inv_inventadas_sin_fuente`
- Strategies con lambdas en `gen` (rompen pickle Optuna)

### Formato output (obligatorio)

Cada batch nuevo = 1 archivo en `strategies_v7/strategies_tv2_batchXXXX.py`:

```python
"""
Batch XXXX — [Nombre temático]
Fuente: [autor + URL + referencia]
Conversión: HUNTER sandbox 2026-04-XX
Estrategias: N
"""

import numpy as np
import pandas as pd

def gen_strat_name(df, **params):
    \"\"\"Genera signals dataframe con columnas: signal (1=long, -1=short, 0=flat), entry_price\"\"\"
    # implementación COMPLETA — sin placeholders, sin lambdas
    ...
    return signals

def space_strat_name(trial):
    return {
        "param1": trial.suggest_int("param1", 5, 50),
        "param2": trial.suggest_float("param2", 0.1, 2.0),
    }

STRATEGY_EXPORT = {
    "strat_name_1": {"gen": gen_strat_name, "space": space_strat_name},
    # ... más estrategias
}
```

### Reglas técnicas (obligatorio — regla 26)

1. **Pickle-safe**: NUNCA lambdas en `gen` — usar `def` nombrado top-level
2. **Sin look-ahead**: no usar `df.shift(-N)`, no `.iloc[i+1]`, no future data
3. **Indicadores correctos**:
   - BB: `df.rolling(n).std(ddof=0)` (NO ddof=1)
   - RSI: Wilder's RMA `ewm(alpha=1/n, adjust=False)`
   - VWAP: reset diario por sesión UTC (no running total)
   - ATR: Wilder's RMA también
4. **Verificación canónica antes de commit**:
   ```python
   import importlib, sys
   sys.path.insert(0, "strategies_v7")
   mod = importlib.import_module("strategies_tv2_batchXXXX")
   print(len(mod.STRATEGY_EXPORT))
   for name, spec in mod.STRATEGY_EXPORT.items():
       assert "gen" in spec and "space" in spec
       assert spec["gen"].__name__ != "<lambda>"
   ```
5. **Numbering**: pedí próximo ID libre antes de crear batch. Actuales ocupados: 3560-3598. Usá 3600+.

### Workflow Rol B

1. Investigar fuente → extraer pseudocódigo/Pine
2. Convertir a Python siguiendo reglas técnicas
3. Verificación canónica local
4. Correr mini-test contra 3 symbols × 2 TFs (BTC/ETH/SOL × 5m/1h si tenés data — o SFP/AGT/LINK que están en el repo) → confirmar que genera signals (>0 trades)
5. Commit: `git add strategies_v7/strategies_tv2_batchXXXX.py`
6. Push → automáticamente aparece en queue para testing
7. Registrar en `coordination/BATCH_REGISTRY.md` con fuente + N strats + status `NOT_TESTED`

### Targets sugeridos de caza
- **Order flow**: delta, imbalance, CVD, footprint (escasas en V8, alto potencial)
- **Market microstructure**: bid-ask spread, Kyle lambda, VPIN
- **Harmonic patterns**: Gartley, Butterfly, Bat, Crab (Pine abundante)
- **Wyckoff**: Accumulation/Distribution phases, Springs/Upthrusts
- **Seasonality**: time-of-day × day-of-week effects en crypto majors
- **Regime-switch**: HMM, markov, volatility clustering breakouts
- **Inter-market**: BTC dominance × altcoin signals, funding rate divergence

---

## 🔒 REGLAS NO NEGOCIABLES (resumen)

| # | Regla | Gate |
|---|-------|------|
| R23 | Full historical backtest antes de grail | Obligatorio pre-inyección |
| R24 | Gap Optuna→Forensic ≤10pp | RECHAZADO si >10pp |
| R26 | NUNCA "100% cumple" sin query empírica | Anti rubber-stamp |
| R28 | Operación >30s sin respuesta → reportar | No silent wait |
| R29 | Filtros per-activo × per-estrategia | Nunca genéricos |
| R30 | TFs = [5m,15m,1h,4h,1d] | 15m obligatorio |
| Golden | SL/TP/lev per-activo × per-estrategia | 100% empírico |
| dynamic_min | WR100%→n≥2, 90%→4, 80%→8, 70%→8 | No decidir sin muestra |
| Pickle-safe | NUNCA lambdas en `gen` | ProcessPoolExecutor crash |
| Canónico | `len(mod.STRATEGY_EXPORT)` via importlib | No grep -c |
| Fees | 0.10% + 0.05% slippage = 0.30% round-trip | En todo PnL |
| Paths | ABSOLUTOS en --combos-file y --progress-file | Bug data/data/ duplicado |

---

## 🚫 LO QUE NO DEBES HACER

- ❌ NO correr batches 3560-3565 (BLOQUEADOS por bug `fn` vs `gen` — esperando HUNTER1 fix)
- ❌ NO repetir combos ya en `already_tested_grails.json` (38 grails) ni `already_tested_forensic_mac_survivors.json` (6 grails)
- ❌ NO tocar `v8_bots.json` (no está en el repo de todos modos — solo Mac lo tiene)
- ❌ NO crear estrategias TV_INV_* sin Pine source
- ❌ NO usar MATIC ni FTM (no hay data en DB Mac)
- ❌ NO hacer `rm -rf` ni borrar candles
- ❌ NO declarar "cumple" sin mostrar query + output raw
- ❌ NO trabajar sobre claims de otros nodos — revisar `coordination/queue_claims.json` (Hetzner tiene `mac_survivors_20260417`, vos tenés `novel_families_hunt`)

---

## 🔄 COORDINACIÓN

Cada 30 min:
```bash
git pull --rebase origin claude/verify-trading-strategies-Fnf0P
```

Al reclamar slice nuevo → editar `coordination/queue_claims.json` → commit → push.

---

## 📤 ENTREGA

Cuando termines un paso:
```bash
git add results/ coordination/mac_inbox.jsonl coordination/BATCH_REGISTRY.md strategies_v7/
git commit -m "Sandbox: <descripción>"
git push origin claude/verify-trading-strategies-Fnf0P
```

Clear tu claim:
```python
# en queue_claims.json: nodes.sandbox.claim = null
```

---

## ❓ SI TENÉS DUDAS

NO alucinar. NO codear a ciegas. Abrir issue en repo O commit con prefijo `QUESTION:` en `coordination/mac_inbox.jsonl` y esperar.

**Máximo 3 preguntas al iniciar** — después ejecutar.

— Mac Claude 2026-04-17
