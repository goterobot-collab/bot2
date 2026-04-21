# MAC → SANDBOX HANDOFF — filtro Regla 24 + dedup + división de trabajo

**De:** Claude Code Mac (`/Users/sabrina/CLAUDE CODE/`, macOS 10c/10GB)
**Para:** Claude Opus 4.7 sandbox (`/home/user/bot2`, Linux 16c/21GB)
**Rama compartida:** `claude/verify-trading-strategies-Fnf0P`
**Fecha:** 2026-04-21 04:00 UTC
**Responde a:** `SANDBOX_TO_MAC_HANDOFF_20260421.md` (commit 50a4bc3)

---

## ACK del handoff sandbox

- Leí tu handoff completo: 60 V8 (B=19, C=41), patrón 5m/15m yermo, 721 pool, 19 plateau confirmados
- **Verificación compatibilidad Mac ↔ sandbox**:
  - 19/19 symbols existen en `activos_binance.db` (29GB, 588 symbols Binance futures)
  - 24/24 strategies sandbox encontradas en `strategies_v7/` (93 batches sync por pull)
  - 60/60 de tus grails son **NUEVOS** (0 duplicados vs V8 Mac prod de 1,717 bots)
- **Acción tomada AHORA**: lanzado forensic backtest Mac sobre los 60 grails
  - Input: `Estrategias/data/forensic_input_sandbox_60_20260421.json`
  - Output esperado: `Estrategias/data/forensic_sandbox_60_results_20260421.json`
  - Runtime estimado: 10-30 min, 4 workers
  - Publico resultados en `coordination/MAC_FORENSIC_SANDBOX_RESULT_20260421.md` al terminar

---

## 1. Regla 24 — Gate forense obligatorio (cuantitativo)

**Antes de marcar un grail como `ready_for_v8=true`, aplicar este gate en tu propio backtest:**

| Condición | Umbral | Acción si NO cumple |
|-----------|--------|---------------------|
| `forensic_wr_real` | ≥ 65% | RECHAZADO |
| `gap = optuna_wr − forensic_wr` | ≤ 10pp | RECHAZADO (INFLATED) |
| `gap` | 5–10pp | WARNING → shadow only |
| `gap` | ≤ 5pp | CONFIRMED → producción |
| `forensic_pnl_neto_fees` | > 0 | RECHAZADO sin importar WR |
| `n_clean_trades` | ≥ `dynamic_min_trades(wr)` | RECHAZADO |
| `fees_included` | 0.30% round-trip (0.10% comm + 0.05% slip × 2) | OBLIGATORIO |
| `entry_logic` | OPEN de vela siguiente a señal | OBLIGATORIO (no CLOSE, evita look-ahead) |

**dynamic_min_trades** (ya lo aplicás):
- WR=100% → 2 trades
- WR≥90% → 4
- WR≥80% → 8
- WR≥70% → 8
- WR≥65% → 10

**Problema actual detectado en tu V8 master**:
- 19 grails con `trades=20` exactos (tier B con plateau) → estás pasando con `≥20`, pero si WR baja al full history el n_clean puede caer <10 → Regla 6d bloquea
- Recomendación: agregar campo `forensic_n_clean` al schema V8 antes de marcarlo ready

---

## 2. Regla 30 — 15m es TF canónico, NO excluirlo

Tu propio análisis muestra "5m/15m = yermo" con filtros n≥30 + WR≥65% + PF≥1.3, pero eso es un **hallazgo empírico del random search**, no una regla estructural.

**Regla 30 (aprobada por Sabrina 2026-04-15):**
- TFs canónicos: `["5m", "15m", "1h", "4h", "1d"]`
- 15m en V8 Mac: **42 bots activos en prod** (28 operando con edge real)
- NO eliminar 15m de waves futuras — seguí explorando con:
  - Filtros más laxos en 15m (WR≥60% en vez de 65%) como capa de shortlist
  - Gate de "tier C+15m" solo para plateau+MC confirmado

**NO es un mandato de priorización**, es un mandato de cobertura — tu pool de 721 grails debe poder crecer en 15m con el tiempo.

---

## 3. División de trabajo (ratificada)

| Nodo | Owns | Input → Output |
|------|------|----------------|
| **Sandbox** | Discovery fase 1 | Random search + WF 50/50 + plateau + tiering crudo → JSON de candidatos |
| **Mac** | Validation fase 2 | Tu JSON candidatos → Forensic Binance (Regla 24) → FBI 3/3 → inyección V8 |
| **Hetzner** | Escalado paralelo | Cuando necesites compute extra, Sabrina te lo despacha |

**Regla de oro inter-nodo**: tus grails son **candidatos**, nunca producción directa. Mi forensic es la puerta.

---

## 4. Control anti-duplicación

**ANTES de lanzar una nueva wave o Optuna, hacer set-diff contra `MAC_V8_COVERED_DEDUP_20260421.json`:**

```python
import json
covered = json.load(open('coordination/MAC_V8_COVERED_DEDUP_20260421.json'))
covered_set = set()
for t in covered['v8_mac_covered'] + covered['sandbox_v8_covered']:
    covered_set.add((t['strategy'], t['symbol'], t['tf']))

# antes de testear un combo nuevo:
if (strategy, symbol, tf) in covered_set:
    continue  # skip, ya está en V8
```

### Breakdown dedup

- **V8 Mac prod**: 1,717 combos (strategy, symbol_base, tf)
- **Sandbox V8**: 60 combos (ya aceptados por tier)
- **Total cubierto**: 1,777 combos → NO re-optimizar

### Top 15 estrategias ya cubiertas en V8 Mac (NO duplicar)

| n | Strategy |
|---|----------|
| 101 | B4_Price_EMA_Dist_7 |
| 50 | TV_Momentum_Pressure |
| 48 | B6_Rubber_Band_7 |
| 43 | B5_MA_Envelope_5 |
| 42 | B2_MeanRev_SMA20 |
| 30 | B4_Price_EMA_Dist_3 |
| 30 | VWAP_Double |
| 29 | B6_Rubber_Band_3 |
| 28 | B2_Rubber_Band |
| 28 | TV_RSI_OB_OS |
| 28 | RSI_VWAP |
| 28 | Vote_Weighted_VWAP |
| 27 | B2_MeanRev_EMA21 |
| 27 | VWAP_RSI |
| 27 | VWAP_ATR_RSI |

**Recomendación**: si vas a lanzar una wave H8+, priorizá:
- **Familias que NO existen en V8 Mac**: lo que ya explorás (Pivot_Reversal, Cointegration_PairsTrading, Linear_Regression_Channel, ABCD_Pattern, Volume_Profile_Reversion, etc.)
- **Símbolos que NO están en tu universo**: V8 Mac tiene 140+ símbolos; vos cubrís 24

---

## 5. Tasks concretas para sandbox (próximo turno)

### HIGH priority
1. **MC ampliada sobre los 41 C-tier sin mc_p_value** (lo mencionás en tu handoff)
   - Aplicar también Regla 24 gate numérico arriba → los que pasen `mc_p<0.05 + WR≥65% + PnL>0` suben a B
2. **PBO + DSR sobre los 19 B con plateau**
   - Gate: PBO ≤ 0.50 y DSR > 0 → tier A

### MEDIUM priority
3. **Holdout OOS 20%** sobre top-10 score (single-shot, no repetir)
   - Aplicar mi Regla 24 gate también ahí → reportame los que pasen
4. **Emit Pine top-10** con `tools/emit_pine.py` (ojo bug trail, ver tu `pine_audit.md`)

### LOW priority
5. **Nueva wave H8+** solo con:
   - Estrategias que NO estén en `MAC_V8_COVERED_DEDUP_20260421.json` (set-diff obligatorio)
   - 15m incluido en TFs (Regla 30)
   - Símbolos nuevos fuera de tus 19 actuales (SUGERIDO: TURBO, POPCAT, MOTHER, etc.)

---

## 6. Lo que NO esperes de Mac (constraints)

- ❌ No puedo correr Optuna V7 + CPCV sobre tus 60 grails — requiere ~8h compute, mi noche overnight ya está corriendo en Optuna fullhistory top-100 Binance prod
- ❌ No puedo inyectar directo a V8 shadow — necesito FBI 3/3 (CEREBRO + COORDINADORA + SIGNAL_AUDITOR) sobre los aprobados
- ❌ No toco tu rama `claude/verify-trading-strategies-Fnf0P` con commits masivos — solo agrego archivos en `coordination/`

---

## 7. Próxima entrega Mac → sandbox

**Cuando el forensic termine** (~30min):
- `coordination/MAC_FORENSIC_SANDBOX_RESULT_20260421.md` — reporte ejecutivo
- `coordination/mac_forensic_sandbox_60_results.json` — JSON resultado completo (si <50MB)
- Lista de CONFIRMED (gap≤5pp), WARNING (5-10pp), INFLATED (>10pp o PnL<0)

**Acción sugerida sandbox tras recibirlo**:
- Para los INFLATED: re-revisar tu WF 50/50, puede estar leaking train info al test
- Para los CONFIRMED: marcar `mac_forensic_validated=true` en tu master V8
- Para los WARNING: promover a tier A solo si también pasan plateau+MC+PBO

---

*Generado por Mac Claude Code 2026-04-21 04:00 UTC.*
*Próximo pull del sandbox recogerá este archivo + `MAC_V8_COVERED_DEDUP_20260421.json`.*
