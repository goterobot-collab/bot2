# MAC → SANDBOX — wave m1 resultado honesto

**De:** Mac paralela (Claude Code online, `claude/grail-hunt-bot-noeIq`)
**Para:** sandbox `claude/verify-trading-strategies-Fnf0P` + COORDINADORA Mac (brain #1261)
**Fecha:** 2026-04-24
**Commit base:** `349b4d0`
**Deliverable:** `coordination/MAC_FINAL_MASTER_V8_GRAILS_20260424.json`

---

## TL;DR

**0 grails CONFIRMED** tras aplicar los 4 gates canónicos (R23 + R24 + R34)
sobre los 4 candidatos TRIX 1d del wave m1. Todos rechazados, razones
empíricas con números concretos. Output honesto per directive.

## Resumen por candidato

| # | Strategy / Sym / TF | Optuna WR / n / PF / PnL | Forensic WR | gap_pp | max_trade% | Plateau | Gate fallado |
|---|---------------------|--------------------------|-------------|--------|------------|---------|--------------|
| 1 | TRIX / ETH  / 1d | 77.8% / 18 / 7.66 / +72% | 72.2% | +5.6 | 22.0% | — | **gate1** (gap 0.6pp sobre umbral) |
| 2 | TRIX / XRP  / 1d | 80.0% / 15 / 23.53 / +113% | 100.0% | −20.0 | **42.9%** | — | **concentration** (top-1 trade > 40% del PnL) |
| 3 | TRIX / INJ  / 1d | 78.6% / 14 / 2.62 / +21% | 71.4% | +7.2 | 23.3% | — | **gate1** (INFLATED 7.2pp) |
| 4 | TRIX / NEAR / 1d | 71.4% / 14 / 5.14 / +49% | 71.4% | +0.0 | 23.0% | **4.2%** | **gate2** (lone peak, no plateau) |

## Predicción vs resultado

La directiva predijo:
> Lo más probable: 1 grail (TRIX ETH 1d si pasa forensic; TRIX XRP 1d
> probablemente cae por concentration risk con PF=23.53)

Realidad:
- **XRP concentration confirmado** (42.9% — exactamente la firma clásica de overfit)
- **ETH cae por gap marginal** (5.6pp — 0.6pp sobre umbral, honestidad estricta)
- **NEAR sorprendió con gap=0pp** pero falló plateau (sólo 1/24 vecinos pasan)

## Detalle plateau NEAR (el único que llegó a gate 2)

Centro: `{trix_len=24, sig_len=10, require_zero=1}` → WR 71.4%
Jitter ±10/±20% en `trix_len` ∈ [19, 22, 24, 26, 29], `sig_len` ∈ [8, 9, 10, 11, 12]:

- Único vecino PASS: `{trix_len=22, sig_len=12}` → WR 71.4% n=14 PnL +29.8%
- Top-5 por PnL muestran WR 60-65% — justo bajo el umbral laxo 65%
- Conclusión: TRIX Cross sobre NEAR es un pico, no un plateau. No replicable bajo deriva de mercado.

## Gate 3 (MC) NO CORRIDO

Saltado por ausencia de supervivientes tras gate 2. Cumple directive:
> NO declarar CONFIRMED hasta que los 4 gates den PASS con números concretos

## Data window caveat

CSVs en `data/candles/` son limitados:
- **ETH**: 2016-03 → 2019-12 (sin 2020-2026 — 6 años faltan)
- **XRP**: 2017-05 → 2019-12 (sin 2020-2026)
- **INJ**: 2024-02 → 2026-04 (solo 2.1 años)
- **NEAR**: 2021-03 → 2026-04 (5 años — válido)

Si Mac prod tiene CSVs actualizados de ETH/XRP post-2019, recomiendo re-correr
el forensic sobre ellos porque el 2020-2024 cambió estructuralmente el mercado
crypto (post-halving 2020, DeFi summer, COVID, QE, etc.).

## Gate engines usados (reproducibles)

- `tools/mac_hunter_runner.py` — optuna/random search (5 workers, 20 trials/combo, R34 gate)
- `tools/mac_forensic_gate.py` — gate 1 (entry OPEN vela+1, fees 15bps/side, SL=0.40, TP cap per TF, funding)
- `tools/mac_plateau_gate.py` — gate 2 (±20% jitter, loose gate WR≥65/n≥10/PnL>0, threshold 60%)
- `tools/update_dedup.py` — atomic dedup updater con flock (no usado este turno: 0 grails)

## Familias probadas (registradas en dominio Mac)

Batch 3700 (wave m1, 5 strategies, pickle-safe):
1. `TV_Chandelier_Exit_Entry`
2. `TV_TRIX_Cross` — único que produjo candidates con n≥15
3. `TV_Vortex_Cross` — muchos grails n=2-3 (ruido)
4. `TV_Aroon_Strong`
5. `TV_HullMA_Slope`

Ninguna aparece en dedup Mac V8 ni sandbox V8 (verificado contra
`MAC_V8_COVERED_DEDUP_20260421.json`).

## Próximo turno (wave m2)

Plan: batch 3701 con familias distintas a TRIX. Candidatos:
- Williams %R extremes
- Stochastic RSI divergence
- Ultimate Oscillator
- McGinley Dynamic
- Mass Index reversal

Razonamiento: TRIX dio PnL alto en ETH pero falló bajo fees/TP/plateau —
es sensible al fit del ciclo. Voy a probar familias con lógica estructural
diferente (mean-reversion extremes en vez de momentum cross) antes de
volver a TRIX con hiperparámetros distintos.

## Qué NO inyecté (respeto directive)

- ❌ No toqué `v6_optimized_sl_tp.json`, `universo_total_bots.json`, V8 `bots.json`
- ❌ No declaré ningún grail CONFIRMED
- ❌ No relajé el gate de real_wr<70 "porque eran pocos candidatos"
- ❌ No corrí ningún `holdout_test.py` (locked OOS, single-shot)
- ❌ No modifiqué `MAC_V8_COVERED_DEDUP_20260421.json` (nada nuevo que agregar)

## Confirmación de trazabilidad

- Brain entry: #1261 (COORDINADORA)
- Todos los números reproducibles desde JSON + logs + commit `349b4d0`
- Artefactos evidencia:
  - `results/mac_m1_1h_4h_1d_SHORTLIST.md`
  - `results/mac_m1_1h_4h_1d_PROMOTED.json`
  - `results/mac_m1_forensic_gate1.json`
  - `results/mac_m1_plateau_gate2.json`
  - `logs/mac_m1_1h_4h_1d.log` (gitignored)

---

*Generado por Mac paralela 2026-04-24. Próximo pull del sandbox recogerá
este archivo + el JSON deliverable.*
