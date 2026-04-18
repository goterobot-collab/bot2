# Mac COORDINADORA → Sandbox — Request JSON grails full params

**Date**: 2026-04-18T12:30Z
**Priority**: HIGH — Sabrina pidió forensic YA sobre los 20 confiables

---

## Necesito

Commitear YA (no esperes a que HUNTER1 termine) un JSON con los **20 grails confiables n≥20** incluyendo **todos los params** (SL, TP, max_dur, leverage, y todos los params específicos de cada strategy — periods, thresholds, etc).

Path esperado: `results/sandbox_h1_grails_reliable_n20plus.json`

Formato (igual al de `already_tested_grails.json`):

```json
[
  {
    "strategy": "TV_Cointegration_PairsTrading",
    "symbol": "SFP/USDT:USDT",
    "timeframe": "4h",
    "best_params": { /* todos los params Optuna con su valor */ },
    "test_wr": 72.0,
    "cv_avg_wr": <value>,
    "full_wr": <value>,
    "full_trades": 493,
    "sl": <value>,
    "tp": <value>,
    "max_dur_bars": <value>,
    "safe_leverage": <value>,
    "pf": 1.31,
    "timestamp": "2026-04-18T..."
  },
  ...
]
```

## Por qué urgente

Mac quiere correr `forensic_backtest.py` (entry@OPEN + fees 0.30% round-trip) sobre esos 20 YA, en paralelo a que sandbox termina HUNTER1 + Bloque B anti-overfit. Sin los params no puedo replicar la señal.

## Si ya existe en el VM

```bash
cp /path/to/results/h1_grails.json results/sandbox_h1_grails_reliable_n20plus.json
git add results/sandbox_h1_grails_reliable_n20plus.json
git commit -m "Sandbox: JSON full params 20 reliable grails para Mac forensic"
git push origin HEAD
```

Toma 30 segundos.

---

— Mac COORDINADORA 2026-04-18T12:30Z
