# MAC PARALLEL → SANDBOX HANDOFF — identity + primer plan

**De:** Claude Code (sesión online, rol "Mac paralela" del playbook)
**Para:** Sandbox `claude/verify-trading-strategies-Fnf0P` + usuario
**Rama de esta sesión:** `claude/grail-hunt-bot-noeIq` (no `claude/mac-hunter-parallel`
como sugería el playbook — el task system me fija la rama y no tengo permiso
explícito para cambiarla)
**Fecha:** 2026-04-24
**Responde a:** `MAC_PARALLEL_SESSION_INSTRUCTIONS.md` (commit a474821) +
`SANDBOX_TO_MAC_HANDOFF_20260421.md`

---

## 1. Identidad

- Soy una instancia de Claude Code corriendo en infra Anthropic (`/home/user/bot2`,
  Linux 16c/21GB), no la Mac de Sabrina ni Hetzner.
- No comparto memoria entre turnos; opero desde lo commiteado.
- No tengo APIs de exchange (todas dan 403). Dataset único:
  `data/candles/*.csv.gz` (24 símbolos × 1h/5m, histórico completo via
  `codeload.github.com`).
- No tengo acceso al Binance DB (`activos_binance.db` 29GB) — no puedo
  correr el forensic backtest Regla 24 de Mac-prod, pero sí ejecuto WF+PF+WR
  sobre full history como gate interno.

## 2. Divergencia del playbook (declarada explícitamente)

| Item playbook | Mandato | Lo que hago |
|---------------|---------|-------------|
| Rama Mac paralela | `claude/mac-hunter-parallel` desde `origin/main` | Me quedo en `claude/grail-hunt-bot-noeIq` (task system) |
| Prefijo commits | `mac-ciclo #N: ...` | Lo aplico igual |
| Dominio archivos | `results/mac_*`, `coordination/MAC_*`, `strategies_v7/strategies_mac_batch37*.py`, `tools/mac_*.py`, `logs/mac_*` | Lo aplico igual |
| Batches | 3700-3799 | Empiezo en 3700 |
| Waves label | m1-m7 | Empiezo en m1 |

**Consecuencia práctica:** sandbox no pierde nada — su rama está aislada, mi
dominio de archivos es disjunto. El único efecto es que `claude/grail-hunt-bot-noeIq`
contiene el trabajo "Mac paralela" en vez de la rama convencional.

## 3. Batch 3700 (wave m1) — familias nuevas

Seleccionadas por NO overlap con:
- Sandbox batches 3566-3610 (Pivot/ABCD/Cointegration/LR_Channel/Volume/
  Fractal/Ehlers/BTCβ/Double_MACD/OBV_Momentum/RSI_Div/BB_Squeeze/HA/KAMA/
  Ichimoku/Elder)
- Mac V8 prod (B4/B5/B6 families, VWAP combos, TV_Momentum_Pressure,
  TV_RSI_OB_OS)

Familias batch 3700:

| # | Nombre | Familia | Racional |
|---|--------|---------|----------|
| 1 | `TV_Chandelier_Exit_Entry` | ATR trailing system (Le Beau) | Flip detection on chandelier line break — no overlap con ATR_Regime_Reversal |
| 2 | `TV_TRIX_Cross` | Triple-smoothed EMA momentum | No hay TRIX en ningún batch existente |
| 3 | `TV_Vortex_Cross` | Vortex VI+/VI- (Etlaes 2009) | No aparece en sandbox ni Mac V8 |
| 4 | `TV_Aroon_Strong` | Aroon up/down con threshold 90+ | Nada de Aroon en el repo |
| 5 | `TV_HullMA_Slope` | Hull MA slope-sign reversal | HMA no está en ningún batch |

Todas: pickle-safe, sin lambdas, `.shift(1)` en señales, signal ∈ {-1, 0, 1}.

## 4. Criterios de aceptación

El hunter aplica el gate R24 reducido (sin forensic Binance, que es Mac-prod only):

| Métrica | Umbral | Fuente |
|---------|--------|--------|
| `WR` | ≥ 70% | R24 user |
| `PF` | ≥ 1.2 | playbook hunter |
| `trades` | dynamic_min_trades(wr) | playbook |
| `total_pnl_pct` | > 0 (implícito vía PF>1) | R24 user |
| TFs | `1h, 4h, 1d, 15m` (R30 incluye 15m) | R30 |

NO corro `holdout_test.py` — es single-shot, reservado a validación final Mac-prod.

## 5. Dedup check (set-diff)

Leído `coordination/MAC_V8_COVERED_DEDUP_20260421.json`:
- 1,717 V8 Mac prod + 60 sandbox V8 = 1,777 combos cubiertos
- **Ninguno** de los 5 strategies de batch 3700 aparece en el dedup
- Filtro en hunter: ignoro combos en el dedup (solo afecta si re-exploro
  familias legacy; por ahora no aplica)

## 6. Primer lanzamiento

Comando:
```bash
python3 tools/mac_hunter_runner.py --wave m1 --tfs 1h 4h 1d
```

Output esperado:
- `results/mac_m1_1h_4h_1d_SHORTLIST.md`
- `results/mac_m1_1h_4h_1d_PROMOTED.json`
- `results/mac_m1_1h_4h_1d_progress.json`
- Logs: `logs/mac_m1.log`

Parametrización:
- 5 strategies × 24 symbols × 3 TFs = **360 combos**, 20 trials c/u = 7,200 evals
- Con 5 workers + early-abort ≈ 10-20min wall time

Segunda ola (si sobrevive ≥5 grails): `--wave m1 --tfs 15m` (R30).

## 7. Lo que sigue si encuentro grails

1. **Plateau ±20% jitter** → `tools/mac_plateau.py` (pendiente crear, copia
   de `plateau_on_hunter_grails.py` con paths `mac_*`)
2. **MC block bootstrap** → `tools/mc_block_bootstrap.py` (pendiente, playbook
   lo asigna a Mac)
3. Grails supervivientes → `coordination/MAC_FINAL_MASTER_V8_GRAILS.json`
4. Update dedup via `tools/update_dedup.py` con flock (pendiente crear)
5. Handoff a sandbox: `coordination/MAC_TO_SANDBOX_M1_GRAILS_<fecha>.md`

## 8. Próximas waves planificadas (si m1 funciona)

| Wave | Batch | Familia objetivo |
|------|-------|------------------|
| m1 | 3700 | Chandelier/TRIX/Vortex/Aroon/HullMA (este turno) |
| m2 | 3701 | Williams %R / StochRSI / UltimateOscillator / McGinley / Mass Index |
| m3 | 3702 | Kagi / Renko-sim / Point-and-Figure / Andrews Pitchfork |
| m4 | 3703 | Coppock / Know Sure Thing / PVT / Chaikin Money Flow / Accumulation-Dist |
| m5 | 3704 | Elder Force Index / TSI / RVI / Fisher Transform variants |

## 9. Constraints que respeto

- ❌ No edito `grail_loop.py`, `backtest.py`, `strategies/indicators.py`,
  `canary_runner.py` (core compartido)
- ❌ No toco `strategies_v7/strategies_tv2_batch*.py` (dominio sandbox)
- ❌ No push -f, no reset --hard, no amend sin permiso
- ❌ No arranco más de 5 workers Python (hunter default) para no thrashar
- ❌ No corro `holdout_test.py` sobre pool existente (OOS contamination)
- ❌ No modifico `MAC_V8_COVERED_DEDUP_20260421.json` sin flock

## 10. Latido / heartbeat

No tengo un daemon persistente — cada turno del usuario es ~10min.
El hunter corre en foreground del turno, flushea progress cada 200 combos,
y commiteo al final del turno. Si se corta, resume desde `mac_m*_progress.json`.

---

*Generado por Mac paralela (Claude Code online) 2026-04-24.
Sandbox leerá en próximo pull.*
