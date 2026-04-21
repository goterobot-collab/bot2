# SANDBOX → MAC HANDOFF — Quien soy y qué hago

**De:** Claude Opus 4.7 sandbox (`/home/user/bot2`, Linux 16c+21GB)
**Para:** Claude Code (Mac del usuario)
**Rama compartida:** `claude/verify-trading-strategies-Fnf0P` en `goterobot-collab/bot2`
**Fecha snapshot:** 2026-04-21 00:30 UTC

---

## Identidad y rol

- Soy una **instancia separada** de Claude que corre en infraestructura
  Anthropic (sandbox `/home/user/bot2`). NO soy tu compañera local.
- **NO comparto memoria entre turnos**: cada mensaje del usuario me reactiva
  desde un context fresco; lo que sé es lo que está commiteado en el repo.
- **NO tengo acceso a APIs de exchange** (todas dan 403). Única fuente de
  datos: `codeload.github.com` → `alimohammadiamirhossein/CryptoPredictions`
  via `tools/convert_cryptopredictions.py`.
- **Sandbox NO persiste entre mis turnos**: nohup/disown procesos se mueren.
  Solo cazo cuando estoy activa (~10min/turno).

## División de trabajo entre nodos

| Nodo | Rol primario |
|------|--------------|
| **Sandbox (yo)** | Random search + walk-forward + plateau + tiering V8. Genera grails. |
| **Mac (tú)** | Validación pesada (MC ampliada, PBO, DSR, holdout OOS). Emite Pine. Dueño de la verdad última. |
| **Hetzner** | Persistencia 24/7 cuando se solicita aparte. |

Tu rol crítico: **antes de promover un grail mío a producción**, correr
contra TV/holdout. Mis grails son candidatos, no truth.

## Pipeline anti-overfit (estado actual)

| # | Etapa | Estado en este snapshot |
|---|-------|-------------------------|
| 1 | Random search (HUNTER waves h1-h7) | **Saturado en 1h/4h/1d** |
| 2 | Walk-forward 50/50 | Integrado en hunter, todos pasan |
| 3 | Plateau ±20% jitter (k=25) | 19 grails confirmados (B-tier) |
| 4 | Monte Carlo 500 perm | Pendiente sobre los 41 C-tier sin MC |
| 5 | Meta-labeling | No corrido para H6/H7 |
| 6 | PBO (López de Prado) | Pendiente |
| 7 | DSR (Bailey-LdP) | Pendiente |
| 8 | Locked OOS 20% holdout | NO TOCAR aún (single-shot) |

## Olas (waves) cazadas hasta hoy

| Wave | Batches | TFs | Estado | Grails brutos |
|------|---------|-----|--------|---------------|
| H1 | 3566–3569 | 1h/4h/1d | 100% | 102 |
| H1 | 3566–3569 | 15m/5m | ~47% | 22 (en curso) |
| H2 | 3594/3598 | 1h/4h/1d | 100% | 294 |
| H2 | 3594/3598 | 5m/15m | ~50% | 7 (en curso) |
| H3 | 3601–3603 | 1h/4h/1d | 100% | 114 |
| H3 | 3601–3603 | 5m/15m | 100% | 7 |
| H4 | 3607 (crypto) | 1h/4h/1d | 100% | 26 |
| H4 | 3607 | 5m/15m | 100% | 0 |
| H5 | 3604–3606 (fractal/Ehlers/regression) | 1h/4h/1d | 100% | 70 |
| H5 | 3604–3606 | 5m/15m | parcial | 3 |
| H6 | 3608 (crypto regime/microstructure) | 1h/4h/1d | 100% | 35 |
| H6 | 3608 | 5m/15m | 100% | 0 |
| H7 | 3609 (RSI div, BB squeeze, vol spike, double MACD, OBV) | 1h/4h/1d | 100% | 22 |
| H7 | 3609 | 5m/15m | 100% | 0 |

**Patrón crítico observado:** TODAS las waves en 5m/15m dan ≤7 grails. La
estructura de TFs cortos no soporta filtros n≥30 + WR≥65% + PF≥1.3.
**Recomendación:** invertir solo en 1h/4h/1d para futuras waves.

## Estado actual (snapshot)

- **Pool total post-dedup:** 721 grails únicos
- **V8 FINAL MASTER:** 60 (A=0, B=19, C=41)
- **Hunters vivos al hacer este doc:** variable (mueren entre mis turnos)

## V8 completo (60 grails)

EL JSON completo está en `coordination/SANDBOX_FINAL_MASTER_V8_GRAILS.json`.
Schema por grail:
```
{strategy, symbol, tf, wr, trades, pf, source, plateau_confirmed,
 score, mc_p_value, tier, validation, ready_for_v8}
```

### Top 60 ordenado por score compuesto

Score = WR × log(1+trades) × min(PF,10) × (1.5 si plateau)

| # | Tier | Strategy | Sym | TF | WR | n | PF | Score | Plateau | MC p |
|---|------|----------|-----|----|----|----|-----|-------|---------|------|
| 1 | B | `TV_Linear_Regression_Channel` | AGT | 4h | 95.0% | 20 | 205.25 | 4338 | Y | — |
| 2 | C | `TV_Gann_Swing_MultiLayer` | NEAR | 1d | 75.4% | 57 | 9.02 | 4142 |  | — |
| 3 | B | `TV_Pivot_Reversal_Backtest` | SWARMS | 1d | 80.6% | 36 | 8.48 | 3702 | Y | — |
| 4 | B | `TV_ABCD_Pattern_Daveatt` | NEAR | 1d | 85.0% | 20 | 8.99 | 3490 | Y | 0.462 |
| 5 | B | `TV_Double_Top_Bottom` | JTO | 4h | 85.0% | 20 | 10.84 | 2588 | Y | — |
| 6 | B | `TV_Pivot_Reversal_Backtest` | GMX | 1d | 71.5% | 123 | 4.54 | 2347 | Y | — |
| 7 | C | `TV_ABCD_Pattern_Daveatt` | SUI | 4h | 78.4% | 37 | 8.17 | 2331 |  | — |
| 8 | B | `TV_Pivot_Reversal_Backtest` | ONDO | 1d | 75.3% | 89 | 4.37 | 2222 | Y | — |
| 9 | C | `TV_ABCD_Pattern_Daveatt` | PYTH | 1d | 78.8% | 33 | 7.97 | 2214 |  | — |
| 10 | B | `TV_Pivot_Reversal_Backtest` | LINK | 1d | 71.1% | 173 | 3.97 | 2184 | Y | — |
| 11 | C | `TV_Linear_Regression_Channel` | GMX | 1d | 86.7% | 30 | 6.54 | 1947 |  | — |
| 12 | C | `TV_Cointegration_PairsTrading` | JTO | 4h | 76.6% | 47 | 5.81 | 1723 |  | — |
| 13 | B | `TV_Double_Top_Bottom` | PYTH | 4h | 82.4% | 34 | 3.90 | 1715 | Y | — |
| 14 | C | `TV_Session_Vol_Regime` | PYTH | 1d | 73.5% | 34 | 6.45 | 1686 |  | — |
| 15 | C | `TV_ABCD_Harmonic_BullBear` | JTO | 4h | 72.3% | 65 | 5.29 | 1603 |  | — |
| 16 | C | `TV_Linear_Regression_Channel` | ARB | 4h | 83.3% | 30 | 5.46 | 1562 |  | — |
| 17 | C | `TV_Double_Top_Bottom` | ONDO | 4h | 72.4% | 76 | 4.92 | 1546 |  | — |
| 18 | C | `TV_Cointegration_PairsTrading` | JUP | 4h | 83.3% | 48 | 4.63 | 1501 |  | — |
| 19 | C | `TV_ABCD_Pattern_Daveatt` | DYDX | 4h | 71.7% | 113 | 4.28 | 1452 |  | — |
| 20 | B | `TV_AlternatingSignals_MultiTFConfirm` | WLD | 1d | 75.0% | 20 | 3.78 | 1295 | Y | 0.440 |
| 21 | B | `TV_Cointegration_PairsTrading` | AGT | 4h | 79.2% | 48 | 2.69 | 1244 | Y | 0.478 |
| 22 | B | `TV_Pivot_Reversal_Backtest` | JUP | 1d | 71.7% | 113 | 2.35 | 1197 | Y | — |
| 23 | C | `TV_Footprint_Volume_Climax` | SUI | 4h | 76.5% | 34 | 4.32 | 1175 |  | — |
| 24 | C | `TV_DOW_Trend_Filter` | ONDO | 1d | 74.2% | 31 | 4.21 | 1083 |  | — |
| 25 | B | `TV_FairValueGap_Trading` | JUP | 1d | 76.2% | 21 | 3.00 | 1060 | Y | 0.610 |
| 26 | C | `TV_BTC_Beta_Residual` | JUP | 1d | 76.5% | 34 | 3.50 | 952 |  | — |
| 27 | B | `TV_VolumeProfile_Reversion` | SFP | 1d | 82.1% | 234 | 1.36 | 914 | Y | 0.574 |
| 28 | B | `TV_VolumeProfile_Reversion` | AGT | 1d | 82.1% | 39 | 2.00 | 909 | Y | 0.436 |
| 29 | C | `TV_Linear_Regression_Channel` | ONDO | 4h | 78.4% | 51 | 2.86 | 886 |  | — |
| 30 | C | `TV_SR_Trendlines` | TIA | 4h | 73.3% | 30 | 3.46 | 872 |  | — |
| 31 | B | `TV_VolumeProfile_Reversion` | SWARMS | 1d | 78.0% | 50 | 1.88 | 865 | Y | 0.580 |
| 32 | C | `TV_OBV_Momentum` | XRP | 1d | 73.5% | 34 | 3.23 | 844 |  | — |
| 33 | C | `TV_Cointegration_PairsTrading` | ARB | 4h | 72.2% | 54 | 2.90 | 839 |  | — |
| 34 | C | `TV_SR_Trendlines` | DYDX | 4h | 70.0% | 30 | 3.38 | 814 |  | — |
| 35 | C | `TV_BreaksAndRetests` | ARB | 1d | 71.4% | 35 | 3.15 | 806 |  | — |
| 36 | B | `TV_Cointegration_PairsTrading` | LINK | 1d | 80.0% | 35 | 1.78 | 765 | Y | 0.566 |
| 37 | C | `TV_Linear_Regression_Channel` | GMX | 4h | 86.0% | 43 | 2.30 | 749 |  | — |
| 38 | C | `TV_Linear_Regression_Channel` | NEAR | 4h | 73.9% | 88 | 2.16 | 716 |  | — |
| 39 | B | `TV_Cointegration_PairsTrading` | ONDO | 4h | 77.8% | 63 | 2.16 | 699 | Y | — |
| 40 | C | `TV_MACD_Divergence_MTF_EMA` | APT | 1d | 71.0% | 31 | 2.77 | 682 |  | — |
| 41 | C | `TV_VolumeDelta_Imbalance` | JUP | 4h | 75.0% | 40 | 2.43 | 677 |  | — |
| 42 | C | `TV_CVD_Crossover` | GMX | 1d | 70.6% | 34 | 2.56 | 643 |  | — |
| 43 | C | `TV_Linear_Regression_Channel` | JTO | 4h | 73.0% | 89 | 1.95 | 641 |  | — |
| 44 | C | `TV_ATR_Regime_Reversal` | ARB | 1d | 73.3% | 30 | 2.52 | 634 |  | — |
| 45 | C | `TV_Cointegration_PairsTrading` | SWARMS | 4h | 79.2% | 48 | 2.01 | 620 |  | — |
| 46 | C | `TV_VolumeDelta_Imbalance` | SUI | 4h | 78.0% | 41 | 2.09 | 609 |  | — |
| 47 | C | `TV_Linear_Regression_Channel` | DYDX | 4h | 77.7% | 166 | 1.49 | 593 |  | — |
| 48 | C | `TV_Cointegration_PairsTrading` | SFP | 4h | 72.0% | 493 | 1.31 | 585 |  | — |
| 49 | B | `TV_Linear_Regression_Channel` | SFP | 4h | 74.3% | 564 | 1.23 | 579 | Y | — |
| 50 | C | `TV_Cointegration_PairsTrading` | NEAR | 4h | 70.0% | 80 | 1.84 | 566 |  | — |
| 51 | C | `TV_VolumeDelta_Imbalance` | OP | 1d | 75.0% | 56 | 1.76 | 534 |  | — |
| 52 | C | `TV_AdaptiveFisherizedCMO` | OP | 1d | 71.2% | 52 | 1.86 | 527 |  | — |
| 53 | C | `TV_Linear_Regression_Channel` | SUI | 4h | 78.3% | 115 | 1.38 | 514 |  | — |
| 54 | C | `TV_SR_Trendlines` | WLD | 1h | 78.9% | 38 | 1.76 | 509 |  | — |
| 55 | C | `TV_Cointegration_PairsTrading` | INJ | 4h | 70.6% | 51 | 1.68 | 469 |  | — |
| 56 | B | `TV_VolumeProfile_Reversion` | TIA | 1d | 77.5% | 111 | 1.27 | 464 | Y | 0.560 |
| 57 | C | `TV_DominantCycleLibrary` | SUI | 1d | 70.0% | 30 | 1.82 | 438 |  | — |
| 58 | C | `TV_Linear_Regression_Channel` | LINK | 1d | 76.8% | 56 | 1.40 | 435 |  | — |
| 59 | C | `TV_Linear_Regression_Channel` | OP | 4h | 72.2% | 72 | 1.32 | 409 |  | — |
| 60 | C | `TV_Weekend_Gap_Fade` | NEAR | 4h | 74.2% | 31 | 1.54 | 396 |  | — |


## Familias de estrategia activas

Ubicadas en `strategies_v7/strategies_tv2_batchXXXX.py`:

- **3566-3569 (h1):** Pivot/SR/Trendlines/Patterns clásicos TradingView
- **3594/3598 (h2):** ABCD/Harmonic/MultiTF/Cointegration
- **3601-3603 (h3):** Volume/Footprint/CVD/Order Flow
- **3604-3606 (h5):** Hurst/DFA/Ehlers DSP/Linear Regression Channel
- **3607 (h4):** Crypto-microstructure (Funding-proxy, Session-vol)
- **3608 (h6):** TV_BTC_Beta_Residual, TV_CumVol_Imbalance,
  TV_ATR_Regime_Reversal, TV_RealizedVol_Anchor, TV_Momentum_Acceleration
- **3609 (h7):** TV_RSI_Divergence, TV_BB_Squeeze_Breakout,
  TV_VolumeSpike_Reversal, TV_Double_MACD, TV_OBV_Momentum

Todas pickle-safe, sin lambdas, `.shift(1)` en señales.

## Optimizaciones del hunter

- **Early-abort:** si primeros 5 trials dan 0 trades → aborta los 15
  restantes (~30% wall time saved).
- **Warmstart:** lee progress.json de h1-h6 anteriores y carga 465 params
  conocidos buenos como trials 0-2 (verificado: hits con trial 0).
- **WORKERS=5** procesos · **TRIALS_PER_COMBO=20** · **COMBO_TIMEOUT_S=30**
- **PROGRESS_EVERY=200** combos → flush a progress.json para resume.

## Símbolos en universo

24 cryptos: AGT, APT, ARB, AVAX, DYDX, ETH, GMX, INJ, JTO, JUP, LINK, LTC,
NEAR, ONDO, OP, PENDLE, PYTH, SEI, SFP, SUI, SWARMS, TIA, WLD, XRP.

## Cómo me llamas

- Edits en `claude/verify-trading-strategies-Fnf0P` → me llegan al hacer
  pull en mi próximo turno.
- Si quieres pedirme algo: deja un `coordination/MAC_REQUEST_<fecha>.md`
  con la tarea. Lo leo al próximo /loop si el usuario me pinge.
- Si necesitas que NO toque algo: pon `LOCK_<archivo>` en coordination/.

## Lo que NO hago

- ❌ Push a otras ramas que no sean `claude/verify-trading-strategies-Fnf0P`
- ❌ `git --amend / reset --hard / push -f` sin permiso del usuario
- ❌ Tracked files >50MB (rechazo GitHub) — `.jsonl` quedan gitignored
- ❌ >16 workers `grail_loop.py` simultáneos
- ❌ Crear emojis en archivos (preferencia del usuario)

## Próximos pasos sugeridos para ti (Mac)

1. **MC ampliada** sobre los 41 C-tier sin `mc_p_value` → upgrade a B.
2. **PBO + DSR** sobre los 19 B con plateau pero sin MC.
3. **Holdout OOS 20%** sobre top-10 score (single-shot, NO repetir).
4. **Emit Pine** de top-20 con `tools/emit_pine.py` (ojo bug trail roto,
   ver `results/pine_audit.md`).
5. Revisar el grail H6 `TV_BTC_Beta_Residual JUP 1d` y H7 `TV_OBV_Momentum
   XRP 1d` que entraron directos a V8 — son las contribuciones nuevas.

---
*Generado por sandbox al pedido del usuario; sincroniza al hacer pull.*
