# CLAUDE.md — Instrucciones para futuras sesiones de Claude Code

## Misión del repo

**Cazar "grails"** = configuraciones de trading con WR > 60%, PnL positivo,
max DD < 30%, Profit Factor > 1 **y que no sean overfits**. Todo sobre
crypto 1h (ETH/ADA/DOGE/BTC), para Pine Script v5 en TradingView.

## Reglas fundamentales del usuario

1. **Todo se testea contra el historial completo del activo**, no sobre
   cortes caprichosos. Si un grail pasa in-sample pero colapsa en TV,
   es overfit — no pasa nunca.
2. **Cuando una estrategia se confirma, pasar a la siguiente familia**.
   El usuario *no* quiere 500 variantes de la misma estrategia (ej. no
   saturar rsi2_regime, aunque sea la que mejor funciona).
3. **Honestidad en resultados.** Si algo no funciona, decirlo. No inventar
   métricas, no meter shortcut a overfit, no fudge.
4. **Sin credenciales externas**: el usuario ofreció login TradingView
   premium varias veces — rechazar siempre (política de seguridad).
5. **Commit + push después de cada cambio relevante**. El hook
   `stop-hook-git-check.sh` se queja si hay cambios sin subir. Rama:
   `claude/verify-trading-strategies-Fnf0P` en `goterobot-collab/bot2`.
6. **Idioma**: usuario habla español; responder en español.
7. **Concisión**: respuestas cortas. Sin emojis en código/archivos.

## Entorno de ejecución

- Sandbox Linux 16 cores + 21GB RAM + 0 swap. **No es la Mac del usuario.**
  Si pregunta "¿usas mi CPU?" — NO, es infra Anthropic en `/home/user/bot2`.
- Todas las APIs de exchange (Binance, Bybit, Yahoo, Polygon) dan 403.
  **Única fuente de datos que funciona**: `codeload.github.com` fetcheando
  `alimohammadiamirhossein/CryptoPredictions`. Ver
  `tools/convert_cryptopredictions.py`.
- CSVs en `data/*.csv` (gitignored). Cada fetch local regenera.

## Pipeline anti-overfit (crítico)

El pipeline es **secuencial**: un grail tiene que pasar TODAS las etapas.

| # | Etapa | Herramienta | Criterio pass |
|---|-------|-------------|---------------|
| 1 | Random search | `grail_loop.py` | WR>60 + PnL+ + DD<30 + PF>1 + trades≥50 |
| 2 | Walk-forward 50/50 | `grail_loop.py` (integrado) | Pasa filtro en full + h1 + h2 |
| 3 | Plateau ±20% jitter | `tools/plateau_test.py` | ≥75% de vecinos siguen pasando |
| 4 | Monte Carlo 500 perm | `tools/monte_carlo.py` | p < 0.01 |
| 5 | Meta-labeling opcional | `tools/meta_label.py` | Mejora WR y PF |
| 6 | PBO (López de Prado) | `tools/pbo.py` | PBO < 0.5 |
| 7 | DSR (Bailey-LdP) | `tools/dsr.py` | DSR > 0.95 |
| 8 | Locked OOS 20% holdout | `tools/holdout_test.py` | Sobrevive (test una sola vez) |

Estado actual (último snapshot): 1093 grails WF → 13 plateau → 3 pasan MC p<0.01.

## Familias de estrategia en SPACES

Archivo: `grail_loop.py` líneas ~172-317 + `strategies/tsmom.py` (plugin).

- **Confirmadas** (weight=1 en sampling): `rsi2_regime`, `bb_trend_rejoin`, `zscore_revert`
- **Weight=4**: `supertrend_atr`, `ema_cross_trend`, `macd_trend`, `donchian_trail`
- **Weight=8 (unexplored priority)**: `keltner_squeeze`, `psar_trend`, `adx_pullback`, `tsmom_voltarget`

Si el usuario pide cazar una NUEVA familia, la regla es:
1. Añadir el signal builder a `strategies/<name>.py` o inline en `grail_loop.py`.
2. Registrar en `SPACES[...]` con `{"sig": fn, "params": grid, "exit": grid}`.
3. Mover a `unexplored` en el sampling para que reciba weight=8.
4. Relanzar workers.

## Comandos típicos

```bash
# Lanzar 16 workers al 95% CPU (satura el box — no lanzar más).
for i in $(seq 1 16); do
  seed=$((RANDOM))
  nohup python3 grail_loop.py --iter 200000 --seed $seed \
    --report-every 50000 > /tmp/wf_$i.log 2>&1 &
done

# Matar todos los workers antes de cambiar SPACES/pesos.
pkill -f "grail_loop.py --iter"

# Re-rankear grails (dedup + score compuesto).
python3 tools/rank_grails.py

# Plateau test sobre el pool actual.
python3 tools/plateau_test.py

# Emitir Pine v5 de los top grails.
python3 tools/emit_pine.py
```

## Archivos / directorios

```
grail_loop.py              # motor principal random-search + WF
backtest.py                # simulator (intrabar SL/TP), metrics_from_trades
strategies/
  indicators.py            # ema, sma, rsi, macd, bbands, stoch, supertrend, atr
  library.py               # 13 estrategias con exit config
  tsmom.py                 # TSMOM + vol-target (plugin agregado por agente)
  pairs_eth_btc.py         # pairs standalone (no está en SPACES)
tools/
  convert_cryptopredictions.py  # fetch data via codeload.github.com
  rank_grails.py           # dedup + score + top_grails.md
  emit_pine.py             # auto-emit Pine v5 desde grails_loop_wf.jsonl
  plateau_test.py          # ±20% jitter stability
  monte_carlo.py           # permutation test
  meta_label.py            # LightGBM wrapper de plateau #1
  pbo.py                   # Probability of Backtest Overfitting
  dsr.py                   # Deflated Sharpe Ratio
  holdout_test.py          # locked OOS 20% (no correr hasta pipeline listo)
results/
  grails_loop_wf.jsonl     # GITIGNORED (>50MB) — cada línea una iter
  grails_loop_wf.md        # grails apendizados (human-readable)
  top_grails.md            # ranking dedup (regenerar con rank_grails.py)
  plateau_report.md        # plateau survivors (primer batch 115)
  plateau_full_pool.md     # plateau survivors (pool completo — agente lo corre)
  pine_audit.md            # audit de Pine emitidos (2 bugs críticos flagged)
  pbo_report.md, dsr_report.md, monte_carlo_report.md, meta_label_report.md
  funding_carry_roadmap.md # roadmap funding-rate carry
  locked_oos_proposal.md   # diseño 40/40/20 split
  ANTI_OVERFIT_RESEARCH.md # bibliografía completa
pine_sources/grails/*.pine # 51+ estrategias emitidas, una por grail top
```

## Bugs conocidos (al 2026-04-17)

1. **`tools/emit_pine.py` emite Pine con trail roto**: usa
   `trail_price=high - N*atr` (sin highest-high tracking) y dos
   `strategy.exit` sin OCA. Ver `results/pine_audit.md` para el fix.
2. **`exits_by_reason` bug**: ver `strategies/library.py`.
3. **`sig_psar_trend` tenía import roto**: ya arreglado.
4. El usuario reportó que el grial #1 en TV dio **DD 37% sobre 375 trades**
   (vs 13% in-sample) — síntoma clásico de sizing 100% equity + falta de
   filtro de régimen. Fix: `pine_sources/grails/grail_001_v4_tight_dd.pine`
   (risk-based sizing 0.5%, daily EMA200 filter, circuit breaker, halt
   si DD > 12%).

## Cómo testear estrategias del usuario

Si el usuario pega código Pine o Python o describe una estrategia en palabras:

1. **Traducir** la lógica a un signal builder Python
   `sig_xxx(df, p) -> (entry_bool_series, exit_bool_series)`.
2. **Registrar** en `SPACES["xxx"]` con un grid razonable de params y exits.
3. **Agregar** a `unexplored` en `main()` para prioritario.
4. **Lanzar** un worker con seed nuevo — dejar correr ~100k iter.
5. **Reportar**: WR/PF/Ret/DD de los mejores, y si pasan plateau.
6. **Emitir Pine** si sobrevive plateau → archivo en `pine_sources/grails/`.

## Decisiones de diseño (por qué esto así)

- **Walk-forward 50/50** (no 60/20/20 aún) porque el usuario necesitaba
  algo rápido y las dos mitades son diagnóstico fuerte. El 40/40/20
  está diseñado (`results/locked_oos_proposal.md`) pero no aplicado para
  no invalidar los 2M evals ya corridos.
- **No mlfinlab/skfolio/optuna**: el sandbox no siempre tiene las libs
  y las implementaciones custom de PBO/DSR/MC son cortas (<200 líneas
  cada una) y auditables.
- **Score compuesto** (ret × PF × trade-pen × DD-pen × consistency) en
  `rank_grails.py` — no es Sharpe, es una función de utilidad que refleja
  preferencias del usuario: **WR alto, DD bajo, consistencia h1/h2**.

## Qué NO hacer

- No dejes `results/grails_loop_wf.jsonl` tracked en git (>50MB → rechazo GH).
- No lances más de 16 workers `grail_loop.py` simultáneos (16 cores = thrash).
- No modifiques `grail_loop.py` mientras hay workers corriendo — importarán
  el código nuevo al reiniciar, no en vivo; pero si uno crashea y se
  relanza, la divergencia de SPACES rompe el pool.
- No uses `git --amend` / `git reset --hard` / `git push -f` sin permiso.
- No delegues a un subagente la MISMA tarea que estás haciendo tú.
- No metas emojis en archivos (tampoco en commits). Usuario no los pidió.
