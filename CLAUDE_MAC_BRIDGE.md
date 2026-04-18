# Puente entre Claude Code (Mac) ↔ Claude Code (sandbox Anthropic)

**Contexto para el Claude que corre en la Mac del usuario:**

Hay otro Claude (yo) corriendo en un sandbox de Anthropic en el repo
`goterobot-collab/bot2`, rama `claude/verify-trading-strategies-Fnf0P`.
Ese Claude:

- Busca "grails" de trading con WR>60%, PnL+, DD<30%, PF>1
- Tiene 34 familias (11 novel no presentes en V8 + 23 V8-covered desactivadas)
- Solo puede fetchear datos via `codeload.github.com` y `raw.githubusercontent.com`
  (Binance/Kraken/CoinGecko/CryptoDataDownload todos 403)
- Corre con 12 workers paralelos en 16 cores / 21 GB RAM
- NO tiene acceso a Binance directo → no puede descargar 5m/15m de ETH/ADA/DOGE

Vos (Claude en la Mac) corrés sobre el V8 bot del usuario que:

- Testea sobre **Binance 17GB / 122M filas** (todo el history real)
- Tiene **24,431 grails** ya probados con gap WR real vs Optuna
- Tiene 5m/15m/1h/4h/1d en muchos activos Binance (BTC/ETH/ADA/DOGE/SOL/…)
- Corre bots vivos con métricas reales desde V8 epoch 2026-04-16

## El problema que resuelve este puente

El sandbox Anthropic está limitado en datos (sólo 1h + BTC 5m/15m).
Los grails que descubre ahí tienen que validarse sobre la historia REAL
(Binance 122M filas) del V8 antes de que el usuario los tire al bot.

Al revés: el V8 ya tiene 24k grails testeados; el sandbox puede hacer
**ingeniería anti-overfit sobre ellos** (plateau ±20% jitter, Monte Carlo
permutación, PBO/DSR, meta-labeling) para separar los 2,612 CONFIRMED
(gap≤5pp) de los overfits.

## Cómo pasarme backtests

Opción A (preferida): **rama dedicada en el mismo repo**

```bash
# 1. Exportás el backtest V8 a un JSONL con schema conocido
#    Cada línea = un backtest, formato:
#    {
#      "strategy":   "ConnorsRSI",
#      "symbol":     "ETHUSDT",
#      "tf":         "1h",
#      "params":     {"rsi_len":2, "oversold":10, ...},
#      "exit":       {"sl_pct":2.0, "tp_pct":3.0, ...},
#      "trades":     237,
#      "wr":         68.3,
#      "pf":         1.42,
#      "ret_pct":    84.1,
#      "dd_pct":    -18.2,
#      "sharpe":     1.85,
#      "gap_wr_pp":  4.2,        # gap WR optuna vs real, negativo = real>optuna
#      "verdict":    "CONFIRMED", # CONFIRMED|WARNING|INFLATED
#      "start_ts":   1650000000,
#      "end_ts":     1776388200,
#      "fees_rt":    0.003,      # round-trip fees aplicadas
#      "trades_real":[...opcional: lista de [entry_ts, exit_ts, pnl_pct]...]
#    }

# 2. Pusheás ese JSONL a una rama puente:
cd ~/bot2  # o donde esté el repo clonado en la Mac
git checkout -b v8/handoff   # solo la primera vez
cp /path/to/V8/export.jsonl results/v8_backtest_snapshot.jsonl
# NO commitees archivos grandes si > 50MB: gzipeá primero
gzip -k results/v8_backtest_snapshot.jsonl
git add results/v8_backtest_snapshot.jsonl.gz
git commit -m "v8 handoff: $(date -u +%Y-%m-%d) snapshot"
git push origin v8/handoff

# 3. En tu próximo mensaje al sandbox, mencionás que hay un snapshot nuevo
#    en la rama v8/handoff. Yo lo leo via raw.githubusercontent.com.
```

Opción B: **gist público / HuggingFace dataset público**

Si el archivo es muy grande (>100MB) y no querés meterlo al repo:

```bash
# Subí a HuggingFace como dataset público (mi sandbox lee HF raw):
#   huggingface-cli upload goterobot/v8-backtests \
#       v8_backtest_snapshot.jsonl.gz --repo-type dataset
# Después en tu mensaje pasame la URL:
#   https://huggingface.co/datasets/goterobot/v8-backtests/resolve/main/v8_backtest_snapshot.jsonl.gz
```

**Importante**: el sandbox tiene bloqueado Binance/Kraken/CoinGecko/CDD,
pero **sí alcanza** huggingface.co si exponés el dataset como público.
(Verificar con una prueba antes — HF también puede dar 403 según endpoint.)

Opción C: **pastebin / raw.githubusercontent de tu propio repo público**

Si el volumen es chico (<1 MB) podés pegarlo directo en el mensaje al
sandbox en formato JSONL o CSV. Yo parseo inline.

## Qué formato me conviene

**Lo que me resuelve el 80%:**

```jsonl
{"strategy":"ConnorsRSI","symbol":"ETHUSDT","tf":"15m","params":{...},"exit":{...},"trades":237,"wr":68.3,"pf":1.42,"ret_pct":84.1,"dd_pct":-18.2,"verdict":"CONFIRMED","start_ts":1704067200,"end_ts":1776000000,"fees_rt":0.003}
{"strategy":"VWAP_Double","symbol":"BTCUSDT","tf":"4h",...}
```

**Lo que me deja hacer Monte Carlo / plateau:**

También incluí la lista de trades individuales (aunque sea resumida):

```jsonl
{"strategy":"ConnorsRSI","symbol":"ETHUSDT","tf":"15m",...,
 "trades_list":[
   [1704067200, 1704070800, 2.3],
   [1704074400, 1704078000, -1.1],
   ...
 ]}
```

Cada tupla = `[entry_unix_ts, exit_unix_ts, pnl_pct]`. Con eso puedo
reconstruir la curva de equity para permutation tests y PBO.

## Qué voy a hacer yo cuando me pases el dump

1. **Dedupe** (el ranking V8 tiene muchos duplicados por ensemble)
2. **Plateau test ±20%** sobre los CONFIRMED → me quedo con los
   que sobreviven ≥75% de los vecinos
3. **Monte Carlo 500 permutaciones** — descartar todo p>0.01
4. **PBO / DSR** para ajustar por multiple-testing
5. **Walk-forward OOS** locked 20% holdout sobre cada survivor
6. **Ranking compuesto**: PF × WR × 1/(1+DD) × trades_log con PBO<0.5
7. Devolver un `results/v8_reranked.md` con los top-N y emitir Pine v5
   listo para pegar en TV

Todo esto lo voy commiteando incrementalmente en la rama principal, vos
lo ves en GitHub y pedís lo que quieras re-testear.

## Cómo invocarme desde la Mac

Cuando termines de pushear el snapshot, lanzame una nueva sesión al
sandbox con un mensaje tipo:

> "Hay un nuevo snapshot V8 en la rama `v8/handoff` archivo
> `results/v8_backtest_snapshot.jsonl.gz` con N grails CONFIRMED.
> Fetch, dedup, plateau-test, MC y devolveme el top-20 re-rankeado."

Yo:

1. `git fetch origin v8/handoff`
2. `git checkout origin/v8/handoff -- results/v8_backtest_snapshot.jsonl.gz`
3. Corro los scripts `tools/plateau_test.py`, `tools/monte_carlo.py`,
   `tools/pbo.py`, `tools/dsr.py` adaptados al schema de V8
4. Commito `results/v8_reranked.md` a la rama principal
5. Te mando el resumen en un comment

## Qué NO me pases

- **Credenciales** (API keys Binance, token GitHub del usuario):
  yo trabajo con mi propio token scoped al repo
- **Base de datos entera**: mandá el snapshot ya filtrado por
  `verdict IN ('CONFIRMED','WARNING')` para no tragar 17 GB de ruido
- **Formato propietario** (pickle, parquet con dependencias raras):
  JSONL plano es lo más portable

## Convenciones compartidas

- **TF strings**: `"5m"`, `"15m"`, `"1h"`, `"4h"`, `"1d"` (como V8)
- **Symbol**: `"BTCUSDT"` para Binance, `"BTCUSD"` para Bitstamp — elegí uno
- **Timestamp**: unix seconds UTC (no ms, no strings)
- **Fees**: expresados como fracción round-trip (0.003 = 0.3% RT)
- **Strategy name**: snake_case del bot, aunque V8 usa CamelCase
  (no importa, yo mapeo)

## Qué me devolvés cada vez

Cuando te pida algo, contestame con:

- Qué hiciste (una línea)
- Archivo en el repo con resultados (path relativo)
- Si hay algo que no pudiste correr, por qué

Sin emojis. Sin "great question!". Sin explicaciones largas. Español,
conciso, directo al grano (instrucciones 1, 3, 6 del CLAUDE.md del repo).

## Info útil para vos

- Rama de desarrollo actual del sandbox: `claude/verify-trading-strategies-Fnf0P`
- Hook git: `stop-hook-git-check.sh` — si ve cambios sin subir se queja;
  commit y push después de cada paso
- No pusheés a `main` ni hagas force-push sin permiso del usuario
- Ver `CLAUDE.md` en el root del repo para las reglas completas

## Recursos del sandbox (capacidad real)

| Recurso | Cantidad |
|---------|----------|
| CPU | 16 vCPU @ 2.1 GHz |
| RAM | 21 GB (sin swap) |
| Disco | 30 GB disponibles |
| Throughput grail-hunt | ~300 it/s total (12 workers) |
| Uso actual del repo | 1.7 GB |

**Podés mandarme hasta ~20 GB de dump** (data OHLCV + backtests JSONL).
Por encima de eso procesalo en chunks o mandame diffs incrementales.

**Archivos > 50 MB**: NO los commitees al git. Subilos a un dataset
público de HuggingFace (el sandbox alcanza `huggingface.co/datasets/.../resolve/main/...`
si el repo es público) o a `codeload.github.com` si cabe en el repo.

## Cómo me indicás sobre qué operar

Cuando el usuario te pida lanzar una búsqueda dirigida, pasame un JSON
así de claro al sandbox:

```json
{
  "action": "hunt",
  "assets": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
  "tfs": ["5m", "15m"],
  "families": "novel",
  "min_trades": 50,
  "min_wr": 62,
  "max_dd": 25,
  "iterations": 500000,
  "workers": 12
}
```

Mapeo:
- `action`: `hunt` | `validate` (plateau+MC sobre lo que ya hay) | `rerank`
- `assets`: símbolos a cargar — tengo que tener los CSVs en `data/` antes,
  así que si son exóticos (SOLUSDT) me los pusheás primero o me mandás
  URL raw de GitHub/HF para fetch
- `tfs`: cualquier subset de `1m 5m 15m 30m 1h 4h 1d`
- `families`: `novel` (solo las 11 microestructura/state-space), `all` (34),
  o lista explícita `["hurst_regime","bvc_ofi"]`
- `min_trades`, `min_wr`, `max_dd`: filtros del WF grail filter
- `iterations`: total random samples (se divide entre workers)
- `workers`: 1-14 (dejo 2 para OS/monitoreo)

Yo arranco los workers con `--tfs 5m 15m --assets BTCUSDT ETHUSDT SOLUSDT`
y reporto cada 30 min.
