# Instrucciones para sesión Claude Code paralela (Mac / online) — sin conflictos

Esta es la guía para que OTRA sesión de Claude Code (Mac o cualquier otra
instancia online) cace grails **en paralelo** con la sesión sandbox actual,
sin pisarse los archivos ni los commits.

**Sesión actual (sandbox)**: rama `claude/verify-trading-strategies-Fnf0P`,
corriendo en infra Anthropic `/home/user/bot2`. Esta sesión genera los
commits con prefix `ciclo watcher #N`.

---

## 1. Separación de ramas git

| Sesión | Rama | Nunca toca |
|--------|------|------------|
| Sandbox (esta) | `claude/verify-trading-strategies-Fnf0P` | `mac-hunter-*` |
| Mac paralela | **crear**: `claude/mac-hunter-parallel` | `claude/verify-trading-strategies-*` |

**Comandos Mac** (primer arranque):
```bash
git fetch origin
git checkout -b claude/mac-hunter-parallel origin/main
git push -u origin claude/mac-hunter-parallel
```

**Regla de oro**: ninguna sesión hace `git push -f` ni `merge` sobre la
rama de la otra. Integración futura por PR a `main`.

---

## 2. Separación de carpetas / archivos (dominio exclusivo)

### Sandbox (esta sesión) SOLO escribe en:
- `strategies_v7/strategies_tv2_batch36*.py` (3600-3699)
- `results/sandbox_*` (`sandbox_h1_*`, `sandbox_h2_*`, ..., `sandbox_master_ranking*`)
- `coordination/SANDBOX_*` y `coordination/SANDBOX_TO_MAC_*`
- `tools/hunter_runner.py`, `tools/rank_all_grails.py`,
  `tools/plateau_on_hunter_grails.py`, `tools/build_final_master.py`
- `tools/mc_c_tier_upgrade.py`, `tools/pbo_dsr_b_tier.py`,
  `tools/holdout_top10_sandbox.py`
- `logs/hunter1_*`, `logs/hunter2_*`, ..., `logs/hunter7_*`

### Mac paralela SOLO escribe en:
- `strategies_v7/strategies_mac_batch37*.py` (3700-3799)
- `results/mac_*` (`mac_h1_*`, `mac_h2_*`, ..., `mac_master_ranking*`)
- `coordination/MAC_*` y `coordination/MAC_TO_SANDBOX_*`
- `tools/mac_hunter_runner.py` (crear — copia de `hunter_runner.py`
  con rutas `mac_*`)
- `tools/mac_rank.py`, `tools/mac_plateau.py`, `tools/mac_build_master.py`
  (crear — copias con output a `mac_*`)
- `tools/emit_pine.py` (fix del bug trail — Mac owns según ACK 2026-04-21)
- `tools/mc_block_bootstrap.py` (crear — nuevo MC approved por Mac)
- `logs/mac_*`

### Ambas pueden LEER (nunca escribir):
- `data/candles/` (solo-lectura, compartido)
- `coordination/MAC_V8_COVERED_DEDUP_20260421.json` (set-diff list)
- `coordination/SANDBOX_FINAL_MASTER_V8_GRAILS.json` (Mac lee; sandbox escribe)
- `coordination/MAC_FINAL_MASTER_V8_GRAILS.json` (sandbox lee; Mac escribe — crear)
- `strategies_v7/strategies_tv2_batch35*.py` (3560-3599, legacy compartido)

---

## 3. Coordinación runtime (sin pisarse)

### Inbox/outbox archivos

| Dirección | Archivo | Regla |
|-----------|---------|-------|
| Sandbox → Mac | `coordination/SANDBOX_TO_MAC_<TAG>_<YYYYMMDD>.md` | Sandbox escribe, Mac lee |
| Mac → Sandbox | `coordination/MAC_TO_SANDBOX_<TAG>_<YYYYMMDD>.md` | Mac escribe, sandbox lee |

Ejemplo en curso:
- `coordination/MAC_TO_SANDBOX_HANDOFF_20260421.md` (Mac → Sandbox)
- `coordination/SANDBOX_TO_MAC_MED3_MED4_LOW5_20260421.md` (Sandbox → Mac)

### Dedup / set-diff

Antes de lanzar cualquier wave, la sesión que va a cazar **lee**
`coordination/MAC_V8_COVERED_DEDUP_20260421.json` (ya contiene 1,777 combos
cubiertos: 1,717 Mac V8 + 60 sandbox V8) y **filtra** su espacio de búsqueda
para NO repetir `(strategy, symbol, tf)` ya cubierto.

Cuando la sesión persiste su V8 local, **actualiza** el JSON de dedup
añadiendo sus nuevas combos. Por seguridad, primero hace:
```bash
python3 tools/update_dedup.py --add-source results/mac_master_ranking.json   # o sandbox_*
```
(crear ese tool; lock via `flock coordination/DEDUP.lock`).

---

## 4. Herramientas específicas a crear en Mac

### `tools/mac_hunter_runner.py`
Copia de `tools/hunter_runner.py` con cambios:
1. Prefixes de output: `results/mac_h*_*` en vez de `results/sandbox_h*_*`
2. Log file: `logs/mac_hunter*.log`
3. Lee la lista de batches desde `strategies_v7/strategies_mac_batch37*.py`
4. Invocación: `python3 tools/mac_hunter_runner.py --wave m1 --tfs 1h 4h 1d`
   (usar `m1`, `m2`, ... para distinguir de `h1`, `h2`, ... de sandbox)

### `tools/mc_block_bootstrap.py` (Mac owns por ACK)
Reemplaza `tools/mc_c_tier_upgrade.py` con **block bootstrap** en vez de
sign-shuffle:
```python
# Block bootstrap (no i.i.d. per-trade)
def block_bootstrap_mc(pnls, n_boot=5000):
    n = len(pnls)
    block_size = max(5, min(10, int(n ** 0.5)))
    observed = sum(pnls)
    count_extreme = 0
    for _ in range(n_boot):
        sample = []
        while len(sample) < n:
            start = np.random.randint(0, n - block_size + 1)
            sample.extend(pnls[start:start + block_size])
        sample = sample[:n]
        # shuffle signs within blocks for permutation test
        np.random.shuffle(sample)
        if sum(sample) >= observed:
            count_extreme += 1
    return count_extreme / n_boot
```
Input: `results/mc_c_tier_41_upgrades_20260421.json`
Output: `results/mc_block_bootstrap_c_tier_20260421.json`

### `tools/emit_pine.py` fix (Mac owns)
Bug de trail stop documentado en `results/pine_audit.md`. Fix esperado:
- Reemplazar `trail_price = high - N*atr` por highest-high tracking + trail_points/trail_offset
- Usar OCA strategy.exit groups para SL/TP/trail
- Submit fix en PR dedicado o commit aparte en `claude/mac-hunter-parallel`

---

## 5. Waves y nombres de batches (no repetir)

| Rango batch | Owner | TFs principales |
|-------------|-------|-----------------|
| 3560-3599 | Legacy compartido (no modificar) | todos |
| 3600-3699 | **Sandbox** (actual) | 1h/4h/1d + algunos 5m/15m |
| 3700-3799 | **Mac** (nuevo) | 1h/4h/1d + 5m/15m + new symbols |
| 3800+ | Reservado futuro | — |

Waves label:
| Sandbox | Mac |
|---------|-----|
| h1, h2, ..., h7 | m1, m2, ..., m7 |

---

## 6. Protocolo commit (evitar race conditions)

### Regla 1: `git pull --rebase` antes de cada commit
```bash
git add -A
git pull --rebase origin $(git branch --show-current)
git commit -m "..."
git push -u origin $(git branch --show-current)
```

### Regla 2: NUNCA commitear archivos cross-domain
Sandbox no debe tener `results/mac_*` staged; Mac no debe tener `results/sandbox_*` staged.

Auto-check antes de commit:
```bash
# Sandbox check (rechazar si hay archivos mac_*)
git diff --cached --name-only | grep -E "(mac_|MAC_[A-Z])" && \
  echo "ERROR: archivos del dominio Mac detectados" && git reset
```

### Regla 3: mensaje de commit con tag de sesión
- Sandbox: `ciclo watcher #N: ...` o `HIGH-X: ...`
- Mac: `mac-ciclo #N: ...` o `mac-HIGH-X: ...`

---

## 7. Archivo de sincronización de estado

Crear `coordination/PARALLEL_STATE.json` con lock `flock`:
```json
{
  "sandbox": {
    "session_id": "...",
    "branch": "claude/verify-trading-strategies-Fnf0P",
    "last_heartbeat": "2026-04-21T15:30:00Z",
    "pool_size": 791,
    "v8_count": 70,
    "active_waves": ["h2", "h3", "h5", "h7"]
  },
  "mac": {
    "session_id": "...",
    "branch": "claude/mac-hunter-parallel",
    "last_heartbeat": "2026-04-21T15:30:00Z",
    "pool_size": 0,
    "v8_count": 0,
    "active_waves": ["m1"]
  }
}
```

Cada sesión actualiza **solo su sub-objeto** con `flock` sobre el archivo.
Esto evita que las dos sesiones lancen hunters de la misma wave a la vez.

---

## 8. Qué NO hacer (para evitar conflictos)

- **Nunca** editar `grail_loop.py`, `backtest.py`, `strategies/indicators.py`
  sin avisar (son core compartido). Si hay que tocarlo, hacer PR a main y
  esperar review de la otra sesión.
- **Nunca** borrar archivos del dominio de la otra sesión.
- **Nunca** hacer `git push -f` sobre `main` ni sobre la rama de la otra
  sesión.
- **Nunca** modificar `coordination/MAC_V8_COVERED_DEDUP_20260421.json`
  sin tool `update_dedup.py` con flock.
- **Nunca** lanzar hunters sin antes leer el dedup JSON.
- **Nunca** correr `holdout_test.py` sobre grails OOS ya evaluados (OOS
  contamination).

---

## 9. Handshake inicial (primera vez que Mac arranca)

Mac debe:
1. `git fetch origin && git checkout -b claude/mac-hunter-parallel origin/main`
2. Leer `coordination/SANDBOX_TO_MAC_HANDOFF_20260421.md` (identity sandbox)
3. Leer `coordination/MAC_V8_COVERED_DEDUP_20260421.json` (qué evitar)
4. Leer `coordination/SANDBOX_FINAL_MASTER_V8_GRAILS.json` (pool actual)
5. Crear `coordination/MAC_PARALLEL_HANDOFF_<YYYYMMDD>.md` con:
   - Su identity (qué va a cazar, qué waves)
   - Primer batch a crear (ej. `batch3700`)
   - Fecha estimada de primer V8 local
6. Lanzar su primer `mac_hunter_runner.py --wave m1 --tfs 1h 4h 1d`

Sandbox (esta sesión) responderá con un `SANDBOX_TO_MAC_PARALLEL_ACK.md`
confirmando recepción.

---

## 10. Preguntas frecuentes

**¿Y si las dos sesiones encuentran el mismo grail?**
Pasa — mismo dataset, mismo algoritmo. El dedup post-caza via
`MAC_V8_COVERED_DEDUP_*.json` lo detecta. Se mantiene la copia con mejor
score.

**¿Puede Mac ejecutar `build_final_master.py` sobre el pool combinado?**
No. Cada sesión construye SU propio V8 (`SANDBOX_FINAL_MASTER_V8_GRAILS.json`
vs `MAC_FINAL_MASTER_V8_GRAILS.json`). El merge final se hace en un
tercer paso manual o en PR review.

**¿Puedo cambiar el branch name?**
Sí, pero avisar en el handshake inicial. Los nombres son convención.

**¿Qué pasa si hay conflicto de merge?**
Resolver a favor del archivo con `mtime` más reciente y confirmar con la
otra sesión vía `coordination/*_TO_*_*.md`.

---

*Documento generado por sandbox 2026-04-21. Aplicable hasta que ambas
sesiones acuerden un cambio vía coordination docs.*
