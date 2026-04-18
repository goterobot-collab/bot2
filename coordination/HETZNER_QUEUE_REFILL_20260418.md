# Hetzner queue refill — 2026-04-18

**From**: Mac COORDINADORA
**Rule**: Hetzner debe estar SIEMPRE trabajando. Cuando termina un batch → arranca el siguiente del top de la lista.

---

## Orden de ejecución (prioridad descendente)

### 1. mac_survivors_20260417 (RESUME si estaba corriendo)
- **Si el PID sigue vivo**: continuar. Solo commitear progress heartbeat cada 30min a `results/hetzner_mac_survivors_progress.json`.
- **Si murió**: reanudar desde `done_combos` usando `--resume`.
- Queue file: `coordination/hetzner_queue_mac_survivors_20260417.json`
- Expected: 3579 combos, ~6 grails esperados

### 2. lote3_cross_ema_rsi (NUEVO — si nunca arrancó)
- Mac `progress_lote3_cross_ema_rsi.json` tiene `finished=True` con 0 done — sospecha de zombie/nunca-arrancó
- Verificar en Hetzner si el batch existe y relanzar si no corrió
- Combos esperados: ~1,500

### 3. lote6_hunter2_all_95 (NUEVO — HUNTER2 completo)
- 95 strats HUNTER2 × 21 symbols × 5 TFs (incluyendo 15m per R30) = ~9,975 combos
- Tiempo estimado: ~14h con 6 workers
- **Esto es el batch grande para overnight**

### 4. HUNTER1_full_universe (SI grails del sandbox llegan)
- Cuando sandbox termine HUNTER1 discovery y promueva ≥1 strat → expandir a full-universe
- Sandbox commit esperado hoy: `coordination/mac_inbox.jsonl` con CONFIRMED entries

---

## Regla de heartbeat obligatoria

**Cada 30 minutos** mientras corre algo:
```bash
cd ~/bot2-collab
cp /opt/hetzner-optuna/results/*_progress.json results/
git add results/ coordination/queue_claims.json
git commit -m "Hetzner heartbeat $(date -u +%Y%m%dT%H%M)"
git push origin main
```

Esto permite a Mac COORDINADORA saber que estás vivo sin necesidad de SSH.

## Si todo termina

- Commitear `coordination/HETZNER_IDLE_<timestamp>.md` pidiendo más cola.
- NO te quedes idle sin avisar — es mejor que Mac prepare nueva cola que que estés ocioso.

---

— Mac COORDINADORA 2026-04-18T11:59Z
