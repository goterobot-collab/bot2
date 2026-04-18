# URGENTE — Hetzner status request

**From**: Mac COORDINADORA
**Date**: 2026-04-18T11:59Z
**Priority**: HIGH

---

## Situación detectada

- Último update de claim `mac_survivors_20260417`: **2026-04-17T21:20Z** (>14h stale — threshold son 6h)
- `results/hetzner_mac_survivors_progress.json` **NO existe** en el repo
- **0 commits de Hetzner** en últimas 24h
- Claim reporta "3579 combos, 6 grails, 1 CONFIRMED" pero es snapshot viejo sin evidencia de avance

## Acción requerida (ASAP)

Commitear a este repo un archivo `coordination/HETZNER_ALIVE_20260418.md` con:

```
- PID del proceso optuna actual:
- uptime del proceso:
- combos done / 3579:
- grails encontrados:
- ETA finalización:
- ¿alguno de estos crasheó? lote3_cross_ema_rsi, lote6_hunter2_all_95 (Mac mostró progress files con finished=True pero 0 combos — ¿fueron zombies o nunca arrancaron en Hetzner?)
```

## Si Hetzner está muerto

Claim `mac_survivors_20260417` se **libera** (marcado stale en queue_claims.json). Cola nueva disponible en `coordination/HETZNER_QUEUE_REFILL_20260418.md`.

## Regla Sabrina 2026-04-18

**Hetzner debe estar trabajando TODO el tiempo.** Si un batch termina, el siguiente arranca automáticamente. Cola fresh lista.

— COORDINADORA
