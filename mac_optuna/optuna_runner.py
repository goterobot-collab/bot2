#!/usr/bin/env python3
"""
optuna_runner.py — Runner orquestador para optuna_v7.py
Procesa 216.948 combos (537 símbolos × 202 estrategias × 2 TFs) en batches por símbolo.

Lógica:
1. Carga run_a_winners_std_1h4h.json → lista de combos
2. Agrupa combos por símbolo
3. Skip símbolo si TODOS sus combos están en done_combos del progress
4. Por cada símbolo pendiente: genera combos-file temporal → llama optuna_v7.py
5. Workers: 10 si 01:00 <= hora local < 09:00, sino 8
6. --max-minutes 30 por batch
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"

RUN_FILE      = DATA_DIR / "run_a_winners_std_1h4h.json"
PROGRESS_FILE = DATA_DIR / "optuna_v7_progress.json"
OPTUNA_SCRIPT = BASE_DIR / "optuna_v7.py"
LOG_FILE      = DATA_DIR / "optuna_runner_log.jsonl"

MAX_MINUTES_PER_BATCH = 30


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_workers() -> int:
    """10 workers en horario valle (01:00–08:59 local), 8 el resto."""
    h = datetime.now().hour
    return 10 if 1 <= h < 9 else 8


def load_done_combos() -> set:
    """Lee done_combos del progress global. Formato: 'estrategia|simbolo|tf'"""
    if not PROGRESS_FILE.exists():
        return set()
    with open(PROGRESS_FILE) as f:
        p = json.load(f)
    return set(p.get("done_combos", []))


def make_combo_key(c: dict) -> str:
    return f"{c['strategy']}|{c['symbol']}|{c['timeframe']}"


def log(msg: str, level: str = "INFO", extra: dict | None = None):
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "msg": msg,
    }
    if extra:
        entry.update(extra)
    line = json.dumps(entry, ensure_ascii=False)
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log("Runner iniciado", extra={"run_file": str(RUN_FILE)})

    # 1. Cargar combos
    with open(RUN_FILE) as f:
        data = json.load(f)
    all_combos: list[dict] = data["combos"]
    log(f"Combos totales en run_file: {len(all_combos)}")

    # 2. Agrupar por símbolo
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for c in all_combos:
        by_symbol[c["symbol"]].append(c)

    symbols_sorted = sorted(by_symbol.keys())
    log(f"Símbolos únicos: {len(symbols_sorted)}")

    # 3. Filtrar símbols ya completos
    done = load_done_combos()
    log(f"done_combos en progress: {len(done)}")

    pending_symbols = []
    for sym in symbols_sorted:
        combos_sym = by_symbol[sym]
        keys_sym   = {make_combo_key(c) for c in combos_sym}
        remaining  = keys_sym - done
        if remaining:
            pending_symbols.append((sym, combos_sym, len(remaining)))
        else:
            log(f"SKIP {sym} — todos sus {len(combos_sym)} combos ya están done", level="DEBUG")

    total_pending_combos = sum(r for _, _, r in pending_symbols)
    log(f"Símbolos pendientes: {len(pending_symbols)} | Combos pendientes: {total_pending_combos}")

    if not pending_symbols:
        log("Nada pendiente. Runner finalizado.", level="INFO")
        return 0

    # 4. Procesar símbolo a símbolo
    for idx, (sym, combos_sym, n_remaining) in enumerate(pending_symbols, 1):
        workers = get_workers()
        log(
            f"[{idx}/{len(pending_symbols)}] Procesando {sym} "
            f"({n_remaining} combos pendientes) — workers={workers}",
        )

        # Recargar done por si optuna_v7 actualizó el progress entre batches
        done = load_done_combos()
        combos_to_run = [c for c in combos_sym if make_combo_key(c) not in done]

        if not combos_to_run:
            log(f"SKIP {sym} — ya completado tras recarga de progress", level="DEBUG")
            continue

        # Escribir combos-file temporal
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"runner_{sym.replace('/', '_').replace(':', '_')}_",
            delete=False,
        ) as tf:
            tmp_path = tf.name
            json.dump(combos_to_run, tf, ensure_ascii=False)

        try:
            cmd = [
                sys.executable,
                str(OPTUNA_SCRIPT),
                "--combos-file", tmp_path,
                "--max-minutes", str(MAX_MINUTES_PER_BATCH),
                "--workers", str(workers),
            ]
            log(f"CMD: {' '.join(cmd)}", level="DEBUG")

            t0 = time.time()
            result = subprocess.run(cmd, cwd=str(BASE_DIR))
            elapsed = round(time.time() - t0, 1)

            if result.returncode == 0:
                log(f"OK {sym} en {elapsed}s", extra={"returncode": 0})
            else:
                log(
                    f"ERROR {sym} returncode={result.returncode} en {elapsed}s",
                    level="ERROR",
                    extra={"returncode": result.returncode},
                )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    log("Runner finalizado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
