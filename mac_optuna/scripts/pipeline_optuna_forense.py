#!/usr/bin/env python3
"""
pipeline_optuna_forense.py — Pipeline completo: Optuna → Forense → Gates → Produccion
======================================================================================

Flujo:
  1. Lee combos pendientes (strategy × symbol × tf) que NO estan en progress
  2. Corre Optuna V7 secuencialmente por estrategia → activo → TF
  3. Cada grail encontrado pasa por forensic_backtest.py automaticamente
  4. Aplica Gates Consenso + Risk Score 3 capas
  5. Si aprueba → lo inyecta al JSON de produccion

Uso:
  python3 scripts/pipeline_optuna_forense.py                    # Todo
  python3 scripts/pipeline_optuna_forense.py --strategy B5_MA_Envelope_3  # Una sola
  python3 scripts/pipeline_optuna_forense.py --dry-run           # Solo ver que haria
  python3 scripts/pipeline_optuna_forense.py --max-hours 8       # Limite tiempo

2026-04-10 — Sabrina
"""
import os
import sys
import json
import time
import sqlite3
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
PROJECT_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
DATA_DIR = PROJECT_DIR / "data"
CANDLES_DB = Path("/Users/sabrina/CLAUDE CODE/data/activos_binance.db")
PROGRESS_FILE = DATA_DIR / "optuna_v7_progress.json"
BOT_V7_DIR = Path("/Users/sabrina/CLAUDE CODE/BOT V7")
PRODUCTION_JSON = BOT_V7_DIR / "live" / "data" / "v6_optimized_sl_tp.json"
FORENSIC_RESULTS = DATA_DIR / "forensic_backtest_results.json"
FORENSIC_RISK = DATA_DIR / "forensic_risk_scores.json"
PIPELINE_LOG = DATA_DIR / "pipeline_optuna_forense_log.json"
OPTUNA_SCRIPT = PROJECT_DIR / "optuna_v7.py"
FORENSIC_SCRIPT = SCRIPTS_DIR / "forensic_backtest.py"

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
# Estrategias ganadoras ordenadas por prioridad (forensic AvgPF)
WINNING_STRATEGIES = [
    "B5_MA_Envelope_5",    # AvgPF 2.74, Top 1 risk score
    "B5_MA_Envelope_3",    # AvgPF 2.11, 6 grails aprobados
    "B2_LinReg_MeanRev",   # AvgPF 1.85, 1 grail aprobado
    "B2_MeanRev_EMA21",    # AvgPF 1.81, 3 grails aprobados
    "B6_Rubber_Band_3",    # AvgPF 1.66, 2 grails aprobados
    "B6_Rubber_Band_5",    # AvgPF 1.60, 3 grails aprobados
    "B2_Rubber_Band",      # AvgPF 1.52, 2 grails aprobados
]

# TFs en orden de prioridad (5m tiene los mejores grails: SOL, ALGO, XRP)
TF_PRIORITY = ["5m", "1h"]

# Minimo de candles para considerar un symbol
MIN_CANDLES = 1000

# Workers para Optuna
OPTUNA_WORKERS = 8

# Trials por combo
OPTUNA_TRIALS = 50

# Gates consenso (de GATES_CONSENSO_FORENSE.md)
GATE_MIN_PNL = 0
GATE_MIN_TRADES = 20
GATE_MIN_WR = 60.0
GATE_MAX_DD = 90.0
GATE_MIN_PF = 1.15
GATE_GAP_HARD = -20.0
GATE_GAP_SOFT = -15.0
GATE_GAP_PF_THRESHOLD = 1.5


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_progress():
    """Load done combos from progress file."""
    if not PROGRESS_FILE.exists():
        return set()
    with open(PROGRESS_FILE) as f:
        data = json.load(f)
    return set(data.get("done_combos", []))


def load_valid_symbols():
    """Load symbols with enough candle data, grouped by TF."""
    if not CANDLES_DB.exists():
        print(f"ERROR: {CANDLES_DB} no existe")
        sys.exit(1)

    conn = sqlite3.connect(str(CANDLES_DB))
    rows = conn.execute("""
        SELECT symbol, timeframe, COUNT(*) as cnt
        FROM candles
        GROUP BY symbol, timeframe
        HAVING cnt >= ?
    """, (MIN_CANDLES,)).fetchall()
    conn.close()

    sym_tf = {}
    for sym, tf, cnt in rows:
        if sym not in sym_tf:
            sym_tf[sym] = {}
        sym_tf[sym][tf] = cnt

    return sym_tf


def build_pending_combos(strategies=None, tfs=None):
    """Build list of combos not yet tested, in priority order."""
    done = load_progress()
    sym_tf = load_valid_symbols()

    strats = strategies or WINNING_STRATEGIES
    timeframes = tfs or TF_PRIORITY

    pending = []
    for strat in strats:
        for tf in timeframes:
            symbols_for_tf = sorted([s for s in sym_tf if tf in sym_tf[s]])
            for sym in symbols_for_tf:
                key = f"{strat}|{sym}|{tf}"
                if key not in done:
                    pending.append({
                        "strategy": strat,
                        "symbol": sym,
                        "timeframe": tf,
                        "candles": sym_tf[sym][tf],
                    })

    return pending


def run_optuna_batch(combos, workers=OPTUNA_WORKERS, trials=OPTUNA_TRIALS):
    """Run Optuna V7 on a batch of combos via --combos-file."""
    if not combos:
        return []

    # Write combos file
    combos_file = DATA_DIR / "pipeline_combos_batch.json"
    with open(combos_file, "w") as f:
        json.dump(combos, f, indent=2)

    # Progress file specific to this pipeline run
    progress_file = str(DATA_DIR / "pipeline_optuna_progress.json")

    cmd = [
        sys.executable, str(OPTUNA_SCRIPT),
        "--combos-file", str(combos_file),
        "--workers", str(workers),
        "--trials", str(trials),
        "--progress-file", progress_file,
    ]

    print(f"\n{'='*60}")
    print(f"  OPTUNA: {len(combos)} combos | {workers} workers | {trials} trials")
    print(f"  Strategy: {combos[0]['strategy']}")
    print(f"  TF: {combos[0]['timeframe']}")
    print(f"  Symbols: {len(combos)}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
        timeout=3600 * 4,  # 4h max per batch
    )
    elapsed = time.time() - start

    # Parse grails from output — only return NEW grails from this batch
    grails_found = []
    if result.returncode == 0:
        # Check progress file for new grails
        if os.path.exists(progress_file):
            with open(progress_file) as f:
                pdata = json.load(f)
            all_grails = pdata.get("grails", [])
            # Filter to only grails that match this batch's combos
            batch_keys = set(f"{c['strategy']}|{c['symbol']}|{c['timeframe']}" for c in combos)
            for g in all_grails:
                key = f"{g.get('strategy','')}|{g.get('symbol','')}|{g.get('timeframe','')}"
                if key in batch_keys:
                    grails_found.append(g)
    else:
        print(f"  ⚠️ Optuna exit code: {result.returncode}")
        if result.stderr:
            # Show last 5 lines of stderr
            lines = result.stderr.strip().split("\n")
            for l in lines[-5:]:
                print(f"  stderr: {l}")

    print(f"  Tiempo: {elapsed/60:.1f} min | Grails: {len(grails_found)}")
    return grails_found


def run_forensic_backtest(grail):
    """Run forensic backtest on a single grail.

    Creates a temp JSON with the grail in forensic format, then runs
    forensic_backtest.py --input <temp> --db <candles_db> --limit 1

    The forensic script expects: strategy, symbol, timeframe, best_params, full_wr, sl, leverage
    Optuna grails have: strategy, symbol, timeframe, best_params, test_wr
    """
    strategy = grail.get("strategy", "?")
    symbol = grail.get("symbol", "?")
    timeframe = grail.get("timeframe", "?")

    # Build forensic-format grail — usar SL/leverage del grail calibrado
    gates = grail.get("gates", {})
    sl_from_grail = grail.get("sl", grail.get("sl_pct", 0))
    lev_from_grail = grail.get("safe_leverage", grail.get("leverage", 1))
    forensic_grail = {
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "best_params": grail.get("best_params", grail.get("params", {})),
        "full_wr": grail.get("test_wr", grail.get("cv_avg_wr", 70)),
        "sl": sl_from_grail if sl_from_grail > 0 else 0.10,  # SL empírico del grail
        "leverage": lev_from_grail if lev_from_grail >= 1 else 1,
    }

    # Write temp input file
    temp_input = DATA_DIR / "_pipeline_forensic_temp.json"
    with open(temp_input, "w") as f:
        json.dump([forensic_grail], f)

    # Use the full DB (27GB) if available, otherwise candles_subset
    full_db = Path("/Users/sabrina/CLAUDE CODE/data/activos_binance.db")
    db_to_use = str(full_db) if full_db.exists() and full_db.stat().st_size > 1_000_000 else str(CANDLES_DB)

    cmd = [
        sys.executable, "-W", "ignore",
        str(FORENSIC_SCRIPT),
        "--input", str(temp_input),
        "--db", db_to_use,
        "--limit", "1",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
        timeout=600,  # 10 min max (full DB is slower)
    )

    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-200:]
        if stderr_tail:
            print(f"    stderr: {stderr_tail}")
        return None

    # Parse result from the forensic output file
    try:
        with open(FORENSIC_RESULTS) as f:
            data = json.load(f)
        results = data.get("results", data) if isinstance(data, dict) else data
        # Find this specific result (last one added)
        for r in reversed(results):
            if (r.get("strategy") == strategy and
                r.get("symbol") == symbol and
                r.get("timeframe") == timeframe):
                return r
    except Exception as e:
        print(f"    Parse error: {e}")

    return None


def apply_gates(result):
    """Apply consensus gates. Returns (approved, tier, reasons)."""
    if not result:
        return False, "BLOCKED", ["no_result"]

    m = result.get("metrics", {})
    real_wr = m.get("win_rate", 0)
    optuna_wr = result.get("optuna_wr", 0)
    pnl = m.get("total_pnl_pct", 0)
    n_trades = m.get("total_trades", 0)
    max_dd = m.get("max_drawdown_pct", 0)
    pf = m.get("profit_factor", 0)
    gap = real_wr - optuna_wr

    reasons = []

    # Hard gates
    if pnl <= GATE_MIN_PNL:
        reasons.append(f"PnL={pnl:.1f}%<=0")
    if n_trades < GATE_MIN_TRADES:
        reasons.append(f"trades={n_trades}<{GATE_MIN_TRADES}")
    if real_wr < GATE_MIN_WR:
        reasons.append(f"WR={real_wr:.1f}%<{GATE_MIN_WR}%")
    if max_dd > GATE_MAX_DD:
        reasons.append(f"DD={max_dd:.1f}%>{GATE_MAX_DD}%")

    # Adjusted gate
    if pf < GATE_MIN_PF:
        reasons.append(f"PF={pf:.2f}<{GATE_MIN_PF}")

    # Gap gate
    if gap < GATE_GAP_HARD:
        reasons.append(f"gap={gap:.1f}pp<{GATE_GAP_HARD}pp")
    elif gap < GATE_GAP_SOFT:
        if pf < GATE_GAP_PF_THRESHOLD:
            reasons.append(f"gap={gap:.1f}pp+PF={pf:.2f}<{GATE_GAP_PF_THRESHOLD}")

    if reasons:
        return False, "BLOCKED", reasons

    # Determine tier
    if gap < GATE_GAP_SOFT and pf >= GATE_GAP_PF_THRESHOLD:
        return True, "Tier 2", ["monitoring_required"]

    return True, "Tier 1", []


def inject_to_production(grail, forensic_result, tier):
    """Inject approved grail to production JSON."""
    if not PRODUCTION_JSON.exists():
        print(f"  ⚠️ Production JSON not found: {PRODUCTION_JSON}")
        return False

    with open(PRODUCTION_JSON) as f:
        production = json.load(f)

    # Check if already exists
    for existing in production:
        if (existing["strategy"] == grail["strategy"] and
            existing["symbol"] == grail["symbol"] and
            existing["timeframe"] == grail["timeframe"]):
            print(f"  ℹ️ Already in production JSON")
            return True

    m = forensic_result.get("metrics", {})

    # Build production entry
    entry = {
        "strategy": grail["strategy"],
        "symbol": grail["symbol"],
        "timeframe": grail["timeframe"],
        "params": grail.get("params", {}),
        "sl_pct": grail.get("sl_pct", 0.10),
        "tp_pct": grail.get("tp_pct", 0.10),
        "leverage": grail.get("leverage", 2),
        "test_wr": round(m.get("win_rate", 0), 1),
        "test_trades": m.get("total_trades", 0),
        "test_pnl": round(m.get("total_pnl_pct", 0), 1),
        "max_dur_h": 720,
        "sl_tp_method": "optuna_v7_empirical",
        "tier": tier,
        "round": "PIPELINE_FORENSE",
        "composite_score": 1.0,
        "suspended": False,
        "forensic": {
            "real_wr": round(m.get("win_rate", 0), 1),
            "optuna_wr": round(grail.get("optuna_wr", grail.get("test_wr", 0)), 1),
            "gap_pp": round(m.get("win_rate", 0) - grail.get("optuna_wr", grail.get("test_wr", 0)), 1),
            "sharpe": round(m.get("sharpe", 0), 2),
            "max_dd_pct": round(m.get("max_drawdown_pct", 0), 1),
            "pnl_pct": round(m.get("total_pnl_pct", 0), 1),
            "n_trades": m.get("total_trades", 0),
            "validated_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    production.append(entry)

    # Backup ANTES de escribir (Regla 14)
    backup = str(PRODUCTION_JSON) + f".bak_{datetime.now().strftime('%H%M')}"
    with open(backup, "w") as f:
        json.dump(production, f, indent=2)

    # Atomic write — evita corrupción si el proceso muere a mitad
    import tempfile, os
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=PRODUCTION_JSON.parent, suffix=".tmp", delete=False
        ) as tmp:
            json.dump(production, tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, PRODUCTION_JSON)  # POSIX-atómico
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        print(f"  ❌ Error escribiendo JSON: {e}")
        return False

    # validate_json.py post-inyección (Regla 14)
    validate_script = BOT_V7_DIR / "scripts" / "validate_json.py"
    if validate_script.exists():
        r = subprocess.run(
            [sys.executable, str(validate_script)],
            capture_output=True, text=True, cwd=str(BOT_V7_DIR)
        )
        if r.returncode == 0:
            print(f"    ✅ validate_json: PASS")
        else:
            print(f"    ⚠️  validate_json: WARN (no revierte)\n{r.stdout[-200:]}")

    return True


def log_pipeline_event(event):
    """Append event to pipeline log."""
    log = []
    if PIPELINE_LOG.exists():
        try:
            with open(PIPELINE_LOG) as f:
                log = json.load(f)
        except:
            log = []

    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    log.append(event)

    with open(PIPELINE_LOG, "w") as f:
        json.dump(log, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pipeline Optuna → Forense → Gates → Produccion")
    parser.add_argument("--strategy", help="Solo una estrategia")
    parser.add_argument("--tf", help="Solo un timeframe (5m, 1h)")
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar que haria")
    parser.add_argument("--max-hours", type=float, default=0, help="Limite de horas (0=ilimitado)")
    parser.add_argument("--workers", type=int, default=OPTUNA_WORKERS)
    parser.add_argument("--trials", type=int, default=OPTUNA_TRIALS)
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Combos por batch de Optuna (default 20)")
    parser.add_argument("--skip-optuna", action="store_true",
                        help="Solo correr forense en grails existentes")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  PIPELINE OPTUNA → FORENSE → GATES → PRODUCCION            ║
║  {datetime.now().strftime('%Y-%m-%d %H:%M')}                                           ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Build pending combos
    strats = [args.strategy] if args.strategy else None
    tfs = [args.tf] if args.tf else None
    pending = build_pending_combos(strategies=strats, tfs=tfs)

    print(f"Combos pendientes: {len(pending)}")

    if not pending:
        print("✅ Todo ya testeado!")
        return

    # Group by strategy + TF for sequential processing
    groups = defaultdict(list)
    for c in pending:
        key = f"{c['strategy']}|{c['timeframe']}"
        groups[key].append(c)

    print(f"Grupos (strategy×TF): {len(groups)}")
    for key, combos in groups.items():
        strat, tf = key.split("|")
        print(f"  {strat} × {tf}: {len(combos)} symbols pendientes")

    if args.dry_run:
        print("\n🔍 DRY RUN — no se ejecuta nada")
        est_hours = len(pending) * args.trials * 5 / args.workers / 3600
        print(f"Tiempo estimado: {est_hours:.1f} horas con {args.workers} workers")
        return

    # Pipeline execution
    start_time = time.time()
    total_grails = 0
    total_approved = 0
    total_injected = 0

    for group_key in sorted(groups.keys()):
        # Check time limit
        if args.max_hours > 0:
            elapsed_h = (time.time() - start_time) / 3600
            if elapsed_h >= args.max_hours:
                print(f"\n⏰ Limite de tiempo alcanzado ({args.max_hours}h)")
                break

        combos = groups[group_key]
        strat, tf = group_key.split("|")

        print(f"\n{'─'*60}")
        print(f"  PROCESANDO: {strat} × {tf}")
        print(f"  Symbols: {len(combos)} pendientes")
        print(f"{'─'*60}")

        if args.skip_optuna:
            print("  [skip-optuna] Saltando Optuna, solo forense...")
            continue

        # Process in batches
        for i in range(0, len(combos), args.batch_size):
            batch = combos[i:i + args.batch_size]

            # Check time limit
            if args.max_hours > 0:
                elapsed_h = (time.time() - start_time) / 3600
                if elapsed_h >= args.max_hours:
                    print(f"\n⏰ Limite de tiempo alcanzado ({args.max_hours}h)")
                    break

            print(f"\n  Batch {i//args.batch_size + 1}/{(len(combos)-1)//args.batch_size + 1}: "
                  f"{len(batch)} combos")

            # Step 1: Run Optuna
            grails = run_optuna_batch(batch, workers=args.workers, trials=args.trials)
            total_grails += len(grails)

            if not grails:
                print(f"  → 0 grails en este batch")
                continue

            print(f"  → {len(grails)} grails encontrados!")

            # Step 2: Forensic backtest each grail
            for grail in grails:
                g_strat = grail.get("strategy", "?")
                g_sym = grail.get("symbol", "?")
                g_tf = grail.get("timeframe", "?")
                g_wr = grail.get("test_wr", 0)

                print(f"\n  🔬 Forense: {g_strat} × {g_sym} {g_tf} (Optuna WR={g_wr:.1f}%)")

                forensic = run_forensic_backtest(grail)

                if not forensic:
                    print(f"    ❌ Forense falló o sin datos")
                    log_pipeline_event({
                        "event": "forensic_failed",
                        "strategy": g_strat, "symbol": g_sym, "timeframe": g_tf,
                    })
                    continue

                real_wr = forensic.get("metrics", {}).get("win_rate", 0)
                print(f"    Forense WR: {real_wr:.1f}% (gap: {real_wr - g_wr:.1f}pp)")

                # Step 3: Apply gates
                approved, tier, reasons = apply_gates(forensic)

                if not approved:
                    print(f"    ❌ BLOQUEADO: {', '.join(reasons)}")
                    log_pipeline_event({
                        "event": "gate_blocked",
                        "strategy": g_strat, "symbol": g_sym, "timeframe": g_tf,
                        "real_wr": real_wr, "reasons": reasons,
                    })
                    continue

                total_approved += 1
                print(f"    ✅ APROBADO ({tier})")

                # Step 4: Inject to production
                injected = inject_to_production(grail, forensic, tier)
                if injected:
                    total_injected += 1
                    print(f"    💉 Inyectado a produccion JSON")
                    log_pipeline_event({
                        "event": "injected",
                        "strategy": g_strat, "symbol": g_sym, "timeframe": g_tf,
                        "real_wr": real_wr, "tier": tier,
                    })

    # Summary
    elapsed = time.time() - start_time
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  PIPELINE COMPLETADO                                         ║
╠══════════════════════════════════════════════════════════════╣
║  Tiempo: {elapsed/3600:.1f} horas                                         ║
║  Combos procesados: {len(pending):<6}                                  ║
║  Grails encontrados: {total_grails:<5}                                  ║
║  Aprobados (Gates): {total_approved:<5}                                  ║
║  Inyectados prod: {total_injected:<5}                                    ║
╚══════════════════════════════════════════════════════════════╝
""")

    log_pipeline_event({
        "event": "pipeline_complete",
        "elapsed_h": round(elapsed / 3600, 2),
        "combos": len(pending),
        "grails": total_grails,
        "approved": total_approved,
        "injected": total_injected,
    })


if __name__ == "__main__":
    main()
