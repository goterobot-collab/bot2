#!/usr/bin/env python3
"""
hetzner_pipeline_v8.py — V8 REAL pipeline (Optuna → Forensic → V8 Gates)
=========================================================================

Extiende hetzner_pipeline.py con gates V8 REAL (Wilson LCB + DSR + Hurst +
regime + direction + fee_rate canónico + n_trades).

DIFERENCIAS vs hetzner_pipeline.py (V7.5):
  - Aplica apply_v8_gates() en vez de 4 gates hardcoded
  - Loop de 100 combos con circuit breaker (errors, DD, timeout)
  - Checkpoint cada N combos (resumable)
  - n_trials = total combos scaneados (de stats de Optuna)
  - Output: v8_real_candidates / v8_real_rejected / v8_real_summary
  - Config V8 profile: default (full) o relaxed_v75plus (hoy)

Convergencia 3/3 A/A/A (COORDINADORA + CEREBRO + SIGNAL_AUDITOR).
Aprobado Sabrina 2026-04-15.

Uso:
  # Full scan con V8 relaxed (hoy):
  python3 scripts/hetzner_pipeline_v8.py \
      --progress data/progress.json \
      --v8-profile relaxed \
      --workers 6 \
      --batch-size 100

  # V8 default (cuando haya regime+direction data):
  python3 scripts/hetzner_pipeline_v8.py \
      --progress data/progress.json \
      --v8-profile default

  # Resume desde checkpoint:
  python3 scripts/hetzner_pipeline_v8.py \
      --resume data/v8_real_checkpoint.json

Paths Hetzner:
  DB: /home/ubuntu/candles_db/activos_binance_fresh.db
  Output: /home/ubuntu/estrategias/data/v8_real_*.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
SCRIPTS_DIR = str(Path(__file__).resolve().parent)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SCRIPTS_DIR)

# DB auto-detect (Hetzner first)
_HETZNER_DB = "/home/ubuntu/candles_db/activos_binance_fresh.db"
_HETZNER_DB_ALT = "/home/ubuntu/candles_db/activos_binance.db"
_MAC_DB = "/Users/sabrina/CLAUDE CODE/data/activos_binance.db"

if os.path.exists(_HETZNER_DB):
    DB_PATH = _HETZNER_DB
elif os.path.exists(_HETZNER_DB_ALT):
    DB_PATH = _HETZNER_DB_ALT
else:
    DB_PATH = _MAC_DB

ALL_TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']

# Circuit breaker thresholds
CB_MAX_ERRORS_PCT = 0.30       # > 30% errores consecutivos en batch → halt
CB_MAX_ELAPSED_HOURS = 6.0     # > 6h sin terminar → halt y salvar checkpoint
CB_MAX_CONSECUTIVE_ERRORS = 15 # 15 errores seguidos → halt (rot subyacente)


# ═════════════════════════════════════════════════════════════
# Forensic + V8 gates (worker)
# ═════════════════════════════════════════════════════════════


def run_forensic_and_v8_gates(combo, v8_profile, n_trials_total):
    """
    Worker: corre forensic + apply_v8_gates para UN combo.

    Args:
        combo: dict con {strategy, symbol, timeframe, best_params, optuna_wr, sl, leverage}
        v8_profile: 'default' o 'relaxed'
        n_trials_total: total combos scaneados por Optuna (para DSR)

    Returns:
        dict con {strategy, symbol, timeframe, forensic_metrics, v8_result, status}
        status ∈ {'V8_PASS', 'V8_FAIL', 'NO_DATA', 'NO_STRATEGY', 'ERROR'}
    """
    try:
        from forensic_backtest import forensic_backtest, load_candles, load_all_strategies
        from v8_gates import V8GateConfig, apply_v8_gates

        strategy = combo['strategy']
        symbol = combo['symbol']
        tf = combo['timeframe']
        params = combo.get('best_params', {})
        sl_pct = combo.get('sl', 0.40)
        leverage = combo.get('leverage', 1)
        optuna_wr = combo.get('optuna_wr', 0)

        # Load candles
        df = load_candles(symbol, tf)
        if df is None or len(df) < 50:
            return {
                'strategy': strategy, 'symbol': symbol, 'timeframe': tf,
                'status': 'NO_DATA', 'error': f'No candles for {symbol} {tf}',
            }

        # Load strategy
        strategies = load_all_strategies()
        gen_fn = strategies.get(strategy)
        if gen_fn is None:
            return {
                'strategy': strategy, 'symbol': symbol, 'timeframe': tf,
                'status': 'NO_STRATEGY', 'error': f'Strategy {strategy} not found',
            }

        # Run forensic
        result = forensic_backtest(
            df, gen_fn, params,
            sl_pct=sl_pct, leverage=leverage,
            symbol=symbol, timeframe=tf,
        )

        if result is None or 'error' in result:
            return {
                'strategy': strategy, 'symbol': symbol, 'timeframe': tf,
                'status': 'ERROR',
                'error': str(result.get('error', 'unknown') if result else 'empty result'),
            }

        metrics = result.get('metrics', {})
        trades = result.get('trades', [])

        # Construct grail dict for apply_v8_gates
        # Forensic trade format: {pnl_pct, direction, win?, regime?, ...}
        v8_trades = []
        returns = []
        for t in trades:
            is_win = t.get('win')
            if is_win is None:
                is_win = (t.get('pnl_pct', 0) > 0)
            v8_trades.append({
                'win': bool(is_win),
                'direction': (t.get('direction') or 'LONG').upper(),
                'regime': (t.get('regime') or 'NEUTRAL').upper(),
                'pnl_pct': t.get('pnl_pct', 0.0),
            })
            # returns for Hurst/skew/kurt (percentage)
            returns.append(float(t.get('pnl_pct', 0.0)) / 100.0)

        grail_for_gates = {
            'wins': metrics.get('wins', 0),
            'total_trades': metrics.get('total_trades', len(trades)),
            'sharpe': metrics.get('sharpe', 0.0),
            'returns': returns,
            'trades': v8_trades,
            'n_trials': n_trials_total,
            'fee_rate': metrics.get('fee_rate', 0.0030),  # persisted by forensic
        }

        # Pick profile
        if v8_profile == 'default':
            cfg = V8GateConfig.default()
        elif v8_profile == 'relaxed':
            cfg = V8GateConfig.relaxed_v75plus()
        else:
            cfg = V8GateConfig.default()

        v8_result = apply_v8_gates(grail_for_gates, cfg)

        # Build output: include forensic + v8 evaluation
        return {
            'strategy': strategy,
            'symbol': symbol,
            'timeframe': tf,
            'best_params': params,
            'sl': sl_pct,
            'leverage': leverage,
            'optuna_wr': optuna_wr,
            'forensic_metrics': {
                'win_rate': metrics.get('win_rate', 0),
                'total_pnl_pct': metrics.get('total_pnl_pct', 0),
                'total_trades': metrics.get('total_trades', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'sharpe': metrics.get('sharpe', 0),
                'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
                'fee_rate': metrics.get('fee_rate', 0.0030),
                'long_wr': metrics.get('long_wr', 0),
                'short_wr': metrics.get('short_wr', 0),
                'avg_mae_pct': metrics.get('avg_mae_pct', 0),
                'avg_mfe_pct': metrics.get('avg_mfe_pct', 0),
            },
            'v8_result': v8_result,
            'v8_profile': v8_profile,
            'status': 'V8_PASS' if v8_result['pass'] else 'V8_FAIL',
            'candles_used': len(df),
        }

    except Exception as e:  # pragma: no cover
        return {
            'strategy': combo.get('strategy', '?'),
            'symbol': combo.get('symbol', '?'),
            'timeframe': combo.get('timeframe', '?'),
            'status': 'ERROR',
            'error': str(e),
        }


# ═════════════════════════════════════════════════════════════
# Input/output helpers
# ═════════════════════════════════════════════════════════════


def load_grails_from_progress(progress_path):
    """Lee grails del progress file de Optuna."""
    with open(progress_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        grails = data.get('grails', [])
        stats = data.get('stats', {})
        n_trials = stats.get('total_tested', len(grails))
    elif isinstance(data, list):
        grails = data
        stats = {}
        n_trials = len(grails)
    else:
        grails = []
        stats = {}
        n_trials = 0
    return grails, stats, n_trials


def expand_grails_all_tf(grails):
    """Cada grail → 5 combos (un TF cada uno)."""
    expanded = []
    seen = set()
    for g in grails:
        strategy = g['strategy']
        symbol = g['symbol']
        params = g.get('best_params', {})
        optuna_wr = g.get('test_wr', g.get('full_wr', g.get('v3_original_wr', 0)))
        for tf in ALL_TIMEFRAMES:
            key = f"{strategy}|{symbol}|{tf}"
            if key not in seen:
                seen.add(key)
                expanded.append({
                    'strategy': strategy,
                    'symbol': symbol,
                    'timeframe': tf,
                    'best_params': params,
                    'optuna_wr': optuna_wr,
                    'sl': g.get('sl', 0.40),
                    'leverage': g.get('safe_leverage', g.get('leverage', 1)),
                    'source_tf': g.get('timeframe', tf),
                })
    return expanded


def save_checkpoint(output_dir, state):
    """Escribe checkpoint atómico (write to tmp → rename)."""
    checkpoint = os.path.join(output_dir, 'v8_real_checkpoint.json')
    tmp = checkpoint + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, checkpoint)
    return checkpoint


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════
# Circuit breaker
# ═════════════════════════════════════════════════════════════


class CircuitBreaker:
    def __init__(self, max_errors_pct=CB_MAX_ERRORS_PCT,
                 max_elapsed_h=CB_MAX_ELAPSED_HOURS,
                 max_consecutive=CB_MAX_CONSECUTIVE_ERRORS):
        self.max_errors_pct = max_errors_pct
        self.max_elapsed_h = max_elapsed_h
        self.max_consecutive = max_consecutive
        self.start_time = time.time()
        self.total = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.halted = False
        self.halt_reason = None

    def record(self, status):
        self.total += 1
        if status in ('ERROR', 'NO_DATA', 'NO_STRATEGY'):
            self.errors += 1
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0

        # Check halts
        elapsed_h = (time.time() - self.start_time) / 3600.0
        if elapsed_h > self.max_elapsed_h:
            self.halted = True
            self.halt_reason = f'Elapsed {elapsed_h:.1f}h > {self.max_elapsed_h}h'
            return
        if self.consecutive_errors >= self.max_consecutive:
            self.halted = True
            self.halt_reason = f'Consecutive errors {self.consecutive_errors} >= {self.max_consecutive}'
            return
        if self.total >= 30:  # solo evaluar pct después de 30 combos
            err_pct = self.errors / self.total
            if err_pct > self.max_errors_pct:
                self.halted = True
                self.halt_reason = f'Error pct {err_pct:.1%} > {self.max_errors_pct:.0%} (over {self.total} combos)'
                return

    def should_continue(self):
        return not self.halted


# ═════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════


def run_v8_pipeline(args):
    print("=" * 72)
    print("  HETZNER V8 REAL PIPELINE — Optuna → Forensic → V8 Gates")
    print("=" * 72)
    print(f"  DB:              {DB_PATH}")
    print(f"  Workers:         {args.workers}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  V8 profile:      {args.v8_profile}")
    print(f"  Expand TFs:      {args.expand_tf}")
    print()

    # ─── Resume from checkpoint? ───
    resume_state = None
    if args.resume:
        resume_state = load_checkpoint(args.resume)
        if resume_state:
            print(f"  RESUMING from checkpoint: {args.resume}")
            print(f"    Already processed: {resume_state.get('processed_count', 0)}")
            print(f"    V8 passed:         {len(resume_state.get('v8_passed', []))}")
            print(f"    V8 failed:         {len(resume_state.get('v8_failed', []))}")

    # ─── Load queue ───
    if resume_state:
        queue = resume_state['queue_remaining']
        v8_passed = resume_state['v8_passed']
        v8_failed = resume_state['v8_failed']
        errors_list = resume_state.get('errors', [])
        processed_keys = set(resume_state.get('processed_keys', []))
        n_trials = resume_state.get('n_trials', 1000)
    else:
        if not args.progress:
            print("  ERROR: Necesito --progress o --resume")
            return 1
        grails, stats, n_trials = load_grails_from_progress(args.progress)
        print(f"  Grails loaded:  {len(grails)}")
        print(f"  Optuna trials:  {n_trials} (para DSR n_trials)")
        if not grails:
            print("  ERROR: 0 grails. Nada que testear.")
            return 1

        if args.expand_tf:
            queue = expand_grails_all_tf(grails)
            print(f"  Expanded to:    {len(queue)} combos ({len(ALL_TIMEFRAMES)} TFs each)")
        else:
            queue = []
            for g in grails:
                queue.append({
                    'strategy': g['strategy'],
                    'symbol': g['symbol'],
                    'timeframe': g.get('timeframe', '1h'),
                    'best_params': g.get('best_params', {}),
                    'optuna_wr': g.get('test_wr', g.get('full_wr', 0)),
                    'sl': g.get('sl', 0.40),
                    'leverage': g.get('safe_leverage', g.get('leverage', 1)),
                })

        v8_passed = []
        v8_failed = []
        errors_list = []
        processed_keys = set()

    # ─── Init circuit breaker ───
    cb = CircuitBreaker(
        max_errors_pct=args.cb_error_pct,
        max_elapsed_h=args.cb_timeout_h,
    )

    # ─── Process in batches ───
    output_dir = args.output_dir or DATA_DIR
    os.makedirs(output_dir, exist_ok=True)

    total = len(queue) + len(processed_keys)
    start = time.time()

    print(f"\n═══ PROCESSING {len(queue)} combos in batches of {args.batch_size} ═══\n")

    batch_num = 0
    while queue and cb.should_continue():
        batch = queue[:args.batch_size]
        queue = queue[args.batch_size:]
        batch_num += 1

        print(f"[Batch {batch_num}] {len(batch)} combos | "
              f"processed={len(processed_keys)}/{total} | "
              f"V8_PASS={len(v8_passed)} V8_FAIL={len(v8_failed)} "
              f"ERR={len(errors_list)}")

        # Run batch in parallel
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_forensic_and_v8_gates, combo, args.v8_profile, n_trials): combo
                for combo in batch
            }
            for fut in as_completed(futures):
                combo = futures[fut]
                key = f"{combo['strategy']}|{combo['symbol']}|{combo['timeframe']}"
                try:
                    r = fut.result(timeout=300)
                except Exception as e:
                    r = {
                        'strategy': combo['strategy'],
                        'symbol': combo['symbol'],
                        'timeframe': combo['timeframe'],
                        'status': 'ERROR',
                        'error': str(e),
                    }

                status = r.get('status', 'ERROR')
                cb.record(status)
                processed_keys.add(key)

                if status == 'V8_PASS':
                    v8_passed.append(r)
                elif status == 'V8_FAIL':
                    v8_failed.append(r)
                else:
                    errors_list.append(r)

                if not cb.should_continue():
                    print(f"  CIRCUIT BREAKER HALT: {cb.halt_reason}")
                    break

        # Checkpoint after each batch
        state = {
            'timestamp': datetime.now().isoformat(),
            'v8_profile': args.v8_profile,
            'n_trials': n_trials,
            'total': total,
            'processed_count': len(processed_keys),
            'processed_keys': list(processed_keys),
            'queue_remaining': queue,
            'v8_passed': v8_passed,
            'v8_failed': v8_failed,
            'errors': errors_list,
            'cb_state': {
                'halted': cb.halted,
                'reason': cb.halt_reason,
                'total': cb.total,
                'errors': cb.errors,
                'consecutive_errors': cb.consecutive_errors,
            },
        }
        save_checkpoint(output_dir, state)

        # Live table of top passes (first 5)
        if v8_passed:
            recent_pass = sorted(v8_passed, key=lambda x: x['v8_result'].get('score', 0), reverse=True)[:5]
            print(f"  TOP 5 V8_PASS (score):")
            for r in recent_pass:
                sym = r['symbol'].replace('/USDT:USDT', '')
                m = r['forensic_metrics']
                print(f"    {r['strategy'][:28]:28} x {sym:10} x {r['timeframe']:3} | "
                      f"WR={m['win_rate']:5.1f}% PnL={m['total_pnl_pct']:+6.1f}% "
                      f"N={m['total_trades']:3} score={r['v8_result']['score']:.2f}")
        print()

    elapsed = time.time() - start

    # ─── Final output ───
    print(f"\n═══ RESULTADOS V8 REAL ═══")
    print(f"  Total processed:     {len(processed_keys)}/{total}")
    print(f"  V8_PASS:             {len(v8_passed)}")
    print(f"  V8_FAIL:             {len(v8_failed)}")
    print(f"  ERRORS:              {len(errors_list)}")
    print(f"  Elapsed:             {elapsed/60:.1f} min")
    if cb.halted:
        print(f"  CIRCUIT BREAKER:     HALTED — {cb.halt_reason}")

    # Breakdown of failure reasons
    if v8_failed:
        fail_gate_counter = {}
        for r in v8_failed:
            for g in r['v8_result'].get('failed_gates', []):
                fail_gate_counter[g] = fail_gate_counter.get(g, 0) + 1
        print(f"\n  Failure breakdown (por gate):")
        for gate, count in sorted(fail_gate_counter.items(), key=lambda x: x[1], reverse=True):
            print(f"    {gate:20} {count:4}  ({count*100//max(1,len(v8_failed)):3}%)")

    # Top 10 passed
    if v8_passed:
        top = sorted(v8_passed,
                     key=lambda x: (x['v8_result'].get('score', 0),
                                   x['forensic_metrics'].get('total_pnl_pct', 0)),
                     reverse=True)[:10]
        print(f"\n  🏆 TOP 10 V8_PASS:")
        print(f"  {'Strategy':<30} {'Symbol':<10} {'TF':>3} {'WR%':>5} {'PnL%':>7} {'N':>4} {'PF':>5} {'Score':>5}")
        print(f"  {'─'*30} {'─'*10} {'─'*3} {'─'*5} {'─'*7} {'─'*4} {'─'*5} {'─'*5}")
        for r in top:
            sym = r['symbol'].replace('/USDT:USDT', '')
            m = r['forensic_metrics']
            print(f"  {r['strategy'][:30]:<30} {sym:<10} {r['timeframe']:>3} "
                  f"{m['win_rate']:>4.1f}% {m['total_pnl_pct']:>+6.1f}% "
                  f"{m['total_trades']:>4} {m['profit_factor']:>5.2f} "
                  f"{r['v8_result']['score']:>5.2f}")

    # ─── Save outputs ───
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    passed_path = os.path.join(output_dir, f'v8_real_candidates_{ts}.json')
    failed_path = os.path.join(output_dir, f'v8_real_rejected_{ts}.json')
    summary_path = os.path.join(output_dir, f'v8_real_summary_{ts}.json')

    with open(passed_path, 'w') as f:
        json.dump(v8_passed, f, indent=2, default=str)
    print(f"\n  V8_PASS saved:   {passed_path} ({len(v8_passed)} candidates)")

    with open(failed_path, 'w') as f:
        json.dump(v8_failed, f, indent=2, default=str)
    print(f"  V8_FAIL saved:   {failed_path}")

    summary = {
        'timestamp': datetime.now().isoformat(),
        'v8_profile': args.v8_profile,
        'n_trials': n_trials,
        'db_path': DB_PATH,
        'total_processed': len(processed_keys),
        'total_queue': total,
        'v8_pass_count': len(v8_passed),
        'v8_fail_count': len(v8_failed),
        'error_count': len(errors_list),
        'elapsed_minutes': round(elapsed / 60, 1),
        'circuit_breaker': {
            'halted': cb.halted,
            'reason': cb.halt_reason,
            'errors': cb.errors,
            'consecutive_errors': cb.consecutive_errors,
        },
        'candidates_file': passed_path,
        'rejected_file': failed_path,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary:         {summary_path}")

    print("\n" + "=" * 72)
    print(f"  V8 REAL PIPELINE DONE — {len(v8_passed)} candidates pending inject")
    print(f"  Next: python3 scripts/inject_v8_real_to_production.py --input {passed_path}")
    print("=" * 72)
    return 0 if not cb.halted else 2


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description='Hetzner V8 REAL Pipeline: Optuna → Forensic → V8 Gates',
    )
    parser.add_argument('--progress', type=str,
                        help='Progress file de Optuna con grails (requerido si no --resume)')
    parser.add_argument('--resume', type=str,
                        help='Resume desde checkpoint file')
    parser.add_argument('--v8-profile', type=str, default='relaxed',
                        choices=['default', 'relaxed'],
                        help='default (full 7 gates) o relaxed (v7.5+, Wilson+fee+n solo)')
    parser.add_argument('--expand-tf', action='store_true', default=True,
                        help='Testear cada grail en TODOS los TFs — default ON')
    parser.add_argument('--no-expand-tf', action='store_true',
                        help='Desactiva expand-tf')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Combos por batch (checkpoint after each batch)')
    parser.add_argument('--cb-error-pct', type=float, default=CB_MAX_ERRORS_PCT)
    parser.add_argument('--cb-timeout-h', type=float, default=CB_MAX_ELAPSED_HOURS)
    parser.add_argument('--output-dir', type=str, help='Override output dir (default: data/)')
    parser.add_argument('--db', type=str, help='Override DB path')

    args = parser.parse_args()

    if args.no_expand_tf:
        args.expand_tf = False

    if args.db:
        global DB_PATH
        DB_PATH = args.db

    if not args.progress and not args.resume:
        print("ERROR: --progress o --resume requerido")
        sys.exit(1)

    sys.exit(run_v8_pipeline(args))


if __name__ == '__main__':
    main()
