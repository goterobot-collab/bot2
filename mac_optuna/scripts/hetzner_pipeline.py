#!/usr/bin/env python3
"""
HETZNER FULL PIPELINE: Optuna → Forensic Backtest → Gates → Resultados
=======================================================================
Corre TODO en Hetzner: Optuna encuentra grails, luego backtest forense
valida cada uno contra TODA la historia en TODOS los timeframes.

Flujo:
  1. Optuna corre sobre combos file → genera grails con params
  2. Lee los grails del progress file
  3. Para cada grail: corre forensic_backtest en el TF original
  4. EXPANDE: testea los mismos params en TODOS los TFs (5m, 15m, 1h, 4h, 1d)
  5. Aplica Gates Finales → clasifica en CONFIRMED / WARNING / INFLATED / REJECTED
  6. Genera JSON final con resultados + resumen

Uso:
  # Paso 1: Correr Optuna normalmente
  python3 optuna_v7.py --combos-file data/combos.json --progress-file data/progress.json --workers 6

  # Paso 2: Correr este pipeline sobre los grails encontrados
  python3 scripts/hetzner_pipeline.py --progress data/progress.json
  python3 scripts/hetzner_pipeline.py --progress data/progress.json --expand-tf   # + todos los TFs
  python3 scripts/hetzner_pipeline.py --progress data/progress.json --workers 4   # paralelo

  # Todo junto (Optuna + Forensic):
  python3 scripts/hetzner_pipeline.py --combos data/combos.json --workers 6 --trials 30

Paths Hetzner:
  DB: /home/ubuntu/candles_db/activos_binance_fresh.db
  Strategies: /home/ubuntu/estrategias/strategies_tv2_batches/
  Scripts: /home/ubuntu/estrategias/scripts/
"""

import json
import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Gates Finales (mismos que forensic_backtest.py)
GATE_MIN_WR = 65.0
GATE_MAX_GAP = 10.0      # gap Optuna→Real máx permitido
GATE_MIN_TRADES = 20
GATE_MIN_PF = 1.15


def load_grails_from_progress(progress_path):
    """Lee grails del progress file de Optuna o de un JSON de grails directo."""
    with open(progress_path) as f:
        data = json.load(f)

    # Soportar ambos formatos:
    # 1. Progress de Optuna: {"grails": [...], "stats": {...}, "done_combos": [...]}
    # 2. Lista directa de grails: [{"strategy": ..., "best_params": ...}, ...]
    if isinstance(data, dict):
        grails = data.get('grails', [])
        stats = data.get('stats', {})
    elif isinstance(data, list):
        grails = data
        stats = {}
    else:
        grails = []
        stats = {}

    print(f"  Progress file: {progress_path}")
    print(f"  Grails encontrados: {len(grails)}")
    if stats:
        print(f"  Combos testeados: {stats.get('total_tested', '?')}")
        print(f"  Elapsed: {stats.get('elapsed_h', '?')}h")
        print(f"  Finished: {stats.get('finished', '?')}")

    return grails


def expand_grails_all_tf(grails):
    """
    Dado un grail en TF X, genera copias para TODOS los TFs.
    Ejemplo: grail en 1h → genera también para 5m, 15m, 4h, 1d
    con los MISMOS params (el forense dirá si funcionan o no).
    """
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

    print(f"  Expandido: {len(grails)} grails → {len(expanded)} combos ({len(ALL_TIMEFRAMES)} TFs)")
    return expanded


def run_forensic_single(grail, db_path):
    """Corre forensic backtest para UN grail. Retorna resultado dict."""
    try:
        from forensic_backtest import forensic_backtest, load_candles, load_all_strategies, \
            get_slippage_for_symbol, COMMISSION, TP_CAP_BY_TIMEFRAME

        strategy_name = grail['strategy']
        symbol = grail['symbol']
        tf = grail['timeframe']
        params = grail.get('best_params', {})
        sl_pct = grail.get('sl', 0.40)
        leverage = grail.get('leverage', 1)
        optuna_wr = grail.get('optuna_wr', 0)

        # Load candles
        df = load_candles(symbol, tf)
        if df is None or len(df) < 50:
            return {'strategy': strategy_name, 'symbol': symbol, 'timeframe': tf,
                    'status': 'NO_DATA', 'error': f'No candles for {symbol} {tf}'}

        # Load strategy
        strategies = load_all_strategies()
        gen_fn = strategies.get(strategy_name)
        if gen_fn is None:
            return {'strategy': strategy_name, 'symbol': symbol, 'timeframe': tf,
                    'status': 'NO_STRATEGY', 'error': f'Strategy {strategy_name} not found'}

        # Run forensic
        result = forensic_backtest(
            df, gen_fn, params,
            sl_pct=sl_pct, leverage=leverage,
            symbol=symbol, timeframe=tf
        )

        if result is None or 'error' in result:
            return {'strategy': strategy_name, 'symbol': symbol, 'timeframe': tf,
                    'status': 'ERROR', 'error': str(result.get('error', 'unknown'))}

        metrics = result.get('metrics', {})
        real_wr = metrics.get('win_rate', 0)
        gap = optuna_wr - real_wr
        pnl = metrics.get('total_pnl_pct', 0)
        trades = metrics.get('total_trades', 0)
        pf = metrics.get('profit_factor', 0)
        max_dd = metrics.get('max_drawdown', 0)

        # Gates
        status = 'CONFIRMED'
        block_reasons = []

        if real_wr < GATE_MIN_WR:
            status = 'REJECTED'
            block_reasons.append(f'WR {real_wr:.1f}% < {GATE_MIN_WR}%')
        if gap > GATE_MAX_GAP:
            status = 'INFLATED'
            block_reasons.append(f'Gap {gap:+.1f}pp > {GATE_MAX_GAP}pp')
        if pnl <= 0:
            status = 'REJECTED'
            block_reasons.append(f'PnL {pnl:.1f}% <= 0')
        if trades < GATE_MIN_TRADES:
            status = 'REJECTED'
            block_reasons.append(f'Trades {trades} < {GATE_MIN_TRADES}')
        if pf < GATE_MIN_PF and status != 'REJECTED':
            status = 'REJECTED'
            block_reasons.append(f'PF {pf:.2f} < {GATE_MIN_PF}')

        if status == 'CONFIRMED' and abs(gap) <= 5:
            status = 'CONFIRMED'
        elif status == 'CONFIRMED' and gap <= 10:
            status = 'WARNING'

        return {
            'strategy': strategy_name,
            'symbol': symbol,
            'timeframe': tf,
            'best_params': params,
            'optuna_wr': optuna_wr,
            'forensic_wr': round(real_wr, 1),
            'gap_pp': round(gap, 1),
            'pnl_pct': round(pnl, 1),
            'trades': trades,
            'profit_factor': round(pf, 2),
            'max_dd': round(max_dd, 1),
            'sharpe': round(metrics.get('sharpe', 0), 2),
            'sl': sl_pct,
            'leverage': leverage,
            'source_tf': grail.get('source_tf', tf),
            'status': status,
            'block_reasons': block_reasons,
            'candles_used': len(df),
        }

    except Exception as e:
        return {
            'strategy': grail.get('strategy', '?'),
            'symbol': grail.get('symbol', '?'),
            'timeframe': grail.get('timeframe', '?'),
            'status': 'ERROR',
            'error': str(e)
        }


def run_pipeline(args):
    """Pipeline principal: Optuna (opcional) → Forensic → Gates → Output."""

    print("=" * 70)
    print("  HETZNER FULL PIPELINE — Optuna → Forensic → Gates")
    print("=" * 70)
    print(f"  DB: {DB_PATH}")
    print(f"  Workers: {args.workers}")
    print(f"  Expand TFs: {args.expand_tf}")
    print()

    # ─── PASO 1: Optuna (opcional) ───
    if args.combos:
        print("═══ PASO 1: OPTUNA ═══")
        progress_file = args.progress or os.path.join(
            DATA_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}_progress.json")

        cmd = [
            sys.executable, os.path.join(PROJECT_DIR, "optuna_v7.py"),
            "--combos-file", args.combos,
            "--progress-file", progress_file,
            "--workers", str(args.workers),
            "--trials", str(args.trials),
        ]
        print(f"  Running: {' '.join(cmd)}")
        ret = subprocess.run(cmd, cwd=PROJECT_DIR)
        if ret.returncode != 0:
            print(f"  ERROR: Optuna failed with exit code {ret.returncode}")
            return
        args.progress = progress_file
        print()

    # ─── PASO 2: Cargar grails ───
    print("═══ PASO 2: CARGAR GRAILS ═══")
    if not args.progress:
        print("  ERROR: Necesito --progress <file> o --combos <file>")
        return

    grails = load_grails_from_progress(args.progress)
    if not grails:
        print("  ERROR: 0 grails encontrados. Nada que testear.")
        return

    # ─── PASO 3: Expandir a todos los TFs (opcional) ───
    if args.expand_tf:
        print("\n═══ PASO 3: EXPANDIR A TODOS LOS TIMEFRAMES ═══")
        test_grails = expand_grails_all_tf(grails)
    else:
        # Solo testear en el TF original
        test_grails = []
        for g in grails:
            test_grails.append({
                'strategy': g['strategy'],
                'symbol': g['symbol'],
                'timeframe': g.get('timeframe', '1h'),
                'best_params': g.get('best_params', {}),
                'optuna_wr': g.get('test_wr', g.get('full_wr', g.get('v3_original_wr', 0))),
                'sl': g.get('sl', 0.40),
                'leverage': g.get('safe_leverage', g.get('leverage', 1)),
                'source_tf': g.get('timeframe', '1h'),
            })
        print(f"  Grails para testear: {len(test_grails)} (TF original)")

    # ─── PASO 4: Forensic backtest ───
    print(f"\n═══ PASO 4: FORENSIC BACKTEST ({len(test_grails)} combos) ═══")
    start = time.time()
    results = []

    if args.workers <= 1:
        # Sequential
        for i, g in enumerate(test_grails):
            sym = g['symbol'].replace('/USDT:USDT', '')
            print(f"  [{i+1}/{len(test_grails)}] {g['strategy']} × {sym} × {g['timeframe']}...", end='', flush=True)
            r = run_forensic_single(g, DB_PATH)
            results.append(r)
            status = r.get('status', '?')
            wr = r.get('forensic_wr', 0)
            print(f" → {status} (WR={wr:.1f}%)")
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for g in test_grails:
                fut = pool.submit(run_forensic_single, g, DB_PATH)
                futures[fut] = g

            done = 0
            for fut in as_completed(futures):
                done += 1
                g = futures[fut]
                sym = g['symbol'].replace('/USDT:USDT', '')
                try:
                    r = fut.result(timeout=300)
                    results.append(r)
                    status = r.get('status', '?')
                    wr = r.get('forensic_wr', 0)
                    print(f"  [{done}/{len(test_grails)}] {g['strategy']} × {sym} × {g['timeframe']} → {status} (WR={wr:.1f}%)")
                except Exception as e:
                    results.append({
                        'strategy': g['strategy'], 'symbol': g['symbol'],
                        'timeframe': g['timeframe'], 'status': 'ERROR', 'error': str(e)
                    })
                    print(f"  [{done}/{len(test_grails)}] {g['strategy']} × {sym} × {g['timeframe']} → ERROR: {e}")

    elapsed = time.time() - start

    # ─── PASO 5: Resumen y output ───
    print(f"\n═══ PASO 5: RESULTADOS ═══")

    confirmed = [r for r in results if r.get('status') == 'CONFIRMED']
    warning = [r for r in results if r.get('status') == 'WARNING']
    inflated = [r for r in results if r.get('status') == 'INFLATED']
    rejected = [r for r in results if r.get('status') == 'REJECTED']
    errors = [r for r in results if r.get('status') in ('ERROR', 'NO_DATA', 'NO_STRATEGY')]

    print(f"  Total testeados:  {len(results)}")
    print(f"  ✅ CONFIRMED:     {len(confirmed)} (gap ≤ 5pp)")
    print(f"  ⚠️  WARNING:       {len(warning)} (gap 5-10pp)")
    print(f"  🔴 INFLATED:      {len(inflated)} (gap > 10pp)")
    print(f"  ❌ REJECTED:      {len(rejected)} (gates)")
    print(f"  💀 ERRORS:        {len(errors)}")
    print(f"  Tiempo:           {elapsed/60:.1f} min")

    # Tabla de aprobados
    approved = confirmed + warning
    if approved:
        approved.sort(key=lambda x: x.get('forensic_wr', 0), reverse=True)
        print(f"\n  {'Strategy':<30} {'Symbol':<12} {'TF':>3} {'OptunaWR':>8} {'RealWR':>7} {'Gap':>6} {'PnL%':>7} {'Trades':>6} {'PF':>5} {'Status'}")
        print(f"  {'─'*30} {'─'*12} {'─'*3} {'─'*8} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*5} {'─'*10}")
        for r in approved:
            sym = r['symbol'].replace('/USDT:USDT', '')
            print(f"  {r['strategy']:<30} {sym:<12} {r['timeframe']:>3} {r['optuna_wr']:>7.1f}% {r['forensic_wr']:>6.1f}% {r['gap_pp']:>+5.1f} {r['pnl_pct']:>+6.1f}% {r['trades']:>6} {r['profit_factor']:>5.2f} {r['status']}")

    # Mejor combo por símbolo (la joya)
    if approved:
        best_by_symbol = {}
        for r in approved:
            sym = r['symbol']
            if sym not in best_by_symbol or r.get('forensic_wr', 0) > best_by_symbol[sym].get('forensic_wr', 0):
                best_by_symbol[sym] = r

        print(f"\n  🏆 MEJOR COMBO POR SÍMBOLO:")
        for sym, r in sorted(best_by_symbol.items(), key=lambda x: x[1].get('forensic_wr', 0), reverse=True):
            sym_short = sym.replace('/USDT:USDT', '')
            print(f"     {r['strategy']} × {sym_short} × {r['timeframe']} → WR={r['forensic_wr']:.1f}% PnL={r['pnl_pct']:+.1f}%")

    # ─── SAVE ───
    output_name = args.output or f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    output_path = os.path.join(DATA_DIR, output_name)

    output = {
        'timestamp': datetime.now().isoformat(),
        'source_progress': args.progress,
        'expand_tf': args.expand_tf,
        'db': DB_PATH,
        'total_tested': len(results),
        'confirmed': len(confirmed),
        'warning': len(warning),
        'inflated': len(inflated),
        'rejected': len(rejected),
        'errors': len(errors),
        'elapsed_min': round(elapsed / 60, 1),
        'gates': {
            'min_wr': GATE_MIN_WR,
            'max_gap': GATE_MAX_GAP,
            'min_trades': GATE_MIN_TRADES,
            'min_pf': GATE_MIN_PF,
        },
        'results': results,
        'approved': approved,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Resultados guardados: {output_path}")

    # Save approved separately for easy injection
    if approved:
        approved_path = os.path.join(DATA_DIR, output_name.replace('.json', '_APPROVED.json'))
        with open(approved_path, 'w') as f:
            json.dump(approved, f, indent=2, default=str)
        print(f"  Aprobados: {approved_path} ({len(approved)} grails)")

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETO — {len(approved)} grails validados de {len(results)} testeados")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hetzner Full Pipeline: Optuna + Forensic Backtest')

    # Optuna (opcional — si se da --combos, corre Optuna primero)
    parser.add_argument('--combos', type=str, help='Combos file para Optuna (si se omite, solo corre forensic)')
    parser.add_argument('--trials', type=int, default=30, help='Trials por combo en Optuna')

    # Forensic (obligatorio — o viene de Optuna o de --progress)
    parser.add_argument('--progress', type=str, help='Progress file de Optuna con grails')
    parser.add_argument('--expand-tf', action='store_true', default=True,
                        help='Testear cada grail en TODOS los TFs (5m,15m,1h,4h,1d) — ACTIVADO por defecto')
    parser.add_argument('--no-expand-tf', action='store_true', help='Solo testear el TF original (desactiva expand)')
    parser.add_argument('--workers', type=int, default=4, help='Workers paralelos')
    parser.add_argument('--output', type=str, help='Nombre del archivo de output')
    parser.add_argument('--db', type=str, help='Override DB path')

    args = parser.parse_args()

    if args.db:
        DB_PATH = args.db

    # --no-expand-tf desactiva el expand
    if args.no_expand_tf:
        args.expand_tf = False

    if not args.progress and not args.combos:
        # Buscar el progress más reciente
        progress_files = sorted(
            [f for f in os.listdir(DATA_DIR) if 'progress' in f and f.endswith('.json')],
            key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)),
            reverse=True
        )
        if progress_files:
            args.progress = os.path.join(DATA_DIR, progress_files[0])
            print(f"  Auto-detected progress: {args.progress}")
        else:
            print("ERROR: Necesito --progress o --combos. No se encontró progress file.")
            sys.exit(1)

    run_pipeline(args)
