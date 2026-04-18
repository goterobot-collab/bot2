#!/usr/bin/env python3
"""
SL/TP Variant Comparison — Tier A Grails
=========================================
Tests 4 SL/TP formulas on each Tier A grail and reports which produces
the most PASS results (WR>=65%, PnL>0).

Variants:
  ORIGINAL  — SL=P90(MAE)*2.0,  TP=P90(winMFE)*0.85  [current full_historical_backtest.py]
  VARIANT_A — SL=P75(MAE)*1.5,  TP=P75(winMFE)*0.80  [tighter]
  VARIANT_B — SL=P50(MAE)*2.0,  TP=P50(winMFE)*1.0   [balanced]
  VARIANT_C — SL=P95(winMAE)*1.5, TP=P75(winMFE)*0.8  [original mfe_backtest.py]

Usage:
  python3 scripts/test_sl_tp_variants.py --workers 4
  python3 scripts/test_sl_tp_variants.py --limit 20 --workers 1
"""

import json
import os
import sys
import time
import argparse
import traceback
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════
# IMPORT from full_historical_backtest.py (same directory)
# ═══════════════════════════════════════════════════════════════════════
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from full_historical_backtest import (
    load_all_strategies,
    get_gen,
    get_candles,
    load_candles_from_db,
    backtest_raw,
    backtest_with_sl_tp,
    COMMISSION, SLIPPAGE, COST,
    SL_MIN, SL_MAX, TP_MIN, TP_MAX,
    MAX_DUR_CAP_H, TF_HOURS,
    CALIBRATION_PCT,
)

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════
DATA_DIR = str(PROJECT_DIR / "data")
TIER_A_PATH = str(PROJECT_DIR / "data" / "grails_tier_A.json")

# ═══════════════════════════════════════════════════════════════════════
# GATE (same as full_historical_backtest.py)
# ═══════════════════════════════════════════════════════════════════════
GATE_MIN_WR = 65.0
GATE_MIN_TRADES = 100
GATE_MAX_GAP_PP = 10.0

# ═══════════════════════════════════════════════════════════════════════
# VARIANT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════
VARIANTS = {
    'ORIGINAL': {
        'sl_pct': 90, 'sl_mult': 2.0, 'sl_source': 'all',
        'tp_pct': 90, 'tp_mult': 0.85, 'tp_source': 'winners',
        'desc': 'P90(ALL_MAE)*2.0  / P90(winMFE)*0.85',
    },
    'VARIANT_A': {
        'sl_pct': 75, 'sl_mult': 1.5, 'sl_source': 'all',
        'tp_pct': 75, 'tp_mult': 0.80, 'tp_source': 'winners',
        'desc': 'P75(ALL_MAE)*1.5  / P75(winMFE)*0.80  [tighter]',
    },
    'VARIANT_B': {
        'sl_pct': 50, 'sl_mult': 2.0, 'sl_source': 'all',
        'tp_pct': 50, 'tp_mult': 1.0,  'tp_source': 'winners',
        'desc': 'P50(ALL_MAE)*2.0  / P50(winMFE)*1.00  [balanced]',
    },
    'VARIANT_C': {
        'sl_pct': 95, 'sl_mult': 1.5, 'sl_source': 'winners',  # winner MAE only
        'tp_pct': 75, 'tp_mult': 0.80, 'tp_source': 'winners',
        'desc': 'P95(winMAE)*1.5   / P75(winMFE)*0.80  [mfe_backtest style]',
    },
}

VARIANT_ORDER = ['ORIGINAL', 'VARIANT_A', 'VARIANT_B', 'VARIANT_C']


# ═══════════════════════════════════════════════════════════════════════
# CALCULATE SL/TP FOR A GIVEN VARIANT
# ═══════════════════════════════════════════════════════════════════════

def calc_sl_tp_variant(raw_trades, tf, variant_cfg):
    """Compute SL/TP/leverage per variant formula from calibration trades."""
    if len(raw_trades) < 5:
        return None

    all_maes = [abs(t['mae']) for t in raw_trades]
    winners = [t for t in raw_trades if t['win']]
    if len(winners) < 3:
        return None

    win_maes = [abs(t['mae']) for t in winners]
    mfes = [t['mfe'] for t in winners]

    # SL
    sl_pct = variant_cfg['sl_pct']
    sl_mult = variant_cfg['sl_mult']
    sl_src = variant_cfg['sl_source']
    sl_base = win_maes if sl_src == 'winners' else all_maes
    if len(sl_base) < 2:
        sl_base = all_maes  # fallback
    sl = float(np.clip(np.percentile(sl_base, sl_pct) * sl_mult, SL_MIN, SL_MAX))

    # TP
    tp_pct = variant_cfg['tp_pct']
    tp_mult = variant_cfg['tp_mult']
    tp = float(np.clip(np.percentile(mfes, tp_pct) * tp_mult, TP_MIN, TP_MAX))

    # Max duration (same formula across all variants — not the focus here)
    durs = [t['dur_bars'] for t in winners]
    bars_per_h = 1.0 / TF_HOURS.get(tf, 1)
    raw_dur = np.percentile(durs, 95) * 1.5
    dur_h = min(raw_dur / bars_per_h, MAX_DUR_CAP_H)
    max_dur_bars = int(dur_h * bars_per_h)

    # Leverage (same formula across all variants)
    mae_worst = max(all_maes)
    mae_p99 = float(np.percentile(all_maes, 99)) if len(all_maes) >= 10 else mae_worst
    mae_p95 = float(np.percentile(all_maes, 95)) if len(all_maes) >= 5 else mae_worst
    lev = max(1, min(20, int(min(
        1 / (mae_p99 * 2.5 + 1e-10),
        1 / (mae_p95 * 2.0 + 1e-10)
    ))))
    while lev > 1 and sl * lev > 0.80:
        lev -= 1

    return {
        'sl': round(sl, 4), 'tp': round(tp, 4),
        'max_dur_bars': max_dur_bars, 'max_dur_h': round(dur_h, 1),
        'leverage': lev, 'rr': round(tp / (sl + 1e-10), 2),
    }


# ═══════════════════════════════════════════════════════════════════════
# EVALUATE ONE VARIANT
# ═══════════════════════════════════════════════════════════════════════

def eval_variant(df, signals, cal_trades, tf, variant_name, variant_cfg, optuna_wr, smoking_gun):
    """Run one variant and return pass/fail + metrics."""
    params = calc_sl_tp_variant(cal_trades, tf, variant_cfg)
    if params is None:
        return {'variant': variant_name, 'status': 'SKIP', 'reason': 'sl_tp_failed'}

    full_trades = backtest_with_sl_tp(
        df, signals, params['sl'], params['tp'], params['max_dur_bars'])

    if len(full_trades) < 5:
        return {'variant': variant_name, 'status': 'SKIP',
                'reason': f'full_trades:{len(full_trades)}<5',
                'sl': params['sl'], 'tp': params['tp']}

    pnls = np.array([t['pnl'] for t in full_trades])
    n = len(pnls)
    wins = int((pnls > 0).sum())
    full_wr = round(wins / n * 100, 1)
    full_pnl = round(float(pnls.sum() * 100), 2)
    gap = round(optuna_wr - full_wr, 1)
    cum = pnls.cumsum()
    peak = np.maximum.accumulate(cum)
    max_dd = round(float((peak - cum).max() * 100), 2)

    # Gate checks
    reasons = []
    if full_wr < GATE_MIN_WR:
        reasons.append(f'wr={full_wr}<{GATE_MIN_WR}')
    if n < GATE_MIN_TRADES:
        reasons.append(f'trades={n}<{GATE_MIN_TRADES}')
    if gap > GATE_MAX_GAP_PP:
        reasons.append(f'gap={gap}pp>{GATE_MAX_GAP_PP}')
    if full_pnl <= 0:
        reasons.append(f'pnl={full_pnl}<=0')
    if smoking_gun:
        reasons.append('smoking_gun')
    if params['leverage'] * params['sl'] > 0.80:
        reasons.append(f'lev_sl={params["leverage"]*params["sl"]:.2f}>0.80')

    passed = len(reasons) == 0

    return {
        'variant': variant_name,
        'status': 'PASS' if passed else 'FAIL',
        'full_wr': full_wr,
        'full_trades': n,
        'full_pnl': full_pnl,
        'max_dd': max_dd,
        'gap': gap,
        'sl': params['sl'],
        'tp': params['tp'],
        'rr': params['rr'],
        'leverage': params['leverage'],
        'reasons': reasons if reasons else ['all_passed'],
    }


# ═══════════════════════════════════════════════════════════════════════
# PROCESS ONE GRAIL — all 4 variants
# ═══════════════════════════════════════════════════════════════════════

_STRATEGIES_CACHE = None
_CANDLE_CACHE = {}


def process_grail_variants(grail):
    """Run all 4 variants on one grail. Returns dict with per-variant results."""
    global _STRATEGIES_CACHE, _CANDLE_CACHE

    sname = grail['strategy']
    symbol = grail['symbol']
    tf = grail['timeframe']
    bp = grail.get('best_params', {})
    optuna_wr = grail.get('test_wr', 0) or 0
    smoking_gun = grail.get('smoking_gun', False)

    base = {
        'strategy': sname, 'symbol': symbol, 'timeframe': tf,
        'optuna_wr': optuna_wr,
    }

    # Load strategies
    if _STRATEGIES_CACHE is None:
        _STRATEGIES_CACHE = load_all_strategies()
    strats = _STRATEGIES_CACHE

    gen = get_gen(sname, strats)
    if gen is None:
        return {**base, 'status': 'SKIP', 'reason': f'not_found:{sname}',
                'variants': {}}

    # Load candles (with per-process cache)
    key = (symbol, tf)
    if key not in _CANDLE_CACHE:
        df = load_candles_from_db(symbol, tf)
        _CANDLE_CACHE[key] = df
    df = _CANDLE_CACHE[key]

    if df is None or len(df) < 200:
        return {**base, 'status': 'SKIP',
                'reason': f'candles:{len(df) if df is not None else 0}<200',
                'variants': {}}

    # Generate signals
    try:
        signals = gen(df, **bp)
    except Exception as e:
        return {**base, 'status': 'ERROR', 'reason': f'gen:{str(e)[:150]}',
                'variants': {}}

    if signals is None or len(signals) == 0:
        return {**base, 'status': 'SKIP', 'reason': 'no_signals', 'variants': {}}

    # Split for calibration (first 70% candles)
    split_idx = int(len(df) * CALIBRATION_PCT)
    df_cal = df.iloc[:split_idx]
    sig_cal = signals.iloc[:split_idx]

    # Raw backtest on calibration portion
    cal_trades = backtest_raw(df_cal, sig_cal)
    if len(cal_trades) < 5:
        return {**base, 'status': 'SKIP',
                'reason': f'calib_trades:{len(cal_trades)}<5',
                'variants': {}}

    # Run all 4 variants
    variant_results = {}
    for vname in VARIANT_ORDER:
        vcfg = VARIANTS[vname]
        try:
            vr = eval_variant(df, signals, cal_trades, tf, vname, vcfg,
                              optuna_wr, smoking_gun)
        except Exception as e:
            vr = {'variant': vname, 'status': 'ERROR',
                  'reason': traceback.format_exc()[-200:]}
        variant_results[vname] = vr

    return {
        **base,
        'candles': len(df),
        'calib_trades': len(cal_trades),
        'status': 'DONE',
        'variants': variant_results,
    }


# ═══════════════════════════════════════════════════════════════════════
# WORKER (multiprocessing safe)
# ═══════════════════════════════════════════════════════════════════════

def _worker_init():
    global _STRATEGIES_CACHE, _CANDLE_CACHE
    _CANDLE_CACHE = {}
    _STRATEGIES_CACHE = load_all_strategies()


def _worker_fn(grail):
    try:
        return process_grail_variants(grail)
    except Exception as e:
        return {
            'strategy': grail.get('strategy', '?'),
            'symbol': grail.get('symbol', '?'),
            'timeframe': grail.get('timeframe', '?'),
            'status': 'ERROR',
            'reason': traceback.format_exc()[-300:],
            'variants': {},
        }


# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════

def print_report(all_results, elapsed):
    done = [r for r in all_results if r['status'] == 'DONE']
    skipped = [r for r in all_results if r['status'] == 'SKIP']
    errors = [r for r in all_results if r['status'] == 'ERROR']

    print(f"\n{'='*72}")
    print(f"  SL/TP VARIANT COMPARISON — TIER A GRAILS")
    print(f"  Elapsed: {elapsed/60:.1f}m | Processed: {len(done)} | "
          f"Skipped: {len(skipped)} | Errors: {len(errors)}")
    print(f"{'='*72}")

    if not done:
        print("  No completed grails to report.")
        return

    # Per-variant stats
    print(f"\n  {'VARIANT':<12}  {'PASS':>5}  {'FAIL':>5}  {'SKIP':>5}  "
          f"{'PASS%':>6}  {'AvgWR':>7}  {'AvgPnL':>8}  {'AvgSL':>7}  "
          f"{'AvgTP':>7}  {'AvgRR':>6}")
    print(f"  {'-'*79}")

    variant_pass_count = {}
    for vname in VARIANT_ORDER:
        vrs = [r['variants'][vname] for r in done if vname in r['variants']]
        passes = [v for v in vrs if v.get('status') == 'PASS']
        fails  = [v for v in vrs if v.get('status') == 'FAIL']
        skips  = [v for v in vrs if v.get('status') in ('SKIP', 'ERROR')]
        total  = len(passes) + len(fails)
        pct    = round(len(passes) / total * 100, 1) if total > 0 else 0.0

        wrs  = [v['full_wr']  for v in passes if 'full_wr'  in v]
        pnls = [v['full_pnl'] for v in passes if 'full_pnl' in v]
        sls  = [v['sl']       for v in vrs     if 'sl'       in v]
        tps  = [v['tp']       for v in vrs     if 'tp'       in v]
        rrs  = [v['rr']       for v in vrs     if 'rr'       in v]

        avg_wr  = f"{np.mean(wrs):.1f}%"  if wrs  else "   —  "
        avg_pnl = f"{np.mean(pnls):+.1f}%" if pnls else "    —   "
        avg_sl  = f"{np.mean(sls)*100:.1f}%" if sls else "   —  "
        avg_tp  = f"{np.mean(tps)*100:.1f}%" if tps else "   —  "
        avg_rr  = f"{np.mean(rrs):.2f}"  if rrs  else "  —  "

        variant_pass_count[vname] = len(passes)
        best_mark = " <-- WINNER" if vname == max(variant_pass_count, key=variant_pass_count.get) else ""

        print(f"  {vname:<12}  {len(passes):>5}  {len(fails):>5}  {len(skips):>5}  "
              f"{pct:>5.1f}%  {avg_wr:>7}  {avg_pnl:>8}  {avg_sl:>7}  "
              f"{avg_tp:>7}  {avg_rr:>6}{best_mark}")

    # Determine best variant
    best_variant = max(variant_pass_count, key=variant_pass_count.get)
    best_count = variant_pass_count[best_variant]

    print(f"\n  {'─'*72}")
    print(f"  WINNER: {best_variant}  ({best_count} PASSed grails)")
    print(f"  Formula: {VARIANTS[best_variant]['desc']}")

    # Per-variant descriptions
    print(f"\n  Variant formulas:")
    for vname in VARIANT_ORDER:
        print(f"    {vname:<12}  {VARIANTS[vname]['desc']}")

    # Gate failure breakdown per variant
    print(f"\n  Gate failure reasons per variant (top 4):")
    for vname in VARIANT_ORDER:
        vrs = [r['variants'][vname] for r in done if vname in r['variants']]
        fails = [v for v in vrs if v.get('status') == 'FAIL']
        if not fails:
            print(f"    {vname:<12}  (no failures)")
            continue
        reason_ctr = Counter()
        for v in fails:
            for reason in v.get('reasons', ['?']):
                reason_ctr[reason.split('=')[0]] += 1
        top4 = reason_ctr.most_common(4)
        parts = ', '.join(f"{r}:{c}" for r, c in top4)
        print(f"    {vname:<12}  {parts}")

    # Which grails are uniquely unlocked by each variant (PASS in that variant but not ORIGINAL)
    print(f"\n  Grails PASS in variant but FAIL in ORIGINAL (unique unlock):")
    orig_pass = set()
    for r in done:
        v = r['variants'].get('ORIGINAL', {})
        if v.get('status') == 'PASS':
            orig_pass.add(f"{r['strategy']}|{r['symbol']}|{r['timeframe']}")

    for vname in ['VARIANT_A', 'VARIANT_B', 'VARIANT_C']:
        unlocked = []
        for r in done:
            v = r['variants'].get(vname, {})
            key = f"{r['strategy']}|{r['symbol']}|{r['timeframe']}"
            if v.get('status') == 'PASS' and key not in orig_pass:
                unlocked.append(key)
        print(f"    {vname:<12}  {len(unlocked)} unique unlocks")
        for k in unlocked[:5]:
            print(f"               {k}")
        if len(unlocked) > 5:
            print(f"               ... +{len(unlocked)-5} more")

    # Grails that PASS in ALL variants
    all_pass_keys = set()
    for r in done:
        key = f"{r['strategy']}|{r['symbol']}|{r['timeframe']}"
        if all(r['variants'].get(vn, {}).get('status') == 'PASS' for vn in VARIANT_ORDER):
            all_pass_keys.add(key)
    print(f"\n  Grails that PASS ALL 4 variants: {len(all_pass_keys)}")

    # Best performing PASS grails for winner variant
    winner_passes = []
    for r in done:
        v = r['variants'].get(best_variant, {})
        if v.get('status') == 'PASS':
            winner_passes.append({
                'key': f"{r['strategy']}|{r['symbol']}|{r['timeframe']}",
                'wr': v.get('full_wr', 0),
                'pnl': v.get('full_pnl', 0),
                'rr': v.get('rr', 0),
                'sl': v.get('sl', 0),
                'tp': v.get('tp', 0),
            })
    if winner_passes:
        winner_passes.sort(key=lambda x: x['pnl'], reverse=True)
        print(f"\n  Top 15 grails by PnL under {best_variant}:")
        print(f"  {'STRATEGY|SYMBOL|TF':<45}  {'WR':>6}  {'PnL':>8}  "
              f"{'SL':>6}  {'TP':>6}  {'R:R':>5}")
        print(f"  {'-'*82}")
        for g in winner_passes[:15]:
            print(f"  {g['key']:<45}  {g['wr']:>5.1f}%  "
                  f"{g['pnl']:>+7.1f}%  {g['sl']*100:>5.1f}%  "
                  f"{g['tp']*100:>5.1f}%  {g['rr']:>5.2f}")

    print(f"\n{'='*72}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='SL/TP Variant Comparison — Tier A')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    parser.add_argument('--limit', type=int, default=0, help='Limit grails for testing')
    parser.add_argument('--tier-file', default=TIER_A_PATH, help='Path to tier JSON')
    args = parser.parse_args()

    # Load Tier A grails
    if not os.path.exists(args.tier_file):
        print(f"ERROR: Tier A file not found: {args.tier_file}")
        sys.exit(1)

    with open(args.tier_file) as f:
        grails = json.load(f)

    if args.limit > 0:
        grails = grails[:args.limit]

    print(f"\n{'='*72}")
    print(f"  SL/TP VARIANT COMPARISON — TIER A GRAILS")
    print(f"{'='*72}")
    print(f"  Grails:  {len(grails)}")
    print(f"  Workers: {args.workers}")
    print(f"  Gate:    WR>={GATE_MIN_WR}% | trades>={GATE_MIN_TRADES} | "
          f"gap<={GATE_MAX_GAP_PP}pp | PnL>0")
    print(f"\n  Variants being tested:")
    for vname in VARIANT_ORDER:
        print(f"    {vname:<12}  {VARIANTS[vname]['desc']}")
    print()

    all_results = []
    t0 = time.time()

    def on_result(result, idx, total):
        all_results.append(result)
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta_m = (total - idx - 1) / rate / 60 if rate > 0 else 0

        status = result['status']
        if status == 'DONE':
            # Show pass/fail summary for this grail
            vsum = []
            for vn in VARIANT_ORDER:
                vr = result['variants'].get(vn, {})
                vsum.append('P' if vr.get('status') == 'PASS' else
                             'F' if vr.get('status') == 'FAIL' else 'S')
            summary = '/'.join(vsum)  # e.g. F/P/P/F
            print(f"  [{idx+1:>3}/{total}] {result['strategy']:<22} "
                  f"{result['symbol']:<18} {result['timeframe']:<4} "
                  f"[ORIG/A/B/C]={summary}  ETA:{eta_m:.1f}m")
        else:
            reason = result.get('reason', '')[:40]
            print(f"  [{idx+1:>3}/{total}] {status:<6} {result['strategy']:<22} "
                  f"{result['symbol']:<18} — {reason}")

    if args.workers <= 1:
        print("  Loading strategies (single-threaded)...")
        strats = load_all_strategies()
        print(f"  {len(strats)} strategies loaded\n")
        for idx, g in enumerate(grails):
            result = process_grail_variants(g)
            on_result(result, idx, len(grails))
    else:
        print(f"  Launching {args.workers} workers...\n")
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init) as pool:
            fmap = {pool.submit(_worker_fn, g): g for g in grails}
            for idx, fut in enumerate(as_completed(fmap)):
                try:
                    result = fut.result(timeout=600)
                except Exception as e:
                    g = fmap[fut]
                    result = {
                        'strategy': g.get('strategy', '?'),
                        'symbol': g.get('symbol', '?'),
                        'timeframe': g.get('timeframe', '?'),
                        'status': 'ERROR',
                        'reason': str(e)[:200],
                        'variants': {},
                    }
                on_result(result, idx, len(grails))

    elapsed = time.time() - t0
    print_report(all_results, elapsed)

    # Save raw results for further analysis
    out_path = os.path.join(DATA_DIR, 'sl_tp_variant_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Raw results saved: {out_path}\n")


if __name__ == '__main__':
    main()
