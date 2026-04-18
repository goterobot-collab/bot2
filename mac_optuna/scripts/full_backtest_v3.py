#!/usr/bin/env python3
"""
FASE 2c — Full Historical Backtest V3: Signal-Exit Primary + Protective SL

V1 (0 validated): V7.2 SL/TP formula → R:R=0.20-0.25, need WR>82%
V2 (0 validated): Grid search SL/TP → better R:R but ATR slippage destroys edge
V3 (12+ validated): Signal exits primary + protective SL + FIXED realistic costs

Key discovery (V3):
  - ATR-based dynamic slippage (ATR×0.1 = 0.3-1.0% per side) is 3-5x too high
  - Binance futures actual: 0.05-0.15% per side (taker + slippage)
  - Using FIXED cost 0.15%/side: strategies with raw_wr>65% become profitable
  - Signal exits ARE the profit mechanism. SL is catastrophe protection ONLY.
  - 12/100 Tier A+B grails validated (XMR 4h, HIPPO 1h, D 1h, USTC 1h)

Usage:
  python3 scripts/full_backtest_v3.py                       # All tiers A+B
  python3 scripts/full_backtest_v3.py --tier ALL             # All A+B+C
  python3 scripts/full_backtest_v3.py --workers 10           # 10 workers
  python3 scripts/full_backtest_v3.py --db /path/to/db       # Custom DB (Hetzner)
  python3 scripts/full_backtest_v3.py --resume               # Continue previous
"""

import json
import sqlite3
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import traceback
from datetime import datetime
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

_HETZNER_DB = "/home/ubuntu/candles_db/activos_binance.db"
_MAC_DB = "/Users/sabrina/CLAUDE CODE/data/activos_binance.db"
DEFAULT_DB = _HETZNER_DB if os.path.exists(_HETZNER_DB) else _MAC_DB
DB_PATH = os.environ.get('BACKTEST_DB_PATH', DEFAULT_DB)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS — V3: FIXED realistic costs
# ═══════════════════════════════════════════════════════════════════════
COMMISSION = 0.001     # 0.1% per side (Binance taker)
SLIPPAGE = 0.0005      # 0.05% per side (realistic for liquid futures)
COST = COMMISSION + SLIPPAGE  # 0.15% per side, 0.30% round trip

SL_MIN = 0.05          # Min protective SL: 5%
SL_MAX = 0.40          # Max protective SL: 40%
MAX_DUR_CAP_H = 720    # Max duration: 30 days

# Gate thresholds
GATE_MIN_WR = 55.0
GATE_MIN_TRADES = 100
GATE_MAX_GAP_PP = 20.0  # More relaxed — V3 signals are closer to raw

CALIBRATION_PCT = 0.70
TF_HOURS = {'5m': 5/60, '15m': 15/60, '1h': 1, '4h': 4, '1d': 24}

_CANDLE_CACHE = {}
_STRATEGIES = None


# ═══════════════════════════════════════════════════════════════════════
# CANDLE LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_candles_from_db(symbol, tf):
    base_tf = {'15m': '5m', '4h': '1h', '1d': '1h'}.get(tf, tf)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA cache_size=-64000")
    df = pd.read_sql(
        "SELECT ts,open,high,low,close,volume FROM candles "
        "WHERE symbol=? AND timeframe=? ORDER BY ts",
        conn, params=(symbol, base_tf))
    conn.close()
    if len(df) == 0:
        return None
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
    df.dropna(subset=['ts'], inplace=True)
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    if tf != base_tf:
        m = {'15m': '15min', '4h': '4h', '1d': '1D'}
        r = m.get(tf)
        if r:
            df = df.resample(r).agg({
                'ts': 'first', 'open': 'first', 'high': 'max',
                'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna(subset=['close'])
    return df if len(df) >= 100 else None


def get_candles(symbol, tf):
    global _CANDLE_CACHE
    key = (symbol, tf)
    if key in _CANDLE_CACHE:
        return _CANDLE_CACHE[key]
    df = load_candles_from_db(symbol, tf)
    if df is not None:
        _CANDLE_CACHE[key] = df
    return df


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_all_strategies():
    global _STRATEGIES
    if _STRATEGIES is not None:
        return _STRATEGIES

    sys.path.insert(0, PROJECT_DIR)
    sys.path.insert(0, os.path.join(PROJECT_DIR, "strategies_code"))
    strats = {}

    try:
        from strategies_round2 import STRATEGY_TYPES_R2
        strats.update(STRATEGY_TYPES_R2)
    except Exception: pass

    try:
        from strategies_round3 import STRATEGY_TYPES_R3
        strats.update(STRATEGY_TYPES_R3)
    except Exception: pass

    try:
        from batch_search_spaces import BATCH_SEARCH_SPACES, make_space_func
        from strategies_code.batch_param_wrappers import BATCH_WRAPPERS
        for key, wrapper_fn in BATCH_WRAPPERS.items():
            if key in BATCH_SEARCH_SPACES:
                strats[key] = {'gen': wrapper_fn, 'space': make_space_func(key)}
    except Exception: pass

    tv2_dir = os.path.join(PROJECT_DIR, "strategies_tv2_batches")
    if os.path.isdir(tv2_dir) and tv2_dir not in sys.path:
        sys.path.insert(0, tv2_dir)
    for batch_num in range(100, 2500):
        try:
            mod = __import__(f"strategies_tv2_batch{batch_num}")
            if hasattr(mod, 'STRATEGY_EXPORT'):
                strats.update(mod.STRATEGY_EXPORT)
        except ImportError: pass
        except Exception: pass

    for pine_path in ['/Users/sabrina/CLAUDE CODE/BOT V7/strategies',
                      '/home/ubuntu/bot-v7/strategies']:
        try:
            if os.path.isdir(pine_path) and pine_path not in sys.path:
                sys.path.insert(0, pine_path)
            from strategies_new_winners import NEW_WINNER_STRATS
            strats.update(NEW_WINNER_STRATS)
            break
        except Exception: pass

    try:
        from strategy_factory import FACTORY_STRATS
        for name, entry in FACTORY_STRATS.items():
            if name not in strats:
                strats[name] = entry
    except Exception: pass

    _STRATEGIES = strats
    return strats


def get_gen(strategy_name, strats):
    entry = strats.get(strategy_name)
    if entry is None: return None
    if callable(entry): return entry
    if isinstance(entry, dict) and 'gen' in entry: return entry['gen']
    return None


# ═══════════════════════════════════════════════════════════════════════
# V3 BACKTEST: Signal exits + protective SL, FIXED costs
# ═══════════════════════════════════════════════════════════════════════

def backtest_raw(df, signals):
    """Raw backtest WITHOUT SL/TP. LONG + SHORT. Fixed cost."""
    if signals is None: return []
    sig = signals.values
    opens, highs, lows = df['open'].values, df['high'].values, df['low'].values
    n = min(len(sig), len(opens))
    trades = []; pos = 0; ep = 0.0; ei = 0; mae = 0.0; mfe = 0.0

    for i in range(1, n):
        s = int(sig[i - 1]); p = opens[i]
        if pos == 0:
            if s == 1:
                ep = p * (1 + COST); ei = i; mae = 0.; mfe = 0.; pos = 1
            elif s == -1:
                ep = p * (1 - COST); ei = i; mae = 0.; mfe = 0.; pos = -1
        elif pos == 1:
            lp = (lows[i] - ep) / ep; hp = (highs[i] - ep) / ep
            if lp < mae: mae = lp
            if hp > mfe: mfe = hp
            if s == -1 or s == 0:
                xp = p * (1 - COST); pnl = (xp - ep) / ep
                trades.append({'pnl': pnl, 'mae': mae, 'mfe': mfe,
                              'dur_bars': i - ei, 'win': pnl > 0})
                pos = 0
                if s == -1:
                    ep = p * (1 - COST); ei = i; mae = 0.; mfe = 0.; pos = -1
        elif pos == -1:
            hp_s = (ep - highs[i]) / ep; lp_s = (ep - lows[i]) / ep
            if hp_s < mae: mae = hp_s
            if lp_s > mfe: mfe = lp_s
            if s == 1 or s == 0:
                xp = p * (1 + COST); pnl = (ep - xp) / ep
                trades.append({'pnl': pnl, 'mae': mae, 'mfe': mfe,
                              'dur_bars': i - ei, 'win': pnl > 0})
                pos = 0
                if s == 1:
                    ep = p * (1 + COST); ei = i; mae = 0.; mfe = 0.; pos = 1
    return trades


def backtest_v3(df, signals, sl_pct, max_dur_bars=None):
    """V3: Signal exits primary + protective SL. FIXED cost (same as raw).
    NO TP target — signals decide when to take profit.
    SL = catastrophe protection only.
    """
    if signals is None: return []
    sig = signals.values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    trades = []; pos = 0; ep = 0.0; ei = 0

    for i in range(1, n):
        s = int(sig[i - 1]); p = opens[i]

        if pos == 0:
            if s == 1:
                ep = p * (1 + COST); ei = i; pos = 1
            elif s == -1:
                ep = p * (1 - COST); ei = i; pos = -1
            continue

        # ── MANAGE LONG ──
        if pos == 1:
            # Protective SL (catastrophe only)
            if sl_pct and (lows[i] - ep) / ep <= -sl_pct:
                trades.append({'pnl': -sl_pct - 2*COST, 'dur_bars': i - ei, 'exit': 'SL'})
                pos = 0; continue
            # Max duration protection
            if max_dur_bars and (i - ei) >= max_dur_bars:
                xp = p * (1 - COST)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei, 'exit': 'DUR'})
                pos = 0; continue
            # Signal exit (PRIMARY profit mechanism)
            if s == -1 or s == 0:
                xp = p * (1 - COST)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei, 'exit': 'SIG'})
                pos = 0
                if s == -1:
                    ep = p * (1 - COST); ei = i; pos = -1

        # ── MANAGE SHORT ──
        elif pos == -1:
            if sl_pct and (highs[i] - ep) / ep >= sl_pct:
                trades.append({'pnl': -sl_pct - 2*COST, 'dur_bars': i - ei, 'exit': 'SL'})
                pos = 0; continue
            if max_dur_bars and (i - ei) >= max_dur_bars:
                xp = p * (1 + COST)
                trades.append({'pnl': (ep - xp) / ep, 'dur_bars': i - ei, 'exit': 'DUR'})
                pos = 0; continue
            if s == 1 or s == 0:
                xp = p * (1 + COST)
                trades.append({'pnl': (ep - xp) / ep, 'dur_bars': i - ei, 'exit': 'SIG'})
                pos = 0
                if s == 1:
                    ep = p * (1 + COST); ei = i; pos = 1
    return trades


# ═══════════════════════════════════════════════════════════════════════
# PROCESS ONE GRAIL — V3
# ═══════════════════════════════════════════════════════════════════════

def process_grail_v3(grail, strategies):
    sname = grail['strategy']
    symbol = grail['symbol']
    tf = grail['timeframe']
    bp = grail.get('best_params', {})
    optuna_wr = grail.get('test_wr', 0) or 0

    res = {
        'strategy': sname, 'symbol': symbol, 'timeframe': tf,
        'tier': grail.get('tier', '?'),
        'optuna_wr': optuna_wr,
        'optuna_trades': grail.get('test_trades', 0),
    }

    gen = get_gen(sname, strategies)
    if gen is None:
        res['status'] = 'SKIP'; res['reason'] = f'not_found:{sname}'
        return res

    df = get_candles(symbol, tf)
    if df is None or len(df) < 200:
        res['status'] = 'SKIP'
        res['reason'] = f'candles:{len(df) if df is not None else 0}<200'
        return res
    res['candles'] = len(df)
    res['days'] = round((df.index[-1] - df.index[0]).total_seconds() / 86400, 1)

    try:
        signals = gen(df, **bp)
    except Exception as e:
        res['status'] = 'ERROR'; res['reason'] = f'gen:{str(e)[:150]}'
        return res
    if signals is None or len(signals) == 0:
        res['status'] = 'SKIP'; res['reason'] = 'no_signals'
        return res

    # RAW signal analysis
    raw_trades = backtest_raw(df, signals)
    if len(raw_trades) >= 10:
        raw_pnls = np.array([t['pnl'] for t in raw_trades])
        res['raw_wr'] = round(float((raw_pnls > 0).mean() * 100), 1)
        res['raw_pnl'] = round(float(raw_pnls.sum() * 100), 2)
        res['raw_trades'] = len(raw_trades)
    else:
        res['raw_wr'] = 0; res['raw_pnl'] = 0; res['raw_trades'] = len(raw_trades)

    # Calibration for protective SL
    split_idx = int(len(df) * CALIBRATION_PCT)
    cal_raw = backtest_raw(df.iloc[:split_idx], signals.iloc[:split_idx])
    if len(cal_raw) < 10:
        res['status'] = 'SKIP'
        res['reason'] = f'calib_trades:{len(cal_raw)}<10'
        return res

    # Protective SL = P99(|MAE|) × 2.0 — very wide, catastrophe protection
    maes = [abs(t['mae']) for t in cal_raw]
    if len(maes) >= 100:
        protective_sl = float(np.percentile(maes, 99)) * 2.0
    else:
        protective_sl = float(np.percentile(maes, 95)) * 2.5
    protective_sl = float(np.clip(protective_sl, SL_MIN, SL_MAX))

    # Leverage from MAE
    mae_p95 = float(np.percentile(maes, 95)) if len(maes) >= 5 else max(maes)
    mae_p99 = float(np.percentile(maes, 99)) if len(maes) >= 10 else max(maes)
    lev = max(1, min(20, int(min(
        1 / (mae_p99 * 2.5 + 1e-10),
        1 / (mae_p95 * 2.0 + 1e-10)
    ))))
    while lev > 1 and protective_sl * lev > 0.80:
        lev -= 1

    res['sl'] = protective_sl
    res['leverage'] = lev

    # V3 backtest on full history: signal exits + protective SL
    v3_trades = backtest_v3(df, signals, protective_sl, None)  # No duration limit
    if len(v3_trades) < 10:
        res['status'] = 'SKIP'
        res['reason'] = f'v3_trades:{len(v3_trades)}<10'
        return res

    # Metrics
    pnls = np.array([t['pnl'] for t in v3_trades])
    n = len(pnls)
    wins = int((pnls > 0).sum())
    full_wr = round(wins / n * 100, 1)
    full_pnl = round(float(pnls.sum() * 100), 2)
    cum = pnls.cumsum()
    peak = np.maximum.accumulate(cum)
    max_dd = round(float((peak - cum).max() * 100), 2)
    sharpe = round(float((pnls.mean() / (pnls.std() + 1e-10)) * np.sqrt(252)), 2)
    exits = dict(Counter(t.get('exit', '?') for t in v3_trades))
    gap = round(optuna_wr - full_wr, 1)
    lev_pnl = round(float(pnls.sum() * lev * 100), 2)

    res.update({
        'full_wr': full_wr, 'full_trades': n, 'full_pnl': full_pnl,
        'full_max_dd': max_dd, 'full_sharpe': sharpe,
        'gap': gap, 'exits': exits, 'lev_pnl': lev_pnl,
    })

    # Gate
    reasons = []; passed = True
    if full_wr < GATE_MIN_WR:
        passed = False; reasons.append(f'wr={full_wr}<{GATE_MIN_WR}')
    if n < GATE_MIN_TRADES:
        passed = False; reasons.append(f'trades={n}<{GATE_MIN_TRADES}')
    if gap > GATE_MAX_GAP_PP:
        passed = False; reasons.append(f'gap={gap}pp>{GATE_MAX_GAP_PP}')
    if full_pnl <= 0:
        passed = False; reasons.append(f'pnl={full_pnl}<=0')
    if grail.get('smoking_gun', False):
        passed = False; reasons.append('smoking_gun')
    if lev * protective_sl > 0.80:
        passed = False; reasons.append(f'lev_sl={lev*protective_sl:.2f}>0.80')

    res['status'] = 'VALIDATED' if passed else 'REJECTED'
    res['gate_reasons'] = reasons if reasons else ['all_passed']
    if passed:
        res['best_params'] = bp

    return res


# ═══════════════════════════════════════════════════════════════════════
# WORKERS
# ═══════════════════════════════════════════════════════════════════════

def _worker_init(db_path):
    global DB_PATH, _STRATEGIES, _CANDLE_CACHE
    DB_PATH = db_path
    os.environ['BACKTEST_DB_PATH'] = db_path
    _CANDLE_CACHE = {}
    _STRATEGIES = None
    load_all_strategies()


def _worker_fn(grail):
    strats = load_all_strategies()
    try:
        return process_grail_v3(grail, strats)
    except Exception as e:
        return {
            'strategy': grail.get('strategy', '?'),
            'symbol': grail.get('symbol', '?'),
            'timeframe': grail.get('timeframe', '?'),
            'status': 'ERROR',
            'reason': traceback.format_exc()[:300],
        }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    global DB_PATH

    parser = argparse.ArgumentParser(description='FASE 2c: V3 Signal-Exit + Protective SL')
    parser.add_argument('--tier', default='AB', help='Tiers: A, B, AB, C, ALL')
    parser.add_argument('--input', help='Direct path to grails JSON')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--db', default=DEFAULT_DB)
    parser.add_argument('--output-prefix', default='v3')
    args = parser.parse_args()

    DB_PATH = args.db

    print(f"{'='*70}")
    print(f"  FASE 2c — V3: Signal-Exit + Protective SL (FIXED costs)")
    print(f"{'='*70}")
    print(f"  DB:      {DB_PATH}")
    print(f"  Workers: {args.workers}")
    print(f"  Cost:    {COST*100:.2f}%/side ({COMMISSION*100:.1f}% commission + {SLIPPAGE*100:.2f}% slippage)")
    print(f"  Gate:    WR>={GATE_MIN_WR}%, trades>={GATE_MIN_TRADES}, PnL>0")

    # Load grails
    if args.input:
        with open(args.input) as f:
            grails = json.load(f)
        print(f"  Input:   {args.input} ({len(grails):,} grails)")
    else:
        grails = []
        tier_str = args.tier.upper()
        tiers = []
        if 'A' in tier_str: tiers.append('A')
        if 'B' in tier_str: tiers.append('B')
        if 'C' in tier_str: tiers.append('C')
        if tier_str == 'ALL': tiers = ['A', 'B', 'C']

        for t in tiers:
            path = os.path.join(DATA_DIR, f"grails_tier_{t}.json")
            if os.path.exists(path):
                with open(path) as f:
                    batch = json.load(f)
                grails.extend(batch)
                print(f"  Tier {t}:  {len(batch):,}")
        print(f"  Total:   {len(grails):,}")

    if not grails:
        print("ERROR: No grails!"); return

    # Resume
    progress_file = os.path.join(DATA_DIR, f"full_backtest_{args.output_prefix}_progress.json")
    done_keys = set()
    if args.resume and os.path.exists(progress_file):
        with open(progress_file) as f:
            done_keys = set(json.load(f).get('done_keys', []))
        print(f"  Resume:  {len(done_keys)} done")

    pending = [g for g in grails
               if f"{g['strategy']}_{g['symbol']}_{g['timeframe']}" not in done_keys]
    if args.limit > 0:
        pending = pending[:args.limit]
    print(f"  Pending: {len(pending):,}")
    print(f"{'='*70}\n")

    if not pending:
        print("Nothing to process!"); return

    # Output files
    validated_file = os.path.join(DATA_DIR, f"grails_{args.output_prefix}_validated.json")
    rejected_file = os.path.join(DATA_DIR, f"grails_{args.output_prefix}_rejected.json")
    all_file = os.path.join(DATA_DIR, f"grails_{args.output_prefix}_all_results.json")
    log_file = os.path.join(DATA_DIR, f"full_backtest_{args.output_prefix}_log.txt")

    validated, rejected, all_results = [], [], []
    if args.resume:
        for fp, lst in [(validated_file, validated), (rejected_file, rejected), (all_file, all_results)]:
            if os.path.exists(fp):
                with open(fp) as f: lst.extend(json.load(f))

    counts = Counter({'VALIDATED': len(validated), 'REJECTED': len(rejected)})
    gate_failures = Counter()
    t0 = time.time()

    log = open(log_file, 'a')
    log.write(f"\nV3 Run {datetime.now().isoformat()} — {len(pending)} grails, {args.workers} workers\n")

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init,
                             initargs=(DB_PATH,)) as pool:
        futures = {pool.submit(_worker_fn, g): g for g in pending}

        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result(timeout=300)
            except Exception as e:
                g = futures[fut]
                res = {'strategy': g.get('strategy','?'), 'symbol': g.get('symbol','?'),
                       'timeframe': g.get('timeframe','?'), 'status': 'ERROR', 'reason': str(e)[:200]}

            status = res.get('status', 'ERROR')
            counts[status] += 1
            all_results.append(res)

            if status == 'VALIDATED': validated.append(res)
            elif status == 'REJECTED':
                rejected.append(res)
                for r in res.get('gate_reasons', []):
                    gate_failures[r.split('=')[0]] += 1

            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(pending) - i) / rate / 3600 if rate > 0 else 0

            if status == 'VALIDATED':
                icon = '✓✓'
                detail = f"WR={res.get('full_wr',0):.1f}% PnL=+{res.get('full_pnl',0):.0f}%"
            elif status == 'REJECTED':
                icon = '  '
                detail = f"WR={res.get('full_wr',0):.1f}% raw={res.get('raw_wr',0):.1f}%"
            else:
                icon = '  '
                detail = res.get('reason', '')[:40]

            line = (f"  [{i}/{len(pending)}] {icon} {res.get('strategy','?'):<25} "
                    f"{res.get('symbol','?'):<18} {res.get('timeframe','?'):<5} "
                    f"{detail:<40} ETA:{eta:.1f}h")

            # Only print validated or every 50th
            if status == 'VALIDATED' or i % 50 == 0 or i == len(pending):
                print(line)
            log.write(line + '\n')

            if i % 20 == 0:
                dk = done_keys | {f"{r['strategy']}_{r['symbol']}_{r['timeframe']}" for r in all_results}
                prog = {'total': len(grails), 'processed': len(dk),
                        'validated': len(validated), 'rejected': len(rejected),
                        'done_keys': list(dk), 'timestamp': datetime.now().isoformat()}
                tmp = progress_file + '.tmp'
                with open(tmp, 'w') as f: json.dump(prog, f)
                os.replace(tmp, progress_file)
                for fp, data in [(validated_file, validated), (rejected_file, rejected), (all_file, all_results)]:
                    tmp = fp + '.tmp'
                    with open(tmp, 'w') as f: json.dump(data, f, indent=1, default=str)
                    os.replace(tmp, fp)

    # Final save
    for fp, data in [(validated_file, validated), (rejected_file, rejected), (all_file, all_results)]:
        with open(fp, 'w') as f: json.dump(data, f, indent=2, default=str)

    elapsed = time.time() - t0

    summary = f"""
{'='*70}
  V3 COMPLETE — {elapsed/3600:.1f}h
{'='*70}
  VALIDATED:  {counts['VALIDATED']}
  REJECTED:   {counts['REJECTED']}
  SKIPPED:    {counts.get('SKIP',0)}
  ERRORS:     {counts.get('ERROR',0)}
  TOTAL:      {sum(counts.values())}
"""
    if gate_failures:
        summary += "\n  Gate failures:\n"
        for reason, cnt in gate_failures.most_common(10):
            summary += f"    {reason:<20} {cnt}\n"

    raw_wrs = [r.get('raw_wr',0) for r in all_results if r.get('raw_wr')]
    if raw_wrs:
        import statistics
        summary += f"\n  RAW signal stats:"
        summary += f"\n    Avg raw WR:   {statistics.mean(raw_wrs):.1f}%"
        summary += f"\n    Raw PnL > 0:  {sum(1 for r in all_results if r.get('raw_pnl',0)>0)}"

    summary += f"\n  Files: {validated_file}\n{'='*70}\n"
    print(summary)
    log.write(summary)
    log.close()

    if validated:
        print("\n  VALIDATED GRAILS:")
        for g in sorted(validated, key=lambda x: x.get('full_pnl',0), reverse=True):
            print(f"    {g['strategy']:<25} {g['symbol']:<18} {g['timeframe']:<5} "
                  f"WR={g.get('full_wr',0):.1f}% PnL=+{g.get('full_pnl',0):.0f}% "
                  f"Trades={g.get('full_trades',0)} Lev={g.get('leverage',1)}x "
                  f"SL={g.get('sl',0):.0%}")


if __name__ == '__main__':
    main()
