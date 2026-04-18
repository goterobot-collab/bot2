#!/usr/bin/env python3
"""
FASE 2 — Full Historical Backtest (el que se pidió 10,000 veces)

Validates grails against ENTIRE candle history of each asset.
For each grail:
  1. Load ALL candles for (symbol, timeframe) — not just Optuna's short window
  2. Run strategy with best_params → signals
  3. Raw backtest on first 70% of CANDLES → calibrate SL/TP/Leverage
  4. Re-simulate on 100% with calibrated SL/TP + real fees
  5. Apply gate: full_wr≥65%, trades≥100, gap≤10pp, PnL>0, no smoking gun

FIXES vs previous version:
  - backtest_with_sl_tp handles BOTH LONG and SHORT (was LONG-only)
  - Calibration uses first 70% of CANDLES (was splitting TRADES — wrong)
  - GATE_MIN_TRADES = 100 (was 30)
  - Worker-level candle cache (was reloading every time)
  - Resume support with atomic progress saves
  - All 4 TFs: 5m, 15m, 1h, 4h (with resampling)

Usage:
  python3 scripts/full_historical_backtest.py                     # All tier A+B
  python3 scripts/full_historical_backtest.py --tier A             # Only tier A
  python3 scripts/full_historical_backtest.py --workers 4          # 4 workers
  python3 scripts/full_historical_backtest.py --limit 10           # Test first 10
  python3 scripts/full_historical_backtest.py --resume             # Continue previous
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
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
DB_PATH = "/Users/sabrina/CLAUDE CODE/data/activos_binance.db"
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS (same as optuna_v7.py V7.2)
# ═══════════════════════════════════════════════════════════════════════
COMMISSION = 0.001     # 0.1% per side
SLIPPAGE = 0.0005      # 0.05%
COST = COMMISSION + SLIPPAGE
SL_MIN, SL_MAX = 0.02, 0.40
TP_MIN, TP_MAX = 0.01, 0.50
MAX_DUR_CAP_H = 720

# Gate thresholds
GATE_MIN_WR = 65.0
GATE_MIN_TRADES = 100
GATE_MAX_GAP_PP = 10.0

# Calibration: first 70% of CANDLES for SL/TP, test on 100%
CALIBRATION_PCT = 0.70

# TF conversion
TF_HOURS = {'5m': 5/60, '15m': 15/60, '1h': 1, '4h': 4, '1d': 24}

# Worker-level caches (populated per process)
_CANDLE_CACHE = {}
_STRATEGIES = None


# ═══════════════════════════════════════════════════════════════════════
# CANDLE LOADING (same logic as optuna_v7.py, with caching)
# ═══════════════════════════════════════════════════════════════════════

def load_candles_from_db(symbol, tf):
    """Load candles from activos_binance.db. For 15m/4h, loads base TF."""
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
    # Resample if target TF != base TF
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
    """Load candles with per-worker caching."""
    global _CANDLE_CACHE
    key = (symbol, tf)
    if key in _CANDLE_CACHE:
        return _CANDLE_CACHE[key]
    df = load_candles_from_db(symbol, tf)
    if df is not None:
        _CANDLE_CACHE[key] = df
    return df


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY LOADING (same sources as optuna_v7.py)
# ═══════════════════════════════════════════════════════════════════════

def load_all_strategies():
    """Load all strategy gen functions from ALL sources."""
    global _STRATEGIES
    if _STRATEGIES is not None:
        return _STRATEGIES

    sys.path.insert(0, PROJECT_DIR)
    sys.path.insert(0, os.path.join(PROJECT_DIR, "strategies_code"))

    strats = {}

    # Round 2
    try:
        from strategies_round2 import STRATEGY_TYPES_R2
        strats.update(STRATEGY_TYPES_R2)
    except Exception:
        pass

    # Round 3
    try:
        from strategies_round3 import STRATEGY_TYPES_R3
        strats.update(STRATEGY_TYPES_R3)
    except Exception:
        pass

    # Batch wrappers
    try:
        from batch_search_spaces import BATCH_SEARCH_SPACES, make_space_func
        from strategies_code.batch_param_wrappers import BATCH_WRAPPERS
        for key, wrapper_fn in BATCH_WRAPPERS.items():
            if key in BATCH_SEARCH_SPACES:
                strats[key] = {'gen': wrapper_fn, 'space': make_space_func(key)}
    except Exception:
        pass

    # TV2 batches (Pine→Python)
    tv2_dir = os.path.join(PROJECT_DIR, "strategies_tv2_batches")
    if os.path.isdir(tv2_dir) and tv2_dir not in sys.path:
        sys.path.insert(0, tv2_dir)
    for batch_num in range(100, 2500):
        try:
            mod = __import__(f"strategies_tv2_batch{batch_num}")
            if hasattr(mod, 'STRATEGY_EXPORT'):
                strats.update(mod.STRATEGY_EXPORT)
        except ImportError:
            pass
        except Exception:
            pass

    # Pine winners
    try:
        pine_path = '/Users/sabrina/CLAUDE CODE/BOT V7/strategies'
        if pine_path not in sys.path:
            sys.path.insert(0, pine_path)
        from strategies_new_winners import NEW_WINNER_STRATS
        strats.update(NEW_WINNER_STRATS)
    except Exception:
        pass

    # strategy_factory (if exists)
    try:
        from strategy_factory import FACTORY_STRATS
        for name, entry in FACTORY_STRATS.items():
            if name not in strats:
                strats[name] = entry
    except Exception:
        pass

    _STRATEGIES = strats
    return strats


def get_gen(strategy_name, strats):
    """Get the signal generator callable for a strategy."""
    entry = strats.get(strategy_name)
    if entry is None:
        return None
    if callable(entry):
        return entry
    if isinstance(entry, dict) and 'gen' in entry:
        return entry['gen']
    return None


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE — LONG + SHORT (fixes optuna_v7 SHORT bug)
# ═══════════════════════════════════════════════════════════════════════

def backtest_raw(df, signals):
    """Raw backtest WITHOUT SL/TP. Handles LONG + SHORT.
    Identical to optuna_v7.py:275-331."""
    if signals is None:
        return []
    sig = signals.values
    opens, highs, lows = df['open'].values, df['high'].values, df['low'].values
    n = min(len(sig), len(opens))
    trades = []
    pos = 0; ep = 0.0; ei = 0; mae = 0.0; mfe = 0.0

    for i in range(1, n):
        s = int(sig[i - 1]); p = opens[i]
        if pos == 0:
            if s == 1:
                ep = p * (1 + COST); ei = i; mae = 0.; mfe = 0.; pos = 1
            elif s == -1:
                ep = p * (1 - COST); ei = i; mae = 0.; mfe = 0.; pos = -1
        elif pos == 1:
            lp = (lows[i] - ep) / ep
            hp = (highs[i] - ep) / ep
            if lp < mae: mae = lp
            if hp > mfe: mfe = hp
            if s == -1 or s == 0:
                xp = p * (1 - COST)
                pnl = (xp - ep) / ep
                yr = df.index[ei].year if hasattr(df.index[ei], 'year') else 2025
                trades.append({'pnl': pnl, 'mae': mae, 'mfe': mfe,
                              'dur_bars': i - ei, 'year': yr, 'win': pnl > 0})
                pos = 0
                if s == -1:
                    ep = p * (1 - COST); ei = i; mae = 0.; mfe = 0.; pos = -1
        elif pos == -1:
            hp_s = (ep - highs[i]) / ep
            lp_s = (ep - lows[i]) / ep
            if hp_s < mae: mae = hp_s
            if lp_s > mfe: mfe = lp_s
            if s == 1 or s == 0:
                xp = p * (1 + COST)
                pnl = (ep - xp) / ep
                yr = df.index[ei].year if hasattr(df.index[ei], 'year') else 2025
                trades.append({'pnl': pnl, 'mae': mae, 'mfe': mfe,
                              'dur_bars': i - ei, 'year': yr, 'win': pnl > 0})
                pos = 0
                if s == 1:
                    ep = p * (1 + COST); ei = i; mae = 0.; mfe = 0.; pos = 1
    return trades


def backtest_with_sl_tp(df, signals, sl_pct, tp_pct, max_dur_bars=None):
    """Backtest WITH SL/TP. Handles BOTH LONG and SHORT.

    FIXED: optuna_v7.py:431-479 only handles LONGs.
    This version handles both directions correctly.
    Uses ATR-based dynamic slippage (same as production).
    """
    if signals is None:
        return []
    sig = signals.values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)

    # ATR-based dynamic slippage (Kissell 2013, same as optuna_v7)
    prev_c = np.empty(n); prev_c[0] = closes[0]; prev_c[1:] = closes[:-1]
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    kernel = np.ones(14) / 14
    atr_full = np.convolve(tr, kernel, mode='full')[:n]
    atr_slip = np.clip(
        (atr_full / np.where(closes > 0, closes, 1)) * 0.1, 0.0005, 0.01)

    trades = []
    pos = 0; ep = 0.0; ei = 0

    for i in range(1, n):
        s = int(sig[i - 1]); p = opens[i]; slip = atr_slip[i]

        # ── OPEN POSITION ──
        if pos == 0:
            if s == 1:
                ep = p * (1 + slip + COMMISSION); ei = i; pos = 1
            elif s == -1:
                ep = p * (1 - slip - COMMISSION); ei = i; pos = -1
            continue

        # ── MANAGE LONG ──
        if pos == 1:
            # SL: price drops by sl_pct
            if (lows[i] - ep) / ep <= -sl_pct:
                trades.append({'pnl': -sl_pct, 'dur_bars': i - ei, 'exit': 'SL'})
                pos = 0; continue
            # TP: price rises by tp_pct
            if (highs[i] - ep) / ep >= tp_pct:
                trades.append({'pnl': tp_pct, 'dur_bars': i - ei, 'exit': 'TP'})
                pos = 0; continue
            # Max duration
            if max_dur_bars and (i - ei) >= max_dur_bars:
                xp = p * (1 - slip - COMMISSION)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei, 'exit': 'DUR'})
                pos = 0; continue
            # Signal exit or flip
            if s == -1 or s == 0:
                xp = p * (1 - slip - COMMISSION)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei, 'exit': 'SIG'})
                pos = 0
                if s == -1:  # Flip to SHORT
                    ep = p * (1 - slip - COMMISSION); ei = i; pos = -1

        # ── MANAGE SHORT ──
        elif pos == -1:
            # SL: price rises by sl_pct above entry
            if (highs[i] - ep) / ep >= sl_pct:
                trades.append({'pnl': -sl_pct, 'dur_bars': i - ei, 'exit': 'SL'})
                pos = 0; continue
            # TP: price drops by tp_pct below entry
            if (lows[i] - ep) / ep <= -tp_pct:
                trades.append({'pnl': tp_pct, 'dur_bars': i - ei, 'exit': 'TP'})
                pos = 0; continue
            # Max duration
            if max_dur_bars and (i - ei) >= max_dur_bars:
                xp = p * (1 + slip + COMMISSION)
                trades.append({'pnl': (ep - xp) / ep, 'dur_bars': i - ei, 'exit': 'DUR'})
                pos = 0; continue
            # Signal exit or flip
            if s == 1 or s == 0:
                xp = p * (1 + slip + COMMISSION)
                trades.append({'pnl': (ep - xp) / ep, 'dur_bars': i - ei, 'exit': 'SIG'})
                pos = 0
                if s == 1:  # Flip to LONG
                    ep = p * (1 + slip + COMMISSION); ei = i; pos = 1

    return trades


# ═══════════════════════════════════════════════════════════════════════
# SL/TP/LEVERAGE CALCULATION (V7.2 formulas)
# ═══════════════════════════════════════════════════════════════════════

def calc_sl_tp_leverage(raw_trades, tf='5m'):
    """V7.2 empirical formulas from calibration trades:
      SL = P90(ALL trades MAE) x 2.0     [2%-40%]
      TP = P90(winners MFE) x 0.85       [1%-50%]
      Lev = min(1/(MAE_P99x2.5), 1/(MAE_P95x2.0))  [1x-20x]
    """
    if len(raw_trades) < 5:
        return None

    all_maes = [abs(t['mae']) for t in raw_trades]
    sl = float(np.clip(np.percentile(all_maes, 90) * 2.0, SL_MIN, SL_MAX))

    winners = [t for t in raw_trades if t['win']]
    if len(winners) < 3:
        return None
    mfes = [t['mfe'] for t in winners]
    tp = float(np.clip(np.percentile(mfes, 90) * 0.85, TP_MIN, TP_MAX))

    # Max duration — FLOOR of 24h (production winners use 48h, not Optuna's 1.8h)
    # Cerebro finding 2026-04-10: 84.4% of trades exit by DUR timeout with Optuna's
    # ultra-low max_dur. Production winners need 24-48h for edge to materialize.
    durs = [t['dur_bars'] for t in winners]
    bars_per_h = 1.0 / TF_HOURS.get(tf, 1)
    raw_dur = np.percentile(durs, 95) * 1.5
    dur_h = max(24.0, min(raw_dur / bars_per_h, MAX_DUR_CAP_H))  # FLOOR 24h
    max_dur_bars = int(dur_h * bars_per_h)

    # Leverage
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
        'mae_p95': round(mae_p95, 4), 'mae_p99': round(mae_p99, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# CORE: Process one grail through entire pipeline
# ═══════════════════════════════════════════════════════════════════════

def process_grail(grail, strategies):
    """Full backtest pipeline for a single grail."""
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

    # 1. Find strategy gen
    gen = get_gen(sname, strategies)
    if gen is None:
        res['status'] = 'SKIP'; res['reason'] = f'not_found:{sname}'
        return res

    # 2. Load ALL candles
    df = get_candles(symbol, tf)
    if df is None or len(df) < 200:
        res['status'] = 'SKIP'
        res['reason'] = f'candles:{len(df) if df is not None else 0}<200'
        return res
    res['candles'] = len(df)
    res['days'] = round((df.index[-1] - df.index[0]).total_seconds() / 86400, 1)

    # 3. Generate signals on ALL candles
    try:
        signals = gen(df, **bp)
    except Exception as e:
        res['status'] = 'ERROR'; res['reason'] = f'gen:{str(e)[:150]}'
        return res
    if signals is None or len(signals) == 0:
        res['status'] = 'SKIP'; res['reason'] = 'no_signals'
        return res

    # 4. Split: first 70% of CANDLES for calibration
    split_idx = int(len(df) * CALIBRATION_PCT)
    df_cal = df.iloc[:split_idx]
    sig_cal = signals.iloc[:split_idx]

    # 5. Raw backtest on calibration portion → MAE/MFE for SL/TP
    cal_trades = backtest_raw(df_cal, sig_cal)
    if len(cal_trades) < 5:
        res['status'] = 'SKIP'
        res['reason'] = f'calib_trades:{len(cal_trades)}<5'
        return res
    res['calib_trades'] = len(cal_trades)

    # 6. Calculate SL/TP/Leverage from calibration
    params = calc_sl_tp_leverage(cal_trades, tf)
    if params is None:
        res['status'] = 'SKIP'; res['reason'] = 'sl_tp_failed'
        return res
    res.update({
        'sl': params['sl'], 'tp': params['tp'],
        'leverage': params['leverage'], 'rr': params['rr'],
        'max_dur_h': params['max_dur_h'],
    })

    # 7. Re-simulate on 100% with calibrated SL/TP + fees
    full_trades = backtest_with_sl_tp(
        df, signals, params['sl'], params['tp'], params['max_dur_bars'])
    if len(full_trades) < 5:
        res['status'] = 'SKIP'
        res['reason'] = f'full_trades:{len(full_trades)}<5'
        return res

    # 8. Compute metrics
    pnls = np.array([t['pnl'] for t in full_trades])
    n = len(pnls)
    wins = int((pnls > 0).sum())
    full_wr = round(wins / n * 100, 1)
    full_pnl = round(float(pnls.sum() * 100), 2)
    cum = pnls.cumsum()
    peak = np.maximum.accumulate(cum)
    max_dd = round(float((peak - cum).max() * 100), 2)
    sharpe = round(float((pnls.mean() / (pnls.std() + 1e-10)) * np.sqrt(252)), 2)
    exits = dict(Counter(t.get('exit', '?') for t in full_trades))
    gap = round(optuna_wr - full_wr, 1)

    res.update({
        'full_wr': full_wr, 'full_trades': n, 'full_pnl': full_pnl,
        'full_max_dd': max_dd, 'full_sharpe': sharpe,
        'gap': gap, 'exits': exits,
        'lev_pnl': round(float(pnls.sum() * params['leverage'] * 100), 2),
    })

    # 9. GATE
    reasons = []
    passed = True

    if full_wr < GATE_MIN_WR:
        passed = False; reasons.append(f'wr={full_wr}<{GATE_MIN_WR}')
    if n < GATE_MIN_TRADES:
        passed = False; reasons.append(f'trades={n}<{GATE_MIN_TRADES}')
    if gap > GATE_MAX_GAP_PP:
        passed = False; reasons.append(f'gap={gap}pp>{GATE_MAX_GAP_PP}')
    if full_pnl <= 0:
        passed = False; reasons.append(f'pnl={full_pnl}<=0')

    smoking = grail.get('smoking_gun', False)
    if smoking:
        passed = False; reasons.append('smoking_gun')

    lev_sl = params['leverage'] * params['sl']
    if lev_sl > 0.80:
        passed = False; reasons.append(f'lev_sl={lev_sl:.2f}>0.80')

    res['status'] = 'VALIDATED' if passed else 'REJECTED'
    res['gate_reasons'] = reasons if reasons else ['all_passed']
    if passed:
        res['best_params'] = bp

    return res


# ═══════════════════════════════════════════════════════════════════════
# WORKER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def _worker_init():
    global _STRATEGIES, _CANDLE_CACHE
    _CANDLE_CACHE = {}
    _STRATEGIES = load_all_strategies()


def _worker_fn(grail):
    global _STRATEGIES
    if _STRATEGIES is None:
        _STRATEGIES = load_all_strategies()
    try:
        return process_grail(grail, _STRATEGIES)
    except Exception as e:
        return {
            'strategy': grail.get('strategy', '?'),
            'symbol': grail.get('symbol', '?'),
            'timeframe': grail.get('timeframe', '?'),
            'status': 'ERROR', 'reason': traceback.format_exc()[-300:],
        }


# ═══════════════════════════════════════════════════════════════════════
# PROGRESS / IO
# ═══════════════════════════════════════════════════════════════════════

def atomic_save(filepath, data):
    tmp = filepath + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, filepath)


def grail_key(g):
    return f"{g['strategy']}|{g['symbol']}|{g['timeframe']}"


def save_checkpoint(out_dir, all_results, done_keys, t0):
    """Save all output files atomically."""
    validated = [r for r in all_results if r.get('status') == 'VALIDATED']
    rejected = [r for r in all_results if r.get('status') == 'REJECTED']
    other = [r for r in all_results if r.get('status') not in ('VALIDATED', 'REJECTED')]

    atomic_save(os.path.join(out_dir, 'grails_validated.json'), validated)
    atomic_save(os.path.join(out_dir, 'grails_rejected.json'), rejected)
    atomic_save(os.path.join(out_dir, 'full_backtest_progress.json'), {
        'timestamp': datetime.now().isoformat(),
        'elapsed_h': round((time.time() - t0) / 3600, 2),
        'total': len(all_results),
        'validated': len(validated),
        'rejected': len(rejected),
        'skipped': len([r for r in other if r.get('status') == 'SKIP']),
        'errors': len([r for r in other if r.get('status') == 'ERROR']),
        'done_keys': list(done_keys),
    })


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='FASE 2: Full Historical Backtest')
    parser.add_argument('--tier', nargs='+', default=['A', 'B'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--out-dir', default=DATA_DIR)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  FASE 2 — FULL HISTORICAL BACKTEST")
    print(f"  Calibrate on 70% CANDLES, test on 100% — all history")
    print(f"  LONG + SHORT support (optuna_v7 SHORT bug FIXED)")
    print(f"{'='*70}")
    print(f"  DB:      {DB_PATH}")
    print(f"  Workers: {args.workers}")
    print(f"  Tiers:   {args.tier}")
    print(f"  Gates:   WR>={GATE_MIN_WR}% | trades>={GATE_MIN_TRADES} | "
          f"gap<={GATE_MAX_GAP_PP}pp | PnL>0 | no smoke")
    print()

    # Load grails
    grails = []
    for tier in args.tier:
        path = os.path.join(args.out_dir, f'grails_tier_{tier}.json')
        if os.path.exists(path):
            with open(path) as f:
                tg = json.load(f)
            grails.extend(tg)
            print(f"  Tier {tier}: {len(tg)} grails")
        else:
            print(f"  Tier {tier}: NOT FOUND — {path}")

    if not grails:
        print("  No grails to process. Run classify_grails_by_coverage.py first.")
        return

    if args.limit > 0:
        grails = grails[:args.limit]

    # Resume
    done_keys = set()
    prev_results = []
    progress_path = os.path.join(args.out_dir, 'full_backtest_progress.json')
    if args.resume and os.path.exists(progress_path):
        with open(progress_path) as f:
            prev = json.load(f)
        done_keys = set(prev.get('done_keys', []))
        for fn in ['grails_validated.json', 'grails_rejected.json']:
            fp = os.path.join(args.out_dir, fn)
            if os.path.exists(fp):
                with open(fp) as f:
                    prev_results.extend(json.load(f))
        print(f"  Resuming: {len(done_keys)} done")

    pending = [g for g in grails if grail_key(g) not in done_keys]
    print(f"\n  Total: {len(grails)} | Done: {len(done_keys)} | Pending: {len(pending)}")

    if not pending:
        print("  Nothing pending. Done.")
        return

    est_h = len(pending) * 30 / max(args.workers, 1) / 3600
    print(f"  Estimated: ~{est_h:.1f}h ({len(pending)} x ~30s / {args.workers} workers)")
    print(f"{'='*70}\n")

    all_results = list(prev_results)
    t0 = time.time()

    def on_result(result, idx, total):
        """Handle a completed result."""
        all_results.append(result)
        done_keys.add(grail_key(result))
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta_h = (total - idx - 1) / rate / 3600 if rate > 0 else 0
        icon = {'VALIDATED': '✓', 'REJECTED': '✗', 'SKIP': '○', 'ERROR': '!'}.get(
            result['status'], '?')
        wr = f"WR={result.get('full_wr', 0):.1f}%" if 'full_wr' in result else (
            result.get('reason', '')[:25])
        print(f"  [{idx+1}/{total}] {icon} {result['strategy']:<22} "
              f"{result['symbol']:<18} {result['timeframe']:<4} "
              f"{wr:<20} ETA:{eta_h:.1f}h")
        if (idx + 1) % 10 == 0:
            save_checkpoint(args.out_dir, all_results, done_keys, t0)

    if args.workers <= 1:
        print("Loading strategies (single-threaded)...")
        strategies = load_all_strategies()
        print(f"  {len(strategies)} strategies loaded\n")
        for idx, g in enumerate(pending):
            result = process_grail(g, strategies)
            on_result(result, idx, len(pending))
    else:
        print(f"Launching {args.workers} workers...\n")
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init) as pool:
            fmap = {pool.submit(_worker_fn, g): g for g in pending}
            for idx, fut in enumerate(as_completed(fmap)):
                try:
                    result = fut.result(timeout=600)
                except Exception as e:
                    g = fmap[fut]
                    result = {'strategy': g.get('strategy', '?'),
                              'symbol': g.get('symbol', '?'),
                              'timeframe': g.get('timeframe', '?'),
                              'status': 'ERROR', 'reason': str(e)[:300]}
                on_result(result, idx, len(pending))

    # Final save
    save_checkpoint(args.out_dir, all_results, done_keys, t0)

    # ── REPORT ──
    elapsed = time.time() - t0
    validated = [r for r in all_results if r.get('status') == 'VALIDATED']
    rejected = [r for r in all_results if r.get('status') == 'REJECTED']
    skipped = [r for r in all_results if r.get('status') == 'SKIP']
    errors = [r for r in all_results if r.get('status') == 'ERROR']

    print(f"\n{'='*70}")
    print(f"  FASE 2 COMPLETE — {elapsed/3600:.1f}h")
    print(f"{'='*70}")
    print(f"  VALIDATED:  {len(validated):>5}")
    print(f"  REJECTED:   {len(rejected):>5}")
    print(f"  SKIPPED:    {len(skipped):>5}")
    print(f"  ERRORS:     {len(errors):>5}")
    print(f"  TOTAL:      {len(all_results):>5}")

    if validated:
        wrs = [r['full_wr'] for r in validated]
        pnls = [r['full_pnl'] for r in validated]
        print(f"\n  Validated stats:")
        print(f"    WR:  avg={np.mean(wrs):.1f}% min={min(wrs):.1f}% max={max(wrs):.1f}%")
        print(f"    PnL: avg={np.mean(pnls):.1f}% total={sum(pnls):.1f}%")
        sc = Counter(r['strategy'] for r in validated)
        print(f"\n  Top validated strategies:")
        for s, c in sc.most_common(10):
            print(f"    {s:<30} {c:>3}")
        print(f"\n  TF: {dict(Counter(r['timeframe'] for r in validated))}")

    if rejected:
        gf = Counter()
        for r in rejected:
            for reason in r.get('gate_reasons', ['?']):
                gf[reason.split('=')[0]] += 1
        print(f"\n  Gate failures:")
        for g, c in gf.most_common():
            print(f"    {g:<15} {c:>5} ({c/len(rejected)*100:.0f}%)")

    print(f"\n  Output files:")
    print(f"    {os.path.join(args.out_dir, 'grails_validated.json')}")
    print(f"    {os.path.join(args.out_dir, 'grails_rejected.json')}")
    print(f"    {progress_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
