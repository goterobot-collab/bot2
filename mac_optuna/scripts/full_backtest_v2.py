#!/usr/bin/env python3
"""
FASE 2b — Full Historical Backtest V2 with SL/TP Grid Search

V1 found 0/855 validated because V7.2 formula produces R:R=0.20-0.25
(SL 14-31%, TP 3-7%). Need WR>82% to break even with that R:R.

V2 fixes this by:
  1. Grid search: try ~40 SL/TP combos per grail on calibration window
  2. R:R floor: skip combos where TP/SL < 0.4
  3. Pick best EV combo, validate on full history
  4. Also reports RAW signal performance (no SL/TP) for edge detection
  5. Relaxed gate: WR>=55% + PnL>0 + trades>=100

Usage:
  python3 scripts/full_backtest_v2.py                        # All tier A+B
  python3 scripts/full_backtest_v2.py --tier C               # Tier C
  python3 scripts/full_backtest_v2.py --workers 10            # 10 workers
  python3 scripts/full_backtest_v2.py --limit 20              # Test first 20
  python3 scripts/full_backtest_v2.py --resume                # Continue previous
  python3 scripts/full_backtest_v2.py --input data/custom.json  # Custom input
  python3 scripts/full_backtest_v2.py --db /path/to/db        # Custom DB (Hetzner)
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
# PATHS — configurable for Mac / Hetzner
# ═══════════════════════════════════════════════════════════════════════
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Auto-detect DB path: Hetzner vs Mac
_HETZNER_DB = "/home/ubuntu/candles_db/activos_binance.db"
_MAC_DB = "/Users/sabrina/CLAUDE CODE/data/activos_binance.db"
DEFAULT_DB = _HETZNER_DB if os.path.exists(_HETZNER_DB) else _MAC_DB

# Will be set by CLI args or environment variable
DB_PATH = os.environ.get('BACKTEST_DB_PATH', DEFAULT_DB)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
COMMISSION = 0.001     # 0.1% per side
SLIPPAGE = 0.0005      # 0.05%
COST = COMMISSION + SLIPPAGE
SL_MIN, SL_MAX = 0.02, 0.40
TP_MIN, TP_MAX = 0.01, 0.50
MAX_DUR_CAP_H = 720
RR_FLOOR = 0.4        # Minimum R:R ratio to consider
MIN_GRID_TRADES = 30   # Min trades for a SL/TP combo to be valid

# Gate thresholds (RELAXED from V1)
GATE_MIN_WR = 55.0     # Was 65.0 — too strict with good R:R
GATE_MIN_TRADES = 100
GATE_MAX_GAP_PP = 15.0  # Was 10.0 — allow more if PnL is positive

# Calibration split
CALIBRATION_PCT = 0.70

# TF conversion
TF_HOURS = {'5m': 5/60, '15m': 15/60, '1h': 1, '4h': 4, '1d': 24}

# Worker-level caches
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

    # TV2 batches
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

    # Pine winners (Mac or Hetzner path)
    for pine_path in ['/Users/sabrina/CLAUDE CODE/BOT V7/strategies',
                      '/home/ubuntu/bot-v7/strategies']:
        try:
            if os.path.isdir(pine_path) and pine_path not in sys.path:
                sys.path.insert(0, pine_path)
            from strategies_new_winners import NEW_WINNER_STRATS
            strats.update(NEW_WINNER_STRATS)
            break
        except Exception:
            pass

    # strategy_factory
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
    entry = strats.get(strategy_name)
    if entry is None:
        return None
    if callable(entry):
        return entry
    if isinstance(entry, dict) and 'gen' in entry:
        return entry['gen']
    return None


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST ENGINES
# ═══════════════════════════════════════════════════════════════════════

def backtest_raw(df, signals):
    """Raw backtest WITHOUT SL/TP. LONG + SHORT."""
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
                trades.append({'pnl': pnl, 'mae': mae, 'mfe': mfe,
                              'dur_bars': i - ei, 'win': pnl > 0})
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
                trades.append({'pnl': pnl, 'mae': mae, 'mfe': mfe,
                              'dur_bars': i - ei, 'win': pnl > 0})
                pos = 0
                if s == 1:
                    ep = p * (1 + COST); ei = i; mae = 0.; mfe = 0.; pos = 1
    return trades


def backtest_with_sl_tp(df, signals, sl_pct, tp_pct, max_dur_bars=None):
    """Backtest WITH SL/TP. LONG + SHORT."""
    if signals is None:
        return []
    sig = signals.values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)

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

        if pos == 0:
            if s == 1:
                ep = p * (1 + slip + COMMISSION); ei = i; pos = 1
            elif s == -1:
                ep = p * (1 - slip - COMMISSION); ei = i; pos = -1
            continue

        if pos == 1:
            if (lows[i] - ep) / ep <= -sl_pct:
                trades.append({'pnl': -sl_pct, 'dur_bars': i - ei, 'exit': 'SL'})
                pos = 0; continue
            if (highs[i] - ep) / ep >= tp_pct:
                trades.append({'pnl': tp_pct, 'dur_bars': i - ei, 'exit': 'TP'})
                pos = 0; continue
            if max_dur_bars and (i - ei) >= max_dur_bars:
                xp = p * (1 - slip - COMMISSION)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei, 'exit': 'DUR'})
                pos = 0; continue
            if s == -1 or s == 0:
                xp = p * (1 - slip - COMMISSION)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei, 'exit': 'SIG'})
                pos = 0
                if s == -1:
                    ep = p * (1 - slip - COMMISSION); ei = i; pos = -1

        elif pos == -1:
            if (highs[i] - ep) / ep >= sl_pct:
                trades.append({'pnl': -sl_pct, 'dur_bars': i - ei, 'exit': 'SL'})
                pos = 0; continue
            if (lows[i] - ep) / ep <= -tp_pct:
                trades.append({'pnl': tp_pct, 'dur_bars': i - ei, 'exit': 'TP'})
                pos = 0; continue
            if max_dur_bars and (i - ei) >= max_dur_bars:
                xp = p * (1 + slip + COMMISSION)
                trades.append({'pnl': (ep - xp) / ep, 'dur_bars': i - ei, 'exit': 'DUR'})
                pos = 0; continue
            if s == 1 or s == 0:
                xp = p * (1 + slip + COMMISSION)
                trades.append({'pnl': (ep - xp) / ep, 'dur_bars': i - ei, 'exit': 'SIG'})
                pos = 0
                if s == 1:
                    ep = p * (1 + slip + COMMISSION); ei = i; pos = 1

    return trades


# ═══════════════════════════════════════════════════════════════════════
# SL/TP GRID SEARCH — The V2 innovation
# ═══════════════════════════════════════════════════════════════════════

def grid_search_sl_tp(cal_trades, df_cal, sig_cal, tf):
    """Try ~40 SL/TP combos on calibration data, return best by EV.

    Grid:
      SL percentiles: P50, P75, P90, P95 of |MAE| × {1.0, 1.5, 2.0}
      TP percentiles: P50, P75, P90, P95 of winners MFE × {0.7, 0.85, 1.0, 1.2}
      R:R floor: skip if TP/SL < 0.4

    Returns dict with best params + all results for analysis.
    """
    all_maes = [abs(t['mae']) for t in cal_trades]
    winners = [t for t in cal_trades if t['win']]
    if len(winners) < 3:
        return None

    mfes = [t['mfe'] for t in winners]

    # Build SL candidates
    sl_vals = set()
    for p in [50, 75, 90, 95]:
        base = float(np.percentile(all_maes, p))
        for mult in [1.0, 1.5, 2.0]:
            sl_vals.add(round(float(np.clip(base * mult, SL_MIN, SL_MAX)), 4))
    sl_list = sorted(sl_vals)

    # Build TP candidates
    tp_vals = set()
    for p in [50, 75, 90, 95]:
        base = float(np.percentile(mfes, p))
        for mult in [0.7, 0.85, 1.0, 1.2]:
            tp_vals.add(round(float(np.clip(base * mult, TP_MIN, TP_MAX)), 4))
    tp_list = sorted(tp_vals)

    # Duration (fixed, not part of grid)
    durs = [t['dur_bars'] for t in winners]
    bars_per_h = 1.0 / TF_HOURS.get(tf, 1)
    raw_dur = float(np.percentile(durs, 95)) * 1.5
    dur_h = min(raw_dur / bars_per_h, MAX_DUR_CAP_H)
    max_dur_bars = int(dur_h * bars_per_h)

    best = None
    best_ev = -999.0
    tested = 0

    for sl in sl_list:
        for tp in tp_list:
            rr = tp / (sl + 1e-10)
            if rr < RR_FLOOR:
                continue

            trades = backtest_with_sl_tp(df_cal, sig_cal, sl, tp, max_dur_bars)
            if len(trades) < MIN_GRID_TRADES:
                continue
            tested += 1

            pnls = np.array([t['pnl'] for t in trades])
            n = len(pnls)
            wr = float((pnls > 0).mean())
            ev = float(pnls.mean())
            total_pnl = float(pnls.sum())

            # Score = EV (avg PnL per trade)
            if ev > best_ev:
                best_ev = ev
                exits = dict(Counter(t.get('exit', '?') for t in trades))
                best = {
                    'sl': sl, 'tp': tp, 'rr': round(rr, 3),
                    'max_dur_bars': max_dur_bars, 'max_dur_h': round(dur_h, 1),
                    'cal_wr': round(wr * 100, 1),
                    'cal_ev': round(ev * 100, 4),
                    'cal_pnl': round(total_pnl * 100, 2),
                    'cal_trades': n,
                    'cal_exits': exits,
                    'combos_tested': tested,
                }

    if best is not None:
        # Calculate leverage for the best combo
        mae_p95 = float(np.percentile(all_maes, 95))
        mae_p99 = float(np.percentile(all_maes, 99)) if len(all_maes) >= 10 else max(all_maes)
        lev = max(1, min(20, int(min(
            1 / (mae_p99 * 2.5 + 1e-10),
            1 / (mae_p95 * 2.0 + 1e-10)
        ))))
        while lev > 1 and best['sl'] * lev > 0.80:
            lev -= 1
        best['leverage'] = lev
        best['combos_tested'] = tested

    return best


# ═══════════════════════════════════════════════════════════════════════
# CORE: Process one grail — V2 with grid search
# ═══════════════════════════════════════════════════════════════════════

def process_grail_v2(grail, strategies):
    """Full backtest V2: grid search SL/TP + raw signal analysis."""
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

    # 1. Find strategy
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

    # 4. RAW signal analysis (no SL/TP — pure signal quality)
    raw_trades = backtest_raw(df, signals)
    if len(raw_trades) >= 10:
        raw_pnls = np.array([t['pnl'] for t in raw_trades])
        res['raw_wr'] = round(float((raw_pnls > 0).mean() * 100), 1)
        res['raw_pnl'] = round(float(raw_pnls.sum() * 100), 2)
        res['raw_trades'] = len(raw_trades)
        res['raw_ev'] = round(float(raw_pnls.mean() * 100), 4)
    else:
        res['raw_wr'] = 0; res['raw_pnl'] = 0; res['raw_trades'] = len(raw_trades)
        res['raw_ev'] = 0

    # 5. Split calibration window
    split_idx = int(len(df) * CALIBRATION_PCT)
    df_cal = df.iloc[:split_idx]
    sig_cal = signals.iloc[:split_idx]

    # 6. Raw backtest on calibration → MAE/MFE for grid
    cal_raw = backtest_raw(df_cal, sig_cal)
    if len(cal_raw) < 10:
        res['status'] = 'SKIP'
        res['reason'] = f'calib_trades:{len(cal_raw)}<10'
        return res
    res['calib_raw_trades'] = len(cal_raw)

    # 7. GRID SEARCH — find best SL/TP combo
    best_params = grid_search_sl_tp(cal_raw, df_cal, sig_cal, tf)
    if best_params is None:
        res['status'] = 'SKIP'; res['reason'] = 'grid_no_valid_combo'
        return res

    res.update({
        'sl': best_params['sl'], 'tp': best_params['tp'],
        'rr': best_params['rr'],
        'leverage': best_params['leverage'],
        'max_dur_h': best_params['max_dur_h'],
        'cal_wr': best_params['cal_wr'],
        'cal_ev': best_params['cal_ev'],
        'combos_tested': best_params['combos_tested'],
    })

    # 8. VALIDATE on 100% with best SL/TP
    full_trades = backtest_with_sl_tp(
        df, signals, best_params['sl'], best_params['tp'],
        best_params['max_dur_bars'])
    if len(full_trades) < 10:
        res['status'] = 'SKIP'
        res['reason'] = f'full_trades:{len(full_trades)}<10'
        return res

    # 9. Full metrics
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
    ev = round(float(pnls.mean() * 100), 4)
    lev_pnl = round(float(pnls.sum() * best_params['leverage'] * 100), 2)

    res.update({
        'full_wr': full_wr, 'full_trades': n, 'full_pnl': full_pnl,
        'full_max_dd': max_dd, 'full_sharpe': sharpe, 'full_ev': ev,
        'gap': gap, 'exits': exits, 'lev_pnl': lev_pnl,
    })

    # 10. GATE (relaxed from V1)
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

    lev_sl = best_params['leverage'] * best_params['sl']
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
        return process_grail_v2(grail, strats)
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

    parser = argparse.ArgumentParser(description='FASE 2b: Full Backtest V2 with SL/TP Grid Search')
    parser.add_argument('--tier', default='AB', help='Tier(s) to process: A, B, AB, C, ALL')
    parser.add_argument('--input', help='Direct path to grails JSON (overrides --tier)')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    parser.add_argument('--limit', type=int, default=0, help='Limit grails (0=all)')
    parser.add_argument('--resume', action='store_true', help='Resume from progress file')
    parser.add_argument('--db', default=DEFAULT_DB, help='Path to activos_binance.db')
    parser.add_argument('--output-prefix', default='v2', help='Prefix for output files')
    args = parser.parse_args()

    DB_PATH = args.db

    print(f"{'='*70}")
    print(f"  FASE 2b — Full Historical Backtest V2 (SL/TP Grid Search)")
    print(f"{'='*70}")
    print(f"  DB:        {DB_PATH}")
    print(f"  Workers:   {args.workers}")
    print(f"  R:R floor: {RR_FLOOR}")
    print(f"  Gate:      WR>={GATE_MIN_WR}%, trades>={GATE_MIN_TRADES}, "
          f"gap<={GATE_MAX_GAP_PP}pp, PnL>0")

    # Load grails
    if args.input:
        with open(args.input) as f:
            grails = json.load(f)
        print(f"  Input:     {args.input} ({len(grails):,} grails)")
    else:
        grails = []
        tier_str = args.tier.upper()
        tiers_to_load = []
        if 'A' in tier_str:
            tiers_to_load.append('A')
        if 'B' in tier_str:
            tiers_to_load.append('B')
        if 'C' in tier_str:
            tiers_to_load.append('C')
        if tier_str == 'ALL':
            tiers_to_load = ['A', 'B', 'C']

        for t in tiers_to_load:
            path = os.path.join(DATA_DIR, f"grails_tier_{t}.json")
            if os.path.exists(path):
                with open(path) as f:
                    batch = json.load(f)
                grails.extend(batch)
                print(f"  Tier {t}:    {len(batch):,} grails from {path}")
        print(f"  Total:     {len(grails):,} grails")

    if not grails:
        print("ERROR: No grails to process!")
        return

    # Resume support
    progress_file = os.path.join(DATA_DIR, f"full_backtest_{args.output_prefix}_progress.json")
    done_keys = set()
    if args.resume and os.path.exists(progress_file):
        with open(progress_file) as f:
            prev = json.load(f)
        done_keys = set(prev.get('done_keys', []))
        print(f"  Resume:    {len(done_keys)} already done")

    # Filter
    pending = []
    for g in grails:
        key = f"{g['strategy']}_{g['symbol']}_{g['timeframe']}"
        if key not in done_keys:
            pending.append(g)
    if args.limit > 0:
        pending = pending[:args.limit]
    print(f"  Pending:   {len(pending):,} grails")
    print(f"{'='*70}\n")

    if not pending:
        print("Nothing to process!")
        return

    # Load existing results for resume
    validated_file = os.path.join(DATA_DIR, f"grails_{args.output_prefix}_validated.json")
    rejected_file = os.path.join(DATA_DIR, f"grails_{args.output_prefix}_rejected.json")
    all_results_file = os.path.join(DATA_DIR, f"grails_{args.output_prefix}_all_results.json")
    log_file = os.path.join(DATA_DIR, f"full_backtest_{args.output_prefix}_log.txt")

    validated = []
    rejected = []
    all_results = []
    if args.resume:
        for fpath, lst in [(validated_file, validated), (rejected_file, rejected),
                           (all_results_file, all_results)]:
            if os.path.exists(fpath):
                with open(fpath) as f:
                    lst.extend(json.load(f))

    counts = Counter({'VALIDATED': len(validated), 'REJECTED': len(rejected),
                      'SKIP': 0, 'ERROR': 0})
    t0 = time.time()
    gate_failures = Counter()

    log = open(log_file, 'a')
    log.write(f"\n{'='*70}\n")
    log.write(f"  V2 Run started {datetime.now().isoformat()}\n")
    log.write(f"  {len(pending)} grails, {args.workers} workers\n")
    log.write(f"{'='*70}\n")

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init,
                             initargs=(DB_PATH,)) as pool:
        futures = {pool.submit(_worker_fn, g): g for g in pending}

        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result(timeout=300)
            except Exception as e:
                g = futures[fut]
                res = {
                    'strategy': g.get('strategy', '?'),
                    'symbol': g.get('symbol', '?'),
                    'timeframe': g.get('timeframe', '?'),
                    'status': 'ERROR', 'reason': str(e)[:200],
                }

            status = res.get('status', 'ERROR')
            counts[status] += 1
            all_results.append(res)

            if status == 'VALIDATED':
                validated.append(res)
            elif status == 'REJECTED':
                rejected.append(res)
                for r in res.get('gate_reasons', []):
                    gate_failures[r.split('=')[0]] += 1

            # Progress line
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(pending) - i) / rate / 3600 if rate > 0 else 0
            icon = '✓' if status == 'VALIDATED' else '✗' if status == 'REJECTED' else '○'

            detail = ''
            if status == 'VALIDATED':
                detail = f"WR={res.get('full_wr',0):.1f}% PnL={res.get('full_pnl',0):.0f}% R:R={res.get('rr',0):.2f}"
            elif status == 'REJECTED':
                detail = f"WR={res.get('full_wr',0):.1f}% raw={res.get('raw_wr',0):.1f}%"
            else:
                detail = res.get('reason', '')[:40]

            line = (f"  [{i}/{len(pending)}] {icon} {res.get('strategy','?'):<25} "
                    f"{res.get('symbol','?'):<18} {res.get('timeframe','?'):<5} "
                    f"{detail:<40} ETA:{eta:.1f}h")
            print(line)
            log.write(line + '\n')

            # Checkpoint every 10
            if i % 10 == 0:
                done_keys_now = done_keys | {
                    f"{r['strategy']}_{r['symbol']}_{r['timeframe']}" for r in all_results
                }
                prog = {
                    'total': len(grails), 'processed': len(done_keys_now),
                    'validated': len(validated), 'rejected': len(rejected),
                    'done_keys': list(done_keys_now),
                    'timestamp': datetime.now().isoformat(),
                }
                tmp = progress_file + '.tmp'
                with open(tmp, 'w') as f:
                    json.dump(prog, f)
                os.replace(tmp, progress_file)

                # Also save results incrementally
                for fpath, data in [(validated_file, validated),
                                    (rejected_file, rejected),
                                    (all_results_file, all_results)]:
                    tmp = fpath + '.tmp'
                    with open(tmp, 'w') as f:
                        json.dump(data, f, indent=1, default=str)
                    os.replace(tmp, fpath)

    # Final save
    for fpath, data in [(validated_file, validated),
                        (rejected_file, rejected),
                        (all_results_file, all_results)]:
        with open(fpath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    elapsed = time.time() - t0

    # Summary
    summary = f"""
{'='*70}
  FASE 2b V2 COMPLETE — {elapsed/3600:.1f}h
{'='*70}
  VALIDATED:      {counts['VALIDATED']}
  REJECTED:       {counts['REJECTED']}
  SKIPPED:        {counts['SKIP']}
  ERRORS:         {counts['ERROR']}
  TOTAL:          {sum(counts.values())}

  Gate failures:
"""
    for reason, cnt in gate_failures.most_common(10):
        total_rej = counts['REJECTED'] or 1
        summary += f"    {reason:<20} {cnt} ({cnt/total_rej*100:.0f}%)\n"

    # RAW signal stats (for grails that were processed)
    raw_wrs = [r.get('raw_wr', 0) for r in all_results if r.get('raw_wr')]
    if raw_wrs:
        import statistics
        summary += f"\n  RAW signal performance (no SL/TP):"
        summary += f"\n    Avg raw WR:   {statistics.mean(raw_wrs):.1f}%"
        summary += f"\n    Max raw WR:   {max(raw_wrs):.1f}%"
        raw_pos_pnl = [r for r in all_results if r.get('raw_pnl', 0) > 0]
        summary += f"\n    Raw PnL > 0:  {len(raw_pos_pnl)} / {len(raw_wrs)} ({len(raw_pos_pnl)/len(raw_wrs)*100:.1f}%)"

    summary += f"""
  Output files:
    {validated_file}
    {rejected_file}
    {all_results_file}
    {progress_file}
{'='*70}
"""
    print(summary)
    log.write(summary)
    log.close()

    # Top 10 validated
    if validated:
        print("\n  TOP VALIDATED GRAILS:")
        top = sorted(validated, key=lambda g: g.get('full_pnl', 0), reverse=True)[:20]
        print(f"  {'Strategy':<25} {'Symbol':<18} {'TF':<5} {'WR%':<6} {'PnL%':<8} {'R:R':<6} {'Trades':<7} {'Lev':<4}")
        print(f"  {'-'*90}")
        for g in top:
            print(f"  {g['strategy']:<25} {g['symbol']:<18} {g['timeframe']:<5} "
                  f"{g.get('full_wr',0):<6.1f} {g.get('full_pnl',0):<8.1f} "
                  f"{g.get('rr',0):<6.2f} {g.get('full_trades',0):<7} "
                  f"{g.get('leverage',0):<4}")

    # Top 10 "almost there" (best rejected by positive EV)
    almost = [r for r in rejected if r.get('full_pnl', 0) > -50 and r.get('full_wr', 0) > 50]
    if almost:
        print(f"\n  CLOSE CALLS (rejected but promising — {len(almost)} total):")
        almost_top = sorted(almost, key=lambda g: g.get('full_wr', 0), reverse=True)[:10]
        for g in almost_top:
            print(f"  {g['strategy']:<25} {g['symbol']:<18} {g['timeframe']:<5} "
                  f"WR={g.get('full_wr',0):.1f}% PnL={g.get('full_pnl',0):.1f}% "
                  f"R:R={g.get('rr',0):.2f} raw_wr={g.get('raw_wr',0):.1f}% "
                  f"reasons={g.get('gate_reasons',[])}")


if __name__ == '__main__':
    main()
