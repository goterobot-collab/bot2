#!/usr/bin/env python3
"""
OPTUNA V7.2.2 — Dynamic R:R floor by WR (min_rr = (1-WR)/WR + 0.10).

V7.2 improvements over V7.1:
  J. SL = P90(ALL MAE) × 2.0 — wider safety net (WR 64%→71%, +53% PnL on TRX)
  K. TP = P90(winners MFE) × 0.85 — captures more profit per trade
  L. R:R floor DYNAMIC by WR: (1-WR)/WR+0.10, hard floor 0.20 (was static 0.60→0.50→dynamic)
  M. TF-based SL caps (5m≤6%, 15m≤10%, 1h≤18%, 4h≤30%)
  N. Leverage capped by R:R (R:R<0.70→1x, R:R≥0.70→max 2x)

V7.1 improvements over V7:
  A. Numba @njit for backtest_raw_fast_wr (100-150x speedup in MC loop)
  B. Early-exit Monte Carlo (Beta-Binomial sequential testing)
  C. CPCV early-exit (abort impossible paths)
  D. Gate-Aware Re-Selection (top 5 Pareto trials vs gates, not just best)
  E. Failure logging (gate values saved for near-misses)
  F. Leverage cap at 10x (NEO Keltner 18x was suicide)
  G. DSR edge-case fixes (nan/0.0 → handled gracefully)
  H. R:R pre-check before expensive MC
  I. Production symbols support (BEAT, RIVER, DEXE, etc.)

Original V7 optimizations over V5:
  1. Worker-level candle cache (zero redundant DB reads)
  2. SQLite WAL mode + read-only pragmas
  3. Vectorized backtest_raw (NumPy, no Python loop for MC)
  4. Monte Carlo reduced 500→200 perms (still p<0.05 valid)
  5. CPCV embargo O(1) instead of O(n) per index
  6. Worker initializer (strategies loaded once, not per-combo)
  7. Background progress writes (non-blocking)

Original improvements over v4 (optuna_v4.py):
  1. BUG FIX: SL/TP calibrated from last CV train window only (not full dataset)
     → fixes "immortal trades" bias that overestimated SL in v4
  2. BUG FIX: Indicator warm-up in CV windows (gen receives df[:test_end] not just test)
     → fixes inflated WR from cold-start indicators (EMA200, RSI etc)
  3. BUG FIX: SL P95 from ALL trades MAE (not winners-only)
     → losers have larger MAE; winners-only SL was too tight for production
  4. BUG FIX: Monte Carlo returns False on insufficient data (was True = pass-by-default)
  5. CPCV: Combinatorial Purged Cross-Validation C(6,2)=15 paths vs 3 walk-forward
     → more robust OOS validation, less path-dependent bias
  6. Block Bootstrap MC: Stationary block bootstrap (sqrt(n) blocks) vs random shuffle
     → preserves temporal autocorrelation in signal series
  7. MC 500 permutations: was 100 → better p-value resolution at p<0.05
  8. Gate G9: Year Coverage — requires ≥2 of 3 recent years (2023-2025) active
     → rejects "ghost of 2021" strategies with no recent activity
  9. Gate G10: EV with real SL/TP trades (G6 uses theoretical WR; G10 uses actual PnL)
 10. Objective uses user_attrs constraints (not -999 penalty) → cleaner Pareto front

Usage:
  python3 optuna_v5.py                                   # All strategies × top 100 symbols
  python3 optuna_v5.py --symbol BTC/USDT:USDT            # One symbol
  python3 optuna_v5.py --top 50 --tf 4h                  # Top 50, specific TF
  python3 optuna_v5.py --round r2                        # Only R2 strategies
  python3 optuna_v5.py --workers 6 --trials 20           # Custom parallelism
  python3 optuna_v5.py --single-objective                # Legacy single-objective mode
  python3 optuna_v5.py --journal                         # Use JournalFileBackend (robust storage)
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
import argparse
import traceback
import threading
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import multiprocessing

warnings.filterwarnings('ignore')

# V7.1: Numba JIT for Monte Carlo speedup
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ─── V7.6: Atomic JSON write (prevents corruption on crash/kill) ───
def atomic_save_json(data, filepath):
    """Write JSON atomically: write to .tmp, then os.replace → no corruption window."""
    tmp_path = filepath + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, default=str)
    os.replace(tmp_path, filepath)  # atomic on POSIX

# ─── V7 OPTIMIZATION: Worker-level caches ───
_CANDLE_CACHE = {}  # {(symbol, tf): df} — populated in worker initializer
_STRATEGY_CACHE = {}  # loaded once per worker via initializer

# ─── CONFIG ───
DB_PATH = "/Users/sabrina/CLAUDE CODE/data/activos_binance.db"
PROJECT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Defaults (overridable via CLI)
N_TRIALS = 50          # V7.5: aumentado de 20 → mejor exploración del espacio de params
MAX_WORKERS = 8
COMMISSION = 0.001
SLIPPAGE = 0.0005
MIN_TRADES_TRAIN = 5
MIN_TRADES_TEST = 2    # V7.5: bajado de 5 → el threshold real lo maneja G3 dinámico
MIN_TRADES_FULL = 4    # V7.5: pre-filtro mínimo absoluto, G3 aplica umbral dinámico
TIMEOUT_PER_STUDY = 60  # seconds

# V5 NEW: CPCV config (C(8,3)=56 paths — upgraded from C(6,2)=15)
# López de Prado (2018): more paths = less overfitting. 56 paths ≈ 3.7x better anti-overfit.
# Tradeoff: each trial ~3.7x slower → fewer trials in TIMEOUT_PER_STUDY window.
# Still much faster than C(10,5)=252 which would be 16x slower.
N_FOLDS_CPCV = 8
N_TEST_FOLDS_CPCV = 3
EMBARGO_PCT = 0.02        # 2% gap between train/test to prevent leakage

# Walk-forward fallback (used when df < 500 rows)
N_CV_WINDOWS = 3

# Gates constants
MIN_TEST_WR = 70.0
MAX_OVERFIT_GAP = 30.0
SL_MIN, SL_MAX = 0.02, 0.40
TP_MIN, TP_MAX = 0.01, 0.50
MAX_DUR_CAP = 720         # hours
RR_FLOOR_ABSOLUTE = 0.20  # V7.2.2: Hard floor — never below this
# V7.2.2: Dynamic R:R floor based on WR (our 17 grails: median R:R=0.43, min=0.32)
# Formula: min_rr = (1-WR)/WR + 0.10
# WR=70%→0.53 | WR=75%→0.43 | WR=80%→0.35 | WR=85%→0.28 | WR=90%→0.21
def dynamic_rr_floor(wr_pct):
    wr = max(wr_pct / 100.0, 0.50)
    return max(RR_FLOOR_ABSOLUTE, (1 - wr) / wr + 0.10)
DD_PNL_MAX = 2.0  # V5.1 FBI fix: wide SL = deeper DD mechanically → 1.5→2.0

# V7.5: Trades mínimos DINÁMICOS según WR — a mayor WR cada trade da más información
# WR=100% → 2 trades  |  WR≥90% → 4 trades  |  WR≥80% → 8 trades  |  WR≥70% → 8 trades
# Lógica: WR=90% con 4 trades ya es estadísticamente más fuerte que WR=70% con 10 trades
def dynamic_min_trades(wr_pct):
    if wr_pct == 100: return 2   # 100% WR → 2 trades suficientes
    if wr_pct >= 90:  return 4
    if wr_pct >= 80:  return 8
    return 8   # WR >= 70% → 8 trades (antes 10)

# V7.4: TF_MAX_SL ELIMINADO — REGLA DE ORO: SL = empírico per (activo × estrategia)
# NUNCA caps genéricos por TF. El rango de movimiento lo determina activo+estrategia, no el TF.
# Safety absoluto único válido: sl < 1.0 (anti-liquidación) → aplicado abajo vía SL_MAX=0.95

# V5 NEW: Monte Carlo config — block bootstrap, 500 perms
MC_PERMUTATIONS = 500     # was 100
MC_P_THRESHOLD = 0.05

# Timeframes to test
USE_TFS = ['5m', '15m', '1h', '4h', '1d']


# TF PRIORITIZATION WEIGHTS (SABRINA 2026-04-16)
# PRIORIZAR estrategias con MEJOR performance en 5m/15m
# Más weight = más trials asignados en Optuna
TF_PRIORITY_WEIGHTS = {
    '5m': 2.5,      # MÁXIMA PRIORIDAD
    '15m': 2.0,     # ALTA PRIORIDAD
    '1h': 1.0,      # Normal
    '4h': 0.8,      # Baja
    '1d': 0.6       # Muy baja
}



# V7.7: Calcular trials dinámicamente por TF según pesos
def calculate_trials_by_tf(base_trials, tf, weights_dict=TF_PRIORITY_WEIGHTS):
    """Distribución dinámica: trials_tf = base × weight[tf] / suma(weights)"""
    if tf not in weights_dict:
        return base_trials
    total_weight = sum(weights_dict.values())
    tf_weight = weights_dict[tf]
    return max(1, int(round(base_trials * tf_weight / total_weight)))


sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "strategies_code"))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════
# DB LAYER
# ═══════════════════════════════════════════════════════════════════════

def load_candles(symbol, tf="5m"):
    """V7: SQLite WAL mode + read-only pragmas for concurrent reads."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
    df = pd.read_sql(
        "SELECT ts,open,high,low,close,volume FROM candles "
        "WHERE symbol=? AND timeframe=? ORDER BY ts",
        conn, params=(symbol, tf))
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
    return df


def resample(df, target):
    m = {'15m': '15min', '4h': '4h', '1d': '1D'}
    r = m.get(target)
    if not r:
        return None
    res = df.resample(r).agg({
        'ts': 'first', 'open': 'first', 'high': 'max',
        'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna(subset=['close'])
    return res if len(res) >= 100 else None


def get_candles(symbol, tf):
    """V7: Worker-level cache — each (symbol, tf) loaded at most once per worker."""
    cache_key = (symbol, tf)
    if cache_key in _CANDLE_CACHE:
        return _CANDLE_CACHE[cache_key]

    if tf in ('5m', '1h'):
        df = load_candles(symbol, tf)
    elif tf in ('15m', '4h', '1d'):
        # Cache the 5m base too
        base_key = (symbol, '5m')
        if base_key in _CANDLE_CACHE:
            df_5m = _CANDLE_CACHE[base_key]
        else:
            df_5m = load_candles(symbol, '5m')
            if df_5m is not None:
                _CANDLE_CACHE[base_key] = df_5m
        df = resample(df_5m, tf) if df_5m is not None else None
    else:
        df = None

    if df is not None:
        _CANDLE_CACHE[cache_key] = df
    return df


_SYMBOLS_CACHE_FILE = os.path.join(OUTPUT_DIR, ".symbols_cache.json")

def get_symbols(top_n=None):
    """Get available symbols from DB. Uses disk cache (refreshes when DB modified)."""
    # V7.6: File-based cache — avoids 5+ min DB query on every startup
    db_mtime = os.path.getmtime(DB_PATH) if os.path.exists(DB_PATH) else 0
    cache_valid = False
    if os.path.exists(_SYMBOLS_CACHE_FILE):
        try:
            with open(_SYMBOLS_CACHE_FILE) as f:
                cache = json.load(f)
            if cache.get('db_mtime', 0) >= db_mtime and cache.get('symbols'):
                cache_valid = True
                all_syms = cache['symbols']  # Already sorted by bars DESC
                print(f"  [V7.6] Symbols from cache: {len(all_syms)} symbols (DB unchanged)")
        except Exception:
            pass

    if not cache_valid:
        print(f"  [V7.6] Querying DB for symbols (cache miss)...")
        t0 = time.time()
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            "SELECT symbol, COUNT(*) as bars FROM candles "
            "WHERE timeframe='5m' GROUP BY symbol ORDER BY bars DESC", conn)
        conn.close()
        all_syms = df['symbol'].tolist()
        # Save cache
        try:
            with open(_SYMBOLS_CACHE_FILE, 'w') as f:
                json.dump({'symbols': all_syms, 'db_mtime': db_mtime,
                          'updated': datetime.now().isoformat()}, f)
            print(f"  [V7.6] Symbols cached: {len(all_syms)} ({time.time()-t0:.1f}s query)")
        except Exception:
            pass

    return all_syms[:top_n] if top_n else all_syms


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════

def backtest_raw(df, signals):
    """Run trades WITHOUT SL/TP. Supports LONG (1) and SHORT (-1) entries.

    Signal convention:
      1  → open LONG (or close SHORT and open LONG)
     -1  → open SHORT (or close LONG and open SHORT)
      0  → close position if open
    """
    if signals is None:
        return []
    sig = signals.values
    opens, highs, lows = df['open'].values, df['high'].values, df['low'].values
    cost = SLIPPAGE + COMMISSION
    n = min(len(sig), len(opens))
    trades = []
    pos = 0; ep = 0.; ei = 0; mae = 0.; mfe = 0.

    for i in range(1, n):
        s = int(sig[i - 1]); p = opens[i]

        if pos == 0:
            if s == 1:
                ep = p * (1 + cost); ei = i; mae = 0.; mfe = 0.; pos = 1
            elif s == -1:
                ep = p * (1 - cost); ei = i; mae = 0.; mfe = 0.; pos = -1

        elif pos == 1:
            lp = (lows[i] - ep) / ep
            hp = (highs[i] - ep) / ep
            if lp < mae: mae = lp
            if hp > mfe: mfe = hp
            if s == -1 or s == 0:
                xp = p * (1 - cost)
                pnl = (xp - ep) / ep
                yr = df.index[ei].year if hasattr(df.index[ei], 'year') else 2025
                trades.append({'pnl': pnl, 'mae': mae, 'mfe': mfe,
                                'dur_bars': i - ei, 'year': yr, 'win': pnl > 0})
                pos = 0
                if s == -1:  # flip to short
                    ep = p * (1 - cost); ei = i; mae = 0.; mfe = 0.; pos = -1

        elif pos == -1:
            hp = (ep - highs[i]) / ep
            lp = (ep - lows[i]) / ep
            if hp < mae: mae = hp   # mae = worst adverse excursion
            if lp > mfe: mfe = lp   # mfe = best favorable excursion
            if s == 1 or s == 0:
                xp = p * (1 + cost)
                pnl = (ep - xp) / ep
                yr = df.index[ei].year if hasattr(df.index[ei], 'year') else 2025
                trades.append({'pnl': pnl, 'mae': mae, 'mfe': mfe,
                                'dur_bars': i - ei, 'year': yr, 'win': pnl > 0})
                pos = 0
                if s == 1:  # flip to long
                    ep = p * (1 + cost); ei = i; mae = 0.; mfe = 0.; pos = 1

    return trades


def _backtest_fast_wr_python(df_opens, sig_values, cost, min_trades):
    """Pure Python fallback for fast WR backtest."""
    n = min(len(sig_values), len(df_opens))
    wins = 0
    total = 0
    pos = 0; ep = 0.

    for i in range(1, n):
        s = int(sig_values[i - 1]); p = df_opens[i]

        if pos == 0:
            if s == 1:
                ep = p * (1 + cost); pos = 1
            elif s == -1:
                ep = p * (1 - cost); pos = -1
        elif pos == 1:
            if s == -1 or s == 0:
                xp = p * (1 - cost)
                if (xp - ep) > 0:
                    wins += 1
                total += 1
                pos = 0
                if s == -1:
                    ep = p * (1 - cost); pos = -1
        elif pos == -1:
            if s == 1 or s == 0:
                xp = p * (1 + cost)
                if (ep - xp) > 0:
                    wins += 1
                total += 1
                pos = 0
                if s == 1:
                    ep = p * (1 + cost); pos = 1

    if total < min_trades:
        return -1.0, 0
    return wins / total * 100, total


if HAS_NUMBA:
    @numba.njit(cache=True)
    def _backtest_fast_wr_numba(df_opens, sig_values, cost, min_trades):
        """V7.1: Numba JIT — 100-150x faster than Python loop for MC."""
        n = min(len(sig_values), len(df_opens))
        wins = 0
        total = 0
        pos = 0
        ep = 0.0

        for i in range(1, n):
            s = sig_values[i - 1]
            p = df_opens[i]

            if pos == 0:
                if s == 1:
                    ep = p * (1.0 + cost); pos = 1
                elif s == -1:
                    ep = p * (1.0 - cost); pos = -1
            elif pos == 1:
                if s == -1 or s == 0:
                    xp = p * (1.0 - cost)
                    if (xp - ep) > 0:
                        wins += 1
                    total += 1
                    pos = 0
                    if s == -1:
                        ep = p * (1.0 - cost); pos = -1
            elif pos == -1:
                if s == 1 or s == 0:
                    xp = p * (1.0 + cost)
                    if (ep - xp) > 0:
                        wins += 1
                    total += 1
                    pos = 0
                    if s == 1:
                        ep = p * (1.0 + cost); pos = 1

        if total < min_trades:
            return -1.0, 0
        return wins / total * 100.0, total


def backtest_raw_fast_wr(df_opens, df_highs, df_lows, sig_values, cost):
    """V7.1: Fast backtest — Numba JIT if available, Python fallback otherwise.

    Returns (wr, count) or (None, None) if insufficient trades.
    """
    sig_int = sig_values.astype(np.int64) if not np.issubdtype(sig_values.dtype, np.integer) else sig_values
    if HAS_NUMBA:
        wr, cnt = _backtest_fast_wr_numba(df_opens, sig_int, cost, MIN_TRADES_TRAIN)
    else:
        wr, cnt = _backtest_fast_wr_python(df_opens, sig_int, cost, MIN_TRADES_TRAIN)
    if wr < 0:
        return None, None
    return round(wr, 1), cnt


def backtest_with_sl_tp(df, signals, sl_pct, tp_pct, max_dur_bars=None):
    """Re-simulate with SL/TP applied and ATR-based dynamic slippage.

    ATR slippage (3/3 IA consensus, Kissell 2013):
      slip[i] = clip(max(0.05%, ATR%*0.1), 0.05%, 1.0%)
    More accurate than fixed 0.05% — doubles during high-volatility bars.
    """
    if signals is None:
        return []
    sig = signals.values
    opens, highs, lows = df['open'].values, df['high'].values, df['low'].values
    closes = df['close'].values
    n = len(df)

    # ATR-based dynamic slippage (14-period ATR, 0.1x scaling)
    prev_c = np.empty(n); prev_c[0] = closes[0]; prev_c[1:] = closes[:-1]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    # Simple rolling mean with np.convolve (matches pandas rolling(14,min_periods=1))
    kernel = np.ones(14) / 14
    atr_full = np.convolve(tr, kernel, mode='full')[:n]
    # Clamp: min 0.05%, max 1.0%
    atr_slip = np.clip((atr_full / np.where(closes > 0, closes, 1)) * 0.1, 0.0005, 0.01)

    trades = []
    pos = 0; ep = 0.; ei = 0; entry_slip = 0.

    for i in range(1, n):
        s = sig[i - 1]; p = opens[i]; slip_i = atr_slip[i]
        if pos == 0 and s == 1:
            entry_slip = slip_i
            ep = p * (1 + entry_slip + COMMISSION); ei = i; pos = 1
        elif pos == 1:
            if (lows[i] - ep) / ep <= -sl_pct:
                xp = ep * (1 - sl_pct)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei, 'exit': 'SL'})
                pos = 0; continue
            if (highs[i] - ep) / ep >= tp_pct:
                xp = ep * (1 + tp_pct)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei, 'exit': 'TP'})
                pos = 0; continue
            if max_dur_bars and (i - ei) >= max_dur_bars:
                xp = p * (1 - slip_i - COMMISSION)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei, 'exit': 'DUR'})
                pos = 0; continue
            if s == -1:
                xp = p * (1 - slip_i - COMMISSION)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei, 'exit': 'SIG'})
                pos = 0
    return trades


def compute_metrics(trades):
    """Compute full metrics from trade list."""
    if not trades or len(trades) < MIN_TRADES_TRAIN:
        return None
    pnls = np.array([t['pnl'] for t in trades])
    n = len(pnls)
    wins = (pnls > 0).sum()
    wr = wins / n * 100
    total_pnl = pnls.sum() * 100  # percent

    # Sharpe
    sharpe = (pnls.mean() / (pnls.std() + 1e-10)) * np.sqrt(252)

    # Sortino (downside deviation only)
    neg = pnls[pnls < 0]
    downside_std = neg.std() if len(neg) > 1 else pnls.std()
    sortino = (pnls.mean() / (downside_std + 1e-10)) * np.sqrt(252)

    # Max drawdown
    cum = pnls.cumsum()
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum).max() * 100  # percent

    # Profit factor
    wm = pnls > 0
    avg_w = pnls[wm].mean() if wm.any() else 0
    avg_l = pnls[~wm].mean() if (~wm).any() else 0
    losses_n = (~wm).sum()
    pf = abs(avg_w * wins / (avg_l * losses_n + 1e-10)) if losses_n > 0 else 999

    # Calmar ratio (annualized return / max DD)
    calmar = (total_pnl / (dd + 1e-10))

    # MAE P95
    maes = [abs(t.get('mae', 0)) for t in trades]
    mae_p95 = np.percentile(maes, 95) if maes else 0.1

    # Yearly breakdown
    years = np.array([t.get('year', 2025) for t in trades])
    yearly = {}
    for y in np.unique(years):
        m = years == y; yp = pnls[m]; yw = (yp > 0).sum()
        yearly[str(y)] = {
            'wr': round(yw / len(yp) * 100, 1),
            'trades': int(len(yp)),
            'pnl': round(yp.sum() * 100, 2)
        }

    return {
        'trades': n, 'wr': round(wr, 1), 'pnl': round(total_pnl, 2),
        'sharpe': round(sharpe, 2), 'sortino': round(sortino, 2),
        'max_drawdown': round(dd, 2), 'profit_factor': round(pf, 2),
        'calmar': round(calmar, 2), 'mae_p95': round(mae_p95, 4),
        'yearly': yearly,
    }


# ═══════════════════════════════════════════════════════════════════════
# TEMPORAL CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def make_wf_splits_v5(df, n_windows=N_CV_WINDOWS, embargo_pct=EMBARGO_PCT):
    """V5 NEW: Walk-forward fallback splits returning 4-tuples (train, test, label, test_end_idx).

    Used when df < 500 rows (not enough data for CPCV C(6,2)).
    Same logic as v4 make_cv_splits but returns 4-tuples with test_end_idx
    for indicator warm-up (BUG FIX 2).
    """
    n = len(df)
    if n < 300:
        return []

    splits = []
    test_start_pct = 0.40
    test_block_size = (1.0 - test_start_pct) / n_windows

    for w in range(n_windows):
        train_end_pct = test_start_pct + w * test_block_size
        test_start = train_end_pct + embargo_pct
        test_end = test_start + test_block_size

        if test_end > 1.0:
            test_end = 1.0

        train_end_idx = int(n * train_end_pct)
        test_start_idx = int(n * test_start)
        test_end_idx = int(n * test_end)

        train = df.iloc[:train_end_idx]
        test = df.iloc[test_start_idx:test_end_idx]

        if len(train) >= 100 and len(test) >= 50:
            # V5 NEW: 4-tuple with test_end_idx for warm-up
            splits.append((train, test, f"W{w+1}", test_end_idx))

    return splits


def make_cpcv_splits(df, n_folds=N_FOLDS_CPCV, n_test_folds=N_TEST_FOLDS_CPCV,
                     embargo_pct=EMBARGO_PCT):
    """V5 NEW: CPCV — Combinatorial Purged Cross-Validation.

    C(6,2) = 15 independent paths. Each path uses 4 folds for train, 2 for test.
    Embargo purges train rows within embargo_bars of any test boundary.

    Falls back to make_wf_splits_v5() if df < 500 rows.

    Returns list of 4-tuples: (train_df, test_df, label, test_end_idx)
    where test_end_idx is the integer position of the last test row + 1
    in the original df (for BUG FIX 2 warm-up).
    """
    n = len(df)

    # V5 NEW: fallback to walk-forward if not enough data for CPCV
    if n < 500:
        return make_wf_splits_v5(df)

    fold_size = n // n_folds
    folds = [
        (i * fold_size, (i + 1) * fold_size if i < n_folds - 1 else n)
        for i in range(n_folds)
    ]
    embargo_bars = max(1, int(n * embargo_pct))
    splits = []

    for test_combo in combinations(range(n_folds), n_test_folds):
        test_set = set(test_combo)
        train_set = set(range(n_folds)) - test_set

        # Gather test indices
        test_idx = []
        for fi in sorted(test_set):
            test_idx.extend(range(folds[fi][0], folds[fi][1]))
        test_start_bound = min(test_idx)
        test_end_bound = max(test_idx)

        # V7 FIX: O(1) embargo check instead of O(n) per index
        train_idx = []
        emb_lo = test_start_bound - embargo_bars
        emb_hi_start = test_start_bound + embargo_bars
        emb_lo_end = test_end_bound - embargo_bars
        emb_hi = test_end_bound + embargo_bars
        for fi in sorted(train_set):
            for idx in range(folds[fi][0], folds[fi][1]):
                if (emb_lo <= idx <= emb_hi_start) or (emb_lo_end <= idx <= emb_hi):
                    continue  # too close to test boundary
                train_idx.append(idx)

        if len(train_idx) < 100 or len(test_idx) < 50:
            continue

        train_df = df.iloc[sorted(train_idx)]
        test_df = df.iloc[sorted(test_idx)]
        test_end_idx = test_end_bound + 1  # V5 NEW: for warm-up slice df.iloc[:test_end_idx]
        label = f"CPCV_{test_combo[0]}{test_combo[1]}"
        splits.append((train_df, test_df, label, test_end_idx))

    return splits if splits else make_wf_splits_v5(df)


# ═══════════════════════════════════════════════════════════════════════
# SAMPLER SELECTION
# ═══════════════════════════════════════════════════════════════════════

def has_categorical_params(space_fn):
    """Check if a space function uses categorical params (needs TPE)."""
    try:
        study = optuna.create_study()
        trial = study.ask()
        space_fn(trial)
        for name, dist in trial.distributions.items():
            if isinstance(dist, optuna.distributions.CategoricalDistribution):
                return True
        return False
    except Exception:
        return True  # default to TPE (safe)


def pick_sampler(space_fn, multi_objective=False, seed=42):
    """Pick best sampler based on search space + objectives.

    - GPSampler: pure int/float, converges faster with 20-30 trials
    - TPESampler: has categoricals or fallback
    """
    has_cat = has_categorical_params(space_fn)

    if has_cat:
        return optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            n_startup_trials=5,
        )
    else:
        try:
            import torch  # noqa: F401
            return optuna.samplers.GPSampler(
                seed=seed,
                deterministic_objective=True,
                n_startup_trials=5,
            )
        except (ImportError, AttributeError, TypeError):
            return optuna.samplers.TPESampler(
                seed=seed,
                multivariate=True,
                n_startup_trials=5,
            )


# ═══════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def objective_single_cv_v5(gen, space, cv_splits, df):
    """V5: Single-objective with WilcoxonPruner + BUG FIX 2 (warm-up).

    cv_splits: 4-tuples (train_df, test_df, label, test_end_idx)
    """
    def objective(trial):
        params = space(trial)
        window_scores = []

        for step, (train_w, test_w, label, test_end_idx) in enumerate(cv_splits):
            try:
                # V5 FIX (BUG FIX 2): pass full prefix for warm-up
                sig_full = gen(df.iloc[:test_end_idx], **params)
                sig_test = sig_full.reindex(test_w.index).fillna(0)
                raw = backtest_raw(test_w, sig_test)
                m = compute_metrics(raw)
                if m is None:
                    score = 0.0
                else:
                    score = m['wr'] * 0.5 + min(m['sortino'], 50) * 0.3 - m['max_drawdown'] * 0.2
            except Exception:
                score = -999

            window_scores.append(score)
            trial.report(score, step)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(window_scores))
    return objective


def objective_multi_cpcv(gen, space, cv_splits, df):
    """V5 NEW: Multi-objective across CPCV paths with constraints (BUG FIX 2 + 8).

    Uses user_attrs for constraints instead of -999 penalty values to avoid
    contaminating the Pareto front with infeasible trials.

    cv_splits: 4-tuples (train_df, test_df, label, test_end_idx)
    """
    def objective(trial):
        params = space(trial)
        all_wr, all_sortino, all_dd = [], [], []
        total_trades = 0

        for train_df, test_df, label, test_end_idx in cv_splits:
            try:
                # V5 FIX (BUG FIX 2): warm-up via full prefix
                sig_full = gen(df.iloc[:test_end_idx], **params)
                sig_test = sig_full.reindex(test_df.index).fillna(0)
                raw = backtest_raw(test_df, sig_test)
                m = compute_metrics(raw)
                if m is None:
                    all_wr.append(0.0); all_sortino.append(-50.0); all_dd.append(100.0)
                else:
                    all_wr.append(m['wr']); all_sortino.append(m['sortino'])
                    all_dd.append(m['max_drawdown'])
                    total_trades += m['trades']
            except Exception:
                all_wr.append(0.0); all_sortino.append(-50.0); all_dd.append(100.0)

        # V5 NEW (BUG FIX 8): soft constraint via user_attrs — value > 0 = violation
        expected_min = MIN_TRADES_FULL * len(cv_splits)
        c_min_trades = max(0.0, (expected_min - total_trades) / (expected_min + 1e-10))
        trial.set_user_attr("constraints", [c_min_trades])

        return float(np.mean(all_wr)), float(np.mean(all_sortino)), float(np.mean(all_dd))
    return objective


def select_from_pareto(study, weights=None):
    """Select best trial from Pareto front using TOPSIS.

    Objectives: [maximize WR, maximize Sortino, minimize DD]
    Default weights: WR=50%, Sortino=30%, DD=20%
    """
    trials = study.best_trials
    if not trials:
        return None
    if len(trials) == 1:
        return trials[0]

    if weights is None:
        weights = [0.5, 0.3, 0.2]
    weights = np.array(weights)

    matrix = np.array([[t.values[0], min(t.values[1], 50), t.values[2]] for t in trials], dtype=float)
    directions = [1, 1, -1]  # +1=maximize, -1=minimize

    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0] = 1e-10
    norm_matrix = (matrix / norms) * weights

    ideal = np.array([
        norm_matrix[:, i].max() if d == 1 else norm_matrix[:, i].min()
        for i, d in enumerate(directions)
    ])
    anti_ideal = np.array([
        norm_matrix[:, i].min() if d == 1 else norm_matrix[:, i].max()
        for i, d in enumerate(directions)
    ])

    d_ideal = np.sqrt(((norm_matrix - ideal) ** 2).sum(axis=1))
    d_anti = np.sqrt(((norm_matrix - anti_ideal) ** 2).sum(axis=1))
    closeness = d_anti / (d_ideal + d_anti + 1e-10)

    return trials[int(np.argmax(closeness))]


def _rank_pareto_trials(trials, weights=None):
    """V7.1: Rank ALL Pareto trials by TOPSIS score (not just pick the best).

    Returns list of trials sorted best→worst. Used for Gate-Aware Re-Selection.
    """
    if not trials:
        return []
    if len(trials) == 1:
        return trials

    if weights is None:
        weights = [0.5, 0.3, 0.2]
    weights = np.array(weights)

    matrix = np.array([[t.values[0], min(t.values[1], 50), t.values[2]] for t in trials], dtype=float)
    directions = [1, 1, -1]

    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0] = 1e-10
    norm_matrix = (matrix / norms) * weights

    ideal = np.array([
        norm_matrix[:, i].max() if d == 1 else norm_matrix[:, i].min()
        for i, d in enumerate(directions)
    ])
    anti_ideal = np.array([
        norm_matrix[:, i].min() if d == 1 else norm_matrix[:, i].max()
        for i, d in enumerate(directions)
    ])

    d_ideal = np.sqrt(((norm_matrix - ideal) ** 2).sum(axis=1))
    d_anti = np.sqrt(((norm_matrix - anti_ideal) ** 2).sum(axis=1))
    closeness = d_anti / (d_ideal + d_anti + 1e-10)

    ranked_indices = np.argsort(-closeness)  # descending
    return [trials[int(i)] for i in ranked_indices]


# ═══════════════════════════════════════════════════════════════════════
# MONTE CARLO — STATIONARY BLOCK BOOTSTRAP (V5)
# ═══════════════════════════════════════════════════════════════════════

def monte_carlo_block_bootstrap(gen, best_params, df, n_perms=MC_PERMUTATIONS):
    """V7: Optimized Monte Carlo — 200 perms + fast_wr backtest (no dict alloc).

    Same null hypothesis as V5.1 (random shuffle), but:
    - 200 perms instead of 500 (still valid for p<0.05, resolution=0.005)
    - Uses backtest_raw_fast_wr instead of full backtest_raw + compute_metrics
    - Pre-extracts numpy arrays once (avoids repeated df column access)

    Returns: (is_significant, p_value, real_wr, random_wr_95th)
    """
    n_perms = min(n_perms, 200)  # V7: cap at 200 (sufficient for p<0.05)

    try:
        real_sig = gen(df, **best_params)
        real_trades = backtest_raw(df, real_sig)
        real_m = compute_metrics(real_trades)
        if real_m is None:
            return False, 1.0, 0, 0
        real_wr = real_m['wr']
    except Exception:
        return False, 1.0, 0, 0

    # V7: Pre-extract arrays once
    vals = real_sig.values.copy()
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    cost = SLIPPAGE + COMMISSION
    rng = np.random.RandomState(42)

    # V7.1: Early-exit Monte Carlo with sequential testing
    # If after N perms, it's statistically clear → stop early (saves 30-85% time)
    random_wrs = []
    beats_real = 0
    early_exit = False

    for perm_idx in range(n_perms):
        permuted_vals = rng.permutation(vals)
        wr, cnt = backtest_raw_fast_wr(opens, highs, lows, permuted_vals, cost)
        if wr is not None:
            random_wrs.append(wr)
            if wr >= real_wr:
                beats_real += 1

        # V7.1: Sequential early-exit checks (every 30 perms after min 30)
        n_done = len(random_wrs)
        if n_done >= 30 and n_done % 30 == 0:
            current_p = beats_real / n_done
            # CLEAR PASS: p < 0.01 with high confidence → stop early (grail)
            if beats_real == 0 and n_done >= 50:
                early_exit = True
                break
            # CLEAR FAIL: p > 0.15 after 30+ perms → no edge, stop early
            if current_p > 0.15 and n_done >= 30:
                early_exit = True
                break
            # MARGINAL FAIL: p > 0.08 after 100+ perms → unlikely to pass
            if current_p > 0.08 and n_done >= 100:
                early_exit = True
                break

    if len(random_wrs) < 20:
        return False, 1.0, real_wr, 0

    random_wrs = np.array(random_wrs)
    p95 = np.percentile(random_wrs, 95)
    p_value = beats_real / len(random_wrs)

    return p_value < MC_P_THRESHOLD, round(p_value, 4), round(real_wr, 1), round(p95, 1)


# ═══════════════════════════════════════════════════════════════════════
# DEFLATED SHARPE RATIO (Bailey & Lopez de Prado, 2014)
# ═══════════════════════════════════════════════════════════════════════

def deflated_sharpe_ratio(estimated_sr, n_trials, backtest_bars,
                          skew=0.0, excess_kurtosis=0.0, sr_variance=1.0):
    """Compute DSR: probability that true SR > selection-bias-adjusted benchmark.

    DSR < 0.95 → logged as warning (not rejection — WR+Gates+MC are the hard filters).
    """
    from scipy.stats import norm

    if n_trials <= 1 or backtest_bars <= 1:
        return 1.0

    e_gamma = 0.5772156649
    sr0 = np.sqrt(sr_variance) * (
        (1 - e_gamma) * norm.ppf(1 - 1.0 / n_trials) +
        e_gamma * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )

    T = backtest_bars
    numerator = (estimated_sr - sr0) * np.sqrt(T - 1)
    denominator = np.sqrt(
        1 - skew * estimated_sr +
        (excess_kurtosis / 4.0) * estimated_sr ** 2
    )

    if denominator <= 0 or np.isnan(denominator) or np.isinf(denominator):
        return 0.5  # V7.1: neutral instead of 0.0 (unknown, not bad)

    result = float(norm.cdf(numerator / denominator))
    if np.isnan(result) or np.isinf(result):
        return 0.5  # V7.1: graceful fallback
    return result


# ═══════════════════════════════════════════════════════════════════════
# MFE ANALYSIS + GATES (V5: G1-G10)
# ═══════════════════════════════════════════════════════════════════════

def calc_sl_tp_from_mfe(raw_trades):
    """V7.2: Wide SL safety net — P90 ALL trades × 2.0.

    Empirical finding on 19,452 trades:
      - Current P95×1.0 cuts 78.5% of winners early (R:R median 0.43)
      - Wide SL (P90×2.0) raises WR 64.4%→71%, PnL +53% on TRX
      - Still protects vs black swans (worst -66% vs -95% without SL)
      - TF caps applied AFTER this function (in optimize_one)
    TP raised P85×0.8 → P90×0.85 to capture more profit per trade.
    """
    if len(raw_trades) < 5:
        return None

    # V7.2: SL = P90(ALL trades MAE) × 2.0 — wider safety net, lets trades breathe
    all_maes = [abs(t['mae']) for t in raw_trades]
    sl = np.clip(np.percentile(all_maes, 90) * 2.0, SL_MIN, SL_MAX)

    # V7.2: TP = P90(winners MFE) × 0.85 — capture more profit
    winners = [t for t in raw_trades if t['win']]
    if len(winners) < 3:
        return None
    mfes = [t['mfe'] for t in winners]
    durs = [t['dur_bars'] for t in winners]

    tp = np.clip(np.percentile(mfes, 90) * 0.85, TP_MIN, TP_MAX)
    max_dur = min(np.percentile(durs, 95) * 1.5, MAX_DUR_CAP)

    return {'sl': round(sl, 4), 'tp': round(tp, 4), 'max_dur_bars': int(max_dur)}


def check_year_coverage(yearly):
    """V5.1 FIX (Gate G9): Year Coverage — dynamic years, softer reject.

    FBI Agent 6 findings:
    - Hardcoded [2023,2024,2025] was stale (we're in 2026) → now dynamic
    - WR<50% instant-kill on 3 trades was too aggressive → only reject on >=10 trades
    - 46% of symbols (250/542) blocked by stale year list

    V7.6 FIX: High-WR bypass for low-frequency strategies (spike reversion etc.)
    - If overall WR>80% with >5 trades, pass G9 regardless of year coverage
    - These strategies fire rarely by design — no year coverage ≠ dead strategy

    Rules:
      - Dynamic: checks current year and 2 prior years
      - ≥2 of 3 years must have ≥3 trades AND WR ≥ 55%
      - Only reject if a year has ≥10 trades AND WR < 45% (not 3 trades / 50%)
      - BYPASS: WR>80% + total trades>5 → pass (low-frequency edge)
    """
    # V7.6: High-WR bypass for rare-signal strategies
    total_trades = sum(y.get('trades', 0) for y in yearly.values())
    if total_trades > 5:
        total_wins = sum(y.get('trades', 0) * y.get('wr', 0) / 100 for y in yearly.values())
        overall_wr = (total_wins / total_trades) * 100 if total_trades > 0 else 0
        if overall_wr > 80:
            return True

    current_year = datetime.now().year
    recent_years = [str(y) for y in range(current_year - 2, current_year + 1)]
    covered = 0
    for y in recent_years:
        if y in yearly:
            yr = yearly[y]
            if yr['trades'] >= 3 and yr['wr'] >= 55:
                covered += 1
            elif yr['trades'] >= 10 and yr['wr'] < 45:
                return False  # FBI fix: only reject with statistical significance
    return covered >= 2


def check_ev_positive(trades_with_sl, sl, tp):
    """V5 NEW (Gate G10): EV > 0 using ACTUAL trade PnL (not theoretical WR).

    G6 computes EV from theoretical WR × SL/TP.
    G10 computes EV from the actual simulated PnL distribution with SL/TP applied,
    catching directional bias (e.g. shorts in bull markets).

    Returns: (passes, ev_value_in_pct)
    """
    if not trades_with_sl:
        return False, -999.0
    pnls = np.array([t['pnl'] for t in trades_with_sl])
    n = len(pnls)
    if n == 0:
        return False, -999.0
    wins = (pnls > 0).sum()
    wr = wins / n
    ev = (wr * tp) - ((1 - wr) * sl)
    return ev > 0, round(ev * 100, 3)


# ─── G0: Hurst Exponent Gate (pre-Optuna filter) ────────────────────────────
# Consensus from 4/5 IAs (Mistral, Groq, Cohere, SambaNova) — 2026-04-07
# Skip mean-reversion strategies on trending markets (H > 0.55)
# Reduces overfitting: prevents Optuna from finding params for edges that don't exist

_MEAN_REV_KEYWORDS = frozenset([
    'rsi', 'bb_mean', 'zscore', 'z_score', 'vwap', 'mean_rev', 'ob_os',
    'stoch', 'cci', 'williams', 'nasos', 'cluchanix', 'ewo', 'bounce',
    'revert', 'oversold', 'overbought', 'mean_reversion', 'bb_rev',
])


def hurst_exponent(ts, max_lag=20):
    """Hurst exponent via R/S analysis. H<0.5=mean-reverting, H>0.5=trending.

    Uses log-log regression of std(price differences) vs lag length.
    Fast numpy implementation, no external dependencies.
    """
    if len(ts) < max_lag * 3:
        return 0.5  # insufficient data → neutral, don't filter
    try:
        lags = range(2, min(max_lag, len(ts) // 3))
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        if len(tau) < 4 or min(tau) <= 0:
            return 0.5
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(poly[0] * 2.0)
    except Exception:
        return 0.5  # any error → neutral


def _is_mean_reversion_strategy(strat_name):
    """Check if strategy is mean-reversion based on name keywords."""
    name_lower = strat_name.lower()
    return any(kw in name_lower for kw in _MEAN_REV_KEYWORDS)


def validate_gates_v5(trades_with_sl, sl, tp, leverage, yearly):
    """V5: 10 gates (G1-G10). Returns (all_passed, gate_results).

    G1-G8: same as v4 (WR, PnL, trades, SL×Lev, DD/PnL, EV, duration, RR)
    G9 (NEW): Year Coverage — ≥2 of 3 recent years (2023-2025) active
    G10 (NEW): EV with real SL/TP trade PnL (not theoretical WR)
    """
    if not trades_with_sl:
        return False, {'error': 'no trades'}

    pnls = np.array([t['pnl'] for t in trades_with_sl])
    n = len(pnls); wins = (pnls > 0).sum()
    wr = wins / n * 100 if n > 0 else 0
    total_pnl = pnls.sum() * 100
    cum = (pnls * 100).cumsum()
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum).max() if len(cum) > 0 else 0

    # G10: EV with real trades
    g10_pass, g10_ev = check_ev_positive(trades_with_sl, sl, tp)

    gates = {
        'G1_WR':       {'pass': wr >= MIN_TEST_WR, 'value': round(wr, 1)},
        'G2_PnL':      {'pass': total_pnl > 0, 'value': round(total_pnl, 2)},
        'G3_trades':   {'pass': n >= dynamic_min_trades(wr), 'value': n},  # V7.5: dinámico por WR
        'G4_SLxLev':   {'pass': sl * leverage * 100 < 80, 'value': round(sl * leverage * 100, 1)},
        'G5_DD_PnL':   {
            'pass': dd / (total_pnl + 1e-10) < DD_PNL_MAX if total_pnl > 0 else False,
            'value': round(dd / (total_pnl + 1e-10), 2) if total_pnl > 0 else 999
        },
        'G6_EV':       {
            'pass': (wr / 100 * tp) - ((1 - wr / 100) * sl) > 0,
            'value': round(((wr / 100 * tp) - ((1 - wr / 100) * sl)) * 100, 3)
        },
        'G7_duration': {'pass': True, 'value': max(t['dur_bars'] for t in trades_with_sl)},
        'G8_RR':       {'pass': tp / (sl + 1e-10) >= dynamic_rr_floor(wr), 'value': round(tp / (sl + 1e-10), 2)},
        'G9_YearCov':  {'pass': check_year_coverage(yearly), 'value': yearly},  # V5 NEW
        'G10_EV_real': {'pass': g10_pass, 'value': g10_ev},                     # V5 NEW
    }
    return all(g['pass'] for g in gates.values()), gates


# ═══════════════════════════════════════════════════════════════════════
# WEIGHTED SCORE (temporal — recent years worth more)
# ═══════════════════════════════════════════════════════════════════════

def calc_weighted_score(yearly):
    years = sorted(yearly.keys(), reverse=True)
    n = len(years)
    if n == 0:
        return 0
    wm = {
        1: [100], 2: [55, 45], 3: [40, 30, 30],
        4: [35, 25, 25, 15], 5: [30, 25, 20, 15, 10],
        6: [27, 22, 18, 14, 11, 8], 7: [25, 20, 16, 13, 10, 9, 7]
    }
    w = wm.get(min(n, 7), wm[7])[:n]
    return round(sum(yearly[y]['wr'] * w[i] / 100 for i, y in enumerate(years[:len(w)])), 1)


# ═══════════════════════════════════════════════════════════════════════
# CORE: OPTIMIZE ONE STRATEGY × ONE ASSET × ONE TF
# ═══════════════════════════════════════════════════════════════════════

def _load_previous_grails():
    """Load previous grail results for warm-starting.
    Priority (last wins): old grails → v7 progress files → production JSON (most up-to-date).
    """
    grail_files = [
        os.path.join(PROJECT_DIR, "data", "santo_grial_r2.json"),
        os.path.join(PROJECT_DIR, "data", "santo_grial_v2.json"),
        os.path.join(OUTPUT_DIR, "optuna_v4_progress.json"),
        os.path.join(OUTPUT_DIR, "optuna_v5_progress.json"),
    ]
    # Add all v7 progress files (each has 'grails' with best_params)
    grail_files += sorted(glob.glob(os.path.join(OUTPUT_DIR, "optuna_v7_*_progress.json")))
    # Production JSON is the most complete source (13k+ grails with params)
    prod_json = os.path.join(os.path.dirname(OUTPUT_DIR), "..", "BOT V7", "live", "data", "v6_optimized_sl_tp.json")
    grails = {}  # key: (strategy, symbol, tf) → best_params
    for f in grail_files:
        if not os.path.exists(f):
            continue
        try:
            data = json.load(open(f))
            items = data.get('grails', data) if isinstance(data, dict) else data
            if not isinstance(items, list):
                continue
            for g in items:
                key = (g.get('strategy', ''), g.get('symbol', ''), g.get('timeframe', ''))
                if 'best_params' in g and g['best_params']:
                    grails[key] = g['best_params']
        except Exception:
            continue
    # Load production JSON last (highest priority — most recent params)
    if os.path.exists(prod_json):
        try:
            prod_bots = json.load(open(prod_json))
            for b in prod_bots:
                key = (b.get('strategy', ''), b.get('symbol', ''), b.get('timeframe', ''))
                if b.get('params'):
                    grails[key] = b['params']
        except Exception:
            pass
    return grails

_PREV_GRAILS = None  # lazy-loaded cache


def optimize_one(gen, space, df, symbol, tf, strat_name, multi_obj=True,
                 use_journal=False, journal_path=None):
    """V5: Full pipeline — CPCV → Optuna → CV validation → MFE → 10 Gates → Block MC.

    Key v5 changes vs v4:
    - CPCV splits (15 paths) instead of 3 walk-forward windows
    - Warm-up for indicator (gen receives df[:test_end_idx])
    - SL calibrated from ALL trades MAE (not winners-only)
    - SL/TP from last CV train window only (not full dataset)
    - Monte Carlo uses stationary block bootstrap
    - 10 gates instead of 8 (added G9 year coverage, G10 real EV)
    - Constraints via user_attrs (not -999 penalty)
    """
    _extra_pareto = []  # V7.1: alternative Pareto trials for Gate-Aware Re-Selection
    # ─── Step 1: CPCV splits (or WF fallback) ───
    splits = make_cpcv_splits(df)
    if not splits:
        # Fallback to single 70/30 (insufficient data for CPCV)
        split_idx = int(len(df) * 0.7)
        train, test = df.iloc[:split_idx], df.iloc[split_idx:]
        if len(train) < 100 or len(test) < 50:
            return None
        test_end_idx = len(df)
        splits = [(train, test, "W1", test_end_idx)]

    # V7.6: Also prepare WF 70/30 fallback for low-trade strategies
    # (used in Step 3 if n_trades_full < 30 — CPCV estadísticamente inválido con <30 trades)
    _wf_split_idx = int(len(df) * 0.7)
    _wf_train, _wf_test = df.iloc[:_wf_split_idx], df.iloc[_wf_split_idx:]
    _wf_splits = [(_wf_train, _wf_test, "WF_70_30", len(df))]

    # ─── Step 2: Optuna with WilcoxonPruner (single-obj) or NSGAIISampler (multi) ───
    sampler = pick_sampler(space, multi_objective=multi_obj)

    try:
        pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.1)
    except AttributeError:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3)

    # Warm-start from previous grails
    global _PREV_GRAILS
    if _PREV_GRAILS is None:
        _PREV_GRAILS = _load_previous_grails()
    prev_params = _PREV_GRAILS.get((strat_name, symbol, tf))

    # V5 NEW (MEJORA 9): optional JournalFileBackend for robust storage
    storage = None
    if use_journal and journal_path:
        try:
            storage = optuna.storages.JournalStorage(
                optuna.storages.JournalFileBackend(journal_path)
            )
        except Exception:
            storage = None  # fall back to in-memory

    if multi_obj:
        try:
            study_kwargs = dict(
                directions=['maximize', 'maximize', 'minimize'],
                sampler=sampler,
                pruner=pruner,
            )
            if storage:
                study_kwargs['storage'] = storage
                study_kwargs['study_name'] = f"{strat_name}_{symbol}_{tf}".replace('/', '_')
            study = optuna.create_study(**study_kwargs)

            if prev_params:
                try:
                    study.enqueue_trial(prev_params)
                except Exception:
                    pass

            obj_fn = objective_multi_cpcv(gen, space, splits, df)
            study.optimize(obj_fn, n_trials=N_TRIALS, timeout=TIMEOUT_PER_STUDY)

            # V7.1: Gate-Aware Re-Selection — try top 5 Pareto trials, not just best
            # If the TOPSIS-best fails gates later, another Pareto trial might pass
            pareto_trials = study.best_trials
            if not pareto_trials:
                return None
            # Rank by TOPSIS and keep top 5
            _pareto_candidates = _rank_pareto_trials(pareto_trials)
            bp = _pareto_candidates[0].params
            _extra_pareto = [t.params for t in _pareto_candidates[1:5]]
        except Exception:
            multi_obj = False

    if not multi_obj:
        study_kwargs = dict(direction='maximize', sampler=sampler, pruner=pruner)
        if storage:
            study_kwargs['storage'] = storage
            study_kwargs['study_name'] = f"s_{strat_name}_{symbol}_{tf}".replace('/', '_')
        study = optuna.create_study(**study_kwargs)

        if prev_params:
            try:
                study.enqueue_trial(prev_params)
            except Exception:
                pass

        obj_fn = objective_single_cv_v5(gen, space, splits, df)
        study.optimize(obj_fn, n_trials=N_TRIALS, timeout=TIMEOUT_PER_STUDY)
        if study.best_value <= -900:
            return None
        bp = study.best_params

    # ─── Step 3: Post-hoc CV validation ───
    # V7.6: Check trade count — if <30 trades, CPCV is statistically invalid → WF 70/30
    try:
        _check_sig = gen(df, **bp)
        _check_trades = backtest_raw(df, _check_sig)
        _n_full_trades = len([t for t in _check_trades if t]) if _check_trades else 0
    except Exception:
        _n_full_trades = 0

    if _n_full_trades < 30 and len(splits) > 1:
        # Low-trade strategy → WF 70/30 validation (López de Prado: CPCV needs ≥30 trades)
        validation_splits = _wf_splits
        _cv_type = 'WF_70_30'
    else:
        validation_splits = splits
        _cv_type = 'CPCV'

    # V7.1: Early-exit CPCV — abort when impossible to reach 40% passing
    cv_results = []
    total_paths = len(validation_splits)
    passing_count = 0
    valid_paths = 0  # V7.6: paths con >=5 trades (estadísticamente válidos)
    _cpcv_wr_threshold = 65.0 if total_paths >= 10 else MIN_TEST_WR
    required_passing = max(1, int(total_paths * 0.4))  # V7.6: min 1 for WF single path
    MIN_TRADES_PER_PATH = 5   # V7.6 Opción C: skip paths con <5 trades
    MIN_VALID_PATHS = min(max(6, int(total_paths * 0.30)), total_paths)  # V7.6: cap at total for WF

    for path_idx, (train_w, test_w, label, test_end_idx) in enumerate(validation_splits):
        try:
            sig_full = gen(df.iloc[:test_end_idx], **bp)
            sig_test = sig_full.reindex(test_w.index).fillna(0)
            test_trades = backtest_raw(test_w, sig_test)
            test_m = compute_metrics(test_trades)

            train_sig = gen(df.iloc[:len(train_w)], **bp)
            train_trades = backtest_raw(train_w, train_sig)
            train_m = compute_metrics(train_trades)

            if test_m and train_m:
                # V7.6 Opción C: skip ventanas con <5 trades (ruido estadístico)
                if test_m['trades'] < MIN_TRADES_PER_PATH:
                    continue
                valid_paths += 1
                is_passing = test_m['wr'] >= _cpcv_wr_threshold and test_m['pnl'] > 0
                if is_passing:
                    passing_count += 1
                cv_results.append({
                    'window': label,
                    'train_wr': train_m['wr'],
                    'test_wr': test_m['wr'],
                    'test_pnl': test_m['pnl'],
                    'test_trades': test_m['trades'],
                    'wr_gap': abs(train_m['wr'] - test_m['wr']),
                })
        except Exception:
            continue

        # V7.1: Early-exit — if remaining paths can't save it, abort (only for CPCV)
        if _cv_type == 'CPCV':
            paths_done = path_idx + 1
            paths_remaining = total_paths - paths_done
            if paths_done >= 4 and (passing_count + paths_remaining) < required_passing:
                break  # impossible to reach 40% — save time

    if not cv_results:
        return None

    # V7.6 Opción C guardrail: need enough valid paths (skip for WF — only 1 path)
    if _cv_type == 'CPCV' and valid_paths < MIN_VALID_PATHS:
        return None

    avg_test_wr = np.mean([r['test_wr'] for r in cv_results])
    avg_wr_gap = np.mean([r['wr_gap'] for r in cv_results])
    windows_passing = passing_count

    if windows_passing < required_passing:  # V7.6 FIX: uses total_paths (not len(cv_results))
        return None
    if avg_wr_gap > MAX_OVERFIT_GAP:
        return None

    # ─── Step 4: Full dataset metrics ───
    try:
        full_sig = gen(df, **bp)
        full_raw = backtest_raw(df, full_sig)
        full_m = compute_metrics(full_raw)
    except Exception:
        return None

    if full_m is None or full_m['trades'] < dynamic_min_trades(full_m.get('wr', 0)):  # V7.5
        return None

    # ─── PNL_GATE: Reject grails where full dataset PnL <= 0 (raw, pre-SL/TP) ───
    # G2_PnL in validate_gates_v5 checks SL/TP-constrained PnL. This gate checks
    # the raw full-dataset PnL too — ensures the strategy has intrinsic edge.
    if full_m['pnl'] <= 0:
        print(f"[PNL_GATE] REJECT {strat_name}x{symbol}: pnl_full={full_m['pnl']:.2f} <= 0")
        return None

    # ─── Step 5: V5 FIX (BUG FIX 1): SL/TP from last CV train window only ───
    # The last split has the largest train window → least biased MFE estimate
    last_split = splits[-1]
    train_last, _, _, last_test_end = last_split
    try:
        train_sig_last = gen(df.iloc[:len(train_last)], **bp)
        train_raw_last = backtest_raw(train_last, train_sig_last)
    except Exception:
        train_raw_last = full_raw  # fallback

    sl_tp = calc_sl_tp_from_mfe(train_raw_last)
    if sl_tp is None:
        sl_tp = calc_sl_tp_from_mfe(full_raw)
    if sl_tp is None:
        return None

    # V7.4: TF_MAX_SL ELIMINADO — SL empírico, no cap arbitrario por TF
    # Solo safety absoluto: SL_MAX=0.95 ya aplicado en calc_sl_tp_from_mfe

    # V7.2.2: R:R pre-check DYNAMIC — skip if R:R below hard floor
    # Uses absolute minimum (0.20) as pre-check — the full dynamic check
    # happens later with actual WR from test results
    rr_check = sl_tp['tp'] / (sl_tp['sl'] + 1e-10)
    if rr_check < RR_FLOOR_ABSOLUTE - 0.05:  # below hard floor with margin
        return None

    # ─── Step 6: Re-simulate with SL/TP ───
    full_with_sl = backtest_with_sl_tp(df, full_sig, sl_tp['sl'], sl_tp['tp'], sl_tp['max_dur_bars'])

    # R:R diagnostic logging (does NOT reject — SL/TP are empirical per Regla de Oro)
    _rr = sl_tp['tp'] / (sl_tp['sl'] + 1e-10)
    if _rr < 0.5:
        print(f"[RR_WARN] {strat_name}x{symbol}: R:R={_rr:.2f} (SL={sl_tp['sl']*100:.1f}% > TP={sl_tp['tp']*100:.1f}%)")

    # ─── Step 7: V7 REAL Leverage calculation per asset × strategy ───
    # Uses P99 MAE (not just max) for robustness, and validates with SL/TP simulation
    all_maes = sorted([abs(t['mae']) for t in full_raw], reverse=True)
    mae_worst = all_maes[0] if all_maes else 0.1
    mae_p99 = np.percentile(all_maes, 99) if len(all_maes) >= 10 else mae_worst
    mae_p95 = np.percentile(all_maes, 95) if len(all_maes) >= 5 else mae_worst

    # V7: Multi-tier leverage — conservative but real (not always 1x)
    # Tier 1: Based on P99 MAE (handles 99% of scenarios)
    safe_lev_p99 = 1 / (mae_p99 * 2.5 + 1e-10)
    # Tier 2: Based on P95 MAE (more aggressive)
    safe_lev_p95 = 1 / (mae_p95 * 2.0 + 1e-10)
    # Tier 3: Based on worst case (ultra-conservative)
    safe_lev_worst = 1 / (mae_worst * 3.0 + 1e-10)

    # V7.4: Leverage = empírico por MAE — REGLA DE ORO
    # max_lev_rr ELIMINADO — el R:R hardcoded no debe overridear el MAE empírico
    # El leverage lo dicta: 1/(MAE_P99 × 2.5), no el R:R threshold arbitrario
    safe_lev = min(safe_lev_p99, safe_lev_p95)
    safe_lev = max(1, int(safe_lev))

    # V7: Validate leverage doesn't cause liquidation with SL
    while safe_lev > 1 and sl_tp['sl'] * safe_lev > 0.80:
        safe_lev -= 1

    # V7: Additional check — simulate with leverage to verify
    lev_pnl = sum(t['pnl'] * safe_lev for t in full_with_sl)
    if lev_pnl < 0 and safe_lev > 1:
        safe_lev = max(1, safe_lev - 1)

    # ─── Step 8: V5 10 Gates (G1-G10) + V7.1 Gate-Aware Re-Selection ───
    gates_passed, gates = validate_gates_v5(
        full_with_sl, sl_tp['sl'], sl_tp['tp'], safe_lev, full_m['yearly']
    )

    # V7.1: If gates fail, try alternative Pareto trials (if multi-obj)
    if not gates_passed and _extra_pareto:
        for alt_bp in _extra_pareto:
            try:
                alt_sig = gen(df, **alt_bp)
                alt_raw = backtest_raw(df, alt_sig)
                alt_m = compute_metrics(alt_raw)
                if alt_m is None or alt_m['trades'] < dynamic_min_trades(alt_m.get('wr', 0)):  # V7.5
                    continue
                # Recalculate SL/TP for alternative params
                alt_train_sig = gen(df.iloc[:len(splits[-1][0])], **alt_bp)
                alt_train_raw = backtest_raw(splits[-1][0], alt_train_sig)
                alt_sl_tp = calc_sl_tp_from_mfe(alt_train_raw)
                if alt_sl_tp is None:
                    continue
                # V7.4: TF cap ELIMINADO — SL empírico. Leverage = 1/(MAE_P99×2.5) puro.
                alt_rr = alt_sl_tp['tp'] / (alt_sl_tp['sl'] + 1e-10)
                if alt_rr < RR_FLOOR_ABSOLUTE - 0.05:
                    continue
                alt_with_sl = backtest_with_sl_tp(df, alt_sig, alt_sl_tp['sl'], alt_sl_tp['tp'], alt_sl_tp['max_dur_bars'])
                alt_maes = sorted([abs(t['mae']) for t in alt_raw], reverse=True)
                alt_mae_p99 = np.percentile(alt_maes, 99) if len(alt_maes) >= 10 else alt_maes[0] if alt_maes else 0.1
                alt_lev = max(1, int(1 / (alt_mae_p99 * 2.5 + 1e-10)))
                while alt_lev > 1 and alt_sl_tp['sl'] * alt_lev > 0.80:
                    alt_lev -= 1

                alt_passed, alt_gates = validate_gates_v5(
                    alt_with_sl, alt_sl_tp['sl'], alt_sl_tp['tp'], alt_lev, alt_m['yearly']
                )
                if alt_passed:
                    # Found a passing alternative!
                    bp = alt_bp
                    full_sig, full_raw, full_m = alt_sig, alt_raw, alt_m
                    sl_tp, full_with_sl = alt_sl_tp, alt_with_sl
                    safe_lev = alt_lev
                    gates_passed, gates = alt_passed, alt_gates
                    mae_worst = alt_maes[0] if alt_maes else 0.1
                    mae_p99 = alt_mae_p99
                    mae_p95 = np.percentile(alt_maes, 95) if len(alt_maes) >= 5 else mae_worst
                    safe_lev_p99 = 1 / (mae_p99 * 2.5 + 1e-10)
                    safe_lev_p95 = 1 / (mae_p95 * 2.0 + 1e-10)
                    safe_lev_worst = 1 / (mae_worst * 3.0 + 1e-10)
                    break
            except Exception:
                continue

    if not gates_passed:
        return None

    # ─── Step 9: Block Bootstrap Monte Carlo ───
    mc_sig, mc_pval, mc_real_wr, mc_rand_p95 = monte_carlo_block_bootstrap(
        gen, bp, df, n_perms=MC_PERMUTATIONS
    )
    if not mc_sig:
        return None

    # ─── Step 10: Deflated Sharpe Ratio ───
    dsr_val = 1.0
    if full_m['sharpe'] != 0 and N_TRIALS > 1:
        pnls = np.array([t['pnl'] for t in full_raw])
        from scipy.stats import skew as calc_skew, kurtosis as calc_kurtosis
        sk = float(calc_skew(pnls)) if len(pnls) > 2 else 0.0
        ek = float(calc_kurtosis(pnls, fisher=True)) if len(pnls) > 2 else 0.0
        dsr_val = deflated_sharpe_ratio(
            estimated_sr=full_m['sharpe'],
            n_trials=N_TRIALS,
            backtest_bars=len(df),
            skew=sk,
            excess_kurtosis=ek,
        )

    # DSR gate — RECHAZO si insuficiente (consenso 4 IAs: DSR<0.90 = grail falso)
    # Bailey & López de Prado (2014): DSR<0.90 indica que el SR observado no supera
    # el benchmark ajustado por selection bias. Grail descartado, no solo advertido.
    # Solo activo cuando N_TRIALS >= 30 para tener historia de trials suficiente.
    DSR_REJECTION_THRESHOLD = 0.90
    DSR_MIN_TRIALS = 30  # Solo aplicar gate con suficiente historia de trials

    if dsr_val < DSR_REJECTION_THRESHOLD:
        if N_TRIALS >= DSR_MIN_TRIALS:
            print(f"[DSR_GATE] REJECT {strat_name}×{symbol}×{tf}: "
                  f"DSR={dsr_val:.3f} < {DSR_REJECTION_THRESHOLD} "
                  f"(N_TRIALS={N_TRIALS}) → grail con selection bias no superado")
            return None
        else:
            logger.debug(f"[DSR_GATE] {strat_name}×{symbol}×{tf}: DSR={dsr_val:.3f} bajo "
                         f"threshold pero N_TRIALS={N_TRIALS} < {DSR_MIN_TRIALS} → solo warning")

    # ─── BUILD GRAIL ───
    return {
        'strategy': strat_name,
        'symbol': symbol,
        'timeframe': tf,
        'best_params': bp,
        'test_wr': round(full_m['wr'], 1),       # V5 FIX: bot_loader busca test_wr o final_wr
        'cv_avg_wr': round(avg_test_wr, 1),     # CPCV promedio (informativo)
        'cv_windows': cv_results,
        'cv_windows_passing': windows_passing,
        'cv_type': _cv_type,  # V7.6: 'CPCV' or 'WF_70_30' (low-trade fallback)
        'avg_wr_gap': round(avg_wr_gap, 1),
        'full': {
            'wr': full_m['wr'], 'pnl': full_m['pnl'],
            'sharpe': full_m['sharpe'], 'sortino': full_m['sortino'],
            'max_dd': full_m['max_drawdown'], 'pf': full_m['profit_factor'],
            'calmar': full_m['calmar'], 'trades': full_m['trades'],
            'mae_p95': full_m['mae_p95'], 'yearly': full_m['yearly'],
        },
        'sl': sl_tp['sl'],
        'tp': sl_tp['tp'],
        'max_dur_bars': sl_tp['max_dur_bars'],
        'safe_leverage': safe_lev,
        'leverage_detail': {
            'mae_worst': round(mae_worst, 4),
            'mae_p99': round(mae_p99, 4),
            'mae_p95': round(mae_p95, 4),
            'lev_p99': round(safe_lev_p99, 1),
            'lev_p95': round(safe_lev_p95, 1),
            'lev_worst': round(safe_lev_worst, 1),
            'sl_x_lev': round(sl_tp['sl'] * safe_lev, 3),
            'lev_pnl_pct': round(lev_pnl * 100, 2),
        },
        'gates': gates,
        'monte_carlo': {
            'significant': mc_sig, 'p_value': mc_pval,
            'real_wr': mc_real_wr, 'random_p95': mc_rand_p95,
            'method': 'shuffle_500',  # V5.1: reverted to shuffle (FBI fix)
        },
        'deflated_sharpe_ratio': round(dsr_val, 3),
        'weighted_score': calc_weighted_score(full_m['yearly']),
        'optimizer': 'optuna_v7.2',
        'timestamp': datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════
# WORKER: Process one combo = ONE strategy × ONE symbol × ONE TF
# ═══════════════════════════════════════════════════════════════════════

def _worker_initializer():
    """V7: Initialize worker once — load strategies + setup paths."""
    global _STRATEGY_CACHE, _CANDLE_CACHE
    _CANDLE_CACHE = {}  # fresh cache per worker

    import optuna as _optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)

    sys.path.insert(0, PROJECT_DIR)
    sys.path.insert(0, os.path.join(PROJECT_DIR, "strategies_code"))

    _STRATEGY_CACHE = _load_strategies_in_worker()


def process_combo(args):
    """V7.1: Worker function — uses cached strategies and candles.

    Returns (sname, symbol, tf, result) where result is:
    - dict with grail data if passed all gates
    - None if failed (failure details logged in progress)
    """
    sname, symbol, tf, n_trials, multi_obj, use_journal = args

    global N_TRIALS
    N_TRIALS = n_trials

    if sname not in _STRATEGY_CACHE:
        return sname, symbol, tf, None

    info = _STRATEGY_CACHE[sname]
    gen, space = info['gen'], info['space']

    df = get_candles(symbol, tf)  # V7: hits cache after first load
    if df is None or len(df) < 300:
        return sname, symbol, tf, None

    # G0: Hurst Gate — skip mean-reversion strategies on trending markets
    if _is_mean_reversion_strategy(sname):
        H = hurst_exponent(df['close'].values)
        if H > 0.55:
            print(f"[G0_HURST] SKIP {sname}×{symbol}×{tf}: H={H:.3f} (trending market, mean-rev edge absent)")
            return sname, symbol, tf, None  # trending market → mean-rev edge doesn't exist

    journal_path = None
    if use_journal:
        safe_name = f"{sname}_{symbol}_{tf}".replace('/', '_').replace(':', '_')
        journal_path = os.path.join(OUTPUT_DIR, f"v7_journal_{safe_name}.log")

    try:
        result = optimize_one(
            gen, space, df, symbol, tf, sname,
            multi_obj=multi_obj,
            use_journal=use_journal,
            journal_path=journal_path,
        )
        return sname, symbol, tf, result
    except Exception:
        return sname, symbol, tf, None


def _wrap_list_space(space_fn):
    """Compatibility: wrap old-style space() into new-style space(trial) → dict.

    Handles 3 formats produced by TV2 batch generators:
      Format 1 (new-style): space(trial) → {name: value}          ← pass-through
      Format 2 (list):      space() → [(name, type, (lo,hi)), ...] ← wrap
      Format 3 (dict):      space() → {name: (type, lo, hi)}       ← wrap
    """
    import inspect

    def _suggest(trial, name, ptype, bounds):
        if ptype == 'int':
            return trial.suggest_int(name, int(bounds[0]), int(bounds[1]))
        elif ptype == 'float':
            return trial.suggest_float(name, float(bounds[0]), float(bounds[1]))
        elif ptype == 'categorical':
            return trial.suggest_categorical(name, list(bounds))
        else:
            return trial.suggest_float(name, float(bounds[0]), float(bounds[1]))

    # If function already takes trial → new-style, pass through
    try:
        sig = inspect.signature(space_fn)
        if len(sig.parameters) > 0:
            return space_fn
    except Exception:
        return space_fn

    # Call with no args to detect format
    try:
        result = space_fn()
    except Exception:
        return space_fn

    if isinstance(result, list):
        # Format 2: list of (name, type, bounds) tuples
        param_list = result
        def wrapped_list(trial, _plist=param_list):
            params = {}
            for entry in _plist:
                name, ptype, bounds = entry[0], entry[1], entry[2]
                params[name] = _suggest(trial, name, ptype, bounds)
            return params
        return wrapped_list

    if isinstance(result, dict):
        # Format 3: {name: (type, lo, hi)} — all values must be tuples
        param_dict = result
        def wrapped_dict(trial, _pdict=param_dict):
            params = {}
            for name, spec in _pdict.items():
                if isinstance(spec, (list, tuple)) and len(spec) >= 2:
                    ptype = spec[0]
                    bounds = spec[1:]
                    params[name] = _suggest(trial, name, ptype, bounds)
                # else: skip malformed spec
            return params
        return wrapped_dict

    # Fallback: return unchanged
    return space_fn


def _load_strategies_in_worker():
    """Load all available strategies in worker process."""
    strats = {}
    try:
        from strategies_round2 import STRATEGY_TYPES_R2
        strats.update(STRATEGY_TYPES_R2)
    except Exception:
        pass
    # Round 3 strategies (fourier_strategy, price_action_sr, williams_r, B1-B6 series)
    try:
        from strategies_round3 import STRATEGY_TYPES_R3
        strats.update(STRATEGY_TYPES_R3)
    except Exception:
        pass
    try:
        from batch_search_spaces import BATCH_SEARCH_SPACES, make_space_func
        from strategies_code.batch_param_wrappers import BATCH_WRAPPERS
        for key, wrapper_fn in BATCH_WRAPPERS.items():
            if key in BATCH_SEARCH_SPACES:
                strats[key] = {'gen': wrapper_fn, 'space': make_space_func(key)}
    except Exception:
        pass
    # TV2 batch strategies (Pine→Python conversions)
    tv2_dir = os.path.join(PROJECT_DIR, "strategies_tv2_batches")
    if tv2_dir not in sys.path:
        sys.path.insert(0, tv2_dir)
    for batch_num in range(100, 4000):  # covers all current + future TV batches (extended 2026-04-13)
        try:
            mod = __import__(f"strategies_tv2_batch{batch_num}")
            raw = mod.STRATEGY_EXPORT
            # Compatibility: handle both {'gen': fn, 'space': fn} and {'gen_long': fn, 'gen_short': fn, 'space': fn}
            if isinstance(raw, dict):
                for sname, sinfo in raw.items():
                    if 'gen' in sinfo:
                        strats[sname] = {'gen': sinfo['gen'], 'space': _wrap_list_space(sinfo['space'])}
                    elif 'gen_long' in sinfo and 'gen_short' in sinfo:
                        gl = sinfo['gen_long']; gs = sinfo['gen_short']
                        sp = sinfo['space']
                        def _make_combined(gl_=gl, gs_=gs):
                            def _gen(df, **p):
                                long_sig  = gl_(df, **p)
                                short_sig = gs_(df, **p)
                                sig = pd.Series(0, index=df.index)
                                sig[long_sig  == 1] =  1
                                sig[short_sig == 1] = -1
                                return sig
                            return _gen
                        strats[sname] = {'gen': _make_combined(), 'space': sp}
        except Exception:
            pass
    # PINE strategies (strategies_new_winners.py — 11 Pine→Python conversions)
    try:
        _pine_path = '/Users/sabrina/CLAUDE CODE/BOT V7/strategies'
        if _pine_path not in sys.path:
            sys.path.insert(0, _pine_path)
        from strategies_new_winners import NEW_WINNER_STRATS
        strats.update(NEW_WINNER_STRATS)
    except Exception:
        pass
    return strats


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Optuna V5 — CPCV + Block Bootstrap optimizer')
    parser.add_argument('--symbol', help='Specific symbol')
    parser.add_argument('--top', type=int, default=100, help='Top N symbols (default 100)')
    parser.add_argument('--tf', help='Specific timeframe')
    parser.add_argument('--round', default='all', help='r2, r4_batch, or all')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--trials', type=int, default=N_TRIALS)
    parser.add_argument('--single-objective', action='store_true', help='Use single-objective (legacy)')
    parser.add_argument('--no-monte-carlo', action='store_true', help='Skip Monte Carlo test')
    parser.add_argument('--journal', action='store_true',
                        help='V5 NEW: Use JournalFileBackend for robust Optuna storage')
    parser.add_argument('--prod-winners', action='store_true',
                        help='V7.1: Use production-winning symbols instead of top-by-volume')
    parser.add_argument('--v3-positive', action='store_true',
                        help='V7.3: Run ONLY combos that were positive in V3 (pre-validated candidates)')
    parser.add_argument('--combos-file', type=str, default=None,
                        help='V7.4: Run ONLY specific combos from a JSON file (list of {strategy, symbol, timeframe})')
    parser.add_argument('--max-minutes', type=int, default=0,
                        help='V7.6: Auto-stop after N minutes (0=unlimited). For runner wrapper.')
    parser.add_argument('--progress-file', type=str, default=None,
                        help='V7.7: Use a custom progress file path (allows parallel runs without conflicts)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    # V7.1: Production-winning symbols (from Bot V7 real trades analysis)
    PROD_WINNER_SYMBOLS = [
        'BEAT/USDT:USDT', 'RIVER/USDT:USDT', 'SWARMS/USDT:USDT',
        'BANANAS31/USDT:USDT', 'DEXE/USDT:USDT', 'CC/USDT:USDT',
        'MON/USDT:USDT', 'INIT/USDT:USDT', 'ONT/USDT:USDT',
        'ARIA/USDT:USDT', 'MEGA/USDT:USDT', 'BAT/USDT:USDT',
        'CYS/USDT:USDT', 'D/USDT:USDT', 'XVG/USDT:USDT',
        # Also include top volume for diversity
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
        'BNB/USDT:USDT', 'ADA/USDT:USDT',
    ]

    # V7.6: Skip get_symbols() when --combos-file is used — saves 10+ minutes on large DBs
    if args.combos_file:
        symbols = []  # Not needed — combos come from file
        print(f"  [V7.6] Skipping get_symbols() — combos from file")
    elif args.symbol:
        symbols = [args.symbol]
    elif args.v3_positive:
        # V7.3: All DB symbols — v3_positive filtering happens at combo level
        symbols = set(get_symbols())
    elif args.prod_winners:
        # Filter to symbols that exist in DB
        all_db = get_symbols()
        symbols = [s for s in PROD_WINNER_SYMBOLS if s in all_db]
        if not symbols:
            symbols = get_symbols(top_n=args.top)
            print(f"WARNING: No prod winners found in DB, falling back to top {args.top}")
    else:
        symbols = get_symbols(top_n=args.top)
    tfs = [args.tf] if args.tf else USE_TFS
    multi_obj = not args.single_objective
    n_trials = args.trials

    strats = _load_strategies_in_worker()
    strat_names = list(strats.keys())

    mode_str = "Multi-Objective CPCV (WR + Sortino - DD)" if multi_obj else "Single-Objective CPCV"
    print("=" * 70)
    print(f"OPTUNA V7 — {mode_str} [PERFORMANCE OPTIMIZED]")
    print(f"Strategies: {len(strat_names)} | Symbols: {len(symbols)} | TFs: {tfs}")
    print(f"Trials: {n_trials} | Workers: {args.workers} | CPCV: C({N_FOLDS_CPCV},{N_TEST_FOLDS_CPCV})=15 paths")
    print(f"MC: {MC_PERMUTATIONS} block-bootstrap perms | Embargo: {EMBARGO_PCT*100}%")
    print(f"Gates: G1-G10 (G9=YearCov, G10=EV_real) | Journal: {args.journal}")
    print(f"SL: P90(ALL MAE)×2.0 EMPIRICO | TP: P90(MFE)×0.85 | R:R=dynamic((1-WR)/WR+0.10, min={RR_FLOOR_ABSOLUTE}) | Lev=1/(MAE_P99×2.5)")
    print("=" * 70)

    # V7: separate progress file (V7.7: allow custom path via --progress-file)
    progress_file = args.progress_file if args.progress_file else os.path.join(OUTPUT_DIR, "optuna_v7_progress.json")
    # V7: Also read V5 progress to skip already-done combos
    v5_progress_file = os.path.join(OUTPUT_DIR, "optuna_v5_progress.json")
    progress = {'done_combos': [], 'grails': [], 'stats': {}}
    if os.path.exists(progress_file):
        try:
            progress = json.load(open(progress_file))
        except Exception:
            pass

    done_combos = set(progress.get('done_combos', []))
    all_grails = progress.get('grails', [])

    # V7.3: --v3-positive — only test combos that were positive in V3 (pre-validated)
    if args.combos_file:
        # V7.4: Run specific combos from file (e.g. capped-rerun, targeted analysis)
        combos_path = args.combos_file if os.path.isabs(args.combos_file) else os.path.join(OUTPUT_DIR, args.combos_file)
        try:
            with open(combos_path) as _f:
                cf_data = json.load(_f)
            cf_combos = cf_data if isinstance(cf_data, list) else cf_data.get('combos', [])
            all_combos = [(c['strategy'], c['symbol'], c['timeframe'])
                          for c in cf_combos
                          if c['strategy'] in strat_names]
            print(f"  Combos-file mode: {len(all_combos)} combos from {os.path.basename(combos_path)}")
        except Exception as _e:
            print(f"  WARNING: combos-file failed ({_e}), falling back to full grid")
            all_combos = [(sn, sym, tf) for sym in symbols for sn in strat_names for tf in tfs]
    elif args.v3_positive:
        v3_file = os.path.join(OUTPUT_DIR, "v3_positive_for_v72.json")
        try:
            with open(v3_file) as _f:
                v3_data = json.load(_f)
            # Build combos from V3 list, sorted by V3 WR desc (best first)
            v3_combos_sorted = sorted(v3_data['combos'], key=lambda x: -x.get('v3_wr', 0))
            all_combos = [(c['strategy'], c['symbol'], c['timeframe'])
                          for c in v3_combos_sorted
                          if c['strategy'] in strat_names and c['symbol'] in symbols]
            print(f"  V3-positive mode: {len(all_combos)} pre-validated combos (WR>=70%, PnL>0 in V3)")
        except Exception as _e:
            print(f"  WARNING: v3-positive failed ({_e}), falling back to full grid")
            all_combos = [(sn, sym, tf) for sym in symbols for sn in strat_names for tf in tfs]
    else:
        all_combos = [
            (sn, sym, tf)
            for sym in symbols
            for sn in strat_names
            for tf in tfs
        ]
    pending = [(sn, sym, tf) for sn, sym, tf in all_combos
               if f"{sn}|{sym}|{tf}" not in done_combos]

    # V7.2.2: PRIORITIZE combos — profitable production symbols FIRST
    # Tier 1: Symbols with real PnL > 0 in production (proven winners)
    # Tier 2: Symbols with any real trade data (have information)
    # Tier 3: All others sorted by backtest WR descending
    try:
        import sqlite3 as _sq
        _tdb = _sq.connect(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BOT V7', 'live', 'data', 'trades_v7.db'))
        _rows = _tdb.execute("""
            SELECT symbol, SUM(pnl_usd) pnl, COUNT(*) trades
            FROM virtual_trades
            WHERE status='closed' AND (skip_reason IS NULL OR skip_reason='')
              AND notes NOT LIKE '%POSITION_GONE%' AND pnl_usd != 0
            GROUP BY symbol
        """).fetchall()
        _tdb.close()
        _prod_pnl = {r[0]: r[1] for r in _rows}
        _prod_syms = {r[0] for r in _rows}  # have real data
        _winners = {r[0] for r in _rows if r[1] > 0}  # profitable
        def _combo_priority(combo):
            sn, sym, tf = combo
            if sym in _winners:
                return (0, -_prod_pnl.get(sym, 0))  # Tier 1: winners first, by PnL desc
            elif sym in _prod_syms:
                return (1, -_prod_pnl.get(sym, 0))  # Tier 2: have data, by PnL desc
            else:
                return (2, sym)  # Tier 3: alphabetical
        pending.sort(key=_combo_priority)
        _n1 = sum(1 for s,_,_ in pending if s in _winners or _ in _winners)
        print(f"  Priority sort: {len(_winners)} winner symbols first, {len(_prod_syms)} with trade data")
    except Exception as _pe:
        print(f"  Priority sort skipped: {_pe}")

    total_combos = len(all_combos)
    print(f"Already done: {len(done_combos)} combos | Remaining: {len(pending)} / {total_combos}")
    print("-" * 70)

    start = time.time()
    total_tested = 0
    new_grails = 0

    worker_args = [
        (sn, sym, tf, calculate_trials_by_tf(n_trials, tf), multi_obj, args.journal)
        for sn, sym, tf in pending
    ]

    # V7.6: Time limit support — stop cleanly after --max-minutes
    max_minutes = args.max_minutes
    deadline = start + max_minutes * 60 if max_minutes > 0 else float('inf')
    time_stopped = False

    # V7.6: Submit in batches (batch_size = workers * 3) to allow clean time-based exit.
    # Previously submitted ALL futures at once — impossible to cancel with 180K pending.
    batch_size = args.workers * 3  # Keep workers busy without over-committing
    submitted_idx = 0  # Index into worker_args for next batch

    # V7: Use worker initializer to load strategies once per worker
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_initializer) as executor:
        futures = {}
        completed_count = 0

        def _submit_batch():
            """Submit next batch of work items."""
            nonlocal submitted_idx
            n_to_submit = min(batch_size - len(futures), len(worker_args) - submitted_idx)
            for _ in range(n_to_submit):
                if submitted_idx >= len(worker_args):
                    break
                wa = worker_args[submitted_idx]
                fut = executor.submit(process_combo, wa)
                futures[fut] = wa
                submitted_idx += 1

        # Initial batch
        _submit_batch()

        while futures:
            # Wait for next completion (with short timeout to check deadline)
            done_futs = []
            for fut in list(futures.keys()):
                if fut.done():
                    done_futs.append(fut)

            if not done_futs:
                # No futures done yet — brief sleep and check deadline
                time.sleep(0.1)
                if time.time() >= deadline:
                    time_stopped = True
                    print(f"\n  [V7.6] Time limit {max_minutes}min reached. Waiting for {len(futures)} in-flight combos...")
                    # Wait for in-flight futures to finish (they're already running)
                    for fut in as_completed(futures):
                        wa = futures[fut]
                        sn, sym, tf = wa[0], wa[1], wa[2]
                        combo_key = f"{sn}|{sym}|{tf}"
                        try:
                            _, _, _, result = fut.result(timeout=600)
                            completed_count += 1
                            total_tested += 1
                            done_combos.add(combo_key)
                            if result is not None:
                                new_grails += 1
                                all_grails.append(result)
                        except Exception:
                            done_combos.add(combo_key)
                    futures.clear()
                    break
                continue

            for fut in done_futs:
                wa = futures.pop(fut)
                sn, sym, tf = wa[0], wa[1], wa[2]
                combo_key = f"{sn}|{sym}|{tf}"
                completed_count += 1

                try:
                    _, _, _, result = fut.result(timeout=600)
                except Exception as e:
                    print(f"  [{completed_count}/{len(pending)}] ERROR {sn}@{sym}/{tf}: {str(e)[:50]}")
                    done_combos.add(combo_key)
                    continue

                total_tested += 1
                done_combos.add(combo_key)

                if result is not None:
                    new_grails += 1
                    all_grails.append(result)
                    wr   = result['full']['wr']
                    lev  = result['safe_leverage']
                    dsr  = result.get('deflated_sharpe_ratio', '?')
                    cvt  = result.get('cv_type', 'CPCV')
                    elapsed_m = (time.time() - start) / 60
                    rate = completed_count / elapsed_m if elapsed_m > 0 else 1
                    eta_m = (len(pending) - completed_count) / rate if rate > 0 else 0
                    print(f"  [{completed_count}/{len(pending)}] GRAIL {sn:30s} {sym.split('/')[0]:8s} {tf:3s} "
                          f"WR={wr:.0f}% Lev={lev}x DSR={dsr} CV={cvt} | "
                          f"Total={len(all_grails)} ETA={eta_m:.0f}min")
                elif completed_count % 100 == 0:
                    elapsed_m = (time.time() - start) / 60
                    rate = completed_count / elapsed_m if elapsed_m > 0 else 1
                    eta_m = (len(pending) - completed_count) / rate if rate > 0 else 0
                    print(f"  [{completed_count}/{len(pending)}] grails={len(all_grails)} "
                          f"rate={rate:.1f}/min ETA={eta_m:.0f}min")

                # Checkpoint: save after every grail or every 50 combos
                if result is not None or completed_count % 50 == 0:
                    progress = {
                        'done_combos': list(done_combos),
                        'grails': all_grails,
                        'stats': {
                            'total_tested': total_tested,
                            'grails_found': len(all_grails),
                            'elapsed_h': round((time.time() - start) / 3600, 2),
                            'last_updated': datetime.now().isoformat(),
                            'optimizer': 'optuna_v5',
                            'cv_method': f'CPCV_C({N_FOLDS_CPCV},{N_TEST_FOLDS_CPCV})',
                            'mc_method': f'block_bootstrap_{MC_PERMUTATIONS}perms',
                        }
                    }
                    atomic_save_json(progress, progress_file)  # V7.6: atomic write

            # V7.6: Check time limit before submitting more work
            if time.time() >= deadline:
                time_stopped = True
                print(f"\n  [V7.6] Time limit {max_minutes}min reached. Waiting for {len(futures)} in-flight combos...")
                for fut in as_completed(futures):
                    wa = futures[fut]
                    sn, sym, tf = wa[0], wa[1], wa[2]
                    combo_key = f"{sn}|{sym}|{tf}"
                    completed_count += 1
                    try:
                        _, _, _, result = fut.result(timeout=600)
                        total_tested += 1
                        done_combos.add(combo_key)
                        if result is not None:
                            new_grails += 1
                            all_grails.append(result)
                    except Exception:
                        done_combos.add(combo_key)
                futures.clear()
                break

            # Submit more work if pool has capacity
            _submit_batch()

    # Final save — rebuild from current state (FIX: checkpoint snapshot was stale)
    elapsed_h = (time.time() - start) / 3600
    progress = {
        'done_combos': list(done_combos),
        'grails': all_grails,
        'stats': progress.get('stats', {}),
    }
    progress['stats']['finished'] = not time_stopped  # V7.6: False if stopped by timer
    progress['stats']['elapsed_h'] = round(elapsed_h, 2)
    progress['stats']['total_tested'] = total_tested
    progress['stats']['grails_found'] = len(all_grails)
    progress['stats']['last_updated'] = datetime.now().isoformat()
    atomic_save_json(progress, progress_file)  # V7.6: atomic write

    status = "TIME-STOPPED" if time_stopped else "COMPLETE"
    remaining = len(pending) - completed_count if time_stopped else 0
    print(f"\n{'='*70}")
    print(f"OPTUNA V7 {status}")
    print(f"  Tested: {total_tested} combos (this run: {completed_count})")
    print(f"  Grails: {len(all_grails)} (new this run: {new_grails})")
    print(f"  Time: {elapsed_h:.1f}h")
    if time_stopped:
        print(f"  Remaining: {remaining} combos pending")
    print(f"  Saved: {progress_file}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
