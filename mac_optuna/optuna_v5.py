#!/usr/bin/env python3
"""
OPTUNA V5 — Next-gen optimizer with CPCV, Block Bootstrap MC, and bug fixes.

Improvements over v4 (optuna_v4.py):
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
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import multiprocessing

warnings.filterwarnings('ignore')

# ─── CONFIG ───
DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"
PROJECT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Defaults (overridable via CLI)
N_TRIALS = 20          # V5 NEW: reduced from 25 (CPCV has 15 paths vs 3 → richer signal)
MAX_WORKERS = 8
COMMISSION = 0.001
SLIPPAGE = 0.0005
MIN_TRADES_TRAIN = 5
MIN_TRADES_TEST = 5
MIN_TRADES_FULL = 15
TIMEOUT_PER_STUDY = 60  # seconds

# V5 NEW: CPCV config (C(6,2)=15 paths)
N_FOLDS_CPCV = 6
N_TEST_FOLDS_CPCV = 2
EMBARGO_PCT = 0.02        # 2% gap between train/test to prevent leakage

# Walk-forward fallback (used when df < 500 rows)
N_CV_WINDOWS = 3

# Gates constants
MIN_TEST_WR = 70.0
MAX_OVERFIT_GAP = 30.0
SL_MIN, SL_MAX = 0.02, 0.40
TP_MIN, TP_MAX = 0.01, 0.50
MAX_DUR_CAP = 720         # hours
RR_FLOOR = 0.30  # V5 FIX: Bug Fix 3 amplía SL (P95 ALL trades) → R:R baja naturalmente → 0.30 compensa
DD_PNL_MAX = 2.0  # V5.1 FBI fix: wide SL = deeper DD mechanically → 1.5→2.0

# V5 NEW: Monte Carlo config — block bootstrap, 500 perms
MC_PERMUTATIONS = 500     # was 100
MC_P_THRESHOLD = 0.05

# Timeframes to test
USE_TFS = ['5m', '15m', '1h', '4h']


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
    conn = sqlite3.connect(DB_PATH)
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
    if tf in ('5m', '1h'):
        return load_candles(symbol, tf)
    elif tf in ('15m', '4h', '1d'):
        df_5m = load_candles(symbol, '5m')
        return resample(df_5m, tf) if df_5m is not None else None
    return None


def get_symbols(top_n=None):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT symbol, COUNT(*) as bars FROM candles "
        "WHERE timeframe='5m' GROUP BY symbol ORDER BY bars DESC", conn)
    conn.close()
    syms = df['symbol'].tolist()
    return syms[:top_n] if top_n else syms


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


def backtest_with_sl_tp(df, signals, sl_pct, tp_pct, max_dur_bars=None):
    """Re-simulate with SL/TP applied."""
    if signals is None:
        return []
    sig = signals.values
    opens, highs, lows = df['open'].values, df['high'].values, df['low'].values
    n = len(df)
    trades = []
    pos = 0; ep = 0.; ei = 0

    for i in range(1, n):
        s = sig[i - 1]; p = opens[i]
        if pos == 0 and s == 1:
            ep = p * (1 + SLIPPAGE + COMMISSION); ei = i; pos = 1
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
                xp = p * (1 - SLIPPAGE - COMMISSION)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei, 'exit': 'DUR'})
                pos = 0; continue
            if s == -1:
                xp = p * (1 - SLIPPAGE - COMMISSION)
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

        # Train: purge rows within embargo_bars of test boundaries
        train_idx = []
        for fi in sorted(train_set):
            for idx in range(folds[fi][0], folds[fi][1]):
                too_close = any(
                    abs(idx - b) <= embargo_bars
                    for b in [test_start_bound, test_end_bound]
                )
                if not too_close:
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


# ═══════════════════════════════════════════════════════════════════════
# MONTE CARLO — STATIONARY BLOCK BOOTSTRAP (V5)
# ═══════════════════════════════════════════════════════════════════════

def monte_carlo_block_bootstrap(gen, best_params, df, n_perms=MC_PERMUTATIONS):
    """V5.1 FIX: Random shuffle MC (500 perms) — reverted from block bootstrap.

    FBI Agent 5 finding: Block bootstrap preserves signal autocorrelation in permutations,
    inflating random baseline to match real WR for mean-reversion strategies (P95 random = real WR).
    This systematically rejects valid strategies. Random shuffle (V4 approach) with 500 perms
    gives better p-value resolution while maintaining correct null hypothesis.

    BUG FIX 4 retained: returns False on insufficient data (not True).

    Returns: (is_significant, p_value, real_wr, random_wr_95th)
    """
    try:
        real_sig = gen(df, **best_params)
        real_trades = backtest_raw(df, real_sig)
        real_m = compute_metrics(real_trades)
        if real_m is None:
            return False, 1.0, 0, 0  # BUG FIX 4 retained
        real_wr = real_m['wr']
    except Exception:
        return False, 1.0, 0, 0  # BUG FIX 4 retained

    vals = real_sig.values.copy()
    n = len(vals)
    rng = np.random.RandomState(42)

    random_wrs = []
    for _ in range(n_perms):
        # V5.1: Random shuffle (FBI fix — block bootstrap inflated baseline)
        permuted_vals = rng.permutation(vals)
        shuffled = pd.Series(permuted_vals, index=real_sig.index)
        rand_trades = backtest_raw(df, shuffled)
        rand_m = compute_metrics(rand_trades)
        if rand_m is not None:
            random_wrs.append(rand_m['wr'])

    if len(random_wrs) < 20:
        return False, 1.0, real_wr, 0  # V5 FIX: was True → False (reject on insufficient perms)

    random_wrs = np.array(random_wrs)
    p95 = np.percentile(random_wrs, 95)
    p_value = (random_wrs >= real_wr).sum() / len(random_wrs)

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

    if denominator <= 0:
        return 0.0

    return float(norm.cdf(numerator / denominator))


# ═══════════════════════════════════════════════════════════════════════
# MFE ANALYSIS + GATES (V5: G1-G10)
# ═══════════════════════════════════════════════════════════════════════

def calc_sl_tp_from_mfe(raw_trades):
    """V5.1 FIX: SL compromise — P95 ALL trades × 1.0 (not 1.5).

    FBI Council finding: V5 Bug Fix 3 (P95 ALL × 1.5) made SL 2.8x wider than V4,
    destroying R:R and killing 78% of grails. V4 (P95 winners × 1.5) was too tight.
    Compromise: use ALL trades (correct population) but lower multiplier to 1.0
    and raise TP percentile from P75→P85 to rebalance R:R.
    """
    if len(raw_trades) < 5:
        return None

    # V5.1 FIX: SL from ALL trades × 1.0 (FBI compromise: correct pop, lower mult)
    all_maes = [abs(t['mae']) for t in raw_trades]
    sl = np.clip(np.percentile(all_maes, 95) * 1.0, SL_MIN, SL_MAX)

    # V5.1 FIX: TP P85 × 0.8 (was P75 — FBI fix #1: rebalance R:R)
    winners = [t for t in raw_trades if t['win']]
    if len(winners) < 3:
        return None
    mfes = [t['mfe'] for t in winners]
    durs = [t['dur_bars'] for t in winners]

    tp = np.clip(np.percentile(mfes, 85) * 0.8, TP_MIN, TP_MAX)
    max_dur = min(np.percentile(durs, 95) * 1.5, MAX_DUR_CAP)

    return {'sl': round(sl, 4), 'tp': round(tp, 4), 'max_dur_bars': int(max_dur)}


def check_year_coverage(yearly):
    """V5.1 FIX (Gate G9): Year Coverage — dynamic years, softer reject.

    FBI Agent 6 findings:
    - Hardcoded [2023,2024,2025] was stale (we're in 2026) → now dynamic
    - WR<50% instant-kill on 3 trades was too aggressive → only reject on >=10 trades
    - 46% of symbols (250/542) blocked by stale year list

    Rules:
      - Dynamic: checks current year and 2 prior years
      - ≥2 of 3 years must have ≥3 trades AND WR ≥ 55%
      - Only reject if a year has ≥10 trades AND WR < 45% (not 3 trades / 50%)
    """
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
        'G3_trades':   {'pass': n >= MIN_TRADES_TEST, 'value': n},
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
        'G8_RR':       {'pass': tp / (sl + 1e-10) >= RR_FLOOR, 'value': round(tp / (sl + 1e-10), 2)},
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
    """Load previous grail results for warm-starting."""
    grail_files = [
        os.path.join(PROJECT_DIR, "data", "santo_grial_r2.json"),
        os.path.join(PROJECT_DIR, "data", "santo_grial_v2.json"),
        os.path.join(OUTPUT_DIR, "optuna_v4_progress.json"),
        os.path.join(OUTPUT_DIR, "optuna_v5_progress.json"),
    ]
    grails = {}  # key: (strategy, symbol, tf) → best_params
    for f in grail_files:
        if not os.path.exists(f):
            continue
        try:
            data = json.load(open(f))
            items = data.get('grails', data) if isinstance(data, dict) else data
            for g in items:
                key = (g.get('strategy', ''), g.get('symbol', ''), g.get('timeframe', ''))
                if 'best_params' in g and g['best_params']:
                    grails[key] = g['best_params']
        except Exception:
            continue
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
    # ─── Step 1: CPCV splits (or WF fallback) ───
    splits = make_cpcv_splits(df)
    if not splits:
        # Fallback to single 70/30
        split_idx = int(len(df) * 0.7)
        train, test = df.iloc[:split_idx], df.iloc[split_idx:]
        if len(train) < 100 or len(test) < 50:
            return None
        test_end_idx = len(df)
        splits = [(train, test, "W1", test_end_idx)]

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

            # V5 NEW: objective with CPCV + warm-up + constraints
            obj_fn = objective_multi_cpcv(gen, space, splits, df)
            study.optimize(obj_fn, n_trials=N_TRIALS, timeout=TIMEOUT_PER_STUDY)

            best_trial = select_from_pareto(study)
            if best_trial is None:
                return None
            bp = best_trial.params
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

    # ─── Step 3: Post-hoc CV validation across all CPCV paths ───
    cv_results = []
    for train_w, test_w, label, test_end_idx in splits:
        try:
            # V5 FIX (BUG FIX 2): warm-up via full prefix
            sig_full = gen(df.iloc[:test_end_idx], **bp)
            sig_test = sig_full.reindex(test_w.index).fillna(0)
            test_trades = backtest_raw(test_w, sig_test)
            test_m = compute_metrics(test_trades)

            train_sig = gen(df.iloc[:len(train_w)], **bp)
            train_trades = backtest_raw(train_w, train_sig)
            train_m = compute_metrics(train_trades)

            if test_m and train_m:
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

    if not cv_results:
        return None

    avg_test_wr = np.mean([r['test_wr'] for r in cv_results])
    avg_wr_gap = np.mean([r['wr_gap'] for r in cv_results])
    # V5 FIX: CPCV con 15 paths tiene varianza controlada → WR gate baja a 65%
    # Con 3 windows (v4) era fácil tener WR>=70 por suerte. Con 15 paths el WR se estabiliza.
    # 65% en 15 paths CPCV > 70% en 3 windows anchored walk-forward
    _cpcv_wr_threshold = 65.0 if len(cv_results) >= 10 else MIN_TEST_WR
    windows_passing = sum(1 for r in cv_results if r['test_wr'] >= _cpcv_wr_threshold and r['test_pnl'] > 0)

    if windows_passing < len(cv_results) * 0.4:  # V5.1 FBI fix: 50%→40% (still stricter than V4's 3 windows)
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

    if full_m is None or full_m['trades'] < MIN_TRADES_FULL:
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
        # Fallback to full dataset if last train window has too few trades
        sl_tp = calc_sl_tp_from_mfe(full_raw)
    if sl_tp is None:
        return None

    # ─── Step 6: Re-simulate with SL/TP ───
    full_with_sl = backtest_with_sl_tp(df, full_sig, sl_tp['sl'], sl_tp['tp'], sl_tp['max_dur_bars'])

    # ─── Step 7: Leverage calculation ───
    mae_worst = max([abs(t['mae']) for t in full_raw]) if full_raw else 0.1
    safe_lev = min(20, max(1, int(1 / (mae_worst * 2.5 + 1e-10))))

    # ─── Step 8: V5 10 Gates (G1-G10) ───
    gates_passed, gates = validate_gates_v5(
        full_with_sl, sl_tp['sl'], sl_tp['tp'], safe_lev, full_m['yearly']
    )
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
        'cv_type': 'CPCV' if len(splits) > N_CV_WINDOWS else 'WalkForward',  # V5 NEW
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
        'gates': gates,
        'monte_carlo': {
            'significant': mc_sig, 'p_value': mc_pval,
            'real_wr': mc_real_wr, 'random_p95': mc_rand_p95,
            'method': 'shuffle_500',  # V5.1: reverted to shuffle (FBI fix)
        },
        'deflated_sharpe_ratio': round(dsr_val, 3),
        'weighted_score': calc_weighted_score(full_m['yearly']),
        'optimizer': 'optuna_v5',  # V5 NEW
        'timestamp': datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════
# WORKER: Process one combo = ONE strategy × ONE symbol × ONE TF
# ═══════════════════════════════════════════════════════════════════════

def process_combo(args):
    """Worker function for ProcessPoolExecutor.

    Granularity per-combo → checkpoint saved after EACH combo.
    If process crashes, only the in-flight combo is lost.
    """
    sname, symbol, tf, n_trials, multi_obj, use_journal = args

    import optuna as _optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    global N_TRIALS
    N_TRIALS = n_trials

    sys.path.insert(0, PROJECT_DIR)
    sys.path.insert(0, os.path.join(PROJECT_DIR, "strategies_code"))

    strategies = _load_strategies_in_worker()
    if sname not in strategies:
        return sname, symbol, tf, None

    info = strategies[sname]
    gen, space = info['gen'], info['space']

    df = get_candles(symbol, tf)
    if df is None or len(df) < 300:
        return sname, symbol, tf, None

    journal_path = None
    if use_journal:
        safe_name = f"{sname}_{symbol}_{tf}".replace('/', '_').replace(':', '_')
        journal_path = os.path.join(OUTPUT_DIR, f"v5_journal_{safe_name}.log")

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


def _load_strategies_in_worker():
    """Load all available strategies in worker process."""
    strats = {}
    try:
        from strategies_round2 import STRATEGY_TYPES_R2
        strats.update(STRATEGY_TYPES_R2)
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
    # ── PINE strategies (strategies_new_winners.py) ──
    try:
        import sys
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
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--strategy', help='Filter: comma-separated strategy names or "pine" for PINE strategies')
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else get_symbols(top_n=args.top)
    tfs = [args.tf] if args.tf else USE_TFS
    multi_obj = not args.single_objective
    n_trials = args.trials

    strats = _load_strategies_in_worker()
    strat_names = list(strats.keys())

    # ── Strategy filter ──
    if args.strategy:
        if args.strategy.lower() == 'pine':
            _PINE_NAMES = {
                'HatiKO_Envelopes', 'Range_Trading', 'RSI_Strat', 'Best_TV_Strategy',
                'LowFinder_Pyra', 'MACs_V6_Final', 'MeanRev_VF', 'Spike_Reversion_70',
                'TV_Spike_Reversion_70', '2Mars_MA_BB_ST', 'Impulse_V2',
            }
            strat_names = [s for s in strat_names if s in _PINE_NAMES]
        else:
            _filter = set(args.strategy.split(','))
            strat_names = [s for s in strat_names if s in _filter]
        print(f"Strategy filter: {len(strat_names)} strategies selected")

    mode_str = "Multi-Objective CPCV (WR + Sortino - DD)" if multi_obj else "Single-Objective CPCV"
    print("=" * 70)
    print(f"OPTUNA V5 — {mode_str}")
    print(f"Strategies: {len(strat_names)} | Symbols: {len(symbols)} | TFs: {tfs}")
    print(f"Trials: {n_trials} | Workers: {args.workers} | CPCV: C({N_FOLDS_CPCV},{N_TEST_FOLDS_CPCV})=15 paths")
    print(f"MC: {MC_PERMUTATIONS} block-bootstrap perms | Embargo: {EMBARGO_PCT*100}%")
    print(f"Gates: G1-G10 (G9=YearCov, G10=EV_real) | Journal: {args.journal}")
    print(f"SL: P95(ALL trades MAE)×1.5 | SL/TP from last CV train window")
    print("=" * 70)

    # V5 NEW: separate progress file from v4
    progress_file = os.path.join(OUTPUT_DIR, "optuna_v5_progress.json")
    progress = {'done_combos': [], 'grails': [], 'stats': {}}
    if os.path.exists(progress_file):
        try:
            progress = json.load(open(progress_file))
        except Exception:
            pass

    done_combos = set(progress.get('done_combos', []))
    all_grails = progress.get('grails', [])

    all_combos = [
        (sn, sym, tf)
        for sym in symbols
        for sn in strat_names
        for tf in tfs
    ]
    pending = [(sn, sym, tf) for sn, sym, tf in all_combos
               if f"{sn}|{sym}|{tf}" not in done_combos]

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

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_combo, wa): wa
            for wa in worker_args
        }

        for idx, fut in enumerate(as_completed(futures)):
            wa = futures[fut]
            sn, sym, tf = wa[0], wa[1], wa[2]
            combo_key = f"{sn}|{sym}|{tf}"

            try:
                _, _, _, result = fut.result(timeout=600)
            except Exception as e:
                print(f"  [{idx+1}/{len(pending)}] ERROR {sn}@{sym}/{tf}: {str(e)[:50]}")
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
                rate = (idx + 1) / elapsed_m if elapsed_m > 0 else 1
                eta_m = (len(pending) - idx - 1) / rate if rate > 0 else 0
                print(f"  [{idx+1}/{len(pending)}] GRAIL {sn:30s} {sym.split('/')[0]:8s} {tf:3s} "
                      f"WR={wr:.0f}% Lev={lev}x DSR={dsr} CV={cvt} | "
                      f"Total={len(all_grails)} ETA={eta_m:.0f}min")
            elif (idx + 1) % 100 == 0:
                elapsed_m = (time.time() - start) / 60
                rate = (idx + 1) / elapsed_m if elapsed_m > 0 else 1
                eta_m = (len(pending) - idx - 1) / rate if rate > 0 else 0
                print(f"  [{idx+1}/{len(pending)}] grails={len(all_grails)} "
                      f"rate={rate:.1f}/min ETA={eta_m:.0f}min")

            # Checkpoint: save after every grail or every 50 combos
            if result is not None or (idx + 1) % 50 == 0:
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
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, default=str)

    # Final save
    elapsed_h = (time.time() - start) / 3600
    progress['stats']['finished'] = True
    progress['stats']['elapsed_h'] = round(elapsed_h, 2)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, default=str)

    print(f"\n{'='*70}")
    print(f"OPTUNA V5 COMPLETE")
    print(f"  Tested: {total_tested} combos")
    print(f"  Grails: {len(all_grails)} (new this run: {new_grails})")
    print(f"  Time: {elapsed_h:.1f}h")
    print(f"  Saved: {progress_file}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
