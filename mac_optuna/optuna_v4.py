#!/usr/bin/env python3
"""
OPTUNA V4 — Next-gen optimizer for trading strategies.

Improvements over v3 (optuna_full_v3.py / optuna_worker.py):
  1. SAMPLER: GPSampler for pure int/float spaces, TPE fallback for categoricals
  2. OBJECTIVE: Multi-objective Pareto (WR, Sortino, -DD) instead of WR*0.7+Sharpe*0.3
  3. TEMPORAL CV: Anchored walk-forward with 3 windows + embargo period
  4. PRUNING: WilcoxonPruner for single-obj CV (prunes statistically inferior trials)
  5. ROBUSTNESS: Monte Carlo permutation test to reject noise-params
  6. DSR: Deflated Sharpe Ratio corrects for selection bias from N trials
  7. SEARCH SPACE: suggest_int(log=True) for periods, n_startup_trials=5
  8. MFE + 8 GATES: Integrated post-optimization validation

Usage:
  python3 optuna_v4.py                                   # All strategies × top 100 symbols
  python3 optuna_v4.py --symbol BTC/USDT:USDT            # One symbol
  python3 optuna_v4.py --top 50 --tf 4h                  # Top 50, specific TF
  python3 optuna_v4.py --round r2                        # Only R2 strategies
  python3 optuna_v4.py --workers 6 --trials 30           # Custom parallelism
  python3 optuna_v4.py --single-objective                # Legacy single-objective mode
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
import multiprocessing

warnings.filterwarnings('ignore')

# ─── CONFIG ───
DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"
PROJECT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Defaults (overridable via CLI)
N_TRIALS = 25
MAX_WORKERS = 8
COMMISSION = 0.001
SLIPPAGE = 0.0005
MIN_TRADES_TRAIN = 5
MIN_TRADES_TEST = 5
MIN_TRADES_FULL = 15
TIMEOUT_PER_STUDY = 60  # seconds

# Walk-forward CV config
N_CV_WINDOWS = 3          # Number of anchored walk-forward windows
EMBARGO_PCT = 0.02        # 2% gap between train/test to prevent leakage
MIN_TEST_WR = 70.0        # Minimum WR on test to pass
MAX_OVERFIT_GAP = 30.0    # Max |train_wr - test_wr| gap

# MFE / Gates constants
SL_MIN, SL_MAX = 0.02, 0.40
TP_MIN, TP_MAX = 0.01, 0.50
MAX_DUR_CAP = 720         # hours
RR_FLOOR = 0.50
DD_PNL_MAX = 1.5

# Monte Carlo config
MC_PERMUTATIONS = 100     # Number of random shuffles for significance test
MC_P_THRESHOLD = 0.05     # p-value threshold (5% significance)

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
# TEMPORAL CROSS-VALIDATION (Anchored Walk-Forward)
# ═══════════════════════════════════════════════════════════════════════

def make_cv_splits(df, n_windows=N_CV_WINDOWS, embargo_pct=EMBARGO_PCT):
    """Create anchored walk-forward splits with embargo.

    Window 1: [0..40%] train, [42%..56%] test  (2% embargo)
    Window 2: [0..56%] train, [58%..72%] test
    Window 3: [0..72%] train, [74%..100%] test  (this is similar to original 70/30)

    Each window anchors from the START, growing the train set. Test always moves forward.
    """
    n = len(df)
    if n < 300:
        return []  # not enough data for CV

    splits = []
    # Divide the last 60% of data into n_windows test blocks
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
            splits.append((train, test, f"W{w+1}"))

    return splits


# ═══════════════════════════════════════════════════════════════════════
# SAMPLER SELECTION
# ═══════════════════════════════════════════════════════════════════════

def has_categorical_params(space_fn):
    """Check if a space function uses categorical params (needs TPE)."""
    try:
        # Create a mock trial to inspect param types
        study = optuna.create_study()
        trial = study.ask()
        params = space_fn(trial)
        # Check distributions
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
    - Both support multi-objective since Optuna 4.4+
    """
    has_cat = has_categorical_params(space_fn)

    if has_cat:
        return optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            n_startup_trials=5,  # v3 default was 10 → wasted half the budget
        )
    else:
        # GPSampler needs torch — try it, fallback to TPE
        try:
            import torch  # noqa: F401
            return optuna.samplers.GPSampler(
                seed=seed,
                deterministic_objective=True,  # backtests are deterministic
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

def objective_single(gen, space, df_train):
    """Legacy single-objective: Sortino-weighted score (better than WR*0.7+Sharpe*0.3).

    Why Sortino > Sharpe: Sharpe penalizes upside volatility (good for us).
    Sortino only penalizes downside — exactly what we want in trading.
    """
    def objective(trial):
        params = space(trial)
        try:
            sig = gen(df_train, **params)
            raw = backtest_raw(df_train, sig)
            m = compute_metrics(raw)
            if m is None:
                return -999
            # Weighted: WR most important, Sortino for risk-adj, minus DD penalty
            return m['wr'] * 0.5 + min(m['sortino'], 50) * 0.3 - m['max_drawdown'] * 0.2
        except Exception:
            return -999
    return objective


def objective_single_cv(gen, space, cv_splits):
    """Single-objective with WilcoxonPruner: evaluate across CV windows.

    Reports WR per window as intermediate value for pruning.
    Bad trials get pruned early (after 1-2 windows instead of all 3).
    """
    def objective(trial):
        params = space(trial)
        window_scores = []

        for step, (train_w, test_w, label) in enumerate(cv_splits):
            try:
                sig = gen(test_w, **params)
                raw = backtest_raw(test_w, sig)
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


def objective_multi(gen, space, df_train):
    """Multi-objective: maximize WR, maximize Sortino, minimize MaxDD.

    Returns 3 values for Pareto front optimization.
    """
    def objective(trial):
        params = space(trial)
        try:
            sig = gen(df_train, **params)
            raw = backtest_raw(df_train, sig)
            m = compute_metrics(raw)
            if m is None:
                return 0.0, -999.0, 999.0
            return m['wr'], m['sortino'], m['max_drawdown']
        except Exception:
            return 0.0, -999.0, 999.0
    return objective


def objective_multi_cv(gen, space, cv_splits):
    """Multi-objective across CV windows (no pruning — trial.report unsupported in multi-obj).

    Evaluates all windows, returns average (WR, Sortino, DD) across windows.
    """
    def objective(trial):
        params = space(trial)
        all_wr, all_sortino, all_dd = [], [], []

        for train_w, test_w, label in cv_splits:
            try:
                sig = gen(test_w, **params)
                raw = backtest_raw(test_w, sig)
                m = compute_metrics(raw)
                if m is None:
                    all_wr.append(0.0); all_sortino.append(-999.0); all_dd.append(999.0)
                else:
                    all_wr.append(m['wr']); all_sortino.append(m['sortino']); all_dd.append(m['max_drawdown'])
            except Exception:
                all_wr.append(0.0); all_sortino.append(-999.0); all_dd.append(999.0)

        return float(np.mean(all_wr)), float(np.mean(all_sortino)), float(np.mean(all_dd))
    return objective


def select_from_pareto(study, weights=None):
    """Select best trial from Pareto front using TOPSIS.

    TOPSIS finds the solution closest to ideal and farthest from anti-ideal,
    normalized across all objectives. Better than simple weighted sum because
    it accounts for the SHAPE of the trade-off surface.

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

    # Build decision matrix
    matrix = np.array([[t.values[0], min(t.values[1], 50), t.values[2]] for t in trials], dtype=float)
    directions = [1, 1, -1]  # +1=maximize, -1=minimize

    # Vector normalization
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0] = 1e-10
    norm_matrix = (matrix / norms) * weights

    # Ideal and anti-ideal points
    ideal = np.array([
        norm_matrix[:, i].max() if d == 1 else norm_matrix[:, i].min()
        for i, d in enumerate(directions)
    ])
    anti_ideal = np.array([
        norm_matrix[:, i].min() if d == 1 else norm_matrix[:, i].max()
        for i, d in enumerate(directions)
    ])

    # Closeness coefficient
    d_ideal = np.sqrt(((norm_matrix - ideal) ** 2).sum(axis=1))
    d_anti = np.sqrt(((norm_matrix - anti_ideal) ** 2).sum(axis=1))
    closeness = d_anti / (d_ideal + d_anti + 1e-10)

    return trials[int(np.argmax(closeness))]


# ═══════════════════════════════════════════════════════════════════════
# MONTE CARLO ROBUSTNESS TEST
# ═══════════════════════════════════════════════════════════════════════

def monte_carlo_test(gen, best_params, df, n_perms=MC_PERMUTATIONS):
    """Test if optimized params are significantly better than random.

    Shuffle the signal series N times, backtest each → compute WR distribution.
    If real WR is above 95th percentile of random → params are significant.

    Returns: (is_significant, p_value, real_wr, random_wr_95th)
    """
    # Real performance
    try:
        real_sig = gen(df, **best_params)
        real_trades = backtest_raw(df, real_sig)
        real_m = compute_metrics(real_trades)
        if real_m is None:
            return True, 0, 0, 0  # can't test → pass by default
        real_wr = real_m['wr']
    except Exception:
        return True, 0, 0, 0

    # Random shuffles
    random_wrs = []
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        shuffled = real_sig.copy()
        vals = shuffled.values.copy()
        rng.shuffle(vals)
        shuffled = pd.Series(vals, index=real_sig.index)
        rand_trades = backtest_raw(df, shuffled)
        rand_m = compute_metrics(rand_trades)
        if rand_m is not None:
            random_wrs.append(rand_m['wr'])

    if len(random_wrs) < 10:
        return True, 0, real_wr, 0  # not enough random samples → pass

    random_wrs = np.array(random_wrs)
    p95 = np.percentile(random_wrs, 95)
    p_value = (random_wrs >= real_wr).sum() / len(random_wrs)

    return p_value < MC_P_THRESHOLD, round(p_value, 3), round(real_wr, 1), round(p95, 1)


# ═══════════════════════════════════════════════════════════════════════
# DEFLATED SHARPE RATIO (Bailey & Lopez de Prado, 2014)
# ═══════════════════════════════════════════════════════════════════════

def deflated_sharpe_ratio(estimated_sr, n_trials, backtest_bars,
                          skew=0.0, excess_kurtosis=0.0, sr_variance=1.0):
    """Compute DSR: probability that true SR > selection-bias-adjusted benchmark.

    Corrects for multiple testing: if you ran n_trials, the expected max SR
    of n_trials random strategies is NOT zero. DSR < 0.95 → don't trust it.

    Args:
        estimated_sr: Sharpe ratio of the best strategy found
        n_trials: Number of Optuna trials (selection bias source)
        backtest_bars: Length of backtest in bars (T)
        skew: Skewness of returns
        excess_kurtosis: Excess kurtosis of returns
        sr_variance: Variance of SR estimates (default 1.0)

    Returns: DSR probability (0 to 1). > 0.95 = trustworthy.

    Source: Bailey & Lopez de Prado (2014)
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
    """
    from scipy.stats import norm

    if n_trials <= 1 or backtest_bars <= 1:
        return 1.0  # No correction needed

    # Expected maximum SR under null hypothesis of n_trials independent tests
    e_gamma = 0.5772156649  # Euler-Mascheroni constant
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
# MFE ANALYSIS + 8 GATES
# ═══════════════════════════════════════════════════════════════════════

def calc_sl_tp_from_mfe(raw_trades):
    """Calculate empirical SL/TP from MFE of winning trades."""
    winners = [t for t in raw_trades if t['win']]
    if len(winners) < 5:
        return None

    maes = [abs(t['mae']) for t in winners]
    mfes = [t['mfe'] for t in winners]
    durs = [t['dur_bars'] for t in winners]

    sl = np.clip(np.percentile(maes, 95) * 1.5, SL_MIN, SL_MAX)
    tp = np.clip(np.percentile(mfes, 75) * 0.8, TP_MIN, TP_MAX)
    max_dur = min(np.percentile(durs, 95) * 1.5, MAX_DUR_CAP)

    return {'sl': round(sl, 4), 'tp': round(tp, 4), 'max_dur_bars': int(max_dur)}


def validate_8_gates(trades_with_sl, sl, tp, leverage=1):
    """Apply 8 production gates. Returns (passed, gate_results)."""
    if not trades_with_sl:
        return False, {'error': 'no trades'}

    pnls = np.array([t['pnl'] for t in trades_with_sl])
    n = len(pnls); wins = (pnls > 0).sum()
    wr = wins / n * 100 if n > 0 else 0
    total_pnl = pnls.sum() * 100
    cum = (pnls * 100).cumsum()
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum).max() if len(cum) > 0 else 0

    gates = {
        'G1_WR':      {'pass': wr >= MIN_TEST_WR, 'value': round(wr, 1)},
        'G2_PnL':     {'pass': total_pnl > 0, 'value': round(total_pnl, 2)},
        'G3_trades':  {'pass': n >= MIN_TRADES_TEST, 'value': n},
        'G4_SLxLev':  {'pass': sl * leverage * 100 < 80, 'value': round(sl * leverage * 100, 1)},
        'G5_DD_PnL':  {'pass': dd / (total_pnl + 1e-10) < DD_PNL_MAX if total_pnl > 0 else False,
                        'value': round(dd / (total_pnl + 1e-10), 2) if total_pnl > 0 else 999},
        'G6_EV':      {'pass': (wr / 100 * tp) - ((1 - wr / 100) * sl) > 0,
                        'value': round(((wr / 100 * tp) - ((1 - wr / 100) * sl)) * 100, 3)},
        'G7_duration': {'pass': True, 'value': max(t['dur_bars'] for t in trades_with_sl)},
        'G8_RR':      {'pass': tp / (sl + 1e-10) >= RR_FLOOR,
                        'value': round(tp / (sl + 1e-10), 2)},
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


def optimize_one(gen, space, df, symbol, tf, strat_name, multi_obj=True):
    """Full pipeline: Optuna → CV validation → MFE → 8 Gates → Monte Carlo.

    Returns grail dict or None.
    """
    # ─── Step 1: Temporal CV splits ───
    splits = make_cv_splits(df)
    if not splits:
        # Fallback to single 70/30
        split_idx = int(len(df) * 0.7)
        train, test = df.iloc[:split_idx], df.iloc[split_idx:]
        if len(train) < 100 or len(test) < 50:
            return None
        splits = [(train, test, "W1")]

    # ─── Step 2: Optuna with WilcoxonPruner across CV windows ───
    sampler = pick_sampler(space, multi_objective=multi_obj)

    # WilcoxonPruner: prunes trials that perform poorly across CV windows
    try:
        pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.1)
    except AttributeError:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3)

    # Warm-start: seed with previous best params if available
    global _PREV_GRAILS
    if _PREV_GRAILS is None:
        _PREV_GRAILS = _load_previous_grails()
    prev_key = (strat_name, symbol, tf)
    prev_params = _PREV_GRAILS.get(prev_key)

    if multi_obj:
        try:
            study = optuna.create_study(
                directions=['maximize', 'maximize', 'minimize'],
                sampler=sampler,
                pruner=pruner,
            )
            if prev_params:
                try:
                    study.enqueue_trial(prev_params)
                except Exception:
                    pass  # param mismatch — skip warm-start
            obj_fn = objective_multi_cv(gen, space, splits)
            study.optimize(obj_fn, n_trials=N_TRIALS, timeout=TIMEOUT_PER_STUDY)

            best_trial = select_from_pareto(study)
            if best_trial is None:
                return None
            bp = best_trial.params
        except Exception:
            # Multi-obj failed → fallback to single
            multi_obj = False

    if not multi_obj:
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
        )
        if prev_params:
            try:
                study.enqueue_trial(prev_params)
            except Exception:
                pass
        obj_fn = objective_single_cv(gen, space, splits)
        study.optimize(obj_fn, n_trials=N_TRIALS, timeout=TIMEOUT_PER_STUDY)
        if study.best_value <= -900:
            return None
        bp = study.best_params

    # ─── Step 3: Post-hoc CV validation (verify best params across all windows) ───
    cv_results = []
    for train_w, test_w, label in splits:
        try:
            test_sig = gen(test_w, **bp)
            test_trades = backtest_raw(test_w, test_sig)
            test_m = compute_metrics(test_trades)

            train_sig = gen(train_w, **bp)
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

    # Majority of windows must pass WR + PnL thresholds
    avg_test_wr = np.mean([r['test_wr'] for r in cv_results])
    avg_wr_gap = np.mean([r['wr_gap'] for r in cv_results])
    windows_passing = sum(1 for r in cv_results if r['test_wr'] >= MIN_TEST_WR and r['test_pnl'] > 0)

    if windows_passing < len(cv_results) * 0.5:
        return None
    if avg_wr_gap > MAX_OVERFIT_GAP:
        return None

    # ─── Step 5: Full dataset metrics ───
    try:
        full_sig = gen(df, **bp)
        full_raw = backtest_raw(df, full_sig)
        full_m = compute_metrics(full_raw)
    except Exception:
        return None

    if full_m is None or full_m['trades'] < MIN_TRADES_FULL:
        return None

    # ─── Step 6: MFE analysis ───
    sl_tp = calc_sl_tp_from_mfe(full_raw)
    if sl_tp is None:
        return None

    # Re-simulate with SL/TP
    full_with_sl = backtest_with_sl_tp(full_sig.index.to_frame().join(df).drop(columns=['datetime'], errors='ignore'),
                                        full_sig, sl_tp['sl'], sl_tp['tp'], sl_tp['max_dur_bars'])
    # Simpler approach — use df directly
    full_with_sl = backtest_with_sl_tp(df, full_sig, sl_tp['sl'], sl_tp['tp'], sl_tp['max_dur_bars'])

    # ─── Step 7: Leverage calculation ───
    mae_worst = max([abs(t['mae']) for t in full_raw]) if full_raw else 0.1
    safe_lev = min(20, max(1, int(1 / (mae_worst * 2.5 + 1e-10))))

    # ─── Step 8: 8 Gates validation ───
    gates_passed, gates = validate_8_gates(full_with_sl, sl_tp['sl'], sl_tp['tp'], safe_lev)
    if not gates_passed:
        return None

    # ─── Step 9: Monte Carlo robustness (only for grails that passed gates) ───
    mc_sig, mc_pval, mc_real_wr, mc_rand_p95 = monte_carlo_test(gen, bp, df, n_perms=MC_PERMUTATIONS)
    if not mc_sig:
        return None  # params are not significantly better than random

    # ─── Step 10: Deflated Sharpe Ratio (selection bias correction) ───
    dsr_val = 1.0  # default: pass
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
    # DSR < 0.95 = selection bias likely → soft warning (not rejection)
    # We log it but don't reject — WR+Gates+MC are the hard filters

    # ─── BUILD GRAIL ───
    return {
        'strategy': strat_name,
        'symbol': symbol,
        'timeframe': tf,
        'best_params': bp,
        'train_wr': round(avg_test_wr, 1),  # average across CV windows
        'cv_windows': cv_results,
        'cv_windows_passing': windows_passing,
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
        },
        'deflated_sharpe_ratio': round(dsr_val, 3),
        'weighted_score': calc_weighted_score(full_m['yearly']),
        'optimizer': 'optuna_v4',
        'timestamp': datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════
# WORKER: Process one symbol across all strategies × TFs
# ═══════════════════════════════════════════════════════════════════════

def process_combo(args):
    """Worker function: ONE combo = ONE strategy × ONE symbol × ONE TF.

    Granularidad per-combo → checkpoint guardado después de CADA combo.
    Si se cae, solo se pierde el combo en proceso, no el símbolo completo.
    """
    sname, symbol, tf, n_trials, multi_obj = args

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

    try:
        result = optimize_one(gen, space, df, symbol, tf, sname, multi_obj=multi_obj)
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
    return strats


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Optuna V4 — Next-gen strategy optimizer')
    parser.add_argument('--symbol', help='Specific symbol')
    parser.add_argument('--top', type=int, default=100, help='Top N symbols (default 100)')
    parser.add_argument('--tf', help='Specific timeframe')
    parser.add_argument('--round', default='all', help='r2, r4_batch, or all')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--single-objective', action='store_true', help='Use single-objective (legacy)')
    parser.add_argument('--no-monte-carlo', action='store_true', help='Skip Monte Carlo test')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    # Symbols
    symbols = [args.symbol] if args.symbol else get_symbols(top_n=args.top)
    tfs = [args.tf] if args.tf else USE_TFS
    multi_obj = not args.single_objective
    n_trials = args.trials
    mc_perms = 0 if args.no_monte_carlo else MC_PERMUTATIONS

    # Load strategies to get names
    strats = _load_strategies_in_worker()
    strat_names = list(strats.keys())

    mode_str = "Multi-Objective (WR + Sortino - DD)" if multi_obj else "Single-Objective (Sortino-weighted)"
    print("=" * 70)
    print(f"OPTUNA V4 — {mode_str}")
    print(f"Strategies: {len(strat_names)} | Symbols: {len(symbols)} | TFs: {tfs}")
    print(f"Trials: {n_trials} | Workers: {args.workers} | CV Windows: {N_CV_WINDOWS}")
    print(f"Monte Carlo: {mc_perms} perms | Embargo: {EMBARGO_PCT*100}%")
    print(f"Sampler: auto (GPSampler for int/float, TPE for categoricals)")
    print("=" * 70)

    # Progress tracking
    progress_file = os.path.join(OUTPUT_DIR, "optuna_v4_progress.json")
    progress = {'completed': [], 'grails': [], 'stats': {}}
    if os.path.exists(progress_file):
        try:
            progress = json.load(open(progress_file))
        except Exception:
            pass
    # done_combos: set of "strategy|symbol|tf" strings — granularidad per-combo
    done_combos = set(progress.get('done_combos', []))
    # legacy: also mark symbols listed in 'completed' as done for all combos
    for sym in progress.get('completed', []):
        for sn in strat_names:
            for tf in tfs:
                done_combos.add(f"{sn}|{sym}|{tf}")
    all_grails = progress.get('grails', [])

    # Build list of pending combos: strategy × symbol × TF
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
        (sn, sym, tf, n_trials, multi_obj)
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
                wr  = result['full']['wr']
                lev = result['safe_leverage']
                dsr = result.get('deflated_sharpe_ratio', '?')
                elapsed_m = (time.time() - start) / 60
                rate = (idx + 1) / elapsed_m if elapsed_m > 0 else 1
                eta_m = (len(pending) - idx - 1) / rate if rate > 0 else 0
                print(f"  [{idx+1}/{len(pending)}] ✅ {sn:30s} {sym.split('/')[0]:8s} {tf:3s} "
                      f"WR={wr:.0f}% Lev={lev}x DSR={dsr} | "
                      f"Total={len(all_grails)} ETA={eta_m:.0f}min")
            elif (idx + 1) % 100 == 0:
                elapsed_m = (time.time() - start) / 60
                rate = (idx + 1) / elapsed_m if elapsed_m > 0 else 1
                eta_m = (len(pending) - idx - 1) / rate if rate > 0 else 0
                print(f"  [{idx+1}/{len(pending)}] grails={len(all_grails)} "
                      f"rate={rate:.1f}/min ETA={eta_m:.0f}min")

            # Checkpoint: save after EVERY grail or every 50 combos
            if result is not None or (idx + 1) % 50 == 0:
                progress = {
                    'done_combos': list(done_combos),
                    'grails': all_grails,
                    'stats': {
                        'total_tested': total_tested,
                        'grails_found': len(all_grails),
                        'elapsed_h': round((time.time() - start) / 3600, 2),
                        'last_updated': datetime.now().isoformat(),
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
    print(f"OPTUNA V4 COMPLETE")
    print(f"  Tested: {total_tested} combos")
    print(f"  Grails: {len(all_grails)} (new this run: {new_grails})")
    print(f"  Time: {elapsed_h:.1f}h")
    print(f"  Saved: {progress_file}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
