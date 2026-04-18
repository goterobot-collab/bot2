#!/usr/bin/env python3
"""
TV2 BATCH MICRO7 — 5 microstructure strategies (5m/15m optimized).
Domain: Statistical Process Theory — Fractal Geometry, Information Entropy,
        Kalman Filter Tracking, Ornstein-Uhlenbeck Process, Hurst Adaptive
(academic, distinct from MICRO1-6 + batches 3237-3318)

Strategies:
  1. HurstAdapt       — Hurst exponent adaptive mean-rev (H<0.5 → anti-persistent)
  2. EntropyBreak     — Shannon entropy regime: low entropy → impending breakout fade
  3. KalmanTrend      — Kalman filter price tracking: divergence→reversion entry
  4. OUProcess        — Ornstein-Uhlenbeck spread half-life mean reversion
  5. FractalDimBand   — Fractal dimension adaptive band reversal

Anti-repainting: ALL OHLCV accessed via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.
"""

import pandas as pd
import numpy as np


# ─── HELPERS ────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _sma(s, p):
    return s.rolling(p).mean()

def _atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / p, adjust=False).mean()

def _zscore(s, p):
    m = s.rolling(p).mean()
    sd = s.rolling(p).std(ddof=0) + 1e-10
    return (s - m) / sd

def _apply_exit_bar(entry_long: pd.Series, entry_short: pd.Series, exit_bar: int) -> pd.Series:
    """
    3-state machine: enters on signal, exits after exit_bar bars.
    Flat (0) → Long (1) on entry_long; Flat (0) → Short (-1) on entry_short.
    Opposite entry can override an active position immediately.
    """
    sig = pd.Series(0, index=entry_long.index, dtype=int)
    el = entry_long.values
    es = entry_short.values
    n = len(sig)
    state = 0
    bars_held = 0
    for i in range(n):
        if state != 0:
            bars_held += 1
            if bars_held >= exit_bar:
                state = 0
                bars_held = 0
        if el[i]:
            state = 1
            bars_held = 0
        elif es[i]:
            state = -1
            bars_held = 0
        sig.iloc[i] = state
    return sig


# ═══════════════════════════════════════════════════════════════════════════
# 1. HurstAdapt — Hurst Exponent Adaptive Mean Reversion
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Hurst (1951), Peters (1994) "Fractal Market Hypothesis"
# Logic: R/S statistic estimates Hurst exponent H.
#   H < 0.5 → anti-persistent / mean-reverting → trade reversals
#   H > 0.5 → persistent / trending → stay flat or trend-follow
# When H < hurst_thresh AND price Z-score is extreme → fade the extreme.

def _hurst_rs(s: pd.Series, lag: int) -> float:
    """Simplified R/S statistic for one lag."""
    if len(s) < lag + 1:
        return np.nan
    sub = s.values[-lag:]
    mean = sub.mean()
    deviations = np.cumsum(sub - mean)
    r_range = deviations.max() - deviations.min()
    std = sub.std(ddof=0)
    if std < 1e-12:
        return np.nan
    return r_range / std

def _rolling_hurst(log_ret: pd.Series, window: int, min_lag: int = 8,
                   max_lag_frac: float = 0.5) -> pd.Series:
    """
    Rolling Hurst estimate: log(R/S) vs log(n) regression over lag sizes.
    Returns H series aligned to log_ret index.
    """
    max_lag = max(min_lag + 2, int(window * max_lag_frac))
    lags = np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag), 6).astype(int))
    log_lags = np.log(lags)
    n = len(log_ret)
    hurst_vals = np.full(n, np.nan)
    for i in range(window, n):
        segment = log_ret.iloc[i - window: i]
        rs_vals = []
        valid_lags = []
        for lag in lags:
            rs = _hurst_rs(segment, lag)
            if not np.isnan(rs) and rs > 0:
                rs_vals.append(np.log(rs))
                valid_lags.append(np.log(lag))
        if len(rs_vals) >= 3:
            coeffs = np.polyfit(valid_lags, rs_vals, 1)
            hurst_vals[i] = np.clip(coeffs[0], 0.1, 0.9)
    return pd.Series(hurst_vals, index=log_ret.index)


def gen_HurstAdapt(df, hurst_window=40, hurst_thresh=0.48,
                   zscore_period=20, zscore_thresh=1.6,
                   atr_period=14, exit_bar=10):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    log_ret = np.log(c / c.shift(1)).fillna(0)
    H = _rolling_hurst(log_ret, hurst_window)

    cz = _zscore(c, zscore_period)
    atr = _atr(h, l, c, atr_period)
    vol_ok = atr > atr.rolling(atr_period * 3).mean() * 0.3  # min liquidity

    mean_rev_regime = H < hurst_thresh
    entry_long  = (mean_rev_regime & (cz < -zscore_thresh) & vol_ok).fillna(False)
    entry_short = (mean_rev_regime & (cz >  zscore_thresh) & vol_ok).fillna(False)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_HurstAdapt(trial):
    return {
        'hurst_window':  trial.suggest_int('hurst_window', 30, 60, step=5),
        'hurst_thresh':  trial.suggest_float('hurst_thresh', 0.42, 0.52, step=0.02),
        'zscore_period': trial.suggest_int('zscore_period', 12, 30, step=4),
        'zscore_thresh': trial.suggest_float('zscore_thresh', 1.2, 2.2, step=0.2),
        'atr_period':    trial.suggest_int('atr_period', 10, 20, step=2),
        'exit_bar':      trial.suggest_int('exit_bar', 6, 16, step=2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. EntropyBreak — Shannon Entropy Regime Detection
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Shannon (1948), Pele et al. (2013) "Entropy of Markets"
# Logic: Approximate price-return entropy via histogram of rolling returns.
#   LOW entropy → returns concentrated in few bins → regime compression →
#   imminent breakout. Fade the extreme move POST-compression breakout.
# When entropy drops below threshold AND price spikes → reversal entry.

def _rolling_entropy(s: pd.Series, window: int, n_bins: int = 8) -> pd.Series:
    """
    Rolling Shannon entropy of return distribution.
    Low entropy = distribution concentrated = low predictive uncertainty.
    """
    n = len(s)
    entropy_vals = np.full(n, np.nan)
    sv = s.values
    for i in range(window, n):
        seg = sv[i - window: i]
        counts, _ = np.histogram(seg, bins=n_bins)
        probs = counts / (counts.sum() + 1e-10)
        # Filter zero probs to avoid log(0)
        probs = probs[probs > 0]
        entropy_vals[i] = -np.sum(probs * np.log(probs + 1e-10))
    return pd.Series(entropy_vals, index=s.index)


def gen_EntropyBreak(df, entropy_window=30, entropy_low_pct=25,
                     n_bins=8, momentum_bars=3,
                     atr_period=14, atr_mult=1.2, exit_bar=8):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    ret = c.pct_change().fillna(0)
    ent = _rolling_entropy(ret, entropy_window, n_bins)

    # Low-entropy threshold: rolling percentile
    ent_thresh = ent.rolling(entropy_window * 2).quantile(entropy_low_pct / 100.0)
    low_ent = ent < ent_thresh

    # Post-compression breakout: sharp move in last momentum_bars bars
    c_change = (c / c.shift(momentum_bars) - 1).fillna(0)
    atr = _atr(h, l, c, atr_period)
    spike_up   = c_change >  (atr / c.replace(0, np.nan) * atr_mult * momentum_bars)
    spike_down = c_change < -(atr / c.replace(0, np.nan) * atr_mult * momentum_bars)

    # Fade the spike after compression: spike_up → short, spike_down → long
    entry_long  = (low_ent & spike_down).fillna(False)
    entry_short = (low_ent & spike_up).fillna(False)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_EntropyBreak(trial):
    return {
        'entropy_window':   trial.suggest_int('entropy_window', 20, 50, step=5),
        'entropy_low_pct':  trial.suggest_int('entropy_low_pct', 15, 35, step=5),
        'n_bins':           trial.suggest_categorical('n_bins', [6, 8, 10, 12]),
        'momentum_bars':    trial.suggest_int('momentum_bars', 2, 5),
        'atr_period':       trial.suggest_int('atr_period', 10, 20, step=2),
        'atr_mult':         trial.suggest_float('atr_mult', 0.8, 2.0, step=0.2),
        'exit_bar':         trial.suggest_int('exit_bar', 5, 14, step=1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. KalmanTrend — Kalman Filter Price Tracking Divergence
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Kalman (1960), Zhu & Martinez (2006) "Kalman Filter for Trading"
# Logic: 1D Kalman filter tracks the "true" price. When actual price diverges
#   from Kalman estimate beyond noise threshold → mean-reversion entry.
#   The Kalman gain adapts based on measurement noise vs process noise ratio.

def _kalman_filter(prices: pd.Series, process_noise: float = 0.01,
                   meas_noise: float = 1.0) -> pd.Series:
    """
    1D Kalman filter: state = price level.
    Returns estimated price series.
    """
    n = len(prices)
    x_est = np.zeros(n)   # State estimate
    p_est = np.zeros(n)   # Estimate covariance

    x_est[0] = prices.iloc[0]
    p_est[0] = 1.0
    pv = prices.values

    for i in range(1, n):
        # Predict
        x_pred = x_est[i - 1]
        p_pred = p_est[i - 1] + process_noise

        # Update (Kalman gain)
        K = p_pred / (p_pred + meas_noise)
        x_est[i] = x_pred + K * (pv[i] - x_pred)
        p_est[i] = (1 - K) * p_pred

    return pd.Series(x_est, index=prices.index)


def gen_KalmanTrend(df, process_noise=0.008, meas_noise=1.5,
                    diverge_period=20, diverge_thresh=1.5,
                    atr_period=14, exit_bar=9):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    kalman_est = _kalman_filter(c, process_noise, meas_noise)
    divergence = c - kalman_est

    div_z = _zscore(divergence, diverge_period)
    atr = _atr(h, l, c, atr_period)
    vol_ok = atr > atr.rolling(30).mean() * 0.25

    # Price above Kalman = overbought → short; below Kalman = oversold → long
    entry_long  = (div_z < -diverge_thresh) & vol_ok
    entry_short = (div_z >  diverge_thresh) & vol_ok

    return _apply_exit_bar(entry_long.fillna(False), entry_short.fillna(False), exit_bar)


def space_KalmanTrend(trial):
    return {
        'process_noise':  trial.suggest_float('process_noise', 0.001, 0.05, log=True),
        'meas_noise':     trial.suggest_float('meas_noise', 0.5, 5.0, step=0.5),
        'diverge_period': trial.suggest_int('diverge_period', 12, 35, step=4),
        'diverge_thresh': trial.suggest_float('diverge_thresh', 1.0, 2.5, step=0.25),
        'atr_period':     trial.suggest_int('atr_period', 10, 20, step=2),
        'exit_bar':       trial.suggest_int('exit_bar', 5, 16, step=2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. OUProcess — Ornstein-Uhlenbeck Spread Half-Life Mean Reversion
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Ornstein & Uhlenbeck (1930), Avellaneda & Lee (2010)
#   "Statistical arbitrage in the US equities market"
# Logic: O-U process: dX = κ(μ - X)dt + σdW
#   Estimate mean-reversion speed κ from rolling regression of ΔX on X(t-1).
#   Half-life = log(2)/κ. Fast half-life → strong reversion.
#   Use O-U residual Z-score as entry signal.

def _ou_halflife(spread: pd.Series, window: int) -> pd.Series:
    """
    Rolling estimate of O-U half-life from LS regression: ΔX = a + b*X(t-1).
    b = -κ → half-life = -log(2)/log(1+b).
    """
    n = len(spread)
    hl_vals = np.full(n, np.nan)
    sv = spread.values
    for i in range(window, n):
        seg = sv[i - window: i]
        delta = np.diff(seg)
        lag = seg[:-1]
        if lag.std() < 1e-10:
            continue
        # OLS: delta ~ a + b * lag
        A = np.column_stack([np.ones(len(lag)), lag])
        try:
            if np.linalg.matrix_rank(A) < 2:
                continue
            coeffs, _, _, _ = np.linalg.lstsq(A, delta, rcond=None)
            b = coeffs[1]
            if -1.0 < b < -1e-6:
                hl_vals[i] = -np.log(2) / np.log(1 + b)
        except Exception:
            pass
    return pd.Series(hl_vals, index=spread.index)


def gen_OUProcess(df, ou_window=40, halflife_max=20,
                  zscore_period=20, zscore_thresh=1.8,
                  atr_period=14, exit_bar=12):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    # Spread = log-price deviation from rolling mean (OU "spread")
    log_c = np.log(c.replace(0, np.nan)).ffill()
    spread = log_c - log_c.rolling(ou_window).mean()

    hl = _ou_halflife(spread, ou_window)
    fast_reversion = (hl > 0) & (hl < halflife_max)

    spread_z = _zscore(spread, zscore_period)
    atr = _atr(h, l, c, atr_period)
    vol_ok = atr > atr.rolling(30).mean() * 0.2

    entry_long  = (fast_reversion & (spread_z < -zscore_thresh) & vol_ok).fillna(False)
    entry_short = (fast_reversion & (spread_z >  zscore_thresh) & vol_ok).fillna(False)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_OUProcess(trial):
    return {
        'ou_window':      trial.suggest_int('ou_window', 25, 60, step=5),
        'halflife_max':   trial.suggest_int('halflife_max', 8, 30, step=4),
        'zscore_period':  trial.suggest_int('zscore_period', 12, 30, step=4),
        'zscore_thresh':  trial.suggest_float('zscore_thresh', 1.3, 2.5, step=0.2),
        'atr_period':     trial.suggest_int('atr_period', 10, 20, step=2),
        'exit_bar':       trial.suggest_int('exit_bar', 6, 18, step=2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. FractalDimBand — Fractal Dimension Adaptive Band Reversal
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Mandelbrot (1982), Chaudhuri & Lo (2001) "Nonstationarity"
# Logic: Fractal Dimension (FD) measures market roughness.
#   FD ≈ 1.0 → trending; FD ≈ 2.0 → random/mean-reverting.
#   Estimate FD via the FRAMA algorithm (Ehlers 2010):
#     FD = (log(N1+N2) - log(N)) / log(2)
#   High FD + price at adaptive band extremes → mean-reversion entry.

def _fractal_dim(h: pd.Series, l: pd.Series, window: int) -> pd.Series:
    """
    Rolling Fractal Dimension via FRAMA method.
    FD ∈ [1, 2]: ~1 = strongly trending, ~2 = noisy/mean-reverting.
    """
    half = window // 2
    n = len(h)
    fd_vals = np.full(n, 1.5)
    hv = h.values
    lv = l.values
    for i in range(window, n):
        # First half
        h1 = hv[i - window: i - half]
        l1 = lv[i - window: i - half]
        N1 = (h1.max() - l1.min()) / half if (h1.max() - l1.min()) > 0 else 1e-10
        # Second half
        h2 = hv[i - half: i]
        l2 = lv[i - half: i]
        N2 = (h2.max() - l2.min()) / half if (h2.max() - l2.min()) > 0 else 1e-10
        # Full window
        hfull = hv[i - window: i]
        lfull = lv[i - window: i]
        N = (hfull.max() - lfull.min()) / window if (hfull.max() - lfull.min()) > 0 else 1e-10
        if N1 + N2 > 0 and N > 0:
            fd_val = (np.log(N1 + N2) - np.log(N)) / np.log(2)
            fd_vals[i] = np.clip(fd_val, 1.0, 2.0)
    return pd.Series(fd_vals, index=h.index)


def gen_FractalDimBand(df, fd_window=20, fd_thresh=1.4,
                       band_period=20, band_mult=1.8,
                       atr_period=14, exit_bar=8):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    fd = _fractal_dim(h, l, fd_window)
    mean_reverting = fd > fd_thresh   # high FD = choppy = mean-reverting

    # Adaptive band: ATR-based rather than std (works better for crypto)
    atr = _atr(h, l, c, atr_period)
    mid = _sma(c, band_period)
    upper = mid + band_mult * atr
    lower = mid - band_mult * atr

    above_band = c > upper
    below_band = c < lower

    entry_long  = (mean_reverting & below_band).fillna(False)
    entry_short = (mean_reverting & above_band).fillna(False)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_FractalDimBand(trial):
    return {
        'fd_window':    trial.suggest_int('fd_window', 14, 40, step=4),
        'fd_thresh':    trial.suggest_float('fd_thresh', 1.25, 1.65, step=0.05),
        'band_period':  trial.suggest_int('band_period', 12, 30, step=4),
        'band_mult':    trial.suggest_float('band_mult', 1.2, 2.8, step=0.2),
        'atr_period':   trial.suggest_int('atr_period', 10, 20, step=2),
        'exit_bar':     trial.suggest_int('exit_bar', 5, 14, step=1),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'HurstAdapt': {
        'gen':   gen_HurstAdapt,
        'space': space_HurstAdapt,
    },
    'EntropyBreak': {
        'gen':   gen_EntropyBreak,
        'space': space_EntropyBreak,
    },
    'KalmanTrend': {
        'gen':   gen_KalmanTrend,
        'space': space_KalmanTrend,
    },
    'OUProcess': {
        'gen':   gen_OUProcess,
        'space': space_OUProcess,
    },
    'FractalDimBand': {
        'gen':   gen_FractalDimBand,
        'space': space_FractalDimBand,
    },
}
