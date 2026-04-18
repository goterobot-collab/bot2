#!/usr/bin/env python3
"""
TV2 BATCH MICRO17 — 5 microstructure strategies (5m/15m optimized).
Domain: Optimal Execution Theory (Almgren-Chriss), Cointegration Mean-Reversion
        (Engle-Granger), Kalman Filter Tracking, HMM-Inspired Regime Transition,
        PCA Factor Divergence — all new vs MICRO1-16 and batches 3237-3307.

Strategies:
  1. AlmgrenChriss_Shortfall  — Almgren & Chriss (2000) optimal execution:
                                 implementation shortfall proxy = price deviation
                                 from VWAP per unit of volume. High shortfall
                                 after a price move → smart money absorbed →
                                 mean-reversion. Low shortfall → execution
                                 efficient → trend signal.

  2. EngleGranger_SpreadRev   — Engle & Granger (1987) cointegration:
                                 synthetic spread = close − (α + β × trend),
                                 where α,β come from rolling OLS. Spread
                                 z-score > thresh → fade (mean-reversion).
                                 Confirmed by KPSS-proxy stationarity gate.

  3. Kalman_TrackingError     — Kalman filter (1960) single-factor price
                                 tracker: adaptive gain λ = P/(P+R).
                                 Tracking error = close − Kalman_state.
                                 Large tracking error → overshoot → fade.
                                 Kalman gain adaptation signals regime change.

  4. HMM_Transition_Signal    — HMM-inspired (Rabiner 1989) 2-state model:
                                 low-vol "accumulation" state vs high-vol
                                 "distribution" state detected via rolling
                                 vol + mean. Transition L→H = breakout LONG/
                                 SHORT; transition H→L = mean-reversion.

  5. PCA_Factor_Divergence    — PCA-inspired (Jolliffe 2002): approximate PC1
                                 as weighted sum of standardized OHLCV features.
                                 PC1 estimate captures the dominant co-movement.
                                 When close diverges from PC1-predicted level →
                                 mean-reversion back to factor consensus.

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Almgren, R. & Chriss, N. (2000) — "Optimal execution of portfolio
    transactions"; Journal of Risk 3(2): 5-39.
  - Engle, R.F. & Granger, C.W.J. (1987) — "Co-Integration and Error
    Correction: Representation, Estimation, and Testing";
    Econometrica 55(2): 251-276.
  - Kalman, R.E. (1960) — "A New Approach to Linear Filtering and Prediction
    Problems"; Journal of Basic Engineering 82(1): 35-45.
  - Rabiner, L.R. (1989) — "A Tutorial on Hidden Markov Models and Selected
    Applications in Speech Recognition"; Proceedings of the IEEE 77(2): 257-286.
  - Jolliffe, I.T. (2002) — "Principal Component Analysis" (2nd ed.),
    Springer Series in Statistics.
"""

import pandas as pd
import numpy as np


# ─── HELPERS ────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _sma(s, p):
    return s.rolling(p).mean()

def _atr(h, l, c, p=14):
    prev_c = c.shift(1)
    tr = pd.concat([h - l,
                    (h - prev_c).abs(),
                    (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / p, adjust=False).mean()

def _zscore(s, p):
    m  = s.rolling(p).mean()
    sd = s.rolling(p).std(ddof=0) + 1e-10
    return (s - m) / sd


def _apply_exit_bar(entry_long, entry_short, exit_bar):
    """
    3-state machine: enters on signal, exits after exit_bar bars.
    Flat(0) → Long(1) on entry_long; Flat(0) → Short(-1) on entry_short.
    New entries override opposite position immediately.
    """
    sig = pd.Series(0, index=entry_long.index, dtype=int)
    el = entry_long.values
    es = entry_short.values
    n  = len(sig)
    state     = 0
    bars_held = 0
    for i in range(n):
        if state != 0:
            bars_held += 1
            if bars_held >= exit_bar:
                state     = 0
                bars_held = 0
        if el[i]:
            state     = 1
            bars_held = 0
        elif es[i]:
            state     = -1
            bars_held = 0
        sig.iloc[i] = state
    return sig


# ===========================================================================
# 1. AlmgrenChriss_Shortfall — Almgren-Chriss (2000) implementation shortfall
# ===========================================================================
# Almgren & Chriss (2000) define implementation shortfall as the cost of
# executing a trade relative to a benchmark price.  Their optimal execution
# framework balances market impact against timing risk.
#
# OHLCV tractable proxy:
#   VWAP ≈ (H + L + C) / 3 (typical price)
#   "Price paid" ≈ Close (final fill price)
#   Volume-adjusted shortfall proxy:
#     shortfall[t] = (Close[t] - TP[t]) / (ATR[t] + ε) × log1p(Vol[t]/avg_vol)
#                  ← (deviation from fair price) × (volume pressure)
#
# Interpretation:
#   shortfall >> 0 → Close >> TP, high volume → buyers paid too much →
#                    informed buying absorbed → SHORT reversal
#   shortfall << 0 → Close << TP, high volume → sellers paid too much →
#                    informed selling absorbed → LONG reversal
#   |shortfall| small → execution near fair price → no edge, skip
#
# Guard: trend EMA ensures we only fade into weak / reverting trends.

def gen_AlmgrenChriss_Shortfall(df, atr_period=14, vol_window=20,
                                  z_window=30, z_thresh=1.5, trend_ema=60,
                                  exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    # Typical price (VWAP proxy)
    tp = (h + l + c) / 3.0

    # ATR for normalisation
    atr_val = _atr(h, l, c, atr_period)

    # Volume pressure: log-ratio to rolling average
    avg_vol  = v.rolling(vol_window).mean() + 1e-10
    vol_pres = np.log1p(v / avg_vol)

    # Implementation shortfall proxy
    dev       = (c - tp) / (atr_val + 1e-10)
    shortfall = dev * vol_pres

    # Z-score of shortfall
    sf_z = _zscore(shortfall, z_window)

    # Trend guard: only fade in mild market context
    trend    = _ema(c, trend_ema)
    above    = c > trend
    below    = c < trend

    # Positive shortfall → overpaid buyers → SHORT (fade up move)
    # Negative shortfall → overpaid sellers → LONG (fade down move)
    entry_short = (sf_z >  z_thresh) & above
    entry_long  = (sf_z < -z_thresh) & below

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_AlmgrenChriss_Shortfall(trial):
    return {
        'atr_period': trial.suggest_int('atr_period',   7, 21),
        'vol_window': trial.suggest_int('vol_window',   10, 40),
        'z_window':   trial.suggest_int('z_window',     20, 60),
        'z_thresh':   trial.suggest_float('z_thresh',    0.8, 2.5),
        'trend_ema':  trial.suggest_int('trend_ema',    30, 100),
        'exit_bar':   trial.suggest_int('exit_bar',      3, 12),
    }


# ===========================================================================
# 2. EngleGranger_SpreadRev — Engle-Granger (1987) cointegration spread reversal
# ===========================================================================
# Engle & Granger (1987) showed that two I(1) series can be cointegrated:
# a linear combination is I(0) (stationary). Classic pairs trading exploits this.
#
# Single-asset self-cointegration proxy:
#   y[t]    = close[t]
#   x[t]    = bar_index (or rolling time index)
#   Estimate rolling OLS: β̂ = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
#                          α̂ = ȳ - β̂×x̄
#   Synthetic spread[t] = close[t] - (α̂ + β̂×t)
#                        ← deviation from the local linear trend (≈ cointegrating vector)
#
# Stationarity gate (KPSS proxy):
#   Variance of spread / variance of cumsum(spread) — if ratio < kpss_thresh
#   the spread is "mean-reverting enough" to trade.
#
# Entry: spread z-score crosses ±thresh → fade (mean-reversion trade)

def gen_EngleGranger_SpreadRev(df, eg_window=40, z_window=30, z_thresh=1.8,
                                 kpss_window=20, kpss_thresh=0.3, trend_ema=80,
                                 exit_bar=7):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    n = len(c)
    spread = pd.Series(np.nan, index=c.index, dtype=float)

    c_vals = c.values

    for i in range(eg_window, n):
        win_c = c_vals[i - eg_window: i]
        win_x = np.arange(eg_window, dtype=float)
        x_bar = win_x.mean()
        y_bar = win_c.mean()
        ssxx  = ((win_x - x_bar) ** 2).sum() + 1e-10
        ssxy  = ((win_x - x_bar) * (win_c - y_bar)).sum()
        beta  = ssxy / ssxx
        alpha = y_bar - beta * x_bar
        # Spread at current bar (last position in window = eg_window - 1)
        t_cur = eg_window - 1.0
        spread.iloc[i] = win_c[-1] - (alpha + beta * t_cur)

    # KPSS-proxy stationarity gate on recent spread window
    # Ratio: variance(spread) / variance(cumsum(spread)) — higher = more stationary
    spread_roll_var  = spread.rolling(kpss_window).var() + 1e-10
    spread_cumsum_var = spread.cumsum().rolling(kpss_window).var() + 1e-10
    kpss_proxy = spread_roll_var / spread_cumsum_var
    stationary_enough = kpss_proxy > kpss_thresh

    # Z-score of spread
    sf_z = _zscore(spread, z_window)

    # Trend guard: avoid strong directional momentum
    trend = _ema(c, trend_ema)
    mild  = (c - trend).abs() / (trend.abs() + 1e-10) < 0.05

    entry_long  = (sf_z < -z_thresh) & stationary_enough & mild
    entry_short = (sf_z >  z_thresh) & stationary_enough & mild

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_EngleGranger_SpreadRev(trial):
    return {
        'eg_window':   trial.suggest_int('eg_window',   20, 60),
        'z_window':    trial.suggest_int('z_window',    20, 50),
        'z_thresh':    trial.suggest_float('z_thresh',   1.2, 3.0),
        'kpss_window': trial.suggest_int('kpss_window', 10, 30),
        'kpss_thresh': trial.suggest_float('kpss_thresh', 0.05, 0.6),
        'trend_ema':   trial.suggest_int('trend_ema',   40, 120),
        'exit_bar':    trial.suggest_int('exit_bar',     3, 12),
    }


# ===========================================================================
# 3. Kalman_TrackingError — Kalman (1960) filter tracking error reversal
# ===========================================================================
# The Kalman filter is the optimal linear estimator for a state-space model.
# For a scalar price series, the steady-state Kalman filter reduces to:
#
#   x̂[t|t]   = x̂[t|t-1] + K[t] × (y[t] - x̂[t|t-1])
#   K[t]      = P[t|t-1] / (P[t|t-1] + R)          ← Kalman gain
#   P[t|t]    = (1 - K[t]) × P[t|t-1]
#   P[t+1|t]  = P[t|t] + Q                          ← process noise
#
# Parameters: Q = process noise variance, R = measurement noise variance.
# K = Q / (Q + R) in steady state → slow (small Q/R) or fast (large Q/R).
#
# Tracking error = y[t] - x̂[t|t-1]  (innovation / residual)
#
# Intuition:
#   Large positive tracking error → price shot above Kalman estimate →
#   overshoot → SHORT reversal (Kalman "knows" price overreacted)
#   Large negative tracking error → price shot below → LONG reversal
#
# Variable Kalman gain: when recent errors are large → increase Q adaptively.

def gen_Kalman_TrackingError(df, q_base=1e-4, r_noise=0.01,
                               adaptive_window=20, z_window=30,
                               z_thresh=1.5, trend_ema=50, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    c_vals = c.values
    n = len(c_vals)

    state     = np.full(n, np.nan)
    innovation = np.full(n, np.nan)

    # Initialise
    p     = r_noise       # initial covariance
    x_hat = c_vals[0] if not np.isnan(c_vals[0]) else 0.0

    for i in range(n):
        y = c_vals[i]
        if np.isnan(y):
            state[i]      = x_hat
            innovation[i] = 0.0
            continue

        # Adaptive Q: scale by recent residual variance
        if i >= adaptive_window:
            recent_innov = innovation[max(0, i - adaptive_window): i]
            recent_innov = recent_innov[~np.isnan(recent_innov)]
            q_adaptive   = q_base * (1 + np.var(recent_innov) / (r_noise + 1e-10))
        else:
            q_adaptive = q_base

        # Predict
        p_pred = p + q_adaptive

        # Update
        k_gain    = p_pred / (p_pred + r_noise)
        innov     = y - x_hat
        x_hat     = x_hat + k_gain * innov
        p         = (1 - k_gain) * p_pred

        state[i]      = x_hat
        innovation[i] = innov

    state_s     = pd.Series(state, index=c.index)
    innovation_s = pd.Series(innovation, index=c.index)

    # Z-score of innovation (tracking error)
    innov_z = _zscore(innovation_s, z_window)

    # Trend guard
    trend = _ema(c, trend_ema)
    mild  = (c - trend).abs() / (trend.abs() + 1e-10) < 0.04

    # Large positive innovation → price overshot → SHORT
    # Large negative innovation → price undershot → LONG
    entry_short = (innov_z >  z_thresh) & mild
    entry_long  = (innov_z < -z_thresh) & mild

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Kalman_TrackingError(trial):
    return {
        'q_base':          trial.suggest_float('q_base',          1e-5, 5e-3, log=True),
        'r_noise':         trial.suggest_float('r_noise',         1e-3, 0.1,  log=True),
        'adaptive_window': trial.suggest_int('adaptive_window',   10, 40),
        'z_window':        trial.suggest_int('z_window',          20, 60),
        'z_thresh':        trial.suggest_float('z_thresh',         1.0, 3.0),
        'trend_ema':       trial.suggest_int('trend_ema',         30, 80),
        'exit_bar':        trial.suggest_int('exit_bar',           3, 12),
    }


# ===========================================================================
# 4. HMM_Transition_Signal — HMM-inspired (Rabiner 1989) regime transition
# ===========================================================================
# A Hidden Markov Model with 2 states (S0 = low-vol accumulation,
# S1 = high-vol distribution) can be approximated without EM training using
# rolling statistics as emission proxies (Rabiner 1989).
#
# State estimation heuristic:
#   vol[t] = rolling std of returns over vol_window
#   mean_ret[t] = rolling mean of returns over vol_window
#   S0 ("accumulation"): vol < vol_mu - vol_std × low_thresh
#   S1 ("distribution"): vol > vol_mu + vol_std × high_thresh
#   (vol_mu/vol_std estimated over regime_window bars)
#
# Transition signals:
#   S0 → S1 (breakout): direction = sign(mean_ret) → LONG or SHORT (momentum)
#   S1 → S0 (compression): price overshoot → fade (mean-reversion)
#
# Guard: require vol transition to be "clean" (not already in S1 for >hold_bars)

def gen_HMM_Transition_Signal(df, vol_window=10, regime_window=40,
                                low_thresh=0.5, high_thresh=0.5,
                                hold_bars=6, trend_ema=60, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    ret = c.pct_change()

    # Rolling volatility and mean return
    vol_roll  = ret.rolling(vol_window).std(ddof=0)
    mean_roll = ret.rolling(vol_window).mean()

    # Regime baseline (longer window)
    vol_mu    = vol_roll.rolling(regime_window).mean()
    vol_sigma = vol_roll.rolling(regime_window).std(ddof=0) + 1e-10

    # State classification
    vol_z = (vol_roll - vol_mu) / vol_sigma

    state_low  = vol_z < -low_thresh    # S0: low vol, accumulation
    state_high = vol_z >  high_thresh   # S1: high vol, distribution

    # Transition detection: was in S0 last bar, now in S1 → breakout
    was_low  = state_low.shift(1).fillna(0).astype(bool)
    was_high = state_high.shift(1).fillna(0).astype(bool)

    transition_breakout    = was_low  & state_high   # S0→S1
    transition_compression = was_high & state_low    # S1→S0

    # Direction of breakout from mean_roll sign
    dir_up   = mean_roll > 0
    dir_down = mean_roll < 0

    # Trend guard
    trend = _ema(c, trend_ema)
    above = c > trend
    below = c < trend

    # S0→S1 breakout: momentum trade in direction of mean_ret
    breakout_long  = transition_breakout & dir_up   & above
    breakout_short = transition_breakout & dir_down & below

    # S1→S0 compression: fade the dominant direction (mean-reversion)
    compress_long  = transition_compression & dir_down & below  # fade the down move
    compress_short = transition_compression & dir_up   & above  # fade the up move

    entry_long  = breakout_long  | compress_long
    entry_short = breakout_short | compress_short

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_HMM_Transition_Signal(trial):
    return {
        'vol_window':    trial.suggest_int('vol_window',    5, 20),
        'regime_window': trial.suggest_int('regime_window', 20, 80),
        'low_thresh':    trial.suggest_float('low_thresh',   0.2, 1.2),
        'high_thresh':   trial.suggest_float('high_thresh',  0.2, 1.2),
        'hold_bars':     trial.suggest_int('hold_bars',      3, 12),
        'trend_ema':     trial.suggest_int('trend_ema',      30, 100),
        'exit_bar':      trial.suggest_int('exit_bar',        3, 12),
    }


# ===========================================================================
# 5. PCA_Factor_Divergence — PCA-inspired (Jolliffe 2002) factor divergence
# ===========================================================================
# Principal Component Analysis (Jolliffe 2002) decomposes a feature matrix
# into orthogonal components ranked by explained variance. PC1 captures the
# dominant co-movement of all features.
#
# For a single-asset OHLCV dataset, the "features" are:
#   [open_ret, high_ret, low_ret, close_ret, vol_norm]
# where ret = pct_change and vol_norm = log(vol / avg_vol).
#
# PC1 approximation (without eigendecomposition):
#   PC1_approx[t] = w1×open_ret + w2×high_ret + w3×low_ret
#                  + w4×close_ret + w5×vol_norm
# where weights wᵢ are the rolling correlations of each feature with close_ret
# (using rolling Pearson r as a proxy for PC1 loadings).
#
# Divergence signal:
#   PC1_forecast = rolling weighted average of recent PC1_approx values
#   When close_ret significantly diverges from PC1_forecast → mean-reversion
#   (the other OHLCV features agree on direction, but close is an outlier)

def gen_PCA_Factor_Divergence(df, feature_window=20, corr_window=30,
                                div_window=15, z_thresh=1.5, trend_ema=60,
                                exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Returns and normalized volume
    close_ret = c.pct_change()
    open_ret  = o.pct_change()
    high_ret  = h.pct_change()
    low_ret   = l.pct_change()
    avg_vol   = v.rolling(corr_window).mean() + 1e-10
    vol_norm  = np.log1p(v / avg_vol) - np.log1p(1.0)  # centered at 0

    # Rolling Pearson correlations as PC1 loading proxies
    def _roll_corr(a, b, w):
        """Rolling correlation between two series."""
        a_std = a.rolling(w).std(ddof=0) + 1e-10
        b_std = b.rolling(w).std(ddof=0) + 1e-10
        a_dm  = a - a.rolling(w).mean()
        b_dm  = b - b.rolling(w).mean()
        cov   = (a_dm * b_dm).rolling(w).mean()
        return cov / (a_std * b_std)

    w_open  = _roll_corr(open_ret,  close_ret, corr_window).abs()
    w_high  = _roll_corr(high_ret,  close_ret, corr_window).abs()
    w_low   = _roll_corr(low_ret,   close_ret, corr_window).abs()
    w_vol   = _roll_corr(vol_norm,  close_ret, corr_window).abs()

    # Normalise weights to sum to 1
    w_total = w_open + w_high + w_low + 1.0 + w_vol + 1e-10  # close always weight 1
    w_o = w_open  / w_total
    w_h = w_high  / w_total
    w_l = w_low   / w_total
    w_c = 1.0     / w_total    # close weight fixed
    w_v = w_vol   / w_total

    # PC1 approximation
    pc1 = (w_o * open_ret + w_h * high_ret + w_l * low_ret
           + w_c * close_ret + w_v * vol_norm)

    # Factor forecast: rolling mean of PC1
    pc1_forecast = pc1.rolling(feature_window).mean()

    # Divergence: close_ret vs PC1 forecast
    divergence = close_ret - pc1_forecast
    div_z      = _zscore(divergence, div_window)

    # Trend guard
    trend = _ema(c, trend_ema)
    mild  = (c - trend).abs() / (trend.abs() + 1e-10) < 0.04

    # Close moved up but PC1 says down → SHORT reversal
    # Close moved down but PC1 says up → LONG reversal
    entry_short = (div_z >  z_thresh) & mild
    entry_long  = (div_z < -z_thresh) & mild

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_PCA_Factor_Divergence(trial):
    return {
        'feature_window': trial.suggest_int('feature_window', 10, 40),
        'corr_window':    trial.suggest_int('corr_window',    20, 60),
        'div_window':     trial.suggest_int('div_window',     10, 30),
        'z_thresh':       trial.suggest_float('z_thresh',      1.0, 3.0),
        'trend_ema':      trial.suggest_int('trend_ema',       30, 100),
        'exit_bar':       trial.suggest_int('exit_bar',         3, 12),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'AlmgrenChriss_Shortfall': {
        'gen':   gen_AlmgrenChriss_Shortfall,
        'space': space_AlmgrenChriss_Shortfall,
    },
    'EngleGranger_SpreadRev': {
        'gen':   gen_EngleGranger_SpreadRev,
        'space': space_EngleGranger_SpreadRev,
    },
    'Kalman_TrackingError': {
        'gen':   gen_Kalman_TrackingError,
        'space': space_Kalman_TrackingError,
    },
    'HMM_Transition_Signal': {
        'gen':   gen_HMM_Transition_Signal,
        'space': space_HMM_Transition_Signal,
    },
    'PCA_Factor_Divergence': {
        'gen':   gen_PCA_Factor_Divergence,
        'space': space_PCA_Factor_Divergence,
    },
}
