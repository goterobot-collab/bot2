#!/usr/bin/env python3
"""
TV2 BATCH MICRO14 — 5 microstructure strategies (5m/15m optimized).
Domain: Empirical Mode Decomposition, Transfer Entropy, Hilbert Transform,
        Optimal Transport (Wasserstein), Lyapunov Exponents.
(academic, distinct from MICRO1-13 + batches 3237-3318)

Strategies:
  1. EMD_IMF_Rev           — Empirical Mode Decomposition (Huang et al. 1998):
                              price decomposed into IMFs via upper/lower envelope.
                              When IMF-1 z-score at extreme → mean-reversion snap.
  2. TransferEntropy_Vol   — Schreiber (2000) transfer entropy proxy: directional
                              information flow from volume to price. High agreement
                              rate → follow volume; low agreement → contrarian fade.
  3. HilbertPhase_Rev      — Ehlers (2001) discrete Hilbert Transform: instantaneous
                              phase of detrended price. Phase near cycle top/bottom
                              with slow phase rate → reversal entry.
  4. WassersteinShift_Break — Villani (2003) optimal transport: Wasserstein-1
                              distance between recent and historical return
                              distributions. High W1 = regime shift → momentum
                              in direction of distributional divergence.
  5. LyapunovPred_Rev      — Oseledets (1968) local Lyapunov exponent proxy:
                              log-ratio of trajectory distances. Negative (convergent
                              = predictable) regime + price at BB extreme → reversal.

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Huang, N.E. et al. (1998) — "The empirical mode decomposition and the
    Hilbert spectrum for nonlinear and non-stationary time series analysis";
    Proc. Royal Society of London A, 454: 903-995.
  - Schreiber, T. (2000) — "Measuring Information Transfer";
    Physical Review Letters 85(2): 461-464.
  - Ehlers, J.F. (2001) — "Rocket Science for Traders: Digital Signal
    Processing Applications"; Wiley Trading.
  - Boashash, B. (1992) — "Estimating and interpreting the instantaneous
    frequency of a signal"; Proc. IEEE 80(4): 520-568.
  - Villani, C. (2003) — "Topics in Optimal Transportation";
    Graduate Studies in Mathematics Vol. 58, AMS.
  - Oseledets, V.I. (1968) — "A multiplicative ergodic theorem. Lyapunov
    characteristic numbers for dynamical systems"; Trans. Moscow Math. Soc.
    19: 197-231.
  - Brock, W.A., Dechert, W.D. & Scheinkman, J.A. (1987) — "A test for
    independence based on the correlation dimension"; University of
    Wisconsin-Madison, SSRI Working Paper 8702.
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

def _bb(c, p=20, ndev=2.0):
    mid = c.rolling(p).mean()
    sd  = c.rolling(p).std(ddof=0)
    return mid - ndev * sd, mid, mid + ndev * sd

def _apply_exit_bar(entry_long, entry_short, exit_bar):
    """
    3-state machine: enters on signal, exits after exit_bar bars.
    Flat(0) -> Long(1) on entry_long; Flat(0) -> Short(-1) on entry_short.
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
# 1. EMD_IMF_Rev — Empirical Mode Decomposition intrinsic mode reversal
# ===========================================================================
# Huang et al. (1998) EMD decomposes a signal into Intrinsic Mode Functions
# (IMFs) ordered from highest to lowest frequency. The first IMF captures
# local oscillations (intraday microstructure noise).
#
# Tractable proxy (no iterative sifting required):
#   upper_env  = rolling max(close, env_period)   ← local maxima envelope
#   lower_env  = rolling min(close, env_period)   ← local minima envelope
#   mean_env   = (upper_env + lower_env) / 2      ← mean envelope
#   imf1_proxy = close - mean_env                 ← 1st IMF approximation
#
# Signal:
#   imf1_z = z-score of imf1_proxy over zscore_period bars
#   imf1_z < -z_thresh (below mean envelope, trend up)  → LONG reversal
#   imf1_z >  z_thresh (above mean envelope, trend down) → SHORT reversal
#
# Trend guard: only fade if EMA trend supports recovery direction.

def gen_EMD_IMF_Rev(df, env_period=12, zscore_period=30, z_thresh=1.8,
                    trend_ema=50, vol_window=20, vol_z_min=0.0, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    # Upper and lower envelopes (local extrema approximation)
    upper_env = c.rolling(env_period).max()
    lower_env = c.rolling(env_period).min()
    mean_env  = (upper_env + lower_env) / 2.0

    # 1st IMF proxy: deviation from mean envelope
    imf1   = c - mean_env
    imf1_z = _zscore(imf1, zscore_period)

    # Trend context
    trend      = _ema(c, trend_ema)
    in_uptrend = c > trend

    # Volume guard (optional — vol_z_min=0 disables)
    vol_z = _zscore(v, vol_window)

    # Reversal: IMF overshoots → fade back toward mean envelope
    entry_long  = (imf1_z < -z_thresh) & in_uptrend  & (vol_z >= vol_z_min)
    entry_short = (imf1_z >  z_thresh) & ~in_uptrend & (vol_z >= vol_z_min)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_EMD_IMF_Rev(trial):
    return {
        'env_period':    trial.suggest_int('env_period',     8, 20),
        'zscore_period': trial.suggest_int('zscore_period', 20, 50),
        'z_thresh':      trial.suggest_float('z_thresh',    1.2, 2.8),
        'trend_ema':     trial.suggest_int('trend_ema',     30, 80),
        'vol_window':    trial.suggest_int('vol_window',    15, 35),
        'vol_z_min':     trial.suggest_float('vol_z_min',   0.0, 1.0),
        'exit_bar':      trial.suggest_int('exit_bar',       3, 12),
    }


# ===========================================================================
# 2. TransferEntropy_Vol — Schreiber (2000) directional information flow
# ===========================================================================
# Transfer entropy T(X→Y) quantifies how much knowing the past of X reduces
# uncertainty about the next state of Y, beyond what Y's own past explains.
# Schreiber (2000) showed T(vol→price) > T(price→vol) in liquid markets
# during informed-trading episodes.
#
# Tractable proxy (no full density estimation):
#   For each lag k = 1..lag:
#     agreement[k, t] = sign(vol_chg[t-k]) == sign(ret[t-k+1])  (1/0)
#   agree_rate = rolling mean of Σ agreement[k] / lag
#
#   agree_rate >> 0.5 → volume consistently led price (information flow active)
#     → follow current volume direction as leading indicator
#   agree_rate << 0.5 → volume was anti-predictive (contrarian signal)
#     → fade current volume direction

def gen_TransferEntropy_Vol(df, lag=5, te_window=30, flow_thresh=0.65,
                             contra_thresh=0.35, price_ema=40, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret     = c.pct_change()
    vol_chg = v.pct_change()

    # Lag-k agreement: did sign(vol[t-k]) match sign(ret[t-k+1])?
    agreement_sum = pd.Series(0.0, index=c.index)
    for k in range(1, lag + 1):
        vol_sign = np.sign(vol_chg.shift(k))
        ret_sign = np.sign(ret.shift(k - 1))
        agreement_sum += (vol_sign == ret_sign).astype(float)

    # Rolling agreement rate (fraction of lags where vol predicted price)
    agree_rate = agreement_sum.rolling(te_window).mean() / lag

    # Current volume direction
    vol_up   = vol_chg > 0
    vol_down = vol_chg < 0

    # Trend context
    price_trend = _ema(c, price_ema)
    up_trend    = c > price_trend

    # Information flow active → follow volume direction (with trend)
    flow_long  = (agree_rate > flow_thresh) & vol_up   & up_trend
    flow_short = (agree_rate > flow_thresh) & vol_down & ~up_trend

    # Contrarian: agreement reversed → fade volume direction
    contra_long  = (agree_rate < contra_thresh) & vol_down & up_trend
    contra_short = (agree_rate < contra_thresh) & vol_up   & ~up_trend

    entry_long  = flow_long  | contra_long
    entry_short = flow_short | contra_short

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_TransferEntropy_Vol(trial):
    return {
        'lag':           trial.suggest_int('lag',             3,  8),
        'te_window':     trial.suggest_int('te_window',      20, 50),
        'flow_thresh':   trial.suggest_float('flow_thresh',  0.58, 0.82),
        'contra_thresh': trial.suggest_float('contra_thresh', 0.18, 0.42),
        'price_ema':     trial.suggest_int('price_ema',      25, 60),
        'exit_bar':      trial.suggest_int('exit_bar',        3, 12),
    }


# ===========================================================================
# 3. HilbertPhase_Rev — Ehlers (2001) discrete Hilbert Transform cycle reversal
# ===========================================================================
# The analytic signal of x(t) is A(t) = x(t) + j·H[x(t)] where H is the
# Hilbert Transform. Instantaneous phase φ(t) = arctan2(H[x], x).
# At cycle peaks: φ ≈ +π/2; at troughs: φ ≈ -π/2.
#
# Ehlers (2001) discrete approximation to H[x]:
#   Q(t) = (0.0962·x[t] + 0.5769·x[t-2] - 0.5769·x[t-4] - 0.0962·x[t-6])
#          × (0.075·period + 0.54)
#   I(t) = x(t)
#   phase = arctan2(Q_smooth, I_smooth)
#
# Price is first detrended by subtracting its EMA (to isolate the cycle).
#
# Signal:
#   phase >  +phase_thresh AND phase slowing (|dφ/dt| < turn_rate) → SHORT
#   phase <  -phase_thresh AND phase slowing                         → LONG
# Trend guard prevents fading a dominant trend direction.

def gen_HilbertPhase_Rev(df, detrend_ema=20, ht_smooth=5, phase_thresh=1.0,
                          turn_rate=0.15, trend_ema=50, vol_window=20,
                          vol_z_min=0.0, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    # Detrend: remove dominant EMA component (isolate cycle)
    trend   = _ema(c, detrend_ema)
    dtrd    = c - trend

    # Ehlers discrete Hilbert Transform (in-phase and quadrature)
    coeff = 0.075 * detrend_ema + 0.54
    Q = (0.0962 * dtrd
         + 0.5769 * dtrd.shift(2)
         - 0.5769 * dtrd.shift(4)
         - 0.0962 * dtrd.shift(6)) * coeff

    I = dtrd  # In-phase component = detrended price

    # Smooth both components
    I_sm = I.rolling(ht_smooth).mean()
    Q_sm = Q.rolling(ht_smooth).mean()

    # Instantaneous phase
    phase_arr = np.arctan2(Q_sm.values, I_sm.values + 1e-10)
    phase     = pd.Series(phase_arr, index=c.index)

    # Phase velocity (rate of change) — small = cycle is turning
    d_phase = phase.diff().abs()

    # Cycle turning condition
    turning = d_phase < turn_rate

    # Cycle top (phase near +π/2) → SHORT; cycle bottom (near -π/2) → LONG
    phase_top    = (phase >  phase_thresh) & turning
    phase_bottom = (phase < -phase_thresh) & turning

    # Trend guard: only reverse against cycle, not against dominant trend
    long_trend = c > _ema(c, trend_ema)

    # Volume guard
    vol_z = _zscore(v, vol_window)

    entry_long  = phase_bottom & long_trend  & (vol_z >= vol_z_min)
    entry_short = phase_top    & ~long_trend & (vol_z >= vol_z_min)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_HilbertPhase_Rev(trial):
    return {
        'detrend_ema':  trial.suggest_int('detrend_ema',    10, 30),
        'ht_smooth':    trial.suggest_int('ht_smooth',       3, 10),
        'phase_thresh': trial.suggest_float('phase_thresh',  0.5, 1.6),
        'turn_rate':    trial.suggest_float('turn_rate',     0.05, 0.30),
        'trend_ema':    trial.suggest_int('trend_ema',       30, 80),
        'vol_window':   trial.suggest_int('vol_window',      15, 35),
        'vol_z_min':    trial.suggest_float('vol_z_min',     0.0, 1.0),
        'exit_bar':     trial.suggest_int('exit_bar',         3, 12),
    }


# ===========================================================================
# 4. WassersteinShift_Break — Wasserstein-1 distribution shift momentum
# ===========================================================================
# Villani (2003) optimal transport: Wasserstein-1 distance W1(P, Q) is the
# minimum "earth-moving cost" between two probability distributions.
# For 1D: W1(P, Q) = ∫|F_P(x) - F_Q(x)|dx = mean(|sort(P) - sort(Q)|).
#
# Market application:
#   recent_dist  = return distribution over last short_window bars
#   hist_dist    = return distribution over last long_window bars
#   W1 ≈ mean(|sorted(recent) - resample(sorted(hist), len(recent))|)
#
#   High W1 → current return distribution diverges from recent history
#           = regime shift is underway → momentum in direction of shift
#   Low W1  → stable distribution (regime unchanged) → reversion
#
# Direction: recent mean return vs long-term mean return signals shift up/down.

def gen_WassersteinShift_Break(df, short_window=15, long_window=60,
                                w1_z_window=40, w1_z_thresh=1.5,
                                trend_ema=40, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change()
    n   = len(ret)

    w1_vals = pd.Series(np.nan, index=c.index)
    ret_arr = ret.values
    sw      = short_window
    lw      = long_window

    for i in range(lw, n):
        r_short = ret_arr[i - sw + 1: i + 1]
        r_long  = ret_arr[i - lw + 1: i + 1]

        # Remove NaN
        rs = r_short[~np.isnan(r_short)]
        rl = r_long[~np.isnan(r_long)]

        if len(rs) < sw // 2 or len(rl) < lw // 2:
            continue

        # W1 approximation: sort both; resample long to len(short) via
        # uniform interpolation; take mean absolute difference
        rs_sorted = np.sort(rs)
        rl_sorted = np.sort(rl)
        indices   = np.linspace(0, len(rl_sorted) - 1, len(rs_sorted))
        rl_interp = np.interp(indices, np.arange(len(rl_sorted)), rl_sorted)

        w1_vals.iloc[i] = np.mean(np.abs(rs_sorted - rl_interp))

    # Z-score of W1 (regime-shift intensity)
    w1_z = _zscore(w1_vals, w1_z_window)

    # Direction of distributional shift
    ret_short_mean = ret.rolling(short_window).mean()
    ret_long_mean  = ret.rolling(long_window).mean()
    shift_up   = ret_short_mean > ret_long_mean
    shift_down = ret_short_mean < ret_long_mean

    # Trend guard
    trend = _ema(c, trend_ema)

    # High W1 → regime shift confirmed → momentum in shift direction
    entry_long  = (w1_z > w1_z_thresh) & shift_up   & (c > trend)
    entry_short = (w1_z > w1_z_thresh) & shift_down & (c < trend)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_WassersteinShift_Break(trial):
    return {
        'short_window': trial.suggest_int('short_window',   8, 25),
        'long_window':  trial.suggest_int('long_window',   40, 100),
        'w1_z_window':  trial.suggest_int('w1_z_window',   25, 60),
        'w1_z_thresh':  trial.suggest_float('w1_z_thresh', 1.0, 2.5),
        'trend_ema':    trial.suggest_int('trend_ema',      25, 70),
        'exit_bar':     trial.suggest_int('exit_bar',        4, 14),
    }


# ===========================================================================
# 5. LyapunovPred_Rev — Local Lyapunov exponent predictability reversal
# ===========================================================================
# Oseledets (1968) Lyapunov exponents measure divergence rate of nearby
# trajectories in phase space. Positive λ = chaotic (unpredictable);
# negative λ = convergent (predictable, bounded dynamics).
#
# Local Lyapunov proxy (Brock, Dechert & Scheinkman 1987):
#   λ_local(t) = log(|c[t] - c[t-k]|) - log(|c[t-1] - c[t-k-1]|)
#   If λ_local < 0: trajectory is converging (distance shrinking) → mean rev
#   If λ_local > 0: trajectory is diverging (distance growing)    → trend
#
# Rolling mean λ_roll(t) = mean(λ_local over lyap_window bars)
# Z-score of λ_roll identifies extreme convergence or divergence episodes.
#
# Signal:
#   λ_roll_z << 0 (highly convergent = predictable) AND price at BB extreme
#   → strong mean-reversion entry (system is "stuck" near equilibrium)
#   λ_roll_z >> 0 (highly divergent = chaotic) → no signal (avoid)
# Trend guard: reversion entry aligned with dominant EMA direction.

def gen_LyapunovPred_Rev(df, lyap_window=20, lyap_lag=3, lyap_z_window=40,
                          lyap_z_thresh=-1.0, bb_period=20, bb_dev=2.0,
                          trend_ema=50, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    # Local Lyapunov proxy: log-ratio of trajectory distances
    k    = lyap_lag
    d_t  = (c - c.shift(k)).abs() + 1e-10
    d_t1 = (c.shift(1) - c.shift(k + 1)).abs() + 1e-10

    lyap_inst = np.log(d_t / d_t1)

    # Rolling mean local Lyapunov exponent
    lyap_roll = lyap_inst.rolling(lyap_window).mean()
    lyap_z    = _zscore(lyap_roll, lyap_z_window)

    # Bollinger Bands for price extreme detection
    bb_lo, bb_mid, bb_hi = _bb(c, bb_period, bb_dev)

    # Trend context
    trend      = _ema(c, trend_ema)
    in_uptrend = c > trend

    # Convergent regime (negative Lyapunov → predictable → mean-rev expected)
    convergent = lyap_z < lyap_z_thresh

    # Price at extreme within convergent regime → reversion entry
    entry_long  = convergent & (c < bb_lo) & in_uptrend
    entry_short = convergent & (c > bb_hi) & ~in_uptrend

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_LyapunovPred_Rev(trial):
    return {
        'lyap_window':   trial.suggest_int('lyap_window',     12, 30),
        'lyap_lag':      trial.suggest_int('lyap_lag',          2,  6),
        'lyap_z_window': trial.suggest_int('lyap_z_window',    25, 60),
        'lyap_z_thresh': trial.suggest_float('lyap_z_thresh', -2.0, -0.3),
        'bb_period':     trial.suggest_int('bb_period',         15, 30),
        'bb_dev':        trial.suggest_float('bb_dev',           1.5, 2.5),
        'trend_ema':     trial.suggest_int('trend_ema',          30, 80),
        'exit_bar':      trial.suggest_int('exit_bar',            3, 12),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'EMD_IMF_Rev': {
        'gen':   gen_EMD_IMF_Rev,
        'space': space_EMD_IMF_Rev,
    },
    'TransferEntropy_Vol': {
        'gen':   gen_TransferEntropy_Vol,
        'space': space_TransferEntropy_Vol,
    },
    'HilbertPhase_Rev': {
        'gen':   gen_HilbertPhase_Rev,
        'space': space_HilbertPhase_Rev,
    },
    'WassersteinShift_Break': {
        'gen':   gen_WassersteinShift_Break,
        'space': space_WassersteinShift_Break,
    },
    'LyapunovPred_Rev': {
        'gen':   gen_LyapunovPred_Rev,
        'space': space_LyapunovPred_Rev,
    },
}
