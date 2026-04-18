#!/usr/bin/env python3
"""
TV2 BATCH MICRO8 — 5 microstructure strategies (5m/15m optimized).
Domain: Spectral Analysis, Structural Break Detection & Causal Inference
(academic, distinct from MICRO1-7 + batches 3237-3318)

Strategies:
  1. DominantCycle    — Ehlers (2001) dominant cycle via DFT proxy: trade turning
                        points when cycle phase reaches extremes (0 or π)
  2. CUSUMBreak       — Page (1954) CUSUM change-point detector: detect mean shift
                        in returns, fade the overextension back to pre-break level
  3. GrangerVol       — Granger causality proxy: volume leads price by 1-3 bars
                        (Lo & MacKinlay 1990); trade when causal signal is strong
  4. SpectralMomRev   — Spectral momentum: FFT low-frequency energy dominance
                        signals cyclical exhaustion → mean reversion entry
  5. StructuralAnchor — Structural break anchor: Zivot-Andrews proxy identifies
                        local level break; price reverts to pre-break anchor

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Ehlers (2001) — "Rocket Science for Traders" dominant cycle measurement
  - Page (1954) — CUSUM sequential change-point detection
  - Granger (1969) — Investigating causal relations by econometric models
  - Lo & MacKinlay (1990) — An Econometric Analysis of Nonsynchronous Trading
  - Zivot & Andrews (1992) — Further evidence on the great crash and unit root
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
    Flat(0) → Long(1) on entry_long; Flat(0) → Short(-1) on entry_short.
    New entries override opposite position immediately.
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
# 1. DominantCycle — Ehlers dominant cycle turning point strategy
# ═══════════════════════════════════════════════════════════════════════════
# Ehlers (2001): Markets have dominant cycles. DFT peak frequency identifies
# the cycle period. Trade turning points at cycle phase extremes.
# Proxy: find dominant period via rolling autocorrelation peak.

def gen_DominantCycle(df, cycle_search_min=8, cycle_search_max=40,
                      phase_z_thresh=1.4, trend_ema=50, exit_bar=8):
    c = df['close'].shift(1)

    ret = c.pct_change().fillna(0)

    # Rolling dominant cycle: find lag with highest autocorrelation
    def _dominant_period(s_vals, search_min, search_max):
        result = np.full(len(s_vals), float(search_min + search_max) / 2)
        for i in range(search_max, len(s_vals)):
            seg = s_vals[i - search_max: i]
            mean_seg = seg.mean()
            var_seg = ((seg - mean_seg) ** 2).mean() + 1e-12
            best_lag = search_min
            best_corr = -1.0
            for lag in range(search_min, search_max + 1):
                # Autocorrelation at lag
                seg1 = seg[:-lag] - mean_seg
                seg2 = seg[lag:] - mean_seg
                if len(seg1) < 3:
                    continue
                corr = (seg1 * seg2).mean() / var_seg
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag
            result[i] = best_lag
        return result

    dom_period = _dominant_period(ret.values, cycle_search_min, cycle_search_max)
    dom_period_s = pd.Series(dom_period, index=c.index)

    # Smooth the dominant period estimate
    dom_period_smooth = dom_period_s.rolling(5).mean().fillna(dom_period_s)

    # Phase oscillator: price position within estimated cycle
    # Approximate by normalizing price within a rolling window = dom_period
    # Use rolling quantile-based position
    roll_window = dom_period_smooth.round().astype(int).clip(cycle_search_min, cycle_search_max)

    # Compute rolling Z-score with dynamic window (use median dom_period)
    median_period = int(dom_period_smooth.median())
    pz = _zscore(c, median_period)

    # Cycle turning points: Z-score crosses zero from extreme
    # Long: Z was < -phase_z_thresh and now crosses up
    pz_prev = pz.shift(1)
    entry_long  = (pz_prev < -phase_z_thresh) & (pz > pz_prev)
    entry_short = (pz_prev >  phase_z_thresh) & (pz < pz_prev)

    # Trend filter
    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.990)
    entry_short = entry_short & (c < trend * 1.010)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_DominantCycle(trial):
    return {
        'cycle_search_min': trial.suggest_int('cycle_search_min', 6, 12),
        'cycle_search_max': trial.suggest_int('cycle_search_max', 25, 50),
        'phase_z_thresh':   trial.suggest_float('phase_z_thresh', 1.0, 2.2, step=0.2),
        'trend_ema':        trial.suggest_int('trend_ema', 30, 70),
        'exit_bar':         trial.suggest_int('exit_bar', 5, 16),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. CUSUMBreak — CUSUM change-point mean reversion
# ═══════════════════════════════════════════════════════════════════════════
# Page (1954): CUSUM accumulates deviations from reference mean.
# When CUSUM exceeds threshold → structural shift detected.
# Post-break: price reverts to pre-break reference level → fade the shift.

def gen_CUSUMBreak(df, ref_window=30, cusum_k=0.5, cusum_h_mult=3.0,
                   zscore_period=20, zscore_thresh=1.2, exit_bar=8):
    c = df['close'].shift(1)

    ret = c.pct_change().fillna(0)
    ref_mean = ret.rolling(ref_window).mean().shift(1)
    ref_std  = ret.rolling(ref_window).std(ddof=0).shift(1).fillna(0.001) + 1e-10

    # Allowance k in std units
    k = cusum_k * ref_std

    # Two-sided CUSUM in return space
    cusum_pos = pd.Series(0.0, index=c.index)
    cusum_neg = pd.Series(0.0, index=c.index)
    cp_vals = cusum_pos.values.copy()
    cn_vals = cusum_neg.values.copy()
    ret_vals = ret.values
    rm_vals  = ref_mean.values
    k_vals   = k.values
    rs_vals  = ref_std.values

    for i in range(1, len(cp_vals)):
        # CUSUM positive: detects upward shift
        cp_vals[i] = max(0.0, cp_vals[i-1] + (ret_vals[i] - rm_vals[i] - k_vals[i]))
        # CUSUM negative: detects downward shift
        cn_vals[i] = max(0.0, cn_vals[i-1] - (ret_vals[i] - rm_vals[i] + k_vals[i]))

    cusum_pos = pd.Series(cp_vals, index=c.index)
    cusum_neg = pd.Series(cn_vals, index=c.index)

    # Threshold h: signal when CUSUM > h * ref_std
    h = cusum_h_mult * ref_std

    # Break detected
    break_up   = cusum_pos > h    # upward shift → price overshot → short
    break_down = cusum_neg > h    # downward shift → price undershoot → long

    # Additional Z-score confirmation: price must be at extreme
    pz = _zscore(c, zscore_period)

    entry_long  = break_down & (pz < -zscore_thresh)
    entry_short = break_up   & (pz >  zscore_thresh)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_CUSUMBreak(trial):
    return {
        'ref_window':     trial.suggest_int('ref_window', 20, 50),
        'cusum_k':        trial.suggest_float('cusum_k', 0.25, 1.0, step=0.25),
        'cusum_h_mult':   trial.suggest_float('cusum_h_mult', 2.0, 5.0, step=0.5),
        'zscore_period':  trial.suggest_int('zscore_period', 12, 28),
        'zscore_thresh':  trial.suggest_float('zscore_thresh', 0.8, 2.0, step=0.2),
        'exit_bar':       trial.suggest_int('exit_bar', 5, 16),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. GrangerVol — Volume-leads-price Granger causality proxy
# ═══════════════════════════════════════════════════════════════════════════
# Granger (1969): X Granger-causes Y if lagged X improves forecast of Y.
# Proxy: compare predictive power of lagged volume Z-score for price direction.
# When volume extremes reliably predict price direction → follow the lead.
# When volume and price diverge → mean reversion likely.

def gen_GrangerVol(df, vol_period=20, vol_lag=2, diverge_z_thresh=1.5,
                   trend_ema=40, exit_bar=7):
    c = df['close'].shift(1)
    v = df['volume'].shift(1)

    # Volume Z-score
    vol_z = _zscore(v, vol_period)

    # Lagged volume direction signal: high vol lag → expect price move
    vol_z_lag = vol_z.shift(vol_lag)

    # Price return Z-score
    ret = c.pct_change().fillna(0)
    ret_z = _zscore(ret, vol_period)

    # Granger-following: lagged vol Z extreme + current price starting to confirm
    # Lagged positive vol Z + price Z still negative → buy (price catching up)
    # Lagged negative vol Z + price Z still positive → sell (price catching up down)
    vol_lead_long  = (vol_z_lag >  diverge_z_thresh) & (ret_z < 0)
    vol_lead_short = (vol_z_lag < -diverge_z_thresh) & (ret_z > 0)

    # Volume-price divergence: price moved but volume didn't confirm → fade price move
    vol_div_fade_short = (ret_z >  diverge_z_thresh) & (vol_z_lag < 0.5)
    vol_div_fade_long  = (ret_z < -diverge_z_thresh) & (vol_z_lag > -0.5)

    # Trend filter
    trend = _ema(c, trend_ema)
    entry_long  = (vol_lead_long  | vol_div_fade_long)  & (c > trend * 0.985)
    entry_short = (vol_lead_short | vol_div_fade_short) & (c < trend * 1.015)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_GrangerVol(trial):
    return {
        'vol_period':       trial.suggest_int('vol_period', 12, 35),
        'vol_lag':          trial.suggest_int('vol_lag', 1, 4),
        'diverge_z_thresh': trial.suggest_float('diverge_z_thresh', 1.0, 2.5, step=0.25),
        'trend_ema':        trial.suggest_int('trend_ema', 25, 60),
        'exit_bar':         trial.suggest_int('exit_bar', 4, 14),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. SpectralMomRev — Spectral momentum exhaustion reversal
# ═══════════════════════════════════════════════════════════════════════════
# Spectral analysis: decompose price into low/high frequency components.
# When low-frequency (trend) component dominates → cycle near peak → fade.
# When high-frequency (noise) component dominates → random walk → wait.
# Proxy: compare smoothed vs raw price momentum energy.

def gen_SpectralMomRev(df, fast_smooth=5, slow_smooth=20, energy_window=30,
                       dom_thresh=0.70, zscore_period=20, zscore_thresh=1.3,
                       exit_bar=8):
    c = df['close'].shift(1)

    ret = c.pct_change().fillna(0)

    # Low-freq component (trend): EMA-smoothed returns
    ret_low  = _ema(ret, slow_smooth)   # smooth = low frequency
    ret_high = ret - _ema(ret, fast_smooth)  # residual = high frequency

    # Energy in each component (rolling variance)
    energy_low  = ret_low.rolling(energy_window).var() + 1e-14
    energy_high = (ret_high.rolling(energy_window).var() + 1e-14)
    total_energy = energy_low + energy_high

    # Fraction of energy in low-frequency component
    low_energy_frac = energy_low / total_energy

    # High low-freq dominance → cyclical structure → near turning point
    cycle_dominant = low_energy_frac > dom_thresh

    # Price Z-score for extreme detection
    pz = _zscore(c, zscore_period)

    # Low-freq momentum direction
    mom_dir = np.sign(ret_low)

    # Trade: cycle dominant + price at extreme in direction of low-freq → reversal
    # (price has been pushed by the cycle, now at peak → fade)
    entry_long  = cycle_dominant & (pz < -zscore_thresh) & (mom_dir <= 0)
    entry_short = cycle_dominant & (pz >  zscore_thresh) & (mom_dir >= 0)

    # Momentum follow when high-freq dominant (trend regime)
    trend_dominant = low_energy_frac < (1.0 - dom_thresh)
    entry_long  = entry_long  | (trend_dominant & (pz > 0.5) & (mom_dir > 0))
    entry_short = entry_short | (trend_dominant & (pz < -0.5) & (mom_dir < 0))

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_SpectralMomRev(trial):
    return {
        'fast_smooth':    trial.suggest_int('fast_smooth', 3, 10),
        'slow_smooth':    trial.suggest_int('slow_smooth', 14, 35),
        'energy_window':  trial.suggest_int('energy_window', 20, 50),
        'dom_thresh':     trial.suggest_float('dom_thresh', 0.55, 0.80, step=0.05),
        'zscore_period':  trial.suggest_int('zscore_period', 14, 30),
        'zscore_thresh':  trial.suggest_float('zscore_thresh', 1.0, 2.0, step=0.2),
        'exit_bar':       trial.suggest_int('exit_bar', 5, 16),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. StructuralAnchor — Structural break anchor reversion
# ═══════════════════════════════════════════════════════════════════════════
# Zivot & Andrews (1992): structural breaks occur when regime changes.
# Post-break: the old level acts as a magnetic anchor (fair value anchor).
# When price deviates too far from its last structural anchor → reversion.
# Proxy: detect level breaks via EWMA divergence, use anchor as target.

def gen_StructuralAnchor(df, anchor_period=40, break_z_thresh=2.0,
                         anchor_smooth=10, revert_z_thresh=1.4,
                         trend_ema=50, exit_bar=9):
    c = df['close'].shift(1)

    # Structural anchor: slow EMA as "consensus value" before a break
    anchor = _ema(c, anchor_period)
    anchor_prev = anchor.shift(anchor_smooth)   # anchor from anchor_smooth bars ago

    # Break detection: price has moved far from where anchor was
    anchor_ret = (c - anchor_prev) / (anchor_prev.abs() + 1e-10)
    anchor_z = _zscore(anchor_ret, anchor_period)

    break_up   = anchor_z >  break_z_thresh   # price broke up from anchor
    break_down = anchor_z < -break_z_thresh   # price broke down from anchor

    # Distance from current anchor (reference level)
    current_dist_z = _zscore(c - anchor, anchor_period)

    # Post-break reversion: price too far from anchor in break direction → fade
    # Long: broke down, now price extreme below anchor → revert up
    # Short: broke up, now price extreme above anchor → revert down
    entry_long  = break_down & (current_dist_z < -revert_z_thresh)
    entry_short = break_up   & (current_dist_z >  revert_z_thresh)

    # Trend filter: only trade against break if macro trend supports
    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.980)
    entry_short = entry_short & (c < trend * 1.020)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_StructuralAnchor(trial):
    return {
        'anchor_period':    trial.suggest_int('anchor_period', 25, 60),
        'break_z_thresh':   trial.suggest_float('break_z_thresh', 1.5, 3.0, step=0.25),
        'anchor_smooth':    trial.suggest_int('anchor_smooth', 5, 20),
        'revert_z_thresh':  trial.suggest_float('revert_z_thresh', 1.0, 2.2, step=0.2),
        'trend_ema':        trial.suggest_int('trend_ema', 30, 70),
        'exit_bar':         trial.suggest_int('exit_bar', 5, 18),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'DominantCycle': {
        'gen':   gen_DominantCycle,
        'space': space_DominantCycle,
    },
    'CUSUMBreak': {
        'gen':   gen_CUSUMBreak,
        'space': space_CUSUMBreak,
    },
    'GrangerVol': {
        'gen':   gen_GrangerVol,
        'space': space_GrangerVol,
    },
    'SpectralMomRev': {
        'gen':   gen_SpectralMomRev,
        'space': space_SpectralMomRev,
    },
    'StructuralAnchor': {
        'gen':   gen_StructuralAnchor,
        'space': space_StructuralAnchor,
    },
}
