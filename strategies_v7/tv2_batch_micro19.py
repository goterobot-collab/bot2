#!/usr/bin/env python3
"""
TV2 BATCH MICRO19 — 5 microstructure strategies (5m/15m optimized).
Domain: Parkinson Extreme-Value Vol (Parkinson 1980), Lehmann Return-Reversal
        (Lehmann 1990), Clark Subordinated Activity Mixture (Clark 1973),
        Grossman-Stiglitz Information Asymmetry (Grossman & Stiglitz 1980),
        Wang Volume Decomposition (Wang 1994) — all new vs MICRO1-18.

Strategies:
  1. Parkinson_VolRange       — Parkinson (1980) extreme-value vol estimator:
                                 PK = 1/(4·ln2) × (ln(H/L))².  Rolling z-score
                                 of PK detects vol spikes (fade) vs vol quiet
                                 directional breakouts (follow).

  2. Lehmann_ReturnReversal   — Lehmann (1990) return-reversal / contrarian:
                                 r_{t-1} z-score + volume confirmation.
                                 Models bid-ask bounce at microstructure level.

  3. Clark_ActivityMixture    — Clark (1973) subordinated stochastic process:
                                 volume-per-unit-return ratio separates efficient
                                 (high-vol, small-move) from noise (low-vol,
                                 large-move) bars.  Noise extreme → fade;
                                 efficient break → follow.

  4. GrossmanStiglitz_Info    — Grossman & Stiglitz (1980) information paradox:
                                 informed traders enter when price deviates from
                                 fundamental.  Actual_move / Kyle-proxy expected
                                 move classifies informed (trend) vs noise (fade).

  5. Wang_VolumeDecomp        — Wang (1994) volume decomposition:
                                 informed_vol = volume × |return_norm|;
                                 noise_vol    = volume × (1 − |return_norm|).
                                 Informed/Noise ratio z-score → momentum when
                                 informed dominates; mean-reversion when noise
                                 dominates.

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Parkinson, M. (1980) — "The Extreme Value Method for Estimating the Variance
    of the Rate of Return"; Journal of Business 53(1): 61-65.
  - Lehmann, B.N. (1990) — "Fads, Martingales, and Market Efficiency";
    Quarterly Journal of Economics 105(1): 1-28.
  - Clark, P.K. (1973) — "A Subordinated Stochastic Process Model with Finite
    Variance for Speculative Prices"; Econometrica 41(1): 135-155.
  - Grossman, S.J. & Stiglitz, J.E. (1980) — "On the Impossibility of
    Informationally Efficient Markets"; American Economic Review 70(3): 393-408.
  - Wang, J. (1994) — "A Model of Intertemporal Asset Prices Under Asymmetric
    Information"; Review of Economic Studies 61(2): 249-282.
"""

import pandas as pd
import numpy as np


# ─── HELPERS ────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

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
    """3-state machine: exits after exit_bar bars in position."""
    sig = pd.Series(0, index=entry_long.index, dtype=int)
    el = entry_long.values
    es = entry_short.values
    n  = len(sig)
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


# ===========================================================================
# 1. Parkinson_VolRange
# ===========================================================================
# Parkinson (1980): PK = 1/(4·ln2) × (ln(H/L))²
# High PK z-score (wide bar) → vol spike → fade close at extreme.
# Low PK z-score (narrow bar) + directional EMA break → breakout follow.

def gen_Parkinson_VolRange(df, pk_window=20, z_thresh_high=1.8,
                            z_thresh_low=-0.5, fade_pct=0.25,
                            break_pct=0.003, trend_ema=40, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    hl_ratio = (h / (l + 1e-10)).clip(lower=1.001)
    pk       = (1.0 / (4.0 * np.log(2))) * (np.log(hl_ratio) ** 2)
    pk_z     = _zscore(pk, pk_window)
    rng      = (h - l).clip(lower=1e-10)

    trend = _ema(c, trend_ema)
    slope = trend.diff(3)

    # HIGH vol spike: fade close near bar extreme
    vol_spike   = pk_z > z_thresh_high
    fade_long   = vol_spike & (c < l + rng * fade_pct)
    fade_short  = vol_spike & (c > h - rng * fade_pct)

    # LOW vol contraction: follow directional EMA break
    vol_quiet   = pk_z < z_thresh_low
    break_long  = vol_quiet & (c > trend * (1 + break_pct)) & (slope > 0)
    break_short = vol_quiet & (c < trend * (1 - break_pct)) & (slope < 0)

    entry_long  = fade_long  | break_long
    entry_short = fade_short | break_short

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Parkinson_VolRange(trial):
    return {
        'pk_window':     trial.suggest_int('pk_window',       10, 40),
        'z_thresh_high': trial.suggest_float('z_thresh_high',  1.0, 3.0),
        'z_thresh_low':  trial.suggest_float('z_thresh_low',  -1.5, 0.0),
        'fade_pct':      trial.suggest_float('fade_pct',       0.10, 0.40),
        'break_pct':     trial.suggest_float('break_pct',      0.001, 0.010),
        'trend_ema':     trial.suggest_int('trend_ema',        20, 80),
        'exit_bar':      trial.suggest_int('exit_bar',          3, 12),
    }


# ===========================================================================
# 2. Lehmann_ReturnReversal
# ===========================================================================
# Lehmann (1990): weekly losers outperform winners via bid-ask bounce.
# On 5m/15m: strong down bar t-1 → LONG reversal; strong up bar → SHORT.
# Volume confirmation: reversal on high volume = real inventory unwind.
# Guard: skip when EMA slope is too steep (trending market).

def gen_Lehmann_ReturnReversal(df, ret_window=20, rev_thresh=1.5,
                                 vol_window=20, vol_factor=1.0,
                                 trend_ema=50, slope_guard=0.002, exit_bar=5):
    c = df['close'].shift(1)
    v = df['volume'].shift(1)

    ret   = c.pct_change()
    ret_z = _zscore(ret, ret_window)

    avg_vol  = v.rolling(vol_window).mean() + 1e-10
    high_vol = v > avg_vol * vol_factor

    trend = _ema(c, trend_ema)
    slope = (trend - trend.shift(3)) / (trend.abs() + 1e-10)
    flat  = slope.abs() < slope_guard

    entry_long  = (ret_z < -rev_thresh) & high_vol & flat
    entry_short = (ret_z >  rev_thresh) & high_vol & flat

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Lehmann_ReturnReversal(trial):
    return {
        'ret_window':   trial.suggest_int('ret_window',    10, 40),
        'rev_thresh':   trial.suggest_float('rev_thresh',   0.8, 2.5),
        'vol_window':   trial.suggest_int('vol_window',    10, 40),
        'vol_factor':   trial.suggest_float('vol_factor',   0.5, 2.0),
        'trend_ema':    trial.suggest_int('trend_ema',     30, 100),
        'slope_guard':  trial.suggest_float('slope_guard',  0.0005, 0.005),
        'exit_bar':     trial.suggest_int('exit_bar',        3, 12),
    }


# ===========================================================================
# 3. Clark_ActivityMixture
# ===========================================================================
# Clark (1973): price changes follow a subordinated process directed by volume.
# activity_ratio = volume / (|bar_return/ATR| + ε)
# Low ratio (big move, little volume) → noise → FADE.
# High ratio (big volume, small move) → efficient pricing → FOLLOW direction.

def gen_Clark_ActivityMixture(df, atr_period=14, ar_window=30,
                                z_high=1.2, z_low=-0.8,
                                trend_ema=40, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    atr_val   = _atr(h, l, c, atr_period)
    ret_abs   = (c - o).abs() / (atr_val + 1e-10)
    act_ratio = v / (ret_abs + 1e-10)
    ar_z      = _zscore(act_ratio, ar_window)

    trend = _ema(c, trend_ema)
    above = c > trend
    below = c < trend

    # Noise: fade extreme move
    noisy       = ar_z < z_low
    fade_long   = noisy & below
    fade_short  = noisy & above

    # Efficient: follow direction with trend confirmation
    efficient    = ar_z > z_high
    follow_long  = efficient & above
    follow_short = efficient & below

    entry_long  = fade_long  | follow_long
    entry_short = fade_short | follow_short

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Clark_ActivityMixture(trial):
    return {
        'atr_period': trial.suggest_int('atr_period',   7, 21),
        'ar_window':  trial.suggest_int('ar_window',    15, 50),
        'z_high':     trial.suggest_float('z_high',      0.5, 2.0),
        'z_low':      trial.suggest_float('z_low',      -2.0, -0.3),
        'trend_ema':  trial.suggest_int('trend_ema',    20, 80),
        'exit_bar':   trial.suggest_int('exit_bar',      3, 12),
    }


# ===========================================================================
# 4. GrossmanStiglitz_Info
# ===========================================================================
# Grossman & Stiglitz (1980): informed traders enter when price deviates from
# fundamental.  Proxy via Kyle square-root law:
#   expected_move ∝ sqrt(volume) / avg(sqrt(volume))
#   excess_z = z-score(|bar_return| / expected_move)
# High excess → informed activity → FOLLOW bar direction.
# Low  excess → noise → FADE bar direction.

def gen_GrossmanStiglitz_Info(df, vol_window=20, exc_window=30,
                                z_informed=1.5, z_noise=-0.8,
                                trend_ema=50, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    bar_ret_abs   = (c - o).abs() / (c.abs() + 1e-10)
    sqrt_vol      = np.sqrt(v + 1e-10)
    expected_move = sqrt_vol / (sqrt_vol.rolling(vol_window).mean() + 1e-10)
    excess_move   = bar_ret_abs / (expected_move + 1e-10)
    excess_z      = _zscore(excess_move, exc_window)

    up_bar   = c > o
    down_bar = c < o

    trend = _ema(c, trend_ema)
    above = c > trend
    below = c < trend

    informed     = excess_z >  z_informed
    follow_long  = informed & up_bar   & above
    follow_short = informed & down_bar & below

    noisy       = excess_z < z_noise
    fade_long   = noisy & down_bar
    fade_short  = noisy & up_bar

    entry_long  = follow_long  | fade_long
    entry_short = follow_short | fade_short

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_GrossmanStiglitz_Info(trial):
    return {
        'vol_window':  trial.suggest_int('vol_window',    10, 40),
        'exc_window':  trial.suggest_int('exc_window',    15, 50),
        'z_informed':  trial.suggest_float('z_informed',   0.8, 2.5),
        'z_noise':     trial.suggest_float('z_noise',     -2.0, -0.3),
        'trend_ema':   trial.suggest_int('trend_ema',     30, 100),
        'exit_bar':    trial.suggest_int('exit_bar',        3, 12),
    }


# ===========================================================================
# 5. Wang_VolumeDecomp
# ===========================================================================
# Wang (1994): volume decomposes into informed and noise components.
#   ret_norm    = |c - o| / (ATR + ε)
#   informed_v  = volume × min(ret_norm, 1)
#   noise_v     = volume × (1 − min(ret_norm, 1))
#   ratio       = informed_v / (noise_v + ε)
# High ratio z-score → informed activity → FOLLOW direction.
# Low  ratio z-score → noise → FADE recent move.

def gen_Wang_VolumeDecomp(df, atr_period=14, ratio_window=30,
                            z_informed=1.3, z_noise=-0.8,
                            trend_ema=40, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    atr_val    = _atr(h, l, c, atr_period)
    ret_norm   = ((c - o).abs() / (atr_val + 1e-10)).clip(upper=2.0)

    informed_v = v * ret_norm.clip(upper=1.0)
    noise_v    = v * (1.0 - ret_norm.clip(upper=1.0)).clip(lower=0.0)

    ratio      = informed_v / (noise_v + 1e-10)
    ratio_z    = _zscore(ratio, ratio_window)

    up_bar   = c > o
    down_bar = c < o

    trend = _ema(c, trend_ema)
    above = c > trend
    below = c < trend

    informed     = ratio_z >  z_informed
    follow_long  = informed & up_bar   & above
    follow_short = informed & down_bar & below

    noisy       = ratio_z < z_noise
    fade_long   = noisy & down_bar
    fade_short  = noisy & up_bar

    entry_long  = follow_long  | fade_long
    entry_short = follow_short | fade_short

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Wang_VolumeDecomp(trial):
    return {
        'atr_period':    trial.suggest_int('atr_period',    7, 21),
        'ratio_window':  trial.suggest_int('ratio_window',  15, 50),
        'z_informed':    trial.suggest_float('z_informed',   0.8, 2.5),
        'z_noise':       trial.suggest_float('z_noise',     -2.0, -0.3),
        'trend_ema':     trial.suggest_int('trend_ema',     20, 80),
        'exit_bar':      trial.suggest_int('exit_bar',        3, 12),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Parkinson_VolRange': {
        'gen':   gen_Parkinson_VolRange,
        'space': space_Parkinson_VolRange,
    },
    'Lehmann_ReturnReversal': {
        'gen':   gen_Lehmann_ReturnReversal,
        'space': space_Lehmann_ReturnReversal,
    },
    'Clark_ActivityMixture': {
        'gen':   gen_Clark_ActivityMixture,
        'space': space_Clark_ActivityMixture,
    },
    'GrossmanStiglitz_Info': {
        'gen':   gen_GrossmanStiglitz_Info,
        'space': space_GrossmanStiglitz_Info,
    },
    'Wang_VolumeDecomp': {
        'gen':   gen_Wang_VolumeDecomp,
        'space': space_Wang_VolumeDecomp,
    },
}
