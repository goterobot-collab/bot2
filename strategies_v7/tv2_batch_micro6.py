#!/usr/bin/env python3
"""
TV2 BATCH MICRO6 — 5 microstructure strategies (5m/15m optimized).
Domain: GARCH Volatility Clustering & Regime-Adaptive Signals
(academic, distinct from MICRO1-5 + batches 3237-3318)

Strategies:
  1. GARCHRegime    — Volatility regime (low→range trade, high→trend follow)
  2. VolCluster     — ARCH-effects clustering: burst detection → fade burst extreme
  3. JumpDetect     — Price jump detection + post-jump mean reversion
  4. RangeBandAdapt — ATR-adaptive normalized range bands with bollinger squeeze
  5. HeteroSig      — Heteroskedasticity signal: variance ratio breakout confirmation

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Engle (1982) ARCH — autoregressive conditional heteroskedasticity
  - Bollerslev (1986) GARCH — generalized ARCH
  - Barndorff-Nielsen & Shephard (2004) — realized volatility jumps
  - Andersen et al. (2001) — intraday volatility clustering
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
# 1. GARCHRegime — Volatility regime-adaptive strategy
# ═══════════════════════════════════════════════════════════════════════════
# Bollerslev (1986): GARCH volatility has regime persistence.
# Low-vol regime → price stays in range → fade extremes.
# High-vol regime → price trends → follow momentum.

def gen_GARCHRegime(df, garch_window=20, low_vol_pct=0.35, high_vol_pct=0.65,
                    bb_period=20, bb_mult=1.8, ema_fast=9, ema_slow=21,
                    exit_bar=8):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    ret = c.pct_change().fillna(0)
    # GARCH proxy: rolling variance of returns (conditional variance)
    cond_var = ret.rolling(garch_window).var()
    var_pctl_low  = cond_var.rolling(120).quantile(low_vol_pct)
    var_pctl_high = cond_var.rolling(120).quantile(high_vol_pct)

    low_vol_regime  = cond_var < var_pctl_low
    high_vol_regime = cond_var > var_pctl_high

    # Bollinger Bands for range-trade signals
    bb_mid  = _sma(c, bb_period)
    bb_std  = c.rolling(bb_period).std(ddof=0)
    bb_up   = bb_mid + bb_mult * bb_std
    bb_dn   = bb_mid - bb_mult * bb_std

    # EMA cross for trend signals
    fast = _ema(c, ema_fast)
    slow = _ema(c, ema_slow)

    # Low-vol: mean-revert at BB extremes
    # High-vol: follow EMA cross momentum
    entry_long = (
        (low_vol_regime  & (c < bb_dn)) |
        (high_vol_regime & (fast > slow) & (fast.shift(1) <= slow.shift(1)))
    )
    entry_short = (
        (low_vol_regime  & (c > bb_up)) |
        (high_vol_regime & (fast < slow) & (fast.shift(1) >= slow.shift(1)))
    )

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_GARCHRegime(trial):
    return {
        'garch_window':  trial.suggest_int('garch_window', 12, 35),
        'low_vol_pct':   trial.suggest_float('low_vol_pct', 0.20, 0.45, step=0.05),
        'high_vol_pct':  trial.suggest_float('high_vol_pct', 0.55, 0.80, step=0.05),
        'bb_period':     trial.suggest_int('bb_period', 14, 30),
        'bb_mult':       trial.suggest_float('bb_mult', 1.5, 2.5, step=0.25),
        'ema_fast':      trial.suggest_int('ema_fast', 5, 14),
        'ema_slow':      trial.suggest_int('ema_slow', 15, 35),
        'exit_bar':      trial.suggest_int('exit_bar', 5, 18),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. VolCluster — ARCH clustering: burst detection → fade the extreme
# ═══════════════════════════════════════════════════════════════════════════
# Engle (1982): Volatility clusters. After a burst, conditional volatility
# stays high. Fade the initial burst extreme, not the continuation.

def gen_VolCluster(df, burst_window=5, cluster_window=20, burst_z_thresh=2.0,
                   price_z_thresh=1.5, exit_bar=7):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    ret = c.pct_change().fillna(0)
    abs_ret = ret.abs()

    # Volatility burst: recent |ret| >> rolling average
    recent_vol  = abs_ret.rolling(burst_window).mean()
    cluster_vol = abs_ret.rolling(cluster_window).mean()
    burst_z = _zscore(recent_vol, cluster_window)

    # Price Z-score to identify the extreme
    price_z = _zscore(c, cluster_window)

    # Clustering state: burst happened recently, vol still elevated
    in_cluster = (burst_z > burst_z_thresh) | (burst_z.rolling(3).max() > burst_z_thresh * 0.8)

    # Fade the extreme: high vol + extreme price → reversal
    entry_long  = in_cluster & (price_z < -price_z_thresh)
    entry_short = in_cluster & (price_z >  price_z_thresh)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_VolCluster(trial):
    return {
        'burst_window':   trial.suggest_int('burst_window', 3, 10),
        'cluster_window': trial.suggest_int('cluster_window', 15, 40),
        'burst_z_thresh': trial.suggest_float('burst_z_thresh', 1.5, 3.0, step=0.25),
        'price_z_thresh': trial.suggest_float('price_z_thresh', 1.0, 2.5, step=0.25),
        'exit_bar':       trial.suggest_int('exit_bar', 4, 15),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. JumpDetect — Price jump detection + post-jump mean reversion
# ═══════════════════════════════════════════════════════════════════════════
# Barndorff-Nielsen & Shephard (2004): Jumps are transient, non-diffusive.
# Post-jump: price reverts toward pre-jump level within short window.

def gen_JumpDetect(df, jump_window=20, jump_z_thresh=2.5, revert_ema=10,
                   confirm_bars=2, exit_bar=8):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    ret = c.pct_change().fillna(0)

    # Jump detection: return Z-score > threshold
    ret_z = _zscore(ret, jump_window)
    jump_up   = ret_z >  jump_z_thresh   # price jumped up
    jump_down = ret_z < -jump_z_thresh   # price jumped down

    # Jump occurred in last confirm_bars bars
    recent_jump_up   = jump_up.rolling(confirm_bars).max().astype(bool)
    recent_jump_down = jump_down.rolling(confirm_bars).max().astype(bool)

    # Reversal EMA as reference
    rev_ema = _ema(c, revert_ema)

    # ATR filter
    atr = _atr(h, l, c)
    vol_ok = atr > atr.rolling(20).mean() * 0.5

    # After upward jump, price above EMA → fade (short)
    # After downward jump, price below EMA → fade (long)
    entry_long  = recent_jump_down & (c < rev_ema) & vol_ok
    entry_short = recent_jump_up   & (c > rev_ema) & vol_ok

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_JumpDetect(trial):
    return {
        'jump_window':    trial.suggest_int('jump_window', 15, 40),
        'jump_z_thresh':  trial.suggest_float('jump_z_thresh', 2.0, 3.5, step=0.25),
        'revert_ema':     trial.suggest_int('revert_ema', 6, 20),
        'confirm_bars':   trial.suggest_int('confirm_bars', 1, 4),
        'exit_bar':       trial.suggest_int('exit_bar', 4, 16),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. RangeBandAdapt — ATR-adaptive normalized range bands
# ═══════════════════════════════════════════════════════════════════════════
# Andersen et al. (2001): Intraday volatility is cyclical. ATR-normalized
# price relative to adaptive bands identifies over/under-extension.

def gen_RangeBandAdapt(df, atr_period=14, band_mult=1.5, bb_period=20,
                        squeeze_thresh=0.8, exit_bar=7):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    atr = _atr(h, l, c, atr_period)

    # Normalized price: (close - SMA) / ATR
    sma = _sma(c, bb_period)
    norm_price = (c - sma) / (atr + 1e-10)

    # Adaptive bands (rolling percentile of norm_price)
    norm_upper = norm_price.rolling(bb_period * 2).quantile(0.85)
    norm_lower = norm_price.rolling(bb_period * 2).quantile(0.15)

    # BB squeeze: BB width / Keltner channel width
    bb_std = c.rolling(bb_period).std(ddof=0)
    kc_width = atr * 2
    squeeze = (2 * bb_std) / (kc_width + 1e-10)
    in_squeeze = squeeze < squeeze_thresh  # compressed range → breakout pending

    # Volume filter
    v = df['volume'].shift(1)
    vol_ok = v > v.rolling(20).mean() * 0.6

    # Extended beyond adaptive bands → reversal
    entry_long  = (norm_price < norm_lower) & ~in_squeeze & vol_ok
    entry_short = (norm_price > norm_upper) & ~in_squeeze & vol_ok

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_RangeBandAdapt(trial):
    return {
        'atr_period':     trial.suggest_int('atr_period', 10, 20),
        'band_mult':      trial.suggest_float('band_mult', 1.0, 2.5, step=0.25),
        'bb_period':      trial.suggest_int('bb_period', 14, 30),
        'squeeze_thresh': trial.suggest_float('squeeze_thresh', 0.6, 1.0, step=0.1),
        'exit_bar':       trial.suggest_int('exit_bar', 4, 15),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. HeteroSig — Variance ratio test breakout confirmation
# ═══════════════════════════════════════════════════════════════════════════
# Lo & MacKinlay (1988): Variance ratio test — VR>1 → momentum, VR<1 → mean rev.
# Use VR trend to confirm or fade breakouts dynamically.

def gen_HeteroSig(df, vr_short=5, vr_long=20, vr_thresh_mom=1.15,
                  vr_thresh_rev=0.88, ema_period=21, exit_bar=9):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    ret = c.pct_change().fillna(0)

    # Variance ratio: Var(k-period ret) / (k * Var(1-period ret))
    var1 = ret.rolling(vr_long).var() + 1e-10
    var_k = ret.rolling(vr_short).std(ddof=0).rolling(vr_long).var() + 1e-10
    # Simpler proxy: std over short vs long window
    std_short = ret.rolling(vr_short).std(ddof=0) + 1e-10
    std_long  = ret.rolling(vr_long).std(ddof=0)  + 1e-10
    # VR proxy: (std_short / std_long) * sqrt(vr_long / vr_short)
    vr = (std_short / std_long) * np.sqrt(vr_long / vr_short)

    ema = _ema(c, ema_period)
    trend_up   = c > ema
    trend_down = c < ema

    atr = _atr(h, l, c)
    vol_ok = atr > atr.rolling(20).mean() * 0.5

    # VR > threshold → momentum → follow trend
    # VR < threshold → mean-rev → fade trend
    momentum_regime    = vr > vr_thresh_mom
    mean_rev_regime    = vr < vr_thresh_rev

    entry_long  = vol_ok & (
        (momentum_regime & trend_up   & (c > c.shift(1))) |
        (mean_rev_regime & trend_down & (c < c.shift(1) * 0.995))
    )
    entry_short = vol_ok & (
        (momentum_regime & trend_down & (c < c.shift(1))) |
        (mean_rev_regime & trend_up   & (c > c.shift(1) * 1.005))
    )

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_HeteroSig(trial):
    return {
        'vr_short':        trial.suggest_int('vr_short', 3, 10),
        'vr_long':         trial.suggest_int('vr_long', 15, 40),
        'vr_thresh_mom':   trial.suggest_float('vr_thresh_mom', 1.05, 1.40, step=0.05),
        'vr_thresh_rev':   trial.suggest_float('vr_thresh_rev', 0.70, 0.95, step=0.05),
        'ema_period':      trial.suggest_int('ema_period', 15, 35),
        'exit_bar':        trial.suggest_int('exit_bar', 5, 18),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'GARCHRegime': {
        'gen':   gen_GARCHRegime,
        'space': space_GARCHRegime,
    },
    'VolCluster': {
        'gen':   gen_VolCluster,
        'space': space_VolCluster,
    },
    'JumpDetect': {
        'gen':   gen_JumpDetect,
        'space': space_JumpDetect,
    },
    'RangeBandAdapt': {
        'gen':   gen_RangeBandAdapt,
        'space': space_RangeBandAdapt,
    },
    'HeteroSig': {
        'gen':   gen_HeteroSig,
        'space': space_HeteroSig,
    },
}
