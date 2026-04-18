#!/usr/bin/env python3
"""
TV2 BATCH MICRO20 — 5 microstructure strategies (5m/15m optimized).
Domain: Chordia Order Flow Reversal (Chordia et al. 2002),
        Garman-Klass OHLC Vol Regime (Garman & Klass 1980),
        Madhavan Inventory Adjustment (Madhavan 1992),
        Blume Volume Quality Signal (Blume, Easley & O'Hara 1994),
        Engle ACD Duration Clustering (Engle & Russell 1998) — all new vs MICRO1-19.

Strategies:
  1. Chordia_FlowReversal      — Chordia, Roll & Subrahmanyam (2002): daily order
                                 imbalance adapted to 5m/15m. Net signed volume flow
                                 (up-volume minus down-volume proxy) z-scored over
                                 rolling window → extremes predict short-term reversal
                                 when confirmed by ATR contraction.

  2. GarmanKlass_VolRegime     — Garman & Klass (1980): OHLC-based variance estimator
                                 more efficient than close-to-close. GK_var captures
                                 intraday drift, detects vol-expansion (breakout follow)
                                 vs vol-compression (mean-reversion fade) regimes.

  3. Madhavan_InventoryAdj     — Madhavan (1992): market-maker inventory model.
                                 Signed cumulative volume deviation from its rolling
                                 mean acts as dealer inventory proxy. Extremes trigger
                                 inventory-rebalancing reversal; direction confirmed
                                 by short-horizon price elasticity.

  4. Blume_VolumeQuality       — Blume, Easley & O'Hara (1994): volume as signal
                                 quality indicator. High volume + small move = informed
                                 consensus → trend-follow. Low volume + large move =
                                 uninformed noise → fade. Dual mode with trend filter.

  5. Engle_DurationCluster     — Engle & Russell (1998): Autoregressive Conditional
                                 Duration (ACD). Bar range (high−low) treated as
                                 "duration" analogue: range-clustering (range rising
                                 after contraction) signals informed activity. Mean
                                 range z-score with exponential decay → entry filter.

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Chordia, T., Roll, R. & Subrahmanyam, A. (2002) — "Order Imbalance, Liquidity,
    and Market Returns"; Journal of Financial Economics 65(1): 111-130.
  - Garman, M.B. & Klass, M.J. (1980) — "On the Estimation of Security Price
    Volatilities from Historical Data"; Journal of Business 53(1): 67-78.
  - Madhavan, A. (1992) — "Trading Mechanisms in Securities Markets";
    Journal of Finance 47(2): 607-641.
  - Blume, L., Easley, D. & O'Hara, M. (1994) — "Market Statistics and Technical
    Analysis: The Role of Volume"; Journal of Finance 49(1): 153-181.
  - Engle, R.F. & Russell, J.R. (1998) — "Autoregressive Conditional Duration:
    A New Model for Irregularly Spaced Transaction Data";
    Econometrica 66(5): 1127-1162.
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
# 1. Chordia_FlowReversal
# ===========================================================================
# Chordia, Roll & Subrahmanyam (2002): order imbalance = buyers - sellers.
# Proxy: signed volume per bar = volume × sign(close − open).
# Net cumulative flow z-scored over `flow_window` bars detects extremes.
# Reversal logic: high positive net flow (buying pressure) → SHORT fade;
# high negative flow → LONG fade. Guard: skip if ATR is expanding (momentum).

def gen_Chordia_FlowReversal(df, flow_window=20, z_thresh=1.6,
                               atr_period=14, atr_window=30,
                               atr_contract_ratio=1.0, exit_bar=6):
    c = df['close'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    # Signed volume: positive = buying pressure, negative = selling pressure
    bar_direction = np.sign(c - o).replace(0, 1)   # ties → long
    signed_vol    = v * bar_direction

    # Cumulative flow z-score over flow_window bars
    flow_z = _zscore(signed_vol.rolling(flow_window).sum(), flow_window)

    # ATR regime: contraction = mean-reversion friendly
    atr_val     = _atr(h, l, c, atr_period)
    atr_avg     = atr_val.rolling(atr_window).mean() + 1e-10
    atr_ratio   = atr_val / atr_avg
    contracting = atr_ratio < atr_contract_ratio   # ATR below its own average

    # Reversal entries
    entry_long  = (flow_z < -z_thresh) & contracting
    entry_short = (flow_z >  z_thresh) & contracting

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Chordia_FlowReversal(trial):
    return {
        'flow_window':        trial.suggest_int('flow_window',         10, 40),
        'z_thresh':           trial.suggest_float('z_thresh',           1.0, 2.8),
        'atr_period':         trial.suggest_int('atr_period',           7, 21),
        'atr_window':         trial.suggest_int('atr_window',          15, 50),
        'atr_contract_ratio': trial.suggest_float('atr_contract_ratio', 0.7, 1.3),
        'exit_bar':           trial.suggest_int('exit_bar',              3, 12),
    }


# ===========================================================================
# 2. GarmanKlass_VolRegime
# ===========================================================================
# Garman & Klass (1980): GK variance = 0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2
# More efficient than Parkinson (uses open as well).
# GK z-score → regime detection:
#   High GK_z (vol expansion after compression): FOLLOW direction of open→close move
#   Low  GK_z (vol compression): FADE extreme open→close move (mean-reversion)

def gen_GarmanKlass_VolRegime(df, gk_window=25, z_high=1.4, z_low=-0.5,
                                 trend_ema=40, follow_pct=0.003,
                                 fade_pct=0.002, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)

    # Garman-Klass estimator (log-price form)
    hl_log  = np.log((h / (l + 1e-10)).clip(lower=1.001))
    co_log  = np.log((c / (o + 1e-10)).clip(lower=1e-6, upper=1e6))
    gk_var  = 0.5 * hl_log ** 2 - (2.0 * np.log(2) - 1.0) * co_log ** 2
    gk_var  = gk_var.clip(lower=0)

    gk_z    = _zscore(gk_var, gk_window)

    trend   = _ema(c, trend_ema)
    slope   = trend.diff(3)
    up_bar  = c > o
    dn_bar  = c < o

    # Vol expansion after squeeze → breakout follow
    expanding     = gk_z > z_high
    follow_long   = expanding & up_bar  & (slope > 0) & (c > trend * (1 + follow_pct))
    follow_short  = expanding & dn_bar  & (slope < 0) & (c < trend * (1 - follow_pct))

    # Vol compression → fade open→close extreme
    compressing   = gk_z < z_low
    rng           = (h - l).clip(lower=1e-10)
    fade_long     = compressing & dn_bar & (c < l + rng * fade_pct)
    fade_short    = compressing & up_bar & (c > h - rng * fade_pct)

    entry_long  = follow_long  | fade_long
    entry_short = follow_short | fade_short

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_GarmanKlass_VolRegime(trial):
    return {
        'gk_window':   trial.suggest_int('gk_window',    10, 50),
        'z_high':      trial.suggest_float('z_high',      0.8, 2.5),
        'z_low':       trial.suggest_float('z_low',      -1.5, 0.0),
        'trend_ema':   trial.suggest_int('trend_ema',    20, 80),
        'follow_pct':  trial.suggest_float('follow_pct',  0.001, 0.008),
        'fade_pct':    trial.suggest_float('fade_pct',    0.001, 0.006),
        'exit_bar':    trial.suggest_int('exit_bar',       3, 12),
    }


# ===========================================================================
# 3. Madhavan_InventoryAdj
# ===========================================================================
# Madhavan (1992): market makers adjust quotes based on inventory position.
# Proxy: signed-volume cumsum deviation from its rolling mean = "inventory load".
# When inventory load extreme → dealer must rebalance → price reverts.
# Price elasticity guard: |c - prev_c| / ATR < elasticity_cap (price moves little
# per unit signed-volume → confirms liquidity provider dominance).

def gen_Madhavan_InventoryAdj(df, inv_window=30, z_thresh=1.5,
                                atr_period=14, elast_window=20,
                                elast_cap=0.8, exit_bar=7):
    c = df['close'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    # Signed volume cumsum as inventory proxy
    sv        = v * np.sign(c - o).replace(0, 1)
    inv_cum   = sv.cumsum()
    inv_dev   = inv_cum - inv_cum.rolling(inv_window).mean()
    inv_z     = _zscore(inv_dev, inv_window)

    # Price elasticity: close-to-close return per unit ATR
    atr_val   = _atr(h, l, c, atr_period)
    ret_abs   = c.diff().abs()
    elast     = (ret_abs / (atr_val + 1e-10)).rolling(elast_window).mean()
    low_elast = elast < elast_cap   # dealer-dominated = low price impact

    # Reversal when inventory extreme + low price elasticity
    entry_long  = (inv_z < -z_thresh) & low_elast
    entry_short = (inv_z >  z_thresh) & low_elast

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Madhavan_InventoryAdj(trial):
    return {
        'inv_window':   trial.suggest_int('inv_window',    15, 50),
        'z_thresh':     trial.suggest_float('z_thresh',     0.8, 2.5),
        'atr_period':   trial.suggest_int('atr_period',     7, 21),
        'elast_window': trial.suggest_int('elast_window',  10, 40),
        'elast_cap':    trial.suggest_float('elast_cap',    0.3, 1.5),
        'exit_bar':     trial.suggest_int('exit_bar',        3, 12),
    }


# ===========================================================================
# 4. Blume_VolumeQuality
# ===========================================================================
# Blume, Easley & O'Hara (1994): volume signals information quality.
# quality = |close - open| / (volume^0.5 + ε)  → high = informed, low = noise.
# High quality + clear direction → trend-FOLLOW.
# Low quality + large move → noise → FADE.
# Trend filter (EMA slope) gates directional regime.

def gen_Blume_VolumeQuality(df, qual_window=25, z_high=1.2, z_low=-0.8,
                              trend_ema=40, move_thresh=0.002,
                              vol_power=0.5, exit_bar=6):
    c = df['close'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    # Volume quality: price movement per unit of sqrt(volume)
    price_move = (c - o).abs() / (c.shift(1).abs() + 1e-10)   # pct move
    vol_adj    = v.clip(lower=1.0) ** vol_power
    quality    = price_move / (vol_adj + 1e-10)
    qual_z     = _zscore(quality, qual_window)

    trend      = _ema(c, trend_ema)
    slope      = (trend - trend.shift(4)) / (trend.abs() + 1e-10)

    up_bar     = c > o * (1 + move_thresh)
    dn_bar     = c < o * (1 - move_thresh)

    # High quality: informed → follow direction + trend confirmation
    informed      = qual_z > z_high
    follow_long   = informed & up_bar  & (slope > 0)
    follow_short  = informed & dn_bar  & (slope < 0)

    # Low quality: noise → fade large move
    noisy         = qual_z < z_low
    fade_long     = noisy & dn_bar
    fade_short    = noisy & up_bar

    entry_long  = follow_long  | fade_long
    entry_short = follow_short | fade_short

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Blume_VolumeQuality(trial):
    return {
        'qual_window':  trial.suggest_int('qual_window',   10, 50),
        'z_high':       trial.suggest_float('z_high',       0.6, 2.5),
        'z_low':        trial.suggest_float('z_low',       -2.0, -0.2),
        'trend_ema':    trial.suggest_int('trend_ema',     20, 80),
        'move_thresh':  trial.suggest_float('move_thresh',  0.0005, 0.005),
        'vol_power':    trial.suggest_float('vol_power',    0.3, 0.8),
        'exit_bar':     trial.suggest_int('exit_bar',        3, 12),
    }


# ===========================================================================
# 5. Engle_DurationCluster
# ===========================================================================
# Engle & Russell (1998): ACD model — duration between events clusters.
# Adapted to OHLCV: bar range (H-L) treated as "activity duration".
# ACD idea: when range rises after a quiet period (compressed range),
# it signals new information arrival → FOLLOW the breakout direction.
# When range falls after elevated range (activity cooling off) → FADE.
# Exponential smoothing of range approximates the ACD conditional mean.

def gen_Engle_DurationCluster(df, range_ema_fast=8, range_ema_slow=25,
                                z_window=20, z_burst=1.3, z_cool=-0.5,
                                trend_ema=50, exit_bar=6):
    c  = df['close'].shift(1)
    h  = df['high'].shift(1)
    l  = df['low'].shift(1)
    o  = df['open'].shift(1)

    rng         = (h - l).clip(lower=1e-10)

    # ACD conditional mean: exponential smoothing at two horizons
    rng_fast    = _ema(rng, range_ema_fast)
    rng_slow    = _ema(rng, range_ema_slow)

    # "Burst" = range recently expanded above slow EMA = new activity arrival
    burst_ratio = rng_fast / (rng_slow + 1e-10)
    burst_z     = _zscore(burst_ratio, z_window)

    trend       = _ema(c, trend_ema)
    up_bar      = c > o
    dn_bar      = c < o
    above_trend = c > trend
    below_trend = c < trend

    # Activity burst: informed arrival → follow direction + trend confirmation
    bursting      = burst_z > z_burst
    follow_long   = bursting & up_bar  & above_trend
    follow_short  = bursting & dn_bar  & below_trend

    # Activity cooling: range collapsing → mean-reversion opportunity
    cooling       = burst_z < z_cool
    fade_long     = cooling & dn_bar
    fade_short    = cooling & up_bar

    entry_long  = follow_long  | fade_long
    entry_short = follow_short | fade_short

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Engle_DurationCluster(trial):
    return {
        'range_ema_fast': trial.suggest_int('range_ema_fast',  4, 20),
        'range_ema_slow': trial.suggest_int('range_ema_slow', 15, 60),
        'z_window':       trial.suggest_int('z_window',        10, 40),
        'z_burst':        trial.suggest_float('z_burst',        0.8, 2.5),
        'z_cool':         trial.suggest_float('z_cool',        -2.0, -0.1),
        'trend_ema':      trial.suggest_int('trend_ema',       20, 80),
        'exit_bar':       trial.suggest_int('exit_bar',          3, 12),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Chordia_FlowReversal': {
        'gen':   gen_Chordia_FlowReversal,
        'space': space_Chordia_FlowReversal,
    },
    'GarmanKlass_VolRegime': {
        'gen':   gen_GarmanKlass_VolRegime,
        'space': space_GarmanKlass_VolRegime,
    },
    'Madhavan_InventoryAdj': {
        'gen':   gen_Madhavan_InventoryAdj,
        'space': space_Madhavan_InventoryAdj,
    },
    'Blume_VolumeQuality': {
        'gen':   gen_Blume_VolumeQuality,
        'space': space_Blume_VolumeQuality,
    },
    'Engle_DurationCluster': {
        'gen':   gen_Engle_DurationCluster,
        'space': space_Engle_DurationCluster,
    },
}
