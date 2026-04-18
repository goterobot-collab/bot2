#!/usr/bin/env python3
"""
TV2 BATCH MICRO11 — 5 microstructure strategies (5m/15m optimized).
Domain: Limit Order Book Dynamics & Trade Classification Theory
(academic, distinct from MICRO1-10 + batches 3237-3318)

Strategies:
  1. LOBSkew_Rev        — LOB shape skewness proxy (bid/ask pressure imbalance
                           via bar structure): extreme wick skew → fade expected
  2. LeeReady_Cont      — Lee & Ready (1991) tick-rule sign continuation:
                           consecutive buy/sell bar runs exhaust → reversal fade
  3. ImpactDecay_Fade   — Bouchaud, Gefen, Potters & Wyart (2004) impact decay:
                           post-large-trade price impact decays → fade residual
                           overshoot one bar after the large impact bar
  4. Hawkes_Burst       — Hawkes (1971) self-exciting process proxy: volume/trade
                           intensity clustering → burst exhaustion fade
  5. GlostenMilgrom_Rev — Glosten & Milgrom (1985) adverse selection model:
                           high spread candle (informed flow) → reversion fade
                           as uninformed flow restores equilibrium price

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Lee & Ready (1991) — "Inferring Trade Direction from Intraday Data";
    J. of Finance 46(2): 733-746.
  - Glosten & Milgrom (1985) — "Bid, Ask and Transaction Prices in a Specialist
    Market with Heterogeneously Informed Traders"; J. of Financial Economics 14(1).
  - Bouchaud, Gefen, Potters & Wyart (2004) — "Fluctuations and Response in
    Financial Markets: the Subtle Nature of Random Price Changes";
    Quantitative Finance 4(2): 176-190.
  - Hawkes (1971) — "Spectra of Some Self-Exciting and Mutually Exciting Point
    Processes"; Biometrika 58(1): 83-90.
  - Cont, Stoikov & Talreja (2010) — "A Stochastic Model for Order Book Dynamics";
    Operations Research 58(3): 549-563.
"""

import pandas as pd
import numpy as np


# ─── HELPERS ────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _atr(h, l, c, p=14):
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / p, adjust=False).mean()

def _zscore(s, p):
    m = s.rolling(p).mean()
    sd = s.rolling(p).std(ddof=0) + 1e-10
    return (s - m) / sd

def _apply_exit_bar(entry_long: pd.Series, entry_short: pd.Series,
                    exit_bar: int) -> pd.Series:
    """
    3-state machine: enters on signal, exits after exit_bar bars.
    Flat(0) -> Long(1) on entry_long; Flat(0) -> Short(-1) on entry_short.
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


# =============================================================================
# 1. LOBSkew_Rev — LOB shape skewness via bar wick structure
# =============================================================================
# Cont, Stoikov & Talreja (2010): LOB shape encodes supply/demand pressure.
# Proxy via bar wicks: upper_wick = high - close (ask-side absorbed pressure),
# lower_wick = close - low (bid-side absorbed pressure).
# LOB skew = (upper_wick - lower_wick) / (range + eps).
# Positive skew -> ask-side heavy (sellers dominating) -> bid bounce expected.
# When Z-score of skew is extreme -> fade the dominant side.

def gen_LOBSkew_Rev(df, lob_window=25, skew_z_thresh=1.4,
                    price_z_window=20, price_z_thresh=1.0,
                    trend_ema=40, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)

    bar_range = (h - l) + 1e-8
    upper_wick = h - c   # ask-side: sellers absorbed upper moves
    lower_wick = c - l   # bid-side: buyers absorbed lower moves

    # LOB skew: >0 -> ask-heavy (bearish skew), <0 -> bid-heavy (bullish skew)
    lob_skew = (upper_wick - lower_wick) / bar_range
    skew_z = _zscore(lob_skew, lob_window)

    # Price Z-score to avoid fading strong persistent trends
    price_z = _zscore(c, price_z_window)

    # Extreme positive skew -> ask overhang exhausted -> bounce up
    entry_long  = (skew_z >  skew_z_thresh) & (price_z < price_z_thresh)
    # Extreme negative skew -> bid overhang exhausted -> fall
    entry_short = (skew_z < -skew_z_thresh) & (price_z > -price_z_thresh)

    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.980)
    entry_short = entry_short & (c < trend * 1.020)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_LOBSkew_Rev(trial):
    return {
        'lob_window':     trial.suggest_int('lob_window', 12, 40),
        'skew_z_thresh':  trial.suggest_float('skew_z_thresh', 1.0, 2.5, step=0.25),
        'price_z_window': trial.suggest_int('price_z_window', 10, 30),
        'price_z_thresh': trial.suggest_float('price_z_thresh', 0.5, 1.8, step=0.25),
        'trend_ema':      trial.suggest_int('trend_ema', 25, 65),
        'exit_bar':       trial.suggest_int('exit_bar', 3, 12),
    }


# =============================================================================
# 2. LeeReady_Cont — Lee & Ready (1991) trade sign streak exhaustion
# =============================================================================
# Lee & Ready (1991): classify each trade as buyer- or seller-initiated via
# the tick rule. Bar-level proxy: close > open -> buy bar (-1 sell bar).
# A long run of same-sign bars indicates momentum that eventually exhausts.
# streak >= thresh -> trend has run too far -> fade in opposite direction.
# Volume filter: excludes single anomalous large-trade candles.

def gen_LeeReady_Cont(df, streak_thresh=4, streak_window=15,
                      vol_z_max=2.5, trend_ema=35, exit_bar=5):
    c = df['close'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    bar_sign = (c - o).fillna(0.0).apply(np.sign)

    # Build consecutive same-sign streak counter
    signs = bar_sign.values
    n = len(signs)
    streak_vals = np.zeros(n)
    current_streak = 0
    for i in range(n):
        s = int(signs[i])
        if i == 0:
            current_streak = s
        elif s == 0:
            current_streak = 0
        elif s == int(signs[i - 1]):
            current_streak += s
        else:
            current_streak = s
        streak_vals[i] = current_streak
    streak = pd.Series(streak_vals, index=bar_sign.index)

    # Exclude mega-candles (single informed trades)
    vol_z = _zscore(v, streak_window)
    vol_ok = vol_z < vol_z_max

    entry_short = (streak >= streak_thresh)  & vol_ok   # buy streak exhaust -> short
    entry_long  = (streak <= -streak_thresh) & vol_ok   # sell streak exhaust -> long

    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.983)
    entry_short = entry_short & (c < trend * 1.017)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_LeeReady_Cont(trial):
    return {
        'streak_thresh': trial.suggest_int('streak_thresh', 3, 7),
        'streak_window': trial.suggest_int('streak_window', 10, 30),
        'vol_z_max':     trial.suggest_float('vol_z_max', 1.5, 3.5, step=0.5),
        'trend_ema':     trial.suggest_int('trend_ema', 20, 55),
        'exit_bar':      trial.suggest_int('exit_bar', 3, 10),
    }


# =============================================================================
# 3. ImpactDecay_Fade — Bouchaud et al. (2004) price impact power-law decay
# =============================================================================
# Bouchaud et al. (2004): market impact decays as t^(-beta), beta ~0.5.
# After a large trade displaces price, the price mean-reverts to remove the
# transient component over the following bars.
# Signal: large ATR-normalized body bar = large market order. Impact decays
# starting next bar. Fade the large bar's direction one bar later.

def gen_ImpactDecay_Fade(df, impact_z_thresh=1.8, impact_window=30,
                         body_atr_ratio=0.6, trend_ema=40, exit_bar=7):
    c = df['close'].shift(1)
    o = df['open'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    atr = _atr(h, l, c, p=14)
    bar_body = (c - o).abs()
    body_ratio = bar_body / (atr + 1e-8)
    body_z = _zscore(body_ratio, impact_window)

    # Large-impact bar: anomalous body AND above absolute threshold
    large_impact = ((body_z > impact_z_thresh) & (body_ratio > body_atr_ratio)).astype(float)
    bar_dir = (c - o).apply(np.sign)

    # One bar after the impact bar, fade its direction
    large_impact_lag = large_impact.shift(1).fillna(0.0) > 0.5
    bar_dir_lag = bar_dir.shift(1).fillna(0)

    # Volume confirmation: the large-impact bar had elevated volume
    vol_z = _zscore(v, impact_window)
    vol_elevated = vol_z.shift(1).fillna(0) > 0.3

    entry_short = large_impact_lag & (bar_dir_lag > 0) & vol_elevated
    entry_long  = large_impact_lag & (bar_dir_lag < 0) & vol_elevated

    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.982)
    entry_short = entry_short & (c < trend * 1.018)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_ImpactDecay_Fade(trial):
    return {
        'impact_z_thresh': trial.suggest_float('impact_z_thresh', 1.2, 2.8, step=0.2),
        'impact_window':   trial.suggest_int('impact_window', 15, 50),
        'body_atr_ratio':  trial.suggest_float('body_atr_ratio', 0.3, 1.2, step=0.1),
        'trend_ema':       trial.suggest_int('trend_ema', 25, 65),
        'exit_bar':        trial.suggest_int('exit_bar', 4, 14),
    }


# =============================================================================
# 4. Hawkes_Burst — Hawkes (1971) self-exciting intensity exhaustion
# =============================================================================
# Hawkes (1971): events cluster — each arrival increases intensity of future
# arrivals (branching ratio mu). Volume burst intensity peaks then decays.
# When the EMA-based intensity is at a Z-score extreme AND declining
# (rate-of-change < 0 for 2 consecutive bars) -> burst exhaustion -> fade.

def gen_Hawkes_Burst(df, decay_kappa=5, intensity_window=30,
                     intensity_z_thresh=1.5, dir_window=10,
                     trend_ema=35, exit_bar=6):
    c = df['close'].shift(1)
    v = df['volume'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    # Exponential kernel (EMA as Hawkes decay approximation)
    intensity = v.ewm(span=decay_kappa, adjust=False).mean()
    intensity_z = _zscore(intensity, intensity_window)

    # Rate of change: is the burst cooling?
    intensity_roc = intensity_z - intensity_z.shift(2).fillna(intensity_z)

    # 2 bars ago intensity was high AND it is now declining
    burst_peak_lag = intensity_z.shift(2).fillna(0)
    cooling = intensity_roc < -0.1
    burst_exhaustion = (burst_peak_lag > intensity_z_thresh) & cooling

    # Net price direction during the burst window
    net_move = c - c.shift(dir_window).fillna(c)
    net_dir = net_move.apply(np.sign)

    # Fade the burst direction
    entry_short = burst_exhaustion & (net_dir > 0)
    entry_long  = burst_exhaustion & (net_dir < 0)

    # Real burst filter: total volume during window > average
    vol_sum = v.rolling(dir_window).sum()
    vol_avg_window = v.rolling(intensity_window).mean() * dir_window
    real_burst = vol_sum > vol_avg_window * 0.9

    entry_short = entry_short & real_burst
    entry_long  = entry_long  & real_burst

    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.984)
    entry_short = entry_short & (c < trend * 1.016)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Hawkes_Burst(trial):
    return {
        'decay_kappa':        trial.suggest_int('decay_kappa', 3, 12),
        'intensity_window':   trial.suggest_int('intensity_window', 15, 45),
        'intensity_z_thresh': trial.suggest_float('intensity_z_thresh', 1.0, 2.5, step=0.25),
        'dir_window':         trial.suggest_int('dir_window', 5, 20),
        'trend_ema':          trial.suggest_int('trend_ema', 20, 55),
        'exit_bar':           trial.suggest_int('exit_bar', 3, 12),
    }


# =============================================================================
# 5. GlostenMilgrom_Rev — Glosten & Milgrom (1985) adverse selection reversion
# =============================================================================
# Glosten & Milgrom (1985): market makers widen spreads proportional to the
# probability of trading with an informed investor. High spread -> high adverse
# selection. Post informed-flow, uninformed (noise) traders restore equilibrium.
# Bar-level spread proxy: (high - low) / mid_price (proportional spread).
# High spread Z-score on an up/down bar -> informed flow overshoot ->
# fade one bar later as noise traders return.

def gen_GlostenMilgrom_Rev(df, spread_window=25, spread_z_thresh=1.6,
                            min_spread_pct=0.005, trend_ema=40, exit_bar=7):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)

    # Proportional spread proxy (Corwin-Schultz simplified bar version)
    mid_price = (h + l) / 2.0 + 1e-8
    rel_spread = (h - l) / mid_price

    # 3-bar smoothing to reduce single-bar noise
    spread_smooth = rel_spread.rolling(3).mean()
    spread_z = _zscore(spread_smooth, spread_window)

    spread_ok = rel_spread > min_spread_pct
    high_spread = ((spread_z > spread_z_thresh) & spread_ok).astype(float)

    # Direction of the informed-flow bar
    bar_dir = (c - o).apply(np.sign)

    # Fade one bar after the high-spread (adverse selection) event
    high_spread_lag = high_spread.shift(1).fillna(0.0) > 0.5
    bar_dir_lag = bar_dir.shift(1).fillna(0)

    entry_short = high_spread_lag & (bar_dir_lag > 0)   # informed BUY -> fade short
    entry_long  = high_spread_lag & (bar_dir_lag < 0)   # informed SELL -> fade long

    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.981)
    entry_short = entry_short & (c < trend * 1.019)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_GlostenMilgrom_Rev(trial):
    return {
        'spread_window':   trial.suggest_int('spread_window', 12, 45),
        'spread_z_thresh': trial.suggest_float('spread_z_thresh', 1.0, 2.8, step=0.2),
        'min_spread_pct':  trial.suggest_float('min_spread_pct', 0.002, 0.015, step=0.001),
        'trend_ema':       trial.suggest_int('trend_ema', 25, 65),
        'exit_bar':        trial.suggest_int('exit_bar', 4, 14),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'LOBSkew_Rev': {
        'gen':   gen_LOBSkew_Rev,
        'space': space_LOBSkew_Rev,
    },
    'LeeReady_Cont': {
        'gen':   gen_LeeReady_Cont,
        'space': space_LeeReady_Cont,
    },
    'ImpactDecay_Fade': {
        'gen':   gen_ImpactDecay_Fade,
        'space': space_ImpactDecay_Fade,
    },
    'Hawkes_Burst': {
        'gen':   gen_Hawkes_Burst,
        'space': space_Hawkes_Burst,
    },
    'GlostenMilgrom_Rev': {
        'gen':   gen_GlostenMilgrom_Rev,
        'space': space_GlostenMilgrom_Rev,
    },
}
