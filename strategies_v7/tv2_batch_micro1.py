#!/usr/bin/env python3
"""
TV2 BATCH MICRO1 — 5 microstructure strategies (5m/15m optimized).
Domain: Order Flow & Microstructure (academic, distinct from batches 3237-3307)

Strategies:
  1. CVD_MeanRev         — Cumulative Volume Delta Z-score mean reversion
  2. OrderFlow_Imbalance — Bar-structure delta volume imbalance oscillator
  3. VolAbsorption       — High-volume / small-range absorption reversal
  4. LiquiditySweep      — Stop hunt wick reversal (wick > ATR * mult)
  5. TickMomentum        — Consecutive-bar momentum exhaustion reversal

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar.
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
    3-state machine: enters on entry signal, exits after exit_bar bars.
    Flat (0) → Long (1) on entry_long; Flat (0) → Short (-1) on entry_short.
    Any active position clears after exit_bar bars.
    New entries can override an opposite position immediately.
    """
    sig = pd.Series(0, index=entry_long.index, dtype=int)
    el = entry_long.values
    es = entry_short.values
    n = len(sig)
    state = 0
    bars_held = 0
    for i in range(n):
        # Check exit first
        if state != 0:
            bars_held += 1
            if bars_held >= exit_bar:
                state = 0
                bars_held = 0
        # Check new entry (can override)
        if el[i]:
            state = 1
            bars_held = 0
        elif es[i]:
            state = -1
            bars_held = 0
        sig.iloc[i] = state
    return sig


# ═══════════════════════════════════════════════════════════════════════════
# 1. CVD_MeanRev — Cumulative Volume Delta mean reversion
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Bessembinder (2003), Easley & O'Hara (1987) order flow
# Logic: Approximate CVD = cumsum(signed_volume). When CVD Z-score is
#   extreme vs price direction → reversal entry.

def gen_CVD_MeanRev(df, cvd_period=20, zscore_thresh=1.5, atr_period=14,
                    trend_ema=50, exit_bar=8):
    # Anti-repainting: shift all OHLCV
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Signed volume: positive if bar closed up, negative if down
    bar_dir = np.sign(c - o).replace(0, 1)
    signed_vol = v * bar_dir

    # Cumulative Volume Delta (rolling window, not all-history to avoid non-stationarity)
    cvd = signed_vol.rolling(cvd_period).sum()

    # Z-score of CVD relative to recent range
    cvd_z = _zscore(cvd, cvd_period)

    # Trend filter: only trade in direction of EMA trend
    trend = _ema(c, trend_ema)
    bull_trend = c > trend
    bear_trend = c < trend

    # Signal: CVD exhaustion — extreme buying but price below trend → short
    #         extreme selling but price above trend support → long
    entry_long  = (cvd_z < -zscore_thresh) & bull_trend
    entry_short = (cvd_z >  zscore_thresh) & bear_trend

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_CVD_MeanRev(trial):
    return {
        'cvd_period':    trial.suggest_int('cvd_period', 10, 40),
        'zscore_thresh': trial.suggest_float('zscore_thresh', 1.0, 3.0, step=0.25),
        'atr_period':    trial.suggest_int('atr_period', 10, 20),
        'trend_ema':     trial.suggest_int('trend_ema', 30, 100),
        'exit_bar':      trial.suggest_int('exit_bar', 3, 20),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. OrderFlow_Imbalance — Bar-structure delta imbalance oscillator
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Chordia, Roll & Subrahmanyam (2002) order imbalance
# Logic: delta_proxy = (close - low) / (high - low) as buy fraction.
#   Imbalance = rolling mean of delta_proxy. Extreme imbalance → reversal.

def gen_OrderFlow_Imbalance(df, ofi_period=15, imb_thresh=0.75, rsi_period=14,
                             rsi_filter=True, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Buy fraction per bar (0=all sell, 1=all buy)
    hl_range = (h - l).replace(0, np.nan)
    buy_frac = (c - l) / hl_range
    buy_frac = buy_frac.fillna(0.5)

    # Volume-weighted imbalance
    vol_w_imb = (buy_frac * v).rolling(ofi_period).sum() / (v.rolling(ofi_period).sum() + 1e-10)

    # RSI filter (optional: only trade against extreme RSI)
    d = c.diff()
    gain = d.where(d > 0, 0.0).ewm(span=rsi_period, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(span=rsi_period, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-10))

    # Entry: extreme buy imbalance + RSI overbought → short reversal
    #        extreme sell imbalance + RSI oversold → long reversal
    overbought = rsi > 60 if rsi_filter else pd.Series(True, index=c.index)
    oversold   = rsi < 40 if rsi_filter else pd.Series(True, index=c.index)

    entry_long  = (vol_w_imb < (1.0 - imb_thresh)) & oversold
    entry_short = (vol_w_imb > imb_thresh) & overbought

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_OrderFlow_Imbalance(trial):
    return {
        'ofi_period':  trial.suggest_int('ofi_period', 8, 30),
        'imb_thresh':  trial.suggest_float('imb_thresh', 0.60, 0.85, step=0.05),
        'rsi_period':  trial.suggest_int('rsi_period', 10, 21),
        'rsi_filter':  trial.suggest_categorical('rsi_filter', [True, False]),
        'exit_bar':    trial.suggest_int('exit_bar', 3, 15),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. VolAbsorption — High-volume small-range absorption reversal
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Glosten & Milgrom (1985) market microstructure
# Logic: When volume spikes (Z > threshold) but price range is small
#   → institutional absorption → reversal likely.
#   Direction determined by bar close position vs OHLC midpoint.

def gen_VolAbsorption(df, vol_period=20, vol_z_thresh=1.5, range_pct_thresh=0.5,
                      ema_trend=30, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Volume Z-score
    vol_z = _zscore(v, vol_period)

    # Normalized range (relative to ATR)
    atr_val = _atr(h, l, c, vol_period)
    bar_range = h - l
    norm_range = bar_range / (atr_val + 1e-10)

    # Absorption: high volume + small range
    is_absorption = (vol_z > vol_z_thresh) & (norm_range < range_pct_thresh)

    # Direction from close position in bar
    midpoint = (h + l) / 2.0
    close_above_mid = c > midpoint  # selling absorbed → bullish
    close_below_mid = c < midpoint  # buying absorbed → bearish

    # Trend context
    trend = _ema(c, ema_trend)

    entry_long  = is_absorption & close_above_mid & (c > trend)
    entry_short = is_absorption & close_below_mid & (c < trend)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_VolAbsorption(trial):
    return {
        'vol_period':       trial.suggest_int('vol_period', 12, 35),
        'vol_z_thresh':     trial.suggest_float('vol_z_thresh', 1.0, 3.0, step=0.25),
        'range_pct_thresh': trial.suggest_float('range_pct_thresh', 0.3, 0.8, step=0.1),
        'ema_trend':        trial.suggest_int('ema_trend', 20, 60),
        'exit_bar':         trial.suggest_int('exit_bar', 3, 15),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. LiquiditySweep — Stop hunt wick reversal
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Pöppe et al. (2016) stop-run microstructure patterns
# Logic: A large wick (> ATR * mult) that breaks recent high/low but
#   closes back inside → liquidity sweep / stop hunt → reversal.

def gen_LiquiditySweep(df, atr_period=14, wick_mult=1.2, lookback=20,
                        wick_body_ratio=2.0, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    atr_val = _atr(h, l, c, atr_period)

    # Upper wick = high - max(open, close)
    body_top = pd.concat([o, c], axis=1).max(axis=1)
    body_bot = pd.concat([o, c], axis=1).min(axis=1)
    upper_wick = h - body_top
    lower_wick = body_bot - l
    body_size  = (body_top - body_bot).abs() + 1e-6

    # Large wick relative to ATR
    long_upper_wick = upper_wick > (atr_val * wick_mult)
    long_lower_wick = lower_wick > (atr_val * wick_mult)

    # Wick >> body (pinbar/shooting star)
    wick_dom_upper = upper_wick > (body_size * wick_body_ratio)
    wick_dom_lower = lower_wick > (body_size * wick_body_ratio)

    # Sweep of recent high/low
    recent_high = h.shift(1).rolling(lookback).max()
    recent_low  = l.shift(1).rolling(lookback).min()
    swept_high  = h > recent_high  # wick swept above highs → bearish reversal
    swept_low   = l < recent_low   # wick swept below lows → bullish reversal

    # Entry: bullish sweep (swept low + large lower wick)
    entry_long  = swept_low  & long_lower_wick & wick_dom_lower
    # Entry: bearish sweep (swept high + large upper wick)
    entry_short = swept_high & long_upper_wick & wick_dom_upper

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_LiquiditySweep(trial):
    return {
        'atr_period':      trial.suggest_int('atr_period', 10, 21),
        'wick_mult':       trial.suggest_float('wick_mult', 0.8, 2.4, step=0.2),
        'lookback':        trial.suggest_int('lookback', 10, 40),
        'wick_body_ratio': trial.suggest_float('wick_body_ratio', 1.0, 3.5, step=0.5),
        'exit_bar':        trial.suggest_int('exit_bar', 3, 15),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. TickMomentum — Consecutive-bar momentum exhaustion reversal
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Jegadeesh & Titman (1993) momentum, Lo & MacKinlay (1988)
# Logic: Count consecutive bars closing in the same direction.
#   After N consecutive up/down bars → momentum exhaustion → reversal.
#   Volume confirmation: exhaustion bar should have declining volume.

def gen_TickMomentum(df, min_streak=3, max_streak=8, vol_confirm=True,
                     vol_decline_bars=2, atr_period=14, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Bar direction: 1 if close > prev close, -1 otherwise
    bar_up   = (c > c.shift(1)).astype(int)
    bar_down = (c < c.shift(1)).astype(int)

    # Streak counters (vectorized using rolling cumsum trick)
    # Streak_up[i] = how many consecutive up bars ending at i
    def _streak(direction_series):
        streaks = []
        streak = 0
        vals = direction_series.values
        for v_val in vals:
            if v_val:
                streak += 1
            else:
                streak = 0
            streaks.append(streak)
        return pd.Series(streaks, index=direction_series.index)

    streak_up   = _streak(bar_up.astype(bool))
    streak_down = _streak(bar_down.astype(bool))

    # Volume declining in last vol_decline_bars bars
    vol_slope = v - v.shift(vol_decline_bars)
    vol_declining = vol_slope < 0

    # Exhaustion: streak within [min_streak, max_streak]
    up_exhaust   = (streak_up   >= min_streak) & (streak_up   <= max_streak)
    down_exhaust = (streak_down >= min_streak) & (streak_down <= max_streak)

    # Volume confirmation
    if vol_confirm:
        entry_long  = down_exhaust & vol_declining
        entry_short = up_exhaust   & vol_declining
    else:
        entry_long  = down_exhaust
        entry_short = up_exhaust

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_TickMomentum(trial):
    return {
        'min_streak':        trial.suggest_int('min_streak', 2, 5),
        'max_streak':        trial.suggest_int('max_streak', 5, 12),
        'vol_confirm':       trial.suggest_categorical('vol_confirm', [True, False]),
        'vol_decline_bars':  trial.suggest_int('vol_decline_bars', 1, 4),
        'atr_period':        trial.suggest_int('atr_period', 10, 20),
        'exit_bar':          trial.suggest_int('exit_bar', 3, 15),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'CVD_MeanRev': {
        'gen':   gen_CVD_MeanRev,
        'space': space_CVD_MeanRev,
    },
    'OrderFlow_Imbalance': {
        'gen':   gen_OrderFlow_Imbalance,
        'space': space_OrderFlow_Imbalance,
    },
    'VolAbsorption': {
        'gen':   gen_VolAbsorption,
        'space': space_VolAbsorption,
    },
    'LiquiditySweep': {
        'gen':   gen_LiquiditySweep,
        'space': space_LiquiditySweep,
    },
    'TickMomentum': {
        'gen':   gen_TickMomentum,
        'space': space_TickMomentum,
    },
}
