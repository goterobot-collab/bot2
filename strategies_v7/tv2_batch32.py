#!/usr/bin/env python3
"""TV2 BATCH 32 — 30 Fibonacci Strategies 2026-04-01"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=int(p), adjust=False).mean()


def _rma(s, p):
    return s.ewm(alpha=1.0 / int(p), adjust=False).mean()


def _sma(s, p):
    return s.rolling(int(p), min_periods=1).mean()


def _atr(df, p):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, p)


def _rsi(s, p):
    d  = s.diff()
    g  = d.clip(lower=0)
    l  = (-d).clip(lower=0)
    rs = _rma(g, p) / _rma(l, p).replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _swing_levels(df, swing_p):
    """Return (swing_low, swing_high) as Series over rolling window."""
    p = int(swing_p)
    sw_low  = df['low'].rolling(p, min_periods=1).min()
    sw_high = df['high'].rolling(p, min_periods=1).max()
    return sw_low, sw_high


def _fib_price(sw_low, sw_high, fib):
    """Fib retracement level in an uptrend: sw_low + fib*(sw_high - sw_low)."""
    return sw_low + fib * (sw_high - sw_low)


def _near_fib(close, fib_level, tol):
    """True where price is within tol of fib_level."""
    ref = fib_level.replace(0, 1e-9)
    return (close - fib_level).abs() / ref < tol


# ── 1. Fib_Retracement_382 ────────────────────────────────────────────────────

def gen_Fib_Retracement_382(df, swing_p=30, fib=0.382, tol=0.01, **kw):
    close         = df['close']
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_lvl       = _fib_price(sw_low, sw_high, fib)
    uptrend       = close > sw_high.shift(1)
    downtrend     = close < sw_low.shift(1)
    near          = _near_fib(close, fib_lvl, tol)
    sig = pd.Series(0, index=df.index)
    sig[near & (close > fib_lvl) & (sw_high > sw_high.shift(1))] =  1
    sig[near & (close < fib_lvl) & (sw_low  < sw_low.shift(1))]  = -1
    return sig.fillna(0)


def space_Fib_Retracement_382(trial):
    return {
        'swing_p': trial.suggest_int('swing_p',     20,  60),
        'fib':     trial.suggest_float('fib',       0.30, 0.45),
        'tol':     trial.suggest_float('tol',       0.005, 0.02),
    }


# ── 2. Fib_Retracement_618 ────────────────────────────────────────────────────

def gen_Fib_Retracement_618(df, swing_p=30, fib=0.618, tol=0.01, rsi_p=14, **kw):
    close         = df['close']
    rsi_p         = int(rsi_p)
    rsi           = _rsi(close, rsi_p)
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_lvl       = _fib_price(sw_low, sw_high, fib)
    fib_short     = _fib_price(sw_low, sw_high, 1 - fib)
    near_long     = _near_fib(close, fib_lvl, tol)
    near_short    = _near_fib(close, fib_short, tol)
    upswing       = sw_high > sw_high.shift(1)
    dnswing       = sw_low  < sw_low.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[near_long  & (rsi < 50) & upswing] =  1
    sig[near_short & (rsi > 50) & dnswing] = -1
    return sig.fillna(0)


def space_Fib_Retracement_618(trial):
    return {
        'swing_p': trial.suggest_int('swing_p',     20,  60),
        'fib':     trial.suggest_float('fib',       0.55, 0.68),
        'tol':     trial.suggest_float('tol',       0.005, 0.02),
        'rsi_p':   trial.suggest_int('rsi_p',        7,  21),
    }


# ── 3. Fib_Retracement_500 ────────────────────────────────────────────────────

def gen_Fib_Retracement_500(df, swing_p=30, tol=0.01, **kw):
    close         = df['close']
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_lvl       = _fib_price(sw_low, sw_high, 0.5)
    near          = _near_fib(close, fib_lvl, tol)
    upswing       = sw_high > sw_high.shift(1)
    dnswing       = sw_low  < sw_low.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[near & upswing & (close > fib_lvl)] =  1
    sig[near & dnswing & (close < fib_lvl)] = -1
    return sig.fillna(0)


def space_Fib_Retracement_500(trial):
    return {
        'swing_p': trial.suggest_int('swing_p', 20, 60),
        'tol':     trial.suggest_float('tol',  0.005, 0.02),
    }


# ── 4. Fib_Extension_1272 ─────────────────────────────────────────────────────

def gen_Fib_Extension_1272(df, swing_p=30, ext=1.272, tol=0.01, **kw):
    close         = df['close']
    sw_low, sw_high = _swing_levels(df, swing_p)
    swing_range   = sw_high - sw_low
    ext_up        = sw_high + ext * swing_range    # extension above swing high
    ext_dn        = sw_low  - ext * swing_range    # extension below swing low
    near_up       = _near_fib(close, ext_up, tol)
    near_dn       = _near_fib(close, ext_dn, tol)
    sig = pd.Series(0, index=df.index)
    sig[near_up & (close > ext_up.shift(1))] = -1   # reversal short at extension
    sig[near_dn & (close < ext_dn.shift(1))] =  1   # reversal long at extension
    return sig.fillna(0)


def space_Fib_Extension_1272(trial):
    return {
        'swing_p': trial.suggest_int('swing_p',   20,  60),
        'ext':     trial.suggest_float('ext',      1.2,  1.35),
        'tol':     trial.suggest_float('tol',      0.005, 0.02),
    }


# ── 5. Fib_Extension_1618 ─────────────────────────────────────────────────────

def gen_Fib_Extension_1618(df, swing_p=30, ext=1.618, tol=0.01, **kw):
    close         = df['close']
    sw_low, sw_high = _swing_levels(df, swing_p)
    swing_range   = sw_high - sw_low
    ext_up        = sw_high + ext * swing_range
    ext_dn        = sw_low  - ext * swing_range
    near_up       = _near_fib(close, ext_up, tol)
    near_dn       = _near_fib(close, ext_dn, tol)
    sig = pd.Series(0, index=df.index)
    sig[near_up & (close >= ext_up)] = -1
    sig[near_dn & (close <= ext_dn)] =  1
    return sig.fillna(0)


def space_Fib_Extension_1618(trial):
    return {
        'swing_p': trial.suggest_int('swing_p',   20,  60),
        'ext':     trial.suggest_float('ext',      1.55, 1.72),
        'tol':     trial.suggest_float('tol',      0.005, 0.02),
    }


# ── 6. Fib_Multiple_Level ─────────────────────────────────────────────────────

def gen_Fib_Multiple_Level(df, swing_p=30, tol=0.01, **kw):
    close         = df['close']
    sw_low, sw_high = _swing_levels(df, swing_p)
    fibs_long  = [0.236, 0.382, 0.5, 0.618, 0.786]
    fibs_short = [1 - f for f in fibs_long]
    upswing    = sw_high > sw_high.shift(2)
    dnswing    = sw_low  < sw_low.shift(2)
    any_long   = pd.Series(False, index=df.index)
    any_short  = pd.Series(False, index=df.index)
    for f in fibs_long:
        lvl = _fib_price(sw_low, sw_high, f)
        any_long  |= _near_fib(close, lvl, tol)
    for f in fibs_short:
        lvl = _fib_price(sw_low, sw_high, f)
        any_short |= _near_fib(close, lvl, tol)
    sig = pd.Series(0, index=df.index)
    sig[any_long  & upswing & (close > sw_low)] =  1
    sig[any_short & dnswing & (close < sw_high)] = -1
    return sig.fillna(0)


def space_Fib_Multiple_Level(trial):
    return {
        'swing_p': trial.suggest_int('swing_p', 20, 60),
        'tol':     trial.suggest_float('tol',  0.005, 0.02),
    }


# ── 7. Fib_EMA_Confluence ─────────────────────────────────────────────────────

def gen_Fib_EMA_Confluence(df, swing_p=30, ema_p=30, tol=0.015, **kw):
    close         = df['close']
    ema_p         = int(ema_p)
    ema           = _ema(close, ema_p)
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_618       = _fib_price(sw_low, sw_high, 0.618)
    fib_382       = _fib_price(sw_low, sw_high, 0.382)
    near_618      = _near_fib(close, fib_618, tol)
    near_382      = _near_fib(close, fib_382, tol)
    near_ema_long  = _near_fib(close, ema, tol)
    confluence_long  = near_618 & near_ema_long & (close > sw_low)
    confluence_short = near_382 & near_ema_long & (close < sw_high)
    sig = pd.Series(0, index=df.index)
    sig[confluence_long]  =  1
    sig[confluence_short] = -1
    return sig.fillna(0)


def space_Fib_EMA_Confluence(trial):
    return {
        'swing_p': trial.suggest_int('swing_p',   20,  60),
        'ema_p':   trial.suggest_int('ema_p',      20,  60),
        'tol':     trial.suggest_float('tol',      0.01, 0.03),
    }


# ── 8. Fib_RSI_Entry ──────────────────────────────────────────────────────────

def gen_Fib_RSI_Entry(df, swing_p=30, fib=0.5, tol=0.015, rsi_p=14, **kw):
    close         = df['close']
    rsi_p         = int(rsi_p)
    rsi           = _rsi(close, rsi_p)
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_long      = _fib_price(sw_low, sw_high, fib)
    fib_short     = _fib_price(sw_low, sw_high, 1 - fib)
    near_long     = _near_fib(close, fib_long,  tol)
    near_short    = _near_fib(close, fib_short, tol)
    sig = pd.Series(0, index=df.index)
    sig[near_long  & (rsi < 50)] =  1
    sig[near_short & (rsi > 50)] = -1
    return sig.fillna(0)


def space_Fib_RSI_Entry(trial):
    return {
        'swing_p': trial.suggest_int('swing_p',   20,  60),
        'fib':     trial.suggest_float('fib',      0.3,  0.7),
        'tol':     trial.suggest_float('tol',      0.01, 0.03),
        'rsi_p':   trial.suggest_int('rsi_p',       7,  21),
    }


# ── 9. Fib_ATR_Zone ───────────────────────────────────────────────────────────

def gen_Fib_ATR_Zone(df, swing_p=30, fib=0.5, atr_p=14, zone_mult=1.0, **kw):
    close         = df['close']
    atr_p         = int(atr_p)
    atr           = _atr(df, atr_p)
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_long      = _fib_price(sw_low, sw_high, fib)
    fib_short     = _fib_price(sw_low, sw_high, 1 - fib)
    zone_up_low   = fib_long  - zone_mult * atr
    zone_up_high  = fib_long  + zone_mult * atr
    zone_dn_low   = fib_short - zone_mult * atr
    zone_dn_high  = fib_short + zone_mult * atr
    in_zone_long  = (close >= zone_up_low) & (close <= zone_up_high)
    in_zone_short = (close >= zone_dn_low) & (close <= zone_dn_high)
    upswing       = sw_high > sw_high.shift(2)
    dnswing       = sw_low  < sw_low.shift(2)
    sig = pd.Series(0, index=df.index)
    sig[in_zone_long  & upswing] =  1
    sig[in_zone_short & dnswing] = -1
    return sig.fillna(0)


def space_Fib_ATR_Zone(trial):
    return {
        'swing_p':   trial.suggest_int('swing_p',     20,  60),
        'fib':       trial.suggest_float('fib',        0.3,  0.7),
        'atr_p':     trial.suggest_int('atr_p',        10,  20),
        'zone_mult': trial.suggest_float('zone_mult',  0.5,  2.0),
    }


# ── 10. Fib_Time ──────────────────────────────────────────────────────────────

def gen_Fib_Time(df, lookback=20, **kw):
    close    = df['close']
    low      = df['low']
    high     = df['high']
    lookback = int(lookback)
    fib_nums = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    sig      = pd.Series(0, index=df.index)
    for i in range(lookback + max(fib_nums), len(df)):
        window = low.iloc[i - lookback: i]
        swing_low_idx = window.idxmin()
        bars_since = i - df.index.get_loc(swing_low_idx)
        if bars_since in fib_nums:
            # at fib time after swing low → long
            sig.iloc[i] = 1
        # swing high
        window_h = high.iloc[i - lookback: i]
        swing_hi_idx = window_h.idxmax()
        bars_since_h = i - df.index.get_loc(swing_hi_idx)
        if bars_since_h in fib_nums:
            sig.iloc[i] = -1
    return sig.fillna(0)


def space_Fib_Time(trial):
    return {'lookback': trial.suggest_int('lookback', 5, 30)}


# ── 11. Fib_Arcs ──────────────────────────────────────────────────────────────

def gen_Fib_Arcs(df, swing_p=30, fib_arc=0.5, **kw):
    close         = df['close']
    sw_low, sw_high = _swing_levels(df, swing_p)
    swing_range   = (sw_high - sw_low).abs()
    arc_radius    = fib_arc * swing_range
    # price within arc_radius of swing low = arc support
    dist_from_low  = (close - sw_low).abs()
    dist_from_high = (close - sw_high).abs()
    at_arc_low  = dist_from_low  <= arc_radius
    at_arc_high = dist_from_high <= arc_radius
    upswing     = sw_high > sw_high.shift(2)
    dnswing     = sw_low  < sw_low.shift(2)
    sig = pd.Series(0, index=df.index)
    sig[at_arc_low  & upswing & (close > sw_low)]  =  1
    sig[at_arc_high & dnswing & (close < sw_high)] = -1
    return sig.fillna(0)


def space_Fib_Arcs(trial):
    return {
        'swing_p':  trial.suggest_int('swing_p',    20, 50),
        'fib_arc':  trial.suggest_float('fib_arc',  0.3, 0.7),
    }


# ── 12. Fib_Fan ───────────────────────────────────────────────────────────────

def gen_Fib_Fan(df, swing_p=30, fib=0.382, **kw):
    close         = df['close']
    sw_low, sw_high = _swing_levels(df, swing_p)
    swing_range   = sw_high - sw_low
    # fan line = pivot low + fib * (swing_range proportional to bar offset)
    # simplified: price at fib fraction of swing range from low
    fan_level_long  = sw_low  + fib * swing_range
    fan_level_short = sw_high - fib * swing_range
    near_long       = _near_fib(close, fan_level_long,  0.015)
    near_short      = _near_fib(close, fan_level_short, 0.015)
    upswing         = sw_high > sw_high.shift(2)
    dnswing         = sw_low  < sw_low.shift(2)
    sig = pd.Series(0, index=df.index)
    sig[near_long  & upswing] =  1
    sig[near_short & dnswing] = -1
    return sig.fillna(0)


def space_Fib_Fan(trial):
    return {
        'swing_p': trial.suggest_int('swing_p', 20, 60),
        'fib':     trial.suggest_float('fib',    0.3, 0.5),
    }


# ── 13. Fib_Channel ───────────────────────────────────────────────────────────

def gen_Fib_Channel(df, trend_p=30, fib=0.5, **kw):
    close         = df['close']
    trend_p       = int(trend_p)
    sw_low, sw_high = _swing_levels(df, trend_p)
    channel_mid   = _fib_price(sw_low, sw_high, fib)
    channel_low   = sw_low
    channel_high  = sw_high
    near_low  = _near_fib(close, channel_low,  0.015)
    near_high = _near_fib(close, channel_high, 0.015)
    sig = pd.Series(0, index=df.index)
    sig[near_low  & (close > channel_low)]  =  1
    sig[near_high & (close < channel_high)] = -1
    return sig.fillna(0)


def space_Fib_Channel(trial):
    return {
        'trend_p': trial.suggest_int('trend_p', 20, 60),
        'fib':     trial.suggest_float('fib',    0.3, 0.7),
    }


# ── 14. Fib_BB_Combo ──────────────────────────────────────────────────────────

def gen_Fib_BB_Combo(df, swing_p=30, fib=0.5, bb_p=20, bb_mult=2.0, tol=0.015, **kw):
    close         = df['close']
    bb_p          = int(bb_p)
    sma           = _sma(close, bb_p)
    std           = close.rolling(bb_p, min_periods=1).std().fillna(0)
    bb_lower      = sma - bb_mult * std
    bb_upper      = sma + bb_mult * std
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_long      = _fib_price(sw_low, sw_high, fib)
    fib_short     = _fib_price(sw_low, sw_high, 1 - fib)
    near_long     = _near_fib(close, fib_long,  tol)
    near_short    = _near_fib(close, fib_short, tol)
    sig = pd.Series(0, index=df.index)
    sig[near_long  & (close < bb_lower)] =  1
    sig[near_short & (close > bb_upper)] = -1
    return sig.fillna(0)


def space_Fib_BB_Combo(trial):
    return {
        'swing_p':  trial.suggest_int('swing_p',   20,  60),
        'fib':      trial.suggest_float('fib',      0.3,  0.7),
        'bb_p':     trial.suggest_int('bb_p',       15,  30),
        'bb_mult':  trial.suggest_float('bb_mult',  1.5,  2.5),
        'tol':      trial.suggest_float('tol',      0.01, 0.03),
    }


# ── 15. Fib_Supertrend ────────────────────────────────────────────────────────

def _supertrend_dir(df, st_p, st_mult):
    atr   = _atr(df, int(st_p))
    hl2   = (df['high'] + df['low']) / 2
    upper = hl2 + st_mult * atr
    lower = hl2 - st_mult * atr
    close = df['close']
    trend = pd.Series(1, index=df.index)
    st    = lower.copy()
    for i in range(1, len(df)):
        if close.iloc[i - 1] > st.iloc[i - 1]:
            st.iloc[i]    = max(lower.iloc[i], st.iloc[i - 1])
            trend.iloc[i] = 1
        else:
            st.iloc[i]    = min(upper.iloc[i], st.iloc[i - 1])
            trend.iloc[i] = -1
    return trend


def gen_Fib_Supertrend(df, swing_p=30, fib=0.5, st_p=10, st_mult=3.0, tol=0.015, **kw):
    close         = df['close']
    st_p          = int(st_p)
    trend         = _supertrend_dir(df, st_p, st_mult)
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_long      = _fib_price(sw_low, sw_high, fib)
    fib_short     = _fib_price(sw_low, sw_high, 1 - fib)
    near_long     = _near_fib(close, fib_long,  tol)
    near_short    = _near_fib(close, fib_short, tol)
    sig = pd.Series(0, index=df.index)
    sig[near_long  & (trend ==  1)] =  1
    sig[near_short & (trend == -1)] = -1
    return sig.fillna(0)


def space_Fib_Supertrend(trial):
    return {
        'swing_p':  trial.suggest_int('swing_p',   20,  60),
        'fib':      trial.suggest_float('fib',      0.3,  0.7),
        'st_p':     trial.suggest_int('st_p',        7,  21),
        'st_mult':  trial.suggest_float('st_mult',  2.0,  4.0),
        'tol':      trial.suggest_float('tol',      0.01, 0.03),
    }


# ── 16. Fib_Volume_Profile ────────────────────────────────────────────────────

def gen_Fib_Volume_Profile(df, swing_p=30, fib=0.5, vol_p=20, tol=0.015, **kw):
    close         = df['close']
    vol           = df['volume']
    vol_p         = int(vol_p)
    vol_avg       = _sma(vol, vol_p)
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_long      = _fib_price(sw_low, sw_high, fib)
    fib_short     = _fib_price(sw_low, sw_high, 1 - fib)
    near_long     = _near_fib(close, fib_long,  tol)
    near_short    = _near_fib(close, fib_short, tol)
    high_vol      = vol > vol_avg
    sig = pd.Series(0, index=df.index)
    sig[near_long  & high_vol & (close > fib_long.shift(1))]  =  1
    sig[near_short & high_vol & (close < fib_short.shift(1))] = -1
    return sig.fillna(0)


def space_Fib_Volume_Profile(trial):
    return {
        'swing_p': trial.suggest_int('swing_p',   20,  60),
        'fib':     trial.suggest_float('fib',      0.3,  0.7),
        'vol_p':   trial.suggest_int('vol_p',       20,  50),
        'tol':     trial.suggest_float('tol',       0.01, 0.03),
    }


# ── 17. Fib_Pivot_Cluster ─────────────────────────────────────────────────────

def gen_Fib_Pivot_Cluster(df, lookback1=20, lookback2=40, tol=0.015, **kw):
    close          = df['close']
    lb1            = int(lookback1); lb2 = int(lookback2)
    sw_low1  = df['low'].rolling(lb1, min_periods=1).min()
    sw_high1 = df['high'].rolling(lb1, min_periods=1).max()
    sw_low2  = df['low'].rolling(lb2, min_periods=1).min()
    sw_high2 = df['high'].rolling(lb2, min_periods=1).max()
    fibs = [0.382, 0.5, 0.618]
    levels_long  = [_fib_price(sw_low1, sw_high1, f) for f in fibs] + \
                   [_fib_price(sw_low2, sw_high2, f) for f in fibs]
    levels_short = [_fib_price(sw_low1, sw_high1, 1-f) for f in fibs] + \
                   [_fib_price(sw_low2, sw_high2, 1-f) for f in fibs]
    cluster_long  = pd.Series(0, index=df.index)
    cluster_short = pd.Series(0, index=df.index)
    ref = close.replace(0, 1e-9)
    for lvl in levels_long:
        cluster_long  += ((close - lvl).abs() / ref < tol).astype(int)
    for lvl in levels_short:
        cluster_short += ((close - lvl).abs() / ref < tol).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[cluster_long  >= 2] =  1
    sig[cluster_short >= 2] = -1
    return sig.fillna(0)


def space_Fib_Pivot_Cluster(trial):
    return {
        'lookback1': trial.suggest_int('lookback1',   10,  30),
        'lookback2': trial.suggest_int('lookback2',   30,  60),
        'tol':       trial.suggest_float('tol',       0.01, 0.03),
    }


# ── 18. Fib_Scalp_382 ────────────────────────────────────────────────────────

def gen_Fib_Scalp_382(df, swing_p=20, fib=0.382, tol=0.01, **kw):
    close         = df['close']
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_long      = _fib_price(sw_low, sw_high, fib)
    fib_short     = _fib_price(sw_low, sw_high, 1 - fib)
    near_long     = _near_fib(close, fib_long,  tol)
    near_short    = _near_fib(close, fib_short, tol)
    # strong trend: swing range expanding
    strong_up  = (sw_high - sw_low) > (sw_high - sw_low).shift(swing_p)
    strong_dn  = (sw_high - sw_low) > (sw_high - sw_low).shift(swing_p)
    sig = pd.Series(0, index=df.index)
    sig[near_long  & strong_up] =  1
    sig[near_short & strong_dn] = -1
    return sig.fillna(0)


def space_Fib_Scalp_382(trial):
    return {
        'swing_p': trial.suggest_int('swing_p',   10,  30),
        'fib':     trial.suggest_float('fib',      0.30, 0.42),
        'tol':     trial.suggest_float('tol',      0.005, 0.015),
    }


# ── 19. Fib_Weekly_Level ──────────────────────────────────────────────────────

def gen_Fib_Weekly_Level(df, base_p=30, **kw):
    close         = df['close']
    base_p        = int(base_p)
    weekly_p      = base_p * 5
    sw_low  = df['low'].rolling(weekly_p, min_periods=1).min()
    sw_high = df['high'].rolling(weekly_p, min_periods=1).max()
    fibs    = [0.382, 0.5, 0.618]
    any_long  = pd.Series(False, index=df.index)
    any_short = pd.Series(False, index=df.index)
    for f in fibs:
        lvl_long  = _fib_price(sw_low, sw_high, f)
        lvl_short = _fib_price(sw_low, sw_high, 1 - f)
        any_long  |= _near_fib(close, lvl_long,  0.015)
        any_short |= _near_fib(close, lvl_short, 0.015)
    upswing = sw_high > sw_high.shift(base_p)
    dnswing = sw_low  < sw_low.shift(base_p)
    sig = pd.Series(0, index=df.index)
    sig[any_long  & upswing] =  1
    sig[any_short & dnswing] = -1
    return sig.fillna(0)


def space_Fib_Weekly_Level(trial):
    return {'base_p': trial.suggest_int('base_p', 20, 60)}


# ── 20. Fib_Breakout ──────────────────────────────────────────────────────────

def gen_Fib_Breakout(df, swing_p=30, fib=0.618, tol=0.01, **kw):
    close         = df['close']
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_res       = _fib_price(sw_low, sw_high, fib)       # retracement as resistance
    fib_sup       = _fib_price(sw_low, sw_high, 1 - fib)   # mirror as support
    # breakout: close crosses fib level from below
    cross_up = (close > fib_res) & (close.shift(1) <= fib_res.shift(1))
    cross_dn = (close < fib_sup) & (close.shift(1) >= fib_sup.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up] =  1
    sig[cross_dn] = -1
    return sig.fillna(0)


def space_Fib_Breakout(trial):
    return {
        'swing_p': trial.suggest_int('swing_p',   20,  60),
        'fib':     trial.suggest_float('fib',      0.55, 0.68),
        'tol':     trial.suggest_float('tol',      0.005, 0.02),
    }


# ── 21. Fib_Sequence_MA ───────────────────────────────────────────────────────

def gen_Fib_Sequence_MA(df, **kw):
    close    = df['close']
    fib_mas  = [8, 13, 21, 34]
    emas     = [_ema(close, p) for p in fib_mas]
    above_all = close > emas[0]
    below_all = close < emas[0]
    for e in emas[1:]:
        above_all &= close > e
        below_all &= close < e
    # pairwise comparison series: fast > slow for bull, fast < slow for bear
    bull_order = pd.Series(True, index=df.index)
    bear_order = pd.Series(True, index=df.index)
    for i in range(len(emas) - 1):
        bull_order &= (emas[i] > emas[i+1])
        bear_order &= (emas[i] < emas[i+1])
    sig = pd.Series(0, index=df.index)
    sig[above_all & bull_order] =  1
    sig[below_all & bear_order] = -1
    return sig.fillna(0)


def space_Fib_Sequence_MA(trial):
    # no meaningful params — return fixed
    return {}


# ── 22. Golden_Ratio_ATR ──────────────────────────────────────────────────────

def gen_Golden_Ratio_ATR(df, ema_p=30, atr_p=14, ratio=1.618, **kw):
    close  = df['close']
    ema_p  = int(ema_p); atr_p = int(atr_p)
    ema    = _ema(close, ema_p)
    atr    = _atr(df, atr_p)
    upper  = ema + ratio * atr
    lower  = ema - ratio * atr
    sig = pd.Series(0, index=df.index)
    sig[close > upper] =  1
    sig[close < lower] = -1
    return sig.fillna(0)


def space_Golden_Ratio_ATR(trial):
    return {
        'ema_p': trial.suggest_int('ema_p',    20, 60),
        'atr_p': trial.suggest_int('atr_p',    10, 20),
        'ratio': trial.suggest_float('ratio',  1.5, 2.0),
    }


# ── 23. Fib_RSI_Level ────────────────────────────────────────────────────────

def gen_Fib_RSI_Level(df, rsi_p=14, **kw):
    close  = df['close']
    rsi_p  = int(rsi_p)
    rsi    = _rsi(close, rsi_p)
    # RSI crosses above 38.2 from below = long
    cross_above_382 = (rsi > 38.2) & (rsi.shift(1) <= 38.2)
    cross_below_618 = (rsi < 61.8) & (rsi.shift(1) >= 61.8)
    sig = pd.Series(0, index=df.index)
    sig[cross_above_382] =  1
    sig[cross_below_618] = -1
    return sig.fillna(0)


def space_Fib_RSI_Level(trial):
    return {'rsi_p': trial.suggest_int('rsi_p', 7, 21)}


# ── 24. Fib_MACD ─────────────────────────────────────────────────────────────

def gen_Fib_MACD(df, swing_p=30, fib=0.5, fast=12, slow=26, sig_p=9, tol=0.015, **kw):
    close         = df['close']
    fast          = int(fast); slow = int(slow); sig_p = int(sig_p)
    macd          = _ema(close, fast) - _ema(close, slow)
    signal_line   = _ema(macd, sig_p)
    macd_cross_up = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
    macd_cross_dn = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_long      = _fib_price(sw_low, sw_high, fib)
    fib_short     = _fib_price(sw_low, sw_high, 1 - fib)
    near_long     = _near_fib(close, fib_long,  tol)
    near_short    = _near_fib(close, fib_short, tol)
    sig = pd.Series(0, index=df.index)
    sig[near_long  & macd_cross_up] =  1
    sig[near_short & macd_cross_dn] = -1
    return sig.fillna(0)


def space_Fib_MACD(trial):
    return {
        'swing_p': trial.suggest_int('swing_p',   20,  60),
        'fib':     trial.suggest_float('fib',      0.3,  0.7),
        'fast':    trial.suggest_int('fast',        8,  16),
        'slow':    trial.suggest_int('slow',        20,  30),
        'sig_p':   trial.suggest_int('sig_p',        5,  12),
        'tol':     trial.suggest_float('tol',       0.01, 0.03),
    }


# ── 25. Fib_Trend_Confluence ──────────────────────────────────────────────────

def gen_Fib_Trend_Confluence(df, swing_p=30, fib=0.5, fast_p=10, slow_p=30, tol=0.015, **kw):
    close         = df['close']
    fast_p        = int(fast_p); slow_p = int(slow_p)
    ema_f         = _ema(close, fast_p)
    ema_s         = _ema(close, slow_p)
    uptrend       = ema_f > ema_s
    dntrend       = ema_f < ema_s
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_long      = _fib_price(sw_low, sw_high, fib)
    fib_short     = _fib_price(sw_low, sw_high, 1 - fib)
    near_long     = _near_fib(close, fib_long,  tol)
    near_short    = _near_fib(close, fib_short, tol)
    sig = pd.Series(0, index=df.index)
    sig[near_long  & uptrend] =  1
    sig[near_short & dntrend] = -1
    return sig.fillna(0)


def space_Fib_Trend_Confluence(trial):
    return {
        'swing_p': trial.suggest_int('swing_p',   20,  60),
        'fib':     trial.suggest_float('fib',      0.3,  0.7),
        'fast_p':  trial.suggest_int('fast_p',      5,  20),
        'slow_p':  trial.suggest_int('slow_p',      20,  60),
        'tol':     trial.suggest_float('tol',       0.01, 0.03),
    }


# ── 26. Fib_2618 ──────────────────────────────────────────────────────────────

def gen_Fib_2618(df, swing_p=30, ext=2.618, tol=0.015, **kw):
    close         = df['close']
    sw_low, sw_high = _swing_levels(df, swing_p)
    swing_range   = sw_high - sw_low
    ext_up        = sw_high + ext * swing_range
    ext_dn        = sw_low  - ext * swing_range
    near_up       = _near_fib(close, ext_up, tol)
    near_dn       = _near_fib(close, ext_dn, tol)
    sig = pd.Series(0, index=df.index)
    sig[near_up] = -1   # potential reversal short at 2.618
    sig[near_dn] =  1   # potential reversal long at 2.618
    return sig.fillna(0)


def space_Fib_2618(trial):
    return {
        'swing_p': trial.suggest_int('swing_p',   20,  60),
        'ext':     trial.suggest_float('ext',      2.5,  2.8),
        'tol':     trial.suggest_float('tol',      0.01, 0.03),
    }


# ── 27. Fib_Inside_Level ──────────────────────────────────────────────────────

def gen_Fib_Inside_Level(df, swing_p=30, fib_low=0.382, fib_high=0.618, tol=0.015, **kw):
    close         = df['close']
    high          = df['high']
    low           = df['low']
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_lo        = _fib_price(sw_low, sw_high, fib_low)
    fib_hi        = _fib_price(sw_low, sw_high, fib_high)
    inside        = (high < high.shift(1)) & (low > low.shift(1))
    near_fib_zone = (close >= fib_lo * (1 - tol)) & (close <= fib_hi * (1 + tol))
    # inside bar at fib zone → breakout direction
    prev_inside_at_fib = inside.shift(1).fillna(False) & near_fib_zone.shift(1).fillna(False)
    sig = pd.Series(0, index=df.index)
    sig[prev_inside_at_fib & (close > high.shift(1))] =  1
    sig[prev_inside_at_fib & (close < low.shift(1))]  = -1
    return sig.fillna(0)


def space_Fib_Inside_Level(trial):
    return {
        'swing_p':  trial.suggest_int('swing_p',    20,  60),
        'fib_low':  trial.suggest_float('fib_low',  0.3, 0.45),
        'fib_high': trial.suggest_float('fib_high', 0.55, 0.7),
        'tol':      trial.suggest_float('tol',       0.01, 0.03),
    }


# ── 28. Fib_Gap_Fill ──────────────────────────────────────────────────────────

def gen_Fib_Gap_Fill(df, swing_p=30, fib=0.5, min_gap=0.005, tol=0.015, **kw):
    close        = df['close']
    open_        = df['open']
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_long     = _fib_price(sw_low, sw_high, fib)
    fib_short    = _fib_price(sw_low, sw_high, 1 - fib)
    prev_close   = close.shift(1)
    gap_pct      = (open_ - prev_close) / prev_close.replace(0, 1e-9)
    gap_down     = gap_pct < -min_gap
    gap_up       = gap_pct >  min_gap
    near_fib_l   = _near_fib(close, fib_long,  tol)
    near_fib_s   = _near_fib(close, fib_short, tol)
    sig = pd.Series(0, index=df.index)
    sig[gap_down & near_fib_l] =  1   # gap down to fib support → fill long
    sig[gap_up   & near_fib_s] = -1   # gap up to fib resistance → fill short
    return sig.fillna(0)


def space_Fib_Gap_Fill(trial):
    return {
        'swing_p': trial.suggest_int('swing_p',   20,  60),
        'fib':     trial.suggest_float('fib',      0.3,  0.7),
        'min_gap': trial.suggest_float('min_gap',  0.002, 0.01),
        'tol':     trial.suggest_float('tol',      0.01, 0.03),
    }


# ── 29. Fib_ATR_Trail ─────────────────────────────────────────────────────────

def gen_Fib_ATR_Trail(df, swing_p=30, fib=0.5, atr_p=14, atr_mult=2.0, tol=0.015, **kw):
    close         = df['close']
    atr_p         = int(atr_p)
    atr           = _atr(df, atr_p)
    sw_low, sw_high = _swing_levels(df, swing_p)
    fib_long      = _fib_price(sw_low, sw_high, fib)
    fib_short     = _fib_price(sw_low, sw_high, 1 - fib)
    near_long     = _near_fib(close, fib_long,  tol)
    near_short    = _near_fib(close, fib_short, tol)
    # dynamic TP trail: ATR * mult above/below entry
    trail_up      = close + atr * atr_mult
    trail_dn      = close - atr * atr_mult
    # entry at fib, expect trail
    sig = pd.Series(0, index=df.index)
    sig[near_long  & (close < trail_up)] =  1
    sig[near_short & (close > trail_dn)] = -1
    return sig.fillna(0)


def space_Fib_ATR_Trail(trial):
    return {
        'swing_p':  trial.suggest_int('swing_p',   20,  60),
        'fib':      trial.suggest_float('fib',      0.3,  0.7),
        'atr_p':    trial.suggest_int('atr_p',      10,  20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5,  3.0),
        'tol':      trial.suggest_float('tol',      0.01, 0.03),
    }


# ── 30. Fib_Cluster_MR ───────────────────────────────────────────────────────

def gen_Fib_Cluster_MR(df, tol=0.015, rsi_p=14, **kw):
    close  = df['close']
    rsi_p  = int(rsi_p)
    rsi    = _rsi(close, rsi_p)
    # use two lookback windows and multiple fibs to find clusters
    fibs   = [0.236, 0.382, 0.5, 0.618, 0.786]
    periods = [20, 40]
    ref    = close.replace(0, 1e-9)
    cluster_long  = pd.Series(0, index=df.index)
    cluster_short = pd.Series(0, index=df.index)
    for p in periods:
        sw_low  = df['low'].rolling(p, min_periods=1).min()
        sw_high = df['high'].rolling(p, min_periods=1).max()
        for f in fibs:
            lvl_l = _fib_price(sw_low, sw_high, f)
            lvl_s = _fib_price(sw_low, sw_high, 1 - f)
            cluster_long  += ((close - lvl_l).abs() / ref < tol).astype(int)
            cluster_short += ((close - lvl_s).abs() / ref < tol).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[(cluster_long  >= 3) & (rsi < 35)] =  1
    sig[(cluster_short >= 3) & (rsi > 65)] = -1
    return sig.fillna(0)


def space_Fib_Cluster_MR(trial):
    return {
        'tol':   trial.suggest_float('tol',   0.01, 0.03),
        'rsi_p': trial.suggest_int('rsi_p',    7,  21),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Fib_Retracement_382': {
        'gen': gen_Fib_Retracement_382,
        'space': space_Fib_Retracement_382,
        'default_params': {'swing_p': 30, 'fib': 0.382, 'tol': 0.01},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': '38.2% Fibonacci retracement entry in uptrend. Bidirectional.'},
    },
    'Fib_Retracement_618': {
        'gen': gen_Fib_Retracement_618,
        'space': space_Fib_Retracement_618,
        'default_params': {'swing_p': 30, 'fib': 0.618, 'tol': 0.01, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 5000,
                 'description': '61.8% golden ratio retracement + RSI<50 confirmation. Bidirectional.'},
    },
    'Fib_Retracement_500': {
        'gen': gen_Fib_Retracement_500,
        'space': space_Fib_Retracement_500,
        'default_params': {'swing_p': 30, 'tol': 0.01},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': '50% midpoint retracement. Bidirectional.'},
    },
    'Fib_Extension_1272': {
        'gen': gen_Fib_Extension_1272,
        'space': space_Fib_Extension_1272,
        'default_params': {'swing_p': 30, 'ext': 1.272, 'tol': 0.01},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': '1.272 Fib extension reversal signal. Bidirectional.'},
    },
    'Fib_Extension_1618': {
        'gen': gen_Fib_Extension_1618,
        'space': space_Fib_Extension_1618,
        'default_params': {'swing_p': 30, 'ext': 1.618, 'tol': 0.01},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': '1.618 golden ratio extension reversal. Bidirectional.'},
    },
    'Fib_Multiple_Level': {
        'gen': gen_Fib_Multiple_Level,
        'space': space_Fib_Multiple_Level,
        'default_params': {'swing_p': 30, 'tol': 0.01},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Any of 5 Fib levels (0.236/0.382/0.5/0.618/0.786). Bidirectional.'},
    },
    'Fib_EMA_Confluence': {
        'gen': gen_Fib_EMA_Confluence,
        'space': space_Fib_EMA_Confluence,
        'default_params': {'swing_p': 30, 'ema_p': 30, 'tol': 0.015},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Fib level + EMA at same price = double confluence. Bidirectional.'},
    },
    'Fib_RSI_Entry': {
        'gen': gen_Fib_RSI_Entry,
        'space': space_Fib_RSI_Entry,
        'default_params': {'swing_p': 30, 'fib': 0.5, 'tol': 0.015, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Fib retracement + RSI<50 or >50 confirmation. Bidirectional.'},
    },
    'Fib_ATR_Zone': {
        'gen': gen_Fib_ATR_Zone,
        'space': space_Fib_ATR_Zone,
        'default_params': {'swing_p': 30, 'fib': 0.5, 'atr_p': 14, 'zone_mult': 1.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Fib zone ± ATR band for dynamic entry zone. Bidirectional.'},
    },
    'Fib_Time': {
        'gen': gen_Fib_Time,
        'space': space_Fib_Time,
        'default_params': {'lookback': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Fibonacci time zones: signal at 1,2,3,5,8,13,21 bars after swing. Bidirectional.'},
    },
    'Fib_Arcs': {
        'gen': gen_Fib_Arcs,
        'space': space_Fib_Arcs,
        'default_params': {'swing_p': 30, 'fib_arc': 0.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Fib arcs: circular levels from pivot at fib ratio. Bidirectional.'},
    },
    'Fib_Fan': {
        'gen': gen_Fib_Fan,
        'space': space_Fib_Fan,
        'default_params': {'swing_p': 30, 'fib': 0.382},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Fib fan bounce from 38.2% fan line. Bidirectional.'},
    },
    'Fib_Channel': {
        'gen': gen_Fib_Channel,
        'space': space_Fib_Channel,
        'default_params': {'trend_p': 30, 'fib': 0.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Fib channel: bounce from lower or upper channel. Bidirectional.'},
    },
    'Fib_BB_Combo': {
        'gen': gen_Fib_BB_Combo,
        'space': space_Fib_BB_Combo,
        'default_params': {'swing_p': 30, 'fib': 0.5, 'bb_p': 20, 'bb_mult': 2.0, 'tol': 0.015},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Fib level + Bollinger Bands double confirmation. Bidirectional.'},
    },
    'Fib_Supertrend': {
        'gen': gen_Fib_Supertrend,
        'space': space_Fib_Supertrend,
        'default_params': {'swing_p': 30, 'fib': 0.5, 'st_p': 10, 'st_mult': 3.0, 'tol': 0.015},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Fib level + SuperTrend direction. Bidirectional.'},
    },
    'Fib_Volume_Profile': {
        'gen': gen_Fib_Volume_Profile,
        'space': space_Fib_Volume_Profile,
        'default_params': {'swing_p': 30, 'fib': 0.5, 'vol_p': 20, 'tol': 0.015},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Fib level + volume cluster confirmation. Bidirectional.'},
    },
    'Fib_Pivot_Cluster': {
        'gen': gen_Fib_Pivot_Cluster,
        'space': space_Fib_Pivot_Cluster,
        'default_params': {'lookback1': 20, 'lookback2': 40, 'tol': 0.015},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': '2+ Fib levels from different swings at same zone. Bidirectional.'},
    },
    'Fib_Scalp_382': {
        'gen': gen_Fib_Scalp_382,
        'space': space_Fib_Scalp_382,
        'default_params': {'swing_p': 20, 'fib': 0.382, 'tol': 0.01},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Quick scalp from 38.2% retracement in strong trends. Bidirectional.'},
    },
    'Fib_Weekly_Level': {
        'gen': gen_Fib_Weekly_Level,
        'space': space_Fib_Weekly_Level,
        'default_params': {'base_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Weekly-equivalent Fib levels (5x period). Bidirectional.'},
    },
    'Fib_Breakout': {
        'gen': gen_Fib_Breakout,
        'space': space_Fib_Breakout,
        'default_params': {'swing_p': 30, 'fib': 0.618, 'tol': 0.01},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Price breaks through 61.8% Fib resistance = continuation. Bidirectional.'},
    },
    'Fib_Sequence_MA': {
        'gen': gen_Fib_Sequence_MA,
        'space': space_Fib_Sequence_MA,
        'default_params': {},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'MA periods at Fib numbers (8,13,21,34). Price above/below all. Bidirectional.'},
    },
    'Golden_Ratio_ATR': {
        'gen': gen_Golden_Ratio_ATR,
        'space': space_Golden_Ratio_ATR,
        'default_params': {'ema_p': 30, 'atr_p': 14, 'ratio': 1.618},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Entry when price moves golden ratio of ATR from EMA. Bidirectional.'},
    },
    'Fib_RSI_Level': {
        'gen': gen_Fib_RSI_Level,
        'space': space_Fib_RSI_Level,
        'default_params': {'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'RSI crosses above 38.2 (long) or below 61.8 (short). Bidirectional.'},
    },
    'Fib_MACD': {
        'gen': gen_Fib_MACD,
        'space': space_Fib_MACD,
        'default_params': {'swing_p': 30, 'fib': 0.5, 'fast': 12, 'slow': 26, 'sig_p': 9, 'tol': 0.015},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Fib support + MACD crossover. Bidirectional.'},
    },
    'Fib_Trend_Confluence': {
        'gen': gen_Fib_Trend_Confluence,
        'space': space_Fib_Trend_Confluence,
        'default_params': {'swing_p': 30, 'fib': 0.5, 'fast_p': 10, 'slow_p': 30, 'tol': 0.015},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'EMA trend + Fib retracement confluence. Bidirectional.'},
    },
    'Fib_2618': {
        'gen': gen_Fib_2618,
        'space': space_Fib_2618,
        'default_params': {'swing_p': 30, 'ext': 2.618, 'tol': 0.015},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': '2.618 double extension reversal zone. Bidirectional.'},
    },
    'Fib_Inside_Level': {
        'gen': gen_Fib_Inside_Level,
        'space': space_Fib_Inside_Level,
        'default_params': {'swing_p': 30, 'fib_low': 0.382, 'fib_high': 0.618, 'tol': 0.015},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Inside bar at 38.2-61.8% Fib zone, breakout direction. Bidirectional.'},
    },
    'Fib_Gap_Fill': {
        'gen': gen_Fib_Gap_Fill,
        'space': space_Fib_Gap_Fill,
        'default_params': {'swing_p': 30, 'fib': 0.5, 'min_gap': 0.005, 'tol': 0.015},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Gap fill target at Fib support/resistance. Bidirectional.'},
    },
    'Fib_ATR_Trail': {
        'gen': gen_Fib_ATR_Trail,
        'space': space_Fib_ATR_Trail,
        'default_params': {'swing_p': 30, 'fib': 0.5, 'atr_p': 14, 'atr_mult': 2.0, 'tol': 0.015},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Fib entry + ATR trailing as dynamic TP target. Bidirectional.'},
    },
    'Fib_Cluster_MR': {
        'gen': gen_Fib_Cluster_MR,
        'space': space_Fib_Cluster_MR,
        'default_params': {'tol': 0.015, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Multiple Fib levels converge + RSI extreme = mean reversion. Bidirectional.'},
    },
}
