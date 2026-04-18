#!/usr/bin/env python3
"""TV2 BATCH 31 — 30 Price Action + S/R Strategies 2026-04-01"""

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


# ── 1. PA_HH_HL_Trend ────────────────────────────────────────────────────────

def gen_PA_HH_HL_Trend(df, lookback=2, **kw):
    high  = df['high']
    low   = df['low']
    lb    = int(lookback)
    ph    = high.shift(lb)
    pl    = low.shift(lb)
    hh    = high > ph
    hl    = low  > pl
    lh    = high < ph
    ll    = low  < pl
    sig   = pd.Series(0, index=df.index)
    sig[hh & hl]  =  1
    sig[lh & ll]  = -1
    return sig.fillna(0)


def space_PA_HH_HL_Trend(trial):
    return {'lookback': trial.suggest_int('lookback', 2, 5)}


# ── 2. PA_Breakout_Vol ────────────────────────────────────────────────────────

def gen_PA_Breakout_Vol(df, break_p=20, vol_p=20, vol_mult=2.0, **kw):
    close    = df['close']
    vol      = df['volume']
    break_p  = int(break_p); vol_p = int(vol_p)
    highest  = close.rolling(break_p, min_periods=1).max().shift(1)
    lowest   = close.rolling(break_p, min_periods=1).min().shift(1)
    vol_avg  = _sma(vol, vol_p)
    vol_ok   = vol > vol_avg * vol_mult
    sig = pd.Series(0, index=df.index)
    sig[(close > highest) & vol_ok] =  1
    sig[(close < lowest)  & vol_ok] = -1
    return sig.fillna(0)


def space_PA_Breakout_Vol(trial):
    return {
        'break_p':  trial.suggest_int('break_p',     10, 50),
        'vol_p':    trial.suggest_int('vol_p',        20, 50),
        'vol_mult': trial.suggest_float('vol_mult',  1.5, 3.0),
    }


# ── 3. PA_Breakdown_Vol ───────────────────────────────────────────────────────

def gen_PA_Breakdown_Vol(df, break_p=20, vol_p=20, vol_mult=1.5, **kw):
    close    = df['close']
    vol      = df['volume']
    break_p  = int(break_p); vol_p = int(vol_p)
    highest  = close.rolling(break_p, min_periods=1).max().shift(1)
    lowest   = close.rolling(break_p, min_periods=1).min().shift(1)
    vol_avg  = _sma(vol, vol_p)
    vol_ok   = vol > vol_avg * vol_mult
    sig = pd.Series(0, index=df.index)
    sig[(close > highest) & vol_ok] =  1   # breakout also valid long
    sig[(close < lowest)  & vol_ok] = -1
    return sig.fillna(0)


def space_PA_Breakdown_Vol(trial):
    return {
        'break_p':  trial.suggest_int('break_p',     10, 50),
        'vol_p':    trial.suggest_int('vol_p',        20, 50),
        'vol_mult': trial.suggest_float('vol_mult',  1.2, 3.0),
    }


# ── 4. PA_Key_Level_Bounce ────────────────────────────────────────────────────

def gen_PA_Key_Level_Bounce(df, level_size=100, atr_p=14, **kw):
    close      = df['close']
    level_size = float(level_size); atr_p = int(atr_p)
    atr        = _atr(df, atr_p)
    # nearest round level
    rounded    = (close / level_size).round() * level_size
    dist       = (close - rounded).abs()
    near_level = dist < atr
    prev_close = close.shift(1)
    bounce_up  = near_level & (close > prev_close) & (close > rounded)
    bounce_dn  = near_level & (close < prev_close) & (close < rounded)
    sig = pd.Series(0, index=df.index)
    sig[bounce_up] =  1
    sig[bounce_dn] = -1
    return sig.fillna(0)


def space_PA_Key_Level_Bounce(trial):
    return {
        'level_size': trial.suggest_int('level_size', 10, 100),
        'atr_p':      trial.suggest_int('atr_p',      10,  20),
    }


# ── 5. PA_Tight_SL ────────────────────────────────────────────────────────────

def gen_PA_Tight_SL(df, look_p=5, **kw):
    high  = df['high']
    low   = df['low']
    close = df['close']
    look_p = int(look_p)
    # inside bar: high < prev high AND low > prev low
    inside = (high < high.shift(1)) & (low > low.shift(1))
    # breakout after inside bar
    bo_long  = inside.shift(1).fillna(False) & (close > high.shift(1))
    bo_short = inside.shift(1).fillna(False) & (close < low.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[bo_long]  =  1
    sig[bo_short] = -1
    return sig.fillna(0)


def space_PA_Tight_SL(trial):
    return {'look_p': trial.suggest_int('look_p', 3, 10)}


# ── 6. PA_Momentum_Candle ─────────────────────────────────────────────────────

def gen_PA_Momentum_Candle(df, body_mult=1.5, atr_p=14, vol_p=20, **kw):
    close    = df['close']
    open_    = df['open']
    vol      = df['volume']
    atr_p    = int(atr_p); vol_p = int(vol_p)
    atr      = _atr(df, atr_p)
    vol_avg  = _sma(vol, vol_p)
    body     = (close - open_).abs()
    large    = body > atr * body_mult
    vol_ok   = vol > vol_avg
    bull     = close > open_
    bear     = close < open_
    sig = pd.Series(0, index=df.index)
    sig[large & vol_ok & bull] =  1
    sig[large & vol_ok & bear] = -1
    return sig.fillna(0)


def space_PA_Momentum_Candle(trial):
    return {
        'body_mult': trial.suggest_float('body_mult', 1.0, 3.0),
        'atr_p':     trial.suggest_int('atr_p',       10,  20),
        'vol_p':     trial.suggest_int('vol_p',        20,  50),
    }


# ── 7. PA_Rejection_Wick ──────────────────────────────────────────────────────

def gen_PA_Rejection_Wick(df, wick_ratio=2.0, support_p=20, ema_p=21, **kw):
    close    = df['close']
    open_    = df['open']
    high     = df['high']
    low      = df['low']
    support_p = int(support_p); ema_p = int(ema_p)
    ema       = _ema(close, ema_p)
    support   = low.rolling(support_p, min_periods=1).min().shift(1)
    resistance = high.rolling(support_p, min_periods=1).max().shift(1)
    body      = (close - open_).abs().replace(0, 1e-9)
    lower_wick = open_.combine(close, min) - low
    upper_wick = high - open_.combine(close, max)
    long_lower = lower_wick > body * wick_ratio
    long_upper = upper_wick > body * wick_ratio
    near_support    = low < support * 1.005
    near_resistance = high > resistance * 0.995
    sig = pd.Series(0, index=df.index)
    sig[long_lower & near_support    & (close > ema)] =  1
    sig[long_upper & near_resistance & (close < ema)] = -1
    return sig.fillna(0)


def space_PA_Rejection_Wick(trial):
    return {
        'wick_ratio': trial.suggest_float('wick_ratio', 1.5, 4.0),
        'support_p':  trial.suggest_int('support_p',    10,  30),
        'ema_p':      trial.suggest_int('ema_p',        14,  50),
    }


# ── 8. PA_Consolidation_Break ─────────────────────────────────────────────────

def gen_PA_Consolidation_Break(df, atr_p=14, consol_bars=5, expand_mult=1.5, **kw):
    close      = df['close']
    atr_p      = int(atr_p); consol_bars = int(consol_bars)
    atr        = _atr(df, atr_p)
    atr_avg    = _sma(atr, atr_p)
    # consolidation: atr was below average for consol_bars bars
    low_atr    = atr < atr_avg * 0.7
    consol     = low_atr.rolling(consol_bars, min_periods=consol_bars).min().fillna(0).astype(bool)
    # expansion now
    expanding  = atr > atr_avg * expand_mult
    prev_consol = consol.shift(1).fillna(False)
    sig = pd.Series(0, index=df.index)
    sig[prev_consol & expanding & (close > close.shift(1))] =  1
    sig[prev_consol & expanding & (close < close.shift(1))] = -1
    return sig.fillna(0)


def space_PA_Consolidation_Break(trial):
    return {
        'atr_p':       trial.suggest_int('atr_p',          10, 20),
        'consol_bars': trial.suggest_int('consol_bars',      3, 10),
        'expand_mult': trial.suggest_float('expand_mult',  1.3, 2.5),
    }


# ── 9. PA_Two_Day_Rule ────────────────────────────────────────────────────────

def gen_PA_Two_Day_Rule(df, n_bars=2, **kw):
    close   = df['close']
    high    = df['high']
    low     = df['low']
    n_bars  = int(n_bars)
    # resistance = highest high n_bars ago
    resistance = high.shift(n_bars)
    support    = low.shift(n_bars)
    sig = pd.Series(0, index=df.index)
    sig[close > resistance] =  1
    sig[close < support]    = -1
    return sig.fillna(0)


def space_PA_Two_Day_Rule(trial):
    return {'n_bars': trial.suggest_int('n_bars', 2, 10)}


# ── 10. PA_Trend_Continuation ─────────────────────────────────────────────────

def gen_PA_Trend_Continuation(df, trend_bars=3, pullback_bars=2, max_retrace=0.5, **kw):
    close        = df['close']
    high         = df['high']
    trend_bars   = int(trend_bars); pullback_bars = int(pullback_bars)
    # count consecutive up/down bars
    up_bar   = (close > close.shift(1)).astype(int)
    dn_bar   = (close < close.shift(1)).astype(int)
    # rolling sum for trend bars
    consec_up = up_bar.rolling(trend_bars, min_periods=trend_bars).sum() == trend_bars
    consec_dn = dn_bar.rolling(trend_bars, min_periods=trend_bars).sum() == trend_bars
    # pullback after uptrend
    pb_dn = dn_bar.rolling(pullback_bars, min_periods=pullback_bars).sum() >= 1
    pb_up = up_bar.rolling(pullback_bars, min_periods=pullback_bars).sum() >= 1
    prior_swing_high = high.rolling(trend_bars + pullback_bars, min_periods=1).max().shift(pullback_bars)
    # continuation: break of prior high after pullback
    continuation_long  = consec_up.shift(pullback_bars).fillna(False) & pb_dn & (close > prior_swing_high)
    continuation_short = consec_dn.shift(pullback_bars).fillna(False) & pb_up & (close < close.rolling(trend_bars + pullback_bars, min_periods=1).min().shift(pullback_bars))
    sig = pd.Series(0, index=df.index)
    sig[continuation_long]  =  1
    sig[continuation_short] = -1
    return sig.fillna(0)


def space_PA_Trend_Continuation(trial):
    return {
        'trend_bars':   trial.suggest_int('trend_bars',         2,   5),
        'pullback_bars':trial.suggest_int('pullback_bars',       1,   3),
        'max_retrace':  trial.suggest_float('max_retrace',      0.3, 0.7),
    }


# ── 11. PA_Opening_Range ──────────────────────────────────────────────────────

def gen_PA_Opening_Range(df, range_bars=10, **kw):
    close      = df['close']
    high       = df['high']
    low        = df['low']
    range_bars = int(range_bars)
    range_high = high.rolling(range_bars, min_periods=range_bars).max().shift(1)
    range_low  = low.rolling(range_bars, min_periods=range_bars).min().shift(1)
    sig = pd.Series(0, index=df.index)
    sig[close > range_high] =  1
    sig[close < range_low]  = -1
    return sig.fillna(0)


def space_PA_Opening_Range(trial):
    return {'range_bars': trial.suggest_int('range_bars', 5, 20)}


# ── 12. PA_Close_Above_High ───────────────────────────────────────────────────

def gen_PA_Close_Above_High(df, ema_p=50, **kw):
    close  = df['close']
    high   = df['high']
    low    = df['low']
    ema_p  = int(ema_p)
    ema    = _ema(close, ema_p)
    sig = pd.Series(0, index=df.index)
    sig[(close > high.shift(1)) & (close > ema)]  =  1
    sig[(close < low.shift(1))  & (close < ema)]  = -1
    return sig.fillna(0)


def space_PA_Close_Above_High(trial):
    return {'ema_p': trial.suggest_int('ema_p', 20, 100)}


# ── 13. PA_Body_Full ──────────────────────────────────────────────────────────

def gen_PA_Body_Full(df, body_pct=0.8, ema_p=21, **kw):
    close   = df['close']
    open_   = df['open']
    high    = df['high']
    low     = df['low']
    ema_p   = int(ema_p)
    ema     = _ema(close, ema_p)
    range_  = (high - low).replace(0, 1e-9)
    body    = (close - open_).abs()
    full    = body / range_ > body_pct
    bull    = close > open_
    bear    = close < open_
    sig = pd.Series(0, index=df.index)
    sig[full & bull & (close > ema)] =  1
    sig[full & bear & (close < ema)] = -1
    return sig.fillna(0)


def space_PA_Body_Full(trial):
    return {
        'body_pct': trial.suggest_float('body_pct', 0.7, 0.95),
        'ema_p':    trial.suggest_int('ema_p',       14,  50),
    }


# ── 14. PA_ATR_Momentum ───────────────────────────────────────────────────────

def gen_PA_ATR_Momentum(df, atr_p=14, mult=1.5, **kw):
    close  = df['close']
    open_  = df['open']
    atr_p  = int(atr_p)
    atr    = _atr(df, atr_p)
    move   = close - open_
    sig = pd.Series(0, index=df.index)
    sig[move >  atr * mult] =  1
    sig[move < -atr * mult] = -1
    return sig.fillna(0)


def space_PA_ATR_Momentum(trial):
    return {
        'atr_p': trial.suggest_int('atr_p',   10, 20),
        'mult':  trial.suggest_float('mult',  1.0, 3.0),
    }


# ── 15. PA_Support_Test ───────────────────────────────────────────────────────

def gen_PA_Support_Test(df, support_p=20, tol=0.002, **kw):
    close     = df['close']
    low       = df['low']
    support_p = int(support_p)
    support   = low.rolling(support_p, min_periods=1).min().shift(1)
    resistance = df['high'].rolling(support_p, min_periods=1).max().shift(1)
    # test support and hold
    test_support    = (low <= support * (1 + tol)) & (close > support)
    test_resistance = (df['high'] >= resistance * (1 - tol)) & (close < resistance)
    sig = pd.Series(0, index=df.index)
    sig[test_support]    =  1
    sig[test_resistance] = -1
    return sig.fillna(0)


def space_PA_Support_Test(trial):
    return {
        'support_p': trial.suggest_int('support_p',   10,  30),
        'tol':       trial.suggest_float('tol',      0.001, 0.005),
    }


# ── 16. PA_Resistance_Break ───────────────────────────────────────────────────

def gen_PA_Resistance_Break(df, res_p=20, **kw):
    close  = df['close']
    high   = df['high']
    low    = df['low']
    res_p  = int(res_p)
    resistance = high.rolling(res_p, min_periods=1).max().shift(1)
    support    = low.rolling(res_p, min_periods=1).min().shift(1)
    break_up   = (close > resistance) & (close.shift(1) > resistance.shift(1))
    break_dn   = (close < support)    & (close.shift(1) < support.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[break_up] =  1
    sig[break_dn] = -1
    return sig.fillna(0)


def space_PA_Resistance_Break(trial):
    return {'res_p': trial.suggest_int('res_p', 10, 50)}


# ── 17. PA_Channel_Trade ──────────────────────────────────────────────────────

def gen_PA_Channel_Trade(df, channel_p=30, k=1.5, **kw):
    close     = df['close']
    channel_p = int(channel_p)
    sma       = _sma(close, channel_p)
    std       = close.rolling(channel_p, min_periods=1).std().fillna(0)
    upper     = sma + k * std
    lower     = sma - k * std
    sig = pd.Series(0, index=df.index)
    sig[close <= lower] =  1
    sig[close >= upper] = -1
    return sig.fillna(0)


def space_PA_Channel_Trade(trial):
    return {
        'channel_p': trial.suggest_int('channel_p',   20, 60),
        'k':         trial.suggest_float('k',          1.0, 2.5),
    }


# ── 18. PA_Trend_Touch_EMA ────────────────────────────────────────────────────

def gen_PA_Trend_Touch_EMA(df, fast_p=10, slow_p=50, tol=0.002, **kw):
    close   = df['close']
    fast_p  = int(fast_p); slow_p = int(slow_p)
    ema_f   = _ema(close, fast_p)
    ema_s   = _ema(close, slow_p)
    uptrend = ema_f > ema_s
    dntrend = ema_f < ema_s
    near_fast_up = (close - ema_f).abs() / ema_f.replace(0, 1e-9) < tol
    near_fast_dn = (close - ema_f).abs() / ema_f.replace(0, 1e-9) < tol
    sig = pd.Series(0, index=df.index)
    sig[uptrend & near_fast_up] =  1
    sig[dntrend & near_fast_dn] = -1
    return sig.fillna(0)


def space_PA_Trend_Touch_EMA(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',   10,  20),
        'slow_p': trial.suggest_int('slow_p',   40, 100),
        'tol':    trial.suggest_float('tol',  0.001, 0.005),
    }


# ── 19. PA_Gap_Fill_EMA ───────────────────────────────────────────────────────

def gen_PA_Gap_Fill_EMA(df, min_gap_pct=0.005, ema_p=30, **kw):
    close       = df['close']
    open_       = df['open']
    ema_p       = int(ema_p)
    ema         = _ema(close, ema_p)
    prev_close  = close.shift(1)
    gap_pct     = (open_ - prev_close) / prev_close.replace(0, 1e-9)
    gap_down    = gap_pct < -min_gap_pct
    gap_up      = gap_pct >  min_gap_pct
    # gap down but EMA is above open → expect fill upward
    fill_long   = gap_down & (ema > open_)
    fill_short  = gap_up   & (ema < open_)
    sig = pd.Series(0, index=df.index)
    sig[fill_long]  =  1
    sig[fill_short] = -1
    return sig.fillna(0)


def space_PA_Gap_Fill_EMA(trial):
    return {
        'min_gap_pct': trial.suggest_float('min_gap_pct', 0.002, 0.01),
        'ema_p':       trial.suggest_int('ema_p',          20,    60),
    }


# ── 20. PA_Exhaust_Rev ────────────────────────────────────────────────────────

def gen_PA_Exhaust_Rev(df, body_mult=2.5, atr_p=14, vol_p=20, vol_mult=3.0, **kw):
    close    = df['close']
    open_    = df['open']
    vol      = df['volume']
    atr_p    = int(atr_p); vol_p = int(vol_p)
    atr      = _atr(df, atr_p)
    vol_avg  = _sma(vol, vol_p)
    body     = (close - open_).abs()
    large    = body > atr * body_mult
    high_vol = vol > vol_avg * vol_mult
    big_bear = large & high_vol & (close < open_)
    big_bull = large & high_vol & (close > open_)
    # reversal: next bar opposite
    exhaust_long  = big_bear.shift(1).fillna(False) & (close > open_)
    exhaust_short = big_bull.shift(1).fillna(False) & (close < open_)
    sig = pd.Series(0, index=df.index)
    sig[exhaust_long]  =  1
    sig[exhaust_short] = -1
    return sig.fillna(0)


def space_PA_Exhaust_Rev(trial):
    return {
        'body_mult': trial.suggest_float('body_mult', 2.0, 4.0),
        'atr_p':     trial.suggest_int('atr_p',       10,  20),
        'vol_p':     trial.suggest_int('vol_p',        20,  50),
        'vol_mult':  trial.suggest_float('vol_mult',  2.0, 5.0),
    }


# ── 21. PA_Range_Fade ─────────────────────────────────────────────────────────

def gen_PA_Range_Fade(df, range_p=20, rsi_p=14, **kw):
    close    = df['close']
    high     = df['high']
    low      = df['low']
    range_p  = int(range_p); rsi_p = int(rsi_p)
    range_high = high.rolling(range_p, min_periods=1).max()
    range_low  = low.rolling(range_p, min_periods=1).min()
    rsi        = _rsi(close, rsi_p)
    at_low     = close <= range_low + (range_high - range_low) * 0.15
    at_high    = close >= range_high - (range_high - range_low) * 0.15
    sig = pd.Series(0, index=df.index)
    sig[at_low  & (rsi < 40)] =  1
    sig[at_high & (rsi > 60)] = -1
    return sig.fillna(0)


def space_PA_Range_Fade(trial):
    return {
        'range_p': trial.suggest_int('range_p', 10, 30),
        'rsi_p':   trial.suggest_int('rsi_p',    7, 21),
    }


# ── 22. PA_Pivot_Trend ────────────────────────────────────────────────────────

def gen_PA_Pivot_Trend(df, pivot_left=5, pivot_right=5, **kw):
    high       = df['high']
    low        = df['low']
    close      = df['close']
    pivot_left  = int(pivot_left); pivot_right = int(pivot_right)
    w = pivot_left + pivot_right + 1
    pivot_high = high.rolling(w, min_periods=w, center=True).max()
    pivot_low  = low.rolling(w, min_periods=w, center=True).min()
    is_ph = (high == pivot_high)
    is_pl = (low  == pivot_low)
    # last pivot high / low
    last_ph = high[is_ph].reindex(high.index).ffill()
    last_pl = low[is_pl].reindex(low.index).ffill()
    # HH: current close breaks last pivot high
    sig = pd.Series(0, index=df.index)
    sig[close > last_ph] =  1
    sig[close < last_pl] = -1
    return sig.fillna(0)


def space_PA_Pivot_Trend(trial):
    return {
        'pivot_left':  trial.suggest_int('pivot_left',  3, 10),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
    }


# ── 23. PA_Double_Bottom ──────────────────────────────────────────────────────

def gen_PA_Double_Bottom(df, lookback=20, tol=0.02, **kw):
    close    = df['close']
    low      = df['low']
    high     = df['high']
    lookback = int(lookback)
    sig      = pd.Series(0, index=df.index)
    for i in range(2 * lookback, len(df)):
        window_l = low.iloc[i - lookback:i]
        window_h = high.iloc[i - lookback:i]
        lo1_idx  = window_l.idxmin()
        if lo1_idx not in window_l.index:
            continue
        # find second low excluding first low region
        mid    = window_l.index.get_loc(lo1_idx)
        if mid >= len(window_l) - 2:
            continue
        second = window_l.iloc[mid + 1:]
        if second.empty:
            continue
        lo2_idx = second.idxmin()
        lo1_val = low[lo1_idx]
        lo2_val = low[lo2_idx]
        avg_low = (lo1_val + lo2_val) / 2
        if abs(lo1_val - lo2_val) / avg_low > tol:
            continue
        # neckline = max between the two lows
        between = high.loc[lo1_idx:lo2_idx]
        if between.empty:
            continue
        neckline = between.max()
        if close.iloc[i] > neckline:
            sig.iloc[i] = 1
    return sig.fillna(0)


def space_PA_Double_Bottom(trial):
    return {
        'lookback': trial.suggest_int('lookback',   10,  40),
        'tol':      trial.suggest_float('tol',      0.01, 0.05),
    }


# ── 24. PA_Double_Top ─────────────────────────────────────────────────────────

def gen_PA_Double_Top(df, lookback=20, tol=0.02, **kw):
    close    = df['close']
    high     = df['high']
    low      = df['low']
    lookback = int(lookback)
    sig      = pd.Series(0, index=df.index)
    for i in range(2 * lookback, len(df)):
        window_h = high.iloc[i - lookback:i]
        hi1_idx  = window_h.idxmax()
        mid      = window_h.index.get_loc(hi1_idx)
        if mid >= len(window_h) - 2:
            continue
        second   = window_h.iloc[mid + 1:]
        if second.empty:
            continue
        hi2_idx  = second.idxmax()
        hi1_val  = high[hi1_idx]
        hi2_val  = high[hi2_idx]
        avg_high = (hi1_val + hi2_val) / 2
        if abs(hi1_val - hi2_val) / avg_high > tol:
            continue
        between  = low.loc[hi1_idx:hi2_idx]
        if between.empty:
            continue
        neckline = between.min()
        if close.iloc[i] < neckline:
            sig.iloc[i] = -1
    return sig.fillna(0)


def space_PA_Double_Top(trial):
    return {
        'lookback': trial.suggest_int('lookback',   10,  40),
        'tol':      trial.suggest_float('tol',      0.01, 0.05),
    }


# ── 25. PA_Three_Drives ───────────────────────────────────────────────────────

def gen_PA_Three_Drives(df, rsi_p=14, lookback=20, **kw):
    close    = df['close']
    low      = df['low']
    rsi_p    = int(rsi_p); lookback = int(lookback)
    rsi      = _rsi(close, rsi_p)
    sig      = pd.Series(0, index=df.index)
    for i in range(lookback * 3, len(df)):
        window = low.iloc[i - lookback * 3:i]
        if len(window) < 6:
            continue
        chunk = len(window) // 3
        lo1   = window.iloc[:chunk].min()
        lo2   = window.iloc[chunk:chunk*2].min()
        lo3   = window.iloc[chunk*2:].min()
        # three drives lower
        if lo2 < lo1 and lo3 < lo2:
            # RSI positive divergence: RSI at lo3 > RSI at lo2
            rsi_at_lo3 = rsi.iloc[i - chunk//2: i].mean()
            rsi_at_lo2 = rsi.iloc[i - chunk - chunk//2: i - chunk].mean()
            if rsi_at_lo3 > rsi_at_lo2:
                sig.iloc[i] = 1
    return sig.fillna(0)


def space_PA_Three_Drives(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p',    7,  21),
        'lookback': trial.suggest_int('lookback', 10,  30),
    }


# ── 26. PA_Flat_Base ──────────────────────────────────────────────────────────

def gen_PA_Flat_Base(df, base_p=10, std_thresh=0.02, vol_mult=2.0, **kw):
    close    = df['close']
    vol      = df['volume']
    base_p   = int(base_p)
    rol_std  = close.rolling(base_p, min_periods=base_p).std()
    rol_mean = close.rolling(base_p, min_periods=base_p).mean().replace(0, 1e-9)
    flat     = (rol_std / rol_mean) < std_thresh
    base_high = close.rolling(base_p, min_periods=base_p).max()
    base_low  = close.rolling(base_p, min_periods=base_p).min()
    vol_avg   = _sma(vol, base_p)
    flat_prev = flat.shift(1).fillna(False)
    breakout  = close > base_high.shift(1)
    breakdown = close < base_low.shift(1)
    vol_ok    = vol > vol_avg * vol_mult
    sig = pd.Series(0, index=df.index)
    sig[flat_prev & breakout  & vol_ok] =  1
    sig[flat_prev & breakdown & vol_ok] = -1
    return sig.fillna(0)


def space_PA_Flat_Base(trial):
    return {
        'base_p':     trial.suggest_int('base_p',         5,  20),
        'std_thresh': trial.suggest_float('std_thresh',  0.01, 0.05),
        'vol_mult':   trial.suggest_float('vol_mult',    1.5,  3.0),
    }


# ── 27. PA_Cup_Handle ─────────────────────────────────────────────────────────

def gen_PA_Cup_Handle(df, cup_p=40, handle_p=10, **kw):
    close    = df['close']
    high     = df['high']
    cup_p    = int(cup_p); handle_p = int(handle_p)
    sig      = pd.Series(0, index=df.index)
    total    = cup_p + handle_p
    for i in range(total, len(df)):
        cup_window    = close.iloc[i - total: i - handle_p]
        handle_window = close.iloc[i - handle_p: i]
        if len(cup_window) < cup_p // 2:
            continue
        cup_start = cup_window.iloc[0]
        cup_end   = cup_window.iloc[-1]
        cup_mid   = cup_window.min()
        # U-shape: start and end higher than mid
        if cup_start < cup_mid or cup_end < cup_mid:
            continue
        handle_high = handle_window.max()
        if close.iloc[i] > handle_high:
            sig.iloc[i] = 1
    return sig.fillna(0)


def space_PA_Cup_Handle(trial):
    return {
        'cup_p':    trial.suggest_int('cup_p',    20, 60),
        'handle_p': trial.suggest_int('handle_p',  5, 15),
    }


# ── 28. PA_Bear_Trap ──────────────────────────────────────────────────────────

def gen_PA_Bear_Trap(df, support_p=20, tol=0.002, **kw):
    close     = df['close']
    support_p = int(support_p)
    support   = close.rolling(support_p, min_periods=1).min().shift(1)
    broke     = close < support * (1 - tol)
    recover   = broke.shift(1).fillna(False) & (close > support.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[recover] = 1
    return sig.fillna(0)


def space_PA_Bear_Trap(trial):
    return {
        'support_p': trial.suggest_int('support_p',    10,  30),
        'tol':       trial.suggest_float('tol',       0.001, 0.005),
    }


# ── 29. PA_Bull_Trap ──────────────────────────────────────────────────────────

def gen_PA_Bull_Trap(df, res_p=20, tol=0.002, **kw):
    close  = df['close']
    res_p  = int(res_p)
    res    = close.rolling(res_p, min_periods=1).max().shift(1)
    broke  = close > res * (1 + tol)
    fall   = broke.shift(1).fillna(False) & (close < res.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[fall] = -1
    return sig.fillna(0)


def space_PA_Bull_Trap(trial):
    return {
        'res_p': trial.suggest_int('res_p',       10,  30),
        'tol':   trial.suggest_float('tol',       0.001, 0.005),
    }


# ── 30. PA_Micro_Pullback ─────────────────────────────────────────────────────

def gen_PA_Micro_Pullback(df, fast_p=9, slow_p=30, **kw):
    close   = df['close']
    open_   = df['open']
    high    = df['high']
    fast_p  = int(fast_p); slow_p = int(slow_p)
    ema_f   = _ema(close, fast_p)
    ema_s   = _ema(close, slow_p)
    uptrend = ema_f > ema_s
    dntrend = ema_f < ema_s
    red_bar = close < open_
    grn_bar = close > open_
    # micro pullback: one red bar in uptrend then break of prior bar high
    prev_red_in_uptrend = uptrend.shift(1).fillna(False) & red_bar.shift(1).fillna(False)
    prev_grn_in_dntrend = dntrend.shift(1).fillna(False) & grn_bar.shift(1).fillna(False)
    sig = pd.Series(0, index=df.index)
    sig[prev_red_in_uptrend & (close > high.shift(1))] =  1
    sig[prev_grn_in_dntrend & (close < df['low'].shift(1))] = -1
    return sig.fillna(0)


def space_PA_Micro_Pullback(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',  5, 20),
        'slow_p': trial.suggest_int('slow_p', 20, 60),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'PA_HH_HL_Trend': {
        'gen': gen_PA_HH_HL_Trend,
        'space': space_PA_HH_HL_Trend,
        'default_params': {'lookback': 2},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Higher highs + higher lows uptrend. Bidirectional.'},
    },
    'PA_Breakout_Vol': {
        'gen': gen_PA_Breakout_Vol,
        'space': space_PA_Breakout_Vol,
        'default_params': {'break_p': 20, 'vol_p': 20, 'vol_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Close above N-bar high with volume confirmation. Bidirectional.'},
    },
    'PA_Breakdown_Vol': {
        'gen': gen_PA_Breakdown_Vol,
        'space': space_PA_Breakdown_Vol,
        'default_params': {'break_p': 20, 'vol_p': 20, 'vol_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Close below N-bar low with volume confirmation. Bidirectional.'},
    },
    'PA_Key_Level_Bounce': {
        'gen': gen_PA_Key_Level_Bounce,
        'space': space_PA_Key_Level_Bounce,
        'default_params': {'level_size': 100, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Bounce from round number key levels using ATR proximity.'},
    },
    'PA_Tight_SL': {
        'gen': gen_PA_Tight_SL,
        'space': space_PA_Tight_SL,
        'default_params': {'look_p': 5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Inside bar + breakout = tight stop setup. Bidirectional.'},
    },
    'PA_Momentum_Candle': {
        'gen': gen_PA_Momentum_Candle,
        'space': space_PA_Momentum_Candle,
        'default_params': {'body_mult': 1.5, 'atr_p': 14, 'vol_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Large body candle > ATR*mult + volume = momentum. Bidirectional.'},
    },
    'PA_Rejection_Wick': {
        'gen': gen_PA_Rejection_Wick,
        'space': space_PA_Rejection_Wick,
        'default_params': {'wick_ratio': 2.0, 'support_p': 20, 'ema_p': 21},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Long wick rejection at support/resistance = reversal. Bidirectional.'},
    },
    'PA_Consolidation_Break': {
        'gen': gen_PA_Consolidation_Break,
        'space': space_PA_Consolidation_Break,
        'default_params': {'atr_p': 14, 'consol_bars': 5, 'expand_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Low-ATR consolidation then ATR expansion breakout. Bidirectional.'},
    },
    'PA_Two_Day_Rule': {
        'gen': gen_PA_Two_Day_Rule,
        'space': space_PA_Two_Day_Rule,
        'default_params': {'n_bars': 2},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Price closes above N-bar high/low = directional signal.'},
    },
    'PA_Trend_Continuation': {
        'gen': gen_PA_Trend_Continuation,
        'space': space_PA_Trend_Continuation,
        'default_params': {'trend_bars': 3, 'pullback_bars': 2, 'max_retrace': 0.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Trend + short pullback + break of last high = continuation. Bidirectional.'},
    },
    'PA_Opening_Range': {
        'gen': gen_PA_Opening_Range,
        'space': space_PA_Opening_Range,
        'default_params': {'range_bars': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Opening range breakout/breakdown. Bidirectional.'},
    },
    'PA_Close_Above_High': {
        'gen': gen_PA_Close_Above_High,
        'space': space_PA_Close_Above_High,
        'default_params': {'ema_p': 50},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Close above/below prior bar high/low with EMA trend filter. Bidirectional.'},
    },
    'PA_Body_Full': {
        'gen': gen_PA_Body_Full,
        'space': space_PA_Body_Full,
        'default_params': {'body_pct': 0.8, 'ema_p': 21},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Full body candle (close~high, open~low) = trend strength. Bidirectional.'},
    },
    'PA_ATR_Momentum': {
        'gen': gen_PA_ATR_Momentum,
        'space': space_PA_ATR_Momentum,
        'default_params': {'atr_p': 14, 'mult': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Candle body > ATR*mult = strong momentum entry. Bidirectional.'},
    },
    'PA_Support_Test': {
        'gen': gen_PA_Support_Test,
        'space': space_PA_Support_Test,
        'default_params': {'support_p': 20, 'tol': 0.002},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Price tests rolling support/resistance and holds. Bidirectional.'},
    },
    'PA_Resistance_Break': {
        'gen': gen_PA_Resistance_Break,
        'space': space_PA_Resistance_Break,
        'default_params': {'res_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Clean break through rolling resistance/support. Bidirectional.'},
    },
    'PA_Channel_Trade': {
        'gen': gen_PA_Channel_Trade,
        'space': space_PA_Channel_Trade,
        'default_params': {'channel_p': 30, 'k': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Price channel: SMA±k*std. Trade channel extremes. Bidirectional.'},
    },
    'PA_Trend_Touch_EMA': {
        'gen': gen_PA_Trend_Touch_EMA,
        'space': space_PA_Trend_Touch_EMA,
        'default_params': {'fast_p': 10, 'slow_p': 50, 'tol': 0.002},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Touch EMA in direction of trend. Bidirectional.'},
    },
    'PA_Gap_Fill_EMA': {
        'gen': gen_PA_Gap_Fill_EMA,
        'space': space_PA_Gap_Fill_EMA,
        'default_params': {'min_gap_pct': 0.005, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Gap fill toward EMA after overnight gap. Bidirectional.'},
    },
    'PA_Exhaust_Rev': {
        'gen': gen_PA_Exhaust_Rev,
        'space': space_PA_Exhaust_Rev,
        'default_params': {'body_mult': 2.5, 'atr_p': 14, 'vol_p': 20, 'vol_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Exhaustion reversal: large bar + extreme volume + next-bar flip. Bidirectional.'},
    },
    'PA_Range_Fade': {
        'gen': gen_PA_Range_Fade,
        'space': space_PA_Range_Fade,
        'default_params': {'range_p': 20, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Fade range extremes with RSI confirmation. Bidirectional.'},
    },
    'PA_Pivot_Trend': {
        'gen': gen_PA_Pivot_Trend,
        'space': space_PA_Pivot_Trend,
        'default_params': {'pivot_left': 5, 'pivot_right': 5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Pivot-based trend: HH/HL=up, LL/LH=down. Bidirectional.'},
    },
    'PA_Double_Bottom': {
        'gen': gen_PA_Double_Bottom,
        'space': space_PA_Double_Bottom,
        'default_params': {'lookback': 20, 'tol': 0.02},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Double bottom pattern with neckline breakout. Long only.'},
    },
    'PA_Double_Top': {
        'gen': gen_PA_Double_Top,
        'space': space_PA_Double_Top,
        'default_params': {'lookback': 20, 'tol': 0.02},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Double top pattern with neckline breakdown. Short only.'},
    },
    'PA_Three_Drives': {
        'gen': gen_PA_Three_Drives,
        'space': space_PA_Three_Drives,
        'default_params': {'rsi_p': 14, 'lookback': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Three drives lower + RSI positive divergence reversal. Long only.'},
    },
    'PA_Flat_Base': {
        'gen': gen_PA_Flat_Base,
        'space': space_PA_Flat_Base,
        'default_params': {'base_p': 10, 'std_thresh': 0.02, 'vol_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Flat base consolidation + volume breakout. Bidirectional.'},
    },
    'PA_Cup_Handle': {
        'gen': gen_PA_Cup_Handle,
        'space': space_PA_Cup_Handle,
        'default_params': {'cup_p': 40, 'handle_p': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4500,
                 'description': 'Cup and handle: U-shape + handle + breakout. Long only.'},
    },
    'PA_Bear_Trap': {
        'gen': gen_PA_Bear_Trap,
        'space': space_PA_Bear_Trap,
        'default_params': {'support_p': 20, 'tol': 0.002},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Bear trap: false break below support then recovery. Long only.'},
    },
    'PA_Bull_Trap': {
        'gen': gen_PA_Bull_Trap,
        'space': space_PA_Bull_Trap,
        'default_params': {'res_p': 20, 'tol': 0.002},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Bull trap: false break above resistance then reversal. Short only.'},
    },
    'PA_Micro_Pullback': {
        'gen': gen_PA_Micro_Pullback,
        'space': space_PA_Micro_Pullback,
        'default_params': {'fast_p': 9, 'slow_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': '1-bar micro pullback in strong trend then continuation. Bidirectional.'},
    },
}
