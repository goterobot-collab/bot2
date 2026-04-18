#!/usr/bin/env python3
"""TV2 BATCH 23 — 30 estrategias Bollinger + ATR Trailing + Volatility 2026-04-01

  BB_Classic          — BB mean reversion: close < lower = long, > upper = short (v4, ~8000L)
  BB_Breakout         — BB momentum: close breaks above upper = long (v4, ~5000L)
  BB_Width_Squeeze    — BB squeeze width expansion breakout (v5, ~3000L)
  BB_W_Pattern        — W-bottom double touch of lower band (v4, ~2500L)
  BB_Percent_B        — %B oscillator crossover entries (v5, ~2500L)
  BB_RSI_OB_OS        — BB + RSI double confirmation (v4, ~3000L)
  BB_MA_Cross         — Price crosses BB midline with RSI filter (v5, ~2000L)
  BB_EMA_Confluence   — BB lower + EMA confluence support (v4, ~2000L)
  BB_Keltner_Width    — BB/KC width ratio as volatility measure (v5, ~1500L)
  BB_Three_Sigma      — 3-sigma extreme reversion (v4, ~2000L)
  ATR_Trail_Long      — ATR trailing stop trend indicator (v4, ~5000L)
  ATR_Trail_Bidir     — Bidirectional ATR trailing (v5, ~3000L)
  ATR_Band_Breakout   — ATR bands around EMA momentum (v4, ~2500L)
  ATR_Mean_Reversion  — EMA +/- 2*ATR extreme reversion (v5, ~2000L)
  ATR_Percentile      — ATR percentile rank volatility filter (v5, ~2000L)
  CE_ATR_Dynamic      — Chandelier Exit with dynamic multiplier (v5, ~2500L)
  CE_MACD             — Chandelier + MACD direction (v4, ~2000L)
  CE_RSI_Vol          — Chandelier + RSI + Volume (v5, ~1500L)
  NR4_NR7             — Narrow Range 4/7 breakout (v4, ~3000L)
  VO_Breakout         — Volatility compression → expansion breakout (v5, ~2500L)
  Heikin_Ashi_ATR     — Heikin Ashi candles + ATR expansion filter (v4, ~3000L)
  PSAR_Classic        — Parabolic SAR classic (v4, ~5000L)
  PSAR_EMA            — PSAR in EMA trend direction only (v5, ~2500L)
  PSAR_RSI            — PSAR flip + RSI confirmation (v4, ~2000L)
  PSAR_MACD           — PSAR + MACD signal alignment (v5, ~1800L)
  VIX_Regime          — Crypto VIX regime switcher (v5, ~2000L)
  BB_Width_Regime     — BB width regime: narrow vs trending (v4, ~1800L)
  ATR_Ratio_Filter    — ATR/close ratio normalized volatility (v5, ~1500L)
  BB_ATR_Confluence   — BB lower + EMA-ATR double oversold (v4, ~2500L)
  BB_ATR_Breakout     — BB upper + EMA+ATR double breakout (v5, ~2000L)
"""

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


def _bb(s, p, mult):
    mid = _sma(s, p)
    std = s.rolling(int(p), min_periods=1).std().fillna(0)
    return mid + mult * std, mid, mid - mult * std


def _macd(s, fast, slow, sig):
    fast, slow, sig = int(fast), int(slow), int(sig)
    line   = _ema(s, fast) - _ema(s, slow)
    signal = _ema(line, sig)
    return line, signal


# ── 1. BB_Classic ─────────────────────────────────────────────────────────────

def gen_BB_Classic(df, bb_p=20, bb_mult=2.0, **kw):
    close = df['close']
    upper, mid, lower = _bb(close, bb_p, bb_mult)
    sig = pd.Series(0, index=df.index)
    sig[close < lower] =  1
    sig[close > upper] = -1
    return sig


def space_BB_Classic(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 15, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0),
    }


# ── 2. BB_Breakout ────────────────────────────────────────────────────────────

def gen_BB_Breakout(df, bb_p=20, bb_mult=2.0, **kw):
    close = df['close']
    upper, mid, lower = _bb(close, bb_p, bb_mult)
    sig = pd.Series(0, index=df.index)
    sig[close > upper] =  1
    sig[close < lower] = -1
    return sig


def space_BB_Breakout(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 15, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0),
    }


# ── 3. BB_Width_Squeeze ───────────────────────────────────────────────────────

def gen_BB_Width_Squeeze(df, bb_p=20, bb_mult=2.0, lookback=10, **kw):
    close = df['close']
    upper, mid, lower = _bb(close, bb_p, bb_mult)
    width = (upper - lower) / mid.replace(0, 1e-9)
    width_min = width.rolling(int(lookback), min_periods=1).min()
    squeeze    = (width <= width_min * 1.01).values.astype(bool)
    squeeze_s  = pd.Series(squeeze, index=df.index)
    expanding  = (width.values > np.concatenate([[np.nan], width.values[:-1]])).astype(bool)
    expanding[0] = False
    expanding_s = pd.Series(expanding, index=df.index)
    was_squeeze = pd.Series(np.concatenate([[False], squeeze[:-1]]), index=df.index)
    sig = pd.Series(0, index=df.index)
    # Long: expanding after squeeze AND price above mid
    cond_long  = was_squeeze & expanding_s & (close > mid)
    # Short: expanding after squeeze AND price below mid
    cond_short = was_squeeze & expanding_s & (close < mid)
    sig[cond_long]  =  1
    sig[cond_short] = -1
    return sig


def space_BB_Width_Squeeze(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 15, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 2.5),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 4. BB_W_Pattern ───────────────────────────────────────────────────────────

def gen_BB_W_Pattern(df, bb_p=20, bb_mult=2.0, lookback=10, **kw):
    close = df['close']
    upper, mid, lower = _bb(close, bb_p, bb_mult)
    lb = int(lookback)
    at_lower  = close <= lower
    # First touch: was at lower at any point in last [lookback] bars
    first_touch = at_lower.rolling(lb, min_periods=1).max().astype(bool)
    # Bounce: price went back above mid after that first touch
    bounced = (close > mid).rolling(lb, min_periods=1).max().astype(bool)
    # Second touch: currently at lower
    second_touch = at_lower
    # W pattern: first touch in window, bounce happened, now second touch
    cond_long = first_touch & bounced & second_touch
    # Short: inverse M-top pattern (two upper band touches with mid dip between)
    at_upper   = close >= upper
    first_top  = at_upper.rolling(lb, min_periods=1).max().astype(bool)
    dipped     = (close < mid).rolling(lb, min_periods=1).max().astype(bool)
    second_top = at_upper
    cond_short = first_top & dipped & second_top
    sig = pd.Series(0, index=df.index)
    sig[cond_long]  =  1
    sig[cond_short] = -1
    return sig


def space_BB_W_Pattern(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 15, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 2.5),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 5. BB_Percent_B ───────────────────────────────────────────────────────────

def gen_BB_Percent_B(df, bb_p=20, bb_mult=2.0, **kw):
    close = df['close']
    upper, mid, lower = _bb(close, bb_p, bb_mult)
    pct_b = (close - lower) / (upper - lower + 1e-9)
    sig = pd.Series(0, index=df.index)
    # Long: %B crosses above 0 from below (was negative / near zero, now above)
    was_below = (pct_b.shift(1) < 0.05).fillna(True)
    now_above = pct_b > 0.05
    # Short: %B crosses below 1 from above
    was_above_1 = (pct_b.shift(1) > 0.95).fillna(False)
    now_below_1 = pct_b < 0.95
    sig[was_below & now_above]   =  1
    sig[was_above_1 & now_below_1] = -1
    return sig


def space_BB_Percent_B(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 15, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0),
    }


# ── 6. BB_RSI_OB_OS ───────────────────────────────────────────────────────────

def gen_BB_RSI_OB_OS(df, bb_p=20, bb_mult=2.0, rsi_p=14, **kw):
    close = df['close']
    upper, mid, lower = _bb(close, bb_p, bb_mult)
    rsi = _rsi(close, rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[(close < lower) & (rsi < 30)] =  1
    sig[(close > upper) & (rsi > 70)] = -1
    return sig


def space_BB_RSI_OB_OS(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 15, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 2.5),
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
    }


# ── 7. BB_MA_Cross ────────────────────────────────────────────────────────────

def gen_BB_MA_Cross(df, bb_p=20, rsi_p=14, **kw):
    close = df['close']
    upper, mid, lower = _bb(close, bb_p, 2.0)
    rsi = _rsi(close, rsi_p)
    above_mid = close > mid
    was_below = (close.shift(1) <= mid.shift(1)).fillna(True)
    cross_up = above_mid & was_below
    below_mid = close < mid
    was_above = (close.shift(1) >= mid.shift(1)).fillna(False)
    cross_dn  = below_mid & was_above
    sig = pd.Series(0, index=df.index)
    sig[cross_up  & (rsi > 50)] =  1
    sig[cross_dn  & (rsi < 50)] = -1
    return sig


def space_BB_MA_Cross(trial):
    return {
        'bb_p':  trial.suggest_int('bb_p', 15, 30),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 8. BB_EMA_Confluence ──────────────────────────────────────────────────────

def gen_BB_EMA_Confluence(df, bb_p=20, bb_mult=2.0, ema_p=30, **kw):
    close = df['close']
    upper, mid, lower = _bb(close, bb_p, bb_mult)
    ema = _ema(close, ema_p)
    near_ema   = (close - ema).abs() / ema.replace(0, 1e-9) < 0.005
    at_lower   = close <= lower * 1.002
    at_upper   = close >= upper * 0.998
    ema_resist = (close - ema).abs() / ema.replace(0, 1e-9) < 0.005
    sig = pd.Series(0, index=df.index)
    sig[at_lower & near_ema]  =  1
    sig[at_upper & ema_resist] = -1
    return sig


def space_BB_EMA_Confluence(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 15, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 2.5),
        'ema_p':   trial.suggest_int('ema_p', 20, 60),
    }


# ── 9. BB_Keltner_Width ───────────────────────────────────────────────────────

def gen_BB_Keltner_Width(df, bb_p=20, bb_mult=2.0, kc_p=20, kc_mult=1.5, **kw):
    close = df['close']
    upper_bb, mid_bb, lower_bb = _bb(close, bb_p, bb_mult)
    bb_width = upper_bb - lower_bb
    ema_kc   = _ema(close, kc_p)
    atr_kc   = _atr(df, kc_p)
    kc_width = 2 * kc_mult * atr_kc
    ratio    = bb_width / kc_width.replace(0, 1e-9)
    expanding = ratio > ratio.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[expanding & (close > mid_bb)] =  1
    sig[expanding & (close < mid_bb)] = -1
    return sig


def space_BB_Keltner_Width(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 15, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 2.5),
        'kc_p':    trial.suggest_int('kc_p', 15, 30),
        'kc_mult': trial.suggest_float('kc_mult', 1.0, 2.5),
    }


# ── 10. BB_Three_Sigma ────────────────────────────────────────────────────────

def gen_BB_Three_Sigma(df, bb_p=20, inner_mult=2.0, outer_mult=3.0, **kw):
    close = df['close']
    mid   = _sma(close, bb_p)
    std   = close.rolling(int(bb_p), min_periods=1).std().fillna(0)
    upper3 = mid + outer_mult * std
    lower3 = mid - outer_mult * std
    sig = pd.Series(0, index=df.index)
    sig[close < lower3] =  1
    sig[close > upper3] = -1
    return sig


def space_BB_Three_Sigma(trial):
    return {
        'bb_p':       trial.suggest_int('bb_p', 15, 30),
        'inner_mult': trial.suggest_float('inner_mult', 2.0, 2.5),
        'outer_mult': trial.suggest_float('outer_mult', 2.5, 4.0),
    }


# ── 11. ATR_Trail_Long ────────────────────────────────────────────────────────

def gen_ATR_Trail_Long(df, atr_p=14, atr_mult=3.0, **kw):
    close  = df['close'].values
    atr_v  = _atr(df, atr_p).values
    trail  = np.full(len(close), np.nan)
    trail[0] = close[0] - atr_mult * atr_v[0]
    for i in range(1, len(close)):
        new_trail = close[i] - atr_mult * atr_v[i]
        trail[i]  = max(trail[i - 1], new_trail) if not np.isnan(trail[i - 1]) else new_trail
    trail_s = pd.Series(trail, index=df.index)
    sig = pd.Series(0, index=df.index)
    sig[df['close'] > trail_s] =  1
    sig[df['close'] < trail_s] = -1
    return sig


def space_ATR_Trail_Long(trial):
    return {
        'atr_p':    trial.suggest_int('atr_p', 10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 5.0),
    }


# ── 12. ATR_Trail_Bidir ───────────────────────────────────────────────────────

def gen_ATR_Trail_Bidir(df, atr_p=14, atr_mult=3.0, **kw):
    close  = df['close'].values
    atr_v  = _atr(df, atr_p).values
    n      = len(close)
    trail  = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)  # 1=long, -1=short
    trail[0]    = close[0] - atr_mult * atr_v[0]
    direction[0] = 1
    for i in range(1, n):
        prev_trail = trail[i - 1] if not np.isnan(trail[i - 1]) else close[i]
        prev_dir   = direction[i - 1]
        if prev_dir == 1:
            new_t = close[i] - atr_mult * atr_v[i]
            t     = max(prev_trail, new_t)
            if close[i] < t:
                direction[i] = -1
                trail[i]     = close[i] + atr_mult * atr_v[i]
            else:
                direction[i] = 1
                trail[i]     = t
        else:
            new_t = close[i] + atr_mult * atr_v[i]
            t     = min(prev_trail, new_t)
            if close[i] > t:
                direction[i] = 1
                trail[i]     = close[i] - atr_mult * atr_v[i]
            else:
                direction[i] = -1
                trail[i]     = t
    return pd.Series(direction, index=df.index)


def space_ATR_Trail_Bidir(trial):
    return {
        'atr_p':    trial.suggest_int('atr_p', 10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 5.0),
    }


# ── 13. ATR_Band_Breakout ─────────────────────────────────────────────────────

def gen_ATR_Band_Breakout(df, ema_p=30, atr_p=14, atr_mult=2.0, **kw):
    close    = df['close']
    ema      = _ema(close, ema_p)
    atr      = _atr(df, atr_p)
    upper    = ema + atr_mult * atr
    lower    = ema - atr_mult * atr
    sig = pd.Series(0, index=df.index)
    sig[close > upper] =  1
    sig[close < lower] = -1
    return sig


def space_ATR_Band_Breakout(trial):
    return {
        'ema_p':    trial.suggest_int('ema_p', 20, 60),
        'atr_p':    trial.suggest_int('atr_p', 10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 4.0),
    }


# ── 14. ATR_Mean_Reversion ────────────────────────────────────────────────────

def gen_ATR_Mean_Reversion(df, ema_p=30, atr_p=14, atr_mult=2.0, **kw):
    close = df['close']
    ema   = _ema(close, ema_p)
    atr   = _atr(df, atr_p)
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    sig = pd.Series(0, index=df.index)
    sig[close < lower] =  1    # oversold reversion
    sig[close > upper] = -1    # overbought reversion
    return sig


def space_ATR_Mean_Reversion(trial):
    return {
        'ema_p':    trial.suggest_int('ema_p', 20, 60),
        'atr_p':    trial.suggest_int('atr_p', 10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 3.0),
    }


# ── 15. ATR_Percentile ────────────────────────────────────────────────────────

def gen_ATR_Percentile(df, atr_p=14, rank_p=100, rank_thresh=60, **kw):
    close      = df['close']
    atr        = _atr(df, atr_p)
    rank_p_int = int(rank_p)
    # Percentile rank of current ATR within rolling window
    atr_rank = atr.rolling(rank_p_int, min_periods=1).rank(pct=True) * 100.0
    low_vol   = atr_rank < rank_thresh
    rsi       = _rsi(close, 14)
    sig = pd.Series(0, index=df.index)
    sig[low_vol & (rsi < 40)] =  1
    sig[low_vol & (rsi > 60)] = -1
    return sig


def space_ATR_Percentile(trial):
    return {
        'atr_p':        trial.suggest_int('atr_p', 10, 20),
        'rank_p':       trial.suggest_int('rank_p', 50, 200),
        'rank_thresh':  trial.suggest_int('rank_thresh', 40, 70),
    }


# ── 16. CE_ATR_Dynamic ────────────────────────────────────────────────────────

def gen_CE_ATR_Dynamic(df, atr_p=22, mult_low=2.0, mult_high=4.0, atr_thresh_pct=65, **kw):
    close      = df['close']
    atr        = _atr(df, atr_p)
    rank_p     = max(int(atr_p) * 3, 50)
    atr_rank   = atr.rolling(rank_p, min_periods=1).rank(pct=True) * 100.0
    high_vol   = atr_rank >= atr_thresh_pct
    mult       = pd.Series(
        np.where(high_vol, mult_high, mult_low), index=df.index
    )
    high_s = df['high'].values
    low_s  = df['low'].values
    close_v = close.values
    atr_v  = atr.values
    mult_v = mult.values
    n = len(close_v)
    # Long CE: highest_high - mult*ATR (running max)
    ce_long  = np.full(n, np.nan)
    ce_short = np.full(n, np.nan)
    ce_long[0]  = high_s[0]  - mult_v[0] * atr_v[0]
    ce_short[0] = low_s[0]   + mult_v[0] * atr_v[0]
    for i in range(1, n):
        new_long  = high_s[i]  - mult_v[i] * atr_v[i]
        new_short = low_s[i]   + mult_v[i] * atr_v[i]
        ce_long[i]  = max(ce_long[i - 1],  new_long)  if not np.isnan(ce_long[i - 1])  else new_long
        ce_short[i] = min(ce_short[i - 1], new_short) if not np.isnan(ce_short[i - 1]) else new_short
    ce_long_s  = pd.Series(ce_long,  index=df.index)
    ce_short_s = pd.Series(ce_short, index=df.index)
    sig = pd.Series(0, index=df.index)
    sig[close > ce_long_s]  =  1
    sig[close < ce_short_s] = -1
    return sig


def space_CE_ATR_Dynamic(trial):
    return {
        'atr_p':          trial.suggest_int('atr_p', 10, 20),
        'mult_low':       trial.suggest_float('mult_low', 1.5, 2.5),
        'mult_high':      trial.suggest_float('mult_high', 3.0, 5.0),
        'atr_thresh_pct': trial.suggest_int('atr_thresh_pct', 50, 80),
    }


# ── 17. CE_MACD ───────────────────────────────────────────────────────────────

def gen_CE_MACD(df, atr_p=22, ce_mult=3.0, fast=12, slow=26, **kw):
    close  = df['close']
    atr    = _atr(df, atr_p)
    high_v = df['high'].values
    low_v  = df['low'].values
    close_v = close.values
    atr_v  = atr.values
    n = len(close_v)
    ce_long  = np.full(n, np.nan)
    ce_short = np.full(n, np.nan)
    ce_long[0]  = high_v[0]  - ce_mult * atr_v[0]
    ce_short[0] = low_v[0]   + ce_mult * atr_v[0]
    for i in range(1, n):
        nl = high_v[i]  - ce_mult * atr_v[i]
        ns = low_v[i]   + ce_mult * atr_v[i]
        ce_long[i]  = max(ce_long[i - 1],  nl) if not np.isnan(ce_long[i - 1])  else nl
        ce_short[i] = min(ce_short[i - 1], ns) if not np.isnan(ce_short[i - 1]) else ns
    ce_long_s  = pd.Series(ce_long,  index=df.index)
    ce_short_s = pd.Series(ce_short, index=df.index)
    macd_line, macd_sig = _macd(close, fast, slow, 9)
    bull_ce = close > ce_long_s
    bear_ce = close < ce_short_s
    sig = pd.Series(0, index=df.index)
    sig[bull_ce & (macd_line > 0)]  =  1
    sig[bear_ce & (macd_line < 0)]  = -1
    return sig


def space_CE_MACD(trial):
    return {
        'atr_p':    trial.suggest_int('atr_p', 10, 20),
        'ce_mult':  trial.suggest_float('ce_mult', 2.0, 5.0),
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
    }


# ── 18. CE_RSI_Vol ────────────────────────────────────────────────────────────

def gen_CE_RSI_Vol(df, atr_p=22, ce_mult=3.0, rsi_p=14, **kw):
    close  = df['close']
    volume = df.get('volume', pd.Series(1, index=df.index))
    atr    = _atr(df, atr_p)
    high_v = df['high'].values
    low_v  = df['low'].values
    atr_v  = atr.values
    n      = len(close)
    ce_long  = np.full(n, np.nan)
    ce_short = np.full(n, np.nan)
    ce_long[0]  = high_v[0]  - ce_mult * atr_v[0]
    ce_short[0] = low_v[0]   + ce_mult * atr_v[0]
    for i in range(1, n):
        nl = high_v[i]  - ce_mult * atr_v[i]
        ns = low_v[i]   + ce_mult * atr_v[i]
        ce_long[i]  = max(ce_long[i - 1],  nl) if not np.isnan(ce_long[i - 1])  else nl
        ce_short[i] = min(ce_short[i - 1], ns) if not np.isnan(ce_short[i - 1]) else ns
    ce_long_s  = pd.Series(ce_long,  index=df.index)
    ce_short_s = pd.Series(ce_short, index=df.index)
    rsi     = _rsi(close, rsi_p)
    vol_sma = _sma(volume, 20)
    high_vol_bar = volume > vol_sma
    sig = pd.Series(0, index=df.index)
    sig[(close > ce_long_s)  & (rsi > 50) & high_vol_bar] =  1
    sig[(close < ce_short_s) & (rsi < 50) & high_vol_bar] = -1
    return sig


def space_CE_RSI_Vol(trial):
    return {
        'atr_p':   trial.suggest_int('atr_p', 10, 20),
        'ce_mult': trial.suggest_float('ce_mult', 2.0, 4.0),
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
    }


# ── 19. NR4_NR7 ───────────────────────────────────────────────────────────────

def gen_NR4_NR7(df, nr_p=7, vol_mult=1.0, **kw):
    nr_p     = int(nr_p)
    rng      = df['high'] - df['low']
    nr_min   = rng.rolling(nr_p, min_periods=nr_p).min()
    is_nr    = (rng <= nr_min * 1.001)
    # Breakout level: high/low of the NR bar
    nr_high  = df['high'].where(is_nr).ffill()
    nr_low   = df['low'].where(is_nr).ffill()
    sig = pd.Series(0, index=df.index)
    sig[df['close'] > nr_high * vol_mult] =  1
    sig[df['close'] < nr_low  / vol_mult] = -1
    return sig


def space_NR4_NR7(trial):
    return {
        'nr_p':      trial.suggest_int('nr_p', 4, 10),
        'vol_mult':  trial.suggest_float('vol_mult', 1.0, 2.5),
    }


# ── 20. VO_Breakout ───────────────────────────────────────────────────────────

def gen_VO_Breakout(df, compress_p=10, vol_thresh_pct=0.5, **kw):
    close     = df['close']
    pct_chg   = close.pct_change().fillna(0)
    vol_std   = pct_chg.rolling(int(compress_p), min_periods=1).std()
    vol_med   = vol_std.rolling(50, min_periods=1).median()
    compressed_v = (vol_std <= vol_med * vol_thresh_pct).values.astype(bool)
    vol_v        = vol_std.values
    expanding_v  = np.concatenate([[False], vol_v[1:] > vol_v[:-1]])
    was_comp     = pd.Series(np.concatenate([[False], compressed_v[:-1]]), index=df.index)
    expanding    = pd.Series(expanding_v, index=df.index)
    sig = pd.Series(0, index=df.index)
    sig[was_comp & expanding & (close > close.shift(1))] =  1
    sig[was_comp & expanding & (close < close.shift(1))] = -1
    return sig


def space_VO_Breakout(trial):
    return {
        'compress_p':      trial.suggest_int('compress_p', 5, 20),
        'vol_thresh_pct':  trial.suggest_float('vol_thresh_pct', 0.3, 0.7),
    }


# ── 21. Heikin_Ashi_ATR ───────────────────────────────────────────────────────

def gen_Heikin_Ashi_ATR(df, consec=2, atr_p=14, atr_mult=1.0, **kw):
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(c)
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty(n)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    ha_h = np.maximum(h, np.maximum(ha_o, ha_c))
    ha_l = np.minimum(l, np.minimum(ha_o, ha_c))
    ha_bull    = ha_c > ha_o                           # green candle
    small_low  = (ha_c - ha_l) / (ha_h - ha_l + 1e-9) > 0.80  # low wick < 20%
    bull_signal = ha_bull & small_low
    bear_signal = ~ha_bull & ((ha_h - ha_c) / (ha_h - ha_l + 1e-9) > 0.80)
    consec_i    = int(consec)
    bull_series = pd.Series(bull_signal.astype(int), index=df.index)
    bear_series = pd.Series(bear_signal.astype(int), index=df.index)
    consec_bull = bull_series.rolling(consec_i, min_periods=consec_i).min().fillna(0).astype(bool)
    consec_bear = bear_series.rolling(consec_i, min_periods=consec_i).min().fillna(0).astype(bool)
    atr     = _atr(df, atr_p)
    atr_exp = atr > atr.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[consec_bull & atr_exp] =  1
    sig[consec_bear & atr_exp] = -1
    return sig


def space_Heikin_Ashi_ATR(trial):
    return {
        'consec':    trial.suggest_int('consec', 1, 4),
        'atr_p':     trial.suggest_int('atr_p', 10, 20),
        'atr_mult':  trial.suggest_float('atr_mult', 0.5, 2.0),
    }


# ── 22. PSAR_Classic ─────────────────────────────────────────────────────────

def _psar(high_v, low_v, close_v, start, inc, max_af):
    n    = len(close_v)
    psar = np.empty(n)
    ep   = np.empty(n)
    af   = np.empty(n)
    bull = np.empty(n, dtype=bool)
    psar[0] = low_v[0]
    ep[0]   = high_v[0]
    af[0]   = start
    bull[0] = True
    for i in range(1, n):
        prev_bull = bull[i - 1]
        prev_psar = psar[i - 1]
        prev_ep   = ep[i - 1]
        prev_af   = af[i - 1]
        if prev_bull:
            new_psar = prev_psar + prev_af * (prev_ep - prev_psar)
            new_psar = min(new_psar, low_v[i - 1], low_v[max(0, i - 2)])
            if low_v[i] < new_psar:
                bull[i] = False
                psar[i] = prev_ep
                ep[i]   = low_v[i]
                af[i]   = start
            else:
                bull[i] = True
                psar[i] = new_psar
                if high_v[i] > prev_ep:
                    ep[i] = high_v[i]
                    af[i] = min(prev_af + inc, max_af)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
        else:
            new_psar = prev_psar + prev_af * (prev_ep - prev_psar)
            new_psar = max(new_psar, high_v[i - 1], high_v[max(0, i - 2)])
            if high_v[i] > new_psar:
                bull[i] = True
                psar[i] = prev_ep
                ep[i]   = high_v[i]
                af[i]   = start
            else:
                bull[i] = False
                psar[i] = new_psar
                if low_v[i] < prev_ep:
                    ep[i] = low_v[i]
                    af[i] = min(prev_af + inc, max_af)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
    return psar, bull


def gen_PSAR_Classic(df, start=0.02, inc=0.02, max_af=0.2, **kw):
    high_v  = df['high'].values
    low_v   = df['low'].values
    close_v = df['close'].values
    psar_v, bull_v = _psar(high_v, low_v, close_v, start, inc, max_af)
    sig = pd.Series(0, index=df.index)
    sig[bull_v]  =  1
    sig[~bull_v] = -1
    return sig


def space_PSAR_Classic(trial):
    return {
        'start':  trial.suggest_float('start', 0.01, 0.04),
        'inc':    trial.suggest_float('inc', 0.01, 0.04),
        'max_af': trial.suggest_float('max_af', 0.1, 0.4),
    }


# ── 23. PSAR_EMA ──────────────────────────────────────────────────────────────

def gen_PSAR_EMA(df, start=0.02, inc=0.02, max_af=0.2, ema_p=50, **kw):
    high_v  = df['high'].values
    low_v   = df['low'].values
    close_v = df['close'].values
    _, bull_v = _psar(high_v, low_v, close_v, start, inc, max_af)
    close   = df['close']
    ema     = _ema(close, ema_p)
    bull_s  = pd.Series(bull_v, index=df.index)
    sig = pd.Series(0, index=df.index)
    sig[bull_s  & (close > ema)] =  1
    sig[~bull_s & (close < ema)] = -1
    return sig


def space_PSAR_EMA(trial):
    return {
        'start':  trial.suggest_float('start', 0.01, 0.04),
        'inc':    trial.suggest_float('inc', 0.01, 0.04),
        'max_af': trial.suggest_float('max_af', 0.1, 0.4),
        'ema_p':  trial.suggest_int('ema_p', 20, 100),
    }


# ── 24. PSAR_RSI ──────────────────────────────────────────────────────────────

def gen_PSAR_RSI(df, start=0.02, inc=0.02, max_af=0.2, rsi_p=14, **kw):
    high_v  = df['high'].values
    low_v   = df['low'].values
    close_v = df['close'].values
    _, bull_v    = _psar(high_v, low_v, close_v, start, inc, max_af)
    bull_s       = pd.Series(bull_v, index=df.index)
    prev_bull    = pd.Series(np.concatenate([[True],  bull_v[:-1]]), index=df.index)
    prev_bear    = pd.Series(np.concatenate([[False], bull_v[:-1]]), index=df.index)
    flipped_bull = bull_s & ~prev_bull
    flipped_bear = ~bull_s & prev_bear
    rsi    = _rsi(df['close'], rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[flipped_bull & (rsi > 45)] =  1
    sig[flipped_bear & (rsi < 55)] = -1
    return sig


def space_PSAR_RSI(trial):
    return {
        'start':  trial.suggest_float('start', 0.01, 0.04),
        'inc':    trial.suggest_float('inc', 0.01, 0.04),
        'max_af': trial.suggest_float('max_af', 0.1, 0.4),
        'rsi_p':  trial.suggest_int('rsi_p', 7, 21),
    }


# ── 25. PSAR_MACD ─────────────────────────────────────────────────────────────

def gen_PSAR_MACD(df, start=0.02, inc=0.02, max_af=0.2, fast=12, slow=26, sig_p=9, **kw):
    high_v  = df['high'].values
    low_v   = df['low'].values
    close_v = df['close'].values
    _, bull_v = _psar(high_v, low_v, close_v, start, inc, max_af)
    bull_s    = pd.Series(bull_v, index=df.index)
    close     = df['close']
    macd_line, macd_signal = _macd(close, fast, slow, sig_p)
    sig = pd.Series(0, index=df.index)
    sig[bull_s  & (macd_line > macd_signal)] =  1
    sig[~bull_s & (macd_line < macd_signal)] = -1
    return sig


def space_PSAR_MACD(trial):
    return {
        'start':  trial.suggest_float('start', 0.01, 0.04),
        'inc':    trial.suggest_float('inc', 0.01, 0.04),
        'max_af': trial.suggest_float('max_af', 0.1, 0.4),
        'fast':   trial.suggest_int('fast', 8, 16),
        'slow':   trial.suggest_int('slow', 20, 30),
        'sig_p':  trial.suggest_int('sig_p', 5, 12),
    }


# ── 26. VIX_Regime ────────────────────────────────────────────────────────────

def gen_VIX_Regime(df, vol_p=20, regime_thresh=0.02, ema_p=30, **kw):
    close     = df['close']
    pct_chg   = close.pct_change().fillna(0)
    vol_std   = pct_chg.rolling(int(vol_p), min_periods=1).std()
    low_vol   = vol_std < regime_thresh     # quiet = trend regime
    high_vol  = vol_std >= regime_thresh    # volatile = mean-reversion regime
    ema       = _ema(close, ema_p)
    rsi       = _rsi(close, 14)
    sig = pd.Series(0, index=df.index)
    # Trend regime: follow the trend
    sig[low_vol  & (close > ema)] =  1
    sig[low_vol  & (close < ema)] = -1
    # Mean-reversion regime: fade extremes
    sig[high_vol & (rsi < 30)] =  1
    sig[high_vol & (rsi > 70)] = -1
    return sig


def space_VIX_Regime(trial):
    return {
        'vol_p':          trial.suggest_int('vol_p', 10, 30),
        'regime_thresh':  trial.suggest_float('regime_thresh', 0.01, 0.05),
        'ema_p':          trial.suggest_int('ema_p', 20, 60),
    }


# ── 27. BB_Width_Regime ───────────────────────────────────────────────────────

def gen_BB_Width_Regime(df, bb_p=20, bb_mult=2.0, width_thresh=0.04, **kw):
    close     = df['close']
    upper, mid, lower = _bb(close, bb_p, bb_mult)
    width     = (upper - lower) / mid.replace(0, 1e-9)
    trending  = width > width_thresh    # wide = trending
    consol    = width <= width_thresh   # narrow = consolidation
    rsi       = _rsi(close, 14)
    sig = pd.Series(0, index=df.index)
    # Trending regime: follow price vs midline
    sig[trending & (close > mid)] =  1
    sig[trending & (close < mid)] = -1
    # Consolidation regime: mean-reversion at bands
    sig[consol & (close < lower)] =  1
    sig[consol & (close > upper)] = -1
    return sig


def space_BB_Width_Regime(trial):
    return {
        'bb_p':         trial.suggest_int('bb_p', 15, 30),
        'bb_mult':      trial.suggest_float('bb_mult', 1.5, 2.5),
        'width_thresh': trial.suggest_float('width_thresh', 0.02, 0.08),
    }


# ── 28. ATR_Ratio_Filter ──────────────────────────────────────────────────────

def gen_ATR_Ratio_Filter(df, atr_p=14, ratio_thresh=0.01, rsi_p=14, **kw):
    close     = df['close']
    atr       = _atr(df, atr_p)
    ratio     = atr / close.replace(0, 1e-9)
    quiet     = ratio < ratio_thresh
    rsi       = _rsi(close, rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[quiet & (rsi < 35)] =  1
    sig[quiet & (rsi > 65)] = -1
    return sig


def space_ATR_Ratio_Filter(trial):
    return {
        'atr_p':        trial.suggest_int('atr_p', 10, 20),
        'ratio_thresh': trial.suggest_float('ratio_thresh', 0.005, 0.02),
        'rsi_p':        trial.suggest_int('rsi_p', 7, 21),
    }


# ── 29. BB_ATR_Confluence ─────────────────────────────────────────────────────

def gen_BB_ATR_Confluence(df, bb_p=20, bb_mult=2.0, atr_p=14, **kw):
    close     = df['close']
    upper, mid, lower = _bb(close, bb_p, bb_mult)
    ema       = _ema(close, 20)
    atr       = _atr(df, atr_p)
    below_bb  = close < lower
    below_ema_atr = close < ema - 1.5 * atr
    above_bb  = close > upper
    above_ema_atr = close > ema + 1.5 * atr
    sig = pd.Series(0, index=df.index)
    sig[below_bb & below_ema_atr] =  1
    sig[above_bb & above_ema_atr] = -1
    return sig


def space_BB_ATR_Confluence(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 15, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 2.5),
        'atr_p':   trial.suggest_int('atr_p', 10, 20),
    }


# ── 30. BB_ATR_Breakout ───────────────────────────────────────────────────────

def gen_BB_ATR_Breakout(df, bb_p=20, bb_mult=2.0, atr_p=14, atr_mult=2.0, **kw):
    close     = df['close']
    upper, mid, lower = _bb(close, bb_p, bb_mult)
    ema       = _ema(close, 20)
    atr       = _atr(df, atr_p)
    ema_atr_upper = ema + atr_mult * atr
    ema_atr_lower = ema - atr_mult * atr
    sig = pd.Series(0, index=df.index)
    sig[(close > upper) & (close > ema_atr_upper)] =  1
    sig[(close < lower) & (close < ema_atr_lower)] = -1
    return sig


def space_BB_ATR_Breakout(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 15, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 2.5),
        'atr_p':   trial.suggest_int('atr_p', 10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 3.0),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'BB_Classic': {
        'gen': gen_BB_Classic,
        'space': space_BB_Classic,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 8000,
                 'description': 'BB mean reversion: close < lower = long, close > upper = short'},
    },
    'BB_Breakout_v2': {
        'gen': gen_BB_Breakout,
        'space': space_BB_Breakout,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 5000,
                 'description': 'BB momentum breakout: close breaks above upper = long'},
    },
    'BB_Width_Squeeze': {
        'gen': gen_BB_Width_Squeeze,
        'space': space_BB_Width_Squeeze,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'BB squeeze: low width followed by expansion breakout'},
    },
    'BB_W_Pattern': {
        'gen': gen_BB_W_Pattern,
        'space': space_BB_W_Pattern,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'W-bottom: two touches of lower band with bounce between'},
    },
    'BB_Percent_B': {
        'gen': gen_BB_Percent_B,
        'space': space_BB_Percent_B,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': '%B oscillator crossover entries (crosses above 0 / below 1)'},
    },
    'BB_RSI_OB_OS': {
        'gen': gen_BB_RSI_OB_OS,
        'space': space_BB_RSI_OB_OS,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'BB + RSI double filter: close < lower AND RSI < 30 = long'},
    },
    'BB_MA_Cross': {
        'gen': gen_BB_MA_Cross,
        'space': space_BB_MA_Cross,
        'default_params': {'bb_p': 20, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Price crosses BB midline with RSI momentum filter'},
    },
    'BB_EMA_Confluence': {
        'gen': gen_BB_EMA_Confluence,
        'space': space_BB_EMA_Confluence,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'BB lower + EMA same level = confluence support'},
    },
    'BB_Keltner_Width': {
        'gen': gen_BB_Keltner_Width,
        'space': space_BB_Keltner_Width,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'kc_p': 20, 'kc_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'BB/KC width ratio as volatility measure; long when expanding + price up'},
    },
    'BB_Three_Sigma': {
        'gen': gen_BB_Three_Sigma,
        'space': space_BB_Three_Sigma,
        'default_params': {'bb_p': 20, 'inner_mult': 2.0, 'outer_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': '3-sigma extreme reversion: close < lower_3sigma = long'},
    },
    'ATR_Trail_Long': {
        'gen': gen_ATR_Trail_Long,
        'space': space_ATR_Trail_Long,
        'default_params': {'atr_p': 14, 'atr_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 5000,
                 'description': 'ATR trailing stop as trend indicator (long-biased running max)'},
    },
    'ATR_Trail_Bidir': {
        'gen': gen_ATR_Trail_Bidir,
        'space': space_ATR_Trail_Bidir,
        'default_params': {'atr_p': 14, 'atr_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Bidirectional ATR trailing stop; flips direction on cross'},
    },
    'ATR_Band_Breakout': {
        'gen': gen_ATR_Band_Breakout,
        'space': space_ATR_Band_Breakout,
        'default_params': {'ema_p': 30, 'atr_p': 14, 'atr_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'ATR bands around EMA momentum breakout'},
    },
    'ATR_Mean_Reversion': {
        'gen': gen_ATR_Mean_Reversion,
        'space': space_ATR_Mean_Reversion,
        'default_params': {'ema_p': 30, 'atr_p': 14, 'atr_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Close outside EMA +/- 2*ATR = extended; fade the extreme'},
    },
    'ATR_Percentile': {
        'gen': gen_ATR_Percentile,
        'space': space_ATR_Percentile,
        'default_params': {'atr_p': 14, 'rank_p': 100, 'rank_thresh': 60},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'ATR percentile rank: trade only when ATR rank < threshold'},
    },
    'CE_ATR_Dynamic': {
        'gen': gen_CE_ATR_Dynamic,
        'space': space_CE_ATR_Dynamic,
        'default_params': {'atr_p': 22, 'mult_low': 2.0, 'mult_high': 4.0, 'atr_thresh_pct': 65},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Chandelier Exit with dynamic multiplier based on ATR volatility regime'},
    },
    'CE_MACD': {
        'gen': gen_CE_MACD,
        'space': space_CE_MACD,
        'default_params': {'atr_p': 22, 'ce_mult': 3.0, 'fast': 12, 'slow': 26},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'Chandelier Exit + MACD direction alignment filter'},
    },
    'CE_RSI_Vol': {
        'gen': gen_CE_RSI_Vol,
        'space': space_CE_RSI_Vol,
        'default_params': {'atr_p': 22, 'ce_mult': 3.0, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'Chandelier Exit + RSI + volume confirmation'},
    },
    'NR4_NR7': {
        'gen': gen_NR4_NR7,
        'space': space_NR4_NR7,
        'default_params': {'nr_p': 7, 'vol_mult': 1.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'Narrow Range 4/7: breakout from narrowest range bar'},
    },
    'VO_Breakout': {
        'gen': gen_VO_Breakout,
        'space': space_VO_Breakout,
        'default_params': {'compress_p': 10, 'vol_thresh_pct': 0.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Volatility compression followed by expansion breakout'},
    },
    'Heikin_Ashi_ATR': {
        'gen': gen_Heikin_Ashi_ATR,
        'space': space_Heikin_Ashi_ATR,
        'default_params': {'consec': 2, 'atr_p': 14, 'atr_mult': 1.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'Heikin Ashi clean candles + ATR expanding confirmation'},
    },
    'PSAR_Classic': {
        'gen': gen_PSAR_Classic,
        'space': space_PSAR_Classic,
        'default_params': {'start': 0.02, 'inc': 0.02, 'max_af': 0.2},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 5000,
                 'description': 'Classic Parabolic SAR trend following'},
    },
    'PSAR_EMA': {
        'gen': gen_PSAR_EMA,
        'space': space_PSAR_EMA,
        'default_params': {'start': 0.02, 'inc': 0.02, 'max_af': 0.2, 'ema_p': 50},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'PSAR in EMA trend direction only (trend filter)'},
    },
    'PSAR_RSI': {
        'gen': gen_PSAR_RSI,
        'space': space_PSAR_RSI,
        'default_params': {'start': 0.02, 'inc': 0.02, 'max_af': 0.2, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'PSAR flip + RSI confirmation filter'},
    },
    'PSAR_MACD': {
        'gen': gen_PSAR_MACD,
        'space': space_PSAR_MACD,
        'default_params': {'start': 0.02, 'inc': 0.02, 'max_af': 0.2, 'fast': 12, 'slow': 26, 'sig_p': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1800,
                 'description': 'PSAR + MACD signal alignment for trend confirmation'},
    },
    'VIX_Regime': {
        'gen': gen_VIX_Regime,
        'space': space_VIX_Regime,
        'default_params': {'vol_p': 20, 'regime_thresh': 0.02, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Crypto VIX regime: low vol = trend, high vol = mean reversion'},
    },
    'BB_Width_Regime': {
        'gen': gen_BB_Width_Regime,
        'space': space_BB_Width_Regime,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'width_thresh': 0.04},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1800,
                 'description': 'BB width regime filter: narrow = range trade, wide = trend follow'},
    },
    'ATR_Ratio_Filter': {
        'gen': gen_ATR_Ratio_Filter,
        'space': space_ATR_Ratio_Filter,
        'default_params': {'atr_p': 14, 'ratio_thresh': 0.01, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'ATR/close ratio normalized volatility: quiet market + RSI extremes'},
    },
    'BB_ATR_Confluence': {
        'gen': gen_BB_ATR_Confluence,
        'space': space_BB_ATR_Confluence,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'BB lower + EMA-ATR double oversold confluence'},
    },
    'BB_ATR_Breakout': {
        'gen': gen_BB_ATR_Breakout,
        'space': space_BB_ATR_Breakout,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'atr_p': 14, 'atr_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Close breaks both BB upper and EMA+ATR band = strong double breakout'},
    },
}
