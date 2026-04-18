#!/usr/bin/env python3
"""
TV2 BATCH 42 — Pine Script v4/v5/v6 strategies converted to Python
Categories covered:
  Nadaraya-Watson Envelope, UT Bot Alerts, Lux Algo Signals, Godmode Oscillator,
  Squeeze Pro, VWAP+Stdev Bands, Half Trend, Wave Trend Divergences, MOST,
  Coral Trend, Andean Oscillator, ZLSMA+UT Bot, Fancy RSI, MACD Histogram Divergence,
  Stochastic Divergence, Money Flow+RSI, Elder Impulse System v2, Parabolic SAR+EMA,
  SuperTrend+MACD, ADX+DMI Signals, CCI+ATR Bands, RSI Stochastic Combo,
  Pivot Point SuperTrend, Linear Regression Bands, SSL Hybrid, Chande Kroll Stop,
  T3 Smooth MA Cross, Gann Hi-Lo Activator, Holt-Winters Double EMA, CCI MTF

Sources: HPotter/Indicators-for-Everyone, everget/TradingView-Pine-Script-Indicators,
         lazybear/indicators, AlgoAlpha public scripts, LuxAlgo public scripts,
         cheatcountry/pinescripts (GitHub public)
"""

import pandas as pd
import numpy as np

# ─── Standard helpers ────────────────────────────────────────────────────────
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()
def _rma(s, n): return s.ewm(alpha=1/n, adjust=False).mean()
def _atr(df, n):
    hi, lo, cl = df['high'], df['low'], df['close']
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    return _rma(tr, n)
def _rsi(s, n):
    d = s.diff()
    return 100 - 100/(1 + _rma(d.clip(lower=0), n) / _rma((-d).clip(lower=0), n))
def _wma(s, n):
    w = np.arange(1, n+1)
    return s.rolling(n).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
def _hma(s, n):
    return _wma(2*_wma(s, n//2) - _wma(s, n), int(np.sqrt(n)))
def _crossover(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))
def _crossunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))
def _macd(s, fast=12, slow=26, sig=9):
    fm = _ema(s, fast); sm = _ema(s, slow)
    m = fm - sm; signal = _ema(m, sig)
    return m, signal, m - signal
def _stochrsi(s, rsi_len=14, stoch_len=14, k_smooth=3, d_smooth=3):
    rsi = _rsi(s, rsi_len)
    lo = rsi.rolling(stoch_len).min(); hi = rsi.rolling(stoch_len).max()
    k = 100 * (rsi - lo) / (hi - lo + 1e-10)
    k = k.rolling(k_smooth).mean(); d = k.rolling(d_smooth).mean()
    return k, d
def _linreg(s, n):
    def _lr(arr):
        if np.any(np.isnan(arr)): return np.nan
        x = np.arange(n); m, b = np.polyfit(x, arr, 1)
        return m * (n-1) + b
    return s.rolling(n).apply(_lr, raw=True)
def _zlsma(s, n):
    lsma = _linreg(s, n); lsma2 = _linreg(lsma, n)
    return lsma + (lsma - lsma2)
def _dmi(df, n=14, lensig=14):
    hi, lo, cl = df['high'], df['low'], df['close']
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    up = hi.diff(); dn = -lo.diff()
    dm_p = np.where((up > dn) & (up > 0), up, 0.0)
    dm_m = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_n = _rma(tr, n)
    di_p = 100 * _rma(pd.Series(dm_p, index=df.index), n) / (atr_n + 1e-10)
    di_m = 100 * _rma(pd.Series(dm_m, index=df.index), n) / (atr_n + 1e-10)
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-10)
    return di_p, di_m, _rma(dx, lensig)
def _supertrend(df, n, mult):
    atr = _atr(df, n); hl2 = (df['high'] + df['low']) / 2
    upper = hl2 + mult * atr; lower = hl2 - mult * atr
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i-1]: trend.iloc[i] = 1
        elif df['close'].iloc[i] < lower.iloc[i-1]: trend.iloc[i] = -1
        else: trend.iloc[i] = trend.iloc[i-1]
    return trend

# ─────────────────────────────────────────────────────────────────────────────
# 1. Nadaraya_Watson_Envelope  (Pine v5 — everget / LuxAlgo)
# Kernel regression envelope; signal on price re-entry from outside bands
# Long: close was below lower band, re-enters above it
# Short: close was above upper band, re-enters below it
# ─────────────────────────────────────────────────────────────────────────────
def gen_Nadaraya_Watson(df, h=8.0, mult=3.0, rep_len=500, **kw):
    cl = df['close']
    n = len(cl)
    nwe = np.full(n, np.nan)
    look = min(rep_len, n)
    for i in range(look - 1, n):
        seg = cl.iloc[i - look + 1: i + 1].values
        x = np.arange(look, dtype=float)
        weights = np.exp(-((x - (look - 1)) ** 2) / (2 * h * h))
        nwe[i] = np.dot(weights, seg) / (weights.sum() + 1e-10)
    nwe_s = pd.Series(nwe, index=df.index)
    # ATR-based envelope
    atr = _atr(df, 14)
    upper = nwe_s + mult * atr
    lower = nwe_s - mult * atr
    # Signal: crossover/under of lower/upper bands
    long_entry  = _crossover(cl, lower)
    short_entry = _crossunder(cl, upper)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(n):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Nadaraya_Watson = {
    'h':       ('float', [4.0, 8.0, 16.0]),
    'mult':    ('float', [2.0, 3.0, 4.0]),
    'rep_len': ('int',   [200, 500, 800]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. UT_Bot_Alerts  (Pine v4 — HPotter/cheatcountry "UT Bot")
# ATR trailing stop; long on close > trail after being below, short vice-versa
# ─────────────────────────────────────────────────────────────────────────────
def gen_UT_Bot_Alerts(df, key_value=1.0, atr_period=10, **kw):
    cl = df['close']
    n = len(cl)
    atr = _atr(df, atr_period) * key_value
    trail = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        prev = trail.iloc[i-1]
        c = cl.iloc[i]; a = atr.iloc[i]
        if c > prev:
            trail.iloc[i] = max(prev, c - a)
        else:
            trail.iloc[i] = min(prev, c + a)
    long_entry  = _crossover(cl, trail)
    short_entry = _crossunder(cl, trail)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(n):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_UT_Bot_Alerts = {
    'key_value':  ('float', [0.5, 1.0, 2.0, 3.0]),
    'atr_period': ('int',   [7, 10, 14]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Lux_Algo_Signals  (Pine v5 — LuxAlgo public)
# Confirmation (EMA cross) + Momentum (RSI>50/50) + Exit (ATR trailing)
# Long: fast>slow + RSI>50 + price > atr_trail
# Short: fast<slow + RSI<50 + price < atr_trail
# ─────────────────────────────────────────────────────────────────────────────
def gen_Lux_Algo_Signals(df, fast=14, slow=28, rsi_len=14,
                          atr_len=14, atr_mult=2.0, **kw):
    cl = df['close']
    fast_ma = _ema(cl, fast)
    slow_ma = _ema(cl, slow)
    rsi = _rsi(cl, rsi_len)
    # ATR trailing baseline
    atr = _atr(df, atr_len) * atr_mult
    trail = pd.Series(0.0, index=df.index)
    for i in range(1, len(cl)):
        prev = trail.iloc[i-1]
        c = cl.iloc[i]; a = atr.iloc[i]
        trail.iloc[i] = max(prev, c - a) if c > prev else min(prev, c + a)
    confirm_long  = fast_ma > slow_ma
    confirm_short = fast_ma < slow_ma
    mom_long  = rsi > 50
    mom_short = rsi < 50
    exit_long  = cl < trail
    exit_short = cl > trail
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if confirm_long.iloc[i] and mom_long.iloc[i] and not exit_long.iloc[i]:
                in_pos = 1
            elif confirm_short.iloc[i] and mom_short.iloc[i] and not exit_short.iloc[i]:
                in_pos = -1
        elif in_pos == 1:
            if exit_long.iloc[i] or confirm_short.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if exit_short.iloc[i] or confirm_long.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Lux_Algo_Signals = {
    'fast':     ('int',   [8, 14, 21]),
    'slow':     ('int',   [20, 28, 50]),
    'rsi_len':  ('int',   [10, 14, 20]),
    'atr_len':  ('int',   [10, 14, 20]),
    'atr_mult': ('float', [1.5, 2.0, 3.0]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Godmode_Oscillator  (Pine v4 — cheatcountry/Godmode)
# Combo: Ultimate Oscillator + Awesome Oscillator + weighted Stochastic
# Signal: oscillator crosses zero line
# ─────────────────────────────────────────────────────────────────────────────
def gen_Godmode_Oscillator(df, uo_fast=7, uo_mid=14, uo_slow=28,
                            ao_fast=5, ao_slow=34, stoch_len=14, **kw):
    hi, lo, cl = df['high'], df['low'], df['close']
    # Ultimate Oscillator
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    bp = cl - pd.concat([lo, cl.shift()], axis=1).min(axis=1)
    avg_f = bp.rolling(uo_fast).sum() / (tr.rolling(uo_fast).sum() + 1e-10)
    avg_m = bp.rolling(uo_mid).sum()  / (tr.rolling(uo_mid).sum()  + 1e-10)
    avg_s = bp.rolling(uo_slow).sum() / (tr.rolling(uo_slow).sum() + 1e-10)
    uo = 100 * (4 * avg_f + 2 * avg_m + avg_s) / 7
    # Awesome Oscillator
    hl2 = (hi + lo) / 2
    ao = _sma(hl2, ao_fast) - _sma(hl2, ao_slow)
    # Stochastic
    lo_k = lo.rolling(stoch_len).min(); hi_k = hi.rolling(stoch_len).max()
    stoch_k = 100 * (cl - lo_k) / (hi_k - lo_k + 1e-10)
    stoch_d = stoch_k.rolling(3).mean()
    # Godmode composite (centered)
    god = (uo - 50) * 0.4 + ao * 0.3 + (stoch_d - 50) * 0.3
    long_entry  = _crossover(god, pd.Series(0.0, index=df.index))
    short_entry = _crossunder(god, pd.Series(0.0, index=df.index))
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Godmode_Oscillator = {
    'uo_fast':   ('int', [5, 7, 10]),
    'uo_mid':    ('int', [10, 14, 20]),
    'uo_slow':   ('int', [20, 28, 40]),
    'stoch_len': ('int', [10, 14, 20]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Squeeze_Pro  (Pine v5 — LazyBear TTM Squeeze Pro)
# 3 squeeze thresholds (narrow/normal/wide KC vs BB); momentum histogram
# Long when histogram turns positive coming from squeeze
# Short when histogram turns negative coming from squeeze
# ─────────────────────────────────────────────────────────────────────────────
def gen_Squeeze_Pro(df, bb_len=20, bb_mult=2.0,
                    kc_len=20, kc_mult_n=1.0, kc_mult_w=2.0,
                    mom_len=12, **kw):
    cl = df['close']; hi = df['high']; lo = df['low']
    # Bollinger Bands
    bb_basis = _sma(cl, bb_len)
    bb_dev = bb_mult * cl.rolling(bb_len).std(ddof=0)
    bb_up = bb_basis + bb_dev; bb_lo = bb_basis - bb_dev
    # Keltner Channel (narrow & wide)
    atr = _atr(df, kc_len)
    kc_up_n = bb_basis + kc_mult_n * atr; kc_lo_n = bb_basis - kc_mult_n * atr
    kc_up_w = bb_basis + kc_mult_w * atr; kc_lo_w = bb_basis - kc_mult_w * atr
    # Squeeze states: True = squeeze on
    sqz_narrow = (bb_up < kc_up_n) & (bb_lo > kc_lo_n)
    sqz_normal = (bb_up < bb_basis + kc_mult_n * 1.5 * atr) & (bb_lo > bb_basis - kc_mult_n * 1.5 * atr)
    sqz_wide   = (bb_up < kc_up_w) & (bb_lo > kc_lo_w)
    any_sqz = sqz_narrow | sqz_normal | sqz_wide
    # Momentum (linear regression of delta)
    delta = cl - ((hi.rolling(mom_len).max() + lo.rolling(mom_len).min()) / 2 + _sma(cl, mom_len)) / 2
    mom = _linreg(delta, mom_len)
    # Signal: momentum crosses zero after squeeze
    _sqz_arr = any_sqz.values.astype(bool)
    was_sqz = pd.Series(np.concatenate([[False], _sqz_arr[:-1]]), index=df.index)
    long_entry  = was_sqz & _crossover(mom, pd.Series(0.0, index=df.index))
    short_entry = was_sqz & _crossunder(mom, pd.Series(0.0, index=df.index))
    # Also allow entries without squeeze condition
    plain_long  = _crossover(mom, pd.Series(0.0, index=df.index))
    plain_short = _crossunder(mom, pd.Series(0.0, index=df.index))
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if plain_long.iloc[i]:  in_pos = 1
            elif plain_short.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if plain_short.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if plain_long.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Squeeze_Pro = {
    'bb_len':     ('int',   [14, 20, 28]),
    'bb_mult':    ('float', [1.5, 2.0, 2.5]),
    'kc_len':     ('int',   [14, 20, 28]),
    'kc_mult_n':  ('float', [0.75, 1.0, 1.5]),
    'kc_mult_w':  ('float', [1.5, 2.0, 3.0]),
    'mom_len':    ('int',   [8, 12, 20]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. VWAP_Stdev_Bands  (Pine v5 — standard VWAP + deviation bands)
# Intraday reversion: long near -1 stdev, short near +1 stdev
# Uses rolling VWAP (approximated as volume-weighted SMA for multi-day data)
# ─────────────────────────────────────────────────────────────────────────────
def gen_VWAP_Stdev_Bands(df, vwap_len=50, stdev_mult1=1.0,
                          stdev_mult2=2.0, stdev_mult3=3.0, **kw):
    cl = df['close']; vol = df['volume']
    tp = (df['high'] + df['low'] + cl) / 3
    # Rolling VWAP
    vwap = (tp * vol).rolling(vwap_len).sum() / (vol.rolling(vwap_len).sum() + 1e-10)
    # Rolling stdev of TP
    stdev = tp.rolling(vwap_len).std(ddof=0)
    upper1 = vwap + stdev_mult1 * stdev
    lower1 = vwap - stdev_mult1 * stdev
    upper2 = vwap + stdev_mult2 * stdev
    lower2 = vwap - stdev_mult2 * stdev
    # Long: price bounces off lower1/lower2 band
    long_entry  = _crossover(cl, lower1) | _crossover(cl, lower2)
    short_entry = _crossunder(cl, upper1) | _crossunder(cl, upper2)
    long_exit   = cl > vwap
    short_exit  = cl < vwap
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i] or short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i] or long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_VWAP_Stdev_Bands = {
    'vwap_len':     ('int',   [20, 50, 100]),
    'stdev_mult1':  ('float', [0.75, 1.0, 1.5]),
    'stdev_mult2':  ('float', [1.5, 2.0, 2.5]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. Half_Trend  (Pine v4 — Alex Orekhov, everget port)
# Trend channel based on ATR half-amplitude; signal on direction change
# ─────────────────────────────────────────────────────────────────────────────
def gen_Half_Trend(df, amplitude=2, channel_dev=2.0, **kw):
    cl = df['close']; hi = df['high']; lo = df['low']
    n = len(df)
    atr2 = _atr(df, 100) / 2
    dev = channel_dev * atr2
    hma_val = _hma(cl, amplitude * 2)
    trend = pd.Series(0, index=df.index)
    next_trend = pd.Series(0, index=df.index)
    up = pd.Series(np.nan, index=df.index)
    dn = pd.Series(np.nan, index=df.index)
    for i in range(1, n):
        nt = next_trend.iloc[i-1]
        if hma_val.iloc[i] < hma_val.iloc[i-1]:
            nt_new = 1 if cl.iloc[i] < hma_val.iloc[i] else nt
        else:
            nt_new = -1 if cl.iloc[i] > hma_val.iloc[i] else nt
        next_trend.iloc[i] = nt_new
        up_v = up.iloc[i-1] if not np.isnan(up.iloc[i-1]) else cl.iloc[i]
        dn_v = dn.iloc[i-1] if not np.isnan(dn.iloc[i-1]) else cl.iloc[i]
        if nt_new == -1:
            up.iloc[i] = max(up_v, lo.iloc[i] - dev.iloc[i])
            dn.iloc[i] = dn_v
        else:
            up.iloc[i] = up_v
            dn.iloc[i] = min(dn_v, hi.iloc[i] + dev.iloc[i])
        if nt_new == trend.iloc[i-1]:
            trend.iloc[i] = trend.iloc[i-1]
        else:
            trend.iloc[i] = nt_new
    long_entry  = (trend == -1) & (trend.shift(1) == 1)
    short_entry = (trend == 1)  & (trend.shift(1) == -1)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(n):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Half_Trend = {
    'amplitude':    ('int',   [1, 2, 3]),
    'channel_dev':  ('float', [1.0, 2.0, 3.0]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 8. Wave_Trend_Divergence  (Pine v4 — LazyBear WaveTrend + divergence)
# WT1/WT2 lines with bull/bear divergence detection
# Long: WT1 crossover WT2 from oversold OR bull divergence
# Short: WT1 crossunder WT2 from overbought OR bear divergence
# ─────────────────────────────────────────────────────────────────────────────
def gen_Wave_Trend_Divergence(df, n1=10, n2=21, ob_level=60, os_level=-60, **kw):
    cl = df['close']; hi = df['high']; lo = df['low']
    hlc3 = (hi + lo + cl) / 3
    esa = _ema(hlc3, n1)
    d = _ema((hlc3 - esa).abs(), n1)
    ci = (hlc3 - esa) / (0.015 * d + 1e-10)
    wt1 = _ema(ci, n2)
    wt2 = _sma(wt1, 4)
    # Basic crossover signals
    cross_up   = _crossover(wt1, wt2) & (wt1 < os_level)
    cross_down = _crossunder(wt1, wt2) & (wt1 > ob_level)
    # Simplified divergence: price makes lower low but WT makes higher low (bull)
    lookback = 5
    price_ll = cl < cl.rolling(lookback).min().shift(1)
    wt_hl    = wt1 > wt1.rolling(lookback).min().shift(1)
    bull_div  = price_ll & wt_hl & (wt1 < os_level + 10)
    price_hh  = cl > cl.rolling(lookback).max().shift(1)
    wt_lh     = wt1 < wt1.rolling(lookback).max().shift(1)
    bear_div  = price_hh & wt_lh & (wt1 > ob_level - 10)
    long_entry  = cross_up | bull_div
    short_entry = cross_down | bear_div
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if cross_down.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if cross_up.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Wave_Trend_Divergence = {
    'n1':        ('int', [8, 10, 14]),
    'n2':        ('int', [16, 21, 30]),
    'ob_level':  ('int', [53, 60, 70]),
    'os_level':  ('int', [-70, -60, -53]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 9. MOST_Strategy  (Pine v4 — Turkish MOST indicator, HPotter port)
# Moving Stop Loss: EMA + adaptive trailing stop
# Long: close crosses above MOST line
# Short: close crosses below MOST line
# ─────────────────────────────────────────────────────────────────────────────
def gen_MOST_Strategy(df, length=14, percent=2.0, ma_type='ema', **kw):
    cl = df['close']
    if ma_type == 'sma':
        src = _sma(cl, length)
    elif ma_type == 'hma':
        src = _hma(cl, length)
    else:
        src = _ema(cl, length)
    stop_pct = percent / 100.0
    most = pd.Series(0.0, index=df.index)
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        prev_most = most.iloc[i-1]
        m = src.iloc[i]
        if trend.iloc[i-1] == 1:
            val = max(prev_most, m * (1 - stop_pct))
            if cl.iloc[i] < val:
                trend.iloc[i] = -1
                most.iloc[i] = m * (1 + stop_pct)
            else:
                trend.iloc[i] = 1
                most.iloc[i] = val
        else:
            val = min(prev_most, m * (1 + stop_pct))
            if cl.iloc[i] > val:
                trend.iloc[i] = 1
                most.iloc[i] = m * (1 - stop_pct)
            else:
                trend.iloc[i] = -1
                most.iloc[i] = val
    long_entry  = (trend == 1) & (trend.shift(1) == -1)
    short_entry = (trend == -1) & (trend.shift(1) == 1)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_MOST_Strategy = {
    'length':  ('int',   [7, 14, 21]),
    'percent': ('float', [1.0, 2.0, 3.0, 5.0]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 10. Coral_Trend  (Pine v5 — Lazybear Coral Trend + RSI confirmation)
# Coral = slow EMA of EMA of EMA (triple smoothing); momentum via RSI
# Long: close crosses above Coral + RSI > 50
# Short: close crosses below Coral + RSI < 50
# ─────────────────────────────────────────────────────────────────────────────
def gen_Coral_Trend(df, sm=21, cd=0.4, rsi_len=14, **kw):
    cl = df['close']
    # Coral calculation: di = (sm - 1) / 2 + 1; coeff from smoothing
    di = (sm - 1) / 2 + 1
    c1 = 2 / (di + 1)
    # Triple smoothed EMA
    ema1 = _ema(cl, int(di))
    ema2 = _ema(ema1, int(di))
    ema3 = _ema(ema2, int(di))
    coral = ema3
    rsi = _rsi(cl, rsi_len)
    long_entry  = _crossover(cl, coral) & (rsi > 50)
    short_entry = _crossunder(cl, coral) & (rsi < 50)
    long_exit   = cl < coral
    short_exit  = cl > coral
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Coral_Trend = {
    'sm':      ('int',   [14, 21, 34]),
    'rsi_len': ('int',   [10, 14, 20]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 11. Andean_Oscillator  (Pine v5 — alexgrover public)
# Bull/Bear component oscillator; signal line crossover
# Long: signal crosses above bear component
# Short: signal crosses below bull component
# ─────────────────────────────────────────────────────────────────────────────
def gen_Andean_Oscillator(df, length=50, sig_len=9, **kw):
    cl = df['close']; op = df['open']
    n = len(df)
    alpha = 2.0 / (length + 1)
    bull = pd.Series(0.0, index=df.index)
    bear = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        c = cl.iloc[i]; o = op.iloc[i]
        bull.iloc[i] = max(c - bull.iloc[i-1], o - bull.iloc[i-1], 0) * alpha + bull.iloc[i-1] * (1 - alpha)
        bear.iloc[i] = max(bull.iloc[i-1] - c, bull.iloc[i-1] - o, 0) * alpha + bear.iloc[i-1] * (1 - alpha)
    sig_line = _ema(bull + bear, sig_len)
    long_entry  = _crossover(bull, sig_line)
    short_entry = _crossunder(bear, sig_line)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(n):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Andean_Oscillator = {
    'length':  ('int', [30, 50, 80]),
    'sig_len': ('int', [5, 9, 14]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 12. ZLSMA_UT_Bot  (Pine v5 — ZLSMA combined with UT Bot exit)
# Entry: ZLSMA crossover (fast vs slow)
# Exit: UT Bot ATR trailing crosses against position
# ─────────────────────────────────────────────────────────────────────────────
def gen_ZLSMA_UT_Bot(df, fast_len=14, slow_len=50,
                     key_value=1.5, atr_period=10, **kw):
    cl = df['close']
    zl_fast = _zlsma(cl, fast_len)
    zl_slow = _zlsma(cl, slow_len)
    # UT Bot trailing stop
    atr = _atr(df, atr_period) * key_value
    trail = pd.Series(0.0, index=df.index)
    for i in range(1, len(cl)):
        prev = trail.iloc[i-1]; c = cl.iloc[i]; a = atr.iloc[i]
        trail.iloc[i] = max(prev, c - a) if c > prev else min(prev, c + a)
    long_entry  = _crossover(zl_fast, zl_slow)
    short_entry = _crossunder(zl_fast, zl_slow)
    long_exit   = _crossunder(cl, trail)
    short_exit  = _crossover(cl, trail)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i] or short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i] or long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_ZLSMA_UT_Bot = {
    'fast_len':   ('int',   [8, 14, 21]),
    'slow_len':   ('int',   [30, 50, 80]),
    'key_value':  ('float', [0.5, 1.0, 1.5, 2.0]),
    'atr_period': ('int',   [7, 10, 14]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 13. Fancy_RSI  (Pine v5 — smoothed RSI with divergence)
# RSI smoothed by SMA; long on cross-up of smoothed RSI from oversold
# Divergence: price lower low but RSI higher low -> bullish
# ─────────────────────────────────────────────────────────────────────────────
def gen_Fancy_RSI(df, rsi_len=14, smooth=3, ob=70, os=30,
                  div_lookback=5, **kw):
    cl = df['close']
    rsi = _rsi(cl, rsi_len)
    srsi = _sma(rsi, smooth)
    # Standard signal
    long_entry  = _crossover(srsi, pd.Series(os, index=df.index))
    short_entry = _crossunder(srsi, pd.Series(ob, index=df.index))
    # Divergence
    price_ll = cl < cl.rolling(div_lookback).min().shift(1)
    rsi_hl   = srsi > srsi.rolling(div_lookback).min().shift(1)
    bull_div  = price_ll & rsi_hl & (srsi < os + 15)
    price_hh  = cl > cl.rolling(div_lookback).max().shift(1)
    rsi_lh    = srsi < srsi.rolling(div_lookback).max().shift(1)
    bear_div  = price_hh & rsi_lh & (srsi > ob - 15)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i] or bull_div.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i] or bear_div.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Fancy_RSI = {
    'rsi_len':     ('int', [10, 14, 21]),
    'smooth':      ('int', [2, 3, 5]),
    'ob':          ('int', [65, 70, 75]),
    'os':          ('int', [25, 30, 35]),
    'div_lookback': ('int', [4, 5, 8]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 14. MACD_Histogram_Divergence  (Pine v5)
# MACD histogram divergence: histogram makes lower high while price makes
# higher high (bear) or histogram makes higher low while price makes lower low (bull)
# ─────────────────────────────────────────────────────────────────────────────
def gen_MACD_Hist_Divergence(df, fast=12, slow=26, sig_len=9,
                              div_lookback=5, **kw):
    cl = df['close']
    _, _, hist = _macd(cl, fast, slow, sig_len)
    # Bullish: price lower low, histogram higher low (positive histogram rising)
    price_ll = cl < cl.rolling(div_lookback).min().shift(1)
    hist_hl  = hist > hist.rolling(div_lookback).min().shift(1)
    bull_div  = price_ll & hist_hl & (hist < 0)
    # Bearish: price higher high, histogram lower high
    price_hh = cl > cl.rolling(div_lookback).max().shift(1)
    hist_lh  = hist < hist.rolling(div_lookback).max().shift(1)
    bear_div  = price_hh & hist_lh & (hist > 0)
    # Standard MACD cross
    _, signal, hist2 = _macd(cl, fast, slow, sig_len)
    macd_line, _ , _ = _macd(cl, fast, slow, sig_len)
    std_long  = _crossover(macd_line, signal)
    std_short = _crossunder(macd_line, signal)
    long_entry  = bull_div | std_long
    short_entry = bear_div | std_short
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_MACD_Hist_Divergence = {
    'fast':        ('int', [8, 12, 16]),
    'slow':        ('int', [20, 26, 34]),
    'sig_len':     ('int', [7, 9, 12]),
    'div_lookback': ('int', [4, 5, 8]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 15. Stochastic_Divergence  (Pine v5)
# Stochastic K/D with bull/bear divergence detection
# Long: stoch crossover from oversold + bull divergence
# Short: stoch crossunder from overbought + bear divergence
# ─────────────────────────────────────────────────────────────────────────────
def gen_Stochastic_Divergence(df, k_len=14, d_smooth=3,
                               ob=80, os=20, div_lb=5, **kw):
    cl = df['close']; hi = df['high']; lo = df['low']
    lo_k = lo.rolling(k_len).min(); hi_k = hi.rolling(k_len).max()
    k = 100 * (cl - lo_k) / (hi_k - lo_k + 1e-10)
    d = k.rolling(d_smooth).mean()
    cross_up   = _crossover(k, d) & (k < os + 5)
    cross_down = _crossunder(k, d) & (k > ob - 5)
    price_ll = cl < cl.rolling(div_lb).min().shift(1)
    k_hl     = k > k.rolling(div_lb).min().shift(1)
    bull_div  = price_ll & k_hl & (k < os + 20)
    price_hh = cl > cl.rolling(div_lb).max().shift(1)
    k_lh     = k < k.rolling(div_lb).max().shift(1)
    bear_div  = price_hh & k_lh & (k > ob - 20)
    long_entry  = cross_up | bull_div
    short_entry = cross_down | bear_div
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if cross_down.iloc[i] or bear_div.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if cross_up.iloc[i] or bull_div.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Stochastic_Divergence = {
    'k_len':   ('int', [10, 14, 21]),
    'd_smooth': ('int', [2, 3, 5]),
    'ob':      ('int', [75, 80, 85]),
    'os':      ('int', [15, 20, 25]),
    'div_lb':  ('int', [4, 5, 8]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 16. Money_Flow_RSI  (Pine v4 — Chaikin Money Flow + RSI combined)
# Long: CMF > threshold AND RSI crosses up 50
# Short: CMF < -threshold AND RSI crosses under 50
# ─────────────────────────────────────────────────────────────────────────────
def gen_Money_Flow_RSI(df, cmf_len=20, rsi_len=14,
                        cmf_thresh=0.05, **kw):
    cl = df['close']; hi = df['high']; lo = df['lo'] if 'lo' in df else df['low']
    vol = df['volume']
    # Chaikin Money Flow
    mfm = ((cl - lo) - (hi - cl)) / (hi - lo + 1e-10)
    mfv = mfm * vol
    cmf = mfv.rolling(cmf_len).sum() / (vol.rolling(cmf_len).sum() + 1e-10)
    rsi = _rsi(cl, rsi_len)
    mid = pd.Series(50.0, index=df.index)
    rsi_cross_up   = _crossover(rsi, mid)
    rsi_cross_down = _crossunder(rsi, mid)
    long_entry  = rsi_cross_up   & (cmf > cmf_thresh)
    short_entry = rsi_cross_down & (cmf < -cmf_thresh)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Money_Flow_RSI = {
    'cmf_len':    ('int',   [10, 20, 30]),
    'rsi_len':    ('int',   [10, 14, 20]),
    'cmf_thresh': ('float', [0.02, 0.05, 0.10]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 17. Elder_Impulse_v2  (Pine v5 — Elder Impulse System)
# EMA slope direction + MACD histogram direction → color system
# Long when both are positive (green), Short when both negative (red)
# ─────────────────────────────────────────────────────────────────────────────
def gen_Elder_Impulse_v2(df, ema_len=13, fast=12, slow=26, sig_len=9, **kw):
    cl = df['close']
    ema_val = _ema(cl, ema_len)
    ema_slope = ema_val - ema_val.shift(1)
    _, _, hist = _macd(cl, fast, slow, sig_len)
    hist_slope = hist - hist.shift(1)
    # Green: both up; Red: both down
    green = (ema_slope > 0) & (hist_slope > 0)
    red   = (ema_slope < 0) & (hist_slope < 0)
    _green_arr = green.values.astype(bool)
    _red_arr   = red.values.astype(bool)
    _prev_green = np.concatenate([[False], _green_arr[:-1]])
    _prev_red   = np.concatenate([[False], _red_arr[:-1]])
    long_entry  = pd.Series(_green_arr & ~_prev_green, index=df.index)
    short_entry = pd.Series(_red_arr   & ~_prev_red,   index=df.index)
    long_exit   = red
    short_exit  = green
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Elder_Impulse_v2 = {
    'ema_len':  ('int', [8, 13, 21]),
    'fast':     ('int', [8, 12, 16]),
    'slow':     ('int', [20, 26, 34]),
    'sig_len':  ('int', [7, 9, 12]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 18. Parabolic_SAR_EMA  (Pine v4 — PSAR + EMA trend filter)
# Long: PSAR flips below price AND price > EMA (uptrend)
# Short: PSAR flips above price AND price < EMA (downtrend)
# ─────────────────────────────────────────────────────────────────────────────
def gen_Parabolic_SAR_EMA(df, sar_start=0.02, sar_inc=0.02,
                           sar_max=0.2, ema_len=200, **kw):
    cl = df['close']; hi = df['high']; lo = df['low']
    n = len(df)
    ema_val = _ema(cl, ema_len)
    # Compute PSAR
    psar = pd.Series(0.0, index=df.index)
    bull = True
    af = sar_start; ep = lo.iloc[0]
    psar.iloc[0] = hi.iloc[0]
    for i in range(1, n):
        if bull:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            psar.iloc[i] = min(psar.iloc[i], lo.iloc[i-1])
            if i > 1: psar.iloc[i] = min(psar.iloc[i], lo.iloc[i-2])
            if lo.iloc[i] < psar.iloc[i]:
                bull = False; psar.iloc[i] = ep
                af = sar_start; ep = lo.iloc[i]
            else:
                if hi.iloc[i] > ep:
                    ep = hi.iloc[i]; af = min(af + sar_inc, sar_max)
        else:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            psar.iloc[i] = max(psar.iloc[i], hi.iloc[i-1])
            if i > 1: psar.iloc[i] = max(psar.iloc[i], hi.iloc[i-2])
            if hi.iloc[i] > psar.iloc[i]:
                bull = True; psar.iloc[i] = ep
                af = sar_start; ep = hi.iloc[i]
            else:
                if lo.iloc[i] < ep:
                    ep = lo.iloc[i]; af = min(af + sar_inc, sar_max)
    sar_bull = cl > psar
    _sb_arr = sar_bull.values.astype(bool)
    _prev_sb = np.concatenate([[False], _sb_arr[:-1]])
    sar_flip_up   = pd.Series(_sb_arr  & ~_prev_sb, index=df.index)
    sar_flip_down = pd.Series(~_sb_arr &  _prev_sb, index=df.index)
    long_entry  = sar_flip_up   & (cl > ema_val)
    short_entry = sar_flip_down & (cl < ema_val)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(n):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if sar_flip_down.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if sar_flip_up.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Parabolic_SAR_EMA = {
    'sar_start': ('float', [0.01, 0.02, 0.03]),
    'sar_inc':   ('float', [0.01, 0.02, 0.03]),
    'sar_max':   ('float', [0.1, 0.2, 0.3]),
    'ema_len':   ('int',   [100, 200, 300]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 19. SuperTrend_MACD  (Pine v5 — dual confirmation: trend + momentum)
# Long: SuperTrend bullish AND MACD crossover signal
# Short: SuperTrend bearish AND MACD crossunder signal
# ─────────────────────────────────────────────────────────────────────────────
def gen_SuperTrend_MACD(df, st_period=10, st_mult=3.0,
                         fast=12, slow=26, sig_len=9, **kw):
    cl = df['close']
    trend = _supertrend(df, st_period, st_mult)
    macd_line, signal, _ = _macd(cl, fast, slow, sig_len)
    long_entry  = (trend == 1) & _crossover(macd_line, signal)
    short_entry = (trend == -1) & _crossunder(macd_line, signal)
    long_exit   = (trend == -1)
    short_exit  = (trend == 1)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_SuperTrend_MACD = {
    'st_period': ('int',   [7, 10, 14]),
    'st_mult':   ('float', [2.0, 3.0, 4.0]),
    'fast':      ('int',   [8, 12, 16]),
    'slow':      ('int',   [20, 26, 34]),
    'sig_len':   ('int',   [7, 9, 12]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 20. ADX_DMI_Signals  (Pine v4 — ADX threshold + DI crossover)
# Long: DI+ crosses above DI- AND ADX > threshold (trend strong)
# Short: DI- crosses above DI+ AND ADX > threshold
# ─────────────────────────────────────────────────────────────────────────────
def gen_ADX_DMI_Signals(df, dmi_len=14, adx_threshold=20, **kw):
    di_plus, di_minus, adx = _dmi(df, dmi_len, dmi_len)
    strong = adx > adx_threshold
    long_entry  = _crossover(di_plus, di_minus) & strong
    short_entry = _crossover(di_minus, di_plus) & strong
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_ADX_DMI_Signals = {
    'dmi_len':       ('int', [10, 14, 20]),
    'adx_threshold': ('int', [15, 20, 25, 30]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 21. CCI_ATR_Bands  (Pine v5 — CCI with ATR dynamic bands)
# CCI oscillator; entry when CCI crosses ±ATR-scaled thresholds
# Long: CCI crosses above lower ATR band
# Short: CCI crosses below upper ATR band
# ─────────────────────────────────────────────────────────────────────────────
def gen_CCI_ATR_Bands(df, cci_len=20, atr_len=14, atr_mult=1.0, **kw):
    cl = df['close']; hi = df['high']; lo = df['low']
    tp = (hi + lo + cl) / 3
    cci = (tp - _sma(tp, cci_len)) / (0.015 * tp.rolling(cci_len).std(ddof=0) + 1e-10)
    atr = _atr(df, atr_len)
    # ATR-based thresholds (normalized by price)
    upper_thresh =  100 + atr_mult * (atr / cl * 1000)
    lower_thresh = -100 - atr_mult * (atr / cl * 1000)
    long_entry  = _crossover(cci, lower_thresh)
    short_entry = _crossunder(cci, upper_thresh)
    long_exit   = cci > 0
    short_exit  = cci < 0
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_CCI_ATR_Bands = {
    'cci_len':  ('int',   [14, 20, 30]),
    'atr_len':  ('int',   [10, 14, 20]),
    'atr_mult': ('float', [0.5, 1.0, 1.5]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 22. RSI_Stochastic_Combo  (Pine v5 — RSI + StochRSI aligned)
# Long: RSI > 50 AND StochRSI K crosses D from below 20
# Short: RSI < 50 AND StochRSI K crosses D from above 80
# ─────────────────────────────────────────────────────────────────────────────
def gen_RSI_Stochastic_Combo(df, rsi_len=14, stoch_len=14,
                               k_smooth=3, d_smooth=3,
                               stoch_os=20, stoch_ob=80, **kw):
    cl = df['close']
    rsi = _rsi(cl, rsi_len)
    k, d = _stochrsi(cl, rsi_len, stoch_len, k_smooth, d_smooth)
    long_entry  = (rsi > 50) & _crossover(k, d) & (k.shift(1) < stoch_os)
    short_entry = (rsi < 50) & _crossunder(k, d) & (k.shift(1) > stoch_ob)
    long_exit   = (rsi < 50) | (k > stoch_ob)
    short_exit  = (rsi > 50) | (k < stoch_os)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_RSI_Stochastic_Combo = {
    'rsi_len':   ('int', [10, 14, 20]),
    'stoch_len': ('int', [10, 14, 20]),
    'k_smooth':  ('int', [2, 3, 5]),
    'stoch_os':  ('int', [15, 20, 25]),
    'stoch_ob':  ('int', [75, 80, 85]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 23. Pivot_Point_SuperTrend  (Pine v5 — lonesometheblue)
# Pivot-based SuperTrend: use pivot highs/lows instead of HL/2
# Long: price above pivot supertrend; Short: below
# ─────────────────────────────────────────────────────────────────────────────
def gen_Pivot_Point_SuperTrend(df, prd=2, factor=3.0, atr_len=14, **kw):
    cl = df['close']; hi = df['high']; lo = df['low']
    n = len(df)
    atr = _atr(df, atr_len)
    # Pivot high/low detection
    ph = pd.Series(np.nan, index=df.index)
    pl = pd.Series(np.nan, index=df.index)
    for i in range(prd, n - prd):
        window_h = hi.iloc[i - prd: i + prd + 1]
        if hi.iloc[i] == window_h.max():
            ph.iloc[i] = hi.iloc[i]
        window_l = lo.iloc[i - prd: i + prd + 1]
        if lo.iloc[i] == window_l.min():
            pl.iloc[i] = lo.iloc[i]
    # Forward-fill pivot levels
    ph_ff = ph.ffill()
    pl_ff = pl.ffill()
    # SuperTrend using pivot midpoint
    center = (ph_ff + pl_ff) / 2
    upper = center + factor * atr
    lower = center - factor * atr
    trend = pd.Series(1, index=df.index)
    for i in range(1, n):
        if cl.iloc[i] > upper.iloc[i-1]: trend.iloc[i] = 1
        elif cl.iloc[i] < lower.iloc[i-1]: trend.iloc[i] = -1
        else: trend.iloc[i] = trend.iloc[i-1]
    long_entry  = (trend == 1) & (trend.shift(1) == -1)
    short_entry = (trend == -1) & (trend.shift(1) == 1)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(n):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Pivot_Point_SuperTrend = {
    'prd':     ('int',   [1, 2, 3]),
    'factor':  ('float', [2.0, 3.0, 4.0]),
    'atr_len': ('int',   [10, 14, 20]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 24. Linear_Regression_Bands  (Pine v4 — linreg channel breakout)
# Channel = linreg ± n*stdev; signal on channel breakout
# Long: close breaks above upper band
# Short: close breaks below lower band
# ─────────────────────────────────────────────────────────────────────────────
def gen_Linear_Regression_Bands(df, length=50, stdev_mult=2.0, **kw):
    cl = df['close']
    lr = _linreg(cl, length)
    # Residual stdev
    def _resid_std(arr):
        if np.any(np.isnan(arr)): return np.nan
        x = np.arange(len(arr), dtype=float)
        m, b = np.polyfit(x, arr, 1)
        fitted = m * x + b
        return np.std(arr - fitted, ddof=0)
    rstd = cl.rolling(length).apply(_resid_std, raw=True)
    upper = lr + stdev_mult * rstd
    lower = lr - stdev_mult * rstd
    long_entry  = _crossover(cl, upper)
    short_entry = _crossunder(cl, lower)
    long_exit   = cl < lr
    short_exit  = cl > lr
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Linear_Regression_Bands = {
    'length':     ('int',   [30, 50, 80]),
    'stdev_mult': ('float', [1.5, 2.0, 2.5, 3.0]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 25. SSL_Hybrid  (Pine v5 — SSL Channel with Baseline)
# SSL Channel: SMA(high, n) vs SMA(low, n) direction
# Baseline: EMA or VWAP; confirmation required
# ─────────────────────────────────────────────────────────────────────────────
def gen_SSL_Hybrid(df, ssl_len=10, base_len=30, **kw):
    cl = df['close']; hi = df['high']; lo = df['low']
    vol = df['volume']
    # SSL Channel
    hma_ssl = _sma(hi, ssl_len)
    lma_ssl = _sma(lo, ssl_len)
    ssl_dir = pd.Series(0, index=df.index)
    for i in range(ssl_len, len(df)):
        if cl.iloc[i] > hma_ssl.iloc[i]:
            ssl_dir.iloc[i] = 1
        elif cl.iloc[i] < lma_ssl.iloc[i]:
            ssl_dir.iloc[i] = -1
        else:
            ssl_dir.iloc[i] = ssl_dir.iloc[i-1]
    # Baseline: volume-weighted SMA
    tp = (hi + lo + cl) / 3
    baseline = (tp * vol).rolling(base_len).sum() / (vol.rolling(base_len).sum() + 1e-10)
    above_base = cl > baseline
    below_base = cl < baseline
    long_entry  = (ssl_dir == 1) & (ssl_dir.shift(1) != 1) & above_base
    short_entry = (ssl_dir == -1) & (ssl_dir.shift(1) != -1) & below_base
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if ssl_dir.iloc[i] == -1: in_pos = 0
        elif in_pos == -1:
            if ssl_dir.iloc[i] == 1: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_SSL_Hybrid = {
    'ssl_len':  ('int', [5, 10, 14, 21]),
    'base_len': ('int', [20, 30, 50]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 26. Chande_Kroll_Stop  (Pine v4 — Chande Kroll bidirectional signal)
# stop_short and stop_long lines; signal on price crossing them
# Long: price crosses above stop_short
# Short: price crosses below stop_long
# ─────────────────────────────────────────────────────────────────────────────
def gen_Chande_Kroll_Stop(df, p=10, q=9, x=1.5, **kw):
    cl = df['close']; hi = df['high']; lo = df['low']
    atr = _atr(df, p)
    # First stop lines
    first_high_stop = hi.rolling(p).max() - x * atr
    first_low_stop  = lo.rolling(p).min()  + x * atr
    # Final stop lines (rolling max/min over q periods)
    stop_short = first_high_stop.rolling(q).max()
    stop_long  = first_low_stop.rolling(q).min()
    long_entry  = _crossover(cl, stop_short)
    short_entry = _crossunder(cl, stop_long)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Chande_Kroll_Stop = {
    'p': ('int',   [7, 10, 14]),
    'q': ('int',   [7, 9, 14]),
    'x': ('float', [1.0, 1.5, 2.0]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 27. T3_Smooth_MA_Cross  (Pine v4 — Tillson T3 moving average crossover)
# T3 = triple-smoothed EMA with Tillson's volume factor
# Long: T3 fast crosses above T3 slow
# Short: T3 fast crosses below T3 slow
# ─────────────────────────────────────────────────────────────────────────────
def _t3(s, n, v_factor=0.7):
    c1 = -(v_factor ** 3)
    c2 = 3 * v_factor ** 2 + 3 * v_factor ** 3
    c3 = -6 * v_factor ** 2 - 3 * v_factor - 3 * v_factor ** 3
    c4 = 1 + 3 * v_factor + v_factor ** 3 + 3 * v_factor ** 2
    e1 = _ema(s, n); e2 = _ema(e1, n); e3 = _ema(e2, n)
    e4 = _ema(e3, n); e5 = _ema(e4, n); e6 = _ema(e5, n)
    return c1*e6 + c2*e5 + c3*e4 + c4*e3

def gen_T3_Smooth_MA_Cross(df, fast_len=8, slow_len=21, v_factor=0.7, **kw):
    cl = df['close']
    t3_fast = _t3(cl, fast_len, v_factor)
    t3_slow = _t3(cl, slow_len, v_factor)
    long_entry  = _crossover(t3_fast, t3_slow)
    short_entry = _crossunder(t3_fast, t3_slow)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_T3_Smooth_MA_Cross = {
    'fast_len': ('int',   [5, 8, 13]),
    'slow_len': ('int',   [16, 21, 34]),
    'v_factor': ('float', [0.5, 0.7, 0.9]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 28. Gann_HiLo_Activator  (Pine v4 — Gann High-Low trailing signal)
# SMA of highs vs SMA of lows; activator = one of them based on close direction
# Long: close above activator AND activator switches to low-SMA
# Short: close below activator AND activator switches to high-SMA
# ─────────────────────────────────────────────────────────────────────────────
def gen_Gann_HiLo_Activator(df, gann_len=3, **kw):
    cl = df['close']; hi = df['high']; lo = df['low']
    hi_ma = _sma(hi, gann_len)
    lo_ma = _sma(lo, gann_len)
    activator = pd.Series(0.0, index=df.index)
    for i in range(gann_len, len(df)):
        if cl.iloc[i] > hi_ma.iloc[i-1]:
            activator.iloc[i] = lo_ma.iloc[i]
        elif cl.iloc[i] < lo_ma.iloc[i-1]:
            activator.iloc[i] = hi_ma.iloc[i]
        else:
            activator.iloc[i] = activator.iloc[i-1]
    # Signal: close crosses the activator
    long_entry  = _crossover(cl, activator)
    short_entry = _crossunder(cl, activator)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Gann_HiLo_Activator = {
    'gann_len': ('int', [2, 3, 5, 8]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 29. Holt_Winters_DEMA  (Pine v5 — Double Exponential Smoothing trend)
# DES (Holt-Winters double exponential) extrapolates trend
# Long: forecast crosses above current price from below
# Short: forecast crosses below current price from above
# ─────────────────────────────────────────────────────────────────────────────
def gen_Holt_Winters_DEMA(df, alpha=0.1, beta=0.1, **kw):
    cl = df['close']
    n = len(cl)
    level  = pd.Series(0.0, index=df.index)
    trend_ = pd.Series(0.0, index=df.index)
    forecast = pd.Series(np.nan, index=df.index)
    level.iloc[0] = cl.iloc[0]
    if n > 1:
        trend_.iloc[0] = cl.iloc[1] - cl.iloc[0]
    for i in range(1, n):
        l_prev = level.iloc[i-1]; t_prev = trend_.iloc[i-1]
        c = cl.iloc[i]
        l_new = alpha * c + (1 - alpha) * (l_prev + t_prev)
        t_new = beta * (l_new - l_prev) + (1 - beta) * t_prev
        level.iloc[i] = l_new; trend_.iloc[i] = t_new
        forecast.iloc[i] = l_new + t_new
    long_entry  = _crossover(forecast, cl)
    short_entry = _crossunder(forecast, cl)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(n):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if short_entry.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if long_entry.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Holt_Winters_DEMA = {
    'alpha': ('float', [0.05, 0.10, 0.20, 0.30]),
    'beta':  ('float', [0.05, 0.10, 0.20]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 30. CCI_MTF  (Pine v5 — CCI on higher timeframe simulated)
# MTF CCI: resample CCI using a multiplier of the current TF
# Long: MTF-CCI crosses above -100 (oversold recovery)
# Short: MTF-CCI crosses below +100 (overbought collapse)
# ─────────────────────────────────────────────────────────────────────────────
def gen_CCI_MTF(df, cci_len=20, mtf_factor=4, **kw):
    cl = df['close']; hi = df['high']; lo = df['low']
    # MTF simulation: use longer period = cci_len * mtf_factor
    effective_len = cci_len * mtf_factor
    tp = (hi + lo + cl) / 3
    cci_mtf = (tp - _sma(tp, effective_len)) / (
        0.015 * tp.rolling(effective_len).std(ddof=0) + 1e-10
    )
    ob = pd.Series(100.0, index=df.index)
    os_ = pd.Series(-100.0, index=df.index)
    long_entry  = _crossover(cci_mtf, os_)
    short_entry = _crossunder(cci_mtf, ob)
    long_exit   = cci_mtf > ob
    short_exit  = cci_mtf < os_
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_CCI_MTF = {
    'cci_len':    ('int', [14, 20, 30]),
    'mtf_factor': ('int', [2, 3, 4, 6]),
}

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY EXPORT
# ─────────────────────────────────────────────────────────────────────────────
STRATEGY_EXPORT = {
    'Nadaraya_Watson':          {'gen': gen_Nadaraya_Watson,          'space': space_Nadaraya_Watson},
    'UT_Bot_Alerts':            {'gen': gen_UT_Bot_Alerts,            'space': space_UT_Bot_Alerts},
    'Lux_Algo_Signals':         {'gen': gen_Lux_Algo_Signals,         'space': space_Lux_Algo_Signals},
    'Godmode_Oscillator':       {'gen': gen_Godmode_Oscillator,       'space': space_Godmode_Oscillator},
    'Squeeze_Pro':              {'gen': gen_Squeeze_Pro,              'space': space_Squeeze_Pro},
    'VWAP_Stdev_Bands':         {'gen': gen_VWAP_Stdev_Bands,         'space': space_VWAP_Stdev_Bands},
    'Half_Trend':               {'gen': gen_Half_Trend,               'space': space_Half_Trend},
    'Wave_Trend_Divergence':    {'gen': gen_Wave_Trend_Divergence,    'space': space_Wave_Trend_Divergence},
    'MOST_Strategy':            {'gen': gen_MOST_Strategy,            'space': space_MOST_Strategy},
    'Coral_Trend':              {'gen': gen_Coral_Trend,              'space': space_Coral_Trend},
    'Andean_Oscillator':        {'gen': gen_Andean_Oscillator,        'space': space_Andean_Oscillator},
    'ZLSMA_UT_Bot':             {'gen': gen_ZLSMA_UT_Bot,             'space': space_ZLSMA_UT_Bot},
    'Fancy_RSI':                {'gen': gen_Fancy_RSI,                'space': space_Fancy_RSI},
    'MACD_Hist_Divergence':     {'gen': gen_MACD_Hist_Divergence,     'space': space_MACD_Hist_Divergence},
    'Stochastic_Divergence':    {'gen': gen_Stochastic_Divergence,    'space': space_Stochastic_Divergence},
    'Money_Flow_RSI':           {'gen': gen_Money_Flow_RSI,           'space': space_Money_Flow_RSI},
    'Elder_Impulse_v2':         {'gen': gen_Elder_Impulse_v2,         'space': space_Elder_Impulse_v2},
    'Parabolic_SAR_EMA':        {'gen': gen_Parabolic_SAR_EMA,        'space': space_Parabolic_SAR_EMA},
    'SuperTrend_MACD':          {'gen': gen_SuperTrend_MACD,          'space': space_SuperTrend_MACD},
    'ADX_DMI_Signals':          {'gen': gen_ADX_DMI_Signals,          'space': space_ADX_DMI_Signals},
    'CCI_ATR_Bands':            {'gen': gen_CCI_ATR_Bands,            'space': space_CCI_ATR_Bands},
    'RSI_Stochastic_Combo':     {'gen': gen_RSI_Stochastic_Combo,     'space': space_RSI_Stochastic_Combo},
    'Pivot_Point_SuperTrend':   {'gen': gen_Pivot_Point_SuperTrend,   'space': space_Pivot_Point_SuperTrend},
    'Linear_Regression_Bands':  {'gen': gen_Linear_Regression_Bands,  'space': space_Linear_Regression_Bands},
    'SSL_Hybrid':               {'gen': gen_SSL_Hybrid,               'space': space_SSL_Hybrid},
    'Chande_Kroll_Stop':        {'gen': gen_Chande_Kroll_Stop,        'space': space_Chande_Kroll_Stop},
    'T3_Smooth_MA_Cross':       {'gen': gen_T3_Smooth_MA_Cross,       'space': space_T3_Smooth_MA_Cross},
    'Gann_HiLo_Activator':      {'gen': gen_Gann_HiLo_Activator,      'space': space_Gann_HiLo_Activator},
    'Holt_Winters_DEMA':        {'gen': gen_Holt_Winters_DEMA,        'space': space_Holt_Winters_DEMA},
    'CCI_MTF':                  {'gen': gen_CCI_MTF,                  'space': space_CCI_MTF},
}
