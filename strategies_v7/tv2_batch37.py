#!/usr/bin/env python3
"""TV2 BATCH 37 — MTF Confluence + Adaptive Systems (30 strategies) 2026-04-01"""

import numpy as np
import pandas as pd

# ── helpers ───────────────────────────────────────────────────────────────────

def _ema(s, n): return s.ewm(span=int(n), adjust=False).mean()
def _sma(s, n): return s.rolling(int(n), min_periods=1).mean()
def _rma(s, n): return s.ewm(alpha=1/int(n), adjust=False).mean()

def _atr(df, n):
    hi, lo, cl = df['high'], df['low'], df['close']
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    return _rma(tr, n)

def _rsi(s, n):
    d = s.diff()
    return 100 - 100/(1 + _rma(d.clip(lower=0), n) / _rma((-d).clip(lower=0), n).replace(0, 1e-9))

def _wma(s, n):
    n = int(n)
    w = np.arange(1, n+1)
    return s.rolling(n).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)

def _hma(s, n):
    n = int(n)
    return _wma(2*_wma(s, n//2) - _wma(s, n), int(np.sqrt(n)))

def _stoch(df, k_period, d_period):
    lo = df['low'].rolling(int(k_period), min_periods=1).min()
    hi = df['high'].rolling(int(k_period), min_periods=1).max()
    k  = 100 * (df['close'] - lo) / (hi - lo + 1e-9)
    d  = _sma(k, d_period)
    return k, d

def _macd(s, fast, slow, sig):
    m = _ema(s, fast) - _ema(s, slow)
    signal = _ema(m, sig)
    return m, signal, m - signal

def _adx(df, n):
    n = int(n)
    hi, lo, cl = df['high'], df['low'], df['close']
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    dmp = (hi - hi.shift()).clip(lower=0)
    dmm = (lo.shift() - lo).clip(lower=0)
    dmp = dmp.where(dmp > dmm, 0)
    dmm = dmm.where(dmm > dmp, 0)
    atr14 = _rma(tr, n)
    pdi = 100 * _rma(dmp, n) / atr14.replace(0, 1e-9)
    mdi = 100 * _rma(dmm, n) / atr14.replace(0, 1e-9)
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    return _rma(dx, n), pdi, mdi

def _obv(df):
    direction = np.sign(df['close'].diff())
    direction.iloc[0] = 0
    return (df['volume'] * direction).cumsum()

def _bb(s, n, mult):
    mid = _sma(s, n)
    std = s.rolling(int(n), min_periods=1).std().fillna(0)
    return mid + mult*std, mid, mid - mult*std

def _cci(df, n):
    tp  = (df['high'] + df['low'] + df['close']) / 3
    sma = _sma(tp, n)
    mad = tp.rolling(int(n), min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, 1e-9))

def _vwap(df):
    tp  = (df['high'] + df['low'] + df['close']) / 3
    cum_tpv = (tp * df['volume']).cumsum()
    cum_vol = df['volume'].cumsum()
    return cum_tpv / cum_vol.replace(0, 1e-9)

def _donchian(df, n):
    hi = df['high'].rolling(int(n), min_periods=1).max()
    lo = df['low'].rolling(int(n), min_periods=1).min()
    return hi, (hi+lo)/2, lo

def _chop(df, n):
    n = int(n)
    atr_sum = _atr(df, 1).rolling(n, min_periods=1).sum()
    hi = df['high'].rolling(n, min_periods=1).max()
    lo = df['low'].rolling(n, min_periods=1).min()
    return 100 * np.log10(atr_sum / (hi - lo + 1e-9)) / np.log10(n)

def _keltner(df, n, mult):
    mid   = _ema(df['close'], n)
    atr   = _atr(df, n)
    return mid + mult*atr, mid, mid - mult*atr

def _pivot_highs(s, left, right):
    """Returns True where bar is a pivot high (strict local max)."""
    left, right = int(left), int(right)
    result = pd.Series(False, index=s.index)
    for i in range(left, len(s)-right):
        window = s.iloc[i-left:i+right+1]
        if s.iloc[i] == window.max() and (window < s.iloc[i]).sum() == len(window)-1:
            result.iloc[i] = True
    return result

def _pivot_lows(s, left, right):
    """Returns True where bar is a pivot low (strict local min)."""
    left, right = int(left), int(right)
    result = pd.Series(False, index=s.index)
    for i in range(left, len(s)-right):
        window = s.iloc[i-left:i+right+1]
        if s.iloc[i] == window.min() and (window > s.iloc[i]).sum() == len(window)-1:
            result.iloc[i] = True
    return result

def _force_index(df, n):
    fi = df['close'].diff() * df['volume']
    return _ema(fi, n)

def _supertrend(df, n, mult):
    atr   = _atr(df, n)
    hl2   = (df['high'] + df['low']) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    cl    = df['close']
    trend = pd.Series(1, index=df.index)
    u_band = upper.copy()
    l_band = lower.copy()
    for i in range(1, len(df)):
        u_band.iloc[i] = min(upper.iloc[i], u_band.iloc[i-1]) if cl.iloc[i-1] > u_band.iloc[i-1] else upper.iloc[i]
        l_band.iloc[i] = max(lower.iloc[i], l_band.iloc[i-1]) if cl.iloc[i-1] < l_band.iloc[i-1] else lower.iloc[i]
        if trend.iloc[i-1] == -1 and cl.iloc[i] > u_band.iloc[i]:
            trend.iloc[i] = 1
        elif trend.iloc[i-1] == 1 and cl.iloc[i] < l_band.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    return trend

# ─────────────────────────────────────────────────────────────────────────────
# 1. MTF_EMA_Confluence
# EMA trend on current TF confirmed by simulated higher TF (4x period)
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_EMA_Confluence(df, fast=10, slow=30, htf_mult=4):
    cl     = df['close']
    f_ema  = _ema(cl, fast)
    s_ema  = _ema(cl, slow)
    # Higher TF simulation: use 4x periods
    f_htf  = _ema(cl, int(fast * htf_mult))
    s_htf  = _ema(cl, int(slow * htf_mult))
    bull   = (f_ema > s_ema) & (f_htf > s_htf)
    bear   = (f_ema < s_ema) & (f_htf < s_htf)
    sig    = pd.Series(0, index=df.index)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_MTF_EMA_Confluence = {
    'fast':     ('int', 5, 25),
    'slow':     ('int', 20, 60),
    'htf_mult': ('int', 2, 6),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. MTF_RSI_Confluence
# RSI oversold/overbought on 2 timeframe simulations
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_RSI_Confluence(df, rsi_len=14, htf_mult=4, ob=65, os=35):
    cl      = df['close']
    rsi_ltf = _rsi(cl, rsi_len)
    rsi_htf = _rsi(cl, int(rsi_len * htf_mult))
    sig     = pd.Series(0, index=df.index)
    bull    = (rsi_ltf < os) & (rsi_htf < os + 5)
    bear    = (rsi_ltf > ob) & (rsi_htf > ob - 5)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_MTF_RSI_Confluence = {
    'rsi_len':  ('int', 7, 21),
    'htf_mult': ('int', 2, 6),
    'ob':       ('int', 60, 75),
    'os':       ('int', 25, 40),
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. MTF_SuperTrend_Confluence
# SuperTrend alignment across 2 simulated TFs
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_SuperTrend_Confluence(df, atr_len=10, mult=3.0, htf_mult=4):
    trend_ltf = _supertrend(df, atr_len, mult)
    trend_htf = _supertrend(df, int(atr_len * htf_mult), mult)
    sig       = pd.Series(0, index=df.index)
    bull      = (trend_ltf == 1) & (trend_htf == 1)
    bear      = (trend_ltf == -1) & (trend_htf == -1)
    cross_up  = bull & ~bull.shift(1).fillna(False)
    cross_dn  = bear & ~bear.shift(1).fillna(False)
    sig[cross_up] = 1
    sig[cross_dn] = -1
    return sig

space_MTF_SuperTrend_Confluence = {
    'atr_len':  ('int', 7, 20),
    'mult':     ('float', 1.5, 5.0),
    'htf_mult': ('int', 2, 6),
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. MTF_MACD_Alignment
# MACD histogram same direction on fast+slow simulation
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_MACD_Alignment(df, fast=12, slow=26, sig_len=9, htf_mult=3):
    cl           = df['close']
    _, _, hist   = _macd(cl, fast, slow, sig_len)
    _, _, hist_h = _macd(cl, int(fast*htf_mult), int(slow*htf_mult), int(sig_len*htf_mult))
    sig          = pd.Series(0, index=df.index)
    bull = (hist > 0) & (hist_h > 0)
    bear = (hist < 0) & (hist_h < 0)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_MTF_MACD_Alignment = {
    'fast':     ('int', 8, 16),
    'slow':     ('int', 20, 34),
    'sig_len':  ('int', 7, 12),
    'htf_mult': ('int', 2, 5),
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. MTF_Stoch_Sync
# Stochastic K line in sync direction across 2 simulated TFs
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_Stoch_Sync(df, k_len=14, d_len=3, htf_mult=4, ob=70, os=30):
    k_ltf, d_ltf = _stoch(df, k_len, d_len)
    # HTF simulated via wider window
    k_htf, d_htf = _stoch(df, int(k_len * htf_mult), int(d_len * htf_mult))
    sig          = pd.Series(0, index=df.index)
    bull = (k_ltf < os) & (k_htf < os + 10) & (k_ltf > k_ltf.shift(1))
    bear = (k_ltf > ob) & (k_htf > ob - 10) & (k_ltf < k_ltf.shift(1))
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_MTF_Stoch_Sync = {
    'k_len':    ('int', 9, 21),
    'd_len':    ('int', 2, 5),
    'htf_mult': ('int', 2, 6),
    'ob':       ('int', 65, 80),
    'os':       ('int', 20, 35),
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. Adaptive_RSI_BB
# RSI with Bollinger Bands dynamically adjusted by ATR
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_RSI_BB(df, rsi_len=14, bb_len=20, base_mult=2.0, atr_len=14):
    cl     = df['close']
    rsi    = _rsi(cl, rsi_len)
    atr    = _atr(df, atr_len)
    # Normalize ATR: high vol → wider RSI bands
    atr_n  = atr / cl
    mult   = base_mult * (1 + atr_n / atr_n.rolling(50, min_periods=1).mean().replace(0, 1e-9))
    mid    = _sma(rsi, bb_len)
    std    = rsi.rolling(int(bb_len), min_periods=1).std().fillna(0)
    upper  = mid + mult * std
    lower  = mid - mult * std
    sig    = pd.Series(0, index=df.index)
    bull   = rsi < lower
    bear   = rsi > upper
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_Adaptive_RSI_BB = {
    'rsi_len':   ('int', 7, 21),
    'bb_len':    ('int', 15, 30),
    'base_mult': ('float', 1.0, 3.0),
    'atr_len':   ('int', 7, 21),
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. Adaptive_EMA_Vol
# EMA period adapted by volatility (ATR ratio)
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_EMA_Vol(df, base_fast=10, base_slow=30, atr_len=14, atr_mult=1.0):
    cl      = df['close']
    atr     = _atr(df, atr_len)
    atr_ma  = _sma(atr, 50)
    # In high volatility, use longer periods (slower to avoid whipsaws)
    vol_r   = (atr / atr_ma.replace(0, 1e-9)).clip(0.5, 3.0)
    # Compute adaptive EMAs bar by bar using rolling span
    fast_ema = _ema(cl, int(base_fast))
    slow_ema = _ema(cl, int(base_slow))
    # Scale: when vol_r > 1 → slow down
    fast_ema_hi = _ema(cl, int(base_fast * 2))
    slow_ema_hi = _ema(cl, int(base_slow * 2))
    high_vol = vol_r > (1.0 + atr_mult * 0.3)
    f_ema = fast_ema_hi.where(high_vol, fast_ema)
    s_ema = slow_ema_hi.where(high_vol, slow_ema)
    sig  = pd.Series(0, index=df.index)
    bull = f_ema > s_ema
    bear = f_ema < s_ema
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_Adaptive_EMA_Vol = {
    'base_fast': ('int', 5, 20),
    'base_slow': ('int', 20, 60),
    'atr_len':   ('int', 7, 21),
    'atr_mult':  ('float', 0.5, 2.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# 8. Adaptive_MA_Crossover
# Fast/slow MA crossover where periods adapt to ADX
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_MA_Crossover(df, base_fast=10, base_slow=30, adx_len=14, adx_thresh=25):
    cl        = df['close']
    adx, _, _ = _adx(df, adx_len)
    # Strong trend (ADX > thresh) → use shorter periods for responsiveness
    fast_tr   = _ema(cl, max(int(base_fast * 0.6), 3))
    slow_tr   = _ema(cl, max(int(base_slow * 0.6), 5))
    fast_rng  = _ema(cl, base_fast)
    slow_rng  = _ema(cl, base_slow)
    trending  = adx > adx_thresh
    f_ema = fast_tr.where(trending, fast_rng)
    s_ema = slow_tr.where(trending, slow_rng)
    sig   = pd.Series(0, index=df.index)
    bull  = f_ema > s_ema
    bear  = f_ema < s_ema
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_Adaptive_MA_Crossover = {
    'base_fast':  ('int', 5, 20),
    'base_slow':  ('int', 20, 60),
    'adx_len':    ('int', 7, 21),
    'adx_thresh': ('int', 15, 40),
}

# ─────────────────────────────────────────────────────────────────────────────
# 9. Adaptive_Momentum_ATR
# Momentum signal scaled by ATR regime
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_Momentum_ATR(df, mom_len=14, atr_len=14, thresh_mult=1.0):
    cl      = df['close']
    mom     = cl - cl.shift(mom_len)
    atr     = _atr(df, atr_len)
    # Threshold adapts to ATR: signal only when momentum > thresh_mult × ATR
    thresh  = thresh_mult * atr
    sig     = pd.Series(0, index=df.index)
    bull    = mom >  thresh
    bear    = mom < -thresh
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_Adaptive_Momentum_ATR = {
    'mom_len':    ('int', 7, 25),
    'atr_len':    ('int', 7, 21),
    'thresh_mult': ('float', 0.5, 3.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# 10. Adaptive_Trend_Strength
# ADX-weighted trend with adaptive thresholds
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_Trend_Strength(df, adx_len=14, ema_len=20, min_adx=20, max_adx=50):
    cl        = df['close']
    adx, pdi, mdi = _adx(df, adx_len)
    ema       = _ema(cl, ema_len)
    # Adaptive threshold: stronger trend required in choppy markets
    adx_n     = (adx - min_adx) / (max_adx - min_adx + 1e-9)
    adx_n     = adx_n.clip(0, 1)
    sig       = pd.Series(0, index=df.index)
    # Bull: price > EMA + pdi > mdi + adx confirms trend
    bull = (cl > ema) & (pdi > mdi) & (adx > min_adx)
    bear = (cl < ema) & (mdi > pdi) & (adx > min_adx)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_Adaptive_Trend_Strength = {
    'adx_len':  ('int', 7, 21),
    'ema_len':  ('int', 10, 50),
    'min_adx':  ('int', 15, 35),
    'max_adx':  ('int', 35, 60),
}

# ─────────────────────────────────────────────────────────────────────────────
# 11. MTF_BB_Squeeze_Confirm
# BB squeeze on current TF confirmed by higher TF trend
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_BB_Squeeze_Confirm(df, bb_len=20, bb_mult=2.0, kc_mult=1.5, htf_mult=4):
    cl         = df['close']
    bb_up, bb_mid, bb_lo = _bb(cl, bb_len, bb_mult)
    kc_up, kc_mid, kc_lo = _keltner(df, bb_len, kc_mult)
    squeeze    = (bb_up < kc_up) & (bb_lo > kc_lo)
    # HTF EMA as trend filter
    htf_ema    = _ema(cl, int(bb_len * htf_mult))
    # Momentum: when squeeze releases, direction of breakout
    delta      = cl - cl.shift(1)
    sig        = pd.Series(0, index=df.index)
    sq_release = ~squeeze & squeeze.shift(1).fillna(False)
    bull       = sq_release & (cl > htf_ema) & (delta > 0)
    bear       = sq_release & (cl < htf_ema) & (delta < 0)
    sig[bull]  = 1
    sig[bear]  = -1
    return sig

space_MTF_BB_Squeeze_Confirm = {
    'bb_len':   ('int', 15, 30),
    'bb_mult':  ('float', 1.5, 2.5),
    'kc_mult':  ('float', 1.0, 2.0),
    'htf_mult': ('int', 2, 6),
}

# ─────────────────────────────────────────────────────────────────────────────
# 12. MTF_Volume_Trend
# OBV trend alignment across simulated MTF
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_Volume_Trend(df, obv_len=20, htf_mult=4):
    obv       = _obv(df)
    obv_sma   = _sma(obv, obv_len)
    obv_htf   = _sma(obv, int(obv_len * htf_mult))
    sig       = pd.Series(0, index=df.index)
    bull = (obv > obv_sma) & (obv > obv_htf)
    bear = (obv < obv_sma) & (obv < obv_htf)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_MTF_Volume_Trend = {
    'obv_len':  ('int', 10, 40),
    'htf_mult': ('int', 2, 6),
}

# ─────────────────────────────────────────────────────────────────────────────
# 13. MTF_Ichimoku_Cloud
# Cloud direction on current TF aligned with Tenkan/Kijun on 3x period
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_Ichimoku_Cloud(df, tenkan=9, kijun=26, senkou_b=52, htf_mult=3):
    hi, lo, cl = df['high'], df['low'], df['close']
    def ichi_lines(t, k, s):
        ten = (hi.rolling(int(t), min_periods=1).max() + lo.rolling(int(t), min_periods=1).min()) / 2
        kij = (hi.rolling(int(k), min_periods=1).max() + lo.rolling(int(k), min_periods=1).min()) / 2
        spa = ((ten + kij) / 2).shift(int(k))
        spb = ((hi.rolling(int(s), min_periods=1).max() + lo.rolling(int(s), min_periods=1).min()) / 2).shift(int(k))
        return ten, kij, spa, spb
    ten, kij, spa, spb = ichi_lines(tenkan, kijun, senkou_b)
    # HTF simulation: 3x periods
    ten_h, kij_h, _, _ = ichi_lines(int(tenkan*htf_mult), int(kijun*htf_mult), int(senkou_b*htf_mult))
    cloud_bull = (spa > spb) & (cl > spa) & (cl > spb)
    cloud_bear = (spb > spa) & (cl < spa) & (cl < spb)
    htf_bull   = ten_h > kij_h
    htf_bear   = ten_h < kij_h
    sig        = pd.Series(0, index=df.index)
    bull       = cloud_bull & htf_bull & (ten > kij)
    bear       = cloud_bear & htf_bear & (ten < kij)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_MTF_Ichimoku_Cloud = {
    'tenkan':   ('int', 7, 14),
    'kijun':    ('int', 20, 35),
    'senkou_b': ('int', 40, 65),
    'htf_mult': ('int', 2, 4),
}

# ─────────────────────────────────────────────────────────────────────────────
# 14. Adaptive_Chandelier
# Chandelier Exit with ATR multiplier adapted by volatility regime
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_Chandelier(df, atr_len=22, base_mult=3.0, vol_window=50):
    cl    = df['close']
    hi    = df['high']
    lo    = df['low']
    atr   = _atr(df, atr_len)
    # Adapt multiplier: higher in high-vol regimes
    atr_ma   = _sma(atr, vol_window)
    vol_r    = (atr / atr_ma.replace(0, 1e-9)).clip(0.5, 2.5)
    mult     = base_mult * vol_r
    # Chandelier exits
    highest  = hi.rolling(int(atr_len), min_periods=1).max()
    lowest   = lo.rolling(int(atr_len), min_periods=1).min()
    long_stop  = highest - mult * atr
    short_stop = lowest  + mult * atr
    trend    = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if trend.iloc[i-1] == 1:
            trend.iloc[i] = -1 if cl.iloc[i] < long_stop.iloc[i] else 1
        else:
            trend.iloc[i] = 1 if cl.iloc[i] > short_stop.iloc[i] else -1
    sig = pd.Series(0, index=df.index)
    sig[trend == 1] = 0  # just track cross events
    cross_up = (trend == 1) & (trend.shift(1) == -1)
    cross_dn = (trend == -1) & (trend.shift(1) == 1)
    sig[cross_up] = 1
    sig[cross_dn] = -1
    return sig

space_Adaptive_Chandelier = {
    'atr_len':    ('int', 14, 30),
    'base_mult':  ('float', 1.5, 4.5),
    'vol_window': ('int', 30, 80),
}

# ─────────────────────────────────────────────────────────────────────────────
# 15. MTF_ADX_Filter
# ADX trend filter applied to RSI signal from lower simulation
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_ADX_Filter(df, rsi_len=14, adx_len=14, adx_thresh=25, ob=65, os=35):
    cl         = df['close']
    rsi        = _rsi(cl, rsi_len)
    adx, _, _  = _adx(df, adx_len)
    trending   = adx > adx_thresh
    sig        = pd.Series(0, index=df.index)
    bull       = (rsi < os) & trending
    bear       = (rsi > ob) & trending
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_MTF_ADX_Filter = {
    'rsi_len':    ('int', 7, 21),
    'adx_len':    ('int', 7, 21),
    'adx_thresh': ('int', 15, 40),
    'ob':         ('int', 60, 75),
    'os':         ('int', 25, 40),
}

# ─────────────────────────────────────────────────────────────────────────────
# 16. Adaptive_Stoch_RSI
# Stoch RSI with adaptive overbought/oversold based on volatility
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_Stoch_RSI(df, rsi_len=14, stoch_len=14, d_len=3, atr_len=14, base_ob=80, base_os=20):
    cl       = df['close']
    rsi      = _rsi(cl, rsi_len)
    rsi_lo   = rsi.rolling(int(stoch_len), min_periods=1).min()
    rsi_hi   = rsi.rolling(int(stoch_len), min_periods=1).max()
    k        = 100 * (rsi - rsi_lo) / (rsi_hi - rsi_lo + 1e-9)
    d        = _sma(k, d_len)
    atr      = _atr(df, atr_len)
    atr_n    = (atr / cl).rolling(50, min_periods=1).mean()
    # High volatility → tighten thresholds
    vol_adj  = (atr / cl / atr_n.replace(0, 1e-9)).clip(0.5, 2.0)
    ob       = (base_ob / vol_adj).clip(60, 95)
    os       = (base_os * vol_adj).clip(5, 40)
    sig      = pd.Series(0, index=df.index)
    bull     = (k > d) & (k.shift(1) <= d.shift(1)) & (k < os)
    bear     = (k < d) & (k.shift(1) >= d.shift(1)) & (k > ob)
    sig[bull] = 1
    sig[bear] = -1
    return sig

space_Adaptive_Stoch_RSI = {
    'rsi_len':   ('int', 7, 21),
    'stoch_len': ('int', 10, 20),
    'd_len':     ('int', 2, 5),
    'atr_len':   ('int', 7, 21),
    'base_ob':   ('int', 70, 90),
    'base_os':   ('int', 10, 30),
}

# ─────────────────────────────────────────────────────────────────────────────
# 17. MTF_VWAP_Trend
# VWAP slope direction confirmed by EMA trend on 4x period sim
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_VWAP_Trend(df, vwap_smooth=20, htf_ema_mult=4, ema_base=20):
    cl        = df['close']
    vwap      = _vwap(df)
    vwap_sma  = _sma(vwap, vwap_smooth)
    htf_ema   = _ema(cl, int(ema_base * htf_ema_mult))
    sig       = pd.Series(0, index=df.index)
    vwap_up   = vwap_sma > vwap_sma.shift(1)
    vwap_dn   = vwap_sma < vwap_sma.shift(1)
    bull      = (cl > vwap) & vwap_up & (cl > htf_ema)
    bear      = (cl < vwap) & vwap_dn & (cl < htf_ema)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_MTF_VWAP_Trend = {
    'vwap_smooth':   ('int', 10, 40),
    'htf_ema_mult':  ('int', 2, 6),
    'ema_base':      ('int', 10, 40),
}

# ─────────────────────────────────────────────────────────────────────────────
# 18. Adaptive_ATR_Channel
# Price channel width adapts to recent ATR expansion
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_ATR_Channel(df, atr_len=14, ch_len=20, mult=2.0, vol_window=50):
    cl      = df['close']
    atr     = _atr(df, atr_len)
    atr_ma  = _sma(atr, vol_window)
    # In ATR expansion, use wider channel
    expansion = (atr > atr_ma * 1.2)
    eff_mult  = mult * 1.5
    mid       = _sma(cl, ch_len)
    upper_std = mid + mult     * atr
    lower_std = mid - mult     * atr
    upper_exp = mid + eff_mult * atr
    lower_exp = mid - eff_mult * atr
    upper     = upper_exp.where(expansion, upper_std)
    lower     = lower_exp.where(expansion, lower_std)
    sig       = pd.Series(0, index=df.index)
    bull      = cl < lower
    bear      = cl > upper
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_Adaptive_ATR_Channel = {
    'atr_len':    ('int', 7, 21),
    'ch_len':     ('int', 10, 40),
    'mult':       ('float', 1.0, 3.0),
    'vol_window': ('int', 30, 80),
}

# ─────────────────────────────────────────────────────────────────────────────
# 19. MTF_Hull_Trend
# HMA direction on current TF plus slower HMA confirmation
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_Hull_Trend(df, hma_fast=14, hma_slow=50):
    cl       = df['close']
    hma_f    = _hma(cl, hma_fast)
    hma_s    = _hma(cl, hma_slow)
    bull     = (hma_f > hma_f.shift(1)) & (hma_s > hma_s.shift(1))
    bear     = (hma_f < hma_f.shift(1)) & (hma_s < hma_s.shift(1))
    sig      = pd.Series(0, index=df.index)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_MTF_Hull_Trend = {
    'hma_fast': ('int', 7, 25),
    'hma_slow': ('int', 30, 80),
}

# ─────────────────────────────────────────────────────────────────────────────
# 20. Adaptive_CCI_ATR
# CCI with ATR-normalized thresholds
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_CCI_ATR(df, cci_len=20, atr_len=14, base_thresh=100):
    cl       = df['close']
    cci      = _cci(df, cci_len)
    atr      = _atr(df, atr_len)
    atr_ma   = _sma(atr, 50)
    # High ATR → raise threshold (reduce noise signals)
    vol_r    = (atr / atr_ma.replace(0, 1e-9)).clip(0.5, 3.0)
    thresh   = base_thresh * vol_r
    sig      = pd.Series(0, index=df.index)
    bull     = (cci > -thresh) & (cci.shift(1) <= -thresh)
    bear     = (cci < thresh) & (cci.shift(1) >= thresh)
    sig[bull] = 1
    sig[bear] = -1
    return sig

space_Adaptive_CCI_ATR = {
    'cci_len':     ('int', 14, 30),
    'atr_len':     ('int', 7, 21),
    'base_thresh': ('int', 80, 150),
}

# ─────────────────────────────────────────────────────────────────────────────
# 21. MTF_Keltner_BB
# Keltner outside BB squeeze on multiple sim periods
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_Keltner_BB(df, length=20, bb_mult=2.0, kc_mult=1.5, htf_mult=3):
    cl          = df['close']
    bb_up, bb_mid, bb_lo  = _bb(cl, length, bb_mult)
    kc_up, kc_mid, kc_lo  = _keltner(df, length, kc_mult)
    bb_up2, _, bb_lo2     = _bb(cl, int(length*htf_mult), bb_mult)
    kc_up2, _, kc_lo2     = _keltner(df, int(length*htf_mult), kc_mult)
    # No squeeze on both TFs = expanded market
    no_sq1  = ~((bb_up < kc_up) & (bb_lo > kc_lo))
    no_sq2  = ~((bb_up2 < kc_up2) & (bb_lo2 > kc_lo2))
    expanded = no_sq1 & no_sq2
    sig      = pd.Series(0, index=df.index)
    bull     = expanded & (cl > kc_up) & (cl > bb_up)
    bear     = expanded & (cl < kc_lo) & (cl < bb_lo)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_MTF_Keltner_BB = {
    'length':   ('int', 15, 30),
    'bb_mult':  ('float', 1.5, 2.5),
    'kc_mult':  ('float', 1.0, 2.0),
    'htf_mult': ('int', 2, 4),
}

# ─────────────────────────────────────────────────────────────────────────────
# 22. Adaptive_WMA_Crossover
# WMA crossover with adaptive periods based on choppiness
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_WMA_Crossover(df, base_fast=10, base_slow=30, chop_len=14, chop_thresh=61.8):
    cl       = df['close']
    chop     = _chop(df, chop_len)
    choppy   = chop > chop_thresh
    # Choppy market → use longer periods to reduce whipsaws
    f_wma_n  = _wma(cl, base_fast)
    s_wma_n  = _wma(cl, base_slow)
    f_wma_c  = _wma(cl, int(base_fast * 1.8))
    s_wma_c  = _wma(cl, int(base_slow * 1.8))
    f_wma    = f_wma_c.where(choppy, f_wma_n)
    s_wma    = s_wma_c.where(choppy, s_wma_n)
    sig      = pd.Series(0, index=df.index)
    bull     = f_wma > s_wma
    bear     = f_wma < s_wma
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_Adaptive_WMA_Crossover = {
    'base_fast':   ('int', 5, 20),
    'base_slow':   ('int', 20, 60),
    'chop_len':    ('int', 10, 25),
    'chop_thresh': ('float', 50.0, 70.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# 23. MTF_RSI_MACD
# RSI + MACD confluence signal
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_RSI_MACD(df, rsi_len=14, fast=12, slow=26, sig_len=9, ob=60, os=40):
    cl           = df['close']
    rsi          = _rsi(cl, rsi_len)
    _, _, hist   = _macd(cl, fast, slow, sig_len)
    sig          = pd.Series(0, index=df.index)
    bull         = (rsi < os) & (hist > 0)
    bear         = (rsi > ob) & (hist < 0)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_MTF_RSI_MACD = {
    'rsi_len': ('int', 7, 21),
    'fast':    ('int', 8, 16),
    'slow':    ('int', 20, 34),
    'sig_len': ('int', 7, 12),
    'ob':      ('int', 55, 70),
    'os':      ('int', 30, 45),
}

# ─────────────────────────────────────────────────────────────────────────────
# 24. Adaptive_Donchian
# Donchian channel period adapts to volatility
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_Donchian(df, base_len=20, atr_len=14, vol_window=50):
    cl       = df['close']
    atr      = _atr(df, atr_len)
    atr_ma   = _sma(atr, vol_window)
    vol_r    = (atr / atr_ma.replace(0, 1e-9)).clip(0.5, 2.5)
    # High vol → shorter Donchian (more reactive)
    eff_len  = (base_len / vol_r).clip(5, base_len * 2).astype(int)
    # Use the median effective length to avoid bar-by-bar changing
    med_len  = int(eff_len.rolling(10, min_periods=1).median().fillna(base_len).iloc[-1])
    med_len  = max(5, med_len)
    hi_ch, mid_ch, lo_ch = _donchian(df, med_len)
    sig      = pd.Series(0, index=df.index)
    bull     = (cl > hi_ch.shift(1)) & (cl.shift(1) <= hi_ch.shift(2))
    bear     = (cl < lo_ch.shift(1)) & (cl.shift(1) >= lo_ch.shift(2))
    sig[bull] = 1
    sig[bear] = -1
    return sig

space_Adaptive_Donchian = {
    'base_len':   ('int', 10, 40),
    'atr_len':    ('int', 7, 21),
    'vol_window': ('int', 30, 80),
}

# ─────────────────────────────────────────────────────────────────────────────
# 25. MTF_Trend_Momentum
# Trend direction + momentum confirmation across sims
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_Trend_Momentum(df, ema_len=20, mom_len=10, htf_mult=4):
    cl       = df['close']
    ema      = _ema(cl, ema_len)
    ema_htf  = _ema(cl, int(ema_len * htf_mult))
    mom      = cl - cl.shift(mom_len)
    mom_htf  = cl - cl.shift(int(mom_len * htf_mult))
    sig      = pd.Series(0, index=df.index)
    bull     = (cl > ema) & (cl > ema_htf) & (mom > 0) & (mom_htf > 0)
    bear     = (cl < ema) & (cl < ema_htf) & (mom < 0) & (mom_htf < 0)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_MTF_Trend_Momentum = {
    'ema_len':  ('int', 10, 50),
    'mom_len':  ('int', 5, 25),
    'htf_mult': ('int', 2, 6),
}

# ─────────────────────────────────────────────────────────────────────────────
# 26. Adaptive_OBV_Trend
# OBV smoothed with adaptive EMA length
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_OBV_Trend(df, base_len=20, atr_len=14, vol_window=50):
    cl       = df['close']
    obv      = _obv(df)
    atr      = _atr(df, atr_len)
    atr_ma   = _sma(atr, vol_window)
    vol_r    = (atr / atr_ma.replace(0, 1e-9)).clip(0.5, 2.5)
    # Low vol → shorter smoothing (quicker to react)
    eff_len  = (base_len * vol_r).clip(5, base_len * 3).astype(int)
    med_len  = int(eff_len.rolling(10, min_periods=1).median().fillna(base_len).iloc[-1])
    med_len  = max(5, med_len)
    obv_sma  = _sma(obv, med_len)
    obv_fast = _ema(obv, max(3, med_len // 3))
    sig      = pd.Series(0, index=df.index)
    bull     = (obv_fast > obv_sma) & (obv > obv_sma)
    bear     = (obv_fast < obv_sma) & (obv < obv_sma)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_Adaptive_OBV_Trend = {
    'base_len':   ('int', 10, 40),
    'atr_len':    ('int', 7, 21),
    'vol_window': ('int', 30, 80),
}

# ─────────────────────────────────────────────────────────────────────────────
# 27. MTF_Pivot_Trend
# Pivot highs/lows on current TF confirmed by higher TF EMA
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_Pivot_Trend(df, pivot_left=5, pivot_right=5, htf_ema_mult=4, ema_base=20):
    cl       = df['close']
    hi       = df['high']
    lo       = df['low']
    htf_ema  = _ema(cl, int(ema_base * htf_ema_mult))
    # Rolling pivot detection (simplified: local high/low)
    hi_max   = hi.rolling(int(pivot_left + pivot_right + 1), min_periods=1).max()
    lo_min   = lo.rolling(int(pivot_left + pivot_right + 1), min_periods=1).min()
    at_ph    = hi == hi_max  # potential pivot high
    at_pl    = lo == lo_min  # potential pivot low
    sig      = pd.Series(0, index=df.index)
    bull     = at_pl & (cl > htf_ema)
    bear     = at_ph & (cl < htf_ema)
    sig[bull] = 1
    sig[bear] = -1
    return sig

space_MTF_Pivot_Trend = {
    'pivot_left':    ('int', 3, 10),
    'pivot_right':   ('int', 3, 10),
    'htf_ema_mult':  ('int', 2, 6),
    'ema_base':      ('int', 10, 40),
}

# ─────────────────────────────────────────────────────────────────────────────
# 28. Adaptive_BB_RSI
# BB mean reversion filtered by adaptive RSI thresholds
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_BB_RSI(df, bb_len=20, bb_mult=2.0, rsi_len=14, atr_len=14):
    cl       = df['close']
    bb_up, bb_mid, bb_lo = _bb(cl, bb_len, bb_mult)
    rsi      = _rsi(cl, rsi_len)
    atr      = _atr(df, atr_len)
    atr_ma   = _sma(atr, 50)
    vol_r    = (atr / atr_ma.replace(0, 1e-9)).clip(0.5, 2.5)
    # Adapt RSI thresholds: high vol → wider bands (looser)
    ob       = (70 * vol_r).clip(65, 85)
    os       = (30 / vol_r).clip(15, 35)
    sig      = pd.Series(0, index=df.index)
    bull     = (cl < bb_lo) & (rsi < os)
    bear     = (cl > bb_up) & (rsi > ob)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_Adaptive_BB_RSI = {
    'bb_len':  ('int', 15, 30),
    'bb_mult': ('float', 1.5, 2.5),
    'rsi_len': ('int', 7, 21),
    'atr_len': ('int', 7, 21),
}

# ─────────────────────────────────────────────────────────────────────────────
# 29. MTF_Force_Index
# Elder Force Index trend on current + 4x simulation
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_Force_Index(df, fi_len=13, htf_mult=4):
    fi      = _force_index(df, fi_len)
    fi_htf  = _force_index(df, int(fi_len * htf_mult))
    sig     = pd.Series(0, index=df.index)
    bull    = (fi > 0) & (fi_htf > 0)
    bear    = (fi < 0) & (fi_htf < 0)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_MTF_Force_Index = {
    'fi_len':   ('int', 7, 25),
    'htf_mult': ('int', 2, 6),
}

# ─────────────────────────────────────────────────────────────────────────────
# 30. Adaptive_MA_Regime
# MA crossover system with volatility regime filter (high vol = wider bands required)
# ─────────────────────────────────────────────────────────────────────────────
def gen_Adaptive_MA_Regime(df, fast=10, slow=30, atr_len=14, vol_window=50, band_mult=0.5):
    cl       = df['close']
    f_ema    = _ema(cl, fast)
    s_ema    = _ema(cl, slow)
    atr      = _atr(df, atr_len)
    atr_ma   = _sma(atr, vol_window)
    vol_r    = (atr / atr_ma.replace(0, 1e-9)).clip(0.5, 3.0)
    # In high-vol regime require bigger spread between EMAs to confirm signal
    required_spread = band_mult * atr * vol_r
    spread          = (f_ema - s_ema).abs()
    sig      = pd.Series(0, index=df.index)
    bull     = (f_ema > s_ema) & (spread > required_spread)
    bear     = (f_ema < s_ema) & (spread > required_spread)
    sig[bull & ~bull.shift(1).fillna(False)] = 1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    return sig

space_Adaptive_MA_Regime = {
    'fast':       ('int', 5, 20),
    'slow':       ('int', 20, 60),
    'atr_len':    ('int', 7, 21),
    'vol_window': ('int', 30, 80),
    'band_mult':  ('float', 0.1, 1.5),
}

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY_EXPORT
# ─────────────────────────────────────────────────────────────────────────────
STRATEGY_EXPORT = {
    'MTF_EMA_Confluence':        {'gen': gen_MTF_EMA_Confluence,        'space': space_MTF_EMA_Confluence},
    'MTF_RSI_Confluence':        {'gen': gen_MTF_RSI_Confluence,        'space': space_MTF_RSI_Confluence},
    'MTF_SuperTrend_Confluence': {'gen': gen_MTF_SuperTrend_Confluence, 'space': space_MTF_SuperTrend_Confluence},
    'MTF_MACD_Alignment':        {'gen': gen_MTF_MACD_Alignment,        'space': space_MTF_MACD_Alignment},
    'MTF_Stoch_Sync':            {'gen': gen_MTF_Stoch_Sync,            'space': space_MTF_Stoch_Sync},
    'Adaptive_RSI_BB':           {'gen': gen_Adaptive_RSI_BB,           'space': space_Adaptive_RSI_BB},
    'Adaptive_EMA_Vol':          {'gen': gen_Adaptive_EMA_Vol,          'space': space_Adaptive_EMA_Vol},
    'Adaptive_MA_Crossover':     {'gen': gen_Adaptive_MA_Crossover,     'space': space_Adaptive_MA_Crossover},
    'Adaptive_Momentum_ATR':     {'gen': gen_Adaptive_Momentum_ATR,     'space': space_Adaptive_Momentum_ATR},
    'Adaptive_Trend_Strength':   {'gen': gen_Adaptive_Trend_Strength,   'space': space_Adaptive_Trend_Strength},
    'MTF_BB_Squeeze_Confirm':    {'gen': gen_MTF_BB_Squeeze_Confirm,    'space': space_MTF_BB_Squeeze_Confirm},
    'MTF_Volume_Trend':          {'gen': gen_MTF_Volume_Trend,          'space': space_MTF_Volume_Trend},
    'MTF_Ichimoku_Cloud':        {'gen': gen_MTF_Ichimoku_Cloud,        'space': space_MTF_Ichimoku_Cloud},
    'Adaptive_Chandelier':       {'gen': gen_Adaptive_Chandelier,       'space': space_Adaptive_Chandelier},
    'MTF_ADX_Filter':            {'gen': gen_MTF_ADX_Filter,            'space': space_MTF_ADX_Filter},
    'Adaptive_Stoch_RSI':        {'gen': gen_Adaptive_Stoch_RSI,        'space': space_Adaptive_Stoch_RSI},
    'MTF_VWAP_Trend':            {'gen': gen_MTF_VWAP_Trend,            'space': space_MTF_VWAP_Trend},
    'Adaptive_ATR_Channel':      {'gen': gen_Adaptive_ATR_Channel,      'space': space_Adaptive_ATR_Channel},
    'MTF_Hull_Trend':            {'gen': gen_MTF_Hull_Trend,            'space': space_MTF_Hull_Trend},
    'Adaptive_CCI_ATR':          {'gen': gen_Adaptive_CCI_ATR,          'space': space_Adaptive_CCI_ATR},
    'MTF_Keltner_BB':            {'gen': gen_MTF_Keltner_BB,            'space': space_MTF_Keltner_BB},
    'Adaptive_WMA_Crossover':    {'gen': gen_Adaptive_WMA_Crossover,    'space': space_Adaptive_WMA_Crossover},
    'MTF_RSI_MACD':              {'gen': gen_MTF_RSI_MACD,              'space': space_MTF_RSI_MACD},
    'Adaptive_Donchian':         {'gen': gen_Adaptive_Donchian,         'space': space_Adaptive_Donchian},
    'MTF_Trend_Momentum':        {'gen': gen_MTF_Trend_Momentum,        'space': space_MTF_Trend_Momentum},
    'Adaptive_OBV_Trend':        {'gen': gen_Adaptive_OBV_Trend,        'space': space_Adaptive_OBV_Trend},
    'MTF_Pivot_Trend':           {'gen': gen_MTF_Pivot_Trend,           'space': space_MTF_Pivot_Trend},
    'Adaptive_BB_RSI':           {'gen': gen_Adaptive_BB_RSI,           'space': space_Adaptive_BB_RSI},
    'MTF_Force_Index':           {'gen': gen_MTF_Force_Index,           'space': space_MTF_Force_Index},
    'Adaptive_MA_Regime':        {'gen': gen_Adaptive_MA_Regime,        'space': space_Adaptive_MA_Regime},
}
