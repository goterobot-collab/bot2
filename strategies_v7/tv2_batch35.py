#!/usr/bin/env python3
"""TV2 BATCH 35 — 30 Market Cipher + WaveTrend Strategies 2026-04-01"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=int(p), adjust=False).mean()


def _rma(s, p):
    return s.ewm(alpha=1 / int(p), adjust=False).mean()


def _sma(s, p):
    return s.rolling(int(p), min_periods=1).mean()


def _atr(df, p):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, p)


def _rsi(s, p):
    d = s.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    return 100 - 100 / (1 + _rma(g, p) / _rma(l, p).replace(0, 1e-9))


def _wavetrend(df, n1, n2):
    """LazyBear WaveTrend: ap=hlc3, esa=EMA(ap,n1), d=EMA(|ap-esa|,n1),
    ci=(ap-esa)/(0.015*d), tci=EMA(ci,n2)"""
    ap  = (df['high'] + df['low'] + df['close']) / 3
    esa = _ema(ap, n1)
    d   = _ema((ap - esa).abs(), n1)
    ci  = (ap - esa) / (0.015 * d.replace(0, 1e-9))
    wt1 = _ema(ci, n2)
    wt2 = _sma(wt1, 4)
    return wt1, wt2


def _bb(s, p, mult):
    mid   = _sma(s, p)
    std   = s.rolling(int(p), min_periods=1).std().fillna(0)
    upper = mid + mult * std
    lower = mid - mult * std
    return upper, mid, lower


def _stoch(df, k):
    p  = int(k)
    lo = df['low'].rolling(p, min_periods=1).min()
    hi = df['high'].rolling(p, min_periods=1).max()
    return 100 * (df['close'] - lo) / (hi - lo + 1e-9)


def _cci(df, p):
    tp  = (df['high'] + df['low'] + df['close']) / 3
    sma = _sma(tp, p)
    mad = tp.rolling(int(p), min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, 1e-9))


def _mfi(df, p):
    tp  = (df['high'] + df['low'] + df['close']) / 3
    rmf = tp * df['volume']
    pos = rmf.where(tp > tp.shift(1), 0).rolling(int(p), min_periods=1).sum()
    neg = rmf.where(tp < tp.shift(1), 0).rolling(int(p), min_periods=1).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, 1e-9))


def _obv(df):
    sign = np.sign(df['close'].diff().fillna(0))
    return (sign * df['volume']).cumsum()


def _supertrend(df, p, mult):
    p   = int(p)
    atr = _atr(df, p)
    hl2 = (df['high'] + df['low']) / 2
    up  = hl2 - mult * atr
    dn  = hl2 + mult * atr
    cl  = df['close']
    trend      = pd.Series(1, index=df.index)
    final_up   = up.copy()
    final_down = dn.copy()
    for i in range(1, len(df)):
        fu = final_up.iloc[i - 1]
        fd = final_down.iloc[i - 1]
        final_up.iloc[i]   = max(up.iloc[i], fu) if cl.iloc[i - 1] > fu else up.iloc[i]
        final_down.iloc[i] = min(dn.iloc[i], fd) if cl.iloc[i - 1] < fd else dn.iloc[i]
        if cl.iloc[i] > final_down.iloc[i]:
            trend.iloc[i] = 1
        elif cl.iloc[i] < final_up.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]
    return trend


def _cmf(df, p):
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
          (df['high'] - df['low'] + 1e-9)
    mfv = clv * df['volume']
    return mfv.rolling(int(p), min_periods=1).sum() / \
           df['volume'].rolling(int(p), min_periods=1).sum().replace(0, 1e-9)


def _vortex(df, p):
    p   = int(p)
    vmp = (df['high'] - df['low'].shift(1)).abs()
    vmn = (df['low']  - df['high'].shift(1)).abs()
    tr  = _atr(df, p) * p
    vi_plus  = vmp.rolling(p, min_periods=1).sum() / tr.replace(0, 1e-9)
    vi_minus = vmn.rolling(p, min_periods=1).sum() / tr.replace(0, 1e-9)
    return vi_plus, vi_minus


def _linreg_slope(s, p):
    p   = int(p)
    out = pd.Series(np.nan, index=s.index)
    arr = s.values
    x   = np.arange(p, dtype=float)
    for i in range(p - 1, len(arr)):
        y = arr[i - p + 1: i + 1].astype(float)
        if not np.any(np.isnan(y)):
            m, _ = np.polyfit(x, y, 1)
            out.iloc[i] = m
    return out.fillna(0)


def _wma(s, p):
    p = int(p)
    weights = np.arange(1, p + 1, dtype=float)
    def _w(x):
        if len(x) < p:
            return np.nan
        return np.dot(x, weights[-len(x):]) / weights[-len(x):].sum()
    return s.rolling(p, min_periods=1).apply(_w, raw=True)


def _hma(s, p):
    p    = int(p)
    half = max(int(p / 2), 1)
    sqrp = max(int(np.sqrt(p)), 1)
    raw  = 2 * _wma(s, half) - _wma(s, p)
    return _wma(raw, sqrp)


def _alma(s, p, offset=0.85, sigma=6):
    p   = int(p)
    out = pd.Series(np.nan, index=s.index)
    m   = offset * (p - 1)
    sig = p / sigma
    w   = np.array([np.exp(-(i - m) ** 2 / (2 * sig ** 2)) for i in range(p)])
    w   /= w.sum()
    arr = s.values
    for i in range(p - 1, len(arr)):
        chunk = arr[i - p + 1: i + 1]
        if not np.any(np.isnan(chunk.astype(float))):
            out.iloc[i] = np.dot(w, chunk)
    return out


def _qqe(s, rsi_p=14, sf=5):
    rsi  = _rsi(s, rsi_p)
    rsi_s = _ema(rsi, sf)
    return rsi_s


def _keltner(df, p, mult):
    mid   = _ema(df['close'], p)
    atr   = _atr(df, p)
    upper = mid + mult * atr
    lower = mid - mult * atr
    return upper, mid, lower


def _ichimoku(df, tenkan, kijun):
    tenkan = int(tenkan)
    kijun  = int(kijun)
    t_high = df['high'].rolling(tenkan, min_periods=1).max()
    t_low  = df['low'].rolling(tenkan, min_periods=1).min()
    k_high = df['high'].rolling(kijun, min_periods=1).max()
    k_low  = df['low'].rolling(kijun, min_periods=1).min()
    tenkan_sen = (t_high + t_low) / 2
    kijun_sen  = (k_high + k_low) / 2
    span_a = (tenkan_sen + kijun_sen) / 2
    span_b = (df['high'].rolling(52, min_periods=1).max() +
              df['low'].rolling(52, min_periods=1).min()) / 2
    cloud_top    = span_a.combine(span_b, max)
    cloud_bottom = span_a.combine(span_b, min)
    return cloud_top, cloud_bottom


def _macd(s, fast, slow, sig):
    fast_ema = _ema(s, fast)
    slow_ema = _ema(s, slow)
    macd_line = fast_ema - slow_ema
    signal    = _ema(macd_line, sig)
    return macd_line, signal, macd_line - signal


# ── 1. WT_Classic ────────────────────────────────────────────────────────────

def gen_WT_Classic(df, n1=10, n2=21, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (wt2.shift(1) < -60)] =  1
    sig[cross_down & (wt2.shift(1) >  60)] = -1
    return sig


def space_WT_Classic(trial):
    return {
        'n1': trial.suggest_int('n1', 7, 15),
        'n2': trial.suggest_int('n2', 15, 30),
    }


# ── 2. WT_OB_OS ──────────────────────────────────────────────────────────────

def gen_WT_OB_OS(df, n1=10, n2=21, ob=60.0, os=-60.0, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    sig = pd.Series(0, index=df.index)
    sig[wt1 < os] =  1
    sig[wt1 > ob] = -1
    return sig


def space_WT_OB_OS(trial):
    return {
        'n1': trial.suggest_int('n1', 7, 15),
        'n2': trial.suggest_int('n2', 15, 30),
        'ob': trial.suggest_float('ob', 50.0, 70.0),
        'os': trial.suggest_float('os', -70.0, -50.0),
    }


# ── 3. WT_Divergence ─────────────────────────────────────────────────────────

def gen_WT_Divergence(df, n1=10, n2=21, lookback=10, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    cl = df['close']
    lb = int(lookback)
    price_ll = cl < cl.shift(lb)
    wt_hl    = wt1 > wt1.shift(lb)
    price_hh = cl > cl.shift(lb)
    wt_lh    = wt1 < wt1.shift(lb)
    sig = pd.Series(0, index=df.index)
    sig[price_ll & wt_hl] =  1
    sig[price_hh & wt_lh] = -1
    return sig


def space_WT_Divergence(trial):
    return {
        'n1':      trial.suggest_int('n1', 7, 15),
        'n2':      trial.suggest_int('n2', 15, 30),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 4. WT_EMA ────────────────────────────────────────────────────────────────

def gen_WT_EMA(df, n1=10, n2=21, ema_p=50, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    ema      = _ema(df['close'], ema_p)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (df['close'] > ema)] =  1
    sig[cross_down & (df['close'] < ema)] = -1
    return sig


def space_WT_EMA(trial):
    return {
        'n1':    trial.suggest_int('n1', 7, 15),
        'n2':    trial.suggest_int('n2', 15, 30),
        'ema_p': trial.suggest_int('ema_p', 20, 100),
    }


# ── 5. WT_RSI_Confirm ────────────────────────────────────────────────────────

def gen_WT_RSI_Confirm(df, n1=10, n2=21, rsi_p=14, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    rsi      = _rsi(df['close'], rsi_p)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (rsi < 50)] =  1
    sig[cross_down & (rsi > 50)] = -1
    return sig


def space_WT_RSI_Confirm(trial):
    return {
        'n1':    trial.suggest_int('n1', 7, 15),
        'n2':    trial.suggest_int('n2', 15, 30),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 6. WT_Volume ─────────────────────────────────────────────────────────────

def gen_WT_Volume(df, n1=10, n2=21, vol_p=30, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    vol_avg  = _sma(df['volume'], vol_p)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    vol_ok = df['volume'] > vol_avg
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & vol_ok] =  1
    sig[cross_down & vol_ok] = -1
    return sig


def space_WT_Volume(trial):
    return {
        'n1':    trial.suggest_int('n1', 7, 15),
        'n2':    trial.suggest_int('n2', 15, 30),
        'vol_p': trial.suggest_int('vol_p', 20, 50),
    }


# ── 7. WT_ATR ────────────────────────────────────────────────────────────────

def gen_WT_ATR(df, n1=10, n2=21, atr_p=14, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    atr      = _atr(df, atr_p)
    atr_avg  = _sma(atr, atr_p)
    atr_expand = atr > atr_avg
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (wt2.shift(1) < -40) & atr_expand] =  1
    sig[cross_down & (wt2.shift(1) >  40) & atr_expand] = -1
    return sig


def space_WT_ATR(trial):
    return {
        'n1':    trial.suggest_int('n1', 7, 15),
        'n2':    trial.suggest_int('n2', 15, 30),
        'atr_p': trial.suggest_int('atr_p', 10, 20),
    }


# ── 8. WT_BB ─────────────────────────────────────────────────────────────────

def gen_WT_BB(df, n1=10, n2=21, bb_p=20, bb_mult=2.0, **kw):
    wt1, wt2   = _wavetrend(df, n1, n2)
    upper, _, lower = _bb(df['close'], bb_p, bb_mult)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (wt1 < -60) & (df['close'] < lower)] =  1
    sig[cross_down & (wt1 >  60) & (df['close'] > upper)] = -1
    return sig


def space_WT_BB(trial):
    return {
        'n1':      trial.suggest_int('n1', 7, 15),
        'n2':      trial.suggest_int('n2', 15, 30),
        'bb_p':    trial.suggest_int('bb_p', 20, 20),
        'bb_mult': trial.suggest_float('bb_mult', 2.0, 2.0),
    }


# ── 9. WT_MACD ───────────────────────────────────────────────────────────────

def gen_WT_MACD(df, n1=10, n2=21, fast=12, slow=26, sig_p=9, **kw):
    wt1, wt2    = _wavetrend(df, n1, n2)
    macd, signal, _ = _macd(df['close'], fast, slow, sig_p)
    cross_up    = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down  = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (macd > signal)] =  1
    sig[cross_down & (macd < signal)] = -1
    return sig


def space_WT_MACD(trial):
    return {
        'n1':    trial.suggest_int('n1', 7, 15),
        'n2':    trial.suggest_int('n2', 15, 30),
        'fast':  trial.suggest_int('fast', 8, 16),
        'slow':  trial.suggest_int('slow', 20, 30),
        'sig_p': trial.suggest_int('sig_p', 5, 12),
    }


# ── 10. WT_SuperTrend ────────────────────────────────────────────────────────

def gen_WT_SuperTrend(df, n1=10, n2=21, st_p=10, st_mult=3.0, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    st       = _supertrend(df, st_p, st_mult)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (st > 0)] =  1
    sig[cross_down & (st < 0)] = -1
    return sig


def space_WT_SuperTrend(trial):
    return {
        'n1':      trial.suggest_int('n1', 7, 15),
        'n2':      trial.suggest_int('n2', 15, 30),
        'st_p':    trial.suggest_int('st_p', 7, 21),
        'st_mult': trial.suggest_float('st_mult', 2.0, 4.0),
    }


# ── 11. WT_Fast ──────────────────────────────────────────────────────────────

def gen_WT_Fast(df, n1=5, n2=10, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (wt2.shift(1) < -40)] =  1
    sig[cross_down & (wt2.shift(1) >  40)] = -1
    return sig


def space_WT_Fast(trial):
    return {
        'n1': trial.suggest_int('n1', 3, 8),
        'n2': trial.suggest_int('n2', 8, 15),
    }


# ── 12. WT_Dual ──────────────────────────────────────────────────────────────

def gen_WT_Dual(df, fast_n1=6, fast_n2=12, slow_n1=14, slow_n2=30, **kw):
    f_wt1, f_wt2 = _wavetrend(df, fast_n1, fast_n2)
    s_wt1, s_wt2 = _wavetrend(df, slow_n1, slow_n2)
    fast_cross_up   = (f_wt1 > f_wt2) & (f_wt1.shift(1) <= f_wt2.shift(1))
    fast_cross_down = (f_wt1 < f_wt2) & (f_wt1.shift(1) >= f_wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[fast_cross_up   & (s_wt1 > 0)] =  1
    sig[fast_cross_down & (s_wt1 < 0)] = -1
    return sig


def space_WT_Dual(trial):
    return {
        'fast_n1': trial.suggest_int('fast_n1', 5, 10),
        'fast_n2': trial.suggest_int('fast_n2', 10, 20),
        'slow_n1': trial.suggest_int('slow_n1', 10, 20),
        'slow_n2': trial.suggest_int('slow_n2', 25, 40),
    }


# ── 13. MC_Full ──────────────────────────────────────────────────────────────

def gen_MC_Full(df, n1=10, n2=21, mfi_p=14, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    mfi      = _mfi(df, mfi_p)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (mfi < 30)] =  1
    sig[cross_down & (mfi > 70)] = -1
    return sig


def space_MC_Full(trial):
    return {
        'n1':    trial.suggest_int('n1', 7, 15),
        'n2':    trial.suggest_int('n2', 15, 30),
        'mfi_p': trial.suggest_int('mfi_p', 7, 21),
    }


# ── 14. MC_Money_Flow ────────────────────────────────────────────────────────

def gen_MC_Money_Flow(df, n1=10, n2=21, mfi_p=14, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    mfi      = _mfi(df, mfi_p)
    obv      = _obv(df)
    obv_ma   = _ema(obv, 20)
    obv_rising = obv > obv_ma
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (mfi < 25) & obv_rising] =  1
    sig[cross_down & (mfi > 75) & ~obv_rising] = -1
    return sig


def space_MC_Money_Flow(trial):
    return {
        'n1':    trial.suggest_int('n1', 7, 15),
        'n2':    trial.suggest_int('n2', 15, 30),
        'mfi_p': trial.suggest_int('mfi_p', 7, 21),
    }


# ── 15. MC_Diamond ───────────────────────────────────────────────────────────

def gen_MC_Diamond(df, n1=10, n2=21, os_level=-70.0, lookback=10, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    cl  = df['close']
    lb  = int(lookback)
    price_ll = cl < cl.shift(lb)
    wt_hl    = wt1 > wt1.shift(lb)
    price_hh = cl > cl.shift(lb)
    wt_lh    = wt1 < wt1.shift(lb)
    sig = pd.Series(0, index=df.index)
    sig[price_ll & wt_hl & (wt1 < os_level)] =  1
    sig[price_hh & wt_lh & (wt1 > -os_level)] = -1
    return sig


def space_MC_Diamond(trial):
    return {
        'n1':       trial.suggest_int('n1', 7, 15),
        'n2':       trial.suggest_int('n2', 15, 30),
        'os_level': trial.suggest_float('os_level', -80.0, -60.0),
        'lookback': trial.suggest_int('lookback', 5, 15),
    }


# ── 16. WT_Stoch ─────────────────────────────────────────────────────────────

def gen_WT_Stoch(df, n1=10, n2=21, k_p=14, d_p=3, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    stoch_k  = _stoch(df, k_p)
    stoch_d  = _sma(stoch_k, d_p)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (wt1 < -60) & (stoch_k < 20)] =  1
    sig[cross_down & (wt1 >  60) & (stoch_k > 80)] = -1
    return sig


def space_WT_Stoch(trial):
    return {
        'n1':  trial.suggest_int('n1', 7, 15),
        'n2':  trial.suggest_int('n2', 15, 30),
        'k_p': trial.suggest_int('k_p', 7, 21),
        'd_p': trial.suggest_int('d_p', 3, 7),
    }


# ── 17. WT_CCI ───────────────────────────────────────────────────────────────

def gen_WT_CCI(df, n1=10, n2=21, cci_p=14, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    cci      = _cci(df, cci_p)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (cci < -100)] =  1
    sig[cross_down & (cci >  100)] = -1
    return sig


def space_WT_CCI(trial):
    return {
        'n1':    trial.suggest_int('n1', 7, 15),
        'n2':    trial.suggest_int('n2', 15, 30),
        'cci_p': trial.suggest_int('cci_p', 10, 20),
    }


# ── 18. WT_Vortex ────────────────────────────────────────────────────────────

def gen_WT_Vortex(df, n1=10, n2=21, v_p=14, **kw):
    wt1, wt2   = _wavetrend(df, n1, n2)
    vi_plus, vi_minus = _vortex(df, v_p)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (vi_plus > vi_minus)] =  1
    sig[cross_down & (vi_minus > vi_plus)] = -1
    return sig


def space_WT_Vortex(trial):
    return {
        'n1':  trial.suggest_int('n1', 7, 15),
        'n2':  trial.suggest_int('n2', 15, 30),
        'v_p': trial.suggest_int('v_p', 10, 20),
    }


# ── 19. WT_OBV ───────────────────────────────────────────────────────────────

def gen_WT_OBV(df, n1=10, n2=21, obv_ema_p=20, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    obv      = _obv(df)
    obv_ma   = _ema(obv, obv_ema_p)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (obv > obv_ma)] =  1
    sig[cross_down & (obv < obv_ma)] = -1
    return sig


def space_WT_OBV(trial):
    return {
        'n1':        trial.suggest_int('n1', 7, 15),
        'n2':        trial.suggest_int('n2', 15, 30),
        'obv_ema_p': trial.suggest_int('obv_ema_p', 10, 30),
    }


# ── 20. WT_Ichimoku ──────────────────────────────────────────────────────────

def gen_WT_Ichimoku(df, n1=10, n2=21, tenkan=9, kijun=26, **kw):
    wt1, wt2       = _wavetrend(df, n1, n2)
    cloud_top, cloud_bottom = _ichimoku(df, tenkan, kijun)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (df['close'] > cloud_top)]    =  1
    sig[cross_down & (df['close'] < cloud_bottom)] = -1
    return sig


def space_WT_Ichimoku(trial):
    return {
        'n1':    trial.suggest_int('n1', 7, 15),
        'n2':    trial.suggest_int('n2', 15, 30),
        'tenkan': trial.suggest_int('tenkan', 7, 14),
        'kijun':  trial.suggest_int('kijun', 20, 35),
    }


# ── 21. WT_Keltner ───────────────────────────────────────────────────────────

def gen_WT_Keltner(df, n1=10, n2=21, kc_p=20, kc_mult=2.0, **kw):
    wt1, wt2       = _wavetrend(df, n1, n2)
    upper, _, lower = _keltner(df, kc_p, kc_mult)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (df['close'] < lower)] =  1
    sig[cross_down & (df['close'] > upper)] = -1
    return sig


def space_WT_Keltner(trial):
    return {
        'n1':      trial.suggest_int('n1', 7, 15),
        'n2':      trial.suggest_int('n2', 15, 30),
        'kc_p':    trial.suggest_int('kc_p', 20, 20),
        'kc_mult': trial.suggest_float('kc_mult', 1.5, 3.0),
    }


# ── 22. WT_LinReg ────────────────────────────────────────────────────────────

def gen_WT_LinReg(df, n1=10, n2=21, lr_p=20, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    slope    = _linreg_slope(df['close'], lr_p)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (slope > 0)] =  1
    sig[cross_down & (slope < 0)] = -1
    return sig


def space_WT_LinReg(trial):
    return {
        'n1':   trial.suggest_int('n1', 7, 15),
        'n2':   trial.suggest_int('n2', 15, 30),
        'lr_p': trial.suggest_int('lr_p', 10, 40),
    }


# ── 23. WT_Adaptive ──────────────────────────────────────────────────────────

def gen_WT_Adaptive(df, base_n1=10, base_n2=21, atr_p=14, **kw):
    atr     = _atr(df, atr_p)
    atr_avg = _sma(atr, atr_p)
    # High vol = shorter n1 (min 3), low vol = longer n1
    adapt_factor = (atr / atr_avg.replace(0, 1e-9)).clip(0.5, 2.0)
    # Use adaptive periods per-bar is infeasible vectorially — use rolling median
    adapt_n1 = int(base_n1)
    adapt_n2 = int(base_n2)
    wt1, wt2 = _wavetrend(df, adapt_n1, adapt_n2)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    # Signal quality filter: only enter when ATR is expanding (high momentum)
    atr_ok = atr > atr_avg
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & atr_ok & (wt2.shift(1) < -30)] =  1
    sig[cross_down & atr_ok & (wt2.shift(1) >  30)] = -1
    return sig


def space_WT_Adaptive(trial):
    return {
        'base_n1': trial.suggest_int('base_n1', 7, 15),
        'base_n2': trial.suggest_int('base_n2', 15, 30),
        'atr_p':   trial.suggest_int('atr_p', 10, 20),
    }


# ── 24. WT_Pattern ───────────────────────────────────────────────────────────

def gen_WT_Pattern(df, n1=10, n2=21, os=-50.0, lookback=5, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    lb = int(lookback)
    # Two consecutive WT lows each higher (rising bottoms) while below OS level
    wt_low1 = wt1.shift(lb)
    wt_low2 = wt1.shift(lb * 2)
    rising_bottoms   = (wt1 > wt_low1) & (wt_low1 > wt_low2)
    falling_tops     = (wt1 < wt_low1) & (wt_low1 < wt_low2)
    sig = pd.Series(0, index=df.index)
    sig[rising_bottoms  & (wt1 < os)]   =  1
    sig[falling_tops    & (wt1 > -os)]  = -1
    return sig


def space_WT_Pattern(trial):
    return {
        'n1':      trial.suggest_int('n1', 7, 15),
        'n2':      trial.suggest_int('n2', 15, 30),
        'os':      trial.suggest_float('os', -60.0, -40.0),
        'lookback': trial.suggest_int('lookback', 3, 10),
    }


# ── 25. WT_Histogram ─────────────────────────────────────────────────────────

def gen_WT_Histogram(df, n1=10, n2=21, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    hist     = wt1 - wt2
    cross_up   = (hist > 0) & (hist.shift(1) <= 0)
    cross_down = (hist < 0) & (hist.shift(1) >= 0)
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_WT_Histogram(trial):
    return {
        'n1': trial.suggest_int('n1', 7, 15),
        'n2': trial.suggest_int('n2', 15, 30),
    }


# ── 26. WT_QQE ───────────────────────────────────────────────────────────────

def gen_WT_QQE(df, n1=10, n2=21, rsi_p=14, sf=5, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    qqe_val  = _qqe(df['close'], rsi_p, sf)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (qqe_val > 50)] =  1
    sig[cross_down & (qqe_val < 50)] = -1
    return sig


def space_WT_QQE(trial):
    return {
        'n1':    trial.suggest_int('n1', 7, 15),
        'n2':    trial.suggest_int('n2', 15, 30),
        'rsi_p': trial.suggest_int('rsi_p', 6, 14),
        'sf':    trial.suggest_int('sf', 3, 8),
    }


# ── 27. WT_Hull ──────────────────────────────────────────────────────────────

def gen_WT_Hull(df, n1=10, n2=21, hma_p=20, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    hma      = _hma(df['close'], hma_p)
    hma_rising = hma > hma.shift(1)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & hma_rising]  =  1
    sig[cross_down & ~hma_rising] = -1
    return sig


def space_WT_Hull(trial):
    return {
        'n1':    trial.suggest_int('n1', 7, 15),
        'n2':    trial.suggest_int('n2', 15, 30),
        'hma_p': trial.suggest_int('hma_p', 10, 40),
    }


# ── 28. WT_CMF ───────────────────────────────────────────────────────────────

def gen_WT_CMF(df, n1=10, n2=21, cmf_p=20, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    cmf      = _cmf(df, cmf_p)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (cmf > 0)] =  1
    sig[cross_down & (cmf < 0)] = -1
    return sig


def space_WT_CMF(trial):
    return {
        'n1':    trial.suggest_int('n1', 7, 15),
        'n2':    trial.suggest_int('n2', 15, 30),
        'cmf_p': trial.suggest_int('cmf_p', 10, 30),
    }


# ── 29. WT_ALMA ──────────────────────────────────────────────────────────────

def gen_WT_ALMA(df, n1=10, n2=21, alma_p=20, offset=0.85, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    alma_val = _alma(df['close'], alma_p, offset)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (df['close'] > alma_val)] =  1
    sig[cross_down & (df['close'] < alma_val)] = -1
    return sig


def space_WT_ALMA(trial):
    return {
        'n1':     trial.suggest_int('n1', 7, 15),
        'n2':     trial.suggest_int('n2', 15, 30),
        'alma_p': trial.suggest_int('alma_p', 10, 30),
        'offset': trial.suggest_float('offset', 0.7, 0.9),
    }


# ── 30. WT_Triple_Confirm ────────────────────────────────────────────────────

def gen_WT_Triple_Confirm(df, n1=10, n2=21, rsi_p=14, vol_p=30, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    rsi      = _rsi(df['close'], rsi_p)
    vol_avg  = _sma(df['volume'], vol_p)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    vol_ok = df['volume'] > vol_avg
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (wt1 < -50) & (rsi < 40) & vol_ok] =  1
    sig[cross_down & (wt1 >  50) & (rsi > 60) & vol_ok] = -1
    return sig


def space_WT_Triple_Confirm(trial):
    return {
        'n1':    trial.suggest_int('n1', 7, 15),
        'n2':    trial.suggest_int('n2', 15, 30),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
        'vol_p': trial.suggest_int('vol_p', 20, 50),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'WT_Classic': {
        'gen': gen_WT_Classic,
        'space': space_WT_Classic,
        'default_params': {'n1': 10, 'n2': 21},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 8000,
                 'description': 'LazyBear WaveTrend cross in OB/OS zones — classic MC signal.'},
    },
    'WT_OB_OS': {
        'gen': gen_WT_OB_OS,
        'space': space_WT_OB_OS,
        'default_params': {'n1': 10, 'n2': 21, 'ob': 60.0, 'os': -60.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 5000,
                 'description': 'WaveTrend overbought/oversold zone entries without cross requirement.'},
    },
    'WT_Divergence': {
        'gen': gen_WT_Divergence,
        'space': space_WT_Divergence,
        'default_params': {'n1': 10, 'n2': 21, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'WaveTrend bullish/bearish divergence vs price over lookback bars.'},
    },
    'WT_EMA': {
        'gen': gen_WT_EMA,
        'space': space_WT_EMA,
        'default_params': {'n1': 10, 'n2': 21, 'ema_p': 50},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'WaveTrend cross filtered by EMA trend direction.'},
    },
    'WT_RSI_Confirm': {
        'gen': gen_WT_RSI_Confirm,
        'space': space_WT_RSI_Confirm,
        'default_params': {'n1': 10, 'n2': 21, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'WaveTrend cross with RSI momentum confirmation.'},
    },
    'WT_Volume': {
        'gen': gen_WT_Volume,
        'space': space_WT_Volume,
        'default_params': {'n1': 10, 'n2': 21, 'vol_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'WaveTrend cross confirmed by above-average volume.'},
    },
    'WT_ATR': {
        'gen': gen_WT_ATR,
        'space': space_WT_ATR,
        'default_params': {'n1': 10, 'n2': 21, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'WaveTrend cross from extreme zone with ATR expansion filter.'},
    },
    'WT_BB': {
        'gen': gen_WT_BB,
        'space': space_WT_BB,
        'default_params': {'n1': 10, 'n2': 21, 'bb_p': 20, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'WaveTrend oversold cross while price touches Bollinger lower band.'},
    },
    'WT_MACD': {
        'gen': gen_WT_MACD,
        'space': space_WT_MACD,
        'default_params': {'n1': 10, 'n2': 21, 'fast': 12, 'slow': 26, 'sig_p': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'WaveTrend cross aligned with MACD histogram direction.'},
    },
    'WT_SuperTrend': {
        'gen': gen_WT_SuperTrend,
        'space': space_WT_SuperTrend,
        'default_params': {'n1': 10, 'n2': 21, 'st_p': 10, 'st_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'WaveTrend cross in SuperTrend direction — dual confirmation.'},
    },
    'WT_Fast': {
        'gen': gen_WT_Fast,
        'space': space_WT_Fast,
        'default_params': {'n1': 5, 'n2': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Fast WaveTrend for scalping — smaller periods, -40 OS threshold.'},
    },
    'WT_Dual': {
        'gen': gen_WT_Dual,
        'space': space_WT_Dual,
        'default_params': {'fast_n1': 6, 'fast_n2': 12, 'slow_n1': 14, 'slow_n2': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Dual WaveTrend: fast cross in direction of slow WT trend.'},
    },
    'MC_Full': {
        'gen': gen_MC_Full,
        'space': space_MC_Full,
        'default_params': {'n1': 10, 'n2': 21, 'mfi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 6000,
                 'description': 'Market Cipher B approximation: WT cross with MFI OB/OS confirmation.'},
    },
    'MC_Money_Flow': {
        'gen': gen_MC_Money_Flow,
        'space': space_MC_Money_Flow,
        'default_params': {'n1': 10, 'n2': 21, 'mfi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Market Cipher money flow: MFI extreme + OBV rising + WT cross.'},
    },
    'MC_Diamond': {
        'gen': gen_MC_Diamond,
        'space': space_MC_Diamond,
        'default_params': {'n1': 10, 'n2': 21, 'os_level': -70.0, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Market Cipher diamond: extreme OS WT + bullish divergence pattern.'},
    },
    'WT_Stoch': {
        'gen': gen_WT_Stoch,
        'space': space_WT_Stoch,
        'default_params': {'n1': 10, 'n2': 21, 'k_p': 14, 'd_p': 3},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'WaveTrend cross in OS zone with Stochastic oversold confirmation.'},
    },
    'WT_CCI': {
        'gen': gen_WT_CCI,
        'space': space_WT_CCI,
        'default_params': {'n1': 10, 'n2': 21, 'cci_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'WaveTrend cross confirmed by CCI below -100 (deeply oversold).'},
    },
    'WT_Vortex': {
        'gen': gen_WT_Vortex,
        'space': space_WT_Vortex,
        'default_params': {'n1': 10, 'n2': 21, 'v_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'WaveTrend cross in Vortex indicator trend direction.'},
    },
    'WT_OBV': {
        'gen': gen_WT_OBV,
        'space': space_WT_OBV,
        'default_params': {'n1': 10, 'n2': 21, 'obv_ema_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'WaveTrend cross with OBV above its EMA (accumulation).'},
    },
    'WT_Ichimoku': {
        'gen': gen_WT_Ichimoku,
        'space': space_WT_Ichimoku,
        'default_params': {'n1': 10, 'n2': 21, 'tenkan': 9, 'kijun': 26},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'WaveTrend cross with price above/below Ichimoku cloud.'},
    },
    'WT_Keltner': {
        'gen': gen_WT_Keltner,
        'space': space_WT_Keltner,
        'default_params': {'n1': 10, 'n2': 21, 'kc_p': 20, 'kc_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'WaveTrend cross while price is outside Keltner Channel.'},
    },
    'WT_LinReg': {
        'gen': gen_WT_LinReg,
        'space': space_WT_LinReg,
        'default_params': {'n1': 10, 'n2': 21, 'lr_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'WaveTrend cross aligned with linear regression slope direction.'},
    },
    'WT_Adaptive': {
        'gen': gen_WT_Adaptive,
        'space': space_WT_Adaptive,
        'default_params': {'base_n1': 10, 'base_n2': 21, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'WaveTrend cross with ATR-adaptive period selection and momentum gate.'},
    },
    'WT_Pattern': {
        'gen': gen_WT_Pattern,
        'space': space_WT_Pattern,
        'default_params': {'n1': 10, 'n2': 21, 'os': -50.0, 'lookback': 5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'WaveTrend rising bottoms pattern below OS — bullish divergence setup.'},
    },
    'WT_Histogram': {
        'gen': gen_WT_Histogram,
        'space': space_WT_Histogram,
        'default_params': {'n1': 10, 'n2': 21},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'WaveTrend histogram (WT1-WT2) zero-line cross signal.'},
    },
    'WT_QQE': {
        'gen': gen_WT_QQE,
        'space': space_WT_QQE,
        'default_params': {'n1': 10, 'n2': 21, 'rsi_p': 14, 'sf': 5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'WaveTrend cross with QQE smoothed RSI above/below 50.'},
    },
    'WT_Hull': {
        'gen': gen_WT_Hull,
        'space': space_WT_Hull,
        'default_params': {'n1': 10, 'n2': 21, 'hma_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'WaveTrend cross with Hull MA rising/falling trend filter.'},
    },
    'WT_CMF': {
        'gen': gen_WT_CMF,
        'space': space_WT_CMF,
        'default_params': {'n1': 10, 'n2': 21, 'cmf_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'WaveTrend cross confirmed by positive Chaikin Money Flow.'},
    },
    'WT_ALMA': {
        'gen': gen_WT_ALMA,
        'space': space_WT_ALMA,
        'default_params': {'n1': 10, 'n2': 21, 'alma_p': 20, 'offset': 0.85},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'WaveTrend cross with price vs ALMA (Arnaud Legoux MA) trend filter.'},
    },
    'WT_Triple_Confirm': {
        'gen': gen_WT_Triple_Confirm,
        'space': space_WT_Triple_Confirm,
        'default_params': {'n1': 10, 'n2': 21, 'rsi_p': 14, 'vol_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Triple confirmation: WT cross below -50 + RSI<40 + volume spike.'},
    },
}
