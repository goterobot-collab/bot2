#!/usr/bin/env python3
"""TV2 BATCH 29 — 30 Mean Reversion Strategies 2026-04-01"""

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


def _bb(s, p, mult):
    mid   = _sma(s, p)
    std   = s.rolling(int(p), min_periods=1).std().fillna(0)
    upper = mid + mult * std
    lower = mid - mult * std
    return upper, mid, lower


def _stoch(df, k):
    p = int(k)
    lo = df['low'].rolling(p, min_periods=1).min()
    hi = df['high'].rolling(p, min_periods=1).max()
    return 100 * (df['close'] - lo) / (hi - lo + 1e-9)


def _cci(df, p):
    tp  = (df['high'] + df['low'] + df['close']) / 3
    sma = _sma(tp, p)
    mad = tp.rolling(int(p), min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
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


def _adx(df, p):
    p   = int(p)
    hi  = df['high']
    lo  = df['low']
    cl  = df['close']
    tr  = _atr(df, p)
    dmp = (hi.diff().clip(lower=0)).where(hi.diff() > (-lo.diff()).clip(lower=0), 0)
    dmn = ((-lo.diff()).clip(lower=0)).where((-lo.diff()) > hi.diff().clip(lower=0), 0)
    dip = 100 * _rma(dmp, p) / tr.replace(0, 1e-9)
    din = 100 * _rma(dmn, p) / tr.replace(0, 1e-9)
    dx  = 100 * (dip - din).abs() / (dip + din).replace(0, 1e-9)
    return _rma(dx, p), dip, din


def _donchian(df, p):
    return (df['high'].rolling(int(p), min_periods=1).max(),
            df['low'].rolling(int(p), min_periods=1).min())


def _supertrend(df, p, mult):
    p    = int(p)
    atr  = _atr(df, p)
    hl2  = (df['high'] + df['low']) / 2
    up   = hl2 - mult * atr
    dn   = hl2 + mult * atr
    cl   = df['close']
    trend = pd.Series(1, index=df.index)
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


def _linreg(s, p):
    """Rolling linear regression last value."""
    p = int(p)
    out = pd.Series(np.nan, index=s.index)
    arr = s.values
    for i in range(p - 1, len(arr)):
        y = arr[i - p + 1: i + 1]
        x = np.arange(p)
        m, b = np.polyfit(x, y, 1)
        out.iloc[i] = m * (p - 1) + b
    return out


def _vwap_rolling(df, p):
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).rolling(int(p), min_periods=1).sum() / \
           df['volume'].rolling(int(p), min_periods=1).sum().replace(0, 1e-9)


# ── 1. MR_RSI_BB ─────────────────────────────────────────────────────────────

def gen_MR_RSI_BB(df, rsi_p=14, bb_p=20, bb_mult=2.0, **kw):
    rsi         = _rsi(df['close'], rsi_p)
    _, _, lower = _bb(df['close'], bb_p, bb_mult)
    _, _, upper_s = _bb(df['close'], bb_p, bb_mult)
    upper, _, _ = _bb(df['close'], bb_p, bb_mult)
    sig = pd.Series(0, index=df.index)
    sig[rsi < 30]               = 1
    sig[(rsi < 30) & (df['close'] < lower)] = 1
    sig[(rsi > 70) & (df['close'] > upper)] = -1
    return sig


def space_MR_RSI_BB(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'bb_p':     trial.suggest_int('bb_p', 20, 20),
        'bb_mult':  trial.suggest_float('bb_mult', 1.5, 3.0),
    }


# ── 2. MR_Zscore_EMA ─────────────────────────────────────────────────────────

def gen_MR_Zscore_EMA(df, z_p=30, thresh=1.5, ema_p=100, **kw):
    cl    = df['close']
    mu    = _sma(cl, z_p)
    std   = cl.rolling(int(z_p), min_periods=1).std().fillna(1e-9)
    z     = (cl - mu) / std
    trend = _ema(cl, ema_p)
    sig   = pd.Series(0, index=df.index)
    sig[(z < -thresh) & (cl > trend)] = 1
    sig[(z >  thresh) & (cl < trend)] = -1
    return sig


def space_MR_Zscore_EMA(trial):
    return {
        'z_p':    trial.suggest_int('z_p', 20, 60),
        'thresh': trial.suggest_float('thresh', 1.0, 2.5),
        'ema_p':  trial.suggest_int('ema_p', 50, 200),
    }


# ── 3. MR_BB_Squeeze_Rev ─────────────────────────────────────────────────────

def gen_MR_BB_Squeeze_Rev(df, bb_p=20, bb_mult=2.0, look_p=10, **kw):
    _, _, lower = _bb(df['close'], bb_p, bb_mult)
    upper, _, _ = _bb(df['close'], bb_p, bb_mult)
    width = upper - lower
    min_w = width.rolling(int(look_p), min_periods=1).min()
    squeezed = width.shift(1) <= min_w.shift(1) * 1.05
    sig = pd.Series(0, index=df.index)
    sig[squeezed & (df['close'] > df['close'].shift(1))] = 1
    sig[squeezed & (df['close'] < df['close'].shift(1))] = -1
    return sig


def space_MR_BB_Squeeze_Rev(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 20, 20),
        'bb_mult': trial.suggest_float('bb_mult', 2.0, 2.0),
        'look_p':  trial.suggest_int('look_p', 5, 20),
    }


# ── 4. MR_ATR_Channel ────────────────────────────────────────────────────────

def gen_MR_ATR_Channel(df, ema_p=40, atr_p=14, atr_mult=3.0, **kw):
    basis = _ema(df['close'], ema_p)
    atr   = _atr(df, atr_p)
    lower = basis - atr_mult * atr
    upper = basis + atr_mult * atr
    sig   = pd.Series(0, index=df.index)
    sig[df['close'] < lower] =  1
    sig[df['close'] > upper] = -1
    return sig


def space_MR_ATR_Channel(trial):
    return {
        'ema_p':    trial.suggest_int('ema_p', 20, 60),
        'atr_p':    trial.suggest_int('atr_p', 10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 2.0, 4.0),
    }


# ── 5. MR_Percentile_Rev ─────────────────────────────────────────────────────

def gen_MR_Percentile_Rev(df, per_p=50, threshold=10, **kw):
    cl  = df['close']
    pct_low  = cl.rolling(int(per_p), min_periods=1).quantile(threshold / 100)
    pct_high = cl.rolling(int(per_p), min_periods=1).quantile(1 - threshold / 100)
    sig = pd.Series(0, index=df.index)
    sig[cl < pct_low]  =  1
    sig[cl > pct_high] = -1
    return sig


def space_MR_Percentile_Rev(trial):
    return {
        'per_p':     trial.suggest_int('per_p', 20, 100),
        'threshold': trial.suggest_int('threshold', 5, 20),
    }


# ── 6. MR_RSI_Stoch_BB ───────────────────────────────────────────────────────

def gen_MR_RSI_Stoch_BB(df, rsi_p=14, stoch_k=14, bb_p=20, bb_mult=2.0, **kw):
    rsi         = _rsi(df['close'], rsi_p)
    stoch       = _stoch(df, stoch_k)
    _, _, lower = _bb(df['close'], bb_p, bb_mult)
    upper, _, _ = _bb(df['close'], bb_p, bb_mult)
    sig = pd.Series(0, index=df.index)
    sig[(rsi < 30) & (stoch < 20) & (df['close'] < lower)] =  1
    sig[(rsi > 70) & (stoch > 80) & (df['close'] > upper)] = -1
    return sig


def space_MR_RSI_Stoch_BB(trial):
    return {
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
        'stoch_k': trial.suggest_int('stoch_k', 7, 21),
        'bb_p':    trial.suggest_int('bb_p', 20, 20),
        'bb_mult': trial.suggest_float('bb_mult', 2.0, 2.0),
    }


# ── 7. MR_OBV_RSI ────────────────────────────────────────────────────────────

def gen_MR_OBV_RSI(df, rsi_p=14, obv_ema_p=20, **kw):
    rsi     = _rsi(df['close'], rsi_p)
    obv     = _obv(df)
    obv_ma  = _ema(obv, obv_ema_p)
    cl      = df['close']
    cl_low  = cl < cl.shift(1)
    obv_ok  = obv >= obv_ma
    sig = pd.Series(0, index=df.index)
    sig[cl_low & obv_ok & (rsi < 35)] =  1
    sig[(~cl_low) & (~obv_ok) & (rsi > 65)] = -1
    return sig


def space_MR_OBV_RSI(trial):
    return {
        'rsi_p':     trial.suggest_int('rsi_p', 7, 21),
        'obv_ema_p': trial.suggest_int('obv_ema_p', 10, 30),
    }


# ── 8. MR_BB_Volume ──────────────────────────────────────────────────────────

def gen_MR_BB_Volume(df, bb_p=20, bb_mult=2.0, vol_p=20, vol_mult=3.0, **kw):
    _, _, lower = _bb(df['close'], bb_p, bb_mult)
    upper, _, _ = _bb(df['close'], bb_p, bb_mult)
    vol_ma = _sma(df['volume'], vol_p)
    vol_surge = df['volume'] > vol_ma * vol_mult
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & vol_surge] =  1
    sig[(df['close'] > upper) & vol_surge] = -1
    return sig


def space_MR_BB_Volume(trial):
    return {
        'bb_p':     trial.suggest_int('bb_p', 20, 20),
        'bb_mult':  trial.suggest_float('bb_mult', 2.0, 2.0),
        'vol_p':    trial.suggest_int('vol_p', 20, 50),
        'vol_mult': trial.suggest_float('vol_mult', 2.0, 5.0),
    }


# ── 9. MR_Williams_R ─────────────────────────────────────────────────────────

def gen_MR_Williams_R(df, wr_p=14, ema_p=200, **kw):
    p     = int(wr_p)
    hi    = df['high'].rolling(p, min_periods=1).max()
    lo    = df['low'].rolling(p, min_periods=1).min()
    wr    = -100 * (hi - df['close']) / (hi - lo + 1e-9)
    ema   = _ema(df['close'], ema_p)
    sig   = pd.Series(0, index=df.index)
    sig[(wr < -90) & (df['close'] > ema)] =  1
    sig[(wr > -10) & (df['close'] < ema)] = -1
    return sig


def space_MR_Williams_R(trial):
    return {
        'wr_p':  trial.suggest_int('wr_p', 7, 21),
        'ema_p': trial.suggest_int('ema_p', 150, 250),
    }


# ── 10. MR_Zscore_Vol ────────────────────────────────────────────────────────

def gen_MR_Zscore_Vol(df, z_p=30, thresh=2.0, vol_p=20, **kw):
    cl   = df['close']
    mu   = _sma(cl, z_p)
    std  = cl.rolling(int(z_p), min_periods=1).std().fillna(1e-9)
    z    = (cl - mu) / std
    vma  = _sma(df['volume'], vol_p)
    vol_ok = df['volume'] > vma
    sig  = pd.Series(0, index=df.index)
    sig[(z < -thresh) & vol_ok] =  1
    sig[(z >  thresh) & vol_ok] = -1
    return sig


def space_MR_Zscore_Vol(trial):
    return {
        'z_p':    trial.suggest_int('z_p', 20, 60),
        'thresh': trial.suggest_float('thresh', 1.5, 3.0),
        'vol_p':  trial.suggest_int('vol_p', 20, 50),
    }


# ── 11. MR_CCI_BB ────────────────────────────────────────────────────────────

def gen_MR_CCI_BB(df, cci_p=14, bb_p=20, bb_mult=2.5, **kw):
    cci         = _cci(df, cci_p)
    _, _, lower = _bb(df['close'], bb_p, bb_mult)
    upper, _, _ = _bb(df['close'], bb_p, bb_mult)
    sig = pd.Series(0, index=df.index)
    sig[(cci < -200) & (df['close'] < lower)] =  1
    sig[(cci >  200) & (df['close'] > upper)] = -1
    return sig


def space_MR_CCI_BB(trial):
    return {
        'cci_p':   trial.suggest_int('cci_p', 10, 20),
        'bb_p':    trial.suggest_int('bb_p', 20, 20),
        'bb_mult': trial.suggest_float('bb_mult', 2.5, 2.5),
    }


# ── 12. MR_Keltner_Rev ───────────────────────────────────────────────────────

def gen_MR_Keltner_Rev(df, kc_p=20, kc_mult=3.0, **kw):
    mid   = _ema(df['close'], kc_p)
    atr   = _atr(df, kc_p)
    lower = mid - kc_mult * atr
    upper = mid + kc_mult * atr
    sig   = pd.Series(0, index=df.index)
    sig[df['close'] < lower] =  1
    sig[df['close'] > upper] = -1
    return sig


def space_MR_Keltner_Rev(trial):
    return {
        'kc_p':    trial.suggest_int('kc_p', 20, 20),
        'kc_mult': trial.suggest_float('kc_mult', 2.0, 4.0),
    }


# ── 13. MR_VWAP_RSI ──────────────────────────────────────────────────────────

def gen_MR_VWAP_RSI(df, vwap_p=50, rsi_p=14, **kw):
    vwap = _vwap_rolling(df, vwap_p)
    rsi  = _rsi(df['close'], rsi_p)
    sig  = pd.Series(0, index=df.index)
    sig[(df['close'] < vwap) & (rsi < 35)] =  1
    sig[(df['close'] > vwap) & (rsi > 65)] = -1
    return sig


def space_MR_VWAP_RSI(trial):
    return {
        'vwap_p': trial.suggest_int('vwap_p', 20, 100),
        'rsi_p':  trial.suggest_int('rsi_p', 7, 21),
    }


# ── 14. MR_ATR_Spike ─────────────────────────────────────────────────────────

def gen_MR_ATR_Spike(df, atr_p=14, spike_mult=3.0, **kw):
    atr    = _atr(df, atr_p)
    atr_ma = _sma(atr, atr_p)
    spike  = atr > atr_ma * spike_mult
    bear   = df['close'] < df['open']
    bull   = df['close'] > df['open']
    sig    = pd.Series(0, index=df.index)
    sig[spike.shift(1, fill_value=False) & bear.shift(1, fill_value=False) & bull] =  1
    sig[spike.shift(1, fill_value=False) & bull.shift(1, fill_value=False) & bear] = -1
    return sig


def space_MR_ATR_Spike(trial):
    return {
        'atr_p':      trial.suggest_int('atr_p', 10, 20),
        'spike_mult': trial.suggest_float('spike_mult', 2.0, 5.0),
    }


# ── 15. MR_Channel_Fade ──────────────────────────────────────────────────────

def gen_MR_Channel_Fade(df, dc_p=20, **kw):
    hi, lo = _donchian(df, dc_p)
    sig    = pd.Series(0, index=df.index)
    sig[df['close'] <= lo] =  1
    sig[df['close'] >= hi] = -1
    return sig


def space_MR_Channel_Fade(trial):
    return {
        'dc_p': trial.suggest_int('dc_p', 10, 50),
    }


# ── 16. MR_RSI2_Pullback ─────────────────────────────────────────────────────

def gen_MR_RSI2_Pullback(df, rsi_p=2, ema_p=200, **kw):
    rsi  = _rsi(df['close'], rsi_p)
    ema  = _ema(df['close'], ema_p)
    sig  = pd.Series(0, index=df.index)
    sig[(rsi < 10) & (df['close'] > ema)] =  1
    sig[(rsi > 90) & (df['close'] < ema)] = -1
    return sig


def space_MR_RSI2_Pullback(trial):
    return {
        'rsi_p': trial.suggest_int('rsi_p', 2, 5),
        'ema_p': trial.suggest_int('ema_p', 150, 250),
    }


# ── 17. MR_BB_RSI_MFI ────────────────────────────────────────────────────────

def gen_MR_BB_RSI_MFI(df, rsi_p=14, bb_p=20, mfi_p=14, **kw):
    rsi         = _rsi(df['close'], rsi_p)
    mfi         = _mfi(df, mfi_p)
    _, _, lower = _bb(df['close'], bb_p, 2.0)
    upper, _, _ = _bb(df['close'], bb_p, 2.0)
    sig = pd.Series(0, index=df.index)
    sig[(rsi < 30) & (df['close'] < lower) & (mfi < 30)] =  1
    sig[(rsi > 70) & (df['close'] > upper) & (mfi > 70)] = -1
    return sig


def space_MR_BB_RSI_MFI(trial):
    return {
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
        'bb_p':  trial.suggest_int('bb_p', 20, 20),
        'mfi_p': trial.suggest_int('mfi_p', 7, 21),
    }


# ── 18. MR_Momentum_Rev ──────────────────────────────────────────────────────

def gen_MR_Momentum_Rev(df, roc_p=10, rev_pct=5.0, rsi_p=14, **kw):
    roc = df['close'].pct_change(int(roc_p)) * 100
    rsi = _rsi(df['close'], rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[(roc < -rev_pct) & (rsi < 35)] =  1
    sig[(roc >  rev_pct) & (rsi > 65)] = -1
    return sig


def space_MR_Momentum_Rev(trial):
    return {
        'roc_p':   trial.suggest_int('roc_p', 5, 20),
        'rev_pct': trial.suggest_float('rev_pct', 3.0, 10.0),
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
    }


# ── 19. MR_HA_Lower_BB ───────────────────────────────────────────────────────

def gen_MR_HA_Lower_BB(df, bb_p=20, bb_mult=2.0, **kw):
    o  = df['open']
    h  = df['high']
    l  = df['low']
    c  = df['close']
    ha_c = (o + h + l + c) / 4
    ha_o = ha_c.copy()
    for i in range(1, len(ha_o)):
        ha_o.iloc[i] = (ha_o.iloc[i - 1] + ha_c.iloc[i - 1]) / 2
    _, _, lower = _bb(ha_c, bb_p, bb_mult)
    upper, _, _ = _bb(ha_c, bb_p, bb_mult)
    sig = pd.Series(0, index=df.index)
    sig[ha_c < lower] =  1
    sig[ha_c > upper] = -1
    return sig


def space_MR_HA_Lower_BB(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 20, 20),
        'bb_mult': trial.suggest_float('bb_mult', 2.0, 2.0),
    }


# ── 20. MR_Linear_Dev ────────────────────────────────────────────────────────

def gen_MR_Linear_Dev(df, lr_p=30, dev_mult=2.0, **kw):
    cl    = df['close']
    p     = int(lr_p)
    lr    = cl.rolling(p, min_periods=p).apply(
        lambda x: np.polyval(np.polyfit(np.arange(len(x)), x, 1), len(x) - 1), raw=True)
    std   = cl.rolling(p, min_periods=p).std().fillna(0)
    lower = lr - dev_mult * std
    upper = lr + dev_mult * std
    sig   = pd.Series(0, index=df.index)
    sig[cl < lower] =  1
    sig[cl > upper] = -1
    return sig


def space_MR_Linear_Dev(trial):
    return {
        'lr_p':     trial.suggest_int('lr_p', 20, 60),
        'dev_mult': trial.suggest_float('dev_mult', 1.5, 3.0),
    }


# ── 21. MR_Dual_BB ───────────────────────────────────────────────────────────

def gen_MR_Dual_BB(df, bb_p=20, inner_mult=1.5, outer_mult=3.0, **kw):
    _, _, inner_l = _bb(df['close'], bb_p, inner_mult)
    _, _, outer_l = _bb(df['close'], bb_p, outer_mult)
    inner_u, _, _ = _bb(df['close'], bb_p, inner_mult)
    outer_u, _, _ = _bb(df['close'], bb_p, outer_mult)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < inner_l) & (df['close'] > outer_l)] =  1
    sig[(df['close'] > inner_u) & (df['close'] < outer_u)] = -1
    return sig


def space_MR_Dual_BB(trial):
    return {
        'bb_p':       trial.suggest_int('bb_p', 20, 20),
        'inner_mult': trial.suggest_float('inner_mult', 1.0, 2.0),
        'outer_mult': trial.suggest_float('outer_mult', 2.5, 4.0),
    }


# ── 22. MR_RSI_Pattern ───────────────────────────────────────────────────────

def gen_MR_RSI_Pattern(df, rsi_p=14, dip=20, recover=35, **kw):
    rsi = _rsi(df['close'], rsi_p)
    was_low  = rsi.rolling(5, min_periods=1).min() < dip
    sig = pd.Series(0, index=df.index)
    sig[was_low & (rsi > recover)] = 1
    was_high = rsi.rolling(5, min_periods=1).max() > (100 - dip)
    sig[was_high & (rsi < (100 - recover))] = -1
    return sig


def space_MR_RSI_Pattern(trial):
    return {
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
        'dip':     trial.suggest_int('dip', 15, 25),
        'recover': trial.suggest_int('recover', 30, 45),
    }


# ── 23. MR_OBV_Price_Div ─────────────────────────────────────────────────────

def gen_MR_OBV_Price_Div(df, lookback=10, **kw):
    cl    = df['close']
    obv   = _obv(df)
    p     = int(lookback)
    price_down = cl < cl.shift(p)
    obv_stable = obv >= obv.shift(p) * 0.98
    sig = pd.Series(0, index=df.index)
    sig[price_down & obv_stable] = 1
    price_up = cl > cl.shift(p)
    obv_weak = obv <= obv.shift(p) * 1.02
    sig[price_up & obv_weak] = -1
    return sig


def space_MR_OBV_Price_Div(trial):
    return {
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 24. MR_ADX_Low ───────────────────────────────────────────────────────────

def gen_MR_ADX_Low(df, adx_p=14, rsi_p=14, **kw):
    adx, _, _ = _adx(df, adx_p)
    rsi       = _rsi(df['close'], rsi_p)
    ranging   = adx < 20
    sig = pd.Series(0, index=df.index)
    sig[ranging & (rsi < 30)] =  1
    sig[ranging & (rsi > 70)] = -1
    return sig


def space_MR_ADX_Low(trial):
    return {
        'adx_p': trial.suggest_int('adx_p', 10, 20),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 25. MR_BB_Width_Low ──────────────────────────────────────────────────────

def gen_MR_BB_Width_Low(df, bb_p=20, bb_mult=2.0, width_p=100, rsi_p=14, **kw):
    upper, mid, lower = _bb(df['close'], bb_p, bb_mult)
    width    = upper - lower
    pct_low  = width.rolling(int(width_p), min_periods=1).quantile(0.25)
    narrow   = width < pct_low
    rsi      = _rsi(df['close'], rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[narrow & (rsi < 35)] =  1
    sig[narrow & (rsi > 65)] = -1
    return sig


def space_MR_BB_Width_Low(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 20, 20),
        'bb_mult': trial.suggest_float('bb_mult', 2.0, 2.0),
        'width_p': trial.suggest_int('width_p', 50, 200),
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
    }


# ── 26. MR_EMA_Touch ─────────────────────────────────────────────────────────

def gen_MR_EMA_Touch(df, ema_p=50, **kw):
    ema   = _ema(df['close'], ema_p)
    cl    = df['close']
    op    = df['open']
    touched_from_below = (cl.shift(1) < ema.shift(1)) & (cl >= ema)
    green = cl > op
    touched_from_above = (cl.shift(1) > ema.shift(1)) & (cl <= ema)
    red   = cl < op
    sig   = pd.Series(0, index=df.index)
    sig[touched_from_below & green] =  1
    sig[touched_from_above & red]   = -1
    return sig


def space_MR_EMA_Touch(trial):
    return {
        'ema_p': trial.suggest_int('ema_p', 20, 100),
    }


# ── 27. MR_Extremes_Vol ──────────────────────────────────────────────────────

def gen_MR_Extremes_Vol(df, low_p=20, vol_p=20, vol_mult=1.5, **kw):
    cl     = df['close']
    lo     = cl.rolling(int(low_p), min_periods=1).min()
    hi     = cl.rolling(int(low_p), min_periods=1).max()
    vol_ma = _sma(df['volume'], vol_p)
    vol_ok = df['volume'] > vol_ma * vol_mult
    sig    = pd.Series(0, index=df.index)
    sig[(cl == lo) & vol_ok] =  1
    sig[(cl == hi) & vol_ok] = -1
    return sig


def space_MR_Extremes_Vol(trial):
    return {
        'low_p':    trial.suggest_int('low_p', 10, 30),
        'vol_p':    trial.suggest_int('vol_p', 20, 50),
        'vol_mult': trial.suggest_float('vol_mult', 1.2, 2.5),
    }


# ── 28. MR_Fibonacci_Rev ─────────────────────────────────────────────────────

def gen_MR_Fibonacci_Rev(df, swing_p=30, fib_level=0.618, **kw):
    p    = int(swing_p)
    hi   = df['high'].rolling(p, min_periods=1).max()
    lo   = df['low'].rolling(p, min_periods=1).min()
    rng  = hi - lo
    fib_sup = hi - fib_level * rng
    fib_res = lo + fib_level * rng
    tol  = rng * 0.02
    cl   = df['close']
    sig  = pd.Series(0, index=df.index)
    sig[(cl >= fib_sup - tol) & (cl <= fib_sup + tol)] =  1
    sig[(cl >= fib_res - tol) & (cl <= fib_res + tol)] = -1
    return sig


def space_MR_Fibonacci_Rev(trial):
    return {
        'swing_p':   trial.suggest_int('swing_p', 20, 60),
        'fib_level': trial.suggest_float('fib_level', 0.3, 0.7),
    }


# ── 29. MR_SuperTrend_Rev ────────────────────────────────────────────────────

def gen_MR_SuperTrend_Rev(df, st_p=10, st_mult=3.0, rsi_p=14, **kw):
    trend = _supertrend(df, st_p, st_mult)
    rsi   = _rsi(df['close'], rsi_p)
    sig   = pd.Series(0, index=df.index)
    sig[(trend == -1) & (rsi < 25)] =  1
    sig[(trend ==  1) & (rsi > 75)] = -1
    return sig


def space_MR_SuperTrend_Rev(trial):
    return {
        'st_p':    trial.suggest_int('st_p', 7, 21),
        'st_mult': trial.suggest_float('st_mult', 2.0, 4.0),
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
    }


# ── 30. MR_Candle_Reversal ───────────────────────────────────────────────────

def gen_MR_Candle_Reversal(df, body_mult=2.0, ema_p=20, **kw):
    cl   = df['close']
    op   = df['open']
    body = (cl - op).abs()
    avg_body = _sma(body, ema_p)
    big_bear_prev = (op.shift(1) > cl.shift(1)) & (body.shift(1) > avg_body.shift(1) * body_mult)
    small_body    = body < avg_body
    green         = cl > op
    big_bull_prev = (cl.shift(1) > op.shift(1)) & (body.shift(1) > avg_body.shift(1) * body_mult)
    red           = cl < op
    sig = pd.Series(0, index=df.index)
    sig[big_bear_prev & small_body & green] =  1
    sig[big_bull_prev & small_body & red]   = -1
    return sig


def space_MR_Candle_Reversal(trial):
    return {
        'body_mult': trial.suggest_float('body_mult', 1.5, 3.5),
        'ema_p':     trial.suggest_int('ema_p', 14, 50),
    }


# ── STRATEGY_EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'MR_RSI_BB': {
        'gen': gen_MR_RSI_BB,
        'space': space_MR_RSI_BB,
        'default_params': {'rsi_p': 14, 'bb_p': 20, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3200,
                 'description': 'RSI<30 + price below Bollinger lower band dual oversold mean reversion.'},
    },
    'MR_Zscore_EMA': {
        'gen': gen_MR_Zscore_EMA,
        'space': space_MR_Zscore_EMA,
        'default_params': {'z_p': 30, 'thresh': 1.5, 'ema_p': 100},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2800,
                 'description': 'Z-score extremes in EMA trend direction for mean reversion entries.'},
    },
    'MR_BB_Squeeze_Rev': {
        'gen': gen_MR_BB_Squeeze_Rev,
        'space': space_MR_BB_Squeeze_Rev,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'look_p': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2400,
                 'description': 'Bollinger Band squeeze release directional entry on expansion.'},
    },
    'MR_ATR_Channel': {
        'gen': gen_MR_ATR_Channel,
        'space': space_MR_ATR_Channel,
        'default_params': {'ema_p': 40, 'atr_p': 14, 'atr_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2600,
                 'description': 'Price outside EMA±ATR*mult dynamic channel mean reversion.'},
    },
    'MR_Percentile_Rev': {
        'gen': gen_MR_Percentile_Rev,
        'space': space_MR_Percentile_Rev,
        'default_params': {'per_p': 50, 'threshold': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2200,
                 'description': 'Price at rolling Nth percentile extreme signals mean reversion.'},
    },
    'MR_RSI_Stoch_BB': {
        'gen': gen_MR_RSI_Stoch_BB,
        'space': space_MR_RSI_Stoch_BB,
        'default_params': {'rsi_p': 14, 'stoch_k': 14, 'bb_p': 20, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Triple oscillator oversold: RSI + Stochastic + BB lower touch.'},
    },
    'MR_OBV_RSI': {
        'gen': gen_MR_OBV_RSI,
        'space': space_MR_OBV_RSI,
        'default_params': {'rsi_p': 14, 'obv_ema_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2100,
                 'description': 'OBV divergence (price lower, OBV stable) + RSI oversold reversal.'},
    },
    'MR_BB_Volume': {
        'gen': gen_MR_BB_Volume,
        'space': space_MR_BB_Volume,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'vol_p': 20, 'vol_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2900,
                 'description': 'BB lower touch + volume surge (selling climax) reversal signal.'},
    },
    'MR_Williams_R': {
        'gen': gen_MR_Williams_R,
        'space': space_MR_Williams_R,
        'default_params': {'wr_p': 14, 'ema_p': 200},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2300,
                 'description': 'Williams %R extreme (<-90) in EMA trend direction mean reversion.'},
    },
    'MR_Zscore_Vol': {
        'gen': gen_MR_Zscore_Vol,
        'space': space_MR_Zscore_Vol,
        'default_params': {'z_p': 30, 'thresh': 2.0, 'vol_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Z-score extreme + volume confirmation for mean reversion entries.'},
    },
    'MR_CCI_BB': {
        'gen': gen_MR_CCI_BB,
        'space': space_MR_CCI_BB,
        'default_params': {'cci_p': 14, 'bb_p': 20, 'bb_mult': 2.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2700,
                 'description': 'CCI<-200 extreme + price below BB lower band triple oversold.'},
    },
    'MR_Keltner_Rev': {
        'gen': gen_MR_Keltner_Rev,
        'space': space_MR_Keltner_Rev,
        'default_params': {'kc_p': 20, 'kc_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2400,
                 'description': 'Price below Keltner Channel lower band mean reversion.'},
    },
    'MR_VWAP_RSI': {
        'gen': gen_MR_VWAP_RSI,
        'space': space_MR_VWAP_RSI,
        'default_params': {'vwap_p': 50, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3100,
                 'description': 'VWAP dip + RSI oversold for intraday mean reversion long entries.'},
    },
    'MR_ATR_Spike': {
        'gen': gen_MR_ATR_Spike,
        'space': space_MR_ATR_Spike,
        'default_params': {'atr_p': 14, 'spike_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1900,
                 'description': 'ATR spike (panic sell) followed by reversal candle entry.'},
    },
    'MR_Channel_Fade': {
        'gen': gen_MR_Channel_Fade,
        'space': space_MR_Channel_Fade,
        'default_params': {'dc_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2200,
                 'description': 'Fade Donchian channel extremes: long at lower, short at upper.'},
    },
    'MR_RSI2_Pullback': {
        'gen': gen_MR_RSI2_Pullback,
        'space': space_MR_RSI2_Pullback,
        'default_params': {'rsi_p': 2, 'ema_p': 200},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4200,
                 'description': 'RSI(2)<10 pullback in EMA uptrend — Larry Connors RSI-2 strategy.'},
    },
    'MR_BB_RSI_MFI': {
        'gen': gen_MR_BB_RSI_MFI,
        'space': space_MR_BB_RSI_MFI,
        'default_params': {'rsi_p': 14, 'bb_p': 20, 'mfi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2600,
                 'description': 'Triple oversold: RSI<30 + BB lower + MFI<30 mean reversion.'},
    },
    'MR_Momentum_Rev': {
        'gen': gen_MR_Momentum_Rev,
        'space': space_MR_Momentum_Rev,
        'default_params': {'roc_p': 10, 'rev_pct': 5.0, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'ROC extreme decline + RSI oversold momentum reversal signal.'},
    },
    'MR_HA_Lower_BB': {
        'gen': gen_MR_HA_Lower_BB,
        'space': space_MR_HA_Lower_BB,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1800,
                 'description': 'Heikin-Ashi smoothed candle below Bollinger lower band reversal.'},
    },
    'MR_Linear_Dev': {
        'gen': gen_MR_Linear_Dev,
        'space': space_MR_Linear_Dev,
        'default_params': {'lr_p': 30, 'dev_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2100,
                 'description': 'Price deviation from linear regression line mean reversion.'},
    },
    'MR_Dual_BB': {
        'gen': gen_MR_Dual_BB,
        'space': space_MR_Dual_BB,
        'default_params': {'bb_p': 20, 'inner_mult': 1.5, 'outer_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2800,
                 'description': 'Dual Bollinger Bands: price between inner and outer lower bands.'},
    },
    'MR_RSI_Pattern': {
        'gen': gen_MR_RSI_Pattern,
        'space': space_MR_RSI_Pattern,
        'default_params': {'rsi_p': 14, 'dip': 20, 'recover': 35},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2300,
                 'description': 'RSI V-shape: dips below threshold then recovers above recover level.'},
    },
    'MR_OBV_Price_Div': {
        'gen': gen_MR_OBV_Price_Div,
        'space': space_MR_OBV_Price_Div,
        'default_params': {'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1900,
                 'description': 'OBV-price divergence: price declining while OBV stable = reversal.'},
    },
    'MR_ADX_Low': {
        'gen': gen_MR_ADX_Low,
        'space': space_MR_ADX_Low,
        'default_params': {'adx_p': 14, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Low ADX (ranging market) + RSI extreme = mean reversion in range.'},
    },
    'MR_BB_Width_Low': {
        'gen': gen_MR_BB_Width_Low,
        'space': space_MR_BB_Width_Low,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'width_p': 100, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Narrow BB width (low volatility squeeze) + RSI edge = coiled spring.'},
    },
    'MR_EMA_Touch': {
        'gen': gen_MR_EMA_Touch,
        'space': space_MR_EMA_Touch,
        'default_params': {'ema_p': 50},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2700,
                 'description': 'Price touches EMA from below with reversal candle = bounce entry.'},
    },
    'MR_Extremes_Vol': {
        'gen': gen_MR_Extremes_Vol,
        'space': space_MR_Extremes_Vol,
        'default_params': {'low_p': 20, 'vol_p': 20, 'vol_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1700,
                 'description': 'Price at N-bar extreme with volume surge confirms reversal.'},
    },
    'MR_Fibonacci_Rev': {
        'gen': gen_MR_Fibonacci_Rev,
        'space': space_MR_Fibonacci_Rev,
        'default_params': {'swing_p': 30, 'fib_level': 0.618},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3400,
                 'description': 'Price retracement to Fibonacci 0.382/0.618 swing levels mean reversion.'},
    },
    'MR_SuperTrend_Rev': {
        'gen': gen_MR_SuperTrend_Rev,
        'space': space_MR_SuperTrend_Rev,
        'default_params': {'st_p': 10, 'st_mult': 3.0, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2200,
                 'description': 'Counter-trend: SuperTrend bearish + RSI<25 extreme oversold reversal.'},
    },
    'MR_Candle_Reversal': {
        'gen': gen_MR_Candle_Reversal,
        'space': space_MR_Candle_Reversal,
        'default_params': {'body_mult': 2.0, 'ema_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2900,
                 'description': 'Large bearish bar + small bullish inside bar = candle reversal pattern.'},
    },
}
