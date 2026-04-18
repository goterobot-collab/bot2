#!/usr/bin/env python3
"""TV2 BATCH 30 — 30 Advanced Oscillator Combo Strategies 2026-04-01"""

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


def _macd(s, fast, slow, sig):
    f = _ema(s, fast)
    sl = _ema(s, slow)
    m = f - sl
    sg = _ema(m, sig)
    return m, sg


def _cci(df, p):
    tp  = (df['high'] + df['low'] + df['close']) / 3
    sma = _sma(tp, p)
    mad = tp.rolling(int(p), min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, 1e-9))


def _stoch(df, k, d=3):
    p  = int(k)
    lo = df['low'].rolling(p, min_periods=1).min()
    hi = df['high'].rolling(p, min_periods=1).max()
    K  = 100 * (df['close'] - lo) / (hi - lo + 1e-9)
    D  = _sma(K, d)
    return K, D


def _williams_r(df, p):
    p  = int(p)
    hi = df['high'].rolling(p, min_periods=1).max()
    lo = df['low'].rolling(p, min_periods=1).min()
    return -100 * (hi - df['close']) / (hi - lo + 1e-9)


def _tsi(s, long_p, short_p):
    d   = s.diff()
    ema1_d   = _ema(d,       long_p)
    ema2_d   = _ema(ema1_d,  short_p)
    ema1_ad  = _ema(d.abs(), long_p)
    ema2_ad  = _ema(ema1_ad, short_p)
    return 100 * ema2_d / ema2_ad.replace(0, 1e-9)


def _ao(df, fast=5, slow=34):
    hl2 = (df['high'] + df['low']) / 2
    return _sma(hl2, fast) - _sma(hl2, slow)


def _cmo(s, p):
    d   = s.diff()
    up  = d.clip(lower=0).rolling(int(p), min_periods=1).sum()
    dn  = (-d).clip(lower=0).rolling(int(p), min_periods=1).sum()
    return 100 * (up - dn) / (up + dn).replace(0, 1e-9)


def _dpo(s, p):
    p  = int(p)
    return s.shift(p // 2 + 1) - _sma(s, p)


def _uo(df, p1=7, p2=14, p3=28):
    pc   = df['close'].shift(1)
    bp   = df['close'] - pd.concat([df['low'], pc], axis=1).min(axis=1)
    tr   = pd.concat([df['high'], pc], axis=1).max(axis=1) - \
           pd.concat([df['low'],  pc], axis=1).min(axis=1)
    a1   = bp.rolling(int(p1), min_periods=1).sum() / \
           tr.rolling(int(p1), min_periods=1).sum().replace(0, 1e-9)
    a2   = bp.rolling(int(p2), min_periods=1).sum() / \
           tr.rolling(int(p2), min_periods=1).sum().replace(0, 1e-9)
    a3   = bp.rolling(int(p3), min_periods=1).sum() / \
           tr.rolling(int(p3), min_periods=1).sum().replace(0, 1e-9)
    return 100 * (4 * a1 + 2 * a2 + a3) / 7


def _fisher(df, p):
    p  = int(p)
    hi = df['high'].rolling(p, min_periods=1).max()
    lo = df['low'].rolling(p, min_periods=1).min()
    v  = 2 * (df['close'] - lo) / (hi - lo + 1e-9) - 1
    v  = v.clip(-0.999, 0.999)
    f  = 0.5 * np.log((1 + v) / (1 - v))
    sig = _ema(f, 1)
    return f, sig


def _trix(s, p, sig_p):
    e1 = _ema(s, p)
    e2 = _ema(e1, p)
    e3 = _ema(e2, p)
    trix = e3.pct_change() * 100
    sg   = _ema(trix, sig_p)
    return trix, sg


def _demarker(df, p):
    p   = int(p)
    hi  = df['high']
    lo  = df['low']
    dh  = (hi - hi.shift(1)).clip(lower=0)
    dl  = (lo.shift(1) - lo).clip(lower=0)
    return _sma(dh, p) / (_sma(dh, p) + _sma(dl, p)).replace(0, 1e-9)


def _rvi(df, p):
    """Relative Vigor Index."""
    p  = int(p)
    cl = df['close']
    op = df['open']
    hi = df['high']
    lo = df['low']
    num = (cl - op + 2 * (cl.shift(1) - op.shift(1)) +
           2 * (cl.shift(2) - op.shift(2)) + (cl.shift(3) - op.shift(3))) / 6
    den = (hi - lo + 2 * (hi.shift(1) - lo.shift(1)) +
           2 * (hi.shift(2) - lo.shift(2)) + (hi.shift(3) - lo.shift(3))) / 6
    n_ma = _sma(num, p)
    d_ma = _sma(den.replace(0, 1e-9), p)
    rvi  = n_ma / d_ma
    sig  = (rvi + 2 * rvi.shift(1) + 2 * rvi.shift(2) + rvi.shift(3)) / 6
    return rvi, sig


def _coppock(s, roc1, roc2, wma_p):
    r1  = s.pct_change(int(roc1)) * 100
    r2  = s.pct_change(int(roc2)) * 100
    cp  = r1 + r2
    w   = np.arange(1, int(wma_p) + 1)
    return cp.rolling(int(wma_p)).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def _aroon(df, p):
    p   = int(p)
    hi  = df['high'].rolling(p + 1, min_periods=1)
    lo  = df['low'].rolling(p + 1, min_periods=1)
    aroon_up  = 100 * (p - hi.apply(lambda x: np.argmax(x[::-1]), raw=True)) / p
    aroon_dn  = 100 * (p - lo.apply(lambda x: np.argmin(x[::-1]), raw=True)) / p
    return aroon_up, aroon_dn


def _bop(df, p):
    raw = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
    return _sma(raw, p)


def _cmf(df, p):
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
          (df['high'] - df['low'] + 1e-9)
    mfv = mfm * df['volume']
    return mfv.rolling(int(p), min_periods=1).sum() / \
           df['volume'].rolling(int(p), min_periods=1).sum().replace(0, 1e-9)


def _klinger(df, fast, slow, sig):
    sv   = (df['close'] - df['low'] - (df['high'] - df['close'])) / \
           (df['high'] - df['low'] + 1e-9) * df['volume']
    kvo  = _ema(sv, fast) - _ema(sv, slow)
    sg   = _ema(kvo, sig)
    return kvo, sg


def _ppo(s, fast_p, slow_p, sig_p):
    fast = _ema(s, fast_p)
    slow = _ema(s, slow_p)
    ppo  = 100 * (fast - slow) / slow.replace(0, 1e-9)
    sg   = _ema(ppo, sig_p)
    return ppo, sg


def _elder_force(df, p):
    efi = (df['close'] - df['close'].shift(1)) * df['volume']
    return _ema(efi, p)


def _vortex(df, p):
    p    = int(p)
    atr  = _atr(df, p)
    vm_p = (df['high'] - df['low'].shift(1)).abs()
    vm_n = (df['low']  - df['high'].shift(1)).abs()
    vi_p = vm_p.rolling(p, min_periods=1).sum() / \
           atr.rolling(p, min_periods=1).sum().replace(0, 1e-9)
    vi_n = vm_n.rolling(p, min_periods=1).sum() / \
           atr.rolling(p, min_periods=1).sum().replace(0, 1e-9)
    return vi_p, vi_n


def _mass_index(df, ema_p, sum_p):
    hl   = df['high'] - df['low']
    e1   = _ema(hl, ema_p)
    e2   = _ema(e1, ema_p)
    ratio = e1 / e2.replace(0, 1e-9)
    return ratio.rolling(int(sum_p), min_periods=1).sum()


def _wave_trend(df, n1, n2):
    hlc3  = (df['high'] + df['low'] + df['close']) / 3
    ema1  = _ema(hlc3, n1)
    d     = (hlc3 - ema1).abs()
    ema2  = _ema(d, n1)
    ci    = (hlc3 - ema1) / (0.015 * ema2.replace(0, 1e-9))
    wt1   = _ema(ci, n2)
    wt2   = _sma(wt1, 4)
    return wt1, wt2


def _stc(s, stc_p, fast, slow):
    """Schaff Trend Cycle."""
    macd_line = _ema(s, fast) - _ema(s, slow)
    p = int(stc_p)
    lo  = macd_line.rolling(p, min_periods=1).min()
    hi  = macd_line.rolling(p, min_periods=1).max()
    k   = 100 * (macd_line - lo) / (hi - lo + 1e-9)
    d   = _ema(k, 3)
    lo2 = d.rolling(p, min_periods=1).min()
    hi2 = d.rolling(p, min_periods=1).max()
    k2  = 100 * (d - lo2) / (hi2 - lo2 + 1e-9)
    stc = _ema(k2, 3)
    return stc


def _qqe(s, rsi_p, sf, qq_f):
    """QQE (Quantitative Qualitative Estimation)."""
    rsi      = _rsi(s, rsi_p)
    rsi_ma   = _ema(rsi, int(sf))
    atr_rsi  = _atr_series(rsi_ma, int(rsi_p))
    qqe_f    = _rma(atr_rsi, int(rsi_p)) * qq_f
    return rsi_ma, qqe_f


def _atr_series(s, p):
    """ATR of a generic series (treat as price movement)."""
    d = s.diff().abs()
    return _rma(d, p)


def _crossover(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


# ── 1. Osc_RSI_CCI_MACD ──────────────────────────────────────────────────────

def gen_Osc_RSI_CCI_MACD(df, rsi_p=14, cci_p=14, fast=12, slow=26, sig=9, **kw):
    rsi       = _rsi(df['close'], rsi_p)
    cci       = _cci(df, cci_p)
    macd, sg  = _macd(df['close'], fast, slow, sig)
    s         = pd.Series(0, index=df.index)
    s[(rsi > 50) & (cci > 0) & (macd > sg)] =  1
    s[(rsi < 50) & (cci < 0) & (macd < sg)] = -1
    return s


def space_Osc_RSI_CCI_MACD(trial):
    return {
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
        'cci_p': trial.suggest_int('cci_p', 10, 20),
        'fast':  trial.suggest_int('fast',  8, 16),
        'slow':  trial.suggest_int('slow',  20, 30),
        'sig':   trial.suggest_int('sig',   5, 12),
    }


# ── 2. Osc_Stoch_Williams_RSI ─────────────────────────────────────────────────

def gen_Osc_Stoch_Williams_RSI(df, stoch_k=14, wr_p=14, rsi_p=14, **kw):
    K, _    = _stoch(df, stoch_k)
    wr      = _williams_r(df, wr_p)
    rsi     = _rsi(df['close'], rsi_p)
    sig     = pd.Series(0, index=df.index)
    sig[(K < 30) & (wr < -80) & (rsi < 35)] =  1
    sig[(K > 70) & (wr > -20) & (rsi > 65)] = -1
    return sig


def space_Osc_Stoch_Williams_RSI(trial):
    return {
        'stoch_k': trial.suggest_int('stoch_k', 7, 21),
        'wr_p':    trial.suggest_int('wr_p', 7, 21),
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
    }


# ── 3. Osc_TSI_MACD ───────────────────────────────────────────────────────────

def gen_Osc_TSI_MACD(df, tsi_long=25, tsi_short=13, fast=12, slow=26, sig=9, **kw):
    tsi       = _tsi(df['close'], tsi_long, tsi_short)
    macd, sg  = _macd(df['close'], fast, slow, sig)
    s         = pd.Series(0, index=df.index)
    s[_crossover(tsi, pd.Series(0, index=df.index))  & (macd > sg)] =  1
    s[_crossunder(tsi, pd.Series(0, index=df.index)) & (macd < sg)] = -1
    return s


def space_Osc_TSI_MACD(trial):
    return {
        'tsi_long':  trial.suggest_int('tsi_long', 13, 30),
        'tsi_short': trial.suggest_int('tsi_short', 3, 10),
        'fast':      trial.suggest_int('fast',  8, 16),
        'slow':      trial.suggest_int('slow',  20, 30),
        'sig':       trial.suggest_int('sig',   5, 12),
    }


# ── 4. Osc_AO_MACD ────────────────────────────────────────────────────────────

def gen_Osc_AO_MACD(df, ao_fast=5, ao_slow=34, fast=12, slow=26, sig=9, **kw):
    ao        = _ao(df, ao_fast, ao_slow)
    macd, sg  = _macd(df['close'], fast, slow, sig)
    s         = pd.Series(0, index=df.index)
    s[(ao > 0) & (macd > sg)] =  1
    s[(ao < 0) & (macd < sg)] = -1
    return s


def space_Osc_AO_MACD(trial):
    return {
        'ao_fast': trial.suggest_int('ao_fast', 3, 8),
        'ao_slow': trial.suggest_int('ao_slow', 25, 40),
        'fast':    trial.suggest_int('fast',  8, 16),
        'slow':    trial.suggest_int('slow',  20, 30),
        'sig':     trial.suggest_int('sig',   5, 12),
    }


# ── 5. Osc_QQE_RSI ────────────────────────────────────────────────────────────

def gen_Osc_QQE_RSI(df, rsi_p=14, sf=5, qq_f=4.236, rsi2_p=14, **kw):
    rsi_ma, qqe_f = _qqe(df['close'], rsi_p, sf, qq_f)
    rsi2          = _rsi(df['close'], rsi2_p)
    bullish       = rsi_ma > (50 + qqe_f * 0.1)
    s             = pd.Series(0, index=df.index)
    s[bullish & (rsi2 > 50)]  =  1
    s[(~bullish) & (rsi2 < 50)] = -1
    return s


def space_Osc_QQE_RSI(trial):
    return {
        'rsi_p':  trial.suggest_int('rsi_p', 6, 14),
        'sf':     trial.suggest_int('sf', 3, 8),
        'qq_f':   trial.suggest_float('qq_f', 3.0, 6.0),
        'rsi2_p': trial.suggest_int('rsi2_p', 7, 21),
    }


# ── 6. Osc_CMO_Stoch ──────────────────────────────────────────────────────────

def gen_Osc_CMO_Stoch(df, cmo_p=14, stoch_k=14, stoch_d=3, **kw):
    cmo  = _cmo(df['close'], cmo_p)
    K, D = _stoch(df, stoch_k, stoch_d)
    sig  = pd.Series(0, index=df.index)
    sig[(cmo > 0) & _crossover(K, D)] =  1
    sig[(cmo < 0) & _crossunder(K, D)] = -1
    return sig


def space_Osc_CMO_Stoch(trial):
    return {
        'cmo_p':   trial.suggest_int('cmo_p', 9, 21),
        'stoch_k': trial.suggest_int('stoch_k', 7, 21),
        'stoch_d': trial.suggest_int('stoch_d', 3, 7),
    }


# ── 7. Osc_STC_MACD ───────────────────────────────────────────────────────────

def gen_Osc_STC_MACD(df, stc_p=12, fast=23, slow=50, macd_fast=12, macd_slow=26, sig=9, **kw):
    stc       = _stc(df['close'], stc_p, fast, slow)
    macd, sg  = _macd(df['close'], macd_fast, macd_slow, sig)
    s         = pd.Series(0, index=df.index)
    s[(stc > 75) & (macd > sg)] =  1
    s[(stc < 25) & (macd < sg)] = -1
    return s


def space_Osc_STC_MACD(trial):
    return {
        'stc_p':     trial.suggest_int('stc_p', 10, 20),
        'fast':      trial.suggest_int('fast',  12, 26),
        'slow':      trial.suggest_int('slow',  26, 52),
        'macd_fast': trial.suggest_int('macd_fast', 8, 16),
        'macd_slow': trial.suggest_int('macd_slow', 20, 30),
        'sig':       trial.suggest_int('sig',   5, 12),
    }


# ── 8. Osc_DPO_RSI ────────────────────────────────────────────────────────────

def gen_Osc_DPO_RSI(df, dpo_p=20, rsi_p=14, **kw):
    dpo = _dpo(df['close'], dpo_p)
    rsi = _rsi(df['close'], rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[(dpo > 0) & (rsi >= 40) & (rsi <= 65)] =  1
    sig[(dpo < 0) & (rsi >= 35) & (rsi <= 60)] = -1
    return sig


def space_Osc_DPO_RSI(trial):
    return {
        'dpo_p': trial.suggest_int('dpo_p', 14, 30),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 9. Osc_UO_Stoch ───────────────────────────────────────────────────────────

def gen_Osc_UO_Stoch(df, p1=7, p2=14, p3=28, stoch_k=14, stoch_d=3, **kw):
    uo      = _uo(df, p1, p2, p3)
    K, D    = _stoch(df, stoch_k, stoch_d)
    sig     = pd.Series(0, index=df.index)
    sig[(uo < 30) & (K < 20)] =  1
    sig[(uo > 70) & (K > 80)] = -1
    return sig


def space_Osc_UO_Stoch(trial):
    return {
        'p1':      trial.suggest_int('p1', 5, 10),
        'p2':      trial.suggest_int('p2', 10, 20),
        'p3':      trial.suggest_int('p3', 20, 40),
        'stoch_k': trial.suggest_int('stoch_k', 7, 21),
        'stoch_d': trial.suggest_int('stoch_d', 3, 7),
    }


# ── 10. Osc_Fisher_RSI ────────────────────────────────────────────────────────

def gen_Osc_Fisher_RSI(df, fish_p=10, rsi_p=14, **kw):
    f, fs = _fisher(df, fish_p)
    rsi   = _rsi(df['close'], rsi_p)
    sig   = pd.Series(0, index=df.index)
    sig[_crossover(f, fs)  & (rsi > 50)] =  1
    sig[_crossunder(f, fs) & (rsi < 50)] = -1
    return sig


def space_Osc_Fisher_RSI(trial):
    return {
        'fish_p': trial.suggest_int('fish_p', 7, 20),
        'rsi_p':  trial.suggest_int('rsi_p', 7, 21),
    }


# ── 11. Osc_TRIX_RSI ──────────────────────────────────────────────────────────

def gen_Osc_TRIX_RSI(df, trix_p=15, sig_p=9, rsi_p=14, **kw):
    trix, sg = _trix(df['close'], trix_p, sig_p)
    rsi      = _rsi(df['close'], rsi_p)
    s        = pd.Series(0, index=df.index)
    s[_crossover(trix, pd.Series(0, index=df.index)) & (rsi >= 45) & (rsi <= 65)] =  1
    s[_crossunder(trix, pd.Series(0, index=df.index)) & (rsi <= 55) & (rsi >= 35)] = -1
    return s


def space_Osc_TRIX_RSI(trial):
    return {
        'trix_p': trial.suggest_int('trix_p', 9, 20),
        'sig_p':  trial.suggest_int('sig_p', 3, 9),
        'rsi_p':  trial.suggest_int('rsi_p', 7, 21),
    }


# ── 12. Osc_DeMarker_BB ───────────────────────────────────────────────────────

def gen_Osc_DeMarker_BB(df, dem_p=14, bb_p=20, bb_mult=2.0, **kw):
    dm          = _demarker(df, dem_p)
    _, _, lower = _bb(df['close'], bb_p, bb_mult)
    upper, _, _ = _bb(df['close'], bb_p, bb_mult)
    sig = pd.Series(0, index=df.index)
    sig[(dm < 0.3) & (df['close'] < lower)] =  1
    sig[(dm > 0.7) & (df['close'] > upper)] = -1
    return sig


def space_Osc_DeMarker_BB(trial):
    return {
        'dem_p':   trial.suggest_int('dem_p', 10, 20),
        'bb_p':    trial.suggest_int('bb_p', 20, 20),
        'bb_mult': trial.suggest_float('bb_mult', 2.0, 2.0),
    }


# ── 13. Osc_RVI_MACD ──────────────────────────────────────────────────────────

def gen_Osc_RVI_MACD(df, rvi_p=14, fast=12, slow=26, sig=9, **kw):
    rvi, rv_sig  = _rvi(df, rvi_p)
    macd, mg     = _macd(df['close'], fast, slow, sig)
    s            = pd.Series(0, index=df.index)
    s[(rvi > rv_sig) & (macd > 0)] =  1
    s[(rvi < rv_sig) & (macd < 0)] = -1
    return s


def space_Osc_RVI_MACD(trial):
    return {
        'rvi_p': trial.suggest_int('rvi_p', 7, 20),
        'fast':  trial.suggest_int('fast', 8, 16),
        'slow':  trial.suggest_int('slow', 20, 30),
        'sig':   trial.suggest_int('sig', 5, 12),
    }


# ── 14. Osc_Coppock_RSI ───────────────────────────────────────────────────────

def gen_Osc_Coppock_RSI(df, roc1=14, roc2=11, wma_p=10, rsi_p=14, **kw):
    cop = _coppock(df['close'], roc1, roc2, wma_p)
    rsi = _rsi(df['close'], rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[(cop > 0) & (rsi > 50)] =  1
    sig[(cop < 0) & (rsi < 50)] = -1
    return sig


def space_Osc_Coppock_RSI(trial):
    return {
        'roc1':  trial.suggest_int('roc1', 10, 16),
        'roc2':  trial.suggest_int('roc2', 7, 13),
        'wma_p': trial.suggest_int('wma_p', 8, 15),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 15. Osc_Aroon_RSI ─────────────────────────────────────────────────────────

def gen_Osc_Aroon_RSI(df, aroon_p=25, rsi_p=14, **kw):
    au, ad = _aroon(df, aroon_p)
    rsi    = _rsi(df['close'], rsi_p)
    sig    = pd.Series(0, index=df.index)
    sig[(au > 70) & (rsi > 50)] =  1
    sig[(ad > 70) & (rsi < 50)] = -1
    return sig


def space_Osc_Aroon_RSI(trial):
    return {
        'aroon_p': trial.suggest_int('aroon_p', 10, 30),
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
    }


# ── 16. Osc_BOP_MACD ──────────────────────────────────────────────────────────

def gen_Osc_BOP_MACD(df, bop_p=14, fast=12, slow=26, sig=9, **kw):
    bop       = _bop(df, bop_p)
    macd, sg  = _macd(df['close'], fast, slow, sig)
    s         = pd.Series(0, index=df.index)
    s[(bop > 0) & _crossover(macd, sg)] =  1
    s[(bop < 0) & _crossunder(macd, sg)] = -1
    return s


def space_Osc_BOP_MACD(trial):
    return {
        'bop_p': trial.suggest_int('bop_p', 5, 20),
        'fast':  trial.suggest_int('fast', 8, 16),
        'slow':  trial.suggest_int('slow', 20, 30),
        'sig':   trial.suggest_int('sig', 5, 12),
    }


# ── 17. Osc_CMF_RSI ───────────────────────────────────────────────────────────

def gen_Osc_CMF_RSI(df, cmf_p=20, rsi_p=14, **kw):
    cmf = _cmf(df, cmf_p)
    rsi = _rsi(df['close'], rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[(cmf > 0.1) & (rsi > 50)] =  1
    sig[(cmf < -0.1) & (rsi < 50)] = -1
    return sig


def space_Osc_CMF_RSI(trial):
    return {
        'cmf_p': trial.suggest_int('cmf_p', 10, 30),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 18. Osc_Klinger_RSI ───────────────────────────────────────────────────────

def gen_Osc_Klinger_RSI(df, fast=34, slow=55, sig=13, rsi_p=14, **kw):
    kvo, kg = _klinger(df, fast, slow, sig)
    rsi     = _rsi(df['close'], rsi_p)
    s       = pd.Series(0, index=df.index)
    s[_crossover(kvo, kg)  & (rsi > 50)] =  1
    s[_crossunder(kvo, kg) & (rsi < 50)] = -1
    return s


def space_Osc_Klinger_RSI(trial):
    return {
        'fast':  trial.suggest_int('fast', 30, 40),
        'slow':  trial.suggest_int('slow', 50, 65),
        'sig':   trial.suggest_int('sig', 10, 15),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 19. Osc_PPO_Stoch ─────────────────────────────────────────────────────────

def gen_Osc_PPO_Stoch(df, fast_p=12, slow_p=26, sig_p=9, stoch_k=14, **kw):
    ppo, pg = _ppo(df['close'], fast_p, slow_p, sig_p)
    K, D    = _stoch(df, stoch_k)
    sig     = pd.Series(0, index=df.index)
    sig[(ppo > 0) & (K < 20)] =  1
    sig[(ppo < 0) & (K > 80)] = -1
    return sig


def space_Osc_PPO_Stoch(trial):
    return {
        'fast_p':  trial.suggest_int('fast_p', 8, 16),
        'slow_p':  trial.suggest_int('slow_p', 20, 35),
        'sig_p':   trial.suggest_int('sig_p', 5, 12),
        'stoch_k': trial.suggest_int('stoch_k', 7, 21),
    }


# ── 20. Osc_Elder_Force_RSI ───────────────────────────────────────────────────

def gen_Osc_Elder_Force_RSI(df, fi_p=13, rsi_p=14, **kw):
    efi = _elder_force(df, fi_p)
    rsi = _rsi(df['close'], rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[(efi > 0) & (rsi >= 40) & (rsi <= 65)] =  1
    sig[(efi < 0) & (rsi <= 60) & (rsi >= 35)] = -1
    return sig


def space_Osc_Elder_Force_RSI(trial):
    return {
        'fi_p':  trial.suggest_int('fi_p', 2, 20),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 21. Osc_Vortex_MACD ───────────────────────────────────────────────────────

def gen_Osc_Vortex_MACD(df, vortex_p=14, fast=12, slow=26, sig=9, **kw):
    vi_p, vi_n = _vortex(df, vortex_p)
    macd, mg   = _macd(df['close'], fast, slow, sig)
    s          = pd.Series(0, index=df.index)
    s[(vi_p > vi_n) & (macd > 0)] =  1
    s[(vi_n > vi_p) & (macd < 0)] = -1
    return s


def space_Osc_Vortex_MACD(trial):
    return {
        'vortex_p': trial.suggest_int('vortex_p', 14, 20),
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig':      trial.suggest_int('sig', 5, 12),
    }


# ── 22. Osc_Mass_RSI ──────────────────────────────────────────────────────────

def gen_Osc_Mass_RSI(df, ema_p=9, sum_p=25, rsi_p=14, **kw):
    mi  = _mass_index(df, ema_p, sum_p)
    rsi = _rsi(df['close'], rsi_p)
    # Reversal bulge: MI crosses above 27 then back below 26.5
    above = mi > 27
    rev   = above.shift(1, fill_value=False) & (mi < 26.5)
    sig   = pd.Series(0, index=df.index)
    sig[rev & (rsi < 50)] = 1
    sig[rev & (rsi > 50)] = -1
    return sig


def space_Osc_Mass_RSI(trial):
    return {
        'ema_p': trial.suggest_int('ema_p', 9, 15),
        'sum_p': trial.suggest_int('sum_p', 20, 30),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 23. Osc_WaveTrend_RSI ─────────────────────────────────────────────────────

def gen_Osc_WaveTrend_RSI(df, wt_n1=10, wt_n2=21, rsi_p=14, **kw):
    wt1, wt2 = _wave_trend(df, wt_n1, wt_n2)
    rsi      = _rsi(df['close'], rsi_p)
    oversold = (wt1 < -60) & (wt2 < -60)
    sig      = pd.Series(0, index=df.index)
    sig[_crossover(wt1, wt2) & oversold & (rsi < 40)] =  1
    overbought = (wt1 > 60) & (wt2 > 60)
    sig[_crossunder(wt1, wt2) & overbought & (rsi > 60)] = -1
    return sig


def space_Osc_WaveTrend_RSI(trial):
    return {
        'wt_n1': trial.suggest_int('wt_n1', 7, 15),
        'wt_n2': trial.suggest_int('wt_n2', 15, 30),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 24. Osc_TSI_Stoch ─────────────────────────────────────────────────────────

def gen_Osc_TSI_Stoch(df, long_p=25, short_p=13, stoch_k=14, stoch_d=3, **kw):
    tsi  = _tsi(df['close'], long_p, short_p)
    K, D = _stoch(df, stoch_k, stoch_d)
    sig  = pd.Series(0, index=df.index)
    sig[(tsi > 0) & _crossover(K, D) & (K < 20)]  =  1
    sig[(tsi < 0) & _crossunder(K, D) & (K > 80)] = -1
    return sig


def space_Osc_TSI_Stoch(trial):
    return {
        'long_p':  trial.suggest_int('long_p', 13, 30),
        'short_p': trial.suggest_int('short_p', 3, 10),
        'stoch_k': trial.suggest_int('stoch_k', 7, 21),
        'stoch_d': trial.suggest_int('stoch_d', 3, 7),
    }


# ── 25. Osc_ROC_RSI ───────────────────────────────────────────────────────────

def gen_Osc_ROC_RSI(df, roc_p=14, rsi_p=14, **kw):
    roc = df['close'].pct_change(int(roc_p)) * 100
    rsi = _rsi(df['close'], rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[(roc > 0) & (rsi >= 50) & (rsi <= 70)] =  1
    sig[(roc < 0) & (rsi <= 50) & (rsi >= 30)] = -1
    return sig


def space_Osc_ROC_RSI(trial):
    return {
        'roc_p': trial.suggest_int('roc_p', 10, 30),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 26. Osc_CMO_EMA_Vol ───────────────────────────────────────────────────────

def gen_Osc_CMO_EMA_Vol(df, cmo_p=14, ema_p=50, vol_p=20, **kw):
    cmo    = _cmo(df['close'], cmo_p)
    ema    = _ema(df['close'], ema_p)
    vol_ma = _sma(df['volume'], vol_p)
    vol_ok = df['volume'] > vol_ma
    sig    = pd.Series(0, index=df.index)
    sig[(cmo > 0) & (df['close'] > ema) & vol_ok] =  1
    sig[(cmo < 0) & (df['close'] < ema) & vol_ok] = -1
    return sig


def space_Osc_CMO_EMA_Vol(trial):
    return {
        'cmo_p': trial.suggest_int('cmo_p', 9, 21),
        'ema_p': trial.suggest_int('ema_p', 20, 100),
        'vol_p': trial.suggest_int('vol_p', 20, 50),
    }


# ── 27. Osc_DM_RSI_BB ─────────────────────────────────────────────────────────

def gen_Osc_DM_RSI_BB(df, dem_p=14, rsi_p=14, bb_p=20, **kw):
    dm          = _demarker(df, dem_p)
    rsi         = _rsi(df['close'], rsi_p)
    _, _, lower = _bb(df['close'], bb_p, 2.0)
    upper, _, _ = _bb(df['close'], bb_p, 2.0)
    sig = pd.Series(0, index=df.index)
    sig[(dm < 0.25) & (rsi < 35) & (df['close'] < lower)] =  1
    sig[(dm > 0.75) & (rsi > 65) & (df['close'] > upper)] = -1
    return sig


def space_Osc_DM_RSI_BB(trial):
    return {
        'dem_p': trial.suggest_int('dem_p', 10, 20),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
        'bb_p':  trial.suggest_int('bb_p', 20, 20),
    }


# ── 28. Osc_Momentum_Cross ────────────────────────────────────────────────────

def gen_Osc_Momentum_Cross(df, fast_roc=5, slow_roc=20, **kw):
    fast = df['close'].pct_change(int(fast_roc)) * 100
    slow = df['close'].pct_change(int(slow_roc)) * 100
    zero = pd.Series(0.0, index=df.index)
    sig  = pd.Series(0, index=df.index)
    sig[_crossover(fast, slow) & (slow < 0)] =  1
    sig[_crossunder(fast, slow) & (slow > 0)] = -1
    return sig


def space_Osc_Momentum_Cross(trial):
    return {
        'fast_roc': trial.suggest_int('fast_roc', 3, 8),
        'slow_roc': trial.suggest_int('slow_roc', 10, 25),
    }


# ── 29. Osc_CCI_RSI_Vol ───────────────────────────────────────────────────────

def gen_Osc_CCI_RSI_Vol(df, cci_p=14, rsi_p=14, vol_p=20, **kw):
    cci    = _cci(df, cci_p)
    rsi    = _rsi(df['close'], rsi_p)
    vol_ma = _sma(df['volume'], vol_p)
    vol_ok = df['volume'] > vol_ma
    sig    = pd.Series(0, index=df.index)
    sig[(cci < -100) & (rsi < 40) & vol_ok] =  1
    sig[(cci >  100) & (rsi > 60) & vol_ok] = -1
    return sig


def space_Osc_CCI_RSI_Vol(trial):
    return {
        'cci_p': trial.suggest_int('cci_p', 10, 20),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
        'vol_p': trial.suggest_int('vol_p', 20, 50),
    }


# ── 30. Osc_Full_Stack ────────────────────────────────────────────────────────

def gen_Osc_Full_Stack(df, rsi_p=14, fast=12, slow=26, sig=9,
                        cmf_p=20, stoch_k=14, stoch_d=3, **kw):
    rsi       = _rsi(df['close'], rsi_p)
    macd, mg  = _macd(df['close'], fast, slow, sig)
    ao        = _ao(df)
    cmf       = _cmf(df, cmf_p)
    K, D      = _stoch(df, stoch_k, stoch_d)
    s         = pd.Series(0, index=df.index)
    all_bull  = (rsi > 50) & (macd > mg) & (ao > 0) & (cmf > 0) & (K > D)
    all_bear  = (rsi < 50) & (macd < mg) & (ao < 0) & (cmf < 0) & (K < D)
    s[all_bull] =  1
    s[all_bear] = -1
    return s


def space_Osc_Full_Stack(trial):
    return {
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
        'fast':    trial.suggest_int('fast', 8, 16),
        'slow':    trial.suggest_int('slow', 20, 30),
        'sig':     trial.suggest_int('sig', 5, 12),
        'cmf_p':   trial.suggest_int('cmf_p', 10, 30),
        'stoch_k': trial.suggest_int('stoch_k', 7, 21),
        'stoch_d': trial.suggest_int('stoch_d', 3, 7),
    }


# ── STRATEGY_EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Osc_RSI_CCI_MACD': {
        'gen': gen_Osc_RSI_CCI_MACD,
        'space': space_Osc_RSI_CCI_MACD,
        'default_params': {'rsi_p': 14, 'cci_p': 14, 'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3400,
                 'description': 'Triple oscillator alignment: RSI + CCI + MACD all confirm direction.'},
    },
    'Osc_Stoch_Williams_RSI': {
        'gen': gen_Osc_Stoch_Williams_RSI,
        'space': space_Osc_Stoch_Williams_RSI,
        'default_params': {'stoch_k': 14, 'wr_p': 14, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3100,
                 'description': 'Stochastic + Williams %R + RSI triple oversold/overbought confirmation.'},
    },
    'Osc_TSI_MACD': {
        'gen': gen_Osc_TSI_MACD,
        'space': space_Osc_TSI_MACD,
        'default_params': {'tsi_long': 25, 'tsi_short': 13, 'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2600,
                 'description': 'TSI zero-cross + MACD direction confirmation dual momentum.'},
    },
    'Osc_AO_MACD': {
        'gen': gen_Osc_AO_MACD,
        'space': space_Osc_AO_MACD,
        'default_params': {'ao_fast': 5, 'ao_slow': 34, 'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2800,
                 'description': 'Awesome Oscillator + MACD same direction dual momentum filter.'},
    },
    'Osc_QQE_RSI': {
        'gen': gen_Osc_QQE_RSI,
        'space': space_Osc_QQE_RSI,
        'default_params': {'rsi_p': 14, 'sf': 5, 'qq_f': 4.236, 'rsi2_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4200,
                 'description': 'QQE (Quantitative Qualitative Estimation) + RSI momentum filter.'},
    },
    'Osc_CMO_Stoch': {
        'gen': gen_Osc_CMO_Stoch,
        'space': space_Osc_CMO_Stoch,
        'default_params': {'cmo_p': 14, 'stoch_k': 14, 'stoch_d': 3},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2200,
                 'description': 'CMO direction + Stochastic K/D crossover dual confirmation.'},
    },
    'Osc_STC_MACD': {
        'gen': gen_Osc_STC_MACD,
        'space': space_Osc_STC_MACD,
        'default_params': {'stc_p': 12, 'fast': 23, 'slow': 50, 'macd_fast': 12, 'macd_slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3600,
                 'description': 'Schaff Trend Cycle (fast trend detector) + MACD alignment.'},
    },
    'Osc_DPO_RSI': {
        'gen': gen_Osc_DPO_RSI,
        'space': space_Osc_DPO_RSI,
        'default_params': {'dpo_p': 20, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1900,
                 'description': 'Detrended Price Oscillator + RSI momentum zone filter.'},
    },
    'Osc_UO_Stoch': {
        'gen': gen_Osc_UO_Stoch,
        'space': space_Osc_UO_Stoch,
        'default_params': {'p1': 7, 'p2': 14, 'p3': 28, 'stoch_k': 14, 'stoch_d': 3},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2400,
                 'description': 'Ultimate Oscillator + Stochastic dual oversold mean reversion.'},
    },
    'Osc_Fisher_RSI': {
        'gen': gen_Osc_Fisher_RSI,
        'space': space_Osc_Fisher_RSI,
        'default_params': {'fish_p': 10, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Fisher Transform crossover signal + RSI momentum confirmation.'},
    },
    'Osc_TRIX_RSI': {
        'gen': gen_Osc_TRIX_RSI,
        'space': space_Osc_TRIX_RSI,
        'default_params': {'trix_p': 15, 'sig_p': 9, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'TRIX zero-cross + RSI momentum zone filter for trend entries.'},
    },
    'Osc_DeMarker_BB': {
        'gen': gen_Osc_DeMarker_BB,
        'space': space_Osc_DeMarker_BB,
        'default_params': {'dem_p': 14, 'bb_p': 20, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2100,
                 'description': 'DeMarker extreme + BB lower touch dual oversold confirmation.'},
    },
    'Osc_RVI_MACD': {
        'gen': gen_Osc_RVI_MACD,
        'space': space_Osc_RVI_MACD,
        'default_params': {'rvi_p': 14, 'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2300,
                 'description': 'Relative Vigor Index signal cross + MACD direction filter.'},
    },
    'Osc_Coppock_RSI': {
        'gen': gen_Osc_Coppock_RSI,
        'space': space_Osc_Coppock_RSI,
        'default_params': {'roc1': 14, 'roc2': 11, 'wma_p': 10, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2700,
                 'description': 'Coppock Curve (long-term momentum) + RSI direction confirmation.'},
    },
    'Osc_Aroon_RSI': {
        'gen': gen_Osc_Aroon_RSI,
        'space': space_Osc_Aroon_RSI,
        'default_params': {'aroon_p': 25, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2900,
                 'description': 'Aroon Up/Down extreme + RSI momentum for trend confirmation.'},
    },
    'Osc_BOP_MACD': {
        'gen': gen_Osc_BOP_MACD,
        'space': space_Osc_BOP_MACD,
        'default_params': {'bop_p': 14, 'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Balance of Power direction + MACD crossover dual confirmation.'},
    },
    'Osc_CMF_RSI': {
        'gen': gen_Osc_CMF_RSI,
        'space': space_Osc_CMF_RSI,
        'default_params': {'cmf_p': 20, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3200,
                 'description': 'Chaikin Money Flow + RSI momentum for smart money direction.'},
    },
    'Osc_Klinger_RSI': {
        'gen': gen_Osc_Klinger_RSI,
        'space': space_Osc_Klinger_RSI,
        'default_params': {'fast': 34, 'slow': 55, 'sig': 13, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2400,
                 'description': 'Klinger Volume Oscillator crossover + RSI momentum confirmation.'},
    },
    'Osc_PPO_Stoch': {
        'gen': gen_Osc_PPO_Stoch,
        'space': space_Osc_PPO_Stoch,
        'default_params': {'fast_p': 12, 'slow_p': 26, 'sig_p': 9, 'stoch_k': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2100,
                 'description': 'PPO (percentage MACD) direction + Stochastic extreme filter.'},
    },
    'Osc_Elder_Force_RSI': {
        'gen': gen_Osc_Elder_Force_RSI,
        'space': space_Osc_Elder_Force_RSI,
        'default_params': {'fi_p': 13, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2600,
                 'description': 'Elder Force Index (price change × volume) + RSI momentum zone.'},
    },
    'Osc_Vortex_MACD': {
        'gen': gen_Osc_Vortex_MACD,
        'space': space_Osc_Vortex_MACD,
        'default_params': {'vortex_p': 14, 'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2800,
                 'description': 'Vortex VI+/VI- direction + MACD zero-line filter.'},
    },
    'Osc_Mass_RSI': {
        'gen': gen_Osc_Mass_RSI,
        'space': space_Osc_Mass_RSI,
        'default_params': {'ema_p': 9, 'sum_p': 25, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1800,
                 'description': 'Mass Index reversal bulge (>27 then <26.5) + RSI direction.'},
    },
    'Osc_WaveTrend_RSI': {
        'gen': gen_Osc_WaveTrend_RSI,
        'space': space_Osc_WaveTrend_RSI,
        'default_params': {'wt_n1': 10, 'wt_n2': 21, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 5100,
                 'description': 'WaveTrend WT1/WT2 crossover in oversold zone + RSI confirmation.'},
    },
    'Osc_TSI_Stoch': {
        'gen': gen_Osc_TSI_Stoch,
        'space': space_Osc_TSI_Stoch,
        'default_params': {'long_p': 25, 'short_p': 13, 'stoch_k': 14, 'stoch_d': 3},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2200,
                 'description': 'TSI positive + Stochastic K/D crossover at extreme levels.'},
    },
    'Osc_ROC_RSI': {
        'gen': gen_Osc_ROC_RSI,
        'space': space_Osc_ROC_RSI,
        'default_params': {'roc_p': 14, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'ROC positive + RSI in 50-70 zone (positive momentum not overbought).'},
    },
    'Osc_CMO_EMA_Vol': {
        'gen': gen_Osc_CMO_EMA_Vol,
        'space': space_Osc_CMO_EMA_Vol,
        'default_params': {'cmo_p': 14, 'ema_p': 50, 'vol_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'CMO direction + EMA trend + volume above average triple filter.'},
    },
    'Osc_DM_RSI_BB': {
        'gen': gen_Osc_DM_RSI_BB,
        'space': space_Osc_DM_RSI_BB,
        'default_params': {'dem_p': 14, 'rsi_p': 14, 'bb_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2300,
                 'description': 'DeMarker + RSI + BB lower: triple oversold confirmation entry.'},
    },
    'Osc_Momentum_Cross': {
        'gen': gen_Osc_Momentum_Cross,
        'space': space_Osc_Momentum_Cross,
        'default_params': {'fast_roc': 5, 'slow_roc': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1700,
                 'description': 'Fast ROC crosses above slow ROC while slow is negative = reversal.'},
    },
    'Osc_CCI_RSI_Vol': {
        'gen': gen_Osc_CCI_RSI_Vol,
        'space': space_Osc_CCI_RSI_Vol,
        'default_params': {'cci_p': 14, 'rsi_p': 14, 'vol_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2900,
                 'description': 'CCI<-100 + RSI<40 + volume surge triple oversold with confirmation.'},
    },
    'Osc_Full_Stack': {
        'gen': gen_Osc_Full_Stack,
        'space': space_Osc_Full_Stack,
        'default_params': {'rsi_p': 14, 'fast': 12, 'slow': 26, 'sig': 9,
                           'cmf_p': 20, 'stoch_k': 14, 'stoch_d': 3},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 6200,
                 'description': '5-oscillator full stack alignment: RSI+MACD+AO+CMF+Stoch all agree.'},
    },
}
