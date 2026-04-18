#!/usr/bin/env python3
"""TV2 BATCH 16 — 30 estrategias Volume + OBV + MFI + VWAP 2026-04-01"""

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


def _crossover(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def _obv(df):
    """On-Balance Volume."""
    direction = np.sign(df['close'].diff()).fillna(0)
    return (direction * df['volume']).cumsum()


def _mfi(df, p):
    """Money Flow Index."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0)
    neg_mf = mf.where(delta < 0, 0.0)
    pos_sum = pos_mf.rolling(int(p), min_periods=1).sum()
    neg_sum = neg_mf.rolling(int(p), min_periods=1).sum()
    ratio = pos_sum / neg_sum.replace(0, 1e-9)
    return 100 - 100 / (1 + ratio)


def _cmf(df, p):
    """Chaikin Money Flow."""
    hl = (df['high'] - df['low']).replace(0, 1e-9)
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl * df['volume']
    return mfv.rolling(int(p), min_periods=1).sum() / df['volume'].rolling(int(p), min_periods=1).sum().replace(0, 1e-9)


def _vwap_rolling(df, p):
    """Rolling VWAP over last p bars."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    cum_tpv = (tp * df['volume']).rolling(int(p), min_periods=1).sum()
    cum_v   = df['volume'].rolling(int(p), min_periods=1).sum().replace(0, 1e-9)
    return cum_tpv / cum_v


def _stoch(df, k):
    """Stochastic %K of close."""
    lo = df['low'].rolling(int(k), min_periods=1).min()
    hi = df['high'].rolling(int(k), min_periods=1).max()
    denom = (hi - lo).replace(0, 1e-9)
    return (df['close'] - lo) / denom * 100


def _klinger(df, fast_p, slow_p):
    """Klinger Volume Oscillator."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    tp_prev = tp.shift(1).fillna(tp)
    direction = np.where(tp > tp_prev, 1, -1)
    sv = pd.Series(direction * df['volume'].values, index=df.index)
    kvo = _ema(sv, fast_p) - _ema(sv, slow_p)
    return kvo


def _vortex(df, p):
    """Vortex Indicator VI+ and VI-."""
    vm_plus  = (df['high'] - df['low'].shift(1)).abs()
    vm_minus = (df['low']  - df['high'].shift(1)).abs()
    atr_sum  = _atr(df, p) * p  # approximate ATR sum
    # proper TR sum
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    tr_sum = tr.rolling(int(p), min_periods=1).sum().replace(0, 1e-9)
    vp_sum = vm_plus.rolling(int(p), min_periods=1).sum()
    vm_sum = vm_minus.rolling(int(p), min_periods=1).sum()
    vi_plus  = vp_sum / tr_sum
    vi_minus = vm_sum / tr_sum
    return vi_plus, vi_minus


def _supertrend(df, p, mult):
    """SuperTrend indicator. Returns Series: 1=bullish, -1=bearish."""
    atr_v  = _atr(df, p)
    hl2    = (df['high'] + df['low']) / 2
    upper  = hl2 + mult * atr_v
    lower  = hl2 - mult * atr_v

    close  = df['close'].values
    up_v   = upper.values
    lo_v   = lower.values
    n      = len(close)
    trend  = np.ones(n)
    st     = np.full(n, np.nan)

    final_upper = np.copy(up_v)
    final_lower = np.copy(lo_v)

    for i in range(1, n):
        # final upper band
        if up_v[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = up_v[i]
        else:
            final_upper[i] = final_upper[i - 1]
        # final lower band
        if lo_v[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lo_v[i]
        else:
            final_lower[i] = final_lower[i - 1]
        # direction
        if close[i] > final_upper[i - 1]:
            trend[i] = 1
        elif close[i] < final_lower[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
        st[i] = final_lower[i] if trend[i] == 1 else final_upper[i]

    return pd.Series(trend, index=df.index)


# ── 1. OBV_EMA ───────────────────────────────────────────────────────────────

def gen_OBV_EMA(df, obv_ema_p=20, **kw):
    """OBV + EMA of OBV. Long: OBV > EMA(OBV). Short: OBV < EMA(OBV)."""
    obv = _obv(df)
    ema_obv = _ema(obv, obv_ema_p)
    sig = pd.Series(0, index=df.index)
    sig[_crossover(obv, ema_obv)]  = 1
    sig[_crossunder(obv, ema_obv)] = -1
    return sig


def space_OBV_EMA(trial):
    return {'obv_ema_p': trial.suggest_int('obv_ema_p', 10, 30)}


# ── 2. OBV_Divergence ────────────────────────────────────────────────────────

def gen_OBV_Divergence(df, lookback=10, pivot_p=5, **kw):
    """OBV divergence vs price. Bull: price LL + OBV HL. Bear: price HH + OBV LH."""
    lb = int(lookback)
    pp = int(pivot_p)
    obv  = _obv(df)
    close = df['close']

    # Rolling min/max for divergence detection
    price_lo  = close.rolling(lb, min_periods=lb).min()
    price_hi  = close.rolling(lb, min_periods=lb).max()
    obv_lo    = obv.rolling(lb, min_periods=lb).min()
    obv_hi    = obv.rolling(lb, min_periods=lb).max()

    prev_price_lo = price_lo.shift(pp)
    prev_price_hi = price_hi.shift(pp)
    prev_obv_lo   = obv_lo.shift(pp)
    prev_obv_hi   = obv_hi.shift(pp)

    # Bullish divergence: price makes lower low, OBV makes higher low
    bull_div = (close <= price_lo) & (close < prev_price_lo) & (obv > prev_obv_lo)
    # Bearish divergence: price makes higher high, OBV makes lower high
    bear_div = (close >= price_hi) & (close > prev_price_hi) & (obv < prev_obv_hi)

    sig = pd.Series(0, index=df.index)
    sig[bull_div] = 1
    sig[bear_div] = -1
    return sig


def space_OBV_Divergence(trial):
    return {
        'lookback': trial.suggest_int('lookback', 5, 20),
        'pivot_p':  trial.suggest_int('pivot_p', 3, 10),
    }


# ── 3. OBV_ATR ───────────────────────────────────────────────────────────────

def gen_OBV_ATR(df, obv_ema_p=20, atr_p=14, **kw):
    """OBV rising AND ATR > ATR avg. Short: OBV falling AND ATR > ATR avg."""
    obv     = _obv(df)
    ema_obv = _ema(obv, obv_ema_p)
    atr     = _atr(df, atr_p)
    atr_avg = _sma(atr, atr_p)

    obv_rising  = obv > ema_obv
    obv_falling = obv < ema_obv
    vol_high    = atr > atr_avg

    sig = pd.Series(0, index=df.index)
    sig[_crossover(obv, ema_obv)  & vol_high] = 1
    sig[_crossunder(obv, ema_obv) & vol_high] = -1
    return sig


def space_OBV_ATR(trial):
    return {
        'obv_ema_p': trial.suggest_int('obv_ema_p', 10, 30),
        'atr_p':     trial.suggest_int('atr_p', 10, 20),
    }


# ── 4. OBV_MACD ──────────────────────────────────────────────────────────────

def gen_OBV_MACD(df, fast=12, slow=26, sig_p=9, **kw):
    """MACD on OBV. Long: OBV_MACD crosses above signal."""
    obv      = _obv(df)
    macd     = _ema(obv, fast) - _ema(obv, slow)
    signal   = _ema(macd, sig_p)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(macd, signal)]  = 1
    sig[_crossunder(macd, signal)] = -1
    return sig


def space_OBV_MACD(trial):
    return {
        'fast':  trial.suggest_int('fast', 8, 16),
        'slow':  trial.suggest_int('slow', 20, 30),
        'sig_p': trial.suggest_int('sig_p', 5, 12),
    }


# ── 5. OBV_RSI ───────────────────────────────────────────────────────────────

def gen_OBV_RSI(df, obv_rsi_p=14, **kw):
    """RSI of OBV. Long: RSI(OBV) < 30 AND OBV rising. Short: RSI(OBV) > 70 AND OBV falling."""
    obv     = _obv(df)
    rsi_obv = _rsi(obv, obv_rsi_p)
    obv_ema = _ema(obv, obv_rsi_p)

    sig = pd.Series(0, index=df.index)
    sig[(rsi_obv < 30) & _crossover(obv, obv_ema)]  = 1
    sig[(rsi_obv > 70) & _crossunder(obv, obv_ema)] = -1
    return sig


def space_OBV_RSI(trial):
    return {'obv_rsi_p': trial.suggest_int('obv_rsi_p', 7, 21)}


# ── 6. MFI_Strategy ──────────────────────────────────────────────────────────

def gen_MFI_Strategy(df, mfi_p=14, **kw):
    """MFI < 20 = oversold long. MFI > 80 = overbought short."""
    mfi = _mfi(df, mfi_p)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(mfi, pd.Series(20, index=df.index))]  = 1
    sig[_crossunder(mfi, pd.Series(80, index=df.index))] = -1
    # Direct threshold entries
    sig[(mfi < 20) & (mfi.shift(1) >= 20)] = 1
    sig[(mfi > 80) & (mfi.shift(1) <= 80)] = -1
    return sig


def space_MFI_Strategy(trial):
    return {'mfi_p': trial.suggest_int('mfi_p', 7, 21)}


# ── 7. MFI_EMA ───────────────────────────────────────────────────────────────

def gen_MFI_EMA(df, mfi_p=14, ema_p=50, **kw):
    """MFI + EMA. Long: MFI < 30 AND close > EMA. Short: MFI > 70 AND close < EMA."""
    mfi   = _mfi(df, mfi_p)
    ema   = _ema(df['close'], ema_p)
    close = df['close']

    sig = pd.Series(0, index=df.index)
    sig[(mfi < 30) & (close > ema) & (mfi.shift(1) >= 30)] = 1
    sig[(mfi > 70) & (close < ema) & (mfi.shift(1) <= 70)] = -1
    return sig


def space_MFI_EMA(trial):
    return {
        'mfi_p': trial.suggest_int('mfi_p', 7, 21),
        'ema_p': trial.suggest_int('ema_p', 20, 60),
    }


# ── 8. MFI_Divergence ────────────────────────────────────────────────────────

def gen_MFI_Divergence(df, mfi_p=14, lookback=10, **kw):
    """MFI divergence. Bull: price LL + MFI HL. Bear: price HH + MFI LH."""
    lb    = int(lookback)
    mfi   = _mfi(df, mfi_p)
    close = df['close']

    price_lo = close.rolling(lb, min_periods=lb).min()
    price_hi = close.rolling(lb, min_periods=lb).max()
    mfi_lo   = mfi.rolling(lb, min_periods=lb).min()
    mfi_hi   = mfi.rolling(lb, min_periods=lb).max()

    prev_pl  = price_lo.shift(lb // 2)
    prev_ph  = price_hi.shift(lb // 2)
    prev_ml  = mfi_lo.shift(lb // 2)
    prev_mh  = mfi_hi.shift(lb // 2)

    bull_div = (close <= price_lo) & (close < prev_pl) & (mfi > prev_ml)
    bear_div = (close >= price_hi) & (close > prev_ph) & (mfi < prev_mh)

    sig = pd.Series(0, index=df.index)
    sig[bull_div] = 1
    sig[bear_div] = -1
    return sig


def space_MFI_Divergence(trial):
    return {
        'mfi_p':    trial.suggest_int('mfi_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 9. MFI_MACD ──────────────────────────────────────────────────────────────

def gen_MFI_MACD(df, mfi_p=14, fast=12, slow=26, **kw):
    """MFI + MACD. Long: MFI < 40 AND MACD > 0. Short: MFI > 60 AND MACD < 0."""
    mfi  = _mfi(df, mfi_p)
    macd = _ema(df['close'], fast) - _ema(df['close'], slow)
    macd_prev = macd.shift(1)

    sig = pd.Series(0, index=df.index)
    sig[(mfi < 40) & (macd > 0) & (macd_prev <= 0)] = 1
    sig[(mfi > 60) & (macd < 0) & (macd_prev >= 0)] = -1
    return sig


def space_MFI_MACD(trial):
    return {
        'mfi_p': trial.suggest_int('mfi_p', 7, 21),
        'fast':  trial.suggest_int('fast', 8, 16),
        'slow':  trial.suggest_int('slow', 20, 30),
    }


# ── 10. MFI_Stoch ────────────────────────────────────────────────────────────

def gen_MFI_Stoch(df, mfi_p=14, stoch_k=14, **kw):
    """MFI + Stoch both oversold/overbought. Long: MFI < 25 AND Stoch < 20."""
    mfi   = _mfi(df, mfi_p)
    stoch = _stoch(df, stoch_k)

    sig = pd.Series(0, index=df.index)
    sig[(mfi < 25) & (stoch < 20) & ((mfi.shift(1) >= 25) | (stoch.shift(1) >= 20))] = 1
    sig[(mfi > 75) & (stoch > 80) & ((mfi.shift(1) <= 75) | (stoch.shift(1) <= 80))] = -1
    return sig


def space_MFI_Stoch(trial):
    return {
        'mfi_p':   trial.suggest_int('mfi_p', 7, 21),
        'stoch_k': trial.suggest_int('stoch_k', 7, 21),
    }


# ── 11. CMF_Strategy ─────────────────────────────────────────────────────────

def gen_CMF_Strategy(df, cmf_p=20, **kw):
    """CMF > 0 AND rising = long. CMF < 0 AND falling = short."""
    cmf     = _cmf(df, cmf_p)
    cmf_dir = cmf.diff()

    sig = pd.Series(0, index=df.index)
    sig[_crossover(cmf, pd.Series(0.0, index=df.index)) & (cmf_dir > 0)]  = 1
    sig[_crossunder(cmf, pd.Series(0.0, index=df.index)) & (cmf_dir < 0)] = -1
    return sig


def space_CMF_Strategy(trial):
    return {'cmf_p': trial.suggest_int('cmf_p', 10, 30)}


# ── 12. CMF_EMA ──────────────────────────────────────────────────────────────

def gen_CMF_EMA(df, cmf_p=20, ema_p=50, **kw):
    """CMF + EMA trend. Long: CMF crosses above 0 AND close > EMA."""
    cmf   = _cmf(df, cmf_p)
    ema   = _ema(df['close'], ema_p)
    close = df['close']

    sig = pd.Series(0, index=df.index)
    sig[_crossover(cmf, pd.Series(0.0, index=df.index)) & (close > ema)]  = 1
    sig[_crossunder(cmf, pd.Series(0.0, index=df.index)) & (close < ema)] = -1
    return sig


def space_CMF_EMA(trial):
    return {
        'cmf_p': trial.suggest_int('cmf_p', 10, 30),
        'ema_p': trial.suggest_int('ema_p', 20, 100),
    }


# ── 13. CMF_RSI ──────────────────────────────────────────────────────────────

def gen_CMF_RSI(df, cmf_p=20, rsi_p=14, **kw):
    """CMF + RSI. Long: CMF > 0 AND RSI 40-65. Short: CMF < 0 AND RSI 35-60."""
    cmf = _cmf(df, cmf_p)
    rsi = _rsi(df['close'], rsi_p)

    sig = pd.Series(0, index=df.index)
    bull = _crossover(cmf, pd.Series(0.0, index=df.index)) & (rsi > 40) & (rsi < 65)
    bear = _crossunder(cmf, pd.Series(0.0, index=df.index)) & (rsi > 35) & (rsi < 60)
    sig[bull] = 1
    sig[bear] = -1
    return sig


def space_CMF_RSI(trial):
    return {
        'cmf_p': trial.suggest_int('cmf_p', 10, 30),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 14. CMF_MACD ─────────────────────────────────────────────────────────────

def gen_CMF_MACD(df, cmf_p=20, fast=12, slow=26, sig_p=9, **kw):
    """CMF + MACD. Long: CMF > 0 AND MACD > signal."""
    cmf    = _cmf(df, cmf_p)
    macd   = _ema(df['close'], fast) - _ema(df['close'], slow)
    signal = _ema(macd, sig_p)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(macd, signal)  & (cmf > 0)] = 1
    sig[_crossunder(macd, signal) & (cmf < 0)] = -1
    return sig


def space_CMF_MACD(trial):
    return {
        'cmf_p': trial.suggest_int('cmf_p', 10, 30),
        'fast':  trial.suggest_int('fast', 8, 16),
        'slow':  trial.suggest_int('slow', 20, 30),
        'sig_p': trial.suggest_int('sig_p', 5, 12),
    }


# ── 15. Klinger_OSC ──────────────────────────────────────────────────────────

def gen_Klinger_OSC(df, fast_p=34, slow_p=55, sig_p=13, **kw):
    """Klinger Volume Oscillator. Long: KVO crosses above signal."""
    kvo    = _klinger(df, fast_p, slow_p)
    signal = _ema(kvo, sig_p)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(kvo, signal)]  = 1
    sig[_crossunder(kvo, signal)] = -1
    return sig


def space_Klinger_OSC(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 28, 40),
        'slow_p': trial.suggest_int('slow_p', 50, 65),
        'sig_p':  trial.suggest_int('sig_p', 9, 18),
    }


# ── 16. Klinger_EMA ──────────────────────────────────────────────────────────

def gen_Klinger_EMA(df, fast_p=34, slow_p=55, ema_p=50, **kw):
    """Klinger + EMA trend. Long: KVO > 0 AND close > EMA."""
    kvo   = _klinger(df, fast_p, slow_p)
    ema   = _ema(df['close'], ema_p)
    close = df['close']

    sig = pd.Series(0, index=df.index)
    sig[_crossover(kvo, pd.Series(0.0, index=df.index)) & (close > ema)]  = 1
    sig[_crossunder(kvo, pd.Series(0.0, index=df.index)) & (close < ema)] = -1
    return sig


def space_Klinger_EMA(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 30, 40),
        'slow_p': trial.suggest_int('slow_p', 50, 65),
        'ema_p':  trial.suggest_int('ema_p', 20, 60),
    }


# ── 17. VWAP_Bands ───────────────────────────────────────────────────────────

def gen_VWAP_Bands(df, vwap_p=50, std_mult=1.5, rsi_p=14, **kw):
    """Rolling VWAP with std deviation bands. Long: close < VWAP - mult*std AND RSI < 40."""
    tp     = (df['high'] + df['low'] + df['close']) / 3
    vwap   = _vwap_rolling(df, vwap_p)
    # Rolling std of typical price
    tp_std = tp.rolling(int(vwap_p), min_periods=1).std().fillna(0)
    upper  = vwap + std_mult * tp_std
    lower  = vwap - std_mult * tp_std
    rsi    = _rsi(df['close'], rsi_p)
    close  = df['close']

    sig = pd.Series(0, index=df.index)
    # Long: price below lower band AND RSI oversold, then crosses back above lower band
    sig[(close < lower) & (rsi < 40) & (close.shift(1) >= lower.shift(1))] = 1
    # Short: price above upper band AND RSI overbought
    sig[(close > upper) & (rsi > 60) & (close.shift(1) <= upper.shift(1))] = -1
    return sig


def space_VWAP_Bands(trial):
    return {
        'vwap_p':   trial.suggest_int('vwap_p', 20, 100),
        'std_mult': trial.suggest_float('std_mult', 1.0, 3.0, step=0.25),
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
    }


# ── 18. VWAP_Pullback ────────────────────────────────────────────────────────

def gen_VWAP_Pullback(df, vwap_p=50, ema_p=20, **kw):
    """VWAP pullback. Long: EMA > VWAP AND close crosses above VWAP after being below."""
    vwap  = _vwap_rolling(df, vwap_p)
    ema   = _ema(df['close'], ema_p)
    close = df['close']

    # Uptrend: EMA above VWAP
    uptrend   = ema > vwap
    downtrend = ema < vwap

    sig = pd.Series(0, index=df.index)
    sig[uptrend   & _crossover(close, vwap)]  = 1
    sig[downtrend & _crossunder(close, vwap)] = -1
    return sig


def space_VWAP_Pullback(trial):
    return {
        'vwap_p': trial.suggest_int('vwap_p', 20, 100),
        'ema_p':  trial.suggest_int('ema_p', 20, 60),
    }


# ── 19. VWAP_MACD ────────────────────────────────────────────────────────────

def gen_VWAP_MACD(df, vwap_p=50, fast=12, slow=26, **kw):
    """VWAP + MACD. Long: close > VWAP AND MACD crosses above 0."""
    vwap  = _vwap_rolling(df, vwap_p)
    macd  = _ema(df['close'], fast) - _ema(df['close'], slow)
    close = df['close']

    sig = pd.Series(0, index=df.index)
    sig[(close > vwap) & _crossover(macd, pd.Series(0.0, index=df.index))]  = 1
    sig[(close < vwap) & _crossunder(macd, pd.Series(0.0, index=df.index))] = -1
    return sig


def space_VWAP_MACD(trial):
    return {
        'vwap_p': trial.suggest_int('vwap_p', 20, 100),
        'fast':   trial.suggest_int('fast', 8, 16),
        'slow':   trial.suggest_int('slow', 20, 30),
    }


# ── 20. VWAP_SuperTrend ──────────────────────────────────────────────────────

def gen_VWAP_SuperTrend(df, vwap_p=50, st_p=10, st_mult=3.0, **kw):
    """VWAP + SuperTrend. Long: close > VWAP AND SuperTrend bullish."""
    vwap  = _vwap_rolling(df, vwap_p)
    st    = _supertrend(df, st_p, st_mult)
    close = df['close']

    sig = pd.Series(0, index=df.index)
    sig[(close > vwap) & _crossover(st, pd.Series(0.0, index=df.index))]  = 1
    sig[(close < vwap) & _crossunder(st, pd.Series(0.0, index=df.index))] = -1
    return sig


def space_VWAP_SuperTrend(trial):
    return {
        'vwap_p':  trial.suggest_int('vwap_p', 20, 100),
        'st_p':    trial.suggest_int('st_p', 7, 21),
        'st_mult': trial.suggest_float('st_mult', 2.0, 5.0, step=0.25),
    }


# ── 21. VWAP_RSI_ATR ─────────────────────────────────────────────────────────

def gen_VWAP_RSI_ATR(df, vwap_p=50, rsi_p=14, atr_p=14, **kw):
    """VWAP + RSI + ATR. Long: close < VWAP*(1-0.005) AND RSI < 35 AND ATR high."""
    vwap    = _vwap_rolling(df, vwap_p)
    rsi     = _rsi(df['close'], rsi_p)
    atr     = _atr(df, atr_p)
    atr_avg = _sma(atr, atr_p)
    close   = df['close']

    vol_high = atr > atr_avg

    sig = pd.Series(0, index=df.index)
    bull = (close < vwap * 0.995) & (rsi < 35) & vol_high & (close.shift(1) >= (vwap.shift(1) * 0.995))
    bear = (close > vwap * 1.005) & (rsi > 65) & vol_high & (close.shift(1) <= (vwap.shift(1) * 1.005))
    sig[bull] = 1
    sig[bear] = -1
    return sig


def space_VWAP_RSI_ATR(trial):
    return {
        'vwap_p': trial.suggest_int('vwap_p', 20, 100),
        'rsi_p':  trial.suggest_int('rsi_p', 7, 21),
        'atr_p':  trial.suggest_int('atr_p', 10, 20),
    }


# ── 22. VP_POC ───────────────────────────────────────────────────────────────

def gen_VP_POC(df, vp_p=50, zone_pct=0.01, **kw):
    """Volume Profile POC. Long when price approaches POC from below and crosses."""
    vp   = int(vp_p)
    zpct = float(zone_pct)
    close = df['close']
    vol   = df['volume']

    # Approximate POC: VWAP is the volume-weighted center (simplified proxy)
    # For true POC we use volume-weighted median via rolling computation
    poc = _vwap_rolling(df, vp)  # volume-weighted average as POC proxy

    # Zone bands around POC
    poc_upper = poc * (1 + zpct)
    poc_lower = poc * (1 - zpct)

    # Long: price crosses from below POC lower band upward
    # Short: price crosses from above POC upper band downward
    sig = pd.Series(0, index=df.index)
    sig[_crossover(close, poc_lower) & (close < poc_upper)] = 1
    sig[_crossunder(close, poc_upper) & (close > poc_lower)] = -1
    return sig


def space_VP_POC(trial):
    return {
        'vp_p':     trial.suggest_int('vp_p', 20, 100),
        'zone_pct': trial.suggest_float('zone_pct', 0.005, 0.02, step=0.005),
    }


# ── 23. VP_ValueArea ─────────────────────────────────────────────────────────

def gen_VP_ValueArea(df, vp_p=50, **kw):
    """Value Area High/Low. Long: price breaks above VAH. Short: price breaks below VAL."""
    vp    = int(vp_p)
    close = df['close']
    tp    = (df['high'] + df['low'] + df['close']) / 3
    vol   = df['volume']

    # VAH = rolling high of volume-weighted typical price + std
    # VAL = rolling low of volume-weighted typical price - std
    vwap  = _vwap_rolling(df, vp)
    tp_std = tp.rolling(vp, min_periods=1).std().fillna(0)

    # Value Area: 70% of volume sits within vwap ± 1 std (simplified)
    vah = vwap + tp_std * 0.7
    val = vwap - tp_std * 0.7

    sig = pd.Series(0, index=df.index)
    sig[_crossover(close, vah)]  = 1
    sig[_crossunder(close, val)] = -1
    return sig


def space_VP_ValueArea(trial):
    return {'vp_p': trial.suggest_int('vp_p', 20, 100)}


# ── 24. Elder_Force ──────────────────────────────────────────────────────────

def gen_Elder_Force(df, fi_p=13, **kw):
    """Elder Force Index. EFI = diff(close) * volume. Long: EFI crosses above 0."""
    fi  = df['close'].diff() * df['volume']
    efi = _ema(fi, fi_p)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(efi, pd.Series(0.0, index=df.index))]  = 1
    sig[_crossunder(efi, pd.Series(0.0, index=df.index))] = -1
    return sig


def space_Elder_Force(trial):
    return {'fi_p': trial.suggest_int('fi_p', 2, 20)}


# ── 25. Elder_Force_EMA ──────────────────────────────────────────────────────

def gen_Elder_Force_EMA(df, fi_p=13, ema_p=50, **kw):
    """EFI + EMA trend. Long: EFI > 0 AND close > EMA."""
    fi    = df['close'].diff() * df['volume']
    efi   = _ema(fi, fi_p)
    ema   = _ema(df['close'], ema_p)
    close = df['close']

    sig = pd.Series(0, index=df.index)
    sig[_crossover(efi, pd.Series(0.0, index=df.index)) & (close > ema)]  = 1
    sig[_crossunder(efi, pd.Series(0.0, index=df.index)) & (close < ema)] = -1
    return sig


def space_Elder_Force_EMA(trial):
    return {
        'fi_p':  trial.suggest_int('fi_p', 2, 20),
        'ema_p': trial.suggest_int('ema_p', 20, 100),
    }


# ── 26. Elder_Force_RSI ──────────────────────────────────────────────────────

def gen_Elder_Force_RSI(df, fi_p=13, rsi_p=14, **kw):
    """EFI + RSI. Long: EFI > 0 AND RSI 40-65."""
    fi    = df['close'].diff() * df['volume']
    efi   = _ema(fi, fi_p)
    rsi   = _rsi(df['close'], rsi_p)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(efi, pd.Series(0.0, index=df.index)) & (rsi > 40) & (rsi < 65)]  = 1
    sig[_crossunder(efi, pd.Series(0.0, index=df.index)) & (rsi > 35) & (rsi < 60)] = -1
    return sig


def space_Elder_Force_RSI(trial):
    return {
        'fi_p':  trial.suggest_int('fi_p', 2, 20),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 27. Mass_Index ───────────────────────────────────────────────────────────

def gen_Mass_Index(df, ema_p=9, sum_p=25, **kw):
    """Mass Index. MI > 27 then drops below 26.5 = reversal signal."""
    ep    = int(ema_p)
    sp    = int(sum_p)
    hl    = df['high'] - df['low']
    ema1  = _ema(hl, ep)
    ema2  = _ema(ema1, ep)
    ratio = ema1 / ema2.replace(0, 1e-9)
    mi    = ratio.rolling(sp, min_periods=1).sum()

    # Reversal bulge: MI crosses below 26.5 after having been above 27
    was_high = (mi.shift(1) > 27) | (mi.shift(2) > 27) | (mi.shift(3) > 27)
    bull_rev = was_high & (mi < 26.5) & (mi.shift(1) >= 26.5)

    # For bearish reversal use price context: if price was rising
    price_rising = df['close'] > df['close'].shift(sp // 2)
    bear_rev = was_high & (mi < 26.5) & (mi.shift(1) >= 26.5) & price_rising

    sig = pd.Series(0, index=df.index)
    sig[bull_rev & ~price_rising] = 1
    sig[bear_rev]                 = -1
    return sig


def space_Mass_Index(trial):
    return {
        'ema_p': trial.suggest_int('ema_p', 9, 15),
        'sum_p': trial.suggest_int('sum_p', 20, 30),
    }


# ── 28. Vortex_Strategy ──────────────────────────────────────────────────────

def gen_Vortex_Strategy(df, vortex_p=14, **kw):
    """Vortex Indicator. Long: VI+ crosses above VI-. Short: VI- crosses above VI+."""
    vi_plus, vi_minus = _vortex(df, vortex_p)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(vi_plus, vi_minus)]  = 1
    sig[_crossunder(vi_plus, vi_minus)] = -1
    return sig


def space_Vortex_Strategy(trial):
    return {'vortex_p': trial.suggest_int('vortex_p', 10, 30)}


# ── 29. Vortex_RSI ───────────────────────────────────────────────────────────

def gen_Vortex_RSI(df, vortex_p=14, rsi_p=14, **kw):
    """Vortex + RSI. Long: VI+ > VI- AND RSI > 50. Short: VI- > VI+ AND RSI < 50."""
    vi_plus, vi_minus = _vortex(df, vortex_p)
    rsi = _rsi(df['close'], rsi_p)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(vi_plus, vi_minus)  & (rsi > 50)] = 1
    sig[_crossunder(vi_plus, vi_minus) & (rsi < 50)] = -1
    return sig


def space_Vortex_RSI(trial):
    return {
        'vortex_p': trial.suggest_int('vortex_p', 10, 30),
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
    }


# ── 30. VWRSI ────────────────────────────────────────────────────────────────

def gen_VWRSI(df, vwrsi_p=14, **kw):
    """Volume-Weighted RSI. RSI applied to VWAP. Long: VWRSI < 30. Short: VWRSI > 70."""
    vwap    = _vwap_rolling(df, vwrsi_p)
    vwrsi   = _rsi(vwap, vwrsi_p)

    sig = pd.Series(0, index=df.index)
    sig[(vwrsi < 30) & (vwrsi.shift(1) >= 30)] = 1
    sig[(vwrsi > 70) & (vwrsi.shift(1) <= 70)] = -1
    return sig


def space_VWRSI(trial):
    return {'vwrsi_p': trial.suggest_int('vwrsi_p', 7, 21)}


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'OBV_EMA': {
        'gen': gen_OBV_EMA,
        'space': space_OBV_EMA,
        'default_params': {'obv_ema_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 3000,
            'description': 'OBV + EMA of OBV crossover. Long: OBV > EMA(OBV). Short: OBV < EMA(OBV).',
        },
    },
    'OBV_Divergence': {
        'gen': gen_OBV_Divergence,
        'space': space_OBV_Divergence,
        'default_params': {'lookback': 10, 'pivot_p': 5},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2500,
            'description': 'OBV divergence vs price. Bullish div: price LL + OBV HL. Bearish: price HH + OBV LH.',
        },
    },
    'OBV_ATR': {
        'gen': gen_OBV_ATR,
        'space': space_OBV_ATR,
        'default_params': {'obv_ema_p': 20, 'atr_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'OBV trend confirmation with ATR volatility filter. Trades only on high volatility.',
        },
    },
    'OBV_MACD': {
        'gen': gen_OBV_MACD,
        'space': space_OBV_MACD,
        'default_params': {'fast': 12, 'slow': 26, 'sig_p': 9},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'MACD applied to OBV series. Long/Short on OBV-MACD vs signal crossovers.',
        },
    },
    'OBV_RSI': {
        'gen': gen_OBV_RSI,
        'space': space_OBV_RSI,
        'default_params': {'obv_rsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'RSI of OBV. Long: RSI(OBV) < 30 AND OBV rising. Short: RSI(OBV) > 70 AND OBV falling.',
        },
    },
    'MFI_Strategy_v2': {
        'gen': gen_MFI_Strategy,
        'space': space_MFI_Strategy,
        'default_params': {'mfi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'Money Flow Index oversold/overbought. Long: MFI < 20. Short: MFI > 80.',
        },
    },
    'MFI_EMA': {
        'gen': gen_MFI_EMA,
        'space': space_MFI_EMA,
        'default_params': {'mfi_p': 14, 'ema_p': 50},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'MFI oversold with EMA trend confirmation. Long: MFI < 30 AND close > EMA.',
        },
    },
    'MFI_Divergence': {
        'gen': gen_MFI_Divergence,
        'space': space_MFI_Divergence,
        'default_params': {'mfi_p': 14, 'lookback': 10},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1800,
            'description': 'MFI divergence detection. Bullish: price LL + MFI HL. Bearish: price HH + MFI LH.',
        },
    },
    'MFI_MACD': {
        'gen': gen_MFI_MACD,
        'space': space_MFI_MACD,
        'default_params': {'mfi_p': 14, 'fast': 12, 'slow': 26},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1200,
            'description': 'MFI oversold with MACD trend confirmation. Long: MFI < 40 AND MACD > 0.',
        },
    },
    'MFI_Stoch': {
        'gen': gen_MFI_Stoch,
        'space': space_MFI_Stoch,
        'default_params': {'mfi_p': 14, 'stoch_k': 14},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1000,
            'description': 'Dual oversold filter: MFI + Stochastic both below threshold simultaneously.',
        },
    },
    'CMF_Strategy': {
        'gen': gen_CMF_Strategy,
        'space': space_CMF_Strategy,
        'default_params': {'cmf_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'Chaikin Money Flow zero-line cross with rising confirmation. Long: CMF > 0 AND rising.',
        },
    },
    'CMF_EMA': {
        'gen': gen_CMF_EMA,
        'space': space_CMF_EMA,
        'default_params': {'cmf_p': 20, 'ema_p': 50},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'CMF zero cross with EMA trend filter. Long: CMF > 0 AND close > EMA.',
        },
    },
    'CMF_RSI': {
        'gen': gen_CMF_RSI,
        'space': space_CMF_RSI,
        'default_params': {'cmf_p': 20, 'rsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1200,
            'description': 'CMF + RSI combo. Long: CMF > 0 AND RSI in 40-65 range.',
        },
    },
    'CMF_MACD': {
        'gen': gen_CMF_MACD,
        'space': space_CMF_MACD,
        'default_params': {'cmf_p': 20, 'fast': 12, 'slow': 26, 'sig_p': 9},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1000,
            'description': 'CMF + MACD confirmation. Long: CMF > 0 AND MACD crosses above signal.',
        },
    },
    'Klinger_OSC': {
        'gen': gen_Klinger_OSC,
        'space': space_Klinger_OSC,
        'default_params': {'fast_p': 34, 'slow_p': 55, 'sig_p': 13},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Klinger Volume Oscillator. Long: KVO crosses above signal line.',
        },
    },
    'Klinger_EMA': {
        'gen': gen_Klinger_EMA,
        'space': space_Klinger_EMA,
        'default_params': {'fast_p': 34, 'slow_p': 55, 'ema_p': 50},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1200,
            'description': 'Klinger OSC with EMA trend filter. Long: KVO > 0 AND close > EMA.',
        },
    },
    'VWAP_Bands': {
        'gen': gen_VWAP_Bands,
        'space': space_VWAP_Bands,
        'default_params': {'vwap_p': 50, 'std_mult': 1.5, 'rsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 4000,
            'description': 'Rolling VWAP with std bands + RSI. Long: price < lower band AND RSI < 40.',
        },
    },
    'VWAP_Pullback': {
        'gen': gen_VWAP_Pullback,
        'space': space_VWAP_Pullback,
        'default_params': {'vwap_p': 50, 'ema_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 3000,
            'description': 'VWAP pullback strategy. Long: EMA > VWAP AND price crosses VWAP upward.',
        },
    },
    'VWAP_MACD': {
        'gen': gen_VWAP_MACD,
        'space': space_VWAP_MACD,
        'default_params': {'vwap_p': 50, 'fast': 12, 'slow': 26},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'VWAP trend + MACD momentum. Long: close > VWAP AND MACD > 0.',
        },
    },
    'VWAP_SuperTrend': {
        'gen': gen_VWAP_SuperTrend,
        'space': space_VWAP_SuperTrend,
        'default_params': {'vwap_p': 50, 'st_p': 10, 'st_mult': 3.0},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'VWAP + SuperTrend dual confirmation. Long: close > VWAP AND ST turns bullish.',
        },
    },
    'VWAP_RSI_ATR': {
        'gen': gen_VWAP_RSI_ATR,
        'space': space_VWAP_RSI_ATR,
        'default_params': {'vwap_p': 50, 'rsi_p': 14, 'atr_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'VWAP mean-reversion with RSI + ATR volatility filter.',
        },
    },
    'VP_POC': {
        'gen': gen_VP_POC,
        'space': space_VP_POC,
        'default_params': {'vp_p': 50, 'zone_pct': 0.01},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2500,
            'description': 'Volume Profile Point of Control. Long when price crosses POC zone from below.',
        },
    },
    'VP_ValueArea': {
        'gen': gen_VP_ValueArea,
        'space': space_VP_ValueArea,
        'default_params': {'vp_p': 50},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'Volume Profile Value Area breakout. Long: price breaks above VAH. Short: below VAL.',
        },
    },
    'Elder_Force': {
        'gen': gen_Elder_Force,
        'space': space_Elder_Force,
        'default_params': {'fi_p': 13},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'Elder Force Index. EFI = diff(close)*volume, smoothed by EMA. Zero-line cross signals.',
        },
    },
    'Elder_Force_EMA': {
        'gen': gen_Elder_Force_EMA,
        'space': space_Elder_Force_EMA,
        'default_params': {'fi_p': 13, 'ema_p': 50},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'EFI + EMA trend filter. Long: EFI > 0 AND close > EMA.',
        },
    },
    'Elder_Force_RSI': {
        'gen': gen_Elder_Force_RSI,
        'space': space_Elder_Force_RSI,
        'default_params': {'fi_p': 13, 'rsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1200,
            'description': 'EFI + RSI quality zone. Long: EFI > 0 AND RSI 40-65.',
        },
    },
    'Mass_Index': {
        'gen': gen_Mass_Index,
        'space': space_Mass_Index,
        'default_params': {'ema_p': 9, 'sum_p': 25},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Mass Index reversal. MI bulge (>27) then drops below 26.5 signals trend reversal.',
        },
    },
    'Vortex_Strategy': {
        'gen': gen_Vortex_Strategy,
        'space': space_Vortex_Strategy,
        'default_params': {'vortex_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Vortex Indicator VI+/VI- crossover. Long: VI+ crosses above VI-.',
        },
    },
    'Vortex_RSI': {
        'gen': gen_Vortex_RSI,
        'space': space_Vortex_RSI,
        'default_params': {'vortex_p': 14, 'rsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1200,
            'description': 'Vortex + RSI confirmation. Long: VI+ > VI- AND RSI > 50.',
        },
    },
    'VWRSI': {
        'gen': gen_VWRSI,
        'space': space_VWRSI,
        'default_params': {'vwrsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'Volume-Weighted RSI: RSI applied to rolling VWAP. Long: VWRSI < 30. Short: VWRSI > 70.',
        },
    },
}
