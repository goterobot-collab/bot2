#!/usr/bin/env python3
"""TV2 BATCH 36 — 30 Squeeze + Divergence Strategies 2026-04-01"""

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


def _keltner(df, p, mult):
    mid   = _ema(df['close'], p)
    atr   = _atr(df, p)
    upper = mid + mult * atr
    lower = mid - mult * atr
    return upper, mid, lower


def _squeeze(df, bb_p, bb_mult, kc_p, kc_mult):
    """LazyBear Squeeze Momentum squeeze detection."""
    bb_upper, bb_mid, bb_lower = _bb(df['close'], bb_p, bb_mult)
    kc_upper, kc_mid, kc_lower = _keltner(df, kc_p, kc_mult)
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    return squeeze_on, bb_mid, kc_mid


def _squeeze_momentum(df, bb_p, bb_mult, kc_p, kc_mult):
    """Squeeze momentum = linear regression of (close - midpoint(BB,KC))."""
    bb_upper, bb_mid, bb_lower = _bb(df['close'], bb_p, bb_mult)
    kc_upper, kc_mid, kc_lower = _keltner(df, kc_p, kc_mult)
    # midpoint of BB and KC midpoints
    mid = (bb_mid + kc_mid) / 2
    delta = df['close'] - mid
    # Linear regression of delta over bb_p bars (last value)
    p   = int(bb_p)
    mom = pd.Series(np.nan, index=df.index)
    arr = delta.values
    x   = np.arange(p, dtype=float)
    for i in range(p - 1, len(arr)):
        y = arr[i - p + 1: i + 1].astype(float)
        if not np.any(np.isnan(y)):
            coef = np.polyfit(x, y, 1)
            mom.iloc[i] = coef[0] * (p - 1) + coef[1]
    return mom.fillna(0)


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


def _cmf(df, p):
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
          (df['high'] - df['low'] + 1e-9)
    mfv = clv * df['volume']
    return mfv.rolling(int(p), min_periods=1).sum() / \
           df['volume'].rolling(int(p), min_periods=1).sum().replace(0, 1e-9)


def _macd(s, fast, slow, sig):
    fast_ema  = _ema(s, fast)
    slow_ema  = _ema(s, slow)
    macd_line = fast_ema - slow_ema
    signal    = _ema(macd_line, sig)
    return macd_line, signal, macd_line - signal


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


def _bull_div(price, osc, lb):
    """Detect regular bullish divergence: price LL + oscillator HL."""
    lb = int(lb)
    # Current vs lb bars ago
    price_ll = price < price.shift(lb)
    osc_hl   = osc > osc.shift(lb)
    return price_ll & osc_hl


def _bear_div(price, osc, lb):
    """Detect regular bearish divergence: price HH + oscillator LH."""
    lb = int(lb)
    price_hh = price > price.shift(lb)
    osc_lh   = osc < osc.shift(lb)
    return price_hh & osc_lh


def _hidden_bull_div(price, osc, lb):
    """Hidden bullish divergence: price HL + oscillator LL (continuation)."""
    lb = int(lb)
    price_hl = price > price.shift(lb)
    osc_ll   = osc < osc.shift(lb)
    return price_hl & osc_ll


def _hidden_bear_div(price, osc, lb):
    """Hidden bearish divergence: price LH + oscillator HH (continuation)."""
    lb = int(lb)
    price_lh = price < price.shift(lb)
    osc_hh   = osc > osc.shift(lb)
    return price_lh & osc_hh


def _ao(df, fast=5, slow=34):
    mid = (df['high'] + df['low']) / 2
    return _sma(mid, fast) - _sma(mid, slow)


def _momentum(s, p):
    return s - s.shift(int(p))


# ── 1. Squeeze_Classic ───────────────────────────────────────────────────────

def gen_Squeeze_Classic(df, bb_p=20, bb_mult=2.0, kc_p=20, kc_mult=1.5, **kw):
    squeeze_on, _, _ = _squeeze(df, bb_p, bb_mult, kc_p, kc_mult)
    mom = _squeeze_momentum(df, bb_p, bb_mult, kc_p, kc_mult)
    # Signal fires when squeeze releases (was on, now off) with momentum direction
    squeeze_release = squeeze_on.shift(1).infer_objects(copy=False).fillna(False) & ~squeeze_on
    sig = pd.Series(0, index=df.index)
    sig[squeeze_release & (mom > 0)] =  1
    sig[squeeze_release & (mom < 0)] = -1
    # Also enter when momentum crosses zero after squeeze
    mom_cross_up   = (mom > 0) & (mom.shift(1) <= 0) & ~squeeze_on
    mom_cross_down = (mom < 0) & (mom.shift(1) >= 0) & ~squeeze_on
    sig[mom_cross_up]   =  1
    sig[mom_cross_down] = -1
    return sig


def space_Squeeze_Classic(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 20, 20),
        'bb_mult': trial.suggest_float('bb_mult', 2.0, 2.0),
        'kc_p':    trial.suggest_int('kc_p', 20, 20),
        'kc_mult': trial.suggest_float('kc_mult', 1.5, 1.5),
    }


# ── 2. Squeeze_RSI ───────────────────────────────────────────────────────────

def gen_Squeeze_RSI(df, bb_p=20, bb_mult=2.0, kc_p=20, kc_mult=1.5, rsi_p=14, **kw):
    squeeze_on, _, _ = _squeeze(df, bb_p, bb_mult, kc_p, kc_mult)
    mom = _squeeze_momentum(df, bb_p, bb_mult, kc_p, kc_mult)
    rsi = _rsi(df['close'], rsi_p)
    squeeze_release = squeeze_on.shift(1).infer_objects(copy=False).fillna(False) & ~squeeze_on
    sig = pd.Series(0, index=df.index)
    sig[squeeze_release & (mom > 0) & (rsi > 50)] =  1
    sig[squeeze_release & (mom < 0) & (rsi < 50)] = -1
    return sig


def space_Squeeze_RSI(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 20, 20),
        'bb_mult': trial.suggest_float('bb_mult', 2.0, 2.0),
        'kc_p':    trial.suggest_int('kc_p', 20, 20),
        'kc_mult': trial.suggest_float('kc_mult', 1.5, 1.5),
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
    }


# ── 3. Squeeze_Vol ───────────────────────────────────────────────────────────

def gen_Squeeze_Vol(df, bb_p=20, bb_mult=2.0, kc_p=20, kc_mult=1.5, vol_p=30, **kw):
    squeeze_on, _, _ = _squeeze(df, bb_p, bb_mult, kc_p, kc_mult)
    mom     = _squeeze_momentum(df, bb_p, bb_mult, kc_p, kc_mult)
    vol_avg = _sma(df['volume'], vol_p)
    squeeze_release = squeeze_on.shift(1).infer_objects(copy=False).fillna(False) & ~squeeze_on
    vol_ok  = df['volume'] > vol_avg
    sig = pd.Series(0, index=df.index)
    sig[squeeze_release & (mom > 0) & vol_ok] =  1
    sig[squeeze_release & (mom < 0) & vol_ok] = -1
    return sig


def space_Squeeze_Vol(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 20, 20),
        'bb_mult': trial.suggest_float('bb_mult', 2.0, 2.0),
        'kc_p':    trial.suggest_int('kc_p', 20, 20),
        'kc_mult': trial.suggest_float('kc_mult', 1.5, 1.5),
        'vol_p':   trial.suggest_int('vol_p', 20, 50),
    }


# ── 4. Squeeze_EMA ───────────────────────────────────────────────────────────

def gen_Squeeze_EMA(df, bb_p=20, bb_mult=2.0, kc_p=20, kc_mult=1.5, ema_p=50, **kw):
    squeeze_on, _, _ = _squeeze(df, bb_p, bb_mult, kc_p, kc_mult)
    mom = _squeeze_momentum(df, bb_p, bb_mult, kc_p, kc_mult)
    ema = _ema(df['close'], ema_p)
    squeeze_release = squeeze_on.shift(1).infer_objects(copy=False).fillna(False) & ~squeeze_on
    sig = pd.Series(0, index=df.index)
    sig[squeeze_release & (mom > 0) & (df['close'] > ema)] =  1
    sig[squeeze_release & (mom < 0) & (df['close'] < ema)] = -1
    return sig


def space_Squeeze_EMA(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 20, 20),
        'bb_mult': trial.suggest_float('bb_mult', 2.0, 2.0),
        'kc_p':    trial.suggest_int('kc_p', 20, 20),
        'kc_mult': trial.suggest_float('kc_mult', 1.5, 1.5),
        'ema_p':   trial.suggest_int('ema_p', 20, 60),
    }


# ── 5. Squeeze_SuperTrend ────────────────────────────────────────────────────

def gen_Squeeze_SuperTrend(df, bb_p=20, bb_mult=2.0, kc_p=20, kc_mult=1.5,
                            st_p=10, st_mult=3.0, **kw):
    squeeze_on, _, _ = _squeeze(df, bb_p, bb_mult, kc_p, kc_mult)
    mom = _squeeze_momentum(df, bb_p, bb_mult, kc_p, kc_mult)
    st  = _supertrend(df, st_p, st_mult)
    squeeze_release = squeeze_on.shift(1).infer_objects(copy=False).fillna(False) & ~squeeze_on
    sig = pd.Series(0, index=df.index)
    sig[squeeze_release & (mom > 0) & (st > 0)] =  1
    sig[squeeze_release & (mom < 0) & (st < 0)] = -1
    return sig


def space_Squeeze_SuperTrend(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 20, 20),
        'bb_mult': trial.suggest_float('bb_mult', 2.0, 2.0),
        'kc_p':    trial.suggest_int('kc_p', 20, 20),
        'kc_mult': trial.suggest_float('kc_mult', 1.5, 1.5),
        'st_p':    trial.suggest_int('st_p', 7, 21),
        'st_mult': trial.suggest_float('st_mult', 2.0, 4.0),
    }


# ── 6. Squeeze_Pro_High ──────────────────────────────────────────────────────

def gen_Squeeze_Pro_High(df, bb_p=20, bb_mult=2.0, kc_p=20, kc_mult_high=1.0, **kw):
    """High squeeze: BB inside tight KC (kc_mult_high < standard 1.5)."""
    squeeze_high, _, _ = _squeeze(df, bb_p, bb_mult, kc_p, kc_mult_high)
    # Standard squeeze with normal mult
    squeeze_std, _, _  = _squeeze(df, bb_p, bb_mult, kc_p, 1.5)
    mom = _squeeze_momentum(df, bb_p, bb_mult, kc_p, kc_mult_high)
    # High squeeze release = stronger signal
    release_high = squeeze_high.shift(1).infer_objects(copy=False).fillna(False) & ~squeeze_high
    sig = pd.Series(0, index=df.index)
    sig[release_high & (mom > 0)] =  1
    sig[release_high & (mom < 0)] = -1
    return sig


def space_Squeeze_Pro_High(trial):
    return {
        'bb_p':         trial.suggest_int('bb_p', 20, 20),
        'bb_mult':      trial.suggest_float('bb_mult', 2.0, 2.0),
        'kc_p':         trial.suggest_int('kc_p', 20, 20),
        'kc_mult_high': trial.suggest_float('kc_mult_high', 1.0, 1.0),
    }


# ── 7. Squeeze_Momentum_Rising ───────────────────────────────────────────────

def gen_Squeeze_Momentum_Rising(df, bb_p=20, bb_mult=2.0, kc_p=20, kc_mult=1.5, **kw):
    squeeze_on, _, _ = _squeeze(df, bb_p, bb_mult, kc_p, kc_mult)
    mom = _squeeze_momentum(df, bb_p, bb_mult, kc_p, kc_mult)
    not_squeeze = ~squeeze_on
    mom_rising  = (mom > 0) & (mom > mom.shift(1))
    mom_falling = (mom < 0) & (mom < mom.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[not_squeeze & mom_rising]  =  1
    sig[not_squeeze & mom_falling] = -1
    return sig


def space_Squeeze_Momentum_Rising(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 20, 20),
        'bb_mult': trial.suggest_float('bb_mult', 2.0, 2.0),
        'kc_p':    trial.suggest_int('kc_p', 20, 20),
        'kc_mult': trial.suggest_float('kc_mult', 1.5, 1.5),
    }


# ── 8. Squeeze_MACD ──────────────────────────────────────────────────────────

def gen_Squeeze_MACD(df, bb_p=20, kc_p=20, fast=12, slow=26, sig_p=9, **kw):
    squeeze_on, _, _ = _squeeze(df, bb_p, 2.0, kc_p, 1.5)
    mom = _squeeze_momentum(df, bb_p, 2.0, kc_p, 1.5)
    macd_line, signal, _ = _macd(df['close'], fast, slow, sig_p)
    squeeze_release = squeeze_on.shift(1).infer_objects(copy=False).fillna(False) & ~squeeze_on
    sig = pd.Series(0, index=df.index)
    sig[squeeze_release & (mom > 0) & (macd_line > signal)] =  1
    sig[squeeze_release & (mom < 0) & (macd_line < signal)] = -1
    return sig


def space_Squeeze_MACD(trial):
    return {
        'bb_p':  trial.suggest_int('bb_p', 20, 20),
        'kc_p':  trial.suggest_int('kc_p', 20, 20),
        'fast':  trial.suggest_int('fast', 8, 16),
        'slow':  trial.suggest_int('slow', 20, 30),
        'sig_p': trial.suggest_int('sig_p', 5, 12),
    }


# ── 9. Squeeze_Divergence ────────────────────────────────────────────────────

def gen_Squeeze_Divergence(df, bb_p=20, bb_mult=2.0, kc_p=20, kc_mult=1.5,
                            lookback=10, **kw):
    squeeze_on, _, _ = _squeeze(df, bb_p, bb_mult, kc_p, kc_mult)
    mom = _squeeze_momentum(df, bb_p, bb_mult, kc_p, kc_mult)
    # Divergence: mom HL + price LL
    bull_div = _bull_div(df['close'], mom, lookback)
    bear_div = _bear_div(df['close'], mom, lookback)
    squeeze_release = squeeze_on.shift(1).infer_objects(copy=False).fillna(False) & ~squeeze_on
    sig = pd.Series(0, index=df.index)
    sig[squeeze_release & bull_div] =  1
    sig[squeeze_release & bear_div] = -1
    return sig


def space_Squeeze_Divergence(trial):
    return {
        'bb_p':     trial.suggest_int('bb_p', 20, 20),
        'bb_mult':  trial.suggest_float('bb_mult', 2.0, 2.0),
        'kc_p':     trial.suggest_int('kc_p', 20, 20),
        'kc_mult':  trial.suggest_float('kc_mult', 1.5, 1.5),
        'lookback': trial.suggest_int('lookback', 5, 15),
    }


# ── 10. Squeeze_ATR ──────────────────────────────────────────────────────────

def gen_Squeeze_ATR(df, bb_p=20, kc_p=20, atr_p=14, **kw):
    squeeze_on, _, _ = _squeeze(df, bb_p, 2.0, kc_p, 1.5)
    mom = _squeeze_momentum(df, bb_p, 2.0, kc_p, 1.5)
    atr     = _atr(df, atr_p)
    atr_avg = _sma(atr, atr_p)
    squeeze_release = squeeze_on.shift(1).infer_objects(copy=False).fillna(False) & ~squeeze_on
    atr_expand = atr > atr_avg
    sig = pd.Series(0, index=df.index)
    sig[squeeze_release & (mom > 0) & atr_expand] =  1
    sig[squeeze_release & (mom < 0) & atr_expand] = -1
    return sig


def space_Squeeze_ATR(trial):
    return {
        'bb_p':  trial.suggest_int('bb_p', 20, 20),
        'kc_p':  trial.suggest_int('kc_p', 20, 20),
        'atr_p': trial.suggest_int('atr_p', 10, 20),
    }


# ── 11. Div_RSI_BB ───────────────────────────────────────────────────────────

def gen_Div_RSI_BB(df, rsi_p=14, bb_p=20, bb_mult=2.0, lookback=10, **kw):
    rsi              = _rsi(df['close'], rsi_p)
    upper, _, lower  = _bb(df['close'], bb_p, bb_mult)
    bull_div = _bull_div(df['close'], rsi, lookback)
    bear_div = _bear_div(df['close'], rsi, lookback)
    sig = pd.Series(0, index=df.index)
    sig[bull_div & (df['close'] <= lower)] =  1
    sig[bear_div & (df['close'] >= upper)] = -1
    return sig


def space_Div_RSI_BB(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'bb_p':     trial.suggest_int('bb_p', 20, 20),
        'bb_mult':  trial.suggest_float('bb_mult', 2.0, 2.0),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 12. Div_MACD_Price ───────────────────────────────────────────────────────

def gen_Div_MACD_Price(df, fast=12, slow=26, sig_p=9, lookback=10, **kw):
    _, _, hist = _macd(df['close'], fast, slow, sig_p)
    bull_div = _bull_div(df['close'], hist, lookback)
    bear_div = _bear_div(df['close'], hist, lookback)
    sig = pd.Series(0, index=df.index)
    sig[bull_div] =  1
    sig[bear_div] = -1
    return sig


def space_Div_MACD_Price(trial):
    return {
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig_p':    trial.suggest_int('sig_p', 5, 12),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 13. Div_Stoch_Price ──────────────────────────────────────────────────────

def gen_Div_Stoch_Price(df, k_p=14, d_p=3, lookback=10, **kw):
    stoch_k  = _stoch(df, k_p)
    stoch_d  = _sma(stoch_k, d_p)
    bull_div = _bull_div(df['close'], stoch_k, lookback)
    bear_div = _bear_div(df['close'], stoch_k, lookback)
    sig = pd.Series(0, index=df.index)
    sig[bull_div & (stoch_k < 30)] =  1
    sig[bear_div & (stoch_k > 70)] = -1
    return sig


def space_Div_Stoch_Price(trial):
    return {
        'k_p':      trial.suggest_int('k_p', 7, 21),
        'd_p':      trial.suggest_int('d_p', 3, 7),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 14. Div_OBV_Price ────────────────────────────────────────────────────────

def gen_Div_OBV_Price(df, obv_ema_p=20, lookback=10, **kw):
    obv      = _obv(df)
    bull_div = _bull_div(df['close'], obv, lookback)
    bear_div = _bear_div(df['close'], obv, lookback)
    sig = pd.Series(0, index=df.index)
    sig[bull_div] =  1
    sig[bear_div] = -1
    return sig


def space_Div_OBV_Price(trial):
    return {
        'obv_ema_p': trial.suggest_int('obv_ema_p', 10, 30),
        'lookback':  trial.suggest_int('lookback', 5, 20),
    }


# ── 15. Div_MFI_Price ────────────────────────────────────────────────────────

def gen_Div_MFI_Price(df, mfi_p=14, lookback=10, **kw):
    mfi      = _mfi(df, mfi_p)
    bull_div = _bull_div(df['close'], mfi, lookback)
    bear_div = _bear_div(df['close'], mfi, lookback)
    sig = pd.Series(0, index=df.index)
    sig[bull_div & (mfi < 40)] =  1
    sig[bear_div & (mfi > 60)] = -1
    return sig


def space_Div_MFI_Price(trial):
    return {
        'mfi_p':    trial.suggest_int('mfi_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 16. Div_CCI_Price ────────────────────────────────────────────────────────

def gen_Div_CCI_Price(df, cci_p=14, lookback=10, **kw):
    cci      = _cci(df, cci_p)
    bull_div = _bull_div(df['close'], cci, lookback)
    bear_div = _bear_div(df['close'], cci, lookback)
    sig = pd.Series(0, index=df.index)
    sig[bull_div & (cci < -50)] =  1
    sig[bear_div & (cci >  50)] = -1
    return sig


def space_Div_CCI_Price(trial):
    return {
        'cci_p':    trial.suggest_int('cci_p', 10, 20),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 17. Div_CMF_Price ────────────────────────────────────────────────────────

def gen_Div_CMF_Price(df, cmf_p=20, lookback=10, **kw):
    cmf      = _cmf(df, cmf_p)
    cl       = df['close']
    lb       = int(lookback)
    # Bull: price declining + CMF improving (less negative)
    price_declining = cl < cl.shift(lb)
    cmf_improving   = cmf > cmf.shift(lb)
    price_rising    = cl > cl.shift(lb)
    cmf_weakening   = cmf < cmf.shift(lb)
    sig = pd.Series(0, index=df.index)
    sig[price_declining & cmf_improving] =  1
    sig[price_rising    & cmf_weakening] = -1
    return sig


def space_Div_CMF_Price(trial):
    return {
        'cmf_p':    trial.suggest_int('cmf_p', 10, 30),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 18. Div_Momentum ─────────────────────────────────────────────────────────

def gen_Div_Momentum(df, mom_p=14, lookback=10, **kw):
    mom      = _momentum(df['close'], mom_p)
    bull_div = _bull_div(df['close'], mom, lookback)
    bear_div = _bear_div(df['close'], mom, lookback)
    sig = pd.Series(0, index=df.index)
    sig[bull_div] =  1
    sig[bear_div] = -1
    return sig


def space_Div_Momentum(trial):
    return {
        'mom_p':    trial.suggest_int('mom_p', 10, 20),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 19. Div_AO_Price ─────────────────────────────────────────────────────────

def gen_Div_AO_Price(df, ao_fast=5, ao_slow=34, lookback=10, **kw):
    ao       = _ao(df, ao_fast, ao_slow)
    # Twin peaks: AO HL (second low above first) while price LL
    bull_div = _bull_div(df['close'], ao, lookback)
    bear_div = _bear_div(df['close'], ao, lookback)
    sig = pd.Series(0, index=df.index)
    sig[bull_div] =  1
    sig[bear_div] = -1
    return sig


def space_Div_AO_Price(trial):
    return {
        'ao_fast':  trial.suggest_int('ao_fast', 3, 8),
        'ao_slow':  trial.suggest_int('ao_slow', 25, 40),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 20. Div_WT_Price ─────────────────────────────────────────────────────────

def gen_Div_WT_Price(df, n1=10, n2=21, lookback=10, **kw):
    wt1, wt2 = _wavetrend(df, n1, n2)
    bull_div = _bull_div(df['close'], wt1, lookback)
    bear_div = _bear_div(df['close'], wt1, lookback)
    sig = pd.Series(0, index=df.index)
    sig[bull_div & (wt1 < -30)] =  1
    sig[bear_div & (wt1 >  30)] = -1
    return sig


def space_Div_WT_Price(trial):
    return {
        'n1':       trial.suggest_int('n1', 7, 15),
        'n2':       trial.suggest_int('n2', 15, 30),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 21. Div_RSI_Hidden ───────────────────────────────────────────────────────

def gen_Div_RSI_Hidden(df, rsi_p=14, lookback=10, **kw):
    rsi      = _rsi(df['close'], rsi_p)
    # Hidden bull: price HL + RSI LL (continuation of uptrend)
    hid_bull = _hidden_bull_div(df['close'], rsi, lookback)
    hid_bear = _hidden_bear_div(df['close'], rsi, lookback)
    sig = pd.Series(0, index=df.index)
    sig[hid_bull] =  1
    sig[hid_bear] = -1
    return sig


def space_Div_RSI_Hidden(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 22. Div_MACD_Hidden ──────────────────────────────────────────────────────

def gen_Div_MACD_Hidden(df, fast=12, slow=26, sig_p=9, lookback=10, **kw):
    _, _, hist = _macd(df['close'], fast, slow, sig_p)
    hid_bull = _hidden_bull_div(df['close'], hist, lookback)
    hid_bear = _hidden_bear_div(df['close'], hist, lookback)
    sig = pd.Series(0, index=df.index)
    sig[hid_bull] =  1
    sig[hid_bear] = -1
    return sig


def space_Div_MACD_Hidden(trial):
    return {
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig_p':    trial.suggest_int('sig_p', 5, 12),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 23. Div_Multi ────────────────────────────────────────────────────────────

def gen_Div_Multi(df, rsi_p=14, fast=12, slow=26, sig_p=9, bb_p=20, lookback=10, **kw):
    rsi              = _rsi(df['close'], rsi_p)
    _, _, hist       = _macd(df['close'], fast, slow, sig_p)
    upper, _, lower  = _bb(df['close'], bb_p, 2.0)
    rsi_bull  = _bull_div(df['close'], rsi,  lookback)
    macd_bull = _bull_div(df['close'], hist, lookback)
    rsi_bear  = _bear_div(df['close'], rsi,  lookback)
    macd_bear = _bear_div(df['close'], hist, lookback)
    sig = pd.Series(0, index=df.index)
    sig[rsi_bull & macd_bull & (df['close'] <= lower)] =  1
    sig[rsi_bear & macd_bear & (df['close'] >= upper)] = -1
    return sig


def space_Div_Multi(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig_p':    trial.suggest_int('sig_p', 5, 12),
        'bb_p':     trial.suggest_int('bb_p', 20, 20),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 24. Div_Amplitude ────────────────────────────────────────────────────────

def gen_Div_Amplitude(df, rsi_p=14, lookback=10, min_div=10.0, **kw):
    rsi = _rsi(df['close'], rsi_p)
    lb  = int(lookback)
    cl  = df['close']
    # Bull: price LL with RSI HL — measure amplitude = RSI gain vs price loss
    price_ll = cl < cl.shift(lb)
    rsi_hl   = rsi > rsi.shift(lb)
    rsi_gain = (rsi - rsi.shift(lb)).abs()
    price_hh = cl > cl.shift(lb)
    rsi_lh   = rsi < rsi.shift(lb)
    rsi_loss = (rsi - rsi.shift(lb)).abs()
    strong_bull = price_ll & rsi_hl & (rsi_gain > min_div)
    strong_bear = price_hh & rsi_lh & (rsi_loss > min_div)
    sig = pd.Series(0, index=df.index)
    sig[strong_bull] =  1
    sig[strong_bear] = -1
    return sig


def space_Div_Amplitude(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
        'min_div':  trial.suggest_float('min_div', 5.0, 20.0),
    }


# ── 25. Div_Volume_Confirm ───────────────────────────────────────────────────

def gen_Div_Volume_Confirm(df, rsi_p=14, lookback=10, vol_p=20, **kw):
    rsi     = _rsi(df['close'], rsi_p)
    vol_avg = _sma(df['volume'], vol_p)
    lb      = int(lookback)
    bull_div = _bull_div(df['close'], rsi, lookback)
    bear_div = _bear_div(df['close'], rsi, lookback)
    # Volume should be increasing for confirmation
    vol_inc = df['volume'] > vol_avg
    sig = pd.Series(0, index=df.index)
    sig[bull_div & vol_inc] =  1
    sig[bear_div & vol_inc] = -1
    return sig


def space_Div_Volume_Confirm(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
        'vol_p':    trial.suggest_int('vol_p', 20, 50),
    }


# ── 26. Div_Squeeze_RSI ──────────────────────────────────────────────────────

def gen_Div_Squeeze_RSI(df, bb_p=20, kc_p=20, rsi_p=14, lookback=10, **kw):
    squeeze_on, _, _ = _squeeze(df, bb_p, 2.0, kc_p, 1.5)
    rsi  = _rsi(df['close'], rsi_p)
    bull_div = _bull_div(df['close'], rsi, lookback)
    bear_div = _bear_div(df['close'], rsi, lookback)
    sig = pd.Series(0, index=df.index)
    sig[squeeze_on & bull_div] =  1
    sig[squeeze_on & bear_div] = -1
    return sig


def space_Div_Squeeze_RSI(trial):
    return {
        'bb_p':     trial.suggest_int('bb_p', 20, 20),
        'kc_p':     trial.suggest_int('kc_p', 20, 20),
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 27. Div_EMA_Confirm ──────────────────────────────────────────────────────

def gen_Div_EMA_Confirm(df, rsi_p=14, lookback=10, ema_p=100, **kw):
    rsi      = _rsi(df['close'], rsi_p)
    ema      = _ema(df['close'], ema_p)
    bull_div = _bull_div(df['close'], rsi, lookback)
    bear_div = _bear_div(df['close'], rsi, lookback)
    sig = pd.Series(0, index=df.index)
    sig[bull_div & (df['close'] > ema)] =  1
    sig[bear_div & (df['close'] < ema)] = -1
    return sig


def space_Div_EMA_Confirm(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
        'ema_p':    trial.suggest_int('ema_p', 50, 200),
    }


# ── 28. Div_ATR_Zone ─────────────────────────────────────────────────────────

def gen_Div_ATR_Zone(df, rsi_p=14, lookback=10, atr_p=14, **kw):
    rsi      = _rsi(df['close'], rsi_p)
    atr      = _atr(df, atr_p)
    atr_avg  = _sma(atr, atr_p)
    high_vol = atr > atr_avg
    bull_div = _bull_div(df['close'], rsi, lookback)
    bear_div = _bear_div(df['close'], rsi, lookback)
    sig = pd.Series(0, index=df.index)
    sig[bull_div & high_vol] =  1
    sig[bear_div & high_vol] = -1
    return sig


def space_Div_ATR_Zone(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
        'atr_p':    trial.suggest_int('atr_p', 10, 20),
    }


# ── 29. Div_Stoch_RSI_Combo ──────────────────────────────────────────────────

def gen_Div_Stoch_RSI_Combo(df, rsi_p=14, stoch_k=14, lookback=10, **kw):
    rsi     = _rsi(df['close'], rsi_p)
    stoch   = _stoch(df, stoch_k)
    rsi_bull  = _bull_div(df['close'], rsi,   lookback)
    stoch_bull = _bull_div(df['close'], stoch, lookback)
    rsi_bear  = _bear_div(df['close'], rsi,   lookback)
    stoch_bear = _bear_div(df['close'], stoch, lookback)
    sig = pd.Series(0, index=df.index)
    sig[rsi_bull & stoch_bull] =  1
    sig[rsi_bear & stoch_bear] = -1
    return sig


def space_Div_Stoch_RSI_Combo(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'stoch_k':  trial.suggest_int('stoch_k', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 30. Div_Full_Scan ────────────────────────────────────────────────────────

def gen_Div_Full_Scan(df, rsi_p=14, fast=12, slow=26, sig_p=9, k_p=14, lookback=10, **kw):
    rsi     = _rsi(df['close'], rsi_p)
    _, _, hist = _macd(df['close'], fast, slow, sig_p)
    stoch   = _stoch(df, k_p)
    obv     = _obv(df)
    rsi_bull   = _bull_div(df['close'], rsi,   lookback).astype(int)
    macd_bull  = _bull_div(df['close'], hist,  lookback).astype(int)
    stoch_bull = _bull_div(df['close'], stoch, lookback).astype(int)
    obv_bull   = _bull_div(df['close'], obv,   lookback).astype(int)
    rsi_bear   = _bear_div(df['close'], rsi,   lookback).astype(int)
    macd_bear  = _bear_div(df['close'], hist,  lookback).astype(int)
    stoch_bear = _bear_div(df['close'], stoch, lookback).astype(int)
    obv_bear   = _bear_div(df['close'], obv,   lookback).astype(int)
    bull_count = rsi_bull + macd_bull + stoch_bull + obv_bull
    bear_count = rsi_bear + macd_bear + stoch_bear + obv_bear
    sig = pd.Series(0, index=df.index)
    sig[bull_count >= 3] =  1
    sig[bear_count >= 3] = -1
    return sig


def space_Div_Full_Scan(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig_p':    trial.suggest_int('sig_p', 5, 12),
        'k_p':      trial.suggest_int('k_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Squeeze_Classic': {
        'gen': gen_Squeeze_Classic,
        'space': space_Squeeze_Classic,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'kc_p': 20, 'kc_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 8000,
                 'description': 'LazyBear Squeeze Momentum — BB inside KC fires directional momentum.'},
    },
    'Squeeze_RSI': {
        'gen': gen_Squeeze_RSI,
        'space': space_Squeeze_RSI,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'kc_p': 20, 'kc_mult': 1.5, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 5000,
                 'description': 'Squeeze release filtered by RSI above/below 50 for momentum direction.'},
    },
    'Squeeze_Vol': {
        'gen': gen_Squeeze_Vol,
        'space': space_Squeeze_Vol,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'kc_p': 20, 'kc_mult': 1.5, 'vol_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Squeeze momentum release with volume expansion confirmation.'},
    },
    'Squeeze_EMA': {
        'gen': gen_Squeeze_EMA,
        'space': space_Squeeze_EMA,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'kc_p': 20, 'kc_mult': 1.5, 'ema_p': 50},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Squeeze fires in EMA trend direction for higher probability entries.'},
    },
    'Squeeze_SuperTrend': {
        'gen': gen_Squeeze_SuperTrend,
        'space': space_Squeeze_SuperTrend,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'kc_p': 20, 'kc_mult': 1.5,
                           'st_p': 10, 'st_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Squeeze release with SuperTrend dual confirmation.'},
    },
    'Squeeze_Pro_High': {
        'gen': gen_Squeeze_Pro_High,
        'space': space_Squeeze_Pro_High,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'kc_p': 20, 'kc_mult_high': 1.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4500,
                 'description': 'High squeeze (tighter KC) = stronger compression = larger breakout.'},
    },
    'Squeeze_Momentum_Rising': {
        'gen': gen_Squeeze_Momentum_Rising,
        'space': space_Squeeze_Momentum_Rising,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'kc_p': 20, 'kc_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3200,
                 'description': 'Momentum rising or falling after squeeze releases — trend continuation.'},
    },
    'Squeeze_MACD': {
        'gen': gen_Squeeze_MACD,
        'space': space_Squeeze_MACD,
        'default_params': {'bb_p': 20, 'kc_p': 20, 'fast': 12, 'slow': 26, 'sig_p': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2800,
                 'description': 'Squeeze fires aligned with MACD cross direction.'},
    },
    'Squeeze_Divergence': {
        'gen': gen_Squeeze_Divergence,
        'space': space_Squeeze_Divergence,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'kc_p': 20, 'kc_mult': 1.5, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3100,
                 'description': 'Squeeze release + momentum divergence from price — reversal setup.'},
    },
    'Squeeze_ATR': {
        'gen': gen_Squeeze_ATR,
        'space': space_Squeeze_ATR,
        'default_params': {'bb_p': 20, 'kc_p': 20, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Squeeze with ATR expansion confirmation — volatility breakout filter.'},
    },
    'Div_RSI_BB': {
        'gen': gen_Div_RSI_BB,
        'space': space_Div_RSI_BB,
        'default_params': {'rsi_p': 14, 'bb_p': 20, 'bb_mult': 2.0, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 5500,
                 'description': 'RSI divergence at Bollinger Band extreme — reversal confluence.'},
    },
    'Div_MACD_Price': {
        'gen': gen_Div_MACD_Price,
        'space': space_Div_MACD_Price,
        'default_params': {'fast': 12, 'slow': 26, 'sig_p': 9, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4800,
                 'description': 'MACD histogram divergence vs price — classic divergence system.'},
    },
    'Div_Stoch_Price': {
        'gen': gen_Div_Stoch_Price,
        'space': space_Div_Stoch_Price,
        'default_params': {'k_p': 14, 'd_p': 3, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4200,
                 'description': 'Stochastic divergence with oversold confirmation for reversals.'},
    },
    'Div_OBV_Price': {
        'gen': gen_Div_OBV_Price,
        'space': space_Div_OBV_Price,
        'default_params': {'obv_ema_p': 20, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3800,
                 'description': 'OBV divergence — volume not confirming price direction.'},
    },
    'Div_MFI_Price': {
        'gen': gen_Div_MFI_Price,
        'space': space_Div_MFI_Price,
        'default_params': {'mfi_p': 14, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Money Flow Index divergence at oversold levels.'},
    },
    'Div_CCI_Price': {
        'gen': gen_Div_CCI_Price,
        'space': space_Div_CCI_Price,
        'default_params': {'cci_p': 14, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3200,
                 'description': 'CCI divergence in extreme territory — strong mean reversion signal.'},
    },
    'Div_CMF_Price': {
        'gen': gen_Div_CMF_Price,
        'space': space_Div_CMF_Price,
        'default_params': {'cmf_p': 20, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2900,
                 'description': 'Chaikin Money Flow improving while price declines — accumulation signal.'},
    },
    'Div_Momentum': {
        'gen': gen_Div_Momentum,
        'space': space_Div_Momentum,
        'default_params': {'mom_p': 14, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2700,
                 'description': 'Raw momentum divergence from price — simple but effective.'},
    },
    'Div_AO_Price': {
        'gen': gen_Div_AO_Price,
        'space': space_Div_AO_Price,
        'default_params': {'ao_fast': 5, 'ao_slow': 34, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3400,
                 'description': 'Awesome Oscillator twin-peaks divergence pattern.'},
    },
    'Div_WT_Price': {
        'gen': gen_Div_WT_Price,
        'space': space_Div_WT_Price,
        'default_params': {'n1': 10, 'n2': 21, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4100,
                 'description': 'WaveTrend divergence from price in OB/OS zones.'},
    },
    'Div_RSI_Hidden': {
        'gen': gen_Div_RSI_Hidden,
        'space': space_Div_RSI_Hidden,
        'default_params': {'rsi_p': 14, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3600,
                 'description': 'Hidden RSI divergence: price HL + RSI LL signals trend continuation.'},
    },
    'Div_MACD_Hidden': {
        'gen': gen_Div_MACD_Hidden,
        'space': space_Div_MACD_Hidden,
        'default_params': {'fast': 12, 'slow': 26, 'sig_p': 9, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3300,
                 'description': 'Hidden MACD divergence — trend continuation after pullback.'},
    },
    'Div_Multi': {
        'gen': gen_Div_Multi,
        'space': space_Div_Multi,
        'default_params': {'rsi_p': 14, 'fast': 12, 'slow': 26, 'sig_p': 9, 'bb_p': 20, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 5200,
                 'description': 'Multi-divergence: RSI + MACD both diverging at BB extreme — high confluence.'},
    },
    'Div_Amplitude': {
        'gen': gen_Div_Amplitude,
        'space': space_Div_Amplitude,
        'default_params': {'rsi_p': 14, 'lookback': 10, 'min_div': 10.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2800,
                 'description': 'Strong divergences only: oscillator delta exceeds amplitude threshold.'},
    },
    'Div_Volume_Confirm': {
        'gen': gen_Div_Volume_Confirm,
        'space': space_Div_Volume_Confirm,
        'default_params': {'rsi_p': 14, 'lookback': 10, 'vol_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3100,
                 'description': 'RSI divergence with increasing volume — accumulation/distribution.'},
    },
    'Div_Squeeze_RSI': {
        'gen': gen_Div_Squeeze_RSI,
        'space': space_Div_Squeeze_RSI,
        'default_params': {'bb_p': 20, 'kc_p': 20, 'rsi_p': 14, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3900,
                 'description': 'RSI divergence inside active squeeze — compression + divergence combo.'},
    },
    'Div_EMA_Confirm': {
        'gen': gen_Div_EMA_Confirm,
        'space': space_Div_EMA_Confirm,
        'default_params': {'rsi_p': 14, 'lookback': 10, 'ema_p': 100},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2900,
                 'description': 'RSI divergence with EMA trend confirmation — higher timeframe filter.'},
    },
    'Div_ATR_Zone': {
        'gen': gen_Div_ATR_Zone,
        'space': space_Div_ATR_Zone,
        'default_params': {'rsi_p': 14, 'lookback': 10, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2700,
                 'description': 'RSI divergence during high volatility — ATR-filtered reversal.'},
    },
    'Div_Stoch_RSI_Combo': {
        'gen': gen_Div_Stoch_RSI_Combo,
        'space': space_Div_Stoch_RSI_Combo,
        'default_params': {'rsi_p': 14, 'stoch_k': 14, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3700,
                 'description': 'Stochastic + RSI both showing divergence simultaneously.'},
    },
    'Div_Full_Scan': {
        'gen': gen_Div_Full_Scan,
        'space': space_Div_Full_Scan,
        'default_params': {'rsi_p': 14, 'fast': 12, 'slow': 26, 'sig_p': 9, 'k_p': 14, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 6500,
                 'description': 'Full divergence scan: 4 indicators, 3+ must agree — maximum confluence.'},
    },
}
