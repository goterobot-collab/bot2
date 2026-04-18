#!/usr/bin/env python3
"""
Estrategias: Multi_Confirm, QuantNomad_V2, Adaptive_RSI, Dynamic_Support, TPSL_Strategy, Trailing_TP
Pine versions: v4/v6
Batch: 5c
"""
import pandas as pd
import numpy as np

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ema(s, p): return s.ewm(span=p, adjust=False).mean()
def _sma(s, p): return s.rolling(p).mean()
def _rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0).ewm(span=p, adjust=False).mean()
    l = (-d.where(d < 0, 0.0)).ewm(span=p, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))
def _atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()
def _wma(s, p):
    w = np.arange(1, p+1, dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)

# ─── 1. Multi_Confirm ─────────────────────────────────────────────────────────

def gen_Multi_Confirm(df, length=20, boll_mult=1.0, upper_q=0.95, lower_q=0.05, ema_length=50):
    close = df['close']
    ema_f = _ema(close, ema_length)

    upper_band = close.rolling(length).quantile(upper_q)
    lower_band = close.rolling(length).quantile(lower_q)

    co_above_lo = (close.shift(1) < lower_band.shift(1)) & (close >= lower_band)
    co_below_hi = (close.shift(1) > upper_band.shift(1)) & (close <= upper_band)

    sig = pd.Series(0, index=df.index)
    sig[co_above_lo & (close > ema_f)] = 1
    sig[co_below_hi & (close < ema_f)] = -1
    return sig


def space_Multi_Confirm(trial):
    return {
        'length':     trial.suggest_int('length', 10, 50),
        'boll_mult':  trial.suggest_float('boll_mult', 0.5, 3.0),
        'upper_q':    trial.suggest_float('upper_q', 0.80, 0.99),
        'lower_q':    trial.suggest_float('lower_q', 0.01, 0.20),
        'ema_length': trial.suggest_int('ema_length', 20, 200),
    }

# ─── 2. QuantNomad_V2 (UT Bot) ────────────────────────────────────────────────

def gen_QuantNomad_V2(df, atr_sensitivity=1, atr_period=10):
    close = df['close']
    atr_vals = _atr(df['high'], df['low'], close, atr_period)
    n_loss = atr_sensitivity * atr_vals

    src = close.values
    nl  = n_loss.values
    n   = len(src)

    stop = np.zeros(n)
    sig_arr = np.zeros(n)

    for i in range(1, n):
        if np.isnan(nl[i]):
            stop[i] = stop[i-1]
            continue
        prev_s = stop[i-1]
        prev_c = src[i-1]
        curr_c = src[i]

        if curr_c > prev_s and prev_c > prev_s:
            stop[i] = max(prev_s, curr_c - nl[i])
        elif curr_c < prev_s and prev_c < prev_s:
            stop[i] = min(prev_s, curr_c + nl[i])
        elif curr_c > prev_s:
            stop[i] = curr_c - nl[i]
        else:
            stop[i] = curr_c + nl[i]

        buy  = (prev_c < prev_s) and (curr_c > stop[i])
        sell = (prev_c > prev_s) and (curr_c < stop[i])

        if buy:
            sig_arr[i] = 1
        elif sell:
            sig_arr[i] = -1

    return pd.Series(sig_arr, index=df.index)


def space_QuantNomad_V2(trial):
    return {
        'atr_sensitivity': trial.suggest_int('atr_sensitivity', 1, 5),
        'atr_period':      trial.suggest_int('atr_period', 5, 30),
    }

# ─── 3. Adaptive_RSI ──────────────────────────────────────────────────────────

def _t3(src, length, factor):
    """Tillson T3 moving average."""
    e1 = _ema(src, length)
    e2 = _ema(e1, length)
    e3 = _ema(e2, length)
    e4 = _ema(e3, length)
    e5 = _ema(e4, length)
    e6 = _ema(e5, length)
    c1 = -(factor ** 3)
    c2 = 3 * factor**2 + 3 * factor**3
    c3 = -6 * factor**2 - 3 * factor - 3 * factor**3
    c4 = 1 + 3 * factor + factor**3 + 3 * factor**2
    return c1*e6 + c2*e5 + c3*e4 + c4*e3


def _psar(high, low, start=0.02, inc=0.02, max_af=0.2):
    """Parabolic SAR (numpy loop, no look-ahead)."""
    h = high.values
    l = low.values
    n = len(h)
    sar = np.zeros(n)
    sar[0] = l[0]
    bull = True
    ep = h[0]
    af = start

    for i in range(1, n):
        if bull:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            sar[i] = min(sar[i], l[i-1], l[i-2] if i > 1 else l[i-1])
            if l[i] < sar[i]:
                bull = False
                sar[i] = ep
                ep = l[i]
                af = start
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + inc, max_af)
        else:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            sar[i] = max(sar[i], h[i-1], h[i-2] if i > 1 else h[i-1])
            if h[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = h[i]
                af = start
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + inc, max_af)

    return pd.Series(sar, index=high.index)


def gen_Adaptive_RSI(df, rsi_len=14, t3_len=6, t3_factor=0.7,
                     sar_start=0.02, sar_inc=0.02, sar_max=0.2):
    close = df['close']
    t3_line = _t3(close, t3_len, t3_factor)
    sar_line = _psar(df['high'], df['low'], sar_start, sar_inc, sar_max)

    # T3 crossover SAR: prev T3 < prev SAR, now T3 >= SAR
    long_cond  = (t3_line.shift(1) < sar_line.shift(1)) & (t3_line >= sar_line)
    short_cond = (t3_line.shift(1) > sar_line.shift(1)) & (t3_line <= sar_line)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_Adaptive_RSI(trial):
    return {
        'rsi_len':   trial.suggest_int('rsi_len', 7, 30),
        't3_len':    trial.suggest_int('t3_len', 3, 20),
        't3_factor': trial.suggest_float('t3_factor', 0.3, 0.9),
        'sar_start': trial.suggest_float('sar_start', 0.01, 0.05),
        'sar_inc':   trial.suggest_float('sar_inc', 0.01, 0.05),
        'sar_max':   trial.suggest_float('sar_max', 0.1, 0.5),
    }

# ─── 4. Dynamic_Support ───────────────────────────────────────────────────────

def gen_Dynamic_Support(df, pivot_len=2, sr_dist_pct=0.4):
    close, high, low = df['close'], df['high'], df['low']
    c = close.values
    h = high.values
    l = low.values
    n = len(df)

    resist = np.full(n, np.nan)
    support = np.full(n, np.nan)

    for i in range(pivot_len, n - pivot_len):
        window_h = h[i-pivot_len:i+pivot_len+1]
        window_l = l[i-pivot_len:i+pivot_len+1]
        if h[i] == np.max(window_h):
            resist[i+pivot_len] = h[i]   # confirmed pivot_len bars later
        if l[i] == np.min(window_l):
            support[i+pivot_len] = l[i]

    # Propagate forward (last known pivot)
    res_prop = np.full(n, np.nan)
    sup_prop = np.full(n, np.nan)
    for i in range(n):
        res_prop[i] = resist[i] if not np.isnan(resist[i]) else (res_prop[i-1] if i > 0 else np.nan)
        sup_prop[i] = support[i] if not np.isnan(support[i]) else (sup_prop[i-1] if i > 0 else np.nan)

    res_s = pd.Series(res_prop, index=df.index)
    sup_s = pd.Series(sup_prop, index=df.index)

    near_res = (close >= res_s * (1 - sr_dist_pct/100)) & (close <= res_s * (1 + sr_dist_pct/100))
    near_sup = (close >= sup_s * (1 - sr_dist_pct/100)) & (close <= sup_s * (1 + sr_dist_pct/100))

    co_above_sup = (close.shift(1) < sup_s.shift(1)) & (close >= sup_s)
    co_below_res = (close.shift(1) > res_s.shift(1)) & (close <= res_s)

    sig = pd.Series(0, index=df.index)
    sig[near_sup & co_above_sup] = 1
    sig[near_res & co_below_res] = -1
    return sig


def space_Dynamic_Support(trial):
    return {
        'pivot_len':    trial.suggest_int('pivot_len', 1, 5),
        'sr_dist_pct':  trial.suggest_float('sr_dist_pct', 0.1, 2.0),
    }

# ─── 5. TPSL_Strategy ─────────────────────────────────────────────────────────

def gen_TPSL_Strategy(df, slow=55, middle=21, fast=9, trend_ma=200, rsi_len=14):
    close = df['close']
    slow_ema = _ema(close, slow)
    mid_ema  = _ema(close, middle)
    fast_ema = _ema(close, fast)
    sma200   = _sma(close, trend_ma)
    rsi_val  = _rsi(close, rsi_len)

    go_long  = (mid_ema.shift(1) < slow_ema.shift(1)) & (mid_ema >= slow_ema)
    go_short = (mid_ema.shift(1) > slow_ema.shift(1)) & (mid_ema <= slow_ema)

    long_cond  = go_long  & (close > sma200) & (rsi_val < 80)
    short_cond = go_short & (close < sma200) & (rsi_val > 20)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_TPSL_Strategy(trial):
    return {
        'slow':     trial.suggest_int('slow', 30, 100),
        'middle':   trial.suggest_int('middle', 10, 50),
        'fast':     trial.suggest_int('fast', 3, 20),
        'trend_ma': trial.suggest_int('trend_ma', 100, 300),
        'rsi_len':  trial.suggest_int('rsi_len', 7, 30),
    }

# ─── 6. Trailing_TP ───────────────────────────────────────────────────────────

def gen_Trailing_TP(df, fast_period=20, slow_period=50):
    """
    Pine naming quirk: fastSMA = sma(src, slowLength=20), slowSMA = sma(src, fastLength=50)
    Signal: SMA(20) crossover SMA(50) → Long; crossunder → Short
    """
    close = df['close']
    fast_sma = _sma(close, fast_period)
    slow_sma = _sma(close, slow_period)

    long_cond  = (fast_sma.shift(1) < slow_sma.shift(1)) & (fast_sma >= slow_sma)
    short_cond = (fast_sma.shift(1) > slow_sma.shift(1)) & (fast_sma <= slow_sma)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_Trailing_TP(trial):
    return {
        'fast_period': trial.suggest_int('fast_period', 5, 50),
        'slow_period': trial.suggest_int('slow_period', 20, 200),
    }

# ─── STRATEGY_EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Multi_Confirm': {
        'gen':   gen_Multi_Confirm,
        'space': space_Multi_Confirm,
        'params': dict(length=20, boll_mult=1.0, upper_q=0.95, lower_q=0.05, ema_length=50),
        'pine_version': 'v6',
    },
    'QuantNomad_V2': {
        'gen':   gen_QuantNomad_V2,
        'space': space_QuantNomad_V2,
        'params': dict(atr_sensitivity=1, atr_period=10),
        'pine_version': 'v4',
        'tv_likes': 5422,
    },
    'Adaptive_RSI': {
        'gen':   gen_Adaptive_RSI,
        'space': space_Adaptive_RSI,
        'params': dict(rsi_len=14, t3_len=6, t3_factor=0.7, sar_start=0.02, sar_inc=0.02, sar_max=0.2),
        'pine_version': 'v6',
        'tv_likes': 2109,
    },
    'Dynamic_Support': {
        'gen':   gen_Dynamic_Support,
        'space': space_Dynamic_Support,
        'params': dict(pivot_len=2, sr_dist_pct=0.4),
        'pine_version': 'v6',
    },
    'TPSL_Strategy': {
        'gen':   gen_TPSL_Strategy,
        'space': space_TPSL_Strategy,
        'params': dict(slow=55, middle=21, fast=9, trend_ma=200, rsi_len=14),
        'pine_version': 'v4',
        'tv_likes': 2351,
    },
    'Trailing_TP': {
        'gen':   gen_Trailing_TP,
        'space': space_Trailing_TP,
        'params': dict(fast_period=20, slow_period=50),
        'pine_version': 'v4',
        'tv_likes': 1684,
    },
}
