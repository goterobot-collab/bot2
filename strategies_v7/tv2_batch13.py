#!/usr/bin/env python3
"""TV2 BATCH 13 — 30 estrategias QQE/Chandelier/Coral/Squeeze/SSL 2026-04-01"""

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


def _stoch_rsi(s, rsi_p, stoch_p, k_p, d_p):
    rsi = _rsi(s, rsi_p)
    lo  = rsi.rolling(int(stoch_p), min_periods=1).min()
    hi  = rsi.rolling(int(stoch_p), min_periods=1).max()
    k   = _sma(((rsi - lo) / (hi - lo + 1e-9) * 100), k_p)
    d   = _sma(k, d_p)
    return k, d


# ── QQE core helper ──────────────────────────────────────────────────────────

def _qqe_lines(close, rsi_p, sf, qq_factor):
    """Returns (rsi_ma, qqe_long, qqe_short) as Series."""
    rsi_p   = int(rsi_p)
    sf      = int(sf)
    rsi     = _rsi(close, rsi_p)
    rsi_ma  = _ema(rsi, sf)

    wilders_p = rsi_p * 2 - 1
    atr_rsi   = _rma((rsi_ma - rsi_ma.shift(1)).abs(), wilders_p) * qq_factor

    vals   = rsi_ma.values
    atr_v  = atr_rsi.values
    n      = len(vals)
    long_  = np.full(n, np.nan)
    short_ = np.full(n, np.nan)
    trend  = np.ones(n)          # 1=bull, -1=bear
    long_[0]  = vals[0] - atr_v[0]
    short_[0] = vals[0] + atr_v[0]

    for i in range(1, n):
        v   = vals[i]
        a   = atr_v[i]
        pL  = long_[i-1]
        pS  = short_[i-1]
        pT  = trend[i-1]

        newL = v - a
        newS = v + a

        if pT == 1:
            newL = max(newL, pL)
        else:
            newL = newL

        if pT == -1:
            newS = min(newS, pS)
        else:
            newS = newS

        if pT == 1 and v < pL:
            trend[i] = -1
            long_[i]  = newL
            short_[i] = newS
        elif pT == -1 and v > pS:
            trend[i] = 1
            long_[i]  = newL
            short_[i] = newS
        else:
            trend[i] = pT
            long_[i]  = newL
            short_[i] = newS

    return (pd.Series(rsi_ma.values, index=close.index),
            pd.Series(long_,  index=close.index),
            pd.Series(short_, index=close.index))


# ══════════════════════════════════════════════════════════════════════════════
# QQE FAMILY
# ══════════════════════════════════════════════════════════════════════════════

def gen_QQE_Mod(df, rsi_p=6, sf=5, qq_factor=4.236, **kw):
    """QQE Mod — RSI smoothed twice + trailing lines. Long: QQE line > 50."""
    close = df['close']
    rsi_ma, ql, qs = _qqe_lines(close, rsi_p, sf, qq_factor)

    sig = pd.Series(0, index=df.index)
    above = rsi_ma > 50
    cross_up   = _crossover(rsi_ma, pd.Series(50, index=rsi_ma.index))
    cross_down = _crossunder(rsi_ma, pd.Series(50, index=rsi_ma.index))

    sig[cross_up]   = 1
    sig[cross_down] = -1
    return sig


def space_QQE_Mod(trial):
    return {
        'rsi_p':     trial.suggest_int('rsi_p',     6,  14),
        'sf':        trial.suggest_int('sf',         3,   8),
        'qq_factor': trial.suggest_float('qq_factor', 3.0, 6.0),
    }


def gen_QQE_Classic(df, rsi_p=9, sf=5, factor=3.0, **kw):
    """Classic QQE — fast and slow QQE lines cross."""
    close = df['close']
    _, ql_fast, qs_fast = _qqe_lines(close, rsi_p, sf, factor)
    _, ql_slow, qs_slow = _qqe_lines(close, rsi_p + 2, sf + 2, factor)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(ql_fast, ql_slow)]   = 1
    sig[_crossunder(ql_fast, ql_slow)]  = -1
    return sig


def space_QQE_Classic(trial):
    return {
        'rsi_p':  trial.suggest_int('rsi_p',  6,  14),
        'sf':     trial.suggest_int('sf',      3,   8),
        'factor': trial.suggest_float('factor', 2.0, 5.0),
    }


def gen_QQE_Histo(df, rsi_p=6, sf=5, **kw):
    """QQE Histogram — QQE histo turns positive/negative."""
    close = df['close']
    rsi_ma, ql, qs = _qqe_lines(close, rsi_p, sf, 4.236)
    histo = rsi_ma - 50

    sig = pd.Series(0, index=df.index)
    sig[_crossover(histo, pd.Series(0, index=histo.index))]   = 1
    sig[_crossunder(histo, pd.Series(0, index=histo.index))]  = -1
    return sig


def space_QQE_Histo(trial):
    return {
        'rsi_p': trial.suggest_int('rsi_p', 6, 14),
        'sf':    trial.suggest_int('sf',    3,  8),
    }


def gen_QQE_ATR_BB(df, rsi_p=6, sf=5, bb_mult=1.0, **kw):
    """QQE + BB bands on RSI smoothed. Long: RSI_EMA crosses above BB lower."""
    close = df['close']
    rsi_ma, ql, qs = _qqe_lines(close, rsi_p, sf, 4.236)

    bb_p  = int(sf) * 4
    mid   = _sma(rsi_ma, bb_p)
    std   = rsi_ma.rolling(bb_p, min_periods=1).std().fillna(0)
    bb_lo = mid - bb_mult * std

    sig = pd.Series(0, index=df.index)
    sig[_crossover(rsi_ma, bb_lo)]  = 1
    sig[_crossunder(rsi_ma, mid)]   = -1
    return sig


def space_QQE_ATR_BB(trial):
    return {
        'rsi_p':   trial.suggest_int('rsi_p',   6, 14),
        'sf':      trial.suggest_int('sf',       3,  8),
        'bb_mult': trial.suggest_float('bb_mult', 0.5, 2.0),
    }


def gen_QQE_RSI_Stoch(df, rsi_p=6, stoch_k=9, **kw):
    """QQE + Stoch RSI — Long: QQE > 50 AND K crosses D from below 20."""
    close = df['close']
    rsi_ma, ql, qs = _qqe_lines(close, rsi_p, 5, 4.236)
    K, D = _stoch_rsi(close, rsi_p, int(stoch_k), 3, 3)

    qqe_bull = rsi_ma > 50
    k_cross  = _crossover(K, D)
    sig = pd.Series(0, index=df.index)
    sig[(k_cross) & (K < 30) & qqe_bull]  = 1
    sig[_crossunder(K, D) & (K > 70) & (rsi_ma < 50)] = -1
    return sig


def space_QQE_RSI_Stoch(trial):
    return {
        'rsi_p':  trial.suggest_int('rsi_p',   6, 14),
        'stoch_k': trial.suggest_int('stoch_k', 7, 14),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CHANDELIER EXIT FAMILY
# ══════════════════════════════════════════════════════════════════════════════

def _chandelier(df, atr_p, mult):
    """Returns (ce_long, ce_short, bull_flag) as Series."""
    atr_p = int(atr_p)
    a     = _atr(df, atr_p)
    hh    = df['high'].rolling(atr_p, min_periods=1).max()
    ll    = df['low'].rolling(atr_p, min_periods=1).min()

    ce_long_raw  = hh - mult * a
    ce_short_raw = ll + mult * a

    close   = df['close'].values
    cl_v    = ce_long_raw.values
    cs_v    = ce_short_raw.values
    n       = len(close)
    cl      = np.full(n, np.nan)
    cs      = np.full(n, np.nan)
    bull    = np.ones(n, dtype=bool)
    cl[0]   = cl_v[0]
    cs[0]   = cs_v[0]

    for i in range(1, n):
        cl[i] = max(cl_v[i], cl[i-1]) if bull[i-1] else cl_v[i]
        cs[i] = min(cs_v[i], cs[i-1]) if not bull[i-1] else cs_v[i]
        if bull[i-1]:
            bull[i] = close[i] >= cl[i]
        else:
            bull[i] = close[i] > cs[i]

    return (pd.Series(cl, index=df.index),
            pd.Series(cs, index=df.index),
            pd.Series(bull.astype(int), index=df.index))


def gen_Chandelier_Exit(df, atr_p=22, mult=3.0, **kw):
    """Chandelier Exit — ATR trailing stop. Bull/bear direction changes."""
    cl, cs, bull = _chandelier(df, atr_p, mult)
    sig = pd.Series(0, index=df.index)
    sig[_crossover(bull, pd.Series(0.5, index=bull.index))]  = 1
    sig[_crossunder(bull, pd.Series(0.5, index=bull.index))] = -1
    return sig


def space_Chandelier_Exit(trial):
    return {
        'atr_p': trial.suggest_int('atr_p',   14, 22),
        'mult':  trial.suggest_float('mult',    2.0, 5.0),
    }


def gen_Chandelier_ATR_EMA(df, atr_p=22, mult=3.0, ema_p=50, **kw):
    """Chandelier + EMA trend filter. Long: CE bullish AND close > EMA."""
    cl, cs, bull = _chandelier(df, atr_p, mult)
    ema = _ema(df['close'], ema_p)
    sig = pd.Series(0, index=df.index)
    change_up   = _crossover(bull, pd.Series(0.5, index=bull.index))
    change_down = _crossunder(bull, pd.Series(0.5, index=bull.index))
    sig[change_up   & (df['close'] > ema)] = 1
    sig[change_down & (df['close'] < ema)] = -1
    return sig


def space_Chandelier_ATR_EMA(trial):
    return {
        'atr_p': trial.suggest_int('atr_p',   14, 22),
        'mult':  trial.suggest_float('mult',    2.0, 4.0),
        'ema_p': trial.suggest_int('ema_p',    20, 60),
    }


def gen_Chandelier_3x(df, fast_mult=2.0, mid_mult=3.0, slow_mult=4.0, **kw):
    """Three Chandelier exits (fast/mid/slow) — Long: all 3 bullish."""
    _, _, bull_f = _chandelier(df, 14, fast_mult)
    _, _, bull_m = _chandelier(df, 18, mid_mult)
    _, _, bull_s = _chandelier(df, 22, slow_mult)

    all_bull = (bull_f == 1) & (bull_m == 1) & (bull_s == 1)
    all_bear = (bull_f == 0) & (bull_m == 0) & (bull_s == 0)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(all_bull.astype(float), pd.Series(0.5, index=df.index))]  = 1
    sig[_crossover(all_bear.astype(float), pd.Series(0.5, index=df.index))]  = -1
    return sig


def space_Chandelier_3x(trial):
    return {
        'fast_mult': trial.suggest_float('fast_mult', 1.5, 2.5),
        'mid_mult':  trial.suggest_float('mid_mult',  2.5, 3.5),
        'slow_mult': trial.suggest_float('slow_mult', 3.5, 5.0),
    }


def gen_Chandelier_RSI(df, atr_p=22, mult=3.0, rsi_p=14, **kw):
    """CE + RSI filter. Long: CE turns bull AND RSI in 40-70 range."""
    cl, cs, bull = _chandelier(df, atr_p, mult)
    rsi = _rsi(df['close'], rsi_p)
    sig = pd.Series(0, index=df.index)
    change_up   = _crossover(bull, pd.Series(0.5, index=bull.index))
    change_down = _crossunder(bull, pd.Series(0.5, index=bull.index))
    sig[change_up   & (rsi >= 40) & (rsi <= 70)] = 1
    sig[change_down & (rsi >= 30) & (rsi <= 60)] = -1
    return sig


def space_Chandelier_RSI(trial):
    return {
        'atr_p': trial.suggest_int('atr_p',   14, 22),
        'mult':  trial.suggest_float('mult',    2.0, 4.0),
        'rsi_p': trial.suggest_int('rsi_p',    7,  21),
    }


def gen_Chandelier_Heikin(df, atr_p=22, mult=3.0, **kw):
    """CE on Heikin Ashi candles — HA OHLC recomputed then CE applied."""
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open_v = ha_close.copy()
    for i in range(1, len(ha_open_v)):
        ha_open_v.iloc[i] = (ha_open_v.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_high = pd.concat([df['high'], ha_open_v, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([df['low'],  ha_open_v, ha_close], axis=1).min(axis=1)

    df_ha = df.copy()
    df_ha['open']  = ha_open_v
    df_ha['high']  = ha_high
    df_ha['low']   = ha_low
    df_ha['close'] = ha_close

    cl, cs, bull = _chandelier(df_ha, atr_p, mult)
    sig = pd.Series(0, index=df.index)
    sig[_crossover(bull, pd.Series(0.5, index=bull.index))]  = 1
    sig[_crossunder(bull, pd.Series(0.5, index=bull.index))] = -1
    return sig


def space_Chandelier_Heikin(trial):
    return {
        'atr_p': trial.suggest_int('atr_p',   14, 22),
        'mult':  trial.suggest_float('mult',    2.0, 4.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CORAL TREND
# ══════════════════════════════════════════════════════════════════════════════

def _coral(close, sm, cd):
    """Coral Trend smoothing — custom exponential smoothing."""
    sm  = int(sm)
    di  = (sm - 1) / 2 + 1
    c1  = 2 / (di + 1)
    c2  = 1 - c1
    # simple iterative EMA-like
    vals = close.values.copy().astype(float)
    out  = np.full(len(vals), np.nan)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = c1 * vals[i] + c2 * out[i-1]
    return pd.Series(out, index=close.index)


def gen_Coral_Trend(df, sm=6, cd=0.4, **kw):
    """Coral Trend — Long: price > coral AND coral rising."""
    coral = _coral(df['close'], sm, cd)
    rising = coral > coral.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] > coral) &  rising]  = 1
    sig[(df['close'] < coral) & ~rising]  = -1
    return sig


def space_Coral_Trend(trial):
    return {
        'sm': trial.suggest_int('sm',     3,  20),
        'cd': trial.suggest_float('cd',   0.3, 0.8),
    }


def gen_Coral_ATR(df, sm=6, atr_p=14, atr_mult=1.5, **kw):
    """Coral + ATR bands. Long: price > coral_upper."""
    coral = _coral(df['close'], sm, 0.4)
    a     = _atr(df, atr_p)
    upper = coral + atr_mult * a
    lower = coral - atr_mult * a

    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], upper)]  = 1
    sig[_crossunder(df['close'], lower)] = -1
    return sig


def space_Coral_ATR(trial):
    return {
        'sm':       trial.suggest_int('sm',       3,  20),
        'atr_p':    trial.suggest_int('atr_p',   10,  20),
        'atr_mult': trial.suggest_float('atr_mult', 1.0, 3.0),
    }


def gen_Coral_EMA_Cross(df, sm_fast=4, sm_slow=14, cd=0.4, **kw):
    """Fast coral vs slow coral cross. Long: fast > slow."""
    fast = _coral(df['close'], sm_fast, cd)
    slow = _coral(df['close'], sm_slow, cd)
    sig = pd.Series(0, index=df.index)
    sig[_crossover(fast, slow)]  = 1
    sig[_crossunder(fast, slow)] = -1
    return sig


def space_Coral_EMA_Cross(trial):
    return {
        'sm_fast': trial.suggest_int('sm_fast',  2,  8),
        'sm_slow': trial.suggest_int('sm_slow',  10, 30),
        'cd':      trial.suggest_float('cd',      0.3, 0.8),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TTM SQUEEZE / LAZYBEAR
# ══════════════════════════════════════════════════════════════════════════════

def _squeeze_momentum(df, bb_p, bb_mult, kc_p, kc_mult):
    """Returns (momentum, in_squeeze) Series."""
    close = df['close']
    bb_p  = int(bb_p)
    kc_p  = int(kc_p)

    # Bollinger Bands
    mid_bb = _sma(close, bb_p)
    std_bb = close.rolling(bb_p, min_periods=1).std().fillna(0)
    bb_hi  = mid_bb + bb_mult * std_bb
    bb_lo  = mid_bb - bb_mult * std_bb

    # Keltner Channels
    mid_kc = _ema(close, kc_p)
    a      = _atr(df, kc_p)
    kc_hi  = mid_kc + kc_mult * a
    kc_lo  = mid_kc - kc_mult * a

    squeeze = (bb_lo > kc_lo) & (bb_hi < kc_hi)

    # Momentum via linear regression of (close - midpoint)
    mid_range = (kc_hi + kc_lo) / 2
    delta     = close - (mid_bb + mid_range) / 2

    # Linear regression slope over bb_p bars
    def linreg_last(x):
        n = len(x)
        if n < 2:
            return x[-1]
        t = np.arange(n)
        m = np.polyfit(t, x, 1)[0]
        return m * (n - 1) + np.polyfit(t, x, 1)[1]

    mom = delta.rolling(bb_p, min_periods=2).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (len(x) - 1) + np.polyfit(np.arange(len(x)), x, 1)[1],
        raw=True
    )
    return mom.fillna(0), squeeze


def gen_TTM_Squeeze(df, bb_p=20, bb_mult=2.0, kc_p=20, kc_mult=1.5, **kw):
    """TTM Squeeze — BB inside KC fires squeeze. Long: fires AND mom > 0 rising."""
    mom, squeeze = _squeeze_momentum(df, bb_p, bb_mult, kc_p, kc_mult)
    fired  = squeeze.shift(1) & ~squeeze     # squeeze released
    rising = mom > mom.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[fired & (mom > 0) & rising]  = 1
    sig[fired & (mom < 0) & ~rising] = -1
    return sig


def space_TTM_Squeeze(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p',      10, 30),
        'bb_mult': trial.suggest_float('bb_mult',  1.5, 2.5),
        'kc_p':    trial.suggest_int('kc_p',      10, 30),
        'kc_mult': trial.suggest_float('kc_mult',  1.0, 2.5),
    }


def gen_Squeeze_Pro(df, bb_p=20, kc_length=20, kc_mult_low=2.0,
                    kc_mult_mid=1.5, kc_mult_hi=1.0, **kw):
    """Squeeze Pro — 3 squeeze levels. Long: any fires AND mom > 0."""
    mom_lo, sq_lo = _squeeze_momentum(df, bb_p, 2.0, kc_length, kc_mult_low)
    _,       sq_mi = _squeeze_momentum(df, bb_p, 2.0, kc_length, kc_mult_mid)
    _,       sq_hi = _squeeze_momentum(df, bb_p, 2.0, kc_length, kc_mult_hi)
    mom = mom_lo

    any_fire = (sq_lo.shift(1) & ~sq_lo) | (sq_mi.shift(1) & ~sq_mi) | (sq_hi.shift(1) & ~sq_hi)
    sig = pd.Series(0, index=df.index)
    sig[any_fire & (mom > 0)]  = 1
    sig[any_fire & (mom < 0)]  = -1
    return sig


def space_Squeeze_Pro(trial):
    return {
        'bb_p':        trial.suggest_int('bb_p',         10, 30),
        'kc_length':   trial.suggest_int('kc_length',    10, 30),
        'kc_mult_low': trial.suggest_float('kc_mult_low', 1.5, 2.5),
        'kc_mult_mid': trial.suggest_float('kc_mult_mid', 1.0, 2.0),
        'kc_mult_hi':  trial.suggest_float('kc_mult_hi',  0.5, 1.5),
    }


def gen_Lazybear_Squeeze(df, length=12, **kw):
    """LazyBear Squeeze — 12-period LR momentum. Long: not squeezed AND mom > 0 rising."""
    length = int(length)
    close  = df['close']
    bb_p   = length

    mid_bb = _sma(close, bb_p)
    std_bb = close.rolling(bb_p, min_periods=1).std().fillna(0)
    bb_hi  = mid_bb + 2.0 * std_bb
    bb_lo  = mid_bb - 2.0 * std_bb

    mid_kc = _ema(close, bb_p)
    a      = _atr(df, bb_p)
    kc_hi  = mid_kc + 1.5 * a
    kc_lo  = mid_kc - 1.5 * a

    squeeze = (bb_lo > kc_lo) & (bb_hi < kc_hi)

    # Linear regression momentum
    highest = df['high'].rolling(bb_p, min_periods=1).max()
    lowest  = df['low'].rolling(bb_p, min_periods=1).min()
    mid_val = (highest + lowest) / 2
    val     = close - (mid_val + mid_bb) / 2

    mom = val.rolling(length, min_periods=2).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (len(x) - 1) + np.polyfit(np.arange(len(x)), x, 1)[1],
        raw=True
    ).fillna(0)

    rising = mom > mom.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[~squeeze & (mom > 0) & rising]  = 1
    sig[~squeeze & (mom < 0) & ~rising] = -1
    return sig


def space_Lazybear_Squeeze(trial):
    return {
        'length': trial.suggest_int('length', 5, 25),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SSL CHANNEL FAMILY
# ══════════════════════════════════════════════════════════════════════════════

def _ssl(df, ssl_p):
    """Returns (ssl_up, ssl_down, bull) — SSL channel lines."""
    ssl_p   = int(ssl_p)
    close   = df['close']
    hi_sma  = _sma(df['high'], ssl_p)
    lo_sma  = _sma(df['low'],  ssl_p)
    hlv     = pd.Series(np.nan, index=close.index)

    # hlv: 1 if close > hi_sma, -1 if close < lo_sma, else previous
    mask_up   = close > hi_sma
    mask_down = close < lo_sma
    hlv_v = np.zeros(len(close))
    for i in range(len(hlv_v)):
        if mask_up.iloc[i]:
            hlv_v[i] = 1
        elif mask_down.iloc[i]:
            hlv_v[i] = -1
        else:
            hlv_v[i] = hlv_v[i-1] if i > 0 else 1

    ssl_up   = np.where(hlv_v < 0, hi_sma.values, lo_sma.values)
    ssl_down = np.where(hlv_v < 0, lo_sma.values, hi_sma.values)
    bull     = (hlv_v == 1)
    return (pd.Series(ssl_up,   index=close.index),
            pd.Series(ssl_down, index=close.index),
            pd.Series(bull,     index=close.index))


def gen_SSL_Channel_ATR(df, ssl_p=10, atr_p=14, atr_mult=2.0, **kw):
    """SSL + ATR bands. Long: SSL bull AND close > ATR upper."""
    sup, sdn, bull = _ssl(df, ssl_p)
    a      = _atr(df, atr_p)
    atr_up = sup + atr_mult * a

    sig = pd.Series(0, index=df.index)
    sig[bull  & (df['close'] > atr_up)] = 1
    sig[~bull & (df['close'] < sdn - atr_mult * a)] = -1
    return sig


def space_SSL_Channel_ATR(trial):
    return {
        'ssl_p':    trial.suggest_int('ssl_p',     7,  21),
        'atr_p':    trial.suggest_int('atr_p',    10,  20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 3.0),
    }


def gen_SSL_EMA_Cross(df, ssl_p=10, ema_fast=9, ema_slow=21, **kw):
    """SSL trend filter + EMA cross entry. Long: SSL bull AND fast EMA > slow EMA."""
    sup, sdn, bull = _ssl(df, ssl_p)
    fast = _ema(df['close'], ema_fast)
    slow = _ema(df['close'], ema_slow)

    sig = pd.Series(0, index=df.index)
    sig[bull  & _crossover(fast, slow)]  = 1
    sig[~bull & _crossunder(fast, slow)] = -1
    return sig


def space_SSL_EMA_Cross(trial):
    return {
        'ssl_p':    trial.suggest_int('ssl_p',    7,  21),
        'ema_fast': trial.suggest_int('ema_fast',  5,  15),
        'ema_slow': trial.suggest_int('ema_slow', 20,  50),
    }


def gen_SSL_QQE(df, ssl_p=10, rsi_p=6, sf=5, **kw):
    """SSL + QQE. Long: SSL bull AND QQE RSI_MA > 50."""
    sup, sdn, bull = _ssl(df, ssl_p)
    rsi_ma, ql, qs = _qqe_lines(df['close'], rsi_p, sf, 4.236)

    sig = pd.Series(0, index=df.index)
    sig[bull  & (rsi_ma > 50)] = 1
    sig[~bull & (rsi_ma < 50)] = -1
    return sig


def space_SSL_QQE(trial):
    return {
        'ssl_p': trial.suggest_int('ssl_p',  7, 21),
        'rsi_p': trial.suggest_int('rsi_p',  6, 14),
        'sf':    trial.suggest_int('sf',      3,  8),
    }


def gen_SSL_Stoch(df, ssl_p=10, stoch_k=14, stoch_d=3, **kw):
    """SSL + Stoch. Long: SSL turns bull AND Stoch K < 25."""
    sup, sdn, bull = _ssl(df, ssl_p)
    close  = df['close']
    lo     = close.rolling(int(stoch_k), min_periods=1).min()
    hi     = close.rolling(int(stoch_k), min_periods=1).max()
    K      = _sma((close - lo) / (hi - lo + 1e-9) * 100, stoch_d)

    ssl_turn_bull = _crossover(bull.astype(float), pd.Series(0.5, index=df.index))
    ssl_turn_bear = _crossunder(bull.astype(float), pd.Series(0.5, index=df.index))

    sig = pd.Series(0, index=df.index)
    sig[ssl_turn_bull & (K < 25)] = 1
    sig[ssl_turn_bear & (K > 75)] = -1
    return sig


def space_SSL_Stoch(trial):
    return {
        'ssl_p':   trial.suggest_int('ssl_p',   7, 21),
        'stoch_k': trial.suggest_int('stoch_k', 7, 21),
        'stoch_d': trial.suggest_int('stoch_d', 3,  7),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TREND MAGIC
# ══════════════════════════════════════════════════════════════════════════════

def _cci(df, p):
    tp   = (df['high'] + df['low'] + df['close']) / 3
    mid  = _sma(tp, p)
    mad  = tp.rolling(int(p), min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).fillna(1e-9)
    return (tp - mid) / (0.015 * mad)


def _trend_magic_line(df, cci_p, atr_p, coeff):
    """CCI-based trend magic — returns trend line Series."""
    cci_p = int(cci_p)
    atr_p = int(atr_p)
    cci   = _cci(df, cci_p)
    a     = _atr(df, atr_p) * coeff
    close = df['close'].values
    cci_v = cci.values
    a_v   = a.values
    n     = len(close)
    tm    = np.full(n, np.nan)
    tm[0] = close[0]

    for i in range(1, n):
        if cci_v[i] > 0:
            tm[i] = max(tm[i-1], close[i] - a_v[i])
        else:
            tm[i] = min(tm[i-1], close[i] + a_v[i])

    return pd.Series(tm, index=df.index)


def gen_Trend_Magic(df, cci_p=20, atr_p=5, coeff=0.8, **kw):
    """Trend Magic — CCI + ATR buffer line. Long: close > TM line."""
    tm  = _trend_magic_line(df, cci_p, atr_p, coeff)
    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], tm)]  = 1
    sig[_crossunder(df['close'], tm)] = -1
    return sig


def space_Trend_Magic(trial):
    return {
        'cci_p': trial.suggest_int('cci_p',   14, 30),
        'atr_p': trial.suggest_int('atr_p',    1,  5),
        'coeff': trial.suggest_float('coeff',  0.5, 2.0),
    }


def gen_Trend_Magic_RSI(df, cci_p=20, atr_p=5, rsi_p=14, **kw):
    """Trend Magic + RSI filter. Long: TM bullish AND RSI > 50."""
    tm  = _trend_magic_line(df, cci_p, atr_p, 0.8)
    rsi = _rsi(df['close'], rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], tm)  & (rsi > 50)] = 1
    sig[_crossunder(df['close'], tm) & (rsi < 50)] = -1
    return sig


def space_Trend_Magic_RSI(trial):
    return {
        'cci_p': trial.suggest_int('cci_p',  14, 30),
        'atr_p': trial.suggest_int('atr_p',   1,  5),
        'rsi_p': trial.suggest_int('rsi_p',   7, 21),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONNORS RSI
# ══════════════════════════════════════════════════════════════════════════════

def _connors_rsi(close, rsi_p, streak_p, rank_p):
    """Returns Connors RSI = avg(RSI_p, StreakRSI_2, PercentRank)."""
    rsi_p    = int(rsi_p)
    streak_p = int(streak_p)
    rank_p   = int(rank_p)

    rsi_comp = _rsi(close, rsi_p)

    # Streak: consecutive up/down days
    vals    = close.values
    streak  = np.zeros(len(vals))
    for i in range(1, len(vals)):
        if vals[i] > vals[i-1]:
            streak[i] = max(1, streak[i-1] + 1)
        elif vals[i] < vals[i-1]:
            streak[i] = min(-1, streak[i-1] - 1)
        else:
            streak[i] = 0
    streak_s    = pd.Series(streak, index=close.index)
    streak_rsi  = _rsi(streak_s, streak_p)

    # PercentRank: % of last rank_p closes strictly below current
    pct_rank = close.rolling(rank_p, min_periods=2).apply(
        lambda x: np.sum(x[:-1] < x[-1]) / (len(x) - 1) * 100,
        raw=True
    ).fillna(50)

    return (rsi_comp + streak_rsi + pct_rank) / 3


def gen_Connors_RSI(df, rsi_p=3, streak_p=2, rank_p=100, **kw):
    """Connors RSI mean-reversion. Long: CRSI < 10. Short: CRSI > 90."""
    crsi = _connors_rsi(df['close'], rsi_p, streak_p, rank_p)
    sig  = pd.Series(0, index=df.index)
    sig[_crossover(crsi, pd.Series(10, index=crsi.index))]   = 1   # exits oversold
    sig[_crossunder(crsi, pd.Series(90, index=crsi.index))]  = -1  # exits overbought
    return sig


def space_Connors_RSI(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p',    2,   5),
        'streak_p': trial.suggest_int('streak_p', 2,   4),
        'rank_p':   trial.suggest_int('rank_p',  80, 130),
    }


def gen_Connors_RSI_EMA(df, rsi_p=3, rank_p=100, ema_p=100, **kw):
    """Connors RSI + EMA trend filter. Long: CRSI oversold AND close > EMA."""
    crsi = _connors_rsi(df['close'], rsi_p, 2, rank_p)
    ema  = _ema(df['close'], ema_p)
    sig  = pd.Series(0, index=df.index)
    sig[_crossover(crsi, pd.Series(10, index=crsi.index)) & (df['close'] > ema)]  = 1
    sig[_crossunder(crsi, pd.Series(90, index=crsi.index)) & (df['close'] < ema)] = -1
    return sig


def space_Connors_RSI_EMA(trial):
    return {
        'rsi_p':  trial.suggest_int('rsi_p',   2,   5),
        'rank_p': trial.suggest_int('rank_p',  80, 130),
        'ema_p':  trial.suggest_int('ema_p',   50, 200),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MCGINLEY DYNAMIC
# ══════════════════════════════════════════════════════════════════════════════

def _mcginley(close, length, k):
    """McGinley Dynamic MA — iterative update."""
    length = int(length)
    vals   = close.values.astype(float)
    md     = np.full(len(vals), np.nan)
    md[0]  = vals[0]
    for i in range(1, len(vals)):
        ratio = vals[i] / md[i-1] if md[i-1] != 0 else 1.0
        md[i] = md[i-1] + (vals[i] - md[i-1]) / (k * length * ratio ** 4)
    return pd.Series(md, index=close.index)


def gen_McGinley_Dynamic(df, length=14, k=0.6, **kw):
    """McGinley Dynamic — Long: price > MD AND MD rising."""
    md      = _mcginley(df['close'], length, k)
    rising  = md > md.shift(1)
    sig     = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], md)  & rising]  = 1
    sig[_crossunder(df['close'], md) & ~rising] = -1
    return sig


def space_McGinley_Dynamic(trial):
    return {
        'length': trial.suggest_int('length',   5, 30),
        'k':      trial.suggest_float('k',      0.3, 0.7),
    }


def gen_McGinley_Cross(df, fast_p=9, slow_p=26, k=0.6, **kw):
    """Fast vs slow McGinley cross. Long: fast MD > slow MD."""
    fast = _mcginley(df['close'], fast_p, k)
    slow = _mcginley(df['close'], slow_p, k)
    sig  = pd.Series(0, index=df.index)
    sig[_crossover(fast, slow)]  = 1
    sig[_crossunder(fast, slow)] = -1
    return sig


def space_McGinley_Cross(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',  5,  15),
        'slow_p': trial.suggest_int('slow_p', 20,  60),
        'k':      trial.suggest_float('k',     0.3, 0.7),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRUE STRENGTH INDEX
# ══════════════════════════════════════════════════════════════════════════════

def _tsi(close, long_p, short_p):
    """TSI = double-smoothed momentum ratio × 100."""
    long_p  = int(long_p)
    short_p = int(short_p)
    mom     = close.diff(1)
    abs_mom = mom.abs()

    num = _ema(_ema(mom,     short_p), long_p)
    den = _ema(_ema(abs_mom, short_p), long_p)
    return (num / den.replace(0, 1e-9)) * 100


def gen_TSI_Strategy(df, long_p=25, short_p=13, sig_p=13, **kw):
    """TSI — Long: TSI crosses above signal above 0."""
    close = df['close']
    tsi   = _tsi(close, long_p, short_p)
    sig_l = _ema(tsi, sig_p)

    sig = pd.Series(0, index=df.index)
    cross_up   = _crossover(tsi, sig_l)
    cross_down = _crossunder(tsi, sig_l)
    sig[cross_up   & (tsi > 0)]  = 1
    sig[cross_down & (tsi < 0)]  = -1
    return sig


def space_TSI_Strategy(trial):
    return {
        'long_p':  trial.suggest_int('long_p',  13, 30),
        'short_p': trial.suggest_int('short_p',  3, 10),
        'sig_p':   trial.suggest_int('sig_p',    5, 15),
    }


def gen_TSI_RSI(df, long_p=25, short_p=13, rsi_p=14, **kw):
    """TSI + RSI filter. Long: TSI bullish cross AND RSI > 50."""
    close = df['close']
    tsi   = _tsi(close, long_p, short_p)
    sig_l = _ema(tsi, 7)
    rsi   = _rsi(close, rsi_p)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(tsi, sig_l)  & (rsi > 50)] = 1
    sig[_crossunder(tsi, sig_l) & (rsi < 50)] = -1
    return sig


def space_TSI_RSI(trial):
    return {
        'long_p':  trial.suggest_int('long_p',  13, 30),
        'short_p': trial.suggest_int('short_p',  3, 10),
        'rsi_p':   trial.suggest_int('rsi_p',    7, 21),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCHAFF TREND CYCLE
# ══════════════════════════════════════════════════════════════════════════════

def _stc(close, stc_p, fast, slow, factor):
    """Schaff Trend Cycle — stochastic of MACD."""
    stc_p  = int(stc_p)
    fast   = int(fast)
    slow   = int(slow)

    macd   = _ema(close, fast) - _ema(close, slow)

    def _stoch_fast(s, p):
        lo = s.rolling(p, min_periods=1).min()
        hi = s.rolling(p, min_periods=1).max()
        return (s - lo) / (hi - lo + 1e-9) * 100

    k1 = _stoch_fast(macd, stc_p)
    # Smooth k1 with factor (Wilder's)
    d1_v = np.full(len(k1), np.nan)
    d1_v[0] = k1.iloc[0]
    for i in range(1, len(d1_v)):
        d1_v[i] = d1_v[i-1] + factor * (k1.iloc[i] - d1_v[i-1])
    d1 = pd.Series(d1_v, index=close.index)

    k2 = _stoch_fast(d1, stc_p)
    d2_v = np.full(len(k2), np.nan)
    d2_v[0] = k2.iloc[0]
    for i in range(1, len(d2_v)):
        d2_v[i] = d2_v[i-1] + factor * (k2.iloc[i] - d2_v[i-1])
    stc_line = pd.Series(d2_v, index=close.index)
    return stc_line.clip(0, 100)


def gen_STC_Strategy(df, stc_p=12, fast=23, slow=50, factor=0.5, **kw):
    """STC — Long: crosses above 25 from below. Short: crosses below 75."""
    stc = _stc(df['close'], stc_p, fast, slow, factor)
    lo  = pd.Series(25, index=stc.index)
    hi  = pd.Series(75, index=stc.index)
    sig = pd.Series(0, index=df.index)
    sig[_crossover(stc, lo)]  = 1
    sig[_crossunder(stc, hi)] = -1
    return sig


def space_STC_Strategy(trial):
    return {
        'stc_p':  trial.suggest_int('stc_p',   10, 20),
        'fast':   trial.suggest_int('fast',     12, 26),
        'slow':   trial.suggest_int('slow',     26, 52),
        'factor': trial.suggest_float('factor', 0.3, 0.7),
    }


def gen_STC_EMA(df, stc_p=12, fast=23, slow=50, ema_p=50, **kw):
    """STC + EMA. Long: STC > 75 AND close > EMA."""
    stc  = _stc(df['close'], stc_p, fast, slow, 0.5)
    ema  = _ema(df['close'], ema_p)
    sig  = pd.Series(0, index=df.index)
    sig[_crossover(stc, pd.Series(75, index=stc.index)) & (df['close'] > ema)]  = 1
    sig[_crossunder(stc, pd.Series(25, index=stc.index)) & (df['close'] < ema)] = -1
    return sig


def space_STC_EMA(trial):
    return {
        'stc_p': trial.suggest_int('stc_p',  10, 20),
        'fast':  trial.suggest_int('fast',    12, 26),
        'slow':  trial.suggest_int('slow',    26, 52),
        'ema_p': trial.suggest_int('ema_p',   20, 60),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY EXPORT (pickle-safe — direct refs only, NO lambdas)
# ══════════════════════════════════════════════════════════════════════════════

STRATEGY_EXPORT = {
    # QQE Family
    'QQE_Mod': {
        'gen':            gen_QQE_Mod,
        'space':          space_QQE_Mod,
        'default_params': {'rsi_p': 6, 'sf': 5, 'qq_factor': 4.236},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'QQE Mod — RSI double-smoothed with SQRT trailing. '
                                'Long/Short on RSI_MA cross of 50.'},
    },
    'QQE_Classic': {
        'gen':            gen_QQE_Classic,
        'space':          space_QQE_Classic,
        'default_params': {'rsi_p': 9, 'sf': 5, 'factor': 3.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'Classic QQE — fast vs slow QQE trailing line cross.'},
    },
    'QQE_Histo': {
        'gen':            gen_QQE_Histo,
        'space':          space_QQE_Histo,
        'default_params': {'rsi_p': 6, 'sf': 5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'QQE Histogram — QQE histo sign changes drive signals.'},
    },
    'QQE_ATR_BB': {
        'gen':            gen_QQE_ATR_BB,
        'space':          space_QQE_ATR_BB,
        'default_params': {'rsi_p': 6, 'sf': 5, 'bb_mult': 1.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1200,
                 'description': 'QQE + Bollinger Band on RSI_EMA. Cross of BB lower.'},
    },
    'QQE_RSI_Stoch': {
        'gen':            gen_QQE_RSI_Stoch,
        'space':          space_QQE_RSI_Stoch,
        'default_params': {'rsi_p': 6, 'stoch_k': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 900,
                 'description': 'QQE + Stoch RSI confirmation. K cross D from <30.'},
    },
    # Chandelier Exit Family
    'Chandelier_Exit': {
        'gen':            gen_Chandelier_Exit,
        'space':          space_Chandelier_Exit,
        'default_params': {'atr_p': 22, 'mult': 3.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 8000,
                 'description': 'Chandelier Exit — ATR trailing stop from HH/LL. Direction changes.'},
    },
    'Chandelier_ATR_EMA': {
        'gen':            gen_Chandelier_ATR_EMA,
        'space':          space_Chandelier_ATR_EMA,
        'default_params': {'atr_p': 22, 'mult': 3.0, 'ema_p': 50},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'Chandelier + EMA trend filter. Long: CE bull AND close > EMA.'},
    },
    'Chandelier_3x': {
        'gen':            gen_Chandelier_3x,
        'space':          space_Chandelier_3x,
        'default_params': {'fast_mult': 2.0, 'mid_mult': 3.0, 'slow_mult': 4.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1800,
                 'description': 'Three Chandelier exits consensus. All 3 must agree.'},
    },
    'Chandelier_RSI': {
        'gen':            gen_Chandelier_RSI,
        'space':          space_Chandelier_RSI,
        'default_params': {'atr_p': 22, 'mult': 3.0, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1200,
                 'description': 'Chandelier + RSI filter. Long: CE bull AND RSI 40-70.'},
    },
    'Chandelier_Heikin': {
        'gen':            gen_Chandelier_Heikin,
        'space':          space_Chandelier_Heikin,
        'default_params': {'atr_p': 22, 'mult': 3.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 800,
                 'description': 'Chandelier Exit applied to Heikin Ashi candles.'},
    },
    # Coral Trend
    'Coral_Trend': {
        'gen':            gen_Coral_Trend,
        'space':          space_Coral_Trend,
        'default_params': {'sm': 6, 'cd': 0.4},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 5000,
                 'description': 'Coral Trend — custom exponential smoothing. Price above rising coral.'},
    },
    'Coral_ATR': {
        'gen':            gen_Coral_ATR,
        'space':          space_Coral_ATR,
        'default_params': {'sm': 6, 'atr_p': 14, 'atr_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Coral + ATR bands. Long: close crosses above coral + ATR.'},
    },
    'Coral_EMA_Cross': {
        'gen':            gen_Coral_EMA_Cross,
        'space':          space_Coral_EMA_Cross,
        'default_params': {'sm_fast': 4, 'sm_slow': 14, 'cd': 0.4},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1500,
                 'description': 'Fast coral vs slow coral cross. Standard MA-cross logic.'},
    },
    # TTM Squeeze
    'TTM_Squeeze': {
        'gen':            gen_TTM_Squeeze,
        'space':          space_TTM_Squeeze,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'kc_p': 20, 'kc_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 6000,
                 'description': 'TTM Squeeze — BB inside KC. Fires on release with momentum direction.'},
    },
    'Squeeze_Pro': {
        'gen':            gen_Squeeze_Pro,
        'space':          space_Squeeze_Pro,
        'default_params': {'bb_p': 20, 'kc_length': 20, 'kc_mult_low': 2.0,
                           'kc_mult_mid': 1.5, 'kc_mult_hi': 1.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Squeeze Pro — 3 squeeze levels (low/mid/high). Any level fires.'},
    },
    'Lazybear_Squeeze': {
        'gen':            gen_Lazybear_Squeeze,
        'space':          space_Lazybear_Squeeze,
        'default_params': {'length': 12},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 4000,
                 'description': "LazyBear's classic squeeze with 12-period LR momentum."},
    },
    # SSL Channel Family
    'SSL_Channel_ATR': {
        'gen':            gen_SSL_Channel_ATR,
        'space':          space_SSL_Channel_ATR,
        'default_params': {'ssl_p': 10, 'atr_p': 14, 'atr_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'SSL + ATR bands. Long: SSL bull AND close above ATR upper.'},
    },
    'SSL_EMA_Cross': {
        'gen':            gen_SSL_EMA_Cross,
        'space':          space_SSL_EMA_Cross,
        'default_params': {'ssl_p': 10, 'ema_fast': 9, 'ema_slow': 21},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'SSL trend filter + EMA cross entry.'},
    },
    'SSL_QQE': {
        'gen':            gen_SSL_QQE,
        'space':          space_SSL_QQE,
        'default_params': {'ssl_p': 10, 'rsi_p': 6, 'sf': 5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'SSL direction filter + QQE RSI_MA > 50 confirmation.'},
    },
    'SSL_Stoch': {
        'gen':            gen_SSL_Stoch,
        'space':          space_SSL_Stoch,
        'default_params': {'ssl_p': 10, 'stoch_k': 14, 'stoch_d': 3},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1000,
                 'description': 'SSL turn + Stoch oversold/overbought confirmation.'},
    },
    # Trend Magic
    'Trend_Magic': {
        'gen':            gen_Trend_Magic,
        'space':          space_Trend_Magic,
        'default_params': {'cci_p': 20, 'atr_p': 5, 'coeff': 0.8},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3500,
                 'description': 'Trend Magic — CCI-based ATR buffer line. Close crosses TM.'},
    },
    'Trend_Magic_RSI': {
        'gen':            gen_Trend_Magic_RSI,
        'space':          space_Trend_Magic_RSI,
        'default_params': {'cci_p': 20, 'atr_p': 5, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'Trend Magic + RSI > 50 filter for trend confirmation.'},
    },
    # Connors RSI
    'Connors_RSI': {
        'gen':            gen_Connors_RSI,
        'space':          space_Connors_RSI,
        'default_params': {'rsi_p': 3, 'streak_p': 2, 'rank_p': 100},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'Connors RSI = avg(RSI_3, StreakRSI_2, PercentRank). Mean reversion.'},
    },
    'Connors_RSI_EMA': {
        'gen':            gen_Connors_RSI_EMA,
        'space':          space_Connors_RSI_EMA,
        'default_params': {'rsi_p': 3, 'rank_p': 100, 'ema_p': 100},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1200,
                 'description': 'Connors RSI + EMA trend filter. Only trade with trend.'},
    },
    # McGinley Dynamic
    'McGinley_Dynamic': {
        'gen':            gen_McGinley_Dynamic,
        'space':          space_McGinley_Dynamic,
        'default_params': {'length': 14, 'k': 0.6},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'McGinley Dynamic MA — smoother than EMA. Price crosses MD.'},
    },
    'McGinley_Cross': {
        'gen':            gen_McGinley_Cross,
        'space':          space_McGinley_Cross,
        'default_params': {'fast_p': 9, 'slow_p': 26, 'k': 0.6},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'Fast vs slow McGinley Dynamic cross.'},
    },
    # TSI
    'TSI_Strategy': {
        'gen':            gen_TSI_Strategy,
        'space':          space_TSI_Strategy,
        'default_params': {'long_p': 25, 'short_p': 13, 'sig_p': 13},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1800,
                 'description': 'True Strength Index — double EMA smoothed momentum. Cross above 0.'},
    },
    'TSI_RSI': {
        'gen':            gen_TSI_RSI,
        'space':          space_TSI_RSI,
        'default_params': {'long_p': 25, 'short_p': 13, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1000,
                 'description': 'TSI bullish cross + RSI > 50 trend confirmation.'},
    },
    # STC
    'STC_Strategy': {
        'gen':            gen_STC_Strategy,
        'space':          space_STC_Strategy,
        'default_params': {'stc_p': 12, 'fast': 23, 'slow': 50, 'factor': 0.5},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'Schaff Trend Cycle — stochastic of MACD. Cross of 25/75 levels.'},
    },
    'STC_EMA': {
        'gen':            gen_STC_EMA,
        'space':          space_STC_EMA,
        'default_params': {'stc_p': 12, 'fast': 23, 'slow': 50, 'ema_p': 50},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'STC > 75 AND close > EMA. Trend + momentum combo.'},
    },
}
