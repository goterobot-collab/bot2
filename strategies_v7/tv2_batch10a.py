#!/usr/bin/env python3
"""
TV2 BATCH 10a — 9 estrategias SuperTrend 2026-03-31

  SuperTrend_Classic    — Classic SuperTrend ATR (v4, 23328L)
  Pivot_SuperTrend      — Pivot Point SuperTrend backtest (v4, 5219L)
  AI_SuperTrend_Pivot   — AI SuperTrend x Pivot Percentile (v5, 4702L)
  SuperTrend_ATR_TSL    — SuperTrend + ATR Trailing SL (v4, 4335L)
  HA_SuperTrend         — Heikin Ashi SuperTrend (v5, 3180L)
  SuperTrend_CCI        — Best SuperTrend CCI (v4, 2818L)
  AI_Volume_SuperTrend  — KNN Volume SuperTrend (v5, 2796L)
  Triple_SuperTrend     — 3x SuperTrend + StochRSI (v4, 2317L)
  ATR_GOD_4ST           — 4 SuperTrend consensus (v5, 2093L)
"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── helpers ──────────────────────────────────────────────────────────────────

def _rma(series, period):
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _atr(df, period):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, period)


def _supertrend_calc(df, period, mult):
    """Returns (trend Series 1/-1, final_lower, final_upper)."""
    hl2 = (df['high'] + df['low']) / 2
    atr  = _atr(df, period)
    ub   = (hl2 + mult * atr).values
    lb   = (hl2 - mult * atr).values
    cl   = df['close'].values
    n    = len(cl)

    fu   = ub.copy()
    fl   = lb.copy()
    tr   = np.ones(n)

    for i in range(1, n):
        fu[i] = ub[i] if (ub[i] < fu[i-1] or cl[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = lb[i] if (lb[i] > fl[i-1] or cl[i-1] < fl[i-1]) else fl[i-1]
        if   tr[i-1] == -1 and cl[i] > fu[i]: tr[i] =  1
        elif tr[i-1] ==  1 and cl[i] < fl[i]: tr[i] = -1
        else:                                   tr[i] =  tr[i-1]

    idx = df.index
    return pd.Series(tr, index=idx), pd.Series(fl, index=idx), pd.Series(fu, index=idx)


def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


# ── 1. SuperTrend Classic ────────────────────────────────────────────────────

def gen_SuperTrend_Classic(df, atr_period=10, mult=3.0, **kw):
    trend, _, _ = _supertrend_calc(df, int(atr_period), mult)
    sig = pd.Series(0, index=df.index)
    sig[trend ==  1] =  1
    sig[trend == -1] = -1
    return sig


def space_SuperTrend_Classic(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 20),
        'mult':       trial.suggest_float('mult', 1.5, 5.0),
    }


# ── 2. Pivot SuperTrend ──────────────────────────────────────────────────────

def gen_Pivot_SuperTrend(df, atr_factor=3.0, atr_period=14, **kw):
    """Crossover entries only (long on bull crossover, short on bear crossover)."""
    trend, _, _ = _supertrend_calc(df, int(atr_period), atr_factor)
    prev = trend.shift(1).fillna(1)
    sig = pd.Series(0, index=df.index)
    sig[(trend ==  1) & (prev == -1)] =  1
    sig[(trend == -1) & (prev ==  1)] = -1
    return sig


def space_Pivot_SuperTrend(trial):
    return {
        'atr_factor': trial.suggest_float('atr_factor', 1.0, 5.0),
        'atr_period': trial.suggest_int('atr_period', 7, 21),
    }


# ── 3. AI SuperTrend × Pivot Percentile ─────────────────────────────────────

def gen_AI_SuperTrend_Pivot(df, atr_period=12, mult=3.0, pct_period=200, **kw):
    trend, _, _ = _supertrend_calc(df, int(atr_period), mult)
    pp = int(pct_period)
    q75 = df['close'].rolling(pp).quantile(0.75)
    q25 = df['close'].rolling(pp).quantile(0.25)
    mid = (q75 + q25) / 2
    sig = pd.Series(0, index=df.index)
    sig[(trend ==  1) & (df['close'] > mid)] =  1
    sig[(trend == -1) & (df['close'] < mid)] = -1
    return sig


def space_AI_SuperTrend_Pivot(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 20),
        'mult':       trial.suggest_float('mult', 1.5, 5.0),
        'pct_period': trial.suggest_int('pct_period', 50, 300),
    }


# ── 4. SuperTrend + ATR Trailing SL ─────────────────────────────────────────

def gen_SuperTrend_ATR_TSL(df, atr_period=10, mult=3.0, tsl_period=5, tsl_mult=0.5, **kw):
    trend, fl, fu = _supertrend_calc(df, int(atr_period), mult)
    atr_fast = _atr(df, int(tsl_period))
    trail_long  = df['close'] - tsl_mult * atr_fast
    trail_short = df['close'] + tsl_mult * atr_fast
    sig = pd.Series(0, index=df.index)
    sig[(trend ==  1) & (df['close'] > trail_long.shift(1))]  =  1
    sig[(trend == -1) & (df['close'] < trail_short.shift(1))] = -1
    return sig


def space_SuperTrend_ATR_TSL(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 20),
        'mult':       trial.suggest_float('mult', 1.5, 5.0),
        'tsl_period': trial.suggest_int('tsl_period', 3, 14),
        'tsl_mult':   trial.suggest_float('tsl_mult', 0.3, 2.0),
    }


# ── 5. Heikin Ashi SuperTrend ────────────────────────────────────────────────

def _ha_df(df):
    ha_c = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    hc_v = ha_c.values
    o_v  = df['open'].values
    c_v  = df['close'].values
    n    = len(df)
    ha_o = np.empty(n)
    ha_o[0] = (o_v[0] + c_v[0]) / 2
    for i in range(1, n):
        ha_o[i] = (ha_o[i-1] + hc_v[i-1]) / 2
    ha_o_s = pd.Series(ha_o, index=df.index)
    ha_h   = pd.concat([df['high'], ha_o_s, ha_c], axis=1).max(axis=1)
    ha_l   = pd.concat([df['low'],  ha_o_s, ha_c], axis=1).min(axis=1)
    return pd.DataFrame({'open': ha_o_s, 'high': ha_h, 'low': ha_l, 'close': ha_c}, index=df.index)


def gen_HA_SuperTrend(df, atr_period=10, mult=3.0, **kw):
    ha = _ha_df(df)
    trend, _, _ = _supertrend_calc(ha, int(atr_period), mult)
    prev = trend.shift(1).fillna(1)
    sig = pd.Series(0, index=df.index)
    sig[(trend ==  1) & (prev == -1)] =  1
    sig[(trend == -1) & (prev ==  1)] = -1
    return sig


def space_HA_SuperTrend(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 20),
        'mult':       trial.suggest_float('mult', 1.5, 5.0),
    }


# ── 6. Best SuperTrend CCI ───────────────────────────────────────────────────

def _cci(df, period):
    tp  = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def gen_SuperTrend_CCI(df, atr_period=10, st_mult=3.0, cci_period=28, **kw):
    trend, _, _ = _supertrend_calc(df, int(atr_period), st_mult)
    cci = _cci(df, int(cci_period))
    sig = pd.Series(0, index=df.index)
    sig[(trend ==  1) & (cci > 0)]  =  1
    sig[(trend == -1) & (cci < 0)]  = -1
    return sig


def space_SuperTrend_CCI(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 20),
        'st_mult':    trial.suggest_float('st_mult', 1.5, 5.0),
        'cci_period': trial.suggest_int('cci_period', 14, 50),
    }


# ── 7. AI Volume SuperTrend (KNN-inspired) ───────────────────────────────────

def gen_AI_Volume_SuperTrend(df, atr_period=10, mult=3.0, vol_period=20, k=5, **kw):
    trend, _, _ = _supertrend_calc(df, int(atr_period), mult)
    vol_ma   = df['volume'].rolling(int(vol_period)).mean().replace(0, np.nan)
    vol_ratio = df['volume'] / vol_ma
    score = (trend * vol_ratio).rolling(int(k)).mean()
    sig = pd.Series(0, index=df.index)
    sig[score >  0.5] =  1
    sig[score < -0.5] = -1
    return sig


def space_AI_Volume_SuperTrend(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 20),
        'mult':       trial.suggest_float('mult', 1.5, 5.0),
        'vol_period': trial.suggest_int('vol_period', 10, 50),
        'k':          trial.suggest_int('k', 3, 15),
    }


# ── 8. 3x SuperTrend + StochRSI ─────────────────────────────────────────────

def _rsi(close, period):
    d     = close.diff()
    gain  = d.clip(lower=0).rolling(period).mean()
    loss  = (-d.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _stochrsi(close, rsi_p=14, stoch_p=14):
    rsi  = _rsi(close, rsi_p)
    rmin = rsi.rolling(stoch_p).min()
    rmax = rsi.rolling(stoch_p).max()
    return (rsi - rmin) / (rmax - rmin).replace(0, np.nan)


def gen_Triple_SuperTrend(df, slow_p=14, slow_m=4.0, mid_p=11, mid_m=3.0, fast_p=8, fast_m=2.0, stoch_p=14, **kw):
    t_slow, _, _ = _supertrend_calc(df, int(slow_p), slow_m)
    t_mid,  _, _ = _supertrend_calc(df, int(mid_p),  mid_m)
    t_fast, _, _ = _supertrend_calc(df, int(fast_p), fast_m)
    srsi = _stochrsi(df['close'], int(stoch_p), int(stoch_p))
    bull = (t_slow == 1) & (t_mid == 1) & (t_fast == 1)
    bear = (t_slow ==-1) & (t_mid ==-1) & (t_fast ==-1)
    sig  = pd.Series(0, index=df.index)
    sig[bull & (srsi < 0.8)] =  1
    sig[bear & (srsi > 0.2)] = -1
    return sig


def space_Triple_SuperTrend(trial):
    return {
        'slow_p': trial.suggest_int('slow_p', 10, 20),
        'slow_m': trial.suggest_float('slow_m', 3.0, 6.0),
        'mid_p':  trial.suggest_int('mid_p', 7, 15),
        'mid_m':  trial.suggest_float('mid_m', 2.0, 5.0),
        'fast_p': trial.suggest_int('fast_p', 5, 12),
        'fast_m': trial.suggest_float('fast_m', 1.0, 4.0),
        'stoch_p': trial.suggest_int('stoch_p', 7, 21),
    }


# ── 9. ATR GOD — 4 SuperTrend consensus ─────────────────────────────────────

def gen_ATR_GOD_4ST(df, p1=10, m1=1.0, p2=11, m2=2.0, p3=12, m3=3.0, p4=13, m4=4.0, **kw):
    t1, _, _ = _supertrend_calc(df, int(p1), m1)
    t2, _, _ = _supertrend_calc(df, int(p2), m2)
    t3, _, _ = _supertrend_calc(df, int(p3), m3)
    t4, _, _ = _supertrend_calc(df, int(p4), m4)
    sig = pd.Series(0, index=df.index)
    sig[(t1 == 1) & (t2 == 1) & (t3 == 1) & (t4 == 1)]   =  1
    sig[(t1 ==-1) & (t2 ==-1) & (t3 ==-1) & (t4 ==-1)]   = -1
    return sig


def space_ATR_GOD_4ST(trial):
    return {
        'p1': trial.suggest_int('p1', 7, 15),
        'm1': trial.suggest_float('m1', 0.5, 2.5),
        'p2': trial.suggest_int('p2', 8, 16),
        'm2': trial.suggest_float('m2', 1.5, 3.5),
        'p3': trial.suggest_int('p3', 9, 18),
        'm3': trial.suggest_float('m3', 2.0, 4.5),
        'p4': trial.suggest_int('p4', 10, 20),
        'm4': trial.suggest_float('m4', 3.0, 6.0),
    }


# ── STRATEGY_EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'SuperTrend_Classic': {
        'gen':            gen_SuperTrend_Classic,
        'space':          space_SuperTrend_Classic,
        'default_params': {'atr_period': 10, 'mult': 3.0},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 23328,
                           'description': 'Classic SuperTrend ATR'},
    },
    'Pivot_SuperTrend': {
        'gen':            gen_Pivot_SuperTrend,
        'space':          space_Pivot_SuperTrend,
        'default_params': {'atr_factor': 3.0, 'atr_period': 14},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 5219,
                           'description': 'Pivot Point SuperTrend — crossover entries'},
    },
    'AI_SuperTrend_Pivot': {
        'gen':            gen_AI_SuperTrend_Pivot,
        'space':          space_AI_SuperTrend_Pivot,
        'default_params': {'atr_period': 12, 'mult': 3.0, 'pct_period': 200},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 4702,
                           'description': 'AI SuperTrend × Pivot Percentile'},
    },
    'SuperTrend_ATR_TSL': {
        'gen':            gen_SuperTrend_ATR_TSL,
        'space':          space_SuperTrend_ATR_TSL,
        'default_params': {'atr_period': 10, 'mult': 3.0, 'tsl_period': 5, 'tsl_mult': 0.5},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 4335,
                           'description': 'SuperTrend + ATR Trailing SL'},
    },
    'HA_SuperTrend': {
        'gen':            gen_HA_SuperTrend,
        'space':          space_HA_SuperTrend,
        'default_params': {'atr_period': 10, 'mult': 3.0},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 3180,
                           'description': 'Heikin Ashi SuperTrend'},
    },
    'SuperTrend_CCI': {
        'gen':            gen_SuperTrend_CCI,
        'space':          space_SuperTrend_CCI,
        'default_params': {'atr_period': 10, 'st_mult': 3.0, 'cci_period': 28},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 2818,
                           'description': 'Best SuperTrend + CCI filter'},
    },
    'AI_Volume_SuperTrend': {
        'gen':            gen_AI_Volume_SuperTrend,
        'space':          space_AI_Volume_SuperTrend,
        'default_params': {'atr_period': 10, 'mult': 3.0, 'vol_period': 20, 'k': 5},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 2796,
                           'description': 'KNN + volume-weighted SuperTrend'},
    },
    'Triple_SuperTrend_v3': {
        'gen':            gen_Triple_SuperTrend,
        'space':          space_Triple_SuperTrend,
        'default_params': {'slow_p': 14, 'slow_m': 4.0, 'mid_p': 11, 'mid_m': 3.0,
                           'fast_p': 8,  'fast_m': 2.0, 'stoch_p': 14},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 2317,
                           'description': '3x SuperTrend + StochRSI + EMA200'},
    },
    'ATR_GOD_4ST': {
        'gen':            gen_ATR_GOD_4ST,
        'space':          space_ATR_GOD_4ST,
        'default_params': {'p1': 10, 'm1': 1.0, 'p2': 11, 'm2': 2.0,
                           'p3': 12, 'm3': 3.0, 'p4': 13, 'm4': 4.0},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 2093,
                           'description': '4 SuperTrend indicators — all must agree'},
    },
}
