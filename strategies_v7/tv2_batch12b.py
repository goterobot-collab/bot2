#!/usr/bin/env python3
"""
TV2 BATCH 12b — 10 estrategias mid-tier 2026-03-31

  VWAP_Fibo_Dev          — VWAP + Fibonacci deviation bands (v4, 5084L)
  WaveTrend_Oscillator   — WaveTrend LazyBear channel osc (v4, 5500L)
  Hammers_Stars          — Hammer/Shooting Star + trend (v4, 4922L)
  Triple_EMA_ATR         — Triple EMA + ATR volatility (v4, 2400L)
  CRYPTO_3EMA_ATR        — 3 EMA + SMA200 + ATR trailing (v4, 2351L)
  Double_AI_SuperTrend   — Dual SuperTrend adaptive (v5, 2112L)
  RSI_Adaptive_T3_SAR    — T3 + PSAR + RSI confirmation (v6, 2109L)
  Hulk_Scalper           — EMA cross + momentum + volume (v4, 2100L)
  Quasimodo_Pattern      — QM pivot structure pattern (v5, 1700L)
  SuperTrend_MTF         — Multi-TF dual SuperTrend (v5, 1000L)
"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def _rma(series, period):
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _atr(df, period):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, period)


def _rsi(close, period):
    d    = close.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs   = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _supertrend_calc(df, period, mult):
    """Full SuperTrend with ATR bands and trend flip logic."""
    hl2 = (df['high'] + df['low']) / 2
    atr = _atr(df, period)
    ub  = (hl2 + mult * atr).values
    lb  = (hl2 - mult * atr).values
    cl  = df['close'].values
    n   = len(cl)
    fu  = ub.copy()
    fl  = lb.copy()
    tr  = np.ones(n)
    for i in range(1, n):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or cl[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or cl[i - 1] < fl[i - 1]) else fl[i - 1]
        if   tr[i - 1] == -1 and cl[i] > fu[i]:  tr[i] =  1
        elif tr[i - 1] ==  1 and cl[i] < fl[i]:  tr[i] = -1
        else:                                      tr[i] = tr[i - 1]
    return (pd.Series(tr, index=df.index),
            pd.Series(fl, index=df.index),
            pd.Series(fu, index=df.index))


def _psar(df, af_start=0.02, af_inc=0.02, af_max=0.20):
    """Parabolic SAR — returns Series of +1 (bullish) / -1 (bearish)."""
    high = df['high'].values
    low  = df['low'].values
    close = df['close'].values
    n = len(close)
    psar_dir = np.zeros(n)
    psar_val = np.zeros(n)
    af = af_start
    bull = True
    ep = high[0]
    psar_val[0] = low[0]
    psar_dir[0] = 1
    for i in range(1, n):
        if bull:
            psar_val[i] = psar_val[i - 1] + af * (ep - psar_val[i - 1])
            psar_val[i] = min(psar_val[i], low[i - 1])
            if i >= 2:
                psar_val[i] = min(psar_val[i], low[i - 2])
            if low[i] < psar_val[i]:
                bull = False
                psar_val[i] = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_inc, af_max)
        else:
            psar_val[i] = psar_val[i - 1] + af * (ep - psar_val[i - 1])
            psar_val[i] = max(psar_val[i], high[i - 1])
            if i >= 2:
                psar_val[i] = max(psar_val[i], high[i - 2])
            if high[i] > psar_val[i]:
                bull = True
                psar_val[i] = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_inc, af_max)
        psar_dir[i] = 1 if bull else -1
    return pd.Series(psar_dir, index=df.index)


def _t3(series, period, vfactor=0.7):
    """Tillson T3 moving average — triple-smoothed EMA."""
    e1 = _ema(series, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    e4 = _ema(e3, period)
    e5 = _ema(e4, period)
    e6 = _ema(e5, period)
    c1 = -vfactor ** 3
    c2 = 3 * vfactor ** 2 + 3 * vfactor ** 3
    c3 = -6 * vfactor ** 2 - 3 * vfactor - 3 * vfactor ** 3
    c4 = 1 + 3 * vfactor + vfactor ** 3 + 3 * vfactor ** 2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def _pivot_high(high, left, right):
    """Detect pivot highs — returns Series with pivot value or NaN."""
    n = len(high)
    vals = high.values
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        candidate = vals[i]
        is_pivot = True
        for j in range(i - left, i):
            if vals[j] >= candidate:
                is_pivot = False
                break
        if is_pivot:
            for j in range(i + 1, i + right + 1):
                if vals[j] >= candidate:
                    is_pivot = False
                    break
        if is_pivot:
            result[i] = candidate
    return pd.Series(result, index=high.index)


def _pivot_low(low, left, right):
    """Detect pivot lows — returns Series with pivot value or NaN."""
    n = len(low)
    vals = low.values
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        candidate = vals[i]
        is_pivot = True
        for j in range(i - left, i):
            if vals[j] <= candidate:
                is_pivot = False
                break
        if is_pivot:
            for j in range(i + 1, i + right + 1):
                if vals[j] <= candidate:
                    is_pivot = False
                    break
        if is_pivot:
            result[i] = candidate
    return pd.Series(result, index=low.index)


# ── 1. VWAP_Fibo_Dev ────────────────────────────────────────────────────────

def gen_VWAP_Fibo_Dev(df, vwap_period=50, fib_mult=2.0, rsi_period=14, **kw):
    close = df['close']
    high  = df['high']
    low   = df['low']
    vol   = df['volume'].replace(0, np.nan).ffill().fillna(1)
    period = int(vwap_period)

    typical = (high + low + close) / 3
    cum_tp_vol = (typical * vol).rolling(period).sum()
    cum_vol    = vol.rolling(period).sum()
    vwap = (cum_tp_vol / cum_vol.replace(0, np.nan)).fillna(close)

    std = close.rolling(period).std().fillna(0)
    upper_fib = vwap + fib_mult * std
    lower_fib = vwap - fib_mult * std

    rsi = _rsi(close, int(rsi_period))

    sig = pd.Series(0, index=df.index)
    sig[(close <= lower_fib) & (rsi < 40)] =  1
    sig[(close >= upper_fib) & (rsi > 60)] = -1
    return sig


def space_VWAP_Fibo_Dev(trial):
    return {
        'vwap_period': trial.suggest_int('vwap_period', 20, 200),
        'fib_mult':    trial.suggest_float('fib_mult', 1.0, 3.0, step=0.1),
        'rsi_period':  trial.suggest_int('rsi_period', 7, 21),
    }


# ── 2. WaveTrend_Oscillator ─────────────────────────────────────────────────

def gen_WaveTrend_Oscillator(df, wt_n1=10, wt_n2=21, ob_level=60, os_level=-60, **kw):
    close = df['close']
    high  = df['high']
    low   = df['low']
    n1 = int(wt_n1)
    n2 = int(wt_n2)

    ap  = (high + low + close) / 3
    esa = _ema(ap, n1)
    d   = _ema((ap - esa).abs(), n1)
    ci  = (ap - esa) / (0.015 * d.replace(0, np.nan)).fillna(0)
    wt1 = _ema(ci, n2)
    wt2 = wt1.rolling(4).mean()

    wt1_prev = wt1.shift(1)
    wt2_prev = wt2.shift(1)

    cross_up   = (wt1_prev <= wt2_prev) & (wt1 > wt2)
    cross_down = (wt1_prev >= wt2_prev) & (wt1 < wt2)

    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (wt1.shift(1) < os_level)] =  1
    sig[cross_down & (wt1.shift(1) > ob_level)]  = -1
    return sig


def space_WaveTrend_Oscillator(trial):
    return {
        'wt_n1':     trial.suggest_int('wt_n1', 5, 15),
        'wt_n2':     trial.suggest_int('wt_n2', 15, 30),
        'ob_level':  trial.suggest_int('ob_level', 50, 80),
        'os_level':  trial.suggest_int('os_level', -80, -50),
    }


# ── 3. Hammers_Stars ────────────────────────────────────────────────────────

def gen_Hammers_Stars(df, ema_fast=8, ema_slow=34, wick_ratio=2.5, **kw):
    close = df['close']
    open_ = df['open']
    high  = df['high']
    low   = df['low']
    ef = int(ema_fast)
    es = int(ema_slow)

    ema_f = _ema(close, ef)
    ema_s = _ema(close, es)

    body       = (close - open_).abs().clip(lower=1e-10)
    upper_wick = high - close.where(close >= open_, open_)
    lower_wick = close.where(close < open_, open_) - low

    hammer = (lower_wick > wick_ratio * body) & (upper_wick < body * 0.3)
    star   = (upper_wick > wick_ratio * body) & (lower_wick < body * 0.3)

    downtrend = ema_f < ema_s
    uptrend   = ema_f > ema_s

    sig = pd.Series(0, index=df.index)
    sig[hammer & downtrend] =  1
    sig[star   & uptrend]   = -1
    return sig


def space_Hammers_Stars(trial):
    return {
        'ema_fast':   trial.suggest_int('ema_fast', 5, 20),
        'ema_slow':   trial.suggest_int('ema_slow', 20, 60),
        'wick_ratio': trial.suggest_float('wick_ratio', 1.5, 4.0, step=0.1),
    }


# ── 4. Triple_EMA_ATR ───────────────────────────────────────────────────────

def gen_Triple_EMA_ATR(df, ema1=8, ema2=21, ema3=50, atr_period=14, atr_mult=1.5, **kw):
    close = df['close']
    e1 = _ema(close, int(ema1))
    e2 = _ema(close, int(ema2))
    e3 = _ema(close, int(ema3))

    atr_val = _atr(df, int(atr_period))
    atr_sma = atr_val.rolling(int(atr_period)).mean()
    vol_ok  = atr_val > atr_sma * atr_mult

    bull = (e1 > e2) & (e2 > e3) & vol_ok
    bear = (e1 < e2) & (e2 < e3) & vol_ok

    sig = pd.Series(0, index=df.index)
    sig[bull] =  1
    sig[bear] = -1
    return sig


def space_Triple_EMA_ATR(trial):
    return {
        'ema1':       trial.suggest_int('ema1', 5, 15),
        'ema2':       trial.suggest_int('ema2', 15, 30),
        'ema3':       trial.suggest_int('ema3', 30, 60),
        'atr_period': trial.suggest_int('atr_period', 10, 20),
        'atr_mult':   trial.suggest_float('atr_mult', 1.0, 3.0, step=0.1),
    }


# ── 5. CRYPTO_3EMA_ATR ──────────────────────────────────────────────────────

def gen_CRYPTO_3EMA_ATR(df, ema1=8, ema2=14, ema3=50, sma_trend=200, atr_period=14, **kw):
    close = df['close']
    e1 = _ema(close, int(ema1))
    e2 = _ema(close, int(ema2))
    e3 = _ema(close, int(ema3))
    sma_t = close.rolling(int(sma_trend)).mean()

    bull = (e1 > e2) & (e2 > e3) & (close > sma_t)
    bear = (e1 < e2) & (e2 < e3) & (close < sma_t)

    sig = pd.Series(0, index=df.index)
    sig[bull] =  1
    sig[bear] = -1
    return sig


def space_CRYPTO_3EMA_ATR(trial):
    return {
        'ema1':       trial.suggest_int('ema1', 5, 12),
        'ema2':       trial.suggest_int('ema2', 12, 20),
        'ema3':       trial.suggest_int('ema3', 40, 60),
        'sma_trend':  trial.suggest_int('sma_trend', 150, 250),
        'atr_period': trial.suggest_int('atr_period', 10, 20),
    }


# ── 6. Double_AI_SuperTrend ─────────────────────────────────────────────────

def gen_Double_AI_SuperTrend(df, fast_period=10, fast_mult=2.0,
                              slow_period=20, slow_mult=3.0, **kw):
    tr_fast, _, _ = _supertrend_calc(df, int(fast_period), fast_mult)
    tr_slow, _, _ = _supertrend_calc(df, int(slow_period), slow_mult)

    sig = pd.Series(0, index=df.index)
    sig[(tr_fast == 1)  & (tr_slow == 1)]  =  1
    sig[(tr_fast == -1) & (tr_slow == -1)] = -1
    return sig


def space_Double_AI_SuperTrend(trial):
    return {
        'fast_period': trial.suggest_int('fast_period', 5, 15),
        'fast_mult':   trial.suggest_float('fast_mult', 1.0, 3.0, step=0.1),
        'slow_period': trial.suggest_int('slow_period', 15, 30),
        'slow_mult':   trial.suggest_float('slow_mult', 2.0, 5.0, step=0.1),
    }


# ── 7. RSI_Adaptive_T3_SAR ──────────────────────────────────────────────────

def gen_RSI_Adaptive_T3_SAR(df, t3_period=8, t3_factor=0.7,
                             sar_start=0.02, sar_inc=0.02, rsi_period=14, **kw):
    close = df['close']
    t3_val  = _t3(close, int(t3_period), t3_factor)
    psar_d  = _psar(df, af_start=sar_start, af_inc=sar_inc)
    rsi_val = _rsi(close, int(rsi_period))

    sig = pd.Series(0, index=df.index)
    sig[(close > t3_val) & (psar_d == 1)  & (rsi_val > 50)] =  1
    sig[(close < t3_val) & (psar_d == -1) & (rsi_val < 50)] = -1
    return sig


def space_RSI_Adaptive_T3_SAR(trial):
    return {
        't3_period':  trial.suggest_int('t3_period', 3, 15),
        't3_factor':  trial.suggest_float('t3_factor', 0.5, 0.9, step=0.05),
        'sar_start':  trial.suggest_float('sar_start', 0.01, 0.04, step=0.005),
        'sar_inc':    trial.suggest_float('sar_inc', 0.01, 0.04, step=0.005),
        'rsi_period': trial.suggest_int('rsi_period', 7, 21),
    }


# ── 8. Hulk_Scalper ─────────────────────────────────────────────────────────

def gen_Hulk_Scalper(df, ema_fast=5, ema_slow=20, mom_period=14, vol_mult=1.5, **kw):
    close = df['close']
    vol   = df['volume'].fillna(0)
    ef = int(ema_fast)
    es = int(ema_slow)
    mp = int(mom_period)

    ema_f = _ema(close, ef)
    ema_s = _ema(close, es)
    mom   = close - close.shift(mp)
    vol_sma = vol.rolling(20).mean()
    vol_ok  = vol > vol_sma * vol_mult

    sig = pd.Series(0, index=df.index)
    sig[(ema_f > ema_s) & (mom > 0) & vol_ok] =  1
    sig[(ema_f < ema_s) & (mom < 0) & vol_ok] = -1
    return sig


def space_Hulk_Scalper(trial):
    return {
        'ema_fast':   trial.suggest_int('ema_fast', 3, 10),
        'ema_slow':   trial.suggest_int('ema_slow', 10, 30),
        'mom_period': trial.suggest_int('mom_period', 7, 21),
        'vol_mult':   trial.suggest_float('vol_mult', 1.0, 3.0, step=0.1),
    }


# ── 9. Quasimodo_Pattern ────────────────────────────────────────────────────

def gen_Quasimodo_Pattern(df, pivot_left=5, pivot_right=5, lookback=20, **kw):
    pl = int(pivot_left)
    pr = int(pivot_right)
    lb = int(lookback)

    ph = _pivot_high(df['high'], pl, pr)
    plw = _pivot_low(df['low'], pl, pr)

    sig = pd.Series(0, index=df.index)
    ph_vals = ph.values
    plw_vals = plw.values
    n = len(df)

    for i in range(pr + lb, n):
        # Collect recent pivot highs and lows in lookback window
        window_start = max(0, i - lb)
        recent_ph = []
        recent_pl = []
        for j in range(window_start, i + 1):
            if not np.isnan(ph_vals[j]):
                recent_ph.append((j, ph_vals[j]))
            if not np.isnan(plw_vals[j]):
                recent_pl.append((j, plw_vals[j]))

        if len(recent_ph) >= 2 and len(recent_pl) >= 2:
            # Bearish QM: higher high then lower low
            last_ph = recent_ph[-1]
            prev_ph = recent_ph[-2]
            last_pl = recent_pl[-1]
            prev_pl = recent_pl[-2]

            # Bullish QM: lower low followed by higher high
            if (last_pl[1] < prev_pl[1] and last_ph[1] > prev_ph[1]
                    and last_pl[0] < last_ph[0]):
                sig.iloc[i] = 1

            # Bearish QM: higher high followed by lower low
            elif (last_ph[1] > prev_ph[1] and last_pl[1] < prev_pl[1]
                  and last_ph[0] < last_pl[0]):
                sig.iloc[i] = -1

    return sig


def space_Quasimodo_Pattern(trial):
    return {
        'pivot_left':  trial.suggest_int('pivot_left', 3, 10),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
        'lookback':    trial.suggest_int('lookback', 10, 30),
    }


# ── 10. SuperTrend_MTF ──────────────────────────────────────────────────────

def gen_SuperTrend_MTF(df, st1_period=10, st1_mult=2.0,
                        st2_period=21, st2_mult=3.5, **kw):
    tr1, _, _ = _supertrend_calc(df, int(st1_period), st1_mult)
    tr2, _, _ = _supertrend_calc(df, int(st2_period), st2_mult)

    sig = pd.Series(0, index=df.index)
    sig[(tr1 == 1)  & (tr2 == 1)]  =  1
    sig[(tr1 == -1) & (tr2 == -1)] = -1
    return sig


def space_SuperTrend_MTF(trial):
    return {
        'st1_period': trial.suggest_int('st1_period', 7, 14),
        'st1_mult':   trial.suggest_float('st1_mult', 1.5, 3.0, step=0.1),
        'st2_period': trial.suggest_int('st2_period', 14, 30),
        'st2_mult':   trial.suggest_float('st2_mult', 2.0, 5.0, step=0.1),
    }


# ── STRATEGY_EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'VWAP_Fibo_Dev': {
        'gen': gen_VWAP_Fibo_Dev,
        'space': space_VWAP_Fibo_Dev,
        'default_params': {'vwap_period': 50, 'fib_mult': 2.0, 'rsi_period': 14},
        'info': {'version': 'v4', 'likes': 5084, 'source': 'TradingView',
                 'description': 'VWAP + Fibonacci deviation bands + RSI filter'},
    },
    'WaveTrend_Oscillator': {
        'gen': gen_WaveTrend_Oscillator,
        'space': space_WaveTrend_Oscillator,
        'default_params': {'wt_n1': 10, 'wt_n2': 21, 'ob_level': 60, 'os_level': -60},
        'info': {'version': 'v4', 'likes': 5500, 'source': 'TradingView',
                 'description': 'WaveTrend LazyBear channel oscillator'},
    },
    'Hammers_Stars': {
        'gen': gen_Hammers_Stars,
        'space': space_Hammers_Stars,
        'default_params': {'ema_fast': 8, 'ema_slow': 34, 'wick_ratio': 2.5},
        'info': {'version': 'v4', 'likes': 4922, 'source': 'TradingView',
                 'description': 'Hammer/Shooting Star candlestick + EMA trend filter'},
    },
    'Triple_EMA_ATR': {
        'gen': gen_Triple_EMA_ATR,
        'space': space_Triple_EMA_ATR,
        'default_params': {'ema1': 8, 'ema2': 21, 'ema3': 50, 'atr_period': 14, 'atr_mult': 1.5},
        'info': {'version': 'v4', 'likes': 2400, 'source': 'TradingView',
                 'description': 'Triple EMA crossover + ATR volatility filter'},
    },
    'CRYPTO_3EMA_ATR': {
        'gen': gen_CRYPTO_3EMA_ATR,
        'space': space_CRYPTO_3EMA_ATR,
        'default_params': {'ema1': 8, 'ema2': 14, 'ema3': 50, 'sma_trend': 200, 'atr_period': 14},
        'info': {'version': 'v4', 'likes': 2351, 'source': 'TradingView',
                 'description': '3 EMA + SMA200 trend filter for crypto'},
    },
    'Double_AI_SuperTrend': {
        'gen': gen_Double_AI_SuperTrend,
        'space': space_Double_AI_SuperTrend,
        'default_params': {'fast_period': 10, 'fast_mult': 2.0, 'slow_period': 20, 'slow_mult': 3.0},
        'info': {'version': 'v5', 'likes': 2112, 'source': 'TradingView',
                 'description': 'Dual SuperTrend adaptive with fast/slow alignment'},
    },
    'RSI_Adaptive_T3_SAR': {
        'gen': gen_RSI_Adaptive_T3_SAR,
        'space': space_RSI_Adaptive_T3_SAR,
        'default_params': {'t3_period': 8, 't3_factor': 0.7, 'sar_start': 0.02, 'sar_inc': 0.02, 'rsi_period': 14},
        'info': {'version': 'v6', 'likes': 2109, 'source': 'TradingView',
                 'description': 'Tillson T3 + Parabolic SAR + RSI confirmation'},
    },
    'Hulk_Scalper': {
        'gen': gen_Hulk_Scalper,
        'space': space_Hulk_Scalper,
        'default_params': {'ema_fast': 5, 'ema_slow': 20, 'mom_period': 14, 'vol_mult': 1.5},
        'info': {'version': 'v4', 'likes': 2100, 'source': 'TradingView',
                 'description': 'EMA cross scalper + momentum + volume filter'},
    },
    'Quasimodo_Pattern': {
        'gen': gen_Quasimodo_Pattern,
        'space': space_Quasimodo_Pattern,
        'default_params': {'pivot_left': 5, 'pivot_right': 5, 'lookback': 20},
        'info': {'version': 'v5', 'likes': 1700, 'source': 'TradingView',
                 'description': 'Quasimodo pivot structure pattern detection'},
    },
    'SuperTrend_MTF': {
        'gen': gen_SuperTrend_MTF,
        'space': space_SuperTrend_MTF,
        'default_params': {'st1_period': 10, 'st1_mult': 2.0, 'st2_period': 21, 'st2_mult': 3.5},
        'info': {'version': 'v5', 'likes': 1000, 'source': 'TradingView',
                 'description': 'Multi-timeframe dual SuperTrend alignment'},
    },
}
