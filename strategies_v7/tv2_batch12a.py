#!/usr/bin/env python3
"""
TV2 BATCH 12a — 10 estrategias top popularity 2026-03-31

  PMax_Explorer        — PMax multi-MA + ATR trailing (v4, 16150L)
  Hull_Suite_Strategy  — HMA/EHMA/THMA suite (v4, 13577L)
  AO_Stoch_RSI_ATR     — AO + Stochastic RSI + ATR (v4, 12591L)
  Flawless_Victory     — RSI+MFI+BB+Volume ML-inspired (v4, 10589L)
  RSI_Divergence_Strat — RSI divergence + pivot detection (v4, 9172L)
  Twin_OTT             — Dual OTT fast/slow cross (v5, 8383L)
  UT_Bot_Strategy      — ATR trailing stop bot (v4, 7528L)
  AlphaTrend           — Dynamic S/R via RSI+ATR (v5, 6722L)
  OTT_Strategy         — Optimized Trend Tracker (v4, 6633L)
  SSL_Channel          — SSL MA channel cross (v4, 6200L)
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


def _sma(series, period):
    return series.rolling(int(period), min_periods=1).mean()


def _wma(series, period):
    period = int(period)
    w = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def _dema(series, period):
    e1 = _ema(series, period)
    e2 = _ema(e1, period)
    return 2 * e1 - e2


def _tema(series, period):
    e1 = _ema(series, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def _atr(df, period):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, period)


def _hull_ma(series, period):
    half   = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    return _wma(2 * _wma(series, half) - _wma(series, period), sqrt_p)


def _rsi(series, period):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _stoch(series, k_period):
    lo = series.rolling(int(k_period), min_periods=1).min()
    hi = series.rolling(int(k_period), min_periods=1).max()
    denom = (hi - lo).replace(0, np.nan)
    return ((series - lo) / denom * 100).fillna(50)


def _mfi(df, period):
    """Money Flow Index."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0)
    neg_mf = mf.where(delta < 0, 0.0)
    pos_sum = pos_mf.rolling(int(period), min_periods=1).sum()
    neg_sum = neg_mf.rolling(int(period), min_periods=1).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50)


def _crossover(a, b):
    """True where a crosses above b."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossunder(a, b):
    """True where a crosses below b."""
    return (a < b) & (a.shift(1) >= b.shift(1))


def _var_ma(series, period):
    """Variable Moving Average (VAR) — iterative adaptive MA."""
    src = series.values
    n = len(src)
    var_arr = np.full(n, np.nan)
    var_arr[0] = src[0]

    # VMA alpha components
    mom   = series.diff(1).abs().values
    vol_s = series.diff(1).abs().rolling(int(period), min_periods=1).sum().values

    for i in range(1, n):
        if np.isnan(src[i]) or np.isnan(var_arr[i - 1]):
            var_arr[i] = var_arr[i - 1] if not np.isnan(var_arr[i - 1]) else src[i]
            continue
        v = vol_s[i]
        if v == 0 or np.isnan(v):
            cmo = 0.0
        else:
            cmo = abs(mom[i]) / v if not np.isnan(mom[i]) else 0.0
        valpha = cmo
        var_arr[i] = valpha * src[i] + (1 - valpha) * var_arr[i - 1]

    return pd.Series(var_arr, index=series.index)


def _ott_calc(series, period, pct):
    """Compute OTT (Optimized Trend Tracker) from a source series."""
    var = _var_ma(series, period)
    pct_val = pct / 100.0
    var_v = var.values
    n = len(var_v)

    long_stop  = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    ott        = np.full(n, np.nan)
    direction  = np.ones(n)  # 1=up, -1=down

    long_stop[0]  = var_v[0] * (1 - pct_val)
    short_stop[0] = var_v[0] * (1 + pct_val)
    ott[0] = long_stop[0]

    for i in range(1, n):
        if np.isnan(var_v[i]):
            long_stop[i]  = long_stop[i - 1]
            short_stop[i] = short_stop[i - 1]
            direction[i]  = direction[i - 1]
            ott[i] = ott[i - 1]
            continue

        ls = var_v[i] * (1 - pct_val)
        ss = var_v[i] * (1 + pct_val)

        long_stop[i]  = max(ls, long_stop[i - 1]) if var_v[i - 1] > long_stop[i - 1] else ls
        short_stop[i] = min(ss, short_stop[i - 1]) if var_v[i - 1] < short_stop[i - 1] else ss

        if var_v[i] > short_stop[i - 1]:
            direction[i] = 1
        elif var_v[i] < long_stop[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        ott[i] = long_stop[i] if direction[i] == 1 else short_stop[i]

    return pd.Series(var_v, index=series.index), pd.Series(ott, index=series.index)


def _pivot_highs(series, left, right):
    """Boolean mask where bar is pivot high."""
    w = left + right + 1
    roll_max = series.rolling(w, center=True).max()
    return series == roll_max


def _pivot_lows(series, left, right):
    """Boolean mask where bar is pivot low."""
    w = left + right + 1
    roll_min = series.rolling(w, center=True).min()
    return series == roll_min


# ── 1. PMax_Explorer ────────────────────────────────────────────────────────

def gen_PMax_Explorer(df, atr_period=10, atr_mult=3.0, ma_period=10, ma_type=0, **kw):
    """PMax: ATR-based trailing stop with multiple MA types."""
    close = df['close']
    atr_p = int(atr_period)
    ma_p  = int(ma_period)

    atr_val = _atr(df, atr_p)

    # Select MA type
    if ma_type == 1:
        ma = _sma(close, ma_p)
    elif ma_type == 2:
        ma = _wma(close, ma_p)
    else:
        ma = _ema(close, ma_p)

    ma_v   = ma.values
    atr_v  = atr_val.values
    n      = len(close)
    pmax   = np.full(n, np.nan)
    d      = np.ones(n)  # 1=up, -1=down

    long_stop  = ma_v[0] - atr_mult * (atr_v[0] if not np.isnan(atr_v[0]) else 0)
    short_stop = ma_v[0] + atr_mult * (atr_v[0] if not np.isnan(atr_v[0]) else 0)
    pmax[0] = long_stop

    for i in range(1, n):
        if np.isnan(ma_v[i]) or np.isnan(atr_v[i]):
            pmax[i] = pmax[i - 1]
            d[i] = d[i - 1]
            continue

        ls = ma_v[i] - atr_mult * atr_v[i]
        ss = ma_v[i] + atr_mult * atr_v[i]

        if ma_v[i - 1] > long_stop:
            long_stop = max(ls, long_stop)
        else:
            long_stop = ls

        if ma_v[i - 1] < short_stop:
            short_stop = min(ss, short_stop)
        else:
            short_stop = ss

        if ma_v[i] > short_stop:
            d[i] = 1
        elif ma_v[i] < long_stop:
            d[i] = -1
        else:
            d[i] = d[i - 1]

        pmax[i] = long_stop if d[i] == 1 else short_stop

    pmax_s = pd.Series(pmax, index=df.index)
    sig = pd.Series(0, index=df.index)
    sig[_crossover(close, pmax_s)]  = 1
    sig[_crossunder(close, pmax_s)] = -1
    return sig


def space_PMax_Explorer(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 21),
        'atr_mult':   trial.suggest_float('atr_mult', 1.0, 5.0, step=0.25),
        'ma_period':  trial.suggest_int('ma_period', 5, 50),
        'ma_type':    trial.suggest_int('ma_type', 0, 2),
    }


# ── 2. Hull_Suite_Strategy ──────────────────────────────────────────────────

def gen_Hull_Suite_Strategy(df, hma_period=55, mode=0, **kw):
    """Hull MA suite: HMA / EHMA / THMA."""
    close = df['close']
    p = int(hma_period)

    if mode == 1:
        # EHMA: EMA of 2*EMA(half) - EMA(full), period sqrt
        half = max(p // 2, 1)
        sqrt_p = max(int(np.sqrt(p)), 1)
        hull = _ema(2 * _ema(close, half) - _ema(close, p), sqrt_p)
    elif mode == 2:
        # THMA: WMA of 3*WMA(p/3) - WMA(p/2), period = p
        p3 = max(p // 3, 1)
        p2 = max(p // 2, 1)
        hull = _wma(3 * _wma(close, p3) - _wma(close, p2), p)
    else:
        # HMA
        hull = _hull_ma(close, p)

    hull_dir = hull.diff()
    sig = pd.Series(0, index=df.index)
    # Turn up: was <=0, now >0
    turn_up   = (hull_dir > 0) & (hull_dir.shift(1) <= 0)
    turn_down = (hull_dir < 0) & (hull_dir.shift(1) >= 0)
    sig[turn_up]   = 1
    sig[turn_down] = -1
    return sig


def space_Hull_Suite_Strategy(trial):
    return {
        'hma_period': trial.suggest_int('hma_period', 10, 100),
        'mode':       trial.suggest_int('mode', 0, 2),
    }


# ── 3. AO_Stoch_RSI_ATR ────────────────────────────────────────────────────

def gen_AO_Stoch_RSI_ATR(df, rsi_period=14, stoch_k=14, stoch_d=3,
                          ao_fast=5, ao_slow=34, **kw):
    """Awesome Oscillator + Stochastic RSI + ATR filter."""
    close = df['close']
    hl2   = (df['high'] + df['low']) / 2

    # AO
    ao = _sma(hl2, int(ao_fast)) - _sma(hl2, int(ao_slow))

    # Stochastic RSI
    rsi_val = _rsi(close, int(rsi_period))
    stoch_rsi_k = _stoch(rsi_val, int(stoch_k))
    stoch_rsi_d = _sma(stoch_rsi_k, int(stoch_d))

    sig = pd.Series(0, index=df.index)

    # Long: AO > 0 AND StochRSI K crosses above D below 20
    long_cond = (ao > 0) & _crossover(stoch_rsi_k, stoch_rsi_d) & (stoch_rsi_k.shift(1) < 20)
    # Short: AO < 0 AND StochRSI K crosses below D above 80
    short_cond = (ao < 0) & _crossunder(stoch_rsi_k, stoch_rsi_d) & (stoch_rsi_k.shift(1) > 80)

    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_AO_Stoch_RSI_ATR(trial):
    return {
        'rsi_period': trial.suggest_int('rsi_period', 7, 21),
        'stoch_k':    trial.suggest_int('stoch_k', 7, 21),
        'stoch_d':    trial.suggest_int('stoch_d', 3, 7),
        'ao_fast':    trial.suggest_int('ao_fast', 3, 8),
        'ao_slow':    trial.suggest_int('ao_slow', 20, 50),
    }


# ── 4. Flawless_Victory ────────────────────────────────────────────────────

def gen_Flawless_Victory(df, rsi_period=14, mfi_period=14, bb_period=20,
                          bb_mult=2.0, vol_period=20, **kw):
    """RSI + MFI + Bollinger Bands + Volume filter."""
    close = df['close']

    rsi_val = _rsi(close, int(rsi_period))
    mfi_val = _mfi(df, int(mfi_period))

    # Bollinger Bands
    bb_p   = int(bb_period)
    bb_mid = _sma(close, bb_p)
    bb_std = close.rolling(bb_p, min_periods=1).std().fillna(0)
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    # Volume filter
    vol_sma = _sma(df['volume'], int(vol_period))
    vol_ok  = df['volume'] > vol_sma

    sig = pd.Series(0, index=df.index)

    long_cond  = (rsi_val < 30) & (mfi_val < 30) & (close < bb_lower) & vol_ok
    short_cond = (rsi_val > 70) & (mfi_val > 70) & (close > bb_upper) & vol_ok

    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_Flawless_Victory(trial):
    return {
        'rsi_period': trial.suggest_int('rsi_period', 7, 21),
        'mfi_period': trial.suggest_int('mfi_period', 7, 21),
        'bb_period':  trial.suggest_int('bb_period', 15, 30),
        'bb_mult':    trial.suggest_float('bb_mult', 1.5, 3.0, step=0.25),
        'vol_period': trial.suggest_int('vol_period', 10, 30),
    }


# ── 5. UT_Bot_Strategy ─────────────────────────────────────────────────────

def gen_UT_Bot_Strategy(df, key_value=1.0, atr_period=10, **kw):
    """ATR Trailing Stop bot."""
    close = df['close']
    atr_val = _atr(df, int(atr_period))
    n_loss = key_value * atr_val

    close_v = close.values
    nloss_v = n_loss.values
    n = len(close_v)

    trail = np.full(n, np.nan)
    trail[0] = close_v[0]

    for i in range(1, n):
        if np.isnan(close_v[i]) or np.isnan(nloss_v[i]):
            trail[i] = trail[i - 1]
            continue

        if close_v[i] > trail[i - 1]:
            trail[i] = max(trail[i - 1], close_v[i] - nloss_v[i])
        elif close_v[i] < trail[i - 1]:
            trail[i] = min(trail[i - 1], close_v[i] + nloss_v[i])
        else:
            trail[i] = trail[i - 1]

    trail_s = pd.Series(trail, index=df.index)
    sig = pd.Series(0, index=df.index)
    sig[_crossover(close, trail_s)]  = 1
    sig[_crossunder(close, trail_s)] = -1
    return sig


def space_UT_Bot_Strategy(trial):
    return {
        'key_value':  trial.suggest_float('key_value', 1.0, 5.0, step=0.25),
        'atr_period': trial.suggest_int('atr_period', 7, 21),
    }


# ── 6. AlphaTrend ──────────────────────────────────────────────────────────

def gen_AlphaTrend(df, atr_period=14, atr_mult=1.0, rsi_period=14, **kw):
    """Dynamic support/resistance using RSI + ATR."""
    close = df['close']
    atr_val = _atr(df, int(atr_period))
    rsi_val = _rsi(close, int(rsi_period))

    low_v  = df['low'].values
    high_v = df['high'].values
    atr_v  = atr_val.values
    rsi_v  = rsi_val.values
    n = len(close)

    alpha = np.full(n, np.nan)

    # Initialize
    alpha[0] = close.values[0]

    for i in range(1, n):
        if np.isnan(atr_v[i]):
            alpha[i] = alpha[i - 1]
            continue

        up_t   = low_v[i] - atr_v[i] * atr_mult
        down_t = high_v[i] + atr_v[i] * atr_mult

        if rsi_v[i] >= 50:
            alpha[i] = max(up_t, alpha[i - 1])
        else:
            alpha[i] = min(down_t, alpha[i - 1])

    alpha_s = pd.Series(alpha, index=df.index)
    # Signal: cross of close vs AlphaTrend shifted by 2
    alpha_lag2 = alpha_s.shift(2)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(close, alpha_lag2)]  = 1
    sig[_crossunder(close, alpha_lag2)] = -1
    return sig


def space_AlphaTrend(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 21),
        'atr_mult':   trial.suggest_float('atr_mult', 0.5, 3.0, step=0.25),
        'rsi_period': trial.suggest_int('rsi_period', 7, 21),
    }


# ── 7. OTT_Strategy ────────────────────────────────────────────────────────

def gen_OTT_Strategy(df, var_period=2, pct=1.4, **kw):
    """Optimized Trend Tracker: VAR + percentage trailing."""
    close = df['close']
    var_line, ott_line = _ott_calc(close, int(var_period), pct)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(var_line, ott_line)]  = 1
    sig[_crossunder(var_line, ott_line)] = -1
    return sig


def space_OTT_Strategy(trial):
    return {
        'var_period': trial.suggest_int('var_period', 2, 30),
        'pct':        trial.suggest_float('pct', 0.3, 3.0, step=0.1),
    }


# ── 8. Twin_OTT ────────────────────────────────────────────────────────────

def gen_Twin_OTT(df, fast_period=5, slow_period=30, fast_pct=0.6, slow_pct=1.4, **kw):
    """Dual OTT with fast and slow. Cross of two OTT lines."""
    close = df['close']
    _, fast_ott = _ott_calc(close, int(fast_period), fast_pct)
    _, slow_ott = _ott_calc(close, int(slow_period), slow_pct)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(fast_ott, slow_ott)]  = 1
    sig[_crossunder(fast_ott, slow_ott)] = -1
    return sig


def space_Twin_OTT(trial):
    return {
        'fast_period': trial.suggest_int('fast_period', 2, 20),
        'slow_period': trial.suggest_int('slow_period', 15, 50),
        'fast_pct':    trial.suggest_float('fast_pct', 0.3, 2.0, step=0.1),
        'slow_pct':    trial.suggest_float('slow_pct', 0.5, 3.0, step=0.1),
    }


# ── 9. RSI_Divergence_Strat ────────────────────────────────────────────────

def gen_RSI_Divergence_Strat(df, rsi_period=14, pivot_left=5, pivot_right=5, **kw):
    """RSI divergence detection: bullish/bearish divergence signals."""
    close = df['close']
    rsi_val = _rsi(close, int(rsi_period))
    pl = int(pivot_left)
    pr = int(pivot_right)

    close_v = close.values
    rsi_v   = rsi_val.values
    n = len(close_v)

    sig = pd.Series(0, index=df.index)

    # Find pivot lows / highs
    is_pivot_low  = _pivot_lows(close, pl, pr)
    is_pivot_high = _pivot_highs(close, pl, pr)

    # Track last pivot values
    last_pl_price = np.nan
    last_pl_rsi   = np.nan
    last_ph_price = np.nan
    last_ph_rsi   = np.nan

    offset = pr  # signals are confirmed after 'right' bars

    for i in range(pl + pr, n):
        idx = i - pr  # the actual pivot bar

        if is_pivot_low.iloc[idx]:
            price_now = close_v[idx]
            rsi_now   = rsi_v[idx]
            # Bullish divergence: price lower low, RSI higher low
            if not np.isnan(last_pl_price):
                if price_now < last_pl_price and rsi_now > last_pl_rsi:
                    sig.iloc[i] = 1
            last_pl_price = price_now
            last_pl_rsi   = rsi_now

        if is_pivot_high.iloc[idx]:
            price_now = close_v[idx]
            rsi_now   = rsi_v[idx]
            # Bearish divergence: price higher high, RSI lower high
            if not np.isnan(last_ph_price):
                if price_now > last_ph_price and rsi_now < last_ph_rsi:
                    sig.iloc[i] = -1
            last_ph_price = price_now
            last_ph_rsi   = rsi_now

    return sig


def space_RSI_Divergence_Strat(trial):
    return {
        'rsi_period':  trial.suggest_int('rsi_period', 7, 21),
        'pivot_left':  trial.suggest_int('pivot_left', 3, 10),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
    }


# ── 10. SSL_Channel ────────────────────────────────────────────────────────

def gen_SSL_Channel(df, ssl_period=10, **kw):
    """SSL = MA channel using high/low. Baseline cross."""
    close = df['close']
    p = int(ssl_period)

    ssl_down_raw = _sma(df['high'], p)
    ssl_up_raw   = _sma(df['low'], p)

    close_v    = close.values
    ssl_d_v    = ssl_down_raw.values
    ssl_u_v    = ssl_up_raw.values
    n = len(close_v)

    hlv = np.zeros(n)
    for i in range(1, n):
        if close_v[i] > ssl_d_v[i]:
            hlv[i] = 1
        elif close_v[i] < ssl_u_v[i]:
            hlv[i] = -1
        else:
            hlv[i] = hlv[i - 1]

    # Build final lines
    ssl_down = np.where(hlv < 0, ssl_d_v, ssl_u_v)
    ssl_up   = np.where(hlv > 0, ssl_d_v, ssl_u_v)

    ssl_up_s   = pd.Series(ssl_up, index=df.index)
    ssl_down_s = pd.Series(ssl_down, index=df.index)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(ssl_up_s, ssl_down_s)]  = 1
    sig[_crossunder(ssl_up_s, ssl_down_s)] = -1
    return sig


def space_SSL_Channel(trial):
    return {
        'ssl_period': trial.suggest_int('ssl_period', 5, 30),
    }


# ── STRATEGY_EXPORT ─────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'PMax_Explorer': {
        'gen': gen_PMax_Explorer,
        'space': space_PMax_Explorer,
        'default_params': {'atr_period': 10, 'atr_mult': 3.0, 'ma_period': 10, 'ma_type': 0},
        'info': {'tv_likes': 16150, 'pine_version': 4, 'type': 'trend', 'batch': 'tv2_batch12a'},
    },
    'Hull_Suite_Strategy': {
        'gen': gen_Hull_Suite_Strategy,
        'space': space_Hull_Suite_Strategy,
        'default_params': {'hma_period': 55, 'mode': 0},
        'info': {'tv_likes': 13577, 'pine_version': 4, 'type': 'trend', 'batch': 'tv2_batch12a'},
    },
    'AO_Stoch_RSI_ATR': {
        'gen': gen_AO_Stoch_RSI_ATR,
        'space': space_AO_Stoch_RSI_ATR,
        'default_params': {'rsi_period': 14, 'stoch_k': 14, 'stoch_d': 3, 'ao_fast': 5, 'ao_slow': 34},
        'info': {'tv_likes': 12591, 'pine_version': 4, 'type': 'oscillator', 'batch': 'tv2_batch12a'},
    },
    'Flawless_Victory': {
        'gen': gen_Flawless_Victory,
        'space': space_Flawless_Victory,
        'default_params': {'rsi_period': 14, 'mfi_period': 14, 'bb_period': 20, 'bb_mult': 2.0, 'vol_period': 20},
        'info': {'tv_likes': 10589, 'pine_version': 4, 'type': 'mean_reversion', 'batch': 'tv2_batch12a'},
    },
    'UT_Bot_Strategy': {
        'gen': gen_UT_Bot_Strategy,
        'space': space_UT_Bot_Strategy,
        'default_params': {'key_value': 1.0, 'atr_period': 10},
        'info': {'tv_likes': 7528, 'pine_version': 4, 'type': 'trend', 'batch': 'tv2_batch12a'},
    },
    'AlphaTrend': {
        'gen': gen_AlphaTrend,
        'space': space_AlphaTrend,
        'default_params': {'atr_period': 14, 'atr_mult': 1.0, 'rsi_period': 14},
        'info': {'tv_likes': 6722, 'pine_version': 5, 'type': 'trend', 'batch': 'tv2_batch12a'},
    },
    'OTT_Strategy': {
        'gen': gen_OTT_Strategy,
        'space': space_OTT_Strategy,
        'default_params': {'var_period': 2, 'pct': 1.4},
        'info': {'tv_likes': 6633, 'pine_version': 4, 'type': 'trend', 'batch': 'tv2_batch12a'},
    },
    'Twin_OTT': {
        'gen': gen_Twin_OTT,
        'space': space_Twin_OTT,
        'default_params': {'fast_period': 5, 'slow_period': 30, 'fast_pct': 0.6, 'slow_pct': 1.4},
        'info': {'tv_likes': 8383, 'pine_version': 5, 'type': 'trend', 'batch': 'tv2_batch12a'},
    },
    'RSI_Divergence_Strat': {
        'gen': gen_RSI_Divergence_Strat,
        'space': space_RSI_Divergence_Strat,
        'default_params': {'rsi_period': 14, 'pivot_left': 5, 'pivot_right': 5},
        'info': {'tv_likes': 9172, 'pine_version': 4, 'type': 'divergence', 'batch': 'tv2_batch12a'},
    },
    'SSL_Channel': {
        'gen': gen_SSL_Channel,
        'space': space_SSL_Channel,
        'default_params': {'ssl_period': 10},
        'info': {'tv_likes': 6200, 'pine_version': 4, 'type': 'trend', 'batch': 'tv2_batch12a'},
    },
}
