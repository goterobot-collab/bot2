#!/usr/bin/env python3
"""
TV2 BATCH 12c — 10 estrategias lower-tier 2026-03-31

  Grid_Spot_Bot          — Grid ATR mean-reversion (v5, 1800L)
  Fibonacci_TR_v6        — Fib retracement reversal (v6, 1055L)
  Trend_Reversal_v6      — EMA cross + RSI + volume spike (v6, 1000L)
  Pivot_Reversal_Range   — Pivot reversal in range ADX (v5, 1000L)
  RSI_OB_OS_Divergence   — RSI OB/OS + divergence (v5, 1000L)
  Reversal_Bot_BullByte  — RSI div + BB + ADX reversal (v6, 977L)
  Stoch_BB_MTF           — Stochastic + BB double filter (v4, 800L)
  MACD_Bidirectional     — MACD full bidirectional (v4, 498L)
  Renko_EMA_Strategy     — Renko brick + EMA trend (v4, 461L)
  Swing_Failure_Pattern  — SFP liquidity grab reversal (v5, 439L)
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
    d     = close.diff()
    gain  = d.clip(lower=0).rolling(period).mean()
    loss  = (-d.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _sma(series, period):
    return series.rolling(period).mean()


def _bb(close, period, mult):
    mid   = _sma(close, period)
    std   = close.rolling(period).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return upper, mid, lower


def _adx(df, period):
    """Full ADX calculation: DI+, DI-, DX, ADX."""
    high = df['high']
    low  = df['low']
    close = df['close']
    up   = high.diff()
    dn   = -low.diff()
    pdm  = np.where((up > dn) & (up > 0), up, 0.0)
    ndm  = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm_s = pd.Series(pdm, index=df.index)
    ndm_s = pd.Series(ndm, index=df.index)
    atr_v = _rma(pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1), period)
    pdi   = 100 * _rma(pdm_s, period) / atr_v.replace(0, np.nan)
    ndi   = 100 * _rma(ndm_s, period) / atr_v.replace(0, np.nan)
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx_v = _rma(dx.fillna(0), period)
    return adx_v.fillna(0), pdi.fillna(0), ndi.fillna(0)


def _stoch(df, k_period, d_period):
    low_min  = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    denom    = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df['close'] - low_min) / denom
    k = k.fillna(50)
    d = k.rolling(d_period).mean().fillna(50)
    return k, d


def _crossover(a, b):
    """a crosses above b."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossunder(a, b):
    """a crosses below b."""
    return (a < b) & (a.shift(1) >= b.shift(1))


def _pivot_highs(high, left, right):
    """Detect pivot highs using rolling max with offset."""
    n = len(high)
    ph = pd.Series(np.nan, index=high.index)
    vals = high.values
    for i in range(left + right, n):
        idx = i - right
        window = vals[idx - left: idx + right + 1]
        if len(window) == left + right + 1 and vals[idx] == window.max():
            ph.iloc[idx] = vals[idx]
    return ph


def _pivot_lows(low, left, right):
    """Detect pivot lows using rolling min with offset."""
    n = len(low)
    pl = pd.Series(np.nan, index=low.index)
    vals = low.values
    for i in range(left + right, n):
        idx = i - right
        window = vals[idx - left: idx + right + 1]
        if len(window) == left + right + 1 and vals[idx] == window.min():
            pl.iloc[idx] = vals[idx]
    return pl


# ── 1. Fibonacci_TR_v6 ──────────────────────────────────────────────────────

def gen_Fibonacci_TR_v6(df, lookback=50, fib_level=3, rsi_period=14, **kw):
    lookback  = int(lookback)
    fib_level = int(fib_level)
    rsi_period = int(rsi_period)

    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    fib = fib_levels[min(fib_level, 4)]

    swing_hi = df['high'].rolling(lookback).max()
    swing_lo = df['low'].rolling(lookback).min()
    rng      = swing_hi - swing_lo

    # Fib support = swing_lo + fib * range (support from bottom)
    fib_support    = swing_lo + fib * rng
    # Fib resistance = swing_hi - fib * range (resistance from top)
    fib_resistance = swing_hi - fib * rng

    rsi = _rsi(df['close'], rsi_period)
    close = df['close']

    # Long: close crosses above fib support AND RSI > 40
    long_sig = (_crossover(close, fib_support)) & (rsi > 40)
    # Short: close crosses below fib resistance AND RSI < 60
    short_sig = (_crossunder(close, fib_resistance)) & (rsi < 60)

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  = 1
    sig[short_sig] = -1
    return sig


def space_Fibonacci_TR_v6(trial):
    return {
        'lookback':   trial.suggest_int('lookback', 20, 100),
        'fib_level':  trial.suggest_int('fib_level', 0, 4),
        'rsi_period': trial.suggest_int('rsi_period', 7, 21),
    }


# ── 2. Reversal_Bot_BullByte ────────────────────────────────────────────────

def gen_Reversal_Bot_BullByte(df, rsi_period=14, bb_period=20, bb_mult=2.0,
                               adx_period=14, adx_thresh=20, **kw):
    rsi_period = int(rsi_period)
    bb_period  = int(bb_period)
    adx_period = int(adx_period)

    rsi = _rsi(df['close'], rsi_period)
    bb_upper, _, bb_lower = _bb(df['close'], bb_period, bb_mult)
    adx_v, _, _ = _adx(df, adx_period)

    close = df['close']

    long_sig  = (rsi < 30) & (close < bb_lower) & (adx_v > adx_thresh)
    short_sig = (rsi > 70) & (close > bb_upper) & (adx_v > adx_thresh)

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  = 1
    sig[short_sig] = -1
    return sig


def space_Reversal_Bot_BullByte(trial):
    return {
        'rsi_period': trial.suggest_int('rsi_period', 7, 21),
        'bb_period':  trial.suggest_int('bb_period', 15, 30),
        'bb_mult':    trial.suggest_float('bb_mult', 1.5, 3.0, step=0.1),
        'adx_period': trial.suggest_int('adx_period', 10, 20),
        'adx_thresh': trial.suggest_int('adx_thresh', 15, 30),
    }


# ── 3. Grid_Spot_Bot ────────────────────────────────────────────────────────

def gen_Grid_Spot_Bot(df, grid_atr_period=14, grid_mult=2.0, ema_period=40, **kw):
    grid_atr_period = int(grid_atr_period)
    ema_period      = int(ema_period)

    atr_v  = _atr(df, grid_atr_period)
    ema_v  = _ema(df['close'], ema_period)
    upper  = ema_v + grid_mult * atr_v
    lower  = ema_v - grid_mult * atr_v
    close  = df['close']

    long_sig  = close < lower
    short_sig = close > upper

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  = 1
    sig[short_sig] = -1
    return sig


def space_Grid_Spot_Bot(trial):
    return {
        'grid_atr_period': trial.suggest_int('grid_atr_period', 10, 30),
        'grid_mult':       trial.suggest_float('grid_mult', 1.0, 4.0, step=0.1),
        'ema_period':      trial.suggest_int('ema_period', 20, 60),
    }


# ── 4. Trend_Reversal_v6 ────────────────────────────────────────────────────

def gen_Trend_Reversal_v6(df, ema_fast=9, ema_slow=30, rsi_period=14,
                           vol_mult=1.5, **kw):
    ema_fast   = int(ema_fast)
    ema_slow   = int(ema_slow)
    rsi_period = int(rsi_period)

    ef  = _ema(df['close'], ema_fast)
    es  = _ema(df['close'], ema_slow)
    rsi = _rsi(df['close'], rsi_period)

    vol     = df['volume']
    vol_sma = _sma(vol, 20).fillna(vol)

    cross_up   = _crossover(ef, es)
    cross_down = _crossunder(ef, es)
    vol_spike  = vol > vol_sma * vol_mult

    long_sig  = cross_up   & (rsi < 60) & vol_spike
    short_sig = cross_down & (rsi > 40) & vol_spike

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  = 1
    sig[short_sig] = -1
    return sig


def space_Trend_Reversal_v6(trial):
    return {
        'ema_fast':   trial.suggest_int('ema_fast', 5, 15),
        'ema_slow':   trial.suggest_int('ema_slow', 20, 50),
        'rsi_period': trial.suggest_int('rsi_period', 7, 21),
        'vol_mult':   trial.suggest_float('vol_mult', 1.2, 3.0, step=0.1),
    }


# ── 5. Swing_Failure_Pattern ────────────────────────────────────────────────

def gen_Swing_Failure_Pattern(df, pivot_left=5, pivot_right=5,
                               min_sweep_pct=0.003, **kw):
    pivot_left  = int(pivot_left)
    pivot_right = int(pivot_right)

    ph = _pivot_highs(df['high'], pivot_left, pivot_right)
    pl = _pivot_lows(df['low'], pivot_left, pivot_right)

    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    n     = len(close)

    # Forward-fill last known pivot to compare against
    last_ph = ph.ffill().values
    last_pl = pl.ffill().values

    sig = np.zeros(n)

    for i in range(pivot_left + pivot_right + 1, n):
        pivot_lo = last_pl[i - 1]  # last known pivot low before this bar
        pivot_hi = last_ph[i - 1]  # last known pivot high before this bar

        if np.isnan(pivot_lo) and np.isnan(pivot_hi):
            continue

        # Bullish SFP: low sweeps below pivot low, then close above it
        if not np.isnan(pivot_lo):
            sweep_below = low[i] < pivot_lo * (1 - min_sweep_pct)
            close_above = close[i] > pivot_lo
            if sweep_below and close_above:
                sig[i] = 1
                continue

        # Bearish SFP: high sweeps above pivot high, then close below it
        if not np.isnan(pivot_hi):
            sweep_above = high[i] > pivot_hi * (1 + min_sweep_pct)
            close_below = close[i] < pivot_hi
            if sweep_above and close_below:
                sig[i] = -1

    return pd.Series(sig, index=df.index).astype(int)


def space_Swing_Failure_Pattern(trial):
    return {
        'pivot_left':    trial.suggest_int('pivot_left', 3, 10),
        'pivot_right':   trial.suggest_int('pivot_right', 3, 10),
        'min_sweep_pct': trial.suggest_float('min_sweep_pct', 0.001, 0.01, step=0.001),
    }


# ── 6. Renko_EMA_Strategy ───────────────────────────────────────────────────

def gen_Renko_EMA_Strategy(df, atr_period=14, ema_period=20, **kw):
    atr_period = int(atr_period)
    ema_period = int(ema_period)

    atr_v  = _atr(df, atr_period)
    ema_v  = _ema(df['close'], ema_period)
    close  = df['close'].values
    atr_vals = atr_v.fillna(0).values
    n      = len(close)

    # Simulate renko brick direction
    brick_dir = np.zeros(n)  # 1=bullish, -1=bearish
    base_price = close[0] if n > 0 else 0.0
    current_dir = 0

    for i in range(1, n):
        bs = atr_vals[i]
        if bs <= 0:
            brick_dir[i] = current_dir
            continue

        diff = close[i] - base_price
        if diff >= bs:
            current_dir = 1
            # Move base price up by full bricks
            bricks = int(diff / bs)
            base_price += bricks * bs
        elif diff <= -bs:
            current_dir = -1
            bricks = int(abs(diff) / bs)
            base_price -= bricks * bs

        brick_dir[i] = current_dir

    renko = pd.Series(brick_dir, index=df.index)
    close_s = df['close']

    long_sig  = (renko == 1)  & (close_s > ema_v)
    short_sig = (renko == -1) & (close_s < ema_v)

    # Only signal on direction change
    renko_change = renko.diff().fillna(0) != 0
    long_sig  = long_sig  & renko_change
    short_sig = short_sig & renko_change

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  = 1
    sig[short_sig] = -1
    return sig


def space_Renko_EMA_Strategy(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 21),
        'ema_period': trial.suggest_int('ema_period', 10, 40),
    }


# ── 7. MACD_Bidirectional ───────────────────────────────────────────────────

def gen_MACD_Bidirectional(df, fast=12, slow=26, signal=9, **kw):
    fast   = int(fast)
    slow   = int(slow)
    signal = int(signal)

    ema_f  = _ema(df['close'], fast)
    ema_s  = _ema(df['close'], slow)
    macd   = ema_f - ema_s
    sig_l  = _ema(macd, signal)
    hist   = macd - sig_l

    long_sig  = _crossover(macd, sig_l)  & (hist > 0)
    short_sig = _crossunder(macd, sig_l) & (hist < 0)

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  = 1
    sig[short_sig] = -1
    return sig


def space_MACD_Bidirectional(trial):
    return {
        'fast':   trial.suggest_int('fast', 8, 16),
        'slow':   trial.suggest_int('slow', 20, 30),
        'signal': trial.suggest_int('signal', 5, 12),
    }


# ── 8. Pivot_Reversal_Range ─────────────────────────────────────────────────

def gen_Pivot_Reversal_Range(df, pivot_left=5, pivot_right=3, adx_period=14,
                              adx_max=40, **kw):
    pivot_left  = int(pivot_left)
    pivot_right = int(pivot_right)
    adx_period  = int(adx_period)

    ph = _pivot_highs(df['high'], pivot_left, pivot_right)
    pl = _pivot_lows(df['low'], pivot_left, pivot_right)

    # Forward-fill pivots for nearest support/resistance
    last_ph = ph.ffill()
    last_pl = pl.ffill()

    adx_v, _, _ = _adx(df, adx_period)
    close = df['close']

    # Near pivot: within 0.3% of pivot level
    near_support    = ((close - last_pl).abs() / last_pl.replace(0, np.nan)) < 0.003
    near_resistance = ((close - last_ph).abs() / last_ph.replace(0, np.nan)) < 0.003
    range_bound     = adx_v < adx_max

    long_sig  = near_support    & range_bound & last_pl.notna()
    short_sig = near_resistance & range_bound & last_ph.notna()

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  = 1
    sig[short_sig] = -1
    return sig


def space_Pivot_Reversal_Range(trial):
    return {
        'pivot_left':  trial.suggest_int('pivot_left', 3, 10),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
        'adx_period':  trial.suggest_int('adx_period', 10, 20),
        'adx_max':     trial.suggest_int('adx_max', 30, 50),
    }


# ── 9. RSI_OB_OS_Divergence ─────────────────────────────────────────────────

def gen_RSI_OB_OS_Divergence(df, rsi_period=14, ob_level=70, os_level=30,
                              lookback=10, **kw):
    rsi_period = int(rsi_period)
    lookback   = int(lookback)

    rsi   = _rsi(df['close'], rsi_period)
    close = df['close']
    n     = len(close)

    sig = np.zeros(n)

    close_vals = close.values
    rsi_vals   = rsi.values

    for i in range(lookback + 1, n):
        window_close = close_vals[i - lookback: i + 1]
        window_rsi   = rsi_vals[i - lookback: i + 1]

        if np.any(np.isnan(window_rsi)):
            continue

        # Current RSI oversold → check bullish divergence
        if rsi_vals[i] < os_level:
            # Price makes lower low but RSI makes higher low
            price_ll = close_vals[i] < np.nanmin(window_close[:-1])
            rsi_hl   = rsi_vals[i] > np.nanmin(window_rsi[:-1])
            if price_ll and rsi_hl:
                sig[i] = 1

        # Current RSI overbought → check bearish divergence
        if rsi_vals[i] > ob_level:
            # Price makes higher high but RSI makes lower high
            price_hh = close_vals[i] > np.nanmax(window_close[:-1])
            rsi_lh   = rsi_vals[i] < np.nanmax(window_rsi[:-1])
            if price_hh and rsi_lh:
                sig[i] = -1

    return pd.Series(sig, index=df.index).astype(int)


def space_RSI_OB_OS_Divergence(trial):
    return {
        'rsi_period': trial.suggest_int('rsi_period', 7, 21),
        'ob_level':   trial.suggest_int('ob_level', 65, 80),
        'os_level':   trial.suggest_int('os_level', 20, 35),
        'lookback':   trial.suggest_int('lookback', 5, 20),
    }


# ── 10. Stoch_BB_MTF ────────────────────────────────────────────────────────

def gen_Stoch_BB_MTF(df, stoch_k=14, stoch_d=3, bb_period=20, bb_mult=2.0, **kw):
    stoch_k  = int(stoch_k)
    stoch_d  = int(stoch_d)
    bb_period = int(bb_period)

    k, d = _stoch(df, stoch_k, stoch_d)
    bb_upper, _, bb_lower = _bb(df['close'], bb_period, bb_mult)

    close = df['close']

    # Long: K crosses above D below 20 AND close < BB lower
    long_sig  = _crossover(k, d)  & (d.shift(1) < 20) & (close < bb_lower)
    # Short: K crosses below D above 80 AND close > BB upper
    short_sig = _crossunder(k, d) & (d.shift(1) > 80) & (close > bb_upper)

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  = 1
    sig[short_sig] = -1
    return sig


def space_Stoch_BB_MTF(trial):
    return {
        'stoch_k':  trial.suggest_int('stoch_k', 7, 21),
        'stoch_d':  trial.suggest_int('stoch_d', 3, 7),
        'bb_period': trial.suggest_int('bb_period', 15, 30),
        'bb_mult':  trial.suggest_float('bb_mult', 1.5, 3.0, step=0.1),
    }


# ── STRATEGY_EXPORT ─────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Fibonacci_TR_v6': {
        'gen':            gen_Fibonacci_TR_v6,
        'space':          space_Fibonacci_TR_v6,
        'default_params': {'lookback': 50, 'fib_level': 3, 'rsi_period': 14},
        'info':           {'source': 'TradingView', 'version': 6, 'likes': 1055,
                           'description': 'Fibonacci retracement trend reversal'},
    },
    'Reversal_Bot_BullByte': {
        'gen':            gen_Reversal_Bot_BullByte,
        'space':          space_Reversal_Bot_BullByte,
        'default_params': {'rsi_period': 14, 'bb_period': 20, 'bb_mult': 2.0,
                           'adx_period': 14, 'adx_thresh': 20},
        'info':           {'source': 'TradingView', 'version': 6, 'likes': 977,
                           'description': 'RSI divergence + BB + ADX trending reversal'},
    },
    'Grid_Spot_Bot': {
        'gen':            gen_Grid_Spot_Bot,
        'space':          space_Grid_Spot_Bot,
        'default_params': {'grid_atr_period': 14, 'grid_mult': 2.0, 'ema_period': 40},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1800,
                           'description': 'Grid ATR mean-reversion from EMA bands'},
    },
    'Trend_Reversal_v6': {
        'gen':            gen_Trend_Reversal_v6,
        'space':          space_Trend_Reversal_v6,
        'default_params': {'ema_fast': 9, 'ema_slow': 30, 'rsi_period': 14, 'vol_mult': 1.5},
        'info':           {'source': 'TradingView', 'version': 6, 'likes': 1000,
                           'description': 'EMA cross + RSI + volume spike reversal'},
    },
    'Swing_Failure_Pattern': {
        'gen':            gen_Swing_Failure_Pattern,
        'space':          space_Swing_Failure_Pattern,
        'default_params': {'pivot_left': 5, 'pivot_right': 5, 'min_sweep_pct': 0.003},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 439,
                           'description': 'SFP liquidity grab reversal'},
    },
    'Renko_EMA_Strategy': {
        'gen':            gen_Renko_EMA_Strategy,
        'space':          space_Renko_EMA_Strategy,
        'default_params': {'atr_period': 14, 'ema_period': 20},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 461,
                           'description': 'Renko ATR brick direction + EMA trend'},
    },
    'MACD_Bidirectional': {
        'gen':            gen_MACD_Bidirectional,
        'space':          space_MACD_Bidirectional,
        'default_params': {'fast': 12, 'slow': 26, 'signal': 9},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 498,
                           'description': 'MACD full bidirectional crossover'},
    },
    'Pivot_Reversal_Range': {
        'gen':            gen_Pivot_Reversal_Range,
        'space':          space_Pivot_Reversal_Range,
        'default_params': {'pivot_left': 5, 'pivot_right': 3, 'adx_period': 14, 'adx_max': 40},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1000,
                           'description': 'Pivot reversal in range-bound ADX filter'},
    },
    'RSI_OB_OS_Divergence': {
        'gen':            gen_RSI_OB_OS_Divergence,
        'space':          space_RSI_OB_OS_Divergence,
        'default_params': {'rsi_period': 14, 'ob_level': 70, 'os_level': 30, 'lookback': 10},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1000,
                           'description': 'RSI OB/OS with divergence confirmation'},
    },
    'Stoch_BB_MTF': {
        'gen':            gen_Stoch_BB_MTF,
        'space':          space_Stoch_BB_MTF,
        'default_params': {'stoch_k': 14, 'stoch_d': 3, 'bb_period': 20, 'bb_mult': 2.0},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 800,
                           'description': 'Stochastic + Bollinger Bands double filter'},
    },
}
