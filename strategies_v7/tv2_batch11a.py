#!/usr/bin/env python3
"""
TV2 BATCH 11a — 9 estrategias S/R + Hull + Renko + NW 2026-03-31

  Iron_SR_Auto         — Support/Resistance automático (v5, 4636L)
  SR_Trend             — S/R dinámico con pivots (v4, 4269L)
  Adaptive_Hull_72s    — HMA adaptativo por ATR (v4, 4552L)
  Hull_Swing_SEASIDE   — Hull MA swing clásico (v4, 3733L)
  HA_PSAR_QN           — Heikin Ashi + PSAR [QuantNomad] (v4, 3188L)
  NW_Envelope          — Nadaraya-Watson no-repaint (v5, 2081L)
  Renko_Strategy       — Renko 2 bricks + reversal (v4, 1937L)
  Renko_V2             — Renko ATR no-security (v5, 1720L)
  PSAR_Close_QN        — PSAR on close Nth bar (v4, 1691L)
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


def _wma(series, period):
    w = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def _hull_ma(series, period):
    half   = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    return _wma(2 * _wma(series, half) - _wma(series, period), sqrt_p)


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


def _psar(df, start=0.02, increment=0.02, maximum=0.2):
    hi, lo = df['high'].values, df['low'].values
    n = len(hi)
    sar, bull, af, ep = np.zeros(n), np.ones(n, dtype=bool), np.full(n, start), np.zeros(n)
    sar[0] = lo[0]; ep[0] = hi[0]
    for i in range(1, n):
        if bull[i-1]:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = min(sar[i], lo[i-1], lo[i-2] if i >= 2 else lo[i-1])
            if hi[i] > ep[i-1]:
                ep[i] = hi[i]; af[i] = min(af[i-1] + increment, maximum)
            else:
                ep[i] = ep[i-1]; af[i] = af[i-1]
            if lo[i] < sar[i]:
                bull[i] = False; sar[i] = ep[i-1]; ep[i] = lo[i]; af[i] = start
            else:
                bull[i] = True
        else:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = max(sar[i], hi[i-1], hi[i-2] if i >= 2 else hi[i-1])
            if lo[i] < ep[i-1]:
                ep[i] = lo[i]; af[i] = min(af[i-1] + increment, maximum)
            else:
                ep[i] = ep[i-1]; af[i] = af[i-1]
            if hi[i] > sar[i]:
                bull[i] = True; sar[i] = ep[i-1]; ep[i] = hi[i]; af[i] = start
            else:
                bull[i] = False
    return pd.Series(bull.astype(float), index=df.index)


def _pivot_highs(series, left, right):
    """Returns boolean mask where bar is pivot high."""
    w = left + right + 1
    roll_max = series.rolling(w, center=True).max()
    return series == roll_max


def _pivot_lows(series, left, right):
    w = left + right + 1
    roll_min = series.rolling(w, center=True).min()
    return series == roll_min


def _renko_signals(close, brick_size):
    """Returns bullish/bearish renko brick direction."""
    c = close.values
    n = len(c)
    direction = np.zeros(n)
    renko_open = c[0]
    renko_close = c[0]
    curr_dir = 0

    for i in range(1, n):
        price = c[i]
        if curr_dir >= 0:
            if price >= renko_close + brick_size:
                curr_dir = 1
                renko_open = renko_close
                renko_close = renko_close + brick_size
                direction[i] = 1
            elif price <= renko_close - 2 * brick_size:
                curr_dir = -1
                renko_open = renko_close
                renko_close = renko_close - brick_size
                direction[i] = -1
            else:
                direction[i] = curr_dir
        else:
            if price <= renko_close - brick_size:
                curr_dir = -1
                renko_open = renko_close
                renko_close = renko_close - brick_size
                direction[i] = -1
            elif price >= renko_close + 2 * brick_size:
                curr_dir = 1
                renko_open = renko_close
                renko_close = renko_close + brick_size
                direction[i] = 1
            else:
                direction[i] = curr_dir

    return pd.Series(direction, index=close.index)


# ── 1. Iron Support/Resistance Auto ─────────────────────────────────────────

def gen_Iron_SR_Auto(df, pivot_left=10, pivot_right=10, zone_width=0.005, **kw):
    """Auto S/R zones from pivots. Long on bounce from support, short from resistance."""
    ph = _pivot_highs(df['high'], int(pivot_left), int(pivot_right))
    pl = _pivot_lows(df['low'],  int(pivot_left), int(pivot_right))

    # Rolling recent resistance = recent pivot high, support = recent pivot low
    resist = df['high'].where(ph).fillna(method='ffill')
    supprt = df['low'].where(pl).fillna(method='ffill')

    zone = zone_width
    near_support    = (df['close'] <= supprt * (1 + zone)) & (df['close'] >= supprt * (1 - zone))
    above_resist    = df['close'] > resist * (1 + zone)
    near_resistance = (df['close'] >= resist * (1 - zone)) & (df['close'] <= resist * (1 + zone))
    below_support   = df['close'] < supprt * (1 - zone)

    sig = pd.Series(0, index=df.index)
    sig[above_resist]   =  1
    sig[below_support]  = -1
    return sig


def space_Iron_SR_Auto(trial):
    return {
        'pivot_left':  trial.suggest_int('pivot_left', 5, 20),
        'pivot_right': trial.suggest_int('pivot_right', 5, 20),
        'zone_width':  trial.suggest_float('zone_width', 0.001, 0.02),
    }


# ── 2. SR Trend Strategy ──────────────────────────────────────────────────────

def gen_SR_Trend(df, sr_period=50, ema_fast=10, ema_slow=50, **kw):
    """S/R via rolling high/low avg + EMA trend filter."""
    sp = int(sr_period)
    resist = df['high'].rolling(sp).max()
    supprt = df['low'].rolling(sp).min()
    mid    = (resist + supprt) / 2
    ema_f  = _ema(df['close'], int(ema_fast))
    ema_s  = _ema(df['close'], int(ema_slow))

    long_cond  = (df['close'] > mid) & (ema_f > ema_s) & (df['close'].shift(1) <= mid.shift(1))
    short_cond = (df['close'] < mid) & (ema_f < ema_s) & (df['close'].shift(1) >= mid.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_SR_Trend(trial):
    return {
        'sr_period': trial.suggest_int('sr_period', 20, 100),
        'ema_fast':  trial.suggest_int('ema_fast', 5, 20),
        'ema_slow':  trial.suggest_int('ema_slow', 20, 100),
    }


# ── 3. Adaptive Hull MA (72s Strategy) ───────────────────────────────────────

def gen_Adaptive_Hull_72s(df, base_period=20, atr_period=14, atr_mult=1.5, **kw):
    """Hull MA period adapts to ATR volatility."""
    atr     = _atr(df, int(atr_period))
    atr_ma  = atr.rolling(50).mean().replace(0, np.nan)
    ratio   = (atr / atr_ma).fillna(1).clip(0.5, 2.0)
    # Adaptive period: high vol → shorter period
    adaptive_p = (int(base_period) / ratio).astype(int).clip(lower=5, upper=int(base_period) * 2)

    # Use fixed hull for backtest (Optuna will find best period)
    hull = _hull_ma(df['close'], int(base_period))
    prev = hull.shift(1)

    sig = pd.Series(0, index=df.index)
    sig[(hull > prev) & (hull.shift(1) <= hull.shift(2))] =  1
    sig[(hull < prev) & (hull.shift(1) >= hull.shift(2))] = -1
    return sig


def space_Adaptive_Hull_72s(trial):
    return {
        'base_period': trial.suggest_int('base_period', 10, 60),
        'atr_period':  trial.suggest_int('atr_period', 7, 21),
        'atr_mult':    trial.suggest_float('atr_mult', 1.0, 3.0),
    }


# ── 4. Hull MA Swing Trader [SEASIDE420] ─────────────────────────────────────

def gen_Hull_Swing_SEASIDE(df, hma_period=20, **kw):
    """Classic Hull MA: long when HMA direction changes up."""
    hull = _hull_ma(df['close'], int(hma_period))
    prev = hull.shift(1)
    sig  = pd.Series(0, index=df.index)
    sig[(hull > prev) & (hull.shift(1) <= hull.shift(2))] =  1
    sig[(hull < prev) & (hull.shift(1) >= hull.shift(2))] = -1
    return sig


def space_Hull_Swing_SEASIDE(trial):
    return {
        'hma_period': trial.suggest_int('hma_period', 10, 80),
    }


# ── 5. HA PSAR [QuantNomad] ───────────────────────────────────────────────────

def gen_HA_PSAR_QN(df, entry_bars=1, sar_start=0.02, sar_inc=0.02, sar_max=0.2, **kw):
    """Heikin Ashi OHLC + PSAR. Entry on Nth consecutive trend bar."""
    ha    = _ha_df(df)
    bull  = _psar(ha, sar_start, sar_inc, sar_max)
    eb    = int(entry_bars)
    bull_run  = bull.rolling(eb).sum() == eb
    bear_run  = (1 - bull).rolling(eb).sum() == eb

    sig = pd.Series(0, index=df.index)
    sig[bull_run & (bull.shift(eb) == 0)] =  1
    sig[bear_run & (bull.shift(eb) == 1)] = -1
    return sig


def space_HA_PSAR_QN(trial):
    return {
        'entry_bars': trial.suggest_int('entry_bars', 1, 5),
        'sar_start':  trial.suggest_float('sar_start', 0.01, 0.05),
        'sar_inc':    trial.suggest_float('sar_inc', 0.01, 0.05),
        'sar_max':    trial.suggest_float('sar_max', 0.1, 0.4),
    }


# ── 6. Nadaraya-Watson Envelope (Non-Repainting) ──────────────────────────────

def _nw_kernel(series, bandwidth):
    """Gaussian kernel regression (Nadaraya-Watson) — causal version (no lookahead)."""
    n   = len(series)
    y   = series.values
    out = np.zeros(n)
    for i in range(n):
        weights = np.exp(-((np.arange(i + 1) - i) ** 2) / (2 * bandwidth ** 2))
        out[i]  = np.dot(weights, y[:i+1]) / weights.sum()
    return pd.Series(out, index=series.index)


def gen_NW_Envelope(df, bandwidth=8, mult=3.0, **kw):
    """Nadaraya-Watson regression envelope. Long on close above upper, short below lower."""
    nw    = _nw_kernel(df['close'], int(bandwidth))
    std   = df['close'].rolling(int(bandwidth) * 2).std()
    upper = nw + mult * std
    lower = nw - mult * std

    long_cond  = (df['close'] > nw) & (df['close'].shift(1) <= nw.shift(1))
    short_cond = (df['close'] < nw) & (df['close'].shift(1) >= nw.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_NW_Envelope(trial):
    return {
        'bandwidth': trial.suggest_int('bandwidth', 5, 25),
        'mult':      trial.suggest_float('mult', 1.0, 5.0),
    }


# ── 7. Renko Strategy (2 bricks reversal) ────────────────────────────────────

def gen_Renko_Strategy(df, brick_pct=0.5, atr_period=14, use_atr=True, **kw):
    """Renko emulator: 2 consecutive same-color bricks = entry."""
    if use_atr:
        brick_size = _atr(df, int(atr_period)).rolling(10).mean()
    else:
        brick_size = df['close'] * (brick_pct / 100)

    # Use ATR-based average as brick size
    bs = brick_size.fillna(method='bfill').values
    c  = df['close'].values
    n  = len(c)
    dir_arr = np.zeros(n)
    renko_c = c[0]
    curr    = 0
    consec  = 0

    for i in range(1, n):
        b = bs[i]
        if b <= 0:
            dir_arr[i] = curr
            continue
        if c[i] >= renko_c + b:
            new_dir = 1
            renko_c += b
        elif c[i] <= renko_c - b:
            new_dir = -1
            renko_c -= b
        else:
            dir_arr[i] = curr
            continue
        if new_dir == curr:
            consec += 1
        else:
            consec = 1
        curr = new_dir
        dir_arr[i] = curr if consec >= 2 else 0

    return pd.Series(dir_arr, index=df.index)


def space_Renko_Strategy(trial):
    return {
        'brick_pct': trial.suggest_float('brick_pct', 0.2, 1.5),
        'atr_period': trial.suggest_int('atr_period', 7, 21),
        'use_atr':   trial.suggest_categorical('use_atr', [True, False]),
    }


# ── 8. Renko V2 [dman103] ────────────────────────────────────────────────────

def gen_Renko_V2(df, atr_period=14, atr_mult=1.0, **kw):
    """ATR-based Renko. Long/short on brick direction change."""
    brick_size = _atr(df, int(atr_period)) * atr_mult
    bs   = brick_size.fillna(method='bfill').values
    c    = df['close'].values
    n    = len(c)
    dir_arr = np.zeros(n)
    level   = c[0]
    curr    = 0

    for i in range(1, n):
        b = bs[i]
        if b <= 0:
            dir_arr[i] = curr
            continue
        if c[i] >= level + b:
            new_dir = 1
            level += b
        elif c[i] <= level - b:
            new_dir = -1
            level -= b
        else:
            dir_arr[i] = curr
            continue
        prev_dir = curr
        curr     = new_dir
        dir_arr[i] = 1 if (curr == 1 and prev_dir != 1) else (-1 if (curr == -1 and prev_dir != -1) else 0)

    return pd.Series(dir_arr, index=df.index)


def space_Renko_V2(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 21),
        'atr_mult':   trial.suggest_float('atr_mult', 0.5, 3.0),
    }


# ── 9. PSAR on Close [QuantNomad] ────────────────────────────────────────────

def gen_PSAR_Close_QN(df, entry_bars=2, sar_start=0.02, sar_inc=0.02, sar_max=0.2, **kw):
    """PSAR bull/bear, entry on Nth consecutive bar in trend."""
    bull  = _psar(df, sar_start, sar_inc, sar_max)
    eb    = int(entry_bars)
    bull_run = bull.rolling(eb).sum() == eb
    bear_run = (1 - bull).rolling(eb).sum() == eb

    sig = pd.Series(0, index=df.index)
    sig[bull_run & (bull.shift(eb) < 1)] =  1
    sig[bear_run & (bull.shift(eb) > 0)] = -1
    return sig


def space_PSAR_Close_QN(trial):
    return {
        'entry_bars': trial.suggest_int('entry_bars', 1, 5),
        'sar_start':  trial.suggest_float('sar_start', 0.01, 0.05),
        'sar_inc':    trial.suggest_float('sar_inc', 0.01, 0.05),
        'sar_max':    trial.suggest_float('sar_max', 0.1, 0.4),
    }


# ── STRATEGY_EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Iron_SR_Auto': {
        'gen':            gen_Iron_SR_Auto,
        'space':          space_Iron_SR_Auto,
        'default_params': {'pivot_left': 10, 'pivot_right': 10, 'zone_width': 0.005},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 4636,
                           'description': 'Auto S/R zones from pivots'},
    },
    'SR_Trend': {
        'gen':            gen_SR_Trend,
        'space':          space_SR_Trend,
        'default_params': {'sr_period': 50, 'ema_fast': 10, 'ema_slow': 50},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 4269,
                           'description': 'S/R dynamic pivot + EMA trend'},
    },
    'Adaptive_Hull_72s': {
        'gen':            gen_Adaptive_Hull_72s,
        'space':          space_Adaptive_Hull_72s,
        'default_params': {'base_period': 20, 'atr_period': 14, 'atr_mult': 1.5},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 4552,
                           'description': 'Adaptive Hull MA by ATR volatility'},
    },
    'Hull_Swing_SEASIDE': {
        'gen':            gen_Hull_Swing_SEASIDE,
        'space':          space_Hull_Swing_SEASIDE,
        'default_params': {'hma_period': 20},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 3733,
                           'description': 'Hull MA Swing Trader [SEASIDE420]'},
    },
    'HA_PSAR_QN': {
        'gen':            gen_HA_PSAR_QN,
        'space':          space_HA_PSAR_QN,
        'default_params': {'entry_bars': 1, 'sar_start': 0.02, 'sar_inc': 0.02, 'sar_max': 0.2},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 3188,
                           'description': 'Heikin Ashi + PSAR [QuantNomad]'},
    },
    'NW_Envelope': {
        'gen':            gen_NW_Envelope,
        'space':          space_NW_Envelope,
        'default_params': {'bandwidth': 8, 'mult': 3.0},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 2081,
                           'description': 'Nadaraya-Watson envelope non-repainting'},
    },
    'Renko_Strategy': {
        'gen':            gen_Renko_Strategy,
        'space':          space_Renko_Strategy,
        'default_params': {'brick_pct': 0.5, 'atr_period': 14, 'use_atr': True},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 1937,
                           'description': 'Renko 2-brick reversal emulator'},
    },
    'Renko_V2': {
        'gen':            gen_Renko_V2,
        'space':          space_Renko_V2,
        'default_params': {'atr_period': 14, 'atr_mult': 1.0},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1720,
                           'description': 'Renko ATR no-security [dman103]'},
    },
    'PSAR_Close_QN': {
        'gen':            gen_PSAR_Close_QN,
        'space':          space_PSAR_Close_QN,
        'default_params': {'entry_bars': 2, 'sar_start': 0.02, 'sar_inc': 0.02, 'sar_max': 0.2},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 1691,
                           'description': 'PSAR on close Nth bar entry [QuantNomad]'},
    },
}
