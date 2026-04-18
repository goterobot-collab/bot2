#!/usr/bin/env python3
"""
TV2 BATCH 11c — 9 estrategias Renko/Turtle/Pivot/Aroon 2026-03-31

  Renko_SMT_ATR        — Renko ATR crossunder/over signals (v5, 1819L)
  Turtle_System_Eugene — Turtle L1/L2 sistema completo (v4, 1588L)
  Renko_Emulator_OCC   — Renko emulator ATR/Traditional (v4, 1553L)
  Renko_Intraday       — Renko intraday + SL/TP/Breakeven (v4, 1591L)
  Pivot_Reversal_RSI   — Pivot reversal + RSI filter (v4, 894L)
  Aroon_Strategy       — Aroon oscillator zero cross (v4, 394L)
  Ichimoku_RSI_NoOff   — Ichimoku no-offset + RSI (v4, 1799L)
  Supertrend_Crossover — SuperTrend bar-by-bar crossover (v4, 2317L)
  Range_Filter_Strat   — Range filter ATR breakout (v5, 1200L)
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
    return 100 - 100 / (1 + rs)


def _supertrend_calc(df, period, mult):
    hl2 = (df['high'] + df['low']) / 2
    atr  = _atr(df, period)
    ub   = (hl2 + mult * atr).values
    lb   = (hl2 - mult * atr).values
    cl   = df['close'].values
    n    = len(cl)
    fu   = ub.copy(); fl = lb.copy(); tr = np.ones(n)
    for i in range(1, n):
        fu[i] = ub[i] if (ub[i] < fu[i-1] or cl[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = lb[i] if (lb[i] > fl[i-1] or cl[i-1] < fl[i-1]) else fl[i-1]
        if   tr[i-1] == -1 and cl[i] > fu[i]: tr[i] =  1
        elif tr[i-1] ==  1 and cl[i] < fl[i]: tr[i] = -1
        else:                                   tr[i] =  tr[i-1]
    return pd.Series(tr, index=df.index), pd.Series(fl, index=df.index), pd.Series(fu, index=df.index)


def _ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    t_hi = df['high'].rolling(tenkan).max()
    t_lo = df['low'].rolling(tenkan).min()
    tenkan_sen  = (t_hi + t_lo) / 2
    k_hi = df['high'].rolling(kijun).max()
    k_lo = df['low'].rolling(kijun).min()
    kijun_sen   = (k_hi + k_lo) / 2
    senkou_a    = (tenkan_sen + kijun_sen) / 2
    sb_hi = df['high'].rolling(senkou_b).max()
    sb_lo = df['low'].rolling(senkou_b).min()
    senkou_b_v  = (sb_hi + sb_lo) / 2
    chikou      = df['close'].shift(-kijun)
    return tenkan_sen, kijun_sen, senkou_a, senkou_b_v, chikou


def _renko_atr(df, atr_period, atr_mult):
    """ATR-based Renko emulator, returns direction series."""
    bs   = (_atr(df, int(atr_period)) * atr_mult).fillna(method='bfill').values
    c    = df['close'].values
    n    = len(c)
    dir_arr = np.zeros(n)
    level = c[0]; curr = 0
    for i in range(1, n):
        b = max(bs[i], 1e-10)
        if   c[i] >= level + b: curr = 1;  level += b
        elif c[i] <= level - b: curr = -1; level -= b
        dir_arr[i] = curr
    return pd.Series(dir_arr, index=df.index)


# ── 1. Renko SMT ATR signals ─────────────────────────────────────────────────

def gen_Renko_SMT_ATR(df, atr_period=14, atr_mult=1.0, **kw):
    """ATR Renko: signal on direction change."""
    d    = _renko_atr(df, int(atr_period), atr_mult)
    prev = d.shift(1).fillna(0)
    sig  = pd.Series(0, index=df.index)
    sig[(d ==  1) & (prev != 1)]  =  1
    sig[(d == -1) & (prev != -1)] = -1
    return sig


def space_Renko_SMT_ATR(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 21),
        'atr_mult':   trial.suggest_float('atr_mult', 0.5, 3.0),
    }


# ── 2. Turtle System Eugene (L1/L2) ──────────────────────────────────────────

def gen_Turtle_System_Eugene(df, l1_enter=20, l1_exit=10, l2_enter=55, l2_exit=20, **kw):
    """Classic Turtle: L1=20-day, L2=55-day breakout. Skip if previous L1 was winner."""
    hh1 = df['high'].rolling(int(l1_enter)).max().shift(1)
    ll1 = df['low'].rolling(int(l1_enter)).min().shift(1)
    hh2 = df['high'].rolling(int(l2_enter)).max().shift(1)
    ll2 = df['low'].rolling(int(l2_enter)).min().shift(1)

    # Use L1 as primary signal
    sig = pd.Series(0, index=df.index)
    sig[df['close'] > hh1] =  1
    sig[df['close'] < ll1] = -1
    # Override with L2 on strong breakouts
    sig[df['close'] > hh2] =  1
    sig[df['close'] < ll2] = -1
    return sig


def space_Turtle_System_Eugene(trial):
    return {
        'l1_enter': trial.suggest_int('l1_enter', 10, 30),
        'l1_exit':  trial.suggest_int('l1_exit', 5, 15),
        'l2_enter': trial.suggest_int('l2_enter', 40, 80),
        'l2_exit':  trial.suggest_int('l2_exit', 15, 30),
    }


# ── 3. Renko Emulator OCC [JustUncleL] ───────────────────────────────────────

def gen_Renko_Emulator_OCC(df, brick_atr_period=10, brick_atr_mult=1.0,
                            confirm_bricks=2, **kw):
    """Renko emulator: N consecutive same-direction bricks = entry."""
    d   = _renko_atr(df, int(brick_atr_period), brick_atr_mult)
    cb  = int(confirm_bricks)

    bull_run = (d == 1).astype(int).rolling(cb).sum() == cb
    bear_run = (d == -1).astype(int).rolling(cb).sum() == cb

    sig = pd.Series(0, index=df.index)
    sig[bull_run & ~bull_run.shift(1).fillna(False)] =  1
    sig[bear_run & ~bear_run.shift(1).fillna(False)] = -1
    return sig


def space_Renko_Emulator_OCC(trial):
    return {
        'brick_atr_period': trial.suggest_int('brick_atr_period', 7, 21),
        'brick_atr_mult':   trial.suggest_float('brick_atr_mult', 0.5, 3.0),
        'confirm_bricks':   trial.suggest_int('confirm_bricks', 1, 4),
    }


# ── 4. Renko Intraday ────────────────────────────────────────────────────────

def gen_Renko_Intraday(df, atr_period=14, atr_mult=1.5, trail_atr=2.0, **kw):
    """Renko ATR with ATR trailing entry filter."""
    d    = _renko_atr(df, int(atr_period), atr_mult)
    atr  = _atr(df, int(atr_period))
    trail_long  = df['close'] - trail_atr * atr
    trail_short = df['close'] + trail_atr * atr

    sig = pd.Series(0, index=df.index)
    prev = d.shift(1).fillna(0)
    sig[(d ==  1) & (prev != 1)  & (df['close'] > trail_long.shift(1))]  =  1
    sig[(d == -1) & (prev != -1) & (df['close'] < trail_short.shift(1))] = -1
    return sig


def space_Renko_Intraday(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 21),
        'atr_mult':   trial.suggest_float('atr_mult', 0.5, 3.0),
        'trail_atr':  trial.suggest_float('trail_atr', 1.0, 4.0),
    }


# ── 5. Pivot Reversal + RSI ───────────────────────────────────────────────────

def gen_Pivot_Reversal_RSI(df, left=5, right=2, rsi_period=14, rsi_ob=70, rsi_os=30, **kw):
    """Pivot high/low reversal filtered by RSI not overbought/oversold."""
    lft, rgt = int(left), int(right)
    w    = lft + rgt + 1
    ph   = df['high'].rolling(w, center=True).max() == df['high']
    pl   = df['low'].rolling(w, center=True).min()  == df['low']
    rsi  = _rsi(df['close'], int(rsi_period))

    sig = pd.Series(0, index=df.index)
    sig[pl & (rsi < rsi_ob)] =  1   # pivot low → long if not overbought
    sig[ph & (rsi > rsi_os)] = -1   # pivot high → short if not oversold
    return sig


def space_Pivot_Reversal_RSI(trial):
    return {
        'left':       trial.suggest_int('left', 3, 10),
        'right':      trial.suggest_int('right', 1, 5),
        'rsi_period': trial.suggest_int('rsi_period', 10, 21),
        'rsi_ob':     trial.suggest_int('rsi_ob', 60, 80),
        'rsi_os':     trial.suggest_int('rsi_os', 20, 40),
    }


# ── 6. Aroon Strategy ────────────────────────────────────────────────────────

def gen_Aroon_Strategy(df, period=25, threshold=0, **kw):
    """Aroon oscillator (Up - Down). Long on cross above threshold."""
    p     = int(period)
    aroon_up  = 100 * (df['high'].rolling(p + 1).apply(lambda x: x.argmax(), raw=True)) / p
    aroon_dn  = 100 * (df['low'].rolling(p + 1).apply(lambda x: x.argmin(), raw=True)) / p
    osc   = aroon_up - aroon_dn

    long_cond  = (osc > threshold) & (osc.shift(1) <= threshold)
    short_cond = (osc < -threshold) & (osc.shift(1) >= -threshold)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Aroon_Strategy(trial):
    return {
        'period':    trial.suggest_int('period', 10, 50),
        'threshold': trial.suggest_int('threshold', -20, 20),
    }


# ── 7. Ichimoku + RSI No-Offset ───────────────────────────────────────────────

def gen_Ichimoku_RSI_NoOff(df, tenkan=9, kijun=26, senkou_b=52,
                            rsi_period=14, rsi_os=30, rsi_ob=70, **kw):
    """Ichimoku (no future offset) + RSI filter."""
    tk, kj, sa, sb, _ = _ichimoku(df, int(tenkan), int(kijun), int(senkou_b))
    rsi = _rsi(df['close'], int(rsi_period))
    cloud_top = pd.concat([sa, sb], axis=1).max(axis=1)
    cloud_bot = pd.concat([sa, sb], axis=1).min(axis=1)

    long_cond  = (tk > kj) & (df['close'] > cloud_top) & (rsi > rsi_os) & (rsi < rsi_ob)
    short_cond = (tk < kj) & (df['close'] < cloud_bot) & (rsi < rsi_ob) & (rsi > rsi_os)
    long_cross  = long_cond  & ~long_cond.shift(1).fillna(False)
    short_cross = short_cond & ~short_cond.shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    sig[long_cross]  =  1
    sig[short_cross] = -1
    return sig


def space_Ichimoku_RSI_NoOff(trial):
    return {
        'tenkan':     trial.suggest_int('tenkan', 7, 13),
        'kijun':      trial.suggest_int('kijun', 20, 34),
        'senkou_b':   trial.suggest_int('senkou_b', 40, 65),
        'rsi_period': trial.suggest_int('rsi_period', 10, 21),
        'rsi_os':     trial.suggest_int('rsi_os', 25, 40),
        'rsi_ob':     trial.suggest_int('rsi_ob', 60, 75),
    }


# ── 8. SuperTrend Crossover only ──────────────────────────────────────────────

def gen_SuperTrend_Crossover(df, atr_period=10, mult=3.0, ema_filter=200, **kw):
    """SuperTrend with EMA trend filter, only crossover entries."""
    trend, _, _ = _supertrend_calc(df, int(atr_period), mult)
    ema    = _ema(df['close'], int(ema_filter))
    prev   = trend.shift(1).fillna(1)

    sig = pd.Series(0, index=df.index)
    sig[(trend ==  1) & (prev == -1) & (df['close'] > ema)] =  1
    sig[(trend == -1) & (prev ==  1) & (df['close'] < ema)] = -1
    return sig


def space_SuperTrend_Crossover(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 20),
        'mult':       trial.suggest_float('mult', 1.5, 5.0),
        'ema_filter': trial.suggest_int('ema_filter', 50, 300),
    }


# ── 9. Range Filter Strategy ─────────────────────────────────────────────────

def _range_filter(close, rf_period, multiplier):
    """Smoothed range filter by DonovanWall."""
    rng = close.diff().abs().rolling(rf_period).mean() * multiplier
    rng_ma = rng.ewm(span=rf_period, adjust=False).mean()

    filt = close.copy().values
    n    = len(close)
    c    = close.values
    r    = rng_ma.values

    for i in range(1, n):
        if np.isnan(r[i]): filt[i] = c[i]; continue
        if   c[i] > filt[i-1] + r[i]: filt[i] = c[i] - r[i]
        elif c[i] < filt[i-1] - r[i]: filt[i] = c[i] + r[i]
        else:                           filt[i] = filt[i-1]

    return pd.Series(filt, index=close.index), rng_ma


def gen_Range_Filter_Strat(df, rf_period=14, rf_mult=1.5, **kw):
    """Range filter: upward break = long, downward break = short."""
    filt, _ = _range_filter(df['close'], int(rf_period), rf_mult)
    long_cond  = (df['close'] > filt) & (df['close'].shift(1) <= filt.shift(1))
    short_cond = (df['close'] < filt) & (df['close'].shift(1) >= filt.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Range_Filter_Strat(trial):
    return {
        'rf_period': trial.suggest_int('rf_period', 7, 30),
        'rf_mult':   trial.suggest_float('rf_mult', 0.5, 4.0),
    }


# ── STRATEGY_EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Renko_SMT_ATR': {
        'gen':            gen_Renko_SMT_ATR,
        'space':          space_Renko_SMT_ATR,
        'default_params': {'atr_period': 14, 'atr_mult': 1.0},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1819,
                           'description': 'Renko ATR crossunder/over [SMT]'},
    },
    'Turtle_System_Eugene': {
        'gen':            gen_Turtle_System_Eugene,
        'space':          space_Turtle_System_Eugene,
        'default_params': {'l1_enter': 20, 'l1_exit': 10, 'l2_enter': 55, 'l2_exit': 20},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 1588,
                           'description': 'Turtle L1/L2 system [Eugene]'},
    },
    'Renko_Emulator_OCC': {
        'gen':            gen_Renko_Emulator_OCC,
        'space':          space_Renko_Emulator_OCC,
        'default_params': {'brick_atr_period': 10, 'brick_atr_mult': 1.0, 'confirm_bricks': 2},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 1553,
                           'description': 'Renko emulator ATR/Traditional [JustUncleL]'},
    },
    'Renko_Intraday': {
        'gen':            gen_Renko_Intraday,
        'space':          space_Renko_Intraday,
        'default_params': {'atr_period': 14, 'atr_mult': 1.5, 'trail_atr': 2.0},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 1591,
                           'description': 'Renko intraday SL/TP/Breakeven'},
    },
    'Pivot_Reversal_RSI': {
        'gen':            gen_Pivot_Reversal_RSI,
        'space':          space_Pivot_Reversal_RSI,
        'default_params': {'left': 5, 'right': 2, 'rsi_period': 14, 'rsi_ob': 70, 'rsi_os': 30},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 894,
                           'description': 'Pivot reversal + RSI filter'},
    },
    'Aroon_Strategy_v2': {
        'gen':            gen_Aroon_Strategy,
        'space':          space_Aroon_Strategy,
        'default_params': {'period': 25, 'threshold': 0},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 394,
                           'description': 'Aroon oscillator zero cross'},
    },
    'Ichimoku_RSI_NoOff': {
        'gen':            gen_Ichimoku_RSI_NoOff,
        'space':          space_Ichimoku_RSI_NoOff,
        'default_params': {'tenkan': 9, 'kijun': 26, 'senkou_b': 52,
                           'rsi_period': 14, 'rsi_os': 30, 'rsi_ob': 70},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 1799,
                           'description': 'Ichimoku no-offset + RSI filter'},
    },
    'SuperTrend_Crossover': {
        'gen':            gen_SuperTrend_Crossover,
        'space':          space_SuperTrend_Crossover,
        'default_params': {'atr_period': 10, 'mult': 3.0, 'ema_filter': 200},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 2317,
                           'description': 'SuperTrend crossover + EMA trend filter'},
    },
    'Range_Filter_Strat': {
        'gen':            gen_Range_Filter_Strat,
        'space':          space_Range_Filter_Strat,
        'default_params': {'rf_period': 14, 'rf_mult': 1.5},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1200,
                           'description': 'Smoothed range filter breakout'},
    },
}
