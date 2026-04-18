#!/usr/bin/env python3
"""TV2 BATCH 27 — 30 estrategias EMA Cross + MTF Simulation 2026-04-01

  EMA_5_20             — EMA 5/20 cross (v4, ~4000L)
  EMA_8_21             — EMA 8/21 Fibonacci cross (v4, ~3500L)
  EMA_13_34            — EMA 13/34 Fibonacci cross (v5, ~3000L)
  EMA_21_55            — EMA 21/55 cross (v4, ~2500L)
  EMA_50_200           — Golden/death cross 50/200 (v4, ~6000L)
  EMA_100_200          — Long-term EMA cross (v4, ~2000L)
  SMA_Cross_Classic    — SMA cross classic (v4, ~3000L)
  EMA_SMA_Combo        — EMA fast + SMA slow cross (v5, ~2500L)
  Triple_EMA_Align     — 3 EMAs aligned trend (v4, ~4000L)
  Quad_EMA_System      — 4 EMAs fully stacked (v5, ~2500L)
  EMA_5_8_13           — Three Fibonacci EMAs (v4, ~3000L)
  Ribbon_Filter_EMA    — 6-EMA ribbon filter (v5, ~2500L)
  MTF_Trend_Filter     — HTF EMA simulation trend filter (v5, ~4000L)
  MTF_RSI_Align        — Simulated HTF RSI alignment (v5, ~3000L)
  MTF_MACD_Confirm     — Fast+slow MACD confirm (v5, ~2500L)
  MTF_SuperTrend       — Two SuperTrends fast+slow (v5, ~3000L)
  MTF_BB_Squeeze       — Normal+wide BB direction (v5, ~2000L)
  EMA_Pullback         — EMA pullback bounce entry (v5, ~3500L)
  EMA_Bounce_RSI       — EMA bounce + RSI confirm (v5, ~2500L)
  EMA_Retest           — EMA cross retest secondary entry (v5, ~2000L)
  ADX_EMA_System       — ADX strength + EMA cross (v4, ~4000L)
  ADX_SuperTrend       — ADX + SuperTrend (v5, ~3000L)
  DI_Cross_EMA         — DI+/DI- cross + EMA filter (v4, ~2500L)
  ADX_RSI_EMA          — Triple ADX+RSI+EMA (v5, ~2000L)
  NNFX_EMA             — No Nonsense Forex EMA baseline (v4, ~3000L)
  Trend_Following_Basic— Simple trend follow: close vs EMA (v4, ~4000L)
  Dual_Momentum_EMA    — Two-momentum EMA system (v5, ~2500L)
  Breakout_EMA_Confirm — Breakout + EMA confirmation (v4, ~3000L)
  EMA_Slope_Strategy   — EMA slope threshold entry (v5, ~2000L)
  Inside_EMA_Break     — Inside bar breakout in EMA trend (v5, ~2000L)
"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=int(p), adjust=False).mean()


def _rma(s, p):
    return s.ewm(alpha=1.0 / int(p), adjust=False).mean()


def _sma(s, p):
    return s.rolling(int(p), min_periods=1).mean()


def _atr(df, p):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, p)


def _rsi(s, p):
    d  = s.diff()
    g  = d.clip(lower=0)
    l  = (-d).clip(lower=0)
    rs = _rma(g, p) / _rma(l, p).replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _adx(df, p):
    """Return (adx, di_plus, di_minus) all as pd.Series."""
    p = int(p)
    high  = df['high']
    low   = df['low']
    close = df['close']
    tr    = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    up   = high.diff()
    down = (-low.diff())
    dm_plus  = up.where((up > down) & (up > 0), 0.0)
    dm_minus = down.where((down > up) & (down > 0), 0.0)
    atr_p   = _rma(tr, p)
    di_plus  = 100 * _rma(dm_plus,  p) / atr_p.replace(0, 1e-9)
    di_minus = 100 * _rma(dm_minus, p) / atr_p.replace(0, 1e-9)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    adx      = _rma(dx, p)
    return adx, di_plus, di_minus


def _supertrend(df, p, mult):
    """Return pd.Series: +1 bullish, -1 bearish."""
    p    = int(p)
    atr  = _atr(df, p)
    hl2  = (df['high'] + df['low']) / 2.0
    upper_basic = hl2 + mult * atr
    lower_basic = hl2 - mult * atr
    close = df['close']
    n = len(df)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    trend = np.full(n, 1)
    ub_arr = upper_basic.values
    lb_arr = lower_basic.values
    cl_arr = close.values
    for i in range(1, n):
        prev_u = upper[i - 1] if not np.isnan(upper[i - 1]) else ub_arr[i]
        prev_l = lower[i - 1] if not np.isnan(lower[i - 1]) else lb_arr[i]
        upper[i] = ub_arr[i] if ub_arr[i] < prev_u or cl_arr[i - 1] > prev_u else prev_u
        lower[i] = lb_arr[i] if lb_arr[i] > prev_l or cl_arr[i - 1] < prev_l else prev_l
        if cl_arr[i] > upper[i]:
            trend[i] = 1
        elif cl_arr[i] < lower[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    return pd.Series(trend, index=df.index)


# ── 1. EMA_5_20 ───────────────────────────────────────────────────────────────

def gen_EMA_5_20(df, fast_p=5, slow_p=20, **kw):
    close  = df['close']
    fast   = _ema(close, fast_p)
    slow   = _ema(close, slow_p)
    cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_EMA_5_20(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 3, 8),
        'slow_p': trial.suggest_int('slow_p', 15, 30),
    }


# ── 2. EMA_8_21 ───────────────────────────────────────────────────────────────

def gen_EMA_8_21(df, fast_p=8, slow_p=21, **kw):
    close  = df['close']
    fast   = _ema(close, fast_p)
    slow   = _ema(close, slow_p)
    cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_EMA_8_21(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 6, 12),
        'slow_p': trial.suggest_int('slow_p', 17, 28),
    }


# ── 3. EMA_13_34 ──────────────────────────────────────────────────────────────

def gen_EMA_13_34(df, fast_p=13, slow_p=34, **kw):
    close  = df['close']
    fast   = _ema(close, fast_p)
    slow   = _ema(close, slow_p)
    cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_EMA_13_34(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 10, 17),
        'slow_p': trial.suggest_int('slow_p', 28, 45),
    }


# ── 4. EMA_21_55 ──────────────────────────────────────────────────────────────

def gen_EMA_21_55(df, fast_p=21, slow_p=55, **kw):
    close  = df['close']
    fast   = _ema(close, fast_p)
    slow   = _ema(close, slow_p)
    cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_EMA_21_55(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 17, 28),
        'slow_p': trial.suggest_int('slow_p', 45, 70),
    }


# ── 5. EMA_50_200 ─────────────────────────────────────────────────────────────

def gen_EMA_50_200(df, fast_p=50, slow_p=200, **kw):
    close  = df['close']
    fast   = _ema(close, fast_p)
    slow   = _ema(close, slow_p)
    cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_EMA_50_200(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 40, 60),
        'slow_p': trial.suggest_int('slow_p', 150, 250),
    }


# ── 6. EMA_100_200 ────────────────────────────────────────────────────────────

def gen_EMA_100_200(df, fast_p=100, slow_p=200, **kw):
    close  = df['close']
    fast   = _ema(close, fast_p)
    slow   = _ema(close, slow_p)
    cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_EMA_100_200(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 80, 120),
        'slow_p': trial.suggest_int('slow_p', 150, 250),
    }


# ── 7. SMA_Cross_Classic ──────────────────────────────────────────────────────

def gen_SMA_Cross_Classic(df, fast_p=10, slow_p=50, **kw):
    close  = df['close']
    fast   = _sma(close, fast_p)
    slow   = _sma(close, slow_p)
    cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_SMA_Cross_Classic(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 5, 25),
        'slow_p': trial.suggest_int('slow_p', 25, 100),
    }


# ── 8. EMA_SMA_Combo ──────────────────────────────────────────────────────────

def gen_EMA_SMA_Combo(df, ema_p=10, sma_p=50, **kw):
    close  = df['close']
    ema_line = _ema(close, ema_p)
    sma_line = _sma(close, sma_p)
    cross_up   = (ema_line > sma_line) & (ema_line.shift(1) <= sma_line.shift(1))
    cross_down = (ema_line < sma_line) & (ema_line.shift(1) >= sma_line.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_EMA_SMA_Combo(trial):
    return {
        'ema_p': trial.suggest_int('ema_p', 5, 30),
        'sma_p': trial.suggest_int('sma_p', 20, 100),
    }


# ── 9. Triple_EMA_Align ───────────────────────────────────────────────────────

def gen_Triple_EMA_Align(df, fast_p=8, mid_p=21, slow_p=55, **kw):
    close = df['close']
    e1    = _ema(close, fast_p)
    e2    = _ema(close, mid_p)
    e3    = _ema(close, slow_p)
    bull  = (e1 > e2) & (e2 > e3)
    bear  = (e1 < e2) & (e2 < e3)
    bull_start = bull & (~bull.shift(1).fillna(False))
    bear_start = bear & (~bear.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[bull_start] =  1
    sig[bear_start] = -1
    return sig


def space_Triple_EMA_Align(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 5, 15),
        'mid_p':  trial.suggest_int('mid_p',  15, 40),
        'slow_p': trial.suggest_int('slow_p', 40, 120),
    }


# ── 10. Quad_EMA_System ───────────────────────────────────────────────────────

def gen_Quad_EMA_System(df, p1=5, p2=13, p3=34, p4=89, **kw):
    close = df['close']
    e1    = _ema(close, p1)
    e2    = _ema(close, p2)
    e3    = _ema(close, p3)
    e4    = _ema(close, p4)
    bull  = (e1 > e2) & (e2 > e3) & (e3 > e4)
    bear  = (e1 < e2) & (e2 < e3) & (e3 < e4)
    bull_start = bull & (~bull.shift(1).fillna(False))
    bear_start = bear & (~bear.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[bull_start] =  1
    sig[bear_start] = -1
    return sig


def space_Quad_EMA_System(trial):
    return {
        'p1': trial.suggest_int('p1', 3,  8),
        'p2': trial.suggest_int('p2', 8,  20),
        'p3': trial.suggest_int('p3', 20, 50),
        'p4': trial.suggest_int('p4', 50, 150),
    }


# ── 11. EMA_5_8_13 ────────────────────────────────────────────────────────────

def gen_EMA_5_8_13(df, p1=5, p2=8, p3=13, **kw):
    close = df['close']
    e1    = _ema(close, p1)
    e2    = _ema(close, p2)
    e3    = _ema(close, p3)
    bull  = (e1 > e2) & (e2 > e3)
    bear  = (e1 < e2) & (e2 < e3)
    bull_start = bull & (~bull.shift(1).fillna(False))
    bear_start = bear & (~bear.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[bull_start] =  1
    sig[bear_start] = -1
    return sig


def space_EMA_5_8_13(trial):
    return {
        'p1': trial.suggest_int('p1', 3,  7),
        'p2': trial.suggest_int('p2', 6,  12),
        'p3': trial.suggest_int('p3', 10, 18),
    }


# ── 12. Ribbon_Filter_EMA ─────────────────────────────────────────────────────

def gen_Ribbon_Filter_EMA(df, base_p=5, step=4, **kw):
    close  = df['close']
    emas   = [_ema(close, int(base_p) + i * int(step)) for i in range(6)]
    # Long: price above all 6 EMAs
    above_all = pd.Series(True, index=df.index)
    below_all = pd.Series(True, index=df.index)
    for e in emas:
        above_all = above_all & (close > e)
        below_all = below_all & (close < e)
    above_start = above_all & (~above_all.shift(1).fillna(False))
    below_start = below_all & (~below_all.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[above_start] =  1
    sig[below_start] = -1
    return sig


def space_Ribbon_Filter_EMA(trial):
    return {
        'base_p': trial.suggest_int('base_p', 5,  10),
        'step':   trial.suggest_int('step',   3,  6),
    }


# ── 13. MTF_Trend_Filter ─────────────────────────────────────────────────────

def gen_MTF_Trend_Filter(df, fast_p=8, slow_p=21, htf_mult=4, **kw):
    close    = df['close']
    fast     = _ema(close, fast_p)
    slow     = _ema(close, slow_p)
    htf_ema  = _ema(close, int(slow_p) * int(htf_mult))
    long_cond  = (close > htf_ema) & (fast > slow)
    short_cond = (close < htf_ema) & (fast < slow)
    long_start  = long_cond  & (~long_cond.shift(1).fillna(False))
    short_start = short_cond & (~short_cond.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[long_start]  =  1
    sig[short_start] = -1
    return sig


def space_MTF_Trend_Filter(trial):
    return {
        'fast_p':   trial.suggest_int('fast_p',   5,  15),
        'slow_p':   trial.suggest_int('slow_p',   15, 50),
        'htf_mult': trial.suggest_int('htf_mult', 3,  6),
    }


# ── 14. MTF_RSI_Align ────────────────────────────────────────────────────────

def gen_MTF_RSI_Align(df, fast_rsi=7, slow_rsi=21, **kw):
    close    = df['close']
    rsi_fast = _rsi(close, fast_rsi)
    rsi_slow = _rsi(close, slow_rsi)
    bull = (rsi_fast > 50) & (rsi_slow > 50)
    bear = (rsi_fast < 50) & (rsi_slow < 50)
    bull_start = bull & (~bull.shift(1).fillna(False))
    bear_start = bear & (~bear.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[bull_start] =  1
    sig[bear_start] = -1
    return sig


def space_MTF_RSI_Align(trial):
    return {
        'fast_rsi': trial.suggest_int('fast_rsi', 5,  14),
        'slow_rsi': trial.suggest_int('slow_rsi', 14, 42),
    }


# ── 15. MTF_MACD_Confirm ─────────────────────────────────────────────────────

def gen_MTF_MACD_Confirm(df, fast_p=7, slow_p=34, sig_p=9, **kw):
    close = df['close']
    # Fast MACD for entry
    macd_fast  = _ema(close, fast_p) - _ema(close, slow_p)
    signal_f   = _ema(macd_fast, sig_p)
    # Slow MACD for trend (2x periods)
    macd_slow  = _ema(close, int(fast_p) * 2) - _ema(close, int(slow_p) * 2)
    long_cond  = (macd_fast > signal_f) & (macd_slow > 0)
    short_cond = (macd_fast < signal_f) & (macd_slow < 0)
    long_start  = long_cond  & (~long_cond.shift(1).fillna(False))
    short_start = short_cond & (~short_cond.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[long_start]  =  1
    sig[short_start] = -1
    return sig


def space_MTF_MACD_Confirm(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 5,  10),
        'slow_p': trial.suggest_int('slow_p', 25, 50),
        'sig_p':  trial.suggest_int('sig_p',  5,  12),
    }


# ── 16. MTF_SuperTrend ───────────────────────────────────────────────────────

def gen_MTF_SuperTrend(df, fast_p=7, fast_mult=2.0, slow_p=21, slow_mult=3.5, **kw):
    st_fast = _supertrend(df, fast_p, fast_mult)
    st_slow = _supertrend(df, slow_p, slow_mult)
    bull    = (st_fast == 1) & (st_slow == 1)
    bear    = (st_fast == -1) & (st_slow == -1)
    bull_start = bull & (~bull.shift(1).fillna(False))
    bear_start = bear & (~bear.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[bull_start] =  1
    sig[bear_start] = -1
    return sig


def space_MTF_SuperTrend(trial):
    return {
        'fast_p':    trial.suggest_int('fast_p',   5,  12),
        'fast_mult': trial.suggest_float('fast_mult', 1.5, 3.0),
        'slow_p':    trial.suggest_int('slow_p',   15, 30),
        'slow_mult': trial.suggest_float('slow_mult', 2.5, 5.0),
    }


# ── 17. MTF_BB_Squeeze ───────────────────────────────────────────────────────

def gen_MTF_BB_Squeeze(df, fast_p=20, slow_p=60, bb_mult=2.0, **kw):
    close  = df['close']
    fast_p = int(fast_p)
    slow_p = int(slow_p)
    # Fast BB
    fast_mid = _sma(close, fast_p)
    fast_std = close.rolling(fast_p, min_periods=1).std().fillna(0)
    fast_upper = fast_mid + bb_mult * fast_std
    fast_lower = fast_mid - bb_mult * fast_std
    fast_width = (fast_upper - fast_lower) / fast_mid.replace(0, 1e-9)
    # Slow BB (simulated HTF)
    slow_mid = _sma(close, slow_p)
    slow_std = close.rolling(slow_p, min_periods=1).std().fillna(0)
    slow_upper = slow_mid + bb_mult * slow_std
    slow_lower = slow_mid - bb_mult * slow_std
    # Squeeze: fast width at local min
    width_min = fast_width.rolling(fast_p, min_periods=1).min()
    was_squeeze = (fast_width.shift(1) <= width_min.shift(1) * 1.05).fillna(False)
    expanding   = fast_width > fast_width.shift(1).fillna(fast_width)
    # Slow BB trending up/down = close relative to slow mid
    slow_up   = close > slow_mid
    slow_down = close < slow_mid
    long_cond  = was_squeeze & expanding & slow_up
    short_cond = was_squeeze & expanding & slow_down
    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_MTF_BB_Squeeze(trial):
    return {
        'fast_p':   trial.suggest_int('fast_p',   15, 25),
        'slow_p':   trial.suggest_int('slow_p',   45, 90),
        'bb_mult':  trial.suggest_float('bb_mult', 1.5, 2.5),
    }


# ── 18. EMA_Pullback ─────────────────────────────────────────────────────────

def gen_EMA_Pullback(df, fast_p=8, slow_p=34, **kw):
    close    = df['close']
    fast_ema = _ema(close, fast_p)
    slow_ema = _ema(close, slow_p)
    uptrend  = fast_ema > slow_ema
    # Pullback: price touches slow EMA (close <= slow_ema) in uptrend
    touching_slow = close <= slow_ema
    # Entry: previous bar touched slow EMA, current bar closes back above it
    prev_touch  = uptrend.shift(1).fillna(False) & touching_slow.shift(1).fillna(False)
    bounce_up   = close > slow_ema
    long_entry  = prev_touch & bounce_up
    # Mirror: downtrend, price touches slow EMA from below, then drops back below
    downtrend       = fast_ema < slow_ema
    touching_slow_b = close >= slow_ema
    prev_touch_b    = downtrend.shift(1).fillna(False) & touching_slow_b.shift(1).fillna(False)
    bounce_down     = close < slow_ema
    short_entry = prev_touch_b & bounce_down
    sig = pd.Series(0, index=df.index)
    sig[long_entry]  =  1
    sig[short_entry] = -1
    return sig


def space_EMA_Pullback(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 5,  20),
        'slow_p': trial.suggest_int('slow_p', 20, 60),
    }


# ── 19. EMA_Bounce_RSI ───────────────────────────────────────────────────────

def gen_EMA_Bounce_RSI(df, ema_p=21, rsi_p=14, touch_pct=0.002, **kw):
    close   = df['close']
    ema_line = _ema(close, ema_p)
    rsi_line = _rsi(close, rsi_p)
    # Touch: price within touch_pct of EMA
    near_ema = ((close - ema_line).abs() / ema_line.replace(0, 1e-9)) <= touch_pct
    # Long: RSI was below 50, price touches EMA, bounces above EMA
    rsi_was_below = (rsi_line.shift(1) < 50).fillna(False)
    close_above   = close > ema_line
    prev_near     = near_ema.shift(1).fillna(False)
    long_entry    = prev_near & rsi_was_below & close_above
    # Short: RSI was above 50, price touches EMA, drops below EMA
    rsi_was_above = (rsi_line.shift(1) > 50).fillna(False)
    close_below   = close < ema_line
    short_entry   = prev_near & rsi_was_above & close_below
    sig = pd.Series(0, index=df.index)
    sig[long_entry]  =  1
    sig[short_entry] = -1
    return sig


def space_EMA_Bounce_RSI(trial):
    return {
        'ema_p':     trial.suggest_int('ema_p',   10, 40),
        'rsi_p':     trial.suggest_int('rsi_p',   7,  21),
        'touch_pct': trial.suggest_float('touch_pct', 0.001, 0.005),
    }


# ── 20. EMA_Retest ───────────────────────────────────────────────────────────

def gen_EMA_Retest(df, fast_p=8, slow_p=34, zone_pct=0.005, **kw):
    close    = df['close']
    fast_ema = _ema(close, fast_p)
    slow_ema = _ema(close, slow_p)
    # Already in bullish cross
    bullish_cross = fast_ema > slow_ema
    # Price came back near slow EMA (retest zone)
    near_slow = ((close - slow_ema).abs() / slow_ema.replace(0, 1e-9)) <= zone_pct
    # Bounce: close now above slow EMA after retest
    bounce_up = close > slow_ema
    long_entry = bullish_cross & near_slow.shift(1).fillna(False) & bounce_up
    # Mirror bearish
    bearish_cross = fast_ema < slow_ema
    near_slow_b   = ((close - slow_ema).abs() / slow_ema.replace(0, 1e-9)) <= zone_pct
    bounce_down   = close < slow_ema
    short_entry   = bearish_cross & near_slow_b.shift(1).fillna(False) & bounce_down
    sig = pd.Series(0, index=df.index)
    sig[long_entry]  =  1
    sig[short_entry] = -1
    return sig


def space_EMA_Retest(trial):
    return {
        'fast_p':   trial.suggest_int('fast_p',   5,  20),
        'slow_p':   trial.suggest_int('slow_p',   20, 60),
        'zone_pct': trial.suggest_float('zone_pct', 0.002, 0.01),
    }


# ── 21. ADX_EMA_System ───────────────────────────────────────────────────────

def gen_ADX_EMA_System(df, adx_p=14, fast_p=8, slow_p=34, adx_thresh=25, **kw):
    close       = df['close']
    adx, _, _   = _adx(df, adx_p)
    fast        = _ema(close, fast_p)
    slow        = _ema(close, slow_p)
    cross_up    = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down  = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    trending    = adx >= adx_thresh
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & trending] =  1
    sig[cross_down & trending] = -1
    return sig


def space_ADX_EMA_System(trial):
    return {
        'adx_p':      trial.suggest_int('adx_p',      10, 20),
        'fast_p':     trial.suggest_int('fast_p',     5,  20),
        'slow_p':     trial.suggest_int('slow_p',     20, 60),
        'adx_thresh': trial.suggest_int('adx_thresh', 20, 30),
    }


# ── 22. ADX_SuperTrend ───────────────────────────────────────────────────────

def gen_ADX_SuperTrend(df, adx_p=14, adx_thresh=20, st_p=10, st_mult=3.0, **kw):
    adx, _, _ = _adx(df, adx_p)
    st        = _supertrend(df, st_p, st_mult)
    trending  = adx >= adx_thresh
    bull      = (st == 1) & trending
    bear      = (st == -1) & trending
    bull_start = bull & (~bull.shift(1).fillna(False))
    bear_start = bear & (~bear.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[bull_start] =  1
    sig[bear_start] = -1
    return sig


def space_ADX_SuperTrend(trial):
    return {
        'adx_p':      trial.suggest_int('adx_p',      10, 20),
        'adx_thresh': trial.suggest_int('adx_thresh', 15, 30),
        'st_p':       trial.suggest_int('st_p',       7,  21),
        'st_mult':    trial.suggest_float('st_mult',  2.0, 4.0),
    }


# ── 23. DI_Cross_EMA ─────────────────────────────────────────────────────────

def gen_DI_Cross_EMA(df, adx_p=14, ema_p=100, **kw):
    close             = df['close']
    _, di_plus, di_minus = _adx(df, adx_p)
    ema_line          = _ema(close, ema_p)
    cross_up   = (di_plus > di_minus) & (di_plus.shift(1) <= di_minus.shift(1))
    cross_down = (di_plus < di_minus) & (di_plus.shift(1) >= di_minus.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (close > ema_line)] =  1
    sig[cross_down & (close < ema_line)] = -1
    return sig


def space_DI_Cross_EMA(trial):
    return {
        'adx_p': trial.suggest_int('adx_p', 10, 20),
        'ema_p': trial.suggest_int('ema_p', 50, 200),
    }


# ── 24. ADX_RSI_EMA ──────────────────────────────────────────────────────────

def gen_ADX_RSI_EMA(df, adx_p=14, rsi_p=14, ema_p=50, adx_thresh=20, **kw):
    close         = df['close']
    adx, _, _     = _adx(df, adx_p)
    rsi_line      = _rsi(close, rsi_p)
    ema_line      = _ema(close, ema_p)
    trending      = adx >= adx_thresh
    long_cond     = trending & (rsi_line > 50) & (close > ema_line)
    short_cond    = trending & (rsi_line < 50) & (close < ema_line)
    long_start    = long_cond  & (~long_cond.shift(1).fillna(False))
    short_start   = short_cond & (~short_cond.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[long_start]  =  1
    sig[short_start] = -1
    return sig


def space_ADX_RSI_EMA(trial):
    return {
        'adx_p':      trial.suggest_int('adx_p',      10, 20),
        'rsi_p':      trial.suggest_int('rsi_p',      7,  21),
        'ema_p':      trial.suggest_int('ema_p',      20, 100),
        'adx_thresh': trial.suggest_int('adx_thresh', 15, 30),
    }


# ── 25. NNFX_EMA ─────────────────────────────────────────────────────────────

def gen_NNFX_EMA(df, baseline_p=100, fast_p=20, atr_p=14, **kw):
    close    = df['close']
    volume   = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
    baseline = _ema(close, baseline_p)
    fast_ema = _ema(close, fast_p)
    slow_ema = _ema(close, int(fast_p) * 2)
    atr_line = _atr(df, atr_p)
    avg_atr  = _sma(atr_line, atr_p)
    avg_vol  = _sma(volume, atr_p)
    atr_ok   = atr_line > avg_atr * 0.8
    vol_ok   = volume > avg_vol * 0.8
    long_cond  = (close > baseline) & (fast_ema > slow_ema) & atr_ok & vol_ok
    short_cond = (close < baseline) & (fast_ema < slow_ema) & atr_ok & vol_ok
    long_start  = long_cond  & (~long_cond.shift(1).fillna(False))
    short_start = short_cond & (~short_cond.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[long_start]  =  1
    sig[short_start] = -1
    return sig


def space_NNFX_EMA(trial):
    return {
        'baseline_p': trial.suggest_int('baseline_p', 50, 200),
        'fast_p':     trial.suggest_int('fast_p',     10, 30),
        'atr_p':      trial.suggest_int('atr_p',      14, 20),
    }


# ── 26. Trend_Following_Basic ────────────────────────────────────────────────

def gen_Trend_Following_Basic(df, ema_p=50, **kw):
    close    = df['close']
    ema_line = _ema(close, ema_p)
    above    = close > ema_line
    below    = close < ema_line
    # Signal on transition: price crosses EMA
    cross_up   = above & (~above.shift(1).fillna(False))
    cross_down = below & (~below.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_Trend_Following_Basic(trial):
    return {
        'ema_p': trial.suggest_int('ema_p', 20, 100),
    }


# ── 27. Dual_Momentum_EMA ────────────────────────────────────────────────────

def gen_Dual_Momentum_EMA(df, fast_p=10, slow_p=40, **kw):
    close     = df['close']
    fast_ema  = _ema(close, fast_p)
    slow_ema  = _ema(close, slow_p)
    # Short-term momentum: close/fast_ema - 1
    mom_fast  = close / fast_ema.replace(0, 1e-9) - 1
    # Long-term momentum: fast_ema/slow_ema - 1
    mom_slow  = fast_ema / slow_ema.replace(0, 1e-9) - 1
    bull      = (mom_fast > 0) & (mom_slow > 0)
    bear      = (mom_fast < 0) & (mom_slow < 0)
    bull_start = bull & (~bull.shift(1).fillna(False))
    bear_start = bear & (~bear.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index)
    sig[bull_start] =  1
    sig[bear_start] = -1
    return sig


def space_Dual_Momentum_EMA(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 5,  20),
        'slow_p': trial.suggest_int('slow_p', 20, 60),
    }


# ── 28. Breakout_EMA_Confirm ─────────────────────────────────────────────────

def gen_Breakout_EMA_Confirm(df, break_p=20, ema_p=50, **kw):
    close    = df['close']
    ema_line = _ema(close, ema_p)
    # Prior period highest close (exclude current bar)
    highest  = close.rolling(int(break_p), min_periods=1).max().shift(1)
    lowest   = close.rolling(int(break_p), min_periods=1).min().shift(1)
    breakout_up   = (close > highest) & (close > ema_line)
    breakout_down = (close < lowest)  & (close < ema_line)
    sig = pd.Series(0, index=df.index)
    sig[breakout_up]   =  1
    sig[breakout_down] = -1
    return sig


def space_Breakout_EMA_Confirm(trial):
    return {
        'break_p': trial.suggest_int('break_p', 10, 50),
        'ema_p':   trial.suggest_int('ema_p',   20, 100),
    }


# ── 29. EMA_Slope_Strategy ───────────────────────────────────────────────────

def gen_EMA_Slope_Strategy(df, ema_p=21, slope_p=5, threshold=0.001, **kw):
    close    = df['close']
    ema_line = _ema(close, ema_p)
    prev_ema = ema_line.shift(int(slope_p))
    slope    = (ema_line - prev_ema) / prev_ema.replace(0, 1e-9)
    slope_up   = (slope >  threshold) & (slope.shift(1) <=  threshold)
    slope_down = (slope < -threshold) & (slope.shift(1) >= -threshold)
    sig = pd.Series(0, index=df.index)
    sig[slope_up]   =  1
    sig[slope_down] = -1
    return sig


def space_EMA_Slope_Strategy(trial):
    return {
        'ema_p':     trial.suggest_int('ema_p',       10, 50),
        'slope_p':   trial.suggest_int('slope_p',     3,  10),
        'threshold': trial.suggest_float('threshold', 0.0, 0.003),
    }


# ── 30. Inside_EMA_Break ─────────────────────────────────────────────────────

def gen_Inside_EMA_Break(df, ema_p=50, **kw):
    close    = df['close']
    high     = df['high']
    low      = df['low']
    ema_line = _ema(close, ema_p)
    # Inside bar: high < prev high AND low > prev low
    prev_high = high.shift(1)
    prev_low  = low.shift(1)
    inside    = (high < prev_high) & (low > prev_low)
    # Uptrend: close > EMA
    uptrend   = close > ema_line
    downtrend = close < ema_line
    # Breakout of inside bar: close breaks prev_high (inside bar) in uptrend
    prev_inside_high = high.where(inside).ffill()
    prev_inside_low  = low.where(inside).ffill()
    long_entry  = uptrend   & ~inside & (close > prev_inside_high.shift(1).fillna(np.inf))
    short_entry = downtrend & ~inside & (close < prev_inside_low.shift(1).fillna(-np.inf))
    sig = pd.Series(0, index=df.index)
    sig[long_entry]  =  1
    sig[short_entry] = -1
    return sig


def space_Inside_EMA_Break(trial):
    return {
        'ema_p': trial.suggest_int('ema_p', 20, 100),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'EMA_5_20': {
        'gen': gen_EMA_5_20,
        'space': space_EMA_5_20,
        'default_params': {'fast_p': 5, 'slow_p': 20},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 4000,
                 'description': 'EMA 5/20 cross: long on cross up, short on cross down'},
    },
    'EMA_8_21': {
        'gen': gen_EMA_8_21,
        'space': space_EMA_8_21,
        'default_params': {'fast_p': 8, 'slow_p': 21},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3500,
                 'description': 'EMA 8/21 Fibonacci cross system'},
    },
    'EMA_13_34': {
        'gen': gen_EMA_13_34,
        'space': space_EMA_13_34,
        'default_params': {'fast_p': 13, 'slow_p': 34},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'EMA 13/34 Fibonacci cross for medium-term trend'},
    },
    'EMA_21_55': {
        'gen': gen_EMA_21_55,
        'space': space_EMA_21_55,
        'default_params': {'fast_p': 21, 'slow_p': 55},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'EMA 21/55 swing trend cross'},
    },
    'EMA_50_200': {
        'gen': gen_EMA_50_200,
        'space': space_EMA_50_200,
        'default_params': {'fast_p': 50, 'slow_p': 200},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 6000,
                 'description': 'Golden/death cross EMA 50/200 — classic trend signal'},
    },
    'EMA_100_200': {
        'gen': gen_EMA_100_200,
        'space': space_EMA_100_200,
        'default_params': {'fast_p': 100, 'slow_p': 200},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'Long-term EMA 100/200 cross for macro trend direction'},
    },
    'SMA_Cross_Classic': {
        'gen': gen_SMA_Cross_Classic,
        'space': space_SMA_Cross_Classic,
        'default_params': {'fast_p': 10, 'slow_p': 50},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'Classic SMA cross (slower than EMA, less noise)'},
    },
    'EMA_SMA_Combo': {
        'gen': gen_EMA_SMA_Combo,
        'space': space_EMA_SMA_Combo,
        'default_params': {'ema_p': 10, 'sma_p': 50},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'EMA fast + SMA slow cross: EMA reacts quicker than SMA'},
    },
    'Triple_EMA_Align': {
        'gen': gen_Triple_EMA_Align,
        'space': space_Triple_EMA_Align,
        'default_params': {'fast_p': 8, 'mid_p': 21, 'slow_p': 55},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 4000,
                 'description': 'Three EMAs fully aligned — high-confidence trend filter'},
    },
    'Quad_EMA_System': {
        'gen': gen_Quad_EMA_System,
        'space': space_Quad_EMA_System,
        'default_params': {'p1': 5, 'p2': 13, 'p3': 34, 'p4': 89},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Four EMAs stacked in perfect alignment — strictest trend filter'},
    },
    'EMA_5_8_13': {
        'gen': gen_EMA_5_8_13,
        'space': space_EMA_5_8_13,
        'default_params': {'p1': 5, 'p2': 8, 'p3': 13},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'Three Fibonacci EMAs 5/8/13 triple alignment'},
    },
    'Ribbon_Filter_EMA': {
        'gen': gen_Ribbon_Filter_EMA,
        'space': space_Ribbon_Filter_EMA,
        'default_params': {'base_p': 5, 'step': 4},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': '6-EMA ribbon: long when price above all 6, short when below all 6'},
    },
    'MTF_Trend_Filter': {
        'gen': gen_MTF_Trend_Filter,
        'space': space_MTF_Trend_Filter,
        'default_params': {'fast_p': 8, 'slow_p': 21, 'htf_mult': 4},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Simulated HTF EMA as trend filter + fast/slow cross for entry'},
    },
    'MTF_RSI_Align': {
        'gen': gen_MTF_RSI_Align,
        'space': space_MTF_RSI_Align,
        'default_params': {'fast_rsi': 7, 'slow_rsi': 21},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Both fast and simulated-HTF RSI above/below 50 for direction'},
    },
    'MTF_MACD_Confirm': {
        'gen': gen_MTF_MACD_Confirm,
        'space': space_MTF_MACD_Confirm,
        'default_params': {'fast_p': 7, 'slow_p': 34, 'sig_p': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Fast MACD cross for entry + slow MACD zero-line for trend filter'},
    },
    'MTF_SuperTrend': {
        'gen': gen_MTF_SuperTrend,
        'space': space_MTF_SuperTrend,
        'default_params': {'fast_p': 7, 'fast_mult': 2.0, 'slow_p': 21, 'slow_mult': 3.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Two SuperTrends: fast entry + slow trend filter both aligned'},
    },
    'MTF_BB_Squeeze': {
        'gen': gen_MTF_BB_Squeeze,
        'space': space_MTF_BB_Squeeze,
        'default_params': {'fast_p': 20, 'slow_p': 60, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'BB squeeze release + simulated-HTF BB direction filter'},
    },
    'EMA_Pullback': {
        'gen': gen_EMA_Pullback,
        'space': space_EMA_Pullback,
        'default_params': {'fast_p': 8, 'slow_p': 34},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'EMA pullback system: uptrend + price touches slow EMA + bounces'},
    },
    'EMA_Bounce_RSI': {
        'gen': gen_EMA_Bounce_RSI,
        'space': space_EMA_Bounce_RSI,
        'default_params': {'ema_p': 21, 'rsi_p': 14, 'touch_pct': 0.002},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'EMA touch + RSI momentum confirmation bounce entry'},
    },
    'EMA_Retest': {
        'gen': gen_EMA_Retest,
        'space': space_EMA_Retest,
        'default_params': {'fast_p': 8, 'slow_p': 34, 'zone_pct': 0.005},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Post-cross retest of slow EMA as secondary entry point'},
    },
    'ADX_EMA_System': {
        'gen': gen_ADX_EMA_System,
        'space': space_ADX_EMA_System,
        'default_params': {'adx_p': 14, 'fast_p': 8, 'slow_p': 34, 'adx_thresh': 25},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 4000,
                 'description': 'ADX trend strength gate + EMA cross directional signal'},
    },
    'ADX_SuperTrend': {
        'gen': gen_ADX_SuperTrend,
        'space': space_ADX_SuperTrend,
        'default_params': {'adx_p': 14, 'adx_thresh': 20, 'st_p': 10, 'st_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'ADX confirms trending market, SuperTrend gives direction'},
    },
    'DI_Cross_EMA': {
        'gen': gen_DI_Cross_EMA,
        'space': space_DI_Cross_EMA,
        'default_params': {'adx_p': 14, 'ema_p': 100},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'DI+ / DI- directional cross filtered by long-term EMA trend'},
    },
    'ADX_RSI_EMA': {
        'gen': gen_ADX_RSI_EMA,
        'space': space_ADX_RSI_EMA,
        'default_params': {'adx_p': 14, 'rsi_p': 14, 'ema_p': 50, 'adx_thresh': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Triple confluence: ADX trending + RSI direction + price vs EMA'},
    },
    'NNFX_EMA': {
        'gen': gen_NNFX_EMA,
        'space': space_NNFX_EMA,
        'default_params': {'baseline_p': 100, 'fast_p': 20, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'No Nonsense Forex: baseline EMA + fast/slow + ATR + volume'},
    },
    'Trend_Following_Basic': {
        'gen': gen_Trend_Following_Basic,
        'space': space_Trend_Following_Basic,
        'default_params': {'ema_p': 50},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 4000,
                 'description': 'Simple trend following: price crosses EMA = signal'},
    },
    'Dual_Momentum_EMA': {
        'gen': gen_Dual_Momentum_EMA,
        'space': space_Dual_Momentum_EMA,
        'default_params': {'fast_p': 10, 'slow_p': 40},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Short + long-term EMA momentum both positive = trend confirmed'},
    },
    'Breakout_EMA_Confirm': {
        'gen': gen_Breakout_EMA_Confirm,
        'space': space_Breakout_EMA_Confirm,
        'default_params': {'break_p': 20, 'ema_p': 50},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'N-bar high/low breakout confirmed by EMA trend direction'},
    },
    'EMA_Slope_Strategy': {
        'gen': gen_EMA_Slope_Strategy,
        'space': space_EMA_Slope_Strategy,
        'default_params': {'ema_p': 21, 'slope_p': 5, 'threshold': 0.001},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'EMA slope normalized rate-of-change exceeds threshold = signal'},
    },
    'Inside_EMA_Break': {
        'gen': gen_Inside_EMA_Break,
        'space': space_Inside_EMA_Break,
        'default_params': {'ema_p': 50},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Inside bar breakout filtered by EMA trend direction'},
    },
}
