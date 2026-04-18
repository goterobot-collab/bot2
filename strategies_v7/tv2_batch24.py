#!/usr/bin/env python3
"""TV2 BATCH 24 — 30 estrategias RSI Advanced Variants 2026-04-01

  RSI_OB_OS           — Classic RSI OB/OS: long RSI<30, short RSI>70 (v4, ~6000L)
  RSI_Midline         — RSI midline cross above/below 50 (v4, ~4000L)
  RSI_Trend           — RSI regime + EMA pullback entry (v5, ~3000L)
  RSI_Two_Level       — Two-level RSI: enter first level, exit second (v5, ~2500L)
  RSI_Smooth          — EMA-smoothed RSI cross of 50 (v4, ~2500L)
  RSI_Bull_Div        — Bullish divergence: price LL + RSI HL (v5, ~4000L)
  RSI_Bear_Div        — Bearish divergence: price HH + RSI LH (v5, ~4000L)
  RSI_Hidden_Bull     — Hidden bull div: price HL + RSI LL continuation (v5, ~2500L)
  RSI_Hidden_Bear     — Hidden bear div: price LH + RSI HH continuation (v5, ~2500L)
  RSI_Div_MA          — RSI divergence + MA trend filter (v5, ~2000L)
  RSI2_Strategy       — Connors 2-period RSI mean reversion above 200 SMA (v4, ~3500L)
  RSI_Connors         — Connors RSI: avg(RSI3, StreakRSI, PercentRank) (v5, ~2500L)
  RSI_MA_Band         — RSI + Bollinger Bands applied to RSI series (v4, ~2000L)
  RSI_Percentile      — RSI percentile rank vs N bars: long when rank<20 (v5, ~1800L)
  RSI_MTF_Align       — Simulated MTF: both RSI(fast) and RSI(slow) aligned (v5, ~3000L)
  RSI_Dual_Cross      — Fast RSI crosses slow RSI (v4, ~2000L)
  RSI_Triple          — Three RSIs all aligned above/below 50 (v5, ~1800L)
  RSI_SuperTrend      — RSI + SuperTrend direction (v5, ~2500L)
  RSI_VWAP            — RSI + VWAP: OB/OS only when price vs VWAP aligns (v5, ~2000L)
  RSI_Stoch_Combo     — RSI + Stoch K/D double oversold entry (v4, ~2500L)
  RSI_Candle_Pattern  — RSI oversold + bullish engulfing candle (v5, ~1500L)
  RSI_Volume          — RSI OB/OS + volume confirmation (v4, ~2000L)
  RSI_ATR_Adapt       — Adaptive RSI: high volatility shortens period (v5, ~1500L)
  RSI_Momentum        — RSI rising N bars from below 50 (v4, ~1800L)
  RSI_V_Shape         — V-shape RSI recovery: dip then cross back (v5, ~1500L)
  RSI_Failure_Swing   — RSI failure swing: peak/trough test failure (v4, ~2000L)
  RSI_Range           — RSI range trading: OB/OS zone entry and exit levels (v5, ~1500L)
  RSI_Fib_Level       — RSI Fibonacci levels 38.2/61.8 as OB/OS (v5, ~2000L)
  RSI_Swing           — RSI pivot low in OS / pivot high in OB (v4, ~2000L)
  RSI_Channel         — RSI rolling channel as dynamic OB/OS (v5, ~1500L)
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


def _bb(s, p, mult):
    mid = _sma(s, p)
    std = s.rolling(int(p), min_periods=1).std().fillna(0)
    return mid + mult * std, mid, mid - mult * std


# ── 1. RSI_OB_OS ─────────────────────────────────────────────────────────────

def gen_RSI_OB_OS(df, rsi_p=14, ob=70, os=30, **kw):
    close  = df['close']
    rsi    = _rsi(close, int(rsi_p))
    sig    = pd.Series(0, index=df.index)
    # Long when RSI below oversold; short when above overbought
    sig[rsi < float(os)] =  1
    sig[rsi > float(ob)] = -1
    return sig


def space_RSI_OB_OS(trial):
    return {
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
        'ob':    trial.suggest_float('ob', 65.0, 80.0),
        'os':    trial.suggest_float('os', 20.0, 35.0),
    }


# ── 2. RSI_Midline ────────────────────────────────────────────────────────────

def gen_RSI_Midline(df, rsi_p=14, **kw):
    close  = df['close']
    rsi    = _rsi(close, int(rsi_p))
    prev   = rsi.shift(1)
    sig    = pd.Series(0, index=df.index)
    # Cross above 50 = long; cross below 50 = short
    sig[(rsi > 50) & (prev <= 50)] =  1
    sig[(rsi < 50) & (prev >= 50)] = -1
    return sig


def space_RSI_Midline(trial):
    return {
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 3. RSI_Trend ─────────────────────────────────────────────────────────────

def gen_RSI_Trend(df, rsi_p=14, ema_p=40, **kw):
    close  = df['close']
    rsi    = _rsi(close, int(rsi_p))
    ema    = _ema(close, int(ema_p))
    sig    = pd.Series(0, index=df.index)
    # Bull regime: RSI > 50; entry when price pulls back to EMA
    bull   = rsi > 50
    bear   = rsi < 50
    prev_c = close.shift(1)
    # Long: bull regime, price crosses EMA from below
    sig[bull & (close > ema) & (prev_c <= ema)] =  1
    # Short: bear regime, price crosses EMA from above
    sig[bear & (close < ema) & (prev_c >= ema)] = -1
    return sig


def space_RSI_Trend(trial):
    return {
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
        'ema_p': trial.suggest_int('ema_p', 20, 60),
    }


# ── 4. RSI_Two_Level ──────────────────────────────────────────────────────────

def gen_RSI_Two_Level(df, rsi_p=14, level1_os=40, level1_ob=60, **kw):
    close  = df['close']
    rsi    = _rsi(close, int(rsi_p))
    prev   = rsi.shift(1)
    sig    = pd.Series(0, index=df.index)
    # Long: RSI crosses above first oversold level (40)
    sig[(rsi > float(level1_os)) & (prev <= float(level1_os))] =  1
    # Short: RSI crosses below first overbought level (60)
    sig[(rsi < float(level1_ob)) & (prev >= float(level1_ob))] = -1
    return sig


def space_RSI_Two_Level(trial):
    return {
        'rsi_p':     trial.suggest_int('rsi_p', 7, 21),
        'level1_os': trial.suggest_float('level1_os', 35.0, 45.0),
        'level1_ob': trial.suggest_float('level1_ob', 55.0, 65.0),
    }


# ── 5. RSI_Smooth ─────────────────────────────────────────────────────────────

def gen_RSI_Smooth(df, rsi_p=14, smooth_p=5, **kw):
    close  = df['close']
    rsi    = _rsi(close, int(rsi_p))
    srsi   = _ema(rsi, int(smooth_p))
    prev   = srsi.shift(1)
    sig    = pd.Series(0, index=df.index)
    # Smoothed RSI cross of 50
    sig[(srsi > 50) & (prev <= 50)] =  1
    sig[(srsi < 50) & (prev >= 50)] = -1
    return sig


def space_RSI_Smooth(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'smooth_p': trial.suggest_int('smooth_p', 3, 9),
    }


# ── 6. RSI_Bull_Div ───────────────────────────────────────────────────────────

def gen_RSI_Bull_Div(df, rsi_p=14, lookback=10, pivot_p=3, **kw):
    close     = df['close']
    rsi       = _rsi(close, int(rsi_p))
    lb        = int(lookback)
    pv        = int(pivot_p)
    lo        = close.values
    ri        = rsi.values
    n         = len(lo)
    sig       = np.zeros(n, dtype=float)

    for i in range(pv + lb, n - pv):
        # Current pivot low in price
        if lo[i] != min(lo[i - pv: i + pv + 1]):
            continue
        # Find prior pivot low in lookback window
        prev_idx = -1
        for j in range(i - pv - 1, max(i - lb - pv, pv) - 1, -1):
            if lo[j] == min(lo[j - pv: j + pv + 1]):
                prev_idx = j
                break
        if prev_idx < 0:
            continue
        # Bullish divergence: price lower low, RSI higher low
        if lo[i] < lo[prev_idx] and ri[i] > ri[prev_idx]:
            sig[i + pv] = 1   # confirm after pivot forms

    return pd.Series(sig, index=df.index)


def space_RSI_Bull_Div(trial):
    return {
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
        'pivot_p':  trial.suggest_int('pivot_p', 3, 8),
    }


# ── 7. RSI_Bear_Div ───────────────────────────────────────────────────────────

def gen_RSI_Bear_Div(df, rsi_p=14, lookback=10, pivot_p=3, **kw):
    close     = df['close']
    rsi       = _rsi(close, int(rsi_p))
    lb        = int(lookback)
    pv        = int(pivot_p)
    hi        = df['high'].values
    ri        = rsi.values
    n         = len(hi)
    sig       = np.zeros(n, dtype=float)

    for i in range(pv + lb, n - pv):
        # Current pivot high in price
        if hi[i] != max(hi[i - pv: i + pv + 1]):
            continue
        # Find prior pivot high in lookback window
        prev_idx = -1
        for j in range(i - pv - 1, max(i - lb - pv, pv) - 1, -1):
            if hi[j] == max(hi[j - pv: j + pv + 1]):
                prev_idx = j
                break
        if prev_idx < 0:
            continue
        # Bearish divergence: price higher high, RSI lower high
        if hi[i] > hi[prev_idx] and ri[i] < ri[prev_idx]:
            sig[i + pv] = -1

    return pd.Series(sig, index=df.index)


def space_RSI_Bear_Div(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
        'pivot_p':  trial.suggest_int('pivot_p', 3, 8),
    }


# ── 8. RSI_Hidden_Bull ────────────────────────────────────────────────────────

def gen_RSI_Hidden_Bull(df, rsi_p=14, lookback=10, **kw):
    close     = df['close']
    rsi       = _rsi(close, int(rsi_p))
    lb        = int(lookback)
    lo        = close.values
    ri        = rsi.values
    n         = len(lo)
    sig       = np.zeros(n, dtype=float)
    pv        = 3  # fixed small pivot window

    for i in range(pv + lb, n - pv):
        if lo[i] != min(lo[i - pv: i + pv + 1]):
            continue
        prev_idx = -1
        for j in range(i - pv - 1, max(i - lb - pv, pv) - 1, -1):
            if lo[j] == min(lo[j - pv: j + pv + 1]):
                prev_idx = j
                break
        if prev_idx < 0:
            continue
        # Hidden bull: price higher low (HL) + RSI lower low (LL) = continuation long
        if lo[i] > lo[prev_idx] and ri[i] < ri[prev_idx]:
            sig[i + pv] = 1

    return pd.Series(sig, index=df.index)


def space_RSI_Hidden_Bull(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 9. RSI_Hidden_Bear ────────────────────────────────────────────────────────

def gen_RSI_Hidden_Bear(df, rsi_p=14, lookback=10, **kw):
    close     = df['close']
    rsi       = _rsi(close, int(rsi_p))
    lb        = int(lookback)
    hi        = df['high'].values
    ri        = rsi.values
    n         = len(hi)
    sig       = np.zeros(n, dtype=float)
    pv        = 3

    for i in range(pv + lb, n - pv):
        if hi[i] != max(hi[i - pv: i + pv + 1]):
            continue
        prev_idx = -1
        for j in range(i - pv - 1, max(i - lb - pv, pv) - 1, -1):
            if hi[j] == max(hi[j - pv: j + pv + 1]):
                prev_idx = j
                break
        if prev_idx < 0:
            continue
        # Hidden bear: price lower high (LH) + RSI higher high (HH) = continuation short
        if hi[i] < hi[prev_idx] and ri[i] > ri[prev_idx]:
            sig[i + pv] = -1

    return pd.Series(sig, index=df.index)


def space_RSI_Hidden_Bear(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 10. RSI_Div_MA ────────────────────────────────────────────────────────────

def gen_RSI_Div_MA(df, rsi_p=14, lookback=10, ma_p=100, **kw):
    close     = df['close']
    rsi       = _rsi(close, int(rsi_p))
    ma        = _sma(close, int(ma_p))
    lb        = int(lookback)
    lo        = close.values
    ri        = rsi.values
    cl        = close.values
    ma_v      = ma.values
    n         = len(lo)
    sig       = np.zeros(n, dtype=float)
    pv        = 3

    for i in range(pv + lb, n - pv):
        if lo[i] != min(lo[i - pv: i + pv + 1]):
            continue
        prev_idx = -1
        for j in range(i - pv - 1, max(i - lb - pv, pv) - 1, -1):
            if lo[j] == min(lo[j - pv: j + pv + 1]):
                prev_idx = j
                break
        if prev_idx < 0:
            continue
        idx = i + pv
        if idx >= n:
            continue
        # Bull div + above MA = long
        if lo[i] < lo[prev_idx] and ri[i] > ri[prev_idx] and cl[idx] > ma_v[idx]:
            sig[idx] = 1

    # Bear side: price HH + RSI LH + price below MA = short
    hi = df['high'].values
    for i in range(pv + lb, n - pv):
        if hi[i] != max(hi[i - pv: i + pv + 1]):
            continue
        prev_idx = -1
        for j in range(i - pv - 1, max(i - lb - pv, pv) - 1, -1):
            if hi[j] == max(hi[j - pv: j + pv + 1]):
                prev_idx = j
                break
        if prev_idx < 0:
            continue
        idx = i + pv
        if idx >= n:
            continue
        if hi[i] > hi[prev_idx] and ri[i] < ri[prev_idx] and cl[idx] < ma_v[idx]:
            sig[idx] = -1

    return pd.Series(sig, index=df.index)


def space_RSI_Div_MA(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
        'ma_p':     trial.suggest_int('ma_p', 50, 200),
    }


# ── 11. RSI2_Strategy ─────────────────────────────────────────────────────────

def gen_RSI2_Strategy(df, rsi_p=2, ma_p=200, ob=90, os=10, **kw):
    close  = df['close']
    rsi    = _rsi(close, int(rsi_p))
    ma     = _sma(close, int(ma_p))
    sig    = pd.Series(0, index=df.index)
    # Long: RSI(2) < os AND price above 200 SMA
    sig[(rsi < float(os)) & (close > ma)] =  1
    # Short: RSI(2) > ob AND price below 200 SMA
    sig[(rsi > float(ob)) & (close < ma)] = -1
    return sig


def space_RSI2_Strategy(trial):
    return {
        'rsi_p': trial.suggest_int('rsi_p', 2, 5),
        'ma_p':  trial.suggest_int('ma_p', 150, 250),
        'ob':    trial.suggest_float('ob', 85.0, 95.0),
        'os':    trial.suggest_float('os', 5.0, 15.0),
    }


# ── 12. RSI_Connors ───────────────────────────────────────────────────────────

def gen_RSI_Connors(df, rsi_p=3, streak_p=2, rank_p=100, os=10, ob=90, **kw):
    close   = df['close']
    rsi3    = _rsi(close, int(rsi_p))

    # Streak: consecutive up/down days
    streak  = pd.Series(0.0, index=df.index)
    c       = close.values
    st      = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            st[i] = max(st[i - 1], 0) + 1
        elif c[i] < c[i - 1]:
            st[i] = min(st[i - 1], 0) - 1
        else:
            st[i] = 0
    streak = pd.Series(st, index=df.index)
    streak_rsi = _rsi(streak, int(streak_p))

    # PercentRank: % of last rank_p RSI values below current RSI
    rp      = int(rank_p)
    ri      = rsi3.values
    n       = len(ri)
    prank   = np.zeros(n)
    for i in range(rp, n):
        window  = ri[i - rp: i]
        prank[i] = np.sum(window < ri[i]) / rp * 100
    prank_s = pd.Series(prank, index=df.index)

    crsi    = (rsi3 + streak_rsi + prank_s) / 3.0
    sig     = pd.Series(0, index=df.index)
    sig[crsi < float(os)] =  1
    sig[crsi > float(ob)] = -1
    return sig


def space_RSI_Connors(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 2, 5),
        'streak_p': trial.suggest_int('streak_p', 2, 4),
        'rank_p':   trial.suggest_int('rank_p', 80, 130),
        'os':       trial.suggest_float('os', 5.0, 20.0),
        'ob':       trial.suggest_float('ob', 80.0, 95.0),
    }


# ── 13. RSI_MA_Band ───────────────────────────────────────────────────────────

def gen_RSI_MA_Band(df, rsi_p=14, rsi_bb_p=20, rsi_bb_mult=2.0, **kw):
    close      = df['close']
    rsi        = _rsi(close, int(rsi_p))
    upper, mid, lower = _bb(rsi, int(rsi_bb_p), float(rsi_bb_mult))
    sig        = pd.Series(0, index=df.index)
    sig[rsi < lower] =  1
    sig[rsi > upper] = -1
    return sig


def space_RSI_MA_Band(trial):
    return {
        'rsi_p':       trial.suggest_int('rsi_p', 7, 21),
        'rsi_bb_p':    trial.suggest_int('rsi_bb_p', 15, 25),
        'rsi_bb_mult': trial.suggest_float('rsi_bb_mult', 1.5, 2.5),
    }


# ── 14. RSI_Percentile ────────────────────────────────────────────────────────

def gen_RSI_Percentile(df, rsi_p=14, rank_p=100, low_thresh=20, high_thresh=80, **kw):
    close   = df['close']
    rsi     = _rsi(close, int(rsi_p))
    rp      = int(rank_p)
    ri      = rsi.values
    n       = len(ri)
    prank   = np.zeros(n)
    for i in range(rp, n):
        window   = ri[i - rp: i]
        prank[i] = np.sum(window < ri[i]) / rp * 100
    prank_s = pd.Series(prank, index=df.index)
    sig     = pd.Series(0, index=df.index)
    sig[prank_s < float(low_thresh)]  =  1
    sig[prank_s > float(high_thresh)] = -1
    return sig


def space_RSI_Percentile(trial):
    return {
        'rsi_p':       trial.suggest_int('rsi_p', 7, 21),
        'rank_p':      trial.suggest_int('rank_p', 50, 200),
        'low_thresh':  trial.suggest_float('low_thresh', 10.0, 25.0),
        'high_thresh': trial.suggest_float('high_thresh', 75.0, 90.0),
    }


# ── 15. RSI_MTF_Align ─────────────────────────────────────────────────────────

def gen_RSI_MTF_Align(df, rsi_fast=14, rsi_slow=42, **kw):
    close      = df['close']
    rsi_f      = _rsi(close, int(rsi_fast))
    rsi_s      = _rsi(close, int(rsi_slow))
    sig        = pd.Series(0, index=df.index)
    sig[(rsi_f > 50) & (rsi_s > 50)] =  1
    sig[(rsi_f < 50) & (rsi_s < 50)] = -1
    return sig


def space_RSI_MTF_Align(trial):
    return {
        'rsi_fast': trial.suggest_int('rsi_fast', 7, 14),
        'rsi_slow': trial.suggest_int('rsi_slow', 30, 60),
    }


# ── 16. RSI_Dual_Cross ────────────────────────────────────────────────────────

def gen_RSI_Dual_Cross(df, fast_p=7, slow_p=21, **kw):
    close   = df['close']
    rf      = _rsi(close, int(fast_p))
    rs      = _rsi(close, int(slow_p))
    prev_f  = rf.shift(1)
    prev_s  = rs.shift(1)
    sig     = pd.Series(0, index=df.index)
    # Fast RSI crosses above slow RSI = long
    sig[(rf > rs) & (prev_f <= prev_s)] =  1
    # Fast RSI crosses below slow RSI = short
    sig[(rf < rs) & (prev_f >= prev_s)] = -1
    return sig


def space_RSI_Dual_Cross(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 3, 8),
        'slow_p': trial.suggest_int('slow_p', 14, 30),
    }


# ── 17. RSI_Triple ────────────────────────────────────────────────────────────

def gen_RSI_Triple(df, p1=7, p2=14, p3=21, **kw):
    close   = df['close']
    r1      = _rsi(close, int(p1))
    r2      = _rsi(close, int(p2))
    r3      = _rsi(close, int(p3))
    sig     = pd.Series(0, index=df.index)
    sig[(r1 > 50) & (r2 > 50) & (r3 > 50)] =  1
    sig[(r1 < 50) & (r2 < 50) & (r3 < 50)] = -1
    return sig


def space_RSI_Triple(trial):
    return {
        'p1': trial.suggest_int('p1', 5, 9),
        'p2': trial.suggest_int('p2', 9, 18),
        'p3': trial.suggest_int('p3', 18, 36),
    }


# ── 18. RSI_SuperTrend ────────────────────────────────────────────────────────

def gen_RSI_SuperTrend(df, rsi_p=14, st_p=10, st_mult=3.0, **kw):
    close   = df['close']
    rsi     = _rsi(close, int(rsi_p))
    atr     = _atr(df, int(st_p))
    src     = (df['high'] + df['low']) / 2.0
    mult    = float(st_mult)

    upper_b = src + mult * atr
    lower_b = src - mult * atr

    n       = len(close)
    trend   = np.zeros(n)       # 1 = bullish, -1 = bearish
    final_u = upper_b.values.copy()
    final_l = lower_b.values.copy()
    cl      = close.values
    ub      = upper_b.values
    lb      = lower_b.values

    for i in range(1, n):
        final_l[i] = lb[i] if lb[i] > final_l[i-1] or cl[i-1] < final_l[i-1] else final_l[i-1]
        final_u[i] = ub[i] if ub[i] < final_u[i-1] or cl[i-1] > final_u[i-1] else final_u[i-1]
        if cl[i] > final_u[i-1]:
            trend[i] = 1
        elif cl[i] < final_l[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    st_bull = pd.Series(trend == 1, index=df.index)
    st_bear = pd.Series(trend == -1, index=df.index)
    sig     = pd.Series(0, index=df.index)
    sig[st_bull & (rsi > 50)] =  1
    sig[st_bear & (rsi < 50)] = -1
    return sig


def space_RSI_SuperTrend(trial):
    return {
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
        'st_p':    trial.suggest_int('st_p', 7, 21),
        'st_mult': trial.suggest_float('st_mult', 2.0, 4.0),
    }


# ── 19. RSI_VWAP ──────────────────────────────────────────────────────────────

def gen_RSI_VWAP(df, rsi_p=14, vwap_p=50, **kw):
    close   = df['close']
    volume  = df.get('volume', pd.Series(1.0, index=df.index))
    rsi     = _rsi(close, int(rsi_p))
    vp      = int(vwap_p)
    # Rolling VWAP approximation
    tp      = (df['high'] + df['low'] + close) / 3.0
    vwap    = (tp * volume).rolling(vp, min_periods=1).sum() / volume.rolling(vp, min_periods=1).sum()
    sig     = pd.Series(0, index=df.index)
    sig[(rsi < 40) & (close < vwap)] =  1
    sig[(rsi > 60) & (close > vwap)] = -1
    return sig


def space_RSI_VWAP(trial):
    return {
        'rsi_p':  trial.suggest_int('rsi_p', 7, 21),
        'vwap_p': trial.suggest_int('vwap_p', 20, 100),
    }


# ── 20. RSI_Stoch_Combo ───────────────────────────────────────────────────────

def gen_RSI_Stoch_Combo(df, rsi_p=14, stoch_k=14, stoch_d=3, rsi_os=35, rsi_ob=65, **kw):
    close   = df['close']
    hi      = df['high']
    lo      = df['low']
    rsi     = _rsi(close, int(rsi_p))
    k_p     = int(stoch_k)
    d_p     = int(stoch_d)
    lowest  = lo.rolling(k_p, min_periods=1).min()
    highest = hi.rolling(k_p, min_periods=1).max()
    k       = 100 * (close - lowest) / (highest - lowest + 1e-9)
    d       = _sma(k, d_p)
    sig     = pd.Series(0, index=df.index)
    # Long: RSI oversold AND Stoch K < 20 AND K crossing above D
    sig[(rsi < float(rsi_os)) & (k < 20) & (k > d) & (k.shift(1) <= d.shift(1))] =  1
    # Short: RSI overbought AND Stoch K > 80 AND K crossing below D
    sig[(rsi > float(rsi_ob)) & (k > 80) & (k < d) & (k.shift(1) >= d.shift(1))] = -1
    return sig


def space_RSI_Stoch_Combo(trial):
    return {
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
        'stoch_k': trial.suggest_int('stoch_k', 7, 21),
        'stoch_d': trial.suggest_int('stoch_d', 3, 7),
        'rsi_os':  trial.suggest_float('rsi_os', 25.0, 40.0),
        'rsi_ob':  trial.suggest_float('rsi_ob', 60.0, 75.0),
    }


# ── 21. RSI_Candle_Pattern ────────────────────────────────────────────────────

def gen_RSI_Candle_Pattern(df, rsi_p=14, os_level=30, **kw):
    close   = df['close']
    open_   = df['open']
    rsi     = _rsi(close, int(rsi_p))
    prev_c  = close.shift(1)
    prev_o  = open_.shift(1)
    # Bullish engulfing: current bar bullish AND opens below prev close AND closes above prev open
    bull_eng = (close > open_) & (open_ < prev_c) & (close > prev_o)
    # Bearish engulfing: current bar bearish AND opens above prev close AND closes below prev open
    bear_eng = (close < open_) & (open_ > prev_c) & (close < prev_o)
    sig     = pd.Series(0, index=df.index)
    sig[(rsi < float(os_level)) & bull_eng] =  1
    sig[(rsi > float(100 - os_level)) & bear_eng] = -1
    return sig


def space_RSI_Candle_Pattern(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'os_level': trial.suggest_float('os_level', 20.0, 35.0),
    }


# ── 22. RSI_Volume ────────────────────────────────────────────────────────────

def gen_RSI_Volume(df, rsi_p=14, vol_p=30, vol_mult=1.5, **kw):
    close   = df['close']
    volume  = df.get('volume', pd.Series(1.0, index=df.index))
    rsi     = _rsi(close, int(rsi_p))
    vol_avg = _sma(volume, int(vol_p))
    high_vol = volume > vol_avg * float(vol_mult)
    sig     = pd.Series(0, index=df.index)
    sig[(rsi < 30) & high_vol] =  1
    sig[(rsi > 70) & high_vol] = -1
    return sig


def space_RSI_Volume(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'vol_p':    trial.suggest_int('vol_p', 20, 50),
        'vol_mult': trial.suggest_float('vol_mult', 1.2, 3.0),
    }


# ── 23. RSI_ATR_Adapt ─────────────────────────────────────────────────────────

def gen_RSI_ATR_Adapt(df, base_rsi=14, atr_p=14, os_level=30, ob_level=70, **kw):
    close   = df['close']
    atr     = _atr(df, int(atr_p))
    atr_med = atr.rolling(50, min_periods=1).median()
    # High volatility = shorter RSI period (more reactive)
    # Low volatility = standard period
    base    = int(base_rsi)
    adapt_p = np.where(atr.values > atr_med.values, max(base // 2, 3), base)
    # Compute RSI with short and long period, blend
    rsi_short = _rsi(close, max(base // 2, 3))
    rsi_long  = _rsi(close, base)
    high_vol  = (atr > atr_med).values
    rsi_val   = pd.Series(
        np.where(high_vol, rsi_short.values, rsi_long.values),
        index=df.index
    )
    sig       = pd.Series(0, index=df.index)
    sig[rsi_val < float(os_level)] =  1
    sig[rsi_val > float(ob_level)] = -1
    return sig


def space_RSI_ATR_Adapt(trial):
    return {
        'base_rsi': trial.suggest_int('base_rsi', 7, 21),
        'atr_p':    trial.suggest_int('atr_p', 10, 20),
        'os_level': trial.suggest_float('os_level', 20.0, 35.0),
        'ob_level': trial.suggest_float('ob_level', 65.0, 80.0),
    }


# ── 24. RSI_Momentum ──────────────────────────────────────────────────────────

def gen_RSI_Momentum(df, rsi_p=14, consec=3, **kw):
    close   = df['close']
    rsi     = _rsi(close, int(rsi_p))
    c       = int(consec)
    rsi_v   = rsi.values
    n       = len(rsi_v)
    sig     = np.zeros(n)

    for i in range(c, n):
        # RSI rising for consec bars from below 50
        rising = all(rsi_v[i - k] > rsi_v[i - k - 1] for k in range(c))
        if rising and rsi_v[i - c] < 50:
            sig[i] = 1
        # RSI falling for consec bars from above 50
        falling = all(rsi_v[i - k] < rsi_v[i - k - 1] for k in range(c))
        if falling and rsi_v[i - c] > 50:
            sig[i] = -1

    return pd.Series(sig, index=df.index)


def space_RSI_Momentum(trial):
    return {
        'rsi_p':  trial.suggest_int('rsi_p', 7, 21),
        'consec': trial.suggest_int('consec', 2, 5),
    }


# ── 25. RSI_V_Shape ───────────────────────────────────────────────────────────

def gen_RSI_V_Shape(df, rsi_p=14, dip_level=30, recover_level=40, **kw):
    close   = df['close']
    rsi     = _rsi(close, int(rsi_p))
    dip     = float(dip_level)
    rec     = float(recover_level)
    rsi_v   = rsi.values
    n       = len(rsi_v)
    sig     = np.zeros(n)

    # Track if RSI was below dip within last 10 bars
    was_dipped = np.zeros(n, dtype=bool)
    dip_window = 10
    for i in range(1, n):
        start = max(0, i - dip_window)
        was_dipped[i] = any(rsi_v[start:i] < dip)

    for i in range(1, n):
        # V-shape long: RSI was below dip and now crosses above recover level
        if was_dipped[i] and rsi_v[i] > rec and rsi_v[i-1] <= rec:
            sig[i] = 1
        # Inverted V short: RSI was above (100-dip) and now crosses below (100-rec)
        ob_level = 100 - dip
        ob_rec   = 100 - rec
        if rsi_v[i] < ob_rec and rsi_v[i-1] >= ob_rec:
            # Check if RSI was above ob_level in last dip_window bars
            start = max(0, i - dip_window)
            if any(rsi_v[start:i] > ob_level):
                sig[i] = -1

    return pd.Series(sig, index=df.index)


def space_RSI_V_Shape(trial):
    return {
        'rsi_p':         trial.suggest_int('rsi_p', 7, 21),
        'dip_level':     trial.suggest_float('dip_level', 20.0, 35.0),
        'recover_level': trial.suggest_float('recover_level', 35.0, 50.0),
    }


# ── 26. RSI_Failure_Swing ─────────────────────────────────────────────────────

def gen_RSI_Failure_Swing(df, rsi_p=14, ob=70, os=30, **kw):
    close   = df['close']
    rsi     = _rsi(close, int(rsi_p))
    ob_f    = float(ob)
    os_f    = float(os)
    rsi_v   = rsi.values
    n       = len(rsi_v)
    sig     = np.zeros(n)
    window  = 20  # lookback for failure swing detection

    for i in range(window, n):
        window_rsi = rsi_v[i - window: i + 1]
        # Bearish failure swing:
        # 1) RSI exceeds ob
        # 2) RSI pulls back below ob
        # 3) RSI rises but fails to exceed prior peak (< ob)
        # 4) RSI breaks prior pullback low → short
        peaks = []
        troughs = []
        for k in range(1, len(window_rsi) - 1):
            if window_rsi[k] > window_rsi[k-1] and window_rsi[k] > window_rsi[k+1]:
                peaks.append((k, window_rsi[k]))
            if window_rsi[k] < window_rsi[k-1] and window_rsi[k] < window_rsi[k+1]:
                troughs.append((k, window_rsi[k]))

        # Bearish: find two peaks where first > ob and second < ob
        if len(peaks) >= 2:
            p1_idx, p1_val = peaks[-2]
            p2_idx, p2_val = peaks[-1]
            if p1_val >= ob_f and p2_val < ob_f and p2_val < p1_val:
                # Find trough between peaks
                between = [t for t in troughs if p1_idx < t[0] < p2_idx]
                if between:
                    trough_val = between[-1][1]
                    if window_rsi[-1] < trough_val:
                        sig[i] = -1

        # Bullish failure swing: symmetric
        if len(peaks) >= 2 and len(troughs) >= 2:
            t1_idx, t1_val = troughs[-2]
            t2_idx, t2_val = troughs[-1]
            if t1_val <= os_f and t2_val > os_f and t2_val > t1_val:
                between = [p for p in peaks if t1_idx < p[0] < t2_idx]
                if between:
                    peak_val = between[-1][1]
                    if window_rsi[-1] > peak_val:
                        sig[i] = 1

    return pd.Series(sig, index=df.index)


def space_RSI_Failure_Swing(trial):
    return {
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
        'ob':    trial.suggest_float('ob', 65.0, 75.0),
        'os':    trial.suggest_float('os', 25.0, 35.0),
    }


# ── 27. RSI_Range ─────────────────────────────────────────────────────────────

def gen_RSI_Range(df, rsi_p=14, ob_entry=75, ob_exit=65, os_entry=25, os_exit=35, **kw):
    close   = df['close']
    rsi     = _rsi(close, int(rsi_p))
    rsi_v   = rsi.values
    n       = len(rsi_v)
    sig     = np.zeros(n)
    in_long  = False
    in_short = False

    for i in range(1, n):
        r = rsi_v[i]
        # Enter long when RSI drops into oversold zone
        if not in_long and r < float(os_entry):
            sig[i] = 1
            in_long = True
        # Exit long when RSI recovers out of zone
        elif in_long and r > float(os_exit):
            in_long = False
        # Enter short when RSI enters overbought zone
        if not in_short and r > float(ob_entry):
            sig[i] = -1
            in_short = True
        # Exit short when RSI comes back down
        elif in_short and r < float(ob_exit):
            in_short = False

    return pd.Series(sig, index=df.index)


def space_RSI_Range(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'ob_entry': trial.suggest_float('ob_entry', 70.0, 80.0),
        'ob_exit':  trial.suggest_float('ob_exit', 60.0, 70.0),
        'os_entry': trial.suggest_float('os_entry', 20.0, 30.0),
        'os_exit':  trial.suggest_float('os_exit', 30.0, 40.0),
    }


# ── 28. RSI_Fib_Level ─────────────────────────────────────────────────────────

def gen_RSI_Fib_Level(df, rsi_p=14, fib_lower=38.2, fib_upper=61.8, **kw):
    close   = df['close']
    rsi     = _rsi(close, int(rsi_p))
    prev    = rsi.shift(1)
    fl      = float(fib_lower)
    fu      = float(fib_upper)
    sig     = pd.Series(0, index=df.index)
    # Long: RSI bounces off 38.2 (crosses back above)
    sig[(rsi > fl) & (prev <= fl)] =  1
    # Short: RSI rejects 61.8 (crosses back below)
    sig[(rsi < fu) & (prev >= fu)] = -1
    return sig


def space_RSI_Fib_Level(trial):
    return {
        'rsi_p':     trial.suggest_int('rsi_p', 7, 21),
        'fib_lower': trial.suggest_float('fib_lower', 35.0, 45.0),
        'fib_upper': trial.suggest_float('fib_upper', 55.0, 65.0),
    }


# ── 29. RSI_Swing ─────────────────────────────────────────────────────────────

def gen_RSI_Swing(df, rsi_p=14, pivot_p=3, os=30, ob=70, **kw):
    close   = df['close']
    rsi     = _rsi(close, int(rsi_p))
    pv      = int(pivot_p)
    os_f    = float(os)
    ob_f    = float(ob)
    rsi_v   = rsi.values
    n       = len(rsi_v)
    sig     = np.zeros(n)

    for i in range(pv, n - pv):
        window = rsi_v[i - pv: i + pv + 1]
        # Pivot low in RSI within oversold zone
        if rsi_v[i] == min(window) and rsi_v[i] < os_f:
            sig[i + pv] = 1   # confirm after pivot forms
        # Pivot high in RSI within overbought zone
        if rsi_v[i] == max(window) and rsi_v[i] > ob_f:
            sig[i + pv] = -1

    return pd.Series(sig, index=df.index)


def space_RSI_Swing(trial):
    return {
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
        'pivot_p': trial.suggest_int('pivot_p', 2, 5),
        'os':      trial.suggest_float('os', 25.0, 35.0),
        'ob':      trial.suggest_float('ob', 65.0, 75.0),
    }


# ── 30. RSI_Channel ───────────────────────────────────────────────────────────

def gen_RSI_Channel(df, rsi_p=14, channel_p=30, **kw):
    close     = df['close']
    rsi       = _rsi(close, int(rsi_p))
    cp        = int(channel_p)
    rsi_high  = rsi.rolling(cp, min_periods=1).max()
    rsi_low   = rsi.rolling(cp, min_periods=1).min()
    # Dynamic OB/OS based on rolling channel
    sig       = pd.Series(0, index=df.index)
    sig[rsi <= rsi_low]  =  1
    sig[rsi >= rsi_high] = -1
    return sig


def space_RSI_Channel(trial):
    return {
        'rsi_p':     trial.suggest_int('rsi_p', 7, 21),
        'channel_p': trial.suggest_int('channel_p', 20, 50),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'RSI_OB_OS': {
        'gen': gen_RSI_OB_OS,
        'space': space_RSI_OB_OS,
        'default_params': {'rsi_p': 14, 'ob': 70, 'os': 30},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 6000,
                 'description': 'Classic RSI OB/OS: long when RSI<30, short when RSI>70'},
    },
    'RSI_Midline': {
        'gen': gen_RSI_Midline,
        'space': space_RSI_Midline,
        'default_params': {'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 4000,
                 'description': 'RSI midline cross: long RSI crosses above 50, short crosses below 50'},
    },
    'RSI_Trend': {
        'gen': gen_RSI_Trend,
        'space': space_RSI_Trend,
        'default_params': {'rsi_p': 14, 'ema_p': 40},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'RSI regime filter + EMA pullback entry (long only in bull, short only in bear)'},
    },
    'RSI_Two_Level': {
        'gen': gen_RSI_Two_Level,
        'space': space_RSI_Two_Level,
        'default_params': {'rsi_p': 14, 'level1_os': 40, 'level1_ob': 60},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Two-level RSI: enter on first level cross (40/60)'},
    },
    'RSI_Smooth': {
        'gen': gen_RSI_Smooth,
        'space': space_RSI_Smooth,
        'default_params': {'rsi_p': 14, 'smooth_p': 5},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'EMA-smoothed RSI cross of 50 to reduce noise'},
    },
    'RSI_Bull_Div': {
        'gen': gen_RSI_Bull_Div,
        'space': space_RSI_Bull_Div,
        'default_params': {'rsi_p': 14, 'lookback': 10, 'pivot_p': 3},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Bullish divergence: price lower low + RSI higher low = long signal'},
    },
    'RSI_Bear_Div': {
        'gen': gen_RSI_Bear_Div,
        'space': space_RSI_Bear_Div,
        'default_params': {'rsi_p': 14, 'lookback': 10, 'pivot_p': 3},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Bearish divergence: price higher high + RSI lower high = short signal'},
    },
    'RSI_Hidden_Bull': {
        'gen': gen_RSI_Hidden_Bull,
        'space': space_RSI_Hidden_Bull,
        'default_params': {'rsi_p': 14, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Hidden bull divergence: price higher low + RSI lower low = trend continuation long'},
    },
    'RSI_Hidden_Bear': {
        'gen': gen_RSI_Hidden_Bear,
        'space': space_RSI_Hidden_Bear,
        'default_params': {'rsi_p': 14, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Hidden bear divergence: price lower high + RSI higher high = trend continuation short'},
    },
    'RSI_Div_MA': {
        'gen': gen_RSI_Div_MA,
        'space': space_RSI_Div_MA,
        'default_params': {'rsi_p': 14, 'lookback': 10, 'ma_p': 100},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'RSI divergence + MA trend filter: bull div AND price > MA = long'},
    },
    'RSI2_Strategy': {
        'gen': gen_RSI2_Strategy,
        'space': space_RSI2_Strategy,
        'default_params': {'rsi_p': 2, 'ma_p': 200, 'ob': 90, 'os': 10},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3500,
                 'description': 'Connors 2-period RSI mean reversion above/below 200 SMA'},
    },
    'RSI_Connors': {
        'gen': gen_RSI_Connors,
        'space': space_RSI_Connors,
        'default_params': {'rsi_p': 3, 'streak_p': 2, 'rank_p': 100, 'os': 10, 'ob': 90},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Connors RSI: composite of RSI3 + StreakRSI + PercentRank'},
    },
    'RSI_MA_Band': {
        'gen': gen_RSI_MA_Band,
        'space': space_RSI_MA_Band,
        'default_params': {'rsi_p': 14, 'rsi_bb_p': 20, 'rsi_bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'Bollinger Bands applied to RSI series: long when RSI < BB_lower of RSI'},
    },
    'RSI_Percentile': {
        'gen': gen_RSI_Percentile,
        'space': space_RSI_Percentile,
        'default_params': {'rsi_p': 14, 'rank_p': 100, 'low_thresh': 20, 'high_thresh': 80},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1800,
                 'description': 'RSI percentile rank vs N bars: long when RSI historically very low'},
    },
    'RSI_MTF_Align': {
        'gen': gen_RSI_MTF_Align,
        'space': space_RSI_MTF_Align,
        'default_params': {'rsi_fast': 14, 'rsi_slow': 42},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Simulated MTF RSI alignment: fast and slow RSI both above/below 50'},
    },
    'RSI_Dual_Cross': {
        'gen': gen_RSI_Dual_Cross,
        'space': space_RSI_Dual_Cross,
        'default_params': {'fast_p': 7, 'slow_p': 21},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'Fast RSI crosses slow RSI: RSI(7) crosses above RSI(21) = long'},
    },
    'RSI_Triple': {
        'gen': gen_RSI_Triple,
        'space': space_RSI_Triple,
        'default_params': {'p1': 7, 'p2': 14, 'p3': 21},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1800,
                 'description': 'Three RSIs all aligned: short, mid, long all above/below 50'},
    },
    'RSI_SuperTrend': {
        'gen': gen_RSI_SuperTrend,
        'space': space_RSI_SuperTrend,
        'default_params': {'rsi_p': 14, 'st_p': 10, 'st_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'RSI + SuperTrend alignment: ST bullish AND RSI > 50 = long'},
    },
    'RSI_VWAP': {
        'gen': gen_RSI_VWAP,
        'space': space_RSI_VWAP,
        'default_params': {'rsi_p': 14, 'vwap_p': 50},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'RSI + rolling VWAP: long RSI<40 AND close<VWAP; short RSI>60 AND close>VWAP'},
    },
    'RSI_Stoch_Combo': {
        'gen': gen_RSI_Stoch_Combo,
        'space': space_RSI_Stoch_Combo,
        'default_params': {'rsi_p': 14, 'stoch_k': 14, 'stoch_d': 3, 'rsi_os': 35, 'rsi_ob': 65},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'RSI + Stochastic double confirmation: both indicators oversold with K/D cross'},
    },
    'RSI_Candle_Pattern': {
        'gen': gen_RSI_Candle_Pattern,
        'space': space_RSI_Candle_Pattern,
        'default_params': {'rsi_p': 14, 'os_level': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'RSI oversold + bullish engulfing candle pattern confirmation'},
    },
    'RSI_Volume': {
        'gen': gen_RSI_Volume,
        'space': space_RSI_Volume,
        'default_params': {'rsi_p': 14, 'vol_p': 30, 'vol_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'RSI OB/OS + volume surge confirmation: volume > avg * multiplier'},
    },
    'RSI_ATR_Adapt': {
        'gen': gen_RSI_ATR_Adapt,
        'space': space_RSI_ATR_Adapt,
        'default_params': {'base_rsi': 14, 'atr_p': 14, 'os_level': 30, 'ob_level': 70},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'Adaptive RSI: high volatility → shorter period, low volatility → standard period'},
    },
    'RSI_Momentum': {
        'gen': gen_RSI_Momentum,
        'space': space_RSI_Momentum,
        'default_params': {'rsi_p': 14, 'consec': 3},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1800,
                 'description': 'RSI momentum: rising N bars from below 50 = long; falling from above 50 = short'},
    },
    'RSI_V_Shape': {
        'gen': gen_RSI_V_Shape,
        'space': space_RSI_V_Shape,
        'default_params': {'rsi_p': 14, 'dip_level': 30, 'recover_level': 40},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'V-shape RSI recovery: RSI dips below threshold then crosses back above recover level'},
    },
    'RSI_Failure_Swing': {
        'gen': gen_RSI_Failure_Swing,
        'space': space_RSI_Failure_Swing,
        'default_params': {'rsi_p': 14, 'ob': 70, 'os': 30},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'RSI failure swing: second peak fails to reach OB then breaks prior trough = short'},
    },
    'RSI_Range': {
        'gen': gen_RSI_Range,
        'space': space_RSI_Range,
        'default_params': {'rsi_p': 14, 'ob_entry': 75, 'ob_exit': 65, 'os_entry': 25, 'os_exit': 35},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'RSI range trading with entry and exit zone levels for OB and OS'},
    },
    'RSI_Fib_Level': {
        'gen': gen_RSI_Fib_Level,
        'space': space_RSI_Fib_Level,
        'default_params': {'rsi_p': 14, 'fib_lower': 38.2, 'fib_upper': 61.8},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'RSI Fibonacci levels: 38.2 as oversold support, 61.8 as overbought resistance'},
    },
    'RSI_Swing': {
        'gen': gen_RSI_Swing,
        'space': space_RSI_Swing,
        'default_params': {'rsi_p': 14, 'pivot_p': 3, 'os': 30, 'ob': 70},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'RSI pivot low in oversold = long; RSI pivot high in overbought = short'},
    },
    'RSI_Channel': {
        'gen': gen_RSI_Channel,
        'space': space_RSI_Channel,
        'default_params': {'rsi_p': 14, 'channel_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'RSI rolling channel: dynamic OB/OS from rolling high/low of RSI'},
    },
}
