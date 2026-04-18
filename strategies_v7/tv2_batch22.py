#!/usr/bin/env python3
"""TV2 BATCH 22 — 30 estrategias Moving Average Family 2026-04-01

  Golden_Death_Cross        — 50/200 EMA cross long/short (v4, ~8000L)
  Golden_Cross_RSI          — EMA cross + RSI confirmation (v5, ~3000L)
  Golden_Cross_Vol          — EMA cross + volume surge filter (v4, ~2000L)
  Triple_Golden_Cross       — 3 EMAs aligned bull/bear (v5, ~2500L)
  DEMA_Cross                — Fast DEMA vs slow DEMA cross (v4, ~3000L)
  DEMA_ATR                  — DEMA + ATR bands (v5, ~2000L)
  DEMA_RSI                  — DEMA rising + RSI > 50 (v4, ~1500L)
  DEMA_BB                   — DEMA trend + BB dip buy (v5, ~1200L)
  TEMA_Cross                — Fast TEMA vs slow TEMA (v4, ~2500L)
  TEMA_EMA_Cross            — TEMA vs EMA same period (v5, ~1800L)
  TEMA_RSI                  — TEMA direction + RSI 45-65 (v4, ~1200L)
  TEMA_ATR                  — TEMA rising + ATR expanding (v5, ~1000L)
  HMA_Cross                 — Fast HMA vs slow HMA (v4, ~3000L)
  HMA_ATR_Band              — HMA + ATR channel (v5, ~2000L)
  HMA_RSI_ATR               — HMA + RSI + ATR combined (v4, ~1500L)
  MA_Rainbow_Bull           — 6 EMAs full sequence alignment (v5, ~4000L)
  MA_Rainbow_Cross          — Rainbow fastest crosses in aligned setup (v4, ~2500L)
  MA_Ribbon                 — 8 EMA ribbon fanned out (v5, ~3000L)
  Adaptive_MA_ATR           — MA period adapts to ATR (v5, ~2500L)
  Adaptive_MA_ROC           — MA period adapts to ROC (v5, ~2000L)
  EMA_SMA_Cross             — EMA vs SMA same period (v4, ~3500L)
  WMA_EMA_Cross             — WMA vs EMA cross (v5, ~2000L)
  HMA_EMA_Cross             — HMA vs EMA (v4, ~2000L)
  TEMA_DEMA_Cross           — TEMA vs DEMA cross (v5, ~1500L)
  McGinley_EMA_Cross        — McGinley dynamic vs EMA (v5, ~2000L)
  VAMA_Strategy             — Volume Adjusted MA (v5, ~2000L)
  PWMA_Strategy             — Polynomial Weighted MA (v4, ~1800L)
  Supertrend_MA_Cross       — SuperTrend + MA cross aligned (v4, ~2500L)
  EMA_Cloud                 — EMA cloud price above/below (v5, ~3000L)
  SMA_Bounce                — Price bounces off key SMA (v4, ~2500L)
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


def _wma(s, p):
    p = int(p)
    w = np.arange(1, p + 1, dtype=float)
    return s.rolling(p, min_periods=1).apply(
        lambda x: np.dot(x[-len(w):], w[-len(x):]) / w[-len(x):].sum(), raw=True
    )


def _dema(s, p):
    e = _ema(s, p)
    return 2 * e - _ema(e, p)


def _tema(s, p):
    e  = _ema(s, p)
    e2 = _ema(e, p)
    e3 = _ema(e2, p)
    return 3 * e - 3 * e2 + e3


def _hma(s, p):
    p    = int(p)
    half = max(p // 2, 1)
    sq   = max(int(np.sqrt(p)), 1)
    return _wma(2 * _wma(s, half) - _wma(s, p), sq)


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


# ── 1. Golden_Death_Cross ────────────────────────────────────────────────────

def gen_Golden_Death_Cross(df, fast_p=50, slow_p=200, **kw):
    fast_p = int(fast_p)
    slow_p = int(slow_p)
    fast   = _ema(df['close'], fast_p)
    slow   = _ema(df['close'], slow_p)
    cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_Golden_Death_Cross(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 30, 70),
        'slow_p': trial.suggest_int('slow_p', 150, 250),
    }


# ── 2. Golden_Cross_RSI ──────────────────────────────────────────────────────

def gen_Golden_Cross_RSI(df, fast_p=30, slow_p=150, rsi_p=14, **kw):
    fast_p = int(fast_p)
    slow_p = int(slow_p)
    rsi_p  = int(rsi_p)
    fast   = _ema(df['close'], fast_p)
    slow   = _ema(df['close'], slow_p)
    rsi    = _rsi(df['close'], rsi_p)
    cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & (rsi < 70)] =  1
    sig[cross_down & (rsi > 30)] = -1
    return sig


def space_Golden_Cross_RSI(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 20, 60),
        'slow_p': trial.suggest_int('slow_p', 100, 200),
        'rsi_p':  trial.suggest_int('rsi_p',  7,  21),
    }


# ── 3. Golden_Cross_Vol ──────────────────────────────────────────────────────

def gen_Golden_Cross_Vol(df, fast_p=20, slow_p=100, vol_p=20, vol_mult=1.5, **kw):
    fast_p   = int(fast_p)
    slow_p   = int(slow_p)
    vol_p    = int(vol_p)
    vol_mult = float(vol_mult)
    fast     = _ema(df['close'], fast_p)
    slow     = _ema(df['close'], slow_p)
    vol_sma  = _sma(df['volume'], vol_p)
    high_vol = df['volume'] > vol_sma * vol_mult
    cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & high_vol] =  1
    sig[cross_down & high_vol] = -1
    return sig


def space_Golden_Cross_Vol(trial):
    return {
        'fast_p':   trial.suggest_int('fast_p',   20,  60),
        'slow_p':   trial.suggest_int('slow_p',  100, 200),
        'vol_p':    trial.suggest_int('vol_p',    20,  50),
        'vol_mult': trial.suggest_float('vol_mult', 1.2, 3.0),
    }


# ── 4. Triple_Golden_Cross ───────────────────────────────────────────────────

def gen_Triple_Golden_Cross(df, p1=9, p2=21, p3=100, **kw):
    p1 = int(p1)
    p2 = int(p2)
    p3 = int(p3)
    e1 = _ema(df['close'], p1)
    e2 = _ema(df['close'], p2)
    e3 = _ema(df['close'], p3)
    bull = (e1 > e2) & (e2 > e3)
    bear = (e1 < e2) & (e2 < e3)
    sig = pd.Series(0, index=df.index)
    sig[bull & ~bull.shift(1).fillna(False)] =  1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    # hold while aligned
    for i in range(1, len(sig)):
        if sig.iloc[i] == 0:
            if bull.iloc[i]:
                sig.iloc[i] =  1
            elif bear.iloc[i]:
                sig.iloc[i] = -1
    return sig


def space_Triple_Golden_Cross(trial):
    return {
        'p1': trial.suggest_int('p1',  5,  20),
        'p2': trial.suggest_int('p2', 20,  60),
        'p3': trial.suggest_int('p3', 60, 200),
    }


# ── 5. DEMA_Cross ────────────────────────────────────────────────────────────

def gen_DEMA_Cross(df, fast_p=8, slow_p=34, **kw):
    fast_p = int(fast_p)
    slow_p = int(slow_p)
    d_fast = _dema(df['close'], fast_p)
    d_slow = _dema(df['close'], slow_p)
    cross_up   = (d_fast > d_slow) & (d_fast.shift(1) <= d_slow.shift(1))
    cross_down = (d_fast < d_slow) & (d_fast.shift(1) >= d_slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_DEMA_Cross(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',  5, 20),
        'slow_p': trial.suggest_int('slow_p', 20, 80),
    }


# ── 6. DEMA_ATR ──────────────────────────────────────────────────────────────

def gen_DEMA_ATR(df, dema_p=20, atr_p=14, atr_mult=1.5, **kw):
    dema_p   = int(dema_p)
    atr_p    = int(atr_p)
    atr_mult = float(atr_mult)
    dema     = _dema(df['close'], dema_p)
    atr      = _atr(df, atr_p)
    upper    = dema + atr_mult * atr
    lower    = dema - atr_mult * atr
    long_sig  = df['close'] > upper
    short_sig = df['close'] < lower
    sig = pd.Series(0, index=df.index)
    sig[long_sig]  =  1
    sig[short_sig] = -1
    return sig


def space_DEMA_ATR(trial):
    return {
        'dema_p':   trial.suggest_int('dema_p',   10, 40),
        'atr_p':    trial.suggest_int('atr_p',    10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.0, 2.5),
    }


# ── 7. DEMA_RSI ──────────────────────────────────────────────────────────────

def gen_DEMA_RSI(df, dema_p=20, rsi_p=14, **kw):
    dema_p = int(dema_p)
    rsi_p  = int(rsi_p)
    dema   = _dema(df['close'], dema_p)
    rsi    = _rsi(df['close'], rsi_p)
    dema_rising  = dema > dema.shift(1)
    dema_falling = dema < dema.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[dema_rising  & (rsi > 50)] =  1
    sig[dema_falling & (rsi < 50)] = -1
    return sig


def space_DEMA_RSI(trial):
    return {
        'dema_p': trial.suggest_int('dema_p', 10, 40),
        'rsi_p':  trial.suggest_int('rsi_p',   7, 21),
    }


# ── 8. DEMA_BB ───────────────────────────────────────────────────────────────

def gen_DEMA_BB(df, dema_p=20, bb_p=20, bb_mult=2.0, **kw):
    dema_p   = int(dema_p)
    bb_p     = int(bb_p)
    bb_mult  = float(bb_mult)
    dema     = _dema(df['close'], dema_p)
    bb_mid   = _sma(df['close'], bb_p)
    bb_std   = df['close'].rolling(bb_p, min_periods=1).std(ddof=0).fillna(0)
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std
    # Long: DEMA rising AND close below BB lower (buy dip in uptrend)
    uptrend   = dema > dema.shift(1)
    downtrend = dema < dema.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[uptrend   & (df['close'] < bb_lower)] =  1
    sig[downtrend & (df['close'] > bb_upper)] = -1
    return sig


def space_DEMA_BB(trial):
    return {
        'dema_p':  trial.suggest_int('dema_p',   10, 40),
        'bb_p':    trial.suggest_int('bb_p',      10, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 2.5),
    }


# ── 9. TEMA_Cross ────────────────────────────────────────────────────────────

def gen_TEMA_Cross(df, fast_p=8, slow_p=30, **kw):
    fast_p  = int(fast_p)
    slow_p  = int(slow_p)
    t_fast  = _tema(df['close'], fast_p)
    t_slow  = _tema(df['close'], slow_p)
    cross_up   = (t_fast > t_slow) & (t_fast.shift(1) <= t_slow.shift(1))
    cross_down = (t_fast < t_slow) & (t_fast.shift(1) >= t_slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_TEMA_Cross(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',  5, 15),
        'slow_p': trial.suggest_int('slow_p', 20, 60),
    }


# ── 10. TEMA_EMA_Cross ───────────────────────────────────────────────────────

def gen_TEMA_EMA_Cross(df, p=20, **kw):
    p    = int(p)
    tema = _tema(df['close'], p)
    ema  = _ema(df['close'], p)
    cross_up   = (tema > ema) & (tema.shift(1) <= ema.shift(1))
    cross_down = (tema < ema) & (tema.shift(1) >= ema.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_TEMA_EMA_Cross(trial):
    return {
        'p': trial.suggest_int('p', 10, 40),
    }


# ── 11. TEMA_RSI ─────────────────────────────────────────────────────────────

def gen_TEMA_RSI(df, tema_p=20, rsi_p=14, **kw):
    tema_p = int(tema_p)
    rsi_p  = int(rsi_p)
    tema   = _tema(df['close'], tema_p)
    rsi    = _rsi(df['close'], rsi_p)
    rising  = tema > tema.shift(1)
    falling = tema < tema.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[rising  & (rsi >= 45) & (rsi <= 65)] =  1
    sig[falling & (rsi >= 35) & (rsi <= 55)] = -1
    return sig


def space_TEMA_RSI(trial):
    return {
        'tema_p': trial.suggest_int('tema_p', 10, 40),
        'rsi_p':  trial.suggest_int('rsi_p',   7, 21),
    }


# ── 12. TEMA_ATR ─────────────────────────────────────────────────────────────

def gen_TEMA_ATR(df, tema_p=20, atr_p=14, **kw):
    tema_p = int(tema_p)
    atr_p  = int(atr_p)
    tema   = _tema(df['close'], tema_p)
    atr    = _atr(df, atr_p)
    tema_rising  = tema > tema.shift(1)
    tema_falling = tema < tema.shift(1)
    atr_expand   = atr > atr.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[tema_rising  & atr_expand] =  1
    sig[tema_falling & atr_expand] = -1
    return sig


def space_TEMA_ATR(trial):
    return {
        'tema_p': trial.suggest_int('tema_p', 10, 40),
        'atr_p':  trial.suggest_int('atr_p',  10, 20),
    }


# ── 13. HMA_Cross ────────────────────────────────────────────────────────────

def gen_HMA_Cross(df, fast_p=9, slow_p=36, **kw):
    fast_p  = int(fast_p)
    slow_p  = int(slow_p)
    h_fast  = _hma(df['close'], fast_p)
    h_slow  = _hma(df['close'], slow_p)
    cross_up   = (h_fast > h_slow) & (h_fast.shift(1) <= h_slow.shift(1))
    cross_down = (h_fast < h_slow) & (h_fast.shift(1) >= h_slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_HMA_Cross(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',  5, 20),
        'slow_p': trial.suggest_int('slow_p', 20, 60),
    }


# ── 14. HMA_ATR_Band ─────────────────────────────────────────────────────────

def gen_HMA_ATR_Band(df, hma_p=20, atr_p=14, atr_mult=2.0, **kw):
    hma_p    = int(hma_p)
    atr_p    = int(atr_p)
    atr_mult = float(atr_mult)
    hma      = _hma(df['close'], hma_p)
    atr      = _atr(df, atr_p)
    hma_up   = hma > hma.shift(1)
    hma_dn   = hma < hma.shift(1)
    # Long: HMA up AND close above HMA - atr*mult (not fallen too far below)
    sig = pd.Series(0, index=df.index)
    sig[hma_up & (df['close'] > hma - atr_mult * atr)] =  1
    sig[hma_dn & (df['close'] < hma + atr_mult * atr)] = -1
    return sig


def space_HMA_ATR_Band(trial):
    return {
        'hma_p':    trial.suggest_int('hma_p',    10, 40),
        'atr_p':    trial.suggest_int('atr_p',    10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 3.0),
    }


# ── 15. HMA_RSI_ATR ──────────────────────────────────────────────────────────

def gen_HMA_RSI_ATR(df, hma_p=20, rsi_p=14, atr_p=14, **kw):
    hma_p = int(hma_p)
    rsi_p = int(rsi_p)
    atr_p = int(atr_p)
    hma   = _hma(df['close'], hma_p)
    rsi   = _rsi(df['close'], rsi_p)
    atr   = _atr(df, atr_p)
    hma_up  = hma > hma.shift(1)
    hma_dn  = hma < hma.shift(1)
    atr_exp = atr > atr.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[hma_up & (rsi > 50) & atr_exp] =  1
    sig[hma_dn & (rsi < 50) & atr_exp] = -1
    return sig


def space_HMA_RSI_ATR(trial):
    return {
        'hma_p': trial.suggest_int('hma_p', 10, 40),
        'rsi_p': trial.suggest_int('rsi_p',  7, 21),
        'atr_p': trial.suggest_int('atr_p', 10, 20),
    }


# ── 16. MA_Rainbow_Bull ──────────────────────────────────────────────────────

def gen_MA_Rainbow_Bull(df, base_p=5, base_mult=2.0, **kw):
    base_p    = int(base_p)
    base_mult = float(base_mult)
    periods   = [max(int(base_p * (base_mult ** i)), 2) for i in range(6)]
    emas      = [_ema(df['close'], p) for p in periods]
    # Bull: e0 > e1 > e2 > e3 > e4 > e5
    bull = emas[0] > emas[1]
    bear = emas[0] < emas[1]
    for i in range(1, 5):
        bull = bull & (emas[i] > emas[i + 1])
        bear = bear & (emas[i] < emas[i + 1])
    sig = pd.Series(0, index=df.index)
    sig[bull & ~bull.shift(1).fillna(False)] =  1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    for i in range(1, len(sig)):
        if sig.iloc[i] == 0:
            if bull.iloc[i]:
                sig.iloc[i] =  1
            elif bear.iloc[i]:
                sig.iloc[i] = -1
    return sig


def space_MA_Rainbow_Bull(trial):
    return {
        'base_p':    trial.suggest_int('base_p', 3, 8),
        'base_mult': trial.suggest_float('base_mult', 1.5, 2.5),
    }


# ── 17. MA_Rainbow_Cross ─────────────────────────────────────────────────────

def gen_MA_Rainbow_Cross(df, p1=5, p2=10, p3=20, p4=40, **kw):
    p1 = int(p1)
    p2 = int(p2)
    p3 = int(p3)
    p4 = int(p4)
    e1 = _ema(df['close'], p1)
    e2 = _ema(df['close'], p2)
    e3 = _ema(df['close'], p3)
    e4 = _ema(df['close'], p4)
    aligned_bull = (e2 > e3) & (e3 > e4)
    aligned_bear = (e2 < e3) & (e3 < e4)
    cross_up   = (e1 > e2) & (e1.shift(1) <= e2.shift(1))
    cross_down = (e1 < e2) & (e1.shift(1) >= e2.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & aligned_bull] =  1
    sig[cross_down & aligned_bear] = -1
    return sig


def space_MA_Rainbow_Cross(trial):
    return {
        'p1': trial.suggest_int('p1',  3,  7),
        'p2': trial.suggest_int('p2',  7, 15),
        'p3': trial.suggest_int('p3', 15, 30),
        'p4': trial.suggest_int('p4', 30, 60),
    }


# ── 18. MA_Ribbon ────────────────────────────────────────────────────────────

def gen_MA_Ribbon(df, base_p=5, multiplier=1.7, **kw):
    base_p     = int(base_p)
    multiplier = float(multiplier)
    periods    = [max(int(base_p * (multiplier ** i)), 2) for i in range(8)]
    emas       = [_ema(df['close'], p) for p in periods]
    # All fanned up: each faster EMA above next slower
    bull = emas[0] > emas[1]
    bear = emas[0] < emas[1]
    for i in range(1, 7):
        bull = bull & (emas[i] > emas[i + 1])
        bear = bear & (emas[i] < emas[i + 1])
    sig = pd.Series(0, index=df.index)
    sig[bull & ~bull.shift(1).fillna(False)] =  1
    sig[bear & ~bear.shift(1).fillna(False)] = -1
    for i in range(1, len(sig)):
        if sig.iloc[i] == 0:
            if bull.iloc[i]:
                sig.iloc[i] =  1
            elif bear.iloc[i]:
                sig.iloc[i] = -1
    return sig


def space_MA_Ribbon(trial):
    return {
        'base_p':     trial.suggest_int('base_p', 3, 8),
        'multiplier': trial.suggest_float('multiplier', 1.5, 2.0),
    }


# ── 19. Adaptive_MA_ATR ──────────────────────────────────────────────────────

def gen_Adaptive_MA_ATR(df, base_p=20, atr_p=14, min_p=3, max_p=80, **kw):
    base_p = int(base_p)
    atr_p  = int(atr_p)
    min_p  = int(min_p)
    max_p  = int(max_p)
    atr    = _atr(df, atr_p)
    # ATR ratio: current atr vs rolling mean atr
    atr_mean  = _sma(atr, base_p)
    atr_ratio = (atr / atr_mean.replace(0, 1e-9)).clip(0.5, 2.0)
    # High vol → shorter period; low vol → longer period
    adapt_p   = (base_p / atr_ratio).clip(min_p, max_p).round().astype(int)
    # Build adaptive MA bar by bar (vectorised approximation via variable EMA)
    close  = df['close'].values
    n      = len(close)
    ama    = np.full(n, np.nan)
    ama[0] = close[0]
    for i in range(1, n):
        p_i   = max(min_p, min(max_p, int(adapt_p.iloc[i])))
        alpha = 2.0 / (p_i + 1.0)
        ama[i] = alpha * close[i] + (1.0 - alpha) * ama[i - 1]
    ama_s = pd.Series(ama, index=df.index)
    rising  = ama_s > ama_s.shift(1)
    falling = ama_s < ama_s.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[rising]  =  1
    sig[falling] = -1
    return sig


def space_Adaptive_MA_ATR(trial):
    return {
        'base_p': trial.suggest_int('base_p', 10,  50),
        'atr_p':  trial.suggest_int('atr_p',  10,  20),
        'min_p':  trial.suggest_int('min_p',   3,  10),
        'max_p':  trial.suggest_int('max_p',  50, 100),
    }


# ── 20. Adaptive_MA_ROC ──────────────────────────────────────────────────────

def gen_Adaptive_MA_ROC(df, base_p=20, roc_p=10, **kw):
    base_p = int(base_p)
    roc_p  = int(roc_p)
    close  = df['close']
    roc    = ((close - close.shift(roc_p)) / close.shift(roc_p).replace(0, 1e-9)).abs()
    roc_mean = _sma(roc, base_p)
    # High ROC → shorter period (more responsive)
    roc_ratio = (roc / roc_mean.replace(0, 1e-9)).clip(0.5, 3.0).fillna(1.0)
    adapt_p   = (base_p / roc_ratio).clip(3, base_p * 2).round().fillna(base_p).astype(int)
    close_v = close.values
    n       = len(close_v)
    ama     = np.full(n, np.nan)
    ama[0]  = close_v[0]
    for i in range(1, n):
        p_i   = max(3, min(base_p * 2, int(adapt_p.iloc[i])))
        alpha = 2.0 / (p_i + 1.0)
        ama[i] = alpha * close_v[i] + (1.0 - alpha) * ama[i - 1]
    ama_s   = pd.Series(ama, index=df.index)
    rising  = ama_s > ama_s.shift(1)
    above   = close > ama_s
    below   = close < ama_s
    sig = pd.Series(0, index=df.index)
    sig[rising  & above] =  1
    sig[~rising & below] = -1
    return sig


def space_Adaptive_MA_ROC(trial):
    return {
        'base_p': trial.suggest_int('base_p', 10, 50),
        'roc_p':  trial.suggest_int('roc_p',  10, 20),
    }


# ── 21. EMA_SMA_Cross ────────────────────────────────────────────────────────

def gen_EMA_SMA_Cross(df, p=21, **kw):
    p   = int(p)
    ema = _ema(df['close'], p)
    sma = _sma(df['close'], p)
    cross_up   = (ema > sma) & (ema.shift(1) <= sma.shift(1))
    cross_down = (ema < sma) & (ema.shift(1) >= sma.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_EMA_SMA_Cross(trial):
    return {
        'p': trial.suggest_int('p', 5, 50),
    }


# ── 22. WMA_EMA_Cross ────────────────────────────────────────────────────────

def gen_WMA_EMA_Cross(df, p=21, **kw):
    p   = int(p)
    wma = _wma(df['close'], p)
    ema = _ema(df['close'], p)
    cross_up   = (wma > ema) & (wma.shift(1) <= ema.shift(1))
    cross_down = (wma < ema) & (wma.shift(1) >= ema.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_WMA_EMA_Cross(trial):
    return {
        'p': trial.suggest_int('p', 5, 50),
    }


# ── 23. HMA_EMA_Cross ────────────────────────────────────────────────────────

def gen_HMA_EMA_Cross(df, hma_p=16, ema_p=40, **kw):
    hma_p = int(hma_p)
    ema_p = int(ema_p)
    hma   = _hma(df['close'], hma_p)
    ema   = _ema(df['close'], ema_p)
    cross_up   = (hma > ema) & (hma.shift(1) <= ema.shift(1))
    cross_down = (hma < ema) & (hma.shift(1) >= ema.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_HMA_EMA_Cross(trial):
    return {
        'hma_p': trial.suggest_int('hma_p',  5, 30),
        'ema_p': trial.suggest_int('ema_p',  20, 80),
    }


# ── 24. TEMA_DEMA_Cross ──────────────────────────────────────────────────────

def gen_TEMA_DEMA_Cross(df, p=20, **kw):
    p    = int(p)
    tema = _tema(df['close'], p)
    dema = _dema(df['close'], p)
    cross_up   = (tema > dema) & (tema.shift(1) <= dema.shift(1))
    cross_down = (tema < dema) & (tema.shift(1) >= dema.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_TEMA_DEMA_Cross(trial):
    return {
        'p': trial.suggest_int('p', 5, 40),
    }


# ── 25. McGinley_EMA_Cross ───────────────────────────────────────────────────

def gen_McGinley_EMA_Cross(df, p=14, k=0.6, **kw):
    p     = int(p)
    k     = float(k)
    close = df['close'].values
    n     = len(close)
    # McGinley Dynamic
    mg    = np.full(n, np.nan)
    mg[0] = close[0]
    for i in range(1, n):
        prev = mg[i - 1]
        if prev <= 0 or close[i] <= 0:
            mg[i] = close[i]
        else:
            ratio = close[i] / prev
            mg[i] = prev + (close[i] - prev) / (k * p * (ratio ** 4))
    mg_s  = pd.Series(mg, index=df.index)
    ema   = _ema(df['close'], p)
    cross_up   = (mg_s > ema) & (mg_s.shift(1) <= ema.shift(1))
    cross_down = (mg_s < ema) & (mg_s.shift(1) >= ema.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def space_McGinley_EMA_Cross(trial):
    return {
        'p': trial.suggest_int('p', 5, 30),
        'k': trial.suggest_float('k', 0.3, 0.7),
    }


# ── 26. VAMA_Strategy ────────────────────────────────────────────────────────

def gen_VAMA_Strategy(df, p=20, base_alpha=0.2, **kw):
    p          = int(p)
    base_alpha = float(base_alpha)
    close      = df['close'].values
    volume     = df['volume'].values
    n          = len(close)
    vol_avg    = pd.Series(volume).rolling(p, min_periods=1).mean().values
    vama       = np.full(n, np.nan)
    vama[0]    = close[0]
    for i in range(1, n):
        vol_ratio = volume[i] / max(vol_avg[i], 1e-9)
        alpha     = float(np.clip(base_alpha * vol_ratio, 0.01, 1.0))
        vama[i]   = alpha * close[i] + (1.0 - alpha) * vama[i - 1]
    vama_s  = pd.Series(vama, index=df.index)
    rising  = vama_s > vama_s.shift(1)
    above   = df['close'] > vama_s
    falling = vama_s < vama_s.shift(1)
    below   = df['close'] < vama_s
    sig = pd.Series(0, index=df.index)
    sig[rising  & above]  =  1
    sig[falling & below]  = -1
    return sig


def space_VAMA_Strategy(trial):
    return {
        'p':          trial.suggest_int('p', 10, 40),
        'base_alpha': trial.suggest_float('base_alpha', 0.1, 0.3),
    }


# ── 27. PWMA_Strategy ────────────────────────────────────────────────────────

def _pwma(s, p):
    """Polynomial Weighted MA: quadratic weights emphasise centre."""
    p  = int(p)
    idx = np.arange(p, dtype=float)
    mid = (p - 1) / 2.0
    w  = -(idx - mid) ** 2 + mid ** 2
    w  = np.maximum(w, 0)
    wsum = w.sum()
    if wsum == 0:
        return s.rolling(p, min_periods=1).mean()
    w = w / wsum
    return s.rolling(p, min_periods=1).apply(
        lambda x: np.dot(x[-len(w):], w[-len(x):]) / w[-len(x):].sum()
        if len(x) < p else np.dot(x, w),
        raw=True,
    )


def gen_PWMA_Strategy(df, p=14, **kw):
    p    = int(p)
    pwma = _pwma(df['close'], p)
    # Signal on cross of price over PWMA and direction confirmation
    cross_up   = (df['close'] > pwma) & (df['close'].shift(1) <= pwma.shift(1))
    cross_down = (df['close'] < pwma) & (df['close'].shift(1) >= pwma.shift(1))
    rising  = pwma > pwma.shift(1)
    falling = pwma < pwma.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[cross_up   & rising]  =  1
    sig[cross_down & falling] = -1
    return sig


def space_PWMA_Strategy(trial):
    return {
        'p': trial.suggest_int('p', 5, 30),
    }


# ── 28. Supertrend_MA_Cross ──────────────────────────────────────────────────

def gen_Supertrend_MA_Cross(df, st_p=10, st_mult=3.0, fast_p=9, slow_p=21, **kw):
    st_p    = int(st_p)
    st_mult = float(st_mult)
    fast_p  = int(fast_p)
    slow_p  = int(slow_p)
    atr     = _atr(df, st_p)
    hl2     = (df['high'] + df['low']) / 2.0
    upper_b = hl2 + st_mult * atr
    lower_b = hl2 - st_mult * atr
    close   = df['close'].values
    n       = len(close)
    upper   = upper_b.values.copy()
    lower   = lower_b.values.copy()
    trend   = np.ones(n, dtype=int)  # 1=bull, -1=bear
    for i in range(1, n):
        # Adjust bands
        if lower[i] < lower[i - 1] or close[i - 1] < lower[i - 1]:
            lower[i] = lower[i]
        else:
            lower[i] = lower[i - 1]
        if upper[i] > upper[i - 1] or close[i - 1] > upper[i - 1]:
            upper[i] = upper[i]
        else:
            upper[i] = upper[i - 1]
        if trend[i - 1] == -1 and close[i] > upper[i - 1]:
            trend[i] = 1
        elif trend[i - 1] == 1 and close[i] < lower[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    st_bull = pd.Series(trend == 1, index=df.index)
    st_bear = pd.Series(trend == -1, index=df.index)
    fast_ema = _ema(df['close'], fast_p)
    slow_ema = _ema(df['close'], slow_p)
    ma_bull  = fast_ema > slow_ema
    ma_bear  = fast_ema < slow_ema
    sig = pd.Series(0, index=df.index)
    enter_long  = st_bull & ma_bull & (~(st_bull & ma_bull)).shift(1).fillna(True)
    enter_short = st_bear & ma_bear & (~(st_bear & ma_bear)).shift(1).fillna(True)
    sig[enter_long]  =  1
    sig[enter_short] = -1
    return sig


def space_Supertrend_MA_Cross(trial):
    return {
        'st_p':    trial.suggest_int('st_p',    7, 21),
        'st_mult': trial.suggest_float('st_mult', 2.0, 4.0),
        'fast_p':  trial.suggest_int('fast_p',   5, 20),
        'slow_p':  trial.suggest_int('slow_p',  20, 60),
    }


# ── 29. EMA_Cloud ────────────────────────────────────────────────────────────

def gen_EMA_Cloud(df, p1=12, p2=26, **kw):
    p1   = int(p1)
    p2   = int(p2)
    e1   = _ema(df['close'], p1)
    e2   = _ema(df['close'], p2)
    top  = pd.concat([e1, e2], axis=1).max(axis=1)
    bot  = pd.concat([e1, e2], axis=1).min(axis=1)
    # Cloud expanding: top rising AND bottom rising (or spread widening)
    spread     = top - bot
    expanding  = spread > spread.shift(1)
    above_cloud = df['close'] > top
    below_cloud = df['close'] < bot
    sig = pd.Series(0, index=df.index)
    sig[above_cloud & expanding] =  1
    sig[below_cloud & expanding] = -1
    return sig


def space_EMA_Cloud(trial):
    return {
        'p1': trial.suggest_int('p1', 10, 30),
        'p2': trial.suggest_int('p2', 20, 60),
    }


# ── 30. SMA_Bounce ───────────────────────────────────────────────────────────

def gen_SMA_Bounce(df, sma_p=50, tolerance_pct=0.005, **kw):
    sma_p         = int(sma_p)
    tolerance_pct = float(tolerance_pct)
    sma           = _sma(df['close'], sma_p)
    close         = df['close']
    # Touched SMA from above: prev close was within tolerance of SMA (or below)
    near_sma_long  = (close.shift(1) >= sma.shift(1) * (1.0 - tolerance_pct)) & \
                     (close.shift(1) <= sma.shift(1) * (1.0 + tolerance_pct))
    # Bounce: current close back above SMA
    bounce_long    = (close > sma) & near_sma_long
    # Touched SMA from below
    near_sma_short = (close.shift(1) <= sma.shift(1) * (1.0 + tolerance_pct)) & \
                     (close.shift(1) >= sma.shift(1) * (1.0 - tolerance_pct))
    bounce_short   = (close < sma) & near_sma_short
    sig = pd.Series(0, index=df.index)
    sig[bounce_long]  =  1
    sig[bounce_short] = -1
    return sig


def space_SMA_Bounce(trial):
    return {
        'sma_p':         trial.suggest_int('sma_p', 20, 200),
        'tolerance_pct': trial.suggest_float('tolerance_pct', 0.001, 0.01),
    }


# ── STRATEGY_EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Golden_Death_Cross': {
        'gen': gen_Golden_Death_Cross,
        'space': space_Golden_Death_Cross,
        'default_params': {'fast_p': 50, 'slow_p': 200},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 8000,
                 'description': '50/200 EMA golden/death cross bidirectional'},
    },
    'Golden_Cross_RSI': {
        'gen': gen_Golden_Cross_RSI,
        'space': space_Golden_Cross_RSI,
        'default_params': {'fast_p': 30, 'slow_p': 150, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'EMA cross with RSI overbought/oversold filter'},
    },
    'Golden_Cross_Vol': {
        'gen': gen_Golden_Cross_Vol,
        'space': space_Golden_Cross_Vol,
        'default_params': {'fast_p': 20, 'slow_p': 100, 'vol_p': 20, 'vol_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'EMA cross confirmed by above-average volume surge'},
    },
    'Triple_Golden_Cross': {
        'gen': gen_Triple_Golden_Cross,
        'space': space_Triple_Golden_Cross,
        'default_params': {'p1': 9, 'p2': 21, 'p3': 100},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Three EMAs fully aligned bull or bear regime'},
    },
    'DEMA_Cross_v2': {
        'gen': gen_DEMA_Cross,
        'space': space_DEMA_Cross,
        'default_params': {'fast_p': 8, 'slow_p': 34},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'Double EMA fast/slow cross for faster signals'},
    },
    'DEMA_ATR_v2': {
        'gen': gen_DEMA_ATR,
        'space': space_DEMA_ATR,
        'default_params': {'dema_p': 20, 'atr_p': 14, 'atr_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'DEMA with ATR envelope breakout entries'},
    },
    'DEMA_RSI': {
        'gen': gen_DEMA_RSI,
        'space': space_DEMA_RSI,
        'default_params': {'dema_p': 20, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1500,
                 'description': 'DEMA trend direction confirmed by RSI above/below 50'},
    },
    'DEMA_BB': {
        'gen': gen_DEMA_BB,
        'space': space_DEMA_BB,
        'default_params': {'dema_p': 20, 'bb_p': 20, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1200,
                 'description': 'Buy dip to BB lower in DEMA uptrend; sell rip in downtrend'},
    },
    'TEMA_Cross': {
        'gen': gen_TEMA_Cross,
        'space': space_TEMA_Cross,
        'default_params': {'fast_p': 8, 'slow_p': 30},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'Triple EMA fast/slow cross for low-lag trend signals'},
    },
    'TEMA_EMA_Cross': {
        'gen': gen_TEMA_EMA_Cross,
        'space': space_TEMA_EMA_Cross,
        'default_params': {'p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1800,
                 'description': 'TEMA vs same-period EMA exploiting TEMA lead'},
    },
    'TEMA_RSI': {
        'gen': gen_TEMA_RSI,
        'space': space_TEMA_RSI,
        'default_params': {'tema_p': 20, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1200,
                 'description': 'TEMA direction with RSI in momentum zone 45-65'},
    },
    'TEMA_ATR': {
        'gen': gen_TEMA_ATR,
        'space': space_TEMA_ATR,
        'default_params': {'tema_p': 20, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1000,
                 'description': 'TEMA trend with ATR expansion confirming momentum'},
    },
    'HMA_Cross': {
        'gen': gen_HMA_Cross,
        'space': space_HMA_Cross,
        'default_params': {'fast_p': 9, 'slow_p': 36},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'Hull MA fast/slow cross — minimal lag trend following'},
    },
    'HMA_ATR_Band': {
        'gen': gen_HMA_ATR_Band,
        'space': space_HMA_ATR_Band,
        'default_params': {'hma_p': 20, 'atr_p': 14, 'atr_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Hull MA direction with ATR channel proximity filter'},
    },
    'HMA_RSI_ATR': {
        'gen': gen_HMA_RSI_ATR,
        'space': space_HMA_RSI_ATR,
        'default_params': {'hma_p': 20, 'rsi_p': 14, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1500,
                 'description': 'Hull MA + RSI momentum + ATR expansion triple filter'},
    },
    'MA_Rainbow_Bull': {
        'gen': gen_MA_Rainbow_Bull,
        'space': space_MA_Rainbow_Bull,
        'default_params': {'base_p': 5, 'base_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': '6 EMAs in full ascending/descending sequence alignment'},
    },
    'MA_Rainbow_Cross': {
        'gen': gen_MA_Rainbow_Cross,
        'space': space_MA_Rainbow_Cross,
        'default_params': {'p1': 5, 'p2': 10, 'p3': 20, 'p4': 40},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'Fastest EMA crosses second in aligned rainbow structure'},
    },
    'MA_Ribbon': {
        'gen': gen_MA_Ribbon,
        'space': space_MA_Ribbon,
        'default_params': {'base_p': 5, 'multiplier': 1.7},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': '8-EMA ribbon fully fanned up or down for regime detection'},
    },
    'Adaptive_MA_ATR': {
        'gen': gen_Adaptive_MA_ATR,
        'space': space_Adaptive_MA_ATR,
        'default_params': {'base_p': 20, 'atr_p': 14, 'min_p': 3, 'max_p': 80},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'MA period shortens in high volatility, lengthens in low vol'},
    },
    'Adaptive_MA_ROC': {
        'gen': gen_Adaptive_MA_ROC,
        'space': space_Adaptive_MA_ROC,
        'default_params': {'base_p': 20, 'roc_p': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'MA period adapts to price ROC: fast in high momentum'},
    },
    'EMA_SMA_Cross': {
        'gen': gen_EMA_SMA_Cross,
        'space': space_EMA_SMA_Cross,
        'default_params': {'p': 21},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3500,
                 'description': 'EMA vs SMA same period — EMA leads SMA on direction changes'},
    },
    'WMA_EMA_Cross': {
        'gen': gen_WMA_EMA_Cross,
        'space': space_WMA_EMA_Cross,
        'default_params': {'p': 21},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'WMA vs EMA cross — WMA emphasises recent prices'},
    },
    'HMA_EMA_Cross': {
        'gen': gen_HMA_EMA_Cross,
        'space': space_HMA_EMA_Cross,
        'default_params': {'hma_p': 16, 'ema_p': 40},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'Hull MA vs EMA cross exploiting HMA speed advantage'},
    },
    'TEMA_DEMA_Cross': {
        'gen': gen_TEMA_DEMA_Cross,
        'space': space_TEMA_DEMA_Cross,
        'default_params': {'p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'TEMA vs DEMA same period — TEMA slightly faster than DEMA'},
    },
    'McGinley_EMA_Cross': {
        'gen': gen_McGinley_EMA_Cross,
        'space': space_McGinley_EMA_Cross,
        'default_params': {'p': 14, 'k': 0.6},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'McGinley Dynamic auto-adjusts speed vs EMA crossover'},
    },
    'VAMA_Strategy': {
        'gen': gen_VAMA_Strategy,
        'space': space_VAMA_Strategy,
        'default_params': {'p': 20, 'base_alpha': 0.2},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Volume-adjusted MA: alpha scales with relative volume'},
    },
    'PWMA_Strategy': {
        'gen': gen_PWMA_Strategy,
        'space': space_PWMA_Strategy,
        'default_params': {'p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1800,
                 'description': 'Polynomial weighted MA — quadratic weights emphasise centre'},
    },
    'Supertrend_MA_Cross': {
        'gen': gen_Supertrend_MA_Cross,
        'space': space_Supertrend_MA_Cross,
        'default_params': {'st_p': 10, 'st_mult': 3.0, 'fast_p': 9, 'slow_p': 21},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'SuperTrend regime filter combined with fast/slow MA cross'},
    },
    'EMA_Cloud': {
        'gen': gen_EMA_Cloud,
        'space': space_EMA_Cloud,
        'default_params': {'p1': 12, 'p2': 26},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Two-EMA cloud: long above expanding cloud, short below'},
    },
    'SMA_Bounce': {
        'gen': gen_SMA_Bounce,
        'space': space_SMA_Bounce,
        'default_params': {'sma_p': 50, 'tolerance_pct': 0.005},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'Price touches key SMA then bounces back through it'},
    },
}
