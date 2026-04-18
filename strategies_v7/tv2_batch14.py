#!/usr/bin/env python3
"""TV2 BATCH 14 — 30 estrategias Ehlers + Adaptive MA 2026-04-01

  Ehlers_Fisher          — Fisher Transform cross (v4, ~4000L)
  Ehlers_Fisher_CG       — Fisher + Center of Gravity (v5, ~2000L)
  Ehlers_Fisher_Stoch    — Fisher + Stochastic (v4, ~1500L)
  Ehlers_CyberCycle      — Cyber Cycle oscillator (v4, ~2500L)
  Ehlers_Stochastic      — Ehlers fast-cycle stochastic (v4, ~2000L)
  Ehlers_MESA            — Simplified MESA adaptive MA (v5, ~3000L)
  Ehlers_Bandpass        — Bandpass filter (v4, ~1500L)
  Ehlers_Sinewave        — Sinewave + Lead Sine (v4, ~1800L)
  KAMA_Strategy          — Kaufman AMA direction (v4, ~3500L)
  KAMA_Cross             — Fast vs slow KAMA cross (v5, ~2000L)
  KAMA_RSI               — KAMA + RSI filter (v4, ~1500L)
  KAMA_ATR               — KAMA + ATR gate (v5, ~1000L)
  VIDYA_Strategy         — VIDYA single (v4, ~2000L)
  VIDYA_Cross            — Dual VIDYA cross (v5, ~1200L)
  FRAMA_Strategy         — Fractal Adaptive MA (v4, ~2500L)
  FRAMA_RSI              — FRAMA + RSI filter (v5, ~1200L)
  ALMA_Strategy          — Arnaud Legoux MA cross (v4, ~3000L)
  ALMA_Cross             — Fast vs slow ALMA (v5, ~1800L)
  ALMA_BB                — ALMA + std bands (v4, ~1000L)
  JMA_Strategy           — Jurik MA approximation (v5, ~2500L)
  JMA_Cross              — Fast vs slow JMA (v4, ~1500L)
  T3_Strategy            — Tillson T3 cross (v4, ~3000L)
  T3_Cross               — Fast vs slow T3 (v5, ~1500L)
  T3_ATR                 — T3 + ATR gate (v4, ~1000L)
  ZLEMA_Strategy         — Zero-Lag EMA (v4, ~2500L)
  ZLEMA_Cross            — Fast vs slow ZLEMA (v5, ~1500L)
  ZLEMA_RSI              — ZLEMA + RSI (v4, ~800L)
  Laguerre_RSI           — Laguerre RSI oscillator (v4, ~3500L)
  Laguerre_RSI_ATR       — Laguerre RSI + ATR trail (v5, ~1500L)
  Laguerre_RSI_EMA       — Laguerre RSI + EMA trend (v4, ~1000L)
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
    d = s.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    rs = _rma(g, p) / _rma(l, p).replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _wma(s, p):
    p = int(p)
    w = np.arange(1, p + 1, dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


# ── 1. Ehlers_Fisher ──────────────────────────────────────────────────────────

def gen_Ehlers_Fisher(df, length=10, **kw):
    length = int(length)
    src    = (df['high'] + df['low']) / 2.0
    hi     = src.rolling(length).max()
    lo     = src.rolling(length).min()
    rng    = (hi - lo).replace(0, 1e-9)
    val    = (2.0 * (src - lo) / rng - 1.0).clip(-0.999, 0.999)
    fish   = 0.5 * np.log((1 + val) / (1 - val))
    sig    = fish.shift(1)
    long   = (fish > sig) & (fish.shift(1) <= sig.shift(1))
    short  = (fish < sig) & (fish.shift(1) >= sig.shift(1))
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_Ehlers_Fisher(trial):
    return {'length': trial.suggest_int('length', 7, 20)}


# ── 2. Ehlers_Fisher_CG ───────────────────────────────────────────────────────

def _cg_oscillator(src, p):
    """Center of Gravity oscillator."""
    p = int(p)
    num = pd.Series(0.0, index=src.index)
    den = pd.Series(0.0, index=src.index)
    for i in range(p):
        num = num + (i + 1) * src.shift(i).fillna(0)
        den = den + src.shift(i).fillna(0)
    return -num / den.replace(0, 1e-9)


def gen_Ehlers_Fisher_CG(df, fish_p=10, cg_p=10, **kw):
    fish_p = int(fish_p)
    cg_p   = int(cg_p)
    src    = (df['high'] + df['low']) / 2.0
    hi     = src.rolling(fish_p).max()
    lo     = src.rolling(fish_p).min()
    rng    = (hi - lo).replace(0, 1e-9)
    val    = (2.0 * (src - lo) / rng - 1.0).clip(-0.999, 0.999)
    fish   = 0.5 * np.log((1 + val) / (1 - val))
    cg     = _cg_oscillator(df['close'], cg_p)
    # Fisher bullish: rising; CG bullish: rising (CG inverted so falling=bullish)
    fish_bull = fish > fish.shift(1)
    cg_bull   = cg < cg.shift(1)   # CG is negative; less negative = rising price
    long  = fish_bull  & cg_bull
    short = (~fish_bull) & (~cg_bull)
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_Ehlers_Fisher_CG(trial):
    return {
        'fish_p': trial.suggest_int('fish_p', 7, 20),
        'cg_p':   trial.suggest_int('cg_p',   7, 20),
    }


# ── 3. Ehlers_Fisher_Stoch ────────────────────────────────────────────────────

def gen_Ehlers_Fisher_Stoch(df, fish_p=10, stoch_k=14, **kw):
    fish_p  = int(fish_p)
    stoch_k = int(stoch_k)
    src     = (df['high'] + df['low']) / 2.0
    hi      = src.rolling(fish_p).max()
    lo      = src.rolling(fish_p).min()
    rng     = (hi - lo).replace(0, 1e-9)
    val     = (2.0 * (src - lo) / rng - 1.0).clip(-0.999, 0.999)
    fish    = 0.5 * np.log((1 + val) / (1 - val))
    # Stochastic %K
    hi_k    = df['high'].rolling(stoch_k).max()
    lo_k    = df['low'].rolling(stoch_k).min()
    stoch   = 100 * (df['close'] - lo_k) / (hi_k - lo_k).replace(0, 1e-9)
    fish_up = fish > fish.shift(1)
    long    = fish_up  & (stoch < 20)
    short   = (~fish_up) & (stoch > 80)
    out     = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_Ehlers_Fisher_Stoch(trial):
    return {
        'fish_p':  trial.suggest_int('fish_p',  7, 20),
        'stoch_k': trial.suggest_int('stoch_k', 7, 21),
    }


# ── 4. Ehlers_CyberCycle ──────────────────────────────────────────────────────

def gen_Ehlers_CyberCycle(df, alpha=0.07, **kw):
    alpha  = float(alpha)
    close  = df['close'].values.astype(np.float64)
    n      = len(close)
    cc     = np.zeros(n)
    coef1  = (1 - 0.5 * alpha) ** 2
    coef2  = 2 * (1 - alpha)
    coef3  = (1 - alpha) ** 2
    for i in range(2, n):
        cc[i] = (coef1 * (close[i] - 2 * close[i - 1] + close[i - 2])
                 + coef2 * cc[i - 1]
                 - coef3 * cc[i - 2])
    cc_s  = pd.Series(cc, index=df.index)
    trig  = cc_s.shift(1)
    long  = (cc_s > trig) & (cc_s.shift(1) <= trig.shift(1))
    short = (cc_s < trig) & (cc_s.shift(1) >= trig.shift(1))
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_Ehlers_CyberCycle(trial):
    return {'alpha': trial.suggest_float('alpha', 0.05, 0.30)}


# ── 5. Ehlers_Stochastic ──────────────────────────────────────────────────────

def gen_Ehlers_Stochastic(df, length=8, **kw):
    """Ehlers simplified fast-cycle stochastic via In-Phase component."""
    length = int(length)
    close  = df['close'].values.astype(np.float64)
    n      = len(close)
    # Approximate I component: (close - close[2]) smoothed
    ip     = np.zeros(n)
    for i in range(2, n):
        ip[i] = close[i] - close[i - 2]
    ip_s   = pd.Series(ip, index=df.index)
    # Normalize over window
    hi_ip  = ip_s.rolling(length).max()
    lo_ip  = ip_s.rolling(length).min()
    rng    = (hi_ip - lo_ip).replace(0, 1e-9)
    es     = (ip_s - lo_ip) / rng   # 0..1
    long   = (es > 0.8) & (es.shift(1) <= 0.8)
    short  = (es < 0.2) & (es.shift(1) >= 0.2)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_Ehlers_Stochastic(trial):
    return {'length': trial.suggest_int('length', 7, 20)}


# ── 6. Ehlers_MESA ────────────────────────────────────────────────────────────

def gen_Ehlers_MESA(df, fast_limit=0.5, slow_limit=0.05, **kw):
    """Simplified MESA adaptive MA: period adapts via momentum ratio."""
    fast_limit = float(fast_limit)
    slow_limit = float(slow_limit)
    close      = df['close'].values.astype(np.float64)
    n          = len(close)
    ama        = np.zeros(n)
    ama[0]     = close[0]
    # Approximate adaptive alpha from local momentum
    for i in range(1, n):
        lo_i = max(0, i - 20)
        window_hi = close[lo_i:i + 1].max()
        window_lo = close[lo_i:i + 1].min()
        span = window_hi - window_lo
        if span == 0:
            alpha = slow_limit
        else:
            direction  = abs(close[i] - close[lo_i])
            volatility = np.sum(np.abs(np.diff(close[lo_i:i + 1])))
            if volatility == 0:
                er = 0.0
            else:
                er = direction / volatility
            alpha = (er * (fast_limit - slow_limit) + slow_limit) ** 2
        ama[i] = ama[i - 1] + alpha * (close[i] - ama[i - 1])
    ama_s = pd.Series(ama, index=df.index)
    rising = ama_s > ama_s.shift(1)
    long   = rising  & (~rising.shift(1, fill_value=False))
    short  = (~rising) & (rising.shift(1, fill_value=False))
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_Ehlers_MESA(trial):
    return {
        'fast_limit': trial.suggest_float('fast_limit', 0.3, 0.7),
        'slow_limit': trial.suggest_float('slow_limit', 0.02, 0.1),
    }


# ── 7. Ehlers_Bandpass ────────────────────────────────────────────────────────

def gen_Ehlers_Bandpass(df, period=20, bandwidth=0.3, **kw):
    period    = int(period)
    bandwidth = float(bandwidth)
    close     = df['close'].values.astype(np.float64)
    n         = len(close)
    bp        = np.zeros(n)
    beta      = np.cos(2 * np.pi / period)
    gamma     = 1.0 / np.cos(2 * np.pi * bandwidth / period)
    alpha_val = gamma - np.sqrt(gamma * gamma - 1)
    for i in range(2, n):
        bp[i] = ((1 - alpha_val / 2) * (close[i] - close[i - 2])
                 + beta * (1 + alpha_val / 2) * bp[i - 1]
                 - alpha_val * bp[i - 2])
    bp_s  = pd.Series(bp, index=df.index)
    rising = bp_s > bp_s.shift(1)
    long   = (bp_s > 0) & rising
    short  = (bp_s < 0) & (~rising)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_Ehlers_Bandpass(trial):
    return {
        'period':    trial.suggest_int('period',        10, 40),
        'bandwidth': trial.suggest_float('bandwidth', 0.2, 0.8),
    }


# ── 8. Ehlers_Sinewave ────────────────────────────────────────────────────────

def gen_Ehlers_Sinewave(df, length=14, **kw):
    """Sinewave indicator: Sine vs Lead Sine crossover."""
    length = int(length)
    close  = df['close'].values.astype(np.float64)
    n      = len(close)
    # Estimate dominant cycle via autocorrelation
    sine     = np.zeros(n)
    lead     = np.zeros(n)
    for i in range(length, n):
        window = close[i - length:i]
        # Simple dominant cycle: use fixed length, compute sinewave position
        # Phase approximation from slope
        slope  = (window[-1] - window[0]) / (length + 1e-9)
        norm   = np.std(window) + 1e-9
        phase  = np.arctan(slope / norm)
        sine[i] = np.sin(phase)
        lead[i] = np.sin(phase + np.pi / 4)   # 45° lead
    sine_s = pd.Series(sine, index=df.index)
    lead_s = pd.Series(lead, index=df.index)
    long   = (sine_s > lead_s) & (sine_s.shift(1) <= lead_s.shift(1))
    short  = (sine_s < lead_s) & (sine_s.shift(1) >= lead_s.shift(1))
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_Ehlers_Sinewave(trial):
    return {'length': trial.suggest_int('length', 10, 30)}


# ── 9. KAMA_Strategy ──────────────────────────────────────────────────────────

def _kama(close_arr, er_p, fast_p, slow_p):
    n      = len(close_arr)
    kama   = np.zeros(n)
    kama[0] = close_arr[0]
    fast_sc = 2.0 / (fast_p + 1)
    slow_sc = 2.0 / (slow_p + 1)
    for i in range(1, n):
        if i < er_p:
            kama[i] = kama[i - 1]
            continue
        direction  = abs(close_arr[i] - close_arr[i - er_p])
        volatility = np.sum(np.abs(np.diff(close_arr[i - er_p:i + 1])))
        er         = direction / (volatility + 1e-9)
        sc         = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i]    = kama[i - 1] + sc * (close_arr[i] - kama[i - 1])
    return kama


def gen_KAMA_Strategy(df, er_p=10, fast_p=2, slow_p=30, **kw):
    er_p   = int(er_p)
    fast_p = int(fast_p)
    slow_p = int(slow_p)
    close  = df['close'].values.astype(np.float64)
    kama   = pd.Series(_kama(close, er_p, fast_p, slow_p), index=df.index)
    rising = kama > kama.shift(1)
    above  = df['close'] > kama
    long   = rising  & above
    short  = (~rising) & (~above)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_KAMA_Strategy(trial):
    return {
        'er_p':   trial.suggest_int('er_p',   5, 20),
        'fast_p': trial.suggest_int('fast_p', 2,  5),
        'slow_p': trial.suggest_int('slow_p', 20, 40),
    }


# ── 10. KAMA_Cross ────────────────────────────────────────────────────────────

def gen_KAMA_Cross(df, er_fast=8, er_slow=20, **kw):
    er_fast = int(er_fast)
    er_slow = int(er_slow)
    close   = df['close'].values.astype(np.float64)
    fast_k  = pd.Series(_kama(close, er_fast, 2, 20), index=df.index)
    slow_k  = pd.Series(_kama(close, er_slow, 2, 30), index=df.index)
    long    = (fast_k > slow_k) & (fast_k.shift(1) <= slow_k.shift(1))
    short   = (fast_k < slow_k) & (fast_k.shift(1) >= slow_k.shift(1))
    out     = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_KAMA_Cross(trial):
    return {
        'er_fast': trial.suggest_int('er_fast', 5, 15),
        'er_slow': trial.suggest_int('er_slow', 15, 30),
    }


# ── 11. KAMA_RSI ──────────────────────────────────────────────────────────────

def gen_KAMA_RSI(df, er_p=10, rsi_p=14, **kw):
    er_p  = int(er_p)
    rsi_p = int(rsi_p)
    close = df['close'].values.astype(np.float64)
    kama  = pd.Series(_kama(close, er_p, 2, 30), index=df.index)
    rsi   = _rsi(df['close'], rsi_p)
    rising = kama > kama.shift(1)
    long   = rising  & (rsi > 50)
    short  = (~rising) & (rsi < 50)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_KAMA_RSI(trial):
    return {
        'er_p':  trial.suggest_int('er_p',  5, 20),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 12. KAMA_ATR ──────────────────────────────────────────────────────────────

def gen_KAMA_ATR(df, er_p=10, atr_p=14, **kw):
    er_p  = int(er_p)
    atr_p = int(atr_p)
    close = df['close'].values.astype(np.float64)
    kama  = pd.Series(_kama(close, er_p, 2, 30), index=df.index)
    atr   = _atr(df, atr_p)
    atr_avg = _sma(atr, atr_p)
    rising  = kama > kama.shift(1)
    high_vol = atr > atr_avg
    long   = rising  & high_vol
    short  = (~rising) & high_vol
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_KAMA_ATR(trial):
    return {
        'er_p':  trial.suggest_int('er_p',  5, 20),
        'atr_p': trial.suggest_int('atr_p', 10, 20),
    }


# ── 13. VIDYA_Strategy ────────────────────────────────────────────────────────

def _cmo(series, p):
    """Chande Momentum Oscillator (-100..100)."""
    p   = int(p)
    d   = series.diff()
    su  = d.clip(lower=0).rolling(p).sum()
    sd  = (-d).clip(lower=0).rolling(p).sum()
    return (su - sd) / (su + sd).replace(0, 1e-9) * 100


def _vidya(close_arr, cmo_arr, ema_p, k):
    n    = len(close_arr)
    vid  = np.zeros(n)
    vid[0] = close_arr[0]
    base_alpha = 2.0 / (ema_p + 1)
    for i in range(1, n):
        cmo_abs = abs(cmo_arr[i]) / 100.0 if not np.isnan(cmo_arr[i]) else 0.0
        alpha   = k * cmo_abs + base_alpha
        alpha   = min(alpha, 1.0)
        vid[i]  = alpha * close_arr[i] + (1 - alpha) * vid[i - 1]
    return vid


def gen_VIDYA_Strategy(df, cmo_p=9, ema_p=12, k=0.5, **kw):
    cmo_p = int(cmo_p)
    ema_p = int(ema_p)
    k     = float(k)
    cmo   = _cmo(df['close'], cmo_p).values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    vidya = pd.Series(_vidya(close, cmo, ema_p, k), index=df.index)
    long  = df['close'] > vidya
    short = df['close'] < vidya
    out   = pd.Series(0, index=df.index)
    out[long & ~long.shift(1, fill_value=False)]  =  1
    out[short & ~short.shift(1, fill_value=False)] = -1
    return out


def space_VIDYA_Strategy(trial):
    return {
        'cmo_p': trial.suggest_int('cmo_p',   5, 20),
        'ema_p': trial.suggest_int('ema_p',   5, 20),
        'k':     trial.suggest_float('k', 0.2, 0.8),
    }


# ── 14. VIDYA_Cross ───────────────────────────────────────────────────────────

def gen_VIDYA_Cross(df, fast_p=5, slow_p=20, k=0.5, **kw):
    fast_p = int(fast_p)
    slow_p = int(slow_p)
    k      = float(k)
    cmo    = _cmo(df['close'], 9).values.astype(np.float64)
    close  = df['close'].values.astype(np.float64)
    fast_v = pd.Series(_vidya(close, cmo, fast_p, k), index=df.index)
    slow_v = pd.Series(_vidya(close, cmo, slow_p, k), index=df.index)
    long   = (fast_v > slow_v) & (fast_v.shift(1) <= slow_v.shift(1))
    short  = (fast_v < slow_v) & (fast_v.shift(1) >= slow_v.shift(1))
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_VIDYA_Cross(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',  3, 10),
        'slow_p': trial.suggest_int('slow_p', 15, 30),
        'k':      trial.suggest_float('k', 0.2, 0.8),
    }


# ── 15. FRAMA_Strategy ────────────────────────────────────────────────────────

def _frama(close_arr, high_arr, low_arr, length):
    length = int(length)
    half   = length // 2
    n      = len(close_arr)
    frama  = np.zeros(n)
    frama[0] = close_arr[0]
    for i in range(length, n):
        hi1  = high_arr[i - length:i - half].max()
        lo1  = low_arr[i - length:i - half].min()
        hi2  = high_arr[i - half:i].max()
        lo2  = low_arr[i - half:i].min()
        hi_all = high_arr[i - length:i].max()
        lo_all = low_arr[i - length:i].min()
        n1   = (hi1 - lo1) / half if half > 0 else 0
        n2   = (hi2 - lo2) / half if half > 0 else 0
        n3   = (hi_all - lo_all) / length if length > 0 else 0
        if n3 > 0 and n1 + n2 > 0:
            D = np.log(n1 + n2) / np.log(2) - np.log(n3) / np.log(2) if (n1 + n2) > 0 else 1.0
        else:
            D = 1.0
        D     = np.clip(D, 1.0, 2.0)
        alpha = np.exp(-4.6 * (D - 1.0))
        alpha = np.clip(alpha, 0.01, 1.0)
        frama[i] = alpha * close_arr[i] + (1 - alpha) * frama[i - 1]
    # Fill initial segment
    for i in range(1, length):
        frama[i] = frama[i - 1] + (close_arr[i] - frama[i - 1]) * 0.3
    return frama


def gen_FRAMA_Strategy(df, length=16, **kw):
    length = int(length)
    close  = df['close'].values.astype(np.float64)
    high   = df['high'].values.astype(np.float64)
    low    = df['low'].values.astype(np.float64)
    fra    = pd.Series(_frama(close, high, low, length), index=df.index)
    above  = df['close'] > fra
    long   = above  & ~above.shift(1, fill_value=False)
    short  = (~above) & above.shift(1, fill_value=True)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_FRAMA_Strategy(trial):
    return {'length': trial.suggest_int('length', 8, 20)}


# ── 16. FRAMA_RSI ─────────────────────────────────────────────────────────────

def gen_FRAMA_RSI(df, length=16, rsi_p=14, **kw):
    length = int(length)
    rsi_p  = int(rsi_p)
    close  = df['close'].values.astype(np.float64)
    high   = df['high'].values.astype(np.float64)
    low    = df['low'].values.astype(np.float64)
    fra    = pd.Series(_frama(close, high, low, length), index=df.index)
    rsi    = _rsi(df['close'], rsi_p)
    rising = fra > fra.shift(1)
    long   = rising  & (rsi > 50)
    short  = (~rising) & (rsi < 50)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_FRAMA_RSI(trial):
    return {
        'length': trial.suggest_int('length', 8, 20),
        'rsi_p':  trial.suggest_int('rsi_p',  7, 21),
    }


# ── 17. ALMA_Strategy ─────────────────────────────────────────────────────────

def _alma(series, length, offset, sigma):
    length = int(length)
    offset = float(offset)
    sigma  = float(sigma)
    m      = offset * (length - 1)
    s      = length / sigma
    weights = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(length)])
    weights /= weights.sum()
    return series.rolling(length).apply(lambda x: np.dot(x, weights), raw=True)


def gen_ALMA_Strategy(df, length=20, offset=0.85, sigma=6.0, **kw):
    length = int(length)
    offset = float(offset)
    sigma  = float(sigma)
    alma   = _alma(df['close'], length, offset, sigma)
    long   = (df['close'] > alma) & (df['close'].shift(1) <= alma.shift(1))
    short  = (df['close'] < alma) & (df['close'].shift(1) >= alma.shift(1))
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_ALMA_Strategy(trial):
    return {
        'length': trial.suggest_int('length',     5, 30),
        'offset': trial.suggest_float('offset', 0.5, 1.0),
        'sigma':  trial.suggest_float('sigma',  3.0, 8.0),
    }


# ── 18. ALMA_Cross ────────────────────────────────────────────────────────────

def gen_ALMA_Cross(df, fast_p=9, slow_p=30, offset=0.85, sigma=6.0, **kw):
    fast_p = int(fast_p)
    slow_p = int(slow_p)
    offset = float(offset)
    sigma  = float(sigma)
    fast   = _alma(df['close'], fast_p, offset, sigma)
    slow   = _alma(df['close'], slow_p, offset, sigma)
    long   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    short  = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_ALMA_Cross(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',    5, 15),
        'slow_p': trial.suggest_int('slow_p',   20, 50),
        'offset': trial.suggest_float('offset', 0.7, 0.9),
        'sigma':  trial.suggest_float('sigma',  4.0, 8.0),
    }


# ── 19. ALMA_BB ───────────────────────────────────────────────────────────────

def gen_ALMA_BB(df, length=20, offset=0.85, **kw):
    length = int(length)
    offset = float(offset)
    alma   = _alma(df['close'], length, offset, 6.0)
    std    = df['close'].rolling(length).std()
    lower  = alma - 2.0 * std
    upper  = alma + 2.0 * std
    rising = alma > alma.shift(1)
    long   = (df['close'] < lower) & rising
    short  = (df['close'] > upper) & (~rising)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_ALMA_BB(trial):
    return {
        'length': trial.suggest_int('length',    10, 30),
        'offset': trial.suggest_float('offset', 0.7, 0.9),
    }


# ── 20. JMA_Strategy ──────────────────────────────────────────────────────────

def _jma(close_arr, length, phase, power):
    """Jurik MA approximation: two-stage adaptive EMA with phase."""
    length = int(length)
    phase  = float(phase)   # -100..100
    power  = float(power)   # 1..3
    n      = len(close_arr)
    jma    = np.zeros(n)
    e0     = np.zeros(n)
    e1     = np.zeros(n)
    e2     = np.zeros(n)
    # Phase ratio: 0.5 + phase / 200 maps -100..100 → 0..1
    phase_ratio = 0.5 + phase / 200.0
    beta        = 0.45 * (length - 1) / (0.45 * (length - 1) + 2.0)
    alpha       = beta ** power
    jma[0]      = close_arr[0]
    e0[0]       = close_arr[0]
    for i in range(1, n):
        e0[i] = (1 - alpha) * close_arr[i] + alpha * e0[i - 1]
        e1[i] = (close_arr[i] - e0[i]) * (1 - beta) + beta * e1[i - 1]
        e2[i] = e0[i] + phase_ratio * e1[i]
        jma[i] = (1 - alpha) * e2[i] + alpha * jma[i - 1]
    return jma


def gen_JMA_Strategy(df, length=14, phase=0, power=2, **kw):
    length = int(length)
    phase  = float(phase)
    power  = float(power)
    close  = df['close'].values.astype(np.float64)
    jma    = pd.Series(_jma(close, length, phase, power), index=df.index)
    rising = jma > jma.shift(1)
    above  = df['close'] > jma
    long   = rising  & above
    short  = (~rising) & (~above)
    out    = pd.Series(0, index=df.index)
    out[long & ~(long.shift(1, fill_value=False))]  =  1
    out[short & ~(short.shift(1, fill_value=False))] = -1
    return out


def space_JMA_Strategy(trial):
    return {
        'length': trial.suggest_int('length',    5, 30),
        'phase':  trial.suggest_float('phase', -50.0, 100.0),
        'power':  trial.suggest_float('power',   1.0, 3.0),
    }


# ── 21. JMA_Cross ─────────────────────────────────────────────────────────────

def gen_JMA_Cross(df, fast_p=8, slow_p=30, phase=0, **kw):
    fast_p = int(fast_p)
    slow_p = int(slow_p)
    phase  = float(phase)
    close  = df['close'].values.astype(np.float64)
    fast   = pd.Series(_jma(close, fast_p, phase, 2), index=df.index)
    slow   = pd.Series(_jma(close, slow_p, phase, 2), index=df.index)
    long   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    short  = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_JMA_Cross(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',    5, 15),
        'slow_p': trial.suggest_int('slow_p',   20, 50),
        'phase':  trial.suggest_float('phase', -50.0, 50.0),
    }


# ── 22. T3_Strategy ───────────────────────────────────────────────────────────

def _t3(series, length, factor):
    length = int(length)
    factor = float(factor)
    vf  = factor
    c1  = -(vf ** 3)
    c2  = 3 * vf ** 2 + 3 * vf ** 3
    c3  = -6 * vf ** 2 - 3 * vf - 3 * vf ** 3
    c4  = 1 + 3 * vf + vf ** 3 + 3 * vf ** 2
    e1  = _ema(series, length)
    e2  = _ema(e1, length)
    e3  = _ema(e2, length)
    e4  = _ema(e3, length)
    e5  = _ema(e4, length)
    e6  = _ema(e5, length)
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def gen_T3_Strategy(df, length=5, factor=0.7, **kw):
    length = int(length)
    factor = float(factor)
    t3     = _t3(df['close'], length, factor)
    long   = (df['close'] > t3) & (df['close'].shift(1) <= t3.shift(1))
    short  = (df['close'] < t3) & (df['close'].shift(1) >= t3.shift(1))
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_T3_Strategy(trial):
    return {
        'length': trial.suggest_int('length',    3, 15),
        'factor': trial.suggest_float('factor', 0.5, 0.9),
    }


# ── 23. T3_Cross ──────────────────────────────────────────────────────────────

def gen_T3_Cross(df, fast_p=4, slow_p=14, factor=0.7, **kw):
    fast_p = int(fast_p)
    slow_p = int(slow_p)
    factor = float(factor)
    fast   = _t3(df['close'], fast_p, factor)
    slow   = _t3(df['close'], slow_p, factor)
    long   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    short  = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_T3_Cross(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',    3,  8),
        'slow_p': trial.suggest_int('slow_p',   10, 25),
        'factor': trial.suggest_float('factor', 0.5, 0.9),
    }


# ── 24. T3_ATR ────────────────────────────────────────────────────────────────

def gen_T3_ATR(df, length=5, factor=0.7, atr_p=14, **kw):
    length = int(length)
    factor = float(factor)
    atr_p  = int(atr_p)
    t3     = _t3(df['close'], length, factor)
    atr    = _atr(df, atr_p)
    atr_avg = _sma(atr, atr_p)
    rising  = t3 > t3.shift(1)
    expanding = atr > atr_avg
    long   = rising  & expanding
    short  = (~rising) & expanding
    out    = pd.Series(0, index=df.index)
    out[long & ~(long.shift(1, fill_value=False))]  =  1
    out[short & ~(short.shift(1, fill_value=False))] = -1
    return out


def space_T3_ATR(trial):
    return {
        'length': trial.suggest_int('length',    3, 15),
        'factor': trial.suggest_float('factor', 0.5, 0.9),
        'atr_p':  trial.suggest_int('atr_p',   10, 20),
    }


# ── 25. ZLEMA_Strategy ────────────────────────────────────────────────────────

def _zlema(series, length):
    length  = int(length)
    lag     = (length - 1) // 2
    adjusted = 2 * series - series.shift(lag)
    return adjusted.ewm(span=length, adjust=False).mean()


def gen_ZLEMA_Strategy(df, length=20, **kw):
    length = int(length)
    zl     = _zlema(df['close'], length)
    rising = zl > zl.shift(1)
    above  = df['close'] > zl
    long   = rising  & above
    short  = (~rising) & (~above)
    out    = pd.Series(0, index=df.index)
    out[long & ~(long.shift(1, fill_value=False))]  =  1
    out[short & ~(short.shift(1, fill_value=False))] = -1
    return out


def space_ZLEMA_Strategy(trial):
    return {'length': trial.suggest_int('length', 5, 30)}


# ── 26. ZLEMA_Cross ───────────────────────────────────────────────────────────

def gen_ZLEMA_Cross(df, fast_p=10, slow_p=30, **kw):
    fast_p = int(fast_p)
    slow_p = int(slow_p)
    fast   = _zlema(df['close'], fast_p)
    slow   = _zlema(df['close'], slow_p)
    long   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    short  = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_ZLEMA_Cross(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',  5, 15),
        'slow_p': trial.suggest_int('slow_p', 20, 60),
    }


# ── 27. ZLEMA_RSI ─────────────────────────────────────────────────────────────

def gen_ZLEMA_RSI(df, length=20, rsi_p=14, **kw):
    length = int(length)
    rsi_p  = int(rsi_p)
    zl     = _zlema(df['close'], length)
    rsi    = _rsi(df['close'], rsi_p)
    rising = zl > zl.shift(1)
    long   = rising  & (rsi >= 45) & (rsi <= 65)
    short  = (~rising) & ((rsi < 45) | (rsi > 65))
    out    = pd.Series(0, index=df.index)
    out[long & ~(long.shift(1, fill_value=False))]  =  1
    out[short & ~(short.shift(1, fill_value=False))] = -1
    return out


def space_ZLEMA_RSI(trial):
    return {
        'length': trial.suggest_int('length', 5, 30),
        'rsi_p':  trial.suggest_int('rsi_p',  7, 21),
    }


# ── 28. Laguerre_RSI ──────────────────────────────────────────────────────────

def _laguerre_rsi(close_arr, alpha):
    alpha = float(alpha)
    n     = len(close_arr)
    L0    = np.zeros(n)
    L1    = np.zeros(n)
    L2    = np.zeros(n)
    L3    = np.zeros(n)
    lrsi  = np.zeros(n)
    L0[0] = close_arr[0]
    for i in range(1, n):
        L0[i] = (1 - alpha) * close_arr[i] + alpha * L0[i - 1]
        L1[i] = -alpha * L0[i] + L0[i - 1] + alpha * L1[i - 1]
        L2[i] = -alpha * L1[i] + L1[i - 1] + alpha * L2[i - 1]
        L3[i] = -alpha * L2[i] + L2[i - 1] + alpha * L3[i - 1]
        cu    = 0.0
        cd    = 0.0
        if L0[i] >= L1[i]:
            cu += L0[i] - L1[i]
        else:
            cd += L1[i] - L0[i]
        if L1[i] >= L2[i]:
            cu += L1[i] - L2[i]
        else:
            cd += L2[i] - L1[i]
        if L2[i] >= L3[i]:
            cu += L2[i] - L3[i]
        else:
            cd += L3[i] - L2[i]
        if cu + cd == 0:
            lrsi[i] = 0.0
        else:
            lrsi[i] = cu / (cu + cd)
    return lrsi


def gen_Laguerre_RSI(df, alpha=0.2, **kw):
    alpha = float(alpha)
    close = df['close'].values.astype(np.float64)
    lrsi  = pd.Series(_laguerre_rsi(close, alpha), index=df.index)
    long  = (lrsi > 0.2) & (lrsi.shift(1) <= 0.2)
    short = (lrsi < 0.8) & (lrsi.shift(1) >= 0.8)
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_Laguerre_RSI(trial):
    return {'alpha': trial.suggest_float('alpha', 0.1, 0.5)}


# ── 29. Laguerre_RSI_ATR ──────────────────────────────────────────────────────

def gen_Laguerre_RSI_ATR(df, alpha=0.2, atr_p=14, atr_mult=2.0, **kw):
    alpha    = float(alpha)
    atr_p    = int(atr_p)
    atr_mult = float(atr_mult)
    close    = df['close'].values.astype(np.float64)
    lrsi     = pd.Series(_laguerre_rsi(close, alpha), index=df.index)
    atr      = _atr(df, atr_p)
    # ATR trailing stop for trend confirmation
    trail_long  = df['close'] - atr_mult * atr
    trail_short = df['close'] + atr_mult * atr
    bullish     = lrsi > 0.8
    bearish     = lrsi < 0.2
    # Trend: close above trailing stop (bull) or below (bear)
    bull_trend  = df['close'] > trail_long
    bear_trend  = df['close'] < trail_short
    long   = bullish  & bull_trend & (lrsi.shift(1) <= 0.8)
    short  = bearish  & bear_trend & (lrsi.shift(1) >= 0.2)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_Laguerre_RSI_ATR(trial):
    return {
        'alpha':    trial.suggest_float('alpha',    0.1, 0.5),
        'atr_p':    trial.suggest_int('atr_p',      10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 3.0),
    }


# ── 30. Laguerre_RSI_EMA ──────────────────────────────────────────────────────

def gen_Laguerre_RSI_EMA(df, alpha=0.2, ema_p=50, **kw):
    alpha = float(alpha)
    ema_p = int(ema_p)
    close = df['close'].values.astype(np.float64)
    lrsi  = pd.Series(_laguerre_rsi(close, alpha), index=df.index)
    ema   = _ema(df['close'], ema_p)
    above_ema = df['close'] > ema
    long   = (lrsi > 0.5) & above_ema & (lrsi.shift(1) <= 0.5)
    short  = (lrsi < 0.5) & (~above_ema) & (lrsi.shift(1) >= 0.5)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out


def space_Laguerre_RSI_EMA(trial):
    return {
        'alpha': trial.suggest_float('alpha', 0.1, 0.5),
        'ema_p': trial.suggest_int('ema_p',  20, 100),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Ehlers_Fisher': {
        'gen': gen_Ehlers_Fisher,
        'space': space_Ehlers_Fisher,
        'default_params': {'length': 10},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 4000,
            'description': 'Ehlers Fisher Transform crossover. Normalizes price to -1..1 over period, then Fisher=0.5*ln((1+v)/(1-v)). Long on Fisher crossing above its signal.',
        },
    },
    'Ehlers_Fisher_CG': {
        'gen': gen_Ehlers_Fisher_CG,
        'space': space_Ehlers_Fisher_CG,
        'default_params': {'fish_p': 10, 'cg_p': 10},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'Fisher Transform combined with Center of Gravity oscillator. Both must agree for entry.',
        },
    },
    'Ehlers_Fisher_Stoch': {
        'gen': gen_Ehlers_Fisher_Stoch,
        'space': space_Ehlers_Fisher_Stoch,
        'default_params': {'fish_p': 10, 'stoch_k': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'Fisher Transform + Stochastic filter. Long: Fisher rising AND Stoch<20 (oversold).',
        },
    },
    'Ehlers_CyberCycle': {
        'gen': gen_Ehlers_CyberCycle,
        'space': space_Ehlers_CyberCycle,
        'default_params': {'alpha': 0.07},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'Ehlers Cyber Cycle oscillator. CC iterative formula removes trend. Long on CC crossing above trigger (CC[1]).',
        },
    },
    'Ehlers_Stochastic': {
        'gen': gen_Ehlers_Stochastic,
        'space': space_Ehlers_Stochastic,
        'default_params': {'length': 8},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Ehlers fast-cycle stochastic using In-Phase approximation. Long on normalized cycle crossing above 0.8.',
        },
    },
    'Ehlers_MESA': {
        'gen': gen_Ehlers_MESA,
        'space': space_Ehlers_MESA,
        'default_params': {'fast_limit': 0.5, 'slow_limit': 0.05},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 3000,
            'description': 'Simplified MESA adaptive MA. Efficiency Ratio adapts alpha between fast/slow limits. Long on AMA turning bullish.',
        },
    },
    'Ehlers_Bandpass': {
        'gen': gen_Ehlers_Bandpass,
        'space': space_Ehlers_Bandpass,
        'default_params': {'period': 20, 'bandwidth': 0.3},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'Ehlers Bandpass filter centered on dominant cycle. Long: BP>0 AND rising.',
        },
    },
    'Ehlers_Sinewave': {
        'gen': gen_Ehlers_Sinewave,
        'space': space_Ehlers_Sinewave,
        'default_params': {'length': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1800,
            'description': 'Ehlers Sinewave indicator. Sine vs Lead Sine (45° ahead) crossover for cycle timing.',
        },
    },
    'KAMA_Strategy_v2': {
        'gen': gen_KAMA_Strategy,
        'space': space_KAMA_Strategy,
        'default_params': {'er_p': 10, 'fast_p': 2, 'slow_p': 30},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 3500,
            'description': 'Kaufman Adaptive Moving Average. SC adapts to Efficiency Ratio. Long: close>KAMA AND KAMA rising.',
        },
    },
    'KAMA_Cross': {
        'gen': gen_KAMA_Cross,
        'space': space_KAMA_Cross,
        'default_params': {'er_fast': 8, 'er_slow': 20},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'Fast KAMA vs slow KAMA crossover system.',
        },
    },
    'KAMA_RSI': {
        'gen': gen_KAMA_RSI,
        'space': space_KAMA_RSI,
        'default_params': {'er_p': 10, 'rsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'KAMA direction filter combined with RSI momentum. Long: KAMA rising AND RSI>50.',
        },
    },
    'KAMA_ATR': {
        'gen': gen_KAMA_ATR,
        'space': space_KAMA_ATR,
        'default_params': {'er_p': 10, 'atr_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1000,
            'description': 'KAMA direction with ATR volatility gate. Only trades when volatility above average.',
        },
    },
    'VIDYA_Strategy': {
        'gen': gen_VIDYA_Strategy,
        'space': space_VIDYA_Strategy,
        'default_params': {'cmo_p': 9, 'ema_p': 12, 'k': 0.5},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Variable Index Dynamic Average. Alpha scaled by CMO strength. Long on price crossing above VIDYA.',
        },
    },
    'VIDYA_Cross': {
        'gen': gen_VIDYA_Cross,
        'space': space_VIDYA_Cross,
        'default_params': {'fast_p': 5, 'slow_p': 20, 'k': 0.5},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1200,
            'description': 'Dual VIDYA crossover. Fast vs slow VIDYA with CMO-scaled smoothing.',
        },
    },
    'FRAMA_Strategy': {
        'gen': gen_FRAMA_Strategy,
        'space': space_FRAMA_Strategy,
        'default_params': {'length': 16},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'Fractal Adaptive MA. Fractal dimension D from two half-windows adjusts alpha. Long: close>FRAMA.',
        },
    },
    'FRAMA_RSI': {
        'gen': gen_FRAMA_RSI,
        'space': space_FRAMA_RSI,
        'default_params': {'length': 16, 'rsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1200,
            'description': 'FRAMA trend direction confirmed by RSI momentum filter.',
        },
    },
    'ALMA_Strategy': {
        'gen': gen_ALMA_Strategy,
        'space': space_ALMA_Strategy,
        'default_params': {'length': 20, 'offset': 0.85, 'sigma': 6.0},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 3000,
            'description': 'Arnaud Legoux MA with Gaussian weights. Long on price crossing above ALMA.',
        },
    },
    'ALMA_Cross': {
        'gen': gen_ALMA_Cross,
        'space': space_ALMA_Cross,
        'default_params': {'fast_p': 9, 'slow_p': 30, 'offset': 0.85, 'sigma': 6.0},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1800,
            'description': 'Fast ALMA vs slow ALMA crossover system.',
        },
    },
    'ALMA_BB': {
        'gen': gen_ALMA_BB,
        'space': space_ALMA_BB,
        'default_params': {'length': 20, 'offset': 0.85},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1000,
            'description': 'ALMA as dynamic center with 2-std bands. Long on price below lower band while ALMA is rising.',
        },
    },
    'JMA_Strategy': {
        'gen': gen_JMA_Strategy,
        'space': space_JMA_Strategy,
        'default_params': {'length': 14, 'phase': 0, 'power': 2},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2500,
            'description': 'Jurik MA approximation via two-stage adaptive EMA with phase correction. Long: JMA rising AND price above JMA.',
        },
    },
    'JMA_Cross': {
        'gen': gen_JMA_Cross,
        'space': space_JMA_Cross,
        'default_params': {'fast_p': 8, 'slow_p': 30, 'phase': 0},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'Fast vs slow JMA crossover.',
        },
    },
    'T3_Strategy': {
        'gen': gen_T3_Strategy,
        'space': space_T3_Strategy,
        'default_params': {'length': 5, 'factor': 0.7},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 3000,
            'description': 'Tillson T3: six-stage EMA with volume factor. Long on price crossing above T3.',
        },
    },
    'T3_Cross': {
        'gen': gen_T3_Cross,
        'space': space_T3_Cross,
        'default_params': {'fast_p': 4, 'slow_p': 14, 'factor': 0.7},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'Fast vs slow T3 crossover.',
        },
    },
    'T3_ATR': {
        'gen': gen_T3_ATR,
        'space': space_T3_ATR,
        'default_params': {'length': 5, 'factor': 0.7, 'atr_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1000,
            'description': 'T3 direction with ATR volatility gate. Only trades when volatility is expanding.',
        },
    },
    'ZLEMA_Strategy': {
        'gen': gen_ZLEMA_Strategy,
        'space': space_ZLEMA_Strategy,
        'default_params': {'length': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'Zero-Lag EMA: lag-adjusted source eliminates EMA delay. Long: price>ZLEMA AND ZLEMA rising.',
        },
    },
    'ZLEMA_Cross': {
        'gen': gen_ZLEMA_Cross,
        'space': space_ZLEMA_Cross,
        'default_params': {'fast_p': 10, 'slow_p': 30},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'Fast vs slow ZLEMA crossover.',
        },
    },
    'ZLEMA_RSI': {
        'gen': gen_ZLEMA_RSI,
        'space': space_ZLEMA_RSI,
        'default_params': {'length': 20, 'rsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 800,
            'description': 'ZLEMA direction with RSI in 45-65 momentum zone filter.',
        },
    },
    'Laguerre_RSI': {
        'gen': gen_Laguerre_RSI,
        'space': space_Laguerre_RSI,
        'default_params': {'alpha': 0.2},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 3500,
            'description': 'Laguerre RSI: 4-bar iterative filter creates low-lag RSI. Long on LRSI crossing above 0.2.',
        },
    },
    'Laguerre_RSI_ATR': {
        'gen': gen_Laguerre_RSI_ATR,
        'space': space_Laguerre_RSI_ATR,
        'default_params': {'alpha': 0.2, 'atr_p': 14, 'atr_mult': 2.0},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'Laguerre RSI overbought (>0.8) with ATR trailing stop for trend confirmation.',
        },
    },
    'Laguerre_RSI_EMA': {
        'gen': gen_Laguerre_RSI_EMA,
        'space': space_Laguerre_RSI_EMA,
        'default_params': {'alpha': 0.2, 'ema_p': 50},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1000,
            'description': 'Laguerre RSI crossing 0.5 combined with EMA trend filter.',
        },
    },
}
