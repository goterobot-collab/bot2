#!/usr/bin/env python3
"""TV2 BATCH 25 — 30 estrategias MACD Advanced Variants 2026-04-01

  MACD_Signal_Cross      — MACD line crosses signal line (v4, ~8000L)
  MACD_Zero_Cross        — MACD crosses zero line (v4, ~5000L)
  MACD_Histogram         — Histogram direction change (v4, ~4000L)
  MACD_Hist_Zero         — Histogram crosses zero (v5, ~3000L)
  MACD_Classic           — Signal cross AND zero line confirmation (v4, ~3000L)
  MACD_Bull_Div          — Bull divergence: price LL + histogram HL (v5, ~3500L)
  MACD_Bear_Div          — Bear divergence: price HH + histogram LH (v5, ~3500L)
  MACD_Hidden_Bull       — Hidden bull div: price HL + MACD LL (v5, ~2000L)
  MACD_Div_EMA           — MACD div + EMA trend filter (v5, ~1500L)
  MACD_Fast              — Fast MACD scalping 3,10,16 (v4, ~2500L)
  MACD_Slow              — Slow MACD swing 24,52,18 (v4, ~2000L)
  MACD_Custom            — Fully custom periods (v5, ~2000L)
  MACD_EMA_Close         — MACD of EMA(close) vs WMA(close) (v4, ~1800L)
  MACD_RSI               — MACD signal cross + RSI filter (v4, ~4000L)
  MACD_BB                — MACD + Bollinger Bands (v5, ~2500L)
  MACD_Stoch             — MACD + Stochastic (v4, ~2500L)
  MACD_Volume            — MACD + volume confirmation (v5, ~2000L)
  MACD_ATR               — MACD + ATR filter (v4, ~2000L)
  MACD_SuperTrend        — MACD + SuperTrend alignment (v5, ~2500L)
  MACD_EMA_Align         — MACD + EMA alignment (v4, ~2000L)
  MACD_CCI               — MACD + CCI (v5, ~1500L)
  MACD_Vortex            — MACD + Vortex Indicator (v5, ~1200L)
  ZLMACD_Strategy        — Zero Lag MACD with DEMA (v5, ~3000L)
  MACD_V                 — Volume-adjusted MACD (v5, ~2500L)
  TEMA_MACD              — MACD using TEMA (v5, ~2000L)
  Hull_MACD              — MACD using Hull MA (v4, ~1800L)
  MACD_Saucer            — Histogram saucer pattern (v4, ~2000L)
  MACD_Squeeze_Release   — Histogram squeeze then expansion (v5, ~1500L)
  MACD_Momentum_Shift    — Histogram peak and reversal (v5, ~1500L)
  MACD_Triple            — Three MACDs alignment (v5, ~2000L)
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
        (df['low'] - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, p)


def _rsi(s, p):
    d = s.diff()
    g = d.clip(lower=0)
    lo = (-d).clip(lower=0)
    return 100 - 100 / (1 + _rma(g, p) / _rma(lo, p).replace(0, 1e-9))


def _macd(s, fast_p, slow_p, sig_p):
    macd = _ema(s, fast_p) - _ema(s, slow_p)
    signal = _ema(macd, sig_p)
    return macd, signal, macd - signal


def _wma(s, p):
    p = int(p)
    weights = np.arange(1, p + 1, dtype=float)
    return s.rolling(p, min_periods=1).apply(
        lambda x: np.dot(x[-len(weights):], weights[-len(x):]) / weights[-len(x):].sum(),
        raw=True
    )


def _hma(s, p):
    p = int(p)
    half = max(int(p / 2), 1)
    sqrt_p = max(int(np.sqrt(p)), 1)
    return _wma(2 * _wma(s, half) - _wma(s, p), sqrt_p)


def _dema(s, p):
    e1 = _ema(s, p)
    e2 = _ema(e1, p)
    return 2 * e1 - e2


def _tema(s, p):
    e1 = _ema(s, p)
    e2 = _ema(e1, p)
    e3 = _ema(e2, p)
    return 3 * e1 - 3 * e2 + e3


def _bb(s, p, mult):
    mid = _sma(s, p)
    std = s.rolling(int(p), min_periods=1).std().fillna(0)
    return mid + mult * std, mid, mid - mult * std


def _stoch(df, k_p, d_p):
    lo = df['low'].rolling(int(k_p), min_periods=1).min()
    hi = df['high'].rolling(int(k_p), min_periods=1).max()
    k = 100 * (df['close'] - lo) / (hi - lo + 1e-9)
    d = _sma(k, d_p)
    return k, d


def _cci(df, p):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = _sma(tp, p)
    mad = tp.rolling(int(p), min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).fillna(1e-9)
    return (tp - sma_tp) / (0.015 * mad)


def _supertrend(df, p, mult):
    atr = _atr(df, p)
    hl2 = (df['high'] + df['low']) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    st = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        prev_upper = upper.iloc[i - 1]
        prev_lower = lower.iloc[i - 1]
        curr_close = df['close'].iloc[i]

        final_upper = upper.iloc[i] if upper.iloc[i] < prev_upper or df['close'].iloc[i - 1] > prev_upper else prev_upper
        final_lower = lower.iloc[i] if lower.iloc[i] > prev_lower or df['close'].iloc[i - 1] < prev_lower else prev_lower

        if direction.iloc[i - 1] == 1 and curr_close < final_lower:
            direction.iloc[i] = -1
        elif direction.iloc[i - 1] == -1 and curr_close > final_upper:
            direction.iloc[i] = 1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        st.iloc[i] = final_lower if direction.iloc[i] == 1 else final_upper

    return direction


# ── 1. MACD_Signal_Cross ──────────────────────────────────────────────────────

def gen_MACD_Signal_Cross(df, fast=12, slow=26, sig=9, **kw):
    macd, signal, _ = _macd(df['close'], fast, slow, sig)
    prev_macd = macd.shift(1)
    prev_sig = signal.shift(1)
    cross_up = (prev_macd <= prev_sig) & (macd > signal)
    cross_dn = (prev_macd >= prev_sig) & (macd < signal)
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up] = 1
    sig_s[cross_dn] = -1
    return sig_s.fillna(0)


def space_MACD_Signal_Cross(trial):
    return {
        'fast': trial.suggest_int('fast', 8, 16),
        'slow': trial.suggest_int('slow', 20, 30),
        'sig':  trial.suggest_int('sig', 5, 12),
    }


# ── 2. MACD_Zero_Cross ───────────────────────────────────────────────────────

def gen_MACD_Zero_Cross(df, fast=12, slow=26, sig=9, **kw):
    macd, _, _ = _macd(df['close'], fast, slow, sig)
    prev = macd.shift(1)
    cross_up = (prev <= 0) & (macd > 0)
    cross_dn = (prev >= 0) & (macd < 0)
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up] = 1
    sig_s[cross_dn] = -1
    return sig_s.fillna(0)


def space_MACD_Zero_Cross(trial):
    return {
        'fast': trial.suggest_int('fast', 8, 16),
        'slow': trial.suggest_int('slow', 20, 30),
        'sig':  trial.suggest_int('sig', 5, 12),
    }


# ── 3. MACD_Histogram ────────────────────────────────────────────────────────

def gen_MACD_Histogram(df, fast=12, slow=26, sig=9, **kw):
    _, _, hist = _macd(df['close'], fast, slow, sig)
    prev_hist = hist.shift(1)
    sig_s = pd.Series(0, index=df.index)
    sig_s[(hist > 0) & (hist > prev_hist)] = 1
    sig_s[(hist < 0) & (hist < prev_hist)] = -1
    return sig_s.fillna(0)


def space_MACD_Histogram(trial):
    return {
        'fast': trial.suggest_int('fast', 8, 16),
        'slow': trial.suggest_int('slow', 20, 30),
        'sig':  trial.suggest_int('sig', 5, 12),
    }


# ── 4. MACD_Hist_Zero ────────────────────────────────────────────────────────

def gen_MACD_Hist_Zero(df, fast=12, slow=26, sig=9, **kw):
    _, _, hist = _macd(df['close'], fast, slow, sig)
    prev = hist.shift(1)
    cross_up = (prev <= 0) & (hist > 0)
    cross_dn = (prev >= 0) & (hist < 0)
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up] = 1
    sig_s[cross_dn] = -1
    return sig_s.fillna(0)


def space_MACD_Hist_Zero(trial):
    return {
        'fast': trial.suggest_int('fast', 8, 16),
        'slow': trial.suggest_int('slow', 20, 30),
        'sig':  trial.suggest_int('sig', 5, 12),
    }


# ── 5. MACD_Classic ──────────────────────────────────────────────────────────

def gen_MACD_Classic(df, fast=12, slow=26, sig=9, **kw):
    macd, signal, _ = _macd(df['close'], fast, slow, sig)
    prev_macd = macd.shift(1)
    prev_sig = signal.shift(1)
    cross_up = (prev_macd <= prev_sig) & (macd > signal)
    cross_dn = (prev_macd >= prev_sig) & (macd < signal)
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up & (macd > 0)] = 1
    sig_s[cross_dn & (macd < 0)] = -1
    return sig_s.fillna(0)


def space_MACD_Classic(trial):
    return {
        'fast': trial.suggest_int('fast', 8, 16),
        'slow': trial.suggest_int('slow', 20, 30),
        'sig':  trial.suggest_int('sig', 5, 12),
    }


# ── 6. MACD_Bull_Div ─────────────────────────────────────────────────────────

def gen_MACD_Bull_Div(df, fast=12, slow=26, sig=9, lookback=10, **kw):
    _, _, hist = _macd(df['close'], fast, slow, sig)
    lb = int(lookback)
    close = df['close']
    sig_s = pd.Series(0, index=df.index)
    for i in range(lb, len(df)):
        window_close = close.iloc[i - lb:i + 1]
        window_hist = hist.iloc[i - lb:i + 1]
        if close.iloc[i] == window_close.min() and hist.iloc[i] > window_hist.min():
            prev_low_idx = window_close.idxmin()
            if prev_low_idx != close.index[i]:
                sig_s.iloc[i] = 1
    return sig_s.fillna(0)


def space_MACD_Bull_Div(trial):
    return {
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig':      trial.suggest_int('sig', 5, 12),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 7. MACD_Bear_Div ─────────────────────────────────────────────────────────

def gen_MACD_Bear_Div(df, fast=12, slow=26, sig=9, lookback=10, **kw):
    _, _, hist = _macd(df['close'], fast, slow, sig)
    lb = int(lookback)
    close = df['close']
    sig_s = pd.Series(0, index=df.index)
    for i in range(lb, len(df)):
        window_close = close.iloc[i - lb:i + 1]
        window_hist = hist.iloc[i - lb:i + 1]
        if close.iloc[i] == window_close.max() and hist.iloc[i] < window_hist.max():
            prev_high_idx = window_close.idxmax()
            if prev_high_idx != close.index[i]:
                sig_s.iloc[i] = -1
    return sig_s.fillna(0)


def space_MACD_Bear_Div(trial):
    return {
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig':      trial.suggest_int('sig', 5, 12),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 8. MACD_Hidden_Bull ──────────────────────────────────────────────────────

def gen_MACD_Hidden_Bull(df, fast=12, slow=26, sig=9, lookback=10, **kw):
    macd, _, _ = _macd(df['close'], fast, slow, sig)
    lb = int(lookback)
    close = df['close']
    sig_s = pd.Series(0, index=df.index)
    for i in range(lb, len(df)):
        window_close = close.iloc[i - lb:i + 1]
        window_macd = macd.iloc[i - lb:i + 1]
        # Hidden bull: price makes higher low, MACD makes lower low
        if close.iloc[i] > window_close.min() and macd.iloc[i] == window_macd.min():
            sig_s.iloc[i] = 1
    return sig_s.fillna(0)


def space_MACD_Hidden_Bull(trial):
    return {
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig':      trial.suggest_int('sig', 5, 12),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# ── 9. MACD_Div_EMA ──────────────────────────────────────────────────────────

def gen_MACD_Div_EMA(df, fast=12, slow=26, sig=9, lookback=10, ema_p=100, **kw):
    _, _, hist = _macd(df['close'], fast, slow, sig)
    ema = _ema(df['close'], ema_p)
    lb = int(lookback)
    close = df['close']
    sig_s = pd.Series(0, index=df.index)
    for i in range(lb, len(df)):
        window_close = close.iloc[i - lb:i + 1]
        window_hist = hist.iloc[i - lb:i + 1]
        # Bull div: price LL + hist HL
        if close.iloc[i] == window_close.min() and hist.iloc[i] > window_hist.min():
            if close.iloc[i] > ema.iloc[i]:
                sig_s.iloc[i] = 1
        # Bear div: price HH + hist LH
        if close.iloc[i] == window_close.max() and hist.iloc[i] < window_hist.max():
            if close.iloc[i] < ema.iloc[i]:
                sig_s.iloc[i] = -1
    return sig_s.fillna(0)


def space_MACD_Div_EMA(trial):
    return {
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig':      trial.suggest_int('sig', 5, 12),
        'lookback': trial.suggest_int('lookback', 5, 20),
        'ema_p':    trial.suggest_int('ema_p', 50, 200),
    }


# ── 10. MACD_Fast ────────────────────────────────────────────────────────────

def gen_MACD_Fast(df, fast=3, slow=10, sig=6, **kw):
    macd, signal, _ = _macd(df['close'], fast, slow, sig)
    prev_macd = macd.shift(1)
    prev_sig = signal.shift(1)
    cross_up = (prev_macd <= prev_sig) & (macd > signal)
    cross_dn = (prev_macd >= prev_sig) & (macd < signal)
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up & (macd > 0)] = 1
    sig_s[cross_dn & (macd < 0)] = -1
    return sig_s.fillna(0)


def space_MACD_Fast(trial):
    return {
        'fast': trial.suggest_int('fast', 2, 6),
        'slow': trial.suggest_int('slow', 7, 15),
        'sig':  trial.suggest_int('sig', 4, 9),
    }


# ── 11. MACD_Slow ────────────────────────────────────────────────────────────

def gen_MACD_Slow(df, fast=24, slow=52, sig=18, **kw):
    macd, signal, _ = _macd(df['close'], fast, slow, sig)
    prev_macd = macd.shift(1)
    prev_sig = signal.shift(1)
    cross_up = (prev_macd <= prev_sig) & (macd > signal)
    cross_dn = (prev_macd >= prev_sig) & (macd < signal)
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up] = 1
    sig_s[cross_dn] = -1
    return sig_s.fillna(0)


def space_MACD_Slow(trial):
    return {
        'fast': trial.suggest_int('fast', 18, 30),
        'slow': trial.suggest_int('slow', 40, 60),
        'sig':  trial.suggest_int('sig', 12, 20),
    }


# ── 12. MACD_Custom ──────────────────────────────────────────────────────────

def gen_MACD_Custom(df, fast=12, slow=26, sig=9, **kw):
    macd, signal, hist = _macd(df['close'], fast, slow, sig)
    prev_macd = macd.shift(1)
    prev_sig = signal.shift(1)
    cross_up = (prev_macd <= prev_sig) & (macd > signal)
    cross_dn = (prev_macd >= prev_sig) & (macd < signal)
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up] = 1
    sig_s[cross_dn] = -1
    return sig_s.fillna(0)


def space_MACD_Custom(trial):
    return {
        'fast': trial.suggest_int('fast', 5, 20),
        'slow': trial.suggest_int('slow', 15, 50),
        'sig':  trial.suggest_int('sig', 5, 15),
    }


# ── 13. MACD_EMA_Close ───────────────────────────────────────────────────────

def gen_MACD_EMA_Close(df, fast=12, slow=26, sig=9, **kw):
    # MACD of EMA(close) source
    ema_src = _ema(df['close'], max(int(fast) // 2, 2))
    wma_src = _wma(df['close'], max(int(slow) // 2, 2))
    # Use EMA as fast source, WMA as slow source
    macd_line = _ema(ema_src, fast) - _ema(wma_src, slow)
    signal_line = _ema(macd_line, sig)
    prev_macd = macd_line.shift(1)
    prev_sig = signal_line.shift(1)
    cross_up = (prev_macd <= prev_sig) & (macd_line > signal_line)
    cross_dn = (prev_macd >= prev_sig) & (macd_line < signal_line)
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up] = 1
    sig_s[cross_dn] = -1
    return sig_s.fillna(0)


def space_MACD_EMA_Close(trial):
    return {
        'fast': trial.suggest_int('fast', 8, 16),
        'slow': trial.suggest_int('slow', 20, 30),
        'sig':  trial.suggest_int('sig', 5, 12),
    }


# ── 14. MACD_RSI ─────────────────────────────────────────────────────────────

def gen_MACD_RSI(df, fast=12, slow=26, sig=9, rsi_p=14, **kw):
    macd, signal, _ = _macd(df['close'], fast, slow, sig)
    rsi = _rsi(df['close'], rsi_p)
    prev_macd = macd.shift(1)
    prev_sig = signal.shift(1)
    cross_up = (prev_macd <= prev_sig) & (macd > signal)
    cross_dn = (prev_macd >= prev_sig) & (macd < signal)
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up & (rsi < 70)] = 1
    sig_s[cross_dn & (rsi > 30)] = -1
    return sig_s.fillna(0)


def space_MACD_RSI(trial):
    return {
        'fast':  trial.suggest_int('fast', 8, 16),
        'slow':  trial.suggest_int('slow', 20, 30),
        'sig':   trial.suggest_int('sig', 5, 12),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 15. MACD_BB ──────────────────────────────────────────────────────────────

def gen_MACD_BB(df, fast=12, slow=26, sig=9, bb_p=20, bb_mult=2.0, **kw):
    macd, signal, _ = _macd(df['close'], fast, slow, sig)
    upper, mid, lower = _bb(df['close'], bb_p, bb_mult)
    sig_s = pd.Series(0, index=df.index)
    # Long: MACD > signal AND close < BB lower (momentum + oversold)
    sig_s[(macd > signal) & (df['close'] < lower)] = 1
    # Short: MACD < signal AND close > BB upper (momentum + overbought)
    sig_s[(macd < signal) & (df['close'] > upper)] = -1
    return sig_s.fillna(0)


def space_MACD_BB(trial):
    return {
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig':      trial.suggest_int('sig', 5, 12),
        'bb_p':     trial.suggest_int('bb_p', 15, 30),
        'bb_mult':  trial.suggest_float('bb_mult', 1.5, 2.5),
    }


# ── 16. MACD_Stoch ───────────────────────────────────────────────────────────

def gen_MACD_Stoch(df, fast=12, slow=26, sig=9, stoch_k=14, stoch_d=3, **kw):
    macd, signal, _ = _macd(df['close'], fast, slow, sig)
    k, d = _stoch(df, stoch_k, stoch_d)
    sig_s = pd.Series(0, index=df.index)
    sig_s[(macd > signal) & (k < 20)] = 1
    sig_s[(macd < signal) & (k > 80)] = -1
    return sig_s.fillna(0)


def space_MACD_Stoch(trial):
    return {
        'fast':    trial.suggest_int('fast', 8, 16),
        'slow':    trial.suggest_int('slow', 20, 30),
        'sig':     trial.suggest_int('sig', 5, 12),
        'stoch_k': trial.suggest_int('stoch_k', 7, 21),
        'stoch_d': trial.suggest_int('stoch_d', 3, 7),
    }


# ── 17. MACD_Volume ──────────────────────────────────────────────────────────

def gen_MACD_Volume(df, fast=12, slow=26, sig=9, vol_p=30, vol_mult=1.5, **kw):
    macd, signal, _ = _macd(df['close'], fast, slow, sig)
    vol = df['volume']
    vol_sma = _sma(vol, vol_p)
    prev_macd = macd.shift(1)
    prev_sig = signal.shift(1)
    cross_up = (prev_macd <= prev_sig) & (macd > signal)
    cross_dn = (prev_macd >= prev_sig) & (macd < signal)
    vol_ok = vol > vol_sma * vol_mult
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up & vol_ok] = 1
    sig_s[cross_dn & vol_ok] = -1
    return sig_s.fillna(0)


def space_MACD_Volume(trial):
    return {
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig':      trial.suggest_int('sig', 5, 12),
        'vol_p':    trial.suggest_int('vol_p', 20, 50),
        'vol_mult': trial.suggest_float('vol_mult', 1.2, 2.5),
    }


# ── 18. MACD_ATR ─────────────────────────────────────────────────────────────

def gen_MACD_ATR(df, fast=12, slow=26, sig=9, atr_p=14, **kw):
    macd, signal, _ = _macd(df['close'], fast, slow, sig)
    atr = _atr(df, atr_p)
    atr_prev = atr.shift(1)
    atr_increasing = atr > atr_prev
    sig_s = pd.Series(0, index=df.index)
    sig_s[(macd > signal) & atr_increasing] = 1
    sig_s[(macd < signal) & atr_increasing] = -1
    return sig_s.fillna(0)


def space_MACD_ATR(trial):
    return {
        'fast':  trial.suggest_int('fast', 8, 16),
        'slow':  trial.suggest_int('slow', 20, 30),
        'sig':   trial.suggest_int('sig', 5, 12),
        'atr_p': trial.suggest_int('atr_p', 10, 20),
    }


# ── 19. MACD_SuperTrend ──────────────────────────────────────────────────────

def gen_MACD_SuperTrend(df, fast=12, slow=26, sig=9, st_p=14, st_mult=3.0, **kw):
    macd, signal, _ = _macd(df['close'], fast, slow, sig)
    direction = _supertrend(df, st_p, st_mult)
    sig_s = pd.Series(0, index=df.index)
    sig_s[(direction == 1) & (macd > signal)] = 1
    sig_s[(direction == -1) & (macd < signal)] = -1
    return sig_s.fillna(0)


def space_MACD_SuperTrend(trial):
    return {
        'fast':    trial.suggest_int('fast', 8, 16),
        'slow':    trial.suggest_int('slow', 20, 30),
        'sig':     trial.suggest_int('sig', 5, 12),
        'st_p':    trial.suggest_int('st_p', 7, 21),
        'st_mult': trial.suggest_float('st_mult', 2.0, 4.0),
    }


# ── 20. MACD_EMA_Align ───────────────────────────────────────────────────────

def gen_MACD_EMA_Align(df, fast=12, slow=26, sig=9, ema_fast=20, ema_slow=100, **kw):
    macd, signal, _ = _macd(df['close'], fast, slow, sig)
    ef = _ema(df['close'], ema_fast)
    es = _ema(df['close'], ema_slow)
    sig_s = pd.Series(0, index=df.index)
    long_cond = (macd > 0) & (df['close'] > ef) & (ef > es)
    short_cond = (macd < 0) & (df['close'] < ef) & (ef < es)
    sig_s[long_cond] = 1
    sig_s[short_cond] = -1
    return sig_s.fillna(0)


def space_MACD_EMA_Align(trial):
    return {
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig':      trial.suggest_int('sig', 5, 12),
        'ema_fast': trial.suggest_int('ema_fast', 10, 30),
        'ema_slow': trial.suggest_int('ema_slow', 50, 200),
    }


# ── 21. MACD_CCI ─────────────────────────────────────────────────────────────

def gen_MACD_CCI(df, fast=12, slow=26, sig=9, cci_p=14, **kw):
    macd, signal, _ = _macd(df['close'], fast, slow, sig)
    cci = _cci(df, cci_p)
    sig_s = pd.Series(0, index=df.index)
    sig_s[(macd > signal) & (cci < -100)] = 1
    sig_s[(macd < signal) & (cci > 100)] = -1
    return sig_s.fillna(0)


def space_MACD_CCI(trial):
    return {
        'fast':  trial.suggest_int('fast', 8, 16),
        'slow':  trial.suggest_int('slow', 20, 30),
        'sig':   trial.suggest_int('sig', 5, 12),
        'cci_p': trial.suggest_int('cci_p', 10, 20),
    }


# ── 22. MACD_Vortex ──────────────────────────────────────────────────────────

def gen_MACD_Vortex(df, fast=12, slow=26, sig=9, vortex_p=14, **kw):
    macd, _, _ = _macd(df['close'], fast, slow, sig)
    p = int(vortex_p)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    vm_plus = (df['high'] - df['low'].shift(1)).abs()
    vm_minus = (df['low'] - df['high'].shift(1)).abs()
    tr_sum = tr.rolling(p, min_periods=1).sum()
    vi_plus = vm_plus.rolling(p, min_periods=1).sum() / tr_sum.replace(0, 1e-9)
    vi_minus = vm_minus.rolling(p, min_periods=1).sum() / tr_sum.replace(0, 1e-9)
    sig_s = pd.Series(0, index=df.index)
    sig_s[(macd > 0) & (vi_plus > vi_minus)] = 1
    sig_s[(macd < 0) & (vi_minus > vi_plus)] = -1
    return sig_s.fillna(0)


def space_MACD_Vortex(trial):
    return {
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig':      trial.suggest_int('sig', 5, 12),
        'vortex_p': trial.suggest_int('vortex_p', 14, 20),
    }


# ── 23. ZLMACD_Strategy ──────────────────────────────────────────────────────

def gen_ZLMACD_Strategy(df, fast=12, slow=26, sig=9, **kw):
    zl_fast = _dema(df['close'], fast)
    zl_slow = _dema(df['close'], slow)
    zl_macd = zl_fast - zl_slow
    signal = _ema(zl_macd, sig)
    prev_macd = zl_macd.shift(1)
    prev_sig = signal.shift(1)
    cross_up = (prev_macd <= prev_sig) & (zl_macd > signal)
    cross_dn = (prev_macd >= prev_sig) & (zl_macd < signal)
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up] = 1
    sig_s[cross_dn] = -1
    return sig_s.fillna(0)


def space_ZLMACD_Strategy(trial):
    return {
        'fast': trial.suggest_int('fast', 8, 16),
        'slow': trial.suggest_int('slow', 20, 30),
        'sig':  trial.suggest_int('sig', 5, 12),
    }


# ── 24. MACD_V ───────────────────────────────────────────────────────────────

def gen_MACD_V(df, fast=12, slow=26, sig=9, vol_p=30, **kw):
    macd, signal, _ = _macd(df['close'], fast, slow, sig)
    vol = df['volume']
    vol_mean = _sma(vol, vol_p).replace(0, 1e-9)
    vol_ratio = vol / vol_mean
    macd_v = macd * vol_ratio
    signal_v = _ema(macd_v, sig)
    prev_mv = macd_v.shift(1)
    prev_sv = signal_v.shift(1)
    cross_up = (prev_mv <= prev_sv) & (macd_v > signal_v) & (macd_v > 0)
    cross_dn = (prev_mv >= prev_sv) & (macd_v < signal_v) & (macd_v < 0)
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up] = 1
    sig_s[cross_dn] = -1
    return sig_s.fillna(0)


def space_MACD_V(trial):
    return {
        'fast':  trial.suggest_int('fast', 8, 16),
        'slow':  trial.suggest_int('slow', 20, 30),
        'sig':   trial.suggest_int('sig', 5, 12),
        'vol_p': trial.suggest_int('vol_p', 20, 50),
    }


# ── 25. TEMA_MACD ────────────────────────────────────────────────────────────

def gen_TEMA_MACD(df, fast=8, slow=21, sig=6, **kw):
    tema_fast = _tema(df['close'], fast)
    tema_slow = _tema(df['close'], slow)
    macd_line = tema_fast - tema_slow
    signal_line = _ema(macd_line, sig)
    prev_macd = macd_line.shift(1)
    prev_sig = signal_line.shift(1)
    cross_up = (prev_macd <= prev_sig) & (macd_line > signal_line)
    cross_dn = (prev_macd >= prev_sig) & (macd_line < signal_line)
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up] = 1
    sig_s[cross_dn] = -1
    return sig_s.fillna(0)


def space_TEMA_MACD(trial):
    return {
        'fast': trial.suggest_int('fast', 5, 14),
        'slow': trial.suggest_int('slow', 15, 28),
        'sig':  trial.suggest_int('sig', 4, 10),
    }


# ── 26. Hull_MACD ────────────────────────────────────────────────────────────

def gen_Hull_MACD(df, fast=9, slow=21, sig=6, **kw):
    hma_fast = _hma(df['close'], fast)
    hma_slow = _hma(df['close'], slow)
    macd_line = hma_fast - hma_slow
    signal_line = _ema(macd_line, sig)
    prev_macd = macd_line.shift(1)
    prev_sig = signal_line.shift(1)
    cross_up = (prev_macd <= prev_sig) & (macd_line > signal_line)
    cross_dn = (prev_macd >= prev_sig) & (macd_line < signal_line)
    sig_s = pd.Series(0, index=df.index)
    sig_s[cross_up] = 1
    sig_s[cross_dn] = -1
    return sig_s.fillna(0)


def space_Hull_MACD(trial):
    return {
        'fast': trial.suggest_int('fast', 5, 14),
        'slow': trial.suggest_int('slow', 15, 30),
        'sig':  trial.suggest_int('sig', 4, 10),
    }


# ── 27. MACD_Saucer ──────────────────────────────────────────────────────────

def gen_MACD_Saucer(df, fast=12, slow=26, sig=9, **kw):
    _, _, hist = _macd(df['close'], fast, slow, sig)
    h0 = hist
    h1 = hist.shift(1)
    h2 = hist.shift(2)
    # Bull saucer: all > 0, h2 > h1 < h0 (dip then recovery)
    bull_saucer = (h2 > 0) & (h1 > 0) & (h0 > 0) & (h2 > h1) & (h0 > h1)
    # Bear saucer: all < 0, h2 < h1 > h0 (peak then drop)
    bear_saucer = (h2 < 0) & (h1 < 0) & (h0 < 0) & (h2 < h1) & (h0 < h1)
    sig_s = pd.Series(0, index=df.index)
    sig_s[bull_saucer] = 1
    sig_s[bear_saucer] = -1
    return sig_s.fillna(0)


def space_MACD_Saucer(trial):
    return {
        'fast': trial.suggest_int('fast', 8, 16),
        'slow': trial.suggest_int('slow', 20, 30),
        'sig':  trial.suggest_int('sig', 5, 12),
    }


# ── 28. MACD_Squeeze_Release ─────────────────────────────────────────────────

def gen_MACD_Squeeze_Release(df, fast=12, slow=26, sig=9, squeeze_bars=5, **kw):
    _, _, hist = _macd(df['close'], fast, slow, sig)
    sb = int(squeeze_bars)
    abs_hist = hist.abs()
    # Rolling min of abs histogram for squeeze detection
    rolling_min = abs_hist.rolling(sb, min_periods=1).mean()
    rolling_max = abs_hist.rolling(sb * 3, min_periods=1).mean()
    # Squeeze: current abs_hist is low relative to recent range
    in_squeeze = abs_hist < rolling_max * 0.5
    # Release: was in squeeze, now expanding
    prev_squeeze = in_squeeze.shift(1).fillna(0).astype(bool)
    expanding_up = prev_squeeze & (hist > 0) & (hist > hist.shift(1))
    expanding_dn = prev_squeeze & (hist < 0) & (hist < hist.shift(1))
    sig_s = pd.Series(0, index=df.index)
    sig_s[expanding_up] = 1
    sig_s[expanding_dn] = -1
    return sig_s.fillna(0)


def space_MACD_Squeeze_Release(trial):
    return {
        'fast':         trial.suggest_int('fast', 8, 16),
        'slow':         trial.suggest_int('slow', 20, 30),
        'sig':          trial.suggest_int('sig', 5, 12),
        'squeeze_bars': trial.suggest_int('squeeze_bars', 3, 8),
    }


# ── 29. MACD_Momentum_Shift ──────────────────────────────────────────────────

def gen_MACD_Momentum_Shift(df, fast=12, slow=26, sig=9, lookback=5, **kw):
    _, _, hist = _macd(df['close'], fast, slow, sig)
    lb = int(lookback)
    # Rolling max and min of histogram
    rolling_max = hist.rolling(lb, min_periods=1).max()
    rolling_min = hist.rolling(lb, min_periods=1).min()
    prev_hist = hist.shift(1)
    # Short: histogram was at peak (== rolling max) and now decreasing from high level
    at_peak = (prev_hist == rolling_max.shift(1)) & (hist < prev_hist) & (prev_hist > 0)
    # Long: histogram was at trough (== rolling min) and now increasing from low level
    at_trough = (prev_hist == rolling_min.shift(1)) & (hist > prev_hist) & (prev_hist < 0)
    sig_s = pd.Series(0, index=df.index)
    sig_s[at_trough] = 1
    sig_s[at_peak] = -1
    return sig_s.fillna(0)


def space_MACD_Momentum_Shift(trial):
    return {
        'fast':     trial.suggest_int('fast', 8, 16),
        'slow':     trial.suggest_int('slow', 20, 30),
        'sig':      trial.suggest_int('sig', 5, 12),
        'lookback': trial.suggest_int('lookback', 3, 10),
    }


# ── 30. MACD_Triple ──────────────────────────────────────────────────────────

def gen_MACD_Triple(df, fast1=3, slow1=10, fast2=12, slow2=26, fast3=24, slow3=52, **kw):
    _, _, hist1 = _macd(df['close'], fast1, slow1, 8)
    _, _, hist2 = _macd(df['close'], fast2, slow2, 9)
    _, _, hist3 = _macd(df['close'], fast3, slow3, 18)
    sig_s = pd.Series(0, index=df.index)
    # Long: all three histograms positive
    sig_s[(hist1 > 0) & (hist2 > 0) & (hist3 > 0)] = 1
    # Short: all three histograms negative
    sig_s[(hist1 < 0) & (hist2 < 0) & (hist3 < 0)] = -1
    return sig_s.fillna(0)


def space_MACD_Triple(trial):
    return {
        'fast1': trial.suggest_int('fast1', 2, 6),
        'slow1': trial.suggest_int('slow1', 8, 15),
        'fast2': trial.suggest_int('fast2', 8, 14),
        'slow2': trial.suggest_int('slow2', 20, 30),
        'fast3': trial.suggest_int('fast3', 20, 28),
        'slow3': trial.suggest_int('slow3', 44, 60),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'MACD_Signal_Cross': {
        'gen': gen_MACD_Signal_Cross,
        'space': space_MACD_Signal_Cross,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 8000,
                 'description': 'MACD line crosses signal line. Long: crosses above. Short: crosses below.'},
    },
    'MACD_Zero_Cross': {
        'gen': gen_MACD_Zero_Cross,
        'space': space_MACD_Zero_Cross,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 5000,
                 'description': 'MACD crosses zero line. Long: MACD crosses above 0.'},
    },
    'MACD_Histogram': {
        'gen': gen_MACD_Histogram,
        'space': space_MACD_Histogram,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 4000,
                 'description': 'Histogram direction change. Long: hist turns positive and increasing.'},
    },
    'MACD_Hist_Zero': {
        'gen': gen_MACD_Hist_Zero,
        'space': space_MACD_Hist_Zero,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Histogram crosses zero line bidirectionally.'},
    },
    'MACD_Classic': {
        'gen': gen_MACD_Classic,
        'space': space_MACD_Classic,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'Full: signal cross AND zero line confirmation.'},
    },
    'MACD_Bull_Div': {
        'gen': gen_MACD_Bull_Div,
        'space': space_MACD_Bull_Div,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Bull divergence: price LL + MACD histogram HL.'},
    },
    'MACD_Bear_Div': {
        'gen': gen_MACD_Bear_Div,
        'space': space_MACD_Bear_Div,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Bear divergence: price HH + MACD histogram LH.'},
    },
    'MACD_Hidden_Bull': {
        'gen': gen_MACD_Hidden_Bull,
        'space': space_MACD_Hidden_Bull,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Hidden bull divergence: price HL + MACD LL. Continuation long.'},
    },
    'MACD_Div_EMA': {
        'gen': gen_MACD_Div_EMA,
        'space': space_MACD_Div_EMA,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'lookback': 10, 'ema_p': 100},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'MACD divergence + EMA trend filter.'},
    },
    'MACD_Fast': {
        'gen': gen_MACD_Fast,
        'space': space_MACD_Fast,
        'default_params': {'fast': 3, 'slow': 10, 'sig': 6},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'Fast MACD for scalping: 3,10,16. Signal cross above 0.'},
    },
    'MACD_Slow': {
        'gen': gen_MACD_Slow,
        'space': space_MACD_Slow,
        'default_params': {'fast': 24, 'slow': 52, 'sig': 18},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'Slow MACD for swing: 24,52,18. Signal cross.'},
    },
    'MACD_Custom': {
        'gen': gen_MACD_Custom,
        'space': space_MACD_Custom,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Fully custom MACD periods, signal cross bidirectional.'},
    },
    'MACD_EMA_Close': {
        'gen': gen_MACD_EMA_Close,
        'space': space_MACD_EMA_Close,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1800,
                 'description': 'MACD of EMA(close) vs WMA(close) source.'},
    },
    'MACD_RSI': {
        'gen': gen_MACD_RSI,
        'space': space_MACD_RSI,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 4000,
                 'description': 'MACD signal cross + RSI filter. Long: cross AND RSI<70.'},
    },
    'MACD_BB': {
        'gen': gen_MACD_BB,
        'space': space_MACD_BB,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'bb_p': 20, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'MACD + BB. Long: MACD>signal AND close<BB_lower.'},
    },
    'MACD_Stoch': {
        'gen': gen_MACD_Stoch,
        'space': space_MACD_Stoch,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'stoch_k': 14, 'stoch_d': 3},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'MACD + Stochastic. Long: MACD>signal AND Stoch<20.'},
    },
    'MACD_Volume': {
        'gen': gen_MACD_Volume,
        'space': space_MACD_Volume,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'vol_p': 30, 'vol_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'MACD cross + volume above average confirmation.'},
    },
    'MACD_ATR': {
        'gen': gen_MACD_ATR,
        'space': space_MACD_ATR,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'MACD + ATR filter. Long: MACD>signal AND ATR increasing.'},
    },
    'MACD_SuperTrend': {
        'gen': gen_MACD_SuperTrend,
        'space': space_MACD_SuperTrend,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'st_p': 14, 'st_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'MACD + SuperTrend alignment. Long: ST bullish AND MACD>signal.'},
    },
    'MACD_EMA_Align': {
        'gen': gen_MACD_EMA_Align,
        'space': space_MACD_EMA_Align,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'ema_fast': 20, 'ema_slow': 100},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'MACD + EMA alignment. Long: MACD>0 AND close>EMA fast AND EMA fast>EMA slow.'},
    },
    'MACD_CCI': {
        'gen': gen_MACD_CCI,
        'space': space_MACD_CCI,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'cci_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'MACD + CCI. Long: MACD>signal AND CCI<-100.'},
    },
    'MACD_Vortex': {
        'gen': gen_MACD_Vortex,
        'space': space_MACD_Vortex,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'vortex_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1200,
                 'description': 'MACD + Vortex Indicator. Long: MACD>0 AND VI+>VI-.'},
    },
    'ZLMACD_Strategy': {
        'gen': gen_ZLMACD_Strategy,
        'space': space_ZLMACD_Strategy,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Zero Lag MACD using DEMA. ZL_MACD crosses signal.'},
    },
    'MACD_V': {
        'gen': gen_MACD_V,
        'space': space_MACD_V,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'vol_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Volume-adjusted MACD. Vol ratio normalizes signal strength.'},
    },
    'TEMA_MACD': {
        'gen': gen_TEMA_MACD,
        'space': space_TEMA_MACD,
        'default_params': {'fast': 8, 'slow': 21, 'sig': 6},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'MACD using TEMA instead of EMA for less lag.'},
    },
    'Hull_MACD': {
        'gen': gen_Hull_MACD,
        'space': space_Hull_MACD,
        'default_params': {'fast': 9, 'slow': 21, 'sig': 6},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1800,
                 'description': 'MACD using Hull Moving Average.'},
    },
    'MACD_Saucer': {
        'gen': gen_MACD_Saucer,
        'space': space_MACD_Saucer,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'MACD histogram saucer pattern. 3 bars: decrease, smallest, increase while same sign.'},
    },
    'MACD_Squeeze_Release': {
        'gen': gen_MACD_Squeeze_Release,
        'space': space_MACD_Squeeze_Release,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'squeeze_bars': 5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'MACD histogram compression then expansion. Low |hist| = squeeze. Expansion = signal.'},
    },
    'MACD_Momentum_Shift': {
        'gen': gen_MACD_Momentum_Shift,
        'space': space_MACD_Momentum_Shift,
        'default_params': {'fast': 12, 'slow': 26, 'sig': 9, 'lookback': 5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'Histogram peak/trough then reversal. Short: peaks from high. Long: recovers from trough.'},
    },
    'MACD_Triple': {
        'gen': gen_MACD_Triple,
        'space': space_MACD_Triple,
        'default_params': {'fast1': 3, 'slow1': 10, 'fast2': 12, 'slow2': 26, 'fast3': 24, 'slow3': 52},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Three MACDs: fast(3,10), mid(12,26), slow(24,52). Long: all three histograms > 0.'},
    },
}
