#!/usr/bin/env python3
"""TV2 BATCH 21 — 30 estrategias Stochastic + Williams + CCI 2026-04-01"""

import pandas as pd
import numpy as np


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=int(p), adjust=False).mean()

def _rma(s, p):
    return s.ewm(alpha=1/int(p), adjust=False).mean()

def _sma(s, p):
    return s.rolling(int(p)).mean()

def _atr(df, p):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return _rma(tr, p)

def _rsi(s, p):
    d = s.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    return 100 - 100 / (1 + _rma(g, p) / _rma(l, p).replace(0, 1e-9))

def _stoch(df, k_p, d_p):
    lo = df['low'].rolling(int(k_p)).min()
    hi = df['high'].rolling(int(k_p)).max()
    k = 100 * (df['close'] - lo) / (hi - lo + 1e-9)
    return k, _sma(k, int(d_p))

def _cci(df, p):
    tp = (df['high'] + df['low'] + df['close']) / 3
    p = int(p)
    tp_mean = tp.rolling(p).mean()
    def _mad(x):
        return np.abs(x - x.mean()).mean()
    mad = tp.rolling(p).apply(_mad, raw=True)
    return (tp - tp_mean) / (0.015 * mad + 1e-9)


# ─── STOCHASTIC VARIANTS ──────────────────────────────────────────────────────

# 1. Stoch_Classic
def gen_Stoch_Classic(df, k_p=14, d_p=3, **kw):
    k, d = _stoch(df, k_p, d_p)
    k_prev = k.shift(1)
    d_prev = d.shift(1)
    long = ((k_prev < d_prev) & (k > d) & (k < 20)).astype(int)
    short = ((k_prev > d_prev) & (k < d) & (k > 80)).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[long.astype(bool)] = 1
    sig[short.astype(bool)] = -1
    return sig.fillna(0).astype(int)

def space_Stoch_Classic(trial):
    return {
        'k_p': trial.suggest_int('k_p', 7, 21),
        'd_p': trial.suggest_int('d_p', 3, 7),
    }


# 2. Stoch_RSI_Classic
def gen_Stoch_RSI_Classic(df, rsi_p=14, stoch_p=14, d_p=3, **kw):
    rsi_vals = _rsi(df['close'], rsi_p)
    rsi_df = df.copy()
    rsi_df['close'] = rsi_vals
    rsi_df['high'] = rsi_vals
    rsi_df['low'] = rsi_vals
    k, d = _stoch(rsi_df, stoch_p, d_p)
    k_prev = k.shift(1)
    d_prev = d.shift(1)
    long = ((k_prev < d_prev) & (k > d) & (k < 20)).astype(int)
    short = ((k_prev > d_prev) & (k < d) & (k > 80)).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[long.astype(bool)] = 1
    sig[short.astype(bool)] = -1
    return sig.fillna(0).astype(int)

def space_Stoch_RSI_Classic(trial):
    return {
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
        'stoch_p': trial.suggest_int('stoch_p', 7, 21),
        'd_p': trial.suggest_int('d_p', 3, 7),
    }


# 3. Stoch_Divergence
def gen_Stoch_Divergence(df, k_p=14, d_p=3, lookback=10, **kw):
    k, _ = _stoch(df, k_p, d_p)
    sig = pd.Series(0, index=df.index)
    lb = int(lookback)
    close = df['close']
    for i in range(lb, len(df)):
        window_close = close.iloc[i - lb:i + 1]
        window_k = k.iloc[i - lb:i + 1]
        if window_close.isna().any() or window_k.isna().any():
            continue
        prev_low_idx = window_close.iloc[:-1].idxmin()
        curr_low = close.iloc[i]
        prev_low = window_close.iloc[:-1].min()
        prev_k_at_low = window_k.loc[prev_low_idx] if prev_low_idx in window_k.index else np.nan
        curr_k = window_k.iloc[-1]
        if np.isnan(prev_k_at_low):
            continue
        # Bull divergence: price LL + K HL
        if curr_low < prev_low and curr_k > prev_k_at_low:
            sig.iloc[i] = 1
        # Bear divergence: price HH + K LH
        prev_high_idx = window_close.iloc[:-1].idxmax()
        curr_high = close.iloc[i]
        prev_high = window_close.iloc[:-1].max()
        prev_k_at_high = window_k.loc[prev_high_idx] if prev_high_idx in window_k.index else np.nan
        if np.isnan(prev_k_at_high):
            continue
        if curr_high > prev_high and curr_k < prev_k_at_high:
            sig.iloc[i] = -1
    return sig.fillna(0).astype(int)

def space_Stoch_Divergence(trial):
    return {
        'k_p': trial.suggest_int('k_p', 7, 21),
        'd_p': trial.suggest_int('d_p', 3, 7),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# 4. Stoch_OB_OS
def gen_Stoch_OB_OS(df, k_p=14, d_p=3, mid=50, **kw):
    k, _ = _stoch(df, k_p, d_p)
    k_prev = k.shift(1)
    mid_val = float(mid)
    # Enter long when K < 20, exit when K crosses above mid
    # Enter short when K > 80, exit when K crosses below mid
    sig = pd.Series(0, index=df.index)
    long_entry = (k < 20)
    short_entry = (k > 80)
    long_cross_mid = ((k_prev < mid_val) & (k >= mid_val))
    short_cross_mid = ((k_prev > mid_val) & (k <= mid_val))
    sig[long_entry] = 1
    sig[short_entry] = -1
    # Reset at midline cross (use 0 as neutral)
    sig[long_cross_mid & ~long_entry & ~short_entry] = 0
    sig[short_cross_mid & ~long_entry & ~short_entry] = 0
    return sig.fillna(0).astype(int)

def space_Stoch_OB_OS(trial):
    return {
        'k_p': trial.suggest_int('k_p', 7, 21),
        'd_p': trial.suggest_int('d_p', 3, 7),
        'mid': trial.suggest_int('mid', 40, 60),
    }


# 5. Stoch_EMA_Trend
def gen_Stoch_EMA_Trend(df, k_p=14, d_p=3, ema_p=100, **kw):
    k, d = _stoch(df, k_p, d_p)
    ema = _ema(df['close'], ema_p)
    k_prev = k.shift(1)
    d_prev = d.shift(1)
    long = ((k_prev < d_prev) & (k > d) & (k < 30) & (df['close'] > ema))
    short = ((k_prev > d_prev) & (k < d) & (k > 70) & (df['close'] < ema))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_Stoch_EMA_Trend(trial):
    return {
        'k_p': trial.suggest_int('k_p', 7, 21),
        'd_p': trial.suggest_int('d_p', 3, 7),
        'ema_p': trial.suggest_int('ema_p', 50, 200),
    }


# 6. Stoch_BB
def gen_Stoch_BB(df, k_p=14, d_p=3, bb_p=20, bb_mult=2.0, **kw):
    k, d = _stoch(df, k_p, d_p)
    k_prev = k.shift(1)
    d_prev = d.shift(1)
    bb_mid = _sma(df['close'], bb_p)
    bb_std = df['close'].rolling(int(bb_p)).std()
    bb_lower = bb_mid - float(bb_mult) * bb_std
    bb_upper = bb_mid + float(bb_mult) * bb_std
    long = ((k_prev < d_prev) & (k > d) & (k < 30) & (df['close'] < bb_lower))
    short = ((k_prev > d_prev) & (k < d) & (k > 70) & (df['close'] > bb_upper))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_Stoch_BB(trial):
    return {
        'k_p': trial.suggest_int('k_p', 7, 21),
        'd_p': trial.suggest_int('d_p', 3, 7),
        'bb_p': trial.suggest_int('bb_p', 10, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0),
    }


# 7. Stoch_ATR
def gen_Stoch_ATR(df, k_p=14, d_p=3, atr_p=14, **kw):
    k, d = _stoch(df, k_p, d_p)
    k_prev = k.shift(1)
    d_prev = d.shift(1)
    atr = _atr(df, atr_p)
    atr_avg = _sma(atr, int(atr_p))
    long = ((k_prev < d_prev) & (k > d) & (k < 25) & (atr > atr_avg))
    short = ((k_prev > d_prev) & (k < d) & (k > 75) & (atr > atr_avg))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_Stoch_ATR(trial):
    return {
        'k_p': trial.suggest_int('k_p', 7, 21),
        'd_p': trial.suggest_int('d_p', 3, 7),
        'atr_p': trial.suggest_int('atr_p', 10, 20),
    }


# 8. Slow_Stoch
def gen_Slow_Stoch(df, k_p=14, d_p=3, **kw):
    k, _ = _stoch(df, k_p, d_p)
    d1 = _sma(k, int(d_p))
    d2 = _sma(d1, int(d_p))
    d1_prev = d1.shift(1)
    d2_prev = d2.shift(1)
    long = ((d1_prev < d2_prev) & (d1 > d2) & (d1 < 25))
    short = ((d1_prev > d2_prev) & (d1 < d2) & (d1 > 75))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_Slow_Stoch(trial):
    return {
        'k_p': trial.suggest_int('k_p', 7, 21),
        'd_p': trial.suggest_int('d_p', 3, 7),
    }


# 9. Full_Stoch
def gen_Full_Stoch(df, k_p=14, slow_k=3, d_p=3, **kw):
    k, _ = _stoch(df, k_p, d_p)
    slk = _sma(k, int(slow_k))
    d = _sma(slk, int(d_p))
    slk_prev = slk.shift(1)
    d_prev = d.shift(1)
    long = ((slk_prev < d_prev) & (slk > d) & (slk < 25))
    short = ((slk_prev > d_prev) & (slk < d) & (slk > 75))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_Full_Stoch(trial):
    return {
        'k_p': trial.suggest_int('k_p', 7, 21),
        'slow_k': trial.suggest_int('slow_k', 2, 5),
        'd_p': trial.suggest_int('d_p', 3, 7),
    }


# 10. StochRSI_Cross
def gen_StochRSI_Cross(df, rsi_p=14, stoch_p=14, d_p=3, **kw):
    rsi_vals = _rsi(df['close'], rsi_p)
    lo = rsi_vals.rolling(int(stoch_p)).min()
    hi = rsi_vals.rolling(int(stoch_p)).max()
    k = 100 * (rsi_vals - lo) / (hi - lo + 1e-9)
    d = _sma(k, int(d_p))
    k_prev = k.shift(1)
    d_prev = d.shift(1)
    long = ((k_prev < d_prev) & (k > d) & (k < 50))
    short = ((k_prev > d_prev) & (k < d) & (k > 50))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_StochRSI_Cross(trial):
    return {
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
        'stoch_p': trial.suggest_int('stoch_p', 7, 21),
        'd_p': trial.suggest_int('d_p', 3, 7),
    }


# ─── WILLIAMS INDICATORS ──────────────────────────────────────────────────────

# 11. Williams_R
def gen_Williams_R(df, wr_p=14, **kw):
    hi = df['high'].rolling(int(wr_p)).max()
    lo = df['low'].rolling(int(wr_p)).min()
    wr = (hi - df['close']) / (hi - lo + 1e-9) * -100
    wr_prev = wr.shift(1)
    long = ((wr_prev < -80) & (wr >= -80))
    short = ((wr_prev > -20) & (wr <= -20))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_Williams_R(trial):
    return {
        'wr_p': trial.suggest_int('wr_p', 7, 21),
    }


# 12. Williams_R_EMA
def gen_Williams_R_EMA(df, wr_p=14, ema_p=50, **kw):
    hi = df['high'].rolling(int(wr_p)).max()
    lo = df['low'].rolling(int(wr_p)).min()
    wr = (hi - df['close']) / (hi - lo + 1e-9) * -100
    ema = _ema(df['close'], ema_p)
    wr_prev = wr.shift(1)
    long = ((wr < -80) & (df['close'] > ema))
    short = ((wr > -20) & (df['close'] < ema))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_Williams_R_EMA(trial):
    return {
        'wr_p': trial.suggest_int('wr_p', 7, 21),
        'ema_p': trial.suggest_int('ema_p', 20, 100),
    }


# 13. Williams_R_Divergence
def gen_Williams_R_Divergence(df, wr_p=14, lookback=10, **kw):
    hi = df['high'].rolling(int(wr_p)).max()
    lo = df['low'].rolling(int(wr_p)).min()
    wr = (hi - df['close']) / (hi - lo + 1e-9) * -100
    sig = pd.Series(0, index=df.index)
    lb = int(lookback)
    close = df['close']
    for i in range(lb, len(df)):
        window_close = close.iloc[i - lb:i + 1]
        window_wr = wr.iloc[i - lb:i + 1]
        if window_close.isna().any() or window_wr.isna().any():
            continue
        prev_low_idx = window_close.iloc[:-1].idxmin()
        curr_low = close.iloc[i]
        prev_low = window_close.iloc[:-1].min()
        prev_wr_at_low = window_wr.loc[prev_low_idx] if prev_low_idx in window_wr.index else np.nan
        curr_wr = window_wr.iloc[-1]
        if np.isnan(prev_wr_at_low):
            continue
        if curr_low < prev_low and curr_wr > prev_wr_at_low:
            sig.iloc[i] = 1
        prev_high_idx = window_close.iloc[:-1].idxmax()
        curr_high = close.iloc[i]
        prev_high = window_close.iloc[:-1].max()
        prev_wr_at_high = window_wr.loc[prev_high_idx] if prev_high_idx in window_wr.index else np.nan
        if np.isnan(prev_wr_at_high):
            continue
        if curr_high > prev_high and curr_wr < prev_wr_at_high:
            sig.iloc[i] = -1
    return sig.fillna(0).astype(int)

def space_Williams_R_Divergence(trial):
    return {
        'wr_p': trial.suggest_int('wr_p', 7, 21),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# 14. Williams_Alligator
def gen_Williams_Alligator(df, jaw_p=13, teeth_p=8, lips_p=5,
                            jaw_shift=8, teeth_shift=5, lips_shift=3, **kw):
    median = (df['high'] + df['low']) / 2
    jaw = _rma(median, jaw_p).shift(int(jaw_shift))
    teeth = _rma(median, teeth_p).shift(int(teeth_shift))
    lips = _rma(median, lips_p).shift(int(lips_shift))
    long = ((df['close'] > jaw) & (lips > teeth) & (teeth > jaw))
    short = ((df['close'] < jaw) & (lips < teeth) & (teeth < jaw))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_Williams_Alligator(trial):
    return {
        'jaw_p': trial.suggest_int('jaw_p', 10, 16),
        'teeth_p': trial.suggest_int('teeth_p', 6, 10),
        'lips_p': trial.suggest_int('lips_p', 3, 6),
        'jaw_shift': trial.suggest_int('jaw_shift', 6, 10),
        'teeth_shift': trial.suggest_int('teeth_shift', 3, 6),
        'lips_shift': trial.suggest_int('lips_shift', 2, 4),
    }


# 15. Williams_Fractals
def gen_Williams_Fractals(df, fractal_p=2, **kw):
    fp = int(fractal_p)
    high = df['high']
    low = df['low']
    close = df['close']
    n = len(df)
    # Detect fractal highs and lows
    fractal_high = pd.Series(np.nan, index=df.index)
    fractal_low = pd.Series(np.nan, index=df.index)
    for i in range(fp, n - fp):
        window_high = high.iloc[i - fp:i + fp + 1]
        window_low = low.iloc[i - fp:i + fp + 1]
        if high.iloc[i] == window_high.max():
            fractal_high.iloc[i] = high.iloc[i]
        if low.iloc[i] == window_low.min():
            fractal_low.iloc[i] = low.iloc[i]
    # Track last fractal levels
    last_frac_high = fractal_high.ffill()
    last_frac_low = fractal_low.ffill()
    close_prev = close.shift(1)
    long = ((close > last_frac_high) & (close_prev <= last_frac_high))
    short = ((close < last_frac_low) & (close_prev >= last_frac_low))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_Williams_Fractals(trial):
    return {
        'fractal_p': trial.suggest_int('fractal_p', 2, 5),
    }


# ─── CCI VARIANTS ─────────────────────────────────────────────────────────────

# 16. CCI_Classic
def gen_CCI_Classic(df, cci_p=20, **kw):
    cci = _cci(df, cci_p)
    cci_prev = cci.shift(1)
    long = ((cci_prev < -100) & (cci >= -100))
    short = ((cci_prev > 100) & (cci <= 100))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_CCI_Classic(trial):
    return {
        'cci_p': trial.suggest_int('cci_p', 10, 30),
    }


# 17. CCI_Zero_Cross
def gen_CCI_Zero_Cross(df, cci_p=20, **kw):
    cci = _cci(df, cci_p)
    cci_prev = cci.shift(1)
    cci_rising = cci > cci_prev
    long = ((cci_prev < 0) & (cci >= 0) & cci_rising)
    short = ((cci_prev > 0) & (cci <= 0) & ~cci_rising)
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_CCI_Zero_Cross(trial):
    return {
        'cci_p': trial.suggest_int('cci_p', 10, 30),
    }


# 18. CCI_EMA
def gen_CCI_EMA(df, cci_p=20, ema_p=50, **kw):
    cci = _cci(df, cci_p)
    ema = _ema(df['close'], ema_p)
    long = ((cci > 0) & (df['close'] > ema))
    short = ((cci < 0) & (df['close'] < ema))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_CCI_EMA(trial):
    return {
        'cci_p': trial.suggest_int('cci_p', 10, 30),
        'ema_p': trial.suggest_int('ema_p', 20, 100),
    }


# 19. CCI_Divergence
def gen_CCI_Divergence(df, cci_p=20, lookback=10, **kw):
    cci = _cci(df, cci_p)
    sig = pd.Series(0, index=df.index)
    lb = int(lookback)
    close = df['close']
    for i in range(lb, len(df)):
        window_close = close.iloc[i - lb:i + 1]
        window_cci = cci.iloc[i - lb:i + 1]
        if window_close.isna().any() or window_cci.isna().any():
            continue
        prev_low_idx = window_close.iloc[:-1].idxmin()
        curr_low = close.iloc[i]
        prev_low = window_close.iloc[:-1].min()
        prev_cci_at_low = window_cci.loc[prev_low_idx] if prev_low_idx in window_cci.index else np.nan
        curr_cci = window_cci.iloc[-1]
        if np.isnan(prev_cci_at_low):
            continue
        if curr_low < prev_low and curr_cci > prev_cci_at_low:
            sig.iloc[i] = 1
        prev_high_idx = window_close.iloc[:-1].idxmax()
        curr_high = close.iloc[i]
        prev_high = window_close.iloc[:-1].max()
        prev_cci_at_high = window_cci.loc[prev_high_idx] if prev_high_idx in window_cci.index else np.nan
        if np.isnan(prev_cci_at_high):
            continue
        if curr_high > prev_high and curr_cci < prev_cci_at_high:
            sig.iloc[i] = -1
    return sig.fillna(0).astype(int)

def space_CCI_Divergence(trial):
    return {
        'cci_p': trial.suggest_int('cci_p', 10, 30),
        'lookback': trial.suggest_int('lookback', 5, 20),
    }


# 20. Woodies_CCI
def gen_Woodies_CCI(df, cci_p=14, turbo_p=6, **kw):
    cci = _cci(df, cci_p)
    turbo_df = df.copy()
    turbo = _cci(turbo_df, turbo_p)
    turbo_prev = turbo.shift(1)
    long = ((cci > 0) & (turbo_prev < 0) & (turbo >= 0))
    short = ((cci < 0) & (turbo_prev > 0) & (turbo <= 0))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_Woodies_CCI(trial):
    return {
        'cci_p': trial.suggest_int('cci_p', 10, 20),
        'turbo_p': trial.suggest_int('turbo_p', 4, 8),
    }


# 21. CCI_ATR
def gen_CCI_ATR(df, cci_p=20, atr_p=14, **kw):
    cci = _cci(df, cci_p)
    cci_prev = cci.shift(1)
    atr = _atr(df, atr_p)
    atr_prev = atr.shift(1)
    atr_expanding = atr > atr_prev
    long = ((cci_prev < -100) & (cci >= -100) & atr_expanding)
    short = ((cci_prev > 100) & (cci <= 100) & atr_expanding)
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_CCI_ATR(trial):
    return {
        'cci_p': trial.suggest_int('cci_p', 10, 30),
        'atr_p': trial.suggest_int('atr_p', 10, 20),
    }


# 22. CCI_RSI
def gen_CCI_RSI(df, cci_p=20, rsi_p=14, **kw):
    cci = _cci(df, cci_p)
    rsi = _rsi(df['close'], rsi_p)
    long = ((cci < -100) & (rsi < 35))
    short = ((cci > 100) & (rsi > 65))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_CCI_RSI(trial):
    return {
        'cci_p': trial.suggest_int('cci_p', 10, 30),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# 23. CCI_BB
def gen_CCI_BB(df, cci_p=20, bb_p=20, bb_mult=2.0, **kw):
    cci = _cci(df, cci_p)
    bb_mid = _sma(df['close'], bb_p)
    bb_std = df['close'].rolling(int(bb_p)).std()
    bb_lower = bb_mid - float(bb_mult) * bb_std
    bb_upper = bb_mid + float(bb_mult) * bb_std
    long = ((cci < -100) & (df['close'] < bb_lower))
    short = ((cci > 100) & (df['close'] > bb_upper))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_CCI_BB(trial):
    return {
        'cci_p': trial.suggest_int('cci_p', 10, 30),
        'bb_p': trial.suggest_int('bb_p', 10, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0),
    }


# ─── SMI (STOCHASTIC MOMENTUM INDEX) ─────────────────────────────────────────

# 24. SMI_Strategy
def gen_SMI_Strategy(df, pct_p=10, pct_d=3, sig_p=3, **kw):
    hl_diff = df['high'] - df['low']
    dist_to_mid = df['close'] - (df['high'] + df['low']) / 2
    # Double smooth both
    ds = _ema(_ema(dist_to_mid, pct_p), pct_d)
    dhl = _ema(_ema(hl_diff, pct_p), pct_d)
    smi = 100 * ds / (0.5 * dhl + 1e-9)
    signal_line = _ema(smi, sig_p)
    smi_prev = smi.shift(1)
    long = ((smi_prev < -40) & (smi >= -40))
    short = ((smi_prev > 40) & (smi <= 40))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_SMI_Strategy(trial):
    return {
        'pct_p': trial.suggest_int('pct_p', 5, 15),
        'pct_d': trial.suggest_int('pct_d', 3, 7),
        'sig_p': trial.suggest_int('sig_p', 3, 7),
    }


# 25. SMI_EMA
def gen_SMI_EMA(df, pct_p=10, pct_d=3, ema_p=40, **kw):
    hl_diff = df['high'] - df['low']
    dist_to_mid = df['close'] - (df['high'] + df['low']) / 2
    ds = _ema(_ema(dist_to_mid, pct_p), pct_d)
    dhl = _ema(_ema(hl_diff, pct_p), pct_d)
    smi = 100 * ds / (0.5 * dhl + 1e-9)
    ema = _ema(df['close'], ema_p)
    long = ((smi > 0) & (df['close'] > ema))
    short = ((smi < 0) & (df['close'] < ema))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_SMI_EMA(trial):
    return {
        'pct_p': trial.suggest_int('pct_p', 5, 15),
        'pct_d': trial.suggest_int('pct_d', 3, 7),
        'ema_p': trial.suggest_int('ema_p', 20, 60),
    }


# ─── AROON VARIANTS ───────────────────────────────────────────────────────────

def _aroon(df, p):
    p = int(p)
    def bars_since_high(x):
        return len(x) - 1 - x.argmax()
    def bars_since_low(x):
        return len(x) - 1 - x.argmin()
    bsh = df['high'].rolling(p + 1).apply(bars_since_high, raw=True)
    bsl = df['low'].rolling(p + 1).apply(bars_since_low, raw=True)
    aroon_up = (p - bsh) / p * 100
    aroon_down = (p - bsl) / p * 100
    return aroon_up, aroon_down


# 26. Aroon_Classic
def gen_Aroon_Classic(df, aroon_p=25, **kw):
    up, down = _aroon(df, aroon_p)
    long = ((up > 70) & (down < 30))
    short = ((down > 70) & (up < 30))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_Aroon_Classic(trial):
    return {
        'aroon_p': trial.suggest_int('aroon_p', 10, 30),
    }


# 27. Aroon_Oscillator
def gen_Aroon_Oscillator(df, aroon_p=25, **kw):
    up, down = _aroon(df, aroon_p)
    osc = up - down
    osc_prev = osc.shift(1)
    long = ((osc_prev < 0) & (osc >= 0))
    short = ((osc_prev > 0) & (osc <= 0))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_Aroon_Oscillator(trial):
    return {
        'aroon_p': trial.suggest_int('aroon_p', 10, 30),
    }


# 28. Aroon_EMA
def gen_Aroon_EMA(df, aroon_p=25, ema_p=40, **kw):
    up, down = _aroon(df, aroon_p)
    up_prev = up.shift(1)
    down_prev = down.shift(1)
    ema = _ema(df['close'], ema_p)
    long = ((up_prev < down_prev) & (up > down) & (df['close'] > ema))
    short = ((up_prev > down_prev) & (up < down) & (df['close'] < ema))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_Aroon_EMA(trial):
    return {
        'aroon_p': trial.suggest_int('aroon_p', 10, 30),
        'ema_p': trial.suggest_int('ema_p', 20, 60),
    }


# ─── BALANCE OF POWER ─────────────────────────────────────────────────────────

# 29. BOP_Strategy
def gen_BOP_Strategy(df, bop_p=14, **kw):
    bop_raw = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
    bop = _ema(bop_raw, bop_p)
    bop_prev = bop.shift(1)
    long = ((bop_prev < 0) & (bop >= 0))
    short = ((bop_prev > 0) & (bop <= 0))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_BOP_Strategy(trial):
    return {
        'bop_p': trial.suggest_int('bop_p', 5, 20),
    }


# 30. BOP_RSI
def gen_BOP_RSI(df, bop_p=14, rsi_p=14, **kw):
    bop_raw = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
    bop = _ema(bop_raw, bop_p)
    rsi = _rsi(df['close'], rsi_p)
    long = ((bop > 0) & (rsi < 50))
    short = ((bop < 0) & (rsi > 50))
    sig = pd.Series(0, index=df.index)
    sig[long] = 1
    sig[short] = -1
    return sig.fillna(0).astype(int)

def space_BOP_RSI(trial):
    return {
        'bop_p': trial.suggest_int('bop_p', 5, 20),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ─── STRATEGY_EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Stoch_Classic': {
        'gen': gen_Stoch_Classic,
        'space': space_Stoch_Classic,
        'default_params': {'k_p': 14, 'd_p': 3},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 5000,
                 'description': 'Classic Stochastic K/D crossover with OB/OS zones'},
    },
    'Stoch_RSI_Classic': {
        'gen': gen_Stoch_RSI_Classic,
        'space': space_Stoch_RSI_Classic,
        'default_params': {'rsi_p': 14, 'stoch_p': 14, 'd_p': 3},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 6000,
                 'description': 'Stochastic RSI K/D cross in OB/OS zones'},
    },
    'Stoch_Divergence': {
        'gen': gen_Stoch_Divergence,
        'space': space_Stoch_Divergence,
        'default_params': {'k_p': 14, 'd_p': 3, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Stochastic divergence: price LL + K HL for bull'},
    },
    'Stoch_OB_OS': {
        'gen': gen_Stoch_OB_OS,
        'space': space_Stoch_OB_OS,
        'default_params': {'k_p': 14, 'd_p': 3, 'mid': 50},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'Stoch mean reversion: enter at extremes, exit at midline'},
    },
    'Stoch_EMA_Trend': {
        'gen': gen_Stoch_EMA_Trend,
        'space': space_Stoch_EMA_Trend,
        'default_params': {'k_p': 14, 'd_p': 3, 'ema_p': 100},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Stoch oversold cross in trend direction (EMA filter)'},
    },
    'Stoch_BB': {
        'gen': gen_Stoch_BB,
        'space': space_Stoch_BB,
        'default_params': {'k_p': 14, 'd_p': 3, 'bb_p': 20, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1800,
                 'description': 'Stoch cross with Bollinger Band position filter'},
    },
    'Stoch_ATR': {
        'gen': gen_Stoch_ATR,
        'space': space_Stoch_ATR,
        'default_params': {'k_p': 14, 'd_p': 3, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'Stoch cross with ATR volatility expansion gate'},
    },
    'Slow_Stoch': {
        'gen': gen_Slow_Stoch,
        'space': space_Slow_Stoch,
        'default_params': {'k_p': 14, 'd_p': 3},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'Slow Stochastic: double-smoothed D1/D2 crossover'},
    },
    'Full_Stoch': {
        'gen': gen_Full_Stoch,
        'space': space_Full_Stoch,
        'default_params': {'k_p': 14, 'slow_k': 3, 'd_p': 3},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'Full Stochastic with slow K smoothing before signal'},
    },
    'StochRSI_Cross': {
        'gen': gen_StochRSI_Cross,
        'space': space_StochRSI_Cross,
        'default_params': {'rsi_p': 14, 'stoch_p': 14, 'd_p': 3},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'StochRSI K/D cross filtered by K < 50 (long) or K > 50 (short)'},
    },
    'Williams_R': {
        'gen': gen_Williams_R,
        'space': space_Williams_R,
        'default_params': {'wr_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'Williams %R OB/OS reversal at -80/-20 levels'},
    },
    'Williams_R_EMA': {
        'gen': gen_Williams_R_EMA,
        'space': space_Williams_R_EMA,
        'default_params': {'wr_p': 14, 'ema_p': 50},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Williams %R extremes filtered by EMA trend direction'},
    },
    'Williams_R_Divergence': {
        'gen': gen_Williams_R_Divergence,
        'space': space_Williams_R_Divergence,
        'default_params': {'wr_p': 14, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1500,
                 'description': 'Williams %R divergence: price LL + %R HL for bull'},
    },
    'Williams_Alligator': {
        'gen': gen_Williams_Alligator,
        'space': space_Williams_Alligator,
        'default_params': {'jaw_p': 13, 'teeth_p': 8, 'lips_p': 5,
                           'jaw_shift': 8, 'teeth_shift': 5, 'lips_shift': 3},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Williams Alligator: 3 SMMA lines with time shifts'},
    },
    'Williams_Fractals_v2': {
        'gen': gen_Williams_Fractals,
        'space': space_Williams_Fractals,
        'default_params': {'fractal_p': 2},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3500,
                 'description': 'Williams Fractals: breakout above/below 5-bar fractal pattern'},
    },
    'CCI_Classic': {
        'gen': gen_CCI_Classic,
        'space': space_CCI_Classic,
        'default_params': {'cci_p': 20},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 4000,
                 'description': 'CCI classic OB/OS reversal at +/-100'},
    },
    'CCI_Zero_Cross': {
        'gen': gen_CCI_Zero_Cross,
        'space': space_CCI_Zero_Cross,
        'default_params': {'cci_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'CCI zero line cross with momentum confirmation'},
    },
    'CCI_EMA': {
        'gen': gen_CCI_EMA,
        'space': space_CCI_EMA,
        'default_params': {'cci_p': 20, 'ema_p': 50},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'CCI momentum above/below zero with EMA trend filter'},
    },
    'CCI_Divergence': {
        'gen': gen_CCI_Divergence,
        'space': space_CCI_Divergence,
        'default_params': {'cci_p': 20, 'lookback': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1800,
                 'description': 'CCI divergence: price LL + CCI HL for bull signal'},
    },
    'Woodies_CCI': {
        'gen': gen_Woodies_CCI,
        'space': space_Woodies_CCI,
        'default_params': {'cci_p': 14, 'turbo_p': 6},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': "Woodies CCI system: 14-CCI trend + 6-period TurboCCI cross"},
    },
    'CCI_ATR': {
        'gen': gen_CCI_ATR,
        'space': space_CCI_ATR,
        'default_params': {'cci_p': 20, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1200,
                 'description': 'CCI oversold/overbought with ATR expansion confirmation'},
    },
    'CCI_RSI': {
        'gen': gen_CCI_RSI,
        'space': space_CCI_RSI,
        'default_params': {'cci_p': 20, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'CCI + RSI double confirmation: both must confirm extreme'},
    },
    'CCI_BB': {
        'gen': gen_CCI_BB,
        'space': space_CCI_BB,
        'default_params': {'cci_p': 20, 'bb_p': 20, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1200,
                 'description': 'CCI extreme + price at Bollinger Band extreme'},
    },
    'SMI_Strategy': {
        'gen': gen_SMI_Strategy,
        'space': space_SMI_Strategy,
        'default_params': {'pct_p': 10, 'pct_d': 3, 'sig_p': 3},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'Stochastic Momentum Index cross at +/-40 thresholds'},
    },
    'SMI_EMA': {
        'gen': gen_SMI_EMA,
        'space': space_SMI_EMA,
        'default_params': {'pct_p': 10, 'pct_d': 3, 'ema_p': 40},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'SMI above/below zero with EMA trend filter'},
    },
    'Aroon_Classic': {
        'gen': gen_Aroon_Classic,
        'space': space_Aroon_Classic,
        'default_params': {'aroon_p': 25},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'Aroon classic: Up>70 + Down<30 for strong uptrend'},
    },
    'Aroon_Oscillator': {
        'gen': gen_Aroon_Oscillator,
        'space': space_Aroon_Oscillator,
        'default_params': {'aroon_p': 25},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Aroon Oscillator (Up-Down) zero line cross'},
    },
    'Aroon_EMA': {
        'gen': gen_Aroon_EMA,
        'space': space_Aroon_EMA,
        'default_params': {'aroon_p': 25, 'ema_p': 40},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1500,
                 'description': 'Aroon Up/Down cross with EMA trend direction filter'},
    },
    'BOP_Strategy': {
        'gen': gen_BOP_Strategy,
        'space': space_BOP_Strategy,
        'default_params': {'bop_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'Balance of Power smoothed zero line cross'},
    },
    'BOP_RSI': {
        'gen': gen_BOP_RSI,
        'space': space_BOP_RSI,
        'default_params': {'bop_p': 14, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1200,
                 'description': 'BOP positive + RSI < 50: bullish momentum not yet overbought'},
    },
}
