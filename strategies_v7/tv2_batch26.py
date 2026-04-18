#!/usr/bin/env python3
"""TV2 BATCH 26 — 30 estrategias Heikin Ashi + Smoothed Price 2026-04-01

  HA_Trend_Basic        — HA bull/bear confirmed by consecutive candles + no-wick check (v4, ~5000L)
  HA_Color_Change       — Signal on HA color flip with optional source smoothing (v4, ~4000L)
  HA_No_Wick_Trend      — N consecutive HA green with no lower wick (v5, ~3000L)
  HA_EMA_Trend          — HA green AND close above EMA (v4, ~3000L)
  HA_ATR_Filter         — HA trend + ATR expansion filter (v5, ~2500L)
  HA_RSI                — HA color + RSI directional filter (v4, ~2500L)
  HA_MACD               — HA green + MACD above signal line (v5, ~2000L)
  HA_Volume             — HA color change + volume surge (v4, ~2000L)
  HA_SuperTrend         — SuperTrend computed on HA OHLC (v5, ~2500L)
  HA_Stoch              — HA trend + Stochastic K>D from below 30 (v4, ~2000L)
  Smooth_HA_EMA         — EMA-smoothed HA close rising signal (v5, ~3000L)
  Smooth_HA_Cross       — Smooth HA close vs open cross (v4, ~2500L)
  Double_Smooth_HA      — Fast vs slow EMA of HA close crossover (v5, ~2000L)
  HA_Ichimoku           — Ichimoku cloud computed on HA OHLC (v5, ~2000L)
  HA_BB                 — BB on HA close — reversion at lower band (v4, ~2000L)
  HA_Keltner            — Keltner breakout on HA close (v5, ~1800L)
  HA_Hull               — Hull MA on HA close directional trend (v4, ~2000L)
  HA_QQE                — QQE oscillator on HA close with HA direction (v5, ~1500L)
  HA_CCI                — CCI on HA OHLC with HA color filter (v4, ~1500L)
  HA_PSAR               — Parabolic SAR computed on HA OHLC (v4, ~2500L)
  HA_Spinning_Top       — HA spinning top reversal after trend (v4, ~2000L)
  HA_Doji_Reversal      — HA doji at low + next green bar confirmation (v5, ~1800L)
  HA_Long_Body          — HA long body = strong trend continuation (v4, ~2000L)
  HA_Consecutive        — Enter on Nth consecutive same-color HA bar (v5, ~2000L)
  HA_Reversal_Pattern   — First HA bar after N consecutive opposite bars (v5, ~1500L)
  HA_WaveTrend          — WaveTrend (LazyBear) + HA direction filter (v5, ~2000L)
  HA_TSI                — TSI + HA direction filter (v5, ~1500L)
  HA_MFI                — MFI + HA color change (v4, ~1500L)
  HA_Chandelier         — Chandelier Exit + HA confirmation (v5, ~2000L)
  HA_SSL                — SSL Channel + HA direction (v4, ~1800L)
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


def _ha(df):
    """Compute Heikin Ashi OHLC from regular OHLC."""
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    n = len(df)
    ha_open_arr = np.zeros(n)
    ha_open_arr[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    hc = ha_close.values
    for i in range(1, n):
        ha_open_arr[i] = (ha_open_arr[i - 1] + hc[i - 1]) / 2
    ha_open = pd.Series(ha_open_arr, index=df.index)
    ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([df['low'],  ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame(
        {'open': ha_open, 'high': ha_high, 'low': ha_low, 'close': ha_close},
        index=df.index
    )


def _consec_count(bool_series):
    """Return series counting consecutive True values (resets to 0 on False)."""
    arr = bool_series.values.astype(int)
    out = np.zeros(len(arr), dtype=int)
    for i in range(len(arr)):
        if arr[i]:
            out[i] = (out[i - 1] + 1) if i > 0 else 1
        else:
            out[i] = 0
    return pd.Series(out, index=bool_series.index)


# ── 1. HA_Trend_Basic ─────────────────────────────────────────────────────────

def gen_HA_Trend_Basic(df, consec=2, **kw):
    ha = _ha(df)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    ha_min  = pd.concat([ha['open'], ha['close']], axis=1).min(axis=1)
    ha_max  = pd.concat([ha['open'], ha['close']], axis=1).max(axis=1)
    no_lower_wick = ha['low'] >= ha_min - 1e-10
    no_upper_wick = ha['high'] <= ha_max + 1e-10
    bull_count = _consec_count(ha_bull & no_lower_wick)
    bear_count = _consec_count(ha_bear & no_upper_wick)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[bull_count >= c] =  1
    sig[bear_count >= c] = -1
    return sig


def space_HA_Trend_Basic(trial):
    return {'consec': trial.suggest_int('consec', 1, 4)}


# ── 2. HA_Color_Change ────────────────────────────────────────────────────────

def gen_HA_Color_Change(df, smooth_p=1, **kw):
    src = df['close']
    if int(smooth_p) > 1:
        src = _ema(src, smooth_p)
    ha_df = df.copy()
    ha_df['close'] = src
    ha = _ha(ha_df)
    ha_bull = ha['close'] > ha['open']
    ha_prev_bull = ha_bull.shift(1).fillna(False)
    turned_bull = ha_bull & ~ha_prev_bull
    turned_bear = ~ha_bull & ha_prev_bull
    sig = pd.Series(0, index=df.index)
    sig[turned_bull] =  1
    sig[turned_bear] = -1
    return sig


def space_HA_Color_Change(trial):
    return {'smooth_p': trial.suggest_int('smooth_p', 1, 5)}


# ── 3. HA_No_Wick_Trend ───────────────────────────────────────────────────────

def gen_HA_No_Wick_Trend(df, consec=2, require_no_wick=1, **kw):
    ha = _ha(df)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    ha_min  = pd.concat([ha['open'], ha['close']], axis=1).min(axis=1)
    ha_max  = pd.concat([ha['open'], ha['close']], axis=1).max(axis=1)
    no_lower = ha['low'] >= ha_min - 1e-10
    no_upper = ha['high'] <= ha_max + 1e-10
    if int(require_no_wick):
        bull_cond = ha_bull & no_lower
        bear_cond = ha_bear & no_upper
    else:
        bull_cond = ha_bull
        bear_cond = ha_bear
    bull_count = _consec_count(bull_cond)
    bear_count = _consec_count(bear_cond)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[bull_count >= c] =  1
    sig[bear_count >= c] = -1
    return sig


def space_HA_No_Wick_Trend(trial):
    return {
        'consec':          trial.suggest_int('consec', 1, 4),
        'require_no_wick': trial.suggest_int('require_no_wick', 0, 1),
    }


# ── 4. HA_EMA_Trend ───────────────────────────────────────────────────────────

def gen_HA_EMA_Trend(df, consec=1, ema_p=50, **kw):
    ha   = _ha(df)
    ema  = _ema(df['close'], ema_p)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[(bull_count >= c) & (df['close'] > ema)] =  1
    sig[(bear_count >= c) & (df['close'] < ema)] = -1
    return sig


def space_HA_EMA_Trend(trial):
    return {
        'consec': trial.suggest_int('consec', 1, 3),
        'ema_p':  trial.suggest_int('ema_p', 20, 100),
    }


# ── 5. HA_ATR_Filter ──────────────────────────────────────────────────────────

def gen_HA_ATR_Filter(df, consec=1, atr_p=14, atr_mult=1.0, **kw):
    ha   = _ha(df)
    atr  = _atr(df, atr_p)
    atr_prev = atr.shift(1).fillna(atr)
    atr_expand = atr > atr_prev * float(atr_mult)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[(bull_count >= c) & atr_expand] =  1
    sig[(bear_count >= c) & atr_expand] = -1
    return sig


def space_HA_ATR_Filter(trial):
    return {
        'consec':   trial.suggest_int('consec', 1, 3),
        'atr_p':    trial.suggest_int('atr_p', 10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 0.8, 2.0),
    }


# ── 6. HA_RSI ─────────────────────────────────────────────────────────────────

def gen_HA_RSI(df, consec=1, rsi_p=14, **kw):
    ha   = _ha(df)
    rsi  = _rsi(df['close'], rsi_p)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[(bull_count >= c) & (rsi > 50)] =  1
    sig[(bear_count >= c) & (rsi < 50)] = -1
    return sig


def space_HA_RSI(trial):
    return {
        'consec': trial.suggest_int('consec', 1, 3),
        'rsi_p':  trial.suggest_int('rsi_p', 7, 21),
    }


# ── 7. HA_MACD ────────────────────────────────────────────────────────────────

def gen_HA_MACD(df, consec=1, fast=12, slow=26, sig_p=9, **kw):
    ha   = _ha(df)
    fast_i, slow_i, sig_i = int(fast), int(slow), int(sig_p)
    macd_line   = _ema(df['close'], fast_i) - _ema(df['close'], slow_i)
    macd_signal = _ema(macd_line, sig_i)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[(bull_count >= c) & (macd_line > macd_signal)] =  1
    sig[(bear_count >= c) & (macd_line < macd_signal)] = -1
    return sig


def space_HA_MACD(trial):
    return {
        'consec': trial.suggest_int('consec', 1, 3),
        'fast':   trial.suggest_int('fast', 8, 16),
        'slow':   trial.suggest_int('slow', 20, 30),
        'sig_p':  trial.suggest_int('sig_p', 5, 12),
    }


# ── 8. HA_Volume ──────────────────────────────────────────────────────────────

def gen_HA_Volume(df, consec=1, vol_p=20, vol_mult=1.5, **kw):
    ha   = _ha(df)
    vol  = df['volume']
    vol_sma = _sma(vol, vol_p)
    vol_surge = vol > vol_sma * float(vol_mult)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    c = int(consec)
    # Volume surge within window of N bars (not required exact same bar)
    vol_surge_window = vol_surge.rolling(int(vol_p) // 2, min_periods=1).max().astype(bool)
    bull_cond = (bull_count >= c) & vol_surge_window
    bear_cond = (bear_count >= c) & vol_surge_window
    sig = pd.Series(0, index=df.index)
    sig[bull_cond] =  1
    sig[bear_cond] = -1
    return sig


def space_HA_Volume(trial):
    return {
        'consec':   trial.suggest_int('consec', 1, 3),
        'vol_p':    trial.suggest_int('vol_p', 20, 50),
        'vol_mult': trial.suggest_float('vol_mult', 1.2, 3.0),
    }


# ── 9. HA_SuperTrend ──────────────────────────────────────────────────────────

def gen_HA_SuperTrend(df, st_p=10, st_mult=3.0, **kw):
    ha = _ha(df)
    atr = _atr(ha, st_p)
    hl2 = (ha['high'] + ha['low']) / 2
    upper_band = hl2 + float(st_mult) * atr
    lower_band = hl2 - float(st_mult) * atr
    n = len(ha)
    trend = np.zeros(n, dtype=int)
    final_upper = upper_band.values.copy()
    final_lower = lower_band.values.copy()
    hc = ha['close'].values
    for i in range(1, n):
        final_upper[i] = min(upper_band.iloc[i], final_upper[i - 1]) if hc[i - 1] < final_upper[i - 1] else upper_band.iloc[i]
        final_lower[i] = max(lower_band.iloc[i], final_lower[i - 1]) if hc[i - 1] > final_lower[i - 1] else lower_band.iloc[i]
        if hc[i] > final_upper[i - 1]:
            trend[i] = 1
        elif hc[i] < final_lower[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    sig = pd.Series(0, index=df.index)
    trend_s = pd.Series(trend, index=df.index)
    trend_prev = trend_s.shift(1).fillna(0)
    sig[trend_s == 1] =  1
    sig[trend_s == -1] = -1
    return sig


def space_HA_SuperTrend(trial):
    return {
        'st_p':    trial.suggest_int('st_p', 7, 21),
        'st_mult': trial.suggest_float('st_mult', 2.0, 5.0),
    }


# ── 10. HA_Stoch ──────────────────────────────────────────────────────────────

def gen_HA_Stoch(df, consec=1, k_p=14, d_p=3, **kw):
    ha   = _ha(df)
    kp   = int(k_p)
    dp   = int(d_p)
    low_min  = df['low'].rolling(kp, min_periods=1).min()
    high_max = df['high'].rolling(kp, min_periods=1).max()
    denom = (high_max - low_min).replace(0, 1e-9)
    k = 100 * (df['close'] - low_min) / denom
    d = _sma(k, dp)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    k_prev = k.shift(1).fillna(50)
    d_prev = d.shift(1).fillna(50)
    k_cross_up   = (k > d) & (k_prev <= d_prev) & (k < 30)
    k_cross_down = (k < d) & (k_prev >= d_prev) & (k > 70)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[(bull_count >= c) & k_cross_up]   =  1
    sig[(bear_count >= c) & k_cross_down] = -1
    return sig


def space_HA_Stoch(trial):
    return {
        'consec': trial.suggest_int('consec', 1, 3),
        'k_p':    trial.suggest_int('k_p', 7, 21),
        'd_p':    trial.suggest_int('d_p', 3, 7),
    }


# ── 11. Smooth_HA_EMA ─────────────────────────────────────────────────────────

def gen_Smooth_HA_EMA(df, ha_smooth=5, ema_p=10, **kw):
    ha = _ha(df)
    sha_close = _ema(ha['close'], ha_smooth)
    smoothed   = _ema(sha_close, ema_p)
    smoothed_prev = smoothed.shift(1).fillna(smoothed)
    sig = pd.Series(0, index=df.index)
    sig[smoothed > smoothed_prev] =  1
    sig[smoothed < smoothed_prev] = -1
    return sig


def space_Smooth_HA_EMA(trial):
    return {
        'ha_smooth': trial.suggest_int('ha_smooth', 3, 10),
        'ema_p':     trial.suggest_int('ema_p', 5, 20),
    }


# ── 12. Smooth_HA_Cross ───────────────────────────────────────────────────────

def gen_Smooth_HA_Cross(df, ha_smooth=5, **kw):
    ha = _ha(df)
    sha_close = _ema(ha['close'], ha_smooth)
    sha_open  = _ema(ha['open'],  ha_smooth)
    sig = pd.Series(0, index=df.index)
    sig[sha_close > sha_open] =  1
    sig[sha_close < sha_open] = -1
    return sig


def space_Smooth_HA_Cross(trial):
    return {'ha_smooth': trial.suggest_int('ha_smooth', 3, 10)}


# ── 13. Double_Smooth_HA ──────────────────────────────────────────────────────

def gen_Double_Smooth_HA(df, fast_p=5, slow_p=20, **kw):
    ha = _ha(df)
    fast = _ema(ha['close'], fast_p)
    slow = _ema(ha['close'], slow_p)
    fast_prev = fast.shift(1).fillna(fast)
    slow_prev = slow.shift(1).fillna(slow)
    cross_up   = (fast > slow) & (fast_prev <= slow_prev)
    cross_down = (fast < slow) & (fast_prev >= slow_prev)
    sig = pd.Series(0, index=df.index)
    sig[fast > slow] =  1
    sig[fast < slow] = -1
    return sig


def space_Double_Smooth_HA(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 3, 10),
        'slow_p': trial.suggest_int('slow_p', 10, 30),
    }


# ── 14. HA_Ichimoku ───────────────────────────────────────────────────────────

def gen_HA_Ichimoku(df, tenkan=9, kijun=26, **kw):
    ha = _ha(df)
    tk = int(tenkan)
    kj = int(kijun)
    tenkan_sen = (ha['high'].rolling(tk, min_periods=1).max() + ha['low'].rolling(tk, min_periods=1).min()) / 2
    kijun_sen  = (ha['high'].rolling(kj, min_periods=1).max() + ha['low'].rolling(kj, min_periods=1).min()) / 2
    span_a = ((tenkan_sen + kijun_sen) / 2).shift(kj)
    span_b = ((ha['high'].rolling(2 * kj, min_periods=1).max() + ha['low'].rolling(2 * kj, min_periods=1).min()) / 2).shift(kj)
    cloud_top    = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
    sig = pd.Series(0, index=df.index)
    sig[ha['close'] > cloud_top]    =  1
    sig[ha['close'] < cloud_bottom] = -1
    return sig


def space_HA_Ichimoku(trial):
    return {
        'tenkan': trial.suggest_int('tenkan', 7, 14),
        'kijun':  trial.suggest_int('kijun', 20, 35),
    }


# ── 15. HA_BB ─────────────────────────────────────────────────────────────────

def gen_HA_BB(df, bb_p=20, bb_mult=2.0, consec=1, **kw):
    ha = _ha(df)
    mid = _sma(ha['close'], bb_p)
    std = ha['close'].rolling(int(bb_p), min_periods=1).std().fillna(0)
    upper = mid + float(bb_mult) * std
    lower = mid - float(bb_mult) * std
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    # Use shifted BB bands: touched band on prior bar → wait for HA confirmation
    lower_prev = lower.shift(1).fillna(lower)
    upper_prev = upper.shift(1).fillna(upper)
    ha_close_prev = ha['close'].shift(1).fillna(ha['close'])
    touched_lower = ha_close_prev <= lower_prev
    touched_upper = ha_close_prev >= upper_prev
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[(bull_count >= c) & touched_lower] =  1
    sig[(bear_count >= c) & touched_upper] = -1
    return sig


def space_HA_BB(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 15, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0),
        'consec':  trial.suggest_int('consec', 1, 3),
    }


# ── 16. HA_Keltner ────────────────────────────────────────────────────────────

def gen_HA_Keltner(df, kc_p=20, kc_mult=2.0, consec=1, **kw):
    ha   = _ha(df)
    ema  = _ema(ha['close'], kc_p)
    atr  = _atr(ha, kc_p)
    upper = ema + float(kc_mult) * atr
    lower = ema - float(kc_mult) * atr
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    # Long: HA green AND close above KC midline (EMA) — trend riding inside channel
    # Short: HA red AND close below KC midline
    sig[(bull_count >= c) & (ha['close'] > ema)] =  1
    sig[(bear_count >= c) & (ha['close'] < ema)] = -1
    return sig


def space_HA_Keltner(trial):
    return {
        'kc_p':    trial.suggest_int('kc_p', 15, 30),
        'kc_mult': trial.suggest_float('kc_mult', 1.5, 3.0),
        'consec':  trial.suggest_int('consec', 1, 3),
    }


# ── 17. HA_Hull ───────────────────────────────────────────────────────────────

def gen_HA_Hull(df, hma_p=20, consec=1, **kw):
    ha  = _ha(df)
    src = ha['close']
    p   = int(hma_p)
    half_p = max(int(p / 2), 2)
    sqrt_p = max(int(np.sqrt(p)), 2)
    wma_full = _sma(src, p)
    wma_half = _sma(src, half_p)
    raw_hma  = 2 * wma_half - wma_full
    hma      = _sma(raw_hma, sqrt_p)
    hma_prev = hma.shift(1).fillna(hma)
    hma_rising  = hma > hma_prev
    hma_falling = hma < hma_prev
    rising_count  = _consec_count(hma_rising)
    falling_count = _consec_count(hma_falling)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[rising_count >= c]  =  1
    sig[falling_count >= c] = -1
    return sig


def space_HA_Hull(trial):
    return {
        'hma_p':  trial.suggest_int('hma_p', 10, 40),
        'consec': trial.suggest_int('consec', 1, 3),
    }


# ── 18. HA_QQE ────────────────────────────────────────────────────────────────

def gen_HA_QQE(df, rsi_p=14, sf=5, **kw):
    ha  = _ha(df)
    rsi = _rsi(ha['close'], rsi_p)
    rsi_smooth = _ema(rsi, sf)
    rsi_delta = rsi_smooth.diff().abs()
    atr_rsi   = _ema(rsi_delta, sf * 4)
    q_upper   = rsi_smooth + atr_rsi
    q_lower   = rsi_smooth - atr_rsi
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    sig = pd.Series(0, index=df.index)
    sig[(rsi_smooth > 50) & ha_bull] =  1
    sig[(rsi_smooth < 50) & ha_bear] = -1
    return sig


def space_HA_QQE(trial):
    return {
        'rsi_p': trial.suggest_int('rsi_p', 6, 14),
        'sf':    trial.suggest_int('sf', 3, 8),
    }


# ── 19. HA_CCI ────────────────────────────────────────────────────────────────

def gen_HA_CCI(df, cci_p=20, consec=1, **kw):
    ha  = _ha(df)
    tp  = (ha['high'] + ha['low'] + ha['close']) / 3
    p   = int(cci_p)
    sma = _sma(tp, p)
    mad = tp.rolling(p, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).fillna(1e-9)
    cci = (tp - sma) / (0.015 * mad.replace(0, 1e-9))
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[(bull_count >= c) & (cci > 0)]  =  1
    sig[(bear_count >= c) & (cci < 0)]  = -1
    return sig


def space_HA_CCI(trial):
    return {
        'cci_p':  trial.suggest_int('cci_p', 10, 30),
        'consec': trial.suggest_int('consec', 1, 3),
    }


# ── 20. HA_PSAR ───────────────────────────────────────────────────────────────

def gen_HA_PSAR(df, start=0.02, inc=0.02, max_af=0.2, **kw):
    ha   = _ha(df)
    n    = len(ha)
    high = ha['high'].values
    low  = ha['low'].values
    af   = float(start)
    af_inc = float(inc)
    af_max = float(max_af)
    ep     = low[0]
    psar   = high[0]
    bull   = False
    psar_arr = np.zeros(n)
    psar_arr[0] = psar
    for i in range(1, n):
        prev_psar = psar
        if bull:
            psar = psar + af * (ep - psar)
            psar = min(psar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < psar:
                bull = False
                psar = ep
                ep   = low[i]
                af   = float(start)
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_inc, af_max)
        else:
            psar = psar + af * (ep - psar)
            psar = max(psar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > psar:
                bull = True
                psar = ep
                ep   = high[i]
                af   = float(start)
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_inc, af_max)
        psar_arr[i] = psar
    psar_s = pd.Series(psar_arr, index=df.index)
    sig = pd.Series(0, index=df.index)
    sig[ha['close'] > psar_s] =  1
    sig[ha['close'] < psar_s] = -1
    return sig


def space_HA_PSAR(trial):
    return {
        'start':  trial.suggest_float('start', 0.01, 0.04),
        'inc':    trial.suggest_float('inc', 0.01, 0.04),
        'max_af': trial.suggest_float('max_af', 0.1, 0.4),
    }


# ── 21. HA_Spinning_Top ───────────────────────────────────────────────────────

def gen_HA_Spinning_Top(df, body_pct=0.2, ema_p=20, **kw):
    ha    = _ha(df)
    body  = (ha['close'] - ha['open']).abs()
    range_ = (ha['high'] - ha['low']).replace(0, 1e-9)
    body_ratio = body / range_
    spinning = body_ratio < float(body_pct)
    ema = _ema(df['close'], ema_p)
    ema_prev = ema.shift(1).fillna(ema)
    ema_declining = ema < ema_prev
    ema_rising    = ema > ema_prev
    # Entry after spinning top signals indecision — next bar direction
    spinning_prev = spinning.shift(1).fillna(False)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    sig = pd.Series(0, index=df.index)
    sig[spinning_prev & ha_bull & ema_declining] =  1
    sig[spinning_prev & ha_bear & ema_rising]    = -1
    return sig


def space_HA_Spinning_Top(trial):
    return {
        'body_pct': trial.suggest_float('body_pct', 0.1, 0.3),
        'ema_p':    trial.suggest_int('ema_p', 14, 50),
    }


# ── 22. HA_Doji_Reversal ──────────────────────────────────────────────────────

def gen_HA_Doji_Reversal(df, doji_pct=0.1, ema_p=20, **kw):
    ha    = _ha(df)
    body  = (ha['close'] - ha['open']).abs()
    range_ = (ha['high'] - ha['low']).replace(0, 1e-9)
    doji  = body / range_ < float(doji_pct)
    ema   = _ema(df['close'], ema_p)
    at_low  = df['close'] < ema
    at_high = df['close'] > ema
    doji_at_low  = doji & at_low
    doji_at_high = doji & at_high
    doji_low_prev  = doji_at_low.shift(1).fillna(False)
    doji_high_prev = doji_at_high.shift(1).fillna(False)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    sig = pd.Series(0, index=df.index)
    sig[doji_low_prev & ha_bull]  =  1
    sig[doji_high_prev & ha_bear] = -1
    return sig


def space_HA_Doji_Reversal(trial):
    return {
        'doji_pct': trial.suggest_float('doji_pct', 0.05, 0.2),
        'ema_p':    trial.suggest_int('ema_p', 14, 50),
    }


# ── 23. HA_Long_Body ──────────────────────────────────────────────────────────

def gen_HA_Long_Body(df, atr_p=14, body_mult=0.5, **kw):
    ha   = _ha(df)
    # Use HA ATR so we compare HA body to HA range volatility
    ha_atr = _atr(ha, atr_p)
    body = (ha['close'] - ha['open']).abs()
    long_body = body > float(body_mult) * ha_atr
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    sig = pd.Series(0, index=df.index)
    sig[long_body & ha_bull] =  1
    sig[long_body & ha_bear] = -1
    return sig


def space_HA_Long_Body(trial):
    return {
        'atr_p':     trial.suggest_int('atr_p', 10, 20),
        'body_mult': trial.suggest_float('body_mult', 0.2, 1.5),
    }


# ── 24. HA_Consecutive ────────────────────────────────────────────────────────

def gen_HA_Consecutive(df, n_bars=3, **kw):
    ha = _ha(df)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    n = int(n_bars)
    sig = pd.Series(0, index=df.index)
    sig[bull_count == n] =  1
    sig[bear_count == n] = -1
    return sig


def space_HA_Consecutive(trial):
    return {'n_bars': trial.suggest_int('n_bars', 2, 6)}


# ── 25. HA_Reversal_Pattern ───────────────────────────────────────────────────

def gen_HA_Reversal_Pattern(df, n_bars=3, rsi_p=14, **kw):
    ha   = _ha(df)
    rsi  = _rsi(df['close'], rsi_p)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    bear_count = _consec_count(ha_bear)
    bull_count = _consec_count(ha_bull)
    n = int(n_bars)
    # First green bar after N+ red bars (prior bar had bear count >= n)
    bear_count_prev = bear_count.shift(1).fillna(0)
    bull_count_prev = bull_count.shift(1).fillna(0)
    first_green_after_red = ha_bull & (bear_count_prev >= n)
    first_red_after_green = ha_bear & (bull_count_prev >= n)
    sig = pd.Series(0, index=df.index)
    sig[first_green_after_red & (rsi < 50)] =  1
    sig[first_red_after_green & (rsi > 50)] = -1
    return sig


def space_HA_Reversal_Pattern(trial):
    return {
        'n_bars': trial.suggest_int('n_bars', 2, 5),
        'rsi_p':  trial.suggest_int('rsi_p', 7, 21),
    }


# ── 26. HA_WaveTrend ──────────────────────────────────────────────────────────

def gen_HA_WaveTrend(df, wt_n1=10, wt_n2=21, consec=1, **kw):
    ha  = _ha(df)
    n1  = int(wt_n1)
    n2  = int(wt_n2)
    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    esa  = _ema(hlc3, n1)
    d    = _ema((hlc3 - esa).abs(), n1)
    ci   = (hlc3 - esa) / (0.015 * d.replace(0, 1e-9))
    tci  = _ema(ci, n2)
    wt1  = tci
    wt2  = _sma(wt1, 4)
    wt1_prev = wt1.shift(1).fillna(wt1)
    wt2_prev = wt2.shift(1).fillna(wt2)
    wt_bull_cross = (wt1 > wt2) & (wt1_prev <= wt2_prev)
    wt_bear_cross = (wt1 < wt2) & (wt1_prev >= wt2_prev)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[wt_bull_cross & (bull_count >= c)] =  1
    sig[wt_bear_cross & (bear_count >= c)] = -1
    return sig


def space_HA_WaveTrend(trial):
    return {
        'wt_n1':  trial.suggest_int('wt_n1', 7, 15),
        'wt_n2':  trial.suggest_int('wt_n2', 15, 30),
        'consec': trial.suggest_int('consec', 1, 3),
    }


# ── 27. HA_TSI ────────────────────────────────────────────────────────────────

def gen_HA_TSI(df, long_p=25, short_p=13, consec=1, **kw):
    ha  = _ha(df)
    pc  = df['close'].diff()
    lp  = int(long_p)
    sp  = int(short_p)
    double_smooth      = _ema(_ema(pc, lp), sp)
    double_smooth_abs  = _ema(_ema(pc.abs(), lp), sp)
    tsi     = 100 * double_smooth / double_smooth_abs.replace(0, 1e-9)
    tsi_sig = _ema(tsi, 7)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[(tsi > tsi_sig) & (bull_count >= c)] =  1
    sig[(tsi < tsi_sig) & (bear_count >= c)] = -1
    return sig


def space_HA_TSI(trial):
    return {
        'long_p':  trial.suggest_int('long_p', 13, 30),
        'short_p': trial.suggest_int('short_p', 3, 10),
        'consec':  trial.suggest_int('consec', 1, 3),
    }


# ── 28. HA_MFI ────────────────────────────────────────────────────────────────

def gen_HA_MFI(df, mfi_p=14, consec=1, **kw):
    ha  = _ha(df)
    p   = int(mfi_p)
    tp  = (df['high'] + df['low'] + df['close']) / 3
    rmf = tp * df['volume']
    prev_tp = tp.shift(1).fillna(tp)
    pos_mf  = rmf.where(tp > prev_tp, 0.0)
    neg_mf  = rmf.where(tp < prev_tp, 0.0)
    pos_sum = pos_mf.rolling(p, min_periods=1).sum()
    neg_sum = neg_mf.rolling(p, min_periods=1).sum()
    mfr = pos_sum / neg_sum.replace(0, 1e-9)
    mfi = 100 - 100 / (1 + mfr)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    ha_prev_bull = ha_bull.shift(1).fillna(False)
    ha_prev_bear = ha_bear.shift(1).fillna(False)
    turned_bull = ha_bull & ~ha_prev_bull
    turned_bear = ha_bear & ~ha_prev_bear
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    c = int(consec)
    if c == 1:
        bull_cond = turned_bull & (mfi < 30)
        bear_cond = turned_bear & (mfi > 70)
    else:
        bull_cond = (bull_count >= c) & (mfi < 50)
        bear_cond = (bear_count >= c) & (mfi > 50)
    sig = pd.Series(0, index=df.index)
    sig[bull_cond] =  1
    sig[bear_cond] = -1
    return sig


def space_HA_MFI(trial):
    return {
        'mfi_p':  trial.suggest_int('mfi_p', 7, 21),
        'consec': trial.suggest_int('consec', 1, 3),
    }


# ── 29. HA_Chandelier ─────────────────────────────────────────────────────────

def gen_HA_Chandelier(df, ce_p=22, ce_mult=3.0, consec=1, **kw):
    ha   = _ha(df)
    p    = int(ce_p)
    mult = float(ce_mult)
    atr  = _atr(ha, p)
    high_max = ha['high'].rolling(p, min_periods=1).max()
    low_min  = ha['low'].rolling(p, min_periods=1).min()
    ce_long  = high_max - mult * atr
    ce_short = low_min  + mult * atr
    n    = len(ha)
    dir_arr = np.zeros(n, dtype=int)
    hc = ha['close'].values
    cl  = ce_long.values
    cs  = ce_short.values
    for i in range(1, n):
        if hc[i] > cs[i - 1]:
            dir_arr[i] = 1
        elif hc[i] < cl[i - 1]:
            dir_arr[i] = -1
        else:
            dir_arr[i] = dir_arr[i - 1]
    dir_s = pd.Series(dir_arr, index=df.index)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    bull_count = _consec_count(ha_bull)
    bear_count = _consec_count(ha_bear)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[(dir_s == 1)  & (bull_count >= c)] =  1
    sig[(dir_s == -1) & (bear_count >= c)] = -1
    return sig


def space_HA_Chandelier(trial):
    return {
        'ce_p':    trial.suggest_int('ce_p', 14, 22),
        'ce_mult': trial.suggest_float('ce_mult', 2.0, 4.0),
        'consec':  trial.suggest_int('consec', 1, 3),
    }


# ── 30. HA_SSL ────────────────────────────────────────────────────────────────

def gen_HA_SSL(df, ssl_p=10, consec=1, **kw):
    ha  = _ha(df)
    p   = int(ssl_p)
    ssl_high = _sma(ha['high'], p)
    ssl_low  = _sma(ha['low'],  p)
    ha_bull = ha['close'] > ha['open']
    ha_bear = ha['close'] < ha['open']
    ssl_bull = ha['close'] > ssl_high
    ssl_bear = ha['close'] < ssl_low
    bull_count = _consec_count(ha_bull & ssl_bull)
    bear_count = _consec_count(ha_bear & ssl_bear)
    c = int(consec)
    sig = pd.Series(0, index=df.index)
    sig[bull_count >= c] =  1
    sig[bear_count >= c] = -1
    return sig


def space_HA_SSL(trial):
    return {
        'ssl_p':  trial.suggest_int('ssl_p', 7, 21),
        'consec': trial.suggest_int('consec', 1, 3),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'HA_Trend_Basic': {
        'gen': gen_HA_Trend_Basic,
        'space': space_HA_Trend_Basic,
        'default_params': {'consec': 2},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 5000,
                 'description': 'HA bull/bear with consecutive bars + no-wick confirmation'},
    },
    'HA_Color_Change': {
        'gen': gen_HA_Color_Change,
        'space': space_HA_Color_Change,
        'default_params': {'smooth_p': 1},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 4000,
                 'description': 'Signal on HA color flip with optional EMA smoothing of source'},
    },
    'HA_No_Wick_Trend': {
        'gen': gen_HA_No_Wick_Trend,
        'space': space_HA_No_Wick_Trend,
        'default_params': {'consec': 2, 'require_no_wick': 1},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'N consecutive HA green with no lower wick for strong trend'},
    },
    'HA_EMA_Trend': {
        'gen': gen_HA_EMA_Trend,
        'space': space_HA_EMA_Trend,
        'default_params': {'consec': 1, 'ema_p': 50},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'HA green AND close above EMA for dual trend confirmation'},
    },
    'HA_ATR_Filter': {
        'gen': gen_HA_ATR_Filter,
        'space': space_HA_ATR_Filter,
        'default_params': {'consec': 1, 'atr_p': 14, 'atr_mult': 1.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'HA trend + ATR expansion: enter only when volatility is growing'},
    },
    'HA_RSI': {
        'gen': gen_HA_RSI,
        'space': space_HA_RSI,
        'default_params': {'consec': 1, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'HA color + RSI direction: HA green AND RSI>50 = long'},
    },
    'HA_MACD': {
        'gen': gen_HA_MACD,
        'space': space_HA_MACD,
        'default_params': {'consec': 1, 'fast': 12, 'slow': 26, 'sig_p': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'HA green + MACD above signal for dual momentum confirmation'},
    },
    'HA_Volume': {
        'gen': gen_HA_Volume,
        'space': space_HA_Volume,
        'default_params': {'consec': 1, 'vol_p': 20, 'vol_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'HA color turn + volume surge above SMA multiple'},
    },
    'HA_SuperTrend_v2': {
        'gen': gen_HA_SuperTrend,
        'space': space_HA_SuperTrend,
        'default_params': {'st_p': 10, 'st_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'SuperTrend computed on HA OHLC for smoother trend signals'},
    },
    'HA_Stoch': {
        'gen': gen_HA_Stoch,
        'space': space_HA_Stoch,
        'default_params': {'consec': 1, 'k_p': 14, 'd_p': 3},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'HA trend + Stochastic K>D cross from oversold/overbought zones'},
    },
    'Smooth_HA_EMA': {
        'gen': gen_Smooth_HA_EMA,
        'space': space_Smooth_HA_EMA,
        'default_params': {'ha_smooth': 5, 'ema_p': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'EMA applied to HA close, then EMA again — rising = long'},
    },
    'Smooth_HA_Cross': {
        'gen': gen_Smooth_HA_Cross,
        'space': space_Smooth_HA_Cross,
        'default_params': {'ha_smooth': 5},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'EMA-smoothed HA close vs open: close > open = long'},
    },
    'Double_Smooth_HA': {
        'gen': gen_Double_Smooth_HA,
        'space': space_Double_Smooth_HA,
        'default_params': {'fast_p': 5, 'slow_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Fast EMA(HA) vs slow EMA(HA) crossover for trend direction'},
    },
    'HA_Ichimoku': {
        'gen': gen_HA_Ichimoku,
        'space': space_HA_Ichimoku,
        'default_params': {'tenkan': 9, 'kijun': 26},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Ichimoku cloud computed on HA OHLC — above/below cloud = long/short'},
    },
    'HA_BB': {
        'gen': gen_HA_BB,
        'space': space_HA_BB,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'consec': 1},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'BB on HA close: reversion at lower band with HA green confirmation'},
    },
    'HA_Keltner': {
        'gen': gen_HA_Keltner,
        'space': space_HA_Keltner,
        'default_params': {'kc_p': 20, 'kc_mult': 2.0, 'consec': 1},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1800,
                 'description': 'Keltner Channel breakout on HA close — above upper = long trend'},
    },
    'HA_Hull': {
        'gen': gen_HA_Hull,
        'space': space_HA_Hull,
        'default_params': {'hma_p': 20, 'consec': 1},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'Hull MA on HA close — rising HMA = long, falling = short'},
    },
    'HA_QQE': {
        'gen': gen_HA_QQE,
        'space': space_HA_QQE,
        'default_params': {'rsi_p': 14, 'sf': 5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'QQE oscillator on HA close with HA direction filter'},
    },
    'HA_CCI': {
        'gen': gen_HA_CCI,
        'space': space_HA_CCI,
        'default_params': {'cci_p': 20, 'consec': 1},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1500,
                 'description': 'CCI on HA OHLC: positive CCI + HA green = long trend'},
    },
    'HA_PSAR': {
        'gen': gen_HA_PSAR,
        'space': space_HA_PSAR,
        'default_params': {'start': 0.02, 'inc': 0.02, 'max_af': 0.2},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'Parabolic SAR iterative loop computed on HA high/low'},
    },
    'HA_Spinning_Top': {
        'gen': gen_HA_Spinning_Top,
        'space': space_HA_Spinning_Top,
        'default_params': {'body_pct': 0.2, 'ema_p': 20},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'HA spinning top (small body) after trend — reversal entry on next bar'},
    },
    'HA_Doji_Reversal': {
        'gen': gen_HA_Doji_Reversal,
        'space': space_HA_Doji_Reversal,
        'default_params': {'doji_pct': 0.1, 'ema_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1800,
                 'description': 'HA doji (close~open) at support/resistance + next bar direction confirm'},
    },
    'HA_Long_Body': {
        'gen': gen_HA_Long_Body,
        'space': space_HA_Long_Body,
        'default_params': {'atr_p': 14, 'body_mult': 0.5},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2000,
                 'description': 'HA body larger than ATR multiple = strong trend continuation signal'},
    },
    'HA_Consecutive': {
        'gen': gen_HA_Consecutive,
        'space': space_HA_Consecutive,
        'default_params': {'n_bars': 3},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Enter exactly on Nth consecutive same-color HA bar for trend entry'},
    },
    'HA_Reversal_Pattern': {
        'gen': gen_HA_Reversal_Pattern,
        'space': space_HA_Reversal_Pattern,
        'default_params': {'n_bars': 3, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'First HA bar in opposite direction after N consecutive bars = reversal'},
    },
    'HA_WaveTrend': {
        'gen': gen_HA_WaveTrend,
        'space': space_HA_WaveTrend,
        'default_params': {'wt_n1': 10, 'wt_n2': 21, 'consec': 1},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'WaveTrend (LazyBear) bullish/bearish cross + HA direction filter'},
    },
    'HA_TSI': {
        'gen': gen_HA_TSI,
        'space': space_HA_TSI,
        'default_params': {'long_p': 25, 'short_p': 13, 'consec': 1},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 1500,
                 'description': 'True Strength Index above signal + HA direction for momentum entries'},
    },
    'HA_MFI': {
        'gen': gen_HA_MFI,
        'space': space_HA_MFI,
        'default_params': {'mfi_p': 14, 'consec': 1},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1500,
                 'description': 'MFI oversold/overbought + HA color change for volume-weighted entries'},
    },
    'HA_Chandelier': {
        'gen': gen_HA_Chandelier,
        'space': space_HA_Chandelier,
        'default_params': {'ce_p': 22, 'ce_mult': 3.0, 'consec': 1},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Chandelier Exit direction on HA OHLC + HA color confirmation'},
    },
    'HA_SSL': {
        'gen': gen_HA_SSL,
        'space': space_HA_SSL,
        'default_params': {'ssl_p': 10, 'consec': 1},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 1800,
                 'description': 'SSL Channel (high/low SMA) on HA OHLC + HA color direction filter'},
    },
}
