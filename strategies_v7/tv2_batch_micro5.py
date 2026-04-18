#!/usr/bin/env python3
"""
TV2 BATCH MICRO5 — 5 microstructure strategies (5m/15m optimized).
Domain: Divergence Detection & Multi-Indicator Confluence
(distinct from MICRO1/2/3/4)

Source Pine scripts (v4plus / EternaHybridExchange / hasnocool):
  1. ChaikinOsc_Rev  — EternaHybridExchange/chaikin_oscillator
                       A/D Line EMA divergence zero-line cross + price divergence
  2. RSI_DivRev      — EternaHybridExchange/rsi_divergence
                       RSI pivot-based bullish/bearish divergence detection
  3. MomCombo_Rev    — EternaHybridExchange/momentum_combo (adapted)
                       RSI + MACD + StochRSI + ROC confluence reversal
  4. ATR_MeanRev     — Bcullen175/ATR_Mean_Reversion_V1
                       Price drops below ATR band → buy at mean EMA recovery
  5. BB_VolBreak_Rev — Alorse/BB_Winner_PRO + Alorse/Bollinger_Breakout
                       BB outer band touch + volume confirmation mean reversion

Anti-repainting: ALL OHLCV via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar.
TF focus: 5m, 15m.
Deadline: 2026-04-26.
"""

import pandas as pd
import numpy as np


# ─── HELPERS ────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _sma(s, p):
    return s.rolling(p).mean()

def _atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / p, adjust=False).mean()

def _rsi(s, p=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / p, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / p, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-10))

def _macd(s, fast=12, slow=26, signal=9):
    fast_ema = s.ewm(span=fast, adjust=False).mean()
    slow_ema = s.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    return macd_line, sig_line, hist

def _stoch_rsi(s, rsi_p=14, stoch_p=14, k_smooth=3, d_smooth=3):
    rsi = _rsi(s, rsi_p)
    lo = rsi.rolling(stoch_p).min()
    hi = rsi.rolling(stoch_p).max()
    k = 100 * (rsi - lo) / (hi - lo + 1e-10)
    k = k.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d

def _apply_exit_bar(entry_long: pd.Series, entry_short: pd.Series, exit_bar: int) -> pd.Series:
    sig = pd.Series(0, index=entry_long.index, dtype=int)
    el = entry_long.values
    es = entry_short.values
    n = len(sig)
    state = 0
    bars_held = 0
    for i in range(n):
        if state != 0:
            bars_held += 1
            if bars_held >= exit_bar:
                state = 0
                bars_held = 0
        if el[i]:
            state = 1
            bars_held = 0
        elif es[i]:
            state = -1
            bars_held = 0
        sig.iloc[i] = state
    return sig


# ═══════════════════════════════════════════════════════════════════════════
# 1. ChaikinOsc_Rev — Chaikin Oscillator Reversal
# ═══════════════════════════════════════════════════════════════════════════
# Source: EternaHybridExchange/tradingview_strategies_chaikin_oscillator.pine (v5)
# Logic: Chaikin = EMA(fast, AD) - EMA(slow, AD) where AD = cum(MFM×volume).
#   Zero-line crossover with momentum confirmation = distribution/accumulation flip.
#   Price divergence from Chaikin: price making new low but Chaikin higher low = bullish.
# Academic ref: Chaikin (1982) Accumulation/Distribution; Williams MFI Chaikin variant.

def gen_ChaikinOsc_Rev(df, fast_len=3, slow_len=10, div_lookback=5, exit_bar=10):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    # Money Flow Multiplier: ((close - low) - (high - close)) / (high - low)
    mfm = ((c - l) - (h - c)) / (h - l + 1e-10)
    # Money Flow Volume
    mfv = mfm * v
    # Accumulation/Distribution Line (cumulative)
    ad = mfv.cumsum()

    # Chaikin Oscillator
    chaikin = _ema(ad, fast_len) - _ema(ad, slow_len)

    # Zero-line crossovers
    cross_up = (chaikin > 0) & (chaikin.shift(1) <= 0)
    cross_dn = (chaikin < 0) & (chaikin.shift(1) >= 0)

    # Momentum confirmation: 2 consecutive bars in direction
    bull_mom = (chaikin > chaikin.shift(1)) & (chaikin.shift(1) > chaikin.shift(2))
    bear_mom = (chaikin < chaikin.shift(1)) & (chaikin.shift(1) < chaikin.shift(2))

    # Divergence: price at period low but Chaikin higher than its period low → bullish
    price_at_low = l == l.rolling(div_lookback).min()
    chaikin_not_at_low = chaikin > chaikin.rolling(div_lookback).min()
    bull_div = price_at_low & chaikin_not_at_low & (chaikin < 0)

    price_at_high = h == h.rolling(div_lookback).max()
    chaikin_not_at_high = chaikin < chaikin.rolling(div_lookback).max()
    bear_div = price_at_high & chaikin_not_at_high & (chaikin > 0)

    long_sig = (cross_up & bull_mom) | bull_div
    short_sig = (cross_dn & bear_mom) | bear_div

    return _apply_exit_bar(long_sig, short_sig, exit_bar)


# ═══════════════════════════════════════════════════════════════════════════
# 2. RSI_DivRev — RSI Price Divergence Reversal
# ═══════════════════════════════════════════════════════════════════════════
# Source: EternaHybridExchange/tradingview_strategies_rsi_divergence.pine (v5)
# Logic: RSI pivot detection. Bullish div: price lower low + RSI higher low in oversold.
#   Bearish div: price higher high + RSI lower high in overbought.
#   Classic leading signal: momentum diverges before price reversal.
# Academic ref: Divergence theory (RSI) — Cardwell (1994); Lo et al. (2000) technical patterns.

def gen_RSI_DivRev(df, rsi_len=14, pivot_lookback=5, rsi_os=35, rsi_ob=65,
                    exit_bar=12):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    rsi = _rsi(c, rsi_len)

    # Detect pivot lows/highs via rolling window comparison
    # Pivot low: low is minimum in [i-lookback, i] window
    l_min = l.rolling(pivot_lookback).min()
    h_max = h.rolling(pivot_lookback).max()
    rsi_min = rsi.rolling(pivot_lookback).min()
    rsi_max = rsi.rolling(pivot_lookback).max()

    # Bullish divergence: price at recent low AND RSI not at its recent low
    # AND RSI recovering from oversold zone
    at_price_low = (l == l_min)
    rsi_recovering = rsi > rsi_min  # RSI higher than its period low
    bull_div = at_price_low & rsi_recovering & (rsi < rsi_os + 15)

    # Bearish divergence: price at recent high AND RSI not at its recent high
    at_price_high = (h == h_max)
    rsi_lagging = rsi < rsi_max  # RSI lower than its period high
    bear_div = at_price_high & rsi_lagging & (rsi > rsi_ob - 15)

    return _apply_exit_bar(bull_div, bear_div, exit_bar)


# ═══════════════════════════════════════════════════════════════════════════
# 3. MomCombo_Rev — Multi-Indicator Momentum Confluence Reversal
# ═══════════════════════════════════════════════════════════════════════════
# Source: EternaHybridExchange/tradingview_strategies_momentum_combo.pine (v5)
# Adapted: Pure mean-reversion version (confluence of reversal signals).
# Logic: Count how many of RSI/MACD/StochRSI/ROC give reversal signals.
#   Minimum N confirmed = high-conviction entry. Avoids false signals.
# Academic ref: Gençay (1998) multi-indicator confluence; Murphy (1999) technical analysis.

def gen_MomCombo_Rev(df, rsi_len=14, rsi_os=30, rsi_ob=70,
                      macd_fast=12, macd_slow=26, macd_sig=9,
                      roc_len=10, min_confirms=2, exit_bar=10):
    c = df['close'].shift(1)

    # RSI reversal signals
    rsi = _rsi(c, rsi_len)
    rsi_bull = (rsi < rsi_os) | ((rsi > rsi_os) & (rsi.shift(1) <= rsi_os) & (rsi < 50))
    rsi_bear = (rsi > rsi_ob) | ((rsi < rsi_ob) & (rsi.shift(1) >= rsi_ob) & (rsi > 50))

    # MACD reversal (histogram turns from negative to positive)
    ml, sl, hist = _macd(c, macd_fast, macd_slow, macd_sig)
    macd_bull = (ml > sl) & (ml.shift(1) <= sl.shift(1))  # bullish cross
    macd_bull |= (ml > sl) & (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2))
    macd_bear = (ml < sl) & (ml.shift(1) >= sl.shift(1))  # bearish cross
    macd_bear |= (ml < sl) & (hist < hist.shift(1)) & (hist.shift(1) < hist.shift(2))

    # StochRSI reversal
    k, d = _stoch_rsi(c, rsi_len)
    stoch_bull = (k > d) & (k.shift(1) <= d.shift(1)) & (k < 30)
    stoch_bear = (k < d) & (k.shift(1) >= d.shift(1)) & (k > 70)

    # ROC reversal (momentum turns)
    roc = c.pct_change(roc_len) * 100
    roc_bull = (roc > 0) & (roc.shift(1) <= 0) & (roc.shift(1) < -1)
    roc_bear = (roc < 0) & (roc.shift(1) >= 0) & (roc.shift(1) > 1)

    # Count confirmations
    bull_count = rsi_bull.astype(int) + macd_bull.astype(int) + stoch_bull.astype(int) + roc_bull.astype(int)
    bear_count = rsi_bear.astype(int) + macd_bear.astype(int) + stoch_bear.astype(int) + roc_bear.astype(int)

    long_sig = bull_count >= min_confirms
    short_sig = bear_count >= min_confirms

    return _apply_exit_bar(long_sig, short_sig, exit_bar)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ATR_MeanRev — ATR Band Mean Reversion
# ═══════════════════════════════════════════════════════════════════════════
# Source: Bcullen175/ATR_Mean_Reversion_Strategy_V1.pine (v5, hasnocool repo)
# Logic: ATR bands around daily open. Price drops below lower band = overextended.
#   Enter long, target recovery to EMA mean. Exit when price returns to mean
#   or exceeds upper band. Short: price pops above upper band → fade to mean.
# Academic ref: Garman & Klass (1980) ATR as volatility measure; mean-reversion literature.

def gen_ATR_MeanRev(df, atr_len=10, atr_mult=1.0, mean_len=20, sl_mult=1.5,
                     exit_bar=8):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)

    atr = _atr(h, l, c, atr_len)
    mean_ema = _ema(c, mean_len)

    # ATR bands around open
    upper_band = o + atr * atr_mult
    lower_band = o - atr * atr_mult

    # Price below lower band (oversold relative to open + volatility)
    below_band = l < lower_band
    # Price above upper band (overbought relative to open + volatility)
    above_band = h > upper_band

    # Recovery back above lower band / below upper band = entry
    long_sig = below_band & (c > lower_band) & (c < mean_ema)
    short_sig = above_band & (c < upper_band) & (c > mean_ema)

    return _apply_exit_bar(long_sig, short_sig, exit_bar)


# ═══════════════════════════════════════════════════════════════════════════
# 5. BB_VolBreak_Rev — Bollinger Band + Volume Breakout Fade
# ═══════════════════════════════════════════════════════════════════════════
# Source: Alorse/BB_Winner_PRO.pine + Alorse/Bollinger_Breakout.pine (v5)
# Logic: Price pierces BB outer band on above-average volume → fade it.
#   Strong volume + outside-band close = exhaustion, not breakout.
#   Close returns inside band = confirmation of reversal.
# Academic ref: Bollinger (2002) Bands; Keltner Channel squeeze (Carter 2005).

def gen_BB_VolBreak_Rev(df, bb_len=20, bb_mult=2.0, vol_len=20, vol_thresh=1.3,
                         exit_bar=9):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    # Bollinger Bands
    bb_mid = _sma(c, bb_len)
    bb_std = c.rolling(bb_len).std(ddof=0)
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    # Volume above average
    avg_vol = v.rolling(vol_len).mean()
    high_vol = v > avg_vol * vol_thresh

    # Fade: price closed outside band AND returning inside on high volume
    # Long: price was below lower band, now closing above it (recovery)
    pierced_low_prev = l.shift(1) < bb_lower.shift(1)
    recovering = (c > bb_lower) & (c.shift(1) <= bb_lower.shift(1))
    long_sig = pierced_low_prev & recovering & high_vol

    # Short: price was above upper band, now closing below it (reversal)
    pierced_high_prev = h.shift(1) > bb_upper.shift(1)
    rejecting = (c < bb_upper) & (c.shift(1) >= bb_upper.shift(1))
    short_sig = pierced_high_prev & rejecting & high_vol

    return _apply_exit_bar(long_sig, short_sig, exit_bar)


# ─── OPTUNA SPACE FUNCTIONS ─────────────────────────────────────────────────

def space_ChaikinOsc_Rev(trial):
    return {
        'fast_len':     trial.suggest_int('fast_len', 2, 8),
        'slow_len':     trial.suggest_int('slow_len', 8, 20),
        'div_lookback': trial.suggest_int('div_lookback', 3, 10),
        'exit_bar':     trial.suggest_int('exit_bar', 6, 16),
    }

def space_RSI_DivRev(trial):
    return {
        'rsi_len':        trial.suggest_int('rsi_len', 10, 28),
        'pivot_lookback': trial.suggest_int('pivot_lookback', 3, 10),
        'rsi_os':         trial.suggest_int('rsi_os', 25, 45),
        'rsi_ob':         trial.suggest_int('rsi_ob', 55, 75),
        'exit_bar':       trial.suggest_int('exit_bar', 8, 20),
    }

def space_MomCombo_Rev(trial):
    return {
        'rsi_len':      trial.suggest_int('rsi_len', 10, 21),
        'rsi_os':       trial.suggest_int('rsi_os', 20, 38),
        'rsi_ob':       trial.suggest_int('rsi_ob', 62, 80),
        'roc_len':      trial.suggest_int('roc_len', 6, 20),
        'min_confirms': trial.suggest_int('min_confirms', 2, 3),
        'exit_bar':     trial.suggest_int('exit_bar', 6, 16),
    }

def space_ATR_MeanRev(trial):
    return {
        'atr_len':  trial.suggest_int('atr_len', 7, 21),
        'atr_mult': trial.suggest_float('atr_mult', 0.6, 2.0, step=0.1),
        'mean_len': trial.suggest_int('mean_len', 12, 40),
        'exit_bar': trial.suggest_int('exit_bar', 5, 14),
    }

def space_BB_VolBreak_Rev(trial):
    return {
        'bb_len':     trial.suggest_int('bb_len', 14, 30),
        'bb_mult':    trial.suggest_float('bb_mult', 1.5, 2.5, step=0.1),
        'vol_len':    trial.suggest_int('vol_len', 10, 30),
        'vol_thresh': trial.suggest_float('vol_thresh', 1.1, 2.0, step=0.1),
        'exit_bar':   trial.suggest_int('exit_bar', 6, 15),
    }


# ─── REGISTRY ───────────────────────────────────────────────────────────────
STRATEGIES = {
    "TV_ChaikinOsc_Rev": gen_ChaikinOsc_Rev,
    "TV_RSI_DivRev": gen_RSI_DivRev,
    "TV_MomCombo_Rev": gen_MomCombo_Rev,
    "TV_ATR_MeanRev": gen_ATR_MeanRev,
    "TV_BB_VolBreak_Rev": gen_BB_VolBreak_Rev,
}

STRATEGY_EXPORT = {
    'TV_ChaikinOsc_Rev':  {'gen': gen_ChaikinOsc_Rev,  'space': space_ChaikinOsc_Rev},
    'TV_RSI_DivRev':      {'gen': gen_RSI_DivRev,      'space': space_RSI_DivRev},
    'TV_MomCombo_Rev':    {'gen': gen_MomCombo_Rev,     'space': space_MomCombo_Rev},
    'TV_ATR_MeanRev':     {'gen': gen_ATR_MeanRev,      'space': space_ATR_MeanRev},
    'TV_BB_VolBreak_Rev': {'gen': gen_BB_VolBreak_Rev,  'space': space_BB_VolBreak_Rev},
}
