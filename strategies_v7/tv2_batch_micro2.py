#!/usr/bin/env python3
"""
TV2 BATCH MICRO2 — 5 microstructure strategies (5m/15m optimized).
Domain: Ultrafast Mean Reversion & Market Inefficiency (distinct from MICRO1 + 3237-3307)

Strategies:
  1. VWAP_Microdev       — VWAP deviation Z-score with declining volume filter
  2. RangeOsc_Zscore     — Normalized price-range oscillator Z-score reversal
  3. VolWeightedMom      — Volume-weighted momentum reversal at extremes
  4. PriceEfficiency     — Kaufman Efficiency Ratio mean-reversion in low-ER zones
  5. BidAskPressure      — Bar-level bid/ask pressure imbalance exhaustion

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
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

def _zscore(s, p):
    m = s.rolling(p).mean()
    sd = s.rolling(p).std(ddof=0) + 1e-10
    return (s - m) / sd

def _apply_exit_bar(entry_long: pd.Series, entry_short: pd.Series, exit_bar: int) -> pd.Series:
    """
    3-state machine: enters on entry signal, exits after exit_bar bars.
    Flat (0) → Long (1) on entry_long; Flat (0) → Short (-1) on entry_short.
    New entries can override an opposite position immediately.
    """
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
# 1. VWAP_Microdev — VWAP deviation Z-score mean reversion
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Berkowitz et al. (1988), Madhavan (2000) VWAP execution
# Logic: Compute rolling VWAP. Price Z-score from VWAP mean.
#   When price deviates > Z threshold AND volume is declining → mean reversion.

def gen_VWAP_Microdev(df, vwap_period=20, zscore_thresh=1.5, vol_ma=10,
                       trend_ema=50, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Rolling VWAP (windowed, not daily-reset — works on perpetual futures)
    typical_price = (h + l + c) / 3.0
    vwap = (typical_price * v).rolling(vwap_period).sum() / (v.rolling(vwap_period).sum() + 1e-10)

    # Price deviation from VWAP (Z-score)
    vwap_dev = c - vwap
    vwap_dev_z = _zscore(vwap_dev, vwap_period)

    # Volume trend: is volume declining (exhaustion) vs recent average
    vol_ma_val = _sma(v, vol_ma)
    vol_declining = v < vol_ma_val

    # Trend filter
    trend = _ema(c, trend_ema)

    # Entry: price Z-score extreme + volume declining → revert to VWAP
    entry_long  = (vwap_dev_z < -zscore_thresh) & vol_declining
    entry_short = (vwap_dev_z >  zscore_thresh) & vol_declining

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_VWAP_Microdev(trial):
    return {
        'vwap_period':   trial.suggest_int('vwap_period', 10, 40),
        'zscore_thresh': trial.suggest_float('zscore_thresh', 1.0, 3.0, step=0.25),
        'vol_ma':        trial.suggest_int('vol_ma', 5, 20),
        'trend_ema':     trial.suggest_int('trend_ema', 30, 100),
        'exit_bar':      trial.suggest_int('exit_bar', 3, 15),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. RangeOsc_Zscore — Normalized range oscillator Z-score reversal
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Williams (1966) %R, Chan (2013) range-based signals
# Logic: Range oscillator = (close - rolling_low) / (rolling_high - rolling_low).
#   Z-score of this oscillator. Extreme Z → reversal entry.

def gen_RangeOsc_Zscore(df, range_period=14, zscore_period=20, zscore_thresh=1.8,
                          confirm_ema=20, exit_bar=7):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # %R-style oscillator (0 = at low, 1 = at high)
    roll_low  = l.rolling(range_period).min()
    roll_high = h.rolling(range_period).max()
    range_osc = (c - roll_low) / (roll_high - roll_low + 1e-10)

    # Z-score of range oscillator
    range_z = _zscore(range_osc, zscore_period)

    # EMA confirmation: price vs EMA for trend context
    ema_c = _ema(c, confirm_ema)

    # Entry: Z-score extreme → reversal
    # Oversold (range_z very low = price near bottom of range, Z extreme negative)
    entry_long  = range_z < -zscore_thresh
    # Overbought (range_z very high)
    entry_short = range_z >  zscore_thresh

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_RangeOsc_Zscore(trial):
    return {
        'range_period':   trial.suggest_int('range_period', 8, 30),
        'zscore_period':  trial.suggest_int('zscore_period', 15, 40),
        'zscore_thresh':  trial.suggest_float('zscore_thresh', 1.2, 3.0, step=0.2),
        'confirm_ema':    trial.suggest_int('confirm_ema', 10, 50),
        'exit_bar':       trial.suggest_int('exit_bar', 3, 15),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. VolWeightedMom — Volume-weighted momentum reversal
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Blume, Easley & O'Hara (1994) volume and information
# Logic: Compute volume-weighted price momentum. When VWMom is extreme
#   (price moved hard WITH volume) → exhaustion → mean reversion.

def gen_VolWeightedMom(df, mom_period=10, zscore_period=25, zscore_thresh=1.8,
                        vol_ratio_thresh=1.3, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Price return per bar
    ret = c.pct_change().fillna(0)

    # Volume-weighted momentum: sum(return * relative_volume, N)
    avg_vol = v.rolling(mom_period).mean() + 1e-10
    rel_vol = v / avg_vol
    vwm = (ret * rel_vol).rolling(mom_period).sum()

    # Z-score of VWMom
    vwm_z = _zscore(vwm, zscore_period)

    # Volume spike confirmation (momentum WITH high volume → exhaustion)
    vol_spike = rel_vol > vol_ratio_thresh

    # Entry: extreme VWMom + volume spike → exhaustion reversal
    entry_long  = (vwm_z < -zscore_thresh) & vol_spike
    entry_short = (vwm_z >  zscore_thresh) & vol_spike

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_VolWeightedMom(trial):
    return {
        'mom_period':       trial.suggest_int('mom_period', 5, 20),
        'zscore_period':    trial.suggest_int('zscore_period', 15, 40),
        'zscore_thresh':    trial.suggest_float('zscore_thresh', 1.2, 3.0, step=0.2),
        'vol_ratio_thresh': trial.suggest_float('vol_ratio_thresh', 1.0, 2.5, step=0.25),
        'exit_bar':         trial.suggest_int('exit_bar', 3, 15),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. PriceEfficiency — Kaufman Efficiency Ratio mean reversion
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Kaufman (1995) Adaptive Moving Average — Efficiency Ratio
# Logic: ER = |price_change_N| / sum(|bar_changes|, N).
#   ER near 0 → noisy, mean-reverting market → optimal for mean-rev strategies.
#   ER near 1 → trending → avoid. Use ER < threshold as entry filter.

def gen_PriceEfficiency(df, er_period=14, er_thresh=0.3, zscore_period=20,
                         zscore_thresh=1.5, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Efficiency Ratio
    direction = (c - c.shift(er_period)).abs()
    noise = c.diff().abs().rolling(er_period).sum() + 1e-10
    er = direction / noise
    er = er.clip(0, 1)

    # Low ER = mean-reverting market
    is_mr_market = er < er_thresh

    # Price Z-score for entry direction
    price_z = _zscore(c, zscore_period)

    # RSI-like: close position relative to recent range
    roll_low  = c.rolling(zscore_period).min()
    roll_high = c.rolling(zscore_period).max()
    c_pct = (c - roll_low) / (roll_high - roll_low + 1e-10)

    # Entry: low ER (mean-rev market) + price at extreme
    entry_long  = is_mr_market & (price_z < -zscore_thresh)
    entry_short = is_mr_market & (price_z >  zscore_thresh)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_PriceEfficiency(trial):
    return {
        'er_period':     trial.suggest_int('er_period', 8, 25),
        'er_thresh':     trial.suggest_float('er_thresh', 0.15, 0.50, step=0.05),
        'zscore_period': trial.suggest_int('zscore_period', 15, 35),
        'zscore_thresh': trial.suggest_float('zscore_thresh', 1.0, 2.5, step=0.25),
        'exit_bar':      trial.suggest_int('exit_bar', 3, 15),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. BidAskPressure — Bar-level bid/ask pressure imbalance exhaustion
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Stoll (1989), Lee & Ready (1991) tick rule
# Logic: Approximate bid/ask pressure from OHLC bar structure.
#   buy_pressure  = (close - low) / (high - low)   — how much buyers won
#   sell_pressure = (high - close) / (high - low)  — how much sellers won
#   Net pressure = buy_pressure - sell_pressure (= 2*(close-low)/(high-low) - 1)
#   Rolling Z-score of net pressure. Extreme → exhaustion reversal.

def gen_BidAskPressure(df, pressure_period=15, zscore_period=25, zscore_thresh=1.8,
                        vol_weight=True, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Net pressure per bar: +1 = all buyers, -1 = all sellers
    hl_range = (h - l).replace(0, np.nan)
    net_pressure = 2.0 * (c - l) / hl_range - 1.0
    net_pressure = net_pressure.fillna(0)

    # Volume-weight the pressure (optional)
    if vol_weight:
        avg_vol = v.rolling(pressure_period).mean() + 1e-10
        rel_vol = v / avg_vol
        weighted_pressure = net_pressure * rel_vol
    else:
        weighted_pressure = net_pressure

    # Rolling mean of pressure
    roll_pressure = weighted_pressure.rolling(pressure_period).mean()

    # Z-score of rolling pressure
    press_z = _zscore(roll_pressure, zscore_period)

    # Entry: sustained extreme pressure → exhaustion reversal
    entry_long  = press_z < -zscore_thresh  # extreme selling pressure → buy
    entry_short = press_z >  zscore_thresh  # extreme buying pressure → sell

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_BidAskPressure(trial):
    return {
        'pressure_period': trial.suggest_int('pressure_period', 8, 30),
        'zscore_period':   trial.suggest_int('zscore_period', 15, 40),
        'zscore_thresh':   trial.suggest_float('zscore_thresh', 1.2, 3.0, step=0.2),
        'vol_weight':      trial.suggest_categorical('vol_weight', [True, False]),
        'exit_bar':        trial.suggest_int('exit_bar', 3, 15),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'VWAP_Microdev': {
        'gen':   gen_VWAP_Microdev,
        'space': space_VWAP_Microdev,
    },
    'RangeOsc_Zscore': {
        'gen':   gen_RangeOsc_Zscore,
        'space': space_RangeOsc_Zscore,
    },
    'VolWeightedMom': {
        'gen':   gen_VolWeightedMom,
        'space': space_VolWeightedMom,
    },
    'PriceEfficiency': {
        'gen':   gen_PriceEfficiency,
        'space': space_PriceEfficiency,
    },
    'BidAskPressure': {
        'gen':   gen_BidAskPressure,
        'space': space_BidAskPressure,
    },
}
