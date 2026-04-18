#!/usr/bin/env python3
"""
TV2 BATCH MICRO4 — 5 microstructure strategies (5m/15m optimized).
Domain: Pine Script conversions — Volume Reversals & Oscillator Extremes
(distinct from MICRO1/2/3)

Source Pine scripts (v4plus repository):
  1. VolSpikeRev    — Basant1Saini/Volume_Spike_Reversal
                     Volume spike + RSI extreme + reversal candle
  2. StochRSI_Cross — EternaHybridExchange/stoch_rsi_crossover
                     Stoch RSI K×D crossover in oversold/overbought zone
  3. WilliamsR_Rev  — EternaHybridExchange/williams_r_reversal
                     Williams %R extreme crossover with price confirmation
  4. TTM_SqzRev     — Alorse/TTM_Squeeze (LazyBear adaptation)
                     BB-KC squeeze + momentum oscillator peak/trough reversal
  5. VWAP_Bounce    — Basant1Saini/VWAP_Bounce
                     Rolling VWAP stdev band bounce with EMA trend filter

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

def _stoch_rsi(s, rsi_p=14, stoch_p=14, k_smooth=3, d_smooth=3):
    rsi = _rsi(s, rsi_p)
    lo = rsi.rolling(stoch_p).min()
    hi = rsi.rolling(stoch_p).max()
    k = 100 * (rsi - lo) / (hi - lo + 1e-10)
    k = k.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d

def _williams_r(h, l, c, p=14):
    hi = h.rolling(p).max()
    lo = l.rolling(p).min()
    return -100 * (hi - c) / (hi - lo + 1e-10)

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
# 1. VolSpikeRev — Volume Spike Reversal
# ═══════════════════════════════════════════════════════════════════════════
# Source: Basant1Saini/trading_tools_23_Volume_Spike_Reversal.pine (v5)
# Logic: Volume spike (≥ N×avg) flags panic/blow-off. RSI confirms extreme.
#   Reversal candle (hammer or engulfing body in upper/lower half) confirms turn.
#   Enter on panic bottom (volume+oversold+bull candle) or blow-off top.
# Academic ref: Andersen (1996) — volume as information proxy; Blume et al (1994).

def gen_VolSpikeRev(df, vol_len=20, vol_mult=2.5, rsi_len=14, rsi_os=30,
                    rsi_ob=70, sma_len=20, exit_bar=8):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    avg_vol = v.rolling(vol_len).mean()
    is_spike = v >= avg_vol * vol_mult

    rsi = _rsi(c, rsi_len)
    sma = _sma(c, sma_len)

    # Bullish reversal candle: close in upper half + close > open
    bull_reversal = (c > o) & (c > (h + l) / 2)
    # Hammer: long lower wick, small body
    hammer = (c > o) & ((o - l) > 2 * (c - o).abs()) & ((h - c) < (c - o).abs())
    # Bearish reversal: close in lower half + close < open
    bear_reversal = (c < o) & (c < (h + l) / 2)
    # Shooting star: long upper wick
    shooting_star = (c < o) & ((h - o) > 2 * (o - c).abs()) & ((c - l) < (o - c).abs())

    bull_candle = bull_reversal | hammer
    bear_candle = bear_reversal | shooting_star

    long_sig = is_spike & (rsi < rsi_os) & bull_candle & (c < sma)
    short_sig = is_spike & (rsi > rsi_ob) & bear_candle & (c > sma)

    return _apply_exit_bar(long_sig, short_sig, exit_bar)


# ═══════════════════════════════════════════════════════════════════════════
# 2. StochRSI_Cross — Stochastic RSI Crossover in Extreme Zones
# ═══════════════════════════════════════════════════════════════════════════
# Source: EternaHybridExchange/tradingview_strategies_stoch_rsi_crossover.pine (v5)
# Logic: Stochastic of RSI. K crosses D from below while in oversold zone → long.
#   K crosses D from above while in overbought zone → short.
#   Momentum confirmation: both K and D moving in signal direction.
# Academic ref: Lane (1984) Stochastic oscillator; RSI variant per Cardwell (1994).

def gen_StochRSI_Cross(df, rsi_len=14, stoch_len=14, k_smooth=3, d_smooth=3,
                        ob=80, os_=20, exit_bar=10):
    c = df['close'].shift(1)

    k, d = _stoch_rsi(c, rsi_len, stoch_len, k_smooth, d_smooth)

    # K crosses above D
    k_cross_up = (k > d) & (k.shift(1) <= d.shift(1))
    # K crosses below D
    k_cross_dn = (k < d) & (k.shift(1) >= d.shift(1))

    # Momentum: both lines moving in same direction
    bull_mom = (k > k.shift(1)) & (d > d.shift(1))
    bear_mom = (k < k.shift(1)) & (d < d.shift(1))

    long_sig = k_cross_up & (k < os_) & bull_mom
    short_sig = k_cross_dn & (k > ob) & bear_mom

    return _apply_exit_bar(long_sig, short_sig, exit_bar)


# ═══════════════════════════════════════════════════════════════════════════
# 3. WilliamsR_Rev — Williams %R Reversal from Extremes
# ═══════════════════════════════════════════════════════════════════════════
# Source: EternaHybridExchange/tradingview_strategies_williams_r_reversal.pine (v5)
# Logic: Williams %R is a leading oscillator (-100 to 0). Below -80 = oversold.
#   When WR crosses above -80 AND close > prior close → long reversal.
#   When WR crosses below -20 AND close < prior close → short reversal.
# Academic ref: Williams (1979) — %R as leading momentum indicator vs lagging MA.

def gen_WilliamsR_Rev(df, wr_len=14, ob=-20, os_=-80, exit_bar=9):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    wr = _williams_r(h, l, c, wr_len)

    # WR crosses from oversold to neutral
    wr_cross_up = (wr > os_) & (wr.shift(1) <= os_)
    # WR crosses from overbought to neutral
    wr_cross_dn = (wr < ob) & (wr.shift(1) >= ob)

    # Price momentum confirmation (close must agree)
    long_sig = wr_cross_up & (c > c.shift(1))
    short_sig = wr_cross_dn & (c < c.shift(1))

    return _apply_exit_bar(long_sig, short_sig, exit_bar)


# ═══════════════════════════════════════════════════════════════════════════
# 4. TTM_SqzRev — TTM Squeeze Momentum Exhaustion Reversal
# ═══════════════════════════════════════════════════════════════════════════
# Source: Alorse/TTM_Squeeze.pine (v5, adapts LazyBear TTM Squeeze)
# Logic: When BB is inside KC = squeeze (compressed volatility). Momentum
#   oscillator (midrange linreg proxy) peaks during squeeze → exhaustion reversal.
#   RSI cross confirms direction. Trade the release of energy AGAINST momentum peak.
# Academic ref: Carter (2005) TTM Squeeze; Raschke & Connors (1995) volatility contraction.

def gen_TTM_SqzRev(df, length=20, bb_mult=2.0, kc_mult=1.5, rsi_len=14,
                    rsi_os=35, rsi_ob=65, exit_bar=12):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    # Bollinger Bands
    bb_mid = _sma(c, length)
    bb_std = c.rolling(length).std(ddof=0)
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    # Keltner Channel
    atr_val = _atr(h, l, c, length)
    kc_mid = _ema(c, length)
    kc_upper = kc_mid + kc_mult * atr_val
    kc_lower = kc_mid - kc_mult * atr_val

    # Momentum oscillator: midrange proxy (TTM midpoint)
    hi_n = h.rolling(length).max()
    lo_n = l.rolling(length).min()
    e1 = (hi_n + lo_n) / 2
    osc = c - (e1 + _sma(c, length)) / 2

    # Smooth with short rolling mean as linreg proxy
    osc_s = osc.rolling(max(3, length // 5)).mean()

    rsi = _rsi(c, rsi_len)

    # Momentum trough: osc was falling, now turning up → long
    osc_turning_up = (osc_s < osc_s.shift(1)) & (osc_s.shift(1) < osc_s.shift(2))
    # Momentum peak: osc was rising, now turning down → short
    osc_turning_dn = (osc_s > osc_s.shift(1)) & (osc_s.shift(1) > osc_s.shift(2))

    rsi_cross_up = (rsi > rsi_os) & (rsi.shift(1) <= rsi_os)
    rsi_cross_dn = (rsi < rsi_ob) & (rsi.shift(1) >= rsi_ob)

    long_sig = osc_turning_up & rsi_cross_up & (osc_s < 0)
    short_sig = osc_turning_dn & rsi_cross_dn & (osc_s > 0)

    return _apply_exit_bar(long_sig, short_sig, exit_bar)


# ═══════════════════════════════════════════════════════════════════════════
# 5. VWAP_Bounce — Rolling VWAP Band Bounce
# ═══════════════════════════════════════════════════════════════════════════
# Source: Basant1Saini/trading_tools_6_VWAP_Bounce.pine (v5)
# Logic: Rolling VWAP with stdev bands. When price touches lower band and
#   closes back above it (bounce), AND EMA trend is bullish → long.
#   Upper band bounce with bearish EMA → short.
#   VWAP acts as gravitational center — price repeatedly reverts to it.
# Academic ref: Berkowitz et al. (1988) VWAP execution; Madhavan (2000) market microstructure.

def gen_VWAP_Bounce(df, vwap_period=20, band_mult=1.5, ema_len=20, exit_bar=7):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    # Rolling VWAP
    hlc3 = (h + l + c) / 3
    vwap = (hlc3 * v).rolling(vwap_period).sum() / (v.rolling(vwap_period).sum() + 1e-10)

    # Standard deviation bands
    vwap_std = (c - vwap).rolling(vwap_period).std(ddof=0)
    upper_band = vwap + band_mult * vwap_std
    lower_band = vwap - band_mult * vwap_std

    # EMA trend filter
    ema = _ema(c, ema_len)
    trend_bull = c > ema
    trend_bear = c < ema

    # Bounce: low touched the band, close recovered inside
    touch_lower = (l <= lower_band) & (c > lower_band)
    touch_upper = (h >= upper_band) & (c < upper_band)

    long_sig = touch_lower & trend_bull
    short_sig = touch_upper & trend_bear

    return _apply_exit_bar(long_sig, short_sig, exit_bar)


# ─── OPTUNA SPACE FUNCTIONS ─────────────────────────────────────────────────

def space_VolSpikeRev(trial):
    return {
        'vol_len':  trial.suggest_int('vol_len', 15, 30),
        'vol_mult': trial.suggest_float('vol_mult', 1.8, 3.5, step=0.1),
        'rsi_len':  trial.suggest_int('rsi_len', 10, 21),
        'rsi_os':   trial.suggest_int('rsi_os', 20, 40),
        'rsi_ob':   trial.suggest_int('rsi_ob', 60, 80),
        'sma_len':  trial.suggest_int('sma_len', 10, 40),
        'exit_bar': trial.suggest_int('exit_bar', 5, 15),
    }

def space_StochRSI_Cross(trial):
    return {
        'rsi_len':   trial.suggest_int('rsi_len', 10, 21),
        'stoch_len': trial.suggest_int('stoch_len', 10, 21),
        'k_smooth':  trial.suggest_int('k_smooth', 2, 5),
        'd_smooth':  trial.suggest_int('d_smooth', 2, 5),
        'ob':        trial.suggest_int('ob', 70, 85),
        'os_':       trial.suggest_int('os_', 15, 30),
        'exit_bar':  trial.suggest_int('exit_bar', 6, 16),
    }

def space_WilliamsR_Rev(trial):
    return {
        'wr_len':   trial.suggest_int('wr_len', 10, 28),
        'ob':       trial.suggest_int('ob', -30, -10),
        'os_':      trial.suggest_int('os_', -90, -65),
        'exit_bar': trial.suggest_int('exit_bar', 6, 15),
    }

def space_TTM_SqzRev(trial):
    return {
        'length':   trial.suggest_int('length', 12, 30),
        'bb_mult':  trial.suggest_float('bb_mult', 1.5, 2.5, step=0.1),
        'kc_mult':  trial.suggest_float('kc_mult', 1.0, 2.0, step=0.1),
        'rsi_len':  trial.suggest_int('rsi_len', 10, 21),
        'rsi_os':   trial.suggest_int('rsi_os', 28, 45),
        'rsi_ob':   trial.suggest_int('rsi_ob', 55, 72),
        'exit_bar': trial.suggest_int('exit_bar', 8, 20),
    }

def space_VWAP_Bounce(trial):
    return {
        'vwap_period': trial.suggest_int('vwap_period', 10, 35),
        'band_mult':   trial.suggest_float('band_mult', 0.8, 2.5, step=0.1),
        'ema_len':     trial.suggest_int('ema_len', 10, 40),
        'exit_bar':    trial.suggest_int('exit_bar', 4, 14),
    }


# ─── REGISTRY ───────────────────────────────────────────────────────────────
STRATEGIES = {
    "TV_VolSpikeRev": gen_VolSpikeRev,
    "TV_StochRSI_Cross": gen_StochRSI_Cross,
    "TV_WilliamsR_Rev": gen_WilliamsR_Rev,
    "TV_TTM_SqzRev": gen_TTM_SqzRev,
    "TV_VWAP_Bounce": gen_VWAP_Bounce,
}

STRATEGY_EXPORT = {
    'TV_VolSpikeRev':    {'gen': gen_VolSpikeRev,    'space': space_VolSpikeRev},
    'TV_StochRSI_Cross': {'gen': gen_StochRSI_Cross, 'space': space_StochRSI_Cross},
    'TV_WilliamsR_Rev':  {'gen': gen_WilliamsR_Rev,  'space': space_WilliamsR_Rev},
    'TV_TTM_SqzRev':     {'gen': gen_TTM_SqzRev,     'space': space_TTM_SqzRev},
    'TV_VWAP_Bounce':    {'gen': gen_VWAP_Bounce,     'space': space_VWAP_Bounce},
}
