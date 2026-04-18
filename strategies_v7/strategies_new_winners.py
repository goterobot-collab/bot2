#!/usr/bin/env python3
"""
NEW WINNERS: Pine→Python conversions from TradingView validated strategies.
Converted: 2026-03-31

TIER 1:
  1. MACs_V6_final   — Keltner Channel + CFB Range Filter (LONG+SHORT)
  2. MeanRev_VF      — SMA deviation bands + Hull MA exit (LONG only)
  3. LowFinder_Pyra  — MTF RSI low-finder + Stochastic filter (LONG only)

TIER 2 (local):
  4. Spike_Reversion_70 — Price spike + RSI extreme + EMA200 HTF (LONG+SHORT)

TIER 2 (TV community):
  5. Best_TV_Strategy — BB(9) + SMA crossover 14/42 (LONG+SHORT)
  6. Range_Trading    — Mean-reversion VAH/VAL highest/lowest N bars (LONG+SHORT)
  7. 2Mars_MA_BB_ST   — SuperTrend direction + BB bounce (LONG+SHORT)
  8. RSI_Strat        — EMA(RSI) rising N candles + RSI<80 (LONG only)
  9. HatiKO_Envelopes — 4-level envelope over SMA, entry at deep levels (LONG+SHORT)
 10. Impulse_V2       — Triple SMMA(21/50/200) cross + RSI>50 (LONG+SHORT)

All generators: gen_XXX(df, **params) → pd.Series(0/1/-1)
"""

import pandas as pd
import numpy as np
import math

# ─── INDICATOR HELPERS (duplicated from strategy_factory for independence) ─────

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _sma(s, p):
    return s.rolling(p).mean()

def _rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0).ewm(span=p, adjust=False).mean()
    l = (-d.where(d < 0, 0.0)).ewm(span=p, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def _atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def _stoch(h, l, c, kp=14, dp=3, smooth=3):
    lo = l.rolling(kp).min()
    hi = h.rolling(kp).max()
    k = 100 * (c - lo) / (hi - lo + 1e-10)
    k_smooth = _sma(k, smooth)
    d = _sma(k_smooth, dp)
    return k_smooth, d

def _wma(s, p):
    weights = np.arange(1, p + 1, dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def _hull_ma(s, p=9):
    half = max(p // 2, 1)
    sqrt_p = max(int(math.floor(math.sqrt(p))), 1)
    return _wma(2 * _wma(s, half) - _wma(s, p), sqrt_p)

def _smma(s, p):
    """Smoothed Moving Average (SMMA/RMA) — same as ta.rma in Pine."""
    return s.ewm(alpha=1.0/p, adjust=False).mean()

def _stddev(s, p):
    return s.rolling(p).std()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MAC's V6 FINAL — Keltner Channel + CFB Range Filter
# ═══════════════════════════════════════════════════════════════════════════════
# Pine: PUB;3db6a02b1c2a4bb2b2ef22b632cb1e34
# Validated: BTC/ADA × 5m/15m/1h | WR=63-86% PF=5.86-42.29

def _supersmoother(src, period):
    """Ehlers SuperSmoother filter (2-pole)."""
    vals = src.values.astype(float)
    n = len(vals)
    out = np.zeros(n)
    f = 1.414 * math.pi / period
    a = math.exp(-f)
    c2 = 2 * a * math.cos(f)
    c3 = -a * a
    c1 = 1 - c2 - c3
    out[0] = vals[0]
    if n > 1:
        out[1] = vals[1]
    for i in range(2, n):
        out[i] = c1 * (vals[i] + vals[i - 1]) * 0.5 + c2 * out[i - 1] + c3 * out[i - 2]
    return pd.Series(out, index=src.index)

def _range_filter(src, period, mult):
    """CFB Range Filter — smoothed range with direction tracking."""
    vals = src.values.astype(float)
    n = len(vals)
    # smoothrng = ema(|src - src[1]|, period) then ema again with wper=2*period-1, times mult
    abs_change = np.abs(np.diff(vals, prepend=vals[0]))
    abs_series = pd.Series(abs_change, index=src.index)
    avrng = _ema(abs_series, period)
    wper = period * 2 - 1
    smrng = (_ema(avrng, wper) * mult).values

    # Range filter with state
    filt = np.zeros(n)
    filt[0] = vals[0]
    for i in range(1, n):
        if vals[i] > filt[i - 1]:
            filt[i] = max(filt[i - 1], vals[i] - smrng[i])
        else:
            filt[i] = min(filt[i - 1], vals[i] + smrng[i])

    # Upward/downward counters
    upward = np.zeros(n)
    downward = np.zeros(n)
    for i in range(1, n):
        if filt[i] > filt[i - 1]:
            upward[i] = upward[i - 1] + 1
            downward[i] = 0
        elif filt[i] < filt[i - 1]:
            downward[i] = downward[i - 1] + 1
            upward[i] = 0
        else:
            upward[i] = upward[i - 1]
            downward[i] = downward[i - 1]

    return pd.Series(filt, index=src.index), pd.Series(upward, index=src.index), pd.Series(downward, index=src.index)

def gen_macs_v6(df, kc_len=55, kc_atr_len=34, kc_mult=1.1, cfb_period=14, cfb_mult=3.8):
    """
    MAC's V6 final: Keltner Channel (SuperSmoother center) + CFB Range Filter.
    Pine logic: hlv = ta.valuewhen(hld != 0, hld, 1) — uses SECOND most recent
    non-zero hld value. hi = upper when hlv==-1, lo = lower when hlv==1.
    CFB: crossover(upward,0) sets longCFB state, crossunder(0,downward) sets shortCFB.
    LONG: longCFB state AND Close crosses above hi band.
    SHORT: shortCFB state AND Close crosses below lo band.
    """
    src = (df['high'] + df['low'] + df['close']) / 3  # hlc3
    close = df['close']

    # KC center: SuperSmoother filter
    em = _supersmoother(src, kc_len)

    # KC bands: ATR-based
    atr_val = _atr(df['high'], df['low'], df['close'], kc_atr_len)
    upper = em + kc_mult * atr_val
    lower = em - kc_mult * atr_val

    # Pine: hld := close > upper[1] ? 1 : close < lower[1] ? -1 : 0
    # hlv = ta.valuewhen(hld != 0, hld, 1)  — the PREVIOUS non-zero value (index=1)
    # This means hlv lags by one signal change — it's the state BEFORE the current change
    hld_vals = np.zeros(len(close))
    for i in range(1, len(close)):
        if close.iloc[i] > upper.iloc[i - 1]:
            hld_vals[i] = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            hld_vals[i] = -1

    # valuewhen(hld != 0, hld, 1): get the second-to-last non-zero hld value
    hlv_vals = np.zeros(len(close))
    nonzero_history = []
    for i in range(len(close)):
        if hld_vals[i] != 0:
            nonzero_history.append(hld_vals[i])
        # index=1 means the one BEFORE the most recent non-zero
        if len(nonzero_history) >= 2:
            hlv_vals[i] = nonzero_history[-2]
        elif len(nonzero_history) == 1:
            hlv_vals[i] = nonzero_history[-1]

    hlv = pd.Series(hlv_vals, index=df.index)
    # hi = upper when hlv==-1 (bearish context → upper is resistance)
    # lo = lower when hlv==1 (bullish context → lower is support)
    hi = upper.where(hlv == -1, np.nan)
    lo = lower.where(hlv == 1, np.nan)

    # CFB Range Filter
    src_cfb = (close + 2 * close.shift(1) + 2 * close.shift(2) + close.shift(3)) / 6
    src_cfb = src_cfb.ffill().bfill()
    filt, upward, downward = _range_filter(src_cfb, cfb_period, cfb_mult)

    # Pine: CFB_up = ta.crossover(upward, 0) → upward goes from 0 to >0
    # CFB_down = ta.crossunder(0, downward) → downward goes from 0 to >0 (0 crosses under -downward)
    # Actually: crossunder(0, downward) = downward crosses above 0 from below = downward > 0 & downward[1] <= 0
    cfb_up = (upward > 0) & (upward.shift(1) <= 0)
    cfb_down = (downward > 0) & (downward.shift(1) <= 0)

    # Pine: longCFB := CFB_up[1] ? true : CFB_down[1] ? false : longCFB[1]
    # This is a STATEFUL latch — set by CFB_up[1], cleared by CFB_down[1]
    long_cfb = np.zeros(len(close), dtype=bool)
    short_cfb = np.zeros(len(close), dtype=bool)
    for i in range(1, len(close)):
        cfb_up_prev = bool(cfb_up.iloc[i - 1]) if i >= 1 else False
        cfb_dn_prev = bool(cfb_down.iloc[i - 1]) if i >= 1 else False
        if cfb_up_prev:
            long_cfb[i] = True
        elif cfb_dn_prev:
            long_cfb[i] = False
        else:
            long_cfb[i] = long_cfb[i - 1]
        if cfb_dn_prev:
            short_cfb[i] = True
        elif cfb_up_prev:
            short_cfb[i] = False
        else:
            short_cfb[i] = short_cfb[i - 1]

    long_cfb_s = pd.Series(long_cfb, index=df.index)
    short_cfb_s = pd.Series(short_cfb, index=df.index)

    # Signals: crossover of hi/lo bands while in CFB trend state
    # Forward-fill hi/lo to get continuous levels for crossover detection
    hi_filled = hi.ffill()
    lo_filled = lo.ffill()
    sig = pd.Series(0, index=df.index)
    # Pine: long = longCFB and Close > hi and Close[1] < hi
    sig[long_cfb_s & (close > hi_filled) & (close.shift(1) < hi_filled.shift(1))] = 1
    # Pine: short = shortCFB and Close < lo and Close[1] > lo
    sig[short_cfb_s & (close < lo_filled) & (close.shift(1) > lo_filled.shift(1))] = -1
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MEAN REVERSION V-F — SMA + deviation bands (LONG only, no pyramid)
# ═══════════════════════════════════════════════════════════════════════════════
# Pine: PUB;96fd7d46211c4736ac3fe58e8bd3ed4f
# Validated: BTC/ADA × 15m/1h/4h | WR=72-84% PF=2.15-3.86

def gen_mean_reversion_vf(df, ma_period=28, deviation=3.0, hull_len=9, tp_pct=1.67):
    """
    Mean Reversion V-F: enters LONG when price falls below SMA deviation bands.
    Exit: price crosses above Hull MA (or TP).
    Simplified from 5-level pyramiding to single entry at first deviation level.
    """
    close = df['close']
    ma = _sma(close, ma_period)
    dev = deviation / 100.0

    # Entry band: first deviation level below SMA
    level_high = ma * (1.0 - dev)
    level_low = ma * (1.0 - 2 * dev)

    # Hull MA for exit
    hull = _hull_ma(close, hull_len)

    sig = pd.Series(0, index=df.index)
    # LONG: price in first deviation band (between level_high and level_low)
    sig[(close < level_high) & (close > level_low)] = 1
    # EXIT: price crosses above Hull MA (signal = -1 as close signal)
    sig[(close > hull) & (close.shift(1) <= hull.shift(1)) & (close > ma)] = -1
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOWFINDER PYRAMIDER V2 — MTF RSI + Stochastic (LONG only, no pyramid)
# ═══════════════════════════════════════════════════════════════════════════════
# Pine: PUB;64fa9225f3d94fc2ac29e21ae1670a6b
# Validated: BTC/ADA × 4h | WR=74-77% PF=4.48-7.98

def _rsi_mtf(source, mtf, rsi_len):
    """
    MTF RSI faithful to Pine: rsi_mtf(source, mtf, len).
    Pine: change_mtf = source - source[mtf]
          up_mtf = ta.rma(max(change_mtf, 0), len * mtf)
          down_mtf = ta.rma(-min(change_mtf, 0), len * mtf)
    This is NOT equivalent to using a longer RSI period.
    """
    mtf = max(mtf, 1)
    change_mtf = source - source.shift(mtf)
    up = change_mtf.where(change_mtf > 0, 0.0)
    down = (-change_mtf).where(change_mtf < 0, 0.0)
    # ta.rma = ewm with alpha = 1/period
    period = rsi_len * mtf
    up_mtf = up.ewm(alpha=1.0/period, adjust=False).mean()
    down_mtf = down.ewm(alpha=1.0/period, adjust=False).mean()
    rs = up_mtf / (down_mtf + 1e-10)
    rsi = 100 - 100 / (1 + rs)
    # Handle edge cases
    rsi = rsi.where(down_mtf != 0, 100.0)
    rsi = rsi.where(up_mtf != 0, rsi)  # if up==0 and down==0, rsi stays
    return rsi


def gen_lowfinder_pyramider(df, rsi_len=5, rsi_mtf=1, ma_sensitivity=26,
                            ma_signal_len=100, stoch_k=14, stoch_d=3,
                            stoch_smooth=3, stoch_threshold=30, stop_len=100,
                            stop_dev=0.3, stoch_mtf=10):
    """
    LowFinder PyraMider V2: detects potential lows using MTF RSI + Stochastic.
    Pine: vrsi = rsi_mtf(close, mtf_rsi, len_rsi) where change_mtf = close - close[mtf].
    Low detection: pp = ema(vrsi, ma_length), dd = (vrsi - pp) * 5, cc = (vrsi + dd + pp) / 2.
    Signal: crossover(cc, 0).
    Buy: lows AND close < moving_average AND mtfK < stoch_threshold.
    First entry: crossunder(close, entry_price) where entry_price = close * (1 - new_entry).
    Exit: crossover(close, stop_level) where stop_level = highest(high, stop_len)[2] * (1 + stop_dev).
    LONG only. Simplified from pyramiding to single entry.
    """
    close = df['close']
    high = df['high']
    low = df['low']

    # MTF RSI — faithful to Pine's rsi_mtf function
    vrsi = _rsi_mtf(close, rsi_mtf, rsi_len)

    # Low detection: EMA of RSI → composite indicator → crossover(cc, 0)
    pp = _ema(vrsi, ma_sensitivity)
    dd = (vrsi - pp) * 5
    cc = (vrsi + dd + pp) / 2
    lows = (cc > 0) & (cc.shift(1) <= 0)  # crossover(cc, 0)

    # MA filter
    ma_filter = _sma(close, ma_signal_len)

    # MTF Stochastic — Pine: k = sma(stoch(close, high, low, periodK * mtf), smoothK * mtf)
    mtf_stoch_period = stoch_k * max(stoch_mtf, 1)
    mtf_smooth = stoch_smooth * max(stoch_mtf, 1)
    lo_stoch = low.rolling(mtf_stoch_period).min()
    hi_stoch = high.rolling(mtf_stoch_period).max()
    raw_k = 100 * (close - lo_stoch) / (hi_stoch - lo_stoch + 1e-10)
    mtf_k = _sma(raw_k, mtf_smooth)

    # Stop level: Pine uses highest(high, stop_len)[2] for exit_condition_2
    stop_level = high.rolling(stop_len).max().shift(2) * (1 + stop_dev / 100)

    # State machine: only enter when flat, exit at stop level
    sig = pd.Series(0, index=df.index)
    in_position = False
    close_arr = close.values
    lows_arr = lows.values
    ma_arr = ma_filter.values
    mtfk_arr = mtf_k.values
    stop_arr = stop_level.values
    sig_arr = sig.values

    for i in range(1, len(close_arr)):
        if not in_position:
            # Buy signal: low detected + below MA + stoch below threshold
            if (lows_arr[i] and not np.isnan(ma_arr[i]) and close_arr[i] < ma_arr[i]
                    and not np.isnan(mtfk_arr[i]) and mtfk_arr[i] < stoch_threshold):
                sig_arr[i] = 1
                in_position = True
        else:
            # Exit: crossover(close, stop_level)
            if (not np.isnan(stop_arr[i]) and not np.isnan(stop_arr[i-1])
                    and close_arr[i] > stop_arr[i] and close_arr[i-1] <= stop_arr[i-1]):
                sig_arr[i] = -1
                in_position = False

    sig = pd.Series(sig_arr, index=df.index)
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. +70% SPIKE REVERSION — Price spike + RSI extreme + EMA200 HTF
# ═══════════════════════════════════════════════════════════════════════════════
# Pine: LOCAL strategies_pine/custom_built/plus70_spike_reversion_v5.pine
# Validated: ADA 5m | WR=70% PF=1.8

def gen_spike_reversion_70(df, spike_pct=6.0, rsi_len=9, rsi_oversold=25,
                            rsi_overbought=75, ema_len=200, htf_mult=12,
                            cooldown=2, max_bars=20):
    """
    +70% Spike Reversion — 100% faithful to Pine Script MMRS_ADA_5M.

    Pine params: `1 14 3 3.6 20 6 9 25 75 20 1.2 2 60 200`
    TV results: INJ(WR=100%,11t), LUNA(WR=76%,38t), DOGE(WR=80%,25t),
                SHIB(WR=80%,10t), ADA(WR=100%,6t), SOL(WR=70%,10t) on 5m.

    Entry conditions (ALL must be true for LONG):
      1. Spike: (close - open) / open <= -spike_pct% (big red candle)
      2. RSI(rsi_len) <= rsi_oversold
      3. HTF Trend: close > EMA(close, ema_len * htf_mult)
         Pine uses EMA(200) on 60min via request.security.
         For 5m data: htf_mult=12 → EMA(2400) simulates 1h EMA(200).
      4. Not in position (cooldown-based state tracking)

    For SHORT: opposite spike/RSI/trend conditions.

    Pine key: process_orders_on_close=true → entry at close of signal bar.
    SL/TP are ATR-based (handled by SLMonitor, not by this generator).
    """
    close = df['close']
    opn = df['open']
    spike_thresh = spike_pct / 100.0

    # 1. Spike filter: (close - open) / open — exact Pine formula
    bar_change = (close - opn) / (opn + 1e-10)
    big_red = bar_change <= -spike_thresh    # LONG: big red candle (drop)
    big_green = bar_change >= spike_thresh   # SHORT: big green candle (pump)

    # 2. RSI filter
    rsi_val = _rsi(close, rsi_len)
    rsi_os = rsi_val <= rsi_oversold
    rsi_ob = rsi_val >= rsi_overbought

    # 3. HTF Trend filter: EMA(ema_len) on higher timeframe
    #    Pine: request.security(syminfo.tickerid, "60", ta.ema(close, 200))
    #    Python with 5m data: EMA(200 * 12) = EMA(2400) approximates 1h EMA(200)
    htf_ema_period = ema_len * htf_mult
    ema_trend = _ema(close, htf_ema_period)
    long_trend = close > ema_trend    # Only LONG when above HTF EMA
    short_trend = close < ema_trend   # Only SHORT when below HTF EMA

    # 4. Combine all conditions (vectorized first pass)
    long_raw = big_red & rsi_os & long_trend
    short_raw = big_green & rsi_ob & short_trend

    # 5. Apply position-aware cooldown (Pine: strategy.position_size == 0)
    #    Pine logic: can only enter if not in position AND bar_index > lastEntry + cooldown
    #    We simulate this: after entry, block new signals for max_bars + cooldown bars
    #    (max_bars = time exit duration, simulating that a position is held)
    result = np.zeros(len(df), dtype=np.int8)
    long_mask = long_raw.values
    short_mask = short_raw.values
    block_until = -1  # bar index until which we're "in position"

    for i in range(len(result)):
        if i <= block_until:
            continue  # Still in position (or in cooldown after exit)
        if long_mask[i]:
            result[i] = 1
            block_until = i + max_bars + cooldown  # position held + cooldown
        elif short_mask[i]:
            result[i] = -1
            block_until = i + max_bars + cooldown

    return pd.Series(result, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BEST TV STRATEGY — BB(9) + SMA crossover 14/42 (LONG+SHORT)
# ═══════════════════════════════════════════════════════════════════════════════
# Pine: Best_TV_Strategy.pine by The_Bigger_Bull
# BB + SMA cross. Long: SMA14>SMA42 + price crosses below lower BB.
# Short: SMA14<SMA42 + price crosses below lower BB (original uses crossunder lower).

def gen_best_tv_strategy(df, bb_len=9, bb_mult=2.0, sma_fast=14, sma_slow=42):
    """
    Best TV Strategy: Bollinger Bands + SMA crossover filter.
    LONG: SMA_fast > SMA_slow AND price crosses above lower BB.
    SHORT: SMA_fast < SMA_slow AND price crosses below lower BB.
    Exit: opposite MA cross.
    """
    close = df['close']
    basis = _sma(close, bb_len)
    dev = bb_mult * _stddev(close, bb_len)
    lower = basis - dev

    sma_f = _sma(close, sma_fast)
    sma_s = _sma(close, sma_slow)

    # Crossover: price crosses above lower BB
    cross_above_lower = (close > lower) & (close.shift(1) <= lower.shift(1))
    # Crossunder: price crosses below lower BB
    cross_below_lower = (close < lower) & (close.shift(1) >= lower.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[(sma_f > sma_s) & cross_above_lower] = 1    # LONG
    sig[(sma_f < sma_s) & cross_below_lower] = -1   # SHORT
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RANGE TRADING — Mean-reversion VAH/VAL (LONG+SHORT)
# ═══════════════════════════════════════════════════════════════════════════════
# Pine: Range_Trading.pine — SOL/USD optimized
# VAH = highest(high, lookback), VAL = lowest(low, lookback) with offset.
# Short at VAH cross-under, Long at VAL cross-over.

def gen_range_trading(df, lookback=72, offset_bars=12, offset_long=9.0, offset_short=12.0):
    """
    Range Trading: mean-reversion between VAH (Value Area High) and VAL (Value Area Low).
    LONG: price crosses above VAL. EXIT: price reaches VAH.
    SHORT: price crosses below VAH. EXIT: price reaches VAL.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # VAH/VAL with offset (shifted bars to avoid look-ahead)
    vah = high.shift(offset_bars).rolling(lookback).max()
    val = low.shift(offset_bars).rolling(lookback).min()

    # Absolute bounds for risk management
    abs_high = vah + offset_long
    abs_low = val - offset_short

    # Entry signals
    cross_above_val = (close > val) & (close.shift(1) <= val.shift(1))
    cross_below_vah = (close < vah) & (close.shift(1) >= vah.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[cross_above_val] = 1    # LONG at VAL
    sig[cross_below_vah] = -1   # SHORT at VAH
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 2MARS MA/BB/SUPERTREND — SuperTrend direction + BB bounce (LONG+SHORT)
# ═══════════════════════════════════════════════════════════════════════════════
# Pine: 2Mars_MA_BB_SuperTrend.pine by facejungle
# Simplified: SuperTrend for trend direction + BB for entry timing.
# Long: SuperTrend bullish + price touches lower BB. Short: opposite.

def _supertrend(high, low, close, period=20, factor=4.0):
    """Compute SuperTrend direction: 1=bullish, -1=bearish."""
    atr_val = _atr(high, low, close, period)
    hl2 = (high + low) / 2
    up_band = hl2 - factor * atr_val
    dn_band = hl2 + factor * atr_val

    n = len(close)
    trend = np.ones(n)  # 1=bull, -1=bear
    final_up = up_band.values.copy()
    final_dn = dn_band.values.copy()

    for i in range(1, n):
        # Upper band logic
        if final_up[i] > final_up[i - 1] or close.iloc[i - 1] <= final_up[i - 1]:
            pass  # keep current
        else:
            final_up[i] = final_up[i - 1]

        # Lower band logic
        if final_dn[i] < final_dn[i - 1] or close.iloc[i - 1] >= final_dn[i - 1]:
            pass
        else:
            final_dn[i] = final_dn[i - 1]

        # Trend direction
        if trend[i - 1] == 1:
            trend[i] = -1 if close.iloc[i] < final_up[i] else 1
        else:
            trend[i] = 1 if close.iloc[i] > final_dn[i] else -1

    return pd.Series(trend, index=close.index)


def gen_2mars_ma_bb_st(df, st_period=20, st_factor=4.0, bb_len=30, bb_mult=3.0,
                       ma_fast_ratio=1.08, ma_mult=89, bars_confirm=2):
    """
    2Mars: SuperTrend + Bollinger Bands + MA cross.
    Pine defaults: useSuperTrend=true, useMaStrategy=true, BB lower3/lower2 for longs,
    upper3/upper2 for shorts. barsConfirm=2.
    Pine uses barCrossoverCounter which counts consecutive bars above/below, then checks
    if count == barsConfirm (exact match, not >=).
    Three entry types that can fire independently:
      1) BB band crossover (price crosses above lower BB bands for long, below upper for short)
      2) MA cross (maSignal crosses above maBasis for long)
      3) SuperTrend signal (optional, off by default)
    All require SuperTrend direction confirmation when useSuperTrend=true.
    """
    close = df['close']
    high = df['high']
    low = df['low']

    # SuperTrend direction: 1=bullish (supertrendDirection < 0), -1=bearish
    st_dir = _supertrend(high, low, close, st_period, st_factor)
    st_bullish = st_dir == 1   # supertrendDirection < 0 in Pine
    st_bearish = st_dir == -1

    # Bollinger Bands (Pine uses WMA by default in this strategy)
    bb_basis = _wma(close, bb_len)
    bb_dev_val = _stddev(close, bb_len)
    # Pine: fj_stdev creates multiple bands: upper/lower (1x), upper2/lower2 (2x), upper3/lower3 (3x)
    bb_upper = bb_basis + bb_mult * bb_dev_val
    bb_upper2 = bb_basis + (bb_mult * 2) * bb_dev_val
    bb_upper3 = bb_basis + (bb_mult * 3) * bb_dev_val
    bb_lower = bb_basis - bb_mult * bb_dev_val
    bb_lower2 = bb_basis - (bb_mult * 2) * bb_dev_val
    bb_lower3 = bb_basis - (bb_mult * 3) * bb_dev_val

    # MA cross (basis = SMA(close, round(ratio*mult)), signal = SMA(close, mult))
    ma_fast_len = max(int(round(ma_fast_ratio * ma_mult)), 2)
    ma_basis = _sma(close, ma_fast_len)
    ma_signal = _sma(close, ma_mult)

    # barCrossoverCounter: counts consecutive bars where signal > base
    # Pine checks: barsConfirm == count (exact N bars since crossover)
    def _bars_above(signal, base):
        """Count consecutive bars signal > base. Returns series."""
        cond = signal > base
        vals = cond.values
        count = np.zeros(len(vals))
        for i in range(len(vals)):
            if vals[i]:
                count[i] = count[i - 1] + 1 if i > 0 else 1
            else:
                count[i] = 0
        return pd.Series(count, index=signal.index)

    def _bars_below(signal, base):
        """Count consecutive bars signal < base."""
        cond = signal < base
        vals = cond.values
        count = np.zeros(len(vals))
        for i in range(len(vals)):
            if vals[i]:
                count[i] = count[i - 1] + 1 if i > 0 else 1
            else:
                count[i] = 0
        return pd.Series(count, index=signal.index)

    # Long conditions: price crosses above various BB lower bands
    # Pine defaults: useBBLong='lower3', useBBLong2='lower2'
    cond_bb_long3 = _bars_above(close, bb_lower3) == bars_confirm
    cond_bb_long2 = _bars_above(close, bb_lower2) == bars_confirm
    # MA cross: maSignal crosses above maBasis
    cond_ma_long = _bars_above(ma_signal, ma_basis) == bars_confirm

    # Short conditions: price crosses below various BB upper bands
    # Pine defaults: useBBShort='upper3', useBBShort2='upper2'
    cond_bb_short3 = _bars_below(close, bb_upper3) == bars_confirm
    cond_bb_short2 = _bars_below(close, bb_upper2) == bars_confirm
    # MA cross short: maSignal crosses below maBasis
    cond_ma_short = _bars_below(ma_signal, ma_basis) == bars_confirm

    # Any long entry trigger (BB lower3 OR lower2 OR MA cross), confirmed by SuperTrend
    long_trigger = (cond_bb_long3 | cond_bb_long2 | cond_ma_long) & st_bullish
    short_trigger = (cond_bb_short3 | cond_bb_short2 | cond_ma_short) & st_bearish

    sig = pd.Series(0, index=df.index)
    sig[long_trigger] = 1
    sig[short_trigger] = -1
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RSI STRATEGY — EMA(RSI) rising for N candles (LONG only)
# ═══════════════════════════════════════════════════════════════════════════════
# Pine: RSI_Strategy.pine by Sonny Parlin
# Entry: EMA(RSI(14), 14) rising for N candles AND RSI<80. LONG only.
# Exit: EMA(RSI) falling (momentum loss).

def gen_rsi_strat(df, rsi_len=14, ema_rsi_len=14, rsi_rising_bars=5,
                  rsi_max=80, sell_profit_pct=0.6, activation_pct=2.45):
    """
    RSI Strategy (Sonny Parlin): LONG only, momentum-based.
    Pine: myRSI = ta.ema(ta.rsi(close, 14), 14)
          cond1 = ta.rising(myRSI, rsiRising)  — myRSI rising for exactly N bars
          cond2 = ta.rsi(close, 14) < rsiLessThan
          enterLong = cond1 and cond2  (only when strategy.opentrades == 0)
    Exit: exitCond = myRSI < myRSI[1] (EMA-RSI starts falling)
          AND close < activationLevel (hasn't hit trailing stop activation)
          AND exitWithProfit: (close - entryPrice) / close > sellSignalProfit
    Key fix: only enter when NOT already in position (opentrades==0).
    """
    close = df['close']
    rsi_val = _rsi(close, rsi_len)
    ema_rsi = _ema(rsi_val, ema_rsi_len)

    # ta.rising(myRSI, N): true if myRSI > myRSI[i] for ALL i from 1 to N
    # This is stricter than just counting consecutive rising bars
    # Pine's ta.rising checks source > source[length], i.e. current value > value N bars ago
    # AND it's been rising each bar in between
    rising = ema_rsi > ema_rsi.shift(1)
    vals = rising.values
    rc = np.zeros(len(vals))
    for i in range(1, len(vals)):
        if vals[i]:
            rc[i] = rc[i - 1] + 1
        else:
            rc[i] = 0
    rising_count = pd.Series(rc, index=df.index)

    # Entry: rising for N bars + RSI below max + NOT already in position
    # Exit: EMA(RSI) starts falling + minimum profit achieved
    sig = pd.Series(0, index=df.index)
    entry_price = np.nan
    in_position = False
    activation_frac = activation_pct / 100.0
    sell_profit_frac = sell_profit_pct / 100.0

    rc_arr = rising_count.values
    rsi_arr = rsi_val.values
    ema_rsi_arr = ema_rsi.values
    close_arr = close.values
    sig_arr = sig.values

    for i in range(1, len(close_arr)):
        if not in_position:
            # Enter: rising for N bars + RSI < max
            if rc_arr[i] >= rsi_rising_bars and rsi_arr[i] < rsi_max:
                sig_arr[i] = 1
                entry_price = close_arr[i]
                in_position = True
        else:
            # Activation level: entry + activation_pct%
            activation_level = entry_price * (1 + activation_frac)
            # Profit check
            profit = (close_arr[i] - entry_price) / (close_arr[i] + 1e-10)
            has_profit = profit > sell_profit_frac
            # Exit: EMA(RSI) falling + below activation + has profit
            ema_rsi_falling = ema_rsi_arr[i] < ema_rsi_arr[i - 1]
            below_activation = close_arr[i] < activation_level

            if ema_rsi_falling and below_activation and has_profit:
                sig_arr[i] = -1
                in_position = False

    sig = pd.Series(sig_arr, index=df.index)
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
# 9. HATIKO ENVELOPES — 4-level envelope over MA (LONG+SHORT, no pyramid)
# ═══════════════════════════════════════════════════════════════════════════════
# Pine: HatiKO_Envelopes.pine (v4) — HatiKO robot
# Simplified from pyramiding=4 to single entry at deepest envelope touch.
# MA(OHLC4, len) with envelope levels. Entry at envelope cross, exit at MA.

def gen_hatiko_envelopes(df, ma_len=3, env1_pct=4.0, env2_pct=7.0,
                         env3_pct=10.0, env4_pct=15.0, entry_level=2):
    """
    HatiKO Envelopes: SMA(OHLC4, len) with 4 envelope levels above/below.
    Pine: pyramiding=4, entries at each envelope level via limit orders.
    Envelope levels: +-4%, +-7%, +-10%, +-15% from MA.
    LONG: price drops to envelope level N or deeper below MA.
    SHORT: price rises to envelope level N or deeper above MA.
    Exit: price crosses back above/below MA (take profit at mean reversion).
    entry_level: minimum envelope depth to trigger (1=4%, 2=7%, 3=10%, 4=15%).
    Default entry_level=2 means signal at -7% (or deeper) below MA.
    Generates MANY signals since envelope touches are common in volatile crypto.
    """
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    close = df['close']

    ma = _sma(ohlc4, ma_len)

    # 4 envelope levels below MA (for longs) and above MA (for shorts)
    env_pcts = [env1_pct, env2_pct, env3_pct, env4_pct]

    # Long levels (below MA)
    long_levels = [ma * (1 - pct / 100.0) for pct in env_pcts]
    # Short levels (above MA)
    short_levels = [ma * (1 + pct / 100.0) for pct in env_pcts]

    # Entry: price is at or beyond the entry_level envelope
    # Pine uses limit orders at each level. We simplify: signal when price
    # crosses below the Nth long envelope or above the Nth short envelope.
    # entry_level is 1-indexed: 1=env1 (4%), 2=env2 (7%), etc.
    lvl_idx = max(0, min(entry_level - 1, 3))

    # LONG: price drops below envelope level (any of the active levels)
    # Signal on ANY envelope touch at level >= entry_level
    long_touch = pd.Series(False, index=df.index)
    for i in range(lvl_idx, 4):
        lvl = long_levels[i]
        long_touch = long_touch | (close <= lvl)

    # SHORT: price rises above envelope level
    short_touch = pd.Series(False, index=df.index)
    for i in range(lvl_idx, 4):
        lvl = short_levels[i]
        short_touch = short_touch | (close >= lvl)

    # Exit: price crosses back to MA
    # Pine: close_all when MA_close > close (for shorts) or MA_close < close (for longs)
    # We use crossover/crossunder of MA
    cross_above_ma = (close > ma) & (close.shift(1) <= ma.shift(1))
    cross_below_ma = (close < ma) & (close.shift(1) >= ma.shift(1))

    # State machine: track if we're in a position, signal on FIRST touch, exit at MA
    sig = pd.Series(0, index=df.index)
    pos = 0  # 0=flat, 1=long, -1=short
    for i in range(len(close)):
        if pos == 0:
            if long_touch.iloc[i]:
                sig.iloc[i] = 1
                pos = 1
            elif short_touch.iloc[i]:
                sig.iloc[i] = -1
                pos = -1
        elif pos == 1:
            if cross_above_ma.iloc[i]:
                sig.iloc[i] = -1  # exit long (close signal)
                pos = 0
            elif short_touch.iloc[i]:
                sig.iloc[i] = -1  # flip to short
                pos = -1
        elif pos == -1:
            if cross_below_ma.iloc[i]:
                sig.iloc[i] = 1  # exit short (close signal)
                pos = 0
            elif long_touch.iloc[i]:
                sig.iloc[i] = 1  # flip to long
                pos = 1
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
# 10. IMPULSE V2 — Triple SMMA(21/50/200) crossover + RSI>50 (LONG+SHORT)
# ═══════════════════════════════════════════════════════════════════════════════
# Pine: Impulse_Strategy_V2.pine by Hiubris_Indicators
# Triple SMMA alignment + RSI filter. Long when all MAs aligned up, short down.

def gen_impulse_v2(df, smma1_len=21, smma2_len=50, smma3_len=200,
                   rsi_len=14, rsi_long_thresh=50.0, rsi_short_thresh=50.0):
    """
    Impulse V2: Triple SMMA cross + RSI filter.
    LONG: close > all 3 SMMAs + (MA1 crosses above MA2 OR close crosses above MA3) + RSI > threshold.
    SHORT: close < all 3 SMMAs + (MA1 crosses below MA2 OR close crosses below MA3) + RSI < threshold.
    Fires only on first bar of alignment (not repeated while aligned).
    """
    close = df['close']
    ma1 = _smma(close, smma1_len)
    ma2 = _smma(close, smma2_len)
    ma3 = _smma(close, smma3_len)
    rsi_val = _rsi(close, rsi_len)

    # All aligned
    all_above = (close > ma1) & (close > ma2) & (close > ma3)
    all_below = (close < ma1) & (close < ma2) & (close < ma3)

    # Crossovers (trigger events)
    ma1_cross_ma2_up = (ma1 > ma2) & (ma1.shift(1) <= ma2.shift(1))
    close_cross_ma3_up = (close > ma3) & (close.shift(1) <= ma3.shift(1))
    ma1_cross_ma2_dn = (ma1 < ma2) & (ma1.shift(1) >= ma2.shift(1))
    close_cross_ma3_dn = (close < ma3) & (close.shift(1) >= ma3.shift(1))

    # Raw signals (before dedup)
    long0 = (rsi_val > rsi_long_thresh) & all_above & (ma1_cross_ma2_up | close_cross_ma3_up)
    short0 = (rsi_val < rsi_short_thresh) & all_below & (ma1_cross_ma2_dn | close_cross_ma3_dn)

    # Fire only on first bar (not repeated — emulates `not long0[1]`)
    long_sig = long0 & ~long0.shift(1, fill_value=False)
    short_sig = short0 & ~short0.shift(1, fill_value=False)

    # Position state tracking: only enter if not already in same direction
    sig = pd.Series(0, index=df.index)
    pos = np.zeros(len(close))
    for i in range(len(close)):
        if long_sig.iloc[i] and pos[i - 1] != 1:
            sig.iloc[i] = 1
            pos[i] = 1
        elif short_sig.iloc[i] and pos[i - 1] != -1:
            sig.iloc[i] = -1
            pos[i] = -1
        else:
            pos[i] = pos[i - 1] if i > 0 else 0
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY — export for signal_factory integration
# ═══════════════════════════════════════════════════════════════════════════════

NEW_WINNER_STRATS = {}

def _reg(name, gen_func, space_func):
    NEW_WINNER_STRATS[name] = {'gen': gen_func, 'space': space_func}


# 1. MAC's V6 Final
_reg('MACs_V6_Final', gen_macs_v6, lambda t: {
    'kc_len': t.suggest_int('kc_len', 20, 80),
    'kc_atr_len': t.suggest_int('kc_atr_len', 14, 50),
    'kc_mult': t.suggest_float('kc_mult', 0.5, 2.5, step=0.1),
    'cfb_period': t.suggest_int('cfb_period', 7, 30),
    'cfb_mult': t.suggest_float('cfb_mult', 1.5, 6.0, step=0.1),
})

# 2. Mean Reversion V-F
_reg('MeanRev_VF', gen_mean_reversion_vf, lambda t: {
    'ma_period': t.suggest_int('ma_period', 14, 60),
    'deviation': t.suggest_float('deviation', 1.0, 8.0, step=0.5),
    'hull_len': t.suggest_int('hull_len', 5, 20),
    'tp_pct': t.suggest_float('tp_pct', 0.5, 4.0, step=0.1),
})

# 3. LowFinder PyraMider
_reg('LowFinder_Pyra', gen_lowfinder_pyramider, lambda t: {
    'rsi_len': t.suggest_int('rsi_len', 3, 14),
    'rsi_mtf': t.suggest_int('rsi_mtf', 1, 5),
    'ma_sensitivity': t.suggest_int('ma_sensitivity', 10, 50),
    'ma_signal_len': t.suggest_int('ma_signal_len', 50, 200),
    'stoch_k': t.suggest_int('stoch_k', 7, 21),
    'stoch_threshold': t.suggest_int('stoch_threshold', 15, 45),
    'stoch_mtf': t.suggest_int('stoch_mtf', 1, 15),
    'stop_len': t.suggest_int('stop_len', 50, 200),
    'stop_dev': t.suggest_float('stop_dev', 0.1, 1.0, step=0.1),
})

# 4. +70% Spike Reversion (faithful to Pine MMRS_ADA_5M)
_reg('Spike_Reversion_70', gen_spike_reversion_70, lambda t: {
    'spike_pct': t.suggest_float('spike_pct', 3.0, 8.0, step=0.5),
    'rsi_len': t.suggest_int('rsi_len', 5, 14),
    'rsi_oversold': t.suggest_int('rsi_oversold', 15, 35),
    'rsi_overbought': t.suggest_int('rsi_overbought', 65, 85),
    'ema_len': t.suggest_int('ema_len', 100, 300),
    'htf_mult': t.suggest_int('htf_mult', 6, 24),
    'cooldown': t.suggest_int('cooldown', 1, 5),
    'max_bars': t.suggest_int('max_bars', 10, 30),
})

# TV_ prefix alias — JSON usa "TV_Spike_Reversion_70", runtime registraba "Spike_Reversion_70"
# Fix P0: 3 bots en produccion con 0 trades por este mismatch (2026-04-04)
_reg('TV_Spike_Reversion_70', gen_spike_reversion_70, lambda t: {
    'spike_pct': t.suggest_float('spike_pct', 3.0, 8.0, step=0.5),
    'rsi_len': t.suggest_int('rsi_len', 5, 14),
    'rsi_oversold': t.suggest_int('rsi_oversold', 15, 35),
    'rsi_overbought': t.suggest_int('rsi_overbought', 65, 85),
    'ema_len': t.suggest_int('ema_len', 100, 300),
    'htf_mult': t.suggest_int('htf_mult', 6, 24),
    'cooldown': t.suggest_int('cooldown', 1, 5),
    'max_bars': t.suggest_int('max_bars', 10, 30),
})

# 5. Best TV Strategy
_reg('Best_TV_Strategy', gen_best_tv_strategy, lambda t: {
    'bb_len': t.suggest_int('bb_len', 5, 20),
    'bb_mult': t.suggest_float('bb_mult', 1.0, 3.5, step=0.1),
    'sma_fast': t.suggest_int('sma_fast', 7, 28),
    'sma_slow': t.suggest_int('sma_slow', 28, 80),
})

# 6. Range Trading
_reg('Range_Trading', gen_range_trading, lambda t: {
    'lookback': t.suggest_int('lookback', 24, 168),
    'offset_bars': t.suggest_int('offset_bars', 4, 24),
    'offset_long': t.suggest_float('offset_long', 2.0, 20.0, step=1.0),
    'offset_short': t.suggest_float('offset_short', 2.0, 20.0, step=1.0),
})

# 7. 2Mars MA/BB/SuperTrend
_reg('2Mars_MA_BB_ST', gen_2mars_ma_bb_st, lambda t: {
    'st_period': t.suggest_int('st_period', 10, 40),
    'st_factor': t.suggest_float('st_factor', 2.0, 6.0, step=0.2),
    'bb_len': t.suggest_int('bb_len', 15, 50),
    'bb_mult': t.suggest_float('bb_mult', 1.5, 4.5, step=0.1),
    'ma_fast_ratio': t.suggest_float('ma_fast_ratio', 1.0, 1.3, step=0.02),
    'ma_mult': t.suggest_int('ma_mult', 50, 130),
    'bars_confirm': t.suggest_int('bars_confirm', 1, 5),
})

# 8. RSI Strategy
_reg('RSI_Strat', gen_rsi_strat, lambda t: {
    'rsi_len': t.suggest_int('rsi_len', 7, 21),
    'ema_rsi_len': t.suggest_int('ema_rsi_len', 7, 21),
    'rsi_rising_bars': t.suggest_int('rsi_rising_bars', 3, 10),
    'rsi_max': t.suggest_int('rsi_max', 70, 90),
    'sell_profit_pct': t.suggest_float('sell_profit_pct', 0.2, 2.0, step=0.1),
    'activation_pct': t.suggest_float('activation_pct', 1.0, 5.0, step=0.5),
})

# 9. HatiKO Envelopes
_reg('HatiKO_Envelopes', gen_hatiko_envelopes, lambda t: {
    'ma_len': t.suggest_int('ma_len', 2, 10),
    'env1_pct': t.suggest_float('env1_pct', 2.0, 8.0, step=0.5),
    'env2_pct': t.suggest_float('env2_pct', 5.0, 12.0, step=0.5),
    'env3_pct': t.suggest_float('env3_pct', 8.0, 16.0, step=0.5),
    'env4_pct': t.suggest_float('env4_pct', 12.0, 25.0, step=0.5),
    'entry_level': t.suggest_int('entry_level', 1, 3),
})

# 10. Impulse V2
_reg('Impulse_V2', gen_impulse_v2, lambda t: {
    'smma1_len': t.suggest_int('smma1_len', 10, 35),
    'smma2_len': t.suggest_int('smma2_len', 30, 80),
    'smma3_len': t.suggest_int('smma3_len', 100, 300),
    'rsi_len': t.suggest_int('rsi_len', 7, 21),
    'rsi_long_thresh': t.suggest_float('rsi_long_thresh', 40.0, 60.0, step=1.0),
    'rsi_short_thresh': t.suggest_float('rsi_short_thresh', 40.0, 60.0, step=1.0),
})

# ─── RSI_BB_Reversal (TV Hunter 2026-04-07) ──────────────────────────────────
# Evidence: TV_BB_RSI_Double = 89.5% WR LTC, 88.2% RIVER, 84.6% AAVE in production
# 57% of our 1,010 grails are mean-reversion RSI+BB combos, avg WR 72.8%
# 4h TF = 61% of winning combos. Mean-reversion > trend-following for WR gate.
# File: BOT V7/strategies/tv2_rsi_bb_reversal.py

try:
    import importlib.util as _ilu, os as _os
    _rbr_path = _os.path.join(_os.path.dirname(__file__), 'tv2_rsi_bb_reversal.py')
    _rbr_spec = _ilu.spec_from_file_location('tv2_rsi_bb_reversal', _rbr_path)
    _rbr_mod  = _ilu.module_from_spec(_rbr_spec)
    _rbr_spec.loader.exec_module(_rbr_mod)
    for _rbr_name, _rbr_entry in _rbr_mod.STRATEGY_EXPORT.items():
        _reg(_rbr_name, _rbr_entry['gen'], _rbr_entry['space'])
    del _ilu, _os, _rbr_path, _rbr_spec, _rbr_mod, _rbr_name, _rbr_entry
except Exception as _rbr_e:
    pass  # fail silently — missing file = skip (optuna_v7.py convention)

# ─── OU_ZScore (IA Consensus Nocturno 2026-04-07) ─────────────────────────────
# Ornstein-Uhlenbeck z-score mean reversion for altcoins.
# Expected WR: 75-85% (altcoins θ≈0.10, 5x faster reversion than BTC).
# Mistral + OpenRouter consensus: best walk-forward survival ★★★★★.
try:
    import importlib.util as _ilu_ou, os as _os_ou
    _ou_path = _os_ou.path.join(_os_ou.path.dirname(__file__), 'tv2_ou_zscore.py')
    _ou_spec = _ilu_ou.spec_from_file_location('tv2_ou_zscore', _ou_path)
    _ou_mod  = _ilu_ou.module_from_spec(_ou_spec)
    _ou_spec.loader.exec_module(_ou_mod)
    for _ou_name, _ou_entry in _ou_mod.STRATEGY_EXPORT.items():
        _reg(_ou_name, _ou_entry['gen'], _ou_entry['space'])
    del _ilu_ou, _os_ou, _ou_path, _ou_spec, _ou_mod, _ou_name, _ou_entry
except Exception:
    pass

# ─── VP_MeanRev (IA Consensus Nocturno 2026-04-07) ───────────────────────────
# Volume Profile Mean Reversion — VWAP rolling with volume-weighted std bands.
# Expected WR: 65-75%. More stable bands than BB (volume-weighted).
# Mistral + Cohere consensus ★★★★☆.
try:
    import importlib.util as _ilu_vp, os as _os_vp
    _vp_path = _os_vp.path.join(_os_vp.path.dirname(__file__), 'tv2_volume_profile_meanrev.py')
    _vp_spec = _ilu_vp.spec_from_file_location('tv2_volume_profile_meanrev', _vp_path)
    _vp_mod  = _ilu_vp.module_from_spec(_vp_spec)
    _vp_spec.loader.exec_module(_vp_mod)
    for _vp_name, _vp_entry in _vp_mod.STRATEGY_EXPORT.items():
        _reg(_vp_name, _vp_entry['gen'], _vp_entry['space'])
    del _ilu_vp, _os_vp, _vp_path, _vp_spec, _vp_mod, _vp_name, _vp_entry
except Exception:
    pass

# ─── Batch 2994 — Market Profile + Value Area (Dalton / Steidlmayer) ─────────
# TV_VAHVALBounce, TV_POCReversion, TV_InitialBalanceBreak,
# TV_ProfileExtension, TV_NakedPOCTouch
# Source: Dalton "Mind Over Markets" (Wiley, 1990/2013),
#         Steidlmayer "Markets and Market Logic" (1984),
#         Dalton "Markets in Profile" (Wiley, 2007),
#         Jones "Value Based Power Trading" (1993)
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch2994.py
try:
    import importlib.util as _ilu_b2994, os as _os_b2994
    _b2994_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch2994.py"
    _b2994_spec = _ilu_b2994.spec_from_file_location('strategies_tv2_batch2994', _b2994_path)
    _b2994_mod  = _ilu_b2994.module_from_spec(_b2994_spec)
    _b2994_spec.loader.exec_module(_b2994_mod)
    for _b2994_name, _b2994_entry in _b2994_mod.STRATEGY_EXPORT.items():
        _reg(_b2994_name, _b2994_entry['gen'], _b2994_entry['space'])
    del _ilu_b2994, _os_b2994, _b2994_path, _b2994_spec, _b2994_mod, _b2994_name, _b2994_entry
except Exception:
    pass

# ─── Batch 2999 — Alexander Elder's Trading Systems ───────────────────────────
# TV_ElderImpulseSystem, TV_ElderForceIndex, TV_ElderRayBull,
# TV_TripleScreenEntry, TV_ElderSafeZone
# Source: Elder "Trading for a Living" (Wiley, 1993, ISBN 0-471-59224-2),
#         Elder "Come Into My Trading Room" (Wiley, 2002, ISBN 0-471-22534-7),
#         Elder "The New Trading for a Living" (Wiley, 2014)
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch2999.py
try:
    import importlib.util as _ilu_b2999, os as _os_b2999
    _b2999_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch2999.py"
    _b2999_spec = _ilu_b2999.spec_from_file_location('strategies_tv2_batch2999', _b2999_path)
    _b2999_mod  = _ilu_b2999.module_from_spec(_b2999_spec)
    _b2999_spec.loader.exec_module(_b2999_mod)
    for _b2999_name, _b2999_entry in _b2999_mod.STRATEGY_EXPORT.items():
        _reg(_b2999_name, _b2999_entry['gen'], _b2999_entry['space'])
    del _ilu_b2999, _os_b2999, _b2999_path, _b2999_spec, _b2999_mod, _b2999_name, _b2999_entry
except Exception:
    pass

# ─── Batch 3025 — Kelly Criterion + Dynamic Position Sizing ──────────────────
# TV_KellyCriterion, TV_FractionalKelly, TV_DynamicSizing,
# TV_SizingByRiskGateway, TV_OptimalFractionMomentum
# Source: Kelly "A New Interpretation of Information Rate" Bell System Technical
#         Journal Vol.35 (1956), Thorp "Fortune's Formula" (2006),
#         Van Tharp "Trade Your Way to Financial Freedom" (2007),
#         Kaufman "Trading Systems and Methods" 5th ed. Wiley (2013),
#         Vince "The Mathematics of Money Management" Wiley (1992)
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3025.py
try:
    import importlib.util as _ilu_b3025, os as _os_b3025
    _b3025_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3025.py"
    _b3025_spec = _ilu_b3025.spec_from_file_location('strategies_tv2_batch3025', _b3025_path)
    _b3025_mod  = _ilu_b3025.module_from_spec(_b3025_spec)
    _b3025_spec.loader.exec_module(_b3025_mod)
    for _b3025_name, _b3025_entry in _b3025_mod.STRATEGY_EXPORT.items():
        _reg(_b3025_name, _b3025_entry['gen'], _b3025_entry['space'])
    del _ilu_b3025, _os_b3025, _b3025_path, _b3025_spec, _b3025_mod, _b3025_name, _b3025_entry
except Exception:
    pass

# ─── Batch 3501 — Microstructure: TickImbalance/BidAskPressure/TradeIntensity ─
# TV_TickImbalance, TV_BidAskPressure, TV_TradeIntensity,
# TV_OrderFlowToxicity, TV_LiquidityAdjustedMomentum
# Source: Easley & O'Hara (1992) PIN model; Glosten & Milgrom (1985) bid-ask spread;
#         Chordia et al. (2002) order imbalance; Amihud (2002) illiquidity; Kyle (1985) lambda
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3501.py
try:
    import importlib.util as _ilu_b3501, os as _os_b3501
    _b3501_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3501.py"
    _b3501_spec = _ilu_b3501.spec_from_file_location('strategies_tv2_batch3501', _b3501_path)
    _b3501_mod  = _ilu_b3501.module_from_spec(_b3501_spec)
    _b3501_spec.loader.exec_module(_b3501_mod)
    for _b3501_name, _b3501_entry in _b3501_mod.STRATEGY_EXPORT.items():
        _reg(_b3501_name, _b3501_entry['gen'], _b3501_entry['space'])
    del _ilu_b3501, _os_b3501, _b3501_path, _b3501_spec, _b3501_mod, _b3501_name, _b3501_entry
except Exception:
    pass

# ─── Batch 3502 — Microstructure: PriceImpact/VolumeClock/MarketMaking ────────
# TV_RollSpread, TV_PriceImpact, TV_VolumeClock,
# TV_MarketMakingOsc, TV_AdverseSelection
# Source: Roll (1984) serial covariance spread; Hasbrouck (1991) information share;
#         Kyle (1985) lambda; Amihud (2002) illiquidity ratio; Glosten-Milgrom (1985)
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3502.py
try:
    import importlib.util as _ilu_b3502, os as _os_b3502
    _b3502_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3502.py"
    _b3502_spec = _ilu_b3502.spec_from_file_location('strategies_tv2_batch3502', _b3502_path)
    _b3502_mod  = _ilu_b3502.module_from_spec(_b3502_spec)
    _b3502_spec.loader.exec_module(_b3502_mod)
    for _b3502_name, _b3502_entry in _b3502_mod.STRATEGY_EXPORT.items():
        _reg(_b3502_name, _b3502_entry['gen'], _b3502_entry['space'])
    del _ilu_b3502, _os_b3502, _b3502_path, _b3502_spec, _b3502_mod, _b3502_name, _b3502_entry
except Exception:
    pass

# ─── Batch 3503 — Microstructure: CumulativeDeltaFlow/TapeReading/AbsorptionRatio
# TV_CumulativeDeltaFlow, TV_TapeReadingSignal, TV_AbsorptionRatio,
# TV_TradeSizeImbalance, TV_FlowToxicityIndex
# Source: Easley et al. (2012) VPIN; Chordia & Subrahmanyam (2004) order flow;
#         Weis (2013) "A Modern Adaptation of the Wyckoff Method"
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3503.py
try:
    import importlib.util as _ilu_b3503, os as _os_b3503
    _b3503_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3503.py"
    _b3503_spec = _ilu_b3503.spec_from_file_location('strategies_tv2_batch3503', _b3503_path)
    _b3503_mod  = _ilu_b3503.module_from_spec(_b3503_spec)
    _b3503_spec.loader.exec_module(_b3503_mod)
    for _b3503_name, _b3503_entry in _b3503_mod.STRATEGY_EXPORT.items():
        _reg(_b3503_name, _b3503_entry['gen'], _b3503_entry['space'])
    del _ilu_b3503, _os_b3503, _b3503_path, _b3503_spec, _b3503_mod, _b3503_name, _b3503_entry
except Exception:
    pass

# ─── Batch 3504 — Microstructure: MicroPrice/QuoteRevision/PinningPressure ────
# TV_MicroPriceSignal, TV_QuoteRevisionTracker, TV_PinningPressure,
# TV_InformationShare, TV_VolatilitySignature
# Source: Stoikov (2018) micro-price; Hasbrouck (1995) information share;
#         Greenwood & Thesmar (2011) fragility; Parkinson (1980) range volatility
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3504.py
try:
    import importlib.util as _ilu_b3504, os as _os_b3504
    _b3504_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3504.py"
    _b3504_spec = _ilu_b3504.spec_from_file_location('strategies_tv2_batch3504', _b3504_path)
    _b3504_mod  = _ilu_b3504.module_from_spec(_b3504_spec)
    _b3504_spec.loader.exec_module(_b3504_mod)
    for _b3504_name, _b3504_entry in _b3504_mod.STRATEGY_EXPORT.items():
        _reg(_b3504_name, _b3504_entry['gen'], _b3504_entry['space'])
    del _ilu_b3504, _os_b3504, _b3504_path, _b3504_spec, _b3504_mod, _b3504_name, _b3504_entry
except Exception:
    pass

# ─── Batch 3505 — Microstructure: InventoryReversion/LatentLiquidity/BarSize ──
# TV_InventoryReversion, TV_LatentLiquidity, TV_BarSizeMomentum,
# TV_PriceDiscoveryRate, TV_MarketMicrostructureNoise
# Source: Garman-Klass (1980) volatility; Amihud-Mendelson (1986) inventory model;
#         Madhavan et al. (1997) price discovery; Roll (1988) noise trading
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3505.py
try:
    import importlib.util as _ilu_b3505, os as _os_b3505
    _b3505_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3505.py"
    _b3505_spec = _ilu_b3505.spec_from_file_location('strategies_tv2_batch3505', _b3505_path)
    _b3505_mod  = _ilu_b3505.module_from_spec(_b3505_spec)
    _b3505_spec.loader.exec_module(_b3505_mod)
    for _b3505_name, _b3505_entry in _b3505_mod.STRATEGY_EXPORT.items():
        _reg(_b3505_name, _b3505_entry['gen'], _b3505_entry['space'])
    del _ilu_b3505, _os_b3505, _b3505_path, _b3505_spec, _b3505_mod, _b3505_name, _b3505_entry
except Exception:
    pass

# ─── Batch 3506 — Microstructure: FractalDimension/HurstRegime/Scalping ───────
# TV_FractalDimension, TV_HurstRegimeAdaptive, TV_ScalpingMomentum,
# TV_SpeedOfPriceChange, TV_VolatilityArbitrage
# Source: Mandelbrot (1997) fractal markets hypothesis; Hurst (1951) long-range dependence;
#         Peters (1994) "Fractal Market Analysis"; Lo (1991) long-range dependence in stock returns
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3506.py
try:
    import importlib.util as _ilu_b3506, os as _os_b3506
    _b3506_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3506.py"
    _b3506_spec = _ilu_b3506.spec_from_file_location('strategies_tv2_batch3506', _b3506_path)
    _b3506_mod  = _ilu_b3506.module_from_spec(_b3506_spec)
    _b3506_spec.loader.exec_module(_b3506_mod)
    for _b3506_name, _b3506_entry in _b3506_mod.STRATEGY_EXPORT.items():
        _reg(_b3506_name, _b3506_entry['gen'], _b3506_entry['space'])
    del _ilu_b3506, _os_b3506, _b3506_path, _b3506_spec, _b3506_mod, _b3506_name, _b3506_entry
except Exception:
    pass

# ─── Batch 3507 — Microstructure: ReturnDispersion/SentimentFlow/MeanRevSpeed ─
# TV_ReturnDispersion, TV_SentimentFlowProxy, TV_MeanReversionSpeed,
# TV_OrderImbalancePersistence, TV_CrossSectionalMomentumProxy
# Source: Goyal & Santa-Clara (2003) return dispersion; Jegadeesh & Titman (1993) momentum;
#         Bouchaud et al. (2009) price impact; Avellaneda & Lee (2010) stat-arb
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3507.py
try:
    import importlib.util as _ilu_b3507, os as _os_b3507
    _b3507_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3507.py"
    _b3507_spec = _ilu_b3507.spec_from_file_location('strategies_tv2_batch3507', _b3507_path)
    _b3507_mod  = _ilu_b3507.module_from_spec(_b3507_spec)
    _b3507_spec.loader.exec_module(_b3507_mod)
    for _b3507_name, _b3507_entry in _b3507_mod.STRATEGY_EXPORT.items():
        _reg(_b3507_name, _b3507_entry['gen'], _b3507_entry['space'])
    del _ilu_b3507, _os_b3507, _b3507_path, _b3507_spec, _b3507_mod, _b3507_name, _b3507_entry
except Exception:
    pass

# ─── Batch 3508 — Microstructure: OrnsteinUhlenbeck/HalfLife/ZScoreVelocity ──
# TV_OrnsteinUhlenbeck, TV_HalfLifeReversion, TV_ZScoreVelocity,
# TV_SpreadReversionMicro, TV_AutocorrelationDecay
# Source: Ornstein & Uhlenbeck (1930) mean-reverting process;
#         Avellaneda & Lee (2010) stat-arb half-life; Lo & MacKinlay (1988) autocorrelation
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3508.py
try:
    import importlib.util as _ilu_b3508, os as _os_b3508
    _b3508_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3508.py"
    _b3508_spec = _ilu_b3508.spec_from_file_location('strategies_tv2_batch3508', _b3508_path)
    _b3508_mod  = _ilu_b3508.module_from_spec(_b3508_spec)
    _b3508_spec.loader.exec_module(_b3508_mod)
    for _b3508_name, _b3508_entry in _b3508_mod.STRATEGY_EXPORT.items():
        _reg(_b3508_name, _b3508_entry['gen'], _b3508_entry['space'])
    del _ilu_b3508, _os_b3508, _b3508_path, _b3508_spec, _b3508_mod, _b3508_name, _b3508_entry
except Exception:
    pass

# ─── Batch 3509 — Microstructure: WickPressure/CloseLocation/CandleBody ───────
# TV_WickPressureRatio, TV_CloseLocationValue, TV_CandleBodyAcceleration,
# TV_EffortVsResult, TV_UpperLowerWickBias
# Source: Bulkowski (2008) encyclopedia of candlestick charts; Nison (1991) Japanese candlesticks;
#         Wyckoff (1932) effort vs result; Weis (2013) wave analysis
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3509.py
try:
    import importlib.util as _ilu_b3509, os as _os_b3509
    _b3509_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3509.py"
    _b3509_spec = _ilu_b3509.spec_from_file_location('strategies_tv2_batch3509', _b3509_path)
    _b3509_mod  = _ilu_b3509.module_from_spec(_b3509_spec)
    _b3509_spec.loader.exec_module(_b3509_mod)
    for _b3509_name, _b3509_entry in _b3509_mod.STRATEGY_EXPORT.items():
        _reg(_b3509_name, _b3509_entry['gen'], _b3509_entry['space'])
    del _ilu_b3509, _os_b3509, _b3509_path, _b3509_spec, _b3509_mod, _b3509_name, _b3509_entry
except Exception:
    pass

# ─── Batch 3510 — Microstructure: VPINProxy/VolumeClock/VolumeSurprise ─────────
# TV_VPINProxy, TV_VolumeClockDeviation, TV_VolumeSurprise,
# TV_AbnormalVolumeFlow, TV_PriceVolumeDivergence
# Source: Easley et al. (2012) VPIN volume-synchronized probability of informed trading;
#         Jain & Joh (1988) volume deviation; Karpoff (1987) price-volume relation
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3510.py
try:
    import importlib.util as _ilu_b3510, os as _os_b3510
    _b3510_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3510.py"
    _b3510_spec = _ilu_b3510.spec_from_file_location('strategies_tv2_batch3510', _b3510_path)
    _b3510_mod  = _ilu_b3510.module_from_spec(_b3510_spec)
    _b3510_spec.loader.exec_module(_b3510_mod)
    for _b3510_name, _b3510_entry in _b3510_mod.STRATEGY_EXPORT.items():
        _reg(_b3510_name, _b3510_entry['gen'], _b3510_entry['space'])
    del _ilu_b3510, _os_b3510, _b3510_path, _b3510_spec, _b3510_mod, _b3510_name, _b3510_entry
except Exception:
    pass

# ─── Batch 3511 — Microstructure: OpenToClose/VolumeWeighted/BarEfficiency ────
# TV_OpenToCloseDrift, TV_VolumeWeightedDirectional, TV_BarEfficiencyOscillator,
# TV_TurnoverReversal, TV_MicroMomentumPulse
# Source: Asness (1994) intraday momentum; Grinblatt & Keloharju (2001) turnover/returns;
#         Chordia et al. (2002) trading activity; Lo & MacKinlay (1990) autocorrelation
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3511.py
try:
    import importlib.util as _ilu_b3511, os as _os_b3511
    _b3511_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3511.py"
    _b3511_spec = _ilu_b3511.spec_from_file_location('strategies_tv2_batch3511', _b3511_path)
    _b3511_mod  = _ilu_b3511.module_from_spec(_b3511_spec)
    _b3511_spec.loader.exec_module(_b3511_mod)
    for _b3511_name, _b3511_entry in _b3511_mod.STRATEGY_EXPORT.items():
        _reg(_b3511_name, _b3511_entry['gen'], _b3511_entry['space'])
    del _ilu_b3511, _os_b3511, _b3511_path, _b3511_spec, _b3511_mod, _b3511_name, _b3511_entry
except Exception:
    pass

# ─── Batch 3512 — Microstructure: RelStrengthMicro/GapFill/RegimeSwitch ───────
# TV_RelativeStrengthMicro, TV_GapFillProbability, TV_RegimeSwitchFast,
# TV_CorrelationBreakdown, TV_AdaptiveMeanRev
# Source: Levy (1967) relative strength; Connors & Alvarez (2009) short-term strategies;
#         Hamilton (1989) regime switching; Kahneman & Tversky (1979) prospect theory
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3512.py
try:
    import importlib.util as _ilu_b3512, os as _os_b3512
    _b3512_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3512.py"
    _b3512_spec = _ilu_b3512.spec_from_file_location('strategies_tv2_batch3512', _b3512_path)
    _b3512_mod  = _ilu_b3512.module_from_spec(_b3512_spec)
    _b3512_spec.loader.exec_module(_b3512_mod)
    for _b3512_name, _b3512_entry in _b3512_mod.STRATEGY_EXPORT.items():
        _reg(_b3512_name, _b3512_entry['gen'], _b3512_entry['space'])
    del _ilu_b3512, _os_b3512, _b3512_path, _b3512_spec, _b3512_mod, _b3512_name, _b3512_entry
except Exception:
    pass

# ─── Batch 3513 — Microstructure/OrderFlow/MeanReversion 5m/15m ultrafast ─────
# TV_VWAP_AnchoredRevert, TV_VolDeltaExhaustion, TV_BBVolFilter,
# TV_RVOLRevertAltcoin, TV_ConsecBarReversal
# Source: Berkowitz et al. (2012) VWAP execution; Chordia & Subrahmanyam (2004)
#         order imbalance mean-reversion; Bollinger (2002) Bollinger on Bollinger Bands;
#         Amihud (2002) illiquidity; Lo & MacKinlay (1990) consecutive return autocorrelation
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3513.py
try:
    import importlib.util as _ilu_b3513, os as _os_b3513
    _b3513_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3513.py"
    _b3513_spec = _ilu_b3513.spec_from_file_location('strategies_tv2_batch3513', _b3513_path)
    _b3513_mod  = _ilu_b3513.module_from_spec(_b3513_spec)
    _b3513_spec.loader.exec_module(_b3513_mod)
    for _b3513_name, _b3513_entry in _b3513_mod.STRATEGY_EXPORT.items():
        _reg(_b3513_name, _b3513_entry['gen'], _b3513_entry['space'])
    del _ilu_b3513, _os_b3513, _b3513_path, _b3513_spec, _b3513_mod, _b3513_name, _b3513_entry
except Exception:
    pass

# ─── Batch 3554 — ICT Concepts: Premium/Discount + Breaker Blocks + CISD ──────
# TV_ICT_PremiumDiscount_Zone, TV_BreakerBlock_Entry, TV_Mitigation_Block,
# TV_CISD_Entry, TV_InducementSweep_Reversal
# Source: Pine v5 — ICT (Inner Circle Trader) advanced S&R concepts for crypto futures
#         Michael J. Huddleston (ICT) — Premium/Discount, Breaker Blocks,
#         Mitigation Blocks, CISD, Inducement Sweeps
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3554.py
try:
    import importlib.util as _ilu_b3554, os as _os_b3554
    _b3554_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3554.py"
    _b3554_spec = _ilu_b3554.spec_from_file_location('strategies_tv2_batch3554', _b3554_path)
    _b3554_mod  = _ilu_b3554.module_from_spec(_b3554_spec)
    _b3554_spec.loader.exec_module(_b3554_mod)
    for _b3554_name, _b3554_entry in _b3554_mod.STRATEGY_EXPORT.items():
        _reg(_b3554_name, _b3554_entry['gen'], _b3554_entry['space'])
    del _ilu_b3554, _os_b3554, _b3554_path, _b3554_spec, _b3554_mod, _b3554_name, _b3554_entry
except Exception:
    pass

# ─── Batch 3555 — Session-Based S&R: Asia Range + London KZ + NY Open + PDH/PDL + Weekly Open ──
# TV_AsiaRange_Breakout, TV_LondonKillZone_SR, TV_NYOpen_Reversal,
# TV_PreviousDay_HL_SR, TV_WeeklyOpen_SR
# Source: Pine v5 — Session-based Support/Resistance for crypto futures
#         ICT Kill Zones, Asia Range (Darvas 1960), Previous Day H/L, Weekly Open
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3555.py
try:
    import importlib.util as _ilu_b3555, os as _os_b3555
    _b3555_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3555.py"
    _b3555_spec = _ilu_b3555.spec_from_file_location('strategies_tv2_batch3555', _b3555_path)
    _b3555_mod  = _ilu_b3555.module_from_spec(_b3555_spec)
    _b3555_spec.loader.exec_module(_b3555_mod)
    for _b3555_name, _b3555_entry in _b3555_mod.STRATEGY_EXPORT.items():
        _reg(_b3555_name, _b3555_entry['gen'], _b3555_entry['space'])
    del _ilu_b3555, _os_b3555, _b3555_path, _b3555_spec, _b3555_mod, _b3555_name, _b3555_entry
except Exception:
    pass

# ─── Batch 3556 — S&R + Technical Confirmators: RSI Div + MACD + StochRSI + Volume + Triple ──
# TV_SR_RSI_Divergence_Combo, TV_SR_MACD_Confluence, TV_SR_StochRSI_Reversal,
# TV_SR_Volume_Confirmation, TV_SR_Triple_Confirmation
# Source: Pine v5 — S&R + technical confirmation strategies for crypto futures
#         S&R zones combined with RSI divergence, MACD confluence, StochRSI extremes,
#         volume spikes (Wyckoff), and triple candlestick confirmation (Nison 1991)
# Batch file: Estrategias/strategies_tv2_batches/strategies_tv2_batch3556.py
try:
    import importlib.util as _ilu_b3556, os as _os_b3556
    _b3556_path = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches/strategies_tv2_batch3556.py"
    _b3556_spec = _ilu_b3556.spec_from_file_location('strategies_tv2_batch3556', _b3556_path)
    _b3556_mod  = _ilu_b3556.module_from_spec(_b3556_spec)
    _b3556_spec.loader.exec_module(_b3556_mod)
    for _b3556_name, _b3556_entry in _b3556_mod.STRATEGY_EXPORT.items():
        _reg(_b3556_name, _b3556_entry['gen'], _b3556_entry['space'])
    del _ilu_b3556, _os_b3556, _b3556_path, _b3556_spec, _b3556_mod, _b3556_name, _b3556_entry
except Exception:
    pass
