#!/usr/bin/env python3
"""TV2 BATCH 46 — 20 estrategias Pine v4/v5/v6 → Python
   VWAPPullback, AlphaTrend, BBRSIBOV, BalancePowerHA, Ichimoku,
   BBofVWAP, BBFib618, SmoothedHA, TripleRSI, ATRRSIv2,
   StochRSIMFIEMA, TripleST, ARRVWAPIntraday, ADXv2,
   8DayRun, 4xEMAVol, FibsMarket, BT-SAR, VADER, AndeanScalping
"""
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _bb(close: pd.Series, n: int = 20, mult: float = 2.0):
    mid = _sma(close, n)
    std = close.rolling(n).std(ddof=0)
    return mid + mult*std, mid, mid - mult*std

def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    hlc3 = (high + low + close) / 3
    cum_tpv = (hlc3 * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tpv / cum_vol.replace(0, np.nan)

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum()

def _stochrsi(close: pd.Series, rsi_len: int = 14, stoch_len: int = 14,
              k_smooth: int = 3, d_smooth: int = 3):
    rsi = _rsi(close, rsi_len)
    lo  = rsi.rolling(stoch_len).min()
    hi  = rsi.rolling(stoch_len).max()
    k   = 100 * (rsi - lo) / (hi - lo + 1e-10)
    k   = _sma(k, k_smooth)
    d   = _sma(k, d_smooth)
    return k, d

def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, n: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    mf = tp * volume
    pos = mf.where(tp > tp.shift(), 0)
    neg = mf.where(tp < tp.shift(), 0)
    pos_r = pos.rolling(n).sum()
    neg_r = neg.rolling(n).sum()
    return 100 - 100 / (1 + pos_r / neg_r.replace(0, np.nan))

def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                mult: float = 3.0, period: int = 7):
    atr_v = _atr(high, low, close, period)
    hl2   = (high + low) / 2
    upper = hl2 + mult * atr_v
    lower = hl2 - mult * atr_v

    trend = pd.Series(1, index=close.index)
    final_up   = lower.copy()
    final_down = upper.copy()

    for i in range(1, len(close)):
        fu_prev = final_up.iloc[i-1]
        fd_prev = final_down.iloc[i-1]
        fu = lower.iloc[i]
        fd = upper.iloc[i]

        final_up.iloc[i]   = max(fu, fu_prev) if close.iloc[i-1] > fu_prev else fu
        final_down.iloc[i] = min(fd, fd_prev) if close.iloc[i-1] < fd_prev else fd

        if close.iloc[i] > final_down.iloc[i-1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < final_up.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]

    return trend

def _parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series,
                   start: float = 0.02, increment: float = 0.02, maximum: float = 0.2):
    """Returns SAR series."""
    n = len(close)
    sar = close.copy()
    bull = True
    af = start
    ep = low.iloc[0]
    hp = high.iloc[0]
    lp = low.iloc[0]

    for i in range(2, n):
        if bull:
            sar.iloc[i] = sar.iloc[i-1] + af * (hp - sar.iloc[i-1])
            sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2])
            if low.iloc[i] < sar.iloc[i]:
                bull = False
                sar.iloc[i] = hp
                lp = low.iloc[i]
                af = start
                ep = lp
            else:
                if high.iloc[i] > hp:
                    hp = high.iloc[i]
                    af = min(af + increment, maximum)
                    ep = hp
        else:
            sar.iloc[i] = sar.iloc[i-1] + af * (lp - sar.iloc[i-1])
            sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2])
            if high.iloc[i] > sar.iloc[i]:
                bull = True
                sar.iloc[i] = lp
                hp = high.iloc[i]
                af = start
                ep = hp
            else:
                if low.iloc[i] < lp:
                    lp = low.iloc[i]
                    af = min(af + increment, maximum)
                    ep = lp
    return sar

def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         di_len: int = 14, adx_len: int = 14) -> pd.Series:
    up   = high.diff()
    down = -low.diff()
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr_v    = tr.ewm(alpha=1/di_len, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/di_len, adjust=False).mean() / atr_v.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/di_len, adjust=False).mean() / atr_v.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val  = dx.ewm(alpha=1/adx_len, adjust=False).mean()
    return adx_val, plus_di, minus_di

def _andean_osc(close: pd.Series, open_: pd.Series,
                length: int = 50, sig: int = 9):
    alpha = 2 / (length + 1)
    up1 = close.copy(); up2 = (close**2).copy()
    dn1 = close.copy(); dn2 = (close**2).copy()
    C = close.values; O = open_.values
    for i in range(1, len(C)):
        up1.iloc[i] = max(C[i], O[i], up1.iloc[i-1] - (up1.iloc[i-1] - C[i]) * alpha)
        up2.iloc[i] = max(C[i]**2, O[i]**2, up2.iloc[i-1] - (up2.iloc[i-1] - C[i]**2) * alpha)
        dn1.iloc[i] = min(C[i], O[i], dn1.iloc[i-1] + (C[i] - dn1.iloc[i-1]) * alpha)
        dn2.iloc[i] = min(C[i]**2, O[i]**2, dn2.iloc[i-1] + (C[i]**2 - dn2.iloc[i-1]) * alpha)
    bull = (dn2 - dn1**2).clip(lower=0).apply(np.sqrt)
    bear = (up2 - up1**2).clip(lower=0).apply(np.sqrt)
    signal = _ema(pd.concat([bull, bear], axis=1).max(axis=1), sig)
    return bull, bear, signal

# ─────────────────────────────────────────────
# 1. VWAP Pullback + Trend Template
# ─────────────────────────────────────────────

def gen_VWAPPullback(df: pd.DataFrame, rsi_os: int = 30, rsi_ob: int = 70,
                     rsi_len: int = 14, sma_len: int = 7, **kw) -> pd.Series:
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df.get('volume', pd.Series(1, index=df.index))

    vwap_v = _vwap(high, low, close, volume)
    ma     = _sma(close, sma_len)
    rsi_v  = _rsi(close, rsi_len)

    # VWAP crossover with MA
    cross_bull = (ma > vwap_v) & (ma.shift() <= vwap_v.shift())
    cross_bear = (ma < vwap_v) & (ma.shift() >= vwap_v.shift())

    # RSI filter
    was_os = rsi_v.rolling(10).min() <= rsi_os
    was_ob = rsi_v.rolling(10).max() >= rsi_ob

    buy  = was_os & cross_bull
    sell = was_ob & cross_bear

    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

def space_VWAPPullback(trial) -> dict:
    return {
        'rsi_os':  trial.suggest_int('rsi_os', 20, 40),
        'rsi_ob':  trial.suggest_int('rsi_ob', 60, 80),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'sma_len': trial.suggest_int('sma_len', 5, 20),
    }

# ─────────────────────────────────────────────
# 2. AlphaTrend + Trailing SL
# ─────────────────────────────────────────────

def gen_AlphaTrend(df: pd.DataFrame, coeff: float = 1.0, period: int = 14, **kw) -> pd.Series:
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df.get('volume', pd.Series(1, index=df.index))

    hlc3   = (high + low + close) / 3
    atr_v  = _sma(pd.concat([high - low,
                               (high - close.shift()).abs(),
                               (low  - close.shift()).abs()], axis=1).max(axis=1), period)

    up_t   = low  - atr_v * coeff
    down_t = high + atr_v * coeff

    # MFI >= 50 → bullish component
    mfi_v  = _mfi(high, low, close, volume, period)
    is_up  = mfi_v >= 50

    at = pd.Series(np.nan, index=close.index)
    at.iloc[0] = up_t.iloc[0] if is_up.iloc[0] else down_t.iloc[0]
    for i in range(1, len(close)):
        prev = at.iloc[i-1]
        if is_up.iloc[i]:
            at.iloc[i] = max(up_t.iloc[i], prev) if not np.isnan(prev) else up_t.iloc[i]
        else:
            at.iloc[i] = min(down_t.iloc[i], prev) if not np.isnan(prev) else down_t.iloc[i]

    # Signal: crossover AT vs AT[2]
    at2 = at.shift(2)
    buy  = (at > at2) & (at.shift() <= at2.shift())
    sell = (at < at2) & (at.shift() >= at2.shift())

    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

def space_AlphaTrend(trial) -> dict:
    return {
        'coeff':  trial.suggest_float('coeff', 0.5, 3.0, step=0.1),
        'period': trial.suggest_int('period', 7, 21),
    }

# ─────────────────────────────────────────────
# 3. BB + RSI + OBV
# ─────────────────────────────────────────────

def gen_BBRSIBOV(df: pd.DataFrame, bb_len: int = 20, bb_mult: float = 2.0,
                 rsi_len: int = 14, **kw) -> pd.Series:
    close  = df['close']
    volume = df.get('volume', pd.Series(1, index=df.index))

    upper, basis, lower = _bb(close, bb_len, bb_mult)
    rsi_v  = _rsi(close, rsi_len)
    obv_v  = _obv(close, volume)

    buy  = (close > basis) & (rsi_v > 50) & (obv_v > obv_v.shift())
    sell = (close < basis) & (rsi_v < 50) & (obv_v < obv_v.shift())

    # Exit on opposite band
    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

def space_BBRSIBOV(trial) -> dict:
    return {
        'bb_len':  trial.suggest_int('bb_len', 10, 40),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0, step=0.1),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
    }

# ─────────────────────────────────────────────
# 4. Balance of Power + Heikin Ashi
# ─────────────────────────────────────────────

def gen_BalancePowerHA(df: pd.DataFrame, bop_len: int = 252,
                       pct_hi: int = 99, pct_lo: int = 1, lookback: int = 75, **kw) -> pd.Series:
    close  = df['close']
    open_  = df['open']
    high   = df['high']
    low    = df['low']

    # Heikin Ashi close approx
    ha_close = (open_ + high + low + close) / 4
    ha_open  = (open_.shift() + close.shift()) / 2

    # Balance of Power = (close - open) / (high - low)
    hl_range = (high - low).replace(0, np.nan)
    bop      = (ha_close - ha_open) / hl_range
    bop_ema  = _ema(bop.ffill(), bop_len)

    # Percentile approximation using rolling quantile
    pnr_hi = bop_ema.rolling(lookback).quantile(pct_hi / 100)
    pnr_lo = bop_ema.rolling(lookback).quantile(pct_lo / 100)

    long_sig  = (pnr_hi > pnr_hi.shift()) & ((bop_ema > 0) | ((bop_ema > -0.35) & (bop_ema.shift() < -0.35)))
    short_sig = ((bop_ema < 0) & (pnr_hi < pnr_hi.shift()) & (pnr_lo < pnr_lo.shift())) | \
                ((bop_ema < 0.6) & (bop_ema.shift() > 0.6))

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  =  1
    sig[short_sig] = -1
    return sig

def space_BalancePowerHA(trial) -> dict:
    return {
        'bop_len':  trial.suggest_int('bop_len', 100, 400),
        'lookback': trial.suggest_int('lookback', 50, 150),
        'pct_hi':   trial.suggest_int('pct_hi', 90, 99),
        'pct_lo':   trial.suggest_int('pct_lo', 1, 10),
    }

# ─────────────────────────────────────────────
# 5. Ichimoku Cloud — All Signals
# ─────────────────────────────────────────────

def gen_Ichimoku(df: pd.DataFrame, conv_p: int = 9, base_p: int = 26,
                 lag_p: int = 52, disp: int = 26, **kw) -> pd.Series:
    close = df['close']
    high  = df['high']
    low   = df['low']

    def donchian(n):
        return (high.rolling(n).max() + low.rolling(n).min()) / 2

    tenkan  = donchian(conv_p)
    kijun   = donchian(base_p)
    span_a  = (tenkan + kijun) / 2
    span_b  = donchian(lag_p)

    # Current cloud (shifted back)
    kumo_hi = pd.concat([span_a.shift(disp - 1), span_b.shift(disp - 1)], axis=1).max(axis=1)
    kumo_lo = pd.concat([span_a.shift(disp - 1), span_b.shift(disp - 1)], axis=1).min(axis=1)

    # Signal: Tenkan/Kijun crossover above Kumo
    tk_cross_up   = (tenkan > kijun) & (tenkan.shift() <= kijun.shift())
    tk_cross_dn   = (tenkan < kijun) & (tenkan.shift() >= kijun.shift())
    above_kumo    = close > kumo_hi
    below_kumo    = close < kumo_lo

    buy  = tk_cross_up & above_kumo
    sell = tk_cross_dn & below_kumo

    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

def space_Ichimoku(trial) -> dict:
    return {
        'conv_p': trial.suggest_int('conv_p', 7, 15),
        'base_p': trial.suggest_int('base_p', 20, 35),
        'lag_p':  trial.suggest_int('lag_p', 40, 60),
        'disp':   trial.suggest_int('disp', 20, 35),
    }

# ─────────────────────────────────────────────
# 6. BB of VWAP + Pivot Points
# ─────────────────────────────────────────────

def gen_BBofVWAP(df: pd.DataFrame, bb_len: int = 50, bb_mult: float = 2.0,
                 sl_pct: float = 5.0, **kw) -> pd.Series:
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df.get('volume', pd.Series(1, index=df.index))

    vwap_v = _vwap(high, low, close, volume)
    basis  = _sma(vwap_v, bb_len)
    std    = vwap_v.rolling(bb_len).std(ddof=0)
    upper  = basis + bb_mult * std
    lower  = basis - bb_mult * std

    # Entry: VWAP crosses above BB midline
    buy   = (vwap_v > basis) & (vwap_v.shift() <= basis.shift())
    # Exit: VWAP crosses below lower band
    exit_ = (vwap_v < lower) & (vwap_v.shift() >= lower.shift())

    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[exit_] = -1
    return sig

def space_BBofVWAP(trial) -> dict:
    return {
        'bb_len':  trial.suggest_int('bb_len', 20, 100),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0, step=0.1),
        'sl_pct':  trial.suggest_float('sl_pct', 2.0, 8.0, step=0.5),
    }

# ─────────────────────────────────────────────
# 7. Bollinger Band + Fibonacci 0.618
# ─────────────────────────────────────────────

def gen_BBFib618(df: pd.DataFrame, bb_len: int = 50, bb_mult: float = 1.5,
                 ema_fast: int = 50, ema_slow: int = 200, **kw) -> pd.Series:
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df.get('volume', pd.Series(1, index=df.index))

    # VWMA basis
    hlc3    = (high + low + close) / 3
    basis   = (hlc3 * volume).rolling(bb_len).sum() / volume.rolling(bb_len).sum()
    std     = hlc3.rolling(bb_len).std(ddof=0)
    fib_up  = basis + 0.618 * bb_mult * std
    fib_lo  = basis - 0.618 * bb_mult * std

    ema50  = _ema(close, ema_fast)
    ema200 = _ema(close, ema_slow)

    long_cond = ema50 > ema200

    buy  = long_cond & ((close < fib_lo) | (low  <= fib_lo))
    sell = (close > fib_up) & (close.shift() <= fib_up.shift())

    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

def space_BBFib618(trial) -> dict:
    return {
        'bb_len':  trial.suggest_int('bb_len', 20, 80),
        'bb_mult': trial.suggest_float('bb_mult', 1.0, 3.0, step=0.1),
        'ema_fast': trial.suggest_int('ema_fast', 20, 80),
        'ema_slow': trial.suggest_int('ema_slow', 100, 300),
    }

# ─────────────────────────────────────────────
# 8. Smoothed Heikin Ashi (Abdoli Rev.4)
# ─────────────────────────────────────────────

def gen_SmoothedHA(df: pd.DataFrame, ma_period: int = 65, **kw) -> pd.Series:
    close  = df['close']
    open_  = df['open']
    high   = df['high']
    low    = df['low']

    ma_o = _ema(open_,  ma_period)
    ma_h = _ema(high,   ma_period)
    ma_l = _ema(low,    ma_period)
    ma_c = _ema(close,  ma_period)

    ha_c = (ma_o + ma_h + ma_l + ma_c) / 4
    ha_o = pd.Series(np.nan, index=df.index)
    ha_o.iloc[0] = (ma_o.iloc[0] + ma_c.iloc[0]) / 2
    for i in range(1, len(ha_c)):
        ha_o.iloc[i] = (ha_o.iloc[i-1] + ha_c.iloc[i-1]) / 2

    b0 = ha_c - ha_o
    b1 = b0.shift(1)
    b2 = b0.shift(2)

    buy  = (b0 > 0) & (b1 > 0) & (b2 > 0) & (ha_c > ha_c.shift()) & (ha_c.shift() > ha_c.shift(2))
    sell = (b0 < 0) & (b1 < 0) & (b2 < 0) & (ha_c < ha_c.shift()) & (ha_c.shift() < ha_c.shift(2))

    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

def space_SmoothedHA(trial) -> dict:
    return {
        'ma_period': trial.suggest_int('ma_period', 20, 100, step=5),
    }

# ─────────────────────────────────────────────
# 9. Triple RSI + EMA Ribbon
# ─────────────────────────────────────────────

def gen_TripleRSI(df: pd.DataFrame, rsi1_len: int = 50, rsi2_len: int = 75,
                  rsi3_len: int = 100, ma_fast: int = 5, ma_mid: int = 30,
                  ma_slow: int = 100, **kw) -> pd.Series:
    close  = df['close']
    src    = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    rsi1 = _rsi(src, rsi1_len)
    rsi2 = _rsi(src, rsi2_len)
    # rsi3 not directly used in entry but acts as slow filter

    # MA ribbon on RSI1
    ma05  = _sma(rsi1, ma_fast)
    ma30  = _sma(rsi1, ma_mid)
    ma100 = _sma(rsi1, ma_slow)

    # MA ribbon on RSI2
    ma051  = _sma(rsi2, ma_fast)
    ma301  = _sma(rsi2, ma_mid)
    ma1001 = _sma(rsi2, ma_slow)

    # Long: ma30 rising & above ma100, and ribbons aligned
    long0 = (ma30 > ma100) & (ma30.diff() >= 0) & (ma05 > ma100)
    long1 = (ma301 > ma1001) & (ma051 > ma1001)

    exit0 = (ma30 < ma100) & (ma30.diff() <= 0)
    exit1 = (ma301 < ma1001)

    buy  = long0 | long1
    sell = exit0 | exit1

    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

def space_TripleRSI(trial) -> dict:
    return {
        'rsi1_len': trial.suggest_int('rsi1_len', 30, 70),
        'rsi2_len': trial.suggest_int('rsi2_len', 60, 100),
        'rsi3_len': trial.suggest_int('rsi3_len', 80, 120),
        'ma_fast':  trial.suggest_int('ma_fast', 3, 10),
        'ma_mid':   trial.suggest_int('ma_mid', 20, 50),
        'ma_slow':  trial.suggest_int('ma_slow', 80, 120),
    }

# ─────────────────────────────────────────────
# 10. ATR + RSI v2 (no-repaint)
# ─────────────────────────────────────────────

def gen_ATRRSIv2(df: pd.DataFrame, atr_len: int = 26, atr_ma_len: int = 45,
                 rsi_len: int = 15, rsi_entry: int = 10,
                 norm_min: float = 0.3, norm_max: float = 0.7, **kw) -> pd.Series:
    close = df['close']
    high  = df['high']

    atr_v    = _atr(high, df['low'], close, atr_len)
    atr_ma   = _sma(atr_v, atr_ma_len)
    rsi_v    = _rsi(close, rsi_len)
    atr_norm = atr_ma / close * 100

    rsi_buy  = 50 + rsi_entry
    rsi_sell = 50 - rsi_entry

    # Normalized ATR in band + ATR above its MA
    vol_ok = (atr_norm >= norm_min) & (atr_norm <= norm_max)
    high_vol = atr_v > atr_ma

    # SMA norm of high
    sma45 = _sma(high, 45)
    lo45  = sma45.rolling(45).min()
    hi45  = sma45.rolling(45).max()
    sma_norm = (sma45 - lo45) / (hi45 - lo45 + 1e-10)

    buy  = vol_ok & high_vol & (rsi_v > rsi_buy)
    sell = vol_ok & high_vol & (rsi_v < rsi_sell)

    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

def space_ATRRSIv2(trial) -> dict:
    return {
        'atr_len':    trial.suggest_int('atr_len', 10, 40),
        'atr_ma_len': trial.suggest_int('atr_ma_len', 20, 80),
        'rsi_len':    trial.suggest_int('rsi_len', 7, 25),
        'rsi_entry':  trial.suggest_int('rsi_entry', 5, 20),
        'norm_min':   trial.suggest_float('norm_min', 0.1, 0.5, step=0.1),
        'norm_max':   trial.suggest_float('norm_max', 0.5, 1.0, step=0.1),
    }

# ─────────────────────────────────────────────
# 11. StochRSI + MFI + EMA
# ─────────────────────────────────────────────

def gen_StochRSIMFIEMA(df: pd.DataFrame, rsi_len: int = 100, stoch_len: int = 100,
                        k_smooth: int = 1, d_smooth: int = 1, mfi_len: int = 30,
                        ema_len: int = 100, srsi_lo: float = 20, srsi_hi: float = 80,
                        mfi_lo: float = 20, mfi_hi: float = 80, **kw) -> pd.Series:
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df.get('volume', pd.Series(1, index=df.index))

    k, d   = _stochrsi(close, rsi_len, stoch_len, k_smooth, d_smooth)
    mfi_v  = _mfi(high, low, close, volume, mfi_len)
    ema_v  = _ema(close, ema_len)

    long_cond  = (d < srsi_lo) & (mfi_v < mfi_lo) & (close < ema_v)
    short_cond = (d > srsi_hi) & (mfi_v > mfi_hi) & (close > ema_v)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig

def space_StochRSIMFIEMA(trial) -> dict:
    return {
        'rsi_len':   trial.suggest_int('rsi_len', 50, 150),
        'stoch_len': trial.suggest_int('stoch_len', 50, 150),
        'mfi_len':   trial.suggest_int('mfi_len', 10, 50),
        'ema_len':   trial.suggest_int('ema_len', 50, 200),
        'srsi_lo':   trial.suggest_float('srsi_lo', 10, 30),
        'srsi_hi':   trial.suggest_float('srsi_hi', 70, 90),
        'mfi_lo':    trial.suggest_float('mfi_lo', 10, 30),
        'mfi_hi':    trial.suggest_float('mfi_hi', 70, 90),
    }

# ─────────────────────────────────────────────
# 12. Triple Supertrend + StochRSI
# ─────────────────────────────────────────────

def gen_TripleSupertrend(df: pd.DataFrame, st1_len: int = 10, st1_mult: float = 1.0,
                          st2_len: int = 11, st2_mult: float = 2.0,
                          st3_len: int = 12, st3_mult: float = 3.0,
                          rsi_len: int = 14, stoch_len: int = 14,
                          k_sm: int = 3, d_sm: int = 3, ema_len: int = 200, **kw) -> pd.Series:
    close = df['close']
    high  = df['high']
    low   = df['low']

    dir1 = _supertrend(high, low, close, st1_mult, st1_len)
    dir2 = _supertrend(high, low, close, st2_mult, st2_len)
    dir3 = _supertrend(high, low, close, st3_mult, st3_len)

    k, d   = _stochrsi(close, rsi_len, stoch_len, k_sm, d_sm)
    ema_v  = _ema(close, ema_len)

    # 2 of 3 STs bullish
    st_bull = ((dir1 == 1) & (dir2 == 1)) | ((dir2 == 1) & (dir3 == 1))
    st_bear = ((dir1 == -1) & (dir2 == -1)) | ((dir2 == -1) & (dir3 == -1))

    k_cross_up = (k > d) & (k.shift() <= d.shift())
    k_cross_dn = (k < d) & (k.shift() >= d.shift())

    buy  = (close > ema_v) & st_bull & k_cross_up
    sell = (close < ema_v) & st_bear & k_cross_dn

    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

def space_TripleSupertrend(trial) -> dict:
    return {
        'st1_len':  trial.suggest_int('st1_len', 7, 15),
        'st1_mult': trial.suggest_float('st1_mult', 0.5, 2.0, step=0.5),
        'st2_len':  trial.suggest_int('st2_len', 8, 16),
        'st2_mult': trial.suggest_float('st2_mult', 1.0, 3.0, step=0.5),
        'st3_len':  trial.suggest_int('st3_len', 9, 17),
        'st3_mult': trial.suggest_float('st3_mult', 2.0, 4.0, step=0.5),
        'rsi_len':  trial.suggest_int('rsi_len', 7, 21),
        'ema_len':  trial.suggest_int('ema_len', 100, 300),
    }

# ─────────────────────────────────────────────
# 13. ARR VWAP Intraday
# ─────────────────────────────────────────────

def gen_ARRVWAPIntraday(df: pd.DataFrame, rsi_period: int = 14,
                         rsi_buy: int = 50, rsi_sell: int = 50, **kw) -> pd.Series:
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df.get('volume', pd.Series(1, index=df.index))

    vwap_v = _vwap(high, low, close, volume)
    rsi_v  = _rsi(close, rsi_period)

    buy  = (vwap_v > vwap_v.shift()) & (rsi_v > rsi_buy)
    sell = (vwap_v < vwap_v.shift()) & (rsi_v < rsi_sell)

    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

def space_ARRVWAPIntraday(trial) -> dict:
    return {
        'rsi_period': trial.suggest_int('rsi_period', 7, 21),
        'rsi_buy':    trial.suggest_int('rsi_buy', 40, 60),
        'rsi_sell':   trial.suggest_int('rsi_sell', 40, 60),
    }

# ─────────────────────────────────────────────
# 14. Average Directional Index v2
# ─────────────────────────────────────────────

def gen_ADXv2(df: pd.DataFrame, di_len: int = 5, adx_len: int = 1,
              adx_min: int = 25, adx_max: int = 75, **kw) -> pd.Series:
    close = df['close']
    high  = df['high']
    low   = df['low']

    adx_v, plus_di, minus_di = _adx(high, low, close, di_len, adx_len)

    # Strong trend: ADX in [adx_min, adx_max]
    strong = (adx_v >= adx_min) & (adx_v <= adx_max)

    buy  = strong & (plus_di > minus_di) & (plus_di.shift() <= minus_di.shift())
    sell = strong & (minus_di > plus_di) & (minus_di.shift() <= plus_di.shift())

    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

def space_ADXv2(trial) -> dict:
    return {
        'di_len':  trial.suggest_int('di_len', 3, 14),
        'adx_len': trial.suggest_int('adx_len', 1, 5),
        'adx_min': trial.suggest_int('adx_min', 15, 35),
        'adx_max': trial.suggest_int('adx_max', 60, 90),
    }

# ─────────────────────────────────────────────
# 15. 8-Day Momentum Run
# ─────────────────────────────────────────────

def gen_EightDayRun(df: pd.DataFrame, sma_len: int = 5, run_bars: int = 8, **kw) -> pd.Series:
    close = df['close']

    sma5 = _sma(close, sma_len)

    # Bars since close < sma (for long trigger)
    bars_above = pd.Series(0, index=df.index)
    bars_below = pd.Series(0, index=df.index)

    cnt_a = 0
    cnt_b = 0
    for i in range(len(close)):
        if pd.isna(sma5.iloc[i]):
            continue
        if close.iloc[i] >= sma5.iloc[i]:
            cnt_a += 1
            cnt_b = 0
        else:
            cnt_b += 1
            cnt_a = 0
        bars_above.iloc[i] = cnt_a
        bars_below.iloc[i] = cnt_b

    # After 8+ bars ABOVE sma → prime sell; first bar back BELOW sma = short
    trigger_sell = (bars_above.shift() >= run_bars)
    sell_entry   = trigger_sell & (close <= sma5)

    # After 8+ bars BELOW sma → prime buy; first bar back ABOVE sma = long
    trigger_buy  = (bars_below.shift() >= run_bars)
    buy_entry    = trigger_buy & (close >= sma5)

    # Exits: cross opposite
    buy_exit  = close > sma5
    sell_exit = close < sma5

    sig = pd.Series(0, index=df.index)
    sig[buy_entry]  =  1
    sig[sell_entry] = -1
    return sig

def space_EightDayRun(trial) -> dict:
    return {
        'sma_len':  trial.suggest_int('sma_len', 3, 10),
        'run_bars': trial.suggest_int('run_bars', 5, 15),
    }

# ─────────────────────────────────────────────
# 16. 4x EMA + Volume (EOM)
# ─────────────────────────────────────────────

def gen_FourEMAVol(df: pd.DataFrame, e1: int = 13, e2: int = 21,
                   e3: int = 50, e4: int = 180, eom_len: int = 14, **kw) -> pd.Series:
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df.get('volume', pd.Series(1, index=df.index))

    ema1 = _ema(close, e1)
    ema2 = _ema(close, e2)
    ema3 = _ema(close, e3)
    ema4 = _ema(close, e4)

    # Ease of Movement
    hl2    = (high + low) / 2
    eom    = _sma(10000 * hl2.diff() * (high - low) / volume.replace(0, np.nan), eom_len)

    # Option 1: all EMAs stacked + EOM filter
    long1  = (close > ema1) & (ema1 > ema2) & (ema2 > ema3) & (ema3 > ema4) & (eom > 0)
    short1 = (close < ema1) & (ema1 < ema2) & (ema2 < ema3) & (ema3 < ema4) & (eom < 0)

    sig = pd.Series(0, index=df.index)
    sig[long1]  =  1
    sig[short1] = -1
    return sig

def space_FourEMAVol(trial) -> dict:
    return {
        'e1':      trial.suggest_int('e1', 8, 20),
        'e2':      trial.suggest_int('e2', 15, 30),
        'e3':      trial.suggest_int('e3', 35, 65),
        'e4':      trial.suggest_int('e4', 120, 250),
        'eom_len': trial.suggest_int('eom_len', 7, 25),
    }

# ─────────────────────────────────────────────
# 17. Fibonacci Market Orders
# ─────────────────────────────────────────────

def gen_FibsMarket(df: pd.DataFrame, pivot_len: int = 60, tp_pct: float = 1.0,
                   fib_idx: int = 5, **kw) -> pd.Series:
    """Simplified: uses fib levels from pivot high/low to find entries."""
    close = df['close']
    high  = df['high']
    low   = df['low']

    hp = high.rolling(pivot_len).max()
    lp = low.rolling(pivot_len).min()

    fib_levels = [
        lp + (hp - lp) * (-1.0),   # fib10 (index 1)
        lp + (hp - lp) * (-0.27),  # fib9  (index 2)
        lp + (hp - lp) * 0.0,      # fib0  (index 3)
        lp + (hp - lp) * 0.21,     # fib1  (index 4)
        lp + (hp - lp) * 0.3,      # fib2  (index 5)
        lp + (hp - lp) * 0.5,      # fib3  (index 6)
        lp + (hp - lp) * 0.62,     # fib4  (index 7)
        lp + (hp - lp) * 0.7,      # fib5  (index 8)
        lp + (hp - lp) * 1.0,      # fib6  (index 9)
        lp + (hp - lp) * 1.27,     # fib7  (index 10)
    ]

    idx = max(0, min(fib_idx - 1, len(fib_levels) - 1))
    entry_level = fib_levels[idx]
    profit      = entry_level + entry_level * (tp_pct / 100)

    # Entry: price crosses below entry level (accumulation)
    filled = (low.shift() > entry_level.shift()) & (low <= entry_level)
    close_it = high > profit

    sig = pd.Series(0, index=df.index)
    sig[filled]   =  1
    sig[close_it] = -1
    return sig

def space_FibsMarket(trial) -> dict:
    return {
        'pivot_len': trial.suggest_int('pivot_len', 30, 120),
        'tp_pct':    trial.suggest_float('tp_pct', 0.5, 3.0, step=0.25),
        'fib_idx':   trial.suggest_int('fib_idx', 3, 8),
    }

# ─────────────────────────────────────────────
# 18. BT-SAR + EMA + Squeeze + Volatility
# ─────────────────────────────────────────────

def gen_BTSAR(df: pd.DataFrame, sar_start: float = 0.02, sar_inc: float = 0.02,
              sar_max: float = 0.2, ema_len: int = 100, sqz_len: int = 20,
              kelt_mult: float = 1.5, vlt_len: int = 100, **kw) -> pd.Series:
    close = df['close']
    high  = df['high']
    low   = df['low']
    open_ = df['open']

    sar_v  = _parabolic_sar(high, low, close, sar_start, sar_inc, sar_max)
    ema_v  = _ema(close, ema_len)

    # Volatility oscillator
    spike    = close - open_
    vlt_std  = spike.rolling(vlt_len).std()
    vlt_bull = spike > vlt_std

    # Squeeze: BB inside Keltner
    bb_upper, bb_mid, bb_lower = _bb(close, sqz_len, 2.0)
    kelt_range = _atr(high, low, close, sqz_len) * kelt_mult
    kelt_upper = bb_mid + kelt_range
    kelt_lower = bb_mid - kelt_range
    sqz_off    = (bb_lower < kelt_lower) & (bb_upper > kelt_upper)  # squeeze released

    # Squeeze momentum
    val = pd.Series(np.nan, index=df.index)
    for i in range(sqz_len, len(close)):
        hi = high.iloc[i-sqz_len:i+1].max()
        lo = low.iloc[i-sqz_len:i+1].min()
        avg_hl = (hi + lo) / 2
        avg_c  = close.iloc[i-sqz_len:i+1].mean()
        val.iloc[i] = close.iloc[i] - (avg_hl + avg_c) / 2
    sqz_bull = val > val.shift()

    # SAR flip: previous bar below SAR, current bar above
    sar_flip = (close.shift() < sar_v.shift()) & (close > sar_v)

    buy = sar_flip & (close > ema_v) & vlt_bull & sqz_bull
    # Exit: SAR flips back (price crosses below SAR)
    sell = (close < sar_v) & (close.shift() >= sar_v.shift())

    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

def space_BTSAR(trial) -> dict:
    return {
        'sar_start': trial.suggest_float('sar_start', 0.01, 0.05, step=0.005),
        'sar_inc':   trial.suggest_float('sar_inc', 0.01, 0.05, step=0.005),
        'sar_max':   trial.suggest_float('sar_max', 0.1, 0.4, step=0.05),
        'ema_len':   trial.suggest_int('ema_len', 50, 200),
        'sqz_len':   trial.suggest_int('sqz_len', 10, 40),
        'vlt_len':   trial.suggest_int('vlt_len', 50, 200),
    }

# ─────────────────────────────────────────────
# 19. VADER — Dual Energy Ratio
# ─────────────────────────────────────────────

def gen_VADER(df: pd.DataFrame, der_len: int = 10, der_avg: int = 5,
              smooth: int = 3, senti_len: int = 20, vol_lookbk: int = 20,
              atr_per: int = 5, hhv_per: int = 10, mult: float = 2.5, **kw) -> pd.Series:
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df.get('volume', pd.Series(1, index=df.index))

    # Relative volume normalisation
    vol_lo = volume.rolling(vol_lookbk).min()
    vol_hi = volume.rolling(vol_lookbk).max()
    rel_vol = (volume - vol_lo) / (vol_hi - vol_lo + 1e-10)

    R = (high.rolling(2).max() - low.rolling(2).min()) / 2
    sr = close.diff() / R.replace(0, np.nan)
    rsr = sr.clip(-1, 1).fillna(0)
    c = (rsr * rel_vol).fillna(0)

    c_plus  = c.clip(lower=0)
    c_minus = (-c).clip(lower=0)

    avg_vol = _wma(rel_vol, der_len)
    dem     = _wma(c_plus,  der_len) / avg_vol.replace(0, np.nan)
    sup     = _wma(c_minus, der_len) / avg_vol.replace(0, np.nan)

    adp   = 100 * _wma(dem.fillna(0), der_avg)
    asp   = 100 * _wma(sup.fillna(0), der_avg)
    anp   = adp - asp
    anp_s = _wma(anp, smooth)

    # Sentiment
    s_adp    = 100 * _wma(dem.fillna(0), senti_len)
    s_asp    = 100 * _wma(sup.fillna(0), senti_len)
    V_senti  = _wma(s_adp - s_asp, smooth)

    # ATR trailing for price trend
    atr_v    = _atr(high, low, close, atr_per)
    hhv_sl   = (high - mult * atr_v).rolling(hhv_per).max()
    ema100   = _ema(close.shift(), 100)
    price_up = ema100 <= close.shift()

    # Entry conditions from Pine (atBottom / atTop)
    bc = adp; bo = asp
    rising   = adp.diff() > 0
    sflag_up = V_senti.abs() >= V_senti.abs().shift()
    s_up     = V_senti >= 0
    up       = anp_s >= 0

    grn_V    = s_up & sflag_up
    grn_bc   = (bc > bo) & rising
    grn_anp  = up

    red_V    = ~s_up & sflag_up
    red_bc   = (bc < bo) & ~rising
    red_anp  = ~up

    at_bottom = red_V & red_bc & red_anp
    at_top    = grn_V & grn_bc & grn_anp

    buy  = at_bottom & ~price_up
    sell = at_top & price_up

    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

def space_VADER(trial) -> dict:
    return {
        'der_len':   trial.suggest_int('der_len', 5, 20),
        'der_avg':   trial.suggest_int('der_avg', 3, 10),
        'smooth':    trial.suggest_int('smooth', 1, 7),
        'senti_len': trial.suggest_int('senti_len', 10, 40),
        'vol_lookbk': trial.suggest_int('vol_lookbk', 10, 40),
        'atr_per':   trial.suggest_int('atr_per', 3, 14),
        'hhv_per':   trial.suggest_int('hhv_per', 5, 20),
        'mult':      trial.suggest_float('mult', 1.5, 4.0, step=0.5),
    }

# ─────────────────────────────────────────────
# 20. Andean Scalping (Andean Oscillator)
# ─────────────────────────────────────────────

def gen_AndeanScalping(df: pd.DataFrame, and_len: int = 50, sig_len: int = 9,
                        threshold_len: int = 100, threshold_mult: float = 1.1,
                        atr_len: int = 14, atr_mult: float = 3.0, **kw) -> pd.Series:
    close  = df['close']
    open_  = df['open']
    high   = df['high']
    low    = df['low']

    bull, bear, signal = _andean_osc(close, open_, and_len, sig_len)

    # Threshold: SMA of signal * multiplier
    avg_thr = _sma(signal, threshold_len) * threshold_mult

    # Long: bull dominant + above signal + signal above threshold
    long_cond  = (bull > bear) & (bull > signal) & (signal > avg_thr)
    short_cond = (bear > bull) & (bear > signal) & (signal > avg_thr)
    end_long   = bull < signal
    end_short  = bear < signal

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    # Close on end condition (don't override entry signal)
    sig[(end_long) & (sig == 0)]  = -1
    sig[(end_short) & (sig == 0)] =  1
    # Simplify: just entry signals
    sig2 = pd.Series(0, index=df.index)
    sig2[long_cond]  =  1
    sig2[short_cond] = -1
    return sig2

def space_AndeanScalping(trial) -> dict:
    return {
        'and_len':        trial.suggest_int('and_len', 20, 100),
        'sig_len':        trial.suggest_int('sig_len', 3, 20),
        'threshold_len':  trial.suggest_int('threshold_len', 50, 200),
        'threshold_mult': trial.suggest_float('threshold_mult', 0.8, 1.5, step=0.1),
        'atr_len':        trial.suggest_int('atr_len', 7, 21),
        'atr_mult':       trial.suggest_float('atr_mult', 1.5, 5.0, step=0.5),
    }


# ─────────────────────────────────────────────
# STRATEGY EXPORT
# ─────────────────────────────────────────────

STRATEGY_EXPORT = {
    'VWAPPullback':        {'gen': gen_VWAPPullback,        'space': space_VWAPPullback},
    'AlphaTrend':          {'gen': gen_AlphaTrend,          'space': space_AlphaTrend},
    'BBRSIBOV':            {'gen': gen_BBRSIBOV,            'space': space_BBRSIBOV},
    'BalancePowerHA':      {'gen': gen_BalancePowerHA,      'space': space_BalancePowerHA},
    'Ichimoku':            {'gen': gen_Ichimoku,            'space': space_Ichimoku},
    'BBofVWAP':            {'gen': gen_BBofVWAP,            'space': space_BBofVWAP},
    'BBFib618':            {'gen': gen_BBFib618,            'space': space_BBFib618},
    'SmoothedHA':          {'gen': gen_SmoothedHA,          'space': space_SmoothedHA},
    'TripleRSI':           {'gen': gen_TripleRSI,           'space': space_TripleRSI},
    'ATRRSIv2':            {'gen': gen_ATRRSIv2,            'space': space_ATRRSIv2},
    'StochRSIMFIEMA':      {'gen': gen_StochRSIMFIEMA,      'space': space_StochRSIMFIEMA},
    'TripleSupertrend':    {'gen': gen_TripleSupertrend,    'space': space_TripleSupertrend},
    'ARRVWAPIntraday':     {'gen': gen_ARRVWAPIntraday,     'space': space_ARRVWAPIntraday},
    'ADXv2':               {'gen': gen_ADXv2,               'space': space_ADXv2},
    'EightDayRun':         {'gen': gen_EightDayRun,         'space': space_EightDayRun},
    'FourEMAVol':          {'gen': gen_FourEMAVol,          'space': space_FourEMAVol},
    'FibsMarket':          {'gen': gen_FibsMarket,          'space': space_FibsMarket},
    'BTSAR':               {'gen': gen_BTSAR,               'space': space_BTSAR},
    'VADER':               {'gen': gen_VADER,               'space': space_VADER},
    'AndeanScalping':      {'gen': gen_AndeanScalping,      'space': space_AndeanScalping},
}
