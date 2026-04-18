#!/usr/bin/env python3
"""
TV2 BATCH 10c — 12 estrategias Fibonacci/Divergencia/Linear Reg/Otros 2026-03-31

  Fibonacci_RSI_VWMA       — VWMA+StdDev Fib bands + RSI (v4, 3285L)
  Price_Divergence_Multi   — Multi-oscillator divergence (v4, 3849L)
  CipherB_WaveTrend        — WaveTrend + MFI + RSI + EMA PB (v4, 2451L)
  ATR_Trail_Dual           — Dual ATR trailing stop (v4, 2284L)
  Crypto_Scalper_SAR       — EMA + MACD + Parabolic SAR (v4, 3254L)
  Bitcoin_Momentum_EMA     — Weekly EMA + ATR volatility (v5, 1346L)
  Gaussian_Channel         — Multi-pole Gaussian filter (v6, 686L)
  Linear_Regression_Bo     — Linear regression channel breakout (v5, 685L)
  Swing_Hull_T3            — Hull MA + T3 average (v4, 1074L)
  HighLow_Channel_Bo       — Highest high / lowest low breakout (v5, 1617L)
  Linear_Regression_Pearson — LR channel + Pearson's R filter (v4, 867L)
  Donchian_HH_LL           — Donchian-style highest high / lowest low (v5, 1617L)
"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def _rma(series, period):
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _atr(df, period):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, period)


def _rsi(close, period):
    d     = close.diff()
    gain  = d.clip(lower=0).rolling(period).mean()
    loss  = (-d.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _wma(series, period):
    w = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def _hull_ma(series, period):
    half   = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    return _wma(2 * _wma(series, half) - _wma(series, period), sqrt_p)


def _t3(series, period, vfactor=0.7):
    """Tillson T3."""
    c1 = -(vfactor ** 3)
    c2 =  3 * vfactor ** 2 + 3 * vfactor ** 3
    c3 = -6 * vfactor ** 2 - 3 * vfactor - 3 * vfactor ** 3
    c4 = 1 + 3 * vfactor + vfactor ** 3 + 3 * vfactor ** 2
    e1 = _ema(series, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    e4 = _ema(e3, period)
    e5 = _ema(e4, period)
    e6 = _ema(e5, period)
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def _vwma(close, volume, period):
    return (close * volume).rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def _macd(close, fast=12, slow=26, sig=9):
    e_fast = _ema(close, fast)
    e_slow = _ema(close, slow)
    line   = e_fast - e_slow
    signal = _ema(line, sig)
    return line, signal, line - signal


def _mfi(df, period):
    tp   = (df['high'] + df['low'] + df['close']) / 3
    mf   = tp * df['volume']
    pos  = (tp > tp.shift(1)) * mf
    neg  = (tp < tp.shift(1)) * mf
    mfr  = pos.rolling(period).sum() / neg.rolling(period).sum().replace(0, np.nan)
    return 100 - 100 / (1 + mfr)


# ── 1. Fibonacci RSI VWMA ────────────────────────────────────────────────────

def gen_Fibonacci_RSI_VWMA(df, vwma_period=20, std_mult=2.0, rsi_period=14, **kw):
    """VWMA + StdDev = Fib bands. Long: below lower band AND RSI crosses above 30."""
    vp    = int(vwma_period)
    vmid  = _vwma(df['close'], df['volume'], vp)
    vstd  = df['close'].rolling(vp).std()
    upper = vmid + std_mult * vstd
    lower = vmid - std_mult * vstd
    rsi   = _rsi(df['close'], int(rsi_period))

    long_cond  = (df['close'] < lower) & (rsi.shift(1) < 30) & (rsi > 30)
    short_cond = (df['close'] > upper) & (rsi.shift(1) > 70) & (rsi < 70)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Fibonacci_RSI_VWMA(trial):
    return {
        'vwma_period': trial.suggest_int('vwma_period', 10, 40),
        'std_mult':    trial.suggest_float('std_mult', 1.0, 3.5),
        'rsi_period':  trial.suggest_int('rsi_period', 10, 21),
    }


# ── 2. Price Divergence Multi-Oscillator ─────────────────────────────────────

def gen_Price_Divergence_Multi(df, rsi_period=14, cci_period=20, pivot_len=5, **kw):
    """Simplified: pivot-high RSI divergence (bearish) + pivot-low RSI divergence (bullish)."""
    rsi  = _rsi(df['close'], int(rsi_period))
    pp   = int(pivot_len)
    cl   = df['close']

    # Pivot lows: bar is lowest in [i-pp .. i+pp]
    lo_min = cl.rolling(2 * pp + 1, center=True).min()
    hi_max = cl.rolling(2 * pp + 1, center=True).max()
    pivot_lo = (cl == lo_min) & (cl.shift(pp + 1) != lo_min)
    pivot_hi = (cl == hi_max) & (cl.shift(pp + 1) != hi_max)

    # Bullish divergence: lower price low, higher RSI low at pivot
    bull_div = pivot_lo & (cl < cl.shift(pp * 2)) & (rsi > rsi.shift(pp * 2))
    # Bearish divergence: higher price high, lower RSI high at pivot
    bear_div = pivot_hi & (cl > cl.shift(pp * 2)) & (rsi < rsi.shift(pp * 2))

    sig = pd.Series(0, index=df.index)
    sig[bull_div] =  1
    sig[bear_div] = -1
    return sig


def space_Price_Divergence_Multi(trial):
    return {
        'rsi_period': trial.suggest_int('rsi_period', 10, 21),
        'cci_period': trial.suggest_int('cci_period', 14, 30),
        'pivot_len':  trial.suggest_int('pivot_len', 3, 10),
    }


# ── 3. CipherB WaveTrend + MFI + RSI + EMA Pullback ─────────────────────────

def _wavetrend(df, n1=10, n2=21):
    ap   = (df['high'] + df['low'] + df['close']) / 3
    esa  = _ema(ap, n1)
    d    = _ema((ap - esa).abs(), n1)
    ci   = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1  = _ema(ci, n2)
    wt2  = wt1.rolling(4).mean()
    return wt1, wt2


def gen_CipherB_WaveTrend(df, wt_n1=10, wt_n2=21, rsi_period=14, mfi_period=60,
                           ema_period=200, **kw):
    wt1, wt2 = _wavetrend(df, int(wt_n1), int(wt_n2))
    rsi  = _rsi(df['close'], int(rsi_period))
    mfi  = _mfi(df, int(mfi_period))
    ema  = _ema(df['close'], int(ema_period))

    # Long: WT cross up from oversold + RSI not overbought + MFI bullish + above EMA
    wt_cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1)) & (wt1 < -60)
    wt_cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1)) & (wt1 > 60)

    long_cond  = wt_cross_up   & (rsi < 60) & (mfi > 0)   & (df['close'] > ema)
    short_cond = wt_cross_down & (rsi > 40) & (mfi < 100)  & (df['close'] < ema)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_CipherB_WaveTrend(trial):
    return {
        'wt_n1':      trial.suggest_int('wt_n1', 5, 15),
        'wt_n2':      trial.suggest_int('wt_n2', 15, 30),
        'rsi_period': trial.suggest_int('rsi_period', 10, 21),
        'mfi_period': trial.suggest_int('mfi_period', 30, 90),
        'ema_period': trial.suggest_int('ema_period', 100, 300),
    }


# ── 4. Dual ATR Trailing Stop [ceyhun] ───────────────────────────────────────

def _atr_trail(close, atr, mult):
    """Vectorized ATR trailing stop."""
    stop = pd.Series(np.nan, index=close.index)
    c    = close.values
    a    = atr.values
    s    = np.zeros(len(c))

    for i in range(1, len(c)):
        prev = s[i-1] if not np.isnan(s[i-1]) else c[i] - mult * a[i]
        raw  = c[i] - mult * a[i]
        if c[i-1] > prev:
            s[i] = max(raw, prev)
        else:
            s[i] = min(raw, prev)
    return pd.Series(s, index=close.index)


def gen_ATR_Trail_Dual(df, fast_period=5, fast_mult=0.5, slow_period=10, slow_mult=3.0, **kw):
    """Two ATR trailing lines — crossover = signal."""
    atr_fast = _atr(df, int(fast_period))
    atr_slow = _atr(df, int(slow_period))
    fast_trail = _atr_trail(df['close'], atr_fast, fast_mult)
    slow_trail = _atr_trail(df['close'], atr_slow, slow_mult)

    long_cond  = (fast_trail > slow_trail) & (fast_trail.shift(1) <= slow_trail.shift(1))
    short_cond = (fast_trail < slow_trail) & (fast_trail.shift(1) >= slow_trail.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_ATR_Trail_Dual(trial):
    return {
        'fast_period': trial.suggest_int('fast_period', 3, 10),
        'fast_mult':   trial.suggest_float('fast_mult', 0.2, 1.5),
        'slow_period': trial.suggest_int('slow_period', 8, 20),
        'slow_mult':   trial.suggest_float('slow_mult', 1.5, 5.0),
    }


# ── 5. Crypto Scalper EMA + MACD + SAR ───────────────────────────────────────

def _psar(df, start=0.02, increment=0.02, maximum=0.2):
    """Simplified Parabolic SAR."""
    hi = df['high'].values
    lo = df['low'].values
    n  = len(hi)
    sar   = np.zeros(n)
    bull  = np.ones(n, dtype=bool)
    af    = np.full(n, start)
    ep    = np.zeros(n)

    sar[0]  = lo[0]
    ep[0]   = hi[0]

    for i in range(1, n):
        if bull[i-1]:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = min(sar[i], lo[i-1], lo[i-2] if i >= 2 else lo[i-1])
            if hi[i] > ep[i-1]:
                ep[i] = hi[i]
                af[i] = min(af[i-1] + increment, maximum)
            else:
                ep[i] = ep[i-1]
                af[i] = af[i-1]
            if lo[i] < sar[i]:
                bull[i] = False
                sar[i]  = ep[i-1]
                ep[i]   = lo[i]
                af[i]   = start
            else:
                bull[i] = True
        else:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = max(sar[i], hi[i-1], hi[i-2] if i >= 2 else hi[i-1])
            if lo[i] < ep[i-1]:
                ep[i] = lo[i]
                af[i] = min(af[i-1] + increment, maximum)
            else:
                ep[i] = ep[i-1]
                af[i] = af[i-1]
            if hi[i] > sar[i]:
                bull[i] = True
                sar[i]  = ep[i-1]
                ep[i]   = hi[i]
                af[i]   = start
            else:
                bull[i] = False

    return pd.Series(bull.astype(float), index=df.index)  # 1=bull, 0=bear


def gen_Crypto_Scalper_SAR(df, ema_period=60, macd_fast=12, macd_slow=26, macd_sig=9,
                            sar_start=0.02, sar_inc=0.02, sar_max=0.2, **kw):
    ema  = _ema(df['close'], int(ema_period))
    _, sig_line, hist = _macd(df['close'], int(macd_fast), int(macd_slow), int(macd_sig))
    sar_bull = _psar(df, sar_start, sar_inc, sar_max)

    long_cond  = (df['close'] > ema) & (hist > 0) & (sar_bull == 1)
    short_cond = (df['close'] < ema) & (hist < 0) & (sar_bull == 0)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Crypto_Scalper_SAR(trial):
    return {
        'ema_period': trial.suggest_int('ema_period', 20, 100),
        'macd_fast':  trial.suggest_int('macd_fast', 8, 16),
        'macd_slow':  trial.suggest_int('macd_slow', 20, 34),
        'macd_sig':   trial.suggest_int('macd_sig', 7, 13),
        'sar_start':  trial.suggest_float('sar_start', 0.01, 0.05),
        'sar_inc':    trial.suggest_float('sar_inc', 0.01, 0.05),
        'sar_max':    trial.suggest_float('sar_max', 0.1, 0.4),
    }


# ── 6. Bitcoin Momentum EMA ───────────────────────────────────────────────────

def gen_Bitcoin_Momentum_EMA(df, ema_period=20, atr_period=14, atr_mult=1.5, **kw):
    """Trend above EMA, ATR-based caution filter, trailing stop."""
    ema       = _ema(df['close'], int(ema_period))
    atr       = _atr(df, int(atr_period))
    high_vol  = atr > atr.rolling(50).mean() * atr_mult   # high volatility flag
    trail_buy = df['close'] - atr                          # momentum trail

    long_cond  = (df['close'] > ema) & ~high_vol & (df['close'] > trail_buy.shift(1))
    short_cond = (df['close'] < ema) & ~high_vol & (df['close'] < (df['close'] + atr).shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Bitcoin_Momentum_EMA(trial):
    return {
        'ema_period': trial.suggest_int('ema_period', 10, 50),
        'atr_period': trial.suggest_int('atr_period', 10, 21),
        'atr_mult':   trial.suggest_float('atr_mult', 1.0, 3.0),
    }


# ── 7. Gaussian Channel ───────────────────────────────────────────────────────

def _gaussian_filter(series, poles=4, period=14):
    """Approximate multi-pole Gaussian filter via iterated EWM."""
    alpha = 1.0 - np.cos(2 * np.pi / period)
    alpha = (-alpha + np.sqrt(alpha ** 2 + 2 * alpha)) / 2
    result = series.copy().astype(float)
    for _ in range(poles):
        result = result.ewm(alpha=alpha, adjust=False).mean()
    return result


def gen_Gaussian_Channel(df, poles=4, period=14, std_mult=1.0, **kw):
    mid  = _gaussian_filter(df['close'], int(poles), int(period))
    std  = df['close'].rolling(int(period)).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std

    long_cond  = (df['close'] > upper) & (df['close'].shift(1) <= upper.shift(1))
    short_cond = (df['close'] < lower) & (df['close'].shift(1) >= lower.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Gaussian_Channel(trial):
    return {
        'poles':     trial.suggest_int('poles', 2, 8),
        'period':    trial.suggest_int('period', 10, 50),
        'std_mult':  trial.suggest_float('std_mult', 0.5, 3.0),
    }


# ── 8. Linear Regression Channel Breakout ────────────────────────────────────

def gen_Linear_Regression_Bo(df, lr_period=20, deviation=2.3, **kw):
    """Linear regression channel. Entry on channel boundary break."""
    lp = int(lr_period)
    x  = np.arange(lp)

    def lr_fit(arr):
        slope = np.polyfit(x, arr, 1)
        return slope[0] * (lp - 1) + slope[1]  # value at last bar

    def lr_std(arr):
        slope, intercept = np.polyfit(x, arr, 1)
        pred  = slope * x + intercept
        return np.std(arr - pred)

    mid_vals = df['close'].rolling(lp).apply(lr_fit, raw=True)
    std_vals = df['close'].rolling(lp).apply(lr_std, raw=True)

    upper = mid_vals + deviation * std_vals
    lower = mid_vals - deviation * std_vals

    sig = pd.Series(0, index=df.index)
    sig[df['close'] > upper.shift(1)] =  1
    sig[df['close'] < lower.shift(1)] = -1
    return sig


def space_Linear_Regression_Bo(trial):
    return {
        'lr_period': trial.suggest_int('lr_period', 10, 50),
        'deviation': trial.suggest_float('deviation', 1.0, 4.0),
    }


# ── 9. Swing Hull + T3 ───────────────────────────────────────────────────────

def gen_Swing_Hull_T3(df, ma_period=50, vfactor=0.7, **kw):
    """Hull MA + T3 average. Long on direction change upward."""
    hull = _hull_ma(df['close'], int(ma_period))
    t3   = _t3(df['close'], int(ma_period), vfactor)
    avg  = (hull + t3) / 2

    long_cond  = (avg > avg.shift(1)) & (avg.shift(1) <= avg.shift(2))
    short_cond = (avg < avg.shift(1)) & (avg.shift(1) >= avg.shift(2))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Swing_Hull_T3(trial):
    return {
        'ma_period': trial.suggest_int('ma_period', 20, 100),
        'vfactor':   trial.suggest_float('vfactor', 0.5, 0.9),
    }


# ── 10. Highest High / Lowest Low Channel Breakout ───────────────────────────

def gen_HighLow_Channel_Bo(df, period=20, **kw):
    """Long on close above prior N-bar highest high, short below lowest low."""
    hh = df['high'].rolling(int(period)).max().shift(1)
    ll = df['low'].rolling(int(period)).min().shift(1)

    sig = pd.Series(0, index=df.index)
    sig[df['close'] > hh] =  1
    sig[df['close'] < ll] = -1
    return sig


def space_HighLow_Channel_Bo(trial):
    return {
        'period': trial.suggest_int('period', 10, 50),
    }


# ── 11. Linear Regression + Pearson's R ──────────────────────────────────────

def gen_Linear_Regression_Pearson(df, lr_period=30, dev_low=2.0, dev_high=3.0,
                                   pearson_ideal=0.85, **kw):
    """LR channel + Pearson's R correlation quality filter."""
    lp = int(lr_period)
    x  = np.arange(lp, dtype=float)

    def pearson_r(arr):
        y     = np.array(arr, dtype=float)
        xm, ym = x.mean(), y.mean()
        num   = ((x - xm) * (y - ym)).sum()
        denom = np.sqrt(((x - xm)**2).sum() * ((y - ym)**2).sum())
        return abs(num / denom) if denom > 0 else 0

    def lr_val(arr):
        s, b = np.polyfit(x, arr, 1)
        return s * (lp - 1) + b

    def lr_std_fn(arr):
        s, b = np.polyfit(x, arr, 1)
        return np.std(np.array(arr) - (s * x + b))

    mid_vals = df['close'].rolling(lp).apply(lr_val, raw=True)
    std_vals = df['close'].rolling(lp).apply(lr_std_fn, raw=True)
    pr_vals  = df['close'].rolling(lp).apply(pearson_r, raw=True)

    upper = mid_vals + dev_high * std_vals
    lower = mid_vals - dev_high * std_vals
    good  = pr_vals >= pearson_ideal

    sig = pd.Series(0, index=df.index)
    sig[good & (df['close'] > upper.shift(1))] =  1
    sig[good & (df['close'] < lower.shift(1))] = -1
    return sig


def space_Linear_Regression_Pearson(trial):
    return {
        'lr_period':      trial.suggest_int('lr_period', 15, 60),
        'dev_low':        trial.suggest_float('dev_low', 1.0, 3.0),
        'dev_high':       trial.suggest_float('dev_high', 2.0, 5.0),
        'pearson_ideal':  trial.suggest_float('pearson_ideal', 0.6, 0.95),
    }


# ── 12. Donchian HH/LL (variant) ─────────────────────────────────────────────

def gen_Donchian_HH_LL(df, long_period=20, short_period=10, **kw):
    """Classic Turtle: enter on 20-bar HH/LL, exit on 10-bar."""
    enter_hi = df['high'].rolling(int(long_period)).max().shift(1)
    enter_lo = df['low'].rolling(int(long_period)).min().shift(1)

    sig = pd.Series(0, index=df.index)
    sig[df['close'] > enter_hi] =  1
    sig[df['close'] < enter_lo] = -1
    return sig


def space_Donchian_HH_LL(trial):
    return {
        'long_period':  trial.suggest_int('long_period', 10, 50),
        'short_period': trial.suggest_int('short_period', 5, 20),
    }


# ── STRATEGY_EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Fibonacci_RSI_VWMA': {
        'gen':            gen_Fibonacci_RSI_VWMA,
        'space':          space_Fibonacci_RSI_VWMA,
        'default_params': {'vwma_period': 20, 'std_mult': 2.0, 'rsi_period': 14},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 3285,
                           'description': 'VWMA+StdDev Fib bands + RSI cross'},
    },
    'Price_Divergence_Multi': {
        'gen':            gen_Price_Divergence_Multi,
        'space':          space_Price_Divergence_Multi,
        'default_params': {'rsi_period': 14, 'cci_period': 20, 'pivot_len': 5},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 3849,
                           'description': 'Multi-oscillator pivot divergence'},
    },
    'CipherB_WaveTrend': {
        'gen':            gen_CipherB_WaveTrend,
        'space':          space_CipherB_WaveTrend,
        'default_params': {'wt_n1': 10, 'wt_n2': 21, 'rsi_period': 14,
                           'mfi_period': 60, 'ema_period': 200},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 2451,
                           'description': 'Cipher B+: WaveTrend+MFI+RSI+EMA PB'},
    },
    'ATR_Trail_Dual': {
        'gen':            gen_ATR_Trail_Dual,
        'space':          space_ATR_Trail_Dual,
        'default_params': {'fast_period': 5, 'fast_mult': 0.5,
                           'slow_period': 10, 'slow_mult': 3.0},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 2284,
                           'description': 'Dual ATR trailing stop crossover'},
    },
    'Crypto_Scalper_SAR': {
        'gen':            gen_Crypto_Scalper_SAR,
        'space':          space_Crypto_Scalper_SAR,
        'default_params': {'ema_period': 60, 'macd_fast': 12, 'macd_slow': 26,
                           'macd_sig': 9, 'sar_start': 0.02, 'sar_inc': 0.02, 'sar_max': 0.2},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 3254,
                           'description': 'Crypto Scalper: EMA + MACD + PSAR'},
    },
    'Bitcoin_Momentum_EMA': {
        'gen':            gen_Bitcoin_Momentum_EMA,
        'space':          space_Bitcoin_Momentum_EMA,
        'default_params': {'ema_period': 20, 'atr_period': 14, 'atr_mult': 1.5},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1346,
                           'description': 'Bitcoin momentum with EMA + ATR caution'},
    },
    'Gaussian_Channel': {
        'gen':            gen_Gaussian_Channel,
        'space':          space_Gaussian_Channel,
        'default_params': {'poles': 4, 'period': 14, 'std_mult': 1.0},
        'info':           {'source': 'TradingView', 'version': 6, 'likes': 686,
                           'description': 'Multi-pole Gaussian channel filter'},
    },
    'Linear_Regression_Bo': {
        'gen':            gen_Linear_Regression_Bo,
        'space':          space_Linear_Regression_Bo,
        'default_params': {'lr_period': 20, 'deviation': 2.3},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 685,
                           'description': 'Linear regression channel breakout'},
    },
    'Swing_Hull_T3': {
        'gen':            gen_Swing_Hull_T3,
        'space':          space_Swing_Hull_T3,
        'default_params': {'ma_period': 50, 'vfactor': 0.7},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 1074,
                           'description': 'Swing Hull MA + T3 direction change'},
    },
    'HighLow_Channel_Bo': {
        'gen':            gen_HighLow_Channel_Bo,
        'space':          space_HighLow_Channel_Bo,
        'default_params': {'period': 20},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1617,
                           'description': 'Highest high / lowest low breakout'},
    },
    'Linear_Regression_Pearson': {
        'gen':            gen_Linear_Regression_Pearson,
        'space':          space_Linear_Regression_Pearson,
        'default_params': {'lr_period': 30, 'dev_low': 2.0,
                           'dev_high': 3.0, 'pearson_ideal': 0.85},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 867,
                           'description': 'LR channel + Pearson R quality filter'},
    },
    'Donchian_HH_LL': {
        'gen':            gen_Donchian_HH_LL,
        'space':          space_Donchian_HH_LL,
        'default_params': {'long_period': 20, 'short_period': 10},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1617,
                           'description': 'Donchian HH/LL Turtle Trading classic'},
    },
}
