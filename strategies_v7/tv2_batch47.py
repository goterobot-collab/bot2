#!/usr/bin/env python3
"""
tv2_batch47.py — Top 10 TradingView strategies (by likes)
Sources: TV API public strategies
- SuperTrend STRATEGY (23,336 likes) — KivancOzbilgic
- RSI Divergence (9,178 likes)
- Flawless Victory (10,590 likes) — DSK36
- PMax Explorer (16,152 likes) — KivancOzbilgic
- OTT Explorer (6,634 likes) — Anil Ozeksi
- UT Bot Strategy (7,537 likes) — HPotter
- Lorentzian Classification (4,858 likes) — jdehorty
- Adaptive HMA+ (4,552 likes) — io72signals
- VWAP + Fibo Dev (5,084 likes) — Mysteriown
- AI SuperTrend x Pivot (4,707 likes) — PresentTrading
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _wma(s: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=False)

def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (RMA)"""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _rma(gain, n)
    avg_loss = _rma(loss, n)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return _rma(tr, n)

def _bb(close: pd.Series, n: int = 20, mult: float = 2.0):
    basis = _sma(close, n)
    dev = close.rolling(n).std(ddof=0)
    return basis, basis + mult * dev, basis - mult * dev

def _crossover(s1: pd.Series, s2: pd.Series) -> pd.Series:
    return (s1 > s2) & (s1.shift(1) <= s2.shift(1))

def _crossunder(s1: pd.Series, s2: pd.Series) -> pd.Series:
    return (s1 < s2) & (s1.shift(1) >= s2.shift(1))

def _highest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).max()

def _lowest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).min()

def _pivot_low(low: pd.Series, left: int = 1, right: int = 1) -> pd.Series:
    """Detect pivot lows"""
    result = pd.Series(False, index=low.index)
    for i in range(left, len(low) - right):
        is_pivot = True
        for j in range(1, left + 1):
            if low.iloc[i] >= low.iloc[i - j]:
                is_pivot = False
                break
        for j in range(1, right + 1):
            if low.iloc[i] >= low.iloc[i + j]:
                is_pivot = False
                break
        result.iloc[i] = is_pivot
    return result

def _pivot_high(high: pd.Series, left: int = 1, right: int = 1) -> pd.Series:
    """Detect pivot highs"""
    result = pd.Series(False, index=high.index)
    for i in range(left, len(high) - right):
        is_pivot = True
        for j in range(1, left + 1):
            if high.iloc[i] <= high.iloc[i - j]:
                is_pivot = False
                break
        for j in range(1, right + 1):
            if high.iloc[i] <= high.iloc[i + j]:
                is_pivot = False
                break
        result.iloc[i] = is_pivot
    return result

# ---------------------------------------------------------------------------
# 1. SuperTrend STRATEGY (23,336 likes)
# ---------------------------------------------------------------------------

def gen_SuperTrendSTRATEGY(df: pd.DataFrame, **params) -> pd.Series:
    """SuperTrend trend following strategy"""
    periods = params.get('periods', 10)
    multiplier = params.get('multiplier', 3.0)

    h, l, c = df['high'], df['low'], df['close']
    src = (h + l) / 2
    atr = _atr(h, l, c, periods)

    up = src - (multiplier * atr)
    dn = src + (multiplier * atr)

    up_prev = up.shift(1).fillna(up)
    dn_prev = dn.shift(1).fillna(dn)

    up = up.where(c.shift(1) > up_prev, up_prev)
    up = up.where(up < up.shift(1).fillna(up), up.shift(1).fillna(up))

    dn = dn.where(c.shift(1) < dn_prev, dn_prev)
    dn = dn.where(dn > dn.shift(1).fillna(dn), dn.shift(1).fillna(dn))

    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if c.iloc[i] < up.iloc[i]:
            trend.iloc[i] = -1
        elif c.iloc[i] > dn.iloc[i]:
            trend.iloc[i] = 1
        else:
            trend.iloc[i] = trend.iloc[i-1]

    buy = (trend == 1) & (trend.shift(1) != 1)
    sell = (trend == -1) & (trend.shift(1) != -1)

    signal = pd.Series(0, index=df.index)
    signal.loc[buy] = 1
    signal.loc[sell] = -1
    return signal

def space_SuperTrendSTRATEGY():
    return {
        'periods': (5, 20, 1),
        'multiplier': (1.0, 4.0, 0.5)
    }

# ---------------------------------------------------------------------------
# 2. RSI Divergence (9,178 likes)
# ---------------------------------------------------------------------------

def gen_RSIDivergence(df: pd.DataFrame, **params) -> pd.Series:
    """RSI divergence detection strategy"""
    rsi_len = params.get('rsi_len', 9)
    lb_r = params.get('lb_r', 3)
    lb_l = params.get('lb_l', 1)
    tp_level = params.get('tp_level', 80)

    rsi = _rsi(df['close'], rsi_len)
    low, high = df['low'], df['high']

    # Pivot detection
    pl = _pivot_low(low, lb_l, lb_r)
    ph = _pivot_high(high, lb_l, lb_r)

    # Regular bullish divergence: pivot low + RSI higher low
    rsi_shifted = rsi.shift(lb_r + 1)
    bull_cond = pl & (rsi > rsi_shifted)
    sell_cond = _crossover(rsi, pd.Series(tp_level, index=rsi.index))

    signal = pd.Series(0, index=df.index)
    signal.loc[bull_cond] = 1
    signal.loc[sell_cond] = -1
    return signal

def space_RSIDivergence():
    return {
        'rsi_len': (5, 14, 1),
        'lb_r': (2, 5, 1),
        'lb_l': (1, 3, 1),
        'tp_level': (70, 85, 5)
    }

# ---------------------------------------------------------------------------
# 3. Flawless Victory Strategy (10,590 likes)
# ---------------------------------------------------------------------------

def gen_FlawlessVictory(df: pd.DataFrame, **params) -> pd.Series:
    """Flawless Victory BB + RSI strategy"""
    rsi_lower = params.get('rsi_lower', 42)
    rsi_upper = params.get('rsi_upper', 70)
    bb_len = params.get('bb_len', 20)
    bb_mult = params.get('bb_mult', 1.0)

    c = df['close']
    rsi = _rsi(c, 14)
    basis, upper, lower = _bb(c, bb_len, bb_mult)

    buy = (c < lower) & (rsi > rsi_lower)
    sell = (c > upper) & (rsi > rsi_upper)

    signal = pd.Series(0, index=df.index)
    signal.loc[buy] = 1
    signal.loc[sell] = -1
    return signal

def space_FlawlessVictory():
    return {
        'rsi_lower': (35, 50, 5),
        'rsi_upper': (65, 80, 5),
        'bb_len': (15, 25, 2),
        'bb_mult': (0.5, 2.0, 0.25)
    }

# ---------------------------------------------------------------------------
# 4. PMax Explorer (16,152 likes)
# ---------------------------------------------------------------------------

def gen_PMAxExplorer(df: pd.DataFrame, **params) -> pd.Series:
    """Trend following with MA + ATR bands"""
    atr_len = params.get('atr_len', 10)
    atr_mult = params.get('atr_mult', 3.0)
    ma_len = params.get('ma_len', 10)

    h, l, c = df['high'], df['low'], df['close']
    src = (h + l) / 2
    atr = _atr(h, l, c, atr_len)
    ma = _ema(src, ma_len)

    long_stop = ma - atr_mult * atr
    short_stop = ma + atr_mult * atr

    long_stop = long_stop.fillna(method='bfill')
    short_stop = short_stop.fillna(method='bfill')

    buy = _crossover(ma, long_stop)
    sell = _crossunder(ma, short_stop)

    signal = pd.Series(0, index=df.index)
    signal.loc[buy] = 1
    signal.loc[sell] = -1
    return signal

def space_PMAxExplorer():
    return {
        'atr_len': (5, 20, 1),
        'atr_mult': (1.5, 4.0, 0.5),
        'ma_len': (5, 20, 2)
    }

# ---------------------------------------------------------------------------
# 5. OTT Explorer (6,634 likes)
# ---------------------------------------------------------------------------

def gen_OTTExplorer(df: pd.DataFrame, **params) -> pd.Series:
    """Optimized Trend Tracker"""
    ott_len = params.get('ott_len', 2)
    ott_pct = params.get('ott_pct', 1.4)

    c = df['close']
    ma = _ema(c, ott_len)

    fark = ma * ott_pct * 0.01
    long_stop = ma - fark
    short_stop = ma + fark

    long_stop = long_stop.fillna(method='bfill')
    short_stop = short_stop.fillna(method='bfill')

    buy = _crossover(ma, long_stop)
    sell = _crossunder(ma, short_stop)

    signal = pd.Series(0, index=df.index)
    signal.loc[buy] = 1
    signal.loc[sell] = -1
    return signal

def space_OTTExplorer():
    return {
        'ott_len': (1, 5, 1),
        'ott_pct': (0.5, 2.5, 0.3)
    }

# ---------------------------------------------------------------------------
# 6. UT Bot Strategy (7,537 likes)
# ---------------------------------------------------------------------------

def gen_UTBotStrategy(df: pd.DataFrame, **params) -> pd.Series:
    """UT Bot ATR trailing stop strategy"""
    key_val = params.get('key_val', 1)
    atr_period = params.get('atr_period', 10)

    c = df['close']
    atr = _atr(df['high'], df['low'], c, atr_period)

    n_loss = key_val * atr

    xatr_stop = pd.Series(0.0, index=df.index)
    pos = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if c.iloc[i] > xatr_stop.iloc[i-1] and c.iloc[i-1] > xatr_stop.iloc[i-1]:
            xatr_stop.iloc[i] = max(xatr_stop.iloc[i-1], c.iloc[i] - n_loss.iloc[i])
        elif c.iloc[i] < xatr_stop.iloc[i-1] and c.iloc[i-1] < xatr_stop.iloc[i-1]:
            xatr_stop.iloc[i] = min(xatr_stop.iloc[i-1], c.iloc[i] + n_loss.iloc[i])
        else:
            xatr_stop.iloc[i] = c.iloc[i] - n_loss.iloc[i] if c.iloc[i] > xatr_stop.iloc[i-1] else c.iloc[i] + n_loss.iloc[i]

        if c.iloc[i-1] < xatr_stop.iloc[i-1] and c.iloc[i] > xatr_stop.iloc[i]:
            pos.iloc[i] = 1
        elif c.iloc[i-1] > xatr_stop.iloc[i-1] and c.iloc[i] < xatr_stop.iloc[i]:
            pos.iloc[i] = -1
        else:
            pos.iloc[i] = pos.iloc[i-1]

    buy = (pos == 1) & (pos.shift(1) != 1)
    sell = (pos == -1) & (pos.shift(1) != -1)

    signal = pd.Series(0, index=df.index)
    signal.loc[buy] = 1
    signal.loc[sell] = -1
    return signal

def space_UTBotStrategy():
    return {
        'key_val': (0.5, 3.0, 0.25),
        'atr_period': (5, 20, 1)
    }

# ---------------------------------------------------------------------------
# 7. Lorentzian Classification (4,858 likes) — Simplified
# ---------------------------------------------------------------------------

def gen_LorentzianClassification(df: pd.DataFrame, **params) -> pd.Series:
    """Lorentzian ML-inspired strategy (simplified)"""
    rsi_len = params.get('rsi_len', 14)
    cci_len = params.get('cci_len', 20)

    # Simple KNN-like approach using multiple indicators
    rsi = _rsi(df['close'], rsi_len)

    # CCI calculation
    typ = (df['high'] + df['low'] + df['close']) / 3
    ma = typ.rolling(cci_len).mean()
    dev = typ.rolling(cci_len).std()
    cci = (typ - ma) / (0.015 * dev + 1e-8)

    buy = (rsi < 30) & (cci < -100)
    sell = (rsi > 70) & (cci > 100)

    signal = pd.Series(0, index=df.index)
    signal.loc[buy] = 1
    signal.loc[sell] = -1
    return signal

def space_LorentzianClassification():
    return {
        'rsi_len': (10, 20, 2),
        'cci_len': (15, 30, 2)
    }

# ---------------------------------------------------------------------------
# 8. Adaptive HMA+ (4,552 likes) — Simplified
# ---------------------------------------------------------------------------

def gen_AdaptiveHMAplus(df: pd.DataFrame, **params) -> pd.Series:
    """Adaptive Hull Moving Average strategy"""
    hma_len = params.get('hma_len', 50)
    atr_mult = params.get('atr_mult', 2.0)

    c = df['close']

    # Hull MA calculation
    hma_half = _wma(c, hma_len // 2)
    hma_full = _wma(c, hma_len)
    hma = _wma(2 * hma_half - hma_full, int(np.sqrt(hma_len)))

    atr = _atr(df['high'], df['low'], c, 14)
    upper = hma + atr_mult * atr
    lower = hma - atr_mult * atr

    buy = _crossover(c, lower)
    sell = _crossunder(c, upper)

    signal = pd.Series(0, index=df.index)
    signal.loc[buy] = 1
    signal.loc[sell] = -1
    return signal

def space_AdaptiveHMAplus():
    return {
        'hma_len': (30, 70, 5),
        'atr_mult': (1.5, 3.0, 0.25)
    }

# ---------------------------------------------------------------------------
# 9. VWAP + Fibo Dev (5,084 likes)
# ---------------------------------------------------------------------------

def gen_VWAPFiboDev(df: pd.DataFrame, **params) -> pd.Series:
    """VWAP + Fibonacci deviation strategy"""
    fib1 = params.get('fib1', 1.618)
    fib2 = params.get('fib2', 2.618)
    dev_min = params.get('dev_min', 150)

    h, l, c, v = df['high'], df['low'], df['close'], df['volume']
    hlc3 = (h + l + c) / 3

    # VWAP calculation
    vwap = (hlc3 * v).rolling(20).sum() / v.rolling(20).sum()

    # Deviation
    sn = ((hlc3 - vwap) * v).rolling(20).sum()
    sd = np.sqrt(sn / v.rolling(20).sum()).clip(lower=1)

    fib_p1 = vwap + fib1 * sd
    fib_m1 = vwap - fib1 * sd

    buy = _crossunder(l, fib_m1) & (sd > dev_min / 10000)
    sell = _crossover(h, fib_p1) & (sd > dev_min / 10000)

    signal = pd.Series(0, index=df.index)
    signal.loc[buy] = 1
    signal.loc[sell] = -1
    return signal

def space_VWAPFiboDev():
    return {
        'fib1': (1.0, 2.0, 0.1),
        'fib2': (2.0, 3.5, 0.2),
        'dev_min': (100, 300, 25)
    }

# ---------------------------------------------------------------------------
# 10. AI SuperTrend x Pivot (4,707 likes) — Simplified
# ---------------------------------------------------------------------------

def gen_AISuperTrendPivot(df: pd.DataFrame, **params) -> pd.Series:
    """AI SuperTrend with Pivot detection"""
    st_len = params.get('st_len', 10)
    st_mult = params.get('st_mult', 3.5)
    pivot_len = params.get('pivot_len', 14)

    h, l, c = df['high'], df['low'], df['close']

    # SuperTrend
    hl2 = (h + l) / 2
    atr = _atr(h, l, c, st_len)

    up = hl2 - st_mult * atr
    dn = hl2 + st_mult * atr

    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if c.iloc[i] < up.iloc[i]:
            trend.iloc[i] = -1
        elif c.iloc[i] > dn.iloc[i]:
            trend.iloc[i] = 1
        else:
            trend.iloc[i] = trend.iloc[i-1]

    # Pivot detection
    pivot_h = _pivot_high(h, 2, 2)
    pivot_l = _pivot_low(l, 2, 2)

    buy = (trend == 1) & pivot_l
    sell = (trend == -1) & pivot_h

    signal = pd.Series(0, index=df.index)
    signal.loc[buy] = 1
    signal.loc[sell] = -1
    return signal

def space_AISuperTrendPivot():
    return {
        'st_len': (5, 20, 1),
        'st_mult': (2.0, 5.0, 0.5),
        'pivot_len': (10, 20, 2)
    }

# ---------------------------------------------------------------------------
# STRATEGY_EXPORT
# ---------------------------------------------------------------------------

STRATEGY_EXPORT = {
    'SuperTrendSTRATEGY': {
        'gen': gen_SuperTrendSTRATEGY,
        'space': space_SuperTrendSTRATEGY
    },
    'RSIDivergence': {
        'gen': gen_RSIDivergence,
        'space': space_RSIDivergence
    },
    'FlawlessVictory': {
        'gen': gen_FlawlessVictory,
        'space': space_FlawlessVictory
    },
    'PMAxExplorer': {
        'gen': gen_PMAxExplorer,
        'space': space_PMAxExplorer
    },
    'OTTExplorer': {
        'gen': gen_OTTExplorer,
        'space': space_OTTExplorer
    },
    'UTBotStrategy': {
        'gen': gen_UTBotStrategy,
        'space': space_UTBotStrategy
    },
    'LorentzianClassification': {
        'gen': gen_LorentzianClassification,
        'space': space_LorentzianClassification
    },
    'AdaptiveHMAplus': {
        'gen': gen_AdaptiveHMAplus,
        'space': space_AdaptiveHMAplus
    },
    'VWAPFiboDev': {
        'gen': gen_VWAPFiboDev,
        'space': space_VWAPFiboDev
    },
    'AISuperTrendPivot': {
        'gen': gen_AISuperTrendPivot,
        'space': space_AISuperTrendPivot
    }
}
