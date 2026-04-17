"""Vectorized technical indicators on pandas Series."""
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    # Wilder's smoothing == ewm alpha=1/n
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0.0)
    dn = (-diff).clip(lower=0.0)
    rs = rma(up, n) / rma(dn, n).replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    sig = ema(macd_line, signal)
    hist = macd_line - sig
    return macd_line, sig, hist


def bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * std, mid, mid + k * std


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    hh = high.rolling(k, min_periods=k).max()
    ll = low.rolling(k, min_periods=k).min()
    kline = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    dline = kline.rolling(d, min_periods=d).mean()
    return kline, dline


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, mult: float = 3.0):
    a = atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    # final bands + direction
    n = len(close)
    st = np.full(n, np.nan)
    dirn = np.ones(n, dtype=int)  # 1 = uptrend, -1 = downtrend
    upper_v = upper.values.copy()
    lower_v = lower.values.copy()
    c = close.values
    for i in range(1, n):
        if np.isnan(upper_v[i - 1]) or np.isnan(lower_v[i - 1]):
            continue
        if upper_v[i] > upper_v[i - 1] and c[i - 1] <= upper_v[i - 1]:
            upper_v[i] = upper_v[i - 1]
        if lower_v[i] < lower_v[i - 1] and c[i - 1] >= lower_v[i - 1]:
            lower_v[i] = lower_v[i - 1]
        if c[i] > upper_v[i - 1]:
            dirn[i] = 1
        elif c[i] < lower_v[i - 1]:
            dirn[i] = -1
        else:
            dirn[i] = dirn[i - 1]
        st[i] = lower_v[i] if dirn[i] == 1 else upper_v[i]
    return pd.Series(st, index=close.index), pd.Series(dirn, index=close.index)


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))
