"""Six new strategy families to diversify the hunt beyond rsi2/bb/zscore/trend.

Registered into grail_loop.SPACES on import.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.indicators import ema, sma, rsi, atr, stoch, crossover, crossunder


# ============================================================================
# 1. ConnorsRSI (CRSI): 3-component oscillator from Larry Connors.
#    CRSI = (RSI(close, n1) + RSI(streak, n2) + PercentRank(roc, n3)) / 3
# ============================================================================
def _streak(close: pd.Series) -> pd.Series:
    d = np.sign(close.diff().fillna(0.0).values)
    s = np.zeros(len(d), dtype=float)
    for i in range(1, len(d)):
        if d[i] > 0:
            s[i] = s[i - 1] + 1 if s[i - 1] >= 0 else 1
        elif d[i] < 0:
            s[i] = s[i - 1] - 1 if s[i - 1] <= 0 else -1
        else:
            s[i] = 0
    return pd.Series(s, index=close.index)


def _pct_rank(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).apply(
        lambda x: 100.0 * (x[-1] > x[:-1]).sum() / (len(x) - 1),
        raw=True,
    )


def crsi(close: pd.Series, n_rsi: int, n_streak: int, n_pct: int) -> pd.Series:
    r1 = rsi(close, n_rsi)
    r2 = rsi(_streak(close), n_streak)
    r3 = _pct_rank(close.pct_change().fillna(0.0), n_pct)
    return (r1 + r2 + r3) / 3.0


def sig_connors_rsi(df, p):
    c = df["close"]
    cr = crsi(c, p["n_rsi"], p["n_streak"], p["n_pct"])
    trend_ok = c > sma(c, p["trend"])
    entry = (cr < p["buy"]) & trend_ok
    ex = cr > p["sell"]
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 2. Williams %R mean reversion: buy on deep oversold in uptrend.
# ============================================================================
def williams_r(high, low, close, n):
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    return -100.0 * (hh - close) / (hh - ll).replace(0, np.nan)


def sig_williams_revert(df, p):
    c = df["close"]
    w = williams_r(df["high"], df["low"], c, p["w_len"])
    trend_ok = c > ema(c, p["trend"])
    entry = (w < -p["os"]) & trend_ok
    ex = w > -p["ob"]
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 3. Ichimoku Cloud: long when price > cloud + tenkan/kijun cross up.
# ============================================================================
def _ichi_line(high, low, n):
    return (high.rolling(n, min_periods=n).max()
            + low.rolling(n, min_periods=n).min()) / 2.0


def sig_ichimoku(df, p):
    h, l, c = df["high"], df["low"], df["close"]
    tenkan = _ichi_line(h, l, p["tenkan"])
    kijun = _ichi_line(h, l, p["kijun"])
    span_a = ((tenkan + kijun) / 2.0).shift(p["shift"])
    span_b = _ichi_line(h, l, p["senkou_b"]).shift(p["shift"])
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    above_cloud = c > cloud_top
    bull_cloud = span_a > span_b
    cross = crossover(tenkan, kijun)
    entry = cross & above_cloud & bull_cloud
    ex = crossunder(tenkan, kijun) | (c < cloud_bot)
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 4. Hull Moving Average (Alan Hull): HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
# ============================================================================
def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n, min_periods=n).apply(
        lambda x: np.dot(x, w) / w.sum(), raw=True,
    )


def hma(s: pd.Series, n: int) -> pd.Series:
    half = max(1, n // 2)
    sqrt_n = max(1, int(round(np.sqrt(n))))
    return wma(2 * wma(s, half) - wma(s, n), sqrt_n)


def sig_hull_cross(df, p):
    c = df["close"]
    h_fast = hma(c, p["fast"])
    h_slow = hma(c, p["slow"])
    trend_ok = c > sma(c, p["trend"])
    entry = crossover(h_fast, h_slow) & trend_ok
    ex = crossunder(h_fast, h_slow)
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 5. Stochastic %K/%D cross from oversold (momentum reversal).
# ============================================================================
def sig_stoch_cross(df, p):
    c = df["close"]
    k, d = stoch(df["high"], df["low"], c, p["k_len"], p["d_len"])
    trend_ok = c > sma(c, p["trend"])
    # cross up while both below oversold threshold
    entry = crossover(k, d) & (k < p["os"]) & trend_ok
    ex = k > p["ob"]
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 6. Volatility breakout (NR-style): enter when close breaks N-bar high after
#    N-bar low-range contraction. Trend-filtered.
# ============================================================================
def sig_vol_breakout(df, p):
    c, h, l = df["close"], df["high"], df["low"]
    rng = (h - l).rolling(p["nr_len"], min_periods=p["nr_len"])
    # contraction signal: current range is the smallest in the last nr_len bars
    contracted = (h - l) == rng.min()
    prior_contract = contracted.shift(1).fillna(False)
    # breakout: close > max(close, p["break_len"]) of prior bars
    brk = c > c.rolling(p["break_len"], min_periods=p["break_len"]).max().shift(1)
    trend_ok = c > sma(c, p["trend"])
    entry = brk & prior_contract & trend_ok
    # exit on close below N-bar low
    ex = c < c.rolling(p["exit_len"], min_periods=p["exit_len"]).min().shift(1)
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# PARAMETER GRIDS & EXIT GRIDS
# ============================================================================
SPACES_NEW = {
    "connors_rsi": {
        "sig": sig_connors_rsi,
        "params": {
            "n_rsi":    [2, 3, 4, 5],
            "n_streak": [2, 3, 5],
            "n_pct":    [50, 100, 150],
            "buy":      [5, 10, 15, 20, 25],
            "sell":     [50, 60, 70, 80],
            "trend":    [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 2.5, 3.0],
            "tp_atr":    [None, 2.0, 3.0, 4.0, 5.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None, 12, 24, 48, 72],
        },
    },
    "williams_revert": {
        "sig": sig_williams_revert,
        "params": {
            "w_len": [5, 7, 10, 14, 21],
            "os":    [80, 85, 90, 95],
            "ob":    [20, 30, 40, 50],
            "trend": [50, 100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 2.5, 3.0],
            "tp_atr":    [None, 2.0, 3.0, 4.0, 5.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None, 12, 24, 48, 72],
        },
    },
    "ichimoku": {
        "sig": sig_ichimoku,
        "params": {
            "tenkan":   [7, 9, 12, 20],
            "kijun":    [22, 26, 30, 52],
            "senkou_b": [44, 52, 78, 104],
            "shift":    [13, 20, 26, 30],
        },
        "exit": {
            "sl_atr":    [None, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 3.0, 4.0, 6.0, 8.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None, 48, 96, 168],
        },
    },
    "hull_cross": {
        "sig": sig_hull_cross,
        "params": {
            "fast":  [9, 14, 21, 34],
            "slow":  [34, 55, 89, 144],
            "trend": [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [None, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 3.0, 5.0, 8.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None, 48, 96, 168],
        },
    },
    "stoch_cross": {
        "sig": sig_stoch_cross,
        "params": {
            "k_len": [9, 14, 21],
            "d_len": [3, 5, 9],
            "os":    [15, 20, 25, 30],
            "ob":    [70, 80, 85],
            "trend": [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 2.5],
            "tp_atr":    [None, 2.0, 3.0, 4.0],
            "trail_atr": [None, 2.0, 3.0],
            "timeout":   [None, 12, 24, 48, 72],
        },
    },
    "vol_breakout": {
        "sig": sig_vol_breakout,
        "params": {
            "nr_len":    [4, 7, 10],
            "break_len": [10, 20, 30, 55],
            "exit_len":  [5, 10, 15, 20],
            "trend":     [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 3.0, 5.0, 8.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None, 24, 48, 96],
        },
    },
}
