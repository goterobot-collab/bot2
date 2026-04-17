"""Round 2 of new families: CCI, CMF (volume), Heikin-Ashi, Aroon, KAMA, GapFade."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.indicators import ema, sma, atr, crossover, crossunder


# ============================================================================
# 1. CCI (Commodity Channel Index) mean reversion.
#    CCI = (TP - SMA(TP,n)) / (0.015 * MAD), buy when CCI < -threshold in uptrend
# ============================================================================
def cci(high, low, close, n):
    tp = (high + low + close) / 3.0
    m = tp.rolling(n, min_periods=n).mean()
    mad = tp.rolling(n, min_periods=n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - m) / (0.015 * mad.replace(0, np.nan))


def sig_cci_revert(df, p):
    c = df["close"]
    x = cci(df["high"], df["low"], c, p["cci_len"])
    trend_ok = c > sma(c, p["trend"])
    entry = (x < -p["os"]) & trend_ok
    ex = x > p["exit_lvl"]
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 2. Chaikin Money Flow (CMF) + trend pullback.
#    Volume-based: positive CMF means accumulation.
# ============================================================================
def cmf(high, low, close, volume, n):
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = (mfm * volume).fillna(0.0)
    return mfv.rolling(n, min_periods=n).sum() / volume.rolling(n, min_periods=n).sum()


def sig_cmf_pullback(df, p):
    c = df["close"]
    cm = cmf(df["high"], df["low"], c, df["volume"], p["cmf_len"])
    e = ema(c, p["ema_len"])
    trend_ok = c > sma(c, p["trend"])
    pullback = (df["low"].rolling(p["lookback"], min_periods=1).min() <= e) & (c > e)
    entry = trend_ok & pullback & (cm > p["cmf_min"])
    ex = (cm < 0) | crossunder(c, e)
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 3. Heikin-Ashi consecutive-bull trend.
#    HA smooths noise: buy when N consecutive HA bars close > open.
# ============================================================================
def heikin_ashi(df):
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
    return ha_open, ha_close


def sig_heikin_trend(df, p):
    c = df["close"]
    ha_o, ha_c = heikin_ashi(df)
    bull = ha_c > ha_o
    consec = bull.rolling(p["consec"], min_periods=p["consec"]).sum() == p["consec"]
    trend_ok = c > sma(c, p["trend"])
    entry = consec & trend_ok & ~consec.shift(1).fillna(False)
    bear = ha_c < ha_o
    ex = bear.rolling(p["exit_consec"], min_periods=p["exit_consec"]).sum() == p["exit_consec"]
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 4. Aroon crossover — timing of high/low recency.
#    Aroon Up = 100*(n - bars_since_high)/n, similar for Down.
# ============================================================================
def aroon(high, low, n):
    au = high.rolling(n + 1, min_periods=n + 1).apply(
        lambda x: 100.0 * (n - (n - np.argmax(x))) / n, raw=True
    )
    ad = low.rolling(n + 1, min_periods=n + 1).apply(
        lambda x: 100.0 * (n - (n - np.argmin(x))) / n, raw=True
    )
    return au, ad


def sig_aroon_cross(df, p):
    c = df["close"]
    au, ad = aroon(df["high"], df["low"], p["ar_len"])
    trend_ok = c > sma(c, p["trend"])
    entry = crossover(au, ad) & (au > p["up_min"]) & trend_ok
    ex = crossunder(au, ad)
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 5. Kaufman Adaptive Moving Average (KAMA) trend crossover.
#    KAMA adapts its speed based on efficiency ratio (signal/noise).
# ============================================================================
def kama(close, n, fast, slow):
    change = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(n, min_periods=n).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0.0)
    sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    k = pd.Series(index=close.index, dtype=float)
    k.iloc[:n] = close.iloc[:n]
    for i in range(n, len(close)):
        k.iloc[i] = k.iloc[i - 1] + sc.iloc[i] * (close.iloc[i] - k.iloc[i - 1])
    return k


def sig_kama_trend(df, p):
    c = df["close"]
    k_fast = kama(c, p["er_len"], p["fast"], p["slow_fast"])
    k_slow = kama(c, p["er_len"], p["fast_slow"], p["slow"])
    trend_ok = c > sma(c, p["trend"])
    entry = crossover(k_fast, k_slow) & (c > k_slow) & trend_ok
    ex = crossunder(c, k_slow)
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 6. Gap fade (open > prev close by X ATR -> fade to mean).
#    On crypto 1d this is the weekend/overnight gap-down edge.
# ============================================================================
def sig_gap_fade(df, p):
    c = df["close"]
    o = df["open"]
    a = atr(df["high"], df["low"], c, 14)
    prev_c = c.shift(1)
    gap_up = (o - prev_c) > p["gap_atr"] * a.shift(1)
    trend_ok = c > sma(c, p["trend"])
    # fade: enter short-like long exit? Here fade = buy gap-DOWN (reversion).
    gap_dn = (prev_c - o) > p["gap_atr"] * a.shift(1)
    entry = gap_dn & trend_ok & (c > o)  # intraday reclaim of open
    ex = (c > prev_c) | gap_up
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# PARAMETER GRIDS
# ============================================================================
SPACES_MORE = {
    "cci_revert": {
        "sig": sig_cci_revert,
        "params": {
            "cci_len":  [10, 14, 20, 30],
            "os":       [100, 150, 200, 250],
            "exit_lvl": [-50, 0, 50, 100],
            "trend":    [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 2.5, 3.0],
            "tp_atr":    [None, 2.0, 3.0, 4.0, 5.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None, 12, 24, 48, 72],
        },
    },
    "cmf_pullback": {
        "sig": sig_cmf_pullback,
        "params": {
            "cmf_len":  [14, 20, 30, 50],
            "cmf_min":  [0.0, 0.05, 0.1, 0.2],
            "ema_len":  [13, 21, 34, 50],
            "lookback": [3, 5, 8, 12],
            "trend":    [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 2.5],
            "tp_atr":    [None, 2.0, 3.0, 4.0],
            "trail_atr": [None, 2.0, 3.0],
            "timeout":   [None, 24, 48, 72],
        },
    },
    "heikin_trend": {
        "sig": sig_heikin_trend,
        "params": {
            "consec":      [3, 4, 5, 7],
            "exit_consec": [1, 2, 3],
            "trend":       [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [None, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 3.0, 5.0, 8.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None, 24, 48, 96],
        },
    },
    "aroon_cross": {
        "sig": sig_aroon_cross,
        "params": {
            "ar_len": [14, 20, 25, 30],
            "up_min": [50, 60, 70, 80],
            "trend":  [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [None, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 3.0, 5.0, 8.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None, 24, 48, 96],
        },
    },
    "kama_trend": {
        "sig": sig_kama_trend,
        "params": {
            "er_len":    [10, 14, 20],
            "fast":      [2, 3, 5],
            "slow_fast": [15, 20, 30],
            "fast_slow": [5, 10, 15],
            "slow":      [30, 50, 100],
            "trend":     [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [None, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 3.0, 5.0, 8.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None, 48, 96, 168],
        },
    },
    "gap_fade": {
        "sig": sig_gap_fade,
        "params": {
            "gap_atr": [0.5, 0.75, 1.0, 1.5, 2.0],
            "trend":   [50, 100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 2.5],
            "tp_atr":    [None, 1.5, 2.0, 3.0],
            "trail_atr": [None, 1.5, 2.0],
            "timeout":   [None, 4, 8, 12, 24],
        },
    },
}
