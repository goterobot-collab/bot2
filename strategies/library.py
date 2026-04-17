"""
Strategy library — long-only signal generators.

Each strategy is a function `(df: pd.DataFrame) -> (entry: Series[bool], exit: Series[bool])`
where df has columns: open, high, low, close, volume (index = timestamp).

Strategies are adapted from public Pine Script sources; original Pine URLs
are listed in SOURCES and archived in pine_sources/ at the repo root.
Author-reported WR values (when available) are metadata only, NOT verified here.
"""
from __future__ import annotations
import pandas as pd
from .indicators import ema, sma, rsi, macd, bbands, stoch, supertrend, crossover, crossunder, atr


SOURCES = {
    "rsi2_connors": {
        "name": "RSI(2) mean-reversion + SMA200 trend filter",
        "author": "Larry Connors (public methodology)",
        "pine_ref": "pine_sources/rsi2_connors.pine",
        "logic": "Long when close > SMA200 and RSI(2) < 10; exit when close > SMA(5).",
    },
    "rsi_oversold_rev": {
        "name": "RSI(14) oversold reversal",
        "author": "Generic mean-reversion (public)",
        "pine_ref": "pine_sources/rsi_oversold_rev.pine",
        "logic": "Long when RSI(14) crosses above 30; exit when RSI(14) crosses below 70.",
    },
    "macd_ema200": {
        "name": "MACD cross + EMA200 filter",
        "author": "Generic (public)",
        "pine_ref": "pine_sources/macd_ema200.pine",
        "logic": "Long when MACD crosses above signal and close > EMA200; exit on MACD<signal.",
    },
    "ema9_21_cross": {
        "name": "EMA 9/21 crossover + EMA200 trend filter",
        "author": "Generic (public)",
        "pine_ref": "pine_sources/ema9_21_cross.pine",
        "logic": "Long when EMA9 crosses above EMA21 and close > EMA200; exit on reverse cross.",
    },
    "bb_meanrev": {
        "name": "Bollinger Band mean reversion",
        "author": "Generic (public)",
        "pine_ref": "pine_sources/bb_meanrev.pine",
        "logic": "Long when close crosses above lower BB(20,2); exit when close crosses mid band.",
    },
    "supertrend_ema": {
        "name": "Supertrend(10,3) + EMA200 filter",
        "author": "Generic Pine community",
        "pine_ref": "pine_sources/supertrend_ema.pine",
        "logic": "Long on Supertrend flip to up while close > EMA200; exit on flip to down.",
    },
    "stoch_rev": {
        "name": "Stochastic(14,3) oversold cross",
        "author": "Generic (public)",
        "pine_ref": "pine_sources/stoch_rev.pine",
        "logic": "Long when %K crosses above %D below 20; exit when %K crosses below %D above 80.",
    },
    "macd_hull_session": {
        "name": "MACD+HMA session strategy (exlux99)",
        "author": "exlux99 (Pine v4, MPL-2.0)",
        "pine_ref": "pine_sources/macd_hull_session.pine",
        "logic": "Adapted: long when 3 consecutive up candles + MACD hist crosses 0 (session window removed for crypto 24/7); ATR-based SL/TP.",
    },
}


# ---------- helpers ----------
def _empty(df: pd.DataFrame):
    s = pd.Series(False, index=df.index)
    return s.copy(), s.copy()


# ---------- strategies ----------
def rsi2_connors(df: pd.DataFrame):
    c = df["close"]
    f = sma(c, 200)
    r = rsi(c, 2)
    exit_ma = sma(c, 5)
    entry = (c > f) & (r < 10)
    # exit when close > SMA(5) (bar-to-bar)
    ex = c > exit_ma
    return entry.fillna(False), ex.fillna(False)


def rsi_oversold_rev(df: pd.DataFrame):
    c = df["close"]
    r = rsi(c, 14)
    entry = crossover(r, pd.Series(30.0, index=c.index))
    ex = crossunder(r, pd.Series(70.0, index=c.index))
    return entry.fillna(False), ex.fillna(False)


def macd_ema200(df: pd.DataFrame):
    c = df["close"]
    m, s, _ = macd(c, 12, 26, 9)
    f = ema(c, 200)
    entry = crossover(m, s) & (c > f)
    ex = crossunder(m, s)
    return entry.fillna(False), ex.fillna(False)


def ema9_21_cross(df: pd.DataFrame):
    c = df["close"]
    e9 = ema(c, 9)
    e21 = ema(c, 21)
    f = ema(c, 200)
    entry = crossover(e9, e21) & (c > f)
    ex = crossunder(e9, e21)
    return entry.fillna(False), ex.fillna(False)


def bb_meanrev(df: pd.DataFrame):
    c = df["close"]
    lo, mid, _ = bbands(c, 20, 2.0)
    entry = crossover(c, lo)
    ex = crossover(c, mid)
    return entry.fillna(False), ex.fillna(False)


def supertrend_ema(df: pd.DataFrame):
    st, dirn = supertrend(df["high"], df["low"], df["close"], 10, 3.0)
    f = ema(df["close"], 200)
    flip_up = (dirn == 1) & (dirn.shift(1) == -1)
    flip_dn = (dirn == -1) & (dirn.shift(1) == 1)
    entry = flip_up & (df["close"] > f)
    return entry.fillna(False), flip_dn.fillna(False)


def stoch_rev(df: pd.DataFrame):
    k, d = stoch(df["high"], df["low"], df["close"], 14, 3)
    entry = crossover(k, d) & (k < 20)
    ex = crossunder(k, d) & (k > 80)
    return entry.fillna(False), ex.fillna(False)


def macd_hull_session(df: pd.DataFrame):
    # Adapted from exlux99 Pine v4 "Very high win rate strategy".
    # Removed forex session gate (crypto trades 24/7). Removed tiny fixed TP/SL
    # (those were pip-scale for forex); use ATR-based exit instead.
    c = df["close"]
    o = df["open"]
    _, _, hist = macd(c, 12, 26, 9)
    up3 = (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2))
    cross_up = crossover(hist, pd.Series(0.0, index=c.index))
    entry = up3 & cross_up
    a = atr(df["high"], df["low"], c, 14)
    # exit when close drops below entry-relative trailing: use hist<0 or close < close.shift(1) - a
    ex = crossunder(hist, pd.Series(0.0, index=c.index))
    return entry.fillna(False), ex.fillna(False)


STRATEGIES = {
    "rsi2_connors": rsi2_connors,
    "rsi_oversold_rev": rsi_oversold_rev,
    "macd_ema200": macd_ema200,
    "ema9_21_cross": ema9_21_cross,
    "bb_meanrev": bb_meanrev,
    "supertrend_ema": supertrend_ema,
    "stoch_rev": stoch_rev,
    "macd_hull_session": macd_hull_session,
}
