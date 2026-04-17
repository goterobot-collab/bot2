"""
Strategy library — long-only signal generators with optional hard SL, hard TP,
and/or ATR trailing stop. Each entry in STRATEGIES has:

    {
      "fn": callable(df) -> (entry: Series[bool], exit: Series[bool]),
      "sl_atr":     float | None,  # hard SL = entry_price - N*ATR(14)
      "tp_atr":     float | None,  # hard TP = entry_price + N*ATR(14)
      "trail_atr":  float | None,  # trailing stop = max(stop, high - N*ATR(14))
      "timeout":    int  | None,   # force-exit after N bars
    }

Strategies are adapted from public Pine Script sources; original Pine URLs
(when public) are listed in SOURCES and archived in pine_sources/.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .indicators import (
    ema, sma, rsi, macd, bbands, stoch, supertrend, atr,
    crossover, crossunder,
)


SOURCES = {
    "rsi2_connors": {
        "name": "RSI(2) mean-reversion + SMA200 trend filter",
        "author": "Larry Connors (public methodology)",
        "pine_ref": "pine_sources/01_rsi2_connors.pine",
        "logic": "Long when close > SMA200 and RSI(2) < 10; exit when close > SMA(5).",
    },
    "rsi_oversold_rev": {
        "name": "RSI(14) oversold reversal",
        "author": "Generic mean-reversion (public)",
        "pine_ref": "pine_sources/02_rsi_oversold_rev.pine",
        "logic": "Long when RSI(14) crosses above 30; exit when RSI(14) crosses below 70.",
    },
    "macd_ema200": {
        "name": "MACD cross + EMA200 filter",
        "author": "Generic (public)",
        "pine_ref": "pine_sources/03_macd_ema200.pine",
        "logic": "Long when MACD crosses above signal and close > EMA200; exit on MACD<signal.",
    },
    "ema9_21_cross": {
        "name": "EMA 9/21 crossover + EMA200 trend filter",
        "author": "Generic (public)",
        "pine_ref": "pine_sources/04_ema9_21_cross.pine",
        "logic": "Long when EMA9 crosses above EMA21 and close > EMA200; exit on reverse cross.",
    },
    "bb_meanrev": {
        "name": "Bollinger Band mean reversion",
        "author": "Generic (public)",
        "pine_ref": "pine_sources/05_bb_meanrev.pine",
        "logic": "Long when close crosses above lower BB(20,2); exit when close crosses mid band.",
    },
    "supertrend_ema": {
        "name": "Supertrend(10,3) + EMA200 filter",
        "author": "Generic Pine community",
        "pine_ref": "pine_sources/06_supertrend_ema.pine",
        "logic": "Long on Supertrend flip up while close > EMA200; exit on flip down.",
    },
    "stoch_rev": {
        "name": "Stochastic(14,3) oversold cross",
        "author": "Generic (public)",
        "pine_ref": "pine_sources/07_stoch_rev.pine",
        "logic": "Long when %K crosses above %D below 20; exit when %K crosses below %D above 80.",
    },
    "macd_hull_session": {
        "name": "MACD hist + 3 up candles (exlux99 adapted)",
        "author": "exlux99 (Pine v4, MPL-2.0)",
        "pine_ref": "pine_sources/08_macd_3up_candles.pine",
        "logic": "Long when 3 consecutive up candles + MACD hist crosses 0; ATR SL/TP.",
    },
    # --- grail-aim: regime filter + hard SL/TP + trailing stop -----------------
    "rsi2_regime_atr": {
        "name": "RSI(2) + rising-SMA200 regime + hard ATR SL (grail-aim)",
        "author": "Connors-style with regime filter",
        "pine_ref": "pine_sources/09_rsi2_regime_atrstop.pine",
        "logic": "SMA200 slope>0 AND close>SMA200 AND RSI(2)<10 → long. Exit: close>SMA5 OR 2×ATR SL OR 24-bar timeout.",
    },
    "holy_grail_raschke": {
        "name": "Linda Raschke Holy Grail (ADX>30 + EMA20 pullback)",
        "author": "Linda Raschke (public)",
        "pine_ref": "pine_sources/10_holy_grail.pine",
        "logic": "ADX(14)>30 AND DI+>DI- AND close pulled back ≤ EMA20 → long on recovery. 2×ATR TP, 1×ATR SL.",
    },
    "donchian_atr_trail": {
        "name": "Donchian(20) breakout + 3×ATR trailing stop",
        "author": "Turtle-style trend (public)",
        "pine_ref": "pine_sources/11_donchian_atr_trail.pine",
        "logic": "Long on close > Donchian upper(20). Trailing stop = 3×ATR(14). No TP; let it trail.",
    },
    "supertrend_atr_grail": {
        "name": "Supertrend flip + EMA200 + 2×ATR SL + 4×ATR TP",
        "author": "VolStop-style (public)",
        "pine_ref": "pine_sources/12_supertrend_atr_stop.pine",
        "logic": "Supertrend flip up AND close>EMA200 → long. Hard 2×ATR SL, hard 4×ATR TP (R:R=2).",
    },
    "rsi_pullback_trend": {
        "name": "RSI pullback in uptrend + 1.5×ATR SL",
        "author": "Generic trend-pullback (public)",
        "pine_ref": "pine_sources/13_rsi_pullback_trend.pine",
        "logic": "EMA50>EMA200 AND EMA200 slope>0 AND RSI(14) crosses above 40. Exit: RSI>70 OR 1.5×ATR SL.",
    },
}


def _empty(df):
    s = pd.Series(False, index=df.index)
    return s.copy(), s.copy()


# ---------- original 8 (no SL/TP) ----------
def rsi2_connors(df):
    c = df["close"]
    entry = (c > sma(c, 200)) & (rsi(c, 2) < 10)
    ex = c > sma(c, 5)
    return entry.fillna(False), ex.fillna(False)


def rsi_oversold_rev(df):
    c = df["close"]
    r = rsi(c, 14)
    entry = crossover(r, pd.Series(30.0, index=c.index))
    ex = crossunder(r, pd.Series(70.0, index=c.index))
    return entry.fillna(False), ex.fillna(False)


def macd_ema200(df):
    c = df["close"]
    m, s, _ = macd(c, 12, 26, 9)
    entry = crossover(m, s) & (c > ema(c, 200))
    ex = crossunder(m, s)
    return entry.fillna(False), ex.fillna(False)


def ema9_21_cross(df):
    c = df["close"]
    e9 = ema(c, 9); e21 = ema(c, 21); et = ema(c, 200)
    entry = crossover(e9, e21) & (c > et)
    ex = crossunder(e9, e21)
    return entry.fillna(False), ex.fillna(False)


def bb_meanrev(df):
    c = df["close"]
    lo, mid, _ = bbands(c, 20, 2.0)
    entry = crossover(c, lo)
    ex = crossover(c, mid)
    return entry.fillna(False), ex.fillna(False)


def supertrend_ema(df):
    _, dirn = supertrend(df["high"], df["low"], df["close"], 10, 3.0)
    trend = ema(df["close"], 200)
    flip_up = (dirn == 1) & (dirn.shift(1) == -1)
    flip_dn = (dirn == -1) & (dirn.shift(1) == 1)
    entry = flip_up & (df["close"] > trend)
    return entry.fillna(False), flip_dn.fillna(False)


def stoch_rev(df):
    k, d = stoch(df["high"], df["low"], df["close"], 14, 3)
    entry = crossover(k, d) & (k < 20)
    ex = crossunder(k, d) & (k > 80)
    return entry.fillna(False), ex.fillna(False)


def macd_hull_session(df):
    c = df["close"]; o = df["open"]
    _, _, hist = macd(c, 12, 26, 9)
    up3 = (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2))
    entry = up3 & crossover(hist, pd.Series(0.0, index=c.index))
    ex = crossunder(hist, pd.Series(0.0, index=c.index))
    return entry.fillna(False), ex.fillna(False)


# ---------- grail-aim 5 (SL/TP/trail applied in simulate) ----------
def rsi2_regime_atr(df):
    c = df["close"]
    s200 = sma(c, 200)
    # SMA200 rising: higher than 20 bars ago
    rising = s200 > s200.shift(20)
    r2 = rsi(c, 2)
    entry = rising & (c > s200) & (r2 < 10)
    ex = c > sma(c, 5)
    return entry.fillna(False), ex.fillna(False)


def _adx(high, low, close, n=14):
    up = high.diff()
    dn = -low.diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_dm  = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    a = atr(high, low, close, n)
    plus_di  = 100 * plus_dm.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()  / a
    minus_di = 100 * minus_dm.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()
    return adx, plus_di, minus_di


def holy_grail_raschke(df):
    c = df["close"]; h = df["high"]; l = df["low"]
    adx, pdi, mdi = _adx(h, l, c, 14)
    ema20 = ema(c, 20)
    # setup = trend confirmed, and low dipped to/below ema20 within last 5 bars
    trend_ok = (adx > 30) & (pdi > mdi)
    touched = (l <= ema20).rolling(5, min_periods=1).max().astype(bool)
    recovery = c > c.shift(1)  # up bar after pullback
    entry = trend_ok & touched & recovery & (c > ema20)
    ex = crossunder(c, ema20)  # time-based exits mainly via SL/TP in simulate
    return entry.fillna(False), ex.fillna(False)


def donchian_atr_trail(df):
    c = df["close"]; h = df["high"]; l = df["low"]
    upper = h.rolling(20, min_periods=20).max().shift(1)  # prior-20 high
    entry = c > upper
    # exit handled by trailing stop; signal "exit" stays False
    ex = pd.Series(False, index=c.index)
    return entry.fillna(False), ex


def supertrend_atr_grail(df):
    _, dirn = supertrend(df["high"], df["low"], df["close"], 10, 3.0)
    trend = ema(df["close"], 200)
    flip_up = (dirn == 1) & (dirn.shift(1) == -1)
    flip_dn = (dirn == -1) & (dirn.shift(1) == 1)
    entry = flip_up & (df["close"] > trend)
    return entry.fillna(False), flip_dn.fillna(False)


def rsi_pullback_trend(df):
    c = df["close"]
    e50 = ema(c, 50); e200 = ema(c, 200)
    rising = e200 > e200.shift(20)
    r = rsi(c, 14)
    entry = (e50 > e200) & rising & crossover(r, pd.Series(40.0, index=c.index))
    ex = crossunder(r, pd.Series(70.0, index=c.index))
    return entry.fillna(False), ex.fillna(False)


STRATEGIES = {
    "rsi2_connors":        {"fn": rsi2_connors,       "sl_atr": None, "tp_atr": None, "trail_atr": None, "timeout": None},
    "rsi_oversold_rev":    {"fn": rsi_oversold_rev,   "sl_atr": None, "tp_atr": None, "trail_atr": None, "timeout": None},
    "macd_ema200":         {"fn": macd_ema200,        "sl_atr": None, "tp_atr": None, "trail_atr": None, "timeout": None},
    "ema9_21_cross":       {"fn": ema9_21_cross,      "sl_atr": None, "tp_atr": None, "trail_atr": None, "timeout": None},
    "bb_meanrev":          {"fn": bb_meanrev,         "sl_atr": None, "tp_atr": None, "trail_atr": None, "timeout": None},
    "supertrend_ema":      {"fn": supertrend_ema,     "sl_atr": None, "tp_atr": None, "trail_atr": None, "timeout": None},
    "stoch_rev":           {"fn": stoch_rev,          "sl_atr": None, "tp_atr": None, "trail_atr": None, "timeout": None},
    "macd_hull_session":   {"fn": macd_hull_session,  "sl_atr": 2.0,  "tp_atr": 3.0,  "trail_atr": None, "timeout": None},
    # grail-aim
    "rsi2_regime_atr":     {"fn": rsi2_regime_atr,    "sl_atr": 2.0,  "tp_atr": None, "trail_atr": None, "timeout": 24},
    "holy_grail_raschke":  {"fn": holy_grail_raschke, "sl_atr": 1.0,  "tp_atr": 2.0,  "trail_atr": None, "timeout": 48},
    "donchian_atr_trail":  {"fn": donchian_atr_trail, "sl_atr": None, "tp_atr": None, "trail_atr": 3.0,  "timeout": None},
    "supertrend_atr_grail":{"fn": supertrend_atr_grail,"sl_atr": 2.0, "tp_atr": 4.0,  "trail_atr": None, "timeout": None},
    "rsi_pullback_trend":  {"fn": rsi_pullback_trend, "sl_atr": 1.5,  "tp_atr": None, "trail_atr": 2.5,  "timeout": None},
}
