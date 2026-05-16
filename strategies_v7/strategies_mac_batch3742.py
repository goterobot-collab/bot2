"""
Batch 3742 - Mac paralela wave m43 - 5 ULTRA-narrow specific-date anchors.

Pushing narrow-window strategy to single-day anchors.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3741.

 1. TV_Dec31_Only          - Dec 31 EXACT (year-end)
 2. TV_Jan1_Only           - Jan 1 EXACT (year-start)
 3. TV_Mar31_Only          - Mar 31 EXACT (Q1 end)
 4. TV_Sep30_Only          - Sep 30 EXACT (US fiscal + Q3 end)
 5. TV_3rdFriday_Narrow    - 3rd Friday exact (day 15-21 + Fri) NO hour filter
"""
import numpy as np
import pandas as pd


def _date_only_fade(df, month, day_lo, day_hi, ret_len):
    c = df['close'].astype(float)
    idx = df.index
    is_match = (idx.month == month) & (idx.day >= day_lo) & (idx.day <= day_hi)
    win = pd.Series(np.asarray(is_match), index=idx)
    ret = (c / c.shift(int(ret_len)) - 1.0)
    long_setup = win & (ret < -0.02)
    short_setup = win & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


# ----- 1) DEC 31 ONLY -----
def gen_TV_Dec31_Only(df, ret_len=10, **kw):
    return _date_only_fade(df, 12, 31, 31, ret_len)


def space_TV_Dec31_Only():
    return {'ret_len': ('int', 3, 30)}


# ----- 2) JAN 1 ONLY -----
def gen_TV_Jan1_Only(df, ret_len=10, **kw):
    return _date_only_fade(df, 1, 1, 1, ret_len)


def space_TV_Jan1_Only():
    return {'ret_len': ('int', 3, 30)}


# ----- 3) MAR 31 ONLY -----
def gen_TV_Mar31_Only(df, ret_len=10, **kw):
    return _date_only_fade(df, 3, 31, 31, ret_len)


def space_TV_Mar31_Only():
    return {'ret_len': ('int', 3, 30)}


# ----- 4) SEP 30 ONLY -----
def gen_TV_Sep30_Only(df, ret_len=10, **kw):
    return _date_only_fade(df, 9, 30, 30, ret_len)


def space_TV_Sep30_Only():
    return {'ret_len': ('int', 3, 30)}


# ----- 5) 3RD FRIDAY NARROW -----
def gen_TV_3rdFriday_Narrow(df, ret_len=12, **kw):
    """3rd Friday exact (day 15-21 + dayofweek=Friday) - same as OpEx but
    without ret threshold filter (looser entry criteria).
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_friday = idx.dayofweek == 4
    is_3rd_week = (idx.day >= 15) & (idx.day <= 21)
    fr3 = pd.Series(np.asarray(is_friday & is_3rd_week), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = fr3 & (ret < -0.015)
    short_setup = fr3 & (ret > 0.015)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_3rdFriday_Narrow():
    return {'ret_len': ('int', 3, 30)}


STRATEGY_EXPORT = {
    "TV_Dec31_Only": {"gen": gen_TV_Dec31_Only, "space": space_TV_Dec31_Only,
                       "source": "mac_batch3742_dec31_exact"},
    "TV_Jan1_Only": {"gen": gen_TV_Jan1_Only, "space": space_TV_Jan1_Only,
                      "source": "mac_batch3742_jan1_exact"},
    "TV_Mar31_Only": {"gen": gen_TV_Mar31_Only, "space": space_TV_Mar31_Only,
                       "source": "mac_batch3742_mar31_exact"},
    "TV_Sep30_Only": {"gen": gen_TV_Sep30_Only, "space": space_TV_Sep30_Only,
                       "source": "mac_batch3742_sep30_exact"},
    "TV_3rdFriday_Narrow": {"gen": gen_TV_3rdFriday_Narrow, "space": space_TV_3rdFriday_Narrow,
                             "source": "mac_batch3742_3rdFriday_loose"},
}
