"""
Batch 3733 - Mac paralela wave m34 - 5 MORE calendar-event-fade variants (3x down).

Tripling down on calendar-event-fade after 4 CONFIRMED V10 in this family.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3732.

 1. TV_PreFOMC_Week           - 2 weeks before estimated FOMC (8/yr)
 2. TV_Quarter_FirstWeek_Fade - first week of new quarter FADE (opposite
                                of m31 QuarterStart_Cont)
 3. TV_BiMonthly_DaySpecific  - day-1 OR day-15 of any month
 4. TV_LastWeek_AnyFriday     - any Friday in last 7 days of month
 5. TV_2ndFriday              - 2nd Friday of any month (vs 3rd Fri = OpEx)
"""
import numpy as np
import pandas as pd


# ----- 1) PRE-FOMC WEEK -----
def gen_TV_PreFOMC_Week(df, ret_len=20, **kw):
    """Approximate FOMC meeting weeks (8/yr): mid-late month of
    Jan/Mar/Apr/Jun/Jul/Sep/Oct/Dec. Pre-FOMC = 2 weeks before, so day 1-14
    of those months. Captures pre-FOMC risk-off / risk-on positioning.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    fomc_months = [1, 3, 4, 6, 7, 9, 10, 12]
    is_fomc_month = idx.month.isin(fomc_months)
    is_first_half = idx.day <= 14
    pre_fomc = pd.Series(np.asarray(is_fomc_month & is_first_half), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = pre_fomc & (ret < -0.04)
    short_setup = pre_fomc & (ret > 0.04)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_PreFOMC_Week():
    return {
        'ret_len': ('int', 8, 40),
    }


# ----- 2) QUARTER FIRST-WEEK FADE -----
def gen_TV_Quarter_FirstWeek_Fade(df, days=5, ret_len=15, **kw):
    """First N days of Apr/Jul/Oct/Jan = new quarter. FADE direction
    (opposite of m31 QuarterStart_Cont which is continuation).
    """
    dd = int(days)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_qs_month = idx.month.isin([4, 7, 10, 1])
    is_first = idx.day <= dd
    qstart = pd.Series(np.asarray(is_qs_month & is_first), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = qstart & (ret < -0.03)
    short_setup = qstart & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Quarter_FirstWeek_Fade():
    return {
        'days': ('int', 2, 14),
        'ret_len': ('int', 4, 30),
    }


# ----- 3) BI-MONTHLY DAY SPECIFIC -----
def gen_TV_BiMonthly_DaySpecific(df, window=1, ret_len=10, **kw):
    """Day-1 OR day-15 of any month (+/- window). Captures bi-monthly
    institutional flow.
    """
    w = int(window)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    near_1 = idx.day <= (1 + w)
    near_15 = (idx.day >= (15 - w)) & (idx.day <= (15 + w))
    bm = pd.Series(np.asarray(near_1 | near_15), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = bm & (ret < -0.03)
    short_setup = bm & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_BiMonthly_DaySpecific():
    return {
        'window': ('int', 0, 3),
        'ret_len': ('int', 4, 25),
    }


# ----- 4) LAST WEEK ANY FRIDAY -----
def gen_TV_LastWeek_AnyFriday(df, ret_len=12, **kw):
    """Any Friday UTC in last 7 days of any month. Captures end-of-month
    Friday liquidity (close-of-week + close-of-month combined).
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_friday = idx.dayofweek == 4
    is_last_week = (idx.days_in_month - idx.day) < 7
    lwf = pd.Series(np.asarray(is_friday & is_last_week), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = lwf & (ret < -0.025)
    short_setup = lwf & (ret > 0.025)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_LastWeek_AnyFriday():
    return {
        'ret_len': ('int', 4, 30),
    }


# ----- 5) 2ND FRIDAY OF MONTH -----
def gen_TV_2ndFriday(df, ret_len=14, **kw):
    """2nd Friday of any month UTC (day 8-14, dayofweek=Friday).
    Distinct from 3rd Friday (OpEx) - captures pre-OpEx positioning.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_friday = idx.dayofweek == 4
    is_2nd_week = (idx.day >= 8) & (idx.day <= 14)
    f2 = pd.Series(np.asarray(is_friday & is_2nd_week), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = f2 & (ret < -0.025)
    short_setup = f2 & (ret > 0.025)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_2ndFriday():
    return {
        'ret_len': ('int', 4, 30),
    }


STRATEGY_EXPORT = {
    "TV_PreFOMC_Week": {
        "gen": gen_TV_PreFOMC_Week, "space": space_TV_PreFOMC_Week,
        "source": "mac_batch3733_pre_FOMC_meeting_week",
    },
    "TV_Quarter_FirstWeek_Fade": {
        "gen": gen_TV_Quarter_FirstWeek_Fade, "space": space_TV_Quarter_FirstWeek_Fade,
        "source": "mac_batch3733_QS_first_week_fade",
    },
    "TV_BiMonthly_DaySpecific": {
        "gen": gen_TV_BiMonthly_DaySpecific, "space": space_TV_BiMonthly_DaySpecific,
        "source": "mac_batch3733_bi_monthly_1_15",
    },
    "TV_LastWeek_AnyFriday": {
        "gen": gen_TV_LastWeek_AnyFriday, "space": space_TV_LastWeek_AnyFriday,
        "source": "mac_batch3733_last_week_friday",
    },
    "TV_2ndFriday": {
        "gen": gen_TV_2ndFriday, "space": space_TV_2ndFriday,
        "source": "mac_batch3733_2nd_friday_pre_OpEx",
    },
}
