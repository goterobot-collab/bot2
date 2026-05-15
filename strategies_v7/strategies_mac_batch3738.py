"""
Batch 3738 - Mac paralela wave m39 - 5 more calendar / event-driven variants.

Continuing calendar-event-fade family (8 CONFIRMED V10).

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3737.

 1. TV_PreEarnings_MidMonth   - day 12-18 of Jan/Apr/Jul/Oct (earnings season)
 2. TV_USHoliday_Weekend      - Memorial Day + Labor Day windows
 3. TV_Bitcoin_HalvingCycle_v2 - synthetic 4yr cycle (anchor approx Apr 19, 2024)
 4. TV_SantaRally_Window      - Dec 24 - Jan 2 (year-end Santa rally)
 5. TV_FirstWeek_NewYear      - Jan 1-7
"""
import numpy as np
import pandas as pd


# ----- 1) PRE-EARNINGS MID-MONTH -----
def gen_TV_PreEarnings_MidMonth(df, ret_len=14, **kw):
    """Day 12-18 of Jan/Apr/Jul/Oct UTC. Stock earnings season starts ~day 15
    of these months. Pre-earnings positioning spills to crypto.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_earnings_month = idx.month.isin([1, 4, 7, 10])
    is_mid_month = (idx.day >= 12) & (idx.day <= 18)
    pe = pd.Series(np.asarray(is_earnings_month & is_mid_month), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = pe & (ret < -0.03)
    short_setup = pe & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_PreEarnings_MidMonth():
    return {
        'ret_len': ('int', 4, 30),
    }


# ----- 2) US HOLIDAY WEEKEND (MEMORIAL DAY + LABOR DAY) -----
def gen_TV_USHoliday_Weekend(df, ret_len=12, **kw):
    """Memorial Day (last Mon May) and Labor Day (1st Mon Sep) extended
    weekends. May 25-31 OR Sep 1-7 UTC. Long weekend low-liq fades.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_memorial = (idx.month == 5) & (idx.day >= 25)
    is_labor = (idx.month == 9) & (idx.day <= 7)
    hw = pd.Series(np.asarray(is_memorial | is_labor), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = hw & (ret < -0.03)
    short_setup = hw & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_USHoliday_Weekend():
    return {
        'ret_len': ('int', 4, 30),
    }


# ----- 3) BITCOIN HALVING CYCLE V2 -----
def gen_TV_Bitcoin_HalvingCycle_v2(df, anchor_year=2024, anchor_month=4,
                                     anchor_day=19, window=14, ret_len=15, **kw):
    """Anchor BTC halving on approximate date (default 2024-04-19).
    Halving cycle = 1460 days (4 years). Within +/- window days of any
    halving anniversary -> fade.
    """
    w = int(window)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    anchor = pd.Timestamp(f"{int(anchor_year)}-{int(anchor_month):02d}-{int(anchor_day):02d}",
                            tz='UTC')
    days_since_anchor = (idx - anchor).days
    phase = days_since_anchor % 1460
    near_halving = (phase < w) | (phase > (1460 - w))
    near = pd.Series(np.asarray(near_halving), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = near & (ret < -0.04)
    short_setup = near & (ret > 0.04)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Bitcoin_HalvingCycle_v2():
    return {
        'anchor_year': ('int', 2020, 2024),
        'anchor_month': ('int', 4, 5),
        'anchor_day': ('int', 11, 25),
        'window': ('int', 7, 30),
        'ret_len': ('int', 5, 30),
    }


# ----- 4) SANTA RALLY WINDOW -----
def gen_TV_SantaRally_Window(df, ret_len=8, **kw):
    """Dec 24 - Jan 2 = traditional "Santa Rally" period.
    Bullish bias historically. Continue if up, only short if very down.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_late_dec = (idx.month == 12) & (idx.day >= 24)
    is_early_jan = (idx.month == 1) & (idx.day <= 2)
    sr = pd.Series(np.asarray(is_late_dec | is_early_jan), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    # Continuation bias (Santa rally bullish)
    long_setup = sr & (ret > 0.015)
    short_setup = sr & (ret < -0.05)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_SantaRally_Window():
    return {
        'ret_len': ('int', 3, 25),
    }


# ----- 5) FIRST WEEK NEW YEAR -----
def gen_TV_FirstWeek_NewYear(df, ret_len=14, **kw):
    """Jan 1-7 UTC. New-year positioning + fresh capital deployment.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_fw = (idx.month == 1) & (idx.day <= 7)
    fw = pd.Series(np.asarray(is_fw), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = fw & (ret < -0.03)
    short_setup = fw & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_FirstWeek_NewYear():
    return {
        'ret_len': ('int', 4, 30),
    }


STRATEGY_EXPORT = {
    "TV_PreEarnings_MidMonth": {
        "gen": gen_TV_PreEarnings_MidMonth, "space": space_TV_PreEarnings_MidMonth,
        "source": "mac_batch3738_pre_earnings_mid_month_4q",
    },
    "TV_USHoliday_Weekend": {
        "gen": gen_TV_USHoliday_Weekend, "space": space_TV_USHoliday_Weekend,
        "source": "mac_batch3738_memorial_labor_extended_weekend",
    },
    "TV_Bitcoin_HalvingCycle_v2": {
        "gen": gen_TV_Bitcoin_HalvingCycle_v2, "space": space_TV_Bitcoin_HalvingCycle_v2,
        "source": "mac_batch3738_BTC_halving_anchor_v2",
    },
    "TV_SantaRally_Window": {
        "gen": gen_TV_SantaRally_Window, "space": space_TV_SantaRally_Window,
        "source": "mac_batch3738_santa_rally_continuation",
    },
    "TV_FirstWeek_NewYear": {
        "gen": gen_TV_FirstWeek_NewYear, "space": space_TV_FirstWeek_NewYear,
        "source": "mac_batch3738_first_week_new_year",
    },
}
