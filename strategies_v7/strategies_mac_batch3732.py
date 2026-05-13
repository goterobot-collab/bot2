"""
Batch 3732 - Mac paralela wave m33 - 5 calendar combos / specific-date.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3731.

 1. TV_NewYear_Window         - last days Dec + first days Jan
 2. TV_DayOfMonth_Specific    - any specific day-of-month fade (1, 5, 15, 25)
 3. TV_LunarPhaseRefined      - lunar phase precise window
 4. TV_BarOfDay_24h           - specific bar-of-day index (0-23) on 1h
 5. TV_MonthEnd_HourFilter    - month-end + specific hour combo
"""
import numpy as np
import pandas as pd


# ----- 1) NEW YEAR WINDOW -----
def gen_TV_NewYear_Window(df, days_dec=5, days_jan=5, ret_len=14, **kw):
    """Last `days_dec` of Dec + first `days_jan` of Jan UTC. Fade recent return.
    Year-end portfolio rebalancing window.
    """
    dd = int(days_dec)
    dj = int(days_jan)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_end_dec = (idx.month == 12) & ((idx.days_in_month - idx.day) < dd)
    is_start_jan = (idx.month == 1) & (idx.day <= dj)
    nyw = pd.Series(np.asarray(is_end_dec | is_start_jan), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = nyw & (ret < -0.04)
    short_setup = nyw & (ret > 0.04)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_NewYear_Window():
    return {
        'days_dec': ('int', 2, 10),
        'days_jan': ('int', 2, 10),
        'ret_len': ('int', 5, 40),
    }


# ----- 2) DAY OF MONTH SPECIFIC -----
def gen_TV_DayOfMonth_Specific(df, target_day=15, window_radius=1, ret_len=10, **kw):
    """Specific day of month UTC +/- window_radius. Fade recent return.
    """
    td = int(target_day)
    wr = int(window_radius)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_target = (idx.day >= (td - wr)) & (idx.day <= (td + wr))
    target = pd.Series(np.asarray(is_target), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = target & (ret < -0.03)
    short_setup = target & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_DayOfMonth_Specific():
    return {
        'target_day': ('int', 1, 28),
        'window_radius': ('int', 0, 3),
        'ret_len': ('int', 4, 30),
    }


# ----- 3) LUNAR PHASE REFINED -----
def gen_TV_LunarPhaseRefined(df, phase_target=14.77, window=2.0, ret_len=14, **kw):
    """Lunar cycle: phase_target ~14.77 = full moon, ~0/29.53 = new moon.
    Tight window around target phase.
    """
    pt = float(phase_target)
    w = float(window)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    day_of_year = np.asarray(idx.dayofyear, dtype=float)
    phase = day_of_year % 29.53
    near_phase = np.abs(phase - pt) < w
    window_s = pd.Series(near_phase, index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = window_s & (ret < -0.03)
    short_setup = window_s & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_LunarPhaseRefined():
    return {
        'phase_target': ('float', 0.0, 29.0),
        'window': ('float', 0.5, 4.0),
        'ret_len': ('int', 5, 30),
    }


# ----- 4) BAR OF DAY 24H -----
def gen_TV_BarOfDay_24h(df, target_hour=9, ret_len=8, fade=1, **kw):
    """Bar-of-day for 1h TF: target_hour UTC (0-23). For 4h TF, only fires
    at the appropriate aggregated bar boundaries.
    """
    th = int(target_hour)
    rl = int(ret_len)
    f = int(fade)
    c = df['close'].astype(float)
    idx = df.index
    is_target = pd.Series(np.asarray(idx.hour == th), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    if f == 1:
        long_setup = is_target & (ret < -0.02)
        short_setup = is_target & (ret > 0.02)
    else:
        long_setup = is_target & (ret > 0.02)
        short_setup = is_target & (ret < -0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_BarOfDay_24h():
    return {
        'target_hour': ('int', 0, 23),
        'ret_len': ('int', 3, 24),
        'fade': ('int', 0, 1),
    }


# ----- 5) MONTH-END + HOUR FILTER -----
def gen_TV_MonthEnd_HourFilter(df, days_window=2, target_hour=23, ret_len=8, **kw):
    """Last `days_window` days of any month AND specific hour UTC.
    Concentrated fade at end-of-month closing hour.
    """
    dw = int(days_window)
    th = int(target_hour)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_me = (idx.days_in_month - idx.day) < dw
    is_hr = idx.hour == th
    window = pd.Series(np.asarray(is_me & is_hr), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = window & (ret < -0.02)
    short_setup = window & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_MonthEnd_HourFilter():
    return {
        'days_window': ('int', 1, 5),
        'target_hour': ('int', 0, 23),
        'ret_len': ('int', 4, 25),
    }


STRATEGY_EXPORT = {
    "TV_NewYear_Window": {
        "gen": gen_TV_NewYear_Window, "space": space_TV_NewYear_Window,
        "source": "mac_batch3732_new_year_rebalance_window",
    },
    "TV_DayOfMonth_Specific": {
        "gen": gen_TV_DayOfMonth_Specific, "space": space_TV_DayOfMonth_Specific,
        "source": "mac_batch3732_specific_day_of_month_fade",
    },
    "TV_LunarPhaseRefined": {
        "gen": gen_TV_LunarPhaseRefined, "space": space_TV_LunarPhaseRefined,
        "source": "mac_batch3732_lunar_phase_precise_window",
    },
    "TV_BarOfDay_24h": {
        "gen": gen_TV_BarOfDay_24h, "space": space_TV_BarOfDay_24h,
        "source": "mac_batch3732_specific_hour_UTC",
    },
    "TV_MonthEnd_HourFilter": {
        "gen": gen_TV_MonthEnd_HourFilter, "space": space_TV_MonthEnd_HourFilter,
        "source": "mac_batch3732_ME_specific_hour_combo",
    },
}
