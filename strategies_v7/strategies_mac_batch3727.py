"""
Batch 3727 - Mac paralela wave m28 - 5 CALENDAR variants (QuarterEnd worked!).

Doubling down on calendar-based patterns after wave m26 yielded 2 CONFIRMED
V10 grails (TV_Calendar_QuarterEnd OP/LINK 1h). Explore other calendar
flow asymmetries.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox (Weekend_Gap_Fade/IntradaySeasonality)
or Mac V8 (B4_Day_Of_Week/B6_Midnight_Rev/B6_Weekend_Effect) or Mac
3700-3726 (QuarterEnd is distinct).

 1. TV_Calendar_MonthEnd       - last N days of ANY month UTC fade
 2. TV_Calendar_HourOfDay      - specific hour-of-day fade (intra-bar pattern)
 3. TV_Calendar_FirstWeek      - first 7 days of new month fade
 4. TV_Calendar_LunarPhase     - synthetic lunar cycle (Renaissance Tech-style
                                  unconventional signal; day-of-year mod 29.5)
 5. TV_Calendar_DOW_Specific   - specific day-of-week (e.g. Friday close fade,
                                  Monday morning continuation)
"""
import numpy as np
import pandas as pd


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


# ----- 1) MONTH-END FADE -----
def gen_TV_Calendar_MonthEnd(df, days_window=3, ret_len=10, **kw):
    """End-of-month: last `days_window` days of ANY month UTC.
    Fade recent N-day return.
    """
    dw = int(days_window)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    days_in_month = idx.days_in_month
    is_last_days = (days_in_month - idx.day) < dw
    monthend = pd.Series(is_last_days, index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = monthend & (ret < -0.03)
    short_setup = monthend & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Calendar_MonthEnd():
    return {
        'days_window': ('int', 1, 7),
        'ret_len': ('int', 3, 30),
    }


# ----- 2) HOUR-OF-DAY FADE -----
def gen_TV_Calendar_HourOfDay(df, target_hour=0, ret_len=6, **kw):
    """Specific hour-of-day UTC fade. Common sessions:
    00 UTC = Asia open / EU pre-open
    08 UTC = EU open
    16 UTC = US close approaching
    Fade recent N-bar return at target hour.
    """
    th = int(target_hour)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_target = pd.Series(idx.hour == th, index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = is_target & (ret < -0.02)
    short_setup = is_target & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Calendar_HourOfDay():
    return {
        'target_hour': ('int', 0, 23),
        'ret_len': ('int', 3, 24),
    }


# ----- 3) FIRST WEEK CONTINUATION -----
def gen_TV_Calendar_FirstWeek(df, days=7, ret_len=10, **kw):
    """First N days of new month: follow direction of last month's close
    (continuation of momentum from prior month).
    """
    dd = int(days)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_first = pd.Series(idx.day <= dd, index=idx)
    ret = (c / c.shift(rl) - 1.0)
    # Continue direction
    long_setup = is_first & (ret > 0.02)
    short_setup = is_first & (ret < -0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Calendar_FirstWeek():
    return {
        'days': ('int', 3, 14),
        'ret_len': ('int', 5, 30),
    }


# ----- 4) LUNAR PHASE -----
def gen_TV_Calendar_LunarPhase(df, ret_len=15, phase_window=5, **kw):
    """Synthetic lunar phase: day-of-year mod 29.5 (full cycle).
    Near full moon (phase ~14-15) OR new moon (phase ~0 or 29) = pivot windows.
    Per Renaissance Technologies-style "unconventional signal".
    Fade recent return when within phase_window of full moon or new moon.
    """
    rl = int(ret_len)
    pw = float(phase_window)
    c = df['close'].astype(float)
    idx = df.index
    # Day of year as numpy float array
    day_of_year = np.asarray(idx.dayofyear, dtype=float)
    lunar_cycle = day_of_year % 29.53
    near_full = np.abs(lunar_cycle - 14.77) < pw
    near_new = (lunar_cycle < pw) | (lunar_cycle > (29.53 - pw))
    pivot_window = pd.Series(near_full | near_new, index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = pivot_window & (ret < -0.03)
    short_setup = pivot_window & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Calendar_LunarPhase():
    return {
        'ret_len': ('int', 5, 40),
        'phase_window': ('float', 1.0, 6.0),
    }


# ----- 5) DAY-OF-WEEK SPECIFIC -----
def gen_TV_Calendar_DOW_Specific(df, target_dow=4, ret_len=10, fade=1, **kw):
    """Specific day-of-week (UTC). Monday=0, Sunday=6.
    fade=1: fade recent return on target day (institutional close).
    fade=0: continue recent direction (momentum day).
    """
    td = int(target_dow)
    rl = int(ret_len)
    f = int(fade)
    c = df['close'].astype(float)
    idx = df.index
    is_target = pd.Series(idx.dayofweek == td, index=idx)
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


def space_TV_Calendar_DOW_Specific():
    return {
        'target_dow': ('int', 0, 6),
        'ret_len': ('int', 3, 30),
        'fade': ('int', 0, 1),
    }


STRATEGY_EXPORT = {
    "TV_Calendar_MonthEnd": {
        "gen": gen_TV_Calendar_MonthEnd, "space": space_TV_Calendar_MonthEnd,
        "source": "mac_batch3727_month_end_fade",
    },
    "TV_Calendar_HourOfDay": {
        "gen": gen_TV_Calendar_HourOfDay, "space": space_TV_Calendar_HourOfDay,
        "source": "mac_batch3727_hour_of_day_session_fade",
    },
    "TV_Calendar_FirstWeek": {
        "gen": gen_TV_Calendar_FirstWeek, "space": space_TV_Calendar_FirstWeek,
        "source": "mac_batch3727_first_week_continuation",
    },
    "TV_Calendar_LunarPhase": {
        "gen": gen_TV_Calendar_LunarPhase, "space": space_TV_Calendar_LunarPhase,
        "source": "mac_batch3727_lunar_cycle_pivot_unconventional",
    },
    "TV_Calendar_DOW_Specific": {
        "gen": gen_TV_Calendar_DOW_Specific, "space": space_TV_Calendar_DOW_Specific,
        "source": "mac_batch3727_day_of_week_specific_fade_or_follow",
    },
}
