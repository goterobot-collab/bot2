"""
Batch 3736 - Mac paralela wave m37 - 5 calendar boundary / week-pattern fade.

Continuing calendar-event-fade family (6 CONFIRMED V10).

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3735.

 1. TV_MonthBoundary_Span    - last 2 + first 2 days of month combined
 2. TV_FirstMonday_Month     - 1st Monday of month
 3. TV_MidWeek_Wednesday     - Wednesday pivot
 4. TV_WeekClose_Sunday      - Sunday UTC (crypto week-close) fade
 5. TV_SettlementWindow_T2   - Tue/Wed (T+2 settlement proxy)
"""
import numpy as np
import pandas as pd


# ----- 1) MONTH BOUNDARY SPAN -----
def gen_TV_MonthBoundary_Span(df, span=2, ret_len=12, **kw):
    """Last `span` days of month + first `span` days of next month.
    Captures month-boundary institutional rebalancing flow.
    """
    s = int(span)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_month_end = (idx.days_in_month - idx.day) < s
    is_month_start = idx.day <= s
    boundary = pd.Series(np.asarray(is_month_end | is_month_start), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = boundary & (ret < -0.03)
    short_setup = boundary & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_MonthBoundary_Span():
    return {
        'span': ('int', 1, 5),
        'ret_len': ('int', 4, 30),
    }


# ----- 2) FIRST MONDAY OF MONTH -----
def gen_TV_FirstMonday_Month(df, ret_len=12, **kw):
    """1st Monday of month UTC (day 1-7, dayofweek=Monday).
    Week+month start positioning.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_monday = idx.dayofweek == 0
    is_1st_week = idx.day <= 7
    fm = pd.Series(np.asarray(is_monday & is_1st_week), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = fm & (ret < -0.03)
    short_setup = fm & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_FirstMonday_Month():
    return {
        'ret_len': ('int', 4, 30),
    }


# ----- 3) MID-WEEK WEDNESDAY PIVOT -----
def gen_TV_MidWeek_Wednesday(df, ret_len=8, **kw):
    """Wednesday UTC = mid-week. Often reversal/pivot point as
    early-week trend exhausts.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_wed = pd.Series(np.asarray(idx.dayofweek == 2), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = is_wed & (ret < -0.025)
    short_setup = is_wed & (ret > 0.025)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_MidWeek_Wednesday():
    return {
        'ret_len': ('int', 3, 24),
    }


# ----- 4) WEEK CLOSE SUNDAY -----
def gen_TV_WeekClose_Sunday(df, ret_len=14, **kw):
    """Sunday UTC = crypto week-close. Weekly bar settlement flow.
    Fade week's accumulated move.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_sunday = pd.Series(np.asarray(idx.dayofweek == 6), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = is_sunday & (ret < -0.03)
    short_setup = is_sunday & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_WeekClose_Sunday():
    return {
        'ret_len': ('int', 4, 35),
    }


# ----- 5) SETTLEMENT WINDOW T+2 -----
def gen_TV_SettlementWindow_T2(df, ret_len=10, **kw):
    """Tuesday + Wednesday UTC = T+2 settlement window for TradFi trades
    placed Friday/Monday. Settlement-driven flow.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_t2 = pd.Series(np.asarray((idx.dayofweek == 1) | (idx.dayofweek == 2)), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = is_t2 & (ret < -0.03)
    short_setup = is_t2 & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_SettlementWindow_T2():
    return {
        'ret_len': ('int', 4, 30),
    }


STRATEGY_EXPORT = {
    "TV_MonthBoundary_Span": {
        "gen": gen_TV_MonthBoundary_Span, "space": space_TV_MonthBoundary_Span,
        "source": "mac_batch3736_month_boundary_span",
    },
    "TV_FirstMonday_Month": {
        "gen": gen_TV_FirstMonday_Month, "space": space_TV_FirstMonday_Month,
        "source": "mac_batch3736_first_monday_month",
    },
    "TV_MidWeek_Wednesday": {
        "gen": gen_TV_MidWeek_Wednesday, "space": space_TV_MidWeek_Wednesday,
        "source": "mac_batch3736_midweek_wednesday_pivot",
    },
    "TV_WeekClose_Sunday": {
        "gen": gen_TV_WeekClose_Sunday, "space": space_TV_WeekClose_Sunday,
        "source": "mac_batch3736_sunday_week_close_fade",
    },
    "TV_SettlementWindow_T2": {
        "gen": gen_TV_SettlementWindow_T2, "space": space_TV_SettlementWindow_T2,
        "source": "mac_batch3736_T2_settlement_window",
    },
}
