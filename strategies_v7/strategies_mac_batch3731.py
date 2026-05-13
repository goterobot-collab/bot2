"""
Batch 3731 - Mac paralela wave m32 - 5 MORE calendar-event-fade variants.

Doubling down on calendar-event-fade family after 3 CONFIRMED V10:
 - TV_Calendar_QuarterEnd OP/LINK 1h (m26)
 - TV_OptionsExpiry_Proxy ONDO 4h (m30)

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3730.

 1. TV_TripleWitching          - 3rd Friday of Mar/Jun/Sep/Dec (more
                                  specific than monthly OpEx; index opts +
                                  index futs + stock opts expire same day)
 2. TV_HalfMonth_Pivot         - around day-15 of month (mid-month pivot)
 3. TV_LastHour_Day            - last 1-3 hours of UTC day (closing flow)
 4. TV_CME_Rollover            - last business Friday before contract roll
                                  (4th Friday approx)
 5. TV_QuarterStart_Cont       - first N days of new quarter
                                  (opposite of QuarterEnd: continuation)
"""
import numpy as np
import pandas as pd


# ----- 1) TRIPLE WITCHING (3RD FRIDAY OF MAR/JUN/SEP/DEC) -----
def gen_TV_TripleWitching(df, ret_len=12, **kw):
    """3rd Friday of Mar/Jun/Sep/Dec = "triple witching" in TradFi.
    Most impactful options expiry of year. Fade recent return.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_friday = idx.dayofweek == 4
    is_3rd_week = (idx.day >= 15) & (idx.day <= 21)
    is_qend_month = idx.month.isin([3, 6, 9, 12])
    is_tw = pd.Series(np.asarray(is_friday & is_3rd_week & is_qend_month), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = is_tw & (ret < -0.02)
    short_setup = is_tw & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_TripleWitching():
    return {
        'ret_len': ('int', 5, 40),
    }


# ----- 2) HALF-MONTH PIVOT (DAY 13-17) -----
def gen_TV_HalfMonth_Pivot(df, window_radius=2, ret_len=10, **kw):
    """Mid-month: day-15 +/- window_radius UTC. Often institutional flow
    rebalance toward intra-month VWAP.
    """
    wr = int(window_radius)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_mid = (idx.day >= (15 - wr)) & (idx.day <= (15 + wr))
    midmonth = pd.Series(np.asarray(is_mid), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = midmonth & (ret < -0.03)
    short_setup = midmonth & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_HalfMonth_Pivot():
    return {
        'window_radius': ('int', 1, 5),
        'ret_len': ('int', 5, 30),
    }


# ----- 3) LAST HOUR OF DAY FADE -----
def gen_TV_LastHour_Day(df, hours_back=2, ret_len=6, **kw):
    """Last `hours_back` hours of UTC day. Captures end-of-day flow rebalance.
    Fade recent return on these bars.
    """
    hb = int(hours_back)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_last_hours = idx.hour >= (24 - hb)
    last_hours = pd.Series(np.asarray(is_last_hours), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = last_hours & (ret < -0.015)
    short_setup = last_hours & (ret > 0.015)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_LastHour_Day():
    return {
        'hours_back': ('int', 1, 4),
        'ret_len': ('int', 3, 20),
    }


# ----- 4) CME ROLLOVER (LAST FRIDAY BEFORE LAST BUSINESS DAY) -----
def gen_TV_CME_Rollover(df, ret_len=14, **kw):
    """CME crypto futures roll on last business Friday of month.
    Approximate: last Friday in calendar month (day >= 22, day_of_week=4).
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_friday = idx.dayofweek == 4
    is_late_month = idx.day >= 22
    is_cme = pd.Series(np.asarray(is_friday & is_late_month), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = is_cme & (ret < -0.025)
    short_setup = is_cme & (ret > 0.025)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_CME_Rollover():
    return {
        'ret_len': ('int', 5, 35),
    }


# ----- 5) QUARTER START CONTINUATION -----
def gen_TV_QuarterStart_Cont(df, days=5, ret_len=20, **kw):
    """First N days of Apr/Jul/Oct/Jan = new quarter. Continue direction
    (opposite of QuarterEnd which fades). Captures fresh institutional
    flow at start of new quarter.
    """
    dd = int(days)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_q_start_month = idx.month.isin([4, 7, 10, 1])
    is_first_days = idx.day <= dd
    qstart = pd.Series(np.asarray(is_q_start_month & is_first_days), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    # Continue direction (opposite of QE)
    long_setup = qstart & (ret > 0.03)
    short_setup = qstart & (ret < -0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_QuarterStart_Cont():
    return {
        'days': ('int', 2, 14),
        'ret_len': ('int', 5, 40),
    }


STRATEGY_EXPORT = {
    "TV_TripleWitching": {
        "gen": gen_TV_TripleWitching, "space": space_TV_TripleWitching,
        "source": "mac_batch3731_triple_witching_3rd_friday_qend_month",
    },
    "TV_HalfMonth_Pivot": {
        "gen": gen_TV_HalfMonth_Pivot, "space": space_TV_HalfMonth_Pivot,
        "source": "mac_batch3731_mid_month_pivot",
    },
    "TV_LastHour_Day": {
        "gen": gen_TV_LastHour_Day, "space": space_TV_LastHour_Day,
        "source": "mac_batch3731_end_of_day_UTC_fade",
    },
    "TV_CME_Rollover": {
        "gen": gen_TV_CME_Rollover, "space": space_TV_CME_Rollover,
        "source": "mac_batch3731_CME_last_friday_rollover",
    },
    "TV_QuarterStart_Cont": {
        "gen": gen_TV_QuarterStart_Cont, "space": space_TV_QuarterStart_Cont,
        "source": "mac_batch3731_quarter_start_continuation",
    },
}
