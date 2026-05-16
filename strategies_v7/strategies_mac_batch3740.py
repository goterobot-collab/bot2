"""
Batch 3740 - Mac paralela wave m41 - 5 HOUR-LEVEL narrow calendar fades.

After 8 CONFIRMED V10 mostly on 1h-4h TF, this batch tests hour-of-day
overlays on calendar events for tighter 1h-bar windows.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3739.

 1. TV_OpEx_LastHours      - 3rd Friday + last 4 hours of UTC day
 2. TV_NFP_LastHours       - 1st Friday + last 4 hours UTC
 3. TV_MonthEnd_FirstHrs   - day 1 + first 4 hours UTC (month start kick)
 4. TV_MonthEnd_LastHrs    - last day of month + last 4 hours UTC
 5. TV_SundayOpen_Crypto   - Sunday hours 22-23 UTC (Mon Asia open lead-in)
"""
import numpy as np
import pandas as pd


# ----- 1) OPEX LAST HOURS -----
def gen_TV_OpEx_LastHours(df, hours_back=4, ret_len=6, **kw):
    """3rd Friday of any month UTC + last `hours_back` hours of UTC day.
    Concentrated post-OpEx-close institutional flow.
    """
    hb = int(hours_back)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_friday = idx.dayofweek == 4
    is_3rd_week = (idx.day >= 15) & (idx.day <= 21)
    is_late_hour = idx.hour >= (24 - hb)
    op_lh = pd.Series(np.asarray(is_friday & is_3rd_week & is_late_hour), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = op_lh & (ret < -0.02)
    short_setup = op_lh & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_OpEx_LastHours():
    return {
        'hours_back': ('int', 1, 6),
        'ret_len': ('int', 3, 20),
    }


# ----- 2) NFP LAST HOURS -----
def gen_TV_NFP_LastHours(df, hours_back=4, ret_len=6, **kw):
    """1st Friday + last `hours_back` UTC hours = post-NFP-volatility close.
    """
    hb = int(hours_back)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_friday = idx.dayofweek == 4
    is_1st_week = idx.day <= 7
    is_late_hour = idx.hour >= (24 - hb)
    nfp_lh = pd.Series(np.asarray(is_friday & is_1st_week & is_late_hour), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = nfp_lh & (ret < -0.02)
    short_setup = nfp_lh & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_NFP_LastHours():
    return {
        'hours_back': ('int', 1, 6),
        'ret_len': ('int', 3, 20),
    }


# ----- 3) MONTH-START FIRST HOURS -----
def gen_TV_MonthEnd_FirstHrs(df, hours_fwd=6, ret_len=8, **kw):
    """Day 1 of month + first `hours_fwd` UTC hours = new-month kickoff flow.
    Fade overreaction at the kickoff.
    """
    hf = int(hours_fwd)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_day_1 = idx.day == 1
    is_early_hour = idx.hour < hf
    ms_fh = pd.Series(np.asarray(is_day_1 & is_early_hour), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = ms_fh & (ret < -0.02)
    short_setup = ms_fh & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_MonthEnd_FirstHrs():
    return {
        'hours_fwd': ('int', 2, 12),
        'ret_len': ('int', 3, 24),
    }


# ----- 4) MONTH-END LAST HOURS -----
def gen_TV_MonthEnd_LastHrs(df, hours_back=6, ret_len=8, **kw):
    """Last day of any month + last `hours_back` UTC hours.
    Concentrates month-end close rebalance flow at exact hour.
    """
    hb = int(hours_back)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_last_day = idx.day == idx.days_in_month
    is_late_hour = idx.hour >= (24 - hb)
    me_lh = pd.Series(np.asarray(is_last_day & is_late_hour), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = me_lh & (ret < -0.02)
    short_setup = me_lh & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_MonthEnd_LastHrs():
    return {
        'hours_back': ('int', 2, 12),
        'ret_len': ('int', 3, 24),
    }


# ----- 5) SUNDAY OPEN CRYPTO -----
def gen_TV_SundayOpen_Crypto(df, hour_start=22, hour_end=23, ret_len=8, **kw):
    """Sunday hours hour_start-hour_end UTC = Monday Asia open lead-in.
    Crypto-specific: liquidity returns at this point. Fade weekend drift.
    """
    hs = int(hour_start)
    he = int(hour_end)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_sun = idx.dayofweek == 6
    is_open_hr = (idx.hour >= hs) & (idx.hour <= he)
    so = pd.Series(np.asarray(is_sun & is_open_hr), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = so & (ret < -0.025)
    short_setup = so & (ret > 0.025)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_SundayOpen_Crypto():
    return {
        'hour_start': ('int', 18, 22),
        'hour_end': ('int', 22, 23),
        'ret_len': ('int', 3, 24),
    }


STRATEGY_EXPORT = {
    "TV_OpEx_LastHours": {
        "gen": gen_TV_OpEx_LastHours, "space": space_TV_OpEx_LastHours,
        "source": "mac_batch3740_opex_last_hours_concentrated",
    },
    "TV_NFP_LastHours": {
        "gen": gen_TV_NFP_LastHours, "space": space_TV_NFP_LastHours,
        "source": "mac_batch3740_nfp_last_hours_concentrated",
    },
    "TV_MonthEnd_FirstHrs": {
        "gen": gen_TV_MonthEnd_FirstHrs, "space": space_TV_MonthEnd_FirstHrs,
        "source": "mac_batch3740_month_start_kickoff_hours",
    },
    "TV_MonthEnd_LastHrs": {
        "gen": gen_TV_MonthEnd_LastHrs, "space": space_TV_MonthEnd_LastHrs,
        "source": "mac_batch3740_month_end_close_hours",
    },
    "TV_SundayOpen_Crypto": {
        "gen": gen_TV_SundayOpen_Crypto, "space": space_TV_SundayOpen_Crypto,
        "source": "mac_batch3740_sunday_asia_open_lead_in",
    },
}
