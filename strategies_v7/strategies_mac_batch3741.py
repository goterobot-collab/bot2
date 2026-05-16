"""
Batch 3741 - Mac paralela wave m42 - 5 hour-level narrow on more calendar events.

Continuing narrow > wide pattern (11 CONFIRMED V10).

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3740.

 1. TV_QE_LastHours       - last day of Mar/Jun/Sep/Dec + last N hours
 2. TV_QE_FirstHours      - first day of Apr/Jul/Oct/Jan + first N hours
 3. TV_NFP_FirstHours     - 1st Fri + first N hours (pre-NFP positioning)
 4. TV_OpEx_FirstHours    - 3rd Fri + first N hours
 5. TV_Sunday_LastHours   - Sunday + last N hours (week-close exact)
"""
import numpy as np
import pandas as pd


# ----- 1) QE LAST HOURS -----
def gen_TV_QE_LastHours(df, hours_back=6, ret_len=4, **kw):
    """Last day of Mar/Jun/Sep/Dec UTC + last hours_back hours.
    """
    hb = int(hours_back)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_qend_month = idx.month.isin([3, 6, 9, 12])
    is_last_day = idx.day == idx.days_in_month
    is_late_hour = idx.hour >= (24 - hb)
    qelh = pd.Series(np.asarray(is_qend_month & is_last_day & is_late_hour), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = qelh & (ret < -0.02)
    short_setup = qelh & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_QE_LastHours():
    return {
        'hours_back': ('int', 2, 12),
        'ret_len': ('int', 3, 24),
    }


# ----- 2) QE FIRST HOURS (NEW QUARTER) -----
def gen_TV_QE_FirstHours(df, hours_fwd=6, ret_len=4, **kw):
    """Day 1 of Apr/Jul/Oct/Jan UTC + first hours_fwd hours.
    """
    hf = int(hours_fwd)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_qstart_month = idx.month.isin([4, 7, 10, 1])
    is_day_1 = idx.day == 1
    is_early_hour = idx.hour < hf
    qefh = pd.Series(np.asarray(is_qstart_month & is_day_1 & is_early_hour), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = qefh & (ret < -0.02)
    short_setup = qefh & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_QE_FirstHours():
    return {
        'hours_fwd': ('int', 2, 12),
        'ret_len': ('int', 3, 24),
    }


# ----- 3) NFP FIRST HOURS -----
def gen_TV_NFP_FirstHours(df, hours_fwd=6, ret_len=4, **kw):
    """1st Friday + first hours_fwd hours UTC = pre-NFP positioning.
    """
    hf = int(hours_fwd)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_friday = idx.dayofweek == 4
    is_1st_week = idx.day <= 7
    is_early_hour = idx.hour < hf
    nfp_fh = pd.Series(np.asarray(is_friday & is_1st_week & is_early_hour), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = nfp_fh & (ret < -0.02)
    short_setup = nfp_fh & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_NFP_FirstHours():
    return {
        'hours_fwd': ('int', 2, 12),
        'ret_len': ('int', 3, 24),
    }


# ----- 4) OPEX FIRST HOURS -----
def gen_TV_OpEx_FirstHours(df, hours_fwd=6, ret_len=4, **kw):
    """3rd Friday + first hours_fwd hours UTC.
    """
    hf = int(hours_fwd)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_friday = idx.dayofweek == 4
    is_3rd_week = (idx.day >= 15) & (idx.day <= 21)
    is_early_hour = idx.hour < hf
    op_fh = pd.Series(np.asarray(is_friday & is_3rd_week & is_early_hour), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = op_fh & (ret < -0.02)
    short_setup = op_fh & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_OpEx_FirstHours():
    return {
        'hours_fwd': ('int', 2, 12),
        'ret_len': ('int', 3, 24),
    }


# ----- 5) SUNDAY LAST HOURS -----
def gen_TV_Sunday_LastHours(df, hours_back=6, ret_len=6, **kw):
    """Sunday + last hours_back hours UTC = exact week-close.
    """
    hb = int(hours_back)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_sun = idx.dayofweek == 6
    is_late_hour = idx.hour >= (24 - hb)
    su_lh = pd.Series(np.asarray(is_sun & is_late_hour), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = su_lh & (ret < -0.025)
    short_setup = su_lh & (ret > 0.025)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Sunday_LastHours():
    return {
        'hours_back': ('int', 2, 12),
        'ret_len': ('int', 3, 24),
    }


STRATEGY_EXPORT = {
    "TV_QE_LastHours": {"gen": gen_TV_QE_LastHours, "space": space_TV_QE_LastHours,
                         "source": "mac_batch3741_QE_last_hours"},
    "TV_QE_FirstHours": {"gen": gen_TV_QE_FirstHours, "space": space_TV_QE_FirstHours,
                          "source": "mac_batch3741_QE_first_hours_new_quarter"},
    "TV_NFP_FirstHours": {"gen": gen_TV_NFP_FirstHours, "space": space_TV_NFP_FirstHours,
                           "source": "mac_batch3741_NFP_first_hours_pre_release"},
    "TV_OpEx_FirstHours": {"gen": gen_TV_OpEx_FirstHours, "space": space_TV_OpEx_FirstHours,
                            "source": "mac_batch3741_OpEx_first_hours"},
    "TV_Sunday_LastHours": {"gen": gen_TV_Sunday_LastHours, "space": space_TV_Sunday_LastHours,
                             "source": "mac_batch3741_sunday_last_hours_week_close"},
}
