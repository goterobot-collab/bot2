"""
Batch 3734 - Mac paralela wave m35 - 5 holiday / cultural calendar fade.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3733.

 1. TV_USHoliday_Window      - around major US holidays (Jul 4, Thanksgiving)
 2. TV_AprilTax_Window       - April 10-17 (US tax day Apr 15)
 3. TV_ChineseNewYear_Window - Jan 21 - Feb 20 (variable - covers most cases)
 4. TV_USFiscal_YearEnd      - Sep 25 - Oct 5 (US Federal fiscal year end)
 5. TV_Halloween_Effect      - Oct 31 - Nov 7 (post-Halloween effect)
"""
import numpy as np
import pandas as pd


# ----- 1) US HOLIDAYS WINDOW -----
def gen_TV_USHoliday_Window(df, ret_len=12, **kw):
    """Around major US trading holidays (low liquidity, fade overreactions):
    - Jul 1-7 (Independence Day)
    - Nov 22-30 (Thanksgiving)
    - Dec 22-31 (Christmas+NewYear)
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_july_4 = (idx.month == 7) & (idx.day >= 1) & (idx.day <= 7)
    is_thanksgiving = (idx.month == 11) & (idx.day >= 22) & (idx.day <= 30)
    is_xmas = (idx.month == 12) & (idx.day >= 22)
    holiday = pd.Series(np.asarray(is_july_4 | is_thanksgiving | is_xmas), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = holiday & (ret < -0.03)
    short_setup = holiday & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_USHoliday_Window():
    return {
        'ret_len': ('int', 5, 30),
    }


# ----- 2) APRIL TAX WINDOW -----
def gen_TV_AprilTax_Window(df, ret_len=15, **kw):
    """April 10-17 (US tax day window). Crypto sales for tax payments
    create selling pressure -> fade direction.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_tax_window = (idx.month == 4) & (idx.day >= 10) & (idx.day <= 17)
    tax = pd.Series(np.asarray(is_tax_window), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = tax & (ret < -0.04)
    short_setup = tax & (ret > 0.04)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_AprilTax_Window():
    return {
        'ret_len': ('int', 5, 30),
    }


# ----- 3) CHINESE NEW YEAR WINDOW -----
def gen_TV_ChineseNewYear_Window(df, ret_len=15, **kw):
    """CNY varies but ~Jan 21 - Feb 20. Asian crypto traders typically
    reduce activity, creating fade opportunities in liquid alts.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_jan_late = (idx.month == 1) & (idx.day >= 21)
    is_feb_early = (idx.month == 2) & (idx.day <= 20)
    cny = pd.Series(np.asarray(is_jan_late | is_feb_early), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = cny & (ret < -0.04)
    short_setup = cny & (ret > 0.04)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_ChineseNewYear_Window():
    return {
        'ret_len': ('int', 5, 30),
    }


# ----- 4) US FISCAL YEAR END -----
def gen_TV_USFiscal_YearEnd(df, ret_len=14, **kw):
    """US Federal fiscal year ends Sep 30. Window Sep 25 - Oct 5.
    Institutional rebalancing distinct from calendar quarter-end.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_sep_late = (idx.month == 9) & (idx.day >= 25)
    is_oct_early = (idx.month == 10) & (idx.day <= 5)
    fy = pd.Series(np.asarray(is_sep_late | is_oct_early), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = fy & (ret < -0.03)
    short_setup = fy & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_USFiscal_YearEnd():
    return {
        'ret_len': ('int', 5, 35),
    }


# ----- 5) HALLOWEEN EFFECT -----
def gen_TV_Halloween_Effect(df, ret_len=14, **kw):
    """Oct 31 - Nov 7. "Sell in May / Buy in Halloween" - bullish bias
    in November. Continue direction if ret>0, else stay flat.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_late_oct = (idx.month == 10) & (idx.day >= 31)
    is_early_nov = (idx.month == 11) & (idx.day <= 7)
    hw = pd.Series(np.asarray(is_late_oct | is_early_nov), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    # Continuation (Halloween bullish bias)
    long_setup = hw & (ret > 0.02)
    short_setup = hw & (ret < -0.05)  # only short if very negative
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Halloween_Effect():
    return {
        'ret_len': ('int', 4, 30),
    }


STRATEGY_EXPORT = {
    "TV_USHoliday_Window": {
        "gen": gen_TV_USHoliday_Window, "space": space_TV_USHoliday_Window,
        "source": "mac_batch3734_us_holiday_low_liq_fade",
    },
    "TV_AprilTax_Window": {
        "gen": gen_TV_AprilTax_Window, "space": space_TV_AprilTax_Window,
        "source": "mac_batch3734_april_tax_day_window",
    },
    "TV_ChineseNewYear_Window": {
        "gen": gen_TV_ChineseNewYear_Window, "space": space_TV_ChineseNewYear_Window,
        "source": "mac_batch3734_CNY_window",
    },
    "TV_USFiscal_YearEnd": {
        "gen": gen_TV_USFiscal_YearEnd, "space": space_TV_USFiscal_YearEnd,
        "source": "mac_batch3734_US_federal_fiscal_year_end",
    },
    "TV_Halloween_Effect": {
        "gen": gen_TV_Halloween_Effect, "space": space_TV_Halloween_Effect,
        "source": "mac_batch3734_halloween_buy_continuation",
    },
}
