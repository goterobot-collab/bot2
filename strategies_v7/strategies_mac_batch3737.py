"""
Batch 3737 - Mac paralela wave m38 - 5 more macro/event calendar fade variants.

Continuing calendar-event-fade family (8 CONFIRMED V10).

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3736.

 1. TV_BlackFriday_Week        - Nov 22-28
 2. TV_TaxLoss_December        - Dec 15-31 (tax-loss harvesting)
 3. TV_Russell_Rebalance       - last week of June (Russell index rebal)
 4. TV_EuroDollar_Settlement   - 3rd Wednesday of Mar/Jun/Sep/Dec
 5. TV_NewYearJan2_Window      - Jan 1-3 (first trading days)
"""
import numpy as np
import pandas as pd


# ----- 1) BLACK FRIDAY WEEK -----
def gen_TV_BlackFriday_Week(df, ret_len=10, **kw):
    """Nov 22-28 UTC = US Thanksgiving + Black Friday week.
    Low institutional liquidity -> fade retail-driven moves.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_bf = (idx.month == 11) & (idx.day >= 22) & (idx.day <= 28)
    bf = pd.Series(np.asarray(is_bf), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = bf & (ret < -0.03)
    short_setup = bf & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_BlackFriday_Week():
    return {
        'ret_len': ('int', 4, 30),
    }


# ----- 2) TAX-LOSS DECEMBER -----
def gen_TV_TaxLoss_December(df, day_start=15, ret_len=15, **kw):
    """Dec 15-31 UTC = tax-loss harvesting window.
    Institutional sell pressure on losers -> fade.
    """
    ds = int(day_start)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_tl = (idx.month == 12) & (idx.day >= ds)
    tl = pd.Series(np.asarray(is_tl), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = tl & (ret < -0.04)
    short_setup = tl & (ret > 0.04)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_TaxLoss_December():
    return {
        'day_start': ('int', 10, 22),
        'ret_len': ('int', 5, 35),
    }


# ----- 3) RUSSELL REBALANCE (LAST WEEK OF JUNE) -----
def gen_TV_Russell_Rebalance(df, ret_len=12, **kw):
    """Last week of June UTC = Russell index annual rebalancing.
    Heavy institutional flow at known dates -> fade overreactions.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_rr = (idx.month == 6) & (idx.day >= 22)
    rr = pd.Series(np.asarray(is_rr), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = rr & (ret < -0.03)
    short_setup = rr & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Russell_Rebalance():
    return {
        'ret_len': ('int', 4, 30),
    }


# ----- 4) EURODOLLAR SETTLEMENT (3RD WEDNESDAY OF Q-END MONTH) -----
def gen_TV_EuroDollar_Settlement(df, ret_len=12, **kw):
    """3rd Wednesday of Mar/Jun/Sep/Dec UTC = Eurodollar futures settlement
    (now SOFR but same calendar). Major rate-related institutional flow.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_wed = idx.dayofweek == 2
    is_3rd_week = (idx.day >= 15) & (idx.day <= 21)
    is_qend_month = idx.month.isin([3, 6, 9, 12])
    is_eds = pd.Series(np.asarray(is_wed & is_3rd_week & is_qend_month), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = is_eds & (ret < -0.025)
    short_setup = is_eds & (ret > 0.025)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_EuroDollar_Settlement():
    return {
        'ret_len': ('int', 4, 30),
    }


# ----- 5) NEW YEAR JAN 1-3 -----
def gen_TV_NewYearJan2_Window(df, ret_len=10, **kw):
    """Jan 1-3 UTC. First trading days of year = re-entry positioning.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_ny = (idx.month == 1) & (idx.day <= 3)
    ny = pd.Series(np.asarray(is_ny), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = ny & (ret < -0.03)
    short_setup = ny & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_NewYearJan2_Window():
    return {
        'ret_len': ('int', 4, 30),
    }


STRATEGY_EXPORT = {
    "TV_BlackFriday_Week": {
        "gen": gen_TV_BlackFriday_Week, "space": space_TV_BlackFriday_Week,
        "source": "mac_batch3737_black_friday_low_liq",
    },
    "TV_TaxLoss_December": {
        "gen": gen_TV_TaxLoss_December, "space": space_TV_TaxLoss_December,
        "source": "mac_batch3737_tax_loss_harvesting",
    },
    "TV_Russell_Rebalance": {
        "gen": gen_TV_Russell_Rebalance, "space": space_TV_Russell_Rebalance,
        "source": "mac_batch3737_russell_june_rebalance",
    },
    "TV_EuroDollar_Settlement": {
        "gen": gen_TV_EuroDollar_Settlement, "space": space_TV_EuroDollar_Settlement,
        "source": "mac_batch3737_eurodollar_3rd_wed_qmonth",
    },
    "TV_NewYearJan2_Window": {
        "gen": gen_TV_NewYearJan2_Window, "space": space_TV_NewYearJan2_Window,
        "source": "mac_batch3737_new_year_first_3_days",
    },
}
