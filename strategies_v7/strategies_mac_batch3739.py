"""
Batch 3739 - Mac paralela wave m40 - 5 NARROW specific-anchor calendar fades.

After loop 20 found 0 CONFIRMED in wide calendar windows (raw=133 m38, 92 m39),
this batch tests TIGHT anchor dates (1-3 day windows).

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3738.

 1. TV_ETHMerge_Anniversary   - Sep 14-16 (ETH Merge Sep 15, 2022)
 2. TV_BitcoinWP_Day          - Oct 30 - Nov 2 (BTC whitepaper Oct 31, 2008)
 3. TV_GenesisBlock_Day       - Jan 3-4 (BTC genesis Jan 3, 2009)
 4. TV_QuarterEnd_LastDay     - EXACT last day of Q-end month (narrowest)
 5. TV_LastFriday_June        - exact last Friday of June (Russell exact)
"""
import numpy as np
import pandas as pd


# ----- 1) ETH MERGE ANNIVERSARY -----
def gen_TV_ETHMerge_Anniversary(df, window=1, ret_len=10, **kw):
    """Sep 15 +/- window UTC = ETH proof-of-stake merge anniversary.
    """
    w = int(window)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_sep = idx.month == 9
    is_near = (idx.day >= (15 - w)) & (idx.day <= (15 + w))
    eth = pd.Series(np.asarray(is_sep & is_near), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = eth & (ret < -0.03)
    short_setup = eth & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_ETHMerge_Anniversary():
    return {
        'window': ('int', 0, 3),
        'ret_len': ('int', 4, 30),
    }


# ----- 2) BITCOIN WHITEPAPER DAY -----
def gen_TV_BitcoinWP_Day(df, ret_len=12, **kw):
    """Oct 30 - Nov 2 UTC = BTC whitepaper Oct 31, 2008.
    Bitcoin community symbolic date, often retail-driven moves.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_late_oct = (idx.month == 10) & (idx.day >= 30)
    is_early_nov = (idx.month == 11) & (idx.day <= 2)
    wp = pd.Series(np.asarray(is_late_oct | is_early_nov), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = wp & (ret < -0.03)
    short_setup = wp & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_BitcoinWP_Day():
    return {
        'ret_len': ('int', 4, 30),
    }


# ----- 3) GENESIS BLOCK DAY -----
def gen_TV_GenesisBlock_Day(df, ret_len=10, **kw):
    """Jan 3-4 UTC = BTC genesis block Jan 3, 2009.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_gb = (idx.month == 1) & (idx.day.isin([3, 4]))
    gb = pd.Series(np.asarray(is_gb), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = gb & (ret < -0.03)
    short_setup = gb & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_GenesisBlock_Day():
    return {
        'ret_len': ('int', 4, 30),
    }


# ----- 4) QUARTER END EXACT LAST DAY -----
def gen_TV_QuarterEnd_LastDay(df, ret_len=10, **kw):
    """ONLY exact last day of Mar/Jun/Sep/Dec (day == days_in_month).
    Narrower than QuarterEnd (which uses last N days window).
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_qend_month = idx.month.isin([3, 6, 9, 12])
    is_last_day = idx.day == idx.days_in_month
    qe = pd.Series(np.asarray(is_qend_month & is_last_day), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = qe & (ret < -0.02)
    short_setup = qe & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_QuarterEnd_LastDay():
    return {
        'ret_len': ('int', 3, 25),
    }


# ----- 5) LAST FRIDAY OF JUNE (RUSSELL REBAL EXACT) -----
def gen_TV_LastFriday_June(df, ret_len=12, **kw):
    """Exact last Friday of June UTC = Russell index annual rebalance.
    Day >= 24 + Friday in June. Narrowest possible Russell event.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_june = idx.month == 6
    is_friday = idx.dayofweek == 4
    is_last_week = idx.day >= 24
    lfj = pd.Series(np.asarray(is_june & is_friday & is_last_week), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = lfj & (ret < -0.025)
    short_setup = lfj & (ret > 0.025)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_LastFriday_June():
    return {
        'ret_len': ('int', 4, 30),
    }


STRATEGY_EXPORT = {
    "TV_ETHMerge_Anniversary": {
        "gen": gen_TV_ETHMerge_Anniversary, "space": space_TV_ETHMerge_Anniversary,
        "source": "mac_batch3739_ETH_merge_sep_15_anchor",
    },
    "TV_BitcoinWP_Day": {
        "gen": gen_TV_BitcoinWP_Day, "space": space_TV_BitcoinWP_Day,
        "source": "mac_batch3739_BTC_whitepaper_oct_31",
    },
    "TV_GenesisBlock_Day": {
        "gen": gen_TV_GenesisBlock_Day, "space": space_TV_GenesisBlock_Day,
        "source": "mac_batch3739_BTC_genesis_jan_3",
    },
    "TV_QuarterEnd_LastDay": {
        "gen": gen_TV_QuarterEnd_LastDay, "space": space_TV_QuarterEnd_LastDay,
        "source": "mac_batch3739_qend_exact_last_day_only",
    },
    "TV_LastFriday_June": {
        "gen": gen_TV_LastFriday_June, "space": space_TV_LastFriday_June,
        "source": "mac_batch3739_russell_last_friday_june_exact",
    },
}
