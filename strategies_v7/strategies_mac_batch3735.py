"""
Batch 3735 - Mac paralela wave m36 - 5 macro-event calendar fade variants.

Continuing calendar-event-fade family (6 CONFIRMED V10 so far).
Macro-release-driven calendar windows.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3734.

 1. TV_FirstTradingDay_Month  - exact day-1 of month (+/- radius) fade
 2. TV_NFP_Friday             - 1st Friday of month (US Non-Farm Payrolls)
 3. TV_CPI_Day_Proxy          - day 10-15 of month (CPI release window)
 4. TV_Weekend_Effect_Crypto  - Sat/Sun UTC (crypto weekend low-liq) fade
 5. TV_PostOpEx_Monday        - Monday after 3rd Friday (post-expiry drift)
"""
import numpy as np
import pandas as pd


# ----- 1) FIRST TRADING DAY OF MONTH -----
def gen_TV_FirstTradingDay_Month(df, radius=0, ret_len=12, **kw):
    """Day-1 of month UTC (+/- radius). New-month institutional inflows.
    """
    r = int(radius)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_d1 = idx.day <= (1 + r)
    d1 = pd.Series(np.asarray(is_d1), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = d1 & (ret < -0.03)
    short_setup = d1 & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_FirstTradingDay_Month():
    return {
        'radius': ('int', 0, 3),
        'ret_len': ('int', 4, 30),
    }


# ----- 2) NFP FRIDAY (1ST FRIDAY OF MONTH) -----
def gen_TV_NFP_Friday(df, ret_len=10, **kw):
    """1st Friday of month UTC = US Non-Farm Payrolls release day.
    High-volatility macro day -> fade overreaction.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_friday = idx.dayofweek == 4
    is_1st_week = idx.day <= 7
    nfp = pd.Series(np.asarray(is_friday & is_1st_week), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = nfp & (ret < -0.025)
    short_setup = nfp & (ret > 0.025)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_NFP_Friday():
    return {
        'ret_len': ('int', 4, 30),
    }


# ----- 3) CPI DAY PROXY (DAY 10-15) -----
def gen_TV_CPI_Day_Proxy(df, day_lo=10, day_hi=15, ret_len=10, **kw):
    """US CPI typically released day 10-15 of month. Macro vol day -> fade.
    """
    dl = int(day_lo)
    dh = int(day_hi)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_cpi = (idx.day >= dl) & (idx.day <= dh)
    cpi = pd.Series(np.asarray(is_cpi), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = cpi & (ret < -0.03)
    short_setup = cpi & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_CPI_Day_Proxy():
    return {
        'day_lo': ('int', 8, 13),
        'day_hi': ('int', 13, 18),
        'ret_len': ('int', 4, 30),
    }


# ----- 4) WEEKEND EFFECT (CRYPTO 24/7) -----
def gen_TV_Weekend_Effect_Crypto(df, ret_len=12, **kw):
    """Crypto trades 24/7. Sat/Sun UTC = low institutional liquidity,
    retail-dominated. Fade weekend overreactions.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_weekend = idx.dayofweek >= 5  # Sat=5, Sun=6
    we = pd.Series(np.asarray(is_weekend), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = we & (ret < -0.025)
    short_setup = we & (ret > 0.025)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Weekend_Effect_Crypto():
    return {
        'ret_len': ('int', 4, 30),
    }


# ----- 5) POST-OPEX MONDAY -----
def gen_TV_PostOpEx_Monday(df, ret_len=12, **kw):
    """Monday after 3rd Friday (day 18-24, dayofweek=Monday).
    Post-options-expiry drift / re-positioning.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_monday = idx.dayofweek == 0
    is_post_opex = (idx.day >= 18) & (idx.day <= 24)
    pox = pd.Series(np.asarray(is_monday & is_post_opex), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = pox & (ret < -0.025)
    short_setup = pox & (ret > 0.025)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_PostOpEx_Monday():
    return {
        'ret_len': ('int', 4, 30),
    }


STRATEGY_EXPORT = {
    "TV_FirstTradingDay_Month": {
        "gen": gen_TV_FirstTradingDay_Month, "space": space_TV_FirstTradingDay_Month,
        "source": "mac_batch3735_first_trading_day_month",
    },
    "TV_NFP_Friday": {
        "gen": gen_TV_NFP_Friday, "space": space_TV_NFP_Friday,
        "source": "mac_batch3735_NFP_1st_friday",
    },
    "TV_CPI_Day_Proxy": {
        "gen": gen_TV_CPI_Day_Proxy, "space": space_TV_CPI_Day_Proxy,
        "source": "mac_batch3735_CPI_release_window",
    },
    "TV_Weekend_Effect_Crypto": {
        "gen": gen_TV_Weekend_Effect_Crypto, "space": space_TV_Weekend_Effect_Crypto,
        "source": "mac_batch3735_crypto_weekend_low_liq_fade",
    },
    "TV_PostOpEx_Monday": {
        "gen": gen_TV_PostOpEx_Monday, "space": space_TV_PostOpEx_Monday,
        "source": "mac_batch3735_post_opex_monday_drift",
    },
}
