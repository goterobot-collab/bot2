"""
Batch 3729 - Mac paralela wave m30 - 5 calendar + technical HYBRID families.

After m26 yielded 2 CONFIRMED V10 (TV_Calendar_QuarterEnd) but m28 calendar
VARIANTS yielded 0 CONFIRMED, this batch tries calendar + technical FILTERS
combinados to see if the edge transfers with confirmation.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3728.

 1. TV_QuarterEnd_RSI_Combo      - QuarterEnd window AND RSI extreme
 2. TV_MonthEnd_BB_Pierce        - MonthEnd window AND BB pierce reversal
 3. TV_Friday_VolumeSpike        - Friday window AND volume spike fade
 4. TV_NewMonth_Continuation     - First N days of month + trend confirm
 5. TV_OptionsExpiry_Proxy       - 3rd Friday window + reversal
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ----- 1) QUARTER-END + RSI COMBO -----
def gen_TV_QuarterEnd_RSI_Combo(df, days_window=2, rsi_len=14, rsi_extreme=70, **kw):
    """In last `days_window` days of Mar/Jun/Sep/Dec UTC,
    short if RSI > rsi_extreme, long if RSI < (100 - rsi_extreme).
    Combines the working QuarterEnd window with RSI confirmation.
    """
    dw = int(days_window)
    rl = int(rsi_len)
    re = float(rsi_extreme)
    c = df['close'].astype(float)
    idx = df.index
    is_qend_month = idx.month.isin([3, 6, 9, 12])
    is_last_days = (idx.days_in_month - idx.day) < dw
    qend_window = pd.Series(np.asarray(is_qend_month & is_last_days), index=idx)
    rsi = _rsi(c, rl)
    long_setup = qend_window & (rsi < (100 - re))
    short_setup = qend_window & (rsi > re)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_QuarterEnd_RSI_Combo():
    return {
        'days_window': ('int', 1, 5),
        'rsi_len': ('int', 7, 21),
        'rsi_extreme': ('int', 60, 80),
    }


# ----- 2) MONTH-END + BB PIERCE -----
def gen_TV_MonthEnd_BB_Pierce(df, days_window=3, bb_len=20, bb_mult=2.0, **kw):
    """Last `days_window` days of any month AND prev bar pierced BB.
    Long: pierced lower BB, then re-enters. Short: pierced upper BB.
    """
    dw = int(days_window)
    n = int(bb_len)
    m = float(bb_mult)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    idx = df.index
    is_last_days = (idx.days_in_month - idx.day) < dw
    monthend = pd.Series(np.asarray(is_last_days), index=idx)
    mid = c.rolling(n, min_periods=n).mean()
    sd = c.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + m * sd
    lower = mid - m * sd
    pierced_lower = l.shift(1) < lower.shift(1)
    pierced_upper = h.shift(1) > upper.shift(1)
    long_setup = monthend & pierced_lower & (c > lower)
    short_setup = monthend & pierced_upper & (c < upper)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_MonthEnd_BB_Pierce():
    return {
        'days_window': ('int', 1, 7),
        'bb_len': ('int', 14, 35),
        'bb_mult': ('float', 1.5, 3.0),
    }


# ----- 3) FRIDAY + VOLUME SPIKE FADE -----
def gen_TV_Friday_VolumeSpike(df, vol_ma=20, vol_mult=2.0, ret_len=8, **kw):
    """Friday UTC + volume > vol_mult * SMA(vol, vol_ma) + recent return
    sign as fade target.
    """
    vma = int(vol_ma)
    vm = float(vol_mult)
    rl = int(ret_len)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    idx = df.index
    is_friday = pd.Series(np.asarray(idx.dayofweek == 4), index=idx)
    vol_avg = _sma(v, vma)
    big_vol = v > vm * vol_avg
    ret = (c / c.shift(rl) - 1.0)
    long_setup = is_friday & big_vol & (ret < -0.02)
    short_setup = is_friday & big_vol & (ret > 0.02)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Friday_VolumeSpike():
    return {
        'vol_ma': ('int', 10, 50),
        'vol_mult': ('float', 1.5, 3.5),
        'ret_len': ('int', 4, 20),
    }


# ----- 4) NEW MONTH CONTINUATION + TREND -----
def gen_TV_NewMonth_Continuation(df, days=5, ma_len=50, **kw):
    """First N days of new month: continue direction if close vs MA(ma_len)
    confirms trend.
    """
    dd = int(days)
    ml = int(ma_len)
    c = df['close'].astype(float)
    idx = df.index
    is_first = pd.Series(np.asarray(idx.day <= dd), index=idx)
    ma = _sma(c, ml)
    long_setup = is_first & (c > ma) & (c.shift(1) <= ma.shift(1))
    short_setup = is_first & (c < ma) & (c.shift(1) >= ma.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_NewMonth_Continuation():
    return {
        'days': ('int', 2, 14),
        'ma_len': ('int', 20, 150),
    }


# ----- 5) 3RD FRIDAY (OPTIONS EXPIRY) PROXY -----
def gen_TV_OptionsExpiry_Proxy(df, ret_len=10, **kw):
    """3rd Friday of month UTC = TradFi monthly options expiry.
    Even crypto can have spill-over. Fade recent return on these days.
    """
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    is_friday = idx.dayofweek == 4
    is_3rd_week = (idx.day >= 15) & (idx.day <= 21)
    is_3rd_friday = pd.Series(np.asarray(is_friday & is_3rd_week), index=idx)
    ret = (c / c.shift(rl) - 1.0)
    long_setup = is_3rd_friday & (ret < -0.03)
    short_setup = is_3rd_friday & (ret > 0.03)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_OptionsExpiry_Proxy():
    return {
        'ret_len': ('int', 4, 30),
    }


STRATEGY_EXPORT = {
    "TV_QuarterEnd_RSI_Combo": {
        "gen": gen_TV_QuarterEnd_RSI_Combo, "space": space_TV_QuarterEnd_RSI_Combo,
        "source": "mac_batch3729_QE_RSI_extreme_combo",
    },
    "TV_MonthEnd_BB_Pierce": {
        "gen": gen_TV_MonthEnd_BB_Pierce, "space": space_TV_MonthEnd_BB_Pierce,
        "source": "mac_batch3729_ME_BB_pierce_revert",
    },
    "TV_Friday_VolumeSpike": {
        "gen": gen_TV_Friday_VolumeSpike, "space": space_TV_Friday_VolumeSpike,
        "source": "mac_batch3729_Friday_vol_spike_fade",
    },
    "TV_NewMonth_Continuation": {
        "gen": gen_TV_NewMonth_Continuation, "space": space_TV_NewMonth_Continuation,
        "source": "mac_batch3729_new_month_trend_continuation",
    },
    "TV_OptionsExpiry_Proxy": {
        "gen": gen_TV_OptionsExpiry_Proxy, "space": space_TV_OptionsExpiry_Proxy,
        "source": "mac_batch3729_3rd_friday_options_expiry",
    },
}
