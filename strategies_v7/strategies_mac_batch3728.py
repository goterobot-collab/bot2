"""
Batch 3728 - Mac paralela wave m29 - 5 novel non-calendar families.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3727.

 1. TV_Compound_Return_Z          - z-score of geometric N-bar compound return
 2. TV_RSI_Slope_Acceleration     - 2nd derivative of RSI (acceleration)
 3. TV_VolPrice_Divergence_Slope  - slope of vol vs price-direction divergence
 4. TV_BB_MeanRev_Landing         - close re-enters BB after pierce (revert
                                     confirmation, distinct from BB_Squeeze)
 5. TV_TwoSpeed_MA_Diff           - fast EMA - slow EMA normalized; surge cross
                                     (distinct from MACD which uses signal line)
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


# ----- 1) COMPOUND RETURN Z-SCORE -----
def gen_TV_Compound_Return_Z(df, ret_len=10, z_win=100, z_thr=2.0, **kw):
    """Geometric N-bar compound return = (close / close.shift(N) - 1).
    z = (compound - mean) / std over z_win.
    Long: z > +z_thr (extreme up move) -> mean-revert short. Symmetric.
    """
    rl = int(ret_len)
    zw = int(z_win)
    th = float(z_thr)
    c = df['close'].astype(float)
    comp = (c / c.shift(rl) - 1.0)
    mu = comp.rolling(zw, min_periods=zw).mean()
    sd = comp.rolling(zw, min_periods=zw).std(ddof=0).replace(0.0, np.nan)
    z = (comp - mu) / sd
    long_cross = (z < -th) & (z.shift(1) >= -th)  # extreme down -> long fade
    short_cross = (z > th) & (z.shift(1) <= th)  # extreme up -> short fade
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Compound_Return_Z():
    return {
        'ret_len': ('int', 3, 30),
        'z_win': ('int', 40, 250),
        'z_thr': ('float', 1.5, 3.5),
    }


# ----- 2) RSI SLOPE ACCELERATION -----
def gen_TV_RSI_Slope_Acceleration(df, rsi_len=14, slope_win=3, accel_thr=2.0, **kw):
    """RSI slope = RSI - RSI.shift(slope_win).
    Acceleration = slope - slope.shift(slope_win).
    Long: acceleration crosses above +thr (momentum building).
    Short: acceleration crosses below -thr.
    """
    rl = int(rsi_len)
    sw = int(slope_win)
    th = float(accel_thr)
    c = df['close'].astype(float)
    rsi = _rsi(c, rl)
    slope = rsi - rsi.shift(sw)
    accel = slope - slope.shift(sw)
    long_cross = (accel > th) & (accel.shift(1) <= th)
    short_cross = (accel < -th) & (accel.shift(1) >= -th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_RSI_Slope_Acceleration():
    return {
        'rsi_len': ('int', 7, 21),
        'slope_win': ('int', 2, 8),
        'accel_thr': ('float', 1.0, 5.0),
    }


# ----- 3) VOL-PRICE DIVERGENCE SLOPE -----
def gen_TV_VolPrice_Divergence_Slope(df, win=30, slope_win=5, thr=0.05, **kw):
    """divergence = sign(price_change) - sign(vol_change) (range [-2, +2])
    Sustained positive divergence = volume confirms price up moves.
    Slope of divergence over slope_win bars.
    """
    w = int(win)
    sw = int(slope_win)
    th = float(thr)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    p_sign = np.sign(c.diff()).fillna(0)
    v_sign = np.sign(v.diff()).fillna(0)
    diverg = p_sign - v_sign
    avg_div = _sma(diverg, w)
    slope = avg_div - avg_div.shift(sw)
    long_cross = (slope > th) & (slope.shift(1) <= th)
    short_cross = (slope < -th) & (slope.shift(1) >= -th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_VolPrice_Divergence_Slope():
    return {
        'win': ('int', 15, 60),
        'slope_win': ('int', 3, 15),
        'thr': ('float', 0.02, 0.20),
    }


# ----- 4) BB MEAN-REVERSION LANDING -----
def gen_TV_BB_MeanRev_Landing(df, bb_len=20, bb_mult=2.0, **kw):
    """After bar PIERCES outside BB (low < lower OR high > upper),
    next bar's close LANDS BACK inside the bands = mean-reversion confirmed.
    Long entry: prev bar pierced lower, current close > lower.
    Short entry: prev bar pierced upper, current close < upper.
    """
    n = int(bb_len)
    m = float(bb_mult)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    mid = c.rolling(n, min_periods=n).mean()
    sd = c.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + m * sd
    lower = mid - m * sd
    pierced_lower = l.shift(1) < lower.shift(1)
    pierced_upper = h.shift(1) > upper.shift(1)
    landed_back_long = pierced_lower & (c > lower)
    landed_back_short = pierced_upper & (c < upper)
    long_entry = landed_back_long & ~(landed_back_long.shift(1).fillna(False))
    short_entry = landed_back_short & ~(landed_back_short.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_BB_MeanRev_Landing():
    return {
        'bb_len': ('int', 14, 35),
        'bb_mult': ('float', 1.5, 3.0),
    }


# ----- 5) TWO-SPEED MA DIFFERENCE -----
def gen_TV_TwoSpeed_MA_Diff(df, fast=12, slow=26, z_win=60, z_thr=1.5, **kw):
    """diff = EMA(fast) - EMA(slow). Normalize by close to get pct.
    z-score over z_win. Long: z > +z_thr cross. Short: z < -z_thr.
    Distinct from MACD which compares to signal-line.
    """
    f = int(fast)
    s = int(slow)
    zw = int(z_win)
    th = float(z_thr)
    c = df['close'].astype(float)
    diff = (_ema(c, f) - _ema(c, s)) / c
    mu = diff.rolling(zw, min_periods=zw).mean()
    sd = diff.rolling(zw, min_periods=zw).std(ddof=0).replace(0.0, np.nan)
    z = (diff - mu) / sd
    long_cross = (z > th) & (z.shift(1) <= th)
    short_cross = (z < -th) & (z.shift(1) >= -th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_TwoSpeed_MA_Diff():
    return {
        'fast': ('int', 5, 20),
        'slow': ('int', 20, 50),
        'z_win': ('int', 30, 150),
        'z_thr': ('float', 1.0, 3.0),
    }


STRATEGY_EXPORT = {
    "TV_Compound_Return_Z": {
        "gen": gen_TV_Compound_Return_Z, "space": space_TV_Compound_Return_Z,
        "source": "mac_batch3728_compound_return_zscore_fade",
    },
    "TV_RSI_Slope_Acceleration": {
        "gen": gen_TV_RSI_Slope_Acceleration,
        "space": space_TV_RSI_Slope_Acceleration,
        "source": "mac_batch3728_RSI_2nd_derivative_acceleration",
    },
    "TV_VolPrice_Divergence_Slope": {
        "gen": gen_TV_VolPrice_Divergence_Slope,
        "space": space_TV_VolPrice_Divergence_Slope,
        "source": "mac_batch3728_vol_price_sign_divergence_slope",
    },
    "TV_BB_MeanRev_Landing": {
        "gen": gen_TV_BB_MeanRev_Landing, "space": space_TV_BB_MeanRev_Landing,
        "source": "mac_batch3728_BB_pierce_landing_back_revert",
    },
    "TV_TwoSpeed_MA_Diff": {
        "gen": gen_TV_TwoSpeed_MA_Diff, "space": space_TV_TwoSpeed_MA_Diff,
        "source": "mac_batch3728_EMA_diff_zscore_cross",
    },
}
