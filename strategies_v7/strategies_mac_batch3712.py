"""
Batch 3712 - Mac paralela wave m13 - 5 vectorized regime/structural families.

REWRITTEN to be FULLY VECTORIZED (no rolling.apply) after first version
hung the hunter for 1+ hour on Sample Entropy O(n^2)/window.

All low-param (1-3 each), pickle-safe, signal int {-1,0,1}, .shift(1).
ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3711.

 1. TV_RealizedVol_Z_Regime    - rolling stdev z-score regime switch
                                  (vectorized regime: low vol -> trend follow,
                                  high vol -> mean revert) - novel framing
 2. TV_BinarySign_Persistence  - count matching sign(return) bars in window
                                  (vectorized "predictability" proxy, NEW)
 3. TV_RangeBreakout_Z         - z-score of bar range vs rolling mean range;
                                  surge breakout direction by close pos (NEW)
 4. TV_AutoCorr1_Approx        - vectorized 1-lag AC via rolling sums
                                  (vs slow np.corrcoef per window, NEW)
 5. TV_RealizedSkew_Surge      - rolling skewness via vectorized formula;
                                  surge -> mean reversal (NEW)
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


# ----- 1) REALIZED VOL Z-SCORE REGIME -----
def gen_TV_RealizedVol_Z_Regime(df, vol_win=20, z_win=100, z_low=-0.5,
                                  z_high=1.0, **kw):
    """Rolling stdev of returns over vol_win, z-scored over z_win bars.
    z < z_low: low vol regime -> follow trend (long if EMA20 > EMA50).
    z > z_high: high vol regime -> mean revert (long if close < SMA(20)).
    """
    vw = int(vol_win)
    zw = int(z_win)
    zl = float(z_low)
    zh = float(z_high)
    c = df['close'].astype(float)
    ret = c.pct_change()
    sd = ret.rolling(vw, min_periods=vw).std(ddof=0)
    mu = sd.rolling(zw, min_periods=zw).mean()
    sig_sd = sd.rolling(zw, min_periods=zw).std(ddof=0).replace(0.0, np.nan)
    z = (sd - mu) / sig_sd
    ema_s = _ema(c, 20)
    ema_l = _ema(c, 50)
    sma_s = _sma(c, 20)
    low_vol = z < zl
    high_vol = z > zh
    long_trend = low_vol & (ema_s > ema_l) & (ema_s.shift(1) <= ema_l.shift(1))
    short_trend = low_vol & (ema_s < ema_l) & (ema_s.shift(1) >= ema_l.shift(1))
    long_revert = high_vol & (c < sma_s) & (c.shift(1) >= sma_s.shift(1))
    short_revert = high_vol & (c > sma_s) & (c.shift(1) <= sma_s.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(long_trend | long_revert).shift(1).fillna(False).astype(bool)] = 1
    sig[(short_trend | short_revert).shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_RealizedVol_Z_Regime():
    return {
        'vol_win': ('int', 10, 40),
        'z_win': ('int', 50, 200),
        'z_low': ('float', -1.5, 0.0),
        'z_high': ('float', 0.5, 2.0),
    }


# ----- 2) BINARY SIGN PERSISTENCE -----
def gen_TV_BinarySign_Persistence(df, win=30, persist_pct=70, **kw):
    """Vectorized: rolling count of bars where sign(return) matches majority.
    persistence = max(up_count, down_count) / win * 100.
    HIGH persistence (>= persist_pct%) = trend regime.
    Direction = majority sign of recent returns.
    """
    w = int(win)
    pp = float(persist_pct)
    c = df['close'].astype(float)
    ret = c.pct_change()
    up_bars = (ret > 0).astype(int).rolling(w, min_periods=w).sum()
    persistence = pd.concat([up_bars, w - up_bars], axis=1).max(axis=1) / w * 100
    long_setup = (persistence >= pp) & (up_bars > w / 2)
    short_setup = (persistence >= pp) & (up_bars <= w / 2)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_BinarySign_Persistence():
    return {
        'win': ('int', 15, 60),
        'persist_pct': ('int', 55, 80),
    }


# ----- 3) RANGE BREAKOUT Z-SCORE -----
def gen_TV_RangeBreakout_Z(df, win=20, z_threshold=2.0, **kw):
    """Bar range = high - low. z = (bar_range - mean_range) / std_range
    over rolling window.
    Surge (z > threshold): direction = sign(close - midpoint of bar).
    """
    w = int(win)
    zt = float(z_threshold)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    rng = h - l
    mu = rng.rolling(w, min_periods=w).mean()
    sd = rng.rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
    z = (rng - mu) / sd
    midpoint = (h + l) / 2.0
    surge = (z > zt) & (z.shift(1) <= zt)
    long_setup = surge & (c > midpoint)
    short_setup = surge & (c < midpoint)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_RangeBreakout_Z():
    return {
        'win': ('int', 10, 50),
        'z_threshold': ('float', 1.5, 3.5),
    }


# ----- 4) AUTOCORR1 APPROXIMATION (vectorized) -----
def gen_TV_AutoCorr1_Approx(df, win=50, **kw):
    """1-lag autocorrelation of returns, computed VECTORIZED via rolling sums:
       AC1 ≈ sum((r[t] - mu)*(r[t-1] - mu)) / sum((r[t] - mu)^2) over win.
    Positive AC: momentum regime -> follow recent direction.
    Negative AC: revert regime -> oppose recent direction.
    """
    w = int(win)
    c = df['close'].astype(float)
    ret = c.pct_change()
    # Rolling mean
    mu = ret.rolling(w, min_periods=w).mean()
    # Rolling cov(r[t], r[t-1])
    r_prev = ret.shift(1)
    cov = (ret * r_prev).rolling(w, min_periods=w).mean() - mu * mu.shift(1)
    var = (ret * ret).rolling(w, min_periods=w).mean() - mu * mu
    ac = cov / var.replace(0.0, np.nan)
    short_ret = ret.rolling(5, min_periods=5).sum()
    pos_regime = ac > 0
    neg_regime = ac < 0
    long_setup = (pos_regime & (short_ret > 0)) | (neg_regime & (short_ret < 0))
    short_setup = (pos_regime & (short_ret < 0)) | (neg_regime & (short_ret > 0))
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_AutoCorr1_Approx():
    return {
        'win': ('int', 30, 100),
    }


# ----- 5) REALIZED SKEW SURGE (vectorized) -----
def gen_TV_RealizedSkew_Surge(df, win=30, threshold=1.0, **kw):
    """Rolling skewness of returns via vectorized formula:
       skew = E[(r-mu)^3] / std^3.
    Skew > +threshold (right tail surge) -> SHORT (mean revert from up surge).
    Skew < -threshold (left tail surge)  -> LONG (mean revert from down surge).
    """
    w = int(win)
    th = float(threshold)
    c = df['close'].astype(float)
    ret = c.pct_change()
    mu = ret.rolling(w, min_periods=w).mean()
    sd = ret.rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
    # Use pandas rolling skew (vectorized internal)
    skew = ret.rolling(w, min_periods=w).skew()
    long_trigger = (skew < -th) & (skew.shift(1) >= -th)
    short_trigger = (skew > th) & (skew.shift(1) <= th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_trigger.shift(1).fillna(False).astype(bool)] = 1
    sig[short_trigger.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_RealizedSkew_Surge():
    return {
        'win': ('int', 15, 60),
        'threshold': ('float', 0.6, 2.5),
    }


STRATEGY_EXPORT = {
    "TV_RealizedVol_Z_Regime": {
        "gen": gen_TV_RealizedVol_Z_Regime,
        "space": space_TV_RealizedVol_Z_Regime,
        "source": "mac_batch3712_v2_vectorized_realized_vol_z_regime",
    },
    "TV_BinarySign_Persistence": {
        "gen": gen_TV_BinarySign_Persistence,
        "space": space_TV_BinarySign_Persistence,
        "source": "mac_batch3712_v2_vectorized_binary_sign_persistence",
    },
    "TV_RangeBreakout_Z": {
        "gen": gen_TV_RangeBreakout_Z,
        "space": space_TV_RangeBreakout_Z,
        "source": "mac_batch3712_v2_vectorized_range_zscore",
    },
    "TV_AutoCorr1_Approx": {
        "gen": gen_TV_AutoCorr1_Approx,
        "space": space_TV_AutoCorr1_Approx,
        "source": "mac_batch3712_v2_vectorized_AC1_approx",
    },
    "TV_RealizedSkew_Surge": {
        "gen": gen_TV_RealizedSkew_Surge,
        "space": space_TV_RealizedSkew_Surge,
        "source": "mac_batch3712_v2_vectorized_realized_skew",
    },
}
