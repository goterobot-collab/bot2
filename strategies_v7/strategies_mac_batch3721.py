"""
Batch 3721 - Mac paralela wave m22 - 5 robust stats + bar shape families.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3720.

 1. TV_Doji_Trend_Reversal      - small body + big range at extreme of EMA
                                   trend = exhaustion reversal (NEW)
 2. TV_RobustZScore_MAD         - Median Absolute Deviation z-score
                                   (robust vs outliers, distinct from
                                   sandbox/Mac z-score variants which use std)
 3. TV_Mid_Close_Asymmetry      - (close - bar_midpoint) / range = where close
                                   lands within bar; asymmetry indicator (NEW)
 4. TV_Trimmed_Mean_Regime      - trimmed mean of returns (drop 10% tails)
                                   regime indicator (robust mean, NEW)
 5. TV_Volatility_Cone_Position - realized vol percentile in 252-bar cone,
                                   regime entry/exit (NEW framing)
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _atr(df, n):
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()


# ----- 1) DOJI + TREND REVERSAL -----
def gen_TV_Doji_Trend_Reversal(df, ema_len=50, body_pct_max=10, atr_len=14,
                                  range_atr_min=0.8, **kw):
    """Doji = body <= body_pct_max% of bar's range AND range >= range_atr_min*ATR.
    At extreme of EMA trend (close > EMA*1.05 or < EMA*0.95), doji = exhaustion.
    Long: doji while close < EMA*0.95.
    Short: doji while close > EMA*1.05.
    """
    el = int(ema_len)
    bm = float(body_pct_max) / 100.0
    al = int(atr_len)
    rm = float(range_atr_min)
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    body = (c - o).abs()
    bar_range = (h - l).replace(0.0, np.nan)
    body_pct = body / bar_range
    atr = _atr(df, al)
    is_doji = (body_pct <= bm) & (bar_range >= rm * atr)
    ema = _ema(c, el)
    long_setup = is_doji & (c < ema * 0.95)
    short_setup = is_doji & (c > ema * 1.05)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Doji_Trend_Reversal():
    return {
        'ema_len': ('int', 20, 100),
        'body_pct_max': ('int', 5, 20),
        'atr_len': ('int', 7, 30),
        'range_atr_min': ('float', 0.5, 1.5),
    }


# ----- 2) ROBUST Z-SCORE (MAD) -----
def gen_TV_RobustZScore_MAD(df, win=30, threshold=2.5, **kw):
    """Modified z-score = 0.6745 * (return - median) / MAD
    where MAD = median(|return - median|).
    Robust to outliers vs std-based z. Threshold cross = mean reversal.
    """
    w = int(win)
    th = float(threshold)
    c = df['close'].astype(float)
    ret = c.pct_change()
    med = ret.rolling(w, min_periods=w).median()
    abs_dev = (ret - med).abs()
    mad = abs_dev.rolling(w, min_periods=w).median().replace(0.0, np.nan)
    rz = 0.6745 * (ret - med) / mad
    long_cross = (rz < -th) & (rz.shift(1) >= -th)  # extreme down -> long
    short_cross = (rz > th) & (rz.shift(1) <= th)  # extreme up -> short
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_RobustZScore_MAD():
    return {
        'win': ('int', 15, 80),
        'threshold': ('float', 1.5, 4.0),
    }


# ----- 3) MID-CLOSE ASYMMETRY -----
def gen_TV_Mid_Close_Asymmetry(df, win=20, asym_thr=0.6, **kw):
    """asym = (close - midpoint) / (high - low). Range [-0.5, 0.5].
    Rolling mean of asym shows where closes tend to land.
    Asym crossing > +thr * 0.5 = persistent close in upper half (bullish)
    Asym crossing < -thr * 0.5 = persistent close in lower half (bearish)
    """
    w = int(win)
    th = float(asym_thr)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    midpoint = (h + l) / 2.0
    bar_range = (h - l).replace(0.0, np.nan)
    asym = (c - midpoint) / bar_range
    avg_asym = asym.rolling(w, min_periods=w).mean()
    long_cross = (avg_asym > th * 0.5) & (avg_asym.shift(1) <= th * 0.5)
    short_cross = (avg_asym < -th * 0.5) & (avg_asym.shift(1) >= -th * 0.5)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Mid_Close_Asymmetry():
    return {
        'win': ('int', 10, 50),
        'asym_thr': ('float', 0.05, 0.6),
    }


# ----- 4) MEAN-MEDIAN DIFFERENCE REGIME -----
def gen_TV_MeanMedian_Diff_Regime(df, win=30, threshold_pct=0.05, **kw):
    """Difference between rolling mean and median of returns.
    Diff > 0: mean above median = right-skewed (positive tails)
    Diff < 0: left-skewed (negative tails)
    Cross zero = distribution shape change (regime).
    Vectorized robust skew proxy without rolling.apply slowness.
    """
    w = int(win)
    th = float(threshold_pct) / 100.0
    c = df['close'].astype(float)
    ret = c.pct_change()
    mean_r = ret.rolling(w, min_periods=w).mean()
    med_r = ret.rolling(w, min_periods=w).median()
    diff = mean_r - med_r
    long_cross = (diff > th) & (diff.shift(1) <= th)
    short_cross = (diff < -th) & (diff.shift(1) >= -th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_MeanMedian_Diff_Regime():
    return {
        'win': ('int', 15, 60),
        'threshold_pct': ('float', 0.01, 0.30),
    }


# ----- 5) VOLATILITY CONE POSITION -----
def gen_TV_Volatility_Cone_Position(df, vol_win=20, cone_win=252, low_pct=20,
                                       high_pct=80, **kw):
    """Realized vol = stdev(returns, vol_win). Percentile rank in cone_win.
    Low pct (<low_pct): low vol regime -> follow recent direction.
    High pct (>high_pct): high vol regime -> mean revert recent direction.
    """
    vw = int(vol_win)
    cw = int(cone_win)
    lp = float(low_pct)
    hp = float(high_pct)
    c = df['close'].astype(float)
    ret = c.pct_change()
    vol = ret.rolling(vw, min_periods=vw).std(ddof=0)
    vol_rank = vol.rolling(cw, min_periods=vw * 3).rank(pct=True) * 100
    short_ret = ret.rolling(5, min_periods=5).sum()
    low_vol = vol_rank < lp
    high_vol = vol_rank > hp
    long_setup = (low_vol & (short_ret > 0)) | (high_vol & (short_ret < 0))
    short_setup = (low_vol & (short_ret < 0)) | (high_vol & (short_ret > 0))
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Volatility_Cone_Position():
    return {
        'vol_win': ('int', 10, 40),
        'cone_win': ('int', 100, 400),
        'low_pct': ('int', 10, 30),
        'high_pct': ('int', 70, 90),
    }


STRATEGY_EXPORT = {
    "TV_Doji_Trend_Reversal": {
        "gen": gen_TV_Doji_Trend_Reversal,
        "space": space_TV_Doji_Trend_Reversal,
        "source": "mac_batch3721_doji_at_trend_extreme_reversal",
    },
    "TV_RobustZScore_MAD": {
        "gen": gen_TV_RobustZScore_MAD, "space": space_TV_RobustZScore_MAD,
        "source": "mac_batch3721_modified_zscore_MAD_robust",
    },
    "TV_Mid_Close_Asymmetry": {
        "gen": gen_TV_Mid_Close_Asymmetry, "space": space_TV_Mid_Close_Asymmetry,
        "source": "mac_batch3721_mid_close_asymmetry_avg",
    },
    "TV_MeanMedian_Diff_Regime": {
        "gen": gen_TV_MeanMedian_Diff_Regime,
        "space": space_TV_MeanMedian_Diff_Regime,
        "source": "mac_batch3721_mean_median_diff_robust_skew_proxy",
    },
    "TV_Volatility_Cone_Position": {
        "gen": gen_TV_Volatility_Cone_Position,
        "space": space_TV_Volatility_Cone_Position,
        "source": "mac_batch3721_volatility_cone_percentile_position",
    },
}
