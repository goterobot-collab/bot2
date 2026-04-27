"""
Batch 3724 - Mac paralela wave m25 - 5 adaptive / persistence / quintile families.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3723.

 1. TV_VolAdapted_RSI         - RSI with vol-adapted period (DISTINCT from
                                 sandbox/Mac Adaptive_RSI which uses Kaufman
                                 efficiency)
 2. TV_TrendPersistence_Score - composite: streak length + ROC sign + RSI
 3. TV_Quintile_BinTransition - close moves up across quintile boundaries
                                 of recent N-bar range
 4. TV_RangeContraction_PreBreak - 3 bars contracting range AND breakout
                                 (different from m17 RangeWithinRange which
                                 is inside-bar pattern)
 5. TV_CumPriceImpact_Slope   - cumulative |return|/volume slope cross
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


# ----- 1) VOL-ADAPTED RSI -----
def gen_TV_VolAdapted_RSI(df, base_len=14, vol_win=50, vol_pct_low=30,
                            vol_pct_high=70, ob=70, os=30, **kw):
    """RSI with adaptive period: in low vol, use longer period (less noise).
    In high vol, use shorter period (more responsive).
    base_len = baseline. period = base_len * (1.5 if low_vol else 0.5 if high_vol else 1).
    """
    bl = int(base_len)
    vw = int(vol_win)
    plo = float(vol_pct_low)
    phi = float(vol_pct_high)
    obl = float(ob)
    osl = float(os)
    c = df['close'].astype(float)
    ret = c.pct_change()
    vol = ret.rolling(vw, min_periods=vw).std(ddof=0)
    vol_rank = vol.rolling(vw * 3, min_periods=vw).rank(pct=True) * 100
    period_short = max(3, bl // 2)
    period_long = bl * 2
    rsi_base = _rsi(c, bl)
    rsi_short = _rsi(c, period_short)
    rsi_long = _rsi(c, period_long)
    adaptive = pd.Series(np.nan, index=df.index)
    adaptive = rsi_base.where((vol_rank >= plo) & (vol_rank <= phi), adaptive)
    adaptive = rsi_long.where(vol_rank < plo, adaptive)
    adaptive = rsi_short.where(vol_rank > phi, adaptive)
    long_cross = (adaptive > osl) & (adaptive.shift(1) <= osl)
    short_cross = (adaptive < obl) & (adaptive.shift(1) >= obl)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_VolAdapted_RSI():
    return {
        'base_len': ('int', 7, 21),
        'vol_win': ('int', 30, 80),
        'vol_pct_low': ('int', 20, 40),
        'vol_pct_high': ('int', 60, 80),
        'ob': ('int', 65, 80),
        'os': ('int', 20, 35),
    }


# ----- 2) TREND PERSISTENCE COMPOSITE SCORE -----
def gen_TV_TrendPersistence_Score(df, win=20, score_thr=2.0, **kw):
    """Composite score (z-summed):
       - streak of same-direction closes / win
       - sign of ROC(win)
       - RSI distance from 50 / 50
    Score = sum of the three normalized signals.
    Long: score crosses above +thr.
    Short: below -thr.
    """
    w = int(win)
    th = float(score_thr)
    c = df['close'].astype(float)
    ret = c.pct_change()
    sign = np.sign(ret).fillna(0)
    # Streak score
    streak_signed = sign.rolling(w, min_periods=w).sum() / w
    # ROC score
    roc = (c / c.shift(w) - 1.0)
    roc_score = np.tanh(roc * 10)
    # RSI distance from 50 normalized
    rsi = _rsi(c, w)
    rsi_score = (rsi - 50) / 50
    composite = streak_signed + roc_score + rsi_score
    long_cross = (composite > th) & (composite.shift(1) <= th)
    short_cross = (composite < -th) & (composite.shift(1) >= -th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_TrendPersistence_Score():
    return {
        'win': ('int', 10, 50),
        'score_thr': ('float', 0.8, 2.5),
    }


# ----- 3) QUINTILE BIN TRANSITION -----
def gen_TV_Quintile_BinTransition(df, win=50, **kw):
    """Bin close into 5 quintiles of N-bar range:
       q0: [0, 20%)  q1: [20, 40%)  q2: [40, 60%)  q3: [60, 80%)  q4: [80, 100%]
    Long: transition q0/q1 -> q3/q4 (large up move).
    Short: transition q3/q4 -> q0/q1.
    """
    w = int(win)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    hh = h.rolling(w, min_periods=w).max()
    ll = l.rolling(w, min_periods=w).min()
    rng = (hh - ll).replace(0.0, np.nan)
    pos = (c - ll) / rng  # 0..1
    bin_now = (pos * 5).clip(0, 4).astype(float).round()
    bin_prev = bin_now.shift(1)
    long_jump = (bin_now >= 3) & (bin_prev <= 1)
    short_jump = (bin_now <= 1) & (bin_prev >= 3)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_jump.shift(1).fillna(False).astype(bool)] = 1
    sig[short_jump.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Quintile_BinTransition():
    return {
        'win': ('int', 20, 120),
    }


# ----- 4) RANGE CONTRACTION PRE-BREAK -----
def gen_TV_RangeContraction_PreBreak(df, win=20, contraction_bars=3, **kw):
    """3+ consecutive bars where (high-low) < SMA(high-low, win).
    Then breakout: close > N-bar high (long) or < N-bar low (short).
    """
    w = int(win)
    cb = int(contraction_bars)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    bar_range = h - l
    avg_range = _sma(bar_range, w)
    contracted = bar_range < avg_range
    contraction_streak = contracted.rolling(cb, min_periods=cb).sum() == cb
    pre_high = h.rolling(w, min_periods=w).max().shift(1)
    pre_low = l.rolling(w, min_periods=w).min().shift(1)
    long_break = contraction_streak.shift(1).fillna(False) & (c > pre_high)
    short_break = contraction_streak.shift(1).fillna(False) & (c < pre_low)
    long_entry = long_break & ~(long_break.shift(1).fillna(False))
    short_entry = short_break & ~(short_break.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_RangeContraction_PreBreak():
    return {
        'win': ('int', 10, 40),
        'contraction_bars': ('int', 2, 6),
    }


# ----- 5) CUMULATIVE PRICE IMPACT SLOPE -----
def gen_TV_CumPriceImpact_Slope(df, win=20, slope_win=10, **kw):
    """Cumulative price impact = cumsum(|return|/volume).
    Slope = cum_impact - cum_impact.shift(slope_win).
    Sustained high impact = trending; declining impact = quiet.
    Direction = sign of recent return.
    Long: slope DECLINING (calming) AND recent ret > 0.
    Short: slope DECLINING AND recent ret < 0.
    Slope rising (impact climbing) = avoid (mean revert risk).
    """
    w = int(win)
    sw = int(slope_win)
    c = df['close'].astype(float)
    v = df['volume'].astype(float).replace(0.0, np.nan)
    ret = c.pct_change()
    impact = ret.abs() / v
    cum_impact = impact.fillna(0).cumsum()
    slope = cum_impact - cum_impact.shift(sw)
    avg_slope = _sma(slope, w)
    declining = slope < avg_slope
    short_ret = ret.rolling(5, min_periods=5).sum()
    long_setup = declining & (short_ret > 0) & ~(declining.shift(1).fillna(False) & (short_ret.shift(1) > 0))
    short_setup = declining & (short_ret < 0) & ~(declining.shift(1).fillna(False) & (short_ret.shift(1) < 0))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_CumPriceImpact_Slope():
    return {
        'win': ('int', 10, 50),
        'slope_win': ('int', 5, 30),
    }


STRATEGY_EXPORT = {
    "TV_VolAdapted_RSI": {
        "gen": gen_TV_VolAdapted_RSI, "space": space_TV_VolAdapted_RSI,
        "source": "mac_batch3724_RSI_vol_adapted_period",
    },
    "TV_TrendPersistence_Score": {
        "gen": gen_TV_TrendPersistence_Score,
        "space": space_TV_TrendPersistence_Score,
        "source": "mac_batch3724_streak_ROC_RSI_composite",
    },
    "TV_Quintile_BinTransition": {
        "gen": gen_TV_Quintile_BinTransition,
        "space": space_TV_Quintile_BinTransition,
        "source": "mac_batch3724_quintile_bin_jump",
    },
    "TV_RangeContraction_PreBreak": {
        "gen": gen_TV_RangeContraction_PreBreak,
        "space": space_TV_RangeContraction_PreBreak,
        "source": "mac_batch3724_range_contraction_then_break",
    },
    "TV_CumPriceImpact_Slope": {
        "gen": gen_TV_CumPriceImpact_Slope,
        "space": space_TV_CumPriceImpact_Slope,
        "source": "mac_batch3724_cumulative_impact_slope",
    },
}
