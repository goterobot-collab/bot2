"""
Batch 3718 - Mac paralela wave m19 - 5 cycle / regime / advanced indicators.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3717.

 1. TV_TimeWeighted_AvgPrice    - TWAP cross (NOT VWAP - uniform time weight,
                                   NEW)
 2. TV_Sigmoid_Momentum         - sigmoid-transformed N-bar return for bounded
                                   momentum signal (NEW)
 3. TV_BB_Penetration_Pct       - % of bar's range outside BB bands (degree
                                   of break) - NEW BB framing
 4. TV_TickCount_Cumulative     - cumulative count of up vs down bars,
                                   slope sign cross
 5. TV_NormalizedHL_Position    - (close - low_n) / (high_n - low_n) regime
                                   indicator (DISTINCT from Williams %R which
                                   maps to [-100,0]; this maps [0,1] and uses
                                   different cross logic)
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


# ----- 1) TIME-WEIGHTED AVERAGE PRICE -----
def gen_TV_TimeWeighted_AvgPrice(df, twap_len=24, signal_thr_pct=1.0, **kw):
    """TWAP = SMA of typical price over twap_len (uniform time weight).
    Long: close > TWAP * (1 + signal_thr_pct/100) AND fresh transition.
    Short: close < TWAP * (1 - signal_thr_pct/100).
    """
    tl = int(twap_len)
    th = float(signal_thr_pct) / 100.0
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    typical = (h + l + c) / 3.0
    twap = _sma(typical, tl)
    upper = twap * (1 + th)
    lower = twap * (1 - th)
    long_setup = c > upper
    short_setup = c < lower
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_TimeWeighted_AvgPrice():
    return {
        'twap_len': ('int', 12, 100),
        'signal_thr_pct': ('float', 0.3, 3.0),
    }


# ----- 2) SIGMOID MOMENTUM -----
def gen_TV_Sigmoid_Momentum(df, mom_len=14, scale=10.0, threshold=0.7, **kw):
    """sig_mom = sigmoid(scale * ROC(mom_len)). Bounded in [0, 1].
    Long: sig_mom crosses above threshold (high positive momentum confirmed).
    Short: sig_mom crosses below (1 - threshold).
    """
    ml = int(mom_len)
    sc = float(scale)
    th = float(threshold)
    c = df['close'].astype(float)
    roc = (c / c.shift(ml) - 1.0)
    sig_mom = 1.0 / (1.0 + np.exp(-sc * roc))
    long_cross = (sig_mom > th) & (sig_mom.shift(1) <= th)
    short_cross = (sig_mom < (1 - th)) & (sig_mom.shift(1) >= (1 - th))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Sigmoid_Momentum():
    return {
        'mom_len': ('int', 5, 30),
        'scale': ('float', 5.0, 30.0),
        'threshold': ('float', 0.6, 0.85),
    }


# ----- 3) BB PENETRATION PERCENT -----
def gen_TV_BB_Penetration_Pct(df, bb_len=20, bb_mult=2.0, pct_thr=50, **kw):
    """% of bar's range outside BB bands.
    upper_pen = max(0, high - bb_upper) / (high - low) * 100.
    lower_pen = max(0, bb_lower - low) / (high - low) * 100.
    Long: lower_pen > pct_thr (deep break below = oversold reversion).
    Short: upper_pen > pct_thr.
    """
    n = int(bb_len)
    m = float(bb_mult)
    th = float(pct_thr)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    mid = c.rolling(n, min_periods=n).mean()
    sd = c.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + m * sd
    lower = mid - m * sd
    bar_rng = (h - l).replace(0.0, np.nan)
    upper_pen = (h - upper).clip(lower=0) / bar_rng * 100
    lower_pen = (lower - l).clip(lower=0) / bar_rng * 100
    long_setup = lower_pen > th
    short_setup = upper_pen > th
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_BB_Penetration_Pct():
    return {
        'bb_len': ('int', 14, 35),
        'bb_mult': ('float', 1.5, 3.0),
        'pct_thr': ('int', 30, 80),
    }


# ----- 4) TICK-COUNT CUMULATIVE SLOPE -----
def gen_TV_TickCount_Cumulative(df, slope_win=20, **kw):
    """Cumulative tick = running sum of (+1 if up, -1 if down).
    Slope = cum_tick - cum_tick.shift(slope_win).
    Long: slope crosses above 0. Short: below 0.
    """
    sw = int(slope_win)
    c = df['close'].astype(float)
    pc = c.shift(1)
    tick = np.where(c > pc, 1, np.where(c < pc, -1, 0))
    cum = pd.Series(tick, index=df.index).cumsum()
    slope = cum - cum.shift(sw)
    long_cross = (slope > 0) & (slope.shift(1) <= 0)
    short_cross = (slope < 0) & (slope.shift(1) >= 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_TickCount_Cumulative():
    return {
        'slope_win': ('int', 10, 60),
    }


# ----- 5) NORMALIZED HIGH-LOW POSITION -----
def gen_TV_NormalizedHL_Position(df, n=20, ob=0.85, os=0.15, **kw):
    """pos = (close - lowest_low(n)) / (highest_high(n) - lowest_low(n))
    Range [0, 1]. NOT same logic as Williams %R (which crosses thresholds).
    Here: regime indicator with momentum confirm.
    Long: pos > ob AND pos.shift(1) <= ob (cross UP into top zone, momentum).
    Short: pos < os AND cross DOWN into bottom zone.
    """
    n = int(n)
    obl = float(ob)
    osl = float(os)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    hh = h.rolling(n, min_periods=n).max()
    ll = l.rolling(n, min_periods=n).min()
    rng = (hh - ll).replace(0.0, np.nan)
    pos = (c - ll) / rng
    long_cross = (pos > obl) & (pos.shift(1) <= obl)
    short_cross = (pos < osl) & (pos.shift(1) >= osl)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_NormalizedHL_Position():
    return {
        'n': ('int', 10, 50),
        'ob': ('float', 0.75, 0.95),
        'os': ('float', 0.05, 0.25),
    }


STRATEGY_EXPORT = {
    "TV_TimeWeighted_AvgPrice": {
        "gen": gen_TV_TimeWeighted_AvgPrice,
        "space": space_TV_TimeWeighted_AvgPrice,
        "source": "mac_batch3718_TWAP_typical_price",
    },
    "TV_Sigmoid_Momentum": {
        "gen": gen_TV_Sigmoid_Momentum, "space": space_TV_Sigmoid_Momentum,
        "source": "mac_batch3718_sigmoid_bounded_momentum",
    },
    "TV_BB_Penetration_Pct": {
        "gen": gen_TV_BB_Penetration_Pct, "space": space_TV_BB_Penetration_Pct,
        "source": "mac_batch3718_BB_penetration_pct_break",
    },
    "TV_TickCount_Cumulative": {
        "gen": gen_TV_TickCount_Cumulative,
        "space": space_TV_TickCount_Cumulative,
        "source": "mac_batch3718_tick_count_cumulative_slope",
    },
    "TV_NormalizedHL_Position": {
        "gen": gen_TV_NormalizedHL_Position,
        "space": space_TV_NormalizedHL_Position,
        "source": "mac_batch3718_normalized_HL_position_threshold_cross",
    },
}
