"""
Batch 3716 - Mac paralela wave m17 - 5 adaptive / pattern composite families.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3715.

 1. TV_KaufmanER_Cross         - Kaufman Efficiency Ratio standalone
                                  cross threshold (DISTINCT from sandbox
                                  TV_KAMA_Crossover which uses adaptive MA;
                                  this uses ER directly as signal)
 2. TV_Bollinger_BandWalk     - close riding upper/lower BB for N consecutive
                                  bars = trending (band walk pattern)
 3. TV_RangeWithinRange_Compression - current bar's range inside prior bar's
                                  range AND prior inside earlier (compression
                                  pattern, NEW)
 4. TV_VolumeBreakout_Asymmetric - one-sided big volume during consolidation,
                                  direction = side of large volume
 5. TV_HighLowRange_Slope     - slope of (H-L) range over N bars; expanding
                                  range = breakout setup, contracting = revert
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


# ----- 1) KAUFMAN EFFICIENCY RATIO CROSS -----
def gen_TV_KaufmanER_Cross(df, er_len=10, threshold=0.5, **kw):
    """ER = |close - close_n_ago| / sum(|close - close_prev|, n).
    ER ∈ [0, 1]. Close to 1 = strong directional move; 0 = pure noise.
    Long: ER crosses above threshold AND close > close_n_ago.
    Short: ER crosses above threshold AND close < close_n_ago.
    """
    n = int(er_len)
    th = float(threshold)
    c = df['close'].astype(float)
    direction = (c - c.shift(n)).abs()
    volatility = c.diff().abs().rolling(n, min_periods=n).sum()
    er = direction / volatility.replace(0.0, np.nan)
    cross = (er > th) & (er.shift(1) <= th)
    long_setup = cross & (c > c.shift(n))
    short_setup = cross & (c < c.shift(n))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_KaufmanER_Cross():
    return {
        'er_len': ('int', 7, 30),
        'threshold': ('float', 0.3, 0.75),
    }


# ----- 2) BOLLINGER BAND WALK -----
def gen_TV_Bollinger_BandWalk(df, bb_len=20, bb_mult=2.0, walk_bars=3, **kw):
    """Close > upper BB for `walk_bars` consecutive bars = trending up
    (band walking). Entry on first triggering bar.
    """
    n = int(bb_len)
    m = float(bb_mult)
    wb = int(walk_bars)
    c = df['close'].astype(float)
    mid = c.rolling(n, min_periods=n).mean()
    sd = c.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + m * sd
    lower = mid - m * sd
    above_upper = (c > upper).astype(int)
    below_lower = (c < lower).astype(int)
    walk_up = above_upper.rolling(wb, min_periods=wb).sum() == wb
    walk_dn = below_lower.rolling(wb, min_periods=wb).sum() == wb
    long_entry = walk_up & ~(walk_up.shift(1).fillna(False))
    short_entry = walk_dn & ~(walk_dn.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Bollinger_BandWalk():
    return {
        'bb_len': ('int', 10, 40),
        'bb_mult': ('float', 1.5, 3.0),
        'walk_bars': ('int', 2, 6),
    }


# ----- 3) RANGE-WITHIN-RANGE COMPRESSION -----
def gen_TV_RangeWithinRange_Compression(df, atr_len=14, **kw):
    """Pattern: current bar inside prior bar's range AND prior inside earlier
    (3-bar compression). Breakout direction = side of escape relative to
    middle of compression range.
    """
    al = int(atr_len)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    inside_1 = (h <= h.shift(1)) & (l >= l.shift(1))
    inside_2 = inside_1.shift(1).fillna(False) & (h.shift(1) <= h.shift(2)) & (l.shift(1) >= l.shift(2))
    compression = inside_1 & inside_2
    # Wait for next-bar breakout
    pre_high = h.shift(1).where(compression).ffill(limit=2)
    pre_low = l.shift(1).where(compression).ffill(limit=2)
    long_break = (c > pre_high) & compression.shift(1).fillna(False)
    short_break = (c < pre_low) & compression.shift(1).fillna(False)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_break.shift(1).fillna(False).astype(bool)] = 1
    sig[short_break.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_RangeWithinRange_Compression():
    return {
        'atr_len': ('int', 7, 30),
    }


# ----- 4) VOLUME BREAKOUT ASYMMETRIC -----
def gen_TV_VolumeBreakout_Asymmetric(df, win=20, asym_ratio=2.0, **kw):
    """During consolidation, one side (up or down) has dramatically higher
    volume than the other. That side wins the eventual breakout.
    Vol_up = sum(vol when close>open in window), Vol_dn = sum(vol when close<open).
    If Vol_up / Vol_dn > asym_ratio AND price breaks N-bar high -> LONG.
    Symmetric for SHORT.
    """
    w = int(win)
    ar = float(asym_ratio)
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    up_vol = v.where(c > o, 0.0).rolling(w, min_periods=w).sum()
    dn_vol = v.where(c < o, 0.0).rolling(w, min_periods=w).sum().replace(0.0, np.nan)
    asym = up_vol / dn_vol
    pre_high = h.rolling(w, min_periods=w).max().shift(1)
    pre_low = l.rolling(w, min_periods=w).min().shift(1)
    long_break = (asym > ar) & (c > pre_high)
    short_break = ((1.0 / asym) > ar) & (c < pre_low)
    long_entry = long_break & ~(long_break.shift(1).fillna(False))
    short_entry = short_break & ~(short_break.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_VolumeBreakout_Asymmetric():
    return {
        'win': ('int', 10, 40),
        'asym_ratio': ('float', 1.5, 3.5),
    }


# ----- 5) HIGH-LOW RANGE SLOPE -----
def gen_TV_HighLowRange_Slope(df, range_win=20, slope_win=5, expansion_thr_pct=20, **kw):
    """Slope of bar range = (current avg_range - past avg_range) / past avg_range.
    Expanding (slope > +thr): breakout setup, direction by recent close pos.
    Contracting (slope < -thr): mean revert setup.
    """
    rw = int(range_win)
    sw = int(slope_win)
    et = float(expansion_thr_pct) / 100.0
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    avg_rng = _sma(h - l, rw)
    past_avg = avg_rng.shift(sw)
    slope = (avg_rng - past_avg) / past_avg.replace(0.0, np.nan)
    short_ret = c.pct_change().rolling(5, min_periods=5).sum()
    expanding = slope > et
    contracting = slope < -et
    long_setup = (expanding & (short_ret > 0)) | (contracting & (short_ret < 0))
    short_setup = (expanding & (short_ret < 0)) | (contracting & (short_ret > 0))
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_HighLowRange_Slope():
    return {
        'range_win': ('int', 10, 40),
        'slope_win': ('int', 3, 15),
        'expansion_thr_pct': ('int', 10, 40),
    }


STRATEGY_EXPORT = {
    "TV_KaufmanER_Cross": {
        "gen": gen_TV_KaufmanER_Cross, "space": space_TV_KaufmanER_Cross,
        "source": "mac_batch3716_Kaufman_EfficiencyRatio_standalone",
    },
    "TV_Bollinger_BandWalk": {
        "gen": gen_TV_Bollinger_BandWalk, "space": space_TV_Bollinger_BandWalk,
        "source": "mac_batch3716_Bollinger_BandWalk_pattern",
    },
    "TV_RangeWithinRange_Compression": {
        "gen": gen_TV_RangeWithinRange_Compression,
        "space": space_TV_RangeWithinRange_Compression,
        "source": "mac_batch3716_3bar_compression_breakout",
    },
    "TV_VolumeBreakout_Asymmetric": {
        "gen": gen_TV_VolumeBreakout_Asymmetric,
        "space": space_TV_VolumeBreakout_Asymmetric,
        "source": "mac_batch3716_asymmetric_volume_breakout",
    },
    "TV_HighLowRange_Slope": {
        "gen": gen_TV_HighLowRange_Slope, "space": space_TV_HighLowRange_Slope,
        "source": "mac_batch3716_HL_range_slope_regime",
    },
}
