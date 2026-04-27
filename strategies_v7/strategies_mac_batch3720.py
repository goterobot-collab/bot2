"""
Batch 3720 - Mac paralela wave m21 - 5 levels / math-based families.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3719.

 1. TV_Murray_Math_Octave       - Murray Math 1/8 octave levels of N-bar
                                   range, bounce off levels (NEW)
 2. TV_VolumeWeighted_Momentum  - momentum * volume normalization
                                   (vol-amplified ROC)
 3. TV_GoldenRatio_Channel      - 1.618 / 0.618 channel from swing extremes
                                   (NEW; distinct from sandbox Linear_Regression
                                   Channel and Andrews Pitchfork in m4)
 4. TV_PriceVolume_Correlation  - rolling correlation of price moves and
                                   volume changes; sign cross
 5. TV_RangePctile_Cross        - current bar range percentile rank vs history
                                   crosses threshold
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


# ----- 1) MURRAY MATH OCTAVE -----
def gen_TV_Murray_Math_Octave(df, octave_win=64, level_pct=12.5, **kw):
    """Murray Math: divide N-bar range into 8 octaves (1/8 levels).
    Long entry: close bounces off lower octave (close near level then up).
    Short entry: close bounces off upper octave.
    `level_pct` controls how close to a level counts as "near" (% of full range).
    """
    ow = int(octave_win)
    lp = float(level_pct) / 100.0
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    hh = h.rolling(ow, min_periods=ow).max()
    ll = l.rolling(ow, min_periods=ow).min()
    rng = (hh - ll).replace(0.0, np.nan)
    pos = (c - ll) / rng  # 0..1
    octave = (pos * 8).round() / 8.0  # nearest 1/8 level
    diff = (pos - octave).abs()
    near_level = diff < lp
    # Direction: bounce up off lower half (octave <= 0.375), down off upper (>= 0.625)
    long_bounce = near_level & (octave <= 0.375) & (c > c.shift(1))
    short_bounce = near_level & (octave >= 0.625) & (c < c.shift(1))
    long_entry = long_bounce & ~(long_bounce.shift(1).fillna(False))
    short_entry = short_bounce & ~(short_bounce.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Murray_Math_Octave():
    return {
        'octave_win': ('int', 32, 128),
        'level_pct': ('float', 4.0, 20.0),
    }


# ----- 2) VOLUME-WEIGHTED MOMENTUM -----
def gen_TV_VolumeWeighted_Momentum(df, mom_len=14, vol_win=20, threshold=1.5, **kw):
    """vol_norm = volume / SMA(volume, vol_win).
    vw_mom = ROC(mom_len) * vol_norm.
    Long: vw_mom crosses above +threshold. Short: below -threshold.
    """
    ml = int(mom_len)
    vw = int(vol_win)
    th = float(threshold)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    roc = (c / c.shift(ml) - 1.0) * 100.0
    vol_avg = v.rolling(vw, min_periods=vw).mean().replace(0.0, np.nan)
    vol_norm = v / vol_avg
    vw_mom = roc * vol_norm
    long_cross = (vw_mom > th) & (vw_mom.shift(1) <= th)
    short_cross = (vw_mom < -th) & (vw_mom.shift(1) >= -th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_VolumeWeighted_Momentum():
    return {
        'mom_len': ('int', 7, 30),
        'vol_win': ('int', 10, 40),
        'threshold': ('float', 0.8, 4.0),
    }


# ----- 3) GOLDEN RATIO CHANNEL -----
def gen_TV_GoldenRatio_Channel(df, swing_len=50, **kw):
    """Channel = swing_high to swing_low.
    Upper extension = swing_high + 0.618 * (swing_high - swing_low)
    Lower extension = swing_low - 0.618 * (swing_high - swing_low)
    Long entry: close breaks above upper extension AND momentum positive.
    Short entry: close breaks below lower extension AND momentum negative.
    """
    sl = int(swing_len)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    swing_high = h.rolling(sl, min_periods=sl).max().shift(1)
    swing_low = l.rolling(sl, min_periods=sl).min().shift(1)
    rng = swing_high - swing_low
    upper_ext = swing_high + 0.618 * rng
    lower_ext = swing_low - 0.618 * rng
    short_ret = c.pct_change().rolling(5, min_periods=5).sum()
    long_break = (c > upper_ext) & (c.shift(1) <= upper_ext.shift(1)) & (short_ret > 0)
    short_break = (c < lower_ext) & (c.shift(1) >= lower_ext.shift(1)) & (short_ret < 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_break.shift(1).fillna(False).astype(bool)] = 1
    sig[short_break.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_GoldenRatio_Channel():
    return {
        'swing_len': ('int', 20, 120),
    }


# ----- 4) PRICE-VOLUME CORRELATION -----
def gen_TV_PriceVolume_Correlation(df, win=30, **kw):
    """Rolling correlation of price changes and volume changes.
    Vectorized via rolling sums:
       corr = (E[xy] - E[x]E[y]) / (std_x * std_y)
    Positive corr: vol confirms price moves -> follow direction.
    Negative corr: vol diverges -> revert.
    """
    w = int(win)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    dp = c.diff()
    dv = v.diff()
    mu_p = dp.rolling(w, min_periods=w).mean()
    mu_v = dv.rolling(w, min_periods=w).mean()
    cov = (dp * dv).rolling(w, min_periods=w).mean() - mu_p * mu_v
    var_p = (dp * dp).rolling(w, min_periods=w).mean() - mu_p * mu_p
    var_v = (dv * dv).rolling(w, min_periods=w).mean() - mu_v * mu_v
    corr = cov / (var_p.pow(0.5) * var_v.pow(0.5)).replace(0.0, np.nan)
    short_ret = dp.rolling(5, min_periods=5).sum()
    pos_corr = corr > 0
    neg_corr = corr < 0
    long_setup = (pos_corr & (short_ret > 0)) | (neg_corr & (short_ret < 0))
    short_setup = (pos_corr & (short_ret < 0)) | (neg_corr & (short_ret > 0))
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_PriceVolume_Correlation():
    return {
        'win': ('int', 15, 80),
    }


# ----- 5) RANGE PERCENTILE CROSS -----
def gen_TV_RangePctile_Cross(df, win=100, high_pct=85, **kw):
    """Bar range = high - low. Percentile rank in last `win` bars.
    HIGH range (> high_pct percentile) = breakout setup.
    Direction = sign of close - prev_close.
    """
    w = int(win)
    hp = float(high_pct)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    rng = h - l
    rng_rank = rng.rolling(w, min_periods=w).rank(pct=True) * 100
    big_range = (rng_rank > hp) & (rng_rank.shift(1) <= hp)
    long_setup = big_range & (c > c.shift(1))
    short_setup = big_range & (c < c.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_RangePctile_Cross():
    return {
        'win': ('int', 50, 250),
        'high_pct': ('int', 75, 95),
    }


STRATEGY_EXPORT = {
    "TV_Murray_Math_Octave": {
        "gen": gen_TV_Murray_Math_Octave, "space": space_TV_Murray_Math_Octave,
        "source": "mac_batch3720_Murray_Math_octave_levels",
    },
    "TV_VolumeWeighted_Momentum": {
        "gen": gen_TV_VolumeWeighted_Momentum,
        "space": space_TV_VolumeWeighted_Momentum,
        "source": "mac_batch3720_vol_weighted_momentum",
    },
    "TV_GoldenRatio_Channel": {
        "gen": gen_TV_GoldenRatio_Channel,
        "space": space_TV_GoldenRatio_Channel,
        "source": "mac_batch3720_Fibonacci_618_extension_channel",
    },
    "TV_PriceVolume_Correlation": {
        "gen": gen_TV_PriceVolume_Correlation,
        "space": space_TV_PriceVolume_Correlation,
        "source": "mac_batch3720_price_volume_rolling_correlation",
    },
    "TV_RangePctile_Cross": {
        "gen": gen_TV_RangePctile_Cross, "space": space_TV_RangePctile_Cross,
        "source": "mac_batch3720_range_percentile_threshold_cross",
    },
}
