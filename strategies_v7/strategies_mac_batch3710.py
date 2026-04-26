"""
Batch 3710 - Mac paralela wave m11 - 5 hybrid / multi-confirm families.

Low-param (1-3 each), pickle-safe, signal int {-1,0,1}, .shift(1).
ZERO overlap with sandbox 3566-3610, Mac V8 prod (1717), Mac 3700-3709.

 1. TV_BB_Width_Slope_Regime  - BB band width slope sign as regime indicator
                                 (NOT Squeeze pattern, NOT Width threshold)
 2. TV_ADX_Pure_Trend         - pure ADX (no DMI direction) entry on ADX
                                 surge + price-vs-MA direction filter
 3. TV_HA_Smoothed_Cross      - Heikin-Ashi smoothed close cross EMA
                                 (DISTINCT from sandbox TV_HeikinAshiTrend
                                 which is direct HA color flip)
 4. TV_FISH_Pullback_Trend    - Fisher Transform pullback entry within
                                 trend (DISTINCT from Ehlers_Fisher_Transform
                                 which is direct cross)
 5. TV_TripleEMA_Alignment    - 3 EMAs (short<mid<long for short, etc) aligned
                                 simultaneously (DISTINCT from sandbox
                                 TV_EMA_TripleChannel which is band-style)
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


# ----- 1) BB WIDTH SLOPE REGIME -----
def gen_TV_BB_Width_Slope_Regime(df, bb_len=20, bb_mult=2.0, slope_len=5, **kw):
    """BB width = (upper - lower) / mid. Slope = width - width.shift(slope_len).
    Long: slope > 0 (expanding) AND close > mid (uptrend leg).
    Short: slope > 0 AND close < mid (downtrend leg).
    """
    n = int(bb_len)
    m = float(bb_mult)
    sl = int(slope_len)
    c = df['close'].astype(float)
    mid = c.rolling(n, min_periods=n).mean()
    sd = c.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + m * sd
    lower = mid - m * sd
    width = (upper - lower) / mid.replace(0.0, np.nan)
    slope = width - width.shift(sl)
    long_setup = (slope > 0) & (c > mid) & (slope.shift(1) <= 0)
    short_setup = (slope > 0) & (c < mid) & (slope.shift(1) <= 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_BB_Width_Slope_Regime():
    return {
        'bb_len': ('int', 14, 30),
        'bb_mult': ('float', 1.5, 3.0),
        'slope_len': ('int', 3, 10),
    }


# ----- 2) PURE ADX TREND -----
def gen_TV_ADX_Pure_Trend(df, adx_len=14, surge_threshold=25, ma_len=50, **kw):
    """ADX surge detector + price-vs-MA filter for direction (no DMI).
    Long: ADX crosses above threshold AND close > MA(ma_len).
    Short: ADX crosses above threshold AND close < MA(ma_len).
    """
    al = int(adx_len)
    th = float(surge_threshold)
    ml = int(ma_len)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    up = h - h.shift(1)
    dn = l.shift(1) - l
    plus_dm = ((up > dn) & (up > 0)).astype(float) * up
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn
    atr = tr.ewm(alpha=1.0 / al, adjust=False, min_periods=al).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / al, adjust=False, min_periods=al).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / al, adjust=False, min_periods=al).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / al, adjust=False, min_periods=al).mean()
    ma = c.rolling(ml, min_periods=ml).mean()
    surge = (adx > th) & (adx.shift(1) <= th)
    long_trigger = surge & (c > ma)
    short_trigger = surge & (c < ma)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_trigger.shift(1).fillna(False).astype(bool)] = 1
    sig[short_trigger.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_ADX_Pure_Trend():
    return {
        'adx_len': ('int', 8, 25),
        'surge_threshold': ('int', 18, 35),
        'ma_len': ('int', 20, 100),
    }


# ----- 3) HEIKIN-ASHI SMOOTHED CROSS -----
def gen_TV_HA_Smoothed_Cross(df, ema_pre=5, ema_post=10, **kw):
    """Smooth OHLC with EMA(pre) before computing Heikin-Ashi.
    Then HA close cross EMA(ha_close, post).
    Long: HA close > EMA(ha_close, post) AND green HA candle (ha_close > ha_open).
    Short: opposite.
    Distinct from sandbox TV_HeikinAshiTrend (direct color flip).
    """
    pre = int(ema_pre)
    post = int(ema_post)
    # Use min_periods=1 to avoid NaN propagation in recursive ha_open
    o = df['open'].astype(float).ewm(span=pre, adjust=False, min_periods=1).mean()
    h = df['high'].astype(float).ewm(span=pre, adjust=False, min_periods=1).mean()
    l = df['low'].astype(float).ewm(span=pre, adjust=False, min_periods=1).mean()
    c = df['close'].astype(float).ewm(span=pre, adjust=False, min_periods=1).mean()
    ha_close = (o + h + l + c) / 4.0
    co = c.to_numpy()
    op = o.to_numpy()
    hco = ha_close.to_numpy()
    hop = np.zeros_like(hco)
    hop[0] = (op[0] + co[0]) / 2.0
    for i in range(1, len(hco)):
        hop[i] = (hop[i - 1] + hco[i - 1]) / 2.0
    ha_open = pd.Series(hop, index=df.index)
    ha_close = pd.Series(hco, index=df.index)
    ema_ha = _ema(ha_close, post)
    cross_up = (ha_close > ema_ha) & (ha_close.shift(1) <= ema_ha.shift(1)) & (ha_close > ha_open)
    cross_dn = (ha_close < ema_ha) & (ha_close.shift(1) >= ema_ha.shift(1)) & (ha_close < ha_open)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_HA_Smoothed_Cross():
    return {
        'ema_pre': ('int', 3, 10),
        'ema_post': ('int', 5, 25),
    }


# ----- 4) FISHER PULLBACK WITHIN TREND -----
def gen_TV_FISH_Pullback_Trend(df, fish_len=10, ma_len=50, threshold=0.5, **kw):
    """Fisher Transform of normalized price.
    Trend filter: close vs MA(ma_len).
    Pullback entry: in uptrend, when Fisher dips < -threshold and crosses
    back above (long). Symmetric for downtrend.
    """
    fl = int(fish_len)
    ml = int(ma_len)
    th = float(threshold)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    median = (h + l) / 2.0
    hh = h.rolling(fl, min_periods=fl).max()
    ll = l.rolling(fl, min_periods=fl).min()
    rng = (hh - ll).replace(0.0, np.nan)
    norm = 0.66 * ((median - ll) / rng - 0.5)
    norm = norm.clip(-0.999, 0.999)
    # Fisher = 0.5 * ln((1+norm)/(1-norm)) recursively smoothed
    fish_arr = np.zeros(len(norm))
    n_arr = norm.to_numpy()
    for i in range(1, len(n_arr)):
        if np.isnan(n_arr[i]):
            fish_arr[i] = fish_arr[i - 1]
            continue
        fish_arr[i] = 0.5 * np.log((1 + n_arr[i]) / (1 - n_arr[i])) + 0.5 * fish_arr[i - 1]
    fish = pd.Series(fish_arr, index=df.index)
    ma = c.rolling(ml, min_periods=ml).mean()
    in_uptrend = c > ma
    in_downtrend = c < ma
    long_pullback = in_uptrend & (fish > -th) & (fish.shift(1) <= -th)
    short_pullback = in_downtrend & (fish < th) & (fish.shift(1) >= th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_pullback.shift(1).fillna(False).astype(bool)] = 1
    sig[short_pullback.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_FISH_Pullback_Trend():
    return {
        'fish_len': ('int', 5, 20),
        'ma_len': ('int', 30, 100),
        'threshold': ('float', 0.3, 1.5),
    }


# ----- 5) TRIPLE EMA ALIGNMENT -----
def gen_TV_TripleEMA_Alignment(df, ema_short=8, ema_mid=21, ema_long=55, **kw):
    """Triple EMA full alignment:
    Long: EMA_short > EMA_mid > EMA_long (full bullish stack)
    Short: EMA_short < EMA_mid < EMA_long
    Entry on transition into alignment.
    """
    s = int(ema_short)
    m = int(ema_mid)
    l = int(ema_long)
    c = df['close'].astype(float)
    e_s = _ema(c, s)
    e_m = _ema(c, m)
    e_l = _ema(c, l)
    bullish = (e_s > e_m) & (e_m > e_l)
    bearish = (e_s < e_m) & (e_m < e_l)
    long_entry = bullish & ~(bullish.shift(1).fillna(False))
    short_entry = bearish & ~(bearish.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_TripleEMA_Alignment():
    return {
        'ema_short': ('int', 5, 15),
        'ema_mid': ('int', 18, 30),
        'ema_long': ('int', 40, 100),
    }


STRATEGY_EXPORT = {
    "TV_BB_Width_Slope_Regime": {
        "gen": gen_TV_BB_Width_Slope_Regime,
        "space": space_TV_BB_Width_Slope_Regime,
        "source": "mac_batch3710_BB_width_slope_regime",
    },
    "TV_ADX_Pure_Trend": {
        "gen": gen_TV_ADX_Pure_Trend, "space": space_TV_ADX_Pure_Trend,
        "source": "mac_batch3710_pure_ADX_surge_MA_filter",
    },
    "TV_HA_Smoothed_Cross": {
        "gen": gen_TV_HA_Smoothed_Cross, "space": space_TV_HA_Smoothed_Cross,
        "source": "mac_batch3710_HA_smoothed_EMA_cross",
    },
    "TV_FISH_Pullback_Trend": {
        "gen": gen_TV_FISH_Pullback_Trend, "space": space_TV_FISH_Pullback_Trend,
        "source": "mac_batch3710_Fisher_pullback_within_trend",
    },
    "TV_TripleEMA_Alignment": {
        "gen": gen_TV_TripleEMA_Alignment, "space": space_TV_TripleEMA_Alignment,
        "source": "mac_batch3710_triple_EMA_full_alignment_transition",
    },
}
