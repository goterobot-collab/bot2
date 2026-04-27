"""
Batch 3714 - Mac paralela wave m15 - 5 pattern + multi-confirm hybrid families.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3713.

 1. TV_LongShortMemory_Combo  - 1-bar momentum opposed by 50-bar mean revert
                                 (Engle-Granger long-short memory mix)
 2. TV_TripleVol_Compression  - 3 vol indicators all in low quartile = entry
                                 setup, breakout direction by close
 3. TV_AsymVolSpike_Confirm   - vol spike + close direction confirms direction
 4. TV_DonchianMid_Bounce     - close bounces off Donchian midline (NEW
                                 concept; sandbox Donchian is edge break)
 5. TV_StreakReversal_VolDiv  - N consecutive same-direction with volume
                                 divergence (declining vol on streak)
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


def _atr(df, n):
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()


# ----- 1) LONG-SHORT MEMORY COMBO -----
def gen_TV_LongShortMemory_Combo(df, short_len=2, long_len=50, **kw):
    """Long-short memory combo:
    Long entry: short_len-bar return > 0 AND long_len-bar return < 0
                (recent up move within longer downtrend = short-term momentum
                 in mean-reversion regime)
    Wait, this is messy. Cleaner:
    Long entry: short return positive (1-2 bar momentum) AND long return positive
                (overall uptrend) = both timeframes aligned bullish, transition.
    Short entry: both negative.
    """
    sl = int(short_len)
    ll = int(long_len)
    c = df['close'].astype(float)
    short_ret = (c / c.shift(sl) - 1.0)
    long_ret = (c / c.shift(ll) - 1.0)
    long_setup = (short_ret > 0) & (long_ret > 0)
    short_setup = (short_ret < 0) & (long_ret < 0)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_LongShortMemory_Combo():
    return {
        'short_len': ('int', 1, 5),
        'long_len': ('int', 30, 120),
    }


# ----- 2) TRIPLE VOLATILITY COMPRESSION -----
def gen_TV_TripleVol_Compression(df, win=20, vol_win=60, low_quartile=25, **kw):
    """3 volatility indicators ALL in low quartile of last vol_win bars:
       - stdev(returns, win)
       - ATR(win)
       - BB band width (mid +- 2*std normalized)
    When all 3 in bottom low_quartile percentile -> compression setup.
    Direction = sign of next bar's close vs prior close (committed forecast).
    """
    w = int(win)
    vw = int(vol_win)
    lq = float(low_quartile)
    c = df['close'].astype(float)
    ret = c.pct_change()
    sd = ret.rolling(w, min_periods=w).std(ddof=0)
    atr = _atr(df, w) / c
    bb_mid = c.rolling(w, min_periods=w).mean()
    bb_std = c.rolling(w, min_periods=w).std(ddof=0)
    bb_width = (4 * bb_std) / bb_mid.replace(0.0, np.nan)
    sd_pct = sd.rolling(vw, min_periods=vw).rank(pct=True) * 100
    atr_pct = atr.rolling(vw, min_periods=vw).rank(pct=True) * 100
    bb_pct = bb_width.rolling(vw, min_periods=vw).rank(pct=True) * 100
    compressed = (sd_pct < lq) & (atr_pct < lq) & (bb_pct < lq)
    long_setup = compressed & (ret > 0)
    short_setup = compressed & (ret < 0)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_TripleVol_Compression():
    return {
        'win': ('int', 10, 30),
        'vol_win': ('int', 30, 120),
        'low_quartile': ('int', 15, 35),
    }


# ----- 3) ASYMMETRIC VOL SPIKE CONFIRM -----
def gen_TV_AsymVolSpike_Confirm(df, vol_win=20, spike_mult=2.5, **kw):
    """Volume spike (vol > spike_mult * SMA(vol, vol_win)) AND close > prev_close
    -> LONG (buying spike). Symmetric for SHORT.
    """
    vw = int(vol_win)
    sm = float(spike_mult)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    vol_ma = _sma(v, vw)
    spike = v > sm * vol_ma
    pc = c.shift(1)
    long_trigger = spike & (c > pc)
    short_trigger = spike & (c < pc)
    long_entry = long_trigger & ~(long_trigger.shift(1).fillna(False))
    short_entry = short_trigger & ~(short_trigger.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_AsymVolSpike_Confirm():
    return {
        'vol_win': ('int', 10, 50),
        'spike_mult': ('float', 1.8, 4.0),
    }


# ----- 4) DONCHIAN MIDLINE BOUNCE -----
def gen_TV_DonchianMid_Bounce(df, donch_len=20, atr_len=14, atr_mult=0.3, **kw):
    """Donchian midline = (highest_high(n) + lowest_low(n)) / 2.
    Long bounce: close approaches midline from below (within atr_mult*ATR)
                 then closes above midline AND in bottom half of donch range.
    Short bounce: opposite.
    """
    dl = int(donch_len)
    al = int(atr_len)
    am = float(atr_mult)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    hh = h.rolling(dl, min_periods=dl).max()
    ll = l.rolling(dl, min_periods=dl).min()
    mid = (hh + ll) / 2.0
    atr = _atr(df, al)
    near_mid = (c - mid).abs() < am * atr
    bounce_up = near_mid & (c > mid) & (c.shift(1) <= mid.shift(1)) & (c < (hh + mid) / 2)
    bounce_dn = near_mid & (c < mid) & (c.shift(1) >= mid.shift(1)) & (c > (ll + mid) / 2)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[bounce_up.shift(1).fillna(False).astype(bool)] = 1
    sig[bounce_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_DonchianMid_Bounce():
    return {
        'donch_len': ('int', 10, 60),
        'atr_len': ('int', 7, 25),
        'atr_mult': ('float', 0.15, 1.0),
    }


# ----- 5) STREAK + VOLUME DIVERGENCE REVERSAL -----
def gen_TV_StreakReversal_VolDiv(df, streak_len=4, vol_decline_pct=80, **kw):
    """N consecutive same-direction bars with DECLINING volume = exhaustion.
    Volume divergence: last bar's vol < vol_decline_pct% of streak's first bar vol.
    Reverse direction.
    """
    sl = int(streak_len)
    vp = float(vol_decline_pct) / 100.0
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    up_bar = c > c.shift(1)
    dn_bar = c < c.shift(1)
    up_streak = up_bar.rolling(sl, min_periods=sl).sum() == sl
    dn_streak = dn_bar.rolling(sl, min_periods=sl).sum() == sl
    vol_first = v.shift(sl - 1)
    vol_decline = v < vp * vol_first
    short_trigger = up_streak & vol_decline  # exhausted up streak -> short
    long_trigger = dn_streak & vol_decline   # exhausted down streak -> long
    long_entry = long_trigger & ~(long_trigger.shift(1).fillna(False))
    short_entry = short_trigger & ~(short_trigger.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_StreakReversal_VolDiv():
    return {
        'streak_len': ('int', 3, 7),
        'vol_decline_pct': ('int', 50, 90),
    }


STRATEGY_EXPORT = {
    "TV_LongShortMemory_Combo": {
        "gen": gen_TV_LongShortMemory_Combo,
        "space": space_TV_LongShortMemory_Combo,
        "source": "mac_batch3714_long_short_horizon_alignment",
    },
    "TV_TripleVol_Compression": {
        "gen": gen_TV_TripleVol_Compression,
        "space": space_TV_TripleVol_Compression,
        "source": "mac_batch3714_triple_volatility_quartile_compression",
    },
    "TV_AsymVolSpike_Confirm": {
        "gen": gen_TV_AsymVolSpike_Confirm,
        "space": space_TV_AsymVolSpike_Confirm,
        "source": "mac_batch3714_asym_volume_spike_directional_confirm",
    },
    "TV_DonchianMid_Bounce": {
        "gen": gen_TV_DonchianMid_Bounce,
        "space": space_TV_DonchianMid_Bounce,
        "source": "mac_batch3714_Donchian_midline_bounce",
    },
    "TV_StreakReversal_VolDiv": {
        "gen": gen_TV_StreakReversal_VolDiv,
        "space": space_TV_StreakReversal_VolDiv,
        "source": "mac_batch3714_streak_volume_divergence_reversal",
    },
}
