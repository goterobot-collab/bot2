"""
Batch 3702 - Mac paralela wave m3 - 5 volume / flow families.

Owner: Mac paralela (Claude Code online, rama claude/grail-hunt-bot-noeIq).

Families chosen for ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac
batches 3700-3701, or sandbox's TV_CVD_Crossover / TV_VolumeProfile_* /
TV_VolumeDelta_* / TV_Footprint_Volume_* / TV_OBV_Momentum:

 1. TV_Coppock_Curve        - long-cycle momentum (ROC sum + WMA)
 2. TV_KnowSureThing        - KST multi-ROC composite oscillator
 3. TV_ChaikinMoneyFlow     - CMF breakout of zero line with trend filter
 4. TV_AccDist_Divergence   - Accumulation/Distribution vs price divergence
 5. TV_PriceVolumeTrend     - PVT slope reversal

All pickle-safe, no lambdas, signal integer in {-1, 0, 1}, .shift(1) applied.
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


def _wma(s, n):
    n = int(n)
    w = np.arange(1, n + 1, dtype=float)
    w_sum = w.sum()
    vals = s.to_numpy(dtype=float)
    out = np.full_like(vals, np.nan)
    for i in range(n - 1, len(vals)):
        window = vals[i - n + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        out[i] = (window * w).sum() / w_sum
    return pd.Series(out, index=s.index)


def _roc(s, n):
    return (s / s.shift(int(n)) - 1.0) * 100.0


# ----- 1) COPPOCK CURVE -----
def gen_TV_Coppock_Curve(df, roc_long=14, roc_short=11, wma_len=10, **kw):
    """Coppock = WMA(ROC(long) + ROC(short), wma_len).
    Long entry: Coppock crosses above zero.
    Short entry: Coppock crosses below zero.
    """
    rl = int(roc_long)
    rs = int(roc_short)
    wl = int(wma_len)
    c = df['close'].astype(float)
    cop = _wma(_roc(c, rl) + _roc(c, rs), wl)
    long_cross = (cop > 0) & (cop.shift(1) <= 0)
    short_cross = (cop < 0) & (cop.shift(1) >= 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Coppock_Curve():
    return {
        'roc_long': ('int', 10, 22),
        'roc_short': ('int', 7, 18),
        'wma_len': ('int', 6, 20),
    }


# ----- 2) KNOW SURE THING (KST) -----
def gen_TV_KnowSureThing(df, roc1=10, roc2=15, roc3=20, roc4=30,
                          sma1=10, sma2=10, sma3=10, sma4=15,
                          sig_len=9, **kw):
    """KST = SMA(ROC1,sma1)*1 + SMA(ROC2,sma2)*2 + SMA(ROC3,sma3)*3 + SMA(ROC4,sma4)*4.
    Signal = SMA(KST, sig_len).
    Long: KST crosses above signal AND KST > 0.
    Short: KST crosses below signal AND KST < 0.
    """
    c = df['close'].astype(float)
    k1 = _sma(_roc(c, int(roc1)), int(sma1)) * 1
    k2 = _sma(_roc(c, int(roc2)), int(sma2)) * 2
    k3 = _sma(_roc(c, int(roc3)), int(sma3)) * 3
    k4 = _sma(_roc(c, int(roc4)), int(sma4)) * 4
    kst = k1 + k2 + k3 + k4
    sigl = _sma(kst, int(sig_len))
    long_cross = (kst > sigl) & (kst.shift(1) <= sigl.shift(1)) & (kst > 0)
    short_cross = (kst < sigl) & (kst.shift(1) >= sigl.shift(1)) & (kst < 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_KnowSureThing():
    return {
        'roc1': ('int', 7, 14), 'roc2': ('int', 12, 20),
        'roc3': ('int', 18, 25), 'roc4': ('int', 25, 40),
        'sma1': ('int', 7, 14), 'sma2': ('int', 7, 14),
        'sma3': ('int', 7, 14), 'sma4': ('int', 10, 20),
        'sig_len': ('int', 5, 15),
    }


# ----- 3) CHAIKIN MONEY FLOW -----
def gen_TV_ChaikinMoneyFlow(df, cmf_len=20, threshold=0.05, trend_len=50, **kw):
    """CMF = sum(MFV, n) / sum(vol, n) where MFV = ((C-L)-(H-C))/(H-L) * V.
    Long: CMF crosses above +threshold AND close > EMA(trend_len).
    Short: CMF crosses below -threshold AND close < EMA(trend_len).
    """
    n = int(cmf_len)
    t = float(threshold)
    tl = int(trend_len)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    rng = (h - l).replace(0.0, np.nan)
    mfm = ((c - l) - (h - c)) / rng
    mfv = mfm * v
    cmf = mfv.rolling(n, min_periods=n).sum() / v.rolling(n, min_periods=n).sum().replace(0.0, np.nan)
    trend = _ema(c, tl)
    long_cross = (cmf > t) & (cmf.shift(1) <= t) & (c > trend)
    short_cross = (cmf < -t) & (cmf.shift(1) >= -t) & (c < trend)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_ChaikinMoneyFlow():
    return {
        'cmf_len': ('int', 10, 40),
        'threshold': ('float', 0.02, 0.15),
        'trend_len': ('int', 20, 100),
    }


# ----- 4) ACCUMULATION / DISTRIBUTION DIVERGENCE -----
def gen_TV_AccDist_Divergence(df, lookback=20, **kw):
    """A/D Line = cumsum( ((C-L)-(H-C))/(H-L) * V ).
    Bullish div: price makes lower low while AD makes higher low.
    Bearish div: price makes higher high while AD makes lower high.
    """
    L = int(lookback)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    rng = (h - l).replace(0.0, np.nan)
    mfv = ((c - l) - (h - c)) / rng * v
    ad = mfv.fillna(0).cumsum()
    p_min_now = c.rolling(L, min_periods=L).min()
    p_min_prev = c.shift(L).rolling(L, min_periods=L).min()
    p_max_now = c.rolling(L, min_periods=L).max()
    p_max_prev = c.shift(L).rolling(L, min_periods=L).max()
    ad_min_now = ad.rolling(L, min_periods=L).min()
    ad_min_prev = ad.shift(L).rolling(L, min_periods=L).min()
    ad_max_now = ad.rolling(L, min_periods=L).max()
    ad_max_prev = ad.shift(L).rolling(L, min_periods=L).max()
    bull_div = (p_min_now < p_min_prev) & (ad_min_now > ad_min_prev)
    bear_div = (p_max_now > p_max_prev) & (ad_max_now < ad_max_prev)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[bull_div.shift(1).fillna(False).astype(bool)] = 1
    sig[bear_div.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_AccDist_Divergence():
    return {
        'lookback': ('int', 10, 60),
    }


# ----- 5) PRICE-VOLUME TREND SLOPE -----
def gen_TV_PriceVolumeTrend(df, slope_len=10, ema_len=30, **kw):
    """PVT = cumsum( (close - prev_close) / prev_close * volume ).
    Signal from slope of PVT vs its own EMA.
    Long: PVT crosses above EMA(PVT, ema_len) AND PVT slope positive.
    Short: PVT crosses below EMA(PVT, ema_len) AND PVT slope negative.
    """
    sl = int(slope_len)
    el = int(ema_len)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    pc = c.shift(1)
    pvt_inc = ((c - pc) / pc.replace(0.0, np.nan)) * v
    pvt = pvt_inc.fillna(0).cumsum()
    pvt_ema = _ema(pvt, el)
    slope = pvt - pvt.shift(sl)
    cross_up = (pvt > pvt_ema) & (pvt.shift(1) <= pvt_ema.shift(1)) & (slope > 0)
    cross_dn = (pvt < pvt_ema) & (pvt.shift(1) >= pvt_ema.shift(1)) & (slope < 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_PriceVolumeTrend():
    return {
        'slope_len': ('int', 3, 20),
        'ema_len': ('int', 15, 60),
    }


STRATEGY_EXPORT = {
    "TV_Coppock_Curve": {
        "gen": gen_TV_Coppock_Curve, "space": space_TV_Coppock_Curve,
        "source": "mac_batch3702",
    },
    "TV_KnowSureThing": {
        "gen": gen_TV_KnowSureThing, "space": space_TV_KnowSureThing,
        "source": "mac_batch3702",
    },
    "TV_ChaikinMoneyFlow": {
        "gen": gen_TV_ChaikinMoneyFlow, "space": space_TV_ChaikinMoneyFlow,
        "source": "mac_batch3702",
    },
    "TV_AccDist_Divergence": {
        "gen": gen_TV_AccDist_Divergence, "space": space_TV_AccDist_Divergence,
        "source": "mac_batch3702",
    },
    "TV_PriceVolumeTrend": {
        "gen": gen_TV_PriceVolumeTrend, "space": space_TV_PriceVolumeTrend,
        "source": "mac_batch3702",
    },
}
