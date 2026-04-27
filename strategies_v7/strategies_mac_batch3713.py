"""
Batch 3713 - Mac paralela wave m14 - 5 statistical / econometric families.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3712.

 1. TV_VarianceRatio_LoMacKinlay - Lo-MacKinlay 1988 VR(q) test for random
                                    walk hypothesis violation
                                    Source: Lo & MacKinlay 1988 RFS
 2. TV_HodrickPrescott_Cycle    - HP-filter cycle component zero-cross
                                    Source: Hodrick-Prescott 1981
 3. TV_AbsRet_Ratio_Regime      - short MA(|ret|) vs long MA(|ret|)
                                    -> volatility expansion regime
 4. TV_UpDownVol_Ratio_Cross    - sum(vol when up) vs sum(vol when down)
                                    over window, cross threshold
 5. TV_TickVol_Momentum         - fraction of up bars in window,
                                    threshold cross
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


# ----- 1) LO-MACKINLAY VARIANCE RATIO -----
def gen_TV_VarianceRatio_LoMacKinlay(df, q=4, win=120, vr_thr=1.3, **kw):
    """VR(q) = Var(r_q) / (q * Var(r_1))
    Where r_q is q-period return. Under random walk: VR(q)=1.
    VR > 1 -> momentum (positive autocorr of returns)
    VR < 1 -> mean reversion
    Long entry: VR > vr_thr AND short_ret > 0 (momentum confirmed)
    Short entry: VR > vr_thr AND short_ret < 0
    Reverse on VR < 1/vr_thr (revert regime, fade direction).
    """
    qq = int(q)
    w = int(win)
    th = float(vr_thr)
    c = df['close'].astype(float)
    r1 = c.pct_change()
    rq = (c / c.shift(qq) - 1.0)
    var1 = r1.rolling(w, min_periods=w).var(ddof=0)
    varq = rq.rolling(w, min_periods=w).var(ddof=0)
    vr = varq / (qq * var1.replace(0.0, np.nan))
    short_ret = r1.rolling(5, min_periods=5).sum()
    momentum_regime = vr > th
    revert_regime = vr < (1.0 / th)
    long_setup = (momentum_regime & (short_ret > 0)) | (revert_regime & (short_ret < 0))
    short_setup = (momentum_regime & (short_ret < 0)) | (revert_regime & (short_ret > 0))
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_VarianceRatio_LoMacKinlay():
    return {
        'q': ('int', 2, 8),
        'win': ('int', 60, 250),
        'vr_thr': ('float', 1.15, 1.6),
    }


# ----- 2) HODRICK-PRESCOTT CYCLE -----
def gen_TV_HodrickPrescott_Cycle(df, lambda_smooth=1600, win=200, **kw):
    """Approximate HP filter via penalized 2nd-difference smoothing.
    Trend = solution to: min sum((y - tau)^2) + lambda * sum((tau[t+1] - 2*tau[t] + tau[t-1])^2)
    Cycle = y - trend.
    Long: cycle crosses above zero (cycle bottoming and rising).
    Short: cycle crosses below zero.

    Implementation: rolling fit over `win` bars, but use a cheap proxy =
    SMA-band cycle (close - SMA(close, win) / SMA(close, win)) which approximates
    HP cycle for low-frequency trends without the matrix inversion cost.
    The lambda_smooth param controls SMA length: ma_len = max(20, lambda_smooth // 100).
    """
    ma_len = max(20, int(lambda_smooth) // 100)
    c = df['close'].astype(float)
    trend = _sma(c, ma_len)
    cycle = (c - trend) / trend.replace(0.0, np.nan)
    cross_up = (cycle > 0) & (cycle.shift(1) <= 0)
    cross_dn = (cycle < 0) & (cycle.shift(1) >= 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_HodrickPrescott_Cycle():
    return {
        'lambda_smooth': ('int', 800, 6400),
        'win': ('int', 100, 300),
    }


# ----- 3) ABSOLUTE RETURN RATIO REGIME -----
def gen_TV_AbsRet_Ratio_Regime(df, short_len=10, long_len=50, ratio_thr=1.5, **kw):
    """Volatility expansion proxy: SMA(|ret|, short) / SMA(|ret|, long).
    Ratio > thr: vol expanding -> direction trade based on short_ret sign.
    Ratio < 1/thr: vol contracting -> revert trade.
    """
    sl = int(short_len)
    ll = int(long_len)
    th = float(ratio_thr)
    c = df['close'].astype(float)
    abs_ret = c.pct_change().abs()
    short_vol = _sma(abs_ret, sl)
    long_vol = _sma(abs_ret, ll)
    ratio = short_vol / long_vol.replace(0.0, np.nan)
    short_ret = c.pct_change().rolling(5, min_periods=5).sum()
    expanding = ratio > th
    contracting = ratio < (1.0 / th)
    long_setup = (expanding & (short_ret > 0)) | (contracting & (short_ret < 0))
    short_setup = (expanding & (short_ret < 0)) | (contracting & (short_ret > 0))
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_AbsRet_Ratio_Regime():
    return {
        'short_len': ('int', 5, 20),
        'long_len': ('int', 30, 80),
        'ratio_thr': ('float', 1.2, 2.5),
    }


# ----- 4) UP/DOWN VOLUME RATIO CROSS -----
def gen_TV_UpDownVol_Ratio_Cross(df, win=20, ratio_thr=1.5, **kw):
    """sum_vol(when close > prev_close) vs sum_vol(when close < prev_close)
    over rolling window. Ratio > thr means strong buying.
    Long entry: ratio crosses above thr.
    Short entry: ratio crosses below 1/thr.
    """
    w = int(win)
    th = float(ratio_thr)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    pc = c.shift(1)
    up_vol = v.where(c > pc, 0.0)
    dn_vol = v.where(c < pc, 0.0)
    sum_up = up_vol.rolling(w, min_periods=w).sum()
    sum_dn = dn_vol.rolling(w, min_periods=w).sum()
    ratio = sum_up / sum_dn.replace(0.0, np.nan)
    long_cross = (ratio > th) & (ratio.shift(1) <= th)
    short_cross = (ratio < (1.0 / th)) & (ratio.shift(1) >= (1.0 / th))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_UpDownVol_Ratio_Cross():
    return {
        'win': ('int', 10, 60),
        'ratio_thr': ('float', 1.2, 2.5),
    }


# ----- 5) TICK-VOLUME MOMENTUM -----
def gen_TV_TickVol_Momentum(df, win=20, up_thr_pct=65, **kw):
    """Fraction of bars where close > open in window.
    > up_thr_pct -> bullish bias (long).
    < (100 - up_thr_pct) -> bearish bias (short).
    """
    w = int(win)
    th = float(up_thr_pct) / 100.0
    o = df['open'].astype(float)
    c = df['close'].astype(float)
    up_bar = (c > o).astype(int)
    frac = up_bar.rolling(w, min_periods=w).mean()
    long_setup = frac > th
    short_setup = frac < (1 - th)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_TickVol_Momentum():
    return {
        'win': ('int', 10, 50),
        'up_thr_pct': ('int', 55, 80),
    }


STRATEGY_EXPORT = {
    "TV_VarianceRatio_LoMacKinlay": {
        "gen": gen_TV_VarianceRatio_LoMacKinlay,
        "space": space_TV_VarianceRatio_LoMacKinlay,
        "source": "mac_batch3713_LoMacKinlay_VR_RFS_1988",
    },
    "TV_HodrickPrescott_Cycle": {
        "gen": gen_TV_HodrickPrescott_Cycle,
        "space": space_TV_HodrickPrescott_Cycle,
        "source": "mac_batch3713_HodrickPrescott_1981_SMAproxy",
    },
    "TV_AbsRet_Ratio_Regime": {
        "gen": gen_TV_AbsRet_Ratio_Regime,
        "space": space_TV_AbsRet_Ratio_Regime,
        "source": "mac_batch3713_abs_return_ratio_regime",
    },
    "TV_UpDownVol_Ratio_Cross": {
        "gen": gen_TV_UpDownVol_Ratio_Cross,
        "space": space_TV_UpDownVol_Ratio_Cross,
        "source": "mac_batch3713_updown_volume_ratio",
    },
    "TV_TickVol_Momentum": {
        "gen": gen_TV_TickVol_Momentum,
        "space": space_TV_TickVol_Momentum,
        "source": "mac_batch3713_tick_volume_up_fraction",
    },
}
