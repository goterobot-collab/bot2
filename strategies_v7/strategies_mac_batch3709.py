"""
Batch 3709 - Mac paralela wave m10 - 5 NOVEL oscillator/composite families.

All low-param (1-3 each), pickle-safe, signal int {-1,0,1}, .shift(1).
ZERO overlap with sandbox 3566-3610, Mac V8 prod (1717), Mac 3700-3708.

 1. TV_TSI_Cross           - Blau True Strength Index cross signal line
                              Source: William Blau "Momentum, Direction and
                                      Divergence" 1995
 2. TV_SchaffTrend_Cycle   - Doug Schaff STC = stochastic of MACD
                              (NOT in repo as standalone)
 3. TV_KVO_Klinger         - Klinger Volume Oscillator
                              Source: Stephen Klinger 1997
 4. TV_QStick_Trend        - Tushar Chande QStick = SMA(close-open, n)
                              Source: Chande "The New Technical Trader" 1994
 5. TV_PriceVelocity_Accel - 2nd derivative (acceleration) of close zero-cross
                              (continuous-time momentum analog, NOT in repo)
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


# ----- 1) BLAU TRUE STRENGTH INDEX -----
def gen_TV_TSI_Cross(df, r=25, s=13, sig_len=7, **kw):
    """TSI = 100 * EMA(EMA(diff, r), s) / EMA(EMA(|diff|, r), s)
       diff = close - prev close.
    Long: TSI crosses above signal line.
    Short: TSI crosses below signal line.
    """
    rr = int(r)
    ss = int(s)
    sl = int(sig_len)
    c = df['close'].astype(float)
    diff = c.diff()
    abs_diff = diff.abs()
    num = _ema(_ema(diff, rr), ss)
    den = _ema(_ema(abs_diff, rr), ss).replace(0.0, np.nan)
    tsi = 100.0 * num / den
    sig_line = _ema(tsi, sl)
    cross_up = (tsi > sig_line) & (tsi.shift(1) <= sig_line.shift(1))
    cross_dn = (tsi < sig_line) & (tsi.shift(1) >= sig_line.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_TSI_Cross():
    return {
        'r': ('int', 15, 35),
        's': ('int', 7, 20),
        'sig_len': ('int', 4, 12),
    }


# ----- 2) SCHAFF TREND CYCLE -----
def gen_TV_SchaffTrend_Cycle(df, fast=23, slow=50, cycle=10, **kw):
    """STC by Doug Schaff: stochastic transformation of MACD.
       MACD = EMA(close, fast) - EMA(close, slow)
       %K1 = (MACD - min(MACD, cycle)) / (max(MACD, cycle) - min(MACD, cycle)) * 100
       D1 = EMA(%K1, 0.5)  -- approximated as EMA span 3
       %K2 = stoch(D1, cycle)
       STC = EMA(%K2, 0.5) -- span 3
    Long: STC crosses above 25.
    Short: STC crosses below 75.
    """
    f = int(fast)
    s = int(slow)
    n = int(cycle)
    c = df['close'].astype(float)
    macd = _ema(c, f) - _ema(c, s)
    macd_min = macd.rolling(n, min_periods=n).min()
    macd_max = macd.rolling(n, min_periods=n).max()
    rng = (macd_max - macd_min).replace(0.0, np.nan)
    k1 = 100.0 * (macd - macd_min) / rng
    d1 = _ema(k1, 3)
    d1_min = d1.rolling(n, min_periods=n).min()
    d1_max = d1.rolling(n, min_periods=n).max()
    rng2 = (d1_max - d1_min).replace(0.0, np.nan)
    k2 = 100.0 * (d1 - d1_min) / rng2
    stc = _ema(k2, 3)
    long_cross = (stc > 25) & (stc.shift(1) <= 25)
    short_cross = (stc < 75) & (stc.shift(1) >= 75)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_SchaffTrend_Cycle():
    return {
        'fast': ('int', 12, 30),
        'slow': ('int', 30, 65),
        'cycle': ('int', 7, 15),
    }


# ----- 3) KLINGER VOLUME OSCILLATOR -----
def gen_TV_KVO_Klinger(df, fast=34, slow=55, sig_len=13, **kw):
    """Klinger VolForce = volume * sign(trend) * 2 * (dm/cm - 1) * 100
       where trend = sign(typical_price - prev_typical_price),
       dm = high - low, cm = cumulative dm of same trend.
    KVO = EMA(VF, fast) - EMA(VF, slow)
    Long: KVO crosses above EMA(KVO, sig_len).
    Short: cross below.
    """
    f = int(fast)
    s = int(slow)
    sl = int(sig_len)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    tp = (h + l + c) / 3.0
    trend = np.sign(tp - tp.shift(1)).fillna(0).astype(int)
    dm = (h - l).replace(0.0, np.nan)
    # cm = cumulative dm where trend stays same; reset when trend flips
    trend_arr = trend.to_numpy()
    dm_arr = dm.fillna(0).to_numpy()
    cm = np.zeros_like(dm_arr)
    for i in range(1, len(dm_arr)):
        if trend_arr[i] == trend_arr[i - 1]:
            cm[i] = cm[i - 1] + dm_arr[i]
        else:
            cm[i] = dm_arr[i - 1] + dm_arr[i]
    cm_s = pd.Series(cm, index=df.index)
    vf = v * trend * 2 * (dm / cm_s.replace(0.0, np.nan) - 1) * 100
    kvo = _ema(vf, f) - _ema(vf, s)
    sig_line = _ema(kvo, sl)
    cross_up = (kvo > sig_line) & (kvo.shift(1) <= sig_line.shift(1))
    cross_dn = (kvo < sig_line) & (kvo.shift(1) >= sig_line.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_KVO_Klinger():
    return {
        'fast': ('int', 20, 50),
        'slow': ('int', 40, 80),
        'sig_len': ('int', 7, 20),
    }


# ----- 4) CHANDE QSTICK TREND -----
def gen_TV_QStick_Trend(df, qstick_len=14, sig_len=7, **kw):
    """QStick = SMA(close - open, n).
    Long: QStick crosses above EMA(QStick, sig_len) AND > 0.
    Short: QStick crosses below signal AND < 0.
    """
    ql = int(qstick_len)
    sl = int(sig_len)
    o = df['open'].astype(float)
    c = df['close'].astype(float)
    qstick = _sma(c - o, ql)
    sig_line = _ema(qstick, sl)
    cross_up = (qstick > sig_line) & (qstick.shift(1) <= sig_line.shift(1)) & (qstick > 0)
    cross_dn = (qstick < sig_line) & (qstick.shift(1) >= sig_line.shift(1)) & (qstick < 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_QStick_Trend():
    return {
        'qstick_len': ('int', 7, 30),
        'sig_len': ('int', 4, 15),
    }


# ----- 5) PRICE VELOCITY ACCELERATION -----
def gen_TV_PriceVelocity_Accel(df, smooth=5, threshold_pct=0.001, **kw):
    """First derivative (velocity) = close - close.shift(smooth).
    Second derivative (acceleration) = velocity - velocity.shift(smooth).
    Long entry: acceleration crosses above +threshold * close.
    Short entry: acceleration crosses below -threshold * close.
    """
    s = int(smooth)
    th = float(threshold_pct)
    c = df['close'].astype(float)
    velocity = c - c.shift(s)
    accel = velocity - velocity.shift(s)
    thr_vec = th * c
    long_cross = (accel > thr_vec) & (accel.shift(1) <= thr_vec.shift(1))
    short_cross = (accel < -thr_vec) & (accel.shift(1) >= -thr_vec.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_PriceVelocity_Accel():
    return {
        'smooth': ('int', 2, 15),
        'threshold_pct': ('float', 0.0005, 0.005),
    }


STRATEGY_EXPORT = {
    "TV_TSI_Cross": {
        "gen": gen_TV_TSI_Cross, "space": space_TV_TSI_Cross,
        "source": "mac_batch3709_Blau_TSI_1995",
    },
    "TV_SchaffTrend_Cycle": {
        "gen": gen_TV_SchaffTrend_Cycle, "space": space_TV_SchaffTrend_Cycle,
        "source": "mac_batch3709_Schaff_STC_1999",
    },
    "TV_KVO_Klinger": {
        "gen": gen_TV_KVO_Klinger, "space": space_TV_KVO_Klinger,
        "source": "mac_batch3709_Klinger_VolumeForce_1997",
    },
    "TV_QStick_Trend": {
        "gen": gen_TV_QStick_Trend, "space": space_TV_QStick_Trend,
        "source": "mac_batch3709_Chande_QStick_NewTechnicalTrader_1994",
    },
    "TV_PriceVelocity_Accel": {
        "gen": gen_TV_PriceVelocity_Accel, "space": space_TV_PriceVelocity_Accel,
        "source": "mac_batch3709_2nd_derivative_acceleration_signal",
    },
}
