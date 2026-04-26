"""
Batch 3708 - Mac paralela wave m9 - 5 specialized regime / structural families.

All low-param (1-3 each), pickle-safe, signal int {-1,0,1}, .shift(1).
ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3707.

 1. TV_GarchVol_RegimeSwitch    - GARCH(1,1)-style cond variance update;
                                   regime-dependent direction (HIGH vol -> revert,
                                   LOW vol -> follow)
 2. TV_FracDiff_Stationary      - López de Prado fractionally differentiated
                                   close (d=0.4) crosses zero
                                   Source: Advances in Financial ML 2018 ch5
 3. TV_HurstExp_Adaptive_Switch - Hurst exponent rolling window: H>0.55 trend
                                   regime, H<0.45 mean-revert regime
                                   (DISTINCT from sandbox TV_Hurst_Regime_Switch
                                   which uses different threshold/exit logic)
 4. TV_Skewness_Surge_Reversal  - rolling skewness of returns surge crossing
                                   threshold = reversal expected
 5. TV_Range_Expansion_Index    - REI (Tom DeMark) - position of price within
                                   N-day range, extreme overshoot reversal
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


# ----- 1) GARCH(1,1)-STYLE VOLATILITY REGIME SWITCH -----
def gen_TV_GarchVol_RegimeSwitch(df, alpha=0.1, beta=0.85, vol_window=30,
                                   high_pct=80, **kw):
    """Simplified GARCH(1,1): sigma2[t] = (1-alpha-beta)*long_var
                                + alpha*r[t-1]^2 + beta*sigma2[t-1].
    Regime: if cond vol percentile rank > high_pct -> HIGH vol regime.
    HIGH regime: mean-revert (long when ret < -1stdev, short when > +1stdev).
    LOW regime: follow trend (long if EMA20 > EMA50, short if reverse).
    """
    a = float(alpha)
    b = float(beta)
    vw = int(vol_window)
    hp = float(high_pct)
    c = df['close'].astype(float)
    ret = c.pct_change().fillna(0)
    long_var = ret.rolling(252, min_periods=50).var(ddof=0)
    omega = 1 - a - b
    sigma2 = pd.Series(np.nan, index=df.index)
    s_prev = float(ret.iloc[:50].var()) if len(ret) > 50 else float(ret.var() or 1e-6)
    for i, r in enumerate(ret.values):
        lv = long_var.iloc[i]
        if np.isnan(lv) or lv <= 0:
            lv = s_prev
        s_prev = max(omega * lv + a * r * r + b * s_prev, 1e-10)
        sigma2.iloc[i] = s_prev
    sigma = sigma2.pow(0.5)
    sigma_rank = sigma.rolling(vw, min_periods=vw).rank(pct=True) * 100
    high_regime = sigma_rank >= hp
    ema_short = _ema(c, 20)
    ema_long = _ema(c, 50)
    in_uptrend = ema_short > ema_long
    long_setup = (high_regime & (ret < -sigma)) | (~high_regime & in_uptrend & (ema_short.shift(1) <= ema_long.shift(1)))
    short_setup = (high_regime & (ret > sigma)) | (~high_regime & ~in_uptrend & (ema_short.shift(1) >= ema_long.shift(1)))
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_GarchVol_RegimeSwitch():
    return {
        'alpha': ('float', 0.05, 0.20),
        'beta': ('float', 0.70, 0.93),
        'vol_window': ('int', 20, 60),
        'high_pct': ('int', 70, 90),
    }


# ----- 2) FRACTIONALLY DIFFERENTIATED CLOSE -----
def _fracdiff_weights(d, k_max):
    w = [1.0]
    for k in range(1, k_max + 1):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
    return np.array(w[::-1])  # most recent first


def gen_TV_FracDiff_Stationary(df, d=0.4, window=50, sma_len=20, **kw):
    """López de Prado AFL ch5 fractionally differentiated log-price.
    Since fracdiff of log-price is not zero-centered, signal is cross of
    fracdiff vs its own SMA (momentum on the partially-stationary series).
    """
    dd = float(d)
    w = int(window)
    sl = int(sma_len)
    c = df['close'].astype(float)
    log_c = np.log(c.replace(0.0, np.nan))
    weights = _fracdiff_weights(dd, w - 1)
    fd = log_c.rolling(w, min_periods=w).apply(
        lambda x: float(np.dot(weights[:len(x)], x.to_numpy())), raw=False
    )
    fd_sma = fd.rolling(sl, min_periods=sl).mean()
    cross_up = (fd > fd_sma) & (fd.shift(1) <= fd_sma.shift(1))
    cross_dn = (fd < fd_sma) & (fd.shift(1) >= fd_sma.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_FracDiff_Stationary():
    return {
        'd': ('float', 0.20, 0.70),
        'window': ('int', 30, 100),
        'sma_len': ('int', 10, 40),
    }


# ----- 3) HURST EXPONENT ADAPTIVE SWITCH -----
def gen_TV_HurstExp_Adaptive_Switch(df, hurst_win=64, ma_short=10, ma_long=30,
                                      h_trend=0.55, h_revert=0.45, **kw):
    """Hurst exponent via rescaled-range over rolling window.
    H > h_trend: trend regime (long if MA short > MA long, short if reverse).
    H < h_revert: mean-revert regime (long when close < MA short, short above).
    """
    hw = int(hurst_win)
    ms = int(ma_short)
    ml = int(ma_long)
    ht = float(h_trend)
    hr = float(h_revert)
    c = df['close'].astype(float)
    log_ret = np.log(c).diff()

    def _hurst(arr):
        n = len(arr)
        if n < 16:
            return np.nan
        a = arr.to_numpy(dtype=float)
        if np.any(np.isnan(a)):
            return np.nan
        rs_list = []
        for sub_n in [8, 16, 32, min(64, n)]:
            if sub_n > n:
                break
            chunks = n // sub_n
            rs_vals = []
            for i in range(chunks):
                seg = a[i * sub_n:(i + 1) * sub_n]
                m = seg.mean()
                z = np.cumsum(seg - m)
                R = z.max() - z.min()
                S = seg.std(ddof=0)
                if S > 0:
                    rs_vals.append(R / S)
            if rs_vals:
                rs_list.append((sub_n, np.mean(rs_vals)))
        if len(rs_list) < 2:
            return np.nan
        ns, rs = zip(*rs_list)
        log_n = np.log(ns)
        log_rs = np.log(rs)
        h = np.polyfit(log_n, log_rs, 1)[0]
        return h

    H = log_ret.rolling(hw, min_periods=hw).apply(_hurst, raw=False)
    sma_s = c.rolling(ms, min_periods=ms).mean()
    sma_l = c.rolling(ml, min_periods=ml).mean()
    trend_long = (H > ht) & (sma_s > sma_l) & (sma_s.shift(1) <= sma_l.shift(1))
    trend_short = (H > ht) & (sma_s < sma_l) & (sma_s.shift(1) >= sma_l.shift(1))
    revert_long = (H < hr) & (c < sma_s) & (c.shift(1) >= sma_s.shift(1))
    revert_short = (H < hr) & (c > sma_s) & (c.shift(1) <= sma_s.shift(1))
    long_trigger = trend_long | revert_long
    short_trigger = trend_short | revert_short
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_trigger.shift(1).fillna(False).astype(bool)] = 1
    sig[short_trigger.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_HurstExp_Adaptive_Switch():
    return {
        'hurst_win': ('int', 32, 96),
        'ma_short': ('int', 5, 15),
        'ma_long': ('int', 20, 50),
        'h_trend': ('float', 0.52, 0.62),
        'h_revert': ('float', 0.38, 0.48),
    }


# ----- 4) SKEWNESS SURGE REVERSAL -----
def gen_TV_Skewness_Surge_Reversal(df, skew_win=20, threshold=1.5, **kw):
    """Rolling skewness of returns. When |skew| > threshold, expect mean
    reversal of the surge:
     - Skew > +threshold (right-tail heavy, recent up surge) -> SHORT
     - Skew < -threshold (left-tail heavy, recent down surge) -> LONG
    """
    sw = int(skew_win)
    th = float(threshold)
    c = df['close'].astype(float)
    ret = c.pct_change()
    skew = ret.rolling(sw, min_periods=sw).skew()
    long_trigger = (skew < -th) & (skew.shift(1) >= -th)
    short_trigger = (skew > th) & (skew.shift(1) <= th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_trigger.shift(1).fillna(False).astype(bool)] = 1
    sig[short_trigger.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Skewness_Surge_Reversal():
    return {
        'skew_win': ('int', 10, 50),
        'threshold': ('float', 0.8, 2.5),
    }


# ----- 5) DEMARK RANGE EXPANSION INDEX -----
def gen_TV_Range_Expansion_Index(df, rei_len=8, ob=60, os=-60, **kw):
    """Tom DeMark's REI: measures price expansion within a window.
    Simplified: REI = sum of (high - high.shift(2)) + (low - low.shift(2))
                       over rei_len, normalized by sum of |H-L| over rei_len.
    REI > +ob: overbought, expect reversal -> SHORT
    REI < -os: oversold, expect reversal -> LONG (cross back inward)
    """
    n = int(rei_len)
    obl = float(ob)
    osl = float(os)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    h_diff = h - h.shift(2)
    l_diff = l - l.shift(2)
    num = (h_diff + l_diff).rolling(n, min_periods=n).sum()
    den = (h - l).abs().rolling(n, min_periods=n).sum().replace(0.0, np.nan)
    rei = 100.0 * num / den
    long_cross = (rei > osl) & (rei.shift(1) <= osl)  # crossing back UP from oversold
    short_cross = (rei < obl) & (rei.shift(1) >= obl)  # crossing back DOWN from OB
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Range_Expansion_Index():
    return {
        'rei_len': ('int', 5, 20),
        'ob': ('int', 40, 80),
        'os': ('int', -80, -40),
    }


STRATEGY_EXPORT = {
    "TV_GarchVol_RegimeSwitch": {
        "gen": gen_TV_GarchVol_RegimeSwitch,
        "space": space_TV_GarchVol_RegimeSwitch,
        "source": "mac_batch3708_GARCH11_Bollerslev1986",
    },
    "TV_FracDiff_Stationary": {
        "gen": gen_TV_FracDiff_Stationary,
        "space": space_TV_FracDiff_Stationary,
        "source": "mac_batch3708_LopezDePrado_AFL_ch5_2018",
    },
    "TV_HurstExp_Adaptive_Switch": {
        "gen": gen_TV_HurstExp_Adaptive_Switch,
        "space": space_TV_HurstExp_Adaptive_Switch,
        "source": "mac_batch3708_Hurst1951_RescaledRange_adaptive_regime",
    },
    "TV_Skewness_Surge_Reversal": {
        "gen": gen_TV_Skewness_Surge_Reversal,
        "space": space_TV_Skewness_Surge_Reversal,
        "source": "mac_batch3708_higherMoment_reversal",
    },
    "TV_Range_Expansion_Index": {
        "gen": gen_TV_Range_Expansion_Index,
        "space": space_TV_Range_Expansion_Index,
        "source": "mac_batch3708_DeMark_REI_NewMarketTimingTechniques",
    },
}
