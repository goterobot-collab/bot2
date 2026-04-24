"""
Batch 3701 - Mac paralela wave m2 - 5 oscillator / volatility families.

Owner: Mac paralela (Claude Code online, rama claude/grail-hunt-bot-noeIq).

Families chosen for ZERO overlap with sandbox 3566-3610, Mac V8 prod, or
Mac batch 3700 (wave m1):

 1. TV_WilliamsR_Extremes  - Williams %R OB/OS reversal
 2. TV_StochRSI_Divergence - StochRSI crossover at extremes
 3. TV_UltimateOsc_OB_OS   - Ultimate Oscillator (3-period composite) extremes
 4. TV_McGinley_Dynamic    - McGinley Dynamic cross with adaptive lag
 5. TV_MassIndex_Reversal  - Mass Index range expansion reversal trigger

All pickle-safe, no lambdas, signal integer in {-1, 0, 1}, .shift(1) applied.
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _rma(s, n):
    # Wilder smoothing
    return s.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ----- 1) WILLIAMS %R EXTREMES -----
def gen_TV_WilliamsR_Extremes(df, wr_len=14, os_level=-80, ob_level=-20,
                               exit_mid=-50, **kw):
    """Williams %R = (highest_high(n) - close) / (highest_high(n) - lowest_low(n)) * -100.
    Range: [-100, 0]. OS = -80 (oversold), OB = -20 (overbought).
    Long entry: %R crosses UP from below os_level.
    Short entry: %R crosses DOWN from above ob_level.
    """
    n = int(wr_len)
    os_l = float(os_level)
    ob_l = float(ob_level)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    hh = h.rolling(n, min_periods=n).max()
    ll = l.rolling(n, min_periods=n).min()
    rng = (hh - ll).replace(0.0, np.nan)
    wr = ((hh - c) / rng) * -100.0
    long_cross = (wr > os_l) & (wr.shift(1) <= os_l)
    short_cross = (wr < ob_l) & (wr.shift(1) >= ob_l)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_WilliamsR_Extremes():
    return {
        'wr_len': ('int', 7, 30),
        'os_level': ('int', -95, -70),
        'ob_level': ('int', -30, -5),
        'exit_mid': ('int', -60, -40),
    }


# ----- 2) STOCH RSI DIVERGENCE -----
def gen_TV_StochRSI_Divergence(df, rsi_len=14, stoch_len=14, k_smooth=3,
                                d_smooth=3, ob=80, os=20, **kw):
    """StochRSI = stochastic of RSI.
    k = smoothed stochrsi; d = smoothed k.
    Long entry: k crosses above d in OS region (both < os).
    Short entry: k crosses below d in OB region (both > ob).
    """
    rl = int(rsi_len)
    sl = int(stoch_len)
    ks = int(k_smooth)
    ds = int(d_smooth)
    ob_l = float(ob)
    os_l = float(os)
    c = df['close'].astype(float)
    r = _rsi(c, rl)
    lo = r.rolling(sl, min_periods=sl).min()
    hi = r.rolling(sl, min_periods=sl).max()
    stoch = (r - lo) / (hi - lo).replace(0.0, np.nan) * 100.0
    k = stoch.rolling(ks, min_periods=ks).mean()
    d = k.rolling(ds, min_periods=ds).mean()
    cross_up = (k > d) & (k.shift(1) <= d.shift(1)) & (k < os_l) & (d < os_l)
    cross_dn = (k < d) & (k.shift(1) >= d.shift(1)) & (k > ob_l) & (d > ob_l)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_StochRSI_Divergence():
    return {
        'rsi_len': ('int', 7, 21),
        'stoch_len': ('int', 7, 21),
        'k_smooth': ('int', 2, 5),
        'd_smooth': ('int', 2, 5),
        'ob': ('int', 70, 90),
        'os': ('int', 10, 30),
    }


# ----- 3) ULTIMATE OSCILLATOR OB/OS -----
def gen_TV_UltimateOsc_OB_OS(df, len1=7, len2=14, len3=28, os=30, ob=70, **kw):
    """Ultimate Oscillator = weighted average of buying pressure over 3 periods.
    BP = close - min(low, prev_close)
    TR = max(high, prev_close) - min(low, prev_close)
    UO = 100 * (4*avg7(BP)/avg7(TR) + 2*avg14(BP)/avg14(TR) + avg28(BP)/avg28(TR)) / 7
    Long: UO crosses above os level. Short: UO crosses below ob level.
    """
    n1, n2, n3 = int(len1), int(len2), int(len3)
    os_l = float(os)
    ob_l = float(ob)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    low_or_pc = pd.concat([l, pc], axis=1).min(axis=1)
    high_or_pc = pd.concat([h, pc], axis=1).max(axis=1)
    bp = c - low_or_pc
    tr = high_or_pc - low_or_pc
    def _avg_ratio(nn):
        return bp.rolling(nn, min_periods=nn).sum() / tr.rolling(nn, min_periods=nn).sum().replace(0.0, np.nan)
    uo = 100.0 * (4 * _avg_ratio(n1) + 2 * _avg_ratio(n2) + _avg_ratio(n3)) / 7.0
    long_cross = (uo > os_l) & (uo.shift(1) <= os_l)
    short_cross = (uo < ob_l) & (uo.shift(1) >= ob_l)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_UltimateOsc_OB_OS():
    return {
        'len1': ('int', 4, 12),
        'len2': ('int', 8, 20),
        'len3': ('int', 20, 40),
        'os': ('int', 15, 40),
        'ob': ('int', 60, 85),
    }


# ----- 4) MCGINLEY DYNAMIC CROSS -----
def gen_TV_McGinley_Dynamic(df, mg_len=14, confirm_bars=2, **kw):
    """McGinley Dynamic MA:
       MG[i] = MG[i-1] + (close - MG[i-1]) / (k * N * (close/MG[i-1])^4)
       k usually = 0.6. Long when close crosses above MG and holds confirm_bars bars.
    """
    n = int(mg_len)
    cb = int(confirm_bars)
    c = df['close'].astype(float)
    mg = pd.Series(index=c.index, dtype=float)
    vals = c.to_numpy()
    mg_vals = np.full_like(vals, np.nan)
    # seed with SMA
    if len(vals) >= n:
        mg_vals[n - 1] = np.nanmean(vals[:n])
        for i in range(n, len(vals)):
            prev = mg_vals[i - 1]
            if np.isnan(prev) or prev == 0:
                mg_vals[i] = vals[i]
                continue
            ratio = vals[i] / prev
            denom = 0.6 * n * (ratio ** 4)
            if denom <= 0:
                mg_vals[i] = prev
            else:
                mg_vals[i] = prev + (vals[i] - prev) / denom
    mg = pd.Series(mg_vals, index=c.index)
    above = c > mg
    below = c < mg
    # require confirm_bars of consistent state before entry
    above_hold = above.rolling(cb, min_periods=cb).sum() == cb
    below_hold = below.rolling(cb, min_periods=cb).sum() == cb
    # entry = transition from not-holding-above to holding-above
    long_trigger = above_hold & ~(above_hold.shift(1).fillna(False))
    short_trigger = below_hold & ~(below_hold.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_trigger.shift(1).fillna(False).astype(bool)] = 1
    sig[short_trigger.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_McGinley_Dynamic():
    return {
        'mg_len': ('int', 8, 50),
        'confirm_bars': ('int', 1, 5),
    }


# ----- 5) MASS INDEX REVERSAL -----
def gen_TV_MassIndex_Reversal(df, mi_len=25, ema_len=9, threshold=27.0,
                               confirm_level=26.5, **kw):
    """Mass Index: sum(EMA9(H-L) / EMA9(EMA9(H-L)), 25 periods).
    Signal: reversal bulge = MI crosses ABOVE threshold then back DOWN below
    confirm_level. Direction determined by trend of close vs its 9-EMA at
    trigger time.
    """
    n = int(mi_len)
    el = int(ema_len)
    thr = float(threshold)
    cf = float(confirm_level)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    hl = h - l
    e1 = _ema(hl, el)
    e2 = _ema(e1, el)
    ratio = e1 / e2.replace(0.0, np.nan)
    mi = ratio.rolling(n, min_periods=n).sum()
    # bulge: has been above thr in last 25 bars AND now just dropped below cf
    above_thr = (mi >= thr)
    was_above = above_thr.rolling(n, min_periods=1).sum() > 0
    just_dropped = was_above & (mi < cf) & (mi.shift(1) >= cf)
    # direction by local trend of close vs EMA9(close)
    c_ema = _ema(c, el)
    uptrend = c > c_ema
    downtrend = c < c_ema
    long_trigger = just_dropped & downtrend  # reversal of downtrend bulge = LONG
    short_trigger = just_dropped & uptrend    # reversal of uptrend bulge = SHORT
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_trigger.shift(1).fillna(False).astype(bool)] = 1
    sig[short_trigger.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_MassIndex_Reversal():
    return {
        'mi_len': ('int', 15, 40),
        'ema_len': ('int', 5, 15),
        'threshold': ('float', 25.0, 30.0),
        'confirm_level': ('float', 24.0, 28.0),
    }


STRATEGY_EXPORT = {
    "TV_WilliamsR_Extremes": {
        "gen": gen_TV_WilliamsR_Extremes,
        "space": space_TV_WilliamsR_Extremes,
        "source": "mac_batch3701",
    },
    "TV_StochRSI_Divergence": {
        "gen": gen_TV_StochRSI_Divergence,
        "space": space_TV_StochRSI_Divergence,
        "source": "mac_batch3701",
    },
    "TV_UltimateOsc_OB_OS": {
        "gen": gen_TV_UltimateOsc_OB_OS,
        "space": space_TV_UltimateOsc_OB_OS,
        "source": "mac_batch3701",
    },
    "TV_McGinley_Dynamic": {
        "gen": gen_TV_McGinley_Dynamic,
        "space": space_TV_McGinley_Dynamic,
        "source": "mac_batch3701",
    },
    "TV_MassIndex_Reversal": {
        "gen": gen_TV_MassIndex_Reversal,
        "space": space_TV_MassIndex_Reversal,
        "source": "mac_batch3701",
    },
}
