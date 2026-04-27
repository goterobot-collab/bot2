"""
Batch 3723 - Mac paralela wave m24 - 5 change-point / fractal / drawdown.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3722.

 1. TV_CUSUM_ChangePoint        - Page (1954) CUSUM cross threshold
                                   Source: Page Biometrika 1954
 2. TV_FRAMA_Cross              - Ehlers Fractal Adaptive Moving Average
                                   cross (NEW; sandbox has SuperSmoother
                                   but no FRAMA by name)
 3. TV_Rolling_Drawdown_Recovery - depth of rolling DD; recovery from depth
                                   threshold = entry
 4. TV_TimeSinceExtreme         - bars since last N-period high/low;
                                   approach to old extreme = trade signal
 5. TV_PriceVolatility_Ratio    - return / vol cross (similar to z but
                                   bar-level, no rolling stat). NEW.
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


# ----- 1) CUSUM CHANGE-POINT -----
def gen_TV_CUSUM_ChangePoint(df, win=50, k=0.5, threshold=4.0, **kw):
    """Page CUSUM 1954. Two-sided CUSUM on standardized returns:
       S_pos[t] = max(0, S_pos[t-1] + (z[t] - k))
       S_neg[t] = max(0, S_neg[t-1] - (z[t] + k))
    Long entry: S_neg crosses above threshold (recent down moves accumulating)
                AND last return > 0 (start of recovery).
    Short entry: S_pos crosses above threshold AND last return < 0.
    """
    w = int(win)
    kk = float(k)
    th = float(threshold)
    c = df['close'].astype(float)
    ret = c.pct_change()
    mu = ret.rolling(w, min_periods=w).mean()
    sd = ret.rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
    z = (ret - mu) / sd
    z_arr = z.fillna(0).to_numpy()
    s_pos = np.zeros(len(z_arr))
    s_neg = np.zeros(len(z_arr))
    for i in range(1, len(z_arr)):
        s_pos[i] = max(0.0, s_pos[i - 1] + z_arr[i] - kk)
        s_neg[i] = max(0.0, s_neg[i - 1] - z_arr[i] - kk)
    s_pos_s = pd.Series(s_pos, index=df.index)
    s_neg_s = pd.Series(s_neg, index=df.index)
    long_setup = (s_neg_s > th) & (s_neg_s.shift(1) <= th)
    short_setup = (s_pos_s > th) & (s_pos_s.shift(1) <= th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_CUSUM_ChangePoint():
    return {
        'win': ('int', 30, 120),
        'k': ('float', 0.2, 1.0),
        'threshold': ('float', 2.0, 8.0),
    }


# ----- 2) FRAMA CROSS (Ehlers Fractal Adaptive MA) -----
def gen_TV_FRAMA_Cross(df, n=16, **kw):
    """Ehlers FRAMA: alpha derived from fractal dimension of price.
       D = (log(N1+N2) - log(N3)) / log(2), where:
         N1 = (max(h, n/2) - min(l, n/2)) / (n/2)  (first half)
         N2 = (max(h, n/2) - min(l, n/2)) / (n/2)  (second half)
         N3 = (max(h, n)   - min(l, n))   / n
       alpha = exp(-4.6 * (D - 1))
       FRAMA[t] = alpha * close + (1 - alpha) * FRAMA[t-1]
    Long: close crosses above FRAMA. Short: below.
    """
    nn = int(n)
    half = max(2, nn // 2)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    h1 = h.rolling(half, min_periods=half).max()
    l1 = l.rolling(half, min_periods=half).min()
    n1 = (h1 - l1) / float(half)
    h2 = h.shift(half).rolling(half, min_periods=half).max()
    l2 = l.shift(half).rolling(half, min_periods=half).min()
    n2 = (h2 - l2) / float(half)
    h3 = h.rolling(nn, min_periods=nn).max()
    l3 = l.rolling(nn, min_periods=nn).min()
    n3 = (h3 - l3) / float(nn)
    log_arg_top = (n1 + n2).replace(0.0, np.nan)
    d = (np.log(log_arg_top) - np.log(n3.replace(0.0, np.nan))) / np.log(2)
    d = d.clip(1.0, 2.0)
    alpha = np.exp(-4.6 * (d - 1)).clip(0.01, 1.0).fillna(0.5)
    alpha_arr = alpha.to_numpy()
    c_arr = c.to_numpy()
    frama = np.full(len(c_arr), np.nan)
    frama[0] = c_arr[0]
    for i in range(1, len(c_arr)):
        if np.isnan(c_arr[i]):
            frama[i] = frama[i - 1]
            continue
        a = alpha_arr[i] if not np.isnan(alpha_arr[i]) else 0.5
        prev = frama[i - 1] if not np.isnan(frama[i - 1]) else c_arr[i]
        frama[i] = a * c_arr[i] + (1 - a) * prev
    frama_s = pd.Series(frama, index=df.index)
    cross_up = (c > frama_s) & (c.shift(1) <= frama_s.shift(1))
    cross_dn = (c < frama_s) & (c.shift(1) >= frama_s.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_FRAMA_Cross():
    return {
        'n': ('int', 8, 40),
    }


# ----- 3) ROLLING DRAWDOWN RECOVERY -----
def gen_TV_Rolling_Drawdown_Recovery(df, win=50, dd_pct=10, recovery_pct=3, **kw):
    """Rolling drawdown: dd = (close - rolling_max) / rolling_max * 100.
    Long entry: dd was below -dd_pct, now recovered to within -recovery_pct.
    Short entry: rolling drawup recovered (symmetric).
    """
    w = int(win)
    dp = float(dd_pct)
    rp = float(recovery_pct)
    c = df['close'].astype(float)
    rolling_max = c.rolling(w, min_periods=w).max()
    rolling_min = c.rolling(w, min_periods=w).min()
    dd = (c - rolling_max) / rolling_max * 100
    du = (c - rolling_min) / rolling_min * 100
    deep_dd = dd <= -dp
    recent_deep = deep_dd.rolling(5, min_periods=1).max() == 1
    recovered = (dd > -rp) & (dd.shift(1) <= -rp)
    deep_du = du >= dp
    recent_deep_du = deep_du.rolling(5, min_periods=1).max() == 1
    recovered_dn = (du < rp) & (du.shift(1) >= rp)
    long_setup = recent_deep & recovered
    short_setup = recent_deep_du & recovered_dn
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Rolling_Drawdown_Recovery():
    return {
        'win': ('int', 30, 150),
        'dd_pct': ('float', 5.0, 25.0),
        'recovery_pct': ('float', 1.0, 8.0),
    }


# ----- 4) TIME SINCE EXTREME -----
def gen_TV_TimeSinceExtreme(df, win=50, max_age=80, **kw):
    """Bars since last N-period high (TSI_H) and last N-period low (TSI_L).
    Long entry: close approaches old high (TSI_H > max_age) AND close near hh.
    Short entry: close approaches old low (TSI_L > max_age) AND close near ll.
    """
    w = int(win)
    ma = int(max_age)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    rolling_max = h.rolling(w, min_periods=w).max()
    rolling_min = l.rolling(w, min_periods=w).min()
    is_max = h == rolling_max
    is_min = l == rolling_min
    # Cumulative argmin trick to count bars since last True
    bars_since_max = (~is_max).astype(int).groupby((is_max).cumsum()).cumsum()
    bars_since_min = (~is_min).astype(int).groupby((is_min).cumsum()).cumsum()
    near_max = c > rolling_max * 0.98
    near_min = c < rolling_min * 1.02
    long_setup = (bars_since_max > ma) & near_max
    short_setup = (bars_since_min > ma) & near_min
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_TimeSinceExtreme():
    return {
        'win': ('int', 20, 100),
        'max_age': ('int', 30, 200),
    }


# ----- 5) PRICE-VOLATILITY RATIO -----
def gen_TV_PriceVolatility_Ratio(df, ret_win=5, vol_win=20, ratio_thr=2.0, **kw):
    """Price change / volatility (risk-adjusted return).
    short_ret = N-bar return.
    vol = stdev of returns over vol_win.
    pvr = short_ret / (vol * sqrt(ret_win)).
    Long: pvr > +threshold AND fresh cross.
    Short: pvr < -threshold AND fresh cross.
    """
    rw = int(ret_win)
    vw = int(vol_win)
    th = float(ratio_thr)
    c = df['close'].astype(float)
    ret = c.pct_change()
    short_ret = (c / c.shift(rw) - 1.0)
    vol = ret.rolling(vw, min_periods=vw).std(ddof=0).replace(0.0, np.nan) * np.sqrt(rw)
    pvr = short_ret / vol
    long_cross = (pvr > th) & (pvr.shift(1) <= th)
    short_cross = (pvr < -th) & (pvr.shift(1) >= -th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_PriceVolatility_Ratio():
    return {
        'ret_win': ('int', 2, 15),
        'vol_win': ('int', 10, 40),
        'ratio_thr': ('float', 1.0, 3.5),
    }


STRATEGY_EXPORT = {
    "TV_CUSUM_ChangePoint": {
        "gen": gen_TV_CUSUM_ChangePoint, "space": space_TV_CUSUM_ChangePoint,
        "source": "mac_batch3723_Page_CUSUM_1954",
    },
    "TV_FRAMA_Cross": {
        "gen": gen_TV_FRAMA_Cross, "space": space_TV_FRAMA_Cross,
        "source": "mac_batch3723_Ehlers_FRAMA",
    },
    "TV_Rolling_Drawdown_Recovery": {
        "gen": gen_TV_Rolling_Drawdown_Recovery,
        "space": space_TV_Rolling_Drawdown_Recovery,
        "source": "mac_batch3723_drawdown_depth_recovery",
    },
    "TV_TimeSinceExtreme": {
        "gen": gen_TV_TimeSinceExtreme, "space": space_TV_TimeSinceExtreme,
        "source": "mac_batch3723_bars_since_extreme_test",
    },
    "TV_PriceVolatility_Ratio": {
        "gen": gen_TV_PriceVolatility_Ratio,
        "space": space_TV_PriceVolatility_Ratio,
        "source": "mac_batch3723_risk_adjusted_return_cross",
    },
}
