"""
Batch 3722 - Mac paralela wave m23 - 5 online filtering / advanced signal.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3721.

 1. TV_Kalman_Trend_Cross    - simple 1-state Kalman filter for trend; close
                                cross of Kalman estimate (NEW; distinct from
                                sandbox AR1 forecast / Linear Regression)
 2. TV_RecursiveOLS_Slope    - rolling OLS slope of close vs time index;
                                slope sign cross zero
 3. TV_Welford_Variance_Surge - Welford-style online variance via EMA of
                                squared returns; surge cross threshold
 4. TV_LogReturn_Skew_Slope  - rolling skew slope (slope of skew vs time)
                                (DISTINCT from m12 RealizedSkew_Surge which
                                is threshold cross)
 5. TV_QuantileRegression_Slope - slope of 50-pctile (median) regression
                                line over window; sign cross
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


# ----- 1) KALMAN TREND ESTIMATE -----
def gen_TV_Kalman_Trend_Cross(df, q=0.001, r=0.1, **kw):
    """1-state Kalman filter for trend (constant velocity model):
       state_t = state_{t-1} + Gaussian(0, q*var)
       obs_t = state_t + Gaussian(0, r*var)
    Recursive update. Long: close > kalman_estimate AND fresh cross UP.
    Short: opposite.
    """
    qq = float(q)
    rr = float(r)
    c = df['close'].astype(float).to_numpy()
    n = len(c)
    if n < 5:
        return pd.Series(0, index=df.index, dtype=int)
    var_obs = float(np.var(np.diff(c)))
    if var_obs <= 0:
        var_obs = 1e-6
    Q = qq * var_obs
    R = rr * var_obs
    x = np.full(n, np.nan)
    P = np.full(n, np.nan)
    x[0] = c[0]
    P[0] = R
    for i in range(1, n):
        # Predict
        x_pred = x[i - 1]
        P_pred = P[i - 1] + Q
        # Update
        K = P_pred / (P_pred + R)
        x[i] = x_pred + K * (c[i] - x_pred)
        P[i] = (1 - K) * P_pred
    kalman = pd.Series(x, index=df.index)
    close_s = df['close'].astype(float)
    cross_up = (close_s > kalman) & (close_s.shift(1) <= kalman.shift(1))
    cross_dn = (close_s < kalman) & (close_s.shift(1) >= kalman.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Kalman_Trend_Cross():
    return {
        'q': ('float', 0.0001, 0.01),
        'r': ('float', 0.01, 1.0),
    }


# ----- 2) RECURSIVE OLS SLOPE -----
def gen_TV_RecursiveOLS_Slope(df, win=30, slope_thr=0.0, **kw):
    """Rolling OLS slope of close vs time index (0..win-1).
    slope = (sum(x*y) - n*mean(x)*mean(y)) / (sum(x^2) - n*mean(x)^2).
    Long: slope crosses above slope_thr.
    Short: crosses below -slope_thr.
    """
    w = int(win)
    th = float(slope_thr)
    c = df['close'].astype(float)
    # Vectorized rolling slope
    x = pd.Series(np.arange(len(c)), index=c.index, dtype=float)
    sum_y = c.rolling(w, min_periods=w).sum()
    sum_x = x.rolling(w, min_periods=w).sum()
    sum_xy = (x * c).rolling(w, min_periods=w).sum()
    sum_xx = (x * x).rolling(w, min_periods=w).sum()
    n_w = float(w)
    slope = (n_w * sum_xy - sum_x * sum_y) / (n_w * sum_xx - sum_x * sum_x).replace(0.0, np.nan)
    long_cross = (slope > th) & (slope.shift(1) <= th)
    short_cross = (slope < -th) & (slope.shift(1) >= -th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_RecursiveOLS_Slope():
    return {
        'win': ('int', 10, 80),
        'slope_thr': ('float', 0.0, 5.0),
    }


# ----- 3) WELFORD VARIANCE SURGE -----
def gen_TV_Welford_Variance_Surge(df, alpha=0.05, surge_mult=2.5, **kw):
    """EWMA variance: V_t = (1-alpha)*V_{t-1} + alpha*r_t^2.
    Long-term variance: long alpha. Surge: V_short > surge_mult * V_long.
    Direction = sign of recent return.
    """
    a = float(alpha)
    sm = float(surge_mult)
    c = df['close'].astype(float)
    ret = c.pct_change()
    sq = ret * ret
    v_short = sq.ewm(alpha=a, adjust=False, min_periods=20).mean()
    v_long = sq.ewm(alpha=a / 5, adjust=False, min_periods=50).mean()
    surge = (v_short > sm * v_long) & (v_short.shift(1) <= sm * v_long.shift(1))
    long_setup = surge & (ret > 0)
    short_setup = surge & (ret < 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Welford_Variance_Surge():
    return {
        'alpha': ('float', 0.02, 0.20),
        'surge_mult': ('float', 1.5, 4.0),
    }


# ----- 4) SKEW SLOPE -----
def gen_TV_LogReturn_Skew_Slope(df, skew_win=30, slope_win=10, slope_thr=0.05, **kw):
    """Slope of rolling skewness: (skew_t - skew_{t-slope_win}) / slope_win.
    Long: slope crosses above +thr (skew rising = upward asymmetry building).
    Short: crosses below -thr.
    """
    sw = int(skew_win)
    slw = int(slope_win)
    th = float(slope_thr)
    c = df['close'].astype(float)
    ret = c.pct_change()
    skew = ret.rolling(sw, min_periods=sw).skew()
    slope = (skew - skew.shift(slw)) / float(slw)
    long_cross = (slope > th) & (slope.shift(1) <= th)
    short_cross = (slope < -th) & (slope.shift(1) >= -th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_LogReturn_Skew_Slope():
    return {
        'skew_win': ('int', 15, 60),
        'slope_win': ('int', 5, 20),
        'slope_thr': ('float', 0.01, 0.20),
    }


# ----- 5) QUANTILE REGRESSION SLOPE -----
def gen_TV_QuantileRegression_Slope(df, win=30, **kw):
    """Median (50-pctile) trend slope over window:
       compute median of close in halves of window; slope = (right - left) / win/2.
    Approximation of quantile regression slope.
    Long: slope crosses above 0.
    Short: crosses below 0.
    """
    w = int(win)
    half = w // 2
    c = df['close'].astype(float)
    left_med = c.shift(half).rolling(half, min_periods=half).median()
    right_med = c.rolling(half, min_periods=half).median()
    slope = (right_med - left_med) / float(half)
    long_cross = (slope > 0) & (slope.shift(1) <= 0)
    short_cross = (slope < 0) & (slope.shift(1) >= 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_QuantileRegression_Slope():
    return {
        'win': ('int', 14, 80),
    }


STRATEGY_EXPORT = {
    "TV_Kalman_Trend_Cross": {
        "gen": gen_TV_Kalman_Trend_Cross, "space": space_TV_Kalman_Trend_Cross,
        "source": "mac_batch3722_Kalman_1state_trend_cross",
    },
    "TV_RecursiveOLS_Slope": {
        "gen": gen_TV_RecursiveOLS_Slope, "space": space_TV_RecursiveOLS_Slope,
        "source": "mac_batch3722_rolling_OLS_slope_cross",
    },
    "TV_Welford_Variance_Surge": {
        "gen": gen_TV_Welford_Variance_Surge,
        "space": space_TV_Welford_Variance_Surge,
        "source": "mac_batch3722_EWMA_variance_short_long_surge",
    },
    "TV_LogReturn_Skew_Slope": {
        "gen": gen_TV_LogReturn_Skew_Slope, "space": space_TV_LogReturn_Skew_Slope,
        "source": "mac_batch3722_rolling_skew_slope_cross",
    },
    "TV_QuantileRegression_Slope": {
        "gen": gen_TV_QuantileRegression_Slope,
        "space": space_TV_QuantileRegression_Slope,
        "source": "mac_batch3722_median_half_slope_proxy",
    },
}
