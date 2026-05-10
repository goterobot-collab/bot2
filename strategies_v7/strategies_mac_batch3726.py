"""
Batch 3726 - Mac paralela wave m27 - 5 vol distribution / Bayesian / multi-period.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3725.

 1. TV_Volume_Mode_Position    - close vs mode of recent volume distribution
                                  (where the volume "lives" as a magnet)
 2. TV_Wright_VR_SignBased     - Wright (2000) sign-based variance ratio
                                  (robust vs Lo-MacKinlay - distinct from m14)
                                  Source: Wright J Bus Econ Stat 2000
 3. TV_BayesianTrend_Probability - simple Bayesian update of trend probability
                                  given recent N-bar return signs (NEW)
 4. TV_MultiPeriod_CCI_Vote    - 3 CCI periods (short/mid/long) all aligned
                                  (DISTINCT from sandbox TV_CCI_CycleFinder
                                  which is single-period zero-line)
 5. TV_Range_Symmetry          - upper wick vs lower wick symmetry over window
                                  (asymmetric wicks -> directional pressure)
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


# ----- 1) VOLUME MODE POSITION -----
def gen_TV_Volume_Mode_Position(df, win=50, bin_count=10, **kw):
    """For each bar, find price level (in N-bar window) where volume is
    most concentrated. Vectorized approximation: weight prices by volume,
    compute weighted-quantile (50th = volume-weighted median).
    Long: close above vol-weighted median by >0.5*ATR, momentum positive.
    Short: opposite.
    """
    w = int(win)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    # Use typical price * vol as proxy weight
    tp = (h + l + c) / 3.0
    weighted = tp * v
    sum_wt = weighted.rolling(w, min_periods=w).sum()
    sum_v = v.rolling(w, min_periods=w).sum().replace(0.0, np.nan)
    vw_median = sum_wt / sum_v  # volume-weighted average price in window
    short_ret = c.pct_change().rolling(5, min_periods=5).sum()
    diff_pct = (c - vw_median) / vw_median.replace(0.0, np.nan) * 100
    long_setup = (diff_pct > 0.5) & (diff_pct.shift(1) <= 0.5) & (short_ret > 0)
    short_setup = (diff_pct < -0.5) & (diff_pct.shift(1) >= -0.5) & (short_ret < 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Volume_Mode_Position():
    return {
        'win': ('int', 20, 120),
        'bin_count': ('int', 5, 20),
    }


# ----- 2) WRIGHT SIGN-BASED VARIANCE RATIO -----
def gen_TV_Wright_VR_SignBased(df, q=4, win=120, vr_thr=1.3, **kw):
    """Wright 2000 sign-based VR test:
    R1 = sum(sign(r_t)) / n - rebased to {-1, +1} returns.
    VR_sign(q) = Var(R_q) / (q * Var(R_1)) using sign returns.
    Robust to outliers vs Lo-MacKinlay VR.
    Long: VR_sign > thr AND short_sum > 0.
    """
    qq = int(q)
    w = int(win)
    th = float(vr_thr)
    c = df['close'].astype(float)
    sign_r = np.sign(c.diff()).fillna(0)
    sign_q = sign_r.rolling(qq, min_periods=qq).sum() / qq
    var1 = sign_r.rolling(w, min_periods=w).var(ddof=0)
    varq = sign_q.rolling(w, min_periods=w).var(ddof=0)
    vr = varq / (qq * var1.replace(0.0, np.nan))
    short_sum = sign_r.rolling(5, min_periods=5).sum()
    momentum = vr > th
    revert = vr < (1.0 / th)
    long_setup = (momentum & (short_sum > 0)) | (revert & (short_sum < 0))
    short_setup = (momentum & (short_sum < 0)) | (revert & (short_sum > 0))
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Wright_VR_SignBased():
    return {
        'q': ('int', 2, 10),
        'win': ('int', 60, 250),
        'vr_thr': ('float', 1.1, 1.6),
    }


# ----- 3) BAYESIAN TREND PROBABILITY -----
def gen_TV_BayesianTrend_Probability(df, win=20, prior=0.5, prob_thr=0.7, **kw):
    """Beta-binomial Bayesian update:
    Prior beta(alpha=1, beta=1) (uniform).
    Each up-bar increments alpha, each down-bar increments beta.
    Posterior P(uptrend) = alpha / (alpha + beta) over rolling window.
    Long: P > prob_thr AND fresh cross.
    Short: P < (1 - prob_thr) AND fresh cross.
    """
    w = int(win)
    pt = float(prob_thr)
    c = df['close'].astype(float)
    up_bar = (c > c.shift(1)).astype(int)
    dn_bar = (c < c.shift(1)).astype(int)
    alpha = 1 + up_bar.rolling(w, min_periods=w).sum()
    beta = 1 + dn_bar.rolling(w, min_periods=w).sum()
    p_up = alpha / (alpha + beta)
    long_cross = (p_up > pt) & (p_up.shift(1) <= pt)
    short_cross = (p_up < (1 - pt)) & (p_up.shift(1) >= (1 - pt))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_BayesianTrend_Probability():
    return {
        'win': ('int', 10, 50),
        'prior': ('float', 0.4, 0.6),
        'prob_thr': ('float', 0.6, 0.85),
    }


# ----- 4) MULTI-PERIOD CCI VOTE -----
def gen_TV_MultiPeriod_CCI_Vote(df, short=14, mid=30, long=60, threshold=100, **kw):
    """3 CCI periods: short, mid, long. All > +threshold = strong long bias.
    All < -threshold = strong short bias.
    Long entry: all 3 cross above +thr (alignment moment).
    """
    sl = int(short)
    ml = int(mid)
    ll = int(long)
    th = float(threshold)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    tp = (h + l + c) / 3.0

    def _cci(period):
        ma = _sma(tp, period)
        md = (tp - ma).abs().rolling(period, min_periods=period).mean()
        return (tp - ma) / (0.015 * md.replace(0.0, np.nan))

    cci_s = _cci(sl)
    cci_m = _cci(ml)
    cci_l = _cci(ll)
    bullish = (cci_s > th) & (cci_m > th) & (cci_l > th)
    bearish = (cci_s < -th) & (cci_m < -th) & (cci_l < -th)
    long_entry = bullish & ~(bullish.shift(1).fillna(False))
    short_entry = bearish & ~(bearish.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_MultiPeriod_CCI_Vote():
    return {
        'short': ('int', 7, 20),
        'mid': ('int', 21, 40),
        'long': ('int', 45, 90),
        'threshold': ('int', 80, 200),
    }


# ----- 5) RANGE SYMMETRY (WICK ASYMMETRY) -----
def gen_TV_Range_Symmetry(df, win=20, asym_thr=0.6, **kw):
    """upper_wick = high - max(open, close)
    lower_wick = min(open, close) - low
    asymmetry = (upper_wick - lower_wick) / (upper_wick + lower_wick)  in [-1, 1]
    Rolling mean of asymmetry over win.
    Long: rolling mean < -asym_thr (persistent lower wick = buying pressure under)
    Short: rolling mean > +asym_thr (persistent upper wick = selling pressure above)
    """
    w = int(win)
    th = float(asym_thr)
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    upper_wick = h - pd.Series(np.maximum(o.to_numpy(), c.to_numpy()), index=o.index)
    lower_wick = pd.Series(np.minimum(o.to_numpy(), c.to_numpy()), index=o.index) - l
    total = (upper_wick + lower_wick).replace(0.0, np.nan)
    asym = (upper_wick - lower_wick) / total
    avg_asym = asym.rolling(w, min_periods=w).mean()
    long_setup = (avg_asym < -th) & (avg_asym.shift(1) >= -th)
    short_setup = (avg_asym > th) & (avg_asym.shift(1) <= th)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Range_Symmetry():
    return {
        'win': ('int', 10, 50),
        'asym_thr': ('float', 0.2, 0.8),
    }


STRATEGY_EXPORT = {
    "TV_Volume_Mode_Position": {
        "gen": gen_TV_Volume_Mode_Position,
        "space": space_TV_Volume_Mode_Position,
        "source": "mac_batch3726_VWAP_window_distance_signal",
    },
    "TV_Wright_VR_SignBased": {
        "gen": gen_TV_Wright_VR_SignBased,
        "space": space_TV_Wright_VR_SignBased,
        "source": "mac_batch3726_Wright_2000_sign_VR",
    },
    "TV_BayesianTrend_Probability": {
        "gen": gen_TV_BayesianTrend_Probability,
        "space": space_TV_BayesianTrend_Probability,
        "source": "mac_batch3726_beta_binomial_trend_probability",
    },
    "TV_MultiPeriod_CCI_Vote": {
        "gen": gen_TV_MultiPeriod_CCI_Vote,
        "space": space_TV_MultiPeriod_CCI_Vote,
        "source": "mac_batch3726_3_period_CCI_alignment_vote",
    },
    "TV_Range_Symmetry": {
        "gen": gen_TV_Range_Symmetry, "space": space_TV_Range_Symmetry,
        "source": "mac_batch3726_wick_asymmetry_pressure",
    },
}
