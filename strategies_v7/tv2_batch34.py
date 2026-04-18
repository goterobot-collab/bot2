#!/usr/bin/env python3
"""TV2 BATCH 34 — 30 Statistical + Quantitative Strategies 2026-04-01

  Stat_Zscore_MR           — Z-score mean reversion: long z<-thresh, short z>thresh
  Stat_Zscore_Trend        — Z-score trend filter: long z>0.5 and rising
  Stat_Percentile_Entry    — Price percentile rank: long < low_pct, short > high_pct
  Stat_Vol_Regime          — Volatility regime: low=RSI signals, high=EMA signals
  Stat_Hurst               — Hurst exponent H>0.6 trending, H<0.4 mean-reverting
  Stat_AutoCorr            — Autocorrelation lag-1: positive=momentum, negative=reversion
  Stat_Skew_Signal         — Rolling skewness: positive skew = long, negative = short
  Stat_Kurtosis            — High kurtosis + RSI<50 = bounce expected
  Stat_LinReg_Slope        — Linear regression slope > 0 and increasing
  Stat_LinReg_RSq          — LR R-squared > thresh AND slope > 0 = strong trend
  Stat_Variance_Ratio      — Variance ratio: VR>1 trending, VR<1 mean-reverting
  Stat_StdDev_Filter       — Low rolling std = calm, enter on breakout
  Stat_Corr_Signal         — Rolling correlation close/volume confirms trend
  Stat_Mean_Reversion_Band — N-sigma bands mean reversion
  Stat_Momentum_Factor     — Composite momentum: ROC + RSI + EMA score
  Stat_Entropy             — Approximate entropy: low = structured = tradeable
  Stat_Fractal_Dim         — Fractal dimension: low D = trending
  Stat_Rolling_Beta        — Rolling beta > 1 with upward close
  Stat_Relative_Strength   — Relative strength vs own MA
  Stat_Momentum_Percentile — ROC ranked vs own history: > 80th pct = long
  Stat_Efficiency_Ratio    — KAMA efficiency ratio: high ER = trending
  Stat_Price_Volume_Corr   — Price-volume correlation confirms direction
  Stat_ATR_Ratio           — ATR/close ratio low + RSI<40 = cheap entry
  Stat_Rolling_Sharpe      — Rolling Sharpe proxy: mean/std of returns
  Stat_Serial_Corr         — Serial correlation: lag-N return correlation
  Stat_Range_Efficiency    — Directed move / total range ratio
  Stat_Vol_Adjusted_Mom    — Volatility-adjusted momentum: ROC/ATR
  Stat_Trend_Strength      — OLS slope normalized by price std
  Stat_Mean_Cross_Vol      — Price crosses rolling mean with volume confirmation
  Stat_Composite_Score     — Multi-factor score: zscore+rsi+ema+vol alignment
"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=int(p), adjust=False).mean()


def _rma(s, p):
    return s.ewm(alpha=1.0 / int(p), adjust=False).mean()


def _sma(s, p):
    return s.rolling(int(p), min_periods=1).mean()


def _atr(df, p):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, p)


def _rsi(s, p):
    d  = s.diff()
    g  = d.clip(lower=0)
    l  = (-d).clip(lower=0)
    rs = _rma(g, p) / _rma(l, p).replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


# ── 1. Stat_Zscore_MR ─────────────────────────────────────────────────────────

def gen_Stat_Zscore_MR(df, z_p=30, z_thresh=2.0, **kw):
    close    = df['close']
    z_p      = int(z_p); z_thresh = float(z_thresh)
    roll_mean = close.rolling(z_p, min_periods=1).mean()
    roll_std  = close.rolling(z_p, min_periods=1).std().replace(0, 1e-9)
    z         = (close - roll_mean) / roll_std
    sig = pd.Series(0, index=df.index)
    sig[z < -z_thresh]  =  1
    sig[z >  z_thresh]  = -1
    return sig.fillna(0)


def space_Stat_Zscore_MR(trial):
    return {
        'z_p':     trial.suggest_int('z_p', 20, 60),
        'z_thresh': trial.suggest_float('z_thresh', 1.5, 3.0),
    }


# ── 2. Stat_Zscore_Trend ──────────────────────────────────────────────────────

def gen_Stat_Zscore_Trend(df, z_p=30, thresh=0.5, **kw):
    close    = df['close']
    z_p      = int(z_p); thresh = float(thresh)
    roll_mean = close.rolling(z_p, min_periods=1).mean()
    roll_std  = close.rolling(z_p, min_periods=1).std().replace(0, 1e-9)
    z         = (close - roll_mean) / roll_std
    z_rising  = z > z.shift(1)
    z_falling = z < z.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[(z >  thresh) & z_rising]   =  1
    sig[(z < -thresh) & z_falling]  = -1
    return sig.fillna(0)


def space_Stat_Zscore_Trend(trial):
    return {
        'z_p':    trial.suggest_int('z_p', 20, 60),
        'thresh': trial.suggest_float('thresh', 0.3, 1.0),
    }


# ── 3. Stat_Percentile_Entry ──────────────────────────────────────────────────

def gen_Stat_Percentile_Entry(df, per_p=50, low_pct=10.0, **kw):
    close   = df['close']
    per_p   = int(per_p); low_pct = float(low_pct)
    high_pct = 100.0 - low_pct

    def _pct_rank(x):
        if len(x) < 2:
            return 50.0
        return float((x[:-1] < x[-1]).sum() / (len(x) - 1) * 100)

    pct = close.rolling(per_p, min_periods=2).apply(_pct_rank, raw=True)
    sig = pd.Series(0, index=df.index)
    sig[pct < low_pct]   =  1
    sig[pct > high_pct]  = -1
    return sig.fillna(0)


def space_Stat_Percentile_Entry(trial):
    return {
        'per_p':   trial.suggest_int('per_p', 20, 100),
        'low_pct': trial.suggest_float('low_pct', 5.0, 20.0),
    }


# ── 4. Stat_Vol_Regime ────────────────────────────────────────────────────────

def gen_Stat_Vol_Regime(df, vol_p=30, vol_thresh=0.02, rsi_p=14, ema_p=30, **kw):
    close     = df['close']
    vol_p     = int(vol_p); rsi_p = int(rsi_p); ema_p = int(ema_p)
    vol_thresh = float(vol_thresh)
    returns   = close.pct_change()
    rolling_vol = returns.rolling(vol_p, min_periods=1).std()
    high_vol  = rolling_vol > vol_thresh
    low_vol   = ~high_vol
    rsi       = _rsi(close, rsi_p)
    ema       = _ema(close, ema_p)
    sig = pd.Series(0, index=df.index)
    sig[low_vol  & (rsi < 35)]  =  1
    sig[low_vol  & (rsi > 65)]  = -1
    sig[high_vol & (close > ema) & (close > close.shift(1))]  =  1
    sig[high_vol & (close < ema) & (close < close.shift(1))]  = -1
    return sig.fillna(0)


def space_Stat_Vol_Regime(trial):
    return {
        'vol_p':      trial.suggest_int('vol_p', 20, 50),
        'vol_thresh': trial.suggest_float('vol_thresh', 0.01, 0.03),
        'rsi_p':      trial.suggest_int('rsi_p', 7, 21),
        'ema_p':      trial.suggest_int('ema_p', 20, 60),
    }


# ── 5. Stat_Hurst ─────────────────────────────────────────────────────────────

def _hurst_rs(series):
    n = len(series)
    if n < 10:
        return 0.5
    half = n // 2
    def rs(x):
        x = np.asarray(x, dtype=float)
        mean_x = np.mean(x)
        dev = np.cumsum(x - mean_x)
        r = dev.max() - dev.min()
        s = np.std(x, ddof=1)
        return r / s if s > 1e-9 else 1.0
    rs1 = rs(series[:half])
    rs2 = rs(series[half:])
    rs_full = rs(series)
    if rs_full < 1e-9:
        return 0.5
    h = np.log((rs1 + rs2) / 2 + 1e-9) / np.log(half + 1e-9)
    return float(np.clip(h, 0.0, 1.0))


def gen_Stat_Hurst(df, hurst_p=50, ema_p=30, **kw):
    close   = df['close']
    hurst_p = int(hurst_p); ema_p = int(ema_p)
    ema     = _ema(close, ema_p)
    returns = close.pct_change().fillna(0)

    hurst_vals = returns.rolling(hurst_p, min_periods=10).apply(
        _hurst_rs, raw=True
    )

    sig = pd.Series(0, index=df.index)
    sig[(hurst_vals > 0.6) & (close > ema)]  =  1
    sig[(hurst_vals > 0.6) & (close < ema)]  = -1
    sig[(hurst_vals < 0.4) & (close < ema)]  =  1
    sig[(hurst_vals < 0.4) & (close > ema)]  = -1
    return sig.fillna(0)


def space_Stat_Hurst(trial):
    return {
        'hurst_p': trial.suggest_int('hurst_p', 30, 100),
        'ema_p':   trial.suggest_int('ema_p',   20,  60),
    }


# ── 6. Stat_AutoCorr ──────────────────────────────────────────────────────────

def gen_Stat_AutoCorr(df, ac_p=20, thresh=0.2, ema_p=30, **kw):
    close   = df['close']
    ac_p    = int(ac_p); ema_p = int(ema_p); thresh = float(thresh)
    ema     = _ema(close, ema_p)
    ret     = close.pct_change()
    ret_lag = ret.shift(1)
    ac      = ret.rolling(ac_p, min_periods=5).corr(ret_lag)
    sig = pd.Series(0, index=df.index)
    sig[(ac >  thresh) & (close > ema)]  =  1
    sig[(ac >  thresh) & (close < ema)]  = -1
    sig[(ac < -thresh) & (close < ema)]  =  1
    sig[(ac < -thresh) & (close > ema)]  = -1
    return sig.fillna(0)


def space_Stat_AutoCorr(trial):
    return {
        'ac_p':   trial.suggest_int('ac_p',  10, 30),
        'thresh': trial.suggest_float('thresh', 0.0, 0.3),
        'ema_p':  trial.suggest_int('ema_p',  20, 60),
    }


# ── 7. Stat_Skew_Signal ───────────────────────────────────────────────────────

def gen_Stat_Skew_Signal(df, skew_p=30, **kw):
    close   = df['close']
    skew_p  = int(skew_p)
    ret     = close.pct_change()
    skew    = ret.rolling(skew_p, min_periods=5).skew()
    sig = pd.Series(0, index=df.index)
    sig[skew > 0]  =  1
    sig[skew < 0]  = -1
    return sig.fillna(0)


def space_Stat_Skew_Signal(trial):
    return {
        'skew_p': trial.suggest_int('skew_p', 20, 60),
    }


# ── 8. Stat_Kurtosis ──────────────────────────────────────────────────────────

def gen_Stat_Kurtosis(df, kurt_p=30, rsi_p=14, **kw):
    close   = df['close']
    kurt_p  = int(kurt_p); rsi_p = int(rsi_p)
    ret     = close.pct_change()
    kurt    = ret.rolling(kurt_p, min_periods=5).kurt()
    rsi     = _rsi(close, rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[(kurt > 3) & (rsi < 50)]  =  1
    sig[(kurt > 3) & (rsi > 50)]  = -1
    return sig.fillna(0)


def space_Stat_Kurtosis(trial):
    return {
        'kurt_p': trial.suggest_int('kurt_p', 20, 60),
        'rsi_p':  trial.suggest_int('rsi_p',   7, 21),
    }


# ── 9. Stat_LinReg_Slope ──────────────────────────────────────────────────────

def _lr_slope(x):
    n = len(x)
    if n < 3:
        return 0.0
    xi = np.arange(n, dtype=float)
    try:
        p = np.polyfit(xi, x, 1)
        return float(p[0])
    except Exception:
        return 0.0


def gen_Stat_LinReg_Slope(df, lr_p=20, slope_thresh=0.0, **kw):
    close       = df['close']
    lr_p        = int(lr_p); slope_thresh = float(slope_thresh)
    slope       = close.rolling(lr_p, min_periods=3).apply(_lr_slope, raw=True)
    slope_rising = slope > slope.shift(1)
    slope_fall   = slope < slope.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[(slope >  slope_thresh) & slope_rising]  =  1
    sig[(slope < -slope_thresh) & slope_fall]    = -1
    return sig.fillna(0)


def space_Stat_LinReg_Slope(trial):
    return {
        'lr_p':        trial.suggest_int('lr_p', 10, 40),
        'slope_thresh': trial.suggest_float('slope_thresh', 0.0, 0.001),
    }


# ── 10. Stat_LinReg_RSq ───────────────────────────────────────────────────────

def _lr_rsq_slope(x):
    n = len(x)
    if n < 3:
        return (0.0, 0.0)
    xi = np.arange(n, dtype=float)
    try:
        p = np.polyfit(xi, x, 1)
        y_hat = np.polyval(p, xi)
        ss_res = np.sum((x - y_hat) ** 2)
        ss_tot = np.sum((x - np.mean(x)) ** 2)
        rsq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return (float(np.clip(rsq, 0.0, 1.0)), float(p[0]))
    except Exception:
        return (0.0, 0.0)


def gen_Stat_LinReg_RSq(df, lr_p=20, rsq_thresh=0.7, **kw):
    close      = df['close']
    lr_p       = int(lr_p); rsq_thresh = float(rsq_thresh)

    rsq_vals   = close.rolling(lr_p, min_periods=3).apply(
        lambda x: _lr_rsq_slope(x)[0], raw=True)
    slope_vals = close.rolling(lr_p, min_periods=3).apply(
        lambda x: _lr_rsq_slope(x)[1], raw=True)

    sig = pd.Series(0, index=df.index)
    sig[(rsq_vals > rsq_thresh) & (slope_vals > 0)]  =  1
    sig[(rsq_vals > rsq_thresh) & (slope_vals < 0)]  = -1
    return sig.fillna(0)


def space_Stat_LinReg_RSq(trial):
    return {
        'lr_p':       trial.suggest_int('lr_p', 10, 40),
        'rsq_thresh': trial.suggest_float('rsq_thresh', 0.6, 0.9),
    }


# ── 11. Stat_Variance_Ratio ───────────────────────────────────────────────────

def gen_Stat_Variance_Ratio(df, vr_p1=5, vr_p2=20, ema_p=30, **kw):
    close  = df['close']
    vr_p1  = int(vr_p1); vr_p2 = int(vr_p2); ema_p = int(ema_p)
    ret    = close.pct_change()
    var1   = ret.rolling(vr_p1, min_periods=2).var().replace(0, 1e-9)
    var2   = ret.rolling(vr_p2, min_periods=2).var().replace(0, 1e-9)
    vr     = var2 / (vr_p2 / vr_p1 * var1)
    ema    = _ema(close, ema_p)
    sig = pd.Series(0, index=df.index)
    sig[(vr > 1) & (close > ema)]  =  1
    sig[(vr > 1) & (close < ema)]  = -1
    sig[(vr < 1) & (close < ema)]  =  1
    sig[(vr < 1) & (close > ema)]  = -1
    return sig.fillna(0)


def space_Stat_Variance_Ratio(trial):
    return {
        'vr_p1': trial.suggest_int('vr_p1',  5, 10),
        'vr_p2': trial.suggest_int('vr_p2', 20, 40),
        'ema_p': trial.suggest_int('ema_p',  20, 60),
    }


# ── 12. Stat_StdDev_Filter ────────────────────────────────────────────────────

def gen_Stat_StdDev_Filter(df, std_p=20, std_thresh=0.02, **kw):
    close      = df['close']
    std_p      = int(std_p); std_thresh = float(std_thresh)
    ret        = close.pct_change()
    rolling_std = ret.rolling(std_p, min_periods=2).std()
    low_vol    = rolling_std < std_thresh
    breakout_up = close > close.rolling(std_p).max().shift(1)
    breakout_dn = close < close.rolling(std_p).min().shift(1)
    sig = pd.Series(0, index=df.index)
    sig[low_vol.shift(1) & breakout_up]  =  1
    sig[low_vol.shift(1) & breakout_dn]  = -1
    return sig.fillna(0)


def space_Stat_StdDev_Filter(trial):
    return {
        'std_p':      trial.suggest_int('std_p', 10, 30),
        'std_thresh': trial.suggest_float('std_thresh', 0.01, 0.03),
    }


# ── 13. Stat_Corr_Signal ──────────────────────────────────────────────────────

def gen_Stat_Corr_Signal(df, corr_p=20, thresh=0.5, ema_p=30, **kw):
    close   = df['close']
    vol     = df['volume']
    corr_p  = int(corr_p); ema_p = int(ema_p); thresh = float(thresh)
    ema     = _ema(close, ema_p)
    corr    = close.rolling(corr_p, min_periods=5).corr(vol)
    sig = pd.Series(0, index=df.index)
    sig[(corr >  thresh) & (close > ema)]  =  1
    sig[(corr >  thresh) & (close < ema)]  = -1
    sig[(corr < -thresh) & (close < ema)]  =  1
    sig[(corr < -thresh) & (close > ema)]  = -1
    return sig.fillna(0)


def space_Stat_Corr_Signal(trial):
    return {
        'corr_p': trial.suggest_int('corr_p', 10, 30),
        'thresh': trial.suggest_float('thresh', 0.3, 0.7),
        'ema_p':  trial.suggest_int('ema_p',   20, 60),
    }


# ── 14. Stat_Mean_Reversion_Band ──────────────────────────────────────────────

def gen_Stat_Mean_Reversion_Band(df, mr_p=30, sigma=2.0, **kw):
    close   = df['close']
    mr_p    = int(mr_p); sigma = float(sigma)
    mean    = close.rolling(mr_p, min_periods=1).mean()
    std     = close.rolling(mr_p, min_periods=1).std().replace(0, 1e-9)
    lower   = mean - sigma * std
    upper   = mean + sigma * std
    sig = pd.Series(0, index=df.index)
    sig[close < lower]  =  1
    sig[close > upper]  = -1
    return sig.fillna(0)


def space_Stat_Mean_Reversion_Band(trial):
    return {
        'mr_p':  trial.suggest_int('mr_p', 20, 60),
        'sigma': trial.suggest_float('sigma', 1.5, 3.0),
    }


# ── 15. Stat_Momentum_Factor ──────────────────────────────────────────────────

def gen_Stat_Momentum_Factor(df, roc_p=20, rsi_p=14, ema_p=30, **kw):
    close   = df['close']
    roc_p   = int(roc_p); rsi_p = int(rsi_p); ema_p = int(ema_p)
    roc     = close.pct_change(roc_p)
    rsi     = _rsi(close, rsi_p)
    ema     = _ema(close, ema_p)
    score = (
        (roc > 0).astype(float) +
        ((rsi - 50) / 50.0).clip(-1, 1) +
        ((close - ema) / ema.replace(0, 1e-9)).clip(-1, 1)
    )
    sig = pd.Series(0, index=df.index)
    sig[score > 0.5]   =  1
    sig[score < -0.5]  = -1
    return sig.fillna(0)


def space_Stat_Momentum_Factor(trial):
    return {
        'roc_p': trial.suggest_int('roc_p', 10, 30),
        'rsi_p': trial.suggest_int('rsi_p',  7, 21),
        'ema_p': trial.suggest_int('ema_p',  20, 60),
    }


# ── 16. Stat_Entropy ──────────────────────────────────────────────────────────

def _approx_entropy(x, m=2, r=0.2):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 2:
        return 1.0
    std_x = np.std(x)
    if std_x < 1e-9:
        return 0.0
    r_val = r * std_x

    def phi(m_val):
        templates = np.array([x[i:i + m_val] for i in range(n - m_val + 1)])
        count = 0
        total = 0
        for i in range(len(templates)):
            diffs = np.abs(templates - templates[i]).max(axis=1)
            count += (diffs <= r_val).sum()
            total += 1
        return np.log(count / total + 1e-9) if total > 0 else 0.0

    return float(phi(m) - phi(m + 1))


def gen_Stat_Entropy(df, entropy_p=20, m=2, r=0.2, **kw):
    close     = df['close']
    entropy_p = int(entropy_p); m = int(m); r = float(r)
    ema       = _ema(close, 30)
    ret       = close.pct_change().fillna(0)

    def _ent_apply(x):
        return _approx_entropy(x, m=m, r=r)

    ent_vals = ret.rolling(entropy_p, min_periods=entropy_p).apply(_ent_apply, raw=True)
    ent_med  = ent_vals.rolling(entropy_p * 2, min_periods=1).median()

    sig = pd.Series(0, index=df.index)
    sig[(ent_vals < ent_med) & (close > ema)]  =  1
    sig[(ent_vals < ent_med) & (close < ema)]  = -1
    return sig.fillna(0)


def space_Stat_Entropy(trial):
    return {
        'entropy_p': trial.suggest_int('entropy_p', 10, 30),
        'm':         trial.suggest_int('m', 2, 3),
        'r':         trial.suggest_float('r', 0.1, 0.3),
    }


# ── 17. Stat_Fractal_Dim ──────────────────────────────────────────────────────

def _fractal_dim(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return 1.5
    lo = np.min(x); hi = np.max(x)
    if hi - lo < 1e-9:
        return 1.0
    n1 = n // 2
    n2 = n - n1
    lo1 = np.min(x[:n1]); hi1 = np.max(x[:n1])
    lo2 = np.min(x[n1:]); hi2 = np.max(x[n1:])
    l1 = hi1 - lo1; l2 = hi2 - lo2
    l_total = hi - lo
    if l_total < 1e-9:
        return 1.0
    d = 1 + np.log((l1 + l2) / l_total + 1e-9) / np.log(2.0)
    return float(np.clip(d, 1.0, 2.0))


def gen_Stat_Fractal_Dim(df, fd_p=20, ema_p=30, **kw):
    close  = df['close']
    fd_p   = int(fd_p); ema_p = int(ema_p)
    ema    = _ema(close, ema_p)
    fd     = close.rolling(fd_p, min_periods=4).apply(_fractal_dim, raw=True)
    sig = pd.Series(0, index=df.index)
    sig[(fd < 1.3) & (close > ema)]  =  1
    sig[(fd < 1.3) & (close < ema)]  = -1
    sig[(fd > 1.7) & (close < ema)]  =  1
    sig[(fd > 1.7) & (close > ema)]  = -1
    return sig.fillna(0)


def space_Stat_Fractal_Dim(trial):
    return {
        'fd_p':  trial.suggest_int('fd_p',  10, 40),
        'ema_p': trial.suggest_int('ema_p', 20, 60),
    }


# ── 18. Stat_Rolling_Beta ─────────────────────────────────────────────────────

def gen_Stat_Rolling_Beta(df, beta_p=30, **kw):
    close   = df['close']
    beta_p  = int(beta_p)
    ret     = close.pct_change()
    ret_lag = ret.shift(1)
    cov     = ret.rolling(beta_p, min_periods=5).cov(ret_lag)
    var     = ret_lag.rolling(beta_p, min_periods=5).var().replace(0, 1e-9)
    beta    = cov / var
    sig = pd.Series(0, index=df.index)
    sig[(beta > 1) & (ret > 0)]  =  1
    sig[(beta > 1) & (ret < 0)]  = -1
    return sig.fillna(0)


def space_Stat_Rolling_Beta(trial):
    return {
        'beta_p': trial.suggest_int('beta_p', 20, 60),
    }


# ── 19. Stat_Relative_Strength ────────────────────────────────────────────────

def gen_Stat_Relative_Strength(df, rs_p=30, threshold=0.02, **kw):
    close     = df['close']
    rs_p      = int(rs_p); threshold = float(threshold)
    sma       = _sma(close, rs_p)
    rs_rel    = close / sma.replace(0, 1e-9) - 1.0
    sig = pd.Series(0, index=df.index)
    sig[rs_rel >  threshold]  =  1
    sig[rs_rel < -threshold]  = -1
    return sig.fillna(0)


def space_Stat_Relative_Strength(trial):
    return {
        'rs_p':      trial.suggest_int('rs_p', 20, 60),
        'threshold': trial.suggest_float('threshold', 0.01, 0.05),
    }


# ── 20. Stat_Momentum_Percentile ──────────────────────────────────────────────

def gen_Stat_Momentum_Percentile(df, roc_p=20, rank_p=100, **kw):
    close   = df['close']
    roc_p   = int(roc_p); rank_p = int(rank_p)
    roc     = close.pct_change(roc_p)

    def _rank(x):
        if len(x) < 2:
            return 50.0
        return float((x[:-1] < x[-1]).sum() / (len(x) - 1) * 100)

    roc_rank = roc.rolling(rank_p, min_periods=2).apply(_rank, raw=True)
    sig = pd.Series(0, index=df.index)
    sig[roc_rank > 80]  =  1
    sig[roc_rank < 20]  = -1
    return sig.fillna(0)


def space_Stat_Momentum_Percentile(trial):
    return {
        'roc_p':  trial.suggest_int('roc_p',  10, 30),
        'rank_p': trial.suggest_int('rank_p', 50, 200),
    }


# ── 21. Stat_Efficiency_Ratio ─────────────────────────────────────────────────

def gen_Stat_Efficiency_Ratio(df, er_p=10, thresh=0.5, ema_p=30, **kw):
    close   = df['close']
    er_p    = int(er_p); ema_p = int(ema_p); thresh = float(thresh)
    ema     = _ema(close, ema_p)
    change  = (close - close.shift(er_p)).abs()
    path    = close.diff().abs().rolling(er_p, min_periods=1).sum().replace(0, 1e-9)
    er      = change / path
    sig = pd.Series(0, index=df.index)
    sig[(er > thresh) & (close > ema)]  =  1
    sig[(er > thresh) & (close < ema)]  = -1
    return sig.fillna(0)


def space_Stat_Efficiency_Ratio(trial):
    return {
        'er_p':   trial.suggest_int('er_p',  5, 20),
        'thresh': trial.suggest_float('thresh', 0.4, 0.7),
        'ema_p':  trial.suggest_int('ema_p',  20, 60),
    }


# ── 22. Stat_Price_Volume_Corr ────────────────────────────────────────────────

def gen_Stat_Price_Volume_Corr(df, pv_p=20, thresh=0.3, ema_p=30, **kw):
    close   = df['close']
    vol     = df['volume']
    pv_p    = int(pv_p); ema_p = int(ema_p); thresh = float(thresh)
    ema     = _ema(close, ema_p)
    pct_chg = close.pct_change()
    corr_pv = pct_chg.rolling(pv_p, min_periods=5).corr(vol)
    sig = pd.Series(0, index=df.index)
    sig[(corr_pv >  thresh) & (close > ema)]  =  1
    sig[(corr_pv >  thresh) & (close < ema)]  = -1
    sig[(corr_pv < -thresh) & (close < ema)]  =  1
    sig[(corr_pv < -thresh) & (close > ema)]  = -1
    return sig.fillna(0)


def space_Stat_Price_Volume_Corr(trial):
    return {
        'pv_p':   trial.suggest_int('pv_p',  10, 30),
        'thresh': trial.suggest_float('thresh', 0.2, 0.5),
        'ema_p':  trial.suggest_int('ema_p',   20, 60),
    }


# ── 23. Stat_ATR_Ratio ────────────────────────────────────────────────────────

def gen_Stat_ATR_Ratio(df, atr_p=14, ratio_max=0.02, rsi_p=14, **kw):
    close     = df['close']
    atr_p     = int(atr_p); rsi_p = int(rsi_p); ratio_max = float(ratio_max)
    atr       = _atr(df, atr_p)
    ratio     = atr / close.replace(0, 1e-9)
    rsi       = _rsi(close, rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[(ratio < ratio_max) & (rsi < 40)]  =  1
    sig[(ratio < ratio_max) & (rsi > 60)]  = -1
    return sig.fillna(0)


def space_Stat_ATR_Ratio(trial):
    return {
        'atr_p':     trial.suggest_int('atr_p', 10, 20),
        'ratio_max': trial.suggest_float('ratio_max', 0.01, 0.04),
        'rsi_p':     trial.suggest_int('rsi_p',  7, 21),
    }


# ── 24. Stat_Rolling_Sharpe ───────────────────────────────────────────────────

def gen_Stat_Rolling_Sharpe(df, sharp_p=30, **kw):
    close    = df['close']
    sharp_p  = int(sharp_p)
    ret      = close.pct_change()
    roll_mean = ret.rolling(sharp_p, min_periods=5).mean()
    roll_std  = ret.rolling(sharp_p, min_periods=5).std().replace(0, 1e-9)
    sharpe   = roll_mean / roll_std * np.sqrt(252)
    sig = pd.Series(0, index=df.index)
    sig[sharpe >  0.5]  =  1
    sig[sharpe < -0.5]  = -1
    return sig.fillna(0)


def space_Stat_Rolling_Sharpe(trial):
    return {
        'sharp_p': trial.suggest_int('sharp_p', 20, 60),
    }


# ── 25. Stat_Serial_Corr ──────────────────────────────────────────────────────

def gen_Stat_Serial_Corr(df, lag_p=5, corr_p=30, thresh=0.2, **kw):
    close   = df['close']
    lag_p   = int(lag_p); corr_p = int(corr_p); thresh = float(thresh)
    ret     = close.pct_change()
    ret_lag = ret.shift(lag_p)
    serial  = ret.rolling(corr_p, min_periods=5).corr(ret_lag)
    sig = pd.Series(0, index=df.index)
    ema = _ema(close, 30)
    sig[(serial >  thresh) & (close > ema)]  =  1
    sig[(serial >  thresh) & (close < ema)]  = -1
    sig[(serial < -thresh) & (close < ema)]  =  1
    sig[(serial < -thresh) & (close > ema)]  = -1
    return sig.fillna(0)


def space_Stat_Serial_Corr(trial):
    return {
        'lag_p':  trial.suggest_int('lag_p',   5, 20),
        'corr_p': trial.suggest_int('corr_p',  20, 60),
        'thresh': trial.suggest_float('thresh', 0.1, 0.3),
    }


# ── 26. Stat_Range_Efficiency ─────────────────────────────────────────────────

def gen_Stat_Range_Efficiency(df, re_p=10, **kw):
    close  = df['close']
    high   = df['high']
    low    = df['low']
    re_p   = int(re_p)
    direct = (close - close.shift(re_p)).abs()
    total  = (high - low).rolling(re_p, min_periods=1).sum().replace(0, 1e-9)
    re     = direct / total
    up_move = close > close.shift(re_p)
    dn_move = close < close.shift(re_p)
    sig = pd.Series(0, index=df.index)
    sig[(re > 0.6) & up_move]  =  1
    sig[(re > 0.6) & dn_move]  = -1
    return sig.fillna(0)


def space_Stat_Range_Efficiency(trial):
    return {
        're_p': trial.suggest_int('re_p', 5, 20),
    }


# ── 27. Stat_Vol_Adjusted_Mom ─────────────────────────────────────────────────

def gen_Stat_Vol_Adjusted_Mom(df, roc_p=20, atr_p=14, threshold=1.0, **kw):
    close     = df['close']
    roc_p     = int(roc_p); atr_p = int(atr_p); threshold = float(threshold)
    roc       = close.pct_change(roc_p)
    atr       = _atr(df, atr_p)
    atr_pct   = atr / close.replace(0, 1e-9)
    vol_adj   = roc / atr_pct.replace(0, 1e-9)
    sig = pd.Series(0, index=df.index)
    sig[vol_adj >  threshold]  =  1
    sig[vol_adj < -threshold]  = -1
    return sig.fillna(0)


def space_Stat_Vol_Adjusted_Mom(trial):
    return {
        'roc_p':     trial.suggest_int('roc_p', 10, 30),
        'atr_p':     trial.suggest_int('atr_p', 10, 20),
        'threshold': trial.suggest_float('threshold', 0.5, 3.0),
    }


# ── 28. Stat_Trend_Strength ───────────────────────────────────────────────────

def gen_Stat_Trend_Strength(df, trend_p=20, threshold=1.0, **kw):
    close     = df['close']
    trend_p   = int(trend_p); threshold = float(threshold)
    slope     = close.rolling(trend_p, min_periods=3).apply(_lr_slope, raw=True)
    std_close = close.rolling(trend_p, min_periods=3).std().replace(0, 1e-9)
    norm_slope = slope / std_close
    sig = pd.Series(0, index=df.index)
    sig[norm_slope >  threshold]  =  1
    sig[norm_slope < -threshold]  = -1
    return sig.fillna(0)


def space_Stat_Trend_Strength(trial):
    return {
        'trend_p':   trial.suggest_int('trend_p', 10, 40),
        'threshold': trial.suggest_float('threshold', 0.5, 2.0),
    }


# ── 29. Stat_Mean_Cross_Vol ───────────────────────────────────────────────────

def gen_Stat_Mean_Cross_Vol(df, mean_p=30, vol_p=20, vol_mult=1.5, **kw):
    close    = df['close']
    vol      = df['volume']
    mean_p   = int(mean_p); vol_p = int(vol_p); vol_mult = float(vol_mult)
    mean     = _sma(close, mean_p)
    vol_avg  = _sma(vol, vol_p)
    cross_up = (close > mean) & (close.shift(1) <= mean.shift(1))
    cross_dn = (close < mean) & (close.shift(1) >= mean.shift(1))
    vol_confirm = vol > vol_avg * vol_mult
    sig = pd.Series(0, index=df.index)
    sig[cross_up & vol_confirm]  =  1
    sig[cross_dn & vol_confirm]  = -1
    return sig.fillna(0)


def space_Stat_Mean_Cross_Vol(trial):
    return {
        'mean_p':   trial.suggest_int('mean_p', 20, 60),
        'vol_p':    trial.suggest_int('vol_p',  20, 50),
        'vol_mult': trial.suggest_float('vol_mult', 1.2, 2.5),
    }


# ── 30. Stat_Composite_Score ──────────────────────────────────────────────────

def gen_Stat_Composite_Score(df, z_p=30, rsi_p=14, ema_p=30, vol_p=20, **kw):
    close   = df['close']
    vol     = df['volume']
    z_p     = int(z_p); rsi_p = int(rsi_p); ema_p = int(ema_p); vol_p = int(vol_p)
    roll_mean = close.rolling(z_p, min_periods=1).mean()
    roll_std  = close.rolling(z_p, min_periods=1).std().replace(0, 1e-9)
    z         = (close - roll_mean) / roll_std
    rsi       = _rsi(close, rsi_p)
    ema       = _ema(close, ema_p)
    vol_avg   = _sma(vol, vol_p)
    bull_score = (
        (z < 0).astype(int) +
        (rsi < 50).astype(int) +
        (close > ema).astype(int) +
        (vol > vol_avg).astype(int)
    )
    bear_score = (
        (z > 0).astype(int) +
        (rsi > 50).astype(int) +
        (close < ema).astype(int) +
        (vol > vol_avg).astype(int)
    )
    sig = pd.Series(0, index=df.index)
    sig[bull_score >= 3]  =  1
    sig[bear_score >= 3]  = -1
    return sig.fillna(0)


def space_Stat_Composite_Score(trial):
    return {
        'z_p':   trial.suggest_int('z_p',   20, 60),
        'rsi_p': trial.suggest_int('rsi_p',  7, 21),
        'ema_p': trial.suggest_int('ema_p',  20, 60),
        'vol_p': trial.suggest_int('vol_p',  20, 50),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Stat_Zscore_MR': {
        'gen': gen_Stat_Zscore_MR,
        'space': space_Stat_Zscore_MR,
        'default_params': {'z_p': 30, 'z_thresh': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Z-score mean reversion: long when z<-thresh, short when z>thresh.'},
    },
    'Stat_Zscore_Trend': {
        'gen': gen_Stat_Zscore_Trend,
        'space': space_Stat_Zscore_Trend,
        'default_params': {'z_p': 30, 'thresh': 0.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2800,
                 'description': 'Z-score trend filter: long when z>thresh and rising.'},
    },
    'Stat_Percentile_Entry': {
        'gen': gen_Stat_Percentile_Entry,
        'space': space_Stat_Percentile_Entry,
        'default_params': {'per_p': 50, 'low_pct': 10.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2600,
                 'description': 'Price percentile rank: long at extreme lows, short at extreme highs.'},
    },
    'Stat_Vol_Regime': {
        'gen': gen_Stat_Vol_Regime,
        'space': space_Stat_Vol_Regime,
        'default_params': {'vol_p': 30, 'vol_thresh': 0.02, 'rsi_p': 14, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3200,
                 'description': 'Volatility regime: low vol uses RSI MR, high vol uses EMA trend.'},
    },
    'Stat_Hurst': {
        'gen': gen_Stat_Hurst,
        'space': space_Stat_Hurst,
        'default_params': {'hurst_p': 50, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3800,
                 'description': 'Hurst exponent: H>0.6 trending (EMA follow), H<0.4 mean-reverting.'},
    },
    'Stat_AutoCorr': {
        'gen': gen_Stat_AutoCorr,
        'space': space_Stat_AutoCorr,
        'default_params': {'ac_p': 20, 'thresh': 0.2, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2900,
                 'description': 'Autocorrelation lag-1: positive=momentum, negative=mean-reversion.'},
    },
    'Stat_Skew_Signal': {
        'gen': gen_Stat_Skew_Signal,
        'space': space_Stat_Skew_Signal,
        'default_params': {'skew_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2400,
                 'description': 'Rolling skewness: positive skew = long, negative skew = short.'},
    },
    'Stat_Kurtosis': {
        'gen': gen_Stat_Kurtosis,
        'space': space_Stat_Kurtosis,
        'default_params': {'kurt_p': 30, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2200,
                 'description': 'High kurtosis (fat tails) + RSI directional filter = extreme move entry.'},
    },
    'Stat_LinReg_Slope': {
        'gen': gen_Stat_LinReg_Slope,
        'space': space_Stat_LinReg_Slope,
        'default_params': {'lr_p': 20, 'slope_thresh': 0.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3100,
                 'description': 'Linear regression slope positive and increasing = trend continuation.'},
    },
    'Stat_LinReg_RSq': {
        'gen': gen_Stat_LinReg_RSq,
        'space': space_Stat_LinReg_RSq,
        'default_params': {'lr_p': 20, 'rsq_thresh': 0.7},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3400,
                 'description': 'LR R-squared high = strong linear trend; combined with slope direction.'},
    },
    'Stat_Variance_Ratio': {
        'gen': gen_Stat_Variance_Ratio,
        'space': space_Stat_Variance_Ratio,
        'default_params': {'vr_p1': 5, 'vr_p2': 20, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Variance ratio test: VR>1 trending, VR<1 mean-reverting with EMA filter.'},
    },
    'Stat_StdDev_Filter': {
        'gen': gen_Stat_StdDev_Filter,
        'space': space_Stat_StdDev_Filter,
        'default_params': {'std_p': 20, 'std_thresh': 0.02},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2700,
                 'description': 'Low rolling std = calm before storm; enter on subsequent breakout.'},
    },
    'Stat_Corr_Signal': {
        'gen': gen_Stat_Corr_Signal,
        'space': space_Stat_Corr_Signal,
        'default_params': {'corr_p': 20, 'thresh': 0.5, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Rolling close/volume correlation confirms directional trend.'},
    },
    'Stat_Mean_Reversion_Band': {
        'gen': gen_Stat_Mean_Reversion_Band,
        'space': space_Stat_Mean_Reversion_Band,
        'default_params': {'mr_p': 30, 'sigma': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3600,
                 'description': 'N-sigma statistical bands: long below lower, short above upper.'},
    },
    'Stat_Momentum_Factor': {
        'gen': gen_Stat_Momentum_Factor,
        'space': space_Stat_Momentum_Factor,
        'default_params': {'roc_p': 20, 'rsi_p': 14, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3300,
                 'description': 'Composite factor: ROC + RSI + EMA deviation combined score.'},
    },
    'Stat_Entropy': {
        'gen': gen_Stat_Entropy,
        'space': space_Stat_Entropy,
        'default_params': {'entropy_p': 20, 'm': 2, 'r': 0.2},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2800,
                 'description': 'Approximate entropy: low entropy = structured market = tradeable.'},
    },
    'Stat_Fractal_Dim': {
        'gen': gen_Stat_Fractal_Dim,
        'space': space_Stat_Fractal_Dim,
        'default_params': {'fd_p': 20, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2600,
                 'description': 'Fractal dimension: low D = trending, high D = random/choppy.'},
    },
    'Stat_Rolling_Beta': {
        'gen': gen_Stat_Rolling_Beta,
        'space': space_Stat_Rolling_Beta,
        'default_params': {'beta_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2300,
                 'description': 'Rolling beta > 1 with directional confirmation.'},
    },
    'Stat_Relative_Strength': {
        'gen': gen_Stat_Relative_Strength,
        'space': space_Stat_Relative_Strength,
        'default_params': {'rs_p': 30, 'threshold': 0.02},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2900,
                 'description': 'Relative strength vs own MA: outperformance = momentum.'},
    },
    'Stat_Momentum_Percentile': {
        'gen': gen_Stat_Momentum_Percentile,
        'space': space_Stat_Momentum_Percentile,
        'default_params': {'roc_p': 20, 'rank_p': 100},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3100,
                 'description': 'ROC percentile rank: >80th = strong momentum, <20th = extreme weakness.'},
    },
    'Stat_Efficiency_Ratio': {
        'gen': gen_Stat_Efficiency_Ratio,
        'space': space_Stat_Efficiency_Ratio,
        'default_params': {'er_p': 10, 'thresh': 0.5, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3400,
                 'description': 'KAMA efficiency ratio: high ER = efficient move = trend signal.'},
    },
    'Stat_Price_Volume_Corr': {
        'gen': gen_Stat_Price_Volume_Corr,
        'space': space_Stat_Price_Volume_Corr,
        'default_params': {'pv_p': 20, 'thresh': 0.3, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2700,
                 'description': 'Price-change / volume correlation confirms directional moves.'},
    },
    'Stat_ATR_Ratio': {
        'gen': gen_Stat_ATR_Ratio,
        'space': space_Stat_ATR_Ratio,
        'default_params': {'atr_p': 14, 'ratio_max': 0.02, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2400,
                 'description': 'ATR/close ratio: low = calm = cheap entry opportunity with RSI filter.'},
    },
    'Stat_Rolling_Sharpe': {
        'gen': gen_Stat_Rolling_Sharpe,
        'space': space_Stat_Rolling_Sharpe,
        'default_params': {'sharp_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3200,
                 'description': 'Rolling Sharpe proxy (mean/std returns * sqrt(252)): risk-adjusted momentum.'},
    },
    'Stat_Serial_Corr': {
        'gen': gen_Stat_Serial_Corr,
        'space': space_Stat_Serial_Corr,
        'default_params': {'lag_p': 5, 'corr_p': 30, 'thresh': 0.2},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2800,
                 'description': 'Serial correlation: positive = momentum, negative = mean reversion.'},
    },
    'Stat_Range_Efficiency': {
        'gen': gen_Stat_Range_Efficiency,
        'space': space_Stat_Range_Efficiency,
        'default_params': {'re_p': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Range efficiency: directed move / total range > 0.6 = directional.'},
    },
    'Stat_Vol_Adjusted_Mom': {
        'gen': gen_Stat_Vol_Adjusted_Mom,
        'space': space_Stat_Vol_Adjusted_Mom,
        'default_params': {'roc_p': 20, 'atr_p': 14, 'threshold': 1.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Volatility-adjusted momentum: ROC/ATR normalizes for regime changes.'},
    },
    'Stat_Trend_Strength': {
        'gen': gen_Stat_Trend_Strength,
        'space': space_Stat_Trend_Strength,
        'default_params': {'trend_p': 20, 'threshold': 1.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3300,
                 'description': 'OLS slope normalized by price std: standardized trend direction signal.'},
    },
    'Stat_Mean_Cross_Vol': {
        'gen': gen_Stat_Mean_Cross_Vol,
        'space': space_Stat_Mean_Cross_Vol,
        'default_params': {'mean_p': 30, 'vol_p': 20, 'vol_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2600,
                 'description': 'Price crosses rolling mean with volume surge confirmation.'},
    },
    'Stat_Composite_Score': {
        'gen': gen_Stat_Composite_Score,
        'space': space_Stat_Composite_Score,
        'default_params': {'z_p': 30, 'rsi_p': 14, 'ema_p': 30, 'vol_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3700,
                 'description': 'Multi-factor score: z-score + RSI + EMA + volume alignment >= 3/4.'},
    },
}
