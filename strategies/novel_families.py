"""Novel families NOT present in the user's V8 dashboard (24k grails tested).

User's V8 already covers VWAP/BB/RSI/Stoch/Williams/CCI/ConnorsRSI/ATR/Keltner/
MACD/ADX/Rubber-band/MA-envelope/Vote/Adaptive/TV_* (common TradingView).

This file implements strategies from academic quant/microstructure literature
that are NOT in V8:

  1. hurst_regime       - Hurst exponent regime switch (Peters 1994)
  2. kalman_residual    - Kalman-filter residual mean reversion
  3. ehlers_mama        - MAMA/FAMA adaptive MA (Ehlers, Hilbert transform)
  4. amihud_contrarian  - Amihud illiquidity spike reversion (Amihud 2002)
  5. vrp_proxy          - Realised-vs-forecast variance risk premium proxy
  6. coint_pair_stat    - ETH-BTC cointegration residual z-score (stat-arb)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.indicators import ema, sma, atr, crossover, crossunder


# ============================================================================
# 1. HURST EXPONENT REGIME SWITCH
#    H > 0.5 => trend-persistent  -> trend-follow
#    H < 0.5 => mean-reverting    -> revert
#    Rolling estimation via rescaled range (R/S) analysis.
# ============================================================================
def hurst_rs(s: pd.Series, lags) -> float:
    """Hurst via R/S (Mandelbrot-Van Ness). `lags` is a list of lag sizes."""
    rs = []
    x = s.values
    mean = x.mean()
    y = np.cumsum(x - mean)
    for k in lags:
        if k >= len(x): continue
        r = np.max(y[:k]) - np.min(y[:k])
        stdev = x[:k].std(ddof=0)
        if stdev == 0: continue
        rs.append((k, r / stdev))
    if len(rs) < 3: return np.nan
    ks, rss = zip(*rs)
    coef = np.polyfit(np.log(ks), np.log(rss), 1)
    return float(coef[0])


def hurst_rolling(close: pd.Series, window: int) -> pd.Series:
    lags = [8, 16, 32, 64]
    returns = close.pct_change().fillna(0.0)
    out = pd.Series(index=close.index, dtype=float)
    for i in range(window, len(returns)):
        out.iloc[i] = hurst_rs(returns.iloc[i - window:i], lags)
    return out


def sig_hurst_regime(df, p):
    c = df["close"]
    h = hurst_rolling(c, p["hurst_win"])
    sma_trend = sma(c, p["trend"])
    # trend mode: H > 0.55 AND price above trend
    trend_entry = (h > p["h_trend"]) & (c > sma_trend) & (c > c.shift(1))
    # revert mode: H < 0.45 AND price far below short MA
    fast = sma(c, p["fast"])
    revert_entry = (h < p["h_revert"]) & (c < fast * (1 - p["revert_pct"] / 100))
    entry = trend_entry | revert_entry
    ex = crossunder(c, fast)
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 2. KALMAN-FILTER RESIDUAL MEAN REVERSION
#    Local-level state-space: x_t = x_{t-1} + w_t, y_t = x_t + v_t.
#    Estimate level; enter when residual (price - level) is > k*std(residual).
# ============================================================================
def kalman_level(close: pd.Series, q: float, r: float) -> pd.Series:
    """1D local-level Kalman (q=process var, r=observation var)."""
    n = len(close)
    x = np.zeros(n)
    P = np.zeros(n)
    x[0] = close.iloc[0]
    P[0] = 1.0
    for t in range(1, n):
        x_prior = x[t - 1]
        P_prior = P[t - 1] + q
        K = P_prior / (P_prior + r)
        x[t] = x_prior + K * (close.iloc[t] - x_prior)
        P[t] = (1 - K) * P_prior
    return pd.Series(x, index=close.index)


def sig_kalman_residual(df, p):
    c = df["close"]
    level = kalman_level(c, q=p["q"], r=p["r"])
    resid = c - level
    z = (resid - resid.rolling(p["zwin"], min_periods=p["zwin"]).mean()) \
        / resid.rolling(p["zwin"], min_periods=p["zwin"]).std(ddof=0)
    trend_ok = c > sma(c, p["trend"])
    entry = (z < -p["zenter"]) & trend_ok
    ex = z > p["zexit"]
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 3. EHLERS MAMA/FAMA ADAPTIVE MOVING AVERAGE
#    Hilbert-transform-derived adaptive MA; speed varies with dominant cycle.
# ============================================================================
def ehlers_mama(close: pd.Series, fast_limit: float = 0.5,
                slow_limit: float = 0.05):
    price = close.values
    n = len(price)
    mama = np.zeros(n)
    fama = np.zeros(n)
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    i1 = np.zeros(n); q1 = np.zeros(n)
    ji = np.zeros(n); jq = np.zeros(n)
    i2 = np.zeros(n); q2 = np.zeros(n)
    re = np.zeros(n); im = np.zeros(n)
    period = np.zeros(n); phase = np.zeros(n)
    for t in range(6, n):
        smooth[t] = (4*price[t] + 3*price[t-1] + 2*price[t-2] + price[t-3]) / 10.0
        detrender[t] = (0.0962*smooth[t] + 0.5769*smooth[t-2] - 0.5769*smooth[t-4] - 0.0962*smooth[t-6]) * (0.075*period[t-1] + 0.54)
        q1[t] = (0.0962*detrender[t] + 0.5769*detrender[t-2] - 0.5769*detrender[t-4] - 0.0962*detrender[t-6]) * (0.075*period[t-1] + 0.54)
        i1[t] = detrender[t-3]
        ji[t] = (0.0962*i1[t] + 0.5769*i1[t-2] - 0.5769*i1[t-4] - 0.0962*i1[t-6]) * (0.075*period[t-1] + 0.54)
        jq[t] = (0.0962*q1[t] + 0.5769*q1[t-2] - 0.5769*q1[t-4] - 0.0962*q1[t-6]) * (0.075*period[t-1] + 0.54)
        i2[t] = i1[t] - jq[t]
        q2[t] = q1[t] + ji[t]
        i2[t] = 0.2*i2[t] + 0.8*i2[t-1]
        q2[t] = 0.2*q2[t] + 0.8*q2[t-1]
        re[t] = i2[t]*i2[t-1] + q2[t]*q2[t-1]
        im[t] = i2[t]*q2[t-1] - q2[t]*i2[t-1]
        re[t] = 0.2*re[t] + 0.8*re[t-1]
        im[t] = 0.2*im[t] + 0.8*im[t-1]
        per = 0 if re[t] == 0 else 360.0/np.degrees(np.arctan(im[t]/re[t])) if re[t] > 0 else period[t-1]
        per = min(max(per, 6), 50)
        per = min(max(per, 0.67*period[t-1]), 1.5*period[t-1]) if period[t-1] > 0 else per
        period[t] = 0.2*per + 0.8*period[t-1]
        phase[t] = 0 if i1[t] == 0 else np.degrees(np.arctan(q1[t]/i1[t]))
        dphase = max(phase[t-1] - phase[t], 1)
        alpha = max(fast_limit / dphase, slow_limit)
        mama[t] = alpha*price[t] + (1-alpha)*mama[t-1]
        fama[t] = 0.5*alpha*mama[t] + (1 - 0.5*alpha)*fama[t-1]
    return pd.Series(mama, index=close.index), pd.Series(fama, index=close.index)


def sig_ehlers_mama(df, p):
    c = df["close"]
    mama, fama = ehlers_mama(c, p["fast_limit"], p["slow_limit"])
    trend_ok = c > sma(c, p["trend"])
    entry = crossover(mama, fama) & trend_ok
    ex = crossunder(mama, fama)
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 4. AMIHUD ILLIQUIDITY CONTRARIAN
#    Illiq = |return| / dollar_volume  (Amihud 2002).
#    Spike in illiquidity followed by price drop -> contrarian long.
# ============================================================================
def amihud(close: pd.Series, volume: pd.Series, n: int) -> pd.Series:
    ret = close.pct_change().abs()
    dv = (close * volume).replace(0, np.nan)
    return (ret / dv).rolling(n, min_periods=n).mean()


def sig_amihud_contrarian(df, p):
    c = df["close"]
    il = amihud(c, df["volume"], p["il_win"])
    il_q = il.rolling(p["q_win"], min_periods=p["q_win"]).quantile(p["q_thr"])
    trend_ok = c > sma(c, p["trend"])
    # entry: illiq above quantile threshold AND price dropped AND in uptrend
    drop = c < c.shift(p["lookback"]) * (1 - p["drop_pct"] / 100)
    entry = (il > il_q) & drop & trend_ok
    ex = c > ema(c, p["exit_ema"])
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 5. VARIANCE RISK PREMIUM PROXY (realised vs GARCH-lite forecast)
#    Proxy for VRP without options data: exp(EWMA log var) - realised var.
#    Negative VRP (realised > forecast) -> vol selling opportunity / revert.
# ============================================================================
def sig_vrp_proxy(df, p):
    c = df["close"]
    ret = c.pct_change()
    var_r = (ret ** 2).rolling(p["rv_win"], min_periods=p["rv_win"]).mean()
    var_f = (ret ** 2).ewm(span=p["ewma_span"], min_periods=p["ewma_span"]).mean()
    vrp = var_r - var_f  # realised minus forecast
    trend_ok = c > sma(c, p["trend"])
    # entry: realised spike beyond forecast + trend intact -> fade the spike
    entry = (vrp > vrp.rolling(p["z_win"], min_periods=p["z_win"]).quantile(0.9)) \
            & trend_ok & (c > c.shift(1))
    ex = vrp < vrp.rolling(p["z_win"], min_periods=p["z_win"]).quantile(0.5)
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 6. ETH-BTC COINTEGRATION RESIDUAL (stat-arb pair, long-only ETH side)
#    beta = cov(log(ETH), log(BTC)) / var(log(BTC)) rolling
#    residual = log(ETH) - beta*log(BTC); revert toward 0 when z-score extreme.
#
#    This strategy only makes sense on ETHUSD with BTC as anchor - it loads BTC
#    data from the same directory at the same TF at evaluation time.
# ============================================================================
_btc_cache = {}

def _get_btc_close(tf: str) -> pd.Series:
    if tf in _btc_cache:
        return _btc_cache[tf]
    from pathlib import Path
    p = Path(f"data/BTCUSD_{tf}.csv")
    if not p.exists():
        _btc_cache[tf] = None
        return None
    df = pd.read_csv(p)
    s = df["close"]
    s.index = df["timestamp"]
    _btc_cache[tf] = s
    return s


def sig_coint_pair(df, p):
    c = df["close"]
    btc = _get_btc_close(df.attrs.get("tf", "1h") if hasattr(df, "attrs") else "1h")
    # fall back: align by timestamp column if btc loaded
    if btc is None or "timestamp" not in df.columns:
        empty = pd.Series(False, index=df.index)
        return empty, empty
    btc_aligned = btc.reindex(df["timestamp"].values)
    btc_aligned.index = df.index
    if btc_aligned.isna().all():
        empty = pd.Series(False, index=df.index)
        return empty, empty
    le = np.log(c.replace(0, np.nan))
    lb = np.log(btc_aligned.replace(0, np.nan))
    n = p["beta_win"]
    cov = le.rolling(n).cov(lb)
    varb = lb.rolling(n).var(ddof=0)
    beta = cov / varb.replace(0, np.nan)
    resid = le - beta * lb
    z = (resid - resid.rolling(p["z_win"]).mean()) / resid.rolling(p["z_win"]).std(ddof=0)
    trend_ok = c > sma(c, p["trend"])
    entry = (z < -p["zenter"]) & trend_ok
    ex = z > p["zexit"]
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 7. CORWIN-SCHULTZ SPREAD ESTIMATOR (liquidity stress)
#    CS 2012 effective spread from H/L only. Spike then normalization = long.
# ============================================================================
def corwin_schultz(high, low, n: int = 2) -> pd.Series:
    beta = (np.log(high / low) ** 2) + (np.log(high.shift(1) / low.shift(1)) ** 2)
    two_day_h = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    two_day_l = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = np.log(two_day_h / two_day_l) ** 2
    denom = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0.0)


def sig_corwin_schultz(df, p):
    c = df["close"]
    sp = corwin_schultz(df["high"], df["low"])
    thr = sp.rolling(p["lookback"], min_periods=p["lookback"]).quantile(p["pct"])
    med = sp.rolling(p["lookback"], min_periods=p["lookback"]).median()
    spike = sp.shift(1) > thr.shift(1)
    normalized = sp < med
    trend_ok = c > sma(c, p["trend"])
    entry = spike & normalized & trend_ok
    ex = sp > thr
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 8. KYLE'S LAMBDA IMPULSE FADE
#    lambda ~ slope of delta_price on signed sqrt(dollar volume). When lambda
#    spikes from forced liquidation and then collapses, price overshoots.
# ============================================================================
def kyle_lambda_bucket(close, volume, bucket: int) -> pd.Series:
    dp = close.diff()
    dv = (close * volume).fillna(0.0)
    sign = np.sign(dp.fillna(0.0))
    sv = sign * np.sqrt(dv.abs())
    # rolling regression slope of dp on sv
    num = (dp * sv).rolling(bucket, min_periods=bucket).sum()
    den = (sv * sv).rolling(bucket, min_periods=bucket).sum().replace(0, np.nan)
    return num / den


def sig_kyle_lambda(df, p):
    c = df["close"]
    lam = kyle_lambda_bucket(c, df["volume"], p["bucket"])
    thr = lam.abs().rolling(p["history"], min_periods=p["history"]).quantile(p["pct"])
    spike_dn = (lam < -thr) & (c < c.shift(p["bucket"]))  # down spike
    reversion = c > c.shift(1)
    trend_ok = c > sma(c, p["trend"])
    entry = spike_dn.shift(1).fillna(False) & reversion & trend_ok
    ex = c > ema(c, p["exit_ema"])
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 9. NAKED POC REVISIT (volume-profile magnet)
#    Previous session's POC (max-volume price level) unrevisited -> magnet.
# ============================================================================
def rolling_poc(close, volume, lookback: int, buckets: int) -> pd.Series:
    out = pd.Series(index=close.index, dtype=float)
    for i in range(lookback, len(close)):
        c_win = close.iloc[i - lookback:i]
        v_win = volume.iloc[i - lookback:i]
        lo, hi = c_win.min(), c_win.max()
        if hi <= lo:
            continue
        edges = np.linspace(lo, hi, buckets + 1)
        idx = np.clip(np.searchsorted(edges, c_win.values) - 1, 0, buckets - 1)
        agg = np.bincount(idx, weights=v_win.values, minlength=buckets)
        mid = (edges[:-1] + edges[1:]) / 2
        out.iloc[i] = mid[int(np.argmax(agg))]
    return out


def sig_naked_poc(df, p):
    c = df["close"]
    poc = rolling_poc(c, df["volume"], p["lookback"], p["buckets"])
    # "naked" if not revisited in last p["unvisited"] bars
    dist = (c - poc).abs() / c
    was_far = (dist.rolling(p["unvisited"], min_periods=p["unvisited"]).min() > p["min_dist"])
    # moving toward POC from below
    below = c < poc
    approaching = c > c.shift(1)
    trend_ok = c > sma(c, p["trend"])
    entry = was_far & below & approaching & trend_ok
    ex = c >= poc
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 10. BULK VOLUME CLASSIFICATION OFI (Easley-Lopez de Prado-O'Hara)
#    Sign volume via normal CDF of (close-open)/range_stdev.
# ============================================================================
try:
    from scipy.stats import norm as _norm
    def _phi(x):
        return _norm.cdf(x)
except Exception:
    def _phi(x):
        # high-accuracy CDF approximation
        return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def sig_bvc_ofi(df, p):
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    ret = (c - o)
    rng_std = (h - l).rolling(p["rng_win"], min_periods=p["rng_win"]).std(ddof=0)
    z = ret / rng_std.replace(0, np.nan)
    phi = pd.Series(_phi(z.fillna(0.0).values), index=c.index)
    sign = 2.0 * phi - 1.0
    ofi = (sign * df["volume"]).rolling(p["ofi_win"], min_periods=p["ofi_win"]).sum()
    pct_hi = ofi.rolling(p["pct_win"], min_periods=p["pct_win"]).quantile(p["pct"])
    pct_lo = ofi.rolling(p["pct_win"], min_periods=p["pct_win"]).quantile(1 - p["pct"])
    trend_ok = c > sma(c, p["trend"])
    # Fade long: OFI at low extreme (heavy selling exhaustion) + reversion bar
    entry = (ofi < pct_lo) & (c > c.shift(1)) & trend_ok
    ex = ofi > 0
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# 11. HAWKES VOLUME BURST FADE
#    Self-exciting intensity proxy; exhaust -> fade.
# ============================================================================
def hawkes_intensity(volume: pd.Series, alpha: float, beta: float) -> pd.Series:
    lam = np.zeros(len(volume))
    v = volume.values
    for t in range(1, len(v)):
        lam[t] = alpha * v[t] + np.exp(-beta) * lam[t - 1]
    return pd.Series(lam, index=volume.index)


def sig_hawkes_burst(df, p):
    c = df["close"]
    up_vol = df["volume"].where(c > c.shift(1), 0.0)
    dn_vol = df["volume"].where(c < c.shift(1), 0.0)
    lam_up = hawkes_intensity(up_vol, p["alpha"], p["beta"])
    lam_dn = hawkes_intensity(dn_vol, p["alpha"], p["beta"])
    thr_up = lam_up.rolling(p["win"], min_periods=p["win"]).quantile(p["pct"])
    # Exhaustion: up-cluster fading -> fade LONG here; actual buy occurs on
    # down-cluster fading in uptrend (buy the dip after panic sell dies).
    exhausted_dn = (lam_dn > thr_up) & (lam_dn < lam_dn.shift(1))
    recovery = c > c.shift(1)
    trend_ok = c > sma(c, p["trend"])
    entry = exhausted_dn & recovery & trend_ok
    ex = lam_up > thr_up
    return entry.fillna(False), ex.fillna(False)


# ============================================================================
# PARAMETER GRIDS
# ============================================================================
SPACES_NOVEL = {
    "hurst_regime": {
        "sig": sig_hurst_regime,
        "params": {
            "hurst_win":  [100, 150, 200, 300],
            "h_trend":    [0.53, 0.55, 0.57, 0.60],
            "h_revert":   [0.40, 0.43, 0.45, 0.47],
            "fast":       [10, 20, 34, 50],
            "trend":      [100, 150, 200, 300],
            "revert_pct": [1.5, 2.5, 4.0, 6.0],
        },
        "exit": {
            "sl_atr":    [1.5, 2.0, 2.5, 3.0],
            "tp_atr":    [None, 2.0, 3.0, 5.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None, 24, 48, 96],
        },
    },
    "kalman_residual": {
        "sig": sig_kalman_residual,
        "params": {
            "q":      [0.001, 0.005, 0.01, 0.05],
            "r":      [0.1, 0.5, 1.0, 2.0],
            "zwin":   [20, 50, 100],
            "zenter": [1.5, 2.0, 2.5, 3.0],
            "zexit":  [0.0, 0.5, 1.0],
            "trend":  [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 2.5, 3.0],
            "tp_atr":    [None, 2.0, 3.0, 4.0],
            "trail_atr": [None, 2.0, 3.0],
            "timeout":   [None, 12, 24, 48, 72],
        },
    },
    "ehlers_mama": {
        "sig": sig_ehlers_mama,
        "params": {
            "fast_limit": [0.3, 0.5, 0.7],
            "slow_limit": [0.01, 0.05, 0.1],
            "trend":      [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [None, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 3.0, 5.0, 8.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None, 48, 96, 168],
        },
    },
    "amihud_contrarian": {
        "sig": sig_amihud_contrarian,
        "params": {
            "il_win":   [5, 10, 14, 20],
            "q_win":    [50, 100, 200],
            "q_thr":    [0.80, 0.90, 0.95],
            "lookback": [2, 3, 5, 8],
            "drop_pct": [2.0, 3.0, 5.0, 8.0],
            "exit_ema": [8, 13, 21],
            "trend":    [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.5, 2.0, 2.5, 3.0],
            "tp_atr":    [None, 2.0, 3.0, 5.0],
            "trail_atr": [None, 2.0, 3.0],
            "timeout":   [None, 24, 48, 72],
        },
    },
    "vrp_proxy": {
        "sig": sig_vrp_proxy,
        "params": {
            "rv_win":    [10, 20, 30],
            "ewma_span": [50, 100, 200],
            "z_win":     [100, 200, 300],
            "trend":     [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.5, 2.0, 2.5, 3.0],
            "tp_atr":    [None, 2.0, 3.0, 4.0],
            "trail_atr": [None, 2.0, 3.0],
            "timeout":   [None, 12, 24, 48],
        },
    },
    "coint_pair": {
        "sig": sig_coint_pair,
        "params": {
            "beta_win": [100, 200, 300, 500],
            "z_win":    [50, 100, 200],
            "zenter":   [1.5, 2.0, 2.5, 3.0],
            "zexit":    [0.0, 0.5, 1.0],
            "trend":    [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.5, 2.0, 2.5, 3.0],
            "tp_atr":    [None, 2.0, 3.0, 4.0],
            "trail_atr": [None, 2.0, 3.0],
            "timeout":   [None, 24, 48, 96, 168],
        },
    },
    "corwin_schultz": {
        "sig": sig_corwin_schultz,
        "params": {
            "lookback": [100, 200, 300],
            "pct":      [0.90, 0.95, 0.98],
            "trend":    [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 2.0, 3.0, 5.0],
            "trail_atr": [None, 2.0, 3.0],
            "timeout":   [None, 8, 12, 24, 48],
        },
    },
    "kyle_lambda": {
        "sig": sig_kyle_lambda,
        "params": {
            "bucket":   [6, 10, 14, 20],
            "history":  [100, 200, 400],
            "pct":      [0.90, 0.95, 0.99],
            "exit_ema": [13, 21, 34],
            "trend":    [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 2.0, 3.0, 4.0],
            "trail_atr": [None, 2.0, 3.0],
            "timeout":   [None, 8, 12, 24, 48],
        },
    },
    "naked_poc": {
        "sig": sig_naked_poc,
        "params": {
            "lookback":  [48, 96, 168, 336],   # 2d / 4d / 7d / 14d on 1h
            "buckets":   [15, 20, 30],
            "unvisited": [5, 10, 20],
            "min_dist":  [0.005, 0.01, 0.02, 0.04],
            "trend":     [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 1.5, 2.0, 3.0],
            "trail_atr": [None, 1.5, 2.0],
            "timeout":   [None, 6, 10, 20, 48],
        },
    },
    "bvc_ofi": {
        "sig": sig_bvc_ofi,
        "params": {
            "rng_win": [50, 100, 200],
            "ofi_win": [5, 10, 20, 30],
            "pct_win": [100, 200, 400],
            "pct":     [0.90, 0.95, 0.98],
            "trend":   [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 2.0, 3.0, 4.0],
            "trail_atr": [None, 2.0, 3.0],
            "timeout":   [None, 6, 12, 24, 48],
        },
    },
    "hawkes_burst": {
        "sig": sig_hawkes_burst,
        "params": {
            "alpha": [0.5, 1.0, 1.5, 2.0],
            "beta":  [0.05, 0.1, 0.2, 0.3],
            "win":   [50, 100, 200],
            "pct":   [0.90, 0.95, 0.99],
            "trend": [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 1.5, 2.0, 3.0],
            "trail_atr": [None, 1.5, 2.0],
            "timeout":   [None, 4, 8, 12, 24],
        },
    },
}
