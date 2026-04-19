"""
Batch 3607 - Crypto-specific microstructure & regime (novel families)
Sources:
 - Funding-rate fade proxy: basis return acceleration (perp funding surrogate)
 - Session vol regime switch: UTC-hour regime + adaptive trend/revert gating
 - Order-flow momentum proxy: (close - low)/(high - low) * sign(vol - sma_vol)
3 strategies, pickle-safe, no lambdas, no look-ahead.
"""
import numpy as np
import pandas as pd


# ----- FUNDING_RATE_FADE (proxy) -----
# Crypto perpetual funding is positive when longs overpay; that manifests as
# persistent positive log-return acceleration relative to vol. We proxy funding
# by normalised return-of-return (jerk) smoothed. Extreme values fade (long
# after extreme-negative, short after extreme-positive) — the classic funding
# cascade fade.
def gen_TV_Funding_Rate_Fade(df, jerk_len=24, z_len=96, z_thresh=2.0, smooth=4, **kw):
    c = df['close'].astype(float)
    r = np.log(c.replace(0.0, np.nan)).diff()
    # jerk = mean of recent returns minus mean of prior returns (momentum proxy of funding pressure)
    n1 = int(jerk_len)
    mom = r.rolling(n1, min_periods=n1).mean()
    jerk = (mom - mom.shift(n1)).ewm(span=int(smooth), adjust=False,
                                      min_periods=int(smooth)).mean()
    w = int(z_len)
    mu = jerk.rolling(w, min_periods=w).mean()
    sd = jerk.rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
    z = ((jerk - mu) / sd).shift(1)
    sig = pd.Series(0, index=df.index, dtype=int)
    # fade: when z very positive, go short; z very negative, go long
    sig[(z >= float(z_thresh)).fillna(False)] = -1
    sig[(z <= -float(z_thresh)).fillna(False)] = 1
    return sig


def space_TV_Funding_Rate_Fade():
    return {'jerk_len': ('int', 8, 96),
            'z_len': ('int', 40, 400),
            'z_thresh': ('float', 1.2, 3.5),
            'smooth': ('int', 1, 12)}


# ----- SESSION_VOL_REGIME_SWITCH -----
# Classifies volatility regime (low / med / high) via rolling quantile of ATR%
# then gates entries:
#   LOW vol  -> mean-revert to short-ema
#   HIGH vol -> trend-follow ema cross
#   MED vol  -> dormant (no signal)
# Captures crypto's 24h vol clustering w/o explicit session-hour dep (robust
# across symbols that publish in different TZs).
def gen_TV_Session_Vol_Regime(df, atr_len=14, q_len=200, q_lo=0.33, q_hi=0.66,
                               ema_fast=20, ema_slow=50, **kw):
    h = df['high'].astype(float); l = df['low'].astype(float); c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(atr_len), adjust=False,
                  min_periods=int(atr_len)).mean()
    atrp = (atr / c.replace(0.0, np.nan))
    w = int(q_len)
    lo = atrp.rolling(w, min_periods=w).quantile(float(q_lo))
    hi = atrp.rolling(w, min_periods=w).quantile(float(q_hi))
    is_low = (atrp < lo).shift(1)
    is_high = (atrp > hi).shift(1)
    ef = c.ewm(span=int(ema_fast), adjust=False, min_periods=int(ema_fast)).mean().shift(1)
    es = c.ewm(span=int(ema_slow), adjust=False, min_periods=int(ema_slow)).mean().shift(1)
    cs = c.shift(1)
    # Trend signal: ef > es -> long; ef < es -> short (only when HIGH vol)
    trend_long = is_high & (ef > es)
    trend_short = is_high & (ef < es)
    # Revert signal: price below ef by >0 -> long; price above ef -> short (LOW vol)
    revert_long = is_low & (cs < ef)
    revert_short = is_low & (cs > ef)
    long_cond = trend_long | revert_long
    short_cond = trend_short | revert_short
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cond.fillna(False)] = 1
    sig[short_cond.fillna(False)] = -1
    return sig


def space_TV_Session_Vol_Regime():
    return {'atr_len': ('int', 7, 50),
            'q_len': ('int', 80, 500),
            'q_lo': ('float', 0.15, 0.45),
            'q_hi': ('float', 0.55, 0.85),
            'ema_fast': ('int', 8, 50),
            'ema_slow': ('int', 30, 200)}


# ----- ORDER_FLOW_MOMENTUM_PROXY -----
# OFM proxy: (close - low)/(high - low) gives "where did price close within the
# bar's range" — near 1.0 = strong buying, near 0.0 = strong selling. Multiply
# by sign(vol - sma_vol) to weight by activity. Signal: exhaustion fade —
# after N bars of extreme OFM, fade the direction.
def gen_TV_OrderFlow_Momentum_Proxy(df, ofm_thresh=0.8, run_len=3, vol_sma=20, **kw):
    h = df['high'].astype(float); l = df['low'].astype(float); c = df['close'].astype(float)
    v = df['volume'].astype(float)
    rng = (h - l).replace(0.0, np.nan)
    loc = ((c - l) / rng).clip(0.0, 1.0)
    vsma = v.rolling(int(vol_sma), min_periods=int(vol_sma)).mean().replace(0.0, np.nan)
    vsign = np.sign((v - vsma).fillna(0.0))
    ofm = (2.0 * loc - 1.0) * vsign  # in [-1, +1]
    # Run-length of same-sign extremes
    up_extreme = (ofm >= float(ofm_thresh)).astype(int)
    dn_extreme = (ofm <= -float(ofm_thresh)).astype(int)
    rl = int(run_len)
    up_run = up_extreme.rolling(rl, min_periods=rl).sum() >= rl
    dn_run = dn_extreme.rolling(rl, min_periods=rl).sum() >= rl
    # Fade after run completes
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[up_run.shift(1).fillna(False)] = -1  # fade up
    sig[dn_run.shift(1).fillna(False)] = 1   # fade down
    return sig


def space_TV_OrderFlow_Momentum_Proxy():
    return {'ofm_thresh': ('float', 0.4, 0.95),
            'run_len': ('int', 2, 8),
            'vol_sma': ('int', 5, 100)}


STRATEGY_EXPORT = {
    'TV_Funding_Rate_Fade': {
        'gen': gen_TV_Funding_Rate_Fade,
        'space': space_TV_Funding_Rate_Fade,
        'source': 'Crypto perp funding-cascade fade proxy (batch 3607)'},
    'TV_Session_Vol_Regime': {
        'gen': gen_TV_Session_Vol_Regime,
        'space': space_TV_Session_Vol_Regime,
        'source': 'ATR-quantile regime-adaptive trend/revert gating (batch 3607)'},
    'TV_OrderFlow_Momentum_Proxy': {
        'gen': gen_TV_OrderFlow_Momentum_Proxy,
        'space': space_TV_OrderFlow_Momentum_Proxy,
        'source': 'Close-location + signed-volume exhaustion fade (batch 3607)'},
}
