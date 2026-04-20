"""
Batch 3608 - Advanced crypto regime + microstructure (5 novel families)

Ideas:
 - TV_BTC_Beta_Residual: rolling OLS residual vs BTC-proxy (uses ETH as market proxy if avail else rolling-mean of log-returns)
 - TV_CumVol_Imbalance: cumulative signed-volume divergence from price (flow diverging)
 - TV_ATR_Regime_Reversal: ATR quantile + trend reversal - fade high-vol spikes
 - TV_RealizedVol_Anchor: entry only when realized vol < N-percentile (quiet market entries)
 - TV_Momentum_Acceleration: jerk (3rd deriv) + sign + threshold

All pickle-safe, no lambdas, no look-ahead (shift(1) on signals).
"""
import numpy as np
import pandas as pd


# ----- 1) BETA_RESIDUAL (single-asset approximation using z-score of detrended returns) -----
def gen_TV_BTC_Beta_Residual(df, win=60, resid_z_len=40, z_thresh=1.8, **kw):
    c = df['close'].astype(float)
    r = np.log(c.replace(0.0, np.nan)).diff()
    # Rolling mean acts as market-proxy; residual = r - mean(r over window)
    mkt = r.rolling(int(win), min_periods=int(win)).mean()
    resid = r - mkt
    w = int(resid_z_len)
    mu = resid.rolling(w, min_periods=w).mean()
    sd = resid.rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
    z = ((resid - mu) / sd).shift(1)
    sig = pd.Series(0, index=df.index, dtype=int)
    # Fade residual extremes
    sig[(z >= float(z_thresh)).fillna(False)] = -1
    sig[(z <= -float(z_thresh)).fillna(False)] = 1
    return sig


def space_TV_BTC_Beta_Residual():
    return {'win': ('int', 20, 240),
            'resid_z_len': ('int', 20, 200),
            'z_thresh': ('float', 1.2, 3.0)}


# ----- 2) CUMULATIVE VOLUME IMBALANCE -----
def gen_TV_CumVol_Imbalance(df, lookback=48, imb_z_len=96, z_thresh=2.0, **kw):
    c = df['close'].astype(float); v = df['volume'].astype(float)
    r = np.log(c.replace(0.0, np.nan)).diff()
    signed_v = v * np.sign(r.fillna(0.0))
    cum = signed_v.rolling(int(lookback), min_periods=int(lookback)).sum()
    # Compare cum-flow to price change over same window
    price_r = c.pct_change(int(lookback))
    # Normalize each
    w = int(imb_z_len)
    cum_z = (cum - cum.rolling(w, min_periods=w).mean()) / cum.rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
    pr_z = (price_r - price_r.rolling(w, min_periods=w).mean()) / price_r.rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
    divergence = (pr_z - cum_z).shift(1)
    sig = pd.Series(0, index=df.index, dtype=int)
    # Price up but flow lagging -> fade short; price down but flow up -> fade long
    sig[(divergence >= float(z_thresh)).fillna(False)] = -1
    sig[(divergence <= -float(z_thresh)).fillna(False)] = 1
    return sig


def space_TV_CumVol_Imbalance():
    return {'lookback': ('int', 12, 200),
            'imb_z_len': ('int', 40, 400),
            'z_thresh': ('float', 1.2, 3.0)}


# ----- 3) ATR REGIME REVERSAL -----
def gen_TV_ATR_Regime_Reversal(df, atr_len=14, q_len=200, q_hi=0.85, rev_len=3, **kw):
    h = df['high'].astype(float); l = df['low'].astype(float); c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(atr_len), adjust=False, min_periods=int(atr_len)).mean()
    atrp = atr / c.replace(0.0, np.nan)
    w = int(q_len)
    hi_q = atrp.rolling(w, min_periods=w).quantile(float(q_hi))
    is_spike = (atrp > hi_q).shift(1)
    # Direction of spike: use N-bar return
    rlen = int(rev_len)
    n_ret = c.pct_change(rlen).shift(1)
    sig = pd.Series(0, index=df.index, dtype=int)
    # After high-vol spike with +return, fade -> short; with -return, fade -> long
    up_fade = is_spike & (n_ret > 0)
    dn_fade = is_spike & (n_ret < 0)
    sig[up_fade.fillna(False)] = -1
    sig[dn_fade.fillna(False)] = 1
    return sig


def space_TV_ATR_Regime_Reversal():
    return {'atr_len': ('int', 7, 50),
            'q_len': ('int', 80, 500),
            'q_hi': ('float', 0.7, 0.95),
            'rev_len': ('int', 2, 10)}


# ----- 4) REALIZED VOL ANCHOR (quiet-market entries) -----
def gen_TV_RealizedVol_Anchor(df, rv_len=24, q_len=200, q_lo=0.20,
                               ema_fast=20, ema_slow=50, **kw):
    c = df['close'].astype(float)
    r = np.log(c.replace(0.0, np.nan)).diff()
    rv = r.rolling(int(rv_len), min_periods=int(rv_len)).std(ddof=0)
    w = int(q_len)
    lo_q = rv.rolling(w, min_periods=w).quantile(float(q_lo))
    is_quiet = (rv < lo_q).shift(1)
    ef = c.ewm(span=int(ema_fast), adjust=False, min_periods=int(ema_fast)).mean().shift(1)
    es = c.ewm(span=int(ema_slow), adjust=False, min_periods=int(ema_slow)).mean().shift(1)
    cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
    cross_dn = (ef < es) & (ef.shift(1) >= es.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    # Trade crossovers only in quiet regime
    sig[(is_quiet & cross_up).fillna(False)] = 1
    sig[(is_quiet & cross_dn).fillna(False)] = -1
    return sig


def space_TV_RealizedVol_Anchor():
    return {'rv_len': ('int', 10, 100),
            'q_len': ('int', 80, 500),
            'q_lo': ('float', 0.10, 0.40),
            'ema_fast': ('int', 5, 50),
            'ema_slow': ('int', 20, 200)}


# ----- 5) MOMENTUM ACCELERATION (jerk) -----
def gen_TV_Momentum_Acceleration(df, mom_len=12, jerk_smooth=4, z_len=80, z_thresh=1.7, **kw):
    c = df['close'].astype(float)
    r = np.log(c.replace(0.0, np.nan)).diff()
    mom = r.rolling(int(mom_len), min_periods=int(mom_len)).sum()
    jerk = mom.diff().ewm(span=int(jerk_smooth), adjust=False,
                           min_periods=int(jerk_smooth)).mean()
    w = int(z_len)
    mu = jerk.rolling(w, min_periods=w).mean()
    sd = jerk.rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
    z = ((jerk - mu) / sd).shift(1)
    sig = pd.Series(0, index=df.index, dtype=int)
    # Accel+ -> long (ride accel), Accel- -> short
    sig[(z >= float(z_thresh)).fillna(False)] = 1
    sig[(z <= -float(z_thresh)).fillna(False)] = -1
    return sig


def space_TV_Momentum_Acceleration():
    return {'mom_len': ('int', 4, 60),
            'jerk_smooth': ('int', 1, 10),
            'z_len': ('int', 30, 300),
            'z_thresh': ('float', 1.0, 3.0)}


STRATEGY_EXPORT = {
    'TV_BTC_Beta_Residual': {
        'gen': gen_TV_BTC_Beta_Residual, 'space': space_TV_BTC_Beta_Residual,
        'source': 'Rolling-mean residual fade (batch 3608)'},
    'TV_CumVol_Imbalance': {
        'gen': gen_TV_CumVol_Imbalance, 'space': space_TV_CumVol_Imbalance,
        'source': 'Cumulative signed-volume vs price divergence (batch 3608)'},
    'TV_ATR_Regime_Reversal': {
        'gen': gen_TV_ATR_Regime_Reversal, 'space': space_TV_ATR_Regime_Reversal,
        'source': 'ATR-quantile spike fade (batch 3608)'},
    'TV_RealizedVol_Anchor': {
        'gen': gen_TV_RealizedVol_Anchor, 'space': space_TV_RealizedVol_Anchor,
        'source': 'Low-vol quiet-market EMA crossover (batch 3608)'},
    'TV_Momentum_Acceleration': {
        'gen': gen_TV_Momentum_Acceleration, 'space': space_TV_Momentum_Acceleration,
        'source': 'Jerk (return-of-return) z-score ride (batch 3608)'},
}
