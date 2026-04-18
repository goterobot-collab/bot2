#!/usr/bin/env python3
"""
TV2 BATCH MICRO15 — 5 microstructure strategies (5m/15m optimized).
Domain: Bayesian Changepoint Detection, Order Book Resilience,
        Almgren-Chriss Execution Theory, Hawkes Branching Ratio,
        Topological Data Analysis (Persistent Homology proxy).
(academic, distinct from MICRO1-14 + batches 3237-3318)

Strategies:
  1. BOCD_Regime_Break      — Bayesian Online Changepoint Detection
                              (Adams & MacKay 2007): posterior over run-length.
                              High hazard rate (short expected run) → changepoint
                              → momentum in break direction.
  2. Resilience_Rev         — Foucault, Pagano & Röell (2013) order book
                              resilience: after volume shock, spread spikes then
                              compresses as MM refills. Rapid spread compression
                              post-spike → mean-reversion entry.
  3. VWAP_Execution_Dev     — Almgren & Chriss (2000) optimal execution: VWAP
                              deviations signal informed trading pressure. Price
                              far from VWAP "execution schedule" → reversion.
  4. BranchingRatio_Vol     — Bacry, Mastromatteo & Muzy (2015) Hawkes branching
                              ratio m = fraction of self-excited events. m→1
                              (near-critical, vol about to cascade) → breakout.
                              m→0 (subcritical, calm) → mean-reversion.
  5. TDA_Shape_Rev          — Carlsson (2009) Topological Data Analysis:
                              0th Betti number proxy (connected components of
                              level set) measures shape complexity of return
                              distribution. High complexity = instability →
                              reversion; low = trend.

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Adams, R.P. & MacKay, D.J.C. (2007) — "Bayesian Online Changepoint
    Detection"; arXiv:0710.3742.
  - Foucault, T., Pagano, M. & Röell, A. (2013) — "Market Liquidity: Theory,
    Evidence, and Policy"; Oxford University Press.
  - Almgren, R. & Chriss, N. (2000) — "Optimal Execution of Portfolio
    Transactions"; Journal of Risk 3(2): 5-39.
  - Bacry, E., Mastromatteo, I. & Muzy, J.-F. (2015) — "Hawkes Processes in
    Finance"; Market Microstructure and Liquidity 1(1): 1550005.
  - Carlsson, G. (2009) — "Topology and Data"; Bulletin of the American
    Mathematical Society 46(2): 255-308.
  - Flajolet, P. & Martin, G.N. (1985) — "Probabilistic Counting Algorithms
    for Data Base Applications"; Journal of Computer and System Sciences
    31(2): 182-209.  (connected-component counting basis)
"""

import pandas as pd
import numpy as np


# ─── HELPERS ────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _sma(s, p):
    return s.rolling(p).mean()

def _atr(h, l, c, p=14):
    prev_c = c.shift(1)
    tr = pd.concat([h - l,
                    (h - prev_c).abs(),
                    (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / p, adjust=False).mean()

def _zscore(s, p):
    m  = s.rolling(p).mean()
    sd = s.rolling(p).std(ddof=0) + 1e-10
    return (s - m) / sd

def _bb(c, p=20, ndev=2.0):
    mid = c.rolling(p).mean()
    sd  = c.rolling(p).std(ddof=0)
    return mid - ndev * sd, mid, mid + ndev * sd

def _apply_exit_bar(entry_long, entry_short, exit_bar):
    """
    3-state machine: enters on signal, exits after exit_bar bars.
    Flat(0) -> Long(1) on entry_long; Flat(0) -> Short(-1) on entry_short.
    New entries override opposite position immediately.
    """
    sig = pd.Series(0, index=entry_long.index, dtype=int)
    el = entry_long.values
    es = entry_short.values
    n  = len(sig)
    state     = 0
    bars_held = 0
    for i in range(n):
        if state != 0:
            bars_held += 1
            if bars_held >= exit_bar:
                state     = 0
                bars_held = 0
        if el[i]:
            state     = 1
            bars_held = 0
        elif es[i]:
            state     = -1
            bars_held = 0
        sig.iloc[i] = state
    return sig


# ===========================================================================
# 1. BOCD_Regime_Break — Bayesian Online Changepoint Detection momentum
# ===========================================================================
# Adams & MacKay (2007) define a run-length r_t = number of time steps since
# the last changepoint. Under the Bayesian model, P(r_t | x_{1:t}) is updated
# recursively each step via the hazard function h and predictive likelihood.
#
# Tractable approximation (no full belief state required):
#   Predictive surprise = |ret[t] - µ_window[t-1]| / σ_window[t-1]
#   Cumulative surprise (Σ over window) ↑ = changepoint pressure
#   When surprise_z crosses threshold (mean of short vs long window diverges
#   strongly) → high posterior probability of changepoint.
#
# Signal:
#   At changepoint:
#     direction = sign(short_mean - long_mean) at detection
#     Entry LONG if break is upward + trend context
#     Entry SHORT if break is downward + trend context
#
# Trend guard: only trade in direction of momentum at detection point.

def gen_BOCD_Regime_Break(df, short_win=10, long_win=40, surp_z_window=30,
                           surp_z_thresh=2.0, trend_ema=50, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change()

    # Rolling predictive mean and std over short window
    mu_short = ret.rolling(short_win).mean()
    sd_short = ret.rolling(short_win).std(ddof=0) + 1e-10

    # Predictive surprise per bar: standardized deviation from short-window mean
    surprise = (ret - mu_short.shift(1)) / sd_short.shift(1)
    surprise = surprise.abs()

    # Cumulative surprise over long_win bars
    cum_surprise = surprise.rolling(long_win).mean()

    # Z-score of cumulative surprise (changepoint intensity)
    surp_z = _zscore(cum_surprise, surp_z_window)

    # Direction: short-window mean vs long-window mean
    mu_long   = ret.rolling(long_win).mean()
    break_up   = mu_short > mu_long
    break_down = mu_short < mu_long

    # Trend guard
    trend     = _ema(c, trend_ema)
    up_trend  = c > trend

    # Changepoint detected → trade in break direction
    entry_long  = (surp_z > surp_z_thresh) & break_up   & up_trend
    entry_short = (surp_z > surp_z_thresh) & break_down & ~up_trend

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_BOCD_Regime_Break(trial):
    return {
        'short_win':     trial.suggest_int('short_win',      6, 20),
        'long_win':      trial.suggest_int('long_win',       30, 80),
        'surp_z_window': trial.suggest_int('surp_z_window',  20, 50),
        'surp_z_thresh': trial.suggest_float('surp_z_thresh', 1.2, 3.0),
        'trend_ema':     trial.suggest_int('trend_ema',       30, 80),
        'exit_bar':      trial.suggest_int('exit_bar',         3, 12),
    }


# ===========================================================================
# 2. Resilience_Rev — Foucault-Pagano-Röell order book resilience reversal
# ===========================================================================
# Foucault, Pagano & Röell (2013) define order book resilience as the speed
# at which the bid-ask spread recovers after a large market order depletes
# one side of the book. In a resilient book, MM limit orders refill quickly
# and price reverts to pre-shock levels.
#
# High-frequency spread is not available from OHLCV. Tractable proxy:
#   spread_proxy[t] = (high[t] - low[t]) / close[t]   ← intra-bar range
#   volume_shock[t] = volume[t] / rolling_avg(volume)  ← volume ratio
#
# Shock episode: volume_shock > shock_thresh (large order)
# Resilience measure: spread_proxy falls back toward normal within rebound bars
#
# Signal (mean-reversion):
#   If volume_shock > thresh AND spread_proxy was elevated AND is now
#   compressing toward its rolling mean → market is refilling → reversion
#   Direction: if close > VWAP-proxy → SHORT (sold into) ; else → LONG
#
# Additional: trend guard ensures direction consistency.

def gen_Resilience_Rev(df, vol_shock_thresh=1.8, spread_window=30,
                        spread_z_thresh=1.2, rebound_bars=4,
                        trend_ema=50, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    # Spread proxy: intra-bar range as fraction of close
    spread = (h - l) / (c + 1e-10)

    # Volume shock ratio
    vol_avg   = v.rolling(spread_window).mean()
    vol_shock = v / (vol_avg + 1e-10)

    # Spread z-score
    spread_z = _zscore(spread, spread_window)

    # Was there a spread spike in the last rebound_bars?
    spread_was_high = spread_z.rolling(rebound_bars).max() > spread_z_thresh

    # Spread is now compressing (current z below the recent spike)
    spread_compressing = spread_z < spread_z.rolling(rebound_bars).max() - 0.5

    # Volume shock happened recently
    vol_shocked = vol_shock.rolling(rebound_bars).max() > vol_shock_thresh

    # Resilience event: large order + spread spike + now compressing
    resilience_event = spread_was_high & spread_compressing & vol_shocked

    # Direction: VWAP proxy for session direction
    vwap_proxy = (c * v).rolling(spread_window).sum() / (v.rolling(spread_window).sum() + 1e-10)
    above_vwap = c > vwap_proxy

    # Trend guard
    trend = _ema(c, trend_ema)

    # After sell shock (price below VWAP, spread spiked then compressed) → LONG
    # After buy shock (price above VWAP, spread spiked then compressed) → SHORT
    # Trend guard: use a medium EMA to avoid fading a strong dominant trend
    medium_trend = _ema(c, trend_ema // 2)
    entry_long  = resilience_event & ~above_vwap & (c > medium_trend * 0.97)
    entry_short = resilience_event &  above_vwap & (c < medium_trend * 1.03)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Resilience_Rev(trial):
    return {
        'vol_shock_thresh': trial.suggest_float('vol_shock_thresh', 1.3, 3.0),
        'spread_window':    trial.suggest_int('spread_window',       15, 50),
        'spread_z_thresh':  trial.suggest_float('spread_z_thresh',   0.8, 2.5),
        'rebound_bars':     trial.suggest_int('rebound_bars',          2,  8),
        'trend_ema':        trial.suggest_int('trend_ema',            30, 80),
        'exit_bar':         trial.suggest_int('exit_bar',              3, 10),
    }


# ===========================================================================
# 3. VWAP_Execution_Dev — Almgren-Chriss execution schedule deviation
# ===========================================================================
# Almgren & Chriss (2000) show that the optimal execution strategy for a
# large order produces a specific price trajectory — roughly, a linear
# decay from arrival price toward VWAP. Deviations from this "schedule"
# signal either informed trading (large player accelerating) or market
# impact reversal (overshoot corrects).
#
# Tractable proxy:
#   session_vwap[t]   = Σ(price×volume, 0..t) / Σ(volume, 0..t) over window
#   linear_schedule   = EMA of price with the same window (smooth drift)
#   execution_dev     = (price - session_vwap) / ATR  ← normalized deviation
#
# Signal:
#   Large positive dev  → price ran ahead of VWAP schedule (impact overshoot)
#     → SHORT reversion back toward VWAP
#   Large negative dev  → price lagged VWAP schedule (demand absorbed)
#     → LONG reversion toward VWAP
#
# Volume acceleration guard: if volume is also accelerating, deviation may
# be informed (not MM overshoot) → skip reversion, wait.

def gen_VWAP_Execution_Dev(df, vwap_window=40, dev_z_window=30,
                            dev_z_thresh=1.8, atr_p=14,
                            vol_accel_thresh=1.5, trend_ema=60, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    # Session VWAP over rolling window
    cum_pv  = (c * v).rolling(vwap_window).sum()
    cum_vol = v.rolling(vwap_window).sum() + 1e-10
    vwap    = cum_pv / cum_vol

    # ATR normalization
    atr = _atr(h, l, c, atr_p)

    # Execution schedule deviation (normalized)
    exec_dev = (c - vwap) / (atr + 1e-10)

    # Z-score of deviation
    dev_z = _zscore(exec_dev, dev_z_window)

    # Volume acceleration: recent vol vs slightly longer window
    vol_short = v.rolling(vwap_window // 2).mean()
    vol_long  = v.rolling(vwap_window).mean() + 1e-10
    vol_accel = vol_short / vol_long

    # Informed trading: volume is accelerating → skip reversion
    vol_informed = vol_accel > vol_accel_thresh

    # Trend guard
    trend = _ema(c, trend_ema)

    # Price ran above VWAP schedule (impact overshoot) → SHORT reversion
    entry_short = (dev_z >  dev_z_thresh) & ~vol_informed & (c < trend)
    # Price lagged VWAP schedule → LONG reversion
    entry_long  = (dev_z < -dev_z_thresh) & ~vol_informed & (c > trend)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_VWAP_Execution_Dev(trial):
    return {
        'vwap_window':       trial.suggest_int('vwap_window',        20, 60),
        'dev_z_window':      trial.suggest_int('dev_z_window',       20, 50),
        'dev_z_thresh':      trial.suggest_float('dev_z_thresh',     1.2, 3.0),
        'atr_p':             trial.suggest_int('atr_p',               8, 20),
        'vol_accel_thresh':  trial.suggest_float('vol_accel_thresh',  1.1, 2.5),
        'trend_ema':         trial.suggest_int('trend_ema',           30, 80),
        'exit_bar':          trial.suggest_int('exit_bar',             3, 10),
    }


# ===========================================================================
# 4. BranchingRatio_Vol — Hawkes branching ratio regime detection
# ===========================================================================
# Bacry, Mastromatteo & Muzy (2015) show the Hawkes branching ratio
# m = ∫ µ(t) dt / λ̄ measures the fraction of events triggered by past events
# (vs exogenous arrivals). For financial volatility:
#   m → 1: near-critical regime (volatility cascade, trending)
#   m → 0: subcritical regime (calm, mean-reverting)
#
# Tractable proxy (moment-matching, Filimonov & Sornette 2012):
#   n_events[t]   = count of |ret| > event_thresh in last window
#   n_triggered   = events that occurred within decay_bars of a prior event
#   m_proxy       = n_triggered / (n_events + 1)
#
# Signal:
#   m_proxy close to 1 (high branching) → vol cascade likely → BREAKOUT
#   m_proxy close to 0 (subcritical)    → mean-reversion favored
#
# Direction:
#   BREAKOUT: sign of return over last few bars
#   REVERSION: opposite of recent return direction, with BB extreme

def gen_BranchingRatio_Vol(df, event_thresh_z=0.8, window=40, decay_bars=3,
                            m_high_thresh=0.7, m_low_thresh=0.3,
                            bb_p=20, bb_dev=2.0, trend_ema=50, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change()

    # Dynamic event threshold: z-score of |ret|
    ret_z = _zscore(ret.abs(), window)
    is_event = (ret_z > event_thresh_z).astype(float)

    n = len(c)
    m_vals = pd.Series(np.nan, index=c.index)

    for i in range(window, n):
        ev_window = is_event.values[i - window: i + 1]
        n_events  = ev_window.sum()
        if n_events < 2:
            m_vals.iloc[i] = 0.0
            continue

        # Count triggered events: event at t preceded by event in [t-decay, t-1]
        triggered = 0
        for j in range(1, len(ev_window)):
            if ev_window[j] == 1:
                # Check if any event in preceding decay_bars
                start_j = max(0, j - decay_bars)
                if ev_window[start_j:j].sum() > 0:
                    triggered += 1

        m_vals.iloc[i] = triggered / (n_events + 1e-5)

    # Rolling smooth branching ratio
    m_smooth = m_vals.rolling(window // 4).mean()

    # Bollinger bands for reversion entries
    bb_lo, bb_mid, bb_hi = _bb(c, bb_p, bb_dev)

    # Recent return direction
    ret_short = ret.rolling(decay_bars + 1).sum()
    ret_up    = ret_short > 0
    ret_down  = ret_short < 0

    # Trend guard
    trend = _ema(c, trend_ema)

    # Near-critical (m high) → breakout momentum in recent direction
    break_long  = (m_smooth > m_high_thresh) & ret_up   & (c > trend)
    break_short = (m_smooth > m_high_thresh) & ret_down & (c < trend)

    # Subcritical (m low) → mean-reversion at BB extremes
    rev_long    = (m_smooth < m_low_thresh) & (c < bb_lo) & (c > trend)
    rev_short   = (m_smooth < m_low_thresh) & (c > bb_hi) & (c < trend)

    entry_long  = break_long  | rev_long
    entry_short = break_short | rev_short

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_BranchingRatio_Vol(trial):
    return {
        'event_thresh_z': trial.suggest_float('event_thresh_z',  0.3, 1.5),
        'window':         trial.suggest_int('window',            25, 60),
        'decay_bars':     trial.suggest_int('decay_bars',          1,  6),
        'm_high_thresh':  trial.suggest_float('m_high_thresh',   0.5, 0.9),
        'm_low_thresh':   trial.suggest_float('m_low_thresh',    0.1, 0.4),
        'bb_p':           trial.suggest_int('bb_p',              15, 30),
        'bb_dev':         trial.suggest_float('bb_dev',           1.5, 2.5),
        'trend_ema':      trial.suggest_int('trend_ema',          30, 80),
        'exit_bar':       trial.suggest_int('exit_bar',            3, 12),
    }


# ===========================================================================
# 5. TDA_Shape_Rev — Topological Data Analysis shape complexity reversal
# ===========================================================================
# Carlsson (2009) Topological Data Analysis uses persistent homology to
# measure the "shape" of data. The 0th Betti number β0 counts connected
# components (clusters) of a level-set filtration. For time series, it
# measures how fragmented the return distribution is.
#
# Tractable proxy (discrete level-set connected components):
#   For a window of returns, count how many distinct "runs" cross ε-threshold:
#     - Partition returns into bins; count transitions between + and - bins
#     - This approximates β0 of the sublevel-set filtration
#   High β0 (many sign changes, oscillating) → high complexity = instability
#   Low β0 (few sign changes, trending/calm) = simple topology = trend
#
# Additional: Euler characteristic proxy = (peaks - valleys) in return series.
#   High |Euler char| → coherent movement (trending)
#   Low |Euler char|  → complex, oscillating → mean-reversion
#
# Signal:
#   High complexity (many zero-crossings) + price at BB extreme → REVERSION
#   Low complexity (coherent) + price in trend direction → TREND follow

def gen_TDA_Shape_Rev(df, tda_window=30, bb_p=20, bb_dev=2.0,
                      complexity_high=0.6, complexity_low=0.25,
                      trend_ema=50, vol_window=20, vol_z_min=0.0, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change()

    n = len(c)
    complexity = pd.Series(np.nan, index=c.index)

    for i in range(tda_window, n):
        r_win = ret.values[i - tda_window + 1: i + 1]
        r_win = r_win[~np.isnan(r_win)]
        if len(r_win) < tda_window // 2:
            continue

        # Count zero-crossings (sign changes) as β0 proxy
        signs          = np.sign(r_win)
        sign_changes   = np.sum(signs[1:] != signs[:-1])

        # Normalize by window length
        complexity.iloc[i] = sign_changes / (len(r_win) - 1 + 1e-5)

    # Smooth complexity
    cx_smooth = complexity.rolling(tda_window // 3).mean()

    # Bollinger bands for reversion entries
    bb_lo, bb_mid, bb_hi = _bb(c, bb_p, bb_dev)

    # Trend guard
    trend      = _ema(c, trend_ema)
    in_uptrend = c > trend

    # Volume guard
    vol_z = _zscore(v, vol_window)

    # High complexity → oscillating/unstable → mean-reversion at BB extremes
    # No directional guard needed — complexity itself is the regime signal
    rev_long  = (cx_smooth > complexity_high) & (c < bb_lo) & (vol_z >= vol_z_min)
    rev_short = (cx_smooth > complexity_high) & (c > bb_hi) & (vol_z >= vol_z_min)

    # Low complexity → coherent/trending → follow dominant EMA direction
    trend_long  = (cx_smooth < complexity_low) & in_uptrend  & (c > bb_mid) & (vol_z >= vol_z_min)
    trend_short = (cx_smooth < complexity_low) & ~in_uptrend & (c < bb_mid) & (vol_z >= vol_z_min)

    entry_long  = rev_long  | trend_long
    entry_short = rev_short | trend_short

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_TDA_Shape_Rev(trial):
    return {
        'tda_window':      trial.suggest_int('tda_window',         20, 50),
        'bb_p':            trial.suggest_int('bb_p',               15, 30),
        'bb_dev':          trial.suggest_float('bb_dev',            1.5, 2.5),
        'complexity_high': trial.suggest_float('complexity_high',   0.4, 0.8),
        'complexity_low':  trial.suggest_float('complexity_low',    0.1, 0.35),
        'trend_ema':       trial.suggest_int('trend_ema',           30, 80),
        'vol_window':      trial.suggest_int('vol_window',          15, 35),
        'vol_z_min':       trial.suggest_float('vol_z_min',         0.0, 1.0),
        'exit_bar':        trial.suggest_int('exit_bar',             3, 12),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'BOCD_Regime_Break': {
        'gen':   gen_BOCD_Regime_Break,
        'space': space_BOCD_Regime_Break,
    },
    'Resilience_Rev': {
        'gen':   gen_Resilience_Rev,
        'space': space_Resilience_Rev,
    },
    'VWAP_Execution_Dev': {
        'gen':   gen_VWAP_Execution_Dev,
        'space': space_VWAP_Execution_Dev,
    },
    'BranchingRatio_Vol': {
        'gen':   gen_BranchingRatio_Vol,
        'space': space_BranchingRatio_Vol,
    },
    'TDA_Shape_Rev': {
        'gen':   gen_TDA_Shape_Rev,
        'space': space_TDA_Shape_Rev,
    },
}
