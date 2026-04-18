#!/usr/bin/env python3
"""
TV2 BATCH MICRO — Game Theory + RL-Inspired Microstructure Strategies (5m/15m).
Batch 4 in session series (target batch 3535+).
Domain: Game Theory (Nash Mixed Strategy), RL Temporal-Difference, Cross-TF Lead-Lag
        Latency Arb, TWAP Execution Pressure, Bulk Volume Classification Divergence.
        All NEW vs MICRO1-18 and batches 3237-3534.

Strategies:
  1. NashMixedRegime     — Game Theory: Mixed-Strategy Nash Equilibrium detection.
                           When agents randomize strategies, price oscillates in a
                           bounded range with no dominant direction. Detect regime
                           transition OUT OF equilibrium (concentration in one
                           direction) as momentum signal.
                           Ref: Nash 1950, Fudenberg & Tirole 1991.

  2. TDLambdaSignal      — RL-inspired: Temporal Difference TD(lambda) eligibility
                           traces. Treats each bar's return as a reward, accumulates
                           a lambda-decayed eligibility trace, and fires when the
                           cumulative trace crosses a threshold. Models how an RL
                           agent would update Q-values across a trajectory.
                           Ref: Sutton & Barto 1998/2018, Watkins 1989.

  3. CrossTFLeadLag      — Latency Arbitrage: Detects when the fast timeframe
                           (2-bar micro return) systematically LEADS the slow
                           timeframe return (N-bar return) by detecting divergence
                           between fast micro-momentum and slow macro-momentum.
                           Ref: Hasbrouck & Schwartz 1988, Chordia et al. 2008.

  4. TWAPExecPressure    — Order Flow: TWAP execution algorithmic pressure detector.
                           Institutional TWAP orders create steady price drift away
                           from rolling VWAP with compressed volatility. Detect
                           TWAP sweep: |close-TWAP| > threshold AND ATR below
                           baseline AND volume acceleration = algorithmic sweep.
                           Ref: Almgren & Chriss 2000, Kissell 2013.

  5. BVCDivergence       — Tick Rule / Trade Sign: Bulk Volume Classification (BVC)
                           divergence. BVC classifies each bar as buy/sell volume
                           using the tick rule proxy. When the rolling BVC ratio
                           DIVERGES from price direction (high buy volume but price
                           falling, or high sell volume but price rising), it signals
                           an impending reversal — exhaustion of directional flow.
                           Ref: Easley, Lopez de Prado & O'Hara 2012 (VPIN/BVC).

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Nash, J.F. (1950) — "Equilibrium Points in N-Person Games";
    Proceedings of the National Academy of Sciences 36(1): 48-49.
  - Fudenberg, D. & Tirole, J. (1991) — "Game Theory"; MIT Press.
  - Sutton, R.S. & Barto, A.G. (2018) — "Reinforcement Learning: An Introduction"
    2nd ed.; MIT Press. (TD(lambda) Chapter 12).
  - Watkins, C.J.C.H. (1989) — "Learning from Delayed Rewards"; PhD Thesis,
    Cambridge University.
  - Hasbrouck, J. & Schwartz, R.A. (1988) — "Liquidity and Execution Costs in
    Equity Markets"; Journal of Portfolio Management 14(3): 10-16.
  - Chordia, T., Roll, R. & Subrahmanyam, A. (2008) — "Liquidity and Market
    Efficiency"; Journal of Financial Economics 87(2): 249-268.
  - Almgren, R. & Chriss, N. (2000) — "Optimal Execution of Portfolio Transactions";
    Journal of Risk 3(2): 5-39.
  - Kissell, R. (2013) — "The Science of Algorithmic Trading and Portfolio Management";
    Academic Press.
  - Easley, D., Lopez de Prado, M.M. & O'Hara, M. (2012) — "Flow Toxicity and
    Liquidity in a High-Frequency World"; Review of Financial Studies 25(5): 1457-1493.
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


def _apply_exit_bar(entry_long, entry_short, exit_bar):
    """
    3-state machine: enters on signal, exits after exit_bar bars.
    Flat(0) → Long(1) on entry_long; Flat(0) → Short(-1) on entry_short.
    New entries override opposite position immediately.
    """
    sig = pd.Series(0, index=entry_long.index, dtype=int)
    el  = entry_long.values
    es  = entry_short.values
    n   = len(sig)
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
# 1. NashMixedRegime — Nash Mixed-Strategy Equilibrium Regime Detector
# ===========================================================================
# Nash (1950) proved that every finite game has at least one mixed-strategy
# equilibrium. In liquid markets, when no single strategy dominates, agents
# mix — prices oscillate in a bounded range (equilibrium). When one side gains
# information advantage or coordinated flow, the equilibrium BREAKS and a
# directional move follows.
#
# OHLCV proxy for mixed vs pure strategy regime:
#   Range concentration index (RCI): rolling std of (high-low) / (close+eps)
#   When RCI is LOW (stable, narrow range) → mixed strategy equilibrium
#   When RCI SPIKES above threshold → regime break → directional signal
#
# Direction filter:
#   Directional bias = EMA(close) slope sign over a lookback
#   When RCI breaks UP AND momentum is positive → LONG
#   When RCI breaks UP AND momentum is negative → SHORT
#
# Confirmation: volume must also spike (concentration of order flow)
#   vol_z > vol_thresh required to confirm the regime break is real.

def gen_NashMixedRegime(df, rci_window=20, rci_break_z=1.6,
                         mom_period=12, trend_ema=40,
                         vol_z_window=25, vol_z_thresh=0.8,
                         exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    # Range Concentration Index per bar
    bar_range = (h - l) / (c + 1e-10)

    # Rolling std of bar_range: LOW = equilibrium, HIGH = break
    rci = bar_range.rolling(rci_window).std(ddof=0)

    # Z-score of RCI to detect spikes
    rci_z = _zscore(rci, rci_window * 2)

    # Momentum: close vs EMA as direction filter
    trend    = _ema(c, trend_ema)
    mom_fast = _ema(c, mom_period)
    up_bias  = mom_fast > trend
    dn_bias  = mom_fast < trend

    # Volume confirmation
    vol_z = _zscore(v, vol_z_window)
    vol_confirm = vol_z > vol_z_thresh

    # Signal: regime break (RCI spike) + direction + volume
    regime_break = rci_z > rci_break_z

    entry_long  = regime_break & up_bias & vol_confirm
    entry_short = regime_break & dn_bias & vol_confirm

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_NashMixedRegime(trial):
    return {
        'rci_window':    trial.suggest_int('rci_window',    10, 35),
        'rci_break_z':   trial.suggest_float('rci_break_z',  1.0, 2.8),
        'mom_period':    trial.suggest_int('mom_period',    6, 25),
        'trend_ema':     trial.suggest_int('trend_ema',     25, 80),
        'vol_z_window':  trial.suggest_int('vol_z_window',  15, 50),
        'vol_z_thresh':  trial.suggest_float('vol_z_thresh', 0.3, 2.0),
        'exit_bar':      trial.suggest_int('exit_bar',       3, 12),
    }


# ===========================================================================
# 2. TDLambdaSignal — Temporal Difference TD(λ) Eligibility Trace Signal
# ===========================================================================
# Sutton & Barto (2018) Chapter 12: TD(lambda) uses eligibility traces to
# assign credit across a trajectory. Each bar's return is treated as a
# "reward". The eligibility trace accumulates returns with lambda-decay,
# simulating how an RL agent would update Q-values over a sequence.
#
# Formulation (OHLCV implementation):
#   r[t]    = (close[t] - close[t-1]) / close[t-1]  ← per-bar return
#   e[t]    = lambda * e[t-1] + r[t]                 ← eligibility trace
#   (implemented as EWM with alpha = 1 - lambda, which gives the same form)
#
# Signal logic:
#   e_z     = z-score of e[t] over lookback window
#   LONG    when e_z > thresh AND close above EMA trend (reinforced uptrend)
#   SHORT   when e_z < -thresh AND close below EMA trend
#
# The lambda parameter controls memory: lambda=0.9 → long memory (trend-following),
# lambda=0.5 → short memory (mean-reverting). Optuna optimizes this.

def gen_TDLambdaSignal(df, td_lambda=0.85, z_window=30, z_thresh=1.5,
                        trend_ema=50, atr_period=14, atr_filter=True,
                        exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    # Per-bar return (reward signal)
    ret = c.pct_change()

    # TD(lambda) eligibility trace via EWM
    # EWM with alpha = (1 - lambda) approximates the recurrence e[t] = lambda*e[t-1] + r[t]
    alpha = max(1e-5, 1.0 - td_lambda)
    trace = ret.ewm(alpha=alpha, adjust=False).mean()

    # Z-score of trace for normalized signal
    trace_z = _zscore(trace, z_window)

    # Trend filter
    trend  = _ema(c, trend_ema)
    above  = c > trend
    below  = c < trend

    # Optional ATR volatility filter: skip during extreme volatility
    if atr_filter:
        atr     = _atr(h, l, c, atr_period)
        atr_z   = _zscore(atr, z_window)
        atr_ok  = atr_z < 2.0          # skip if ATR anomalously high
    else:
        atr_ok  = pd.Series(True, index=c.index)

    entry_long  = (trace_z >  z_thresh) & above & atr_ok
    entry_short = (trace_z < -z_thresh) & below & atr_ok

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_TDLambdaSignal(trial):
    return {
        'td_lambda':  trial.suggest_float('td_lambda',  0.50, 0.97),
        'z_window':   trial.suggest_int('z_window',     15, 60),
        'z_thresh':   trial.suggest_float('z_thresh',    0.8, 2.5),
        'trend_ema':  trial.suggest_int('trend_ema',    25, 100),
        'atr_period': trial.suggest_int('atr_period',   7, 21),
        'atr_filter': trial.suggest_categorical('atr_filter', [True, False]),
        'exit_bar':   trial.suggest_int('exit_bar',      3, 12),
    }


# ===========================================================================
# 3. CrossTFLeadLag — Cross-Timeframe Price Discovery Lead-Lag Signal
# ===========================================================================
# Hasbrouck & Schwartz (1988) and Chordia et al. (2008): price discovery in
# liquid markets exhibits lead-lag structure where fast-reacting market
# participants anticipate slow-reacting ones.
#
# OHLCV implementation (single timeframe):
#   fast_ret[t] = micro-return over 2 bars  (short lookback = "fast TF")
#   slow_ret[t] = macro-return over N bars  (long lookback = "slow TF")
#
# Lead-lag divergence:
#   lead_lag = fast_ret - slow_ret_normalized
#   When fast_ret STRONGLY POSITIVE but slow_ret still NEGATIVE (or flat):
#     → fast side is "discovering" new price → LONG (front-run the slow side)
#   When fast_ret STRONGLY NEGATIVE but slow_ret still POSITIVE (or flat):
#     → fast side discovering downside → SHORT
#
# Confirmation: the fast/slow spread must be statistically significant (z-score).

def gen_CrossTFLeadLag(df, fast_period=3, slow_period=20,
                        ll_z_window=40, ll_z_thresh=1.5,
                        trend_filter_period=60,
                        vol_confirm_period=20,
                        exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    # Fast return (micro): N-bar return over fast_period
    fast_ret = c.pct_change(fast_period)

    # Slow return (macro): N-bar return over slow_period
    slow_ret = c.pct_change(slow_period)

    # Normalize slow_ret to fast_ret scale to make them comparable
    # Use ratio of std over matching window
    fast_std = fast_ret.rolling(ll_z_window).std(ddof=0) + 1e-10
    slow_std = slow_ret.rolling(ll_z_window).std(ddof=0) + 1e-10
    slow_ret_norm = slow_ret * (fast_std / slow_std)

    # Lead-lag divergence: fast ahead of slow
    ll_div = fast_ret - slow_ret_norm

    # Z-score the divergence
    ll_z = _zscore(ll_div, ll_z_window)

    # Trend guard: position in market structure
    trend  = _ema(c, trend_filter_period)
    above  = c > trend
    below  = c < trend

    # Volume confirmation: above rolling average
    vol_avg  = _sma(v, vol_confirm_period)
    vol_ok   = v > vol_avg * 0.8

    # Fast leads slow UP → entry long (fast already moving, slow lagging)
    entry_long  = (ll_z >  ll_z_thresh) & above & vol_ok
    # Fast leads slow DOWN → entry short
    entry_short = (ll_z < -ll_z_thresh) & below & vol_ok

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_CrossTFLeadLag(trial):
    return {
        'fast_period':          trial.suggest_int('fast_period',          2, 8),
        'slow_period':          trial.suggest_int('slow_period',          12, 40),
        'll_z_window':          trial.suggest_int('ll_z_window',          20, 70),
        'll_z_thresh':          trial.suggest_float('ll_z_thresh',         1.0, 2.5),
        'trend_filter_period':  trial.suggest_int('trend_filter_period',  30, 120),
        'vol_confirm_period':   trial.suggest_int('vol_confirm_period',   10, 40),
        'exit_bar':             trial.suggest_int('exit_bar',              3, 12),
    }


# ===========================================================================
# 4. TWAPExecPressure — TWAP Algorithmic Execution Pressure Detector
# ===========================================================================
# Almgren & Chriss (2000) and Kissell (2013): institutional TWAP algorithms
# slice large orders into time-equal buckets, creating a predictable pattern:
#   - Steady price drift away from rolling average (execution footprint)
#   - Compressed intra-bar volatility (algo smooths execution)
#   - Volume near average (TWAP distributes volume evenly by design)
#
# Detection proxy:
#   twap[t] = rolling mean of close over twap_window (time-weighted avg price proxy)
#   dev[t]  = (close - twap) / atr  ← normalized deviation from TWAP
#   atr_compressed: rolling ATR / long_ATR < compress_thresh → compressed volatility
#   vol_steady:     |vol_z| < vol_steady_thresh → no volume spike (TWAP behavior)
#
# When dev > thresh AND atr compressed AND vol steady → TWAP sweep in progress
# → follow the sweep direction (momentum during execution pressure)
# When dev > thresh but atr EXPANDING → natural move, not TWAP → skip

def gen_TWAPExecPressure(df, twap_window=25, dev_thresh=1.2,
                          atr_short=8, atr_long=40,
                          compress_thresh=0.75,
                          vol_z_window=25, vol_steady_thresh=1.5,
                          trend_ema=50, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    # TWAP proxy: rolling mean of close
    twap = _sma(c, twap_window)

    # Short and long ATR for compression detection
    atr_s = _atr(h, l, c, atr_short)
    atr_l = _atr(h, l, c, atr_long)
    atr_ratio = atr_s / (atr_l + 1e-10)
    atr_compressed = atr_ratio < compress_thresh

    # Normalized deviation from TWAP
    dev = (c - twap) / (atr_l + 1e-10)

    # Volume steadiness: |vol_z| < threshold means no anomalous vol spike
    vol_z = _zscore(v, vol_z_window)
    vol_steady = vol_z.abs() < vol_steady_thresh

    # Trend guard
    trend = _ema(c, trend_ema)
    above = c > trend
    below = c < trend

    # TWAP sweep: sustained deviation + compressed ATR + steady volume
    sweep_up   = (dev >  dev_thresh) & atr_compressed & vol_steady & above
    sweep_down = (dev < -dev_thresh) & atr_compressed & vol_steady & below

    return _apply_exit_bar(sweep_up, sweep_down, exit_bar)


def space_TWAPExecPressure(trial):
    return {
        'twap_window':        trial.suggest_int('twap_window',        12, 50),
        'dev_thresh':         trial.suggest_float('dev_thresh',         0.6, 2.5),
        'atr_short':          trial.suggest_int('atr_short',           4, 14),
        'atr_long':           trial.suggest_int('atr_long',            20, 70),
        'compress_thresh':    trial.suggest_float('compress_thresh',    0.4, 0.95),
        'vol_z_window':       trial.suggest_int('vol_z_window',        15, 50),
        'vol_steady_thresh':  trial.suggest_float('vol_steady_thresh',  0.5, 2.5),
        'trend_ema':          trial.suggest_int('trend_ema',           25, 100),
        'exit_bar':           trial.suggest_int('exit_bar',             3, 12),
    }


# ===========================================================================
# 5. BVCDivergence — Bulk Volume Classification Divergence Signal
# ===========================================================================
# Easley, Lopez de Prado & O'Hara (2012): Bulk Volume Classification (BVC)
# assigns each bar's volume as buy or sell using the tick rule:
#   If close[t] > close[t-1] → buy volume  (up-tick)
#   If close[t] < close[t-1] → sell volume (down-tick)
#   If close[t] == close[t-1] → prior classification (simplification: 0.5 each)
#
# BVC ratio: rolling_buy_vol / (rolling_buy_vol + rolling_sell_vol + eps)
# BVC_ratio > 0.5 → net buy pressure; < 0.5 → net sell pressure
#
# DIVERGENCE signal (key novelty vs LeeReady_Cont which is continuation):
#   Price going UP but BVC_ratio FALLING → informed sellers absorbing buys → SHORT
#   Price going DOWN but BVC_ratio RISING → informed buyers absorbing sells → LONG
#
# This is the "smart money vs dumb money" divergence:
# When retail drives price in one direction but BVC shows opposite flow,
# institutional (informed) flow is the reality → fade the retail direction.

def gen_BVCDivergence(df, bvc_window=20, bvc_z_window=40,
                       price_ret_period=10,
                       div_thresh=1.3,
                       trend_ema=60,
                       exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    # Tick rule classification
    c_prev   = c.shift(1)
    up_tick  = c > c_prev
    dn_tick  = c < c_prev

    # Assign volume to buy/sell per BVC tick rule
    buy_vol  = v.where(up_tick,  other=v * 0.5)   # tie → 50% each
    sell_vol = v.where(dn_tick,  other=v * 0.5)

    # Rolling BVC ratio
    rolling_buy  = buy_vol.rolling(bvc_window).sum()
    rolling_sell = sell_vol.rolling(bvc_window).sum()
    bvc_ratio    = rolling_buy / (rolling_buy + rolling_sell + 1e-10)

    # Z-score of BVC ratio (normalized flow balance)
    bvc_z = _zscore(bvc_ratio, bvc_z_window)

    # Price momentum direction over lookback
    price_ret = c.pct_change(price_ret_period)
    price_up  = price_ret > 0
    price_dn  = price_ret < 0

    # DIVERGENCE: price going one way, BVC going the other
    # Price UP but BVC declining (sell pressure hidden in up-move) → SHORT
    div_short = price_up  & (bvc_z < -div_thresh)
    # Price DOWN but BVC rising (buy pressure hidden in down-move) → LONG
    div_long  = price_dn  & (bvc_z >  div_thresh)

    # Trend guard: divergence against extreme trend is dangerous
    trend  = _ema(c, trend_ema)
    # For short: don't fade strong uptrend; for long: don't fade strong downtrend
    # Use: short only if close < trend*1.02 (not in extreme bull), long if close > trend*0.98
    safe_short = c < trend * 1.02
    safe_long  = c > trend * 0.98

    entry_long  = div_long  & safe_long
    entry_short = div_short & safe_short

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_BVCDivergence(trial):
    return {
        'bvc_window':       trial.suggest_int('bvc_window',       10, 40),
        'bvc_z_window':     trial.suggest_int('bvc_z_window',     25, 80),
        'price_ret_period': trial.suggest_int('price_ret_period',  5, 25),
        'div_thresh':       trial.suggest_float('div_thresh',       0.8, 2.5),
        'trend_ema':        trial.suggest_int('trend_ema',         30, 120),
        'exit_bar':         trial.suggest_int('exit_bar',           3, 12),
    }


# ─── WRAPPER CLASSES ────────────────────────────────────────────────────────
# Each strategy is wrapped in a class for the standard pipeline interface.

class NashMixedRegimeStrategy:
    def __init__(self, **params):
        self.params = params

    def generate_signal(self, df):
        return gen_NashMixedRegime(df, **self.params)

    def get_optuna_space(self, trial):
        return space_NashMixedRegime(trial)


class TDLambdaSignalStrategy:
    def __init__(self, **params):
        self.params = params

    def generate_signal(self, df):
        return gen_TDLambdaSignal(df, **self.params)

    def get_optuna_space(self, trial):
        return space_TDLambdaSignal(trial)


class CrossTFLeadLagStrategy:
    def __init__(self, **params):
        self.params = params

    def generate_signal(self, df):
        return gen_CrossTFLeadLag(df, **self.params)

    def get_optuna_space(self, trial):
        return space_CrossTFLeadLag(trial)


class TWAPExecPressureStrategy:
    def __init__(self, **params):
        self.params = params

    def generate_signal(self, df):
        return gen_TWAPExecPressure(df, **self.params)

    def get_optuna_space(self, trial):
        return space_TWAPExecPressure(trial)


class BVCDivergenceStrategy:
    def __init__(self, **params):
        self.params = params

    def generate_signal(self, df):
        return gen_BVCDivergence(df, **self.params)

    def get_optuna_space(self, trial):
        return space_BVCDivergence(trial)


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────
# Format: dict keyed by strategy name.
# 'gen'   — pickleable function reference (NEVER lambda)
# 'space' — Optuna space function
# 'class' — strategy class for pipeline instantiation
# 'default_params' — default param values (from function signatures)
# 'timeframes' — supported timeframes

def _gen_NashMixedRegime_export(df, **p):
    return gen_NashMixedRegime(df, **p)

def _gen_TDLambdaSignal_export(df, **p):
    return gen_TDLambdaSignal(df, **p)

def _gen_CrossTFLeadLag_export(df, **p):
    return gen_CrossTFLeadLag(df, **p)

def _gen_TWAPExecPressure_export(df, **p):
    return gen_TWAPExecPressure(df, **p)

def _gen_BVCDivergence_export(df, **p):
    return gen_BVCDivergence(df, **p)


STRATEGY_EXPORT = {
    'TV_NashMixedRegime': {
        'name':           'TV_NashMixedRegime',
        'gen':            _gen_NashMixedRegime_export,
        'space':          space_NashMixedRegime,
        'class':          NashMixedRegimeStrategy,
        'default_params': {
            'rci_window':   20,
            'rci_break_z':  1.6,
            'mom_period':   12,
            'trend_ema':    40,
            'vol_z_window': 25,
            'vol_z_thresh': 0.8,
            'exit_bar':     5,
        },
        'timeframes': ['5m', '15m'],
    },
    'TV_TDLambdaSignal': {
        'name':           'TV_TDLambdaSignal',
        'gen':            _gen_TDLambdaSignal_export,
        'space':          space_TDLambdaSignal,
        'class':          TDLambdaSignalStrategy,
        'default_params': {
            'td_lambda':  0.85,
            'z_window':   30,
            'z_thresh':   1.5,
            'trend_ema':  50,
            'atr_period': 14,
            'atr_filter': True,
            'exit_bar':   6,
        },
        'timeframes': ['5m', '15m'],
    },
    'TV_CrossTFLeadLag': {
        'name':           'TV_CrossTFLeadLag',
        'gen':            _gen_CrossTFLeadLag_export,
        'space':          space_CrossTFLeadLag,
        'class':          CrossTFLeadLagStrategy,
        'default_params': {
            'fast_period':         3,
            'slow_period':         20,
            'll_z_window':         40,
            'll_z_thresh':         1.5,
            'trend_filter_period': 60,
            'vol_confirm_period':  20,
            'exit_bar':            5,
        },
        'timeframes': ['5m', '15m'],
    },
    'TV_TWAPExecPressure': {
        'name':           'TV_TWAPExecPressure',
        'gen':            _gen_TWAPExecPressure_export,
        'space':          space_TWAPExecPressure,
        'class':          TWAPExecPressureStrategy,
        'default_params': {
            'twap_window':       25,
            'dev_thresh':        1.2,
            'atr_short':         8,
            'atr_long':          40,
            'compress_thresh':   0.75,
            'vol_z_window':      25,
            'vol_steady_thresh': 1.5,
            'trend_ema':         50,
            'exit_bar':          5,
        },
        'timeframes': ['5m', '15m'],
    },
    'TV_BVCDivergence': {
        'name':           'TV_BVCDivergence',
        'gen':            _gen_BVCDivergence_export,
        'space':          space_BVCDivergence,
        'class':          BVCDivergenceStrategy,
        'default_params': {
            'bvc_window':       20,
            'bvc_z_window':     40,
            'price_ret_period': 10,
            'div_thresh':       1.3,
            'trend_ema':        60,
            'exit_bar':         6,
        },
        'timeframes': ['5m', '15m'],
    },
}


# ─── VALIDATION SUITE (run directly for 5-check validation) ─────────────────
if __name__ == '__main__':
    import sys

    print("=" * 65)
    print("TV2 BATCH MICRO GT/RL — 5-Check Validation Suite")
    print("=" * 65)

    # Build a synthetic OHLCV DataFrame (400 bars, enough warmup)
    np.random.seed(42)
    n = 400
    price = 100.0 * np.cumprod(1 + np.random.normal(0, 0.002, n))
    noise = np.random.uniform(-0.5, 0.5, n)
    df_test = pd.DataFrame({
        'open':   price * (1 + np.random.uniform(-0.001, 0.001, n)),
        'high':   price * (1 + np.abs(noise) * 0.005 + 0.002),
        'low':    price * (1 - np.abs(noise) * 0.005 - 0.002),
        'close':  price,
        'volume': np.abs(np.random.normal(1000, 200, n)),
    })

    all_pass = True
    for strat_name, cfg in STRATEGY_EXPORT.items():
        print(f"\n--- {strat_name} ---")
        errors = []

        # Check 1: instantiate class
        try:
            inst = cfg['class'](**cfg['default_params'])
            print("  [1] Type check (instantiable): PASS")
        except Exception as e:
            errors.append(f"Type check FAIL: {e}")
            print(f"  [1] Type check FAIL: {e}")

        # Check 2: generate_signal returns pd.Series with values in {-1,0,1}
        try:
            sig = inst.generate_signal(df_test)
            assert isinstance(sig, pd.Series), "not a pd.Series"
            bad = set(sig.dropna().unique()) - {-1, 0, 1}
            assert len(bad) == 0, f"unexpected values: {bad}"
            n_signals = (sig != 0).sum()
            print(f"  [2] Signature check (Series {{-1,0,1}}): PASS  "
                  f"(signals={n_signals}/{len(sig)})")
        except Exception as e:
            errors.append(f"Signature check FAIL: {e}")
            print(f"  [2] Signature check FAIL: {e}")

        # Check 3: Optuna space works with mock trial
        try:
            class MockTrial:
                def suggest_int(self, n, lo, hi, **kw):   return (lo + hi) // 2
                def suggest_float(self, n, lo, hi, **kw): return (lo + hi) / 2.0
                def suggest_categorical(self, n, choices, **kw): return choices[0]
            space = cfg['space'](MockTrial())
            assert isinstance(space, dict), "space is not dict"
            assert len(space) > 0, "empty space"
            print(f"  [3] Optuna space check ({len(space)} params): PASS")
        except Exception as e:
            errors.append(f"Optuna space FAIL: {e}")
            print(f"  [3] Optuna space FAIL: {e}")

        # Check 4: STRATEGY_EXPORT fields
        required = {'name', 'gen', 'space', 'class', 'default_params', 'timeframes'}
        missing  = required - set(cfg.keys())
        if missing:
            errors.append(f"STRATEGY_EXPORT missing fields: {missing}")
            print(f"  [4] STRATEGY_EXPORT check FAIL — missing: {missing}")
        else:
            tfs = cfg['timeframes']
            assert '5m' in tfs and '15m' in tfs, f"missing 5m/15m in timeframes: {tfs}"
            print(f"  [4] STRATEGY_EXPORT check (fields + timeframes): PASS")

        # Check 5: gen function is pickleable (not a lambda)
        import pickle
        try:
            pickle.dumps(cfg['gen'])
            print(f"  [5] Pickle (gen not lambda): PASS")
        except Exception as e:
            errors.append(f"Pickle FAIL: {e}")
            print(f"  [5] Pickle FAIL: {e}")

        if errors:
            all_pass = False
            print(f"  *** ERRORS: {errors}")

    print("\n" + "=" * 65)
    if all_pass:
        print("RESULT: ALL 5 CHECKS PASSED for all 5 strategies.")
        print("File ready for Optuna pipeline.")
    else:
        print("RESULT: SOME CHECKS FAILED — see above.")
        sys.exit(1)
