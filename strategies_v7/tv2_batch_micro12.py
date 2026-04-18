#!/usr/bin/env python3
"""
TV2 BATCH MICRO12 — 5 microstructure strategies (5m/15m optimized).
Domain: Game Theory, Reinforcement Learning Signals, Optimal Stopping,
        Kyle Information Asymmetry, Latency Arbitrage Proxy.
(academic, distinct from MICRO1-11 + batches 3237-3318)

Strategies:
  1. StackelbergSpread_Rev — Stackelberg (1934) leader-follower game: market
                             maker compresses spread when sensing informed flow.
                             Proxy: HL range much narrower than ATR → inventory
                             reversion fade signal.
  2. BellmanMomentum       — Bellman (1957) dynamic programming / RL Q-value
                             proxy: Q(state, action) = directional momentum
                             strength / normalized volatility. High Q → entry.
  3. KyleAlpha_Dir         — Kyle (1985) strategic insider model: order flow
                             imbalance (OFI) = Lee-Ready buy_vol - sell_vol.
                             Sustained OFI in one direction → informed flow →
                             trend-following signal.
  4. DixitOptStop          — Dixit & Pindyck (1994) real options: traders wait
                             while uncertain. Volatility compression AFTER a
                             directional move = uncertainty resolved = momentum
                             continuation entry.
  5. LatencyArb_Fade       — Budish, Cramton & Shim (2015) latency arbitrage:
                             narrow counter-trend bar = HFT arb clearing a stale
                             quote dislocation → fade back to trend direction.

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Stackelberg, H. von (1934) — "Marktform und Gleichgewicht" (leader-follower
    strategic equilibrium in oligopoly markets).
  - Bellman, R. (1957) — "Dynamic Programming"; Princeton University Press
    (Markov decision processes, value functions, Q-values).
  - Kyle, A.S. (1985) — "Continuous Auctions and Insider Trading";
    Econometrica 53(6): 1315-1335.
  - Dixit, A.K. & Pindyck, R.S. (1994) — "Investment under Uncertainty";
    Princeton University Press (Optimal stopping, real options).
  - Budish, E., Cramton, P. & Shim, J. (2015) — "The High-Frequency Trading
    Arms Race: Frequent Batch Auctions as a Market Design Response"; Quarterly
    Journal of Economics 130(4): 1547-1621.
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
    m = s.rolling(p).mean()
    sd = s.rolling(p).std(ddof=0) + 1e-10
    return (s - m) / sd

def _apply_exit_bar(entry_long, entry_short, exit_bar):
    """
    3-state machine: enters on signal, exits after exit_bar bars.
    Flat(0) -> Long(1) on entry_long; Flat(0) -> Short(-1) on entry_short.
    New entries override opposite position immediately.
    """
    sig = pd.Series(0, index=entry_long.index, dtype=int)
    el = entry_long.values
    es = entry_short.values
    n = len(sig)
    state = 0
    bars_held = 0
    for i in range(n):
        if state != 0:
            bars_held += 1
            if bars_held >= exit_bar:
                state = 0
                bars_held = 0
        if el[i]:
            state = 1
            bars_held = 0
        elif es[i]:
            state = -1
            bars_held = 0
        sig.iloc[i] = state
    return sig


# ===========================================================================
# 1. StackelbergSpread_Rev — von Stackelberg (1934) leader-follower game
# ===========================================================================
# In the Stackelberg model the market maker (leader) sets spread first;
# informed traders (followers) react to it. When the intra-bar HL range is
# abnormally NARROW relative to ATR, the market maker has compressed the
# spread — typically done when sensing informed order flow and wanting to
# protect inventory. After filling the informed side, the maker reverts
# inventory in the opposite direction.
#
# Signal:
#   spread_ratio = (high - low) / ATR  ->  low ratio = compressed spread
#   bar_ret      = (close - open) / open
#   Compressed spread + positive bar_ret -> short (maker inventory reversion)
#   Compressed spread + negative bar_ret -> long  (maker inventory reversion)

def gen_StackelbergSpread_Rev(df, atr_period=14, spread_ratio_thresh=0.45,
                               bar_ret_thresh=0.0008, trend_ema=50,
                               vol_window=20, vol_z_min=0.2, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    atr    = _atr(h, l, c, atr_period)
    range_ = (h - l).clip(lower=1e-12)
    spread_ratio = range_ / (atr + 1e-12)

    bar_ret = (c - o) / (o + 1e-12)

    compressed = spread_ratio < spread_ratio_thresh

    vol_z    = _zscore(v, vol_window)
    vol_ok   = vol_z > vol_z_min

    entry_short = compressed & vol_ok & (bar_ret >  bar_ret_thresh)
    entry_long  = compressed & vol_ok & (bar_ret < -bar_ret_thresh)

    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.980)
    entry_short = entry_short & (c < trend * 1.020)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_StackelbergSpread_Rev(trial):
    return {
        'atr_period':          trial.suggest_int('atr_period', 8, 20),
        'spread_ratio_thresh': trial.suggest_float('spread_ratio_thresh', 0.30, 0.65, step=0.05),
        'bar_ret_thresh':      trial.suggest_float('bar_ret_thresh', 0.0003, 0.0020, step=0.0001),
        'trend_ema':           trial.suggest_int('trend_ema', 30, 70),
        'vol_window':          trial.suggest_int('vol_window', 12, 30),
        'vol_z_min':           trial.suggest_float('vol_z_min', -0.5, 0.5, step=0.25),
        'exit_bar':            trial.suggest_int('exit_bar', 3, 12),
    }


# ===========================================================================
# 2. BellmanMomentum — Bellman (1957) dynamic programming Q-value proxy
# ===========================================================================
# Bellman's principle of optimality: optimal policy depends only on current
# state (Markov). Q(state, action) = expected reward of taking action in state.
#
# Proxy Q-value:
#   momentum  = EMA(ret, fast) - EMA(ret, slow)   (directional strength)
#   norm_vol  = short_vol / long_vol               (relative uncertainty)
#   Q_long    = positive_momentum / (norm_vol + e) (strength / uncertainty)
#   Q_short   = negative_momentum / (norm_vol + e)
#
# High Q_long (strong up momentum + low relative vol) -> enter long.
# Normalize Q by rolling max for cross-asset comparability.

def gen_BellmanMomentum(df, fast_ema=5, slow_ema=20, vol_window=15,
                         long_vol_window=60, q_thresh_long=0.5,
                         q_thresh_short=0.5, trend_ema=40, exit_bar=7):
    c = df['close'].shift(1)

    ret      = c.pct_change().fillna(0)
    fast_ret = _ema(ret, fast_ema)
    slow_ret = _ema(ret, slow_ema)
    momentum = fast_ret - slow_ret

    vol_short = ret.rolling(vol_window).std(ddof=0) + 1e-10
    vol_long  = ret.rolling(long_vol_window).std(ddof=0) + 1e-10
    norm_vol  = vol_short / (vol_long + 1e-10)

    q_long  = momentum.clip(lower=0)    / (norm_vol + 1e-6)
    q_short = (-momentum).clip(lower=0) / (norm_vol + 1e-6)

    q_long_norm  = q_long  / (q_long.rolling(long_vol_window).max()  + 1e-10)
    q_short_norm = q_short / (q_short.rolling(long_vol_window).max() + 1e-10)

    entry_long  = q_long_norm  > q_thresh_long
    entry_short = q_short_norm > q_thresh_short

    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.985)
    entry_short = entry_short & (c < trend * 1.015)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_BellmanMomentum(trial):
    return {
        'fast_ema':         trial.suggest_int('fast_ema', 3, 12),
        'slow_ema':         trial.suggest_int('slow_ema', 15, 35),
        'vol_window':       trial.suggest_int('vol_window', 8, 25),
        'long_vol_window':  trial.suggest_int('long_vol_window', 40, 90),
        'q_thresh_long':    trial.suggest_float('q_thresh_long', 0.3, 0.85, step=0.05),
        'q_thresh_short':   trial.suggest_float('q_thresh_short', 0.3, 0.85, step=0.05),
        'trend_ema':        trial.suggest_int('trend_ema', 25, 60),
        'exit_bar':         trial.suggest_int('exit_bar', 4, 14),
    }


# ===========================================================================
# 3. KyleAlpha_Dir — Kyle (1985) strategic insider trading / order flow
# ===========================================================================
# Kyle (1985): informed traders leak private information into prices via order
# flow. Market maker infers from aggregate flow and adjusts prices.
# OFI proxy (no tick data needed):
#   buy_vol  = volume * (close - low) / (high - low)   [Lee-Ready proxy]
#   sell_vol = volume - buy_vol
#   OFI      = buy_vol - sell_vol
#   OFI_z    = Z-score(OFI, window)
#
# Sustained OFI_z > threshold for N consecutive bars -> informed buying ->
# trend-following long entry. Symmetric for sellers.

def gen_KyleAlpha_Dir(df, ofi_window=20, ofi_z_thresh=1.0,
                       sustain_bars=3, trend_ema=30, exit_bar=8):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    hl_range  = (h - l).clip(lower=1e-10)
    buy_frac  = ((c - l) / hl_range).clip(0.0, 1.0)
    buy_vol   = v * buy_frac
    sell_vol  = v * (1.0 - buy_frac)
    ofi       = buy_vol - sell_vol

    ofi_z = _zscore(ofi, ofi_window)

    strong_buy  = ofi_z > ofi_z_thresh
    strong_sell = ofi_z < -ofi_z_thresh

    buy_sustained  = strong_buy.rolling(sustain_bars).min().fillna(0).astype(bool)
    sell_sustained = strong_sell.rolling(sustain_bars).min().fillna(0).astype(bool)

    trend = _ema(c, trend_ema)
    entry_long  = buy_sustained  & (c > trend * 0.990)
    entry_short = sell_sustained & (c < trend * 1.010)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_KyleAlpha_Dir(trial):
    return {
        'ofi_window':    trial.suggest_int('ofi_window', 10, 35),
        'ofi_z_thresh':  trial.suggest_float('ofi_z_thresh', 0.5, 2.0, step=0.25),
        'sustain_bars':  trial.suggest_int('sustain_bars', 2, 6),
        'trend_ema':     trial.suggest_int('trend_ema', 20, 55),
        'exit_bar':      trial.suggest_int('exit_bar', 4, 14),
    }


# ===========================================================================
# 4. DixitOptStop — Dixit & Pindyck (1994) real options / optimal stopping
# ===========================================================================
# Dixit & Pindyck (1994): under uncertainty, the option to WAIT has value.
# Rational traders delay entry until uncertainty resolves. The signal that
# uncertainty has resolved: volatility compresses sharply AFTER a directional
# price move. At this point, the option is exercised and participants enter.
#
# Signal:
#   vol_current  = rolling_std(ret, short_win)
#   vol_prior    = vol_current.shift(prior_offset)   <- volatility N bars ago
#   vol_drop     = (vol_prior - vol_current) / vol_prior
#   prior_ret    = (close - close[N]) / close[N]    <- N-bar net move
#
# vol_drop > thresh AND prior_ret > thresh -> long (uncertainty resolved up)
# vol_drop > thresh AND prior_ret < -thresh -> short (uncertainty resolved down)

def gen_DixitOptStop(df, short_vol_win=10, vol_prior_offset=8,
                      vol_drop_thresh=0.15, breakout_bars=5,
                      breakout_ret_thresh=0.003, trend_ema=45, exit_bar=8):
    c = df['close'].shift(1)

    ret         = c.pct_change().fillna(0)
    vol_current = ret.rolling(short_vol_win).std(ddof=0) + 1e-10
    vol_prior   = vol_current.shift(vol_prior_offset)
    vol_drop    = ((vol_prior - vol_current) / (vol_prior + 1e-10)).fillna(0.0)

    vol_compressed = vol_drop > vol_drop_thresh

    prior_ret = (c - c.shift(breakout_bars)) / (c.shift(breakout_bars) + 1e-10)

    entry_long  = vol_compressed & (prior_ret >  breakout_ret_thresh)
    entry_short = vol_compressed & (prior_ret < -breakout_ret_thresh)

    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.982)
    entry_short = entry_short & (c < trend * 1.018)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_DixitOptStop(trial):
    return {
        'short_vol_win':       trial.suggest_int('short_vol_win', 6, 18),
        'vol_prior_offset':    trial.suggest_int('vol_prior_offset', 5, 15),
        'vol_drop_thresh':     trial.suggest_float('vol_drop_thresh', 0.08, 0.35, step=0.05),
        'breakout_bars':       trial.suggest_int('breakout_bars', 3, 10),
        'breakout_ret_thresh': trial.suggest_float('breakout_ret_thresh', 0.001, 0.008, step=0.001),
        'trend_ema':           trial.suggest_int('trend_ema', 25, 65),
        'exit_bar':            trial.suggest_int('exit_bar', 4, 14),
    }


# ===========================================================================
# 5. LatencyArb_Fade — Budish, Cramton & Shim (2015) latency arb correction
# ===========================================================================
# Budish et al. (2015): HFTs continuously arbitrage stale cross-market quotes.
# At 5m/15m resolution, the residual is: narrow-range directional bars that
# counter the local trend. These represent HFT clearing of a latency-arb
# imbalance, NOT genuine new price information. After the arb clears, price
# reverts to the local trend direction.
#
# Signal:
#   narrow_bar    = HL / ATR < narrow_thresh      (tight intra-bar range)
#   bar_ret       = (close - open) / open         (net bar direction)
#   counter_trend = bar_ret opposes local EMA direction
#   entry: fade the counter-trend narrow bar back toward the local trend

def gen_LatencyArb_Fade(df, atr_period=14, narrow_thresh=0.55,
                         bar_ret_thresh=0.0005, local_ema=15,
                         trend_ema=40, vol_window=20, vol_z_min=-0.3,
                         exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    atr    = _atr(h, l, c, atr_period)
    hl     = (h - l).clip(lower=1e-12)
    narrow = (hl / (atr + 1e-12)) < narrow_thresh

    bar_ret   = (c - o) / (o + 1e-12)
    local     = _ema(c, local_ema)
    trend     = _ema(c, trend_ema)

    local_up   = c > local
    local_down = c < local

    # Narrow counter-trend bar: down bar but local trend up -> long fade
    entry_long  = narrow & (bar_ret < -bar_ret_thresh) & local_up
    # Narrow counter-trend bar: up bar but local trend down -> short fade
    entry_short = narrow & (bar_ret >  bar_ret_thresh) & local_down

    vol_z  = _zscore(v, vol_window)
    vol_ok = vol_z > vol_z_min
    entry_long  = entry_long  & vol_ok
    entry_short = entry_short & vol_ok

    entry_long  = entry_long  & (c > trend * 0.975)
    entry_short = entry_short & (c < trend * 1.025)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_LatencyArb_Fade(trial):
    return {
        'atr_period':      trial.suggest_int('atr_period', 8, 20),
        'narrow_thresh':   trial.suggest_float('narrow_thresh', 0.35, 0.75, step=0.05),
        'bar_ret_thresh':  trial.suggest_float('bar_ret_thresh', 0.0002, 0.0015, step=0.0001),
        'local_ema':       trial.suggest_int('local_ema', 8, 25),
        'trend_ema':       trial.suggest_int('trend_ema', 25, 60),
        'vol_window':      trial.suggest_int('vol_window', 12, 30),
        'vol_z_min':       trial.suggest_float('vol_z_min', -0.5, 0.5, step=0.25),
        'exit_bar':        trial.suggest_int('exit_bar', 3, 10),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'StackelbergSpread_Rev': {
        'gen':   gen_StackelbergSpread_Rev,
        'space': space_StackelbergSpread_Rev,
    },
    'BellmanMomentum': {
        'gen':   gen_BellmanMomentum,
        'space': space_BellmanMomentum,
    },
    'KyleAlpha_Dir': {
        'gen':   gen_KyleAlpha_Dir,
        'space': space_KyleAlpha_Dir,
    },
    'DixitOptStop': {
        'gen':   gen_DixitOptStop,
        'space': space_DixitOptStop,
    },
    'LatencyArb_Fade': {
        'gen':   gen_LatencyArb_Fade,
        'space': space_LatencyArb_Fade,
    },
}
