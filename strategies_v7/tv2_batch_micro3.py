#!/usr/bin/env python3
"""
TV2 BATCH MICRO3 — 5 microstructure strategies (5m/15m optimized).
Domain: Game Theory, RL-Inspired Signals & Latency Arbitrage
(academic, distinct from MICRO1/MICRO2 + batches 3237-3318)

Strategies:
  1. NashEquil_Reversal   — Game-theory coordination failure: both buyers+sellers
                            agree on direction → exhaustion reversal (Nash trap)
  2. QLearning_Threshold  — RL-inspired adaptive state: reward signal from
                            recent trade outcomes drives dynamic entry threshold
  3. InfoLeakage_Signal   — Information asymmetry proxy: open-to-high/low gaps
                            that persist suggest informed trading → fade at exhaustion
  4. LatencyArb_Spread    — Microstructure latency: slow-update spread proxy
                            using EMA lag vs realized price → lead/lag entry
  5. MarketMaker_Fade     — Market maker inventory rebalancing: detect MM
                            one-sided accumulation via spread skew, fade the move

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references embedded per strategy.
"""

import pandas as pd
import numpy as np


# ─── HELPERS ────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _sma(s, p):
    return s.rolling(p).mean()

def _atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / p, adjust=False).mean()

def _zscore(s, p):
    m = s.rolling(p).mean()
    sd = s.rolling(p).std(ddof=0) + 1e-10
    return (s - m) / sd

def _apply_exit_bar(entry_long: pd.Series, entry_short: pd.Series, exit_bar: int) -> pd.Series:
    """
    3-state machine: enters on entry signal, exits after exit_bar bars.
    Flat (0) → Long (1) on entry_long; Flat (0) → Short (-1) on entry_short.
    New entries can override an opposite position immediately.
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


# ═══════════════════════════════════════════════════════════════════════════
# 1. NashEquil_Reversal — Game theory coordination failure / Nash trap
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Nash (1951) equilibrium; Rochet & Tirole (2003) coordination games;
#   Brunnermeier (2001) asset pricing under asymmetric info.
# Logic: When BOTH buyers and sellers signal the same direction simultaneously
#   (open > close AND high-low range is small = both trapped), market is at a
#   Nash trap: no dominant strategy → price reverts to equilibrium.
# Proxy:
#   - "buyer pressure" = (close - open) / ATR  (positive = buyers won)
#   - "seller pressure" = (high - close) / ATR  (high wick = sellers defended)
#   - When buyer_pressure > thresh AND seller_pressure > thresh simultaneously
#     → both forces exhausted → Nash trap → fade.

def gen_NashEquil_Reversal(df, atr_period=14, buyer_thresh=0.3, seller_thresh=0.25,
                             zscore_period=20, zscore_thresh=1.5, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    atr = _atr(h, l, c, atr_period)
    atr_safe = atr.replace(0, np.nan).ffill()

    # Buyer pressure: close > open normalized by ATR
    buyer_press = (c - o) / atr_safe
    # Seller pressure: upper wick = sellers defended the high
    seller_press = (h - c) / atr_safe
    # Lower wick pressure (buyers defended the low)
    lower_press = (c - l) / atr_safe

    # Nash trap LONG: sellers clearly dominated (c < o) AND upper wick is small
    # (sellers already pushed hard, lower wick shows buyers came in → revert up)
    # buyer_press < -buyer_thresh (bearish bar) AND lower_press > seller_thresh (buyers defended)
    nash_long_cond = (buyer_press < -buyer_thresh) & (lower_press > seller_thresh)

    # Nash trap SHORT: buyers clearly dominated (c > o) AND lower wick small
    # (buyers already pushed hard, upper wick shows sellers came in → revert down)
    nash_short_cond = (buyer_press > buyer_thresh) & (seller_press > seller_thresh)

    # Z-score filter on price to avoid mean-rev in trending market
    price_z = _zscore(c, zscore_period)

    # LONG: Nash trap bearish bar with price Z near oversold
    entry_long  = nash_long_cond  & (price_z < -zscore_thresh)
    # SHORT: Nash trap bullish bar with price Z near overbought
    entry_short = nash_short_cond & (price_z >  zscore_thresh)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_NashEquil_Reversal(trial):
    return {
        'atr_period':    trial.suggest_int('atr_period', 8, 21),
        'buyer_thresh':  trial.suggest_float('buyer_thresh', 0.15, 0.60, step=0.05),
        'seller_thresh': trial.suggest_float('seller_thresh', 0.10, 0.50, step=0.05),
        'zscore_period': trial.suggest_int('zscore_period', 14, 40),
        'zscore_thresh': trial.suggest_float('zscore_thresh', 1.0, 2.5, step=0.25),
        'exit_bar':      trial.suggest_int('exit_bar', 3, 12),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. QLearning_Threshold — RL-inspired adaptive entry threshold
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Watkins & Dayan (1992) Q-learning; Moody & Saffell (2001)
#   "Learning to trade via direct reinforcement", Journal of Neural Networks.
# Logic: Maintain a "Q-value" approximation as a decaying EMA of recent
#   per-bar rewards (signed returns). When Q-signal is at an extreme
#   (large negative Q = regime of recent losses = mean-rev opportunity),
#   enter against the recent trend (RL "exploration" at extreme Q-values).
# Note: This is a signal-only RL proxy (no live Q-table), optimized by Optuna.

def gen_QLearning_Threshold(df, q_fast=5, q_slow=20, reward_period=10,
                              q_thresh=1.5, confirm_vol_ratio=1.2, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # "Reward" per bar: bar return (signed)
    bar_ret = (c - o) / (o.replace(0, np.nan).ffill())

    # Fast Q-signal: EMA of recent rewards (exploitation estimate)
    q_fast_val = _ema(bar_ret, q_fast)
    # Slow Q-signal: longer-term baseline
    q_slow_val = _ema(bar_ret, q_slow)

    # Q-divergence: fast Q deviates from slow Q → RL non-stationarity detected
    q_div = q_fast_val - q_slow_val

    # Z-score of Q-divergence
    q_z = _zscore(q_div, reward_period)

    # Volume confirmation: only act when volume is elevated (information event)
    avg_vol = _sma(v, reward_period)
    vol_elevated = v > avg_vol * confirm_vol_ratio

    # Entry: Q-signal at extreme → RL mean-reversion (fade the recent regime)
    # q_z << 0 means recent fast-rewards crashed → buyers exhausted → long
    entry_long  = (q_z < -q_thresh) & vol_elevated
    entry_short = (q_z >  q_thresh) & vol_elevated

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_QLearning_Threshold(trial):
    return {
        'q_fast':            trial.suggest_int('q_fast', 3, 10),
        'q_slow':            trial.suggest_int('q_slow', 12, 35),
        'reward_period':     trial.suggest_int('reward_period', 8, 25),
        'q_thresh':          trial.suggest_float('q_thresh', 1.0, 2.5, step=0.25),
        'confirm_vol_ratio': trial.suggest_float('confirm_vol_ratio', 1.0, 2.0, step=0.1),
        'exit_bar':          trial.suggest_int('exit_bar', 3, 12),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. InfoLeakage_Signal — Information asymmetry / informed trading proxy
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Kyle (1985) insider trading model; Glosten & Milgrom (1985)
#   bid-ask spread model; Easley & O'Hara (1987) information and the bid-ask spread.
# Logic: Informed traders tend to push price directionally during the bar
#   (open-to-close spread is large relative to bar range = directional intent).
#   When this "information ratio" is sustained for N bars in one direction,
#   it signals informed accumulation/distribution. Fade at exhaustion.
# Proxy: info_ratio = |close - open| / (high - low + 1e-8)
#   High info_ratio + directional → informed push → potential exhaustion reversal.

def gen_InfoLeakage_Signal(df, atr_period=14, info_ratio_thresh=0.65,
                            persist_bars=3, zscore_period=20, zscore_thresh=1.5,
                            exit_bar=7):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    bar_range = (h - l).replace(0, np.nan).ffill()
    bar_body  = (c - o)

    # Information ratio: how directional was each bar (0=doji, 1=full marubozu)
    info_ratio = bar_body.abs() / bar_range

    # High information ratio above threshold
    high_info = info_ratio > info_ratio_thresh

    # Directional: bar_body sign
    bullish_bar = bar_body > 0
    bearish_bar = bar_body < 0

    # Sustained informed push: N consecutive high-info bars in same direction
    # Rolling sum of (high_info & direction) over persist_bars window
    bull_persist = (high_info & bullish_bar).rolling(persist_bars).sum()
    bear_persist = (high_info & bearish_bar).rolling(persist_bars).sum()

    informed_bull = bull_persist >= persist_bars  # N bars of informed buying
    informed_bear = bear_persist >= persist_bars  # N bars of informed selling

    # Z-score of price for amplitude context
    price_z = _zscore(c, zscore_period)

    # Volume confirmation: informed trading has above-avg volume
    avg_vol = _sma(v, atr_period)
    vol_confirm = v > avg_vol

    # Entry: informed push exhaustion → fade
    # Sustained buying + overbought Z → sell (informed buyers done)
    entry_short = informed_bull & (price_z >  zscore_thresh) & vol_confirm
    # Sustained selling + oversold Z → buy (informed sellers done)
    entry_long  = informed_bear & (price_z < -zscore_thresh) & vol_confirm

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_InfoLeakage_Signal(trial):
    return {
        'atr_period':        trial.suggest_int('atr_period', 8, 21),
        'info_ratio_thresh': trial.suggest_float('info_ratio_thresh', 0.45, 0.85, step=0.05),
        'persist_bars':      trial.suggest_int('persist_bars', 2, 5),
        'zscore_period':     trial.suggest_int('zscore_period', 14, 40),
        'zscore_thresh':     trial.suggest_float('zscore_thresh', 1.0, 2.5, step=0.25),
        'exit_bar':          trial.suggest_int('exit_bar', 3, 12),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. LatencyArb_Spread — Latency arbitrage via EMA lag divergence
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Biais, Foucault & Moinas (2015) "Equilibrium fast trading",
#   J. Financial Economics; Budish, Cramton & Shim (2015) "The high-frequency
#   trading arms race", QJE.
# Logic: In real markets, slow participants (lagged EMAs) and fast participants
#   (price itself) create temporary divergence during news events or order flow
#   imbalances. This spread between fast price and slow EMA widens → then closes.
# Proxy:
#   "latency spread" = (current_price - slow_EMA) normalized by ATR
#   When spread is extreme in one direction → slow participants haven't caught up
#   → price will revert to the EMA (the "true price" slow participants see).

def gen_LatencyArb_Spread(df, fast_ema=3, slow_ema=21, atr_period=14,
                           spread_thresh=1.5, vol_ma=15, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    ema_fast = _ema(c, fast_ema)
    ema_slow = _ema(c, slow_ema)

    # Latency spread: fast EMA (rapid price tracker) vs slow EMA (lagged participants)
    lat_spread = ema_fast - ema_slow

    atr = _atr(h, l, c, atr_period)
    atr_safe = atr.replace(0, np.nan).ffill()

    # Normalize spread by ATR → units of ATR
    spread_atr = lat_spread / atr_safe

    # Z-score of the spread (how extreme is the current divergence historically)
    spread_z = _zscore(spread_atr, slow_ema)

    # Volume filter: latency arb events are associated with volume spikes
    avg_vol = _sma(v, vol_ma)
    vol_spike = v > avg_vol * 1.5

    # Entry: extreme spread → price will close back to slow EMA (reversion)
    # spread_z >> 0: fast price >> slow EMA → overbought relative to "true" price
    entry_short = (spread_z >  spread_thresh) & vol_spike
    # spread_z << 0: fast price << slow EMA → oversold relative to "true" price
    entry_long  = (spread_z < -spread_thresh) & vol_spike

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_LatencyArb_Spread(trial):
    return {
        'fast_ema':      trial.suggest_int('fast_ema', 2, 8),
        'slow_ema':      trial.suggest_int('slow_ema', 15, 40),
        'atr_period':    trial.suggest_int('atr_period', 8, 21),
        'spread_thresh': trial.suggest_float('spread_thresh', 1.0, 2.5, step=0.25),
        'vol_ma':        trial.suggest_int('vol_ma', 8, 25),
        'exit_bar':      trial.suggest_int('exit_bar', 3, 10),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. MarketMaker_Fade — Market maker inventory rebalancing detection
# ═══════════════════════════════════════════════════════════════════════════
# Academic ref: Ho & Stoll (1981) optimal dealer pricing; Amihud & Mendelson (1980)
#   dealership markets; Madhavan & Smidt (1993) inventory and adverse selection.
# Logic: Market makers must rebalance their inventory after one-sided order flow.
#   After a directional move WITH volume (MM had to accumulate inventory),
#   they will lean against the market to offload → price fades.
# Proxy:
#   mm_inventory_signal = rolling sum of (bar_return * relative_volume, N)
#   → cumulative weighted return = proxy for MM inventory imbalance
#   When this saturates (Z-score extreme) + volume returns to average
#   (MM no longer absorbing) → fade.

def gen_MarketMaker_Fade(df, inv_period=12, vol_period=20, inv_thresh=1.8,
                          vol_return_thresh=0.8, trend_ema=30, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Bar return
    bar_ret = (c - c.shift(1)) / (c.shift(1).replace(0, np.nan).ffill())

    # Relative volume
    avg_vol = _sma(v, vol_period)
    rel_vol = v / (avg_vol + 1e-10)

    # MM inventory proxy: accumulated directional volume-weighted return
    mm_inv = (bar_ret * rel_vol).rolling(inv_period).sum()

    # Z-score of inventory signal
    inv_z = _zscore(mm_inv, vol_period)

    # Volume "returned to normal": after a spike, MM has finished absorbing
    vol_normalized = rel_vol < vol_return_thresh  # volume has come back down

    # Trend filter: fade only when price is at extreme (trending too far)
    trend = _ema(c, trend_ema)
    above_trend = c > trend
    below_trend = c < trend

    # Entry: extreme MM inventory + volume normalized (MM done absorbing) → fade
    # inv_z >> 0: MM accumulated long inventory → they will sell → short entry
    entry_short = (inv_z >  inv_thresh) & vol_normalized & above_trend
    # inv_z << 0: MM accumulated short inventory → they will buy → long entry
    entry_long  = (inv_z < -inv_thresh) & vol_normalized & below_trend

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_MarketMaker_Fade(trial):
    return {
        'inv_period':         trial.suggest_int('inv_period', 8, 25),
        'vol_period':         trial.suggest_int('vol_period', 12, 35),
        'inv_thresh':         trial.suggest_float('inv_thresh', 1.2, 3.0, step=0.2),
        'vol_return_thresh':  trial.suggest_float('vol_return_thresh', 0.6, 1.0, step=0.1),
        'trend_ema':          trial.suggest_int('trend_ema', 20, 60),
        'exit_bar':           trial.suggest_int('exit_bar', 3, 12),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'NashEquil_Reversal': {
        'gen':   gen_NashEquil_Reversal,
        'space': space_NashEquil_Reversal,
    },
    'QLearning_Threshold': {
        'gen':   gen_QLearning_Threshold,
        'space': space_QLearning_Threshold,
    },
    'InfoLeakage_Signal': {
        'gen':   gen_InfoLeakage_Signal,
        'space': space_InfoLeakage_Signal,
    },
    'LatencyArb_Spread': {
        'gen':   gen_LatencyArb_Spread,
        'space': space_LatencyArb_Spread,
    },
    'MarketMaker_Fade': {
        'gen':   gen_MarketMaker_Fade,
        'space': space_MarketMaker_Fade,
    },
}
