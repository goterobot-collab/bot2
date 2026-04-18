#!/usr/bin/env python3
"""
TV2 BATCH MICRO10 — 5 microstructure strategies (5m/15m optimized).
Domain: Adaptive Market Hypothesis (AMH) & Behavioral Finance Signals
(academic, distinct from MICRO1-9 + batches 3237-3318)

Strategies:
  1. AMH_AutoCorr_Rev     — Lo (2004) Adaptive Market Hypothesis: rolling
                            autocorrelation of returns detects mean-reversion
                            regime. Negative autocorr → fade directional moves.
  2. ProspectLoss_Fade    — Kahneman & Tversky (1979) loss aversion: traders
                            overreact to loss-bars (negative return + high vol)
                            creating predictable bounce.
  3. AttentionBuying_Rev  — Barber & Odean (2008) attention-driven buying:
                            abnormal UP bar + high volume = attention-driven
                            overbuy → fade the spike.
  4. RoundLevel_Break     — Tversky & Kahneman (1974) anchoring at round
                            numbers: price breaches ×N levels → momentum
                            continuation signal.
  5. CognDissonance_Rev   — Price-volume divergence: new price high + falling
                            volume = cognitive dissonance in buyers →
                            distribution phase → reversal.

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Lo, A.W. (2004) — "The Adaptive Markets Hypothesis"; J. of Portfolio
    Management 30(5): 15-29.
  - Kahneman, D. & Tversky, A. (1979) — "Prospect Theory: An Analysis of
    Decision under Risk"; Econometrica 47(2): 263-292.
  - Barber, B.M. & Odean, T. (2008) — "All That Glitters: The Effect of
    Attention and News on the Buying Behavior of Individual and Institutional
    Investors"; Review of Financial Studies 21(2): 785-818.
  - Tversky, A. & Kahneman, D. (1974) — "Judgment under Uncertainty:
    Heuristics and Biases"; Science 185(4157): 1124-1131.
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
    3-state machine: enters on signal, exits after exit_bar bars.
    Flat(0) → Long(1) on entry_long; Flat(0) → Short(-1) on entry_short.
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


# ═══════════════════════════════════════════════════════════════════════════
# 1. AMH_AutoCorr_Rev — Lo (2004) Adaptive Markets rolling autocorrelation
# ═══════════════════════════════════════════════════════════════════════════
# Lo (2004): Market efficiency is NOT constant — it adapts.
# The serial autocorrelation of returns cycles between positive (trending
# regime) and negative (mean-reverting regime) as conditions change.
# Key: when rolling autocorr is HIGHLY NEGATIVE → market is actively
# mean-reverting → fade directional price moves for high-probability bounce.
# When autocorr → 0 or positive → trending, avoid fading.
#
# Implementation:
#   autocorr_k = rolling corr( ret_t, ret_{t-1} )  over autocorr_window bars
#   Negative autocorr → mean-rev regime active
#   Entry: autocorr below threshold AND price made a move (Z-score extreme)
#   → enter opposite direction

def gen_AMH_AutoCorr_Rev(df, autocorr_window=25, autocorr_thresh=-0.15,
                         zscore_period=20, zscore_thresh=1.3,
                         trend_ema=40, exit_bar=7):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    ret = c.pct_change().fillna(0)
    ret_lag1 = ret.shift(1)

    # Rolling lag-1 autocorrelation proxy via rolling covariance / variance
    cov = (ret * ret_lag1).rolling(autocorr_window).mean() - \
          ret.rolling(autocorr_window).mean() * ret_lag1.rolling(autocorr_window).mean()
    var = ret.rolling(autocorr_window).var(ddof=0) + 1e-12
    autocorr = (cov / var).fillna(0.0)

    # Mean-reverting regime: autocorr < threshold (negative)
    mean_rev_regime = autocorr < autocorr_thresh

    # Price Z-score for entry timing
    pz = _zscore(c, zscore_period)

    # Fade moves in mean-reversion regime
    entry_long  = mean_rev_regime & (pz < -zscore_thresh)
    entry_short = mean_rev_regime & (pz >  zscore_thresh)

    # Trend filter: only fade if price is near its trend (not runaway)
    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.982)
    entry_short = entry_short & (c < trend * 1.018)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_AMH_AutoCorr_Rev(trial):
    return {
        'autocorr_window':  trial.suggest_int('autocorr_window', 15, 40),
        'autocorr_thresh':  trial.suggest_float('autocorr_thresh', -0.30, -0.05, step=0.05),
        'zscore_period':    trial.suggest_int('zscore_period', 12, 30),
        'zscore_thresh':    trial.suggest_float('zscore_thresh', 0.8, 2.2, step=0.2),
        'trend_ema':        trial.suggest_int('trend_ema', 25, 60),
        'exit_bar':         trial.suggest_int('exit_bar', 4, 14),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. ProspectLoss_Fade — Kahneman & Tversky (1979) loss-aversion overreaction
# ═══════════════════════════════════════════════════════════════════════════
# Prospect Theory (1979): losses are felt ~2.5× more intensely than equivalent
# gains. In markets: when a strong negative bar occurs (perceived loss),
# participants OVERREACT — selling beyond fundamental value.
# Signal: detect "loss-aversion spike" bars:
#   (a) Abnormally large negative return (loss_z < -threshold)
#   (b) Abnormally high volume (volume spike confirms panic selling)
#   (c) Price has NOT been in a sustained downtrend (trend filter)
# These conditions → overreaction → bounce long.
# Symmetric: overreaction to perceived "gain lock-in" (sell winners too early)
#   when a strongly positive bar occurs with volume → short fade.

def gen_ProspectLoss_Fade(df, ret_window=20, ret_z_thresh=1.6,
                          vol_window=20, vol_z_thresh=1.2,
                          trend_ema=35, exit_bar=6):
    c = df['close'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change().fillna(0)

    # Normalize return
    ret_z = _zscore(ret, ret_window)

    # Normalize volume
    vol_z = _zscore(v, vol_window)

    # Loss-aversion spike: strong negative return + high volume
    # → panic selling → overreaction → buy the bounce
    loss_spike = (ret_z < -ret_z_thresh) & (vol_z > vol_z_thresh)

    # Disposition sell: strong positive return + high volume
    # → "take profit" selling pressure → short fade
    gain_spike = (ret_z >  ret_z_thresh) & (vol_z > vol_z_thresh)

    # Trend filter: only fade if price not in runaway trend
    trend = _ema(c, trend_ema)
    loss_spike = loss_spike & (c > trend * 0.975)  # not in free-fall
    gain_spike = gain_spike & (c < trend * 1.025)  # not in blow-off

    return _apply_exit_bar(loss_spike, gain_spike, exit_bar)


def space_ProspectLoss_Fade(trial):
    return {
        'ret_window':     trial.suggest_int('ret_window', 12, 35),
        'ret_z_thresh':   trial.suggest_float('ret_z_thresh', 1.2, 2.5, step=0.2),
        'vol_window':     trial.suggest_int('vol_window', 12, 35),
        'vol_z_thresh':   trial.suggest_float('vol_z_thresh', 0.8, 2.0, step=0.2),
        'trend_ema':      trial.suggest_int('trend_ema', 20, 55),
        'exit_bar':       trial.suggest_int('exit_bar', 3, 12),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. AttentionBuying_Rev — Barber & Odean (2008) attention-driven overbuy
# ═══════════════════════════════════════════════════════════════════════════
# Barber & Odean (2008): retail investors are NET BUYERS of attention-
# grabbing assets (high abnormal volume, large price moves). This creates
# predictable OVERBUY situations: strong positive return + high volume
# signals that "attention buyers" have pushed price beyond fundamental value.
# Attention-buying = symmetric: abnormal high/low + unusual volume.
# Fade: after attention event, institutional/smart money sells into the
# retail buying wave → mean reversion follows.
#
# Signal:
#   Attention = abs(ret_z) > thresh AND vol_z > thresh → attention bar
#   Direction from ret_z sign
#   Fade: if attention buying (ret > 0 + high vol) → short
#         if attention selling (ret < 0 + high vol) → long

def gen_AttentionBuying_Rev(df, ret_window=25, ret_z_thresh=1.5,
                             vol_window=25, vol_z_thresh=1.4,
                             min_bars_since=3, trend_ema=40, exit_bar=8):
    c = df['close'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change().fillna(0)
    ret_z = _zscore(ret, ret_window)
    vol_z = _zscore(v, vol_window)

    # Attention event: extreme return + high volume
    attention_up   = (ret_z > ret_z_thresh)  & (vol_z > vol_z_thresh)
    attention_down = (ret_z < -ret_z_thresh) & (vol_z > vol_z_thresh)

    # min_bars_since: wait at least N bars after attention event before entry
    # (let the initial momentum exhaust)
    # Proxy: require attention event was set in rolling window but NOT last bar
    attention_up_window   = attention_up.shift(1).rolling(min_bars_since).max().fillna(0).astype(bool)
    attention_down_window = attention_down.shift(1).rolling(min_bars_since).max().fillna(0).astype(bool)

    # Exclude if another attention event just happened (still in spike)
    no_fresh_spike = ~(attention_up | attention_down)

    # Fade: attention-driven buy → short; attention-driven sell → long
    entry_short = attention_up_window   & no_fresh_spike
    entry_long  = attention_down_window & no_fresh_spike

    # Trend filter
    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.980)
    entry_short = entry_short & (c < trend * 1.020)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_AttentionBuying_Rev(trial):
    return {
        'ret_window':      trial.suggest_int('ret_window', 15, 40),
        'ret_z_thresh':    trial.suggest_float('ret_z_thresh', 1.0, 2.5, step=0.25),
        'vol_window':      trial.suggest_int('vol_window', 15, 40),
        'vol_z_thresh':    trial.suggest_float('vol_z_thresh', 0.8, 2.2, step=0.2),
        'min_bars_since':  trial.suggest_int('min_bars_since', 2, 6),
        'trend_ema':       trial.suggest_int('trend_ema', 25, 60),
        'exit_bar':        trial.suggest_int('exit_bar', 4, 14),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. RoundLevel_Break — Tversky & Kahneman (1974) anchoring at round numbers
# ═══════════════════════════════════════════════════════════════════════════
# Anchoring heuristic (1974): traders anchor to psychologically salient
# price levels — round numbers (×100, ×1000, ×10). These act as:
#   (a) Support/resistance (price stalls near them)
#   (b) Breakout triggers (clean break → momentum continuation)
# In crypto: BTC×1000, ETH×100, altcoins×10, micro-caps×1 all function as
# anchor levels. A confirmed break (close > round level × N bars) triggers
# momentum: anchored sellers/buyers are now squeezed.
#
# Signal:
#   round_level = round(price / round_unit) × round_unit
#   If price was BELOW round level for lookback bars AND now CLOSES ABOVE
#   → breakout momentum LONG (squeezed sellers)
#   Vice versa for SHORT (breakdown)

def gen_RoundLevel_Break(df, round_unit_factor=0.01, lookback=12,
                         confirm_bars=2, breakout_buffer=0.002,
                         trend_ema=30, exit_bar=8):
    c = df['close'].shift(1)

    # Dynamic round unit: fraction of median price (avoids hardcoded USD levels)
    # round_unit ~ round_unit_factor × median_price
    # e.g., factor=0.01 → for price=100, unit=1; price=10000, unit=100
    med_price = c.rolling(100).median() + 1e-8
    round_unit = (med_price * round_unit_factor).apply(lambda x: max(x, 1e-6))

    # Nearest round level
    round_level = (c / round_unit).round() * round_unit

    # Proximity to round level: how close is price to the nearest round?
    # (used to confirm we were near it before breaking)
    dist_to_round = (c - round_level).abs() / (round_unit + 1e-8)

    # Was price BELOW round level consistently (sellers dominated near it)?
    below_round = c < round_level
    above_round = c > round_level

    # Lookback: price was below round for most of the window
    below_fraction = below_round.rolling(lookback).mean().fillna(0)
    above_fraction = above_round.rolling(lookback).mean().fillna(0)

    # Now we're ABOVE and confirm with distance buffer
    now_above = c > round_level * (1 + breakout_buffer)
    now_below = c < round_level * (1 - breakout_buffer)

    # Breakout: was mostly below, now confirmed above → LONG momentum
    entry_long  = (below_fraction > 0.65) & now_above
    entry_short = (above_fraction > 0.65) & now_below

    # Confirm signal must persist for N bars
    entry_long  = entry_long.rolling(confirm_bars).sum() == confirm_bars
    entry_short = entry_short.rolling(confirm_bars).sum() == confirm_bars

    # Trend filter: momentum only if not violently extended
    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c < trend * 1.040)
    entry_short = entry_short & (c > trend * 0.960)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_RoundLevel_Break(trial):
    return {
        'round_unit_factor': trial.suggest_float('round_unit_factor', 0.005, 0.02, step=0.005),
        'lookback':          trial.suggest_int('lookback', 8, 25),
        'confirm_bars':      trial.suggest_int('confirm_bars', 1, 4),
        'breakout_buffer':   trial.suggest_float('breakout_buffer', 0.001, 0.005, step=0.001),
        'trend_ema':         trial.suggest_int('trend_ema', 20, 50),
        'exit_bar':          trial.suggest_int('exit_bar', 5, 15),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. CognDissonance_Rev — Price-volume divergence (distribution detection)
# ═══════════════════════════════════════════════════════════════════════════
# Cognitive dissonance in markets: when price makes new highs (N-bar high)
# but volume is declining (vol_z < 0), smart money is DISTRIBUTING to
# retail buyers still buying on momentum. Buyers are in cognitive dissonance:
# they see the price going up but the underlying conviction (volume) is fading.
# This is a classic distribution pattern → reversal signal.
# Symmetric: price new low + falling volume = capitulation exhaustion → bounce.
#
# Signal:
#   price_high_N: close = max over lookback bars → new high made
#   vol_declining: volume trend (EMA slope < 0) over the high
#   Divergence: price pushes high BUT volume EMA falling → SHORT
#   price_low_N + vol declining → LONG (exhaustion bounce)

def gen_CognDissonance_Rev(df, lookback=15, vol_ema_span=10,
                            vol_slope_window=8, vol_slope_thresh=-0.02,
                            zscore_period=20, pz_thresh=1.0,
                            trend_ema=35, exit_bar=7):
    c = df['close'].shift(1)
    v = df['volume'].shift(1)

    # Price at N-bar high or low
    roll_high = c.rolling(lookback).max()
    roll_low  = c.rolling(lookback).min()

    at_high = c >= roll_high * 0.998   # within 0.2% of N-bar high
    at_low  = c <= roll_low  * 1.002   # within 0.2% of N-bar low

    # Volume EMA slope (is volume trend rising or falling?)
    vol_ema = _ema(v, vol_ema_span)
    vol_ema_lag = vol_ema.shift(vol_slope_window)
    vol_slope = (vol_ema - vol_ema_lag) / (vol_ema_lag + 1e-8)

    # Declining volume trend
    vol_declining = vol_slope < vol_slope_thresh

    # Price Z-score for confirming extreme
    pz = _zscore(c, zscore_period)

    # Distribution: price at high + volume falling → SHORT
    entry_short = at_high & vol_declining & (pz > pz_thresh)

    # Exhaustion: price at low + volume falling → LONG (sellers exhausted)
    entry_long  = at_low  & vol_declining & (pz < -pz_thresh)

    # Trend filter
    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.978)
    entry_short = entry_short & (c < trend * 1.022)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_CognDissonance_Rev(trial):
    return {
        'lookback':           trial.suggest_int('lookback', 8, 25),
        'vol_ema_span':       trial.suggest_int('vol_ema_span', 5, 18),
        'vol_slope_window':   trial.suggest_int('vol_slope_window', 4, 15),
        'vol_slope_thresh':   trial.suggest_float('vol_slope_thresh', -0.05, -0.01, step=0.01),
        'zscore_period':      trial.suggest_int('zscore_period', 12, 30),
        'pz_thresh':          trial.suggest_float('pz_thresh', 0.6, 1.8, step=0.2),
        'trend_ema':          trial.suggest_int('trend_ema', 20, 55),
        'exit_bar':           trial.suggest_int('exit_bar', 4, 14),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'AMH_AutoCorr_Rev': {
        'gen':   gen_AMH_AutoCorr_Rev,
        'space': space_AMH_AutoCorr_Rev,
    },
    'ProspectLoss_Fade': {
        'gen':   gen_ProspectLoss_Fade,
        'space': space_ProspectLoss_Fade,
    },
    'AttentionBuying_Rev': {
        'gen':   gen_AttentionBuying_Rev,
        'space': space_AttentionBuying_Rev,
    },
    'RoundLevel_Break': {
        'gen':   gen_RoundLevel_Break,
        'space': space_RoundLevel_Break,
    },
    'CognDissonance_Rev': {
        'gen':   gen_CognDissonance_Rev,
        'space': space_CognDissonance_Rev,
    },
}
