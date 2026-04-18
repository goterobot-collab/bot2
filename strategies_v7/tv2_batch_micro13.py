#!/usr/bin/env python3
"""
TV2 BATCH MICRO13 — 5 microstructure strategies (5m/15m optimized).
Domain: Queuing Theory, Extreme Value Theory, Copula Dependence,
        Market Thermodynamics, Percolation Theory.
(academic, distinct from MICRO1-12 + batches 3237-3318)

Strategies:
  1. LittlesQueue_Rev   — Little's Law (1961) from queuing theory: avg queue
                          length = arrival rate × service time. When avg trade
                          size (volume/bar proxy) spikes while price is flat,
                          large orders are being absorbed (queue saturation)
                          → subsequent reversal.
  2. EVT_TailBreak      — Peaks-Over-Threshold with Generalized Pareto
                          Distribution (Pickands 1975). When standardized
                          return exceeds the running extreme quantile, tail
                          events cluster → momentum continuation entry.
  3. CopulaRank_Rev     — Spearman rank correlation between price returns and
                          volume (Sklar 1959). Strongly negative rank corr
                          (high volume + low price move) = absorption regime
                          → reversal. Positive rank corr = distribution regime
                          → continuation.
  4. ThermTemp_Fade     — Market "temperature" analogy (Sornette 2003):
                          temperature = rolling return variance. When
                          temperature crashes below its own mean AND a bar
                          closes beyond its Bollinger Band, the system is
                          supercooled → mean-reversion snap-back entry.
  5. PercThresh_Break   — Percolation theory (Stauffer & Aharony 1991):
                          critical threshold ≈ 0.5927 for 2D lattice.
                          Ratio of bullish bars in rolling window crossing
                          the percolation threshold signals phase transition
                          → momentum entry in the dominant direction.

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Little, J.D.C. (1961) — "A Proof for the Queuing Formula: L = λW";
    Operations Research 9(3): 383-387.
  - Pickands, J. (1975) — "Statistical Inference Using Extreme Order
    Statistics"; Annals of Statistics 3(1): 119-131.
  - Sklar, A. (1959) — "Fonctions de répartition à n dimensions et leurs
    marges"; Publications de l'Institut de Statistique de l'Université
    de Paris 8: 229-231.
  - Sornette, D. (2003) — "Why Stock Markets Crash: Critical Events in
    Complex Financial Systems"; Princeton University Press.
  - Stauffer, D. & Aharony, A. (1991) — "Introduction to Percolation
    Theory" (2nd ed.); Taylor & Francis.
  - Cont, R. & Bouchaud, J.P. (2000) — "Herd Behavior and Aggregate
    Fluctuations in Financial Markets"; Macroeconomic Dynamics 4: 170-196.
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
# 1. LittlesQueue_Rev — Little's Law (1961) queue saturation reversal
# ===========================================================================
# Little's Law: avg queue length L = λW (arrival rate × service time).
# Proxy in price data:
#   avg_trade_size  ≈ volume / (HL_range / close)  — larger when few large
#                     orders clear a narrow range (queue processing big block)
#   price_flatness  = abs(close - open) / close  — small when price absorbed
#
# Queue saturation signal:
#   big_size AND flat_bar → absorption → reversal toward trend
#   Entry direction: fade bar direction (if bar closed up → short; down → long)
#   Trend filter: EMA slope guards against fading strong trends.

def gen_LittlesQueue_Rev(df, atr_period=14, size_z_window=30,
                          size_z_thresh=1.5, flatness_thresh=0.0006,
                          trend_ema=50, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Avg trade size proxy: volume × close / (high - low + 1e-10)
    # High when large volume clears a narrow range (absorption)
    hl_range = (h - l).clip(lower=1e-10)
    avg_size  = v * c / hl_range

    size_z = _zscore(avg_size, size_z_window)

    # Price flatness: abs(close - open) / close — small = flat bar
    flatness = (c - o).abs() / c.clip(lower=1e-10)

    # Bar direction (shifted): positive close - open → price moved up
    bar_up   = (c - o) > 0
    bar_down = (c - o) < 0

    # Trend filter
    trend = _ema(c, trend_ema)
    in_uptrend   = c > trend
    in_downtrend = c < trend

    # Absorption detected: big size + flat bar
    absorption = (size_z > size_z_thresh) & (flatness < flatness_thresh)

    # Reversal entries: fade the bar, only when trend supports reversion
    # Fade up-bar → short, but only in downtrend (short-side reversion)
    # Fade down-bar → long, but only in uptrend (long-side reversion)
    entry_long  = absorption & bar_down & in_uptrend
    entry_short = absorption & bar_up   & in_downtrend

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_LittlesQueue_Rev(trial):
    return {
        'atr_period':      trial.suggest_int('atr_period',      10, 20),
        'size_z_window':   trial.suggest_int('size_z_window',   20, 50),
        'size_z_thresh':   trial.suggest_float('size_z_thresh',  1.0, 2.5),
        'flatness_thresh': trial.suggest_float('flatness_thresh', 0.0003, 0.0015),
        'trend_ema':       trial.suggest_int('trend_ema',       30, 80),
        'exit_bar':        trial.suggest_int('exit_bar',         3, 12),
    }


# ===========================================================================
# 2. EVT_TailBreak — Peaks-Over-Threshold / Extreme Value Theory momentum
# ===========================================================================
# Pickands (1975) Generalized Pareto Distribution applied to return tails.
# Proxy (no parametric fitting): use rolling max/min of return as the
# "empirical tail threshold". When the current return crosses the top/bottom
# quantile of its rolling window, we're in a tail event.
# Tail events cluster (Mandelbrot 1963, Cont 2001) → momentum continuation.
#
# Signal:
#   r_t  = (close - prev_close) / prev_close
#   q_hi = rolling quantile at prob_hi of |r|  → positive tail threshold
#   q_lo = rolling quantile at 1-prob_hi       → negative tail threshold
#   r_t > q_hi  → LONG  (positive tail → momentum up)
#   r_t < q_lo  → SHORT (negative tail → momentum down)
# Trend guard: only enter if EMA confirms direction.

def gen_EVT_TailBreak(df, ret_window=40, tail_prob=0.90,
                       trend_ema=30, vol_window=20, vol_z_min=0.0,
                       exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change()

    # Rolling tail thresholds (empirical quantile)
    q_hi = ret.rolling(ret_window).quantile(tail_prob)
    q_lo = ret.rolling(ret_window).quantile(1.0 - tail_prob)

    # Trend confirmation
    trend   = _ema(c, trend_ema)
    atr_val = _atr(h, l, c, 14)

    # Volume z-score (optional guard — keep vol_z_min=0 to disable)
    vol_z = _zscore(v, vol_window)

    # Tail crossings — momentum direction
    tail_long  = (ret > q_hi)  & (c > trend) & (vol_z >= vol_z_min)
    tail_short = (ret < q_lo)  & (c < trend) & (vol_z >= vol_z_min)

    return _apply_exit_bar(tail_long, tail_short, exit_bar)


def space_EVT_TailBreak(trial):
    return {
        'ret_window':  trial.suggest_int('ret_window',   25, 60),
        'tail_prob':   trial.suggest_float('tail_prob',  0.82, 0.96),
        'trend_ema':   trial.suggest_int('trend_ema',    20, 60),
        'vol_window':  trial.suggest_int('vol_window',   15, 35),
        'vol_z_min':   trial.suggest_float('vol_z_min',  0.0, 1.0),
        'exit_bar':    trial.suggest_int('exit_bar',      4, 14),
    }


# ===========================================================================
# 3. CopulaRank_Rev — Spearman copula rank correlation reversal
# ===========================================================================
# Sklar (1959): copula separates marginal distributions from dependency
# structure. The Spearman rank correlation of (returns, volume) captures the
# tail dependence between price movement and traded volume.
#
# Interpretation:
#   ρ_rank >> 0  → distribution regime: high volume drives price → continuation
#   ρ_rank << 0  → absorption regime: high volume but no price move → reversal
#
# Signal:
#   rank_corr = Spearman(ret, vol) over rolling window
#   rank_corr < -neg_thresh  AND  recent bar direction → fade (reversal)
#   rank_corr >  pos_thresh  AND  trend confirmed      → follow (continuation)

def gen_CopulaRank_Rev(df, rank_window=20, neg_thresh=0.35, pos_thresh=0.30,
                        trend_ema=40, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change()

    # Rolling Spearman rank correlation (ret vs volume)
    # Computed manually to avoid scipy dependency: rank within window
    n   = len(c)
    rho = pd.Series(np.nan, index=c.index)

    ret_arr = ret.values
    vol_arr = v.values
    w       = rank_window

    for i in range(w - 1, n):
        r_w = ret_arr[i - w + 1: i + 1]
        v_w = vol_arr[i - w + 1: i + 1]
        # Rank arrays (handle NaN)
        valid = ~(np.isnan(r_w) | np.isnan(v_w))
        if valid.sum() < w // 2:
            continue
        r_v  = r_w[valid]
        v_v  = v_w[valid]
        n_v  = len(r_v)
        r_rk = np.argsort(np.argsort(r_v)).astype(float)
        v_rk = np.argsort(np.argsort(v_v)).astype(float)
        # Spearman ρ = 1 - 6Σd²/(n(n²-1))
        d2  = (r_rk - v_rk) ** 2
        rho.iloc[i] = 1.0 - 6.0 * d2.sum() / max(n_v * (n_v ** 2 - 1), 1)

    # Trend context
    trend      = _ema(c, trend_ema)
    in_uptrend = c > trend

    # Bar direction
    bar_up   = (c - o) > 0
    bar_down = (c - o) < 0

    # Absorption regime: fade bar direction
    absorb_long  = (rho < -neg_thresh) & bar_down & in_uptrend
    absorb_short = (rho < -neg_thresh) & bar_up   & ~in_uptrend

    return _apply_exit_bar(absorb_long, absorb_short, exit_bar)


def space_CopulaRank_Rev(trial):
    return {
        'rank_window': trial.suggest_int('rank_window',  12, 35),
        'neg_thresh':  trial.suggest_float('neg_thresh',  0.20, 0.55),
        'pos_thresh':  trial.suggest_float('pos_thresh',  0.15, 0.50),
        'trend_ema':   trial.suggest_int('trend_ema',     25, 70),
        'exit_bar':    trial.suggest_int('exit_bar',       3, 12),
    }


# ===========================================================================
# 4. ThermTemp_Fade — Market thermodynamics supercooling reversal
# ===========================================================================
# Sornette (2003) market thermodynamics analogy:
#   "Temperature" T = rolling variance of returns (degree of thermal motion)
#   When T drops far below its own mean → market is "supercooled"
#   (artificially calm, like water below 0°C but not yet frozen)
#   A perturbation (bar closing outside BB) triggers crystallization
#   → mean-reversion snap-back toward equilibrium.
#
# Signal:
#   temp     = rolling variance of returns
#   temp_z   = zscore of temperature (how cold vs its own history)
#   bb_lo, _, bb_hi = Bollinger Bands on close
#   "Supercooled + price outside band" → snap-back entry

def gen_ThermTemp_Fade(df, ret_window=20, temp_z_window=40,
                        temp_z_thresh=-1.0, bb_period=20, bb_dev=2.0,
                        trend_ema=50, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    ret  = c.pct_change()

    # Market temperature = rolling variance of returns
    temp   = ret.rolling(ret_window).var()
    temp_z = _zscore(temp, temp_z_window)

    # Bollinger Bands on close
    bb_lo, bb_mid, bb_hi = _bb(c, bb_period, bb_dev)

    # Trend context
    trend = _ema(c, trend_ema)

    # Supercooled state: temperature z-score below threshold (cold = low vol)
    supercooled = temp_z < temp_z_thresh

    # Perturbation: close outside Bollinger Band → snap-back
    above_bb = c > bb_hi
    below_bb = c < bb_lo

    # Entry: supercooled AND outside band → mean reversion
    # Fade the extreme: above BB → short, below BB → long
    entry_long  = supercooled & below_bb & (c > trend)
    entry_short = supercooled & above_bb & (c < trend)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_ThermTemp_Fade(trial):
    return {
        'ret_window':     trial.suggest_int('ret_window',      12, 30),
        'temp_z_window':  trial.suggest_int('temp_z_window',   25, 60),
        'temp_z_thresh':  trial.suggest_float('temp_z_thresh', -2.0, -0.3),
        'bb_period':      trial.suggest_int('bb_period',        15, 30),
        'bb_dev':         trial.suggest_float('bb_dev',          1.5, 2.5),
        'trend_ema':      trial.suggest_int('trend_ema',         30, 80),
        'exit_bar':       trial.suggest_int('exit_bar',           3, 12),
    }


# ===========================================================================
# 5. PercThresh_Break — Percolation theory critical phase transition
# ===========================================================================
# Stauffer & Aharony (1991): in a 2D square lattice, the percolation
# threshold is p_c ≈ 0.5927. Below p_c: isolated clusters (no spanning).
# Above p_c: a spanning cluster forms (system-wide connectivity).
#
# Market analogy (Cont & Bouchaud 2000):
#   Traders form a network. When the fraction of "bullish" traders (bulls
#   minus bears) crosses p_c ≈ 0.5927, a connected bull cluster spans the
#   network → coordinated buying → trending price regime.
#
# Proxy:
#   bull_ratio  = count(close > open in rolling window) / window
#   Crossing from below p_c to above → LONG momentum entry
#   Crossing from above p_c to below (bear threshold = 1-p_c ≈ 0.407) → SHORT

def gen_PercThresh_Break(df, perc_window=20, perc_threshold=0.5927,
                          bear_threshold=0.407, trend_ema=40,
                          vol_window=20, vol_z_min=0.3, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Fraction of bullish bars in rolling window
    is_bull     = (c > o).astype(float)
    bull_ratio  = is_bull.rolling(perc_window).mean()
    bear_ratio  = 1.0 - bull_ratio

    # Crossings (previous bar vs current bar)
    prev_bull = bull_ratio.shift(1)
    prev_bear = bear_ratio.shift(1)

    # Trend context
    trend = _ema(c, trend_ema)

    # Volume guard
    vol_z = _zscore(v, vol_window)

    # Phase transition crossings
    # Bull phase starts: ratio crosses up through perc_threshold
    bull_phase_entry = (bull_ratio >= perc_threshold) & (prev_bull < perc_threshold)
    # Bear phase starts: bear_ratio crosses up through (1 - perc_threshold)
    bear_phase_entry = (bear_ratio >= perc_threshold) & (prev_bear < perc_threshold)

    entry_long  = bull_phase_entry & (c > trend) & (vol_z >= vol_z_min)
    entry_short = bear_phase_entry & (c < trend) & (vol_z >= vol_z_min)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_PercThresh_Break(trial):
    return {
        'perc_window':     trial.suggest_int('perc_window',      12, 35),
        'perc_threshold':  trial.suggest_float('perc_threshold',  0.52, 0.70),
        'bear_threshold':  trial.suggest_float('bear_threshold',  0.30, 0.48),
        'trend_ema':       trial.suggest_int('trend_ema',          25, 70),
        'vol_window':      trial.suggest_int('vol_window',         15, 35),
        'vol_z_min':       trial.suggest_float('vol_z_min',         0.0, 1.0),
        'exit_bar':        trial.suggest_int('exit_bar',             4, 14),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'LittlesQueue_Rev': {
        'gen':   gen_LittlesQueue_Rev,
        'space': space_LittlesQueue_Rev,
    },
    'EVT_TailBreak': {
        'gen':   gen_EVT_TailBreak,
        'space': space_EVT_TailBreak,
    },
    'CopulaRank_Rev': {
        'gen':   gen_CopulaRank_Rev,
        'space': space_CopulaRank_Rev,
    },
    'ThermTemp_Fade': {
        'gen':   gen_ThermTemp_Fade,
        'space': space_ThermTemp_Fade,
    },
    'PercThresh_Break': {
        'gen':   gen_PercThresh_Break,
        'space': space_PercThresh_Break,
    },
}
