#!/usr/bin/env python3
"""
TV2 BATCH MICRO16 — 5 microstructure strategies (5m/15m optimized).
Domain: Avellaneda-Stoikov Market Making Theory, Kyle Lambda Price Impact,
        Rough Volatility (Gatheral-Jaisson-Rosenbaum), Transient Impact Decay
        (Bouchaud-Gefen-Potters-Wyart), Momentum-Volume Correlation Breakdown.
(academic, distinct from MICRO1-15 + batches 3237-3307)

Strategies:
  1. AS_Reservation_Rev     — Avellaneda & Stoikov (2008) market-making model:
                              "reservation price" r = mid - q×γ×σ²×(T−t).
                              Inventory proxy from directional volume; γ = risk
                              aversion, σ² from ATR². When price >> reservation
                              → informed selling → SHORT; price << reservation
                              → informed buying → LONG.

  2. Kyle_Lambda_Break      — Kyle (1985) price impact lambda: Δp = λ × x
                              (price change proportional to signed order flow).
                              Lambda estimated as rolling |ΔP| / |signed_vol|.
                              Lambda spike → informed trader → trend-follow.
                              Lambda collapse after spike → impact exhausted →
                              mean-reversion.

  3. RoughVol_Reversion     — Gatheral, Jaisson & Rosenbaum (2018) Rough
                              Fractional Brownian Motion (H≈0.1 in equity vol):
                              realized vol is anti-persistent. Proxy: short-window
                              realized vol z-score. When vol bursts above μ+2σ →
                              high probability of rapid vol reversion → fade the
                              price move that caused the vol spike.

  4. Impact_Decay_Fade      — Bouchaud, Gefen, Potters & Muzy (2004) transient
                              impact model: G(t)∝t^{−β}, β≈0.5. Large order
                              pushes price; impact decays as ~√t. Signal: detect
                              an impact event (price shock + volume spike), then
                              fade in the opposite direction after N bars.

  5. MomVol_Corr_Break      — Rolling Pearson correlation between price returns
                              and signed volume. High correlation = informed flow
                              (trend). Sudden collapse of this correlation signals
                              liquidity traders dominating → mean-reversion.
                              Direction: last dominant move before correlation
                              collapse is the direction to fade.

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Avellaneda, M. & Stoikov, S. (2008) — "High-frequency trading in a limit
    order book"; Quantitative Finance 8(3): 217-224.
  - Kyle, A.S. (1985) — "Continuous Auctions and Insider Trading";
    Econometrica 53(6): 1315-1335.
  - Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018) — "Volatility is rough";
    Quantitative Finance 18(6): 933-949.
  - Bouchaud, J.-P., Gefen, Y., Potters, M. & Wyart, M. (2004) — "Fluctuations
    and response in financial markets: the subtle nature of 'random' price
    changes"; Quantitative Finance 4(2): 176-190.
  - Engle, R.F. & Granger, C.W.J. (1987) — "Co-Integration and Error
    Correction: Representation, Estimation, and Testing";
    Econometrica 55(2): 251-276.
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
# 1. AS_Reservation_Rev — Avellaneda-Stoikov reservation price reversal
# ===========================================================================
# In Avellaneda & Stoikov (2008), the market maker's reservation price is:
#   r(s, q, t) = s - q × γ × σ² × (T - t)
# where s = mid-price, q = inventory, γ = risk aversion, σ² = variance.
#
# Tractable OHLCV proxy:
#   inventory proxy q̂ = rolling sum of signed returns (pos = accumulated
#   buying pressure, neg = selling); σ² = ATR²; γ = tunable param.
#
# Intuition:
#   When close >> reservation_price → asset is overpriced vs MM's cost to
#   hold inventory → likely mean-reversion → SHORT
#   When close << reservation_price → underpriced → LONG
#
# Implementation:
#   q̂[t] = rolling sum(sign(ret) × abs(ret)) over inv_window bars
#   r[t] = close[t] - q̂[t] × gamma × (ATR[t])²
#   deviation_z = zscore(close - r, dev_window)
#   SHORT when deviation_z > thresh, LONG when deviation_z < -thresh

def gen_AS_Reservation_Rev(df, inv_window=20, gamma=5.0, atr_period=14,
                            dev_window=40, z_thresh=1.5, trend_ema=60,
                            exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change()

    # Inventory proxy: signed return momentum (positive = long inventory,
    # negative = short inventory in market-maker sense)
    signed_ret = ret * ret.apply(np.sign)  # always ≥ 0 — need direction
    # Direction from close vs shifted close
    direction = ret.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    q_hat = (direction * ret.abs()).rolling(inv_window).sum()

    # Volatility proxy: ATR²
    atr_val = _atr(h, l, c, atr_period)
    sigma2  = atr_val ** 2

    # Reservation price
    reservation = c - q_hat * gamma * sigma2

    # Deviation of close from reservation
    deviation = c - reservation

    # Z-score of deviation
    dev_z = _zscore(deviation, dev_window)

    # Trend guard: don't trade against strong trend
    trend = _ema(c, trend_ema)

    # Entry signals
    # SHORT when asset is overpriced vs reservation (overextended up)
    entry_short = (dev_z > z_thresh) & (c < trend)   # weakening trend context
    # LONG when asset is underpriced vs reservation (overextended down)
    entry_long  = (dev_z < -z_thresh) & (c > trend)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_AS_Reservation_Rev(trial):
    return {
        'inv_window': trial.suggest_int('inv_window',  10, 40),
        'gamma':      trial.suggest_float('gamma',      1.0, 15.0),
        'atr_period': trial.suggest_int('atr_period',   7, 21),
        'dev_window': trial.suggest_int('dev_window',  20, 60),
        'z_thresh':   trial.suggest_float('z_thresh',   0.8, 2.5),
        'trend_ema':  trial.suggest_int('trend_ema',   30, 100),
        'exit_bar':   trial.suggest_int('exit_bar',     3, 12),
    }


# ===========================================================================
# 2. Kyle_Lambda_Break — Kyle (1985) impact lambda regime shift
# ===========================================================================
# Kyle (1985) shows that price impact is linear in aggregate order flow:
#   ΔP = λ × x   (x = signed order flow, λ = price impact coefficient)
#
# Interpretation:
#   λ high → informed traders dominate → price moves carry information →
#            trend-follow in direction of the move
#   λ collapsing after a spike → impact absorbed, noise traders dominating →
#            price mean-reverts
#
# Tractable OHLCV proxy:
#   signed_vol[t] = volume[t] × sign(ret[t])   (buy/sell volume proxy)
#   lambda_hat[t] = |ret[t]| / (|signed_vol[t]| / avg_vol + ε)
#                 ← price move per unit order flow
#
# Signal modes:
#   A. Lambda spike (z > spike_thresh) AND positive ret → LONG (informed buying)
#   B. Lambda spike AND negative ret → SHORT (informed selling)
#   C. Lambda collapses after spike (z drops below collapse_thresh after spike) →
#      mean-reversion: fade last direction

def gen_Kyle_Lambda_Break(df, lambda_window=30, spike_thresh=2.0,
                          collapse_window=8, collapse_thresh=-0.5,
                          vol_window=20, exit_bar=5, mode='trend'):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change()

    # Volume normalization
    avg_vol   = v.rolling(vol_window).mean() + 1e-10
    norm_vol  = v / avg_vol

    # Signed volume proxy (positive = net buying)
    signed_vol = norm_vol * ret.apply(lambda x: 1.0 if x >= 0 else -1.0)

    # Lambda hat: |return| per unit of |signed_vol|
    lambda_hat = ret.abs() / (norm_vol.abs() + 1e-6)

    # Z-score of lambda
    lambda_z = _zscore(lambda_hat, lambda_window)

    # Trend guard via EMA
    trend_ema = _ema(c, lambda_window)
    up_trend  = c > trend_ema

    if mode == 'trend':
        # Trade in direction of informed spike
        spike = lambda_z > spike_thresh
        entry_long  = spike & (ret > 0) & up_trend
        entry_short = spike & (ret < 0) & ~up_trend
    else:
        # Reversion mode: lambda collapses after being high
        was_spiked = lambda_z.rolling(collapse_window).max() > spike_thresh
        collapsing = lambda_z < collapse_thresh
        # Last dominant direction (sign of sum of recent returns when lambda was high)
        last_dir_up = ret.rolling(collapse_window).sum() > 0

        entry_long  = was_spiked & collapsing & ~last_dir_up  # fade down
        entry_short = was_spiked & collapsing & last_dir_up   # fade up

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Kyle_Lambda_Break(trial):
    return {
        'lambda_window':   trial.suggest_int('lambda_window',   15, 50),
        'spike_thresh':    trial.suggest_float('spike_thresh',   1.2, 3.0),
        'collapse_window': trial.suggest_int('collapse_window',   4, 15),
        'collapse_thresh': trial.suggest_float('collapse_thresh', -1.5, 0.0),
        'vol_window':      trial.suggest_int('vol_window',       10, 30),
        'exit_bar':        trial.suggest_int('exit_bar',          3, 12),
        'mode':            trial.suggest_categorical('mode', ['trend', 'reversion']),
    }


# ===========================================================================
# 3. RoughVol_Reversion — Gatheral-Jaisson-Rosenbaum rough volatility reversal
# ===========================================================================
# Gatheral, Jaisson & Rosenbaum (2018) show that equity volatility follows
# a rough fractional Brownian motion with Hurst exponent H ≈ 0.1 (strongly
# anti-persistent). This means realized vol spikes strongly tend to revert.
#
# Signal logic:
#   1. Short-window realized vol: rv_short = std(ret, short_win) × √(bars/day)
#   2. Long-window baseline: rv_long = std(ret, long_win)
#   3. Vol z-score = zscore(rv_short, z_window)
#   4. When vol spikes (z > thresh) → anti-persistence prediction: vol reverts
#      → Price that caused the vol spike is likely to mean-revert
#   5. Fade direction: price moved up during spike → SHORT; moved down → LONG
#
# Additional guard: only enter if vol is still elevated but beginning to
# compress (current rv_short < peak of last compress_bars).

def gen_RoughVol_Reversion(df, short_win=8, long_win=40, z_window=30,
                            vol_thresh=2.0, compress_bars=5, trend_ema=50,
                            exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change()

    # Short and long realized volatility
    rv_short = ret.rolling(short_win).std(ddof=0) + 1e-10
    rv_long  = ret.rolling(long_win).std(ddof=0)  + 1e-10

    # Vol ratio and z-score
    vol_ratio = rv_short / rv_long
    vol_z     = _zscore(vol_ratio, z_window)

    # Vol was spiking in last compress_bars
    vol_was_high = vol_z.rolling(compress_bars).max() > vol_thresh

    # Vol is now compressing (current below recent max)
    vol_compressing = vol_z < vol_z.rolling(compress_bars).max() - 0.3

    # Direction of the move that caused the vol spike
    # Cumulative return over the spike window
    spike_ret = ret.rolling(compress_bars).sum()
    move_was_up   = spike_ret > 0
    move_was_down = spike_ret < 0

    # Trend guard: very strong trends are not safe to fade
    trend     = _ema(c, trend_ema)
    mild_mkt  = (c - trend).abs() / (trend + 1e-10) < 0.03  # within 3% of EMA

    # Fade direction: price moved up in spike → SHORT now
    entry_short = vol_was_high & vol_compressing & move_was_up   & mild_mkt
    entry_long  = vol_was_high & vol_compressing & move_was_down & mild_mkt

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_RoughVol_Reversion(trial):
    return {
        'short_win':      trial.suggest_int('short_win',       4, 20),
        'long_win':       trial.suggest_int('long_win',        25, 60),
        'z_window':       trial.suggest_int('z_window',        20, 50),
        'vol_thresh':     trial.suggest_float('vol_thresh',     1.2, 3.0),
        'compress_bars':  trial.suggest_int('compress_bars',    3, 10),
        'trend_ema':      trial.suggest_int('trend_ema',        30, 80),
        'exit_bar':       trial.suggest_int('exit_bar',          3, 12),
    }


# ===========================================================================
# 4. Impact_Decay_Fade — Bouchaud et al. (2004) transient price impact decay
# ===========================================================================
# Bouchaud, Gefen, Potters & Wyart (2004) model price impact as:
#   G(t) ∝ t^{-β}, β ≈ 0.5   (impact decays as roughly √t)
#
# Intuition: A large order temporarily moves the price by more than it "should"
# (based on order size). Over subsequent bars, market makers refill liquidity
# and the price decays back toward fair value.
#
# Tractable signal:
#   impact_event[t]: detected when BOTH conditions hold:
#     (a) |ret[t]| > ret_thresh × rolling_std  (large price shock)
#     (b) volume[t] > vol_thresh × rolling_avg  (accompanied by large volume)
#   After detecting an impact event, fade (trade against) the price move
#   in the next N=decay_bars bars.
#   Direction of fade: if ret > 0 at impact → SHORT; ret < 0 → LONG.
#
# Guard: only if overall trend is neutral (not a strong breakout).

def gen_Impact_Decay_Fade(df, ret_window=30, ret_thresh=2.0,
                          vol_window=30, vol_thresh=2.0,
                          decay_bars=4, trend_ema=60, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change()

    # Rolling return std for shock detection
    ret_std = ret.rolling(ret_window).std(ddof=0) + 1e-10
    large_ret = ret.abs() > ret_thresh * ret_std

    # Rolling volume mean for shock detection
    vol_avg = v.rolling(vol_window).mean() + 1e-10
    large_vol = v > vol_thresh * vol_avg

    # Impact event: both conditions
    impact_event = large_ret & large_vol

    # Direction at impact
    impact_up   = impact_event & (ret > 0)
    impact_down = impact_event & (ret < 0)

    # Carry the impact signal forward for decay_bars bars (fade signal)
    # Rolling max: if any impact in last decay_bars → still in decay window
    in_decay_from_up   = impact_up.rolling(decay_bars).max().astype(bool)
    in_decay_from_down = impact_down.rolling(decay_bars).max().astype(bool)

    # Remove the bar of impact itself (trade starting from bar+1)
    # Impact is from shift(1) data, so entry is from bar after
    fade_short = in_decay_from_up   & ~impact_event  # fade the up-impact
    fade_long  = in_decay_from_down & ~impact_event  # fade the down-impact

    # Trend guard: avoid fading in strong directional markets
    trend = _ema(c, trend_ema)
    trend_neutral = (c - trend).abs() / (trend + 1e-10) < 0.025

    entry_short = fade_short & trend_neutral
    entry_long  = fade_long  & trend_neutral

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Impact_Decay_Fade(trial):
    return {
        'ret_window':  trial.suggest_int('ret_window',   15, 50),
        'ret_thresh':  trial.suggest_float('ret_thresh',  1.2, 3.0),
        'vol_window':  trial.suggest_int('vol_window',   15, 50),
        'vol_thresh':  trial.suggest_float('vol_thresh',  1.5, 3.5),
        'decay_bars':  trial.suggest_int('decay_bars',    2, 10),
        'trend_ema':   trial.suggest_int('trend_ema',    30, 100),
        'exit_bar':    trial.suggest_int('exit_bar',      3, 12),
    }


# ===========================================================================
# 5. MomVol_Corr_Break — Momentum-volume correlation breakdown regime signal
# ===========================================================================
# When price momentum and volume are positively correlated → informed order
# flow dominates → trending regime.
# When this correlation suddenly breaks down (drops sharply) → regime switch:
# liquidity traders (noise) dominate → mean-reversion likely.
#
# Academic basis: Easley & O'Hara (1992) information-based trading model;
# Chordia & Swaminathan (2000) trading volume and cross-autocorrelations.
#
# Signal:
#   corr[t] = rolling Pearson corr(abs(ret), volume) over corr_window
#   corr_z  = zscore(corr, z_window)
#   Breakdown detected: corr was high (> high_thresh) but now drops
#   sharply (current < drop_thresh) within collapse_bars
#
# Direction of entry:
#   Fade the dominant recent directional move before the breakdown.
#   Momentum before breakdown = sign(rolling sum of ret over corr_window)
#   If positive momentum → SHORT (fade), if negative → LONG (fade)

def gen_MomVol_Corr_Break(df, corr_window=20, z_window=30,
                           high_thresh=0.5, drop_thresh=-0.2,
                           collapse_bars=5, mom_window=10,
                           trend_ema=50, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change()

    # Rolling Pearson correlation between |ret| and volume
    # Use rolling covariance / (std × std)
    abs_ret = ret.abs()
    corr = abs_ret.rolling(corr_window).corr(v)

    # Handle NaN edges
    corr = corr.fillna(0.0)

    # Z-score of correlation (to detect breakdowns relative to history)
    corr_z = _zscore(corr, z_window)

    # Was correlation high recently?
    corr_was_high = corr.rolling(collapse_bars).max() > high_thresh

    # Correlation has now collapsed
    corr_collapsed = corr < drop_thresh

    # Breakdown event
    breakdown = corr_was_high & corr_collapsed

    # Dominant momentum before breakdown (sign of net return over mom_window)
    prior_mom = ret.rolling(mom_window).sum()
    prior_up  = prior_mom > 0   # recent up-trend to fade
    prior_dn  = prior_mom < 0   # recent down-trend to fade

    # Trend guard
    trend     = _ema(c, trend_ema)
    trend_neut = (c - trend).abs() / (trend + 1e-10) < 0.03

    entry_short = breakdown & prior_up & trend_neut   # fade the up-momentum
    entry_long  = breakdown & prior_dn & trend_neut   # fade the down-momentum

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_MomVol_Corr_Break(trial):
    return {
        'corr_window':   trial.suggest_int('corr_window',    10, 35),
        'z_window':      trial.suggest_int('z_window',       20, 50),
        'high_thresh':   trial.suggest_float('high_thresh',   0.2, 0.7),
        'drop_thresh':   trial.suggest_float('drop_thresh',  -0.5, 0.0),
        'collapse_bars': trial.suggest_int('collapse_bars',    3, 10),
        'mom_window':    trial.suggest_int('mom_window',       5, 20),
        'trend_ema':     trial.suggest_int('trend_ema',       30, 80),
        'exit_bar':      trial.suggest_int('exit_bar',         3, 12),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'AS_Reservation_Rev': {
        'gen':   gen_AS_Reservation_Rev,
        'space': space_AS_Reservation_Rev,
    },
    'Kyle_Lambda_Break': {
        'gen':   gen_Kyle_Lambda_Break,
        'space': space_Kyle_Lambda_Break,
    },
    'RoughVol_Reversion': {
        'gen':   gen_RoughVol_Reversion,
        'space': space_RoughVol_Reversion,
    },
    'Impact_Decay_Fade': {
        'gen':   gen_Impact_Decay_Fade,
        'space': space_Impact_Decay_Fade,
    },
    'MomVol_Corr_Break': {
        'gen':   gen_MomVol_Corr_Break,
        'space': space_MomVol_Corr_Break,
    },
}
