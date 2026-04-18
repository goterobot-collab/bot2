#!/usr/bin/env python3
"""
TV2 BATCH MICRO9 — 5 microstructure strategies (5m/15m optimized).
Domain: Market Impact Theory & Microstructure Noise Decomposition
(academic, distinct from MICRO1-8 + batches 3237-3318)

Strategies:
  1. RollSpread_Rev     — Roll (1984) serial covariance bid-ask bounce reversal
  2. KyleLambda_Fade    — Kyle (1985) λ price-impact per volume: high λ → informed
                          trader activity → fade the overshoot
  3. AmihudIlliq_Rev    — Amihud (2002) ILLIQ ratio: |return|/volume spike → reversion
  4. VPIN_Proxy         — Easley et al. (2012) volume-synchronized informed-trading
                          proxy: volume imbalance toxicity → fade direction
  5. HasbrouckTmp       — Hasbrouck (1991) temporary price impact component:
                          isolate transient microstructure impact → mean-revert it

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Roll (1984) — "A Simple Implicit Measure of the Effective Bid-Ask Spread"
    in Prices; J. of Finance 39(4): 1127-1139.
  - Kyle (1985) — "Continuous Auctions and Insider Trading"; Econometrica 53(6).
  - Amihud (2002) — "Illiquidity and Stock Returns: Cross-Section and
    Time-Series Effects"; J. of Financial Markets 5(1): 31-56.
  - Easley, Lopez de Prado, O'Hara (2012) — "Flow Toxicity and Liquidity in
    a High-Frequency World"; Review of Financial Studies 25(5): 1457-1493.
  - Hasbrouck (1991) — "Measuring the Information Content of Stock Trades";
    J. of Finance 46(1): 179-207.
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
# 1. RollSpread_Rev — Roll (1984) bid-ask bounce reversal
# ═══════════════════════════════════════════════════════════════════════════
# Roll (1984): The bid-ask spread induces NEGATIVE serial covariance in price
# changes. cov(Δp_t, Δp_{t-1}) = -s²/4 where s = effective spread.
# High negative serial covariance → large effective spread → strong bounce.
# Trading logic: detect bars with unusually negative serial covariance (high
# microstructure friction), then fade the next price move (bounce expected).
# Entry trigger: price moved strongly in one direction on a high-friction bar.

def gen_RollSpread_Rev(df, cov_window=20, spread_z_thresh=1.5,
                       price_move_thresh=0.6, trend_ema=40, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)

    ret = c.pct_change().fillna(0)
    ret_lag = ret.shift(1)

    # Rolling serial covariance proxy: cov(ret_t, ret_{t-1})
    # Use rolling product mean (centered)
    product = ret * ret_lag
    rolling_cov = product.rolling(cov_window).mean()

    # Normalize: negative = Roll spread present
    cov_z = _zscore(rolling_cov, cov_window)

    # High negative serial cov → high spread regime
    high_spread = cov_z < -spread_z_thresh

    # Price move on this bar (bar body as fraction of ATR)
    atr = _atr(h, l, c, p=14)
    bar_move = (c - c.shift(1)) / (atr + 1e-8)
    bar_move_z = _zscore(bar_move, cov_window)

    # Fade the move: strong up bar in high-spread regime → short (bounce down)
    entry_short = high_spread & (bar_move_z > price_move_thresh)
    entry_long  = high_spread & (bar_move_z < -price_move_thresh)

    # Trend filter: only fade against trend if spread is very extreme
    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.982)
    entry_short = entry_short & (c < trend * 1.018)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_RollSpread_Rev(trial):
    return {
        'cov_window':        trial.suggest_int('cov_window', 12, 35),
        'spread_z_thresh':   trial.suggest_float('spread_z_thresh', 1.0, 2.5, step=0.25),
        'price_move_thresh': trial.suggest_float('price_move_thresh', 0.4, 1.2, step=0.2),
        'trend_ema':         trial.suggest_int('trend_ema', 25, 60),
        'exit_bar':          trial.suggest_int('exit_bar', 4, 14),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. KyleLambda_Fade — Kyle's lambda market-impact informed-trader fade
# ═══════════════════════════════════════════════════════════════════════════
# Kyle (1985): λ = ΔPrice / ΔVolume (OLS slope of price change on order flow).
# High λ → informed traders moving price efficiently per unit of volume.
# Proxy: rolling OLS of |return| on normalized volume. High λ → informed
# trading → price has moved with purpose. When λ spikes AND price is at
# Z-score extreme → the informed move is overdone → fade back.

def gen_KyleLambda_Fade(df, lambda_window=30, lambda_z_thresh=1.4,
                        zscore_period=20, zscore_thresh=1.2,
                        trend_ema=35, exit_bar=7):
    c = df['close'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change().fillna(0).abs()
    vol_norm = (v / (v.rolling(lambda_window).mean() + 1e-8)).fillna(1.0)

    # Kyle's lambda proxy: rolling OLS slope of |ret| ~ vol_norm
    # Simplified: rolling covariance / rolling variance
    cov_rv = (ret * vol_norm).rolling(lambda_window).mean() - \
             ret.rolling(lambda_window).mean() * vol_norm.rolling(lambda_window).mean()
    var_v  = vol_norm.rolling(lambda_window).var(ddof=0) + 1e-12
    kyle_lambda = (cov_rv / var_v).fillna(0.0)

    # Normalize lambda
    lambda_z = _zscore(kyle_lambda, lambda_window)

    # High lambda regime → informed activity → overshoot
    high_lambda = lambda_z > lambda_z_thresh

    # Price direction: Z-score of close
    pz = _zscore(df['close'].shift(1), zscore_period)

    # Fade: high informed-impact + price extended → reverse
    entry_long  = high_lambda & (pz < -zscore_thresh)
    entry_short = high_lambda & (pz >  zscore_thresh)

    # Trend filter
    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.985)
    entry_short = entry_short & (c < trend * 1.015)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_KyleLambda_Fade(trial):
    return {
        'lambda_window':    trial.suggest_int('lambda_window', 15, 45),
        'lambda_z_thresh':  trial.suggest_float('lambda_z_thresh', 1.0, 2.5, step=0.25),
        'zscore_period':    trial.suggest_int('zscore_period', 12, 30),
        'zscore_thresh':    trial.suggest_float('zscore_thresh', 0.8, 2.0, step=0.2),
        'trend_ema':        trial.suggest_int('trend_ema', 20, 55),
        'exit_bar':         trial.suggest_int('exit_bar', 4, 14),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. AmihudIlliq_Rev — Amihud (2002) illiquidity-spike reversion
# ═══════════════════════════════════════════════════════════════════════════
# Amihud (2002): ILLIQ = |return| / dollar_volume. High ILLIQ → price moves
# a lot per unit of volume → illiquid bar. In crypto: ILLIQ spike on a bar
# implies low depth / thin orderbook → overshoot likely → fade the bar direction.
# Additional: after high-ILLIQ bar, next bar often corrects (transient impact).

def gen_AmihudIlliq_Rev(df, illiq_window=30, illiq_z_thresh=1.5,
                        min_volume=100.0, trend_ema=40, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret_abs = c.pct_change().abs().fillna(0)
    dv = (c * v + 1e-8)  # dollar volume proxy

    # Amihud ILLIQ = |return| / dollar_volume
    illiq = (ret_abs / dv).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Smooth and normalize
    illiq_smooth = illiq.rolling(5).mean()
    illiq_z = _zscore(illiq_smooth, illiq_window)

    # Price direction on high-ILLIQ bar (which direction was the illiquid move?)
    bar_dir = (c - c.shift(1)).apply(np.sign)  # +1 up, -1 down

    # Volume filter: need minimum volume to distinguish thin from zero
    vol_ok = v > min_volume

    # High illiquidity → thin bar overshoot → fade direction of move
    high_illiq = illiq_z > illiq_z_thresh
    entry_long  = high_illiq & (bar_dir < 0) & vol_ok   # illiquid DOWN → buy bounce
    entry_short = high_illiq & (bar_dir > 0) & vol_ok   # illiquid UP → sell bounce

    # Trend filter
    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.980)
    entry_short = entry_short & (c < trend * 1.020)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_AmihudIlliq_Rev(trial):
    return {
        'illiq_window':    trial.suggest_int('illiq_window', 15, 50),
        'illiq_z_thresh':  trial.suggest_float('illiq_z_thresh', 1.0, 2.5, step=0.25),
        'min_volume':      trial.suggest_float('min_volume', 50.0, 500.0, step=50.0),
        'trend_ema':       trial.suggest_int('trend_ema', 25, 60),
        'exit_bar':        trial.suggest_int('exit_bar', 3, 12),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. VPIN_Proxy — Volume-Synchronized Informed-Trading toxicity fade
# ═══════════════════════════════════════════════════════════════════════════
# Easley, Lopez de Prado & O'Hara (2012): VPIN = volume imbalance in volume
# buckets. High VPIN → toxic order flow (informed traders dominate) → adverse
# selection for market makers → liquidity withdrawal → overshoot + reversion.
# Proxy (bar-level): classify each bar as buy/sell volume using bulk classification
# (Lee & Ready 1991 tick-rule proxy via bar close vs open direction).
# Rolling buy/sell imbalance over N volume buckets → VPIN proxy.

def gen_VPIN_Proxy(df, bucket_window=50, vpin_z_thresh=1.6,
                   confirmation_bars=2, trend_ema=35, exit_bar=8):
    c = df['close'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Tick-rule proxy: if close > open → buy volume; else sell volume
    buy_vol  = v * ((c > o).astype(float) * 0.5 + 0.5 * (c == o).astype(float))
    sell_vol = v - buy_vol

    # Volume imbalance per bar
    imbalance = (buy_vol - sell_vol).abs()
    total_vol = v + 1e-8

    # VPIN proxy: rolling imbalance fraction over bucket_window bars
    vpin = imbalance.rolling(bucket_window).sum() / (total_vol.rolling(bucket_window).sum() + 1e-8)

    # Normalize
    vpin_z = _zscore(vpin, bucket_window)

    # Net buy pressure direction
    net_dir = (buy_vol - sell_vol).rolling(bucket_window).sum().apply(np.sign)

    # High VPIN + directional pressure → informed flow → eventual reversion
    high_toxicity = vpin_z > vpin_z_thresh

    # Fade the dominant direction after high toxicity
    entry_long  = high_toxicity & (net_dir < 0)   # sell pressure dominant → buy fade
    entry_short = high_toxicity & (net_dir > 0)   # buy pressure dominant → sell fade

    # Confirmation: signal must persist for N bars (avoid single-bar noise)
    entry_long  = entry_long.rolling(confirmation_bars).sum() == confirmation_bars
    entry_short = entry_short.rolling(confirmation_bars).sum() == confirmation_bars

    # Trend filter
    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.983)
    entry_short = entry_short & (c < trend * 1.017)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_VPIN_Proxy(trial):
    return {
        'bucket_window':      trial.suggest_int('bucket_window', 20, 80),
        'vpin_z_thresh':      trial.suggest_float('vpin_z_thresh', 1.0, 2.5, step=0.25),
        'confirmation_bars':  trial.suggest_int('confirmation_bars', 1, 4),
        'trend_ema':          trial.suggest_int('trend_ema', 20, 55),
        'exit_bar':           trial.suggest_int('exit_bar', 5, 16),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. HasbrouckTmp — Hasbrouck (1991) temporary price impact reversal
# ═══════════════════════════════════════════════════════════════════════════
# Hasbrouck (1991): price changes decompose into a PERMANENT component
# (information) and a TEMPORARY component (microstructure noise / bid-ask).
# The temporary component mean-reverts to zero.
# Proxy: use a two-speed EMA framework — fast EMA approximates total price,
# slow EMA approximates the permanent (fundamental) component.
# Temporary = fast_EMA - slow_EMA = deviation driven by noise.
# Extreme temporary component → noise overshoot → trade reversion.

def gen_HasbrouckTmp(df, fast_span=5, slow_span=30, tmp_z_window=25,
                     tmp_z_thresh=1.3, volume_confirm_z=0.5,
                     trend_ema=45, exit_bar=7):
    c = df['close'].shift(1)
    v = df['volume'].shift(1)

    # Permanent component: slow EMA
    perm = _ema(c, slow_span)

    # Total: fast EMA (approximates real-time signal)
    total = _ema(c, fast_span)

    # Temporary impact: deviation from permanent
    tmp_impact = (total - perm) / (perm + 1e-8)

    # Normalize temporary component
    tmp_z = _zscore(tmp_impact, tmp_z_window)

    # Volume confirmation: temporary impact more reliable when volume is above avg
    vol_z = _zscore(v, tmp_z_window)
    vol_confirm = vol_z > volume_confirm_z

    # Fade the temporary impact: large positive tmp_z → overshoot up → short
    entry_long  = (tmp_z < -tmp_z_thresh) & vol_confirm
    entry_short = (tmp_z >  tmp_z_thresh) & vol_confirm

    # Trend filter
    trend = _ema(c, trend_ema)
    entry_long  = entry_long  & (c > trend * 0.987)
    entry_short = entry_short & (c < trend * 1.013)

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_HasbrouckTmp(trial):
    return {
        'fast_span':          trial.suggest_int('fast_span', 3, 10),
        'slow_span':          trial.suggest_int('slow_span', 20, 50),
        'tmp_z_window':       trial.suggest_int('tmp_z_window', 15, 40),
        'tmp_z_thresh':       trial.suggest_float('tmp_z_thresh', 0.8, 2.2, step=0.2),
        'volume_confirm_z':   trial.suggest_float('volume_confirm_z', -0.5, 1.0, step=0.25),
        'trend_ema':          trial.suggest_int('trend_ema', 25, 65),
        'exit_bar':           trial.suggest_int('exit_bar', 4, 14),
    }


# ─── STRATEGY_EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'RollSpread_Rev': {
        'gen':   gen_RollSpread_Rev,
        'space': space_RollSpread_Rev,
    },
    'KyleLambda_Fade': {
        'gen':   gen_KyleLambda_Fade,
        'space': space_KyleLambda_Fade,
    },
    'AmihudIlliq_Rev': {
        'gen':   gen_AmihudIlliq_Rev,
        'space': space_AmihudIlliq_Rev,
    },
    'VPIN_Proxy': {
        'gen':   gen_VPIN_Proxy,
        'space': space_VPIN_Proxy,
    },
    'HasbrouckTmp': {
        'gen':   gen_HasbrouckTmp,
        'space': space_HasbrouckTmp,
    },
}
