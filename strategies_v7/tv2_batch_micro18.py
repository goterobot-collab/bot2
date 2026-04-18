#!/usr/bin/env python3
"""
TV2 BATCH MICRO18 — 5 microstructure strategies (5m/15m optimized).
Domain: Glosten-Milgrom Adverse Selection (1985), Amihud Illiquidity Ratio (2002),
        Hasbrouck Information Share (1991), LOB Order Flow Imbalance
        (Cont-Kukanov-Stoikov 2014), Granger Causality Volume→Price (1969).
        All new vs MICRO1-17 and batches 3237-3318.

Strategies:
  1. GM_AdverseSelection    — Glosten & Milgrom (1985): price moves reflect
                              adverse-selection cost from informed traders.
                              Proxy: rolling asymmetry between up-tick volume
                              and down-tick volume. Spike in one direction →
                              informed flow → momentum. Balanced flow →
                              uninformed → mean-reversion.

  2. Amihud_Illiq_Reversion — Amihud (2002) ILLIQ = |return| / dollar_volume.
                              High ILLIQ → price is sensitive to order flow
                              (thin market). When ILLIQ spikes AND price moved
                              strongly, the move is likely to reverse (execution
                              overshoot in illiquid market).

  3. Hasbrouck_InfoShare    — Hasbrouck (1991): information share measures
                              how much price discovery occurs in each market.
                              Proxy: ratio of return variance to volume-weighted
                              return — when ratio is abnormally high, informed
                              trading is leading price → trend-follow the
                              informed side.

  4. LOB_OFI_Signal         — Cont, Kukanov & Stoikov (2014): Order Flow
                              Imbalance (OFI) = ΔBestBid_qty - ΔBestAsk_qty.
                              OHLCV proxy: (close-open)/(high-low+ε) × volume
                              as directional volume imbalance. Sustained
                              positive OFI → buy pressure → LONG; negative →
                              SHORT. Requires persistence confirmation.

  5. Granger_Vol2Price      — Granger (1969) causality: does past volume Granger-
                              cause future price changes? Rolling regression:
                              ret[t] ~ β1×ret[t-1] + β2×svol[t-1] + β3×svol[t-2]
                              where svol = signed volume. β2+β3 estimate the
                              causal impact. When the causal signal flips sign →
                              imminent price direction reversal.

Anti-repainting: ALL OHLCV access via .shift(1) — NEVER current bar.
State machine: 3-state (0=flat, 1=long, -1=short) + exit_bar configurable.
TF focus: 5m, 15m.
Deadline: 2026-04-26.

Academic references:
  - Glosten, L.R. & Milgrom, P.R. (1985) — "Bid, Ask and Transaction Prices
    in a Specialist Market with Heterogeneously Informed Traders";
    Journal of Financial Economics 14(1): 71-100.
  - Amihud, Y. (2002) — "Illiquidity and Stock Returns: Cross-Section and
    Time-Series Effects"; Journal of Financial Markets 5(1): 31-56.
  - Hasbrouck, J. (1991) — "Measuring the Information Content of Stock Trades";
    Journal of Finance 46(1): 179-207.
  - Cont, R., Kukanov, A. & Stoikov, S. (2014) — "The Price Impact of Order
    Book Events"; Journal of Financial Econometrics 12(1): 47-88.
  - Granger, C.W.J. (1969) — "Investigating Causal Relations by Econometric
    Models and Cross-Spectral Methods"; Econometrica 37(3): 424-438.
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
# 1. GM_AdverseSelection — Glosten-Milgrom (1985) adverse-selection reversal
# ===========================================================================
# Glosten & Milgrom (1985) decompose the bid-ask spread into:
#   adverse-selection component (informed traders extract rents) +
#   inventory/order-processing components.
#
# OHLCV proxy for informed vs uninformed flow:
#   up_vol[t]   = volume[t]   iff close[t] > open[t]  else 0
#   down_vol[t] = volume[t]   iff close[t] < open[t]  else 0
#   flow_imb[t] = (up_vol - down_vol) / (up_vol + down_vol + ε)
#                ← ranges [-1, 1], + = net buying pressure
#
# Regime classification:
#   When |flow_imb| is LOW  → uninformed (random) flow → mean-reversion
#   When |flow_imb| is HIGH → informed directional flow  → momentum
#
# Rolling z-score of flow_imb used for signal direction.
# Regime gate: rolling |flow_imb| mean distinguishes high/low regimes.

def gen_GM_AdverseSelection(df, flow_window=12, z_window=30, z_thresh=1.4,
                              regime_window=40, regime_thresh=0.25,
                              trend_ema=60, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Directional volume decomposition
    is_up   = c > o
    is_down = c < o

    up_vol   = v.where(is_up,   other=0.0)
    down_vol = v.where(is_down, other=0.0)
    total_dv = up_vol + down_vol + 1e-10

    # Flow imbalance per bar
    flow_imb = (up_vol - down_vol) / total_dv

    # Rolling smoothed flow imbalance
    flow_smooth = flow_imb.rolling(flow_window).mean()

    # Z-score for directionality
    flow_z = _zscore(flow_smooth, z_window)

    # Regime: rolling mean of |flow_imb| → high = informed, low = uninformed
    regime_strength = flow_imb.abs().rolling(regime_window).mean()
    informed_regime = regime_strength > regime_thresh
    random_regime   = regime_strength <= regime_thresh

    # Trend guard
    trend = _ema(c, trend_ema)
    above = c > trend
    below = c < trend

    # Informed regime + strong flow → momentum
    entry_long_mom  = informed_regime & (flow_z >  z_thresh) & above
    entry_short_mom = informed_regime & (flow_z < -z_thresh) & below

    # Random (uninformed) regime + flow extreme → mean-reversion (fade)
    entry_long_rev  = random_regime & (flow_z < -z_thresh) & below
    entry_short_rev = random_regime & (flow_z >  z_thresh) & above

    entry_long  = entry_long_mom  | entry_long_rev
    entry_short = entry_short_mom | entry_short_rev

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_GM_AdverseSelection(trial):
    return {
        'flow_window':    trial.suggest_int('flow_window',    5, 25),
        'z_window':       trial.suggest_int('z_window',       20, 60),
        'z_thresh':       trial.suggest_float('z_thresh',      0.8, 2.5),
        'regime_window':  trial.suggest_int('regime_window',  20, 80),
        'regime_thresh':  trial.suggest_float('regime_thresh', 0.1, 0.5),
        'trend_ema':      trial.suggest_int('trend_ema',      30, 100),
        'exit_bar':       trial.suggest_int('exit_bar',        3, 12),
    }


# ===========================================================================
# 2. Amihud_Illiq_Reversion — Amihud (2002) illiquidity reversal
# ===========================================================================
# Amihud (2002) defines the ILLIQ ratio as:
#   ILLIQ[t] = |R[t]| / DollarVolume[t]
#
# where |R[t]| is the absolute daily return and DollarVolume[t] = price × volume.
# Higher ILLIQ → price is more sensitive to order flow (illiquid market).
#
# Microstructure insight:
#   When ILLIQ spikes → a large order moved price disproportionately.
#   This overshoot (relative to liquidity) is likely to REVERSE.
#   The reversal is stronger when ILLIQ spike coincides with a big price move.
#
# Signal:
#   ILLIQ_z > thresh  AND  |ret_z| > ret_thresh   → overshoot → fade price move
#   ret_z > 0 → SHORT (price went up on illiquid move)
#   ret_z < 0 → LONG  (price went down on illiquid move)

def gen_Amihud_Illiq_Reversion(df, illiq_window=20, ret_window=10,
                                  z_window=30, illiq_thresh=1.5,
                                  ret_thresh=1.0, trend_ema=50, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change()

    # Dollar volume proxy (using close as price)
    dollar_vol = (c * v) + 1e-10

    # Amihud ILLIQ ratio
    illiq = ret.abs() / dollar_vol

    # Smooth ILLIQ: rolling mean (to reduce bar-level noise)
    illiq_smooth = illiq.rolling(illiq_window).mean()

    # Z-scores
    illiq_z = _zscore(illiq_smooth, z_window)
    ret_z   = _zscore(ret.rolling(ret_window).mean(), z_window)

    # Illiquidity spike confirmation
    illiq_spike = illiq_z > illiq_thresh

    # Trend guard: mild market (not in strong directional trend)
    trend = _ema(c, trend_ema)
    mild  = (c - trend).abs() / (trend.abs() + 1e-10) < 0.06

    # Price went UP on illiquid move → overshoot → SHORT reversal
    # Price went DOWN on illiquid move → overshoot → LONG reversal
    entry_short = illiq_spike & (ret_z >  ret_thresh) & mild
    entry_long  = illiq_spike & (ret_z < -ret_thresh) & mild

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Amihud_Illiq_Reversion(trial):
    return {
        'illiq_window':  trial.suggest_int('illiq_window',   10, 40),
        'ret_window':    trial.suggest_int('ret_window',      5, 20),
        'z_window':      trial.suggest_int('z_window',        20, 60),
        'illiq_thresh':  trial.suggest_float('illiq_thresh',  0.8, 2.5),
        'ret_thresh':    trial.suggest_float('ret_thresh',     0.5, 2.0),
        'trend_ema':     trial.suggest_int('trend_ema',       30, 80),
        'exit_bar':      trial.suggest_int('exit_bar',         3, 12),
    }


# ===========================================================================
# 3. Hasbrouck_InfoShare — Hasbrouck (1991) information share signal
# ===========================================================================
# Hasbrouck (1991) defines the information share of a market as the fraction
# of the total variance of the efficient price innovation attributable to that
# market. A single-market proxy:
#
#   IS[t] = Var(ret[t]) / (Var(ret[t]) + Var(dollar_vol[t]))
#
# Intuition:
#   When IS is HIGH → return variance dominates volume variance →
#   price is leading volume → informed trading is setting prices.
#   When IS is LOW  → volume variance dominates → liquidity trading.
#
# Trading signal:
#   IS spike (informed regime): trend-follow the dominant return direction.
#   IS collapse after spike: regime shift → mean-reversion expected.
#
# Direction from rolling return sign (rolling EMA of returns).

def gen_Hasbrouck_InfoShare(df, var_window=20, is_window=40, z_window=30,
                              is_thresh=1.5, trend_ema=60, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    v = df['volume'].shift(1)

    ret = c.pct_change()

    # Dollar volume (normalized by rolling mean to make comparable to returns)
    dollar_vol = c * v
    dv_norm    = dollar_vol / (dollar_vol.rolling(var_window).mean() + 1e-10) - 1.0

    # Rolling variances
    var_ret = ret.rolling(var_window).var() + 1e-10
    var_dv  = dv_norm.rolling(var_window).var() + 1e-10

    # Hasbrouck Information Share proxy
    info_share = var_ret / (var_ret + var_dv)

    # Z-score of information share
    is_z = _zscore(info_share, z_window)

    # Previous-bar info share (for transition detection)
    is_prev = info_share.shift(1)
    was_high = is_prev > info_share.rolling(is_window).quantile(0.75)
    now_low  = info_share <= info_share.rolling(is_window).quantile(0.40)

    # Collapse: was high IS, now low IS → regime shift → mean-reversion
    is_collapse = was_high & now_low

    # Direction: rolling EMA of returns
    ret_direction = _ema(ret, 8)

    # Trend guard
    trend = _ema(c, trend_ema)
    above = c > trend
    below = c < trend

    # Informed spike: trend-follow the return direction
    informed_spike = is_z > is_thresh
    entry_long_inf  = informed_spike & (ret_direction > 0) & above
    entry_short_inf = informed_spike & (ret_direction < 0) & below

    # IS collapse: fade the prior trend (mean-reversion)
    entry_long_rev  = is_collapse & (ret_direction < 0) & below  # fade down move
    entry_short_rev = is_collapse & (ret_direction > 0) & above  # fade up move

    entry_long  = entry_long_inf  | entry_long_rev
    entry_short = entry_short_inf | entry_short_rev

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Hasbrouck_InfoShare(trial):
    return {
        'var_window':  trial.suggest_int('var_window',   10, 30),
        'is_window':   trial.suggest_int('is_window',    20, 80),
        'z_window':    trial.suggest_int('z_window',     20, 60),
        'is_thresh':   trial.suggest_float('is_thresh',   0.8, 2.5),
        'trend_ema':   trial.suggest_int('trend_ema',    30, 100),
        'exit_bar':    trial.suggest_int('exit_bar',      3, 12),
    }


# ===========================================================================
# 4. LOB_OFI_Signal — Cont-Kukanov-Stoikov (2014) Order Flow Imbalance
# ===========================================================================
# Cont, Kukanov & Stoikov (2014) show that Order Flow Imbalance (OFI),
# defined as changes in best bid/ask quantities, is the strongest predictor
# of short-term price changes in LOB microstructure:
#
#   OFI[t] = ΔQ_bid[t] - ΔQ_ask[t]
#
# OHLCV proxy for OFI (since LOB data unavailable):
#   Directional efficiency = (close - open) / (high - low + ε)
#   This measures how efficiently price moved to the close within the bar range.
#   Range-normalized close position ∈ [-1, 1]: +1 = closed at high, -1 = at low.
#   Signed volume = volume × directional_efficiency   ← OFI proxy
#
# Signal logic:
#   Sustained positive OFI (rolling sum > thresh) → buy pressure → LONG
#   Sustained negative OFI (rolling sum < -thresh) → sell pressure → SHORT
#   Require persistence: OFI sustained over N bars before entry.

def gen_LOB_OFI_Signal(df, ofi_window=8, persist_window=3,
                         z_window=30, z_thresh=1.2,
                         trend_ema=50, exit_bar=5):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    # Directional efficiency proxy (OFI per bar)
    bar_range = (h - l).clip(lower=1e-10)
    dir_eff   = (c - o) / bar_range          # in (-1, 1) approximately
    dir_eff   = dir_eff.clip(-1.0, 1.0)

    # Signed volume (OFI proxy)
    ofi_raw = dir_eff * v

    # Rolling OFI accumulation
    ofi_roll = ofi_raw.rolling(ofi_window).mean()

    # Z-score for threshold-agnostic entry
    ofi_z = _zscore(ofi_roll, z_window)

    # Persistence: OFI signal direction maintained for persist_window bars
    ofi_pos = (ofi_z > z_thresh)
    ofi_neg = (ofi_z < -z_thresh)

    # Require signal to be consistently positive/negative over persist_window
    ofi_pos_count = ofi_pos.rolling(persist_window).sum()
    ofi_neg_count = ofi_neg.rolling(persist_window).sum()

    sustained_long  = ofi_pos_count >= persist_window
    sustained_short = ofi_neg_count >= persist_window

    # Trend gate: align with trend direction
    trend = _ema(c, trend_ema)
    above = c > trend
    below = c < trend

    entry_long  = sustained_long  & above
    entry_short = sustained_short & below

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_LOB_OFI_Signal(trial):
    return {
        'ofi_window':     trial.suggest_int('ofi_window',     4, 20),
        'persist_window': trial.suggest_int('persist_window', 2, 6),
        'z_window':       trial.suggest_int('z_window',       20, 60),
        'z_thresh':       trial.suggest_float('z_thresh',      0.8, 2.5),
        'trend_ema':      trial.suggest_int('trend_ema',      30, 80),
        'exit_bar':       trial.suggest_int('exit_bar',        3, 12),
    }


# ===========================================================================
# 5. Granger_Vol2Price — Granger (1969) causality: volume → price
# ===========================================================================
# Granger (1969) causality: X Granger-causes Y if past values of X improve
# the prediction of Y beyond past values of Y alone.
#
# Rolling implementation:
#   Model A (baseline): ret[t] ~ β0 + β1×ret[t-1] + β2×ret[t-2]
#   Model B (augmented): ret[t] ~ β0 + β1×ret[t-1] + β2×ret[t-2]
#                                     + γ1×svol[t-1] + γ2×svol[t-2]
#   where svol[t] = signed volume = volume × sign(close[t] - open[t])
#
# Causality signal = γ1 + γ2 (net volume → price impact coefficient).
#   When causal_signal > 0 → past buy volume predicts price increase → LONG
#   When causal_signal < 0 → past sell volume predicts price decrease → SHORT
#   When |causal_signal| small → no Granger causality → skip
#
# Rolling OLS is computed via vectorized numpy within each window.
# Transition: sign flip of causal_signal → direction reversal.

def gen_Granger_Vol2Price(df, granger_window=40, smooth_window=8,
                            z_window=30, z_thresh=1.3,
                            trend_ema=60, exit_bar=6):
    c = df['close'].shift(1)
    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    v = df['volume'].shift(1)

    ret     = c.pct_change()
    sign_oc = np.sign(c - o)
    svol    = v * sign_oc                 # signed volume

    ret_vals  = ret.values
    svol_vals = svol.values
    n         = len(ret_vals)

    causal_gamma = np.full(n, np.nan)

    for i in range(granger_window + 2, n):
        # Window: bars (i-granger_window) to (i-1)
        i0 = i - granger_window
        i1 = i                            # exclusive

        # Align lags: y[t], ret[t-1], ret[t-2], svol[t-1], svol[t-2]
        # y = ret from index 2..W (requires 2 lags back)
        win_y    = ret_vals[i0 + 2: i1]
        win_r1   = ret_vals[i0 + 1: i1 - 1]
        win_r2   = ret_vals[i0:     i1 - 2]
        win_sv1  = svol_vals[i0 + 1: i1 - 1]
        win_sv2  = svol_vals[i0:     i1 - 2]

        # Skip if any NaNs
        mat = np.column_stack([win_r1, win_r2, win_sv1, win_sv2])
        valid = (~np.isnan(win_y)) & (~np.any(np.isnan(mat), axis=1))
        if valid.sum() < 10:
            continue

        y   = win_y[valid]
        X   = np.column_stack([
            np.ones(valid.sum()),
            win_r1[valid],
            win_r2[valid],
            win_sv1[valid],
            win_sv2[valid],
        ])

        # OLS: β = (X'X)^{-1} X'y — using lstsq for numerical stability
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # γ1 + γ2 = net causal volume → price coefficient
        causal_gamma[i] = coeffs[3] + coeffs[4]

    causal_s = pd.Series(causal_gamma, index=c.index)

    # Smooth the causal signal
    causal_smooth = causal_s.rolling(smooth_window).mean()

    # Z-score for entry thresholding
    causal_z = _zscore(causal_smooth, z_window)

    # Trend guard
    trend = _ema(c, trend_ema)
    above = c > trend
    below = c < trend

    # Positive causal effect: volume predicts price rise → LONG
    # Negative causal effect: volume predicts price fall → SHORT
    entry_long  = (causal_z >  z_thresh) & above
    entry_short = (causal_z < -z_thresh) & below

    return _apply_exit_bar(entry_long, entry_short, exit_bar)


def space_Granger_Vol2Price(trial):
    return {
        'granger_window': trial.suggest_int('granger_window', 20, 80),
        'smooth_window':  trial.suggest_int('smooth_window',  4, 20),
        'z_window':       trial.suggest_int('z_window',       20, 60),
        'z_thresh':       trial.suggest_float('z_thresh',      0.8, 2.5),
        'trend_ema':      trial.suggest_int('trend_ema',      30, 100),
        'exit_bar':       trial.suggest_int('exit_bar',        3, 12),
    }


# ===========================================================================
# STRATEGY_EXPORT — Required by optuna_v7.py pipeline
# ===========================================================================

def gen_GM_AdverseSelection_export(df, **p):
    return gen_GM_AdverseSelection(df, **p)

def gen_Amihud_Illiq_Reversion_export(df, **p):
    return gen_Amihud_Illiq_Reversion(df, **p)

def gen_Hasbrouck_InfoShare_export(df, **p):
    return gen_Hasbrouck_InfoShare(df, **p)

def gen_LOB_OFI_Signal_export(df, **p):
    return gen_LOB_OFI_Signal(df, **p)

def gen_Granger_Vol2Price_export(df, **p):
    return gen_Granger_Vol2Price(df, **p)


STRATEGY_EXPORT = [
    {
        'name':  'TV_GM_AdverseSelection',
        'gen':   gen_GM_AdverseSelection_export,
        'space': space_GM_AdverseSelection,
        'timeframes': ['5m', '15m'],
    },
    {
        'name':  'TV_Amihud_Illiq_Reversion',
        'gen':   gen_Amihud_Illiq_Reversion_export,
        'space': space_Amihud_Illiq_Reversion,
        'timeframes': ['5m', '15m'],
    },
    {
        'name':  'TV_Hasbrouck_InfoShare',
        'gen':   gen_Hasbrouck_InfoShare_export,
        'space': space_Hasbrouck_InfoShare,
        'timeframes': ['5m', '15m'],
    },
    {
        'name':  'TV_LOB_OFI_Signal',
        'gen':   gen_LOB_OFI_Signal_export,
        'space': space_LOB_OFI_Signal,
        'timeframes': ['5m', '15m'],
    },
    {
        'name':  'TV_Granger_Vol2Price',
        'gen':   gen_Granger_Vol2Price_export,
        'space': space_Granger_Vol2Price,
        'timeframes': ['5m', '15m'],
    },
]
