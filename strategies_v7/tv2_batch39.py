"""
TV2 BATCH 39 — 26 TradingView Pine v4/v5/v6 strategies (WR>=60% validated)
Real TradingView strategy names. Bidirectional (LONG=1, SHORT=-1, flat=0).
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Standard helpers
# ---------------------------------------------------------------------------

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()
def _rma(s, n): return s.ewm(alpha=1/n, adjust=False).mean()
def _atr(df, n):
    hi, lo, cl = df['high'], df['low'], df['close']
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    return _rma(tr, n)
def _rsi(s, n):
    d = s.diff()
    return 100 - 100/(1 + _rma(d.clip(lower=0), n) / _rma((-d).clip(lower=0), n))
def _wma(s, n):
    w = np.arange(1, n+1)
    return s.rolling(n).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
def _hma(s, n):
    return _wma(2*_wma(s, n//2) - _wma(s, n), int(np.sqrt(n)))
def _supertrend(df, n, mult):
    atr = _atr(df, n)
    hl2 = (df['high'] + df['low']) / 2
    basic_upper = hl2 + mult * atr
    basic_lower = hl2 - mult * atr

    cl = df['close'].values
    bu = basic_upper.values
    bl = basic_lower.values

    final_upper = bu.copy()
    final_lower = bl.copy()
    trend = np.ones(len(cl), dtype=int)

    for i in range(1, len(cl)):
        # Upper band: only tighten (lower) if in uptrend, else reset
        if bu[i] < final_upper[i-1] or cl[i-1] > final_upper[i-1]:
            final_upper[i] = bu[i]
        else:
            final_upper[i] = final_upper[i-1]
        # Lower band: only tighten (raise) if in downtrend, else reset
        if bl[i] > final_lower[i-1] or cl[i-1] < final_lower[i-1]:
            final_lower[i] = bl[i]
        else:
            final_lower[i] = final_lower[i-1]
        # Trend direction
        if trend[i-1] == 1:
            trend[i] = -1 if cl[i] < final_lower[i] else 1
        else:
            trend[i] = 1 if cl[i] > final_upper[i] else -1

    return pd.Series(trend, index=df.index)

# ---------------------------------------------------------------------------
# 1. IK_Grid_Script  (Pine v4, WR=100%)
# Grid: divide price range into equidistant levels.
# Buy at lower level retrace, sell at upper level recovery.
# ---------------------------------------------------------------------------

def gen_IK_Grid_Script(df, grid_levels=8, lookback=50, atr_mult=0.5):
    cl = df['close']
    hi = df['high']
    lo = df['low']
    sig = pd.Series(0, index=df.index)

    roll_hi = hi.rolling(lookback).max()
    roll_lo = lo.rolling(lookback).min()
    rng = roll_hi - roll_lo
    step = rng / grid_levels

    # Current level bucket
    level = ((cl - roll_lo) / step.replace(0, np.nan)).fillna(0).astype(int).clip(0, grid_levels)
    level_prev = level.shift(1)

    # LONG: price drops to lower level (retrace), SHORT: rises to upper level
    long_cond = (level < level_prev) & (level <= grid_levels // 3)
    short_cond = (level > level_prev) & (level >= grid_levels * 2 // 3)

    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

space_IK_Grid_Script = {
    'grid_levels': ('int', 4, 16),
    'lookback':    ('int', 20, 100),
    'atr_mult':    ('float', 0.3, 1.5),
}

# ---------------------------------------------------------------------------
# 2. CoRelation_StDev_Strategy  (Pine v4, WR=100%)
# Uses correlation and std dev between close and SMA to detect momentum extremes.
# ---------------------------------------------------------------------------

def gen_CoRelation_StDev_Strategy(df, sma_len=20, corr_len=20, std_mult=1.5):
    cl = df['close']
    sma = _sma(cl, sma_len)
    diff = cl - sma

    rolling_std = diff.rolling(corr_len).std()
    rolling_mean = diff.rolling(corr_len).mean()

    # Z-score of (close - sma) deviation
    z = (diff - rolling_mean) / rolling_std.replace(0, np.nan)

    sig = pd.Series(0, index=df.index)
    # Extreme positive momentum: z > std_mult → SHORT (mean reversion)
    # Extreme negative momentum: z < -std_mult → LONG
    sig[z < -std_mult] = 1
    sig[z > std_mult] = -1
    return sig

space_CoRelation_StDev_Strategy = {
    'sma_len':   ('int', 10, 50),
    'corr_len':  ('int', 10, 40),
    'std_mult':  ('float', 1.0, 3.0),
}

# ---------------------------------------------------------------------------
# 3. RSI_Strategy_Professional  (Pine v6, WR=86%)
# RSI with dynamic zones adjusted by ATR volatility.
# Signal when RSI crosses zone ± 1 ATR-based offset.
# ---------------------------------------------------------------------------

def gen_RSI_Strategy_Professional(df, rsi_len=14, atr_len=14, ob=70, os=30, atr_mult=1.0):
    cl = df['close']
    rsi = _rsi(cl, rsi_len)
    atr = _atr(df, atr_len)
    # Normalize ATR to RSI scale (atr as % of price → scale to 0-100 RSI space)
    atr_pct = (atr / cl) * 100
    dyn_offset = atr_pct * atr_mult

    upper = ob + dyn_offset
    lower = os - dyn_offset

    sig = pd.Series(0, index=df.index)
    sig[rsi < lower] = 1
    sig[rsi > upper] = -1
    return sig

space_RSI_Strategy_Professional = {
    'rsi_len':  ('int', 7, 21),
    'atr_len':  ('int', 7, 21),
    'ob':       ('int', 65, 80),
    'os':       ('int', 20, 35),
    'atr_mult': ('float', 0.5, 3.0),
}

# ---------------------------------------------------------------------------
# 4. Orion_Algo_v2  (Pine v4, WR=85%)
# EMA trend filter + Stoch RSI entry + ATR trailing as signal.
# ---------------------------------------------------------------------------

def gen_Orion_Algo_v2(df, ema_len=50, stoch_len=14, stoch_smooth=3,
                       stoch_ob=80, stoch_os=20, atr_len=14):
    cl = df['close']
    hi = df['high']
    lo = df['low']

    trend_ema = _ema(cl, ema_len)

    # Stoch RSI
    rsi = _rsi(cl, stoch_len)
    rsi_min = rsi.rolling(stoch_len).min()
    rsi_max = rsi.rolling(stoch_len).max()
    stoch_rsi_raw = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100
    stoch_k = _sma(stoch_rsi_raw, stoch_smooth)
    stoch_d = _sma(stoch_k, stoch_smooth)

    # ATR trailing — used as confirmation
    atr = _atr(df, atr_len)
    atr_trail = cl - atr

    bull_trend = cl > trend_ema
    bear_trend = cl < trend_ema

    # LONG: bull trend + stoch RSI crosses up from oversold
    long_cross = (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)) & (stoch_k < stoch_ob)
    # SHORT: bear trend + stoch RSI crosses down from overbought
    short_cross = (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1)) & (stoch_k > stoch_os)

    sig = pd.Series(0, index=df.index)
    sig[bull_trend & long_cross] = 1
    sig[bear_trend & short_cross] = -1
    return sig

space_Orion_Algo_v2 = {
    'ema_len':      ('int', 20, 100),
    'stoch_len':    ('int', 7, 21),
    'stoch_smooth': ('int', 2, 5),
    'stoch_ob':     ('int', 70, 90),
    'stoch_os':     ('int', 10, 30),
    'atr_len':      ('int', 7, 21),
}

# ---------------------------------------------------------------------------
# 5. Mean_Reversion_VF  (Pine v4, WR=84%)
# price > 2 ATR from SMA → SHORT, price < 2 ATR from SMA → LONG
# ---------------------------------------------------------------------------

def gen_Mean_Reversion_VF(df, sma_len=20, atr_len=14, atr_mult=2.0):
    cl = df['close']
    sma = _sma(cl, sma_len)
    atr = _atr(df, atr_len)

    upper_band = sma + atr_mult * atr
    lower_band = sma - atr_mult * atr

    sig = pd.Series(0, index=df.index)
    sig[cl < lower_band] = 1    # LONG: price too far below SMA
    sig[cl > upper_band] = -1   # SHORT: price too far above SMA
    return sig

space_Mean_Reversion_VF = {
    'sma_len':  ('int', 10, 50),
    'atr_len':  ('int', 7, 21),
    'atr_mult': ('float', 1.0, 3.5),
}

# ---------------------------------------------------------------------------
# 6. Antigravity_OCC  (Pine v5, WR=84%)
# Oscillator Crossover Confirmation: fast MA cross + Delayed TSI confirmation.
# ---------------------------------------------------------------------------

def gen_Antigravity_OCC(df, fast_len=5, slow_len=20, tsi_fast=13,
                         tsi_slow=25, tsi_signal=13, delay=2):
    cl = df['close']

    fast_ma = _ema(cl, fast_len)
    slow_ma = _ema(cl, slow_len)

    # True Strength Index (TSI)
    mtm = cl.diff(1)
    abs_mtm = mtm.abs()
    tsi_num = _ema(_ema(mtm, tsi_fast), tsi_slow)
    tsi_den = _ema(_ema(abs_mtm, tsi_fast), tsi_slow)
    tsi = 100 * tsi_num / tsi_den.replace(0, np.nan)
    tsi_sig = _ema(tsi, tsi_signal)

    # MA cross
    ma_cross_up   = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    ma_cross_down = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

    # TSI confirmation (delayed by `delay` bars)
    tsi_bull = tsi > tsi_sig
    tsi_bear = tsi < tsi_sig

    sig = pd.Series(0, index=df.index)
    for i in range(delay, len(df)):
        if ma_cross_up.iloc[i - delay] and tsi_bull.iloc[i]:
            sig.iloc[i] = 1
        elif ma_cross_down.iloc[i - delay] and tsi_bear.iloc[i]:
            sig.iloc[i] = -1
    return sig

space_Antigravity_OCC = {
    'fast_len':   ('int', 3, 10),
    'slow_len':   ('int', 15, 50),
    'tsi_fast':   ('int', 5, 20),
    'tsi_slow':   ('int', 15, 35),
    'tsi_signal': ('int', 5, 20),
    'delay':      ('int', 1, 5),
}

# ---------------------------------------------------------------------------
# 7. RSI_Strategy_Simple  (Pine v5, WR=80%)
# RSI 14 level-50 crossover + volume confirmation.
# ---------------------------------------------------------------------------

def gen_RSI_Strategy_Simple(df, rsi_len=14, vol_mult=1.2):
    cl = df['close']
    vol = df['volume']
    rsi = _rsi(cl, rsi_len)
    avg_vol = _sma(vol, 20)

    vol_confirm = vol > avg_vol * vol_mult

    cross_up   = (rsi > 50) & (rsi.shift(1) <= 50)
    cross_down = (rsi < 50) & (rsi.shift(1) >= 50)

    sig = pd.Series(0, index=df.index)
    sig[cross_up   & vol_confirm] = 1
    sig[cross_down & vol_confirm] = -1
    return sig

space_RSI_Strategy_Simple = {
    'rsi_len':  ('int', 7, 21),
    'vol_mult': ('float', 1.0, 2.5),
}

# ---------------------------------------------------------------------------
# 8. Supertrend_Furkan  (Pine v4, WR=78%)
# Classic SuperTrend with optimized params, bidirectional.
# ---------------------------------------------------------------------------

def gen_Supertrend_Furkan(df, atr_len=10, mult=3.0):
    trend = _supertrend(df, atr_len, mult)
    sig = pd.Series(0, index=df.index)
    sig[trend == 1] = 1
    sig[trend == -1] = -1
    return sig

space_Supertrend_Furkan = {
    'atr_len': ('int', 5, 20),
    'mult':    ('float', 1.5, 5.0),
}

# ---------------------------------------------------------------------------
# 9. Super8_30M_BTC  (Pine v5, WR=76%)
# 8 EMAs (8,13,21,34,55,89,144,233): entry when all aligned same direction.
# ---------------------------------------------------------------------------

def gen_Super8_30M_BTC(df, e1=8, e2=13, e3=21, e4=34, e5=55, e6=89, e7=144, e8=233):
    cl = df['close']
    emas = [_ema(cl, n) for n in [e1, e2, e3, e4, e5, e6, e7, e8]]

    # All aligned up: each EMA > next longer EMA
    all_bull = emas[0] > emas[1]
    all_bear = emas[0] < emas[1]
    for i in range(1, len(emas) - 1):
        all_bull = all_bull & (emas[i] > emas[i+1])
        all_bear = all_bear & (emas[i] < emas[i+1])

    # Signal on transition
    sig = pd.Series(0, index=df.index)
    sig[all_bull] = 1
    sig[all_bear] = -1
    return sig

space_Super8_30M_BTC = {
    'e1': ('int', 5, 13),
    'e2': ('int', 10, 21),
    'e3': ('int', 15, 34),
    'e4': ('int', 25, 55),
    'e5': ('int', 40, 89),
    'e6': ('int', 60, 120),
    'e7': ('int', 100, 200),
    'e8': ('int', 150, 300),
}

# ---------------------------------------------------------------------------
# 10. Fifty_Pips_Kaspricci  (Pine v5, WR=75%)
# Bollinger Bands + RSI: entry at extreme band + RSI confirms, target opposite band.
# ---------------------------------------------------------------------------

def gen_Fifty_Pips_Kaspricci(df, bb_len=20, bb_mult=2.0, rsi_len=14,
                               rsi_ob=65, rsi_os=35):
    cl = df['close']
    mid = _sma(cl, bb_len)
    std = cl.rolling(bb_len).std()
    upper = mid + bb_mult * std
    lower = mid - bb_mult * std

    rsi = _rsi(cl, rsi_len)

    sig = pd.Series(0, index=df.index)
    # LONG: close near/below lower band + RSI oversold
    sig[(cl <= lower) & (rsi < rsi_os)] = 1
    # SHORT: close near/above upper band + RSI overbought
    sig[(cl >= upper) & (rsi > rsi_ob)] = -1
    return sig

space_Fifty_Pips_Kaspricci = {
    'bb_len':  ('int', 10, 30),
    'bb_mult': ('float', 1.5, 3.0),
    'rsi_len': ('int', 7, 21),
    'rsi_ob':  ('int', 60, 80),
    'rsi_os':  ('int', 20, 40),
}

# ---------------------------------------------------------------------------
# 11. LowFinder_PyRaMider_v2  (Pine v6, WR=74%)
# Detects local lows with pivot points, pyramid entries in accumulation.
# ---------------------------------------------------------------------------

def gen_LowFinder_PyRaMider_v2(df, pivot_len=5, atr_len=14, atr_mult=0.5):
    cl = df['close']
    hi = df['high']
    lo = df['low']
    atr = _atr(df, atr_len)

    # Pivot low: lowest in window
    pivot_low  = lo.rolling(2 * pivot_len + 1, center=True).min()
    pivot_high = hi.rolling(2 * pivot_len + 1, center=True).max()

    is_pivot_low  = (lo == pivot_low)
    is_pivot_high = (hi == pivot_high)

    sig = pd.Series(0, index=df.index)
    # LONG on pivot low confirmation (price bouncing from local low)
    sig[is_pivot_low & (cl > lo + atr_mult * atr)] = 1
    # SHORT on pivot high confirmation
    sig[is_pivot_high & (cl < hi - atr_mult * atr)] = -1
    return sig

space_LowFinder_PyRaMider_v2 = {
    'pivot_len': ('int', 3, 10),
    'atr_len':   ('int', 7, 21),
    'atr_mult':  ('float', 0.2, 1.5),
}

# ---------------------------------------------------------------------------
# 12. Best_TV_Strategy_NASDAQ  (Pine v5, WR=71%)
# Triple EMA (9,21,50) + RSI + MACD confluence.
# ---------------------------------------------------------------------------

def gen_Best_TV_Strategy_NASDAQ(df, ema_fast=9, ema_mid=21, ema_slow=50,
                                  rsi_len=14, rsi_ob=60, rsi_os=40,
                                  macd_fast=12, macd_slow=26, macd_signal=9):
    cl = df['close']

    ef = _ema(cl, ema_fast)
    em = _ema(cl, ema_mid)
    es = _ema(cl, ema_slow)

    rsi = _rsi(cl, rsi_len)

    macd_line   = _ema(cl, macd_fast) - _ema(cl, macd_slow)
    macd_sig    = _ema(macd_line, macd_signal)
    macd_hist   = macd_line - macd_sig

    bull_ema  = (ef > em) & (em > es)
    bear_ema  = (ef < em) & (em < es)
    bull_rsi  = rsi > rsi_os
    bear_rsi  = rsi < rsi_ob
    bull_macd = macd_hist > 0
    bear_macd = macd_hist < 0

    sig = pd.Series(0, index=df.index)
    sig[bull_ema & bull_rsi & bull_macd] = 1
    sig[bear_ema & bear_rsi & bear_macd] = -1
    return sig

space_Best_TV_Strategy_NASDAQ = {
    'ema_fast':    ('int', 5, 15),
    'ema_mid':     ('int', 15, 30),
    'ema_slow':    ('int', 40, 100),
    'rsi_len':     ('int', 7, 21),
    'rsi_ob':      ('int', 55, 75),
    'rsi_os':      ('int', 25, 45),
    'macd_fast':   ('int', 8, 16),
    'macd_slow':   ('int', 20, 30),
    'macd_signal': ('int', 7, 12),
}

# ---------------------------------------------------------------------------
# 13. Two_HL_Strategy  (Pine v5, WR=70%)
# 2 Higher Lows (bull) / 2 Lower Highs (bear) pattern + volume confirmation.
# ---------------------------------------------------------------------------

def gen_Two_HL_Strategy(df, vol_mult=1.1):
    lo = df['low']
    hi = df['high']
    vol = df['volume']
    avg_vol = _sma(vol, 20)

    vol_ok = vol > avg_vol * vol_mult

    # 2 higher lows: lo[i] > lo[i-1] > lo[i-2]
    hl2 = (lo > lo.shift(1)) & (lo.shift(1) > lo.shift(2))
    # 2 lower highs: hi[i] < hi[i-1] < hi[i-2]
    lh2 = (hi < hi.shift(1)) & (hi.shift(1) < hi.shift(2))

    sig = pd.Series(0, index=df.index)
    sig[hl2 & vol_ok] = 1
    sig[lh2 & vol_ok] = -1
    return sig

space_Two_HL_Strategy = {
    'vol_mult': ('float', 0.8, 2.5),
}

# ---------------------------------------------------------------------------
# 14. Volume_LinReg_Trend  (Pine v4, WR=69%)
# Volume-weighted linear regression of price; signal on cross with price.
# ---------------------------------------------------------------------------

def gen_Volume_LinReg_Trend(df, linreg_len=20):
    cl = df['close']
    vol = df['volume']

    # Volume-weighted price (VWAP-like rolling)
    vwp = (cl * vol).rolling(linreg_len).sum() / vol.rolling(linreg_len).sum().replace(0, np.nan)

    # Linear regression of vwp
    def linreg_val(x):
        n = len(x)
        if np.isnan(x).any():
            return np.nan
        xs = np.arange(n)
        m, b = np.polyfit(xs, x, 1)
        return m * (n - 1) + b  # value at last bar

    linreg = vwp.rolling(linreg_len).apply(linreg_val, raw=True)

    sig = pd.Series(0, index=df.index)
    cross_up   = (cl > linreg) & (cl.shift(1) <= linreg.shift(1))
    cross_down = (cl < linreg) & (cl.shift(1) >= linreg.shift(1))
    sig[cross_up]   = 1
    sig[cross_down] = -1
    return sig

space_Volume_LinReg_Trend = {
    'linreg_len': ('int', 10, 50),
}

# ---------------------------------------------------------------------------
# 15. TwoMars_MA_BB_ST  (Pine v4, WR=68%)
# Triple combo: MA cross + BB breakout + SuperTrend all aligned.
# ---------------------------------------------------------------------------

def gen_TwoMars_MA_BB_ST(df, fast_ma=10, slow_ma=30, bb_len=20, bb_mult=2.0,
                           st_len=10, st_mult=3.0):
    cl = df['close']

    fma = _ema(cl, fast_ma)
    sma = _ema(cl, slow_ma)

    mid = _sma(cl, bb_len)
    std = cl.rolling(bb_len).std()
    bb_upper = mid + bb_mult * std
    bb_lower = mid - bb_mult * std

    trend = _supertrend(df, st_len, st_mult)

    ma_bull = fma > sma
    ma_bear = fma < sma
    bb_bull = cl > mid       # above midline
    bb_bear = cl < mid
    st_bull = trend == 1
    st_bear = trend == -1

    sig = pd.Series(0, index=df.index)
    sig[ma_bull & bb_bull & st_bull] = 1
    sig[ma_bear & bb_bear & st_bear] = -1
    return sig

space_TwoMars_MA_BB_ST = {
    'fast_ma':  ('int', 5, 20),
    'slow_ma':  ('int', 20, 60),
    'bb_len':   ('int', 10, 30),
    'bb_mult':  ('float', 1.5, 3.0),
    'st_len':   ('int', 5, 20),
    'st_mult':  ('float', 1.5, 5.0),
}

# ---------------------------------------------------------------------------
# 16. Impulse_Strategy_v2  (Pine v4, WR=67%)
# Impulse System (Elder): EMA slope + MACD histogram direction.
# Both bullish → LONG, both bearish → SHORT.
# ---------------------------------------------------------------------------

def gen_Impulse_Strategy_v2(df, ema_len=13, macd_fast=12, macd_slow=26, macd_signal=9):
    cl = df['close']

    ema = _ema(cl, ema_len)
    ema_slope_up   = ema > ema.shift(1)
    ema_slope_down = ema < ema.shift(1)

    macd_line = _ema(cl, macd_fast) - _ema(cl, macd_slow)
    macd_sig  = _ema(macd_line, macd_signal)
    hist      = macd_line - macd_sig
    hist_up   = hist > hist.shift(1)
    hist_down = hist < hist.shift(1)

    sig = pd.Series(0, index=df.index)
    sig[ema_slope_up   & hist_up]   = 1
    sig[ema_slope_down & hist_down] = -1
    return sig

space_Impulse_Strategy_v2 = {
    'ema_len':     ('int', 7, 26),
    'macd_fast':   ('int', 8, 16),
    'macd_slow':   ('int', 20, 30),
    'macd_signal': ('int', 7, 12),
}

# ---------------------------------------------------------------------------
# 17. Range_Trading_Strategy  (Pine v4, WR=67%)
# Identifies range using historical ATR; trades bounces at range extremes.
# ---------------------------------------------------------------------------

def gen_Range_Trading_Strategy(df, range_len=50, atr_len=14, atr_mult=1.5):
    cl = df['close']
    hi = df['high']
    lo = df['low']
    atr = _atr(df, atr_len)

    range_hi = hi.rolling(range_len).max()
    range_lo = lo.rolling(range_len).min()
    range_mid = (range_hi + range_lo) / 2

    # Narrow range: ATR / range_size < threshold (we're in a ranging market)
    range_size = (range_hi - range_lo).replace(0, np.nan)
    is_range = (atr / range_size) < 0.3

    sig = pd.Series(0, index=df.index)
    near_lo = cl < (range_lo + atr * atr_mult)
    near_hi = cl > (range_hi - atr * atr_mult)

    sig[is_range & near_lo] = 1     # LONG at range bottom
    sig[is_range & near_hi] = -1    # SHORT at range top
    return sig

space_Range_Trading_Strategy = {
    'range_len': ('int', 20, 100),
    'atr_len':   ('int', 7, 21),
    'atr_mult':  ('float', 0.5, 3.0),
}

# ---------------------------------------------------------------------------
# 18. Larry_Connors_Strategy  (Pine v4, WR=66%)
# Connors RSI: CRSI = (RSI(3) + UpDown streak + ROC percentile) / 3
# ---------------------------------------------------------------------------

def gen_Larry_Connors_Strategy(df, rsi_short=3, streak_len=2,
                                 pct_rank_len=100, ob=70, os=30):
    cl = df['close']

    r3 = _rsi(cl, rsi_short)

    # UpDown streak
    diff = cl.diff()
    streak = pd.Series(0.0, index=df.index)
    streak_arr = streak.values
    diff_arr = diff.values
    for i in range(1, len(diff_arr)):
        if np.isnan(diff_arr[i]):
            streak_arr[i] = 0
        elif diff_arr[i] > 0:
            streak_arr[i] = max(streak_arr[i-1], 0) + 1
        elif diff_arr[i] < 0:
            streak_arr[i] = min(streak_arr[i-1], 0) - 1
        else:
            streak_arr[i] = 0
    streak = pd.Series(streak_arr, index=df.index)

    # Streak RSI
    d = streak.diff()
    streak_rsi = 100 - 100 / (1 + _rma(d.clip(lower=0), streak_len) /
                               _rma((-d).clip(lower=0), streak_len))

    # ROC percentile rank
    roc = cl.pct_change(1) * 100
    pct_rank = roc.rolling(pct_rank_len).apply(
        lambda x: (x[:-1] < x[-1]).sum() / (len(x)-1) * 100 if len(x) > 1 else 50,
        raw=True
    )

    crsi = (r3 + streak_rsi + pct_rank) / 3

    sig = pd.Series(0, index=df.index)
    sig[crsi < os] = 1
    sig[crsi > ob] = -1
    return sig

space_Larry_Connors_Strategy = {
    'rsi_short':     ('int', 2, 5),
    'streak_len':    ('int', 2, 5),
    'pct_rank_len':  ('int', 50, 200),
    'ob':            ('int', 65, 85),
    'os':            ('int', 15, 35),
}

# ---------------------------------------------------------------------------
# 19. RSI_VWAP_Upgraded  (Pine v6, WR=65%)
# RSI calculated on VWAP instead of close; signals on 30/70 crossovers.
# ---------------------------------------------------------------------------

def gen_RSI_VWAP_Upgraded(df, rsi_len=14, ob=70, os=30):
    cl = df['close']
    hi = df['high']
    lo = df['low']
    vol = df['volume']

    tp = (hi + lo + cl) / 3
    vwap_roll = (tp * vol).rolling(rsi_len * 2).sum() / vol.rolling(rsi_len * 2).sum().replace(0, np.nan)

    rsi_vwap = _rsi(vwap_roll, rsi_len)

    sig = pd.Series(0, index=df.index)
    cross_up   = (rsi_vwap > os) & (rsi_vwap.shift(1) <= os)
    cross_down = (rsi_vwap < ob) & (rsi_vwap.shift(1) >= ob)
    sig[cross_up]   = 1
    sig[cross_down] = -1
    return sig

space_RSI_VWAP_Upgraded = {
    'rsi_len': ('int', 7, 21),
    'ob':      ('int', 65, 80),
    'os':      ('int', 20, 35),
}

# ---------------------------------------------------------------------------
# 20. HatiKO_Envelopes  (Pine v6, WR=64%)
# Adaptive envelopes (EMA ± k×ATR), entry on retrace to band + momentum.
# ---------------------------------------------------------------------------

def gen_HatiKO_Envelopes(df, ema_len=20, atr_len=14, k=1.5, mom_len=3):
    cl = df['close']
    ema = _ema(cl, ema_len)
    atr = _atr(df, atr_len)

    upper = ema + k * atr
    lower = ema - k * atr

    # Momentum confirmation: current bar closing higher/lower than previous bar
    mom_up   = cl > cl.shift(1)   # bar closed up  → recovering from low
    mom_down = cl < cl.shift(1)   # bar closed down → extending from high

    sig = pd.Series(0, index=df.index)
    # LONG: price touches/crosses below lower envelope, then first up bar
    touched_lower = (cl.shift(1) <= lower.shift(1))
    touched_upper = (cl.shift(1) >= upper.shift(1))
    sig[touched_lower & mom_up]   = 1
    sig[touched_upper & mom_down] = -1
    return sig

space_HatiKO_Envelopes = {
    'ema_len': ('int', 10, 50),
    'atr_len': ('int', 7, 21),
    'k':       ('float', 1.0, 4.0),
    'mom_len': ('int', 2, 10),
}

# ---------------------------------------------------------------------------
# 21. Mean_Reversion_v2_KL  (Pine v6, WR=64%)
# Z-score of price vs SMA50; LONG when z < -2, SHORT when z > +2.
# ---------------------------------------------------------------------------

def gen_Mean_Reversion_v2_KL(df, sma_len=50, z_entry=2.0, z_smooth=3):
    cl = df['close']
    sma = _sma(cl, sma_len)
    std = cl.rolling(sma_len).std()
    z = (cl - sma) / std.replace(0, np.nan)
    z_s = _sma(z, z_smooth)

    sig = pd.Series(0, index=df.index)
    sig[z_s < -z_entry] = 1
    sig[z_s > z_entry]  = -1
    return sig

space_Mean_Reversion_v2_KL = {
    'sma_len':  ('int', 20, 100),
    'z_entry':  ('float', 1.0, 3.5),
    'z_smooth': ('int', 1, 7),
}

# ---------------------------------------------------------------------------
# 22. Chande_CMO_Strategy  (Pine v4, WR=63%)
# Chande Momentum Oscillator: (sum_up - sum_dn) / (sum_up + sum_dn) × 100
# Signal at ±50 threshold.
# ---------------------------------------------------------------------------

def gen_Chande_CMO_Strategy(df, cmo_len=14, ob=50, os=-50):
    cl = df['close']
    diff = cl.diff()
    up   = diff.clip(lower=0)
    dn   = (-diff).clip(lower=0)

    sum_up = up.rolling(cmo_len).sum()
    sum_dn = dn.rolling(cmo_len).sum()
    cmo = 100 * (sum_up - sum_dn) / (sum_up + sum_dn).replace(0, np.nan)

    sig = pd.Series(0, index=df.index)
    cross_up   = (cmo > os) & (cmo.shift(1) <= os)
    cross_down = (cmo < ob) & (cmo.shift(1) >= ob)
    sig[cross_up]   = 1
    sig[cross_down] = -1
    return sig

space_Chande_CMO_Strategy = {
    'cmo_len': ('int', 7, 30),
    'ob':      ('int', 30, 70),
    'os':      ('int', -70, -30),
}

# ---------------------------------------------------------------------------
# 23. Vix_Fix_StochRSI  (Pine v4, WR=63%)
# Vix Fix (Williams %R on High) + Stoch RSI: both at extreme → entry.
# ---------------------------------------------------------------------------

def gen_Vix_Fix_StochRSI(df, vf_len=22, rsi_len=14, stoch_len=14,
                           stoch_smooth=3, vf_thresh=0.3, stoch_os=20, stoch_ob=80):
    cl = df['close']
    hi = df['high']

    # Vix Fix: synthetic VIX — measures how far we are from recent high
    highest_cl = cl.rolling(vf_len).max()
    vix_fix = (highest_cl - cl) / highest_cl.replace(0, np.nan) * 100

    # Stoch RSI
    rsi = _rsi(cl, rsi_len)
    rsi_min = rsi.rolling(stoch_len).min()
    rsi_max = rsi.rolling(stoch_len).max()
    stoch_raw = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100
    stoch_k = _sma(stoch_raw, stoch_smooth)

    vf_extreme_low  = vix_fix > vf_thresh * 100 * 0.3    # high vix fix = fear = potential LONG
    stoch_os_cond   = stoch_k < stoch_os
    stoch_ob_cond   = stoch_k > stoch_ob
    vf_low_val      = vix_fix < vf_thresh * 10            # low vix fix = complacency = potential SHORT

    sig = pd.Series(0, index=df.index)
    sig[vf_extreme_low & stoch_os_cond]  = 1
    sig[vf_low_val     & stoch_ob_cond]  = -1
    return sig

space_Vix_Fix_StochRSI = {
    'vf_len':       ('int', 10, 40),
    'rsi_len':      ('int', 7, 21),
    'stoch_len':    ('int', 7, 21),
    'stoch_smooth': ('int', 2, 5),
    'vf_thresh':    ('float', 0.1, 0.5),
    'stoch_os':     ('int', 10, 30),
    'stoch_ob':     ('int', 70, 90),
}

# ---------------------------------------------------------------------------
# 24. Advanced_Position_Mgmt  (Pine v4, WR=61%)
# Multiple staggered entries based on ATR offsets from EMA.
# ---------------------------------------------------------------------------

def gen_Advanced_Position_Mgmt(df, ema_len=21, atr_len=14,
                                 entry1_mult=0.5, entry2_mult=1.0, entry3_mult=1.5):
    cl = df['close']
    ema = _ema(cl, ema_len)
    atr = _atr(df, atr_len)

    # Broader trend: longer-term EMA slope over several bars
    trend_window = max(ema_len // 3, 3)
    ema_bull = ema > ema.shift(trend_window)
    ema_bear = ema < ema.shift(trend_window)

    # Entry levels: price extends beyond EMA ± n*ATR (extension entries)
    long1  = cl < ema - entry1_mult * atr
    long2  = cl < ema - entry2_mult * atr
    long3  = cl < ema - entry3_mult * atr
    short1 = cl > ema + entry1_mult * atr
    short2 = cl > ema + entry2_mult * atr
    short3 = cl > ema + entry3_mult * atr

    # Signal — deepest reached level wins (assign all, last write wins per bar)
    sig = pd.Series(0, index=df.index)
    # In bull trend: buy pullbacks below EMA
    sig[ema_bull & long1]  = 1
    sig[ema_bull & long2]  = 1
    sig[ema_bull & long3]  = 1
    # In bear trend: sell extensions above EMA
    sig[ema_bear & short1] = -1
    sig[ema_bear & short2] = -1
    sig[ema_bear & short3] = -1
    # When trend is unclear, allow mean-reversion in both directions
    neutral = ~ema_bull & ~ema_bear
    sig[neutral & long2]  = 1
    sig[neutral & short2] = -1
    return sig

space_Advanced_Position_Mgmt = {
    'ema_len':      ('int', 10, 50),
    'atr_len':      ('int', 7, 21),
    'entry1_mult':  ('float', 0.2, 1.0),
    'entry2_mult':  ('float', 0.5, 2.0),
    'entry3_mult':  ('float', 1.0, 3.0),
}

# ---------------------------------------------------------------------------
# 25. Entry_Fragger  (Pine v5, WR=61%)
# Volume Spread Analysis: high-volume bull/bear bar as directional entry signal.
# ---------------------------------------------------------------------------

def gen_Entry_Fragger(df, vol_mult=1.2, body_pct=0.3):
    cl = df['close']
    op = df['open']
    hi = df['high']
    lo = df['low']
    vol = df['volume']

    avg_vol = _sma(vol, 20)
    high_vol = vol > avg_vol * vol_mult

    bar_range = (hi - lo).replace(0, np.nan)
    body      = (cl - op).abs()
    body_ratio = body / bar_range

    bull_bar = (cl > op) & (body_ratio > body_pct)   # directional bull close
    bear_bar = (cl < op) & (body_ratio > body_pct)   # directional bear close

    sig = pd.Series(0, index=df.index)
    sig[high_vol & bull_bar] = 1
    sig[high_vol & bear_bar] = -1
    return sig

space_Entry_Fragger = {
    'vol_mult':  ('float', 1.0, 3.0),
    'body_pct':  ('float', 0.2, 0.8),
}

# ---------------------------------------------------------------------------
# 26. Connors_RSI_Composite  (Pine v4, WR=61%)
# CRSI = (RSI(3) + UpDown_streak_RSI + Percentile_Rank(ROC,100)) / 3
# ---------------------------------------------------------------------------

def gen_Connors_RSI_Composite(df, rsi_len=3, streak_rsi_len=2,
                                pct_rank_len=100, ob=70, os=30):
    cl = df['close']

    r = _rsi(cl, rsi_len)

    # Consecutive up/down streak
    diff = cl.diff()
    streak_arr = np.zeros(len(cl))
    diff_arr = diff.values
    for i in range(1, len(diff_arr)):
        if np.isnan(diff_arr[i]):
            streak_arr[i] = 0
        elif diff_arr[i] > 0:
            streak_arr[i] = max(streak_arr[i-1], 0) + 1
        elif diff_arr[i] < 0:
            streak_arr[i] = min(streak_arr[i-1], 0) - 1
        else:
            streak_arr[i] = 0
    streak = pd.Series(streak_arr, index=df.index)

    d_streak = streak.diff()
    ud_rsi = 100 - 100 / (1 + _rma(d_streak.clip(lower=0), streak_rsi_len) /
                           _rma((-d_streak).clip(lower=0), streak_rsi_len))

    # Percentile rank of 1-bar ROC
    roc = cl.pct_change(1) * 100
    pct_rank = roc.rolling(pct_rank_len).apply(
        lambda x: (x[:-1] < x[-1]).sum() / max(len(x)-1, 1) * 100,
        raw=True
    )

    crsi = (r + ud_rsi + pct_rank) / 3

    sig = pd.Series(0, index=df.index)
    sig[crsi < os] = 1
    sig[crsi > ob] = -1
    return sig

space_Connors_RSI_Composite = {
    'rsi_len':        ('int', 2, 5),
    'streak_rsi_len': ('int', 2, 5),
    'pct_rank_len':   ('int', 50, 200),
    'ob':             ('int', 65, 85),
    'os':             ('int', 15, 35),
}

# ---------------------------------------------------------------------------
# STRATEGY_EXPORT  — direct function references (pickle-safe, no lambdas)
# ---------------------------------------------------------------------------

STRATEGY_EXPORT = {
    'IK_Grid_Script': {
        'gen': gen_IK_Grid_Script,
        'space': space_IK_Grid_Script,
    },
    'CoRelation_StDev_Strategy': {
        'gen': gen_CoRelation_StDev_Strategy,
        'space': space_CoRelation_StDev_Strategy,
    },
    'RSI_Strategy_Professional': {
        'gen': gen_RSI_Strategy_Professional,
        'space': space_RSI_Strategy_Professional,
    },
    'Orion_Algo_v2': {
        'gen': gen_Orion_Algo_v2,
        'space': space_Orion_Algo_v2,
    },
    'Mean_Reversion_VF': {
        'gen': gen_Mean_Reversion_VF,
        'space': space_Mean_Reversion_VF,
    },
    'Antigravity_OCC': {
        'gen': gen_Antigravity_OCC,
        'space': space_Antigravity_OCC,
    },
    'RSI_Strategy_Simple': {
        'gen': gen_RSI_Strategy_Simple,
        'space': space_RSI_Strategy_Simple,
    },
    'Supertrend_Furkan': {
        'gen': gen_Supertrend_Furkan,
        'space': space_Supertrend_Furkan,
    },
    'Super8_30M_BTC': {
        'gen': gen_Super8_30M_BTC,
        'space': space_Super8_30M_BTC,
    },
    'Fifty_Pips_Kaspricci': {
        'gen': gen_Fifty_Pips_Kaspricci,
        'space': space_Fifty_Pips_Kaspricci,
    },
    'LowFinder_PyRaMider_v2': {
        'gen': gen_LowFinder_PyRaMider_v2,
        'space': space_LowFinder_PyRaMider_v2,
    },
    'Best_TV_Strategy_NASDAQ': {
        'gen': gen_Best_TV_Strategy_NASDAQ,
        'space': space_Best_TV_Strategy_NASDAQ,
    },
    'Two_HL_Strategy': {
        'gen': gen_Two_HL_Strategy,
        'space': space_Two_HL_Strategy,
    },
    'Volume_LinReg_Trend': {
        'gen': gen_Volume_LinReg_Trend,
        'space': space_Volume_LinReg_Trend,
    },
    'TwoMars_MA_BB_ST': {
        'gen': gen_TwoMars_MA_BB_ST,
        'space': space_TwoMars_MA_BB_ST,
    },
    'Impulse_Strategy_v2': {
        'gen': gen_Impulse_Strategy_v2,
        'space': space_Impulse_Strategy_v2,
    },
    'Range_Trading_Strategy': {
        'gen': gen_Range_Trading_Strategy,
        'space': space_Range_Trading_Strategy,
    },
    'Larry_Connors_Strategy': {
        'gen': gen_Larry_Connors_Strategy,
        'space': space_Larry_Connors_Strategy,
    },
    'RSI_VWAP_Upgraded': {
        'gen': gen_RSI_VWAP_Upgraded,
        'space': space_RSI_VWAP_Upgraded,
    },
    'HatiKO_Envelopes': {
        'gen': gen_HatiKO_Envelopes,
        'space': space_HatiKO_Envelopes,
    },
    'Mean_Reversion_v2_KL': {
        'gen': gen_Mean_Reversion_v2_KL,
        'space': space_Mean_Reversion_v2_KL,
    },
    'Chande_CMO_Strategy': {
        'gen': gen_Chande_CMO_Strategy,
        'space': space_Chande_CMO_Strategy,
    },
    'Vix_Fix_StochRSI': {
        'gen': gen_Vix_Fix_StochRSI,
        'space': space_Vix_Fix_StochRSI,
    },
    'Advanced_Position_Mgmt': {
        'gen': gen_Advanced_Position_Mgmt,
        'space': space_Advanced_Position_Mgmt,
    },
    'Entry_Fragger': {
        'gen': gen_Entry_Fragger,
        'space': space_Entry_Fragger,
    },
    'Connors_RSI_Composite': {
        'gen': gen_Connors_RSI_Composite,
        'space': space_Connors_RSI_Composite,
    },
}
