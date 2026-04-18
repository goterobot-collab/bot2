"""
TV2 BATCH 38 — ML-Inspired + Advanced Pattern Recognition (34 strategies)
Real TradingView Pine Script v4/v5/v6 algorithm implementations.
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

# ---------------------------------------------------------------------------
# 1. Nearest_Neighbor_Signal
# ---------------------------------------------------------------------------

def gen_Nearest_Neighbor_Signal(df, k=5, lookback=50, rsi_len=14, cci_len=20):
    cl = df['close']
    hi = df['high']
    lo = df['low']
    rsi = _rsi(cl, rsi_len)
    tp = (hi + lo + cl) / 3
    cci_raw = tp - _sma(tp, cci_len)
    mad = tp.rolling(cci_len).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = cci_raw / (0.015 * mad.replace(0, np.nan))

    sig = pd.Series(0, index=df.index)
    rsi_arr = rsi.values
    cci_arr = cci.values
    cl_arr = cl.values
    n = len(cl_arr)

    for i in range(lookback + 1, n):
        cur_rsi = rsi_arr[i]
        cur_cci = cci_arr[i]
        if np.isnan(cur_rsi) or np.isnan(cur_cci):
            continue
        dists = []
        for j in range(i - lookback, i):
            if np.isnan(rsi_arr[j]) or np.isnan(cci_arr[j]):
                continue
            d = abs(cur_rsi - rsi_arr[j]) + abs(cur_cci - cci_arr[j])
            future = 1 if cl_arr[j+1] > cl_arr[j] else -1
            dists.append((d, future))
        if len(dists) < k:
            continue
        dists.sort(key=lambda x: x[0])
        votes = sum(v for _, v in dists[:k])
        sig.iloc[i] = 1 if votes > 0 else (-1 if votes < 0 else 0)
    return sig

space_Nearest_Neighbor_Signal = {
    'k': ('int', 3, 10),
    'lookback': ('int', 30, 100),
    'rsi_len': ('int', 7, 21),
    'cci_len': ('int', 10, 30),
}

# ---------------------------------------------------------------------------
# 2. Gaussian_Kernel_Trend
# ---------------------------------------------------------------------------

def gen_Gaussian_Kernel_Trend(df, bandwidth=10, fast_len=5, slow_len=20):
    cl = df['close']
    n = len(cl)
    kernel_vals = np.zeros(n)
    cl_arr = cl.values

    for i in range(bandwidth, n):
        weights = np.array([
            np.exp(-0.5 * ((i - j) / bandwidth) ** 2)
            for j in range(i - bandwidth, i + 1)
        ])
        window = cl_arr[i - bandwidth: i + 1]
        kernel_vals[i] = np.dot(weights, window) / weights.sum()

    gk = pd.Series(kernel_vals, index=cl.index)
    gk_fast = _ema(gk, fast_len)
    gk_slow = _ema(gk, slow_len)

    sig = pd.Series(0, index=df.index)
    sig[gk_fast > gk_slow] = 1
    sig[gk_fast < gk_slow] = -1
    return sig

space_Gaussian_Kernel_Trend = {
    'bandwidth': ('int', 5, 30),
    'fast_len': ('int', 3, 10),
    'slow_len': ('int', 10, 40),
}

# ---------------------------------------------------------------------------
# 3. Lorentzian_KNN
# ---------------------------------------------------------------------------

def gen_Lorentzian_KNN(df, k=8, lookback=50, rsi_len=14, wt_len=10):
    cl = df['close']
    hi = df['high']
    lo = df['low']
    rsi = _rsi(cl, rsi_len)
    # Williams %R as second feature
    highest_hi = hi.rolling(wt_len).max()
    lowest_lo = lo.rolling(wt_len).min()
    wr = -100 * (highest_hi - cl) / (highest_hi - lowest_lo).replace(0, np.nan)

    sig = pd.Series(0, index=df.index)
    rsi_arr = rsi.values
    wr_arr = wr.values
    cl_arr = cl.values
    n = len(cl_arr)

    for i in range(lookback + 1, n):
        cur_r = rsi_arr[i]
        cur_w = wr_arr[i]
        if np.isnan(cur_r) or np.isnan(cur_w):
            continue
        dists = []
        for j in range(i - lookback, i):
            if np.isnan(rsi_arr[j]) or np.isnan(wr_arr[j]):
                continue
            # Lorentzian distance: log(1 + |delta|)
            d = (np.log1p(abs(cur_r - rsi_arr[j])) +
                 np.log1p(abs(cur_w - wr_arr[j])))
            future = 1 if cl_arr[j+1] > cl_arr[j] else -1
            dists.append((d, future))
        if len(dists) < k:
            continue
        dists.sort(key=lambda x: x[0])
        votes = sum(v for _, v in dists[:k])
        sig.iloc[i] = 1 if votes > 0 else (-1 if votes < 0 else 0)
    return sig

space_Lorentzian_KNN = {
    'k': ('int', 4, 12),
    'lookback': ('int', 30, 80),
    'rsi_len': ('int', 7, 21),
    'wt_len': ('int', 7, 20),
}

# ---------------------------------------------------------------------------
# 4. Linear_Regression_Slope
# ---------------------------------------------------------------------------

def gen_Linear_Regression_Slope(df, lr_len=20, slope_thresh=0.0, accel_len=5):
    cl = df['close']
    n = len(cl)
    slope = np.full(n, np.nan)
    for i in range(lr_len - 1, n):
        y = cl.values[i - lr_len + 1: i + 1]
        x = np.arange(lr_len)
        coeffs = np.polyfit(x, y, 1)
        slope[i] = coeffs[0]

    slope_s = pd.Series(slope, index=cl.index)
    accel = slope_s - slope_s.shift(accel_len)

    sig = pd.Series(0, index=df.index)
    sig[(slope_s > slope_thresh) & (accel > 0)] = 1
    sig[(slope_s < -slope_thresh) & (accel < 0)] = -1
    return sig

space_Linear_Regression_Slope = {
    'lr_len': ('int', 10, 40),
    'slope_thresh': ('float', 0.0, 0.5),
    'accel_len': ('int', 2, 10),
}

# ---------------------------------------------------------------------------
# 5. Polynomial_Regression_Channel
# ---------------------------------------------------------------------------

def gen_Polynomial_Regression_Channel(df, poly_len=30, dev_mult=2.0):
    cl = df['close']
    n = len(cl)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    fitted = np.full(n, np.nan)

    for i in range(poly_len - 1, n):
        y = cl.values[i - poly_len + 1: i + 1]
        x = np.arange(poly_len)
        coeffs = np.polyfit(x, y, 2)
        yhat = np.polyval(coeffs, x)
        residuals = y - yhat
        std = residuals.std()
        fitted[i] = yhat[-1]
        upper[i] = yhat[-1] + dev_mult * std
        lower[i] = yhat[-1] - dev_mult * std

    fitted_s = pd.Series(fitted, index=cl.index)
    upper_s = pd.Series(upper, index=cl.index)
    lower_s = pd.Series(lower, index=cl.index)

    sig = pd.Series(0, index=df.index)
    sig[(cl > upper_s) & (cl.shift(1) <= upper_s.shift(1))] = 1
    sig[(cl < lower_s) & (cl.shift(1) >= lower_s.shift(1))] = -1
    return sig

space_Polynomial_Regression_Channel = {
    'poly_len': ('int', 15, 50),
    'dev_mult': ('float', 1.0, 3.0),
}

# ---------------------------------------------------------------------------
# 6. LSMA_Crossover
# ---------------------------------------------------------------------------

def gen_LSMA_Crossover(df, fast_len=10, slow_len=30):
    cl = df['close']
    n = len(cl)

    def lsma(series, length):
        vals = np.full(len(series), np.nan)
        arr = series.values
        for i in range(length - 1, len(arr)):
            y = arr[i - length + 1: i + 1]
            x = np.arange(length)
            coeffs = np.polyfit(x, y, 1)
            vals[i] = np.polyval(coeffs, length - 1)
        return pd.Series(vals, index=series.index)

    fast = lsma(cl, fast_len)
    slow = lsma(cl, slow_len)

    sig = pd.Series(0, index=df.index)
    sig[fast > slow] = 1
    sig[fast < slow] = -1
    return sig

space_LSMA_Crossover = {
    'fast_len': ('int', 5, 20),
    'slow_len': ('int', 20, 60),
}

# ---------------------------------------------------------------------------
# 7. Deviation_Channel_ML
# ---------------------------------------------------------------------------

def gen_Deviation_Channel_ML(df, reg_len=30, z_entry=1.5, z_exit=0.5):
    cl = df['close']
    n = len(cl)
    fitted = np.full(n, np.nan)
    for i in range(reg_len - 1, n):
        y = cl.values[i - reg_len + 1: i + 1]
        x = np.arange(reg_len)
        coeffs = np.polyfit(x, y, 1)
        fitted[i] = np.polyval(coeffs, reg_len - 1)

    fitted_s = pd.Series(fitted, index=cl.index)
    deviation = cl - fitted_s
    roll_std = deviation.rolling(reg_len).std()
    z = deviation / roll_std.replace(0, np.nan)

    sig = pd.Series(0, index=df.index)
    # Mean reversion: price far below regression → long
    sig[z < -z_entry] = 1
    sig[z > z_entry] = -1
    return sig

space_Deviation_Channel_ML = {
    'reg_len': ('int', 15, 50),
    'z_entry': ('float', 1.0, 3.0),
    'z_exit': ('float', 0.2, 1.0),
}

# ---------------------------------------------------------------------------
# 8. Gradient_Descent_MA
# ---------------------------------------------------------------------------

def gen_Gradient_Descent_MA(df, ma_len=20, slope_thresh=0.001, curv_thresh=0.0):
    cl = df['close']
    ma = _ema(cl, ma_len)
    slope = ma.diff()
    curvature = slope.diff()

    sig = pd.Series(0, index=df.index)
    sig[(slope > slope_thresh) & (curvature >= curv_thresh)] = 1
    sig[(slope < -slope_thresh) & (curvature <= -curv_thresh)] = -1
    return sig

space_Gradient_Descent_MA = {
    'ma_len': ('int', 10, 50),
    'slope_thresh': ('float', 0.0001, 0.01),
    'curv_thresh': ('float', 0.0, 0.005),
}

# ---------------------------------------------------------------------------
# 9. Adaptive_Lookback_RSI
# ---------------------------------------------------------------------------

def gen_Adaptive_Lookback_RSI(df, base_len=14, min_len=5, max_len=30,
                               ob=70, os=30, cycle_len=20):
    cl = df['close']
    # Estimate cycle via zero-crossings of detrended price
    detrended = cl - _sma(cl, cycle_len)
    signs = np.sign(detrended)
    crossings = (signs != signs.shift(1)).rolling(cycle_len).sum()
    # More crossings → shorter period
    half_cycle = (cycle_len / crossings.replace(0, np.nan)).clip(min_len, max_len).round().fillna(base_len).astype(int)

    rsi_vals = np.full(len(cl), np.nan)
    for i in range(max_len, len(cl)):
        p = int(half_cycle.iloc[i])
        p = max(min_len, min(max_len, p))
        d = cl.diff().iloc[max(0, i - p * 2): i + 1]
        gain = _rma(d.clip(lower=0), p).iloc[-1]
        loss = _rma((-d).clip(lower=0), p).iloc[-1]
        if loss == 0:
            rsi_vals[i] = 100.0
        else:
            rsi_vals[i] = 100 - 100 / (1 + gain / loss)

    rsi = pd.Series(rsi_vals, index=cl.index)
    sig = pd.Series(0, index=df.index)
    sig[rsi < os] = 1
    sig[rsi > ob] = -1
    return sig

space_Adaptive_Lookback_RSI = {
    'base_len': ('int', 7, 21),
    'min_len': ('int', 3, 10),
    'max_len': ('int', 20, 50),
    'ob': ('int', 65, 80),
    'os': ('int', 20, 35),
    'cycle_len': ('int', 10, 40),
}

# ---------------------------------------------------------------------------
# 10. Fractal_Efficiency_MA
# ---------------------------------------------------------------------------

def gen_Fractal_Efficiency_MA(df, fast_len=3, slow_len=30, er_len=10):
    cl = df['close']
    direction = (cl - cl.shift(er_len)).abs()
    volatility = cl.diff().abs().rolling(er_len).sum()
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0).clip(0, 1)

    fast_sc = (2.0 / (fast_len + 1))
    slow_sc = (2.0 / (slow_len + 1))
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = cl.copy().astype(float)
    for i in range(1, len(cl)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (cl.iloc[i] - kama.iloc[i-1])

    sig = pd.Series(0, index=df.index)
    sig[cl > kama] = 1
    sig[cl < kama] = -1
    return sig

space_Fractal_Efficiency_MA = {
    'fast_len': ('int', 2, 5),
    'slow_len': ('int', 20, 60),
    'er_len': ('int', 5, 20),
}

# ---------------------------------------------------------------------------
# 11. Pattern_3Bar_Rev
# ---------------------------------------------------------------------------

def gen_Pattern_3Bar_Rev(df, confirm=True):
    cl = df['close']
    op = df['open']
    # 3-bar reversal: 3 consecutive up bars then bearish close, vice versa
    up3 = (cl > cl.shift(1)) & (cl.shift(1) > cl.shift(2)) & (cl.shift(2) > cl.shift(3))
    dn3 = (cl < cl.shift(1)) & (cl.shift(1) < cl.shift(2)) & (cl.shift(2) < cl.shift(3))
    bear_bar = cl < op
    bull_bar = cl > op

    sig = pd.Series(0, index=df.index)
    # After 3 up bars, bearish reversal candle → SHORT
    sig[up3.shift(1) & bear_bar] = -1
    # After 3 down bars, bullish reversal candle → LONG
    sig[dn3.shift(1) & bull_bar] = 1
    return sig

space_Pattern_3Bar_Rev = {
    'confirm': ('categorical', [True, False]),
}

# ---------------------------------------------------------------------------
# 12. Pattern_Morning_Star
# ---------------------------------------------------------------------------

def gen_Pattern_Morning_Star(df, doji_thresh=0.1, trend_len=5):
    op = df['open']
    hi = df['high']
    lo = df['low']
    cl = df['close']
    body = (cl - op).abs()
    rng = hi - lo

    ratio = body / rng.replace(0, np.nan)
    is_doji = ratio < doji_thresh
    is_bearish = cl < op
    is_bullish = cl > op

    downtrend = cl.shift(2) > cl.shift(3)  # bar2 was bearish move
    uptrend = cl.shift(2) < cl.shift(3)

    # Morning Star: bearish bar, doji/small bar gap down, bullish bar
    morning = (is_bearish.shift(2) &
               (body.shift(1) < body.shift(2) * 0.5) &
               is_bullish &
               (cl > (op.shift(2) + cl.shift(2)) / 2))
    # Evening Star: bullish, doji/small, bearish
    evening = (is_bullish.shift(2) &
               (body.shift(1) < body.shift(2) * 0.5) &
               is_bearish &
               (cl < (op.shift(2) + cl.shift(2)) / 2))

    sig = pd.Series(0, index=df.index)
    sig[morning] = 1
    sig[evening] = -1
    return sig

space_Pattern_Morning_Star = {
    'doji_thresh': ('float', 0.05, 0.25),
    'trend_len': ('int', 3, 10),
}

# ---------------------------------------------------------------------------
# 13. Pattern_Engulfing_ATR
# ---------------------------------------------------------------------------

def gen_Pattern_Engulfing_ATR(df, atr_len=14, min_atr_mult=1.0):
    op = df['open']
    cl = df['close']
    atr = _atr(df, atr_len)

    body_curr = (cl - op).abs()
    body_prev = (cl.shift(1) - op.shift(1)).abs()

    bull_engulf = (cl.shift(1) < op.shift(1)) & (cl > op) & (cl > op.shift(1)) & (op < cl.shift(1))
    bear_engulf = (cl.shift(1) > op.shift(1)) & (cl < op) & (cl < op.shift(1)) & (op > cl.shift(1))

    large_enough = body_curr > atr * min_atr_mult

    sig = pd.Series(0, index=df.index)
    sig[bull_engulf & large_enough] = 1
    sig[bear_engulf & large_enough] = -1
    return sig

space_Pattern_Engulfing_ATR = {
    'atr_len': ('int', 7, 21),
    'min_atr_mult': ('float', 0.5, 2.0),
}

# ---------------------------------------------------------------------------
# 14. Pattern_Pinbar_Signal
# ---------------------------------------------------------------------------

def gen_Pattern_Pinbar_Signal(df, nose_ratio=0.6, body_ratio=0.25, trend_len=20):
    op = df['open']
    hi = df['high']
    lo = df['low']
    cl = df['close']
    rng = hi - lo
    body = (cl - op).abs()
    upper_wick = hi - pd.concat([cl, op], axis=1).max(axis=1)
    lower_wick = pd.concat([cl, op], axis=1).min(axis=1) - lo

    rng_nz = rng.replace(0, np.nan)

    # Hammer: long lower wick, small body, small upper wick
    hammer = (lower_wick / rng_nz > nose_ratio) & (body / rng_nz < body_ratio)
    # Shooting star: long upper wick, small body, small lower wick
    shooting = (upper_wick / rng_nz > nose_ratio) & (body / rng_nz < body_ratio)

    trend_up = cl > _sma(cl, trend_len)
    trend_dn = cl < _sma(cl, trend_len)

    sig = pd.Series(0, index=df.index)
    sig[hammer & ~trend_up] = 1   # hammer in downtrend
    sig[shooting & trend_up] = -1  # shooting star in uptrend
    return sig

space_Pattern_Pinbar_Signal = {
    'nose_ratio': ('float', 0.5, 0.75),
    'body_ratio': ('float', 0.1, 0.35),
    'trend_len': ('int', 10, 50),
}

# ---------------------------------------------------------------------------
# 15. Pattern_Outside_Bar
# ---------------------------------------------------------------------------

def gen_Pattern_Outside_Bar(df, trend_len=20):
    hi = df['high']
    lo = df['low']
    cl = df['close']
    op = df['open']

    outside = (hi > hi.shift(1)) & (lo < lo.shift(1))
    bullish_close = cl > (hi + lo) / 2
    bearish_close = cl < (hi + lo) / 2

    sig = pd.Series(0, index=df.index)
    sig[outside & bullish_close] = 1
    sig[outside & bearish_close] = -1
    return sig

space_Pattern_Outside_Bar = {
    'trend_len': ('int', 10, 40),
}

# ---------------------------------------------------------------------------
# 16. Pattern_Doji_Trend
# ---------------------------------------------------------------------------

def gen_Pattern_Doji_Trend(df, doji_ratio=0.1, trend_len=20, confirm_bars=1):
    op = df['open']
    hi = df['high']
    lo = df['low']
    cl = df['close']
    body = (cl - op).abs()
    rng = (hi - lo).replace(0, np.nan)
    is_doji = (body / rng) < doji_ratio

    trend_up = cl > _sma(cl, trend_len)
    trend_dn = cl < _sma(cl, trend_len)

    # Doji after uptrend confirmed by next bearish bar → SHORT
    # Doji after downtrend confirmed by next bullish bar → LONG
    next_bull = cl > cl.shift(1)
    next_bear = cl < cl.shift(1)

    sig = pd.Series(0, index=df.index)
    sig[is_doji.shift(1) & trend_dn.shift(1) & next_bull] = 1
    sig[is_doji.shift(1) & trend_up.shift(1) & next_bear] = -1
    return sig

space_Pattern_Doji_Trend = {
    'doji_ratio': ('float', 0.05, 0.2),
    'trend_len': ('int', 10, 50),
    'confirm_bars': ('int', 1, 3),
}

# ---------------------------------------------------------------------------
# 17. Pattern_Harami
# ---------------------------------------------------------------------------

def gen_Pattern_Harami(df, trend_len=20):
    op = df['open']
    cl = df['close']
    hi = df['high']
    lo = df['low']

    prev_bull = cl.shift(1) > op.shift(1)
    prev_bear = cl.shift(1) < op.shift(1)
    prev_hi = pd.concat([op.shift(1), cl.shift(1)], axis=1).max(axis=1)
    prev_lo = pd.concat([op.shift(1), cl.shift(1)], axis=1).min(axis=1)

    curr_hi = pd.concat([op, cl], axis=1).max(axis=1)
    curr_lo = pd.concat([op, cl], axis=1).min(axis=1)

    inside = (curr_hi < prev_hi) & (curr_lo > prev_lo)

    trend_up = cl > _sma(cl, trend_len)
    trend_dn = cl < _sma(cl, trend_len)

    # Bearish harami: large bullish prev, small inside bar in uptrend → SHORT
    # Bullish harami: large bearish prev, small inside bar in downtrend → LONG
    sig = pd.Series(0, index=df.index)
    sig[inside & prev_bull & trend_up] = -1
    sig[inside & prev_bear & trend_dn] = 1
    return sig

space_Pattern_Harami = {
    'trend_len': ('int', 10, 50),
}

# ---------------------------------------------------------------------------
# 18. Pattern_ThreeBar_Push
# ---------------------------------------------------------------------------

def gen_Pattern_ThreeBar_Push(df, min_consecutive=3):
    cl = df['close']
    up = cl > cl.shift(1)
    dn = cl < cl.shift(1)

    consec_up = up.rolling(min_consecutive).min().astype(bool)
    consec_dn = dn.rolling(min_consecutive).min().astype(bool)

    sig = pd.Series(0, index=df.index)
    sig[consec_up] = 1
    sig[consec_dn] = -1
    return sig

space_Pattern_ThreeBar_Push = {
    'min_consecutive': ('int', 2, 5),
}

# ---------------------------------------------------------------------------
# 19. Pattern_EqualHL
# ---------------------------------------------------------------------------

def gen_Pattern_EqualHL(df, lookback=20, tolerance=0.003, trend_len=30):
    hi = df['high']
    lo = df['low']
    cl = df['close']

    roll_max = hi.rolling(lookback).max()
    roll_min = lo.rolling(lookback).min()

    # Equal highs: current high within tolerance of recent max (double top proxy)
    eq_high = (hi - roll_max).abs() / roll_max < tolerance
    # Equal lows: current low within tolerance of recent min (double bottom proxy)
    eq_low = (lo - roll_min).abs() / roll_min.replace(0, np.nan).abs() < tolerance

    trend_up = cl > _sma(cl, trend_len)
    trend_dn = cl < _sma(cl, trend_len)

    sig = pd.Series(0, index=df.index)
    sig[eq_high & trend_up] = -1   # double top → reversal short
    sig[eq_low & trend_dn] = 1     # double bottom → reversal long
    return sig

space_Pattern_EqualHL = {
    'lookback': ('int', 10, 40),
    'tolerance': ('float', 0.001, 0.01),
    'trend_len': ('int', 15, 50),
}

# ---------------------------------------------------------------------------
# 20. Pattern_WideRange_Bar
# ---------------------------------------------------------------------------

def gen_Pattern_WideRange_Bar(df, avg_len=20, mult=2.0, trend_len=20):
    hi = df['high']
    lo = df['low']
    cl = df['close']
    op = df['open']

    bar_range = hi - lo
    avg_range = _sma(bar_range, avg_len)
    wide = bar_range > avg_range * mult

    bullish = cl > op
    bearish = cl < op
    trend_up = cl > _sma(cl, trend_len)
    trend_dn = cl < _sma(cl, trend_len)

    sig = pd.Series(0, index=df.index)
    sig[wide & bullish & trend_up] = 1
    sig[wide & bearish & trend_dn] = -1
    return sig

space_Pattern_WideRange_Bar = {
    'avg_len': ('int', 10, 30),
    'mult': ('float', 1.5, 3.0),
    'trend_len': ('int', 10, 40),
}

# ---------------------------------------------------------------------------
# 21. Volatility_Regime_Switch
# ---------------------------------------------------------------------------

def gen_Volatility_Regime_Switch(df, atr_len=14, atr_sma_len=50,
                                  trend_fast=10, trend_slow=30,
                                  mr_len=20, z_thresh=1.5):
    cl = df['close']
    atr = _atr(df, atr_len)
    atr_avg = _sma(atr, atr_sma_len)

    # High vol → trend-following; Low vol → mean reversion
    high_vol = atr > atr_avg

    # Trend signal
    trend_sig = pd.Series(0, index=df.index)
    fast_ma = _ema(cl, trend_fast)
    slow_ma = _ema(cl, trend_slow)
    trend_sig[fast_ma > slow_ma] = 1
    trend_sig[fast_ma < slow_ma] = -1

    # Mean reversion signal
    ma = _sma(cl, mr_len)
    std = cl.rolling(mr_len).std()
    z = (cl - ma) / std.replace(0, np.nan)
    mr_sig = pd.Series(0, index=df.index)
    mr_sig[z < -z_thresh] = 1
    mr_sig[z > z_thresh] = -1

    sig = pd.Series(0, index=df.index)
    sig[high_vol] = trend_sig[high_vol]
    sig[~high_vol] = mr_sig[~high_vol]
    return sig

space_Volatility_Regime_Switch = {
    'atr_len': ('int', 7, 21),
    'atr_sma_len': ('int', 30, 100),
    'trend_fast': ('int', 5, 20),
    'trend_slow': ('int', 20, 60),
    'mr_len': ('int', 10, 40),
    'z_thresh': ('float', 1.0, 2.5),
}

# ---------------------------------------------------------------------------
# 22. Trend_Quality_Index
# ---------------------------------------------------------------------------

def gen_Trend_Quality_Index(df, tqi_len=20, entry_thresh=0.6, fast=5, slow=20):
    cl = df['close']
    hi = df['high']
    lo = df['low']

    # TQI: sum of |close_i - close_{i-1}| vs total range over period
    directed_move = (cl - cl.shift(tqi_len)).abs()
    total_path = cl.diff().abs().rolling(tqi_len).sum()
    tqi = directed_move / total_path.replace(0, np.nan)

    fast_ma = _ema(cl, fast)
    slow_ma = _ema(cl, slow)

    sig = pd.Series(0, index=df.index)
    sig[(tqi > entry_thresh) & (fast_ma > slow_ma)] = 1
    sig[(tqi > entry_thresh) & (fast_ma < slow_ma)] = -1
    return sig

space_Trend_Quality_Index = {
    'tqi_len': ('int', 10, 40),
    'entry_thresh': ('float', 0.4, 0.8),
    'fast': ('int', 3, 15),
    'slow': ('int', 15, 50),
}

# ---------------------------------------------------------------------------
# 23. Choppiness_Breakout
# ---------------------------------------------------------------------------

def gen_Choppiness_Breakout(df, chop_len=14, chop_thresh=38.2, atr_len=14):
    hi = df['high']
    lo = df['low']
    cl = df['close']

    atr = _atr(df, atr_len)
    atr_sum = atr.rolling(chop_len).sum()
    hl_range = hi.rolling(chop_len).max() - lo.rolling(chop_len).min()
    chop = 100 * np.log10(atr_sum / hl_range.replace(0, np.nan)) / np.log10(chop_len)

    fast_ma = _ema(cl, 5)
    slow_ma = _ema(cl, 20)

    sig = pd.Series(0, index=df.index)
    trending = chop < chop_thresh
    sig[trending & (fast_ma > slow_ma)] = 1
    sig[trending & (fast_ma < slow_ma)] = -1
    return sig

space_Choppiness_Breakout = {
    'chop_len': ('int', 7, 30),
    'chop_thresh': ('float', 30.0, 50.0),
    'atr_len': ('int', 7, 21),
}

# ---------------------------------------------------------------------------
# 24. Efficiency_Ratio_Signal
# ---------------------------------------------------------------------------

def gen_Efficiency_Ratio_Signal(df, er_len=10, er_high=0.6, er_low=0.3,
                                 fast=5, slow=20, mr_len=20):
    cl = df['close']
    direction = (cl - cl.shift(er_len)).abs()
    volatility = cl.diff().abs().rolling(er_len).sum()
    er = (direction / volatility.replace(0, np.nan)).fillna(0).clip(0, 1)

    fast_ma = _ema(cl, fast)
    slow_ma = _ema(cl, slow)
    ma = _sma(cl, mr_len)

    sig = pd.Series(0, index=df.index)
    # High ER → trending: follow EMA cross
    sig[(er > er_high) & (fast_ma > slow_ma)] = 1
    sig[(er > er_high) & (fast_ma < slow_ma)] = -1
    # Low ER → mean reversion: fade moves
    sig[(er < er_low) & (cl > ma)] = -1
    sig[(er < er_low) & (cl < ma)] = 1
    return sig

space_Efficiency_Ratio_Signal = {
    'er_len': ('int', 5, 20),
    'er_high': ('float', 0.5, 0.8),
    'er_low': ('float', 0.1, 0.4),
    'fast': ('int', 3, 15),
    'slow': ('int', 15, 40),
    'mr_len': ('int', 10, 30),
}

# ---------------------------------------------------------------------------
# 25. Cycle_Period_RSI
# ---------------------------------------------------------------------------

def gen_Cycle_Period_RSI(df, base_period=14, ob=65, os=35, smooth=3):
    cl = df['close']
    # Dominant cycle via autocorrelation peak
    n = len(cl)
    cycle_period = np.full(n, float(base_period))
    arr = cl.values

    window = 40
    for i in range(window, n):
        seg = arr[i - window: i]
        seg_d = seg - seg.mean()
        acf = np.correlate(seg_d, seg_d, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / (acf[0] + 1e-10)
        # Find first peak after lag=2
        peaks = []
        for lag in range(2, min(window//2, len(acf)-1)):
            if acf[lag] > acf[lag-1] and acf[lag] > acf[lag+1]:
                peaks.append((acf[lag], lag))
        if peaks:
            best_lag = max(peaks, key=lambda x: x[0])[1]
            cycle_period[i] = float(np.clip(best_lag, 5, 30))

    cycle_s = pd.Series(cycle_period, index=cl.index)
    cycle_smooth = cycle_s.rolling(smooth).mean().fillna(base_period).round().astype(int)

    rsi_arr = np.full(n, np.nan)
    for i in range(30, n):
        p = int(cycle_smooth.iloc[i])
        p = max(5, min(30, p))
        slice_ = cl.iloc[max(0, i - p*3): i+1]
        d = slice_.diff()
        gain = _rma(d.clip(lower=0), p).iloc[-1]
        loss = _rma((-d).clip(lower=0), p).iloc[-1]
        if loss == 0:
            rsi_arr[i] = 100.0
        else:
            rsi_arr[i] = 100 - 100 / (1 + gain / loss)

    rsi = pd.Series(rsi_arr, index=cl.index)
    sig = pd.Series(0, index=df.index)
    sig[rsi < os] = 1
    sig[rsi > ob] = -1
    return sig

space_Cycle_Period_RSI = {
    'base_period': ('int', 7, 21),
    'ob': ('int', 60, 80),
    'os': ('int', 20, 40),
    'smooth': ('int', 2, 5),
}

# ---------------------------------------------------------------------------
# 26. Spectral_MA
# ---------------------------------------------------------------------------

def gen_Spectral_MA(df, w1=8, w2=13, w3=21, w4=34, thresh=2):
    cl = df['close']
    e1 = _ema(cl, w1)
    e2 = _ema(cl, w2)
    e3 = _ema(cl, w3)
    e4 = _ema(cl, w4)

    # Count how many EMAs are bullish (price above EMA)
    bull_count = (cl > e1).astype(int) + (cl > e2).astype(int) + \
                 (cl > e3).astype(int) + (cl > e4).astype(int)
    bear_count = (cl < e1).astype(int) + (cl < e2).astype(int) + \
                 (cl < e3).astype(int) + (cl < e4).astype(int)

    sig = pd.Series(0, index=df.index)
    sig[bull_count >= thresh + 1] = 1
    sig[bear_count >= thresh + 1] = -1
    return sig

space_Spectral_MA = {
    'w1': ('int', 5, 13),
    'w2': ('int', 10, 21),
    'w3': ('int', 17, 34),
    'w4': ('int', 28, 55),
    'thresh': ('int', 2, 3),
}

# ---------------------------------------------------------------------------
# 27. Entropy_Signal
# ---------------------------------------------------------------------------

def gen_Entropy_Signal(df, entropy_len=20, high_entropy_thresh=0.8,
                        fast=5, slow=20):
    cl = df['close']
    returns = cl.pct_change()

    def approx_entropy(series_values, m=2, r_mult=0.2):
        n = len(series_values)
        if n < m + 2:
            return np.nan
        r = r_mult * np.std(series_values)
        if r == 0:
            return 0.0
        def phi(m_val):
            count = 0
            total = 0
            for i in range(n - m_val):
                template = series_values[i: i + m_val]
                for j in range(n - m_val):
                    if np.max(np.abs(series_values[j: j + m_val] - template)) <= r:
                        count += 1
                total += 1
            return np.log(count / max(total * (n - m_val), 1))
        return abs(phi(m) - phi(m + 1))

    ent_vals = np.full(len(cl), np.nan)
    ret_arr = returns.values
    for i in range(entropy_len, len(ret_arr)):
        seg = ret_arr[i - entropy_len: i]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= entropy_len // 2:
            ent_vals[i] = approx_entropy(seg)

    ent = pd.Series(ent_vals, index=cl.index)
    ent_norm = (ent - ent.rolling(50).min()) / \
               (ent.rolling(50).max() - ent.rolling(50).min() + 1e-10)

    fast_ma = _ema(cl, fast)
    slow_ma = _ema(cl, slow)

    sig = pd.Series(0, index=df.index)
    low_ent = ent_norm < (1 - high_entropy_thresh)
    sig[low_ent & (fast_ma > slow_ma)] = 1
    sig[low_ent & (fast_ma < slow_ma)] = -1
    return sig

space_Entropy_Signal = {
    'entropy_len': ('int', 10, 30),
    'high_entropy_thresh': ('float', 0.5, 0.9),
    'fast': ('int', 3, 15),
    'slow': ('int', 15, 40),
}

# ---------------------------------------------------------------------------
# 28. Fractal_Dimension_MA
# ---------------------------------------------------------------------------

def gen_Fractal_Dimension_MA(df, fd_len=30, fast_base=5, slow_base=30):
    cl = df['close']
    n = len(cl)
    fd_vals = np.full(n, np.nan)

    for i in range(fd_len, n):
        seg = cl.values[i - fd_len: i]
        hi_seg = np.max(seg)
        lo_seg = np.min(seg)
        if hi_seg == lo_seg:
            fd_vals[i] = 1.5
            continue
        # Higuchi-inspired: N1 = path length, N2 = Euclidean distance
        path_len = np.sum(np.abs(np.diff(seg)))
        eucl = hi_seg - lo_seg
        fd_vals[i] = 1 + np.log(path_len / (eucl + 1e-10)) / np.log(fd_len)

    fd = pd.Series(fd_vals, index=cl.index).clip(1.0, 2.0)

    # Scale EMA period based on fractal dimension: high FD (choppy) → longer period
    period = (fast_base + (fd - 1.0) * (slow_base - fast_base)).round().fillna(fast_base)
    period = period.clip(fast_base, slow_base).astype(int)

    ema_vals = np.full(n, np.nan)
    cl_arr = cl.values
    for i in range(1, n):
        p = int(period.iloc[i])
        alpha = 2.0 / (p + 1)
        if np.isnan(ema_vals[i-1]):
            ema_vals[i] = cl_arr[i]
        else:
            ema_vals[i] = alpha * cl_arr[i] + (1 - alpha) * ema_vals[i-1]

    adaptive_ema = pd.Series(ema_vals, index=cl.index)
    sig = pd.Series(0, index=df.index)
    sig[cl > adaptive_ema] = 1
    sig[cl < adaptive_ema] = -1
    return sig

space_Fractal_Dimension_MA = {
    'fd_len': ('int', 15, 50),
    'fast_base': ('int', 3, 10),
    'slow_base': ('int', 20, 60),
}

# ---------------------------------------------------------------------------
# 29. Self_Adjusting_RSI
# ---------------------------------------------------------------------------

def gen_Self_Adjusting_RSI(df, rsi_len=14, pct_len=100, ob_pct=80, os_pct=20):
    cl = df['close']
    rsi = _rsi(cl, rsi_len)
    dyn_ob = rsi.rolling(pct_len).quantile(ob_pct / 100.0)
    dyn_os = rsi.rolling(pct_len).quantile(os_pct / 100.0)

    sig = pd.Series(0, index=df.index)
    sig[rsi < dyn_os] = 1
    sig[rsi > dyn_ob] = -1
    return sig

space_Self_Adjusting_RSI = {
    'rsi_len': ('int', 7, 21),
    'pct_len': ('int', 50, 200),
    'ob_pct': ('int', 70, 90),
    'os_pct': ('int', 10, 30),
}

# ---------------------------------------------------------------------------
# 30. Regression_RSI_Combo
# ---------------------------------------------------------------------------

def gen_Regression_RSI_Combo(df, lr_len=20, rsi_len=14, dev_mult=2.0,
                               ob=60, os=40):
    cl = df['close']
    n = len(cl)

    fitted = np.full(n, np.nan)
    for i in range(lr_len - 1, n):
        y = cl.values[i - lr_len + 1: i + 1]
        x = np.arange(lr_len)
        coeffs = np.polyfit(x, y, 1)
        fitted[i] = np.polyval(coeffs, lr_len - 1)

    fitted_s = pd.Series(fitted, index=cl.index)
    residuals = cl - fitted_s
    std = residuals.rolling(lr_len).std()
    upper = fitted_s + dev_mult * std
    lower = fitted_s - dev_mult * std

    rsi = _rsi(cl, rsi_len)

    sig = pd.Series(0, index=df.index)
    # Long: price at lower channel AND RSI oversold
    sig[(cl <= lower) & (rsi < os)] = 1
    # Short: price at upper channel AND RSI overbought
    sig[(cl >= upper) & (rsi > ob)] = -1
    return sig

space_Regression_RSI_Combo = {
    'lr_len': ('int', 10, 40),
    'rsi_len': ('int', 7, 21),
    'dev_mult': ('float', 1.0, 3.0),
    'ob': ('int', 55, 75),
    'os': ('int', 25, 45),
}

# ---------------------------------------------------------------------------
# 31. Pattern_Three_White_Soldiers
# ---------------------------------------------------------------------------

def gen_Pattern_Three_White_Soldiers(df, min_body_ratio=0.6, trend_len=20):
    op = df['open']
    hi = df['high']
    lo = df['low']
    cl = df['close']

    body = cl - op
    rng = (hi - lo).replace(0, np.nan)
    body_ratio = body / rng

    bull_bar = cl > op
    bear_bar = cl < op

    # Three White Soldiers: 3 bullish bars, each opening within prior body, each closing higher
    tws = (bull_bar & bull_bar.shift(1) & bull_bar.shift(2) &
           (body_ratio > min_body_ratio) &
           (body_ratio.shift(1) > min_body_ratio) &
           (body_ratio.shift(2) > min_body_ratio) &
           (op > op.shift(1)) & (cl > cl.shift(1)) &
           (op.shift(1) > op.shift(2)) & (cl.shift(1) > cl.shift(2)))

    # Three Black Crows: opposite
    tbc = (bear_bar & bear_bar.shift(1) & bear_bar.shift(2) &
           ((-body / rng) > min_body_ratio) &
           ((-body.shift(1) / rng.shift(1)) > min_body_ratio) &
           ((-body.shift(2) / rng.shift(2)) > min_body_ratio) &
           (op < op.shift(1)) & (cl < cl.shift(1)) &
           (op.shift(1) < op.shift(2)) & (cl.shift(1) < cl.shift(2)))

    sig = pd.Series(0, index=df.index)
    sig[tws] = 1
    sig[tbc] = -1
    return sig

space_Pattern_Three_White_Soldiers = {
    'min_body_ratio': ('float', 0.4, 0.8),
    'trend_len': ('int', 10, 40),
}

# ---------------------------------------------------------------------------
# 32. Pattern_Rising_Falling_Three
# ---------------------------------------------------------------------------

def gen_Pattern_Rising_Falling_Three(df, small_body_mult=0.5):
    op = df['open']
    cl = df['close']
    hi = df['high']
    lo = df['low']

    body = (cl - op).abs()
    avg_body = _sma(body, 10)

    bull1 = cl.shift(4) > op.shift(4)
    small_2 = body.shift(3) < avg_body.shift(3) * small_body_mult
    small_3 = body.shift(2) < avg_body.shift(2) * small_body_mult
    small_4 = body.shift(1) < avg_body.shift(1) * small_body_mult
    bull5 = cl > op
    # Rising three: bar1 bullish, 3 small bars, bar5 bullish closes above bar1
    rising_three = (bull1 & small_2 & small_3 & small_4 & bull5 &
                    (cl > cl.shift(4)) &
                    (cl.shift(3) > op.shift(4)) & (cl.shift(1) < cl.shift(4)))

    bear1 = cl.shift(4) < op.shift(4)
    bear5 = cl < op
    falling_three = (bear1 & small_2 & small_3 & small_4 & bear5 &
                     (cl < cl.shift(4)) &
                     (cl.shift(3) < op.shift(4)) & (cl.shift(1) > cl.shift(4)))

    sig = pd.Series(0, index=df.index)
    sig[rising_three] = 1
    sig[falling_three] = -1
    return sig

space_Pattern_Rising_Falling_Three = {
    'small_body_mult': ('float', 0.3, 0.7),
}

# ---------------------------------------------------------------------------
# 33. Pattern_Kicker
# ---------------------------------------------------------------------------

def gen_Pattern_Kicker(df, gap_mult=0.5, atr_len=14):
    op = df['open']
    cl = df['close']
    atr = _atr(df, atr_len)

    prev_bull = cl.shift(1) > op.shift(1)
    prev_bear = cl.shift(1) < op.shift(1)

    # Bullish kicker: previous bar was bearish, current opens above prev open (gap up) and is bullish
    bull_kicker = (prev_bear &
                   (op > op.shift(1) + atr * gap_mult) &
                   (cl > op))

    # Bearish kicker: previous bar was bullish, current opens below prev open (gap down) and is bearish
    bear_kicker = (prev_bull &
                   (op < op.shift(1) - atr * gap_mult) &
                   (cl < op))

    sig = pd.Series(0, index=df.index)
    sig[bull_kicker] = 1
    sig[bear_kicker] = -1
    return sig

space_Pattern_Kicker = {
    'gap_mult': ('float', 0.2, 1.5),
    'atr_len': ('int', 7, 21),
}

# ---------------------------------------------------------------------------
# 34. Adaptive_Volatility_Breakout
# ---------------------------------------------------------------------------

def gen_Adaptive_Volatility_Breakout(df, lookback=20, atr_len=14,
                                      vol_fast=10, vol_slow=40, mult=1.5):
    cl = df['close']
    hi = df['high']
    lo = df['low']

    atr = _atr(df, atr_len)
    atr_fast = _sma(atr, vol_fast)
    atr_slow = _sma(atr, vol_slow)

    # Adaptive multiplier: in high-vol regime use larger mult
    vol_ratio = (atr_fast / atr_slow.replace(0, np.nan)).fillna(1.0).clip(0.5, 2.0)
    adaptive_mult = mult * vol_ratio

    roll_high = hi.rolling(lookback).max()
    roll_low = lo.rolling(lookback).min()

    upper_band = roll_high.shift(1)
    lower_band = roll_low.shift(1)

    # Breakout: close breaks above upper band by adaptive ATR threshold
    breakout_up = cl > upper_band + atr * (adaptive_mult - mult)
    breakout_dn = cl < lower_band - atr * (adaptive_mult - mult)

    # Also require trend confirmation via EMA
    ema_trend = _ema(cl, lookback)

    sig = pd.Series(0, index=df.index)
    sig[(cl > upper_band) & (cl > ema_trend)] = 1
    sig[(cl < lower_band) & (cl < ema_trend)] = -1
    return sig

space_Adaptive_Volatility_Breakout = {
    'lookback': ('int', 10, 40),
    'atr_len': ('int', 7, 21),
    'vol_fast': ('int', 5, 20),
    'vol_slow': ('int', 20, 60),
    'mult': ('float', 1.0, 3.0),
}

# ---------------------------------------------------------------------------
# STRATEGY_EXPORT
# ---------------------------------------------------------------------------

STRATEGY_EXPORT = {
    'Nearest_Neighbor_Signal': {'gen': gen_Nearest_Neighbor_Signal, 'space': space_Nearest_Neighbor_Signal},
    'Gaussian_Kernel_Trend': {'gen': gen_Gaussian_Kernel_Trend, 'space': space_Gaussian_Kernel_Trend},
    'Lorentzian_KNN': {'gen': gen_Lorentzian_KNN, 'space': space_Lorentzian_KNN},
    'Linear_Regression_Slope': {'gen': gen_Linear_Regression_Slope, 'space': space_Linear_Regression_Slope},
    'Polynomial_Regression_Channel': {'gen': gen_Polynomial_Regression_Channel, 'space': space_Polynomial_Regression_Channel},
    'LSMA_Crossover': {'gen': gen_LSMA_Crossover, 'space': space_LSMA_Crossover},
    'Deviation_Channel_ML': {'gen': gen_Deviation_Channel_ML, 'space': space_Deviation_Channel_ML},
    'Gradient_Descent_MA': {'gen': gen_Gradient_Descent_MA, 'space': space_Gradient_Descent_MA},
    'Adaptive_Lookback_RSI': {'gen': gen_Adaptive_Lookback_RSI, 'space': space_Adaptive_Lookback_RSI},
    'Fractal_Efficiency_MA': {'gen': gen_Fractal_Efficiency_MA, 'space': space_Fractal_Efficiency_MA},
    'Pattern_3Bar_Rev': {'gen': gen_Pattern_3Bar_Rev, 'space': space_Pattern_3Bar_Rev},
    'Pattern_Morning_Star': {'gen': gen_Pattern_Morning_Star, 'space': space_Pattern_Morning_Star},
    'Pattern_Engulfing_ATR': {'gen': gen_Pattern_Engulfing_ATR, 'space': space_Pattern_Engulfing_ATR},
    'Pattern_Pinbar_Signal': {'gen': gen_Pattern_Pinbar_Signal, 'space': space_Pattern_Pinbar_Signal},
    'Pattern_Outside_Bar': {'gen': gen_Pattern_Outside_Bar, 'space': space_Pattern_Outside_Bar},
    'Pattern_Doji_Trend': {'gen': gen_Pattern_Doji_Trend, 'space': space_Pattern_Doji_Trend},
    'Pattern_Harami': {'gen': gen_Pattern_Harami, 'space': space_Pattern_Harami},
    'Pattern_ThreeBar_Push': {'gen': gen_Pattern_ThreeBar_Push, 'space': space_Pattern_ThreeBar_Push},
    'Pattern_EqualHL': {'gen': gen_Pattern_EqualHL, 'space': space_Pattern_EqualHL},
    'Pattern_WideRange_Bar': {'gen': gen_Pattern_WideRange_Bar, 'space': space_Pattern_WideRange_Bar},
    'Volatility_Regime_Switch': {'gen': gen_Volatility_Regime_Switch, 'space': space_Volatility_Regime_Switch},
    'Trend_Quality_Index': {'gen': gen_Trend_Quality_Index, 'space': space_Trend_Quality_Index},
    'Choppiness_Breakout': {'gen': gen_Choppiness_Breakout, 'space': space_Choppiness_Breakout},
    'Efficiency_Ratio_Signal': {'gen': gen_Efficiency_Ratio_Signal, 'space': space_Efficiency_Ratio_Signal},
    'Cycle_Period_RSI': {'gen': gen_Cycle_Period_RSI, 'space': space_Cycle_Period_RSI},
    'Spectral_MA': {'gen': gen_Spectral_MA, 'space': space_Spectral_MA},
    'Entropy_Signal': {'gen': gen_Entropy_Signal, 'space': space_Entropy_Signal},
    'Fractal_Dimension_MA': {'gen': gen_Fractal_Dimension_MA, 'space': space_Fractal_Dimension_MA},
    'Self_Adjusting_RSI': {'gen': gen_Self_Adjusting_RSI, 'space': space_Self_Adjusting_RSI},
    'Regression_RSI_Combo': {'gen': gen_Regression_RSI_Combo, 'space': space_Regression_RSI_Combo},
    'Pattern_Three_White_Soldiers': {'gen': gen_Pattern_Three_White_Soldiers, 'space': space_Pattern_Three_White_Soldiers},
    'Pattern_Rising_Falling_Three': {'gen': gen_Pattern_Rising_Falling_Three, 'space': space_Pattern_Rising_Falling_Three},
    'Pattern_Kicker': {'gen': gen_Pattern_Kicker, 'space': space_Pattern_Kicker},
    'Adaptive_Volatility_Breakout': {'gen': gen_Adaptive_Volatility_Breakout, 'space': space_Adaptive_Volatility_Breakout},
}
