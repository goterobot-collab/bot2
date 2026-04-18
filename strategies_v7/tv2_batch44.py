#!/usr/bin/env python3
"""TV2 BATCH 44 — Geraked (3MAF/ABSAR/BBRSI/CEZLSMA/CP1/DHLAOS/LRCUTB/MLCEL/NWERSI)
   + harryguiacorn (BBForce/HOLP/Kijun/OneBurPursuit)
   + JN842 (MTF_BB) + RSC809 (kNN) + zemulist (BCGA/TheOneRing) + RodrigoBrito (Setup92)
   18 estrategias Pine v4/v5/v6 → Python
"""
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _bb(close: pd.Series, n: int = 20, mult: float = 2.0):
    mid = _sma(close, n)
    std = close.rolling(n).std(ddof=0)
    return mid + mult*std, mid, mid - mult*std

def _zlsma(close: pd.Series, n: int) -> pd.Series:
    """Zero-Lag LSMA = linreg + (linreg - linreg(linreg))"""
    def linreg(s, p):
        return s.rolling(p).apply(
            lambda x: np.polyval(np.polyfit(range(p), x, 1), p-1), raw=True)
    lsma = linreg(close, n)
    lsma2 = linreg(lsma.ffill(), n)
    return lsma + (lsma - lsma2)

def _chandelier_exit(close: pd.Series, high: pd.Series, low: pd.Series,
                     length: int = 22, mult: float = 3.0):
    """Returns direction series: 1=up, -1=down"""
    atr_v = _atr(high, low, close, length)
    highest = high.rolling(length).max()
    lowest  = low.rolling(length).min()
    long_stop  = highest - mult * atr_v
    short_stop = lowest  + mult * atr_v

    dir_ = pd.Series(1, index=close.index)
    for i in range(1, len(close)):
        if pd.isna(long_stop.iloc[i]) or pd.isna(short_stop.iloc[i]):
            continue
        prev_dir = dir_.iloc[i-1]
        if close.iloc[i] > short_stop.iloc[i-1]:
            dir_.iloc[i] = 1
        elif close.iloc[i] < long_stop.iloc[i-1]:
            dir_.iloc[i] = -1
        else:
            dir_.iloc[i] = prev_dir
    return dir_

def _ut_bot(close: pd.Series, high: pd.Series, low: pd.Series,
            key_value: float = 2.0, atr_period: int = 1) -> pd.Series:
    """UT Bot — ATR trailing stop direction. Returns buy/sell series."""
    atr_v = _atr(high, low, close, atr_period)
    n_loss = key_value * atr_v

    trailing = pd.Series(0.0, index=close.index)
    for i in range(1, len(close)):
        prev_trail = trailing.iloc[i-1]
        c = close.iloc[i]
        c_prev = close.iloc[i-1]
        nl = n_loss.iloc[i] if not pd.isna(n_loss.iloc[i]) else 0
        if c > prev_trail and c_prev > prev_trail:
            trailing.iloc[i] = max(prev_trail, c - nl)
        elif c < prev_trail and c_prev < prev_trail:
            trailing.iloc[i] = min(prev_trail, c + nl)
        else:
            trailing.iloc[i] = c - nl if c > prev_trail else c + nl

    ema1 = _ema(close, 1)
    buy  = (ema1 > trailing) & (ema1.shift() <= trailing.shift())
    sell = (ema1 < trailing) & (ema1.shift() >= trailing.shift())
    return buy.astype(int) - sell.astype(int)

def _williams_fractal(high: pd.Series, low: pd.Series, n: int = 2):
    """Williams Fractals: up_fractal (local high), down_fractal (local low)."""
    up   = pd.Series(False, index=high.index)
    down = pd.Series(False, index=low.index)
    arr_h = high.values
    arr_l = low.values
    for i in range(n, len(arr_h) - n):
        if all(arr_h[i] > arr_h[i-k] for k in range(1, n+1)) and \
           all(arr_h[i] > arr_h[i+k] for k in range(1, n+1)):
            up.iloc[i] = True
        if all(arr_l[i] < arr_l[i-k] for k in range(1, n+1)) and \
           all(arr_l[i] < arr_l[i+k] for k in range(1, n+1)):
            down.iloc[i] = True
    return up, down

def _andean_osc(close: pd.Series, open_: pd.Series,
                length: int = 50, sig: int = 9):
    """Andean Oscillator — bull/bear components."""
    alpha = 2 / (length + 1)
    up1 = close.copy(); up2 = (close**2).copy()
    dn1 = close.copy(); dn2 = (close**2).copy()
    C = close.values; O = open_.values
    for i in range(1, len(C)):
        up1.iloc[i] = max(C[i], O[i], up1.iloc[i-1] - (up1.iloc[i-1] - C[i]) * alpha)
        up2.iloc[i] = max(C[i]**2, O[i]**2, up2.iloc[i-1] - (up2.iloc[i-1] - C[i]**2) * alpha)
        dn1.iloc[i] = min(C[i], O[i], dn1.iloc[i-1] + (C[i] - dn1.iloc[i-1]) * alpha)
        dn2.iloc[i] = min(C[i]**2, O[i]**2, dn2.iloc[i-1] + (C[i]**2 - dn2.iloc[i-1]) * alpha)
    bull = (dn2 - dn1**2).clip(lower=0).apply(np.sqrt)
    bear = (up2 - up1**2).clip(lower=0).apply(np.sqrt)
    signal = _ema(pd.concat([bull, bear], axis=1).max(axis=1), sig)
    return bull, bear, signal

def _linreg_candles(open_: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series, n: int = 11):
    """Linear regression candles."""
    def lr(s):
        return s.rolling(n).apply(
            lambda x: np.polyval(np.polyfit(range(n), x, 1), n-1), raw=True)
    return lr(open_), lr(high), lr(low), lr(close)

def _kijun(high: pd.Series, low: pd.Series, n: int = 26) -> pd.Series:
    return (high.rolling(n).max() + low.rolling(n).min()) / 2

def _parabolic_sar(high: pd.Series, low: pd.Series,
                   af_start: float = 0.02, af_step: float = 0.02,
                   af_max: float = 0.2) -> pd.Series:
    """Simple Parabolic SAR."""
    sar = pd.Series(np.nan, index=high.index)
    bull = True
    ep = low.iloc[0]
    af = af_start
    sar.iloc[0] = high.iloc[0]
    for i in range(1, len(high)):
        prev_sar = sar.iloc[i-1]
        if bull:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = min(new_sar, low.iloc[i-1],
                         low.iloc[i-2] if i > 1 else low.iloc[i-1])
            if low.iloc[i] < new_sar:
                bull = False; new_sar = ep; ep = low.iloc[i]; af = af_start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]; af = min(af + af_step, af_max)
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = max(new_sar, high.iloc[i-1],
                         high.iloc[i-2] if i > 1 else high.iloc[i-1])
            if high.iloc[i] > new_sar:
                bull = True; new_sar = ep; ep = high.iloc[i]; af = af_start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]; af = min(af + af_step, af_max)
        sar.iloc[i] = new_sar
    return sar

def _crossover(s1: pd.Series, s2) -> pd.Series:
    if isinstance(s2, (int, float)):
        s2 = pd.Series(s2, index=s1.index)
    return (s1 > s2) & (s1.shift() <= s2.shift())

def _crossunder(s1: pd.Series, s2) -> pd.Series:
    if isinstance(s2, (int, float)):
        s2 = pd.Series(s2, index=s1.index)
    return (s1 < s2) & (s1.shift() >= s2.shift())

def _minimax(x: pd.Series, p: int) -> pd.Series:
    """Normalize to [0,100] over rolling window p."""
    hi = x.rolling(p).max()
    lo = x.rolling(p).min()
    rng = (hi - lo).replace(0, np.nan)
    return 100 * (x - lo) / rng


# ─────────────────────────────────────────────
# 1. 3MAF_Geraked — 3 EMAs + Williams Fractals
# ─────────────────────────────────────────────
def gen_3MAF_Geraked(df: pd.DataFrame, params: dict) -> pd.Series:
    o, h, l, c = df.open, df.high, df.low, df.close
    l1 = int(params.get('len1', 20))
    l2 = int(params.get('len2', 50))
    l3 = int(params.get('len3', 100))
    n  = int(params.get('fractal_n', 2))
    ma1 = _ema(c, l1); ma2 = _ema(c, l2); ma3 = _ema(c, l3)
    up_frac, dn_frac = _williams_fractal(h, l, n)
    # Long: ma1>ma2>ma3 + low between ma3 and ma1 + downFractal (low at fractal)
    long_  = (ma1 > ma2) & (ma2 > ma3) & (l > ma3) & (l < ma1) & dn_frac
    # Short: ma3>ma2>ma1 + high between ma1 and ma3 + upFractal
    short_ = (ma3 > ma2) & (ma2 > ma1) & (h < ma3) & (h > ma1) & up_frac
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_3MAF_Geraked(trial) -> dict:
    return {
        'len1':      trial.suggest_int('len1', 5, 30),
        'len2':      trial.suggest_int('len2', 20, 80),
        'len3':      trial.suggest_int('len3', 50, 200),
        'fractal_n': trial.suggest_int('fractal_n', 1, 5),
    }


# ─────────────────────────────────────────────
# 2. ABSAR_Geraked — SAR + Price Structure (simplified from Pine ABC+SAR v4)
# ─────────────────────────────────────────────
def gen_ABSAR_Geraked(df: pd.DataFrame, params: dict) -> pd.Series:
    h, l, c = df.high, df.low, df.close
    af_start = params.get('af_start', 0.02)
    af_max   = params.get('af_max', 0.2)
    prd      = int(params.get('prd', 8))
    # SAR direction flip
    sar = _parabolic_sar(h, l, af_start=af_start, af_step=af_start, af_max=af_max)
    sar_bull = c > sar
    # ABC structure: A = N-bar high, B = pullback, C = breakout above B
    swing_h = h.rolling(prd).max()
    swing_l = l.rolling(prd).min()
    # Long: SAR flips bullish AND close breaks above recent swing high (C > A)
    flip_bull = sar_bull & ~sar_bull.shift().fillna(False)
    flip_bear = ~sar_bull & sar_bull.shift().fillna(True)
    # Confirm with price above recent swing
    long_  = flip_bull & (c > swing_h.shift(prd))
    short_ = flip_bear & (c < swing_l.shift(prd))
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_ABSAR_Geraked(trial) -> dict:
    return {
        'af_start': trial.suggest_float('af_start', 0.01, 0.05),
        'af_max':   trial.suggest_float('af_max', 0.1, 0.4),
        'prd':      trial.suggest_int('prd', 5, 20),
    }


# ─────────────────────────────────────────────
# 3. BBRSI_Geraked — Bollinger Bands + RSI (v5)
# ─────────────────────────────────────────────
def gen_BBRSI_Geraked(df: pd.DataFrame, params: dict) -> pd.Series:
    h, l, c = df.high, df.low, df.close
    bb_len   = int(params.get('bb_len', 20))
    bb_mult  = params.get('bb_mult', 2.0)
    rsi_len  = int(params.get('rsi_len', 3))
    rsi_os   = params.get('rsi_os', 30)
    rsi_ob   = params.get('rsi_ob', 70)
    upper, mid, lower = _bb(c, bb_len, bb_mult)
    rsi_ = _rsi(c, rsi_len)
    # Long: RSI crosses up from oversold AND close below lower BB
    long_  = _crossover(rsi_, rsi_os) & (c < lower)
    # Short: RSI crosses down from overbought AND close above upper BB
    short_ = _crossunder(rsi_, rsi_ob) & (c > upper)
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_BBRSI_Geraked(trial) -> dict:
    return {
        'bb_len':   trial.suggest_int('bb_len', 10, 50),
        'bb_mult':  trial.suggest_float('bb_mult', 1.0, 3.0),
        'rsi_len':  trial.suggest_int('rsi_len', 2, 14),
        'rsi_os':   trial.suggest_float('rsi_os', 20, 45),
        'rsi_ob':   trial.suggest_float('rsi_ob', 55, 80),
    }


# ─────────────────────────────────────────────
# 4. CEZLSMA_Geraked — Chandelier Exit + ZLSMA (v5)
# ─────────────────────────────────────────────
def gen_CEZLSMA_Geraked(df: pd.DataFrame, params: dict) -> pd.Series:
    h, l, c = df.high, df.low, df.close
    ce_len   = int(params.get('ce_len', 1))
    ce_mult  = params.get('ce_mult', 2.0)
    zlsma_n  = int(params.get('zlsma_n', 50))
    dir_ = _chandelier_exit(c, h, l, length=max(ce_len, 2), mult=ce_mult)
    zl   = _zlsma(c, zlsma_n)
    buy_signal  = (dir_ == 1) & (dir_.shift().fillna(1) == -1) & (c > zl)
    sell_signal = (dir_ == -1) & (dir_.shift().fillna(-1) == 1) & (c < zl)
    # Also close on ZLSMA cross
    close_long  = _crossunder(c, zl)
    close_short = _crossover(c, zl)
    sig = pd.Series(0, index=c.index)
    sig[buy_signal]  =  1
    sig[sell_signal] = -1
    # Neutralize on opposite cross
    sig[close_long & (sig == 0)] = -1  # exit signal only if no new entry
    return sig

def space_CEZLSMA_Geraked(trial) -> dict:
    return {
        'ce_len':   trial.suggest_int('ce_len', 1, 5),
        'ce_mult':  trial.suggest_float('ce_mult', 1.0, 4.0),
        'zlsma_n':  trial.suggest_int('zlsma_n', 20, 100),
    }


# ─────────────────────────────────────────────
# 5. CP1_Geraked — Candlestick Pattern 1 (v5)
# ─────────────────────────────────────────────
def gen_CP1_Geraked(df: pd.DataFrame, params: dict) -> pd.Series:
    o, h, l, c = df.open, df.high, df.low, df.close
    # Long: 2 bull candles (c>o), 2nd opens lower, closes higher, with lower wick below prev
    long_ = (
        (c > o) & (c.shift() > o.shift()) &
        (o.shift() < o) & (c > c.shift()) &
        (l < l.shift()) & (l < o.shift())
    )
    # Short: 2 bear candles, 2nd opens higher, closes lower, upper wick above prev
    short_ = (
        (c < o) & (c.shift() < o.shift()) &
        (o.shift() > o) & (c < c.shift()) &
        (h > h.shift()) & (h > o.shift())
    )
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_CP1_Geraked(trial) -> dict:
    # No numeric params in this strategy — add a trend filter
    return {
        'trend_filter': trial.suggest_int('trend_filter', 0, 1),
        'ema_trend':    trial.suggest_int('ema_trend', 20, 200),
    }

# Override with trend filter version
def gen_CP1_Geraked(df: pd.DataFrame, params: dict) -> pd.Series:
    o, h, l, c = df.open, df.high, df.low, df.close
    use_trend = bool(params.get('trend_filter', 0))
    ema_t     = int(params.get('ema_trend', 50))
    long_ = (
        (c > o) & (c.shift() > o.shift()) &
        (o.shift() < o) & (c > c.shift()) &
        (l < l.shift()) & (l < o.shift())
    )
    short_ = (
        (c < o) & (c.shift() < o.shift()) &
        (o.shift() > o) & (c < c.shift()) &
        (h > h.shift()) & (h > o.shift())
    )
    if use_trend:
        trend = _ema(c, ema_t)
        long_  = long_  & (c > trend)
        short_ = short_ & (c < trend)
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig


# ─────────────────────────────────────────────
# 6. DHLAOS_Geraked — Andean Oscillator (simplified, no security()) (v5)
# ─────────────────────────────────────────────
def gen_DHLAOS_Geraked(df: pd.DataFrame, params: dict) -> pd.Series:
    o, h, l, c = df.open, df.high, df.low, df.close
    ao_len   = int(params.get('ao_len', 50))
    sig_len  = int(params.get('sig_len', 9))
    swing    = int(params.get('swing', 5))
    bull, bear, signal_ = _andean_osc(c, o, ao_len, sig_len)
    # Long: bull crosses above signal (impulse)
    long_  = _crossover(bull, signal_)
    # Short: bear crosses above signal
    short_ = _crossover(bear, signal_)
    # Filter: long only when DHL confirms (simplified: use rolling high > c)
    swing_h = h.rolling(swing).max()
    swing_l = l.rolling(swing).min()
    long_  = long_  & (c > swing_l.shift())
    short_ = short_ & (c < swing_h.shift())
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_DHLAOS_Geraked(trial) -> dict:
    return {
        'ao_len':   trial.suggest_int('ao_len', 20, 100),
        'sig_len':  trial.suggest_int('sig_len', 5, 20),
        'swing':    trial.suggest_int('swing', 3, 15),
    }


# ─────────────────────────────────────────────
# 7. LRCUTB_Geraked — LinReg Candles + UT Bot (v5)
# ─────────────────────────────────────────────
def gen_LRCUTB_Geraked(df: pd.DataFrame, params: dict) -> pd.Series:
    o, h, l, c = df.open, df.high, df.low, df.close
    lr_n      = int(params.get('lr_n', 11))
    sig_len   = int(params.get('sig_len', 7))
    ut_kv     = params.get('ut_kv', 2.0)
    ut_atr    = int(params.get('ut_atr', 1))
    # LinReg Candles
    bo, bh, bl, bc = _linreg_candles(o, h, l, c, lr_n)
    signal_ = _ema(bc, sig_len)
    # UT Bot
    ut = _ut_bot(c, h, l, ut_kv, ut_atr)
    buy_ut  = ut > 0
    sell_ut = ut < 0
    # Long: UT Bot buy + linreg close > signal
    long_  = buy_ut  & (bc > signal_)
    # Short: UT Bot sell + linreg close < signal
    short_ = sell_ut & (bc < signal_)
    # Exit on linreg candle color change
    lr_red  = (bc < bo)  # LinReg red candle → close long
    lr_green = (bc > bo) # LinReg green → close short
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_LRCUTB_Geraked(trial) -> dict:
    return {
        'lr_n':     trial.suggest_int('lr_n', 5, 30),
        'sig_len':  trial.suggest_int('sig_len', 3, 20),
        'ut_kv':    trial.suggest_float('ut_kv', 1.0, 5.0),
        'ut_atr':   trial.suggest_int('ut_atr', 1, 3),
    }


# ─────────────────────────────────────────────
# 8. MLCEL_MACD_Geraked — Rational Quadratic + MACD (v5, simplified)
# ─────────────────────────────────────────────
def gen_MLCEL_MACD_Geraked(df: pd.DataFrame, params: dict) -> pd.Series:
    c = df.close
    rq_lb    = int(params.get('rq_lb', 8))
    rq_rw    = params.get('rq_rw', 8.0)
    macd_f   = int(params.get('macd_f', 12))
    macd_s   = int(params.get('macd_s', 26))
    macd_sig = int(params.get('macd_sig', 9))
    # Rational Quadratic Kernel (lorentzian-inspired smoothing)
    # Approximation: weighted EMA where weights decay polynomially
    def rq_kernel(s: pd.Series, lb: int, rw: float) -> pd.Series:
        w = np.array([(1 + (i**2) / (lb**2 * 2 * rw))**(-rw)
                      for i in range(lb)])
        w /= w.sum()
        return s.rolling(lb).apply(lambda x: np.dot(x[::-1], w[:len(x)]), raw=True)
    rq = rq_kernel(c, rq_lb, rq_rw)
    rq_prev = rq.shift()
    # MACD
    ema_f = _ema(c, macd_f); ema_s = _ema(c, macd_s)
    macd_line = ema_f - ema_s
    macd_signal = _ema(macd_line, macd_sig)
    # Long: RQ trending up + MACD crosses above signal
    long_  = (rq > rq_prev) & _crossover(macd_line, macd_signal)
    short_ = (rq < rq_prev) & _crossunder(macd_line, macd_signal)
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_MLCEL_MACD_Geraked(trial) -> dict:
    return {
        'rq_lb':    trial.suggest_int('rq_lb', 4, 20),
        'rq_rw':    trial.suggest_float('rq_rw', 2.0, 16.0),
        'macd_f':   trial.suggest_int('macd_f', 5, 20),
        'macd_s':   trial.suggest_int('macd_s', 15, 50),
        'macd_sig': trial.suggest_int('macd_sig', 3, 15),
    }


# ─────────────────────────────────────────────
# 9. NWERSI_Geraked — Nadaraya-Watson + RSI (v5, simplified)
# ─────────────────────────────────────────────
def gen_NWERSI_Geraked(df: pd.DataFrame, params: dict) -> pd.Series:
    h, l, c = df.high, df.low, df.close
    nw_h    = params.get('nw_h', 8.0)
    nw_mult = params.get('nw_mult', 3.0)
    rsi_n   = int(params.get('rsi_n', 14))
    rsi_os  = params.get('rsi_os', 30)
    rsi_ob  = params.get('rsi_ob', 70)
    # Nadaraya-Watson Envelope (Gaussian kernel)
    def nw_envelope(s: pd.Series, h_bw: float, mult: float):
        n = len(s)
        nw = pd.Series(np.nan, index=s.index)
        vals = s.values
        for i in range(n):
            weights = np.array([np.exp(-(j**2) / (h_bw**2 * 2))
                                 for j in range(min(i+1, n))])
            w_sum = weights.sum()
            if w_sum > 0:
                nw.iloc[i] = np.dot(vals[max(0, i-len(weights)+1):i+1][::-1],
                                    weights[:min(i+1, n)]) / w_sum
        # Envelope bands
        residuals = (s - nw).abs()
        band = mult * residuals.rolling(20).mean()
        return nw, nw + band, nw - band
    # Use a fast approximation with rolling weighted avg
    def fast_nw(s: pd.Series, h_bw: float) -> pd.Series:
        lb = max(3, int(h_bw * 3))
        weights = np.array([np.exp(-(j**2) / (h_bw**2 * 2)) for j in range(lb)])
        weights /= weights.sum()
        return s.rolling(lb).apply(lambda x: np.dot(x[::-1], weights[:len(x)]), raw=True)
    nw = fast_nw(c, nw_h)
    residuals = (c - nw).abs()
    band = nw_mult * residuals.rolling(20).mean()
    upper = nw + band
    lower = nw - band
    rsi_ = _rsi(c, rsi_n)
    # Long: close crosses above lower NW band + RSI oversold
    long_  = _crossover(c, lower) & (rsi_ < rsi_os + 20)
    # Short: close crosses below upper NW band + RSI overbought
    short_ = _crossunder(c, upper) & (rsi_ > rsi_ob - 20)
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_NWERSI_Geraked(trial) -> dict:
    return {
        'nw_h':    trial.suggest_float('nw_h', 3.0, 15.0),
        'nw_mult': trial.suggest_float('nw_mult', 1.0, 5.0),
        'rsi_n':   trial.suggest_int('rsi_n', 7, 21),
        'rsi_os':  trial.suggest_float('rsi_os', 20, 40),
        'rsi_ob':  trial.suggest_float('rsi_ob', 60, 80),
    }


# ─────────────────────────────────────────────
# 10. BBForce_Harry — BB Force Direction (v5)
# ─────────────────────────────────────────────
def gen_BBForce_Harry(df: pd.DataFrame, params: dict) -> pd.Series:
    c = df.close
    bb_len  = int(params.get('bb_len', 26))
    bb_mult = params.get('bb_mult', 2.0)
    upper, basis, lower = _bb(c, bb_len, bb_mult)
    # Bull: all three bands moving up
    bull = (upper > upper.shift()) & (lower > lower.shift()) & (basis > basis.shift())
    bear = (upper < upper.shift()) & (lower < lower.shift()) & (basis < basis.shift())
    # Signal on state change
    prev_bull = bull.shift().fillna(False)
    prev_bear = bear.shift().fillna(False)
    long_  = bull & ~prev_bull & ~prev_bear  # entering bull from neutral
    short_ = bear & ~prev_bear & ~prev_bull  # entering bear from neutral
    # Also: if was bull, now bear → short; if was bear, now bull → long
    long_  = long_  | (bull & prev_bear)
    short_ = short_ | (bear & prev_bull)
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_BBForce_Harry(trial) -> dict:
    return {
        'bb_len':  trial.suggest_int('bb_len', 10, 60),
        'bb_mult': trial.suggest_float('bb_mult', 1.0, 3.5),
    }


# ─────────────────────────────────────────────
# 11. HOLP_Harry — High of Low Period (v5)
# ─────────────────────────────────────────────
def gen_HOLP_Harry(df: pd.DataFrame, params: dict) -> pd.Series:
    h, l, c = df.high, df.low, df.close
    n_look = int(params.get('n_look', 10))
    rr     = params.get('rr', 3.0)
    # HOLP: find N-bar low, record its high, enter when close breaks above that high
    n_bar_low_idx = l.rolling(n_look).apply(lambda x: np.argmin(x), raw=True).astype(float)
    # High at the N-bar low bar
    holp = pd.Series(np.nan, index=c.index)
    for i in range(n_look, len(c)):
        lo_idx = int(n_bar_low_idx.iloc[i])
        holp.iloc[i] = h.iloc[i - (n_look - 1 - lo_idx)]
    # LOHP: find N-bar high, record its low, enter when close breaks below that low
    n_bar_high_idx = h.rolling(n_look).apply(lambda x: np.argmax(x), raw=True).astype(float)
    lohp = pd.Series(np.nan, index=c.index)
    for i in range(n_look, len(c)):
        hi_idx = int(n_bar_high_idx.iloc[i])
        lohp.iloc[i] = l.iloc[i - (n_look - 1 - hi_idx)]
    long_  = _crossover(c, holp.shift())
    short_ = _crossunder(c, lohp.shift())
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_HOLP_Harry(trial) -> dict:
    return {
        'n_look': trial.suggest_int('n_look', 5, 25),
        'rr':     trial.suggest_float('rr', 1.5, 5.0),
    }


# ─────────────────────────────────────────────
# 12. KijunArrow_Harry — Ichimoku Kijun Crossover (v5)
# ─────────────────────────────────────────────
def gen_KijunArrow_Harry(df: pd.DataFrame, params: dict) -> pd.Series:
    h, l, c = df.high, df.low, df.close
    kijun_n  = int(params.get('kijun_n', 26))
    tenkan_n = int(params.get('tenkan_n', 9))
    atr_sl   = int(params.get('atr_sl', 3))
    kijun  = _kijun(h, l, kijun_n)
    tenkan = _kijun(h, l, tenkan_n)
    # Long: close crosses above Kijun + tenkan > kijun (weak chikou filter)
    long_  = _crossover(c, kijun) & (tenkan >= kijun)
    # Short: close crosses below Kijun + tenkan < kijun
    short_ = _crossunder(c, kijun) & (tenkan <= kijun)
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_KijunArrow_Harry(trial) -> dict:
    return {
        'kijun_n':  trial.suggest_int('kijun_n', 10, 52),
        'tenkan_n': trial.suggest_int('tenkan_n', 3, 20),
        'atr_sl':   trial.suggest_int('atr_sl', 1, 5),
    }


# ─────────────────────────────────────────────
# 13. OneBurPursuit_Harry — One Bar Pursuit (ATR entry) (v5)
# ─────────────────────────────────────────────
def gen_OneBurPursuit_Harry(df: pd.DataFrame, params: dict) -> pd.Series:
    o, h, l, c = df.open, df.high, df.low, df.close
    atr_n    = int(params.get('atr_n', 14))
    atr_mult = params.get('atr_mult', 1.0)
    # Bull bar: close > open. Bear bar: close < open
    bull_bar  = c > o
    bear_bar  = c < o
    # Long: bearish bar [1] then bullish bar [0]
    long_entry  = bear_bar.shift().fillna(False) & bull_bar
    # Short: bullish bar [1] then bearish bar [0]
    short_entry = bull_bar.shift().fillna(False) & bear_bar
    # Entry price would be close[1] + ATR * mult, but for signal we fire on candle close
    atr_v = _atr(h, l, c, atr_n)
    # Use ATR as additional filter: only enter if ATR is above median (volatile enough)
    # ATR filter: volatility above rolling min * mult (permissive by default)
    atr_min = atr_v.rolling(20).min()
    long_  = long_entry  & (atr_v > atr_min * atr_mult)
    short_ = short_entry & (atr_v > atr_min * atr_mult)
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_OneBurPursuit_Harry(trial) -> dict:
    return {
        'atr_n':    trial.suggest_int('atr_n', 7, 21),
        'atr_mult': trial.suggest_float('atr_mult', 0.5, 2.0),
    }


# ─────────────────────────────────────────────
# 14. MTF_BB_JN842 — Multi-TF Bollinger Bands (v6, simplified HTF via longer period)
# ─────────────────────────────────────────────
def gen_MTF_BB_JN842(df: pd.DataFrame, params: dict) -> pd.Series:
    h, l, c = df.high, df.low, df.close
    ltf_len  = int(params.get('ltf_len', 20))
    htf_mult = int(params.get('htf_mult', 12))  # HTF = LTF × mult
    bb_mult  = params.get('bb_mult', 2.0)
    htf_len  = ltf_len * htf_mult
    # LTF bands (on high for long, on low for short)
    ltf_up_u, ltf_up_m, ltf_up_l = _bb(h, ltf_len, bb_mult)  # Long bands (high-based)
    ltf_dn_u, ltf_dn_m, ltf_dn_l = _bb(l, ltf_len, bb_mult)  # Short bands (low-based)
    # HTF bands (close-based, longer period)
    htf_u, htf_m, htf_l = _bb(c, min(htf_len, 500), bb_mult)
    # Long: LTF close crosses above lower long band + HTF is above HTF lower
    long_  = _crossover(c, ltf_up_l) & (c > htf_l) & (c < htf_m)
    # Short: LTF close crosses below upper short band + HTF below upper
    short_ = _crossunder(c, ltf_dn_u) & (c < htf_u) & (c > htf_m)
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_MTF_BB_JN842(trial) -> dict:
    return {
        'ltf_len':  trial.suggest_int('ltf_len', 10, 40),
        'htf_mult': trial.suggest_int('htf_mult', 4, 24),
        'bb_mult':  trial.suggest_float('bb_mult', 1.0, 3.0),
    }


# ─────────────────────────────────────────────
# 15. kNN_Trend_RSC809 — kNN-based Trend Following (v5)
# ─────────────────────────────────────────────
def _knn_predict(f1: np.ndarray, f2: np.ndarray, k: int,
                 lookback: int) -> np.ndarray:
    """Vectorized kNN with fixed lookback window."""
    n = len(f1)
    preds = np.zeros(n)
    for i in range(lookback, n):
        start = max(0, i - lookback)
        h_f1 = f1[start:i]
        h_f2 = f2[start:i]
        # Direction: was next bar up or down
        next_close_change = np.diff(np.append(h_f1, f1[i]))  # rough proxy
        directions = np.sign(next_close_change[:-1]) if len(next_close_change) > 1 else np.array([0])
        if len(h_f1) < k + 1:
            continue
        dist = np.sqrt((h_f1[:-1] - f1[i])**2 + (h_f2[:-1] - f2[i])**2)
        # k-nearest
        kk = min(k, len(dist))
        k_idx = np.argpartition(dist, kk-1)[:kk]
        k_dir = directions[:len(dist)][k_idx]
        preds[i] = 1.0 if np.sum(k_dir > 0) > np.sum(k_dir < 0) else (
                  -1.0 if np.sum(k_dir < 0) > np.sum(k_dir > 0) else 0.0)
    return preds

def gen_kNN_Trend_RSC809(df: pd.DataFrame, params: dict) -> pd.Series:
    h, l, c, v = df.high, df.low, df.close, df.volume
    short_w  = int(params.get('short_w', 8))
    long_w   = int(params.get('long_w', 29))
    k_val    = int(params.get('k_val', 20))
    lookback = int(params.get('lookback', 300))
    # Features
    rsi_long  = _rsi(c, long_w)
    rsi_short = _rsi(c, short_w)
    # Minimax normalize
    f1 = _minimax(rsi_long, long_w)
    f2 = _minimax(rsi_short, short_w)
    f1_arr = f1.fillna(50).values
    f2_arr = f2.fillna(50).values
    k = int(np.floor(np.sqrt(k_val)))
    preds = _knn_predict(f1_arr, f2_arr, k, lookback)
    sig = pd.Series(preds, index=c.index)
    return sig

def space_kNN_Trend_RSC809(trial) -> dict:
    return {
        'short_w':  trial.suggest_int('short_w', 4, 15),
        'long_w':   trial.suggest_int('long_w', 15, 50),
        'k_val':    trial.suggest_int('k_val', 50, 600),
        'lookback': trial.suggest_int('lookback', 100, 500),
    }


# ─────────────────────────────────────────────
# 16. BCGA_zemulist — BC-GA Optimized v5 (composite CF score)
# ─────────────────────────────────────────────
def gen_BCGA_zemulist(df: pd.DataFrame, params: dict) -> pd.Series:
    h, l, c, v = df.high, df.low, df.close, df.volume
    cf_thresh  = params.get('cf_thresh', 70.0)
    rsi_n      = int(params.get('rsi_n', 14))
    macd_f     = int(params.get('macd_f', 12))
    macd_s     = int(params.get('macd_s', 26))
    macd_sig   = int(params.get('macd_sig', 9))
    adx_n      = int(params.get('adx_n', 14))
    # Indicators
    rsi_ = _rsi(c, rsi_n)
    ema_f = _ema(c, macd_f); ema_s = _ema(c, macd_s)
    macd_line = ema_f - ema_s; macd_signal_ = _ema(macd_line, macd_sig)
    upper_bb, mid_bb, lower_bb = _bb(c, 20, 2.0)
    atr_ = _atr(h, l, c, 14)
    sma20 = _sma(c, 20); sma50 = _sma(c, 50); sma200 = _sma(c, 200)
    ema9 = _ema(c, 9); ema21 = _ema(c, 21)
    vol_sma = _sma(v, 20)
    # CF Score (0–100): composite of signals
    # Each sub-signal contributes ~10–25 points
    cf = pd.Series(0.0, index=c.index)
    cf += (rsi_ > 50).astype(float) * 15        # RSI bullish
    cf += (macd_line > macd_signal_).astype(float) * 20  # MACD bullish
    cf += (c > mid_bb).astype(float) * 10        # BB above mid
    cf += (sma20 > sma50).astype(float) * 15     # EMA cascade
    cf += (sma50 > sma200).astype(float) * 15    # long-term trend
    cf += (ema9 > ema21).astype(float) * 10      # short-term momentum
    cf += (v > vol_sma).astype(float) * 10       # volume confirm
    cf += ((c - lower_bb) / (upper_bb - lower_bb + 1e-9) * 5)  # BB position
    # Bear CF
    cf_bear = pd.Series(0.0, index=c.index)
    cf_bear += (rsi_ < 50).astype(float) * 15
    cf_bear += (macd_line < macd_signal_).astype(float) * 20
    cf_bear += (c < mid_bb).astype(float) * 10
    cf_bear += (sma20 < sma50).astype(float) * 15
    cf_bear += (sma50 < sma200).astype(float) * 15
    cf_bear += (ema9 < ema21).astype(float) * 10
    cf_bear += (v > vol_sma).astype(float) * 10
    # Signals when CF crosses threshold
    long_  = (cf >= cf_thresh) & (cf.shift().fillna(0) < cf_thresh)
    short_ = (cf_bear >= cf_thresh) & (cf_bear.shift().fillna(0) < cf_thresh)
    sig = pd.Series(0, index=c.index)
    sig[long_]  =  1
    sig[short_] = -1
    return sig

def space_BCGA_zemulist(trial) -> dict:
    return {
        'cf_thresh':  trial.suggest_float('cf_thresh', 55.0, 85.0),
        'rsi_n':      trial.suggest_int('rsi_n', 7, 21),
        'macd_f':     trial.suggest_int('macd_f', 5, 20),
        'macd_s':     trial.suggest_int('macd_s', 15, 50),
        'macd_sig':   trial.suggest_int('macd_sig', 3, 15),
        'adx_n':      trial.suggest_int('adx_n', 7, 21),
    }


# ─────────────────────────────────────────────
# 17. TheOneRing_zemulist — Regime-Adaptive Trading (v5)
# ─────────────────────────────────────────────
def gen_TheOneRing_zemulist(df: pd.DataFrame, params: dict) -> pd.Series:
    h, l, c = df.high, df.low, df.close
    adx_n      = int(params.get('adx_n', 14))
    adx_thresh = params.get('adx_thresh', 25.0)
    rsi_n      = int(params.get('rsi_n', 14))
    bb_n       = int(params.get('bb_n', 20))
    bb_mult    = params.get('bb_mult', 2.0)
    ema_n      = int(params.get('ema_n', 50))
    # Regime Detection: ADX > threshold = trending; else = ranging
    # ADX proxy: ratio of directional movement to ATR
    atr_ = _atr(h, l, c, adx_n)
    dm_plus  = (h - h.shift()).clip(lower=0)
    dm_minus = (l.shift() - l).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    di_plus  = 100 * dm_plus.ewm(alpha=1/adx_n, adjust=False).mean() / atr_.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(alpha=1/adx_n, adjust=False).mean() / atr_.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1/adx_n, adjust=False).mean()
    trending = adx_ >= adx_thresh
    # Trend-following signals
    ema_ = _ema(c, ema_n)
    tf_long  = trending & (c > ema_) & _crossover(c, ema_)
    tf_short = trending & (c < ema_) & _crossunder(c, ema_)
    # Mean-reversion signals (ranging)
    upper_bb, mid_bb, lower_bb = _bb(c, bb_n, bb_mult)
    rsi_ = _rsi(c, rsi_n)
    mr_long  = ~trending & _crossover(c, lower_bb) & (rsi_ < 40)
    mr_short = ~trending & _crossunder(c, upper_bb) & (rsi_ > 60)
    sig = pd.Series(0, index=c.index)
    sig[tf_long | mr_long]  =  1
    sig[tf_short | mr_short] = -1
    return sig

def space_TheOneRing_zemulist(trial) -> dict:
    return {
        'adx_n':      trial.suggest_int('adx_n', 7, 21),
        'adx_thresh': trial.suggest_float('adx_thresh', 15.0, 40.0),
        'rsi_n':      trial.suggest_int('rsi_n', 7, 21),
        'bb_n':       trial.suggest_int('bb_n', 10, 40),
        'bb_mult':    trial.suggest_float('bb_mult', 1.0, 3.0),
        'ema_n':      trial.suggest_int('ema_n', 20, 100),
    }


# ─────────────────────────────────────────────
# 18. Setup92_RodrigoBrito — Larry Williams Setup 9.2 (v4)
# ─────────────────────────────────────────────
def gen_Setup92_RodrigoBrito(df: pd.DataFrame, params: dict) -> pd.Series:
    c = df.close
    ema9_n   = int(params.get('ema9_n', 9))
    sma20_n  = int(params.get('sma20_n', 20))
    sma54_n  = int(params.get('sma54_n', 54))
    sma200_n = int(params.get('sma200_n', 200))
    # EMAs
    ema9_   = _ema(c, ema9_n)
    sma20_  = _sma(c, sma20_n)
    sma54_  = _sma(c, sma54_n)
    sma200_ = _sma(c, sma200_n)
    # Trend: ema9 rising
    ema9_rising  = ema9_ > ema9_.shift()
    ema9_falling = ema9_ < ema9_.shift()
    # Setup: close < low[1] (pullback) while trend up
    pullback_long  = (c < df.low.shift()) & ema9_rising  & (c > sma54_)
    pullback_short = (c > df.high.shift()) & ema9_falling & (c < sma54_)
    # Entry: next bar breaks above high of setup bar
    entry_long  = pullback_long.shift().fillna(False) & (c > df.high.shift(2).fillna(0))
    entry_short = pullback_short.shift().fillna(False) & (c < df.low.shift(2).fillna(np.inf))
    sig = pd.Series(0, index=c.index)
    sig[entry_long]  =  1
    sig[entry_short] = -1
    return sig

def space_Setup92_RodrigoBrito(trial) -> dict:
    return {
        'ema9_n':    trial.suggest_int('ema9_n', 5, 15),
        'sma20_n':   trial.suggest_int('sma20_n', 10, 30),
        'sma54_n':   trial.suggest_int('sma54_n', 30, 80),
        'sma200_n':  trial.suggest_int('sma200_n', 100, 250),
    }


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
STRATEGY_EXPORT = {
    '3MAF_Geraked':            {'gen': gen_3MAF_Geraked,            'space': space_3MAF_Geraked},
    'ABSAR_Geraked':           {'gen': gen_ABSAR_Geraked,           'space': space_ABSAR_Geraked},
    'BBRSI_Geraked':           {'gen': gen_BBRSI_Geraked,           'space': space_BBRSI_Geraked},
    'CEZLSMA_Geraked':         {'gen': gen_CEZLSMA_Geraked,         'space': space_CEZLSMA_Geraked},
    'CP1_Geraked':             {'gen': gen_CP1_Geraked,             'space': space_CP1_Geraked},
    'DHLAOS_Geraked':          {'gen': gen_DHLAOS_Geraked,          'space': space_DHLAOS_Geraked},
    'LRCUTB_Geraked':          {'gen': gen_LRCUTB_Geraked,          'space': space_LRCUTB_Geraked},
    'MLCEL_MACD_Geraked':      {'gen': gen_MLCEL_MACD_Geraked,      'space': space_MLCEL_MACD_Geraked},
    'NWERSI_Geraked':          {'gen': gen_NWERSI_Geraked,          'space': space_NWERSI_Geraked},
    'BBForce_Harry':           {'gen': gen_BBForce_Harry,           'space': space_BBForce_Harry},
    'HOLP_Harry':              {'gen': gen_HOLP_Harry,              'space': space_HOLP_Harry},
    'KijunArrow_Harry':        {'gen': gen_KijunArrow_Harry,        'space': space_KijunArrow_Harry},
    'OneBurPursuit_Harry':     {'gen': gen_OneBurPursuit_Harry,     'space': space_OneBurPursuit_Harry},
    'MTF_BB_JN842':            {'gen': gen_MTF_BB_JN842,            'space': space_MTF_BB_JN842},
    'kNN_Trend_RSC809':        {'gen': gen_kNN_Trend_RSC809,        'space': space_kNN_Trend_RSC809},
    'BCGA_zemulist':           {'gen': gen_BCGA_zemulist,           'space': space_BCGA_zemulist},
    'TheOneRing_zemulist':     {'gen': gen_TheOneRing_zemulist,     'space': space_TheOneRing_zemulist},
    'Setup92_RodrigoBrito':    {'gen': gen_Setup92_RodrigoBrito,    'space': space_Setup92_RodrigoBrito},
}
