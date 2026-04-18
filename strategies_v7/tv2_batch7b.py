#!/usr/bin/env python3
"""
TV2 BATCH 7B — 10 estrategias convertidas de Pine Script a Python.

Estrategias:
  DEMA_ATR             — DEMA + ATR trailing band crossover (v6, 1438L)
  Triple_EMA_Turtle    — Triple EMA + ADX Turtle (v6, 1398L)
  BB_Enhanced          — Bollinger Bands + EMA200 + ATR (v5, 1280L)
  Triple_EMA_QQE       — Triple EMA (TEMA) + QQE signal (v5, 1265L)
  Hull_T3_Swing        — HULL MA + T3 Tillson average (v4, 1074L)
  ATR_Stop_Multiple    — SMA crossover + ATR stop (v4, 1014L)
  ATR_MA_Trend         — ATR breakout + MA trend (v6, 1003L)
  Crypto_Squeeze       — Lazy Bear Momentum + MFI (v4, 974L)
  Commas_Bollinger     — Bollinger crossover DCA-style (v4, 962L)
  EMA_RSI_ADX          — EMA cross + RSI filter + ADX (v6, 898L)
"""

import numpy as np
import pandas as pd

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _sma(s, p):
    return s.rolling(p).mean()

def _rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0).ewm(span=p, adjust=False).mean()
    l = (-d.where(d < 0, 0.0)).ewm(span=p, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def _atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def _wma(s, p):
    w = np.arange(1, p + 1, dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def _hma(s, p):
    return _wma(2 * _wma(s, p // 2) - _wma(s, p), max(1, int(p ** 0.5)))

def _bb(c, p=20, m=2.0):
    b = _sma(c, p)
    s = c.rolling(p).std()
    return b, b + m * s, b - m * s

def _stoch(c, h, l, k=14, d=3):
    lo = l.rolling(k).min()
    hi = h.rolling(k).max()
    ks = 100 * (c - lo) / (hi - lo + 1e-10)
    ds = _sma(ks, d)
    return ks, ds

def _dema(s, p):
    e = _ema(s, p)
    return 2 * e - _ema(e, p)

def _adx(h, l, c, p=14):
    up = h.diff()
    dn = -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=p, adjust=False).mean()
    pdi = 100 * pdm.ewm(span=p, adjust=False).mean() / (atr14 + 1e-10)
    ndi = 100 * ndm.ewm(span=p, adjust=False).mean() / (atr14 + 1e-10)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-10)
    return dx.ewm(span=p, adjust=False).mean(), pdi, ndi

def _t3(s, p, v=0.7):
    """Tillson T3: chain of 6 EMAs with Tillson coefficients."""
    c1 = -(v ** 3)
    c2 = 3 * v ** 2 + 3 * v ** 3
    c3 = -6 * v ** 2 - 3 * v - 3 * v ** 3
    c4 = 1 + 3 * v + v ** 3 + 3 * v ** 2
    e1 = _ema(s, p)
    e2 = _ema(e1, p)
    e3 = _ema(e2, p)
    e4 = _ema(e3, p)
    e5 = _ema(e4, p)
    e6 = _ema(e5, p)
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

def _tema(s, p):
    """Triple EMA (TEMA): 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    e1 = _ema(s, p)
    e2 = _ema(e1, p)
    e3 = _ema(e2, p)
    return 3 * e1 - 3 * e2 + e3

def _crossover(a, b):
    """Returns True where a crosses above b."""
    return (a > b) & (a.shift(1) <= b.shift(1))

def _crossunder(a, b):
    """Returns True where a crosses below b."""
    return (a < b) & (a.shift(1) >= b.shift(1))

# ─── 1. DEMA_ATR_Strategy ───────────────────────────────────────────────────────
# Logic: DEMA + ATR band trailing → crossover/crossunder of trailing line

def gen_DEMA_ATR(df, **params):
    period_dema = int(params.get('period_dema', 75))
    period_atr  = int(params.get('period_atr', 20))
    factor_atr  = float(params.get('factor_atr', 1.20))

    c = df['close']
    h = df['high']
    l = df['low']

    dema_out = _dema(c, period_dema)
    true_range = _atr(h, l, c, period_atr) * factor_atr

    upper = dema_out + true_range
    lower = dema_out - true_range

    # Build DemaAtr trailing line iteratively
    dema_atr = dema_out.copy()
    da_vals = dema_atr.values.copy().astype(float)
    lo_vals = lower.values
    up_vals = upper.values

    for i in range(1, len(da_vals)):
        prev = da_vals[i - 1]
        if np.isnan(prev):
            prev = da_vals[i]
        if lo_vals[i] > prev:
            da_vals[i] = lo_vals[i]
        elif up_vals[i] < prev:
            da_vals[i] = up_vals[i]
        else:
            da_vals[i] = prev

    da = pd.Series(da_vals, index=df.index)

    # Signal: crossover(da, da[2]) → long; crossunder(da, da[2]) → short
    da2 = da.shift(2)
    long_cond  = _crossover(da, da2)
    short_cond = _crossunder(da, da2)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig.fillna(0).astype(int)


def space_DEMA_ATR(trial):
    return {
        'period_dema': trial.suggest_int('period_dema', 20, 150),
        'period_atr':  trial.suggest_int('period_atr', 7, 50),
        'factor_atr':  trial.suggest_float('factor_atr', 0.5, 3.0, step=0.05),
    }


# ─── 2. Triple_EMA_Turtle ───────────────────────────────────────────────────────
# Logic: Triple EMA alignment + ADX filter

def gen_Triple_EMA_Turtle(df, **params):
    fast_len  = int(params.get('ema_fast', 30))
    mid_len   = int(params.get('ema_mid', 46))
    slow_len  = int(params.get('ema_slow', 80))
    adx_thresh = float(params.get('adx_thresh', 43.0))
    adx_len   = int(params.get('adx_len', 14))

    c = df['close']
    h = df['high']
    l = df['low']

    ema_fast = _ema(c, fast_len)
    ema_mid  = _ema(c, mid_len)
    ema_slow = _ema(c, slow_len)
    adx_val, _, _ = _adx(h, l, c, adx_len)

    long_cond  = (adx_val > adx_thresh) & (ema_fast > ema_mid) & (ema_mid > ema_slow) & (c > ema_fast)
    short_cond = (adx_val > adx_thresh) & (ema_fast < ema_mid) & (ema_mid < ema_slow) & (c < ema_fast)

    # Entry only on first bar of condition (transition)
    long_entry  = long_cond & ~long_cond.shift(1).fillna(False)
    short_entry = short_cond & ~short_cond.shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    sig[long_entry]  = 1
    sig[short_entry] = -1
    return sig.fillna(0).astype(int)


def space_Triple_EMA_Turtle(trial):
    return {
        'ema_fast':   trial.suggest_int('ema_fast', 10, 60),
        'ema_mid':    trial.suggest_int('ema_mid', 30, 80),
        'ema_slow':   trial.suggest_int('ema_slow', 60, 150),
        'adx_thresh': trial.suggest_float('adx_thresh', 20.0, 60.0, step=1.0),
        'adx_len':    trial.suggest_int('adx_len', 7, 21),
    }


# ─── 3. BB_Enhanced ─────────────────────────────────────────────────────────────
# Logic: Price below lower BB + above EMA200 → long entry
# Exit: price crosses below basis SMA (trailing TP) or ATR stop

def gen_BB_Enhanced(df, **params):
    bb_period = int(params.get('bb_period', 20))
    bb_mult   = float(params.get('bb_mult', 2.0))
    ema_len   = int(params.get('ema_len', 200))

    c = df['close']
    l = df['low']
    h = df['high']

    basis, upper, lower = _bb(c, bb_period, bb_mult)
    filter_ema = _ema(c, ema_len)

    # Long condition: low touches/below lower band AND above EMA200
    long_cond = (l <= lower) & (l > filter_ema)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    return sig.fillna(0).astype(int)


def space_BB_Enhanced(trial):
    return {
        'bb_period': trial.suggest_int('bb_period', 10, 50),
        'bb_mult':   trial.suggest_float('bb_mult', 1.0, 3.5, step=0.25),
        'ema_len':   trial.suggest_int('ema_len', 100, 300),
    }


# ─── 4. Triple_EMA_QQE ──────────────────────────────────────────────────────────
# Logic: TEMA1 > TEMA2 uptrend + QQE crossover → long
# QQE: RSI smoothed + trailing stop line from RSI ATR

def gen_Triple_EMA_QQE(df, **params):
    tema1_len = int(params.get('tema1_len', 20))
    tema2_len = int(params.get('tema2_len', 40))
    rsi_period = int(params.get('rsi_period', 14))
    rsi_smooth = int(params.get('rsi_smooth', 5))
    qqe_factor = float(params.get('qqe_factor', 4.238))

    c = df['close']

    tema1 = _tema(c, tema1_len)
    tema2 = _tema(c, tema2_len)

    # QQE calculation
    rsi_val = _rsi(c, rsi_period)
    rsi_ma  = _ema(rsi_val, rsi_smooth)
    atr_rsi = (rsi_ma - rsi_ma.shift(1)).abs()

    final_smooth = rsi_period * 2 - 1
    ma_atr_rsi = _ema(atr_rsi, final_smooth)
    dar = _ema(ma_atr_rsi, final_smooth) * qqe_factor

    # Build QQE trailing line iteratively
    ts_vals = rsi_ma.values.copy()
    rm_vals = rsi_ma.values
    dar_vals = dar.values

    for i in range(1, len(ts_vals)):
        prev_ts = ts_vals[i - 1]
        rm = rm_vals[i]
        d = dar_vals[i]
        if np.isnan(d) or np.isnan(rm):
            continue
        # crossover / crossunder logic
        if rm_vals[i - 1] < prev_ts and rm > prev_ts:
            # crossover: ts = rsiMa - dar
            ts_vals[i] = rm - d
        elif rm_vals[i - 1] > prev_ts and rm < prev_ts:
            # crossunder: ts = rsiMa + dar
            ts_vals[i] = rm + d
        elif rm > prev_ts:
            ts_vals[i] = max(rm - d, prev_ts)
        else:
            ts_vals[i] = min(rm + d, prev_ts)

    ts = pd.Series(ts_vals, index=df.index)

    # Entry conditions
    # Long: close > tema1 AND tema1 > tema2 AND tema2 rising AND rsiMa crossover ts
    tema2_rising = tema2 > tema2.shift(1)
    qqe_cross_up   = _crossover(rsi_ma, ts)
    qqe_cross_down = _crossunder(rsi_ma, ts)

    long_cond  = (c > tema1) & (tema1 > tema2) & tema2_rising & qqe_cross_up
    short_cond = (c < tema1) & (tema1 < tema2) & (~tema2_rising) & qqe_cross_down

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig.fillna(0).astype(int)


def space_Triple_EMA_QQE(trial):
    return {
        'tema1_len':  trial.suggest_int('tema1_len', 10, 40),
        'tema2_len':  trial.suggest_int('tema2_len', 30, 80),
        'rsi_period': trial.suggest_int('rsi_period', 7, 21),
        'rsi_smooth': trial.suggest_int('rsi_smooth', 3, 10),
        'qqe_factor': trial.suggest_float('qqe_factor', 2.0, 6.0, step=0.1),
    }


# ─── 5. Hull_T3_Swing ────────────────────────────────────────────────────────────
# Logic: (HMA + T3) / 2 average direction → long when avg rising, short when falling

def gen_Hull_T3_Swing(df, **params):
    length = int(params.get('length', 50))
    t3_factor = float(params.get('t3_factor', 0.7))

    c = df['close']

    hma_val = _hma(c, length)
    t3_val  = _t3(c, length, t3_factor)
    avg = (hma_val + t3_val) / 2

    long_cond  = avg > avg.shift(1)
    short_cond = avg < avg.shift(1)

    # Entry on direction change
    long_entry  = long_cond & ~long_cond.shift(1).fillna(False)
    short_entry = short_cond & ~short_cond.shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    sig[long_entry]  = 1
    sig[short_entry] = -1
    return sig.fillna(0).astype(int)


def space_Hull_T3_Swing(trial):
    return {
        'length':    trial.suggest_int('length', 20, 100),
        't3_factor': trial.suggest_float('t3_factor', 0.3, 0.9, step=0.05),
    }


# ─── 6. ATR_Stop_Multiple ────────────────────────────────────────────────────────
# Logic: fast/slow SMA crossover entry, ATR-multiple stop loss

def gen_ATR_Stop_Multiple(df, **params):
    fast_period = int(params.get('fast_period', 15))
    slow_period = int(params.get('slow_period', 45))

    c = df['close']

    fast_sma = _sma(c, fast_period)
    slow_sma = _sma(c, slow_period)

    long_cond  = _crossover(fast_sma, slow_sma)
    short_cond = _crossunder(fast_sma, slow_sma)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig.fillna(0).astype(int)


def space_ATR_Stop_Multiple(trial):
    return {
        'fast_period': trial.suggest_int('fast_period', 5, 30),
        'slow_period': trial.suggest_int('slow_period', 20, 100),
    }


# ─── 7. ATR_MA_Trend ─────────────────────────────────────────────────────────────
# Logic: ATR band breakout (close > MA + ATR*mult → uptrend), confirmed N bars
# Entry on trend change

def gen_ATR_MA_Trend(df, **params):
    ma_len     = int(params.get('ma_len', 100))
    atr_len    = int(params.get('atr_len', 7))
    atr_mult   = float(params.get('atr_mult', 3.0))
    confirm_bars = int(params.get('confirm_bars', 1))

    c = df['close']
    h = df['high']
    l = df['low']

    ma_val  = _ema(c, ma_len)
    atr_val = _atr(h, l, c, atr_len)

    # Use lagged values like the Pine script (MAtmp[1] / ATRtmp[1])
    ma_lag  = ma_val.shift(1)
    atr_lag = atr_val.shift(1)

    upper = ma_lag + atr_lag * atr_mult
    lower = ma_lag - atr_lag * atr_mult

    is_up_raw   = c > upper
    is_down_raw = c < lower

    # Confirm: N consecutive bars in raw state
    def _consecutive(s, n):
        if n <= 1:
            return s
        result = s.copy()
        for i in range(1, n):
            result = result & s.shift(i).fillna(False)
        return result

    up_confirmed   = _consecutive(is_up_raw, confirm_bars)
    down_confirmed = _consecutive(is_down_raw, confirm_bars)

    # Trend state transitions
    trend = pd.Series(0, index=df.index)
    trend_vals = trend.values.copy()
    up_c   = up_confirmed.values
    down_c = down_confirmed.values

    current_trend = 0
    for i in range(len(trend_vals)):
        if up_c[i] and current_trend != 1:
            current_trend = 1
            trend_vals[i] = 1   # breakout long
        elif down_c[i] and current_trend != -1:
            current_trend = -1
            trend_vals[i] = -1  # breakout short

    sig = pd.Series(trend_vals, index=df.index).astype(int)
    return sig.fillna(0).astype(int)


def space_ATR_MA_Trend(trial):
    return {
        'ma_len':       trial.suggest_int('ma_len', 30, 200),
        'atr_len':      trial.suggest_int('atr_len', 3, 21),
        'atr_mult':     trial.suggest_float('atr_mult', 1.0, 5.0, step=0.5),
        'confirm_bars': trial.suggest_int('confirm_bars', 1, 4),
    }


# ─── 8. Crypto_Squeeze ──────────────────────────────────────────────────────────
# Logic: Lazy Bear BlueWave momentum (linreg on (close - midpoint - SMA20))
#        + MFI oscillator
# Long: BlueWave crossover 0 AND mfi > 0
# Short: BlueWave crossunder 0 AND mfi < 0

def gen_Crypto_Squeeze(df, **params):
    sq_len  = int(params.get('sq_len', 20))
    mfi_len = int(params.get('mfi_len', 58))

    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']

    # BlueWave: linreg(close - avg(avg(highest(h,20), lowest(l,20)), sma(close,20)), 20, 0)
    highest = h.rolling(sq_len).max()
    lowest  = l.rolling(sq_len).min()
    mid     = (highest + lowest) / 2
    basis   = _sma(c, sq_len)
    source  = c - (mid + basis) / 2

    # Linear regression of source over sq_len periods (value at end, lag=0)
    def _linreg(s, p):
        x = np.arange(p, dtype=float)
        result = s.rolling(p).apply(
            lambda y: np.polyfit(x, y, 1)[0] * (p - 1) + np.polyfit(x, y, 1)[1],
            raw=True
        )
        return result

    blue_wave = _linreg(source, sq_len)

    # MFI oscillator (Crypto Face style)
    hlc3 = (h + l + c) / 3
    hlc3_change = hlc3.diff()
    mfi_upper = (v * hlc3.where(hlc3_change > 0, 0.0)).rolling(mfi_len).sum()
    mfi_lower = (v * (-hlc3.where(hlc3_change < 0, 0.0))).rolling(mfi_len).sum()
    mf = 100.0 - (100.0 / (1.0 + mfi_upper / (mfi_lower + 1e-10)))
    mfi_osc = (mf - 50) * 3

    long_cond  = _crossover(blue_wave, pd.Series(0.0, index=df.index)) & (mfi_osc > 0)
    short_cond = _crossunder(blue_wave, pd.Series(0.0, index=df.index)) & (mfi_osc < 0)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig.fillna(0).astype(int)


def space_Crypto_Squeeze(trial):
    return {
        'sq_len':  trial.suggest_int('sq_len', 10, 40),
        'mfi_len': trial.suggest_int('mfi_len', 20, 100),
    }


# ─── 9. Commas_Bollinger ────────────────────────────────────────────────────────
# Logic: close crossover/crossunder lower Bollinger Band → long entry
# Uses asymmetric offsets for upper/lower bands

def gen_Commas_Bollinger(df, **params):
    sma_short  = int(params.get('sma_short', 20))
    sma_long   = int(params.get('sma_long', 100))
    ub_offset  = float(params.get('ub_offset', 2.5))
    lb_offset  = float(params.get('lb_offset', 2.5))
    entry_mode = str(params.get('entry_mode', 'over'))  # 'over' or 'under'

    c = df['close']

    short_ma  = _sma(c, sma_short)
    std_dev   = c.rolling(sma_short).std()
    upper_band = short_ma + std_dev * ub_offset
    lower_band = short_ma - std_dev * lb_offset

    if entry_mode == 'over':
        good_buy = _crossover(c, lower_band)
    else:
        good_buy = _crossunder(c, lower_band)

    # Exit: take profit when price reaches take profit level (simplified: upper band)
    good_sell = _crossover(c, upper_band)

    sig = pd.Series(0, index=df.index)
    sig[good_buy]  = 1
    sig[good_sell] = -1
    return sig.fillna(0).astype(int)


def space_Commas_Bollinger(trial):
    return {
        'sma_short': trial.suggest_int('sma_short', 10, 50),
        'sma_long':  trial.suggest_int('sma_long', 50, 200),
        'ub_offset': trial.suggest_float('ub_offset', 1.5, 4.0, step=0.5),
        'lb_offset': trial.suggest_float('lb_offset', 1.5, 4.0, step=0.5),
    }


# ─── 10. EMA_RSI_ADX ────────────────────────────────────────────────────────────
# Logic: EMA fast/slow crossover + RSI threshold + ADX filter

def gen_EMA_RSI_ADX(df, **params):
    fast_len    = int(params.get('fast_len', 9))
    slow_len    = int(params.get('slow_len', 21))
    rsi_len     = int(params.get('rsi_len', 14))
    rsi_long    = float(params.get('rsi_long', 55.0))
    rsi_short   = float(params.get('rsi_short', 40.0))
    adx_len     = int(params.get('adx_len', 14))
    adx_thresh  = float(params.get('adx_thresh', 0.0))
    use_adx     = bool(params.get('use_adx', True))

    c = df['close']
    h = df['high']
    l = df['low']

    ema_fast = _ema(c, fast_len)
    ema_slow = _ema(c, slow_len)
    rsi_val  = _rsi(c, rsi_len)
    adx_val, _, _ = _adx(h, l, c, adx_len)

    adx_ok = (adx_val > adx_thresh) if use_adx else pd.Series(True, index=df.index)

    bull_cross = _crossover(ema_fast, ema_slow)
    bear_cross = _crossunder(ema_fast, ema_slow)

    long_cond  = bull_cross & (rsi_val > rsi_long)  & adx_ok
    short_cond = bear_cross & (rsi_val < rsi_short) & adx_ok

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig.fillna(0).astype(int)


def space_EMA_RSI_ADX(trial):
    return {
        'fast_len':   trial.suggest_int('fast_len', 5, 20),
        'slow_len':   trial.suggest_int('slow_len', 15, 50),
        'rsi_len':    trial.suggest_int('rsi_len', 7, 21),
        'rsi_long':   trial.suggest_float('rsi_long', 45.0, 70.0, step=1.0),
        'rsi_short':  trial.suggest_float('rsi_short', 30.0, 55.0, step=1.0),
        'adx_len':    trial.suggest_int('adx_len', 7, 21),
        'adx_thresh': trial.suggest_float('adx_thresh', 0.0, 40.0, step=1.0),
    }


# ─── STRATEGY EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'DEMA_ATR': {
        'gen':     gen_DEMA_ATR,
        'space':   space_DEMA_ATR,
        'default_params': {'period_dema': 75, 'period_atr': 20, 'factor_atr': 1.20},
        'info': {'source': 'DEMA_ATR_Strategy__PrimeAutomation', 'version': 'v6', 'likes': 1438},
    },
    'Triple_EMA_Turtle': {
        'gen':     gen_Triple_EMA_Turtle,
        'space':   space_Triple_EMA_Turtle,
        'default_params': {'ema_fast': 30, 'ema_mid': 46, 'ema_slow': 80, 'adx_thresh': 43.0, 'adx_len': 14},
        'info': {'source': 'Turtle_Strategy___Triple_EMA_Trend', 'version': 'v6', 'likes': 1398},
    },
    'BB_Enhanced': {
        'gen':     gen_BB_Enhanced,
        'space':   space_BB_Enhanced,
        'default_params': {'bb_period': 20, 'bb_mult': 2.0, 'ema_len': 200},
        'info': {'source': 'Bollinger_Bands_Enhanced_Strategy', 'version': 'v5', 'likes': 1280},
    },
    'Triple_EMA_QQE': {
        'gen':     gen_Triple_EMA_QQE,
        'space':   space_Triple_EMA_QQE,
        'default_params': {'tema1_len': 20, 'tema2_len': 40, 'rsi_period': 14, 'rsi_smooth': 5, 'qqe_factor': 4.238},
        'info': {'source': 'Triple_EMA___QQE_Trend_Following_St', 'version': 'v5', 'likes': 1265},
    },
    'Hull_T3_Swing': {
        'gen':     gen_Hull_T3_Swing,
        'space':   space_Hull_T3_Swing,
        'default_params': {'length': 50, 't3_factor': 0.7},
        'info': {'source': 'Swing_Scalper_HULL___T3_avg_Crypto', 'version': 'v4', 'likes': 1074},
    },
    'ATR_Stop_Multiple': {
        'gen':     gen_ATR_Stop_Multiple,
        'space':   space_ATR_Stop_Multiple,
        'default_params': {'fast_period': 15, 'slow_period': 45},
        'info': {'source': 'BEST_ATR_Stop_Multiple_Strategy', 'version': 'v4', 'likes': 1014},
    },
    'ATR_MA_Trend': {
        'gen':     gen_ATR_MA_Trend,
        'space':   space_ATR_MA_Trend,
        'default_params': {'ma_len': 100, 'atr_len': 7, 'atr_mult': 3.0, 'confirm_bars': 1},
        'info': {'source': 'ATR_Trend_Strategy_with_Moving_Aver', 'version': 'v6', 'likes': 1003},
    },
    'Crypto_Squeeze': {
        'gen':     gen_Crypto_Squeeze,
        'space':   space_Crypto_Squeeze,
        'default_params': {'sq_len': 20, 'mfi_len': 58},
        'info': {'source': 'Crypto_Squeeze_Strategy', 'version': 'v4', 'likes': 974},
    },
    'Commas_Bollinger': {
        'gen':     gen_Commas_Bollinger,
        'space':   space_Commas_Bollinger,
        'default_params': {'sma_short': 20, 'sma_long': 100, 'ub_offset': 2.5, 'lb_offset': 2.5},
        'info': {'source': '3Commas_Bollinger_Strategy', 'version': 'v4', 'likes': 962},
    },
    'EMA_RSI_ADX': {
        'gen':     gen_EMA_RSI_ADX,
        'space':   space_EMA_RSI_ADX,
        'default_params': {'fast_len': 9, 'slow_len': 21, 'rsi_len': 14, 'rsi_long': 55.0, 'rsi_short': 40.0, 'adx_len': 14, 'adx_thresh': 0.0},
        'info': {'source': 'EMA_Cross___RSI___ADX___Autotrade_S', 'version': 'v6', 'likes': 898},
    },
}
