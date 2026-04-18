"""
tv2_batch6b.py — Batch 6B: 8 Pine Script strategies converted to Python
Strategies: MACD_ATR_Strategy, ATR_Dual_Trail, Swing_Hull_RSI_EMA,
            ORB_Heikin_Ashi, Squeeze_Momentum, Ichimoku_RSI_NoOffset,
            Triple_SuperTrend, ZigZag_RSI
sig=1(LONG), -1(SHORT), 0(none). No look-ahead. Pine defaults preserved.
"""

import numpy as np
import pandas as pd


# ─── HELPERS ────────────────────────────────────────────────────────────────

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

def _supertrend(high, low, close, period=10, factor=3.0):
    atr_v = _atr(high, low, close, period)
    hl2 = (high + low) / 2
    up = (hl2 - factor * atr_v).values
    dn = (hl2 + factor * atr_v).values
    c = close.values
    n = len(c)
    fu = np.copy(up)
    fd = np.copy(dn)
    direction = np.ones(n)
    for i in range(1, n):
        fu[i] = fu[i] if (up[i] > fu[i-1] or c[i-1] < fu[i-1]) else fu[i-1]
        fd[i] = fd[i] if (dn[i] < fd[i-1] or c[i-1] > fd[i-1]) else fd[i-1]
        if direction[i-1] == 1:
            direction[i] = -1 if c[i] < fu[i] else 1
        else:
            direction[i] = 1 if c[i] > fd[i] else -1
    return pd.Series(direction, index=close.index)


# ─── 1. MACD_ATR_Strategy ───────────────────────────────────────────────────

def gen_MACD_ATR_Strategy(df, fast=3, slow=5, signal_p=2, ribbon=34):
    close = df['close']
    fast_ma = _ema(close, fast)
    slow_ma = _ema(close, slow)
    macd = fast_ma - slow_ma
    sig = _ema(macd, signal_p)
    hist = macd - sig
    lead1 = _ema(close, ribbon)
    lead2 = _sma(close, ribbon)
    co_up   = (hist.shift(1) < 0) & (hist >= 0)
    co_down = (hist.shift(1) > 0) & (hist <= 0)
    s = pd.Series(0, index=df.index)
    s[co_up   & (close > lead1)] = 1
    s[co_down & (close < lead1)] = -1
    return s

def space_MACD_ATR_Strategy(trial):
    return {
        'fast':     trial.suggest_int('fast', 2, 8),
        'slow':     trial.suggest_int('slow', 3, 15),
        'signal_p': trial.suggest_int('signal_p', 2, 5),
        'ribbon':   trial.suggest_int('ribbon', 20, 55),
    }


# ─── 2. ATR_Dual_Trail ──────────────────────────────────────────────────────

def gen_ATR_Dual_Trail(df, fast_period=5, fast_mult=0.5, slow_period=10, slow_mult=3.0):
    close = df['close']
    atr_fast = _atr(df['high'], df['low'], close, fast_period)
    atr_slow = _atr(df['high'], df['low'], close, slow_period)
    sl1 = (fast_mult * atr_fast).values
    sl2 = (slow_mult * atr_slow).values
    c = close.values
    n = len(c)
    t1 = np.zeros(n)
    t2 = np.zeros(n)
    for i in range(1, n):
        # Fast trail
        if np.isnan(sl1[i]):
            t1[i] = t1[i-1]
        elif c[i] > t1[i-1] and c[i-1] > t1[i-1]:
            t1[i] = max(t1[i-1], c[i] - sl1[i])
        elif c[i] < t1[i-1] and c[i-1] < t1[i-1]:
            t1[i] = min(t1[i-1], c[i] + sl1[i])
        elif c[i] > t1[i-1]:
            t1[i] = c[i] - sl1[i]
        else:
            t1[i] = c[i] + sl1[i]
        # Slow trail
        if np.isnan(sl2[i]):
            t2[i] = t2[i-1]
        elif c[i] > t2[i-1] and c[i-1] > t2[i-1]:
            t2[i] = max(t2[i-1], c[i] - sl2[i])
        elif c[i] < t2[i-1] and c[i-1] < t2[i-1]:
            t2[i] = min(t2[i-1], c[i] + sl2[i])
        elif c[i] > t2[i-1]:
            t2[i] = c[i] - sl2[i]
        else:
            t2[i] = c[i] + sl2[i]
    t1s = pd.Series(t1, index=df.index)
    t2s = pd.Series(t2, index=df.index)
    co_up   = (t1s.shift(1) < t2s.shift(1)) & (t1s >= t2s)
    co_down = (t1s.shift(1) > t2s.shift(1)) & (t1s <= t2s)
    s = pd.Series(0, index=df.index)
    s[co_up] = 1
    s[co_down] = -1
    return s

def space_ATR_Dual_Trail(trial):
    return {
        'fast_period': trial.suggest_int('fast_period', 3, 10),
        'fast_mult':   trial.suggest_float('fast_mult', 0.3, 1.5),
        'slow_period': trial.suggest_int('slow_period', 8, 20),
        'slow_mult':   trial.suggest_float('slow_mult', 1.5, 5.0),
    }


# ─── 3. Swing_Hull_RSI_EMA ──────────────────────────────────────────────────

def gen_Swing_Hull_RSI_EMA(df, n=100, fast_ema=50, slow_ema=100, rsi_len=14):
    close = df['close']
    hma = _hma(close, n)
    bull = hma > hma.shift(1)
    mafast = _ema(close, fast_ema)
    # Long: HMA bullish + close crosses below fast EMA (pullback entry)
    long_cond  = bull & (close < mafast) & (close.shift(1) >= mafast.shift(1))
    # Short: HMA bearish + close crosses above fast EMA (pullback entry)
    short_cond = (~bull) & (close > mafast) & (close.shift(1) <= mafast.shift(1))
    s = pd.Series(0, index=df.index)
    s[long_cond] = 1
    s[short_cond] = -1
    return s

def space_Swing_Hull_RSI_EMA(trial):
    return {
        'n':        trial.suggest_int('n', 50, 200),
        'fast_ema': trial.suggest_int('fast_ema', 20, 80),
        'slow_ema': trial.suggest_int('slow_ema', 80, 200),
        'rsi_len':  trial.suggest_int('rsi_len', 7, 21),
    }


# ─── 4. ORB_Heikin_Ashi ─────────────────────────────────────────────────────

def gen_ORB_Heikin_Ashi(df, vol_mult=1.5):
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    o = df['open'].values
    n = len(df)
    ha_c = (o + h + l + c) / 4
    ha_o = np.zeros(n)
    ha_o[0] = (o[0] + c[0]) / 2
    for i in range(1, n):
        ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2
    hac = pd.Series(ha_c, index=df.index)
    hao = pd.Series(ha_o, index=df.index)
    vol = df['volume']
    bull_ha = hac > hao
    vol_surge = vol > vol.shift(1) * vol_mult
    long_cond  = bull_ha & (~bull_ha.shift(1).fillna(True))  & vol_surge
    short_cond = (~bull_ha) & bull_ha.shift(1).fillna(False) & vol_surge
    s = pd.Series(0, index=df.index)
    s[long_cond] = 1
    s[short_cond] = -1
    return s

def space_ORB_Heikin_Ashi(trial):
    return {
        'vol_mult': trial.suggest_float('vol_mult', 1.1, 3.0),
    }


# ─── 5. Squeeze_Momentum ────────────────────────────────────────────────────

def gen_Squeeze_Momentum(df, bb_len=20, bb_mult=2.0, kc_len=20, kc_mult=1.5, mom_len=12):
    c = df['close']
    h = df['high']
    l = df['low']
    # Bollinger Bands
    bb_basis = _sma(c, bb_len)
    bb_std   = c.rolling(bb_len).std()
    bb_up    = bb_basis + bb_mult * bb_std
    bb_lo    = bb_basis - bb_mult * bb_std
    # Keltner Channels
    kc_basis = _ema(c, kc_len)
    atr_v    = _atr(h, l, c, kc_len)
    kc_up    = kc_basis + kc_mult * atr_v
    kc_lo    = kc_basis - kc_mult * atr_v
    # Squeeze: BB inside KC
    sqz = (bb_up < kc_up) & (bb_lo > kc_lo)  # noqa: F841 (kept for clarity)
    # Momentum: linear regression of delta
    highest = h.rolling(mom_len).max()
    lowest  = l.rolling(mom_len).min()
    delta   = c - (highest + lowest) / 2 - bb_basis

    def linreg_val(x):
        if np.any(np.isnan(x)):
            return np.nan
        n_pts = len(x)
        xi = np.arange(n_pts)
        return np.polyfit(xi, x, 1)[0]

    mom = delta.rolling(mom_len).apply(linreg_val, raw=True)
    co_up   = (mom.shift(1) <= 0) & (mom > 0)
    co_down = (mom.shift(1) >= 0) & (mom < 0)
    s = pd.Series(0, index=df.index)
    s[co_up] = 1
    s[co_down] = -1
    return s

def space_Squeeze_Momentum(trial):
    return {
        'bb_len':   trial.suggest_int('bb_len', 10, 30),
        'bb_mult':  trial.suggest_float('bb_mult', 1.5, 3.0),
        'kc_len':   trial.suggest_int('kc_len', 10, 30),
        'kc_mult':  trial.suggest_float('kc_mult', 1.0, 2.5),
        'mom_len':  trial.suggest_int('mom_len', 8, 20),
    }


# ─── 6. Ichimoku_RSI_NoOffset ───────────────────────────────────────────────

def gen_Ichimoku_RSI_NoOffset(df, conv=9, base=26, span_b=52,
                               rsi_len=14, rsi_long=55, rsi_short=45):
    h = df['high']
    l = df['low']
    c = df['close']

    def donchian(p):
        return (h.rolling(p).max() + l.rolling(p).min()) / 2

    conv_line = donchian(conv)
    base_line = donchian(base)
    lead_a    = (conv_line + base_line) / 2
    lead_b    = donchian(span_b)
    cloud_top = pd.concat([lead_a, lead_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([lead_a, lead_b], axis=1).min(axis=1)
    rsi = _rsi(c, rsi_len)
    long_cond  = (c > cloud_top) & (conv_line > base_line) & (rsi > rsi_long)
    short_cond = (c < cloud_bot) & (conv_line < base_line) & (rsi < rsi_short)
    # Entry on condition change (edge trigger)
    long_entry  = long_cond  & ~long_cond.shift(1).fillna(False)
    short_entry = short_cond & ~short_cond.shift(1).fillna(False)
    s = pd.Series(0, index=df.index)
    s[long_entry]  = 1
    s[short_entry] = -1
    return s

def space_Ichimoku_RSI_NoOffset(trial):
    return {
        'conv':        trial.suggest_int('conv', 6, 14),
        'base':        trial.suggest_int('base', 18, 36),
        'span_b':      trial.suggest_int('span_b', 40, 70),
        'rsi_len':     trial.suggest_int('rsi_len', 10, 21),
        'rsi_long':    trial.suggest_int('rsi_long', 50, 65),
        'rsi_short':   trial.suggest_int('rsi_short', 35, 50),
    }


# ─── 7. Triple_SuperTrend ───────────────────────────────────────────────────

def _supertrend_v2(high, low, close, period=10, factor=3.0):
    """SuperTrend with proper Pine-style initialization (no zero-bias)."""
    atr_v = _atr(high, low, close, period)
    hl2 = (high + low) / 2
    up_raw = (hl2 - factor * atr_v)
    dn_raw = (hl2 + factor * atr_v)
    c = close.values
    up = up_raw.values
    dn = dn_raw.values
    n = len(c)
    fu = np.full(n, np.nan)
    fd = np.full(n, np.nan)
    direction = np.zeros(n)
    # Find first valid ATR bar
    first = period
    while first < n and np.isnan(up[first]):
        first += 1
    if first >= n:
        return pd.Series(direction, index=close.index)
    fu[first] = up[first]
    fd[first] = dn[first]
    # Pine default: start bullish if close > upper band, else bearish
    direction[first] = 1 if c[first] >= fu[first] else -1
    for i in range(first + 1, n):
        if np.isnan(up[i]) or np.isnan(dn[i]):
            fu[i] = fu[i-1]; fd[i] = fd[i-1]; direction[i] = direction[i-1]
            continue
        fu[i] = up[i] if (up[i] > fu[i-1] or c[i-1] < fu[i-1]) else fu[i-1]
        fd[i] = dn[i] if (dn[i] < fd[i-1] or c[i-1] > fd[i-1]) else fd[i-1]
        if direction[i-1] == 1:
            direction[i] = -1 if c[i] < fu[i] else 1
        else:
            direction[i] = 1 if c[i] > fd[i] else -1
    return pd.Series(direction, index=close.index)


def gen_Triple_SuperTrend(df, p1=10, m1=1.0, p2=11, m2=2.0,
                           p3=12, m3=3.0, ema_len=200):
    h = df['high']
    l = df['low']
    c = df['close']
    st1 = _supertrend_v2(h, l, c, p1, m1)
    st2 = _supertrend_v2(h, l, c, p2, m2)
    st3 = _supertrend_v2(h, l, c, p3, m3)
    ema = _ema(c, ema_len)
    # Individual flips as triggers; majority (>=2 of 3) as direction filter
    score = st1 + st2 + st3  # -3 to +3
    bull_majority = score > 0
    bear_majority = score < 0
    st1_bull_flip = (st1 == 1) & (st1.shift(1) == -1)
    st2_bull_flip = (st2 == 1) & (st2.shift(1) == -1)
    st3_bull_flip = (st3 == 1) & (st3.shift(1) == -1)
    st1_bear_flip = (st1 == -1) & (st1.shift(1) == 1)
    st2_bear_flip = (st2 == -1) & (st2.shift(1) == 1)
    st3_bear_flip = (st3 == -1) & (st3.shift(1) == 1)
    any_bull_flip = st1_bull_flip | st2_bull_flip | st3_bull_flip
    any_bear_flip = st1_bear_flip | st2_bear_flip | st3_bear_flip
    long_entry  = any_bull_flip & bull_majority & (c > ema)
    short_entry = any_bear_flip & bear_majority & (c < ema)
    s = pd.Series(0, index=df.index)
    s[long_entry]  = 1
    s[short_entry] = -1
    return s

def space_Triple_SuperTrend(trial):
    return {
        'p1':      trial.suggest_int('p1', 7, 14),
        'm1':      trial.suggest_float('m1', 0.5, 2.0),
        'p2':      trial.suggest_int('p2', 9, 16),
        'm2':      trial.suggest_float('m2', 1.5, 3.0),
        'p3':      trial.suggest_int('p3', 10, 18),
        'm3':      trial.suggest_float('m3', 2.5, 4.5),
        'ema_len': trial.suggest_int('ema_len', 100, 300),
    }


# ─── 8. ZigZag_RSI ──────────────────────────────────────────────────────────

def gen_ZigZag_RSI(df, rsi_len=5, oversold=25, overbought=75):
    c = df['close']
    rsi = _rsi(c, rsi_len)
    long_cond  = (rsi.shift(1) < oversold)   & (rsi >= oversold)
    short_cond = (rsi.shift(1) > overbought) & (rsi <= overbought)
    s = pd.Series(0, index=df.index)
    s[long_cond]  = 1
    s[short_cond] = -1
    return s

def space_ZigZag_RSI(trial):
    return {
        'rsi_len':    trial.suggest_int('rsi_len', 3, 14),
        'oversold':   trial.suggest_int('oversold', 15, 35),
        'overbought': trial.suggest_int('overbought', 65, 85),
    }


# ─── STRATEGY EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'MACD_ATR_Strategy': {
        'gen':   gen_MACD_ATR_Strategy,
        'space': space_MACD_ATR_Strategy,
        'params': {'fast': 3, 'slow': 5, 'signal_p': 2, 'ribbon': 34},
    },
    'ATR_Dual_Trail': {
        'gen':   gen_ATR_Dual_Trail,
        'space': space_ATR_Dual_Trail,
        'params': {'fast_period': 5, 'fast_mult': 0.5, 'slow_period': 10, 'slow_mult': 3.0},
    },
    'Swing_Hull_RSI_EMA': {
        'gen':   gen_Swing_Hull_RSI_EMA,
        'space': space_Swing_Hull_RSI_EMA,
        'params': {'n': 100, 'fast_ema': 50, 'slow_ema': 100, 'rsi_len': 14},
    },
    'ORB_Heikin_Ashi': {
        'gen':   gen_ORB_Heikin_Ashi,
        'space': space_ORB_Heikin_Ashi,
        'params': {'vol_mult': 1.5},
    },
    'Squeeze_Momentum': {
        'gen':   gen_Squeeze_Momentum,
        'space': space_Squeeze_Momentum,
        'params': {'bb_len': 20, 'bb_mult': 2.0, 'kc_len': 20, 'kc_mult': 1.5, 'mom_len': 12},
    },
    'Ichimoku_RSI_NoOffset': {
        'gen':   gen_Ichimoku_RSI_NoOffset,
        'space': space_Ichimoku_RSI_NoOffset,
        'params': {'conv': 9, 'base': 26, 'span_b': 52, 'rsi_len': 14,
                   'rsi_long': 55, 'rsi_short': 45},
    },
    'Triple_SuperTrend_v2': {
        'gen':   gen_Triple_SuperTrend,
        'space': space_Triple_SuperTrend,
        'params': {'p1': 10, 'm1': 1.0, 'p2': 11, 'm2': 2.0,
                   'p3': 12, 'm3': 3.0, 'ema_len': 200},
    },
    'ZigZag_RSI_v2': {
        'gen':   gen_ZigZag_RSI,
        'space': space_ZigZag_RSI,
        'params': {'rsi_len': 5, 'oversold': 25, 'overbought': 75},
    },
}
