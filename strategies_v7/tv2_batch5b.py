#!/usr/bin/env python3
"""
Estrategias: Liquidity_Sweep, Fib_Strategy, Vol_Spread, High_WR_Crypto,
             Heikin_Ashi_V2, Supertrend_V2
Pine version: v4/v5/v6
Batch: 5b
"""
import pandas as pd
import numpy as np

# ─── Helper functions ─────────────────────────────────────────────────────────

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
    return _wma(2 * _wma(s, p // 2) - _wma(s, p), int(p ** 0.5))

def _supertrend(h, l, c, atr_period=10, atr_mult=3.0):
    """Standard Supertrend algorithm. Returns pd.Series: 1=bullish, -1=bearish."""
    atr = _atr(h, l, c, atr_period)
    hl2 = (h + l) / 2

    upper_basic = (hl2 + atr_mult * atr).values.copy()
    lower_basic = (hl2 - atr_mult * atr).values.copy()

    c_arr   = c.values
    n       = len(c_arr)
    upper   = upper_basic.copy()
    lower   = lower_basic.copy()
    dir_arr = np.ones(n, dtype=int)

    for i in range(1, n):
        # Lower band (support): only tighten upward, reset if price closed below prev lower
        if lower_basic[i] > lower[i - 1] or c_arr[i - 1] < lower[i - 1]:
            lower[i] = lower_basic[i]
        else:
            lower[i] = lower[i - 1]

        # Upper band (resistance): only tighten downward, reset if price closed above prev upper
        if upper_basic[i] < upper[i - 1] or c_arr[i - 1] > upper[i - 1]:
            upper[i] = upper_basic[i]
        else:
            upper[i] = upper[i - 1]

        # Direction: 1 = bullish (price above lower band), -1 = bearish (price below upper band)
        if dir_arr[i - 1] == 1:
            dir_arr[i] = 1 if c_arr[i] >= lower[i] else -1
        else:
            dir_arr[i] = -1 if c_arr[i] <= upper[i] else 1

    return pd.Series(dir_arr, index=c.index)

# ─── Strategy 1: Liquidity_Sweep ─────────────────────────────────────────────

def gen_Liquidity_Sweep(df, ema_len=200, left=3, right=3, max_age=10,
                        rev_strength=0.60, atr_len=14):
    """
    Pivot-based liquidity sweep with reversal confirmation.
    Long: wick sweeps below pivot low → close reclaims + bull trend + reversal candle.
    Short: wick sweeps above pivot high → close falls below + bear trend + reversal candle.
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']
    open_  = df['open']

    ema_trend = _ema(close, ema_len)
    bull = close > ema_trend
    bear = close < ema_trend

    h_arr = high.values
    l_arr = low.values
    c_arr = close.values
    o_arr = open_.values

    n = len(df)

    # Detect pivot highs / lows (classical: bar i is pivot if it's highest/lowest
    # over [i-left .. i+right], confirmed after right bars)
    pivot_h = np.zeros(n, dtype=bool)
    pivot_l = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        if np.all(h_arr[i] >= h_arr[i - left:i]) and np.all(h_arr[i] >= h_arr[i + 1:i + right + 1]):
            pivot_h[i] = True
        if np.all(l_arr[i] <= l_arr[i - left:i]) and np.all(l_arr[i] <= l_arr[i + 1:i + right + 1]):
            pivot_l[i] = True

    # Propagate most-recent pivot level and age forward
    # Pivot at index pi is confirmed at bar pi+right
    ph_level = np.full(n, np.nan)
    pl_level = np.full(n, np.nan)
    ph_age   = np.full(n, 999, dtype=float)
    pl_age   = np.full(n, 999, dtype=float)

    for i in range(1, n):
        ph_level[i] = ph_level[i - 1]
        pl_level[i] = pl_level[i - 1]
        ph_age[i]   = ph_age[i - 1] + 1
        pl_age[i]   = pl_age[i - 1] + 1

        # Pivot confirmed right bars after it formed
        pi = i - right
        if pi >= 0:
            if pivot_h[pi]:
                ph_level[i] = h_arr[pi]
                ph_age[i]   = right
            if pivot_l[pi]:
                pl_level[i] = l_arr[pi]
                pl_age[i]   = right

    # Reversal candle filter
    body = np.abs(c_arr - o_arr)
    rng  = np.where(h_arr - l_arr > 0, h_arr - l_arr, 1e-10)
    rev_cond = (body / rng) >= rev_strength

    bull_arr = bull.values
    bear_arr = bear.values

    long_arr  = np.zeros(n, dtype=bool)
    short_arr = np.zeros(n, dtype=bool)

    for i in range(1, n):
        # LONG: wick below pivot low, close reclaims above it
        if (not np.isnan(pl_level[i]) and pl_age[i] <= max_age and
                l_arr[i] < pl_level[i] and c_arr[i] > pl_level[i] and
                bull_arr[i] and rev_cond[i]):
            long_arr[i] = True

        # SHORT: wick above pivot high, close falls back below it
        if (not np.isnan(ph_level[i]) and ph_age[i] <= max_age and
                h_arr[i] > ph_level[i] and c_arr[i] < ph_level[i] and
                bear_arr[i] and rev_cond[i]):
            short_arr[i] = True

    sig = pd.Series(0, index=df.index)
    sig[long_arr]  = 1
    sig[short_arr] = -1
    return sig

def space_Liquidity_Sweep(trial):
    return {
        'ema_len':      trial.suggest_int('ema_len', 100, 300),
        'left':         trial.suggest_int('left', 2, 6),
        'right':        trial.suggest_int('right', 2, 6),
        'max_age':      trial.suggest_int('max_age', 5, 20),
        'rev_strength': trial.suggest_float('rev_strength', 0.40, 0.80),
        'atr_len':      trial.suggest_int('atr_len', 7, 21),
    }

# ─── Strategy 2: Fib_Strategy ────────────────────────────────────────────────

def gen_Fib_Strategy(df, lookback=50):
    """
    Fibonacci retracement levels from rolling high/low.
    Long:  close crosses above 61.8% level (retracement from top).
    Short: close crosses below 38.2% level.
    """
    close = df['close']

    highest = close.rolling(lookback).max()
    lowest  = close.rolling(lookback).min()
    diff    = highest - lowest

    level_618 = highest - diff * 0.618   # 61.8% fib retracement from top
    level_382 = highest - diff * 0.382   # 38.2% fib retracement from top

    # Crossover: previous bar below level, current bar above
    prev_close = close.shift(1)

    long_cond  = (prev_close < level_618.shift(1)) & (close >= level_618)
    short_cond = (prev_close > level_382.shift(1)) & (close <= level_382)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig

def space_Fib_Strategy(trial):
    return {
        'lookback': trial.suggest_int('lookback', 20, 100),
    }

# ─── Strategy 3: Vol_Spread ───────────────────────────────────────────────────

def gen_Vol_Spread(df, lenth=10, malenth=20):
    """
    Volume spread analysis (simplified Fourier-free version).
    vd = EMA(close, lenth)*vol_ratio - EMA(open, lenth)*vol_ratio
    vdma = EMA(vd, malenth)
    Long:  vdma crosses above 0 (vdma>0 and vd>0).
    Short: vdma crosses below 0 (vdma<0 and vd<0).
    State-tracking: only signal on crossover/crossunder.
    """
    close  = df['close']
    open_  = df['open']
    volume = df['volume']

    vol_ratio = volume / (volume.shift(1) + 1e-10)

    ema_c = _ema(close, lenth)
    ema_o = _ema(open_, lenth)

    vd   = ema_c * vol_ratio - ema_o * vol_ratio
    vdma = _ema(vd, malenth)

    bull_zone = (vdma > 0) & (vd > 0)
    bear_zone = (vdma < 0) & (vd < 0)

    # Crossover style: condition just turned true (wasn't true previous bar)
    long_cond  = bull_zone & ~bull_zone.shift(1).fillna(False)
    short_cond = bear_zone & ~bear_zone.shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig

def space_Vol_Spread(trial):
    return {
        'lenth':   trial.suggest_int('lenth', 5, 20),
        'malenth': trial.suggest_int('malenth', 10, 40),
    }

# ─── Strategy 4: High_WR_Crypto ───────────────────────────────────────────────

def gen_High_WR_Crypto(df, hma_len=9, ema_short_len=18, ema_long_len=36,
                       rsi_len=14, rsi_long=50, rsi_short=50):
    """
    HMA crossover with EMA filter and RSI.
    Long:  HMA crosses above EMA_short AND low>EMA_long AND rsi>rsi_long.
    Short: HMA crosses below EMA_short AND high<EMA_long AND rsi<rsi_short.
    """
    close = df['close']
    high  = df['high']
    low   = df['low']

    hma_val   = _hma(close, hma_len)
    ema_short = _ema(close, ema_short_len)
    ema_long  = _ema(close, ema_long_len)
    rsi       = _rsi(close, rsi_len)

    # HMA crossover EMA_short
    hma_cross_up   = (hma_val.shift(1) < ema_short.shift(1)) & (hma_val >= ema_short)
    hma_cross_down = (hma_val.shift(1) > ema_short.shift(1)) & (hma_val <= ema_short)

    long_cond  = hma_cross_up   & (low  > ema_long) & (rsi > rsi_long)
    short_cond = hma_cross_down & (high < ema_long) & (rsi < rsi_short)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig

def space_High_WR_Crypto(trial):
    return {
        'hma_len':       trial.suggest_int('hma_len', 5, 20),
        'ema_short_len': trial.suggest_int('ema_short_len', 10, 30),
        'ema_long_len':  trial.suggest_int('ema_long_len', 20, 60),
        'rsi_len':       trial.suggest_int('rsi_len', 7, 21),
        'rsi_long':      trial.suggest_int('rsi_long', 40, 60),
        'rsi_short':     trial.suggest_int('rsi_short', 40, 60),
    }

# ─── Strategy 5: Heikin_Ashi_V2 ──────────────────────────────────────────────

def gen_Heikin_Ashi_V2(df):
    """
    Heikin Ashi calculated from OHLC. LONG ONLY (v4 original).
    Long: ha_close > ha_high[1] AND ha_close > ha_open AND ha_high[1] > ha_high[2]
    """
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)

    ha_open  = np.zeros(n)
    ha_close = np.zeros(n)
    ha_high  = np.zeros(n)
    ha_low   = np.zeros(n)

    # First bar: ha_open = (open+close)/2
    ha_close[0] = (o[0] + h[0] + l[0] + c[0]) / 4
    ha_open[0]  = (o[0] + c[0]) / 2
    ha_high[0]  = max(h[0], ha_open[0], ha_close[0])
    ha_low[0]   = min(l[0], ha_open[0], ha_close[0])

    for i in range(1, n):
        ha_close[i] = (o[i] + h[i] + l[i] + c[i]) / 4
        ha_open[i]  = (ha_open[i - 1] + ha_close[i - 1]) / 2
        ha_high[i]  = max(h[i], ha_open[i], ha_close[i])
        ha_low[i]   = min(l[i], ha_open[i], ha_close[i])

    # Long condition: needs at least 3 bars
    long_arr = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if (ha_close[i] > ha_high[i - 1] and
                ha_close[i] > ha_open[i] and
                ha_high[i - 1] > ha_high[i - 2]):
            long_arr[i] = True

    sig = pd.Series(0, index=df.index)
    sig[long_arr] = 1
    return sig

def space_Heikin_Ashi_V2(trial):
    # No tunable params in original; expose a dummy to satisfy Optuna interface
    return {}

# ─── Strategy 6: Supertrend_V2 ───────────────────────────────────────────────

def gen_Supertrend_V2(df, atr_period=10, atr_multiplier=3.0, ema_length=21,
                      dema_length=200, vol_ema_length=20, rsi_length=14,
                      rsi_threshold=50, allow_short=False):
    """
    Supertrend + EMA + DEMA + Volume + RSI filters. LONG ONLY by default.
    Long:  supertrend flips bullish (+1) AND close>EMA AND close>DEMA
           AND volume>vol_ema AND rsi>rsi_threshold.
    Short: supertrend flips bearish (-1) [only if allow_short=True].
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']

    st_dir = _supertrend(high, low, close, atr_period, atr_multiplier)

    ema_val = _ema(close, ema_length)

    dema_inner = _ema(close, dema_length)
    dema_val   = 2 * dema_inner - _ema(dema_inner, dema_length)

    vol_ema = _ema(volume, vol_ema_length)
    rsi     = _rsi(close, rsi_length)

    # Supertrend flip: direction changes from -1 to 1 (bullish flip)
    st_flip_bull = (st_dir.shift(1) == -1) & (st_dir == 1)
    st_flip_bear = (st_dir.shift(1) == 1)  & (st_dir == -1)

    long_cond = (st_flip_bull &
                 (close > ema_val) &
                 (close > dema_val) &
                 (volume > vol_ema) &
                 (rsi > rsi_threshold))

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1

    if allow_short:
        short_cond = (st_flip_bear &
                      (close < ema_val) &
                      (close < dema_val) &
                      (volume > vol_ema) &
                      (rsi < rsi_threshold))
        sig[short_cond] = -1

    return sig

def space_Supertrend_V2(trial):
    return {
        'atr_period':     trial.suggest_int('atr_period', 7, 20),
        'atr_multiplier': trial.suggest_float('atr_multiplier', 1.5, 5.0),
        'ema_length':     trial.suggest_int('ema_length', 10, 50),
        'dema_length':    trial.suggest_int('dema_length', 100, 300),
        'vol_ema_length': trial.suggest_int('vol_ema_length', 10, 40),
        'rsi_length':     trial.suggest_int('rsi_length', 7, 21),
        'rsi_threshold':  trial.suggest_int('rsi_threshold', 40, 60),
    }

# ─── STRATEGY_EXPORT ─────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Liquidity_Sweep': {
        'gen':   gen_Liquidity_Sweep,
        'space': space_Liquidity_Sweep,
        'params': {
            'ema_len': 200, 'left': 3, 'right': 3, 'max_age': 10,
            'rev_strength': 0.60, 'atr_len': 14,
        },
    },
    'Fib_Strategy': {
        'gen':   gen_Fib_Strategy,
        'space': space_Fib_Strategy,
        'params': {'lookback': 50},
    },
    'Vol_Spread': {
        'gen':   gen_Vol_Spread,
        'space': space_Vol_Spread,
        'params': {'lenth': 10, 'malenth': 20},
    },
    'High_WR_Crypto': {
        'gen':   gen_High_WR_Crypto,
        'space': space_High_WR_Crypto,
        'params': {
            'hma_len': 9, 'ema_short_len': 18, 'ema_long_len': 36,
            'rsi_len': 14, 'rsi_long': 50, 'rsi_short': 50,
        },
    },
    'Heikin_Ashi_V2': {
        'gen':   gen_Heikin_Ashi_V2,
        'space': space_Heikin_Ashi_V2,
        'params': {},
    },
    'Supertrend_V2': {
        'gen':   gen_Supertrend_V2,
        'space': space_Supertrend_V2,
        'params': {
            'atr_period': 10, 'atr_multiplier': 3.0, 'ema_length': 21,
            'dema_length': 200, 'vol_ema_length': 20, 'rsi_length': 14,
            'rsi_threshold': 50, 'allow_short': False,
        },
    },
}
