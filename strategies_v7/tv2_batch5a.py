#!/usr/bin/env python3
"""
Estrategias: EMA_Cross_V2, Aroon_Strategy, Ichimoku_V2, Kaufman_AMA, Fractal_Break, Percent_B
Pine version: v4/v5/v6
TradingView WR: varies per strategy
Batch: 5a
"""
import pandas as pd
import numpy as np

# ─── Helper functions ─────────────────────────────────────────────────────────

def _ema(s, p): return s.ewm(span=p, adjust=False).mean()
def _sma(s, p): return s.rolling(p).mean()
def _rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0).ewm(span=p, adjust=False).mean()
    l = (-d.where(d < 0, 0.0)).ewm(span=p, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))
def _atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()
def _wma(s, p):
    w = np.arange(1, p+1, dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
def _hma(s, p):
    return _wma(2*_wma(s,p//2) - _wma(s,p), int(p**0.5))
def _bb(c, p=20, std=2.0):
    m = _sma(c, p); s = c.rolling(p).std()
    return m, m + std*s, m - std*s

# ─── Strategy 1: EMA Cross V2 ────────────────────────────────────────────────

def gen_EMA_Cross_V2(df, fast=9, slow=21, exit_p=5, conf1=1, conf2=4):
    """
    5 EMAs on open. Long: fast crossover slow AND conf1>conf2 AND fast<exit.
    Short: fast crossunder slow AND conf1<conf2 AND fast>exit.
    """
    o = df['open']
    fast_ema  = _ema(o, fast)
    slow_ema  = _ema(o, slow)
    exit_ema  = _ema(o, exit_p)
    conf1_ema = _ema(o, max(conf1, 2))
    conf2_ema = _ema(o, conf2)

    co_long  = (fast_ema.shift(1) < slow_ema.shift(1)) & (fast_ema >= slow_ema)
    co_short = (fast_ema.shift(1) > slow_ema.shift(1)) & (fast_ema <= slow_ema)

    long_cond  = co_long  & (conf1_ema > conf2_ema) & (fast_ema < exit_ema)
    short_cond = co_short & (conf1_ema < conf2_ema) & (fast_ema > exit_ema)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig

def space_EMA_Cross_V2(trial):
    return {
        'fast':   trial.suggest_int('fast', 5, 20),
        'slow':   trial.suggest_int('slow', 15, 50),
        'exit_p': trial.suggest_int('exit_p', 3, 10),
        'conf1':  trial.suggest_int('conf1', 2, 5),
        'conf2':  trial.suggest_int('conf2', 3, 8),
    }

# ─── Strategy 2: Aroon Strategy ──────────────────────────────────────────────

def gen_Aroon_Strategy(df, aroon_len=169, rsi_len=13):
    """
    Aroon oscillator crossover 0. LONG ONLY (original has no short logic).
    Long: aroonOsc crosses above 0.
    """
    high, low = df['high'], df['low']
    n = aroon_len

    # highestbars / lowestbars: distance from the highest/lowest bar in window
    # Pine: aroonUpper = 100 * (highestbars(high, n+1) + n) / n
    # highestbars returns 0 when highest is current bar, negative otherwise
    # Equivalent: position of highest in rolling window from the right
    roll_high = high.rolling(n+1).apply(lambda x: (len(x) - 1 - np.argmax(x)), raw=True)
    roll_low  = low.rolling(n+1).apply(lambda x: (len(x) - 1 - np.argmin(x)), raw=True)

    aroon_upper = 100 * (n - roll_high) / n
    aroon_lower = 100 * (n - roll_low)  / n
    aroon_osc   = aroon_upper - aroon_lower

    co_up = (aroon_osc.shift(1) < 0) & (aroon_osc >= 0)

    sig = pd.Series(0, index=df.index)
    sig[co_up] = 1
    return sig

def space_Aroon_Strategy(trial):
    return {
        'aroon_len': trial.suggest_int('aroon_len', 50, 200),
        'rsi_len':   trial.suggest_int('rsi_len', 7, 21),
    }

# ─── Strategy 3: Ichimoku V2 ─────────────────────────────────────────────────

def gen_Ichimoku_V2(df, conv_p=20, base_p=60, span_b_p=120, displace=30):
    """
    Ichimoku Cloud - LONG ONLY.
    Long: TKcross AND aboveCloud AND greenCloud AND lagLong.
    """
    high, low, close = df['high'], df['low'], df['close']

    def donchian(h, l, p):
        return (h.rolling(p).max() + l.rolling(p).min()) / 2

    conv  = donchian(high, low, conv_p)
    base  = donchian(high, low, base_p)
    lead1 = (conv + base) / 2
    lead2 = donchian(high, low, span_b_p)

    # TK cross (conversion > base)
    tk_cross = conv > base
    # Above cloud (both lead lines)
    above_cloud = (close > lead1) & (close > lead2)
    # Green cloud
    green_cloud = lead1 > lead2
    # Lagging span: close vs lead lines displaced bars ago
    lag_long = (
        (close > lead1.shift(2*displace)) &
        (close > lead2.shift(2*displace)) &
        (close > close.shift(displace))
    )

    all_cond = tk_cross & above_cloud & green_cloud & lag_long

    # Pine strategy.entry fires only on the FIRST bar the condition becomes true
    # (equivalent to crossover from False → True)
    entry = all_cond & (~all_cond.shift(1).astype(bool).fillna(False))

    sig = pd.Series(0, index=df.index)
    sig[entry] = 1
    return sig

def space_Ichimoku_V2(trial):
    return {
        'conv_p':   trial.suggest_int('conv_p', 9, 30),
        'base_p':   trial.suggest_int('base_p', 26, 80),
        'span_b_p': trial.suggest_int('span_b_p', 52, 150),
        'displace': trial.suggest_int('displace', 15, 45),
    }

# ─── Strategy 4: Kaufman AMA ─────────────────────────────────────────────────

def _kama(src, length, fast_len, slow_len):
    """Kaufman Adaptive Moving Average (needs loop for state)."""
    fast_alpha = 2.0 / (fast_len + 1)
    slow_alpha = 2.0 / (slow_len + 1)
    arr = src.values
    n = len(arr)
    out = np.full(n, np.nan)

    # Find first valid index
    first_valid = 0
    for i in range(n):
        if not np.isnan(arr[i]):
            first_valid = i
            break

    start = first_valid + length
    if start >= n:
        return pd.Series(out, index=src.index)

    # Seed the KAMA with the first valid value
    out[first_valid:start] = arr[first_valid]
    out[start-1] = arr[start-1]  # seed start-1 so loop from start works

    for i in range(start, n):
        window = arr[i-length:i+1]
        if np.any(np.isnan(window)):
            out[i] = out[i-1] if not np.isnan(out[i-1]) else arr[i]
            continue
        direction = abs(arr[i] - arr[i-length])
        volatility = np.sum(np.abs(np.diff(window)))
        er = direction / volatility if volatility != 0 else 0
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        prev = out[i-1] if not np.isnan(out[i-1]) else arr[i]
        out[i] = prev + sc * (arr[i] - prev)

    return pd.Series(out, index=src.index)

def gen_Kaufman_AMA(df, l1=14, f1=2, s1=20, l5=18, f5=6, s5=28):
    """
    KAMA Strategy: fast KAMA crosses above/below slow KAMA.
    Long: kama1 crossover kama5. Short: kama1 crossunder kama5.
    """
    close = df['close']
    kama_fast = _kama(close, l1, f1, s1)
    kama_slow = _kama(close, l5, f5, s5)

    co_up   = (kama_fast.shift(1) < kama_slow.shift(1)) & (kama_fast >= kama_slow)
    co_down = (kama_fast.shift(1) > kama_slow.shift(1)) & (kama_fast <= kama_slow)

    sig = pd.Series(0, index=df.index)
    sig[co_up]   = 1
    sig[co_down] = -1
    return sig

def space_Kaufman_AMA(trial):
    return {
        'l1': trial.suggest_int('l1', 5, 25),
        'f1': trial.suggest_int('f1', 2, 5),
        's1': trial.suggest_int('s1', 15, 30),
        'l5': trial.suggest_int('l5', 15, 30),
        'f5': trial.suggest_int('f5', 4, 10),
        's5': trial.suggest_int('s5', 20, 40),
    }

# ─── Strategy 5: Fractal Break ───────────────────────────────────────────────

def gen_Fractal_Break(df, atr_mult=2.0, atr_len=14, rel_vol_len=6):
    """
    Fractal Breakout: 5-bar fractal pattern with volume confirmation.
    Long: price breaks above fractal resistance.
    Short: price breaks below fractal support.
    Pine fractal: high[2]>high[3]>high[4] and high[1]<high[2] and high[0]<high[1] and vol[2]>rel_vol[2]
    (confirmed 2 bars after the peak)
    """
    high, low, close, volume = df['high'], df['low'], df['close'], df['volume']

    rel_vol = volume.rolling(rel_vol_len).mean()

    h = high
    v = volume

    # Fractal confirmed 2 bars after the peak (Williams fractal: peak at bar-2)
    # At current bar i: fractal peak was at i-2
    frac_high_cond = (
        (h.shift(2) > h.shift(3)) &
        (h.shift(3) > h.shift(4)) &
        (h.shift(1) < h.shift(2)) &
        (h       < h.shift(1)) &
        (v.shift(2) > rel_vol.shift(2))
    )
    frac_low_cond = (
        (low.shift(2) < low.shift(3)) &
        (low.shift(3) < low.shift(4)) &
        (low.shift(1) > low.shift(2)) &
        (low         > low.shift(1)) &
        (v.shift(2) > rel_vol.shift(2))
    )

    # Track last fractal resistance and support levels
    arr_h = h.values
    arr_l = low.values
    fhc = frac_high_cond.values
    flc = frac_low_cond.values
    fr = np.full(len(df), np.nan)
    fs = np.full(len(df), np.nan)

    for i in range(len(df)):
        if i > 0:
            fr[i] = fr[i-1]
            fs[i] = fs[i-1]
        if fhc[i]:
            fr[i] = arr_h[i-2] if i >= 2 else arr_h[i]
        if flc[i]:
            fs[i] = arr_l[i-2] if i >= 2 else arr_l[i]

    frac_res = pd.Series(fr, index=df.index)
    frac_sup = pd.Series(fs, index=df.index)

    # Long: close breaks above resistance
    long_cond  = (close.shift(1) < frac_res.shift(1)) & (close >= frac_res) & (~frac_res.isna())
    # Short: close breaks below support
    short_cond = (close.shift(1) > frac_sup.shift(1)) & (close <= frac_sup) & (~frac_sup.isna())

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig

def space_Fractal_Break(trial):
    return {
        'atr_mult':    trial.suggest_float('atr_mult', 1.0, 4.0, step=0.5),
        'atr_len':     trial.suggest_int('atr_len', 7, 21),
        'rel_vol_len': trial.suggest_int('rel_vol_len', 3, 10),
    }

# ─── Strategy 6: Percent B ───────────────────────────────────────────────────

def gen_Percent_B(df, ema_len=100, inner_pct=1.0, outer_pct=2.0, bb_len=20, bb_mult=2.0):
    """
    EMA% Channel + Bollinger Band trending strategy.
    Long: close crossover lower BB AND close < inner_lower AND close > outer_lower.
    Short: close crossunder upper BB AND close > inner_upper AND close < outer_upper.
    """
    close = df['close']

    ema1 = _ema(close, ema_len)
    val_in  = (inner_pct * ema1) / 100
    val_out = (outer_pct * ema1) / 100
    inner_up  = ema1 + val_in
    inner_lo  = ema1 - val_in
    outer_up  = ema1 + val_out
    outer_lo  = ema1 - val_out

    _, bb_up, bb_lo = _bb(close, bb_len, bb_mult)

    # Long: cross above lower BB and close between outer_lo and inner_lo
    co_above_bb_lo = (close.shift(1) < bb_lo) & (close >= bb_lo)
    long_cond = co_above_bb_lo & (close < inner_lo) & (close > outer_lo)

    # Short: cross below upper BB and close between inner_up and outer_up
    co_below_bb_up = (close.shift(1) > bb_up) & (close <= bb_up)
    short_cond = co_below_bb_up & (close > inner_up) & (close < outer_up)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig

def space_Percent_B(trial):
    return {
        'ema_len':   trial.suggest_int('ema_len', 50, 200),
        'inner_pct': trial.suggest_float('inner_pct', 0.5, 2.0, step=0.5),
        'outer_pct': trial.suggest_float('outer_pct', 1.0, 4.0, step=0.5),
        'bb_len':    trial.suggest_int('bb_len', 10, 30),
        'bb_mult':   trial.suggest_float('bb_mult', 1.5, 3.0, step=0.5),
    }

# ─── Export ───────────────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'EMA_Cross_V2':   {'gen': gen_EMA_Cross_V2,   'space': space_EMA_Cross_V2},
    'Aroon_Strategy': {'gen': gen_Aroon_Strategy,  'space': space_Aroon_Strategy},
    'Ichimoku_V2':    {'gen': gen_Ichimoku_V2,     'space': space_Ichimoku_V2},
    'Kaufman_AMA':    {'gen': gen_Kaufman_AMA,     'space': space_Kaufman_AMA},
    'Fractal_Break':  {'gen': gen_Fractal_Break,   'space': space_Fractal_Break},
    'Percent_B':      {'gen': gen_Percent_B,       'space': space_Percent_B},
}
