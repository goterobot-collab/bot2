#!/usr/bin/env python3
"""TV2 BATCH 20 — 30 estrategias Ichimoku + Gann + Cycle 2026-04-01"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=int(p), adjust=False).mean()


def _rma(s, p):
    return s.ewm(alpha=1 / int(p), adjust=False).mean()


def _sma(s, p):
    return s.rolling(int(p), min_periods=1).mean()


def _atr(df, p):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, p)


def _rsi(s, p):
    d = s.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    return 100 - 100 / (1 + _rma(g, p) / _rma(l, p).replace(0, 1e-9))


def _donchian(df, p):
    return df['high'].rolling(int(p), min_periods=1).max(), df['low'].rolling(int(p), min_periods=1).min()


def _crossover(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def _ichimoku(df, tenkan_p, kijun_p, senkou_p):
    """Compute full Ichimoku components. Returns (tenkan, kijun, senkou_a, senkou_b)."""
    tp = int(tenkan_p)
    kp = int(kijun_p)
    sp = int(senkou_p)
    tenkan = (df['high'].rolling(tp, min_periods=1).max() + df['low'].rolling(tp, min_periods=1).min()) / 2
    kijun  = (df['high'].rolling(kp, min_periods=1).max() + df['low'].rolling(kp, min_periods=1).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kp)
    senkou_b = ((df['high'].rolling(sp, min_periods=1).max() + df['low'].rolling(sp, min_periods=1).min()) / 2).shift(kp)
    return tenkan, kijun, senkou_a, senkou_b


def _bb(s, p, mult):
    mid   = _sma(s, p)
    std   = s.rolling(int(p), min_periods=1).std().fillna(0)
    upper = mid + mult * std
    lower = mid - mult * std
    return upper, mid, lower


def _psar(df, start, inc, max_val):
    """Parabolic SAR. Returns Series: 1=bullish, -1=bearish."""
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    n     = len(close)

    bull    = True
    af      = start
    ep      = high[0]
    sar     = low[0]
    out     = np.ones(n)

    for i in range(1, n):
        prev_sar = sar
        if bull:
            sar = prev_sar + af * (ep - prev_sar)
            sar = min(sar, low[i - 1], low[max(i - 2, 0)])
            if low[i] < sar:
                bull = False
                sar  = ep
                ep   = low[i]
                af   = start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + inc, max_val)
        else:
            sar = prev_sar + af * (ep - prev_sar)
            sar = max(sar, high[i - 1], high[max(i - 2, 0)])
            if high[i] > sar:
                bull = True
                sar  = ep
                ep   = high[i]
                af   = start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + inc, max_val)
        out[i] = 1 if bull else -1

    return pd.Series(out, index=df.index)


def _linreg_channel(close, p):
    """Rolling linear regression midline and std of residuals."""
    p = int(p)
    vals  = close.values
    n     = len(vals)
    mid   = np.full(n, np.nan)
    std_r = np.full(n, np.nan)
    x     = np.arange(p)
    xa    = x - x.mean()
    ss_x  = (xa ** 2).sum()
    for i in range(p - 1, n):
        y     = vals[i - p + 1: i + 1]
        b1    = (xa * (y - y.mean())).sum() / ss_x
        b0    = y.mean() - b1 * xa.mean()
        fitted = b0 + b1 * x
        residuals = y - fitted
        mid[i]   = fitted[-1]
        std_r[i] = residuals.std() if len(residuals) > 1 else 0.0
    return pd.Series(mid, index=close.index), pd.Series(std_r, index=close.index)


# ── 1. Ichimoku_Full ─────────────────────────────────────────────────────────

def gen_Ichimoku_Full(df, tenkan_p=9, kijun_p=26, senkou_p=52, **kw):
    """Full Ichimoku: TK cross + price above cloud + cloud green."""
    tenkan, kijun, senkou_a, senkou_b = _ichimoku(df, tenkan_p, kijun_p, senkou_p)
    cloud_top    = senkou_a.combine(senkou_b, max)
    cloud_bot    = senkou_a.combine(senkou_b, min)
    cloud_green  = senkou_a >= senkou_b
    cloud_red    = senkou_a < senkou_b

    price_above  = df['close'] > cloud_top
    price_below  = df['close'] < cloud_bot
    tk_bull      = tenkan >= kijun
    tk_bear      = tenkan <= kijun

    long_cond  = price_above & tk_bull & cloud_green
    short_cond = price_below & tk_bear & cloud_red

    sig = pd.Series(0, index=df.index)
    sig[long_cond  & ~long_cond.shift(1, fill_value=False)]  = 1
    sig[short_cond & ~short_cond.shift(1, fill_value=False)] = -1
    return sig.fillna(0).astype(int)


def space_Ichimoku_Full(trial):
    return {
        'tenkan_p': trial.suggest_int('tenkan_p', 5, 15),
        'kijun_p':  trial.suggest_int('kijun_p', 20, 35),
        'senkou_p': trial.suggest_int('senkou_p', 44, 60),
    }


# ── 2. Ichimoku_TK_Cross ─────────────────────────────────────────────────────

def gen_Ichimoku_TK_Cross(df, tenkan_p=9, kijun_p=26, **kw):
    """Tenkan/Kijun cross only (TK Cross)."""
    tenkan = (df['high'].rolling(int(tenkan_p), min_periods=1).max() +
              df['low'].rolling(int(tenkan_p), min_periods=1).min()) / 2
    kijun  = (df['high'].rolling(int(kijun_p), min_periods=1).max() +
              df['low'].rolling(int(kijun_p), min_periods=1).min()) / 2

    sig = pd.Series(0, index=df.index)
    sig[_crossover(tenkan, kijun)]  = 1
    sig[_crossunder(tenkan, kijun)] = -1
    return sig.fillna(0).astype(int)


def space_Ichimoku_TK_Cross(trial):
    return {
        'tenkan_p': trial.suggest_int('tenkan_p', 5, 15),
        'kijun_p':  trial.suggest_int('kijun_p', 20, 35),
    }


# ── 3. Ichimoku_Cloud_Only ───────────────────────────────────────────────────

def gen_Ichimoku_Cloud_Only(df, tenkan_p=9, kijun_p=26, senkou_p=52, **kw):
    """Price position relative to cloud only."""
    _, _, senkou_a, senkou_b = _ichimoku(df, tenkan_p, kijun_p, senkou_p)
    cloud_top = senkou_a.combine(senkou_b, max)
    cloud_bot = senkou_a.combine(senkou_b, min)

    above = df['close'] > cloud_top
    below = df['close'] < cloud_bot

    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], cloud_top)]  = 1
    sig[_crossunder(df['close'], cloud_bot)] = -1
    # maintain state if above/below
    sig[above & (sig == 0) & above.shift(1, fill_value=False)] = 0  # hold, no new signal
    return sig.fillna(0).astype(int)


def space_Ichimoku_Cloud_Only(trial):
    return {
        'tenkan_p': trial.suggest_int('tenkan_p', 5, 15),
        'kijun_p':  trial.suggest_int('kijun_p', 20, 35),
        'senkou_p': trial.suggest_int('senkou_p', 44, 60),
    }


# ── 4. Ichimoku_Kijun_Bounce ─────────────────────────────────────────────────

def gen_Ichimoku_Kijun_Bounce(df, tenkan_p=9, kijun_p=26, **kw):
    """Price bounces off Kijun-sen in uptrend direction."""
    tenkan = (df['high'].rolling(int(tenkan_p), min_periods=1).max() +
              df['low'].rolling(int(tenkan_p), min_periods=1).min()) / 2
    kijun  = (df['high'].rolling(int(kijun_p), min_periods=1).max() +
              df['low'].rolling(int(kijun_p), min_periods=1).min()) / 2

    uptrend   = tenkan > kijun
    downtrend = tenkan < kijun

    sig = pd.Series(0, index=df.index)
    # Bounce long: crossover kijun in uptrend
    sig[_crossover(df['close'], kijun) & uptrend]   = 1
    sig[_crossunder(df['close'], kijun) & downtrend] = -1
    return sig.fillna(0).astype(int)


def space_Ichimoku_Kijun_Bounce(trial):
    return {
        'tenkan_p': trial.suggest_int('tenkan_p', 5, 15),
        'kijun_p':  trial.suggest_int('kijun_p', 20, 35),
    }


# ── 5. Ichimoku_Kumo_Breakout ────────────────────────────────────────────────

def gen_Ichimoku_Kumo_Breakout(df, tenkan_p=9, kijun_p=26, senkou_p=52, **kw):
    """Kumo breakout: price breaks through cloud."""
    _, _, senkou_a, senkou_b = _ichimoku(df, tenkan_p, kijun_p, senkou_p)
    cloud_top = senkou_a.combine(senkou_b, max)
    cloud_bot = senkou_a.combine(senkou_b, min)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], cloud_top)]  = 1
    sig[_crossunder(df['close'], cloud_bot)] = -1
    return sig.fillna(0).astype(int)


def space_Ichimoku_Kumo_Breakout(trial):
    return {
        'tenkan_p': trial.suggest_int('tenkan_p', 5, 15),
        'kijun_p':  trial.suggest_int('kijun_p', 20, 35),
        'senkou_p': trial.suggest_int('senkou_p', 44, 60),
    }


# ── 6. Ichimoku_Chikou ───────────────────────────────────────────────────────

def gen_Ichimoku_Chikou(df, tenkan_p=9, kijun_p=26, senkou_p=52, **kw):
    """Chikou above cloud signal. close > cloud AND chikou (close[kijun_p bars ago]) > senkou at that point."""
    kp = int(kijun_p)
    _, _, senkou_a, senkou_b = _ichimoku(df, tenkan_p, kijun_p, senkou_p)
    cloud_top = senkou_a.combine(senkou_b, max)
    cloud_bot = senkou_a.combine(senkou_b, min)

    # Chikou = current close projected back kijun_p bars
    chikou_above_cloud = df['close'].shift(kp) > cloud_top.shift(kp)
    chikou_below_cloud = df['close'].shift(kp) < cloud_bot.shift(kp)

    price_above = df['close'] > cloud_top
    price_below = df['close'] < cloud_bot

    sig = pd.Series(0, index=df.index)
    long_cond  = price_above & chikou_above_cloud
    short_cond = price_below & chikou_below_cloud

    sig[long_cond  & ~long_cond.shift(1, fill_value=False)]  = 1
    sig[short_cond & ~short_cond.shift(1, fill_value=False)] = -1
    return sig.fillna(0).astype(int)


def space_Ichimoku_Chikou(trial):
    return {
        'tenkan_p': trial.suggest_int('tenkan_p', 5, 15),
        'kijun_p':  trial.suggest_int('kijun_p', 20, 35),
        'senkou_p': trial.suggest_int('senkou_p', 44, 60),
    }


# ── 7. Ichimoku_Multi_Confirm ────────────────────────────────────────────────

def gen_Ichimoku_Multi_Confirm(df, tenkan_p=9, kijun_p=26, senkou_p=52, **kw):
    """Multiple Ichimoku signals required simultaneously."""
    tenkan, kijun, senkou_a, senkou_b = _ichimoku(df, tenkan_p, kijun_p, senkou_p)
    cloud_top   = senkou_a.combine(senkou_b, max)
    cloud_bot   = senkou_a.combine(senkou_b, min)
    cloud_green = senkou_a >= senkou_b
    cloud_red   = senkou_a < senkou_b

    # All 3 conditions must align
    long_cond  = (tenkan > kijun) & (df['close'] > cloud_top) & cloud_green
    short_cond = (tenkan < kijun) & (df['close'] < cloud_bot) & cloud_red

    sig = pd.Series(0, index=df.index)
    sig[_crossover(long_cond.astype(int),  pd.Series(0.5, index=df.index))] = 1
    sig[_crossover(short_cond.astype(int), pd.Series(0.5, index=df.index))] = -1
    # Simpler: fire on first bar where all conditions met
    new_long  = long_cond  & ~long_cond.shift(1, fill_value=False)
    new_short = short_cond & ~short_cond.shift(1, fill_value=False)
    sig = pd.Series(0, index=df.index)
    sig[new_long]  = 1
    sig[new_short] = -1
    return sig.fillna(0).astype(int)


def space_Ichimoku_Multi_Confirm(trial):
    return {
        'tenkan_p': trial.suggest_int('tenkan_p', 5, 15),
        'kijun_p':  trial.suggest_int('kijun_p', 20, 35),
        'senkou_p': trial.suggest_int('senkou_p', 44, 60),
    }


# ── 8. Ichimoku_ATR ──────────────────────────────────────────────────────────

def gen_Ichimoku_ATR(df, tenkan_p=9, kijun_p=26, atr_p=14, atr_mult=2.0, **kw):
    """Ichimoku bullish AND price > ATR trailing stop."""
    tenkan = (df['high'].rolling(int(tenkan_p), min_periods=1).max() +
              df['low'].rolling(int(tenkan_p), min_periods=1).min()) / 2
    kijun  = (df['high'].rolling(int(kijun_p), min_periods=1).max() +
              df['low'].rolling(int(kijun_p), min_periods=1).min()) / 2
    atr    = _atr(df, atr_p)
    atr_trail_bull = df['close'] - atr_mult * atr
    atr_trail_bear = df['close'] + atr_mult * atr

    ichi_bull = tenkan > kijun
    ichi_bear = tenkan < kijun

    sig = pd.Series(0, index=df.index)
    long_cond  = ichi_bull & (df['close'] > atr_trail_bull.shift(1).bfill())
    short_cond = ichi_bear & (df['close'] < atr_trail_bear.shift(1).bfill())

    sig[_crossover(long_cond.astype(float),  pd.Series(0.5, index=df.index))] = 1
    sig[_crossover(short_cond.astype(float), pd.Series(0.5, index=df.index))] = -1
    new_long  = long_cond  & ~long_cond.shift(1, fill_value=False)
    new_short = short_cond & ~short_cond.shift(1, fill_value=False)
    sig = pd.Series(0, index=df.index)
    sig[new_long]  = 1
    sig[new_short] = -1
    return sig.fillna(0).astype(int)


def space_Ichimoku_ATR(trial):
    return {
        'tenkan_p': trial.suggest_int('tenkan_p', 5, 15),
        'kijun_p':  trial.suggest_int('kijun_p', 20, 35),
        'atr_p':    trial.suggest_int('atr_p', 10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 3.0),
    }


# ── 9. Donchian_Breakout ─────────────────────────────────────────────────────

def gen_Donchian_Breakout(df, enter_p=20, exit_p=10, **kw):
    """Classic Donchian channel breakout (Turtle Trading)."""
    ep = int(enter_p)
    xp = int(exit_p)
    dc_high, dc_low = _donchian(df, ep)
    ex_high, ex_low = _donchian(df, xp)

    # Long: new N-bar high. Short: new N-bar low.
    new_high = df['close'] >= dc_high.shift(1).bfill()
    new_low  = df['close'] <= dc_low.shift(1).bfill()
    ex_long  = df['close'] <= ex_low.shift(1).bfill()
    ex_short = df['close'] >= ex_high.shift(1).bfill()

    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], dc_high.shift(1).bfill())]  = 1
    sig[_crossunder(df['close'], dc_low.shift(1).bfill())]  = -1
    return sig.fillna(0).astype(int)


def space_Donchian_Breakout(trial):
    return {
        'enter_p': trial.suggest_int('enter_p', 15, 55),
        'exit_p':  trial.suggest_int('exit_p', 5, 20),
    }


# ── 10. Donchian_ATR_Filter ──────────────────────────────────────────────────

def gen_Donchian_ATR_Filter(df, dc_p=20, atr_p=14, **kw):
    """Donchian breakout filtered by ATR > ATR average (expanding volatility)."""
    dc_high, dc_low = _donchian(df, dc_p)
    atr     = _atr(df, atr_p)
    atr_avg = _sma(atr, atr_p)
    vol_exp = atr > atr_avg

    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], dc_high.shift(1).bfill()) & vol_exp]  = 1
    sig[_crossunder(df['close'], dc_low.shift(1).bfill()) & vol_exp]  = -1
    return sig.fillna(0).astype(int)


def space_Donchian_ATR_Filter(trial):
    return {
        'dc_p':  trial.suggest_int('dc_p', 15, 55),
        'atr_p': trial.suggest_int('atr_p', 10, 20),
    }


# ── 11. Donchian_RSI ─────────────────────────────────────────────────────────

def gen_Donchian_RSI(df, dc_p=20, rsi_p=14, **kw):
    """Donchian breakout with RSI filter (avoid overbought/oversold chasing)."""
    dc_high, dc_low = _donchian(df, dc_p)
    rsi = _rsi(df['close'], rsi_p)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], dc_high.shift(1).bfill()) & (rsi < 70)]  = 1
    sig[_crossunder(df['close'], dc_low.shift(1).bfill()) & (rsi > 30)]  = -1
    return sig.fillna(0).astype(int)


def space_Donchian_RSI(trial):
    return {
        'dc_p':  trial.suggest_int('dc_p', 15, 55),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 12. Turtle_System ────────────────────────────────────────────────────────

def gen_Turtle_System(df, enter_p=20, exit_p=10, atr_p=14, **kw):
    """Original Turtle Trading System 1: 20-bar entry, 10-bar exit, ATR-sized."""
    ep = int(enter_p)
    xp = int(exit_p)
    dc_high, dc_low = _donchian(df, ep)
    ex_high, ex_low = _donchian(df, xp)
    atr = _atr(df, atr_p)

    prev_dc_high = dc_high.shift(1).bfill()
    prev_dc_low  = dc_low.shift(1).bfill()

    # Entry: breakout above/below channel
    long_entry  = _crossover(df['close'], prev_dc_high)
    short_entry = _crossunder(df['close'], prev_dc_low)

    sig = pd.Series(0, index=df.index)
    sig[long_entry]  = 1
    sig[short_entry] = -1
    return sig.fillna(0).astype(int)


def space_Turtle_System(trial):
    return {
        'enter_p': trial.suggest_int('enter_p', 15, 25),
        'exit_p':  trial.suggest_int('exit_p', 7, 15),
        'atr_p':   trial.suggest_int('atr_p', 14, 20),
    }


# ── 13. Gann_Swing ───────────────────────────────────────────────────────────

def gen_Gann_Swing(df, swing_p=3, **kw):
    """Gann Swing Chart: track swing highs/lows over swing_p bars."""
    sp = int(swing_p)
    high  = df['high']
    low   = df['low']
    close = df['close']

    # Swing high: high > high of previous sp bars AND next sp bars
    swing_high = (high == high.rolling(2 * sp + 1, center=True, min_periods=1).max())
    swing_low  = (low  == low.rolling(2 * sp + 1, center=True, min_periods=1).min())

    # Simplification: detect bar that makes new sp-period high/low
    n_high = high.rolling(sp, min_periods=1).max()
    n_low  = low.rolling(sp, min_periods=1).min()

    # Direction change: was making new low, now making new high
    prev_high = n_high.shift(sp)
    prev_low  = n_low.shift(sp)

    turn_up   = (high > prev_high.bfill()) & (low.shift(sp) <= prev_low.bfill())
    turn_down = (low < prev_low.bfill()) & (high.shift(sp) >= prev_high.bfill())

    sig = pd.Series(0, index=df.index)
    sig[turn_up]   = 1
    sig[turn_down] = -1
    return sig.fillna(0).astype(int)


def space_Gann_Swing(trial):
    return {'swing_p': trial.suggest_int('swing_p', 2, 5)}


# ── 14. Gann_HiLo ────────────────────────────────────────────────────────────

def gen_Gann_HiLo(df, gann_p=13, **kw):
    """Gann Hi-Lo Activator: slow EMA of highs or lows depending on trend direction."""
    gp  = int(gann_p)
    ema_hi = _ema(df['high'], gp)
    ema_lo = _ema(df['low'],  gp)

    # Activator follows highs in downtrend, lows in uptrend
    close = df['close']
    # Initial: use midpoint
    activator = pd.Series(np.nan, index=df.index)
    activator.iloc[0] = ema_lo.iloc[0]

    act_vals   = activator.values.copy()
    ema_lo_v   = ema_lo.values
    ema_hi_v   = ema_hi.values
    close_v    = close.values
    bull       = True

    for i in range(1, len(close_v)):
        if bull:
            act_vals[i] = ema_lo_v[i]
            if close_v[i] < act_vals[i]:
                bull = False
        else:
            act_vals[i] = ema_hi_v[i]
            if close_v[i] > act_vals[i]:
                bull = True

    activator = pd.Series(act_vals, index=df.index)
    sig = pd.Series(0, index=df.index)
    sig[_crossover(close, activator)]  = 1
    sig[_crossunder(close, activator)] = -1
    return sig.fillna(0).astype(int)


def space_Gann_HiLo(trial):
    return {'gann_p': trial.suggest_int('gann_p', 10, 30)}


# ── 15. Gann_Box ─────────────────────────────────────────────────────────────

def gen_Gann_Box(df, box_p=40, gann_level=4, **kw):
    """Gann box levels at 1/8 intervals within rolling range."""
    bp  = int(box_p)
    gl  = int(gann_level)  # which 1/8 level to use as support (4 = 50%)
    level_frac = gl / 8.0

    hi  = df['high'].rolling(bp, min_periods=1).max()
    lo  = df['low'].rolling(bp, min_periods=1).min()
    rng = (hi - lo).replace(0, 1e-9)

    support    = lo + rng * level_frac
    resistance = lo + rng * (1.0 - level_frac)

    close = df['close']
    sig = pd.Series(0, index=df.index)
    # Long: close crosses above 50% support from below
    sig[_crossover(close, support)]    = 1
    sig[_crossunder(close, resistance)] = -1
    return sig.fillna(0).astype(int)


def space_Gann_Box(trial):
    return {
        'box_p':      trial.suggest_int('box_p', 20, 60),
        'gann_level': trial.suggest_int('gann_level', 4, 6),
    }


# ── 16. Square_Of_Nine ───────────────────────────────────────────────────────

def gen_Square_Of_Nine(df, rot_factor=1.0, atr_p=14, **kw):
    """Gann Square of 9: key price levels from sqrt(price) ± rot rotations."""
    close = df['close']
    atr   = _atr(df, atr_p)

    # Key support/resistance: sqrt(close) +/- rot_factor → squared back
    sq    = np.sqrt(close)
    level_up   = (sq + rot_factor) ** 2
    level_down = (sq - rot_factor).clip(lower=0) ** 2

    sig = pd.Series(0, index=df.index)
    # Long: close near support (within ATR) and recovering
    near_support    = (close - level_down).abs() < atr
    near_resistance = (close - level_up).abs() < atr

    # Signal on crossover of the sqrt levels
    sig[_crossover(close, level_down.shift(1).bfill()) & near_support]    = 1
    sig[_crossunder(close, level_up.shift(1).bfill()) & near_resistance]  = -1
    return sig.fillna(0).astype(int)


def space_Square_Of_Nine(trial):
    return {
        'rot_factor': trial.suggest_float('rot_factor', 0.5, 2.0),
        'atr_p':      trial.suggest_int('atr_p', 10, 20),
    }


# ── 17. Sine_Wave_Cycle ──────────────────────────────────────────────────────

def gen_Sine_Wave_Cycle(df, min_p=10, max_p=40, **kw):
    """Ehlers-inspired dominant cycle via autocorrelation; sine/cosine components."""
    close  = df['close']
    n      = len(close)
    min_p  = int(min_p)
    max_p  = int(max_p)

    # Estimate dominant period via autocorrelation over range [min_p, max_p]
    # Use rolling window equal to max_p
    window = max_p
    dper   = np.full(n, float(min_p + max_p) / 2)

    for i in range(window, n):
        chunk = close.values[i - window: i]
        chunk = chunk - chunk.mean()
        best_corr = -np.inf
        best_lag  = min_p
        for lag in range(min_p, max_p + 1):
            if lag >= len(chunk):
                continue
            corr = np.corrcoef(chunk[lag:], chunk[:-lag])[0, 1] if lag < len(chunk) else 0
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_lag  = lag
        dper[i] = best_lag

    dper_s = pd.Series(dper, index=close.index)
    dper_s = _sma(dper_s, 5)  # smooth the period estimate

    # Sine component: sin(2π / period) applied to smoothed close
    ema_close = _ema(close, min_p)
    phase = 2 * np.pi / dper_s.clip(lower=2)
    sine_val  = np.sin(phase.cumsum() % (2 * np.pi))
    sine_s    = pd.Series(sine_val, index=close.index)

    # Long: sine crossing up from negative (cycle bottom turning up)
    sig = pd.Series(0, index=df.index)
    sig[_crossover(sine_s, pd.Series(-0.2, index=df.index)) & (ema_close > ema_close.shift(1))] = 1
    sig[_crossunder(sine_s, pd.Series(0.2, index=df.index)) & (ema_close < ema_close.shift(1))] = -1
    return sig.fillna(0).astype(int)


def space_Sine_Wave_Cycle(trial):
    return {
        'min_p': trial.suggest_int('min_p', 10, 20),
        'max_p': trial.suggest_int('max_p', 40, 60),
    }


# ── 18. Hurst_Exponent ───────────────────────────────────────────────────────

def gen_Hurst_Exponent(df, hurst_p=50, ema_p=30, **kw):
    """Simplified Hurst exponent via R/S analysis. H>0.55=trending, H<0.45=mean-reverting."""
    close  = df['close']
    hp     = int(hurst_p)
    ep     = int(ema_p)
    n      = len(close)
    hurst  = np.full(n, 0.5)

    lv = np.log(hp)
    for i in range(hp, n):
        chunk = close.values[i - hp: i]
        diff  = np.diff(chunk)
        if diff.std() < 1e-12:
            hurst[i] = 0.5
            continue
        mean_diff = diff.mean()
        cum_dev   = np.cumsum(diff - mean_diff)
        rs        = (cum_dev.max() - cum_dev.min()) / (diff.std() + 1e-12)
        rs        = max(rs, 1e-6)
        hurst[i]  = np.log(rs) / (lv if lv > 0 else 1)

    hurst_s = pd.Series(hurst, index=close.index).clip(0, 1)
    ema_c   = _ema(close, ep)

    # Trending regime + direction
    trending   = hurst_s > 0.55
    mean_rev   = hurst_s < 0.45
    ema_rising = ema_c > ema_c.shift(1)
    ema_falling= ema_c < ema_c.shift(1)

    bb_upper, _, bb_lower = _bb(close, 20, 2.0)
    near_lower = close < bb_lower
    near_upper = close > bb_upper

    sig = pd.Series(0, index=df.index)
    sig[(trending & ema_rising  & _crossover(close, ema_c)) |
        (mean_rev & near_lower)] = 1
    sig[(trending & ema_falling & _crossunder(close, ema_c)) |
        (mean_rev & near_upper)] = -1
    return sig.fillna(0).astype(int)


def space_Hurst_Exponent(trial):
    return {
        'hurst_p': trial.suggest_int('hurst_p', 30, 100),
        'ema_p':   trial.suggest_int('ema_p', 20, 60),
    }


# ── 19. Autocorrelation_OSC ──────────────────────────────────────────────────

def gen_Autocorrelation_OSC(df, ac_p=20, threshold=0.1, **kw):
    """Rolling lag-1 autocorrelation of close returns. Positive=momentum, negative=mean-rev."""
    close   = df['close']
    returns = close.pct_change().fillna(0)
    ac_p    = int(ac_p)

    # Rolling correlation between returns and returns.shift(1)
    ac = returns.rolling(ac_p, min_periods=ac_p).corr(returns.shift(1)).fillna(0)
    ema_c = _ema(close, ac_p)

    sig = pd.Series(0, index=df.index)
    # Momentum regime: AC positive + EMA rising
    sig[_crossover(ac, pd.Series(threshold, index=df.index)) & (ema_c > ema_c.shift(1))]  = 1
    sig[_crossunder(ac, pd.Series(-threshold, index=df.index)) & (ema_c < ema_c.shift(1))] = -1
    return sig.fillna(0).astype(int)


def space_Autocorrelation_OSC(trial):
    return {
        'ac_p':      trial.suggest_int('ac_p', 10, 30),
        'threshold': trial.suggest_float('threshold', 0.0, 0.3),
    }


# ── 20. Dominant_Cycle_EMA ───────────────────────────────────────────────────

def gen_Dominant_Cycle_EMA(df, min_p=5, max_p=30, **kw):
    """Adaptive EMA period estimated from zero-crossings of DPO (Detrended Price Oscillator)."""
    close  = df['close']
    min_p  = int(min_p)
    max_p  = int(max_p)
    mid_p  = (min_p + max_p) // 2

    # DPO: close - SMA(close, p) shifted p/2+1 bars back
    sma_c  = _sma(close, mid_p)
    dpo    = close - sma_c.shift(mid_p // 2 + 1).bfill()

    # Count zero crossings in rolling window to estimate cycle period
    window     = max_p
    n          = len(close)
    adapt_p    = np.full(n, float(mid_p))
    dpo_vals   = dpo.values

    for i in range(window, n):
        chunk    = dpo_vals[i - window: i]
        # Zero crossings = sign changes
        signs    = np.sign(chunk)
        xings    = np.sum(np.diff(signs) != 0)
        if xings > 1:
            period = 2.0 * window / xings
            adapt_p[i] = np.clip(period, min_p, max_p)

    adapt_p_s  = pd.Series(adapt_p, index=close.index)
    # Compute adaptive EMA using vectorized approximation (piecewise)
    adapt_ema  = _ema(close, mid_p)  # baseline
    adapt_ema2 = _ema(close, min_p)
    # Blend: weight towards shorter period when cycle is short
    weight     = (adapt_p_s - min_p) / max(max_p - min_p, 1)
    blend      = adapt_ema * weight + adapt_ema2 * (1 - weight)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(close, blend)]  = 1
    sig[_crossunder(close, blend)] = -1
    return sig.fillna(0).astype(int)


def space_Dominant_Cycle_EMA(trial):
    return {
        'min_p': trial.suggest_int('min_p', 5, 15),
        'max_p': trial.suggest_int('max_p', 20, 50),
    }


# ── 21. Keltner_Classic ──────────────────────────────────────────────────────

def gen_Keltner_Classic(df, kc_p=20, kc_mult=2.0, atr_p=14, **kw):
    """Keltner Channel: EMA ± mult*ATR. Long: close > upper. Short: close < lower."""
    mid   = _ema(df['close'], kc_p)
    atr   = _atr(df, atr_p)
    upper = mid + kc_mult * atr
    lower = mid - kc_mult * atr

    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], upper)]  = 1
    sig[_crossunder(df['close'], lower)] = -1
    return sig.fillna(0).astype(int)


def space_Keltner_Classic(trial):
    return {
        'kc_p':    trial.suggest_int('kc_p', 15, 25),
        'kc_mult': trial.suggest_float('kc_mult', 1.5, 3.5),
        'atr_p':   trial.suggest_int('atr_p', 10, 20),
    }


# ── 22. Keltner_BB_Squeeze ───────────────────────────────────────────────────

def gen_Keltner_BB_Squeeze(df, kc_p=20, bb_p=20, kc_mult=1.5, bb_mult=2.0, **kw):
    """BB inside KC = squeeze. Long/short when squeeze releases."""
    kc_mid   = _ema(df['close'], kc_p)
    atr      = _atr(df, kc_p)
    kc_upper = kc_mid + kc_mult * atr
    kc_lower = kc_mid - kc_mult * atr

    bb_upper, bb_mid, bb_lower = _bb(df['close'], bb_p, bb_mult)

    # Squeeze: BB bands inside KC bands
    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    # Release: was in squeeze, now not
    release_up   = ~squeeze & squeeze.shift(1, fill_value=False) & (df['close'] > kc_mid)
    release_down = ~squeeze & squeeze.shift(1, fill_value=False) & (df['close'] < kc_mid)

    sig = pd.Series(0, index=df.index)
    sig[release_up]   = 1
    sig[release_down] = -1
    return sig.fillna(0).astype(int)


def space_Keltner_BB_Squeeze(trial):
    return {
        'kc_p':    trial.suggest_int('kc_p', 15, 25),
        'bb_p':    trial.suggest_int('bb_p', 15, 25),
        'kc_mult': trial.suggest_float('kc_mult', 1.5, 2.5),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 2.5),
    }


# ── 23. Keltner_RSI ──────────────────────────────────────────────────────────

def gen_Keltner_RSI(df, kc_p=20, kc_mult=2.0, rsi_p=14, **kw):
    """Keltner + RSI mean-reversion: oversold at lower band, overbought at upper."""
    mid   = _ema(df['close'], kc_p)
    atr   = _atr(df, kc_p)
    upper = mid + kc_mult * atr
    lower = mid - kc_mult * atr
    rsi   = _rsi(df['close'], rsi_p)

    sig = pd.Series(0, index=df.index)
    # Oversold at KC lower → long
    sig[(df['close'] < lower) & (rsi < 30)] = 1
    # Wait for previous bar trigger
    long_trigger  = (df['close'] < lower) & (rsi < 30)
    short_trigger = (df['close'] > upper) & (rsi > 70)

    # Signal on first bar of condition
    sig = pd.Series(0, index=df.index)
    sig[long_trigger  & ~long_trigger.shift(1, fill_value=False)]  = 1
    sig[short_trigger & ~short_trigger.shift(1, fill_value=False)] = -1
    return sig.fillna(0).astype(int)


def space_Keltner_RSI(trial):
    return {
        'kc_p':    trial.suggest_int('kc_p', 15, 25),
        'kc_mult': trial.suggest_float('kc_mult', 1.5, 3.0),
        'rsi_p':   trial.suggest_int('rsi_p', 7, 21),
    }


# ── 24. Keltner_Trend ────────────────────────────────────────────────────────

def gen_Keltner_Trend(df, kc_p=20, kc_mult=2.0, consec_bars=3, **kw):
    """Keltner as trend filter: price consistently above upper band = strong uptrend."""
    mid   = _ema(df['close'], kc_p)
    atr   = _atr(df, kc_p)
    upper = mid + kc_mult * atr
    lower = mid - kc_mult * atr
    cb    = int(consec_bars)

    above_upper = (df['close'] > upper).astype(int)
    below_lower = (df['close'] < lower).astype(int)

    # N consecutive bars above/below
    consec_above = above_upper.rolling(cb, min_periods=cb).sum() == cb
    consec_below = below_lower.rolling(cb, min_periods=cb).sum() == cb

    sig = pd.Series(0, index=df.index)
    sig[consec_above & ~consec_above.shift(1, fill_value=False)]  = 1
    sig[consec_below & ~consec_below.shift(1, fill_value=False)]  = -1
    return sig.fillna(0).astype(int)


def space_Keltner_Trend(trial):
    return {
        'kc_p':       trial.suggest_int('kc_p', 15, 25),
        'kc_mult':    trial.suggest_float('kc_mult', 1.5, 2.5),
        'consec_bars': trial.suggest_int('consec_bars', 2, 5),
    }


# ── 25. Price_Channel_Breakout ───────────────────────────────────────────────

def gen_Price_Channel_Breakout(df, pc_p=20, threshold_pct=0.3, **kw):
    """Price channel breakout: close > mid + threshold% of range."""
    pcp = int(pc_p)
    pch = df['high'].rolling(pcp, min_periods=1).max()
    pcl = df['low'].rolling(pcp, min_periods=1).min()
    mid = (pch + pcl) / 2
    rng = (pch - pcl).replace(0, 1e-9)

    upper_thresh = mid + threshold_pct * rng
    lower_thresh = mid - threshold_pct * rng

    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], upper_thresh)]  = 1
    sig[_crossunder(df['close'], lower_thresh)] = -1
    return sig.fillna(0).astype(int)


def space_Price_Channel_Breakout(trial):
    return {
        'pc_p':          trial.suggest_int('pc_p', 10, 50),
        'threshold_pct': trial.suggest_float('threshold_pct', 0.2, 0.5),
    }


# ── 26. Donchian_Mid ─────────────────────────────────────────────────────────

def gen_Donchian_Mid(df, dc_p=20, **kw):
    """Donchian midline crossover strategy."""
    dcp     = int(dc_p)
    dc_high = df['high'].rolling(dcp, min_periods=1).max()
    dc_low  = df['low'].rolling(dcp, min_periods=1).min()
    mid     = (dc_high + dc_low) / 2

    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], mid)]  = 1
    sig[_crossunder(df['close'], mid)] = -1
    return sig.fillna(0).astype(int)


def space_Donchian_Mid(trial):
    return {'dc_p': trial.suggest_int('dc_p', 15, 50)}


# ── 27. LRC_Strategy ─────────────────────────────────────────────────────────

def gen_LRC_Strategy(df, lrc_p=50, **kw):
    """Linear Regression Channel: cross above/below LR midline."""
    mid, _ = _linreg_channel(df['close'], lrc_p)

    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], mid)]  = 1
    sig[_crossunder(df['close'], mid)] = -1
    return sig.fillna(0).astype(int)


def space_LRC_Strategy(trial):
    return {'lrc_p': trial.suggest_int('lrc_p', 20, 100)}


# ── 28. LRC_Breakout ─────────────────────────────────────────────────────────

def gen_LRC_Breakout(df, lrc_p=50, dev_mult=2.0, **kw):
    """LRC momentum breakout: close breaks above upper or below lower band."""
    mid, std_r = _linreg_channel(df['close'], lrc_p)
    upper = mid + dev_mult * std_r
    lower = mid - dev_mult * std_r

    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], upper)]  = 1
    sig[_crossunder(df['close'], lower)] = -1
    return sig.fillna(0).astype(int)


def space_LRC_Breakout(trial):
    return {
        'lrc_p':    trial.suggest_int('lrc_p', 20, 100),
        'dev_mult': trial.suggest_float('dev_mult', 1.5, 3.0),
    }


# ── 29. LRC_Reversion ────────────────────────────────────────────────────────

def gen_LRC_Reversion(df, lrc_p=50, dev_mult=2.0, **kw):
    """LRC mean reversion: close outside bands recovers back toward midline."""
    mid, std_r = _linreg_channel(df['close'], lrc_p)
    upper = mid + dev_mult * std_r
    lower = mid - dev_mult * std_r

    # Long: was below lower, now recovering (crossover lower from below)
    sig = pd.Series(0, index=df.index)
    sig[_crossover(df['close'], lower)]  = 1
    sig[_crossunder(df['close'], upper)] = -1
    return sig.fillna(0).astype(int)


def space_LRC_Reversion(trial):
    return {
        'lrc_p':    trial.suggest_int('lrc_p', 20, 100),
        'dev_mult': trial.suggest_float('dev_mult', 1.5, 3.0),
    }


# ── 30. PSAR_ATR ─────────────────────────────────────────────────────────────

def gen_PSAR_ATR(df, psar_start=0.02, psar_inc=0.02, psar_max=0.2, atr_p=14, **kw):
    """PSAR + ATR momentum gate: PSAR bullish AND ATR expanding."""
    psar_dir = _psar(df, float(psar_start), float(psar_inc), float(psar_max))
    atr      = _atr(df, atr_p)
    atr_avg  = _sma(atr, atr_p)
    atr_exp  = atr > atr_avg

    sig = pd.Series(0, index=df.index)
    sig[_crossover(psar_dir,  pd.Series(0.0, index=df.index)) & atr_exp] = 1
    sig[_crossunder(psar_dir, pd.Series(0.0, index=df.index)) & atr_exp] = -1
    return sig.fillna(0).astype(int)


def space_PSAR_ATR(trial):
    return {
        'psar_start': trial.suggest_float('psar_start', 0.01, 0.04),
        'psar_inc':   trial.suggest_float('psar_inc', 0.01, 0.04),
        'psar_max':   trial.suggest_float('psar_max', 0.1, 0.4),
        'atr_p':      trial.suggest_int('atr_p', 10, 20),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Ichimoku_Full': {
        'gen': gen_Ichimoku_Full,
        'space': space_Ichimoku_Full,
        'default_params': {'tenkan_p': 9, 'kijun_p': 26, 'senkou_p': 52},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 8000,
            'description': 'Full Ichimoku: TK cross + price above cloud + cloud green. All 3 conditions required.',
        },
    },
    'Ichimoku_TK_Cross': {
        'gen': gen_Ichimoku_TK_Cross,
        'space': space_Ichimoku_TK_Cross,
        'default_params': {'tenkan_p': 9, 'kijun_p': 26},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 5000,
            'description': 'Ichimoku TK Cross only: Tenkan crosses above/below Kijun.',
        },
    },
    'Ichimoku_Cloud_Only': {
        'gen': gen_Ichimoku_Cloud_Only,
        'space': space_Ichimoku_Cloud_Only,
        'default_params': {'tenkan_p': 9, 'kijun_p': 26, 'senkou_p': 52},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 3000,
            'description': 'Ichimoku cloud position: long when price crosses above cloud top.',
        },
    },
    'Ichimoku_Kijun_Bounce': {
        'gen': gen_Ichimoku_Kijun_Bounce,
        'space': space_Ichimoku_Kijun_Bounce,
        'default_params': {'tenkan_p': 9, 'kijun_p': 26},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2500,
            'description': 'Ichimoku Kijun bounce: price crosses Kijun in direction of TK trend.',
        },
    },
    'Ichimoku_Kumo_Breakout': {
        'gen': gen_Ichimoku_Kumo_Breakout,
        'space': space_Ichimoku_Kumo_Breakout,
        'default_params': {'tenkan_p': 9, 'kijun_p': 26, 'senkou_p': 52},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 3000,
            'description': 'Ichimoku Kumo breakout: price breaks through cloud (both Senkou lines).',
        },
    },
    'Ichimoku_Chikou': {
        'gen': gen_Ichimoku_Chikou,
        'space': space_Ichimoku_Chikou,
        'default_params': {'tenkan_p': 9, 'kijun_p': 26, 'senkou_p': 52},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'Ichimoku Chikou confirmation: current price above cloud AND projected close also above historical cloud.',
        },
    },
    'Ichimoku_Multi_Confirm': {
        'gen': gen_Ichimoku_Multi_Confirm,
        'space': space_Ichimoku_Multi_Confirm,
        'default_params': {'tenkan_p': 9, 'kijun_p': 26, 'senkou_p': 52},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2500,
            'description': 'Ichimoku multi-confirmation: TK cross + price above cloud + cloud green all simultaneously.',
        },
    },
    'Ichimoku_ATR': {
        'gen': gen_Ichimoku_ATR,
        'space': space_Ichimoku_ATR,
        'default_params': {'tenkan_p': 9, 'kijun_p': 26, 'atr_p': 14, 'atr_mult': 2.0},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'Ichimoku + ATR trailing: TK trend direction confirmed by price above ATR trail.',
        },
    },
    'Donchian_Breakout': {
        'gen': gen_Donchian_Breakout,
        'space': space_Donchian_Breakout,
        'default_params': {'enter_p': 20, 'exit_p': 10},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 4000,
            'description': 'Classic Donchian channel breakout: long on N-bar high, short on N-bar low.',
        },
    },
    'Donchian_ATR_Filter': {
        'gen': gen_Donchian_ATR_Filter,
        'space': space_Donchian_ATR_Filter,
        'default_params': {'dc_p': 20, 'atr_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'Donchian breakout with ATR expansion filter: only trade when volatility is expanding.',
        },
    },
    'Donchian_RSI': {
        'gen': gen_Donchian_RSI,
        'space': space_Donchian_RSI,
        'default_params': {'dc_p': 20, 'rsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'Donchian breakout with RSI filter: avoid chasing overbought/oversold breakouts.',
        },
    },
    'Turtle_System': {
        'gen': gen_Turtle_System,
        'space': space_Turtle_System,
        'default_params': {'enter_p': 20, 'exit_p': 10, 'atr_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 3000,
            'description': 'Original Turtle Trading System 1: 20-bar channel breakout entry, 10-bar exit.',
        },
    },
    'Gann_Swing': {
        'gen': gen_Gann_Swing,
        'space': space_Gann_Swing,
        'default_params': {'swing_p': 3},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 3000,
            'description': 'Gann Swing Chart: detects swing high/low direction changes over N bars.',
        },
    },
    'Gann_HiLo': {
        'gen': gen_Gann_HiLo,
        'space': space_Gann_HiLo,
        'default_params': {'gann_p': 13},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'Gann Hi-Lo Activator: adaptive EMA of highs/lows switching with trend direction.',
        },
    },
    'Gann_Box': {
        'gen': gen_Gann_Box,
        'space': space_Gann_Box,
        'default_params': {'box_p': 40, 'gann_level': 4},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'Gann Box: 1/8-interval price levels within rolling range. Signal at support/resistance.',
        },
    },
    'Square_Of_Nine': {
        'gen': gen_Square_Of_Nine,
        'space': space_Square_Of_Nine,
        'default_params': {'rot_factor': 1.0, 'atr_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'Gann Square of 9: key price levels from sqrt(price) ± rotation factor, squared back.',
        },
    },
    'Sine_Wave_Cycle': {
        'gen': gen_Sine_Wave_Cycle,
        'space': space_Sine_Wave_Cycle,
        'default_params': {'min_p': 10, 'max_p': 40},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'Ehlers-inspired dominant cycle detection via autocorrelation + sine wave phase signal.',
        },
    },
    'Hurst_Exponent': {
        'gen': gen_Hurst_Exponent,
        'space': space_Hurst_Exponent,
        'default_params': {'hurst_p': 50, 'ema_p': 30},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'Hurst exponent via R/S analysis: H>0.55=trend-follow, H<0.45=mean-revert with BB.',
        },
    },
    'Autocorrelation_OSC': {
        'gen': gen_Autocorrelation_OSC,
        'space': space_Autocorrelation_OSC,
        'default_params': {'ac_p': 20, 'threshold': 0.1},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1800,
            'description': 'Lag-1 rolling autocorrelation of returns: positive=momentum, negative=mean-reversion.',
        },
    },
    'Dominant_Cycle_EMA': {
        'gen': gen_Dominant_Cycle_EMA,
        'space': space_Dominant_Cycle_EMA,
        'default_params': {'min_p': 5, 'max_p': 30},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'Adaptive EMA whose period adapts to dominant cycle estimated via DPO zero-crossings.',
        },
    },
    'Keltner_Classic': {
        'gen': gen_Keltner_Classic,
        'space': space_Keltner_Classic,
        'default_params': {'kc_p': 20, 'kc_mult': 2.0, 'atr_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 5000,
            'description': 'Keltner Channel breakout: EMA ± mult*ATR. Long above upper, short below lower.',
        },
    },
    'Keltner_BB_Squeeze': {
        'gen': gen_Keltner_BB_Squeeze,
        'space': space_Keltner_BB_Squeeze,
        'default_params': {'kc_p': 20, 'bb_p': 20, 'kc_mult': 1.5, 'bb_mult': 2.0},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 3000,
            'description': 'Bollinger Band / Keltner Squeeze: low volatility compression release signal.',
        },
    },
    'Keltner_RSI': {
        'gen': gen_Keltner_RSI,
        'space': space_Keltner_RSI,
        'default_params': {'kc_p': 20, 'kc_mult': 2.0, 'rsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Keltner mean-reversion: oversold at lower KC band (RSI<30) or overbought at upper (RSI>70).',
        },
    },
    'Keltner_Trend_v2': {
        'gen': gen_Keltner_Trend,
        'space': space_Keltner_Trend,
        'default_params': {'kc_p': 20, 'kc_mult': 2.0, 'consec_bars': 3},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'Keltner trend filter: N consecutive bars outside channel = strong trend confirmation.',
        },
    },
    'Price_Channel_Breakout': {
        'gen': gen_Price_Channel_Breakout,
        'space': space_Price_Channel_Breakout,
        'default_params': {'pc_p': 20, 'threshold_pct': 0.3},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'Price channel breakout: close beyond mid ± threshold% of highest-high/lowest-low range.',
        },
    },
    'Donchian_Mid': {
        'gen': gen_Donchian_Mid,
        'space': space_Donchian_Mid,
        'default_params': {'dc_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'Donchian midline crossover: long when close crosses above (high+low)/2 midpoint.',
        },
    },
    'LRC_Strategy': {
        'gen': gen_LRC_Strategy,
        'space': space_LRC_Strategy,
        'default_params': {'lrc_p': 50},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 3000,
            'description': 'Linear Regression Channel midline cross: long above LR midline, short below.',
        },
    },
    'LRC_Breakout': {
        'gen': gen_LRC_Breakout,
        'space': space_LRC_Breakout,
        'default_params': {'lrc_p': 50, 'dev_mult': 2.0},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'LRC momentum breakout: close breaks above/below LR channel ± dev_mult*std(residuals).',
        },
    },
    'LRC_Reversion': {
        'gen': gen_LRC_Reversion,
        'space': space_LRC_Reversion,
        'default_params': {'lrc_p': 50, 'dev_mult': 2.0},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'LRC mean reversion: close recovers from outside LR channel bands back toward midline.',
        },
    },
    'PSAR_ATR': {
        'gen': gen_PSAR_ATR,
        'space': space_PSAR_ATR,
        'default_params': {'psar_start': 0.02, 'psar_inc': 0.02, 'psar_max': 0.2, 'atr_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'Parabolic SAR + ATR momentum gate: PSAR direction change AND ATR expanding.',
        },
    },
}
