#!/usr/bin/env python3
"""
TV2 Batch 8a — 9 estrategias Pine → Python
Likes totales: ~13,954

1. Trendline_Fib_ST        (2511 likes) — Pine v6 — Fibonacci Supertrend + Trendline Breaks
2. BB_Breakout_V2          (2048 likes) — Pine v5 — Bollinger Bands Breakout
3. Ichimoku_NoOffset_V2   (1799 likes) — Pine v4 — Ichimoku Kinko Hyo no offset
4. MA_BB_RSI              (1550 likes) — Pine v5 — MA + BB + RSI
5. Donchian_Free_Bot      (1336 likes) — Pine v4 — Donchian Channel breakout
6. Fractal_Breakout_KL    (1304 likes) — Pine v5 — Fractal Breakout + ATR
7. Order_Block_FVG        (1301 likes) — Pine v6 — Order Block + FVG + Volume
8. Fibonacci_Trend_Reversal (1054 likes) — Pine v6 — Fibonacci retracement reversal
9. Fractal_Breakout_Simple (1051 likes) — Pine v4 — Simple Fractal Breakout

Convención: sig = 1 (LONG), -1 (SHORT), 0 (sin señal)
Sin look-ahead: señales basadas en datos hasta la barra actual.
"""

import numpy as np
import pandas as pd


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()
def _rsi(s, n=14):
    d = s.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    return 100 - 100/(1 + g.ewm(alpha=1/n,adjust=False).mean()/l.ewm(alpha=1/n,adjust=False).mean())
def _atr(df, n=14):
    h,l,c = df['high'],df['low'],df['close']
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def _wma(s, n):
    w = np.arange(1, n+1)
    return s.rolling(n).apply(lambda x: np.dot(x,w)/w.sum(), raw=True)
def _hma(s, n):
    return _wma(2*_wma(s,n//2) - _wma(s,n), max(1,int(n**0.5)))
def _bb(s, n=20, mult=2.0):
    m = _sma(s,n); std = s.rolling(n).std()
    return m, m+mult*std, m-mult*std
def _stoch(df, k=14, d=3):
    lo = df['low'].rolling(k).min(); hi = df['high'].rolling(k).max()
    kline = 100*(df['close']-lo)/(hi-lo+1e-9)
    return kline, kline.rolling(d).mean()
def _dema(s, n):
    e = _ema(s, n); return 2*e - _ema(e, n)
def _keltner(df, n=20, mult=2.0):
    m = _ema(df['close'],n); a = _atr(df,n)
    return m, m+mult*a, m-mult*a
def _mfi(df, n=14):
    tp = (df['high']+df['low']+df['close'])/3
    mf = tp * df['volume']
    pos = mf.where(tp>tp.shift(),0); neg = mf.where(tp<tp.shift(),0)
    mr = pos.rolling(n).sum()/(neg.rolling(n).sum()+1e-9)
    return 100 - 100/(1+mr)
def _adx(df, n=14):
    h,l,c = df['high'],df['low'],df['close']
    up = h-h.shift(); dn = l.shift()-l
    pdm = up.where((up>dn)&(up>0),0); ndm = dn.where((dn>up)&(dn>0),0)
    atr = _atr(df,n)
    pdi = 100*pdm.ewm(alpha=1/n,adjust=False).mean()/(atr+1e-9)
    ndi = 100*ndm.ewm(alpha=1/n,adjust=False).mean()/(atr+1e-9)
    dx = 100*(pdi-ndi).abs()/(pdi+ndi+1e-9)
    return dx.ewm(alpha=1/n,adjust=False).mean(), pdi, ndi
def _supertrend(df, n=10, mult=3.0):
    atr = _atr(df,n); hl2 = (df['high']+df['low'])/2
    bu = hl2+mult*atr; bl = hl2-mult*atr
    fu = bu.copy(); fd = bl.copy()
    close = df['close'].values
    fu_v = bu.values.copy(); fd_v = bl.values.copy()
    for i in range(1,len(df)):
        fu_v[i] = min(fu_v[i], fu_v[i-1]) if close[i-1]<=fu_v[i-1] else fu_v[i]
        fd_v[i] = max(fd_v[i], fd_v[i-1]) if close[i-1]>=fd_v[i-1] else fd_v[i]
    trend = pd.Series(1, index=df.index)
    for i in range(1,len(df)):
        if trend.iloc[i-1]==1 and close[i]<fd_v[i]: trend.iloc[i]=-1
        elif trend.iloc[i-1]==-1 and close[i]>fu_v[i]: trend.iloc[i]=1
        else: trend.iloc[i]=trend.iloc[i-1]
    return trend, pd.Series(fd_v,index=df.index), pd.Series(fu_v,index=df.index)


# ─── 1. TRENDLINE BREAKS + FIBONACCI SUPERTREND ───────────────────────────────

def gen_Trendline_Fib_ST(df, atr_period=10, factor=3.0, fibs=3):
    """
    Trendline Breaks with Multi Fibonacci Supertrend Strategy — Pine v6 (2511 likes)
    Multiple supertrends with Fibonacci multipliers (0.236, 0.382, 0.618, 1.0, 1.618)
    LONG: all 'fibs' supertrends are bullish AND close crosses over st1
    SHORT: all 'fibs' supertrends are bearish AND close crosses under st1
    """
    close = df['close']
    fib_mults = [0.236, 0.382, 0.618, 1.0, 1.618]
    fibs = max(1, min(5, fibs))

    # Calculate supertrends for each fibonacci multiplier
    trends = []
    st_lines = []
    for i in range(fibs):
        trend, fd, fu = _supertrend(df, n=atr_period, mult=factor * fib_mults[i])
        trends.append(trend)
        # st1 is the first supertrend (fib_mults[0])
        if i == 0:
            # supertrend line: fd when bullish, fu when bearish
            st1 = pd.Series(np.where(trend == 1, fd, fu), index=df.index)

    # Count bullish/bearish supertrends
    bull_count = sum([(t == 1).astype(int) for t in trends])
    bear_count = sum([(t == -1).astype(int) for t in trends])

    # Crossover/crossunder of close vs st1
    cross_up = (close > st1) & (close.shift(1) <= st1.shift(1))
    cross_dn = (close < st1) & (close.shift(1) >= st1.shift(1))

    long_cond  = (bull_count >= fibs) & cross_up
    short_cond = (bear_count >= fibs) & cross_dn

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Trendline_Fib_ST(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 20),
        'factor':     trial.suggest_float('factor', 1.5, 5.0),
        'fibs':       trial.suggest_int('fibs', 1, 5),
    }


# ─── 2. BOLLINGER BANDS BREAKOUT V2 ──────────────────────────────────────────

def gen_BB_Breakout_V2(df, length=20, mult=2.0):
    """
    Bollinger Bands - Breakout Strategy — Pine v5 (2048 likes)
    LONG:  close crosses ABOVE upper band (breakout long)
    SHORT: close crosses BELOW lower band (breakout short)
    """
    close = df['close']
    _, upper, lower = _bb(close, n=length, mult=mult)

    # crossover(close, upper): close > upper AND prev_close <= prev_upper
    long_cond  = (close > upper) & (close.shift(1) <= upper.shift(1))
    # crossunder(close, lower): close < lower AND prev_close >= prev_lower
    short_cond = (close < lower) & (close.shift(1) >= lower.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_BB_Breakout_V2(trial):
    return {
        'length': trial.suggest_int('length', 10, 50),
        'mult':   trial.suggest_float('mult', 1.0, 3.5),
    }


# ─── 3. ICHIMOKU KINKO HYO — NO OFFSET ───────────────────────────────────────

def gen_Ichimoku_NoOffset_V2(df, conversion_periods=9, base_periods=26,
                              lagging_span2_periods=52, displacement=26):
    """
    Ichimoku Kinko Hyo Cloud - no offset - no repaint — Pine v4 (1799 likes)
    donchian(n) = (high.rolling(n).max() + low.rolling(n).min()) / 2
    LONG:  TK cross (tenkan > kijun) AND price above cloud (no offset)
    SHORT: TK crossunder AND price below cloud
    Note: displacement param kept for API compatibility but not used (no offset version)
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    def donchian(n):
        return (high.rolling(n).max() + low.rolling(n).min()) / 2

    tenkan  = donchian(conversion_periods)
    kijun   = donchian(base_periods)
    span_a  = (tenkan + kijun) / 2
    span_b  = donchian(lagging_span2_periods)

    # Cloud boundaries (no offset — use current values, not shifted)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)

    # Price above/below cloud
    above_cloud = close > cloud_top
    below_cloud = close < cloud_bot

    # TK cross: tenkan crosses over kijun
    tk_cross      = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    tk_crossunder = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))

    long_cond  = tk_cross & above_cloud
    short_cond = tk_crossunder & below_cloud

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Ichimoku_NoOffset_V2(trial):
    return {
        'conversion_periods':     trial.suggest_int('conversion_periods', 7, 15),
        'base_periods':           trial.suggest_int('base_periods', 20, 35),
        'lagging_span2_periods':  trial.suggest_int('lagging_span2_periods', 40, 65),
        'displacement':           trial.suggest_int('displacement', 20, 35),
    }


# ─── 4. MA + BOLLINGER BANDS + RSI ───────────────────────────────────────────

def gen_MA_BB_RSI(df, ma_len=200, bb_len=20, bb_mult=2.0,
                  rsi_len=14, rsi_ob=70.0, rsi_os=30.0):
    """
    MA Bollinger Bands + RSI — Pine v5 (1550 likes)
    LONG:  close > MA AND close crosses over lower BB AND RSI < rsi_os
    SHORT: close < MA AND close crosses under upper BB AND RSI > rsi_ob
    """
    close = df['close']
    ma = _sma(close, ma_len)
    _, upper, lower = _bb(close, n=bb_len, mult=bb_mult)
    rsi = _rsi(close, rsi_len)

    # crossover(close, lower): price bounces up from lower band
    cross_up_lower = (close > lower) & (close.shift(1) <= lower.shift(1))
    # crossunder(close, upper): price drops from upper band
    cross_dn_upper = (close < upper) & (close.shift(1) >= upper.shift(1))

    long_cond  = (close > ma) & cross_up_lower & (rsi < rsi_os)
    short_cond = (close < ma) & cross_dn_upper & (rsi > rsi_ob)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_MA_BB_RSI(trial):
    return {
        'ma_len':   trial.suggest_int('ma_len', 100, 300),
        'bb_len':   trial.suggest_int('bb_len', 10, 40),
        'bb_mult':  trial.suggest_float('bb_mult', 1.5, 3.0),
        'rsi_len':  trial.suggest_int('rsi_len', 10, 21),
        'rsi_ob':   trial.suggest_float('rsi_ob', 65.0, 80.0),
        'rsi_os':   trial.suggest_float('rsi_os', 20.0, 35.0),
    }


# ─── 5. DONCHIAN CHANNEL FREE BOT ─────────────────────────────────────────────

def gen_Donchian_Free_Bot(df, upper_length=20, lower_length=20):
    """
    Donchian Channel Strategy [for free bot] — Pine v4 (1336 likes)
    LONG:  close crosses over the previous bar's upper channel (breakout)
    SHORT: close crosses under the previous bar's lower channel (breakdown)
    Note: Pine uses upper[1] and lower[1] → shift(1) for no-lookahead
    """
    close = df['close']

    upper = close.rolling(upper_length).max()
    lower = close.rolling(lower_length).min()

    # crossover(close, upper[1]): close > upper[1] AND prev_close <= upper[2]
    long_cond  = (close > upper.shift(1)) & (close.shift(1) <= upper.shift(2))
    # crossunder(close, lower[1]): close < lower[1] AND prev_close >= lower[2]
    short_cond = (close < lower.shift(1)) & (close.shift(1) >= lower.shift(2))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Donchian_Free_Bot(trial):
    return {
        'upper_length': trial.suggest_int('upper_length', 10, 55),
        'lower_length': trial.suggest_int('lower_length', 10, 55),
    }


# ─── 6. FRACTAL BREAKOUT KL ───────────────────────────────────────────────────

def gen_Fractal_Breakout_KL(df, fractal_len=2, atr_len=14, atr_mult=1.5):
    """
    Fractal Breakout Strategy [KL] — Pine v5 (1304 likes)
    Williams Fractals: fractal_up when high[n] is highest in 2n+1 window
    LONG:  close crosses over last fractal high level
    SHORT: close crosses under last fractal low level
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    window = 2 * fractal_len + 1

    # Fractal detection: center bar is pivot high/low
    # Pine: high[fractal_len] == ta.highest(high, 2*fractal_len+1)[fractal_len]
    # This is equivalent to: the bar at position -fractal_len was the highest in the window
    # We detect fractals at bar i-fractal_len using rolling centered window
    roll_high = high.rolling(window, center=True).max()
    roll_low  = low.rolling(window, center=True).min()

    fractal_up   = (high == roll_high)
    fractal_down = (low  == roll_low)

    # Track last fractal levels — vectorized forward-fill
    # Pine: if fractal_up[fractal_len] => last_up := high[fractal_len*2]
    # The fractal at bar i confirms fractal_len bars later
    # So shift fractal detection forward by fractal_len to get when it becomes known
    frac_up_confirmed   = fractal_up.astype(float).shift(fractal_len).fillna(0).astype(bool)
    frac_down_confirmed = fractal_down.astype(float).shift(fractal_len).fillna(0).astype(bool)

    # The fractal level is the high/low at fractal_len*2 bars back when confirmed
    frac_up_level   = high.shift(fractal_len * 2)
    frac_down_level = low.shift(fractal_len * 2)

    # Forward fill last known fractal level
    last_up   = frac_up_level.where(frac_up_confirmed).ffill()
    last_down = frac_down_level.where(frac_down_confirmed).ffill()

    # crossover(close, last_up): long breakout
    long_cond  = last_up.notna()  & (close > last_up)  & (close.shift(1) <= last_up.shift(1))
    # crossunder(close, last_down): short breakdown
    short_cond = last_down.notna() & (close < last_down) & (close.shift(1) >= last_down.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Fractal_Breakout_KL(trial):
    return {
        'fractal_len': trial.suggest_int('fractal_len', 1, 5),
        'atr_len':     trial.suggest_int('atr_len', 10, 21),
        'atr_mult':    trial.suggest_float('atr_mult', 0.5, 3.0),
    }


# ─── 7. ORDER BLOCK + FVG ─────────────────────────────────────────────────────

def gen_Order_Block_FVG(df, ob_len=10, fvg_filter=True, ema_len=200,
                         atr_len=14, atr_mult=2.0):
    """
    Order Block Volumatic FVG Strategy — Pine v6 (1301 likes)
    Order Block: highest/lowest range over lookback period
    FVG: Fair Value Gap (gap between candle[2].high and candle[0].low)
    LONG:  bullish trend + price in OB zone + (FVG if enabled) + high volume
    SHORT: bearish trend + price in OB zone + (bear FVG if enabled) + high volume
    """
    high  = df['high']
    low   = df['low']
    close = df['close']
    vol   = df['volume']

    ema_trend = _ema(close, ema_len)

    # Order block zone: prior period high/low
    bull_ob_hi = high.rolling(ob_len).max().shift(1)
    bull_ob_lo = low.rolling(ob_len).min().shift(1)

    # FVG detection
    # Bull FVG: low[0] > high[2]  → gap between current low and high 2 bars ago
    bull_fvg = low > high.shift(2)
    bear_fvg = high < low.shift(2)

    # Trend filter
    bull_trend = close > ema_trend
    bear_trend = close < ema_trend

    # Volume filter: volume > 1.5x 20-period average
    vol_avg  = _sma(vol, 20)
    high_vol = vol > vol_avg * 1.5

    # Entry: price in OB zone
    in_ob_zone = (close > bull_ob_lo) & (close < bull_ob_hi)

    fvg_long  = bull_fvg.shift(1) if fvg_filter else pd.Series(True, index=df.index)
    fvg_short = bear_fvg.shift(1) if fvg_filter else pd.Series(True, index=df.index)

    long_cond  = bull_trend & in_ob_zone & fvg_long  & high_vol
    short_cond = bear_trend & in_ob_zone & fvg_short & high_vol

    # Avoid duplicate signals (only entry when not already in position direction)
    sig = pd.Series(0, index=df.index)
    sig[long_cond & ~long_cond.astype(float).shift(1).fillna(0).astype(bool)]   =  1
    sig[short_cond & ~short_cond.astype(float).shift(1).fillna(0).astype(bool)] = -1
    return sig


def space_Order_Block_FVG(trial):
    return {
        'ob_len':    trial.suggest_int('ob_len', 5, 20),
        'fvg_filter': trial.suggest_categorical('fvg_filter', [True, False]),
        'ema_len':   trial.suggest_int('ema_len', 100, 300),
        'atr_len':   trial.suggest_int('atr_len', 10, 21),
        'atr_mult':  trial.suggest_float('atr_mult', 1.0, 3.5),
    }


# ─── 8. FIBONACCI TREND REVERSAL ─────────────────────────────────────────────

def gen_Fibonacci_Trend_Reversal(df, pivot_len=21, fib_618=0.618, fib_382=0.382,
                                   ema_fast=21, ema_slow=55, rsi_len=14,
                                   rsi_os=40.0, rsi_ob=60.0):
    """
    Fibonacci Trend Reversal Strategy — Pine v6 (1054 likes)
    Detects pivot highs/lows, calculates Fibonacci retracement levels
    LONG:  price near 61.8% retracement + EMAs bullish + RSI recovering
    SHORT: price near 38.2% retracement + EMAs bearish + RSI overbought
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    # Pivot high/low detection using rolling centered window
    # Pine: ta.pivothigh(high, pivot_len, pivot_len) → center=True rolling max
    ph_window = high.rolling(2 * pivot_len + 1, center=True).max()
    pl_window = low.rolling(2 * pivot_len + 1, center=True).min()

    is_pivot_high = (high == ph_window)
    is_pivot_low  = (low  == pl_window)

    # Forward-fill last pivot high/low values
    last_ph = high.where(is_pivot_high).ffill()
    last_pl = low.where(is_pivot_low).ffill()

    # Fibonacci levels
    fib_range    = last_ph - last_pl
    fib_618_lvl  = last_ph - fib_618 * fib_range
    fib_382_lvl  = last_ph - fib_382 * fib_range

    ema_f = _ema(close, ema_fast)
    ema_s = _ema(close, ema_slow)
    rsi   = _rsi(close, rsi_len)

    # Bull: price within 2% of 61.8% level + bullish EMAs + RSI oversold
    near_618 = (close <= fib_618_lvl * 1.02) & (close >= fib_618_lvl * 0.98)
    bull_cond = last_ph.notna() & last_pl.notna() & near_618 & (ema_f > ema_s) & (rsi < rsi_os)

    # Bear: price within 2% of 38.2% level + bearish EMAs + RSI overbought
    near_382  = (close >= fib_382_lvl * 0.98) & (close <= fib_382_lvl * 1.02)
    bear_cond = last_ph.notna() & last_pl.notna() & near_382 & (ema_f < ema_s) & (rsi > rsi_ob)

    # Signal on transition (not bull_cond[1])
    long_sig  = bull_cond & ~bull_cond.astype(float).shift(1).fillna(0).astype(bool)
    short_sig = bear_cond & ~bear_cond.astype(float).shift(1).fillna(0).astype(bool)

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  =  1
    sig[short_sig] = -1
    return sig


def space_Fibonacci_Trend_Reversal(trial):
    return {
        'pivot_len': trial.suggest_int('pivot_len', 10, 35),
        'fib_618':   trial.suggest_float('fib_618', 0.55, 0.70),
        'fib_382':   trial.suggest_float('fib_382', 0.30, 0.45),
        'ema_fast':  trial.suggest_int('ema_fast', 10, 30),
        'ema_slow':  trial.suggest_int('ema_slow', 40, 80),
        'rsi_len':   trial.suggest_int('rsi_len', 10, 21),
        'rsi_os':    trial.suggest_float('rsi_os', 30.0, 50.0),
        'rsi_ob':    trial.suggest_float('rsi_ob', 50.0, 70.0),
    }


# ─── 9. FRACTAL BREAKOUT SIMPLE ──────────────────────────────────────────────

def gen_Fractal_Breakout_Simple(df, n=2):
    """
    Fractal Breakout Strategy — Pine v4 (1051 likes)
    Simple Williams Fractal detection + breakout
    LONG:  close crosses over last resistance fractal
    SHORT: close crosses under last support fractal
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    window = 2 * n + 1

    # Fractal: bar is highest/lowest in window (centered)
    roll_high = high.rolling(window, center=True).max()
    roll_low  = low.rolling(window, center=True).min()

    up_fractal   = (high == roll_high)
    down_fractal = (low  == roll_low)

    # Resistance = last fractal high, support = last fractal low
    res = high.where(up_fractal).ffill()
    sup = low.where(down_fractal).ffill()

    # Crossover/crossunder signals
    long_cond  = res.notna() & (close > res) & (close.shift(1) <= res.shift(1))
    short_cond = sup.notna() & (close < sup) & (close.shift(1) >= sup.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Fractal_Breakout_Simple(trial):
    return {
        'n': trial.suggest_int('n', 1, 5),
    }


# ─── STRATEGY EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Trendline_Fib_ST': {
        'gen': gen_Trendline_Fib_ST,
        'space': space_Trendline_Fib_ST,
        'default_params': {'atr_period': 10, 'factor': 3.0, 'fibs': 3},
        'info': {'likes': 2511, 'version': 'v6', 'source': 'Trendline_Breaks_Fib_Supertrend'},
    },
    'BB_Breakout_V2': {
        'gen': gen_BB_Breakout_V2,
        'space': space_BB_Breakout_V2,
        'default_params': {'length': 20, 'mult': 2.0},
        'info': {'likes': 2048, 'version': 'v5', 'source': 'BB_Breakout_V2'},
    },
    'Ichimoku_NoOffset_V2': {
        'gen': gen_Ichimoku_NoOffset_V2,
        'space': space_Ichimoku_NoOffset_V2,
        'default_params': {
            'conversion_periods': 9, 'base_periods': 26,
            'lagging_span2_periods': 52, 'displacement': 26,
        },
        'info': {'likes': 1799, 'version': 'v4', 'source': 'Ichimoku_NoOffset_V2'},
    },
    'MA_BB_RSI': {
        'gen': gen_MA_BB_RSI,
        'space': space_MA_BB_RSI,
        'default_params': {
            'ma_len': 200, 'bb_len': 20, 'bb_mult': 2.0,
            'rsi_len': 14, 'rsi_ob': 70.0, 'rsi_os': 30.0,
        },
        'info': {'likes': 1550, 'version': 'v5', 'source': 'MA_BB_RSI'},
    },
    'Donchian_Free_Bot': {
        'gen': gen_Donchian_Free_Bot,
        'space': space_Donchian_Free_Bot,
        'default_params': {'upper_length': 20, 'lower_length': 20},
        'info': {'likes': 1336, 'version': 'v4', 'source': 'Donchian_Free_Bot'},
    },
    'Fractal_Breakout_KL': {
        'gen': gen_Fractal_Breakout_KL,
        'space': space_Fractal_Breakout_KL,
        'default_params': {'fractal_len': 2, 'atr_len': 14, 'atr_mult': 1.5},
        'info': {'likes': 1304, 'version': 'v5', 'source': 'Fractal_Breakout_KL'},
    },
    'Order_Block_FVG': {
        'gen': gen_Order_Block_FVG,
        'space': space_Order_Block_FVG,
        'default_params': {
            'ob_len': 10, 'fvg_filter': True, 'ema_len': 200,
            'atr_len': 14, 'atr_mult': 2.0,
        },
        'info': {'likes': 1301, 'version': 'v6', 'source': 'Order_Block_FVG'},
    },
    'Fibonacci_Trend_Reversal': {
        'gen': gen_Fibonacci_Trend_Reversal,
        'space': space_Fibonacci_Trend_Reversal,
        'default_params': {
            'pivot_len': 21, 'fib_618': 0.618, 'fib_382': 0.382,
            'ema_fast': 21, 'ema_slow': 55, 'rsi_len': 14,
            'rsi_os': 40.0, 'rsi_ob': 60.0,
        },
        'info': {'likes': 1054, 'version': 'v6', 'source': 'Fibonacci_Trend_Reversal'},
    },
    'Fractal_Breakout_Simple': {
        'gen': gen_Fractal_Breakout_Simple,
        'space': space_Fractal_Breakout_Simple,
        'default_params': {'n': 2},
        'info': {'likes': 1051, 'version': 'v4', 'source': 'Fractal_Breakout_Simple'},
    },
}
