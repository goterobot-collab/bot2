#!/usr/bin/env python3
"""
TV2 Batch 9a — 13 estrategias Pine → Python
Likes totales: ~8,458

 1. MVRV_ZScore          (692L)  — Pine v5 — MVRV Z-Score approximation
 2. Williams_Fractals    (681L)  — Pine v4 — Williams Fractals breakout
 3. Flash_Minervini      (678L)  — Pine v5 — Flash + Minervini Stage 2
 4. ATR_Kijun_WR_NNFX   (672L)  — Pine v4 — ATR + Kijun + Williams %R (NNFX)
 5. MACD_RSI_Signal_V2  (658L)  — Pine v4 — MACD Signal + RSI filter
 6. Aggressive_Scalper_V2 (648L) — Pine v4 — Range + Vortex indicator
 7. Ichimoku_Long_Short  (644L)  — Pine v5 — Ichimoku TK cross + cloud
 8. DCA_BB_MeanReversion (630L)  — Pine v5 — BB mean-reversion + RSI + EMA
 9. HA_Wick              (624L)  — Pine v5 — Heikin Ashi wick strategy
10. HA_Smoothed_Abdoli   (578L)  — Pine v4 — Smoothed HA buy/sell
11. Donchian_KrisWaters  (577L)  — Pine v4 — Donchian channel breakout
12. AllInOne_NoRSI       (573L)  — Pine v4 — Multi-MA alignment + BB
13. EMA_RSI_Pump_Drop    (550L)  — Pine v4 — EMA crossover + RSI pump/drop

Convención: sig = 1 (LONG), -1 (SHORT), 0 (sin señal)
Sin look-ahead: señales basadas en datos hasta la barra actual.
"""

import numpy as np
import pandas as pd

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
    close = df['close'].values
    fu_v = bu.values.copy(); fd_v = bl.values.copy()
    first = (~np.isnan(fu_v)).argmax()
    for i in range(first+1,len(df)):
        fu_v[i] = min(fu_v[i], fu_v[i-1]) if close[i-1]<=fu_v[i-1] else fu_v[i]
        fd_v[i] = max(fd_v[i], fd_v[i-1]) if close[i-1]>=fd_v[i-1] else fd_v[i]
    trend = pd.Series(np.nan, index=df.index)
    trend.iloc[first] = 1
    for i in range(first+1,len(df)):
        if trend.iloc[i-1]==1 and close[i]<fd_v[i]: trend.iloc[i]=-1
        elif trend.iloc[i-1]==-1 and close[i]>fu_v[i]: trend.iloc[i]=1
        else: trend.iloc[i]=trend.iloc[i-1]
    return trend, pd.Series(fd_v,index=df.index), pd.Series(fu_v,index=df.index)


# ─── 1. MVRV_ZScore ──────────────────────────────────────────────────────────

def gen_MVRV_ZScore(df, length_fast=30, length_slow=365, z_buy=-0.5, z_sell=2.0, **kwargs):
    """
    MVRV Z-Score approximation: (fast_SMA - slow_SMA) / stdev(close, slow)
    Long on crossover above z_buy, Short on crossunder below z_sell.
    """
    close = df['close']
    fast_ma = _sma(close, length_fast)
    slow_ma = _sma(close, length_slow)
    std_dev = close.rolling(length_slow).std()
    z_score = (fast_ma - slow_ma) / (std_dev + 1e-9)

    long_cond  = (z_score > z_buy) & (z_score.shift(1) <= z_buy)
    short_cond = (z_score < z_sell) & (z_score.shift(1) >= z_sell)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_MVRV_ZScore(trial):
    return {
        'length_fast': trial.suggest_int('length_fast', 10, 60),
        'length_slow': trial.suggest_int('length_slow', 100, 500),
        'z_buy':  trial.suggest_float('z_buy',  -2.0, 0.5),
        'z_sell': trial.suggest_float('z_sell',  0.5, 4.0),
    }


# ─── 2. Williams_Fractals ─────────────────────────────────────────────────────

def gen_Williams_Fractals(df, n=2, **kwargs):
    """
    Williams Fractals: up fractal = high[n] is highest in 2n+1 window.
    Entry when close crosses above last up fractal (long) or below last down fractal (short).
    Uses bar[n] shift to avoid look-ahead (fractal confirmed n bars later).
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    window = 2 * n + 1
    # Fractal at bar [n] is confirmed when we're at bar [0] (n bars later)
    # high.shift(n) = value of high n bars ago
    roll_max = high.shift(n).rolling(window).max()
    roll_min = low.shift(n).rolling(window).min()

    up_fractal   = high.shift(n) == roll_max
    down_fractal = low.shift(n)  == roll_min

    # Track last fractal level (stateful-style using forward-fill)
    up_level   = high.shift(n).where(up_fractal).ffill()
    down_level = low.shift(n).where(down_fractal).ffill()

    long_cond  = (~up_level.isna()) & (close > up_level) & (close.shift(1) <= up_level.shift(1))
    short_cond = (~down_level.isna()) & (close < down_level) & (close.shift(1) >= down_level.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Williams_Fractals(trial):
    return {
        'n': trial.suggest_int('n', 1, 5),
    }


# ─── 3. Flash_Minervini ───────────────────────────────────────────────────────

def gen_Flash_Minervini(df, ema_fast=8, ema_slow=21, ema_200=200, ema_150=150,
                        rsi_len=14, rsi_buy=50.0, **kwargs):
    """
    Flash Strategy + Minervini Stage 2 analysis.
    Long: fast EMA crosses above slow EMA + price > 200 EMA + 150 EMA > 200 EMA + RSI > rsi_buy.
    Short: fast EMA crosses below slow EMA + NOT stage 2.
    """
    close = df['close']
    ema1  = _ema(close, ema_fast)
    ema2  = _ema(close, ema_slow)
    ema_l = _ema(close, ema_150)
    ema_t = _ema(close, ema_200)
    rsi   = _rsi(close, rsi_len)

    stage2 = (close > ema_t) & (ema_l > ema_t)

    cross_up   = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
    cross_down = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))

    long_cond  = cross_up   & stage2 & (rsi > rsi_buy)
    short_cond = cross_down & (~stage2)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Flash_Minervini(trial):
    return {
        'ema_fast': trial.suggest_int('ema_fast', 5, 20),
        'ema_slow': trial.suggest_int('ema_slow', 15, 50),
        'ema_200':  trial.suggest_int('ema_200',  100, 250),
        'ema_150':  trial.suggest_int('ema_150',  80, 200),
        'rsi_len':  trial.suggest_int('rsi_len',  10, 20),
        'rsi_buy':  trial.suggest_float('rsi_buy', 40.0, 65.0),
    }


# ─── 4. ATR_Kijun_WR_NNFX ────────────────────────────────────────────────────

def gen_ATR_Kijun_WR_NNFX(df, kijun_len=26, wr_len=14, wr_ob=-20.0, wr_os=-80.0, **kwargs):
    """
    NNFX style: Kijun-Sen as baseline, Williams %R as confirmation.
    Long: close crosses above Kijun AND WR > wr_os (not oversold).
    Short: close crosses below Kijun AND WR < wr_ob (not overbought).
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    kijun = (high.rolling(kijun_len).max() + low.rolling(kijun_len).min()) / 2
    wr    = -100.0 * (high.rolling(wr_len).max() - close) / (
                high.rolling(wr_len).max() - low.rolling(wr_len).min() + 1e-9)

    cross_up   = (close > kijun) & (close.shift(1) <= kijun.shift(1))
    cross_down = (close < kijun) & (close.shift(1) >= kijun.shift(1))

    long_cond  = cross_up   & (wr > wr_os)
    short_cond = cross_down & (wr < wr_ob)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_ATR_Kijun_WR_NNFX(trial):
    return {
        'kijun_len': trial.suggest_int('kijun_len', 9, 52),
        'wr_len':    trial.suggest_int('wr_len',    7, 28),
        'wr_ob':     trial.suggest_float('wr_ob',  -30.0, -5.0),
        'wr_os':     trial.suggest_float('wr_os',  -95.0, -60.0),
    }


# ─── 5. MACD_RSI_Signal_V2 ────────────────────────────────────────────────────

def gen_MACD_RSI_Signal_V2(df, fast_len=12, slow_len=26, signal_len=9,
                            rsi_len=14, rsi_ob=70.0, rsi_os=30.0, **kwargs):
    """
    MACD Signal line crossover with RSI filter.
    Long: MACD crosses above signal AND RSI < rsi_ob.
    Short: MACD crosses below signal AND RSI > rsi_os.
    """
    close = df['close']
    macd_line = _ema(close, fast_len) - _ema(close, slow_len)
    sig_line  = _ema(macd_line, signal_len)
    rsi_v     = _rsi(close, rsi_len)

    cross_up   = (macd_line > sig_line) & (macd_line.shift(1) <= sig_line.shift(1))
    cross_down = (macd_line < sig_line) & (macd_line.shift(1) >= sig_line.shift(1))

    long_cond  = cross_up   & (rsi_v < rsi_ob)
    short_cond = cross_down & (rsi_v > rsi_os)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_MACD_RSI_Signal_V2(trial):
    return {
        'fast_len':   trial.suggest_int('fast_len',   5, 20),
        'slow_len':   trial.suggest_int('slow_len',  15, 50),
        'signal_len': trial.suggest_int('signal_len', 5, 15),
        'rsi_len':    trial.suggest_int('rsi_len',    7, 21),
        'rsi_ob':     trial.suggest_float('rsi_ob',  60.0, 85.0),
        'rsi_os':     trial.suggest_float('rsi_os',  15.0, 40.0),
    }


# ─── 6. Aggressive_Scalper_V2 ─────────────────────────────────────────────────

def gen_Aggressive_Scalper_V2(df, range_p=60, vortex_p=14, **kwargs):
    """
    Range mid-point filter + Vortex indicator crossover.
    Long: close > mid_range AND VI+ crosses above VI-.
    Short: close < mid_range AND VI+ crosses below VI-.
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    highest_h = high.rolling(range_p).max()
    lowest_l  = low.rolling(range_p).min()
    mid_range = (highest_h + lowest_l) / 2

    tr_v = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    vm_plus  = (high - low.shift()).abs()
    vm_minus = (low  - high.shift()).abs()

    vip = vm_plus.rolling(vortex_p).sum()  / (tr_v.rolling(vortex_p).sum() + 1e-9)
    vim = vm_minus.rolling(vortex_p).sum() / (tr_v.rolling(vortex_p).sum() + 1e-9)

    cross_up   = (vip > vim) & (vip.shift(1) <= vim.shift(1))
    cross_down = (vip < vim) & (vip.shift(1) >= vim.shift(1))

    long_cond  = (close > mid_range) & cross_up
    short_cond = (close < mid_range) & cross_down

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Aggressive_Scalper_V2(trial):
    return {
        'range_p':  trial.suggest_int('range_p',  20, 120),
        'vortex_p': trial.suggest_int('vortex_p',  7,  28),
    }


# ─── 7. Ichimoku_Long_Short ───────────────────────────────────────────────────

def gen_Ichimoku_Long_Short(df, conv_len=9, base_len=26, span2_len=52, **kwargs):
    """
    Ichimoku Clouds TK crossover + price vs cloud (no offset = no look-ahead).
    Long: Tenkan crosses above Kijun AND close above cloud top.
    Short: Tenkan crosses below Kijun AND close below cloud bottom.
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    def donchian(n):
        return (high.rolling(n).max() + low.rolling(n).min()) / 2

    tenkan = donchian(conv_len)
    kijun  = donchian(base_len)
    spanA  = (tenkan + kijun) / 2
    spanB  = donchian(span2_len)

    cloud_top = pd.concat([spanA, spanB], axis=1).max(axis=1)
    cloud_bot = pd.concat([spanA, spanB], axis=1).min(axis=1)

    tk_bull = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    tk_bear = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))

    above_cloud = close > cloud_top
    below_cloud = close < cloud_bot

    long_cond  = tk_bull & above_cloud
    short_cond = tk_bear & below_cloud

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Ichimoku_Long_Short(trial):
    return {
        'conv_len':  trial.suggest_int('conv_len',   5, 20),
        'base_len':  trial.suggest_int('base_len',  15, 52),
        'span2_len': trial.suggest_int('span2_len', 30, 104),
    }


# ─── 8. DCA_BB_MeanReversion ──────────────────────────────────────────────────

def gen_DCA_BB_MeanReversion(df, bb_len=20, bb_mult=2.0, rsi_len=14,
                              rsi_os=30.0, rsi_ob=70.0, ema_len=50, **kwargs):
    """
    BB mean-reversion: buy on lower band touch + RSI oversold + above EMA filter.
    Short on upper band touch + RSI overbought.
    """
    close = df['close']
    _, upper, lower = _bb(close, bb_len, bb_mult)
    rsi   = _rsi(close, rsi_len)
    ema   = _ema(close, ema_len)

    long_cond  = (close > lower) & (close.shift(1) <= lower.shift(1)) & (rsi < rsi_os)
    short_cond = (close < upper) & (close.shift(1) >= upper.shift(1)) & (rsi > rsi_ob)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_DCA_BB_MeanReversion(trial):
    return {
        'bb_len':  trial.suggest_int('bb_len',   10, 50),
        'bb_mult': trial.suggest_float('bb_mult', 1.0, 3.0),
        'rsi_len': trial.suggest_int('rsi_len',   7, 21),
        'rsi_os':  trial.suggest_float('rsi_os',  15.0, 40.0),
        'rsi_ob':  trial.suggest_float('rsi_ob',  60.0, 85.0),
        'ema_len': trial.suggest_int('ema_len',   20, 100),
    }


# ─── 9. HA_Wick ───────────────────────────────────────────────────────────────

def gen_HA_Wick(df, ema_len=21, **kwargs):
    """
    Heikin Ashi Wick Strategy.
    Long: no upper wick (HA high == max(haOpen, haClose)) AND haClose > haOpen AND close > EMA.
    Short: no lower wick (HA low == min(haOpen, haClose)) AND haClose < haOpen AND close < EMA.
    Only on first bar of new condition (edge detection).
    """
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)

    ha_close = (o + h + l + c) / 4.0
    ha_open  = np.empty(n)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0

    ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
    ha_low  = np.minimum(l, np.minimum(ha_open, ha_close))

    ha_close_s = pd.Series(ha_close, index=df.index)
    ha_open_s  = pd.Series(ha_open,  index=df.index)
    ha_high_s  = pd.Series(ha_high,  index=df.index)
    ha_low_s   = pd.Series(ha_low,   index=df.index)

    ema = _ema(df['close'], ema_len)

    # Pine: noUpperWick = haHigh == max(haOpen, haClose)
    # In real OHLC → HA conversion, haHigh = max(real_high, haOpen, haClose).
    # "No upper wick" in HA means the real high did not extend above the HA body top.
    # Practical interpretation: upper wick (above body) is tiny relative to body size.
    # We use: upper_wick / body_size < threshold to catch "effectively no wick" bars.
    body_size = (ha_close_s - ha_open_s).abs() + 1e-9
    upper_wick = ha_high_s - pd.concat([ha_open_s, ha_close_s], axis=1).max(axis=1)
    lower_wick = pd.concat([ha_open_s, ha_close_s], axis=1).min(axis=1) - ha_low_s

    bull_body = ha_close_s > ha_open_s
    bear_body = ha_close_s < ha_open_s

    # Strong bull HA: very small upper wick relative to body
    no_upper_wick = upper_wick / body_size < 0.2
    # Strong bear HA: very small lower wick relative to body
    no_lower_wick = lower_wick / body_size < 0.2

    bull = no_upper_wick & bull_body & (df['close'] > ema)
    bear = no_lower_wick & bear_body & (df['close'] < ema)

    # Only on first bar of new condition
    long_cond  = bull & (~bull.shift(1).astype('boolean').fillna(False))
    short_cond = bear & (~bear.shift(1).astype('boolean').fillna(False))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_HA_Wick(trial):
    return {
        'ema_len': trial.suggest_int('ema_len', 10, 50),
    }


# ─── 10. HA_Smoothed_Abdoli ───────────────────────────────────────────────────

def gen_HA_Smoothed_Abdoli(df, smooth=3, **kwargs):
    """
    Abdoli Smoothed Heikin Ashi: apply EMA smoothing to HA candles.
    Long: smoothed HA close crosses above smoothed HA open (bull transition).
    Short: smoothed HA close crosses below smoothed HA open (bear transition).
    """
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)

    ha_close_arr = (o + h + l + c) / 4.0
    ha_open_arr  = np.empty(n)
    ha_open_arr[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open_arr[i] = (ha_open_arr[i-1] + ha_close_arr[i-1]) / 2.0

    ha_close_s = pd.Series(ha_close_arr, index=df.index)
    ha_open_s  = pd.Series(ha_open_arr,  index=df.index)

    s_close = _ema(ha_close_s, smooth)
    s_open  = _ema(ha_open_s,  smooth)

    ha_bull = s_close > s_open
    ha_bear = s_close < s_open

    long_cond  = ha_bull  & (~ha_bull.shift(1).astype('boolean').fillna(True))
    short_cond = ha_bear  & (~ha_bear.shift(1).astype('boolean').fillna(True))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_HA_Smoothed_Abdoli(trial):
    return {
        'smooth': trial.suggest_int('smooth', 2, 10),
    }


# ─── 11. Donchian_KrisWaters ──────────────────────────────────────────────────

def gen_Donchian_KrisWaters(df, dc_len=20, **kwargs):
    """
    Donchian Channel breakout by KrisWaters.
    Long: close crosses above previous upper channel (breakout).
    Short: close crosses below previous lower channel (breakdown).
    Uses dc_len-1 shift (upper[1] in Pine = previous bar's upper).
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    upper = high.rolling(dc_len).max()
    lower = low.rolling(dc_len).min()
    mid   = (upper + lower) / 2

    # Breakout: close crosses above upper of previous bar
    long_cond  = (close > upper.shift(1)) & (close.shift(1) <= upper.shift(1))
    short_cond = (close < lower.shift(1)) & (close.shift(1) >= lower.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Donchian_KrisWaters(trial):
    return {
        'dc_len': trial.suggest_int('dc_len', 10, 60),
    }


# ─── 12. AllInOne_NoRSI ───────────────────────────────────────────────────────

def gen_AllInOne_NoRSI(df, sma_len=50, ema_len=20, wma_len=10,
                        bb_len=20, bb_mult=2.0, **kwargs):
    """
    Multi-MA alignment + Bollinger Bands.
    Long: all MAs aligned bull (close > SMA > EMA check + EMA > SMA) AND close crosses above lower BB.
    Short: all MAs aligned bear AND close crosses below upper BB.
    """
    close = df['close']
    sma_v = _sma(close, sma_len)
    ema_v = _ema(close, ema_len)
    wma_v = _wma(close, wma_len)
    _, upper, lower = _bb(close, bb_len, bb_mult)

    all_bull = (close > sma_v) & (close > ema_v) & (close > wma_v) & (ema_v > sma_v)
    all_bear = (close < sma_v) & (close < ema_v) & (close < wma_v) & (ema_v < sma_v)

    long_cond  = all_bull & (close > lower) & (close.shift(1) <= lower.shift(1))
    short_cond = all_bear & (close < upper) & (close.shift(1) >= upper.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_AllInOne_NoRSI(trial):
    return {
        'sma_len':  trial.suggest_int('sma_len',  20, 100),
        'ema_len':  trial.suggest_int('ema_len',  10,  50),
        'wma_len':  trial.suggest_int('wma_len',   5,  30),
        'bb_len':   trial.suggest_int('bb_len',   10,  40),
        'bb_mult':  trial.suggest_float('bb_mult', 1.0, 3.0),
    }


# ─── 13. EMA_RSI_Pump_Drop ────────────────────────────────────────────────────

def gen_EMA_RSI_Pump_Drop(df, ema_fast=9, ema_slow=21, rsi_len=14,
                           rsi_ob=70.0, rsi_os=30.0, **kwargs):
    """
    EMA crossover + RSI pump/drop filter (Swing Sniper).
    Long: fast EMA crosses above slow EMA AND RSI < rsi_ob AND RSI > 40.
    Short: fast EMA crosses below slow EMA AND RSI > rsi_os AND RSI < 60.
    """
    close = df['close']
    ema1  = _ema(close, ema_fast)
    ema2  = _ema(close, ema_slow)
    rsi   = _rsi(close, rsi_len)

    cross_up   = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
    cross_down = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))

    long_cond  = cross_up   & (rsi < rsi_ob) & (rsi > 40)
    short_cond = cross_down & (rsi > rsi_os) & (rsi < 60)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_EMA_RSI_Pump_Drop(trial):
    return {
        'ema_fast': trial.suggest_int('ema_fast',  5, 20),
        'ema_slow': trial.suggest_int('ema_slow', 15, 50),
        'rsi_len':  trial.suggest_int('rsi_len',   7, 21),
        'rsi_ob':   trial.suggest_float('rsi_ob',  55.0, 80.0),
        'rsi_os':   trial.suggest_float('rsi_os',  20.0, 45.0),
    }


# ─── EXPORT ───────────────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'MVRV_ZScore': {
        'gen': gen_MVRV_ZScore,
        'space': space_MVRV_ZScore,
        'default_params': {
            'length_fast': 30,
            'length_slow': 365,
            'z_buy': -0.5,
            'z_sell': 2.0,
        },
        'info': {'likes': 692, 'version': 'v5', 'source': 'Crypto_MVRV_ZScore___Strategy__Pres'},
    },
    'Williams_Fractals': {
        'gen': gen_Williams_Fractals,
        'space': space_Williams_Fractals,
        'default_params': {'n': 2},
        'info': {'likes': 681, 'version': 'v4', 'source': 'Williams_Fractals_Strategy'},
    },
    'Flash_Minervini': {
        'gen': gen_Flash_Minervini,
        'space': space_Flash_Minervini,
        'default_params': {
            'ema_fast': 8,
            'ema_slow': 21,
            'ema_200': 200,
            'ema_150': 150,
            'rsi_len': 14,
            'rsi_buy': 50.0,
        },
        'info': {'likes': 678, 'version': 'v5', 'source': 'The_Flash_Strategy_with_Minervini_S'},
    },
    'ATR_Kijun_WR_NNFX': {
        'gen': gen_ATR_Kijun_WR_NNFX,
        'space': space_ATR_Kijun_WR_NNFX,
        'default_params': {
            'kijun_len': 26,
            'wr_len': 14,
            'wr_ob': -20.0,
            'wr_os': -80.0,
        },
        'info': {'likes': 672, 'version': 'v4', 'source': 'ATR__Kijun_Sen___R_Strategy__No_Non'},
    },
    'MACD_RSI_Signal_V2': {
        'gen': gen_MACD_RSI_Signal_V2,
        'space': space_MACD_RSI_Signal_V2,
        'default_params': {
            'fast_len': 12,
            'slow_len': 26,
            'signal_len': 9,
            'rsi_len': 14,
            'rsi_ob': 70.0,
            'rsi_os': 30.0,
        },
        'info': {'likes': 658, 'version': 'v4', 'source': 'MACD_Signal_with_RSI_Indicating_Str'},
    },
    'Aggressive_Scalper_V2': {
        'gen': gen_Aggressive_Scalper_V2,
        'space': space_Aggressive_Scalper_V2,
        'default_params': {
            'range_p': 60,
            'vortex_p': 14,
        },
        'info': {'likes': 648, 'version': 'v4', 'source': 'Aggresive_Scalper_Swing_Crypto_Stra'},
    },
    'Ichimoku_Long_Short': {
        'gen': gen_Ichimoku_Long_Short,
        'space': space_Ichimoku_Long_Short,
        'default_params': {
            'conv_len': 9,
            'base_len': 26,
            'span2_len': 52,
        },
        'info': {'likes': 644, 'version': 'v5', 'source': 'Ichimoku_Clouds_Strategy_Long_and_S'},
    },
    'DCA_BB_MeanReversion': {
        'gen': gen_DCA_BB_MeanReversion,
        'space': space_DCA_BB_MeanReversion,
        'default_params': {
            'bb_len': 20,
            'bb_mult': 2.0,
            'rsi_len': 14,
            'rsi_os': 30.0,
            'rsi_ob': 70.0,
            'ema_len': 50,
        },
        'info': {'likes': 630, 'version': 'v5', 'source': 'DCA_Strategy_with_Mean_Reversion_an'},
    },
    'HA_Wick': {
        'gen': gen_HA_Wick,
        'space': space_HA_Wick,
        'default_params': {'ema_len': 21},
        'info': {'likes': 624, 'version': 'v5', 'source': 'Heikin_Ashi_Wick_Strategy'},
    },
    'HA_Smoothed_Abdoli': {
        'gen': gen_HA_Smoothed_Abdoli,
        'space': space_HA_Smoothed_Abdoli,
        'default_params': {'smooth': 3},
        'info': {'likes': 578, 'version': 'v4', 'source': 'Abdoli_s_Heikin_Ashi_Smoothed_Buy'},
    },
    'Donchian_KrisWaters': {
        'gen': gen_Donchian_KrisWaters,
        'space': space_Donchian_KrisWaters,
        'default_params': {'dc_len': 20},
        'info': {'likes': 577, 'version': 'v4', 'source': 'Donchian_Channels_Strategy_by_KrisW'},
    },
    'AllInOne_NoRSI': {
        'gen': gen_AllInOne_NoRSI,
        'space': space_AllInOne_NoRSI,
        'default_params': {
            'sma_len': 50,
            'ema_len': 20,
            'wma_len': 10,
            'bb_len': 20,
            'bb_mult': 2.0,
        },
        'info': {'likes': 573, 'version': 'v4', 'source': 'All_in_One_Strategy_no_RSI_Label'},
    },
    'EMA_RSI_Pump_Drop': {
        'gen': gen_EMA_RSI_Pump_Drop,
        'space': space_EMA_RSI_Pump_Drop,
        'default_params': {
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_len': 14,
            'rsi_ob': 70.0,
            'rsi_os': 30.0,
        },
        'info': {'likes': 550, 'version': 'v4', 'source': 'EMA_RSI_Pump___Drop_Swing_Sniper__S'},
    },
}
