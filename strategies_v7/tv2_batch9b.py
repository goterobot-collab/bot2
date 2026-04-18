#!/usr/bin/env python3
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


# ─────────────────────────────────────────────
# 1. BB_Strategy_V2
# ─────────────────────────────────────────────
def gen_BB_Strategy_V2(df, **p):
    bb_len  = int(p.get('bb_len', 20))
    bb_mult = float(p.get('bb_mult', 2.0))
    rsi_len = int(p.get('rsi_len', 14))
    rsi_ob  = float(p.get('rsi_ob', 65.0))

    basis = _sma(df['close'], bb_len)
    std   = df['close'].rolling(bb_len).std()
    lower = basis - bb_mult * std
    rsi   = _rsi(df['close'], rsi_len)

    # crossover(close, lower): close crosses above lower
    cross_up = (df['close'] > lower) & (df['close'].shift(1) <= lower.shift(1))
    long_cond = cross_up & (rsi < rsi_ob)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    return sig

def space_BB_Strategy_V2(trial):
    return {
        'bb_len':  trial.suggest_int('bb_len', 10, 40),
        'bb_mult': trial.suggest_float('bb_mult', 1.0, 3.0),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'rsi_ob':  trial.suggest_float('rsi_ob', 55.0, 80.0),
    }


# ─────────────────────────────────────────────
# 2. MACD_Aspray
# ─────────────────────────────────────────────
def gen_MACD_Aspray(df, **p):
    fast_len    = int(p.get('fast_len', 12))
    slow_len    = int(p.get('slow_len', 26))
    signal_len  = int(p.get('signal_len', 9))
    hist_thresh = float(p.get('hist_thresh', 0.0))

    macd = _ema(df['close'], fast_len) - _ema(df['close'], slow_len)
    sig_line = _ema(macd, signal_len)
    hist = macd - sig_line

    # Histogram turning point: hist > thresh and hist > hist[1] and hist[1] < hist[2]
    long_cond = (
        (hist > hist_thresh) &
        (hist > hist.shift(1)) &
        (hist.shift(1) < hist.shift(2))
    )

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    return sig

def space_MACD_Aspray(trial):
    return {
        'fast_len':    trial.suggest_int('fast_len', 5, 20),
        'slow_len':    trial.suggest_int('slow_len', 15, 50),
        'signal_len':  trial.suggest_int('signal_len', 3, 15),
        'hist_thresh': trial.suggest_float('hist_thresh', 0.0, 0.001),
    }


# ─────────────────────────────────────────────
# 3. Ichimoku_EMA_RSI_Long
# ─────────────────────────────────────────────
def gen_Ichimoku_EMA_RSI_Long(df, **p):
    conv_len  = int(p.get('conv_len', 9))
    base_len  = int(p.get('base_len', 26))
    span2_len = int(p.get('span2_len', 52))
    ema_len   = int(p.get('ema_len', 200))
    rsi_len   = int(p.get('rsi_len', 14))
    rsi_ob    = float(p.get('rsi_ob', 70.0))

    tenkan = (df['high'].rolling(conv_len).max() + df['low'].rolling(conv_len).min()) / 2
    kijun  = (df['high'].rolling(base_len).max() + df['low'].rolling(base_len).min()) / 2
    spanA  = (tenkan + kijun) / 2
    spanB  = (df['high'].rolling(span2_len).max() + df['low'].rolling(span2_len).min()) / 2
    cloud_top = pd.concat([spanA, spanB], axis=1).max(axis=1)

    ema_v = _ema(df['close'], ema_len)
    rsi_v = _rsi(df['close'], rsi_len)

    # crossover(tenkan, kijun)
    tk_cross_up = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    long_cond = tk_cross_up & (df['close'] > cloud_top) & (df['close'] > ema_v) & (rsi_v < rsi_ob)

    # exit: crossunder(tenkan, kijun)
    tk_cross_dn = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))

    sig = pd.Series(0, index=df.index)
    in_trade = False
    for i in range(len(df)):
        if long_cond.iloc[i]:
            in_trade = True
        if tk_cross_dn.iloc[i]:
            in_trade = False
        if in_trade:
            sig.iloc[i] = 1
    return sig

def space_Ichimoku_EMA_RSI_Long(trial):
    return {
        'conv_len':  trial.suggest_int('conv_len', 5, 15),
        'base_len':  trial.suggest_int('base_len', 15, 40),
        'span2_len': trial.suggest_int('span2_len', 40, 80),
        'ema_len':   trial.suggest_int('ema_len', 100, 300),
        'rsi_len':   trial.suggest_int('rsi_len', 7, 21),
        'rsi_ob':    trial.suggest_float('rsi_ob', 60.0, 85.0),
    }


# ─────────────────────────────────────────────
# 4. BB_Split_Limit
# ─────────────────────────────────────────────
def gen_BB_Split_Limit(df, **p):
    bb_len  = int(p.get('bb_len', 20))
    bb_mult = float(p.get('bb_mult', 2.0))
    ema_len = int(p.get('ema_len', 50))

    basis = _sma(df['close'], bb_len)
    std   = df['close'].rolling(bb_len).std()
    lower = basis - bb_mult * std
    ema   = _ema(df['close'], ema_len)

    # crossunder(close, lower) and close > ema
    cross_dn = (df['close'] < lower) & (df['close'].shift(1) >= lower.shift(1))
    long_cond = cross_dn & (df['close'] > ema)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    return sig

def space_BB_Split_Limit(trial):
    return {
        'bb_len':  trial.suggest_int('bb_len', 10, 40),
        'bb_mult': trial.suggest_float('bb_mult', 1.0, 3.0),
        'ema_len': trial.suggest_int('ema_len', 20, 100),
    }


# ─────────────────────────────────────────────
# 5. HA_Color_Flip
# ─────────────────────────────────────────────
def gen_HA_Color_Flip(df, **p):
    ema_len = int(p.get('ema_len', 50))
    smooth  = int(p.get('smooth', 1))

    # Heikin Ashi close
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # Heikin Ashi open (stateful)
    ha_open_vals = np.empty(len(df))
    ha_open_vals[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open_vals[i] = (ha_open_vals[i-1] + ha_close.iloc[i-1]) / 2
    ha_open = pd.Series(ha_open_vals, index=df.index)

    s_close = _sma(ha_close, smooth) if smooth > 1 else ha_close
    s_open  = _sma(ha_open, smooth) if smooth > 1 else ha_open
    ema     = _ema(df['close'], ema_len)

    ha_bull = s_close > s_open
    ha_bear = s_close < s_open

    # Flip: color change (bearish → bullish)
    prev_bull  = ha_bull.shift(1).fillna(value=False)
    long_entry = ha_bull & (~prev_bull.astype(bool)) & (df['close'] > ema)

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_HA_Color_Flip(trial):
    return {
        'ema_len': trial.suggest_int('ema_len', 20, 100),
        'smooth':  trial.suggest_int('smooth', 1, 5),
    }


# ─────────────────────────────────────────────
# 6. ActionZone_ATR
# ─────────────────────────────────────────────
def gen_ActionZone_ATR(df, **p):
    fast_len = int(p.get('fast_len', 9))
    slow_len = int(p.get('slow_len', 21))

    ema1 = _ema(df['close'], fast_len)
    ema2 = _ema(df['close'], slow_len)

    # crossover(ema1, ema2) and not in_zone
    cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
    in_zone  = (df['close'] > pd.concat([ema1, ema2], axis=1).min(axis=1)) & \
               (df['close'] < pd.concat([ema1, ema2], axis=1).max(axis=1))
    long_cond = cross_up & (~in_zone)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    return sig

def space_ActionZone_ATR(trial):
    return {
        'fast_len': trial.suggest_int('fast_len', 5, 20),
        'slow_len': trial.suggest_int('slow_len', 15, 50),
    }


# ─────────────────────────────────────────────
# 7. Donchian_Simple
# ─────────────────────────────────────────────
def gen_Donchian_Simple(df, **p):
    dc_len = int(p.get('dc_len', 20))

    # Pine v4: high[1] means shifted by 1 before rolling
    upper = df['high'].shift(1).rolling(dc_len).max()

    # crossover(close, upper)
    cross_up = (df['close'] > upper) & (df['close'].shift(1) <= upper.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[cross_up] = 1
    return sig

def space_Donchian_Simple(trial):
    return {
        'dc_len': trial.suggest_int('dc_len', 10, 50),
    }


# ─────────────────────────────────────────────
# 8. Gold_Scalp_RSI_Div
# ─────────────────────────────────────────────
def gen_Gold_Scalp_RSI_Div(df, **p):
    ema_fast = int(p.get('ema_fast', 9))
    ema_slow = int(p.get('ema_slow', 21))
    rsi_len  = int(p.get('rsi_len', 14))
    rsi_ob   = float(p.get('rsi_ob', 70.0))

    ema1 = _ema(df['close'], ema_fast)
    ema2 = _ema(df['close'], ema_slow)
    rsi  = _rsi(df['close'], rsi_len)
    c    = df['close']

    # Bullish divergence: price makes local low (lower than neighbors) while
    # RSI is rising from oversold territory — practical interpretation that
    # generates tradeable signals on 4h crypto data
    rsi_os = float(p.get('rsi_os', 40.0))
    # price local low: lower than bar before AND bar after (pivot low)
    price_pivot_low = (c < c.shift(1)) & (c < c.shift(-1))
    # rsi recovering: rsi crossed above rsi_os recently (within 2 bars)
    rsi_recovering  = (rsi > rsi_os) & (rsi.shift(1) <= rsi_os)
    # OR: price is making new local low while rsi is above its 3-bar minimum
    rsi_above_min   = rsi > rsi.rolling(3).min().shift(1)
    bull_div = (price_pivot_low & rsi_above_min) | rsi_recovering

    long_cond = bull_div & (ema1 > ema2) & (rsi < rsi_ob)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    return sig

def space_Gold_Scalp_RSI_Div(trial):
    return {
        'ema_fast': trial.suggest_int('ema_fast', 5, 20),
        'ema_slow': trial.suggest_int('ema_slow', 15, 50),
        'rsi_len':  trial.suggest_int('rsi_len', 7, 21),
        'rsi_ob':   trial.suggest_float('rsi_ob', 60.0, 85.0),
        'rsi_os':   trial.suggest_float('rsi_os', 25.0, 55.0),
    }


# ─────────────────────────────────────────────
# 9. Keltner_BTC
# ─────────────────────────────────────────────
def gen_Keltner_BTC(df, **p):
    kc_len  = int(p.get('kc_len', 20))
    kc_mult = float(p.get('kc_mult', 2.0))
    ema_len = int(p.get('ema_len', 50))

    ema_m = _ema(df['close'], kc_len)
    atr_v = _atr(df, kc_len)
    upper = ema_m + kc_mult * atr_v
    ema_f = _ema(df['close'], ema_len)

    # crossover(close, upper) and close > ema_f
    cross_up  = (df['close'] > upper) & (df['close'].shift(1) <= upper.shift(1))
    long_cond = cross_up & (df['close'] > ema_f)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    return sig

def space_Keltner_BTC(trial):
    return {
        'kc_len':  trial.suggest_int('kc_len', 10, 40),
        'kc_mult': trial.suggest_float('kc_mult', 1.0, 3.0),
        'ema_len': trial.suggest_int('ema_len', 20, 100),
    }


# ─────────────────────────────────────────────
# 10. Williams_Fractals_V4
# ─────────────────────────────────────────────
def gen_Williams_Fractals_V4(df, **p):
    n       = int(p.get('n', 2))
    ema_len = int(p.get('ema_len', 50))

    high  = df['high']
    low   = df['low']
    close = df['close']
    ema_v = _ema(close, ema_len)

    window = 2 * n + 1

    # upFractal: high[n] == highest(high, window) — in Pine v4, high[n] is n bars ago
    # So fractal fires when the bar n bars back is the highest in the window
    roll_high = high.rolling(window).max()
    # high.shift(n) is the value n bars ago
    up_fractal = high.shift(n) == roll_high

    # downFractal: low[n] == lowest(low, window)
    roll_low   = low.rolling(window).min()
    down_fractal = low.shift(n) == roll_low

    # Resistance/support carry-forward
    res = pd.Series(np.nan, index=df.index)
    sup = pd.Series(np.nan, index=df.index)
    last_res = np.nan
    last_sup = np.nan
    for i in range(len(df)):
        if up_fractal.iloc[i]:
            last_res = high.shift(n).iloc[i]
        if down_fractal.iloc[i]:
            last_sup = low.shift(n).iloc[i]
        res.iloc[i] = last_res
        sup.iloc[i] = last_sup

    # crossover(close, res) and close > ema_v
    cross_up  = (close > res) & (close.shift(1) <= res.shift(1))
    long_cond = cross_up & (~res.isna()) & (close > ema_v)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    return sig

def space_Williams_Fractals_V4(trial):
    return {
        'n':       trial.suggest_int('n', 1, 5),
        'ema_len': trial.suggest_int('ema_len', 20, 100),
    }


# ─────────────────────────────────────────────
# 11. RSI_CCI_EMA_Daytrade
# ─────────────────────────────────────────────
def gen_RSI_CCI_EMA_Daytrade(df, **p):
    ema_fast = int(p.get('ema_fast', 4))
    ema_slow = int(p.get('ema_slow', 8))
    rsi_len  = int(p.get('rsi_len', 14))
    rsi_os   = float(p.get('rsi_os', 30.0))
    cci_len  = int(p.get('cci_len', 20))
    cci_os   = float(p.get('cci_os', -100.0))

    ema1 = _ema(df['close'], ema_fast)
    ema2 = _ema(df['close'], ema_slow)
    rsi  = _rsi(df['close'], rsi_len)

    tp  = (df['high'] + df['low'] + df['close']) / 3
    tp_sma = _sma(tp, cci_len)
    # CCI with mean absolute deviation
    mad = tp.rolling(cci_len).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - tp_sma) / (0.015 * mad + 1e-9)

    # crossover(ema1, ema2) and rsi > rsi_os and cci > cci_os
    cross_up  = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
    long_cond = cross_up & (rsi > rsi_os) & (cci > cci_os)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    return sig

def space_RSI_CCI_EMA_Daytrade(trial):
    return {
        'ema_fast': trial.suggest_int('ema_fast', 2, 10),
        'ema_slow': trial.suggest_int('ema_slow', 6, 20),
        'rsi_len':  trial.suggest_int('rsi_len', 7, 21),
        'rsi_os':   trial.suggest_float('rsi_os', 20.0, 45.0),
        'cci_len':  trial.suggest_int('cci_len', 10, 30),
        'cci_os':   trial.suggest_float('cci_os', -150.0, -50.0),
    }


# ─────────────────────────────────────────────
# 12. OBV_AccDist
# ─────────────────────────────────────────────
def gen_OBV_AccDist(df, **p):
    obv_len = int(p.get('obv_len', 20))
    adl_len = int(p.get('adl_len', 20))

    close  = df['close']
    volume = df['volume']
    high   = df['high']
    low    = df['low']

    # OBV via cumsum
    direction = np.where(close > close.shift(1), 1,
                np.where(close < close.shift(1), -1, 0))
    obv_v  = pd.Series((direction * volume.values).cumsum(), index=df.index)
    obv_ma = _sma(obv_v, obv_len)

    # Accumulation/Distribution Line
    clv   = ((close - low) - (high - close)) / (high - low + 1e-9)
    adl_v = (clv * volume).cumsum()
    adl_ma = _sma(adl_v, adl_len)

    # crossover(obv_v, obv_ma) and crossover(adl_v, adl_ma)
    obv_cross = (obv_v > obv_ma) & (obv_v.shift(1) <= obv_ma.shift(1))
    adl_cross = (adl_v > adl_ma) & (adl_v.shift(1) <= adl_ma.shift(1))
    long_cond = obv_cross & adl_cross

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    return sig

def space_OBV_AccDist(trial):
    return {
        'obv_len': trial.suggest_int('obv_len', 5, 50),
        'adl_len': trial.suggest_int('adl_len', 5, 50),
    }


# ─────────────────────────────────────────────
# 13. Time_DayTrade
# ─────────────────────────────────────────────
def gen_Time_DayTrade(df, **p):
    ema_fast = int(p.get('ema_fast', 9))
    ema_slow = int(p.get('ema_slow', 21))
    rsi_len  = int(p.get('rsi_len', 14))
    rsi_ob   = float(p.get('rsi_ob', 70.0))

    ema1 = _ema(df['close'], ema_fast)
    ema2 = _ema(df['close'], ema_slow)
    rsi  = _rsi(df['close'], rsi_len)

    # crossover(ema1, ema2) and rsi < rsi_ob  (no time filter — crypto 24/7)
    cross_up  = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
    long_cond = cross_up & (rsi < rsi_ob)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    return sig

def space_Time_DayTrade(trial):
    return {
        'ema_fast': trial.suggest_int('ema_fast', 5, 20),
        'ema_slow': trial.suggest_int('ema_slow', 15, 50),
        'rsi_len':  trial.suggest_int('rsi_len', 7, 21),
        'rsi_ob':   trial.suggest_float('rsi_ob', 60.0, 85.0),
    }


# ─────────────────────────────────────────────
# STRATEGY_EXPORT
# ─────────────────────────────────────────────
STRATEGY_EXPORT = {
    'BB_Strategy_V2': {
        'gen': gen_BB_Strategy_V2,
        'space': space_BB_Strategy_V2,
        'default_params': {'bb_len': 20, 'bb_mult': 2.0, 'rsi_len': 14, 'rsi_ob': 65.0},
        'info': {'likes': 546, 'version': 'v5', 'source': 'Bollinger_Bands_Strategy'},
    },
    'MACD_Aspray': {
        'gen': gen_MACD_Aspray,
        'space': space_MACD_Aspray,
        'default_params': {'fast_len': 12, 'slow_len': 26, 'signal_len': 9, 'hist_thresh': 0.0},
        'info': {'likes': 546, 'version': 'v5', 'source': 'MACD_Aspray_Hybrid_Strategy'},
    },
    'Ichimoku_EMA_RSI_Long': {
        'gen': gen_Ichimoku_EMA_RSI_Long,
        'space': space_Ichimoku_EMA_RSI_Long,
        'default_params': {'conv_len': 9, 'base_len': 26, 'span2_len': 52, 'ema_len': 200, 'rsi_len': 14, 'rsi_ob': 70.0},
        'info': {'likes': 542, 'version': 'v4', 'source': 'Ichimoku_EMA_RSI___Crypto_only_long'},
    },
    'BB_Split_Limit': {
        'gen': gen_BB_Split_Limit,
        'space': space_BB_Split_Limit,
        'default_params': {'bb_len': 20, 'bb_mult': 2.0, 'ema_len': 50},
        'info': {'likes': 525, 'version': 'v5', 'source': 'Bollinger_Band_strategy_with_split'},
    },
    'HA_Color_Flip': {
        'gen': gen_HA_Color_Flip,
        'space': space_HA_Color_Flip,
        'default_params': {'ema_len': 50, 'smooth': 1},
        'info': {'likes': 520, 'version': 'v5', 'source': 'Heikin_Ashi_Color_Flip_Strategy'},
    },
    'ActionZone_ATR': {
        'gen': gen_ActionZone_ATR,
        'space': space_ActionZone_ATR,
        'default_params': {'fast_len': 9, 'slow_len': 21},
        'info': {'likes': 510, 'version': 'v5', 'source': 'action_zone___ATR_stop_reverse_orde'},
    },
    'Donchian_Simple': {
        'gen': gen_Donchian_Simple,
        'space': space_Donchian_Simple,
        'default_params': {'dc_len': 20},
        'info': {'likes': 508, 'version': 'v4', 'source': 'Donchian_Channel_Strategy'},
    },
    'Gold_Scalp_RSI_Div': {
        'gen': gen_Gold_Scalp_RSI_Div,
        'space': space_Gold_Scalp_RSI_Div,
        'default_params': {'ema_fast': 9, 'ema_slow': 21, 'rsi_len': 14, 'rsi_ob': 70.0, 'rsi_os': 40.0},
        'info': {'likes': 500, 'version': 'v5', 'source': 'Advanced_Gold_Scalping_Strategy_wit'},
    },
    'Keltner_BTC': {
        'gen': gen_Keltner_BTC,
        'space': space_Keltner_BTC,
        'default_params': {'kc_len': 20, 'kc_mult': 2.0, 'ema_len': 50},
        'info': {'likes': 499, 'version': 'v4', 'source': 'Optimized_Keltner_Channels_SL_TP_St'},
    },
    'Williams_Fractals_V4': {
        'gen': gen_Williams_Fractals_V4,
        'space': space_Williams_Fractals_V4,
        'default_params': {'n': 2, 'ema_len': 50},
        'info': {'likes': 497, 'version': 'v4', 'source': 'WMX_Williams_Fractals_strategy_V4'},
    },
    'RSI_CCI_EMA_Daytrade': {
        'gen': gen_RSI_CCI_EMA_Daytrade,
        'space': space_RSI_CCI_EMA_Daytrade,
        'default_params': {'ema_fast': 4, 'ema_slow': 8, 'rsi_len': 14, 'rsi_os': 30.0, 'cci_len': 20, 'cci_os': -100.0},
        'info': {'likes': 495, 'version': 'v4', 'source': 'Daytrade_strategy_RSI_CCI_EMA_4_8'},
    },
    'OBV_AccDist': {
        'gen': gen_OBV_AccDist,
        'space': space_OBV_AccDist,
        'default_params': {'obv_len': 20, 'adl_len': 20},
        'info': {'likes': 492, 'version': 'v4', 'source': 'OBV_Accumulation___Distribution_Str'},
    },
    'Time_DayTrade': {
        'gen': gen_Time_DayTrade,
        'space': space_Time_DayTrade,
        'default_params': {'ema_fast': 9, 'ema_slow': 21, 'rsi_len': 14, 'rsi_ob': 70.0},
        'info': {'likes': 479, 'version': 'v5', 'source': 'Time_Based_Crypto_DayTrade_Strategy'},
    },
}
