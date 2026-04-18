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
# 1. Ichimoku_QQE
# ─────────────────────────────────────────────
def gen_Ichimoku_QQE(df, conversion=9, base=26, lagging=52,
                     rsi_period=14, sf=5, qqe_f=4.238, **kwargs):
    close = df['close']
    high  = df['high']
    low   = df['low']

    def donchian(n):
        return (high.rolling(n).max() + low.rolling(n).min()) / 2

    tenkan = donchian(conversion)
    kijun  = donchian(base)
    spanA  = (tenkan + kijun) / 2
    spanB  = donchian(lagging)

    cloud_top = pd.concat([spanA, spanB], axis=1).max(axis=1)
    cloud_bot = pd.concat([spanA, spanB], axis=1).min(axis=1)

    # QQE
    rsi_v = _rsi(close, rsi_period)
    rsi_s = _ema(rsi_v, sf)
    delta = rsi_s.diff().abs()
    atr_rsi = _ema(delta, sf)
    qqe_dn = rsi_s - qqe_f * atr_rsi
    qqe_up = rsi_s + qqe_f * atr_rsi

    # Trailing stop for QQE
    rsi_s_arr  = rsi_s.values
    qqe_dn_arr = qqe_dn.values
    qqe_up_arr = qqe_up.values
    ts_arr = np.zeros(len(df))
    for i in range(1, len(df)):
        if np.isnan(rsi_s_arr[i]) or np.isnan(qqe_dn_arr[i]):
            ts_arr[i] = ts_arr[i-1]
            continue
        if rsi_s_arr[i-1] > ts_arr[i-1]:
            ts_arr[i] = max(ts_arr[i-1], qqe_dn_arr[i])
        else:
            ts_arr[i] = min(ts_arr[i-1], qqe_up_arr[i])

    trail = pd.Series(ts_arr, index=df.index)
    qqe_bull = rsi_s > trail
    qqe_bear = rsi_s < trail

    above_cloud = close > cloud_top
    below_cloud = close < cloud_bot

    tk_cross      = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    tk_crossunder = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))

    long_cond  = tk_cross & above_cloud & qqe_bull
    short_cond = tk_crossunder & below_cloud & qqe_bear

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig

def space_Ichimoku_QQE(trial):
    return {
        'conversion':  trial.suggest_int('conversion', 7, 13),
        'base':        trial.suggest_int('base', 20, 34),
        'lagging':     trial.suggest_int('lagging', 44, 60),
        'rsi_period':  trial.suggest_int('rsi_period', 10, 20),
        'sf':          trial.suggest_int('sf', 3, 8),
        'qqe_f':       trial.suggest_float('qqe_f', 2.0, 6.0),
    }


# ─────────────────────────────────────────────
# 2. HA_PriceAction_Long
# ─────────────────────────────────────────────
def gen_HA_PriceAction_Long(df, **kwargs):
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    ha_open_arr = np.empty(len(df))
    ha_open_arr[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    hc = ha_close.values
    for i in range(1, len(df)):
        ha_open_arr[i] = (ha_open_arr[i-1] + hc[i-1]) / 2

    ha_open = pd.Series(ha_open_arr, index=df.index)

    ha_bull = ha_close > ha_open
    ha_bear = ha_close < ha_open

    ha_bull_prev = pd.Series(np.concatenate([[False], ha_bull.values[:-1]]), index=df.index)
    ha_bear_prev = pd.Series(np.concatenate([[False], ha_bear.values[:-1]]), index=df.index)
    long_entry = ha_bull & ~ha_bull_prev
    long_exit  = ha_bear & ~ha_bear_prev

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    sig[long_exit]  = 0   # close signals — use 0 (hold/close handled by bot)
    return sig

def space_HA_PriceAction_Long(trial):
    return {}   # No tunable params in this strategy


# ─────────────────────────────────────────────
# 3. HighLow_Channel_Swing
# ─────────────────────────────────────────────
def gen_HighLow_Channel_Swing(df, channel_len=20, ema_fast=9,
                               ema_slow=21, rsi_len=14, **kwargs):
    close = df['close']
    high  = df['high']
    low   = df['low']

    chan_hi  = high.rolling(channel_len).max()
    chan_lo  = low.rolling(channel_len).min()
    chan_mid = (chan_hi + chan_lo) / 2

    ema1 = _ema(close, ema_fast)
    ema2 = _ema(close, ema_slow)
    rsi_v = _rsi(close, rsi_len)

    long_cond  = (close > chan_mid) & (close.shift(1) <= chan_mid.shift(1)) & (ema1 > ema2) & (rsi_v > 50)
    short_cond = (close < chan_mid) & (close.shift(1) >= chan_mid.shift(1)) & (ema1 < ema2) & (rsi_v < 50)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig

def space_HighLow_Channel_Swing(trial):
    return {
        'channel_len': trial.suggest_int('channel_len', 10, 40),
        'ema_fast':    trial.suggest_int('ema_fast', 5, 20),
        'ema_slow':    trial.suggest_int('ema_slow', 15, 50),
        'rsi_len':     trial.suggest_int('rsi_len', 10, 20),
    }


# ─────────────────────────────────────────────
# 4. ATR_Trailing_SL
# ─────────────────────────────────────────────
def gen_ATR_Trailing_SL(df, atr_period=14, atr_mult=3.0, ema_len=20, **kwargs):
    close = df['close']
    atr_v = _atr(df, atr_period)
    ema_v = _ema(close, ema_len)

    close_arr = close.values
    atr_arr   = atr_v.values
    ema_arr   = ema_v.values

    trail_arr = np.full(len(df), np.nan)
    # Find first valid atr index
    first = np.argmax(~np.isnan(atr_arr))
    trail_arr[first] = close_arr[first] - atr_mult * atr_arr[first]

    for i in range(first + 1, len(df)):
        if np.isnan(atr_arr[i]):
            trail_arr[i] = trail_arr[i-1]
            continue
        if close_arr[i] > ema_arr[i]:
            trail_arr[i] = max(trail_arr[i-1], close_arr[i] - atr_mult * atr_arr[i])
        else:
            trail_arr[i] = min(trail_arr[i-1], close_arr[i] + atr_mult * atr_arr[i])

    trail = pd.Series(trail_arr, index=df.index)

    long_cond  = (close > trail) & (close.shift(1) <= trail.shift(1))
    short_cond = (close < trail) & (close.shift(1) >= trail.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig

def space_ATR_Trailing_SL(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 10, 20),
        'atr_mult':   trial.suggest_float('atr_mult', 1.5, 5.0),
        'ema_len':    trial.suggest_int('ema_len', 10, 50),
    }


# ─────────────────────────────────────────────
# 5. Fractal_Proximity_Scalp
# ─────────────────────────────────────────────
def gen_Fractal_Proximity_Scalp(df, fractal_len=5, proximity=0.005,
                                  ema_fast=10, ema_slow=50, ema_trend=200, **kwargs):
    close = df['close']
    high  = df['high']
    low   = df['low']

    win = 2 * fractal_len + 1

    # Fractal: bar[fractal_len] is highest/lowest in window
    roll_hi = high.rolling(win, center=True).max()
    roll_lo = low.rolling(win, center=True).min()

    up_fractal   = high == roll_hi
    down_fractal = low  == roll_lo

    # Track last fractal levels (shift by fractal_len to avoid lookahead)
    up_fractal_shifted   = up_fractal.shift(fractal_len)
    down_fractal_shifted = down_fractal.shift(fractal_len)
    high_shifted = high.shift(fractal_len)
    low_shifted  = low.shift(fractal_len)

    last_up   = high_shifted.where(up_fractal_shifted).ffill()
    last_down = low_shifted.where(down_fractal_shifted).ffill()

    ema1 = _ema(close, ema_fast)
    ema2 = _ema(close, ema_slow)
    ema3 = _ema(close, ema_trend)

    near_down = (last_down.notna() &
                 (close >= last_down * (1 - proximity)) &
                 (close <= last_down * (1 + proximity)))
    near_up   = (last_up.notna() &
                 (close >= last_up * (1 - proximity)) &
                 (close <= last_up * (1 + proximity)))

    long_cond  = near_down & (ema1 > ema2) & (ema2 > ema3)
    short_cond = near_up   & (ema1 < ema2) & (ema2 < ema3)

    # Edge trigger: only on new condition
    long_prev  = pd.Series(np.concatenate([[False], long_cond.values[:-1]]),  index=df.index)
    short_prev = pd.Series(np.concatenate([[False], short_cond.values[:-1]]), index=df.index)
    long_sig  = long_cond  & ~long_prev
    short_sig = short_cond & ~short_prev

    sig = pd.Series(0, index=df.index)
    sig[long_sig]  =  1
    sig[short_sig] = -1
    return sig

def space_Fractal_Proximity_Scalp(trial):
    return {
        'fractal_len': trial.suggest_int('fractal_len', 3, 8),
        'proximity':   trial.suggest_float('proximity', 0.002, 0.02),
        'ema_fast':    trial.suggest_int('ema_fast', 5, 20),
        'ema_slow':    trial.suggest_int('ema_slow', 30, 80),
        'ema_trend':   trial.suggest_int('ema_trend', 150, 300),
    }


# ─────────────────────────────────────────────
# 6. BB_RSI_MACD
# ─────────────────────────────────────────────
def gen_BB_RSI_MACD(df, bb_len=20, bb_mult=2.0,
                    rsi_len=14, rsi_ob=70, rsi_os=30,
                    fast_len=12, slow_len=26, signal_len=9, **kwargs):
    close = df['close']

    basis, upper, lower = _bb(close, bb_len, bb_mult)
    rsi_v     = _rsi(close, rsi_len)
    macd_line = _ema(close, fast_len) - _ema(close, slow_len)
    signal    = _ema(macd_line, signal_len)

    long_cond  = (close <= lower) & (rsi_v < rsi_os) & (macd_line > signal)
    short_cond = (close >= upper) & (rsi_v > rsi_ob) & (macd_line < signal)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig

def space_BB_RSI_MACD(trial):
    return {
        'bb_len':     trial.suggest_int('bb_len', 15, 30),
        'bb_mult':    trial.suggest_float('bb_mult', 1.5, 3.0),
        'rsi_len':    trial.suggest_int('rsi_len', 10, 20),
        'rsi_ob':     trial.suggest_int('rsi_ob', 65, 80),
        'rsi_os':     trial.suggest_int('rsi_os', 20, 35),
        'fast_len':   trial.suggest_int('fast_len', 8, 16),
        'slow_len':   trial.suggest_int('slow_len', 20, 34),
        'signal_len': trial.suggest_int('signal_len', 7, 12),
    }


# ─────────────────────────────────────────────
# 7. Noro_MA_ATR_V2
# ─────────────────────────────────────────────
def gen_Noro_MA_ATR_V2(df, ma_len=20, atr_mult=2.0,
                        use_rsi=True, rsi_len=14, **kwargs):
    close = df['close']

    ma    = _sma(close, ma_len)
    atr_v = _atr(df, 14)

    upper = ma + atr_mult * atr_v
    lower = ma - atr_mult * atr_v

    rsi_v = _rsi(close, rsi_len)

    long_cond  = (close > lower) & (close.shift(1) <= lower.shift(1))
    short_cond = (close < upper) & (close.shift(1) >= upper.shift(1))

    if use_rsi:
        long_cond  = long_cond  & (rsi_v < 50)
        short_cond = short_cond & (rsi_v > 50)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig

def space_Noro_MA_ATR_V2(trial):
    return {
        'ma_len':   trial.suggest_int('ma_len', 10, 50),
        'atr_mult': trial.suggest_float('atr_mult', 1.0, 4.0),
        'use_rsi':  trial.suggest_categorical('use_rsi', [True, False]),
        'rsi_len':  trial.suggest_int('rsi_len', 10, 20),
    }


# ─────────────────────────────────────────────
# 8. KAMA_TradeDots_V2
# ─────────────────────────────────────────────
def _kama(src, length, fast, slow):
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    src_arr = src.values
    kama_arr = np.full(len(src), np.nan)

    # Find first valid bar
    first = length  # need `length` bars of history
    kama_arr[first] = src_arr[first]

    for i in range(first + 1, len(src)):
        direction = abs(src_arr[i] - src_arr[i - length])
        volatility = 0.0
        for j in range(1, length + 1):
            if i - j + 1 < len(src) and i - j >= 0:
                volatility += abs(src_arr[i - j + 1] - src_arr[i - j])
        er  = direction / volatility if volatility != 0 else 0
        sc  = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_arr[i] = kama_arr[i-1] + sc * (src_arr[i] - kama_arr[i-1])

    return pd.Series(kama_arr, index=src.index)

def gen_KAMA_TradeDots_V2(df, kama_length=14, fast_period=2,
                           slow_period=30, ema_len=50, **kwargs):
    close     = df['close']
    ema_trend = _ema(close, ema_len)
    kama      = _kama(close, kama_length, fast_period, slow_period)

    long_entry  = (close > kama) & (close.shift(1) <= kama.shift(1)) & (close > ema_trend)
    short_entry = (close < kama) & (close.shift(1) >= kama.shift(1)) & (close < ema_trend)

    sig = pd.Series(0, index=df.index)
    sig[long_entry]  =  1
    sig[short_entry] = -1
    return sig

def space_KAMA_TradeDots_V2(trial):
    return {
        'kama_length':  trial.suggest_int('kama_length', 8, 24),
        'fast_period':  trial.suggest_int('fast_period', 2, 5),
        'slow_period':  trial.suggest_int('slow_period', 20, 50),
        'ema_len':      trial.suggest_int('ema_len', 30, 100),
    }


# ─────────────────────────────────────────────
# 9. EMA_SMA_RSI_MACD
# ─────────────────────────────────────────────
def gen_EMA_SMA_RSI_MACD(df, ema_len=20, sma_len=50,
                          rsi_len=14, rsi_ob=70, rsi_os=30,
                          fast_m=12, slow_m=26, signal_m=9, **kwargs):
    close = df['close']

    ema_v  = _ema(close, ema_len)
    sma_v  = _sma(close, sma_len)
    rsi_v  = _rsi(close, rsi_len)
    macd_l = _ema(close, fast_m) - _ema(close, slow_m)
    sig_l  = _ema(macd_l, signal_m)

    macd_cross_up   = (macd_l > sig_l) & (macd_l.shift(1) <= sig_l.shift(1))
    macd_cross_down = (macd_l < sig_l) & (macd_l.shift(1) >= sig_l.shift(1))

    long_cond  = (ema_v > sma_v) & (rsi_v < rsi_ob) & macd_cross_up
    short_cond = (ema_v < sma_v) & (rsi_v > rsi_os) & macd_cross_down

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig

def space_EMA_SMA_RSI_MACD(trial):
    return {
        'ema_len':  trial.suggest_int('ema_len', 10, 30),
        'sma_len':  trial.suggest_int('sma_len', 30, 80),
        'rsi_len':  trial.suggest_int('rsi_len', 10, 20),
        'rsi_ob':   trial.suggest_int('rsi_ob', 65, 80),
        'rsi_os':   trial.suggest_int('rsi_os', 20, 35),
        'fast_m':   trial.suggest_int('fast_m', 8, 16),
        'slow_m':   trial.suggest_int('slow_m', 20, 34),
        'signal_m': trial.suggest_int('signal_m', 7, 12),
    }


# ─────────────────────────────────────────────
# STRATEGY EXPORT
# ─────────────────────────────────────────────
STRATEGY_EXPORT = {
    'Ichimoku_QQE': {
        'gen': gen_Ichimoku_QQE,
        'space': space_Ichimoku_QQE,
        'default_params': {
            'conversion': 9, 'base': 26, 'lagging': 52,
            'rsi_period': 14, 'sf': 5, 'qqe_f': 4.238,
        },
        'info': {'likes': 800, 'version': 'v4', 'source': 'TV_Ichimoku_QQE'},
    },
    'HA_PriceAction_Long': {
        'gen': gen_HA_PriceAction_Long,
        'space': space_HA_PriceAction_Long,
        'default_params': {},
        'info': {'likes': 796, 'version': 'v4', 'source': 'TV_HA_PriceAction_Long'},
    },
    'HighLow_Channel_Swing': {
        'gen': gen_HighLow_Channel_Swing,
        'space': space_HighLow_Channel_Swing,
        'default_params': {
            'channel_len': 20, 'ema_fast': 9, 'ema_slow': 21, 'rsi_len': 14,
        },
        'info': {'likes': 751, 'version': 'v4', 'source': 'TV_HighLow_Channel_Swing'},
    },
    'ATR_Trailing_SL': {
        'gen': gen_ATR_Trailing_SL,
        'space': space_ATR_Trailing_SL,
        'default_params': {'atr_period': 14, 'atr_mult': 3.0, 'ema_len': 20},
        'info': {'likes': 745, 'version': 'v4', 'source': 'TV_ATR_Trailing_SL'},
    },
    'Fractal_Proximity_Scalp': {
        'gen': gen_Fractal_Proximity_Scalp,
        'space': space_Fractal_Proximity_Scalp,
        'default_params': {
            'fractal_len': 5, 'proximity': 0.005,
            'ema_fast': 10, 'ema_slow': 50, 'ema_trend': 200,
        },
        'info': {'likes': 738, 'version': 'v5', 'source': 'TV_Fractal_Proximity_Scalp'},
    },
    'BB_RSI_MACD': {
        'gen': gen_BB_RSI_MACD,
        'space': space_BB_RSI_MACD,
        'default_params': {
            'bb_len': 20, 'bb_mult': 2.0,
            'rsi_len': 14, 'rsi_ob': 70, 'rsi_os': 30,
            'fast_len': 12, 'slow_len': 26, 'signal_len': 9,
        },
        'info': {'likes': 728, 'version': 'v4', 'source': 'TV_BB_RSI_MACD'},
    },
    'Noro_MA_ATR_V2': {
        'gen': gen_Noro_MA_ATR_V2,
        'space': space_Noro_MA_ATR_V2,
        'default_params': {
            'ma_len': 20, 'atr_mult': 2.0, 'use_rsi': True, 'rsi_len': 14,
        },
        'info': {'likes': 707, 'version': 'v4', 'source': 'TV_Noro_MA_ATR_V2'},
    },
    'KAMA_TradeDots_V2': {
        'gen': gen_KAMA_TradeDots_V2,
        'space': space_KAMA_TradeDots_V2,
        'default_params': {
            'kama_length': 14, 'fast_period': 2, 'slow_period': 30, 'ema_len': 50,
        },
        'info': {'likes': 704, 'version': 'v5', 'source': 'TV_KAMA_TradeDots_V2'},
    },
    'EMA_SMA_RSI_MACD': {
        'gen': gen_EMA_SMA_RSI_MACD,
        'space': space_EMA_SMA_RSI_MACD,
        'default_params': {
            'ema_len': 20, 'sma_len': 50,
            'rsi_len': 14, 'rsi_ob': 70, 'rsi_os': 30,
            'fast_m': 12, 'slow_m': 26, 'signal_m': 9,
        },
        'info': {'likes': 693, 'version': 'v4', 'source': 'TV_EMA_SMA_RSI_MACD'},
    },
}
