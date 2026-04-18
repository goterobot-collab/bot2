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


# ─────────────────────────────────────────────────────────────
# 1. HL_Breakout_ATR
# ─────────────────────────────────────────────────────────────
def gen_HL_Breakout_ATR(df, dc_len=20, atr_len=14, atr_mult=2.0):
    close  = df['close']
    high   = df['high']
    low    = df['low']

    upper  = high.rolling(dc_len).max().shift(1)
    atr    = _atr(df, atr_len)

    # Long entry: close crosses above Donchian upper band
    long_entry  = (close > upper) & (close.shift(1) <= upper.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_HL_Breakout_ATR(trial):
    return {
        'dc_len':   trial.suggest_int('dc_len', 10, 50),
        'atr_len':  trial.suggest_int('atr_len', 7, 21),
        'atr_mult': trial.suggest_float('atr_mult', 1.0, 4.0),
    }


# ─────────────────────────────────────────────────────────────
# 2. BB_Fib_Ratios
# ─────────────────────────────────────────────────────────────
def gen_BB_Fib_Ratios(df, bb_len=20, fib1_mult=1.618, fib2_mult=2.618, rsi_len=14):
    close  = df['close']
    basis  = _sma(close, bb_len)
    dev    = close.rolling(bb_len).std()
    lower1 = basis - fib1_mult * dev
    rsi    = _rsi(close, rsi_len)

    # Buy near lower Fib band: close < lower1 AND rsi < 40, new signal only
    long_cond  = (close < lower1) & (rsi < 40)
    long_entry = long_cond & ~long_cond.shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_BB_Fib_Ratios(trial):
    return {
        'bb_len':    trial.suggest_int('bb_len', 10, 50),
        'fib1_mult': trial.suggest_float('fib1_mult', 1.0, 2.5),
        'fib2_mult': trial.suggest_float('fib2_mult', 2.0, 4.0),
        'rsi_len':   trial.suggest_int('rsi_len', 7, 21),
    }


# ─────────────────────────────────────────────────────────────
# 3. Low_Scanner_Crypto
# ─────────────────────────────────────────────────────────────
def gen_Low_Scanner_Crypto(df, length=14, rsi_os=30, bb_len=20, bb_mult=2.0):
    close  = df['close']
    rsi_v  = _rsi(close, length)
    basis  = _sma(close, bb_len)
    dev    = bb_mult * close.rolling(bb_len).std()
    lower  = basis - dev

    # Long: RSI oversold + price at lower BB, new signal only
    long_cond  = (rsi_v < rsi_os) & (close < lower)
    long_entry = long_cond & ~long_cond.shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_Low_Scanner_Crypto(trial):
    return {
        'length':   trial.suggest_int('length', 7, 21),
        'rsi_os':   trial.suggest_int('rsi_os', 20, 40),
        'bb_len':   trial.suggest_int('bb_len', 10, 50),
        'bb_mult':  trial.suggest_float('bb_mult', 1.5, 3.0),
    }


# ─────────────────────────────────────────────────────────────
# 4. MACD_EMA200
# ─────────────────────────────────────────────────────────────
def gen_MACD_EMA200(df, fast_len=12, slow_len=26, signal_len=9, ema_len=200):
    close  = df['close']
    macd   = _ema(close, fast_len) - _ema(close, slow_len)
    sig_   = _ema(macd, signal_len)
    ema_t  = _ema(close, ema_len)

    # Crossover: macd crosses above signal AND price above EMA200
    cross_up   = (macd > sig_) & (macd.shift(1) <= sig_.shift(1))
    long_entry = cross_up & (close > ema_t)

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_MACD_EMA200(trial):
    return {
        'fast_len':   trial.suggest_int('fast_len', 5, 20),
        'slow_len':   trial.suggest_int('slow_len', 15, 50),
        'signal_len': trial.suggest_int('signal_len', 5, 15),
        'ema_len':    trial.suggest_int('ema_len', 100, 300),
    }


# ─────────────────────────────────────────────────────────────
# 5. Mean_Reversion_KL
# ─────────────────────────────────────────────────────────────
def gen_Mean_Reversion_KL(df, bb_len=20, bb_mult=2.0, rsi_len=14,
                           rsi_ob=65.0, rsi_os=35.0, atr_len=14,
                           sl_mult=1.5, tp_mult=2.5):
    close  = df['close']
    basis  = _sma(close, bb_len)
    dev    = bb_mult * close.rolling(bb_len).std()
    lower  = basis - dev
    rsi    = _rsi(close, rsi_len)

    # Long: close below lower BB and RSI oversold
    long_entry = (close < lower) & (rsi < rsi_os)

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_Mean_Reversion_KL(trial):
    return {
        'bb_len':   trial.suggest_int('bb_len', 10, 50),
        'bb_mult':  trial.suggest_float('bb_mult', 1.5, 3.0),
        'rsi_len':  trial.suggest_int('rsi_len', 7, 21),
        'rsi_ob':   trial.suggest_float('rsi_ob', 60.0, 80.0),
        'rsi_os':   trial.suggest_float('rsi_os', 20.0, 45.0),
        'atr_len':  trial.suggest_int('atr_len', 7, 21),
        'sl_mult':  trial.suggest_float('sl_mult', 0.5, 3.0),
        'tp_mult':  trial.suggest_float('tp_mult', 1.0, 5.0),
    }


# ─────────────────────────────────────────────────────────────
# 6. VWAP_Swing
# ─────────────────────────────────────────────────────────────
def gen_VWAP_Swing(df, ema_len=21, rsi_len=14, rsi_ob=70.0, rsi_os=30.0):
    close  = df['close']
    high   = df['high']
    low    = df['low']
    vol    = df['volume']

    tp     = (high + low + close) / 3
    vwap   = (tp * vol).cumsum() / vol.cumsum()
    ema_v  = _ema(close, ema_len)
    rsi_v  = _rsi(close, rsi_len)

    # Long: close crosses above VWAP AND close > EMA AND RSI not overbought
    cross_up   = (close > vwap) & (close.shift(1) <= vwap.shift(1))
    long_entry = cross_up & (close > ema_v) & (rsi_v < rsi_ob)

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_VWAP_Swing(trial):
    return {
        'ema_len': trial.suggest_int('ema_len', 10, 50),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'rsi_ob':  trial.suggest_float('rsi_ob', 60.0, 85.0),
        'rsi_os':  trial.suggest_float('rsi_os', 15.0, 40.0),
    }


# ─────────────────────────────────────────────────────────────
# 7. HA_Long_Term
# ─────────────────────────────────────────────────────────────
def gen_HA_Long_Term(df, ema_len=200, rsi_len=14):
    close  = df['close']
    open_  = df['open']
    high   = df['high']
    low    = df['low']

    # Heikin-Ashi calculation (stateful open)
    ha_close = (open_ + high + low + close) / 4
    ha_open  = ha_close.copy()
    ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

    ema_v  = _ema(close, ema_len)
    ha_bull = ha_close > ha_open

    # Long entry: HA turns bullish (was not bullish before) AND price above EMA
    long_entry = ha_bull & ~ha_bull.shift(1).fillna(False) & (close > ema_v)
    # Long exit: HA turns bearish
    long_exit  = ~ha_bull & ha_bull.shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    in_trade = False
    for i in range(len(df)):
        if long_entry.iloc[i] and not in_trade:
            sig.iloc[i] = 1
            in_trade = True
        elif long_exit.iloc[i] and in_trade:
            in_trade = False
        elif in_trade:
            sig.iloc[i] = 1
    return sig

def space_HA_Long_Term(trial):
    return {
        'ema_len': trial.suggest_int('ema_len', 50, 300),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
    }


# ─────────────────────────────────────────────────────────────
# 8. Trend_BB_Valente
# ─────────────────────────────────────────────────────────────
def gen_Trend_BB_Valente(df, bb_len=20, bb_mult=2.0, ema_len=50, atr_len=14):
    close  = df['close']
    basis  = _sma(close, bb_len)
    dev    = bb_mult * close.rolling(bb_len).std()
    ema_v  = _ema(close, ema_len)

    # Long: close crosses above BB basis AND price above EMA
    cross_up   = (close > basis) & (close.shift(1) <= basis.shift(1))
    long_entry = cross_up & (close > ema_v)

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_Trend_BB_Valente(trial):
    return {
        'bb_len':  trial.suggest_int('bb_len', 10, 50),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0),
        'ema_len': trial.suggest_int('ema_len', 20, 100),
        'atr_len': trial.suggest_int('atr_len', 7, 21),
    }


# ─────────────────────────────────────────────────────────────
# 9. MACD_200_SMA
# ─────────────────────────────────────────────────────────────
def gen_MACD_200_SMA(df, fast_len=12, slow_len=26, signal_len=9, sma_len=200):
    close  = df['close']
    macd   = _ema(close, fast_len) - _ema(close, slow_len)
    sig_   = _ema(macd, signal_len)
    sma_v  = _sma(close, sma_len)

    # Long: MACD crosses above signal AND price above SMA200
    cross_up   = (macd > sig_) & (macd.shift(1) <= sig_.shift(1))
    long_entry = cross_up & (close > sma_v)

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_MACD_200_SMA(trial):
    return {
        'fast_len':   trial.suggest_int('fast_len', 5, 20),
        'slow_len':   trial.suggest_int('slow_len', 15, 50),
        'signal_len': trial.suggest_int('signal_len', 5, 15),
        'sma_len':    trial.suggest_int('sma_len', 100, 300),
    }


# ─────────────────────────────────────────────────────────────
# 10. BB_Squeeze_Momentum
# ─────────────────────────────────────────────────────────────
def gen_BB_Squeeze_Momentum(df, bb_len=20, bb_mult=2.0, kc_len=20,
                              kc_mult=1.5, mom_len=12):
    close    = df['close']
    basis    = _sma(close, bb_len)
    bb_upper = basis + bb_mult * close.rolling(bb_len).std()
    bb_lower = basis - bb_mult * close.rolling(bb_len).std()

    kc_mid   = _ema(close, kc_len)
    kc_atr   = _atr(df, kc_len)
    kc_upper = kc_mid + kc_mult * kc_atr
    kc_lower = kc_mid - kc_mult * kc_atr

    # Squeeze: BB inside KC
    squeeze  = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    # Momentum
    mom      = close - _sma(close, mom_len)
    mom_pos  = mom > 0

    # Signal: squeeze just released AND momentum positive
    sq_prev  = squeeze.shift(1).fillna(True)
    long_entry = ~squeeze & sq_prev & mom_pos

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_BB_Squeeze_Momentum(trial):
    return {
        'bb_len':  trial.suggest_int('bb_len', 10, 50),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0),
        'kc_len':  trial.suggest_int('kc_len', 10, 50),
        'kc_mult': trial.suggest_float('kc_mult', 1.0, 3.0),
        'mom_len': trial.suggest_int('mom_len', 5, 30),
    }


# ─────────────────────────────────────────────────────────────
# 11. Mutanabby_ATR_Trend
# ─────────────────────────────────────────────────────────────
def gen_Mutanabby_ATR_Trend(df, atr_len=14, atr_mult=3.0, ema_fast=9,
                              ema_slow=21, ema_trend=200):
    close  = df['close']
    atr    = _atr(df, atr_len)
    ema1   = _ema(close, ema_fast)
    ema2   = _ema(close, ema_slow)
    ema3   = _ema(close, ema_trend)
    atr_lower = ema2 - atr_mult * atr

    # Long: fast EMA crosses above slow EMA AND price above trend EMA AND price above ATR lower
    cross_up   = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
    long_entry = cross_up & (close > ema3) & (close > atr_lower)

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_Mutanabby_ATR_Trend(trial):
    return {
        'atr_len':   trial.suggest_int('atr_len', 7, 21),
        'atr_mult':  trial.suggest_float('atr_mult', 1.5, 5.0),
        'ema_fast':  trial.suggest_int('ema_fast', 5, 20),
        'ema_slow':  trial.suggest_int('ema_slow', 15, 50),
        'ema_trend': trial.suggest_int('ema_trend', 100, 300),
    }


# ─────────────────────────────────────────────────────────────
# 12. BTC_Correlation_Scalper
# ─────────────────────────────────────────────────────────────
def gen_BTC_Correlation_Scalper(df, ema_fast=8, ema_slow=21, rsi_len=14,
                                  rsi_ob=70.0, rsi_os=30.0, atr_len=14,
                                  gap_mult=1.0):
    close  = df['close']
    open_  = df['open']
    ema1   = _ema(close, ema_fast)
    ema2   = _ema(close, ema_slow)
    rsi    = _rsi(close, rsi_len)
    atr    = _atr(df, atr_len)

    # Gap detection: open gaps up relative to previous close by gap_mult * ATR
    gap_up = open_ > close.shift(1) + gap_mult * atr.shift(1)

    # Long: EMA crossover OR gap up, AND RSI not overbought; new signal only
    ema_cross = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
    long_cond  = (ema_cross | gap_up) & (rsi < rsi_ob)
    long_entry = long_cond & ~long_cond.shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_BTC_Correlation_Scalper(trial):
    return {
        'ema_fast': trial.suggest_int('ema_fast', 3, 20),
        'ema_slow': trial.suggest_int('ema_slow', 15, 50),
        'rsi_len':  trial.suggest_int('rsi_len', 7, 21),
        'rsi_ob':   trial.suggest_float('rsi_ob', 60.0, 85.0),
        'rsi_os':   trial.suggest_float('rsi_os', 15.0, 40.0),
        'atr_len':  trial.suggest_int('atr_len', 7, 21),
        'gap_mult': trial.suggest_float('gap_mult', 0.5, 3.0),
    }


# ─────────────────────────────────────────────────────────────
# 13. BB_RSI_KL
# ─────────────────────────────────────────────────────────────
def gen_BB_RSI_KL(df, bb_len=20, bb_mult=2.0, rsi_len=14, rsi_ob=70, rsi_os=30):
    close  = df['close']
    basis  = _sma(close, bb_len)
    dev    = bb_mult * close.rolling(bb_len).std()
    lower  = basis - dev
    rsi_v  = _rsi(close, rsi_len)

    # Long: close < lower BB AND RSI oversold, new signal only
    long_cond  = (close < lower) & (rsi_v < rsi_os)
    long_entry = long_cond & ~long_cond.shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_BB_RSI_KL(trial):
    return {
        'bb_len':  trial.suggest_int('bb_len', 10, 50),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'rsi_ob':  trial.suggest_int('rsi_ob', 60, 80),
        'rsi_os':  trial.suggest_int('rsi_os', 20, 40),
    }


# ─────────────────────────────────────────────────────────────
# 14. Williams_R_Strategy
# ─────────────────────────────────────────────────────────────
def gen_Williams_R_Strategy(df, wr_len=14, wr_ob=-20.0, wr_os=-80.0, ema_len=50):
    close  = df['close']
    high   = df['high']
    low    = df['low']

    hh = high.rolling(wr_len).max()
    ll = low.rolling(wr_len).min()
    wr = -100 * (hh - close) / (hh - ll + 1e-9)
    ema_v = _ema(close, ema_len)

    # Long: WR crosses above oversold level AND price above EMA
    cross_up   = (wr > wr_os) & (wr.shift(1) <= wr_os)
    long_entry = cross_up & (close > ema_v)

    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

def space_Williams_R_Strategy(trial):
    return {
        'wr_len':  trial.suggest_int('wr_len', 7, 30),
        'wr_ob':   trial.suggest_float('wr_ob', -30.0, -5.0),
        'wr_os':   trial.suggest_float('wr_os', -95.0, -60.0),
        'ema_len': trial.suggest_int('ema_len', 20, 100),
    }


# ─────────────────────────────────────────────────────────────
# STRATEGY EXPORT
# ─────────────────────────────────────────────────────────────
STRATEGY_EXPORT = {
    'HL_Breakout_ATR': {
        'gen': gen_HL_Breakout_ATR,
        'space': space_HL_Breakout_ATR,
        'default_params': {'dc_len': 20, 'atr_len': 14, 'atr_mult': 2.0},
        'info': {'likes': 476, 'version': 'v5', 'source': 'High_Low_Breakout_Strategy_with_ATR'},
    },
    'BB_Fib_Ratios': {
        'gen': gen_BB_Fib_Ratios,
        'space': space_BB_Fib_Ratios,
        'default_params': {'bb_len': 20, 'fib1_mult': 1.618, 'fib2_mult': 2.618, 'rsi_len': 14},
        'info': {'likes': 469, 'version': 'v4', 'source': 'Bollinger_Bands_Fibonacci_Ratios_St'},
    },
    'Low_Scanner_Crypto': {
        'gen': gen_Low_Scanner_Crypto,
        'space': space_Low_Scanner_Crypto,
        'default_params': {'length': 14, 'rsi_os': 30, 'bb_len': 20, 'bb_mult': 2.0},
        'info': {'likes': 464, 'version': 'v4', 'source': 'Low_Scanner_strategy_crypto'},
    },
    'MACD_EMA200': {
        'gen': gen_MACD_EMA200,
        'space': space_MACD_EMA200,
        'default_params': {'fast_len': 12, 'slow_len': 26, 'signal_len': 9, 'ema_len': 200},
        'info': {'likes': 450, 'version': 'v4', 'source': 'MACD_Crossover_Strategy_with_EMA200'},
    },
    'Mean_Reversion_KL': {
        'gen': gen_Mean_Reversion_KL,
        'space': space_Mean_Reversion_KL,
        'default_params': {
            'bb_len': 20, 'bb_mult': 2.0, 'rsi_len': 14,
            'rsi_ob': 65.0, 'rsi_os': 35.0, 'atr_len': 14,
            'sl_mult': 1.5, 'tp_mult': 2.5,
        },
        'info': {'likes': 445, 'version': 'v5', 'source': 'Mean_Reversion_Strategy_v2__KL'},
    },
    'VWAP_Swing': {
        'gen': gen_VWAP_Swing,
        'space': space_VWAP_Swing,
        'default_params': {'ema_len': 21, 'rsi_len': 14, 'rsi_ob': 70.0, 'rsi_os': 30.0},
        'info': {'likes': 435, 'version': 'v5', 'source': 'Swing_VWAP_Crypto_and_Stocks_Strate'},
    },
    'HA_Long_Term': {
        'gen': gen_HA_Long_Term,
        'space': space_HA_Long_Term,
        'default_params': {'ema_len': 200, 'rsi_len': 14},
        'info': {'likes': 426, 'version': 'v4', 'source': 'CRYPTO_HA_Strategy_money_maker_long'},
    },
    'Trend_BB_Valente': {
        'gen': gen_Trend_BB_Valente,
        'space': space_Trend_BB_Valente,
        'default_params': {'bb_len': 20, 'bb_mult': 2.0, 'ema_len': 50, 'atr_len': 14},
        'info': {'likes': 417, 'version': 'v4', 'source': 'Setup_Trend_Following_Bollinger_Ban'},
    },
    'MACD_200_SMA': {
        'gen': gen_MACD_200_SMA,
        'space': space_MACD_200_SMA,
        'default_params': {'fast_len': 12, 'slow_len': 26, 'signal_len': 9, 'sma_len': 200},
        'info': {'likes': 413, 'version': 'v4', 'source': 'MacD_200_Day_Moving_Average_Signal'},
    },
    'BB_Squeeze_Momentum': {
        'gen': gen_BB_Squeeze_Momentum,
        'space': space_BB_Squeeze_Momentum,
        'default_params': {'bb_len': 20, 'bb_mult': 2.0, 'kc_len': 20, 'kc_mult': 1.5, 'mom_len': 12},
        'info': {'likes': 402, 'version': 'v6', 'source': 'BB_Breakout___Momentum_Squeeze__Str'},
    },
    'Mutanabby_ATR_Trend': {
        'gen': gen_Mutanabby_ATR_Trend,
        'space': space_Mutanabby_ATR_Trend,
        'default_params': {
            'atr_len': 14, 'atr_mult': 3.0,
            'ema_fast': 9, 'ema_slow': 21, 'ema_trend': 200,
        },
        'info': {'likes': 383, 'version': 'v6', 'source': 'Mutanabby_AI___ATR____Trend_Followi'},
    },
    'BTC_Correlation_Scalper': {
        'gen': gen_BTC_Correlation_Scalper,
        'space': space_BTC_Correlation_Scalper,
        'default_params': {
            'ema_fast': 8, 'ema_slow': 21, 'rsi_len': 14,
            'rsi_ob': 70.0, 'rsi_os': 30.0, 'atr_len': 14, 'gap_mult': 1.0,
        },
        'info': {'likes': 381, 'version': 'v5', 'source': 'Crypto_BTC_Correlation_Scalper_Gaps'},
    },
    'BB_RSI_KL': {
        'gen': gen_BB_RSI_KL,
        'space': space_BB_RSI_KL,
        'default_params': {'bb_len': 20, 'bb_mult': 2.0, 'rsi_len': 14, 'rsi_ob': 70, 'rsi_os': 30},
        'info': {'likes': 369, 'version': 'v4', 'source': 'KL__Bollinger_bands___RSI_Strategy'},
    },
    'Williams_R_Strategy': {
        'gen': gen_Williams_R_Strategy,
        'space': space_Williams_R_Strategy,
        'default_params': {'wr_len': 14, 'wr_ob': -20.0, 'wr_os': -80.0, 'ema_len': 50},
        'info': {'likes': 369, 'version': 'v5', 'source': 'Williams__R_Strategy'},
    },
}
