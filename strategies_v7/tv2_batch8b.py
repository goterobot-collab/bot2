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


# ─────────────────────────────────────────────────────────────────────────────
# 1. Optimized_HA
# ─────────────────────────────────────────────────────────────────────────────
def gen_Optimized_HA(df, ha_smooth=1, ema_len=21, use_ema_filter=True, **kwargs):
    """Heikin Ashi with optional EMA filter. Long/Short."""
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # Iterative ha_open
    ha_open = ha_close.copy()
    ha_open_vals = ha_open.values.copy()
    ha_close_vals = ha_close.values
    for i in range(1, len(df)):
        ha_open_vals[i] = (ha_open_vals[i-1] + ha_close_vals[i-1]) / 2
    ha_open = pd.Series(ha_open_vals, index=df.index)

    smooth_close = _sma(ha_close, ha_smooth)
    smooth_open  = _sma(ha_open,  ha_smooth)

    ha_bull = smooth_close > smooth_open
    ha_bear = smooth_close < smooth_open

    ema = _ema(df['close'], ema_len)

    long_cond  = ha_bull & ~ha_bull.shift(1).infer_objects(copy=False).fillna(False).astype(bool)
    short_cond = ha_bear & ~ha_bear.shift(1).infer_objects(copy=False).fillna(False).astype(bool)

    if use_ema_filter:
        long_cond  = long_cond  & (df['close'] > ema)
        short_cond = short_cond & (df['close'] < ema)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_Optimized_HA(trial):
    return {
        'ha_smooth':      trial.suggest_int('ha_smooth', 1, 5),
        'ema_len':        trial.suggest_int('ema_len', 10, 50),
        'use_ema_filter': trial.suggest_categorical('use_ema_filter', [True, False]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. KAMA_TradeDots
# ─────────────────────────────────────────────────────────────────────────────
def _kama(src, length=10, fast=2, slow=30):
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    vals = src.values
    kama_vals = np.full(len(vals), np.nan)
    # Seed with first valid value
    for i in range(length, len(vals)):
        if np.isnan(kama_vals[i-1]):
            kama_vals[i] = vals[i]
            continue
        direction = abs(vals[i] - vals[i - length])
        noise = sum(abs(vals[j] - vals[j-1]) for j in range(i - length + 1, i + 1))
        er = direction / (noise + 1e-9)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_vals[i] = kama_vals[i-1] + sc * (vals[i] - kama_vals[i-1])
    return pd.Series(kama_vals, index=src.index)


def gen_KAMA_TradeDots(df, length=10, fast_len=2, slow_len=30, **kwargs):
    """KAMA crossover. Long when price crosses above KAMA, short below."""
    kama = _kama(df['close'], length=length, fast=fast_len, slow=slow_len)

    long_cond  = (df['close'] > kama) & (df['close'].shift(1) <= kama.shift(1))
    short_cond = (df['close'] < kama) & (df['close'].shift(1) >= kama.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_KAMA_TradeDots(trial):
    return {
        'length':   trial.suggest_int('length', 5, 20),
        'fast_len': trial.suggest_int('fast_len', 2, 5),
        'slow_len': trial.suggest_int('slow_len', 20, 50),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CCI_EMA_ATR
# ─────────────────────────────────────────────────────────────────────────────
def _cci(df, n=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = _sma(tp, n)
    mean_dev = (tp - sma).abs().rolling(n).mean()
    return (tp - sma) / (0.015 * mean_dev + 1e-9)


def gen_CCI_EMA_ATR(df, cci_length=20, cci_ob=100.0, cci_os=-100.0, ema_len=50, **kwargs):
    """CCI crosses oversold/overbought with EMA filter."""
    cci = _cci(df, cci_length)
    ema = _ema(df['close'], ema_len)

    long_cond  = (cci > cci_os) & (cci.shift(1) <= cci_os) & (df['close'] > ema)
    short_cond = (cci < cci_ob) & (cci.shift(1) >= cci_ob) & (df['close'] < ema)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_CCI_EMA_ATR(trial):
    return {
        'cci_length': trial.suggest_int('cci_length', 10, 30),
        'cci_ob':     trial.suggest_float('cci_ob', 80.0, 150.0),
        'cci_os':     trial.suggest_float('cci_os', -150.0, -80.0),
        'ema_len':    trial.suggest_int('ema_len', 20, 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. BB_Pending_Alerts
# ─────────────────────────────────────────────────────────────────────────────
def gen_BB_Pending_Alerts(df, length=20, mult=2.0, **kwargs):
    """Bollinger Bands: buy on cross above lower, sell on cross below upper."""
    basis, upper, lower = _bb(df['close'], n=length, mult=mult)

    buy_sig  = (df['close'] > lower) & (df['close'].shift(1) <= lower.shift(1))
    sell_sig = (df['close'] < upper) & (df['close'].shift(1) >= upper.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[buy_sig]  = 1
    sig[sell_sig] = -1
    return sig


def space_BB_Pending_Alerts(trial):
    return {
        'length': trial.suggest_int('length', 10, 40),
        'mult':   trial.suggest_float('mult', 1.5, 3.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Flash_Strategy_ATR
# ─────────────────────────────────────────────────────────────────────────────
def gen_Flash_Strategy_ATR(df, rsi_len=14, rsi_buy=50.0, rsi_sell=50.0,
                            ema_fast=8, ema_slow=21, atr_len=14, atr_mult=1.5,
                            mom_len=10, **kwargs):
    """EMA crossover + RSI filter + momentum."""
    rsi  = _rsi(df['close'], rsi_len)
    ema1 = _ema(df['close'], ema_fast)
    ema2 = _ema(df['close'], ema_slow)
    mom  = df['close'] - df['close'].shift(mom_len)

    long_cond  = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1)) & (rsi > rsi_buy)  & (mom > 0)
    short_cond = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1)) & (rsi < rsi_sell) & (mom < 0)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_Flash_Strategy_ATR(trial):
    return {
        'rsi_len':  trial.suggest_int('rsi_len', 7, 21),
        'rsi_buy':  trial.suggest_float('rsi_buy', 40.0, 60.0),
        'rsi_sell': trial.suggest_float('rsi_sell', 40.0, 60.0),
        'ema_fast': trial.suggest_int('ema_fast', 5, 15),
        'ema_slow': trial.suggest_int('ema_slow', 15, 35),
        'atr_len':  trial.suggest_int('atr_len', 7, 21),
        'atr_mult': trial.suggest_float('atr_mult', 1.0, 3.0),
        'mom_len':  trial.suggest_int('mom_len', 5, 20),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. TradePro_2EMA_Stoch_ATR
# ─────────────────────────────────────────────────────────────────────────────
def gen_TradePro_2EMA_Stoch_ATR(df, ema_fast=8, ema_slow=21,
                                  stoch_len=14, stoch_smooth_k=3, stoch_smooth_d=3,
                                  stoch_ob=80.0, stoch_os=20.0,
                                  atr_len=14, atr_mult=2.0, **kwargs):
    """2 EMA crossover + Stochastic RSI filter."""
    ema1 = _ema(df['close'], ema_fast)
    ema2 = _ema(df['close'], ema_slow)

    # Stochastic RSI
    rsi_series = _rsi(df['close'], stoch_len)
    rsi_min = rsi_series.rolling(stoch_len).min()
    rsi_max = rsi_series.rolling(stoch_len).max()
    stoch_k_raw = 100 * (rsi_series - rsi_min) / (rsi_max - rsi_min + 1e-9)
    stoch_k = _sma(stoch_k_raw, stoch_smooth_k)

    bull_cross = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
    bear_cross = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))

    long_cond  = bull_cross & (stoch_k < stoch_os)
    short_cond = bear_cross & (stoch_k > stoch_ob)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_TradePro_2EMA_Stoch_ATR(trial):
    return {
        'ema_fast':      trial.suggest_int('ema_fast', 5, 15),
        'ema_slow':      trial.suggest_int('ema_slow', 15, 35),
        'stoch_len':     trial.suggest_int('stoch_len', 10, 20),
        'stoch_smooth_k':trial.suggest_int('stoch_smooth_k', 2, 5),
        'stoch_smooth_d':trial.suggest_int('stoch_smooth_d', 2, 5),
        'stoch_ob':      trial.suggest_float('stoch_ob', 70.0, 90.0),
        'stoch_os':      trial.suggest_float('stoch_os', 10.0, 30.0),
        'atr_len':       trial.suggest_int('atr_len', 7, 21),
        'atr_mult':      trial.suggest_float('atr_mult', 1.0, 3.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Bollinger_Stop
# ─────────────────────────────────────────────────────────────────────────────
def gen_Bollinger_Stop(df, length=20, mult=2.0, **kwargs):
    """
    Long: price crosses under lower band (oversold bounce entry).
    Short: price crosses over upper band (overbought reversal entry).
    Exits at basis cross — encoded as new signals flipping direction.
    """
    basis, upper, lower = _bb(df['close'], n=length, mult=mult)
    src = df['close']

    long_entry  = (src < lower) & (src.shift(1) >= lower.shift(1))
    short_entry = (src > upper) & (src.shift(1) <= upper.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_entry]  = 1
    sig[short_entry] = -1
    return sig


def space_Bollinger_Stop(trial):
    return {
        'length': trial.suggest_int('length', 10, 40),
        'mult':   trial.suggest_float('mult', 1.5, 3.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Crypto_Momentum_V4
# ─────────────────────────────────────────────────────────────────────────────
def gen_Crypto_Momentum_V4(df, fast_len=12, slow_len=26, signal_len=9,
                             rsi_len=14, rsi_ob=70.0, rsi_os=30.0, **kwargs):
    """MACD crossover with RSI filter (not overbought/oversold)."""
    macd_line   = _ema(df['close'], fast_len) - _ema(df['close'], slow_len)
    signal_line = _ema(macd_line, signal_len)
    rsi         = _rsi(df['close'], rsi_len)

    macd_cross_up   = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    long_cond  = macd_cross_up   & (rsi > rsi_os) & (rsi < rsi_ob)
    short_cond = macd_cross_down & (rsi < rsi_ob) & (rsi > rsi_os)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_Crypto_Momentum_V4(trial):
    return {
        'fast_len':   trial.suggest_int('fast_len', 8, 20),
        'slow_len':   trial.suggest_int('slow_len', 20, 40),
        'signal_len': trial.suggest_int('signal_len', 5, 15),
        'rsi_len':    trial.suggest_int('rsi_len', 7, 21),
        'rsi_ob':     trial.suggest_float('rsi_ob', 60.0, 80.0),
        'rsi_os':     trial.suggest_float('rsi_os', 20.0, 40.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. PriceAction_BB_V2
# ─────────────────────────────────────────────────────────────────────────────
def gen_PriceAction_BB_V2(df, bb_len=20, bb_mult=2.0, band_tolerance=0.5, **kwargs):
    """
    Bullish/bearish engulfing candles near BB extremes.
    band_tolerance: ATR multiplier to widen the band touch zone (default 0.5).
    Pine original used close<=lower / close>=upper; tolerance relaxes for higher TFs.
    """
    o = df['open']; c = df['close']
    o1 = o.shift(1); c1 = c.shift(1)

    bull_engulf = (c > o) & (c1 < o1) & (c > o1) & (o < c1)
    bear_engulf = (c < o) & (c1 > o1) & (c < o1) & (o > c1)

    basis, upper, lower = _bb(df['close'], n=bb_len, mult=bb_mult)
    atr = _atr(df, 14)

    long_cond  = bull_engulf & (df['close'] <= lower + band_tolerance * atr)
    short_cond = bear_engulf & (df['close'] >= upper - band_tolerance * atr)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_PriceAction_BB_V2(trial):
    return {
        'bb_len':          trial.suggest_int('bb_len', 10, 40),
        'bb_mult':         trial.suggest_float('bb_mult', 1.5, 3.0),
        'band_tolerance':  trial.suggest_float('band_tolerance', 0.0, 1.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY EXPORT
# ─────────────────────────────────────────────────────────────────────────────
STRATEGY_EXPORT = {
    'Optimized_HA': {
        'gen': gen_Optimized_HA,
        'space': space_Optimized_HA,
        'default_params': {'ha_smooth': 1, 'ema_len': 21, 'use_ema_filter': True},
        'info': {'likes': 950, 'version': 'v5', 'source': 'TV_Optimized_HA'},
    },
    'KAMA_TradeDots': {
        'gen': gen_KAMA_TradeDots,
        'space': space_KAMA_TradeDots,
        'default_params': {'length': 10, 'fast_len': 2, 'slow_len': 30},
        'info': {'likes': 916, 'version': 'v5', 'source': 'TV_KAMA_TradeDots'},
    },
    'CCI_EMA_ATR': {
        'gen': gen_CCI_EMA_ATR,
        'space': space_CCI_EMA_ATR,
        'default_params': {'cci_length': 20, 'cci_ob': 100.0, 'cci_os': -100.0, 'ema_len': 50},
        'info': {'likes': 887, 'version': 'v5', 'source': 'TV_CCI_EMA_ATR'},
    },
    'BB_Pending_Alerts': {
        'gen': gen_BB_Pending_Alerts,
        'space': space_BB_Pending_Alerts,
        'default_params': {'length': 20, 'mult': 2.0},
        'info': {'likes': 875, 'version': 'v5', 'source': 'TV_BB_Pending_Alerts'},
    },
    'Flash_Strategy_ATR': {
        'gen': gen_Flash_Strategy_ATR,
        'space': space_Flash_Strategy_ATR,
        'default_params': {'rsi_len': 14, 'rsi_buy': 50.0, 'rsi_sell': 50.0,
                           'ema_fast': 8, 'ema_slow': 21, 'atr_len': 14,
                           'atr_mult': 1.5, 'mom_len': 10},
        'info': {'likes': 871, 'version': 'v5', 'source': 'TV_Flash_Strategy_ATR'},
    },
    'TradePro_2EMA_Stoch_ATR': {
        'gen': gen_TradePro_2EMA_Stoch_ATR,
        'space': space_TradePro_2EMA_Stoch_ATR,
        'default_params': {'ema_fast': 8, 'ema_slow': 21, 'stoch_len': 14,
                           'stoch_smooth_k': 3, 'stoch_smooth_d': 3,
                           'stoch_ob': 80.0, 'stoch_os': 20.0,
                           'atr_len': 14, 'atr_mult': 2.0},
        'info': {'likes': 843, 'version': 'v5', 'source': 'TV_TradePro_2EMA_Stoch_ATR'},
    },
    'Bollinger_Stop': {
        'gen': gen_Bollinger_Stop,
        'space': space_Bollinger_Stop,
        'default_params': {'length': 20, 'mult': 2.0},
        'info': {'likes': 836, 'version': 'v5', 'source': 'TV_Bollinger_Stop'},
    },
    'Crypto_Momentum_V4': {
        'gen': gen_Crypto_Momentum_V4,
        'space': space_Crypto_Momentum_V4,
        'default_params': {'fast_len': 12, 'slow_len': 26, 'signal_len': 9,
                           'rsi_len': 14, 'rsi_ob': 70.0, 'rsi_os': 30.0},
        'info': {'likes': 827, 'version': 'v4', 'source': 'TV_Crypto_Momentum_V4'},
    },
    'PriceAction_BB_V2': {
        'gen': gen_PriceAction_BB_V2,
        'space': space_PriceAction_BB_V2,
        'default_params': {'bb_len': 20, 'bb_mult': 2.0, 'band_tolerance': 0.5},
        'info': {'likes': 819, 'version': 'v4', 'source': 'TV_PriceAction_BB_V2'},
    },
}
