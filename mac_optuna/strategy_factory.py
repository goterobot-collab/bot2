#!/usr/bin/env python3
"""
STRATEGY FACTORY: 215+ unique strategy generators with Optuna search spaces.
Importable: from strategy_factory import FACTORY_STRATS
All generators are vectorized (pandas/numpy). Each takes (df, **params) → pd.Series of signals.
"""

import pandas as pd
import numpy as np

# ─── INDICATOR HELPERS ───────────────────────────────────────────────────────

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0).ewm(span=p, adjust=False).mean()
    l = (-d.where(d < 0, 0.0)).ewm(span=p, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def bb(c, p=20, std=2.0):
    m = c.rolling(p).mean()
    s = c.rolling(p).std()
    return m, m + std * s, m - std * s

def macd_calc(c, f=12, s=26, sig=9):
    m = ema(c, f) - ema(c, s)
    si = ema(m, sig)
    return m, si, m - si

def stoch(h, l, c, kp=14, dp=3):
    lo = l.rolling(kp).min()
    hi = h.rolling(kp).max()
    k = 100 * (c - lo) / (hi - lo + 1e-10)
    return k, k.rolling(dp).mean()

def adx_calc(h, l, c, p=14):
    tr1 = atr(h, l, c, 1)
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0)
    pdi = 100 * pd.Series(pdm, index=c.index).rolling(p).mean() / (tr1.rolling(p).mean() + 1e-10)
    mdi = 100 * pd.Series(mdm, index=c.index).rolling(p).mean() / (tr1.rolling(p).mean() + 1e-10)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    return dx.rolling(p).mean(), pdi, mdi

def cci_calc(h, l, c, p=20):
    tp = (h + l + c) / 3
    m = tp.rolling(p).mean()
    md = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - m) / (0.015 * md + 1e-10)

def williams_r(h, l, c, p=14):
    hi = h.rolling(p).max()
    lo = l.rolling(p).min()
    return -100 * (hi - c) / (hi - lo + 1e-10)

def vwap_calc(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    cumvol = df['volume'].cumsum()
    return (tp * df['volume']).cumsum() / (cumvol + 1e-10)

def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha['open'] = (df['open'].shift() + df['close'].shift()) / 2
    ha['open'].iloc[0] = df['open'].iloc[0]
    ha['high'] = pd.concat([df['high'], ha['open'], ha['close']], axis=1).max(axis=1)
    ha['low'] = pd.concat([df['low'], ha['open'], ha['close']], axis=1).min(axis=1)
    ha['volume'] = df['volume']
    return ha

def keltner(c, h, l, p=20, mult=2.0):
    e = ema(c, p)
    a = atr(h, l, c, p)
    return e, e + mult * a, e - mult * a

def obv_calc(c, v):
    return (np.sign(c.diff()) * v).cumsum()

# ─── REGISTRY ────────────────────────────────────────────────────────────────

FACTORY_STRATS = {}

def reg(name, gen_func, space_func):
    FACTORY_STRATS[name] = {'gen': gen_func, 'space': space_func}

# ==============================================================================
# CATEGORY 1: VWAP VARIATIONS (40)
# ==============================================================================

def gen_vwap_basic(df, dist_pct=0.01):
    vw = vwap_calc(df)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (df['close'] > df['close'].shift())] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (df['close'] < df['close'].shift())] = -1
    return sig
reg('VWAP_Basic', gen_vwap_basic,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001)})

def gen_vwap_rsi(df, dist_pct=0.01, rsi_buy=30, rsi_sell=70):
    vw = vwap_calc(df)
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (r < rsi_buy)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (r > rsi_sell)] = -1
    return sig
reg('VWAP_RSI', gen_vwap_rsi,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'rsi_sell': t.suggest_int('rsi_sell', 60, 85)})

def gen_vwap_vol(df, dist_pct=0.01, vol_mult=1.5):
    vw = vwap_calc(df)
    vm = df['volume'].rolling(20).mean()
    high_vol = df['volume'] > vol_mult * vm
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & high_vol] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & high_vol] = -1
    return sig
reg('VWAP_Volume', gen_vwap_vol,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'vol_mult': t.suggest_float('vol_mult', 1.0, 4.0, step=0.2)})

def gen_vwap_atr(df, atr_mult=1.5, atr_period=14):
    vw = vwap_calc(df)
    a = atr(df['high'], df['low'], df['close'], atr_period)
    dist = atr_mult * a
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw - dist) & (df['close'] > df['close'].shift())] = 1
    sig[(df['close'] > vw + dist) & (df['close'] < df['close'].shift())] = -1
    return sig
reg('VWAP_ATR', gen_vwap_atr,
    lambda t: {'atr_mult': t.suggest_float('atr_mult', 0.5, 4.0, step=0.25),
               'atr_period': t.suggest_int('atr_period', 7, 28)})

def gen_vwap_stoch(df, dist_pct=0.01, k_period=14, stoch_buy=20, stoch_sell=80):
    vw = vwap_calc(df)
    k, d = stoch(df['high'], df['low'], df['close'], k_period, 3)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (k < stoch_buy)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (k > stoch_sell)] = -1
    return sig
reg('VWAP_Stoch', gen_vwap_stoch,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'k_period': t.suggest_int('k_period', 5, 28),
               'stoch_buy': t.suggest_int('stoch_buy', 8, 30), 'stoch_sell': t.suggest_int('stoch_sell', 70, 92)})

def gen_vwap_bb(df, dist_pct=0.01, bb_period=20, bb_std=2.0):
    vw = vwap_calc(df)
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (df['close'] < lower)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (df['close'] > upper)] = -1
    return sig
reg('VWAP_BB', gen_vwap_bb,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.05, step=0.001),
               'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_vwap_momentum(df, dist_pct=0.01, mom_period=10):
    vw = vwap_calc(df)
    mom = df['close'] / df['close'].shift(mom_period) - 1
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (mom > -0.02)] = 1  # not falling knife
    sig[(df['close'] > vw * (1 + dist_pct)) & (mom < 0.02)] = -1
    return sig
reg('VWAP_Momentum', gen_vwap_momentum,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'mom_period': t.suggest_int('mom_period', 3, 25)})

def gen_vwap_ema_trend(df, dist_pct=0.01, ema_period=50):
    vw = vwap_calc(df)
    trend = ema(df['close'], ema_period)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (df['close'] > trend)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (df['close'] < trend)] = -1
    return sig
reg('VWAP_EMA_Trend', gen_vwap_ema_trend,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'ema_period': t.suggest_int('ema_period', 20, 200)})

def gen_vwap_heikin(df, dist_pct=0.01):
    ha = heikin_ashi(df)
    vw = vwap_calc(df)
    ha_bull = ha['close'] > ha['open']
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & ha_bull] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & ~ha_bull] = -1
    return sig
reg('VWAP_HeikinAshi', gen_vwap_heikin,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001)})

def gen_vwap_double(df, dist_pct=0.01, session_len=20):
    vw_cum = vwap_calc(df)
    tp = (df['high'] + df['low'] + df['close']) / 3
    vw_session = (tp * df['volume']).rolling(session_len).sum() / (df['volume'].rolling(session_len).sum() + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw_cum * (1 - dist_pct)) & (df['close'] < vw_session)] = 1
    sig[(df['close'] > vw_cum * (1 + dist_pct)) & (df['close'] > vw_session)] = -1
    return sig
reg('VWAP_Double', gen_vwap_double,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.05, step=0.001),
               'session_len': t.suggest_int('session_len', 10, 60)})

def gen_vwap_rsi_vol(df, dist_pct=0.01, rsi_buy=30, vol_mult=1.3):
    vw = vwap_calc(df)
    r = rsi(df['close'], 14)
    vm = df['volume'].rolling(20).mean()
    high_vol = df['volume'] > vol_mult * vm
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (r < rsi_buy) & high_vol] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (r > (100 - rsi_buy)) & high_vol] = -1
    return sig
reg('VWAP_RSI_Vol', gen_vwap_rsi_vol,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.05, step=0.001),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40),
               'vol_mult': t.suggest_float('vol_mult', 1.0, 3.5, step=0.2)})

def gen_vwap_macd(df, dist_pct=0.01):
    vw = vwap_calc(df)
    _, _, hist = macd_calc(df['close'])
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (hist > 0)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (hist < 0)] = -1
    return sig
reg('VWAP_MACD', gen_vwap_macd,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001)})

def gen_vwap_cci(df, dist_pct=0.01, cci_level=100):
    vw = vwap_calc(df)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (c < -cci_level)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (c > cci_level)] = -1
    return sig
reg('VWAP_CCI', gen_vwap_cci,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'cci_level': t.suggest_int('cci_level', 50, 200)})

def gen_vwap_williams(df, dist_pct=0.01, wr_buy=-80):
    vw = vwap_calc(df)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (wr < wr_buy)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (wr > -100 - wr_buy)] = -1
    return sig
reg('VWAP_WilliamsR', gen_vwap_williams,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'wr_buy': t.suggest_int('wr_buy', -95, -65)})

def gen_vwap_adx_filter(df, dist_pct=0.01, adx_thresh=25):
    vw = vwap_calc(df)
    a, _, _ = adx_calc(df['high'], df['low'], df['close'], 14)
    ranging = a < adx_thresh
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & ranging] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & ranging] = -1
    return sig
reg('VWAP_ADX_Range', gen_vwap_adx_filter,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'adx_thresh': t.suggest_int('adx_thresh', 15, 35)})

def gen_vwap_obv(df, dist_pct=0.01, obv_period=20):
    vw = vwap_calc(df)
    o = obv_calc(df['close'], df['volume'])
    o_sma = o.rolling(obv_period).mean()
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (o > o_sma)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (o < o_sma)] = -1
    return sig
reg('VWAP_OBV', gen_vwap_obv,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'obv_period': t.suggest_int('obv_period', 10, 40)})

def gen_vwap_atr_rsi(df, atr_mult=1.5, atr_period=14, rsi_buy=30):
    vw = vwap_calc(df)
    a = atr(df['high'], df['low'], df['close'], atr_period)
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw - atr_mult * a) & (r < rsi_buy)] = 1
    sig[(df['close'] > vw + atr_mult * a) & (r > 100 - rsi_buy)] = -1
    return sig
reg('VWAP_ATR_RSI', gen_vwap_atr_rsi,
    lambda t: {'atr_mult': t.suggest_float('atr_mult', 0.5, 4.0, step=0.25),
               'atr_period': t.suggest_int('atr_period', 7, 28),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40)})

def gen_vwap_squeeze(df, dist_pct=0.01, bb_period=20, kc_mult=1.5):
    vw = vwap_calc(df)
    mid, bb_up, bb_lo = bb(df['close'], bb_period)
    e = ema(df['close'], bb_period)
    a = atr(df['high'], df['low'], df['close'], bb_period)
    kc_up = e + kc_mult * a
    squeezed = (bb_lo > e - kc_mult * a) & (bb_up < kc_up)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & squeezed] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & squeezed] = -1
    return sig
reg('VWAP_Squeeze', gen_vwap_squeeze,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.05, step=0.001),
               'bb_period': t.suggest_int('bb_period', 10, 35),
               'kc_mult': t.suggest_float('kc_mult', 1.0, 2.5, step=0.25)})

def gen_vwap_rsi_stoch(df, dist_pct=0.01, rsi_buy=30, stoch_buy=20):
    vw = vwap_calc(df)
    r = rsi(df['close'], 14)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (r < rsi_buy) & (k < stoch_buy)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (r > 100 - rsi_buy) & (k > 100 - stoch_buy)] = -1
    return sig
reg('VWAP_RSI_Stoch', gen_vwap_rsi_stoch,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.05, step=0.001),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40),
               'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_vwap_mean_cross(df, dist_pct=0.01, ma_period=20):
    vw = vwap_calc(df)
    ma = df['close'].rolling(ma_period).mean()
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (df['close'] > ma)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (df['close'] < ma)] = -1
    return sig
reg('VWAP_MA_Cross', gen_vwap_mean_cross,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'ma_period': t.suggest_int('ma_period', 10, 100)})

def gen_vwap_donchian(df, dist_pct=0.01, dc_period=20):
    vw = vwap_calc(df)
    dc_lo = df['low'].rolling(dc_period).min()
    dc_hi = df['high'].rolling(dc_period).max()
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (df['close'] <= dc_lo.shift())] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (df['close'] >= dc_hi.shift())] = -1
    return sig
reg('VWAP_Donchian', gen_vwap_donchian,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'dc_period': t.suggest_int('dc_period', 5, 50)})

def gen_vwap_keltner(df, dist_pct=0.01, kc_period=20, kc_mult=2.0):
    vw = vwap_calc(df)
    _, kc_up, kc_lo = keltner(df['close'], df['high'], df['low'], kc_period, kc_mult)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (df['close'] < kc_lo)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (df['close'] > kc_up)] = -1
    return sig
reg('VWAP_Keltner', gen_vwap_keltner,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'kc_period': t.suggest_int('kc_period', 10, 40),
               'kc_mult': t.suggest_float('kc_mult', 1.0, 3.5, step=0.25)})

def gen_vwap_vol_profile(df, dist_pct=0.01, lookback=50):
    vw = vwap_calc(df)
    vol_ma = df['volume'].rolling(lookback).mean()
    vol_std = df['volume'].rolling(lookback).std()
    vol_spike = df['volume'] > vol_ma + vol_std
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & vol_spike & (df['close'] > df['close'].shift())] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & vol_spike & (df['close'] < df['close'].shift())] = -1
    return sig
reg('VWAP_VolProfile', gen_vwap_vol_profile,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'lookback': t.suggest_int('lookback', 20, 100)})

def gen_vwap_rsi_divergence(df, dist_pct=0.01, rsi_period=14, lookback=10):
    vw = vwap_calc(df)
    r = rsi(df['close'], rsi_period)
    price_lower = df['close'] < df['close'].rolling(lookback).min().shift()
    rsi_higher = r > r.rolling(lookback).min().shift()
    bull_div = price_lower & rsi_higher
    price_higher = df['close'] > df['close'].rolling(lookback).max().shift()
    rsi_lower = r < r.rolling(lookback).max().shift()
    bear_div = price_higher & rsi_lower
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & bull_div] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & bear_div] = -1
    return sig
reg('VWAP_RSI_Divergence', gen_vwap_rsi_divergence,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.05, step=0.001),
               'rsi_period': t.suggest_int('rsi_period', 7, 21),
               'lookback': t.suggest_int('lookback', 5, 20)})

def gen_vwap_ichimoku(df, dist_pct=0.01, tenkan=9, kijun=26):
    vw = vwap_calc(df)
    th = df['high'].rolling(tenkan).max()
    tl = df['low'].rolling(tenkan).min()
    tk = (th + tl) / 2
    kh = df['high'].rolling(kijun).max()
    kl = df['low'].rolling(kijun).min()
    kj = (kh + kl) / 2
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (tk > kj)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (tk < kj)] = -1
    return sig
reg('VWAP_Ichimoku', gen_vwap_ichimoku,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'tenkan': t.suggest_int('tenkan', 5, 15), 'kijun': t.suggest_int('kijun', 15, 40)})

def gen_vwap_tsi(df, dist_pct=0.01, long_p=25, short_p=13):
    vw = vwap_calc(df)
    d = df['close'].diff()
    ds = ema(ema(d, long_p), short_p)
    ads = ema(ema(d.abs(), long_p), short_p)
    tsi = 100 * ds / (ads + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (tsi > tsi.shift())] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (tsi < tsi.shift())] = -1
    return sig
reg('VWAP_TSI', gen_vwap_tsi,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'long_p': t.suggest_int('long_p', 15, 40), 'short_p': t.suggest_int('short_p', 5, 18)})

def gen_vwap_range_pct(df, dist_pct=0.01, range_lookback=20, range_thresh=0.02):
    vw = vwap_calc(df)
    hi = df['high'].rolling(range_lookback).max()
    lo = df['low'].rolling(range_lookback).min()
    rng = (hi - lo) / (lo + 1e-10)
    tight = rng < range_thresh
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & tight] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & tight] = -1
    return sig
reg('VWAP_TightRange', gen_vwap_range_pct,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'range_lookback': t.suggest_int('range_lookback', 10, 50),
               'range_thresh': t.suggest_float('range_thresh', 0.005, 0.05, step=0.005)})

def gen_vwap_ema_cross(df, dist_pct=0.01, fast=9, slow=21):
    vw = vwap_calc(df)
    ef = ema(df['close'], fast)
    es = ema(df['close'], slow)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (ef > es)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (ef < es)] = -1
    return sig
reg('VWAP_EMA_Cross', gen_vwap_ema_cross,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'fast': t.suggest_int('fast', 3, 15), 'slow': t.suggest_int('slow', 15, 60)})

def gen_vwap_cmf(df, dist_pct=0.01, cmf_period=20):
    vw = vwap_calc(df)
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    mfv = mfm * df['volume']
    cmf = mfv.rolling(cmf_period).sum() / (df['volume'].rolling(cmf_period).sum() + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (cmf > 0)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (cmf < 0)] = -1
    return sig
reg('VWAP_CMF', gen_vwap_cmf,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'cmf_period': t.suggest_int('cmf_period', 10, 40)})

def gen_vwap_roc(df, dist_pct=0.01, roc_period=10, roc_thresh=0.0):
    vw = vwap_calc(df)
    roc = (df['close'] / df['close'].shift(roc_period) - 1) * 100
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (roc > roc_thresh)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (roc < -roc_thresh)] = -1
    return sig
reg('VWAP_ROC', gen_vwap_roc,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'roc_period': t.suggest_int('roc_period', 3, 25),
               'roc_thresh': t.suggest_float('roc_thresh', -2.0, 2.0, step=0.5)})

def gen_vwap_mfi(df, dist_pct=0.01, mfi_period=14, mfi_buy=20, mfi_sell=80):
    vw = vwap_calc(df)
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos_mf = mf.where(tp > tp.shift(), 0.0).rolling(mfi_period).sum()
    neg_mf = mf.where(tp < tp.shift(), 0.0).rolling(mfi_period).sum()
    mfi = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-10))
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (mfi < mfi_buy)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (mfi > mfi_sell)] = -1
    return sig
reg('VWAP_MFI', gen_vwap_mfi,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'mfi_period': t.suggest_int('mfi_period', 7, 28),
               'mfi_buy': t.suggest_int('mfi_buy', 10, 35), 'mfi_sell': t.suggest_int('mfi_sell', 65, 90)})

def gen_vwap_bb_width(df, dist_pct=0.01, bb_period=20, width_thresh=0.02):
    vw = vwap_calc(df)
    mid, upper, lower = bb(df['close'], bb_period)
    width = (upper - lower) / (mid + 1e-10)
    narrow = width < width_thresh
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & narrow] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & narrow] = -1
    return sig
reg('VWAP_BB_Width', gen_vwap_bb_width,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'bb_period': t.suggest_int('bb_period', 10, 40),
               'width_thresh': t.suggest_float('width_thresh', 0.005, 0.06, step=0.005)})

def gen_vwap_triple_confirm(df, dist_pct=0.01, rsi_buy=35, stoch_buy=25, vol_mult=1.2):
    vw = vwap_calc(df)
    r = rsi(df['close'], 14)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    vm = df['volume'].rolling(20).mean()
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (r < rsi_buy) & (k < stoch_buy) & (df['volume'] > vol_mult * vm)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (r > 100 - rsi_buy) & (k > 100 - stoch_buy) & (df['volume'] > vol_mult * vm)] = -1
    return sig
reg('VWAP_TripleConfirm', gen_vwap_triple_confirm,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.05, step=0.001),
               'rsi_buy': t.suggest_int('rsi_buy', 20, 40),
               'stoch_buy': t.suggest_int('stoch_buy', 10, 35),
               'vol_mult': t.suggest_float('vol_mult', 1.0, 3.0, step=0.2)})

def gen_vwap_pivot(df, dist_pct=0.01):
    vw = vwap_calc(df)
    pp = (df['high'].shift() + df['low'].shift() + df['close'].shift()) / 3
    s1 = 2 * pp - df['high'].shift()
    r1 = 2 * pp - df['low'].shift()
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (df['close'] <= s1)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (df['close'] >= r1)] = -1
    return sig
reg('VWAP_Pivot', gen_vwap_pivot,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001)})

def gen_vwap_dema(df, dist_pct=0.01, dema_period=20):
    vw = vwap_calc(df)
    e1 = ema(df['close'], dema_period)
    dema = 2 * e1 - ema(e1, dema_period)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (df['close'] > dema)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (df['close'] < dema)] = -1
    return sig
reg('VWAP_DEMA', gen_vwap_dema,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'dema_period': t.suggest_int('dema_period', 10, 60)})

# ==============================================================================
# CATEGORY 2: ZSCORE VARIATIONS (30)
# ==============================================================================

def gen_zscore_price(df, lookback=50, z_thresh=2.0):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[z < -z_thresh] = 1
    sig[z > z_thresh] = -1
    return sig
reg('ZScore_Price', gen_zscore_price,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_zscore_vwap_dist(df, lookback=50, z_thresh=2.0):
    vw = vwap_calc(df)
    dist = (df['close'] - vw) / (vw + 1e-10)
    m = dist.rolling(lookback).mean()
    s = dist.rolling(lookback).std()
    z = (dist - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[z < -z_thresh] = 1
    sig[z > z_thresh] = -1
    return sig
reg('ZScore_VWAP_Dist', gen_zscore_vwap_dist,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 100), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_zscore_rsi(df, lookback=50, z_thresh=2.0, rsi_period=14):
    r = rsi(df['close'], rsi_period)
    m = r.rolling(lookback).mean()
    s = r.rolling(lookback).std()
    z = (r - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[z < -z_thresh] = 1
    sig[z > z_thresh] = -1
    return sig
reg('ZScore_RSI', gen_zscore_rsi,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 100), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'rsi_period': t.suggest_int('rsi_period', 7, 21)})

def gen_zscore_volume(df, lookback=50, z_thresh=2.0):
    m = df['volume'].rolling(lookback).mean()
    s = df['volume'].rolling(lookback).std()
    z = (df['volume'] - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(z > z_thresh) & (df['close'] > df['open'])] = 1
    sig[(z > z_thresh) & (df['close'] < df['open'])] = -1
    return sig
reg('ZScore_Volume', gen_zscore_volume,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 100), 'z_thresh': t.suggest_float('z_thresh', 1.0, 4.0, step=0.2)})

def gen_zscore_regime(df, lookback=50, z_thresh=2.0, adx_thresh=25):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    a, _, _ = adx_calc(df['high'], df['low'], df['close'], 14)
    ranging = a < adx_thresh
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & ranging] = 1
    sig[(z > z_thresh) & ranging] = -1
    return sig
reg('ZScore_Regime', gen_zscore_regime,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'adx_thresh': t.suggest_int('adx_thresh', 15, 35)})

def gen_zscore_momentum(df, lookback=50, z_thresh=2.0, mom_period=10):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    mom = df['close'] / df['close'].shift(mom_period) - 1
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (mom > -0.01)] = 1
    sig[(z > z_thresh) & (mom < 0.01)] = -1
    return sig
reg('ZScore_Momentum', gen_zscore_momentum,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'mom_period': t.suggest_int('mom_period', 3, 25)})

def gen_zscore_adaptive(df, base_lookback=50, z_thresh=2.0, vol_period=20):
    vol = df['close'].rolling(vol_period).std() / df['close'].rolling(vol_period).mean()
    vol_ma = vol.rolling(vol_period).mean()
    ratio = vol / (vol_ma + 1e-10)
    ratio = ratio.fillna(1.0).replace([np.inf, -np.inf], 1.0)
    # Use base lookback as approximation for vectorized
    m = df['close'].rolling(base_lookback).mean()
    s = df['close'].rolling(base_lookback).std()
    adj_thresh = z_thresh * ratio
    z = (df['close'] - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[z < -adj_thresh] = 1
    sig[z > adj_thresh] = -1
    return sig
reg('ZScore_Adaptive', gen_zscore_adaptive,
    lambda t: {'base_lookback': t.suggest_int('base_lookback', 15, 150),
               'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'vol_period': t.suggest_int('vol_period', 10, 40)})

def gen_zscore_bb_width(df, lookback=50, z_thresh=2.0, bb_period=20, width_pct=0.03):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    mid, upper, lower = bb(df['close'], bb_period)
    width = (upper - lower) / (mid + 1e-10)
    narrow = width < width_pct
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & narrow] = 1
    sig[(z > z_thresh) & narrow] = -1
    return sig
reg('ZScore_BB_Width', gen_zscore_bb_width,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'bb_period': t.suggest_int('bb_period', 10, 40), 'width_pct': t.suggest_float('width_pct', 0.005, 0.06, step=0.005)})

def gen_zscore_ema(df, lookback=50, z_thresh=2.0, ema_period=50):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    trend = ema(df['close'], ema_period)
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (df['close'] > trend)] = 1
    sig[(z > z_thresh) & (df['close'] < trend)] = -1
    return sig
reg('ZScore_EMA', gen_zscore_ema,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'ema_period': t.suggest_int('ema_period', 20, 200)})

def gen_zscore_rsi_confirm(df, lookback=50, z_thresh=2.0, rsi_buy=30):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (r < rsi_buy)] = 1
    sig[(z > z_thresh) & (r > 100 - rsi_buy)] = -1
    return sig
reg('ZScore_RSI_Confirm', gen_zscore_rsi_confirm,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40)})

def gen_zscore_vol_confirm(df, lookback=50, z_thresh=2.0, vol_mult=1.5):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    vm = df['volume'].rolling(20).mean()
    high_vol = df['volume'] > vol_mult * vm
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & high_vol] = 1
    sig[(z > z_thresh) & high_vol] = -1
    return sig
reg('ZScore_Vol_Confirm', gen_zscore_vol_confirm,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'vol_mult': t.suggest_float('vol_mult', 1.0, 4.0, step=0.2)})

def gen_zscore_stoch(df, lookback=50, z_thresh=2.0, stoch_buy=20):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (k < stoch_buy)] = 1
    sig[(z > z_thresh) & (k > 100 - stoch_buy)] = -1
    return sig
reg('ZScore_Stoch', gen_zscore_stoch,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_zscore_macd(df, lookback=50, z_thresh=2.0):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    _, _, hist = macd_calc(df['close'])
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (hist > 0)] = 1
    sig[(z > z_thresh) & (hist < 0)] = -1
    return sig
reg('ZScore_MACD', gen_zscore_macd,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_zscore_cci(df, lookback=50, z_thresh=2.0, cci_level=100):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (c < -cci_level)] = 1
    sig[(z > z_thresh) & (c > cci_level)] = -1
    return sig
reg('ZScore_CCI', gen_zscore_cci,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'cci_level': t.suggest_int('cci_level', 50, 200)})

def gen_zscore_of_cci(df, lookback=50, z_thresh=2.0, cci_period=20):
    c = cci_calc(df['high'], df['low'], df['close'], cci_period)
    m = c.rolling(lookback).mean()
    s = c.rolling(lookback).std()
    z = (c - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[z < -z_thresh] = 1
    sig[z > z_thresh] = -1
    return sig
reg('ZScore_of_CCI', gen_zscore_of_cci,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 100), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'cci_period': t.suggest_int('cci_period', 10, 30)})

def gen_zscore_of_stoch(df, lookback=50, z_thresh=2.0, k_period=14):
    k, _ = stoch(df['high'], df['low'], df['close'], k_period, 3)
    m = k.rolling(lookback).mean()
    s = k.rolling(lookback).std()
    z = (k - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[z < -z_thresh] = 1
    sig[z > z_thresh] = -1
    return sig
reg('ZScore_of_Stoch', gen_zscore_of_stoch,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 100), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'k_period': t.suggest_int('k_period', 5, 28)})

def gen_zscore_of_williams(df, lookback=50, z_thresh=2.0, wr_period=14):
    wr = williams_r(df['high'], df['low'], df['close'], wr_period)
    m = wr.rolling(lookback).mean()
    s = wr.rolling(lookback).std()
    z = (wr - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[z < -z_thresh] = 1
    sig[z > z_thresh] = -1
    return sig
reg('ZScore_of_WilliamsR', gen_zscore_of_williams,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 100), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'wr_period': t.suggest_int('wr_period', 5, 28)})

def gen_zscore_of_macd_hist(df, lookback=50, z_thresh=2.0):
    _, _, hist = macd_calc(df['close'])
    m = hist.rolling(lookback).mean()
    s = hist.rolling(lookback).std()
    z = (hist - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (z > z.shift())] = 1
    sig[(z > z_thresh) & (z < z.shift())] = -1
    return sig
reg('ZScore_of_MACD_Hist', gen_zscore_of_macd_hist,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 100), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_zscore_of_atr(df, lookback=50, z_thresh=2.0, atr_period=14):
    a = atr(df['high'], df['low'], df['close'], atr_period)
    m = a.rolling(lookback).mean()
    s = a.rolling(lookback).std()
    z = (a - m) / (s + 1e-10)
    price_z_m = df['close'].rolling(lookback).mean()
    price_z_s = df['close'].rolling(lookback).std()
    price_z = (df['close'] - price_z_m) / (price_z_s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(z > z_thresh) & (price_z < -1)] = 1
    sig[(z > z_thresh) & (price_z > 1)] = -1
    return sig
reg('ZScore_of_ATR', gen_zscore_of_atr,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 100), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'atr_period': t.suggest_int('atr_period', 7, 28)})

def gen_zscore_dual(df, short_lb=20, long_lb=100, z_thresh=2.0):
    m_s = df['close'].rolling(short_lb).mean()
    s_s = df['close'].rolling(short_lb).std()
    z_short = (df['close'] - m_s) / (s_s + 1e-10)
    m_l = df['close'].rolling(long_lb).mean()
    s_l = df['close'].rolling(long_lb).std()
    z_long = (df['close'] - m_l) / (s_l + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(z_short < -z_thresh) & (z_long < 0)] = 1
    sig[(z_short > z_thresh) & (z_long > 0)] = -1
    return sig
reg('ZScore_Dual', gen_zscore_dual,
    lambda t: {'short_lb': t.suggest_int('short_lb', 5, 40), 'long_lb': t.suggest_int('long_lb', 50, 200),
               'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_zscore_obv(df, lookback=50, z_thresh=2.0, obv_period=20):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    o = obv_calc(df['close'], df['volume'])
    o_sma = o.rolling(obv_period).mean()
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (o > o_sma)] = 1
    sig[(z > z_thresh) & (o < o_sma)] = -1
    return sig
reg('ZScore_OBV', gen_zscore_obv,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'obv_period': t.suggest_int('obv_period', 10, 40)})

def gen_zscore_heikin(df, lookback=50, z_thresh=2.0):
    ha = heikin_ashi(df)
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    ha_bull = ha['close'] > ha['open']
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & ha_bull] = 1
    sig[(z > z_thresh) & ~ha_bull] = -1
    return sig
reg('ZScore_HeikinAshi', gen_zscore_heikin,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_zscore_vwap_rsi(df, lookback=50, z_thresh=2.0, rsi_buy=30):
    vw = vwap_calc(df)
    dist = (df['close'] - vw) / (vw + 1e-10)
    m = dist.rolling(lookback).mean()
    s = dist.rolling(lookback).std()
    z = (dist - m) / (s + 1e-10)
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (r < rsi_buy)] = 1
    sig[(z > z_thresh) & (r > 100 - rsi_buy)] = -1
    return sig
reg('ZScore_VWAP_RSI', gen_zscore_vwap_rsi,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 100), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40)})

def gen_zscore_mean_revert(df, lookback=50, z_entry=2.0, z_exit=0.5):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[z < -z_entry] = 1
    sig[(z > -z_exit) & (z.shift() < -z_exit)] = -1  # exit when reverting
    sig[z > z_entry] = -1
    return sig
reg('ZScore_MeanRevert', gen_zscore_mean_revert,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150),
               'z_entry': t.suggest_float('z_entry', 1.0, 3.5, step=0.1),
               'z_exit': t.suggest_float('z_exit', 0.0, 1.5, step=0.1)})

def gen_zscore_keltner(df, lookback=50, z_thresh=2.0, kc_period=20, kc_mult=2.0):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    _, kc_up, kc_lo = keltner(df['close'], df['high'], df['low'], kc_period, kc_mult)
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (df['close'] < kc_lo)] = 1
    sig[(z > z_thresh) & (df['close'] > kc_up)] = -1
    return sig
reg('ZScore_Keltner', gen_zscore_keltner,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'kc_period': t.suggest_int('kc_period', 10, 40), 'kc_mult': t.suggest_float('kc_mult', 1.0, 3.5, step=0.25)})

# ==============================================================================
# CATEGORY 3: BB VARIATIONS (25)
# ==============================================================================

def gen_bb_bounce_rsi(df, bb_period=20, bb_std=2.0, rsi_buy=30):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (r < rsi_buy)] = 1
    sig[(df['close'] > upper) & (r > 100 - rsi_buy)] = -1
    return sig
reg('BB_Bounce_RSI', gen_bb_bounce_rsi,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40)})

def gen_bb_bounce_vol(df, bb_period=20, bb_std=2.0, vol_mult=1.5):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    vm = df['volume'].rolling(20).mean()
    high_vol = df['volume'] > vol_mult * vm
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & high_vol & (df['close'] > df['close'].shift())] = 1
    sig[(df['close'] > upper) & high_vol & (df['close'] < df['close'].shift())] = -1
    return sig
reg('BB_Bounce_Vol', gen_bb_bounce_vol,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'vol_mult': t.suggest_float('vol_mult', 1.0, 4.0, step=0.2)})

def gen_bb_squeeze_mr(df, bb_period=20, kc_mult=1.5):
    mid, bb_up, bb_lo = bb(df['close'], bb_period)
    e = ema(df['close'], bb_period)
    a = atr(df['high'], df['low'], df['close'], bb_period)
    kc_up = e + kc_mult * a
    kc_lo = e - kc_mult * a
    sq = (bb_lo > kc_lo) & (bb_up < kc_up)
    released = ~sq & sq.shift().fillna(False)
    mom = df['close'] - mid
    sig = pd.Series(0, index=df.index)
    sig[released & (mom > 0)] = 1
    sig[released & (mom < 0)] = -1
    return sig
reg('BB_Squeeze_MR', gen_bb_squeeze_mr,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 35), 'kc_mult': t.suggest_float('kc_mult', 1.0, 2.5, step=0.25)})

def gen_bb_walk(df, bb_period=20, bb_std=2.0, consec=3):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    above_upper = (df['close'] > upper).rolling(consec).sum() == consec
    below_lower = (df['close'] < lower).rolling(consec).sum() == consec
    sig = pd.Series(0, index=df.index)
    sig[above_upper & (df['close'] < upper)] = -1
    sig[below_lower & (df['close'] > lower)] = 1
    return sig
reg('BB_Walk', gen_bb_walk,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'consec': t.suggest_int('consec', 2, 5)})

def gen_bb_width_filter(df, bb_period=20, bb_std=2.0, width_thresh=0.03):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    width = (upper - lower) / (mid + 1e-10)
    narrow = width < width_thresh
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & narrow] = 1
    sig[(df['close'] > upper) & narrow] = -1
    return sig
reg('BB_Width_Filter', gen_bb_width_filter,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'width_thresh': t.suggest_float('width_thresh', 0.005, 0.06, step=0.005)})

def gen_bb_keltner_combo(df, bb_period=20, kc_mult=2.0):
    mid, bb_up, bb_lo = bb(df['close'], bb_period)
    _, kc_up, kc_lo = keltner(df['close'], df['high'], df['low'], bb_period, kc_mult)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < bb_lo) & (df['close'] < kc_lo)] = 1
    sig[(df['close'] > bb_up) & (df['close'] > kc_up)] = -1
    return sig
reg('BB_Keltner_Combo', gen_bb_keltner_combo,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'kc_mult': t.suggest_float('kc_mult', 1.0, 3.5, step=0.25)})

def gen_bb_vwap(df, bb_period=20, bb_std=2.0):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    vw = vwap_calc(df)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (df['close'] < vw)] = 1
    sig[(df['close'] > upper) & (df['close'] > vw)] = -1
    return sig
reg('BB_VWAP', gen_bb_vwap,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_bb_macd(df, bb_period=20, bb_std=2.0):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    _, _, hist = macd_calc(df['close'])
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (hist > 0)] = 1
    sig[(df['close'] > upper) & (hist < 0)] = -1
    return sig
reg('BB_MACD', gen_bb_macd,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_bb_stoch(df, bb_period=20, bb_std=2.0, stoch_buy=20):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (k < stoch_buy)] = 1
    sig[(df['close'] > upper) & (k > 100 - stoch_buy)] = -1
    return sig
reg('BB_Stoch', gen_bb_stoch,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_bb_cci(df, bb_period=20, bb_std=2.0, cci_level=100):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (c < -cci_level)] = 1
    sig[(df['close'] > upper) & (c > cci_level)] = -1
    return sig
reg('BB_CCI', gen_bb_cci,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'cci_level': t.suggest_int('cci_level', 50, 200)})

def gen_bb_williams(df, bb_period=20, bb_std=2.0, wr_buy=-80):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (wr < wr_buy)] = 1
    sig[(df['close'] > upper) & (wr > -100 - wr_buy)] = -1
    return sig
reg('BB_WilliamsR', gen_bb_williams,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'wr_buy': t.suggest_int('wr_buy', -95, -65)})

def gen_bb_adx(df, bb_period=20, bb_std=2.0, adx_thresh=25):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    a, _, _ = adx_calc(df['high'], df['low'], df['close'], 14)
    ranging = a < adx_thresh
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & ranging] = 1
    sig[(df['close'] > upper) & ranging] = -1
    return sig
reg('BB_ADX_Range', gen_bb_adx,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'adx_thresh': t.suggest_int('adx_thresh', 15, 35)})

def gen_bb_ema_trend(df, bb_period=20, bb_std=2.0, ema_period=50):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    trend = ema(df['close'], ema_period)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (df['close'] > trend)] = 1
    sig[(df['close'] > upper) & (df['close'] < trend)] = -1
    return sig
reg('BB_EMA_Trend', gen_bb_ema_trend,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'ema_period': t.suggest_int('ema_period', 20, 200)})

def gen_bb_momentum(df, bb_period=20, bb_std=2.0, mom_period=10):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    mom = df['close'] / df['close'].shift(mom_period) - 1
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (mom > -0.02)] = 1
    sig[(df['close'] > upper) & (mom < 0.02)] = -1
    return sig
reg('BB_Momentum', gen_bb_momentum,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'mom_period': t.suggest_int('mom_period', 3, 25)})

def gen_bb_obv(df, bb_period=20, bb_std=2.0, obv_period=20):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    o = obv_calc(df['close'], df['volume'])
    o_sma = o.rolling(obv_period).mean()
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (o > o_sma)] = 1
    sig[(df['close'] > upper) & (o < o_sma)] = -1
    return sig
reg('BB_OBV', gen_bb_obv,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'obv_period': t.suggest_int('obv_period', 10, 40)})

def gen_bb_pctb(df, bb_period=20, bb_std=2.0, low_thresh=0.0, high_thresh=1.0):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    pctb = (df['close'] - lower) / (upper - lower + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(pctb < low_thresh) & (pctb > pctb.shift())] = 1
    sig[(pctb > high_thresh) & (pctb < pctb.shift())] = -1
    return sig
reg('BB_PctB', gen_bb_pctb,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'low_thresh': t.suggest_float('low_thresh', -0.2, 0.2, step=0.05),
               'high_thresh': t.suggest_float('high_thresh', 0.8, 1.2, step=0.05)})

def gen_bb_mean_revert(df, bb_period=20, bb_std=2.0):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (df['close'] > df['close'].shift())] = 1
    sig[(df['close'] > mid) & (df['close'].shift() < mid.shift())] = -1  # exit at mid
    return sig
reg('BB_MeanRevert', gen_bb_mean_revert,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_bb_double(df, short_period=10, long_period=30, bb_std=2.0):
    _, su, sl = bb(df['close'], short_period, bb_std)
    _, lu, ll = bb(df['close'], long_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < sl) & (df['close'] < ll)] = 1
    sig[(df['close'] > su) & (df['close'] > lu)] = -1
    return sig
reg('BB_Double', gen_bb_double,
    lambda t: {'short_period': t.suggest_int('short_period', 5, 15), 'long_period': t.suggest_int('long_period', 20, 50),
               'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_bb_rsi_stoch(df, bb_period=20, bb_std=2.0, rsi_buy=30, stoch_buy=20):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    r = rsi(df['close'], 14)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (r < rsi_buy) & (k < stoch_buy)] = 1
    sig[(df['close'] > upper) & (r > 100 - rsi_buy) & (k > 100 - stoch_buy)] = -1
    return sig
reg('BB_RSI_Stoch', gen_bb_rsi_stoch,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_bb_heikin(df, bb_period=20, bb_std=2.0):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    ha = heikin_ashi(df)
    ha_bull = ha['close'] > ha['open']
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & ha_bull] = 1
    sig[(df['close'] > upper) & ~ha_bull] = -1
    return sig
reg('BB_HeikinAshi', gen_bb_heikin,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_bb_atr_adaptive(df, bb_period=20, atr_period=14, atr_mult=1.0):
    a = atr(df['high'], df['low'], df['close'], atr_period)
    atr_norm = a / (df['close'] + 1e-10)
    adaptive_std = 2.0 + atr_mult * atr_norm * 100
    mid = df['close'].rolling(bb_period).mean()
    s = df['close'].rolling(bb_period).std()
    upper = mid + adaptive_std * s
    lower = mid - adaptive_std * s
    sig = pd.Series(0, index=df.index)
    sig[df['close'] < lower] = 1
    sig[df['close'] > upper] = -1
    return sig
reg('BB_ATR_Adaptive', gen_bb_atr_adaptive,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'atr_period': t.suggest_int('atr_period', 7, 28),
               'atr_mult': t.suggest_float('atr_mult', 0.5, 3.0, step=0.25)})

def gen_bb_cmf(df, bb_period=20, bb_std=2.0, cmf_period=20):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    mfv = mfm * df['volume']
    cmf = mfv.rolling(cmf_period).sum() / (df['volume'].rolling(cmf_period).sum() + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (cmf > 0)] = 1
    sig[(df['close'] > upper) & (cmf < 0)] = -1
    return sig
reg('BB_CMF', gen_bb_cmf,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'cmf_period': t.suggest_int('cmf_period', 10, 40)})

# ==============================================================================
# CATEGORY 4: RSI VARIATIONS (25)
# ==============================================================================

def gen_rsi_basic(df, period=14, buy=30, sell=70):
    r = rsi(df['close'], period)
    sig = pd.Series(0, index=df.index)
    sig[r < buy] = 1
    sig[r > sell] = -1
    return sig
reg('RSI_Basic', gen_rsi_basic,
    lambda t: {'period': t.suggest_int('period', 5, 30), 'buy': t.suggest_int('buy', 15, 40), 'sell': t.suggest_int('sell', 60, 85)})

def gen_rsi_macd(df, rsi_buy=30, rsi_sell=70):
    r = rsi(df['close'], 14)
    _, _, hist = macd_calc(df['close'])
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (hist > 0) & (hist.shift() <= 0)] = 1
    sig[(r > rsi_sell) & (hist < 0) & (hist.shift() >= 0)] = -1
    return sig
reg('RSI_MACD', gen_rsi_macd,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 18, 45), 'rsi_sell': t.suggest_int('rsi_sell', 55, 82)})

def gen_rsi_vol(df, rsi_buy=30, vol_mult=1.5):
    r = rsi(df['close'], 14)
    vm = df['volume'].rolling(20).mean()
    high_vol = df['volume'] > vol_mult * vm
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & high_vol] = 1
    sig[(r > 100 - rsi_buy) & high_vol] = -1
    return sig
reg('RSI_Volume', gen_rsi_vol,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'vol_mult': t.suggest_float('vol_mult', 1.0, 4.0, step=0.2)})

def gen_rsi_divergence(df, rsi_period=14, lookback=10):
    r = rsi(df['close'], rsi_period)
    price_lower = df['close'] < df['close'].rolling(lookback).min().shift()
    rsi_higher = r > r.rolling(lookback).min().shift()
    bull_div = price_lower & rsi_higher
    price_higher = df['close'] > df['close'].rolling(lookback).max().shift()
    rsi_lower = r < r.rolling(lookback).max().shift()
    bear_div = price_higher & rsi_lower
    sig = pd.Series(0, index=df.index)
    sig[bull_div] = 1
    sig[bear_div] = -1
    return sig
reg('RSI_Divergence', gen_rsi_divergence,
    lambda t: {'rsi_period': t.suggest_int('rsi_period', 7, 21), 'lookback': t.suggest_int('lookback', 5, 20)})

def gen_rsi_ema(df, rsi_buy=30, ema_period=50):
    r = rsi(df['close'], 14)
    trend = ema(df['close'], ema_period)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (df['close'] > trend)] = 1
    sig[(r > 100 - rsi_buy) & (df['close'] < trend)] = -1
    return sig
reg('RSI_EMA', gen_rsi_ema,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'ema_period': t.suggest_int('ema_period', 20, 200)})

def gen_connors_rsi(df, rsi_period=3, streak_period=2, pctrank_period=100, buy=10, sell=90):
    r = rsi(df['close'], rsi_period)
    streak = pd.Series(0.0, index=df.index)
    up = df['close'] > df['close'].shift()
    down = df['close'] < df['close'].shift()
    streak = up.astype(float) - down.astype(float)
    streak_rsi = rsi(streak.cumsum(), streak_period)
    pct_rank = df['close'].rolling(pctrank_period).apply(lambda x: (x[-1] > x[:-1]).mean() * 100, raw=True)
    crsi = (r + streak_rsi + pct_rank) / 3
    sig = pd.Series(0, index=df.index)
    sig[crsi < buy] = 1
    sig[crsi > sell] = -1
    return sig
reg('ConnorsRSI', gen_connors_rsi,
    lambda t: {'rsi_period': t.suggest_int('rsi_period', 2, 5), 'streak_period': t.suggest_int('streak_period', 2, 5),
               'pctrank_period': t.suggest_int('pctrank_period', 50, 200),
               'buy': t.suggest_int('buy', 5, 25), 'sell': t.suggest_int('sell', 75, 95)})

def gen_rsi_stoch(df, rsi_buy=30, stoch_buy=20):
    r = rsi(df['close'], 14)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (k < stoch_buy)] = 1
    sig[(r > 100 - rsi_buy) & (k > 100 - stoch_buy)] = -1
    return sig
reg('RSI_Stoch', gen_rsi_stoch,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_rsi_dynamic(df, rsi_period=14, lookback=100):
    r = rsi(df['close'], rsi_period)
    r_mean = r.rolling(lookback).mean()
    r_std = r.rolling(lookback).std()
    buy_level = r_mean - r_std
    sell_level = r_mean + r_std
    sig = pd.Series(0, index=df.index)
    sig[r < buy_level] = 1
    sig[r > sell_level] = -1
    return sig
reg('RSI_Dynamic', gen_rsi_dynamic,
    lambda t: {'rsi_period': t.suggest_int('rsi_period', 7, 21), 'lookback': t.suggest_int('lookback', 50, 200)})

def gen_rsi_bb(df, rsi_period=14, bb_period=20, bb_std=2.0):
    r = rsi(df['close'], rsi_period)
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[(r < 30) & (df['close'] < lower)] = 1
    sig[(r > 70) & (df['close'] > upper)] = -1
    return sig
reg('RSI_BB', gen_rsi_bb,
    lambda t: {'rsi_period': t.suggest_int('rsi_period', 7, 21), 'bb_period': t.suggest_int('bb_period', 10, 40),
               'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_rsi_adx(df, rsi_buy=30, adx_thresh=25):
    r = rsi(df['close'], 14)
    a, _, _ = adx_calc(df['high'], df['low'], df['close'], 14)
    ranging = a < adx_thresh
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & ranging] = 1
    sig[(r > 100 - rsi_buy) & ranging] = -1
    return sig
reg('RSI_ADX', gen_rsi_adx,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'adx_thresh': t.suggest_int('adx_thresh', 15, 35)})

def gen_rsi_cci(df, rsi_buy=30, cci_level=100):
    r = rsi(df['close'], 14)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (c < -cci_level)] = 1
    sig[(r > 100 - rsi_buy) & (c > cci_level)] = -1
    return sig
reg('RSI_CCI', gen_rsi_cci,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'cci_level': t.suggest_int('cci_level', 50, 200)})

def gen_rsi_williams(df, rsi_buy=30, wr_buy=-80):
    r = rsi(df['close'], 14)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (wr < wr_buy)] = 1
    sig[(r > 100 - rsi_buy) & (wr > -100 - wr_buy)] = -1
    return sig
reg('RSI_WilliamsR', gen_rsi_williams,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'wr_buy': t.suggest_int('wr_buy', -95, -65)})

def gen_rsi_obv(df, rsi_buy=30, obv_period=20):
    r = rsi(df['close'], 14)
    o = obv_calc(df['close'], df['volume'])
    o_sma = o.rolling(obv_period).mean()
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (o > o_sma)] = 1
    sig[(r > 100 - rsi_buy) & (o < o_sma)] = -1
    return sig
reg('RSI_OBV', gen_rsi_obv,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'obv_period': t.suggest_int('obv_period', 10, 40)})

def gen_rsi_keltner(df, rsi_buy=30, kc_period=20, kc_mult=2.0):
    r = rsi(df['close'], 14)
    _, kc_up, kc_lo = keltner(df['close'], df['high'], df['low'], kc_period, kc_mult)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (df['close'] < kc_lo)] = 1
    sig[(r > 100 - rsi_buy) & (df['close'] > kc_up)] = -1
    return sig
reg('RSI_Keltner', gen_rsi_keltner,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40),
               'kc_period': t.suggest_int('kc_period', 10, 40), 'kc_mult': t.suggest_float('kc_mult', 1.0, 3.5, step=0.25)})

def gen_rsi_cmf(df, rsi_buy=30, cmf_period=20):
    r = rsi(df['close'], 14)
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    mfv = mfm * df['volume']
    cmf = mfv.rolling(cmf_period).sum() / (df['volume'].rolling(cmf_period).sum() + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (cmf > 0)] = 1
    sig[(r > 100 - rsi_buy) & (cmf < 0)] = -1
    return sig
reg('RSI_CMF', gen_rsi_cmf,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'cmf_period': t.suggest_int('cmf_period', 10, 40)})

def gen_rsi_mfi(df, rsi_buy=30, mfi_buy=20, mfi_period=14):
    r = rsi(df['close'], 14)
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos_mf = mf.where(tp > tp.shift(), 0.0).rolling(mfi_period).sum()
    neg_mf = mf.where(tp < tp.shift(), 0.0).rolling(mfi_period).sum()
    mfi = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-10))
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (mfi < mfi_buy)] = 1
    sig[(r > 100 - rsi_buy) & (mfi > 100 - mfi_buy)] = -1
    return sig
reg('RSI_MFI', gen_rsi_mfi,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'mfi_buy': t.suggest_int('mfi_buy', 10, 35),
               'mfi_period': t.suggest_int('mfi_period', 7, 28)})

def gen_rsi_heikin(df, rsi_buy=30):
    r = rsi(df['close'], 14)
    ha = heikin_ashi(df)
    ha_bull = ha['close'] > ha['open']
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & ha_bull] = 1
    sig[(r > 100 - rsi_buy) & ~ha_bull] = -1
    return sig
reg('RSI_HeikinAshi', gen_rsi_heikin,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40)})

def gen_rsi_vwap(df, rsi_buy=30, dist_pct=0.01):
    r = rsi(df['close'], 14)
    vw = vwap_calc(df)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (df['close'] < vw * (1 - dist_pct))] = 1
    sig[(r > 100 - rsi_buy) & (df['close'] > vw * (1 + dist_pct))] = -1
    return sig
reg('RSI_VWAP', gen_rsi_vwap,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001)})

def gen_rsi_zscore(df, rsi_buy=30, lookback=50, z_thresh=2.0):
    r = rsi(df['close'], 14)
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (z < -z_thresh)] = 1
    sig[(r > 100 - rsi_buy) & (z > z_thresh)] = -1
    return sig
reg('RSI_ZScore', gen_rsi_zscore,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40),
               'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_rsi_donchian(df, rsi_buy=30, dc_period=20):
    r = rsi(df['close'], 14)
    dc_lo = df['low'].rolling(dc_period).min()
    dc_hi = df['high'].rolling(dc_period).max()
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (df['close'] <= dc_lo.shift())] = 1
    sig[(r > 100 - rsi_buy) & (df['close'] >= dc_hi.shift())] = -1
    return sig
reg('RSI_Donchian', gen_rsi_donchian,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'dc_period': t.suggest_int('dc_period', 5, 50)})

def gen_rsi_tsi(df, rsi_buy=30, long_p=25, short_p=13):
    r = rsi(df['close'], 14)
    d = df['close'].diff()
    ds = ema(ema(d, long_p), short_p)
    ads = ema(ema(d.abs(), long_p), short_p)
    tsi = 100 * ds / (ads + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (tsi > tsi.shift())] = 1
    sig[(r > 100 - rsi_buy) & (tsi < tsi.shift())] = -1
    return sig
reg('RSI_TSI', gen_rsi_tsi,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40),
               'long_p': t.suggest_int('long_p', 15, 40), 'short_p': t.suggest_int('short_p', 5, 18)})

def gen_rsi_pivot(df, rsi_buy=30):
    r = rsi(df['close'], 14)
    pp = (df['high'].shift() + df['low'].shift() + df['close'].shift()) / 3
    s1 = 2 * pp - df['high'].shift()
    r1 = 2 * pp - df['low'].shift()
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (df['close'] <= s1)] = 1
    sig[(r > 100 - rsi_buy) & (df['close'] >= r1)] = -1
    return sig
reg('RSI_Pivot', gen_rsi_pivot,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40)})

# ==============================================================================
# CATEGORY 5: MULTI-INDICATOR COMBOS (30 - all 2-of-N pairs)
# ==============================================================================

def gen_combo_vwap_bb(df, dist_pct=0.01, bb_period=20, bb_std=2.0):
    vw = vwap_calc(df)
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (df['close'] < lower)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (df['close'] > upper)] = -1
    return sig
reg('Combo_VWAP_BB', gen_combo_vwap_bb,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_combo_vwap_zscore(df, dist_pct=0.01, lookback=50, z_thresh=2.0):
    vw = vwap_calc(df)
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (z < -z_thresh)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (z > z_thresh)] = -1
    return sig
reg('Combo_VWAP_ZScore', gen_combo_vwap_zscore,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_combo_vwap_cci(df, dist_pct=0.01, cci_level=100):
    vw = vwap_calc(df)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (c < -cci_level)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (c > cci_level)] = -1
    return sig
reg('Combo_VWAP_CCI', gen_combo_vwap_cci,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'cci_level': t.suggest_int('cci_level', 50, 200)})

def gen_combo_vwap_macd(df, dist_pct=0.01):
    vw = vwap_calc(df)
    _, _, hist = macd_calc(df['close'])
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (hist > 0)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (hist < 0)] = -1
    return sig
reg('Combo_VWAP_MACD', gen_combo_vwap_macd,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001)})

def gen_combo_rsi_bb(df, rsi_buy=30, bb_period=20, bb_std=2.0):
    r = rsi(df['close'], 14)
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (df['close'] < lower)] = 1
    sig[(r > 100 - rsi_buy) & (df['close'] > upper)] = -1
    return sig
reg('Combo_RSI_BB', gen_combo_rsi_bb,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40),
               'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_combo_rsi_zscore(df, rsi_buy=30, lookback=50, z_thresh=2.0):
    r = rsi(df['close'], 14)
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (z < -z_thresh)] = 1
    sig[(r > 100 - rsi_buy) & (z > z_thresh)] = -1
    return sig
reg('Combo_RSI_ZScore', gen_combo_rsi_zscore,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40),
               'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_combo_rsi_cci(df, rsi_buy=30, cci_level=100):
    r = rsi(df['close'], 14)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (c < -cci_level)] = 1
    sig[(r > 100 - rsi_buy) & (c > cci_level)] = -1
    return sig
reg('Combo_RSI_CCI', gen_combo_rsi_cci,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'cci_level': t.suggest_int('cci_level', 50, 200)})

def gen_combo_rsi_stoch(df, rsi_buy=30, stoch_buy=20):
    r = rsi(df['close'], 14)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (k < stoch_buy)] = 1
    sig[(r > 100 - rsi_buy) & (k > 100 - stoch_buy)] = -1
    return sig
reg('Combo_RSI_Stoch', gen_combo_rsi_stoch,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_combo_rsi_wr(df, rsi_buy=30, wr_buy=-80):
    r = rsi(df['close'], 14)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (wr < wr_buy)] = 1
    sig[(r > 100 - rsi_buy) & (wr > -100 - wr_buy)] = -1
    return sig
reg('Combo_RSI_WR', gen_combo_rsi_wr,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'wr_buy': t.suggest_int('wr_buy', -95, -65)})

def gen_combo_rsi_macd(df, rsi_buy=30):
    r = rsi(df['close'], 14)
    _, _, hist = macd_calc(df['close'])
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (hist > 0)] = 1
    sig[(r > 100 - rsi_buy) & (hist < 0)] = -1
    return sig
reg('Combo_RSI_MACD', gen_combo_rsi_macd,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40)})

def gen_combo_bb_zscore(df, bb_period=20, bb_std=2.0, lookback=50, z_thresh=2.0):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (z < -z_thresh)] = 1
    sig[(df['close'] > upper) & (z > z_thresh)] = -1
    return sig
reg('Combo_BB_ZScore', gen_combo_bb_zscore,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_combo_bb_cci(df, bb_period=20, bb_std=2.0, cci_level=100):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (c < -cci_level)] = 1
    sig[(df['close'] > upper) & (c > cci_level)] = -1
    return sig
reg('Combo_BB_CCI', gen_combo_bb_cci,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'cci_level': t.suggest_int('cci_level', 50, 200)})

def gen_combo_bb_stoch(df, bb_period=20, bb_std=2.0, stoch_buy=20):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (k < stoch_buy)] = 1
    sig[(df['close'] > upper) & (k > 100 - stoch_buy)] = -1
    return sig
reg('Combo_BB_Stoch', gen_combo_bb_stoch,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_combo_bb_wr(df, bb_period=20, bb_std=2.0, wr_buy=-80):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (wr < wr_buy)] = 1
    sig[(df['close'] > upper) & (wr > -100 - wr_buy)] = -1
    return sig
reg('Combo_BB_WR', gen_combo_bb_wr,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'wr_buy': t.suggest_int('wr_buy', -95, -65)})

def gen_combo_bb_macd(df, bb_period=20, bb_std=2.0):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    _, _, hist = macd_calc(df['close'])
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (hist > 0)] = 1
    sig[(df['close'] > upper) & (hist < 0)] = -1
    return sig
reg('Combo_BB_MACD', gen_combo_bb_macd,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_combo_zscore_cci(df, lookback=50, z_thresh=2.0, cci_level=100):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (c < -cci_level)] = 1
    sig[(z > z_thresh) & (c > cci_level)] = -1
    return sig
reg('Combo_ZScore_CCI', gen_combo_zscore_cci,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'cci_level': t.suggest_int('cci_level', 50, 200)})

def gen_combo_zscore_stoch(df, lookback=50, z_thresh=2.0, stoch_buy=20):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (k < stoch_buy)] = 1
    sig[(z > z_thresh) & (k > 100 - stoch_buy)] = -1
    return sig
reg('Combo_ZScore_Stoch', gen_combo_zscore_stoch,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_combo_zscore_wr(df, lookback=50, z_thresh=2.0, wr_buy=-80):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (wr < wr_buy)] = 1
    sig[(z > z_thresh) & (wr > -100 - wr_buy)] = -1
    return sig
reg('Combo_ZScore_WR', gen_combo_zscore_wr,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'wr_buy': t.suggest_int('wr_buy', -95, -65)})

def gen_combo_zscore_macd(df, lookback=50, z_thresh=2.0):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    _, _, hist = macd_calc(df['close'])
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (hist > 0)] = 1
    sig[(z > z_thresh) & (hist < 0)] = -1
    return sig
reg('Combo_ZScore_MACD', gen_combo_zscore_macd,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_combo_cci_stoch(df, cci_level=100, stoch_buy=20):
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    sig = pd.Series(0, index=df.index)
    sig[(c < -cci_level) & (k < stoch_buy)] = 1
    sig[(c > cci_level) & (k > 100 - stoch_buy)] = -1
    return sig
reg('Combo_CCI_Stoch', gen_combo_cci_stoch,
    lambda t: {'cci_level': t.suggest_int('cci_level', 50, 200), 'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_combo_cci_wr(df, cci_level=100, wr_buy=-80):
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(c < -cci_level) & (wr < wr_buy)] = 1
    sig[(c > cci_level) & (wr > -100 - wr_buy)] = -1
    return sig
reg('Combo_CCI_WR', gen_combo_cci_wr,
    lambda t: {'cci_level': t.suggest_int('cci_level', 50, 200), 'wr_buy': t.suggest_int('wr_buy', -95, -65)})

def gen_combo_cci_macd(df, cci_level=100):
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    _, _, hist = macd_calc(df['close'])
    sig = pd.Series(0, index=df.index)
    sig[(c < -cci_level) & (hist > 0)] = 1
    sig[(c > cci_level) & (hist < 0)] = -1
    return sig
reg('Combo_CCI_MACD', gen_combo_cci_macd,
    lambda t: {'cci_level': t.suggest_int('cci_level', 50, 200)})

def gen_combo_stoch_wr(df, stoch_buy=20, wr_buy=-80):
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(k < stoch_buy) & (wr < wr_buy)] = 1
    sig[(k > 100 - stoch_buy) & (wr > -100 - wr_buy)] = -1
    return sig
reg('Combo_Stoch_WR', gen_combo_stoch_wr,
    lambda t: {'stoch_buy': t.suggest_int('stoch_buy', 8, 30), 'wr_buy': t.suggest_int('wr_buy', -95, -65)})

def gen_combo_stoch_macd(df, stoch_buy=20):
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    _, _, hist = macd_calc(df['close'])
    sig = pd.Series(0, index=df.index)
    sig[(k < stoch_buy) & (hist > 0)] = 1
    sig[(k > 100 - stoch_buy) & (hist < 0)] = -1
    return sig
reg('Combo_Stoch_MACD', gen_combo_stoch_macd,
    lambda t: {'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_combo_wr_macd(df, wr_buy=-80):
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    _, _, hist = macd_calc(df['close'])
    sig = pd.Series(0, index=df.index)
    sig[(wr < wr_buy) & (hist > 0)] = 1
    sig[(wr > -100 - wr_buy) & (hist < 0)] = -1
    return sig
reg('Combo_WR_MACD', gen_combo_wr_macd,
    lambda t: {'wr_buy': t.suggest_int('wr_buy', -95, -65)})

def gen_combo_vwap_stoch(df, dist_pct=0.01, stoch_buy=20):
    vw = vwap_calc(df)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (k < stoch_buy)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (k > 100 - stoch_buy)] = -1
    return sig
reg('Combo_VWAP_Stoch', gen_combo_vwap_stoch,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_combo_vwap_wr(df, dist_pct=0.01, wr_buy=-80):
    vw = vwap_calc(df)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (wr < wr_buy)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (wr > -100 - wr_buy)] = -1
    return sig
reg('Combo_VWAP_WR', gen_combo_vwap_wr,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'wr_buy': t.suggest_int('wr_buy', -95, -65)})

# ==============================================================================
# CATEGORY 6: ENSEMBLE / VOTING STRATEGIES (20)
# ==============================================================================

def _vote_signals(signals_list, threshold):
    """Sum of signals from multiple indicator lists, threshold for agreement."""
    combined = sum(signals_list)
    sig = pd.Series(0, index=combined.index)
    sig[combined >= threshold] = 1
    sig[combined <= -threshold] = -1
    return sig

def gen_vote_3of5_mr(df, rsi_buy=30, z_thresh=2.0, bb_std=2.0, stoch_buy=20, cci_level=100):
    r = rsi(df['close'], 14)
    m = df['close'].rolling(50).mean()
    s = df['close'].rolling(50).std()
    z = (df['close'] - m) / (s + 1e-10)
    mid, upper, lower = bb(df['close'], 20, bb_std)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    buy_votes = (r < rsi_buy).astype(int) + (z < -z_thresh).astype(int) + (df['close'] < lower).astype(int) + (k < stoch_buy).astype(int) + (c < -cci_level).astype(int)
    sell_votes = (r > 100 - rsi_buy).astype(int) + (z > z_thresh).astype(int) + (df['close'] > upper).astype(int) + (k > 100 - stoch_buy).astype(int) + (c > cci_level).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_votes >= 3] = 1
    sig[sell_votes >= 3] = -1
    return sig
reg('Vote_3of5_MR', gen_vote_3of5_mr,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'stoch_buy': t.suggest_int('stoch_buy', 8, 30), 'cci_level': t.suggest_int('cci_level', 50, 200)})

def gen_vote_2of3_rsi_bb_z(df, rsi_buy=30, bb_std=2.0, z_thresh=2.0):
    r = rsi(df['close'], 14)
    mid, upper, lower = bb(df['close'], 20, bb_std)
    m = df['close'].rolling(50).mean()
    s = df['close'].rolling(50).std()
    z = (df['close'] - m) / (s + 1e-10)
    buy_v = (r < rsi_buy).astype(int) + (df['close'] < lower).astype(int) + (z < -z_thresh).astype(int)
    sell_v = (r > 100 - rsi_buy).astype(int) + (df['close'] > upper).astype(int) + (z > z_thresh).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 2] = 1
    sig[sell_v >= 2] = -1
    return sig
reg('Vote_2of3_RSI_BB_Z', gen_vote_2of3_rsi_bb_z,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_vote_2of3_vwap_rsi_stoch(df, dist_pct=0.01, rsi_buy=30, stoch_buy=20):
    vw = vwap_calc(df)
    r = rsi(df['close'], 14)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    buy_v = (df['close'] < vw * (1 - dist_pct)).astype(int) + (r < rsi_buy).astype(int) + (k < stoch_buy).astype(int)
    sell_v = (df['close'] > vw * (1 + dist_pct)).astype(int) + (r > 100 - rsi_buy).astype(int) + (k > 100 - stoch_buy).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 2] = 1
    sig[sell_v >= 2] = -1
    return sig
reg('Vote_2of3_VWAP_RSI_Stoch', gen_vote_2of3_vwap_rsi_stoch,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_vote_2of3_vwap_bb_cci(df, dist_pct=0.01, bb_std=2.0, cci_level=100):
    vw = vwap_calc(df)
    mid, upper, lower = bb(df['close'], 20, bb_std)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    buy_v = (df['close'] < vw * (1 - dist_pct)).astype(int) + (df['close'] < lower).astype(int) + (c < -cci_level).astype(int)
    sell_v = (df['close'] > vw * (1 + dist_pct)).astype(int) + (df['close'] > upper).astype(int) + (c > cci_level).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 2] = 1
    sig[sell_v >= 2] = -1
    return sig
reg('Vote_2of3_VWAP_BB_CCI', gen_vote_2of3_vwap_bb_cci,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25), 'cci_level': t.suggest_int('cci_level', 50, 200)})

def gen_vote_4of5_strict(df, rsi_buy=25, z_thresh=2.0, bb_std=2.0, stoch_buy=15, cci_level=150):
    r = rsi(df['close'], 14)
    m = df['close'].rolling(50).mean()
    s = df['close'].rolling(50).std()
    z = (df['close'] - m) / (s + 1e-10)
    mid, upper, lower = bb(df['close'], 20, bb_std)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    buy_v = (r < rsi_buy).astype(int) + (z < -z_thresh).astype(int) + (df['close'] < lower).astype(int) + (k < stoch_buy).astype(int) + (c < -cci_level).astype(int)
    sell_v = (r > 100 - rsi_buy).astype(int) + (z > z_thresh).astype(int) + (df['close'] > upper).astype(int) + (k > 100 - stoch_buy).astype(int) + (c > cci_level).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 4] = 1
    sig[sell_v >= 4] = -1
    return sig
reg('Vote_4of5_Strict', gen_vote_4of5_strict,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 10, 35), 'z_thresh': t.suggest_float('z_thresh', 1.0, 4.0, step=0.1),
               'bb_std': t.suggest_float('bb_std', 1.5, 4.0, step=0.25),
               'stoch_buy': t.suggest_int('stoch_buy', 5, 25), 'cci_level': t.suggest_int('cci_level', 100, 250)})

def gen_vote_weighted_vwap(df, dist_pct=0.01, rsi_buy=30, bb_std=2.0, vwap_weight=2.0):
    vw = vwap_calc(df)
    r = rsi(df['close'], 14)
    mid, upper, lower = bb(df['close'], 20, bb_std)
    buy_score = vwap_weight * (df['close'] < vw * (1 - dist_pct)).astype(float) + (r < rsi_buy).astype(float) + (df['close'] < lower).astype(float)
    sell_score = vwap_weight * (df['close'] > vw * (1 + dist_pct)).astype(float) + (r > 100 - rsi_buy).astype(float) + (df['close'] > upper).astype(float)
    thresh = 1 + vwap_weight
    sig = pd.Series(0, index=df.index)
    sig[buy_score >= thresh] = 1
    sig[sell_score >= thresh] = -1
    return sig
reg('Vote_Weighted_VWAP', gen_vote_weighted_vwap,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'vwap_weight': t.suggest_float('vwap_weight', 1.0, 4.0, step=0.5)})

def gen_vote_2of3_z_stoch_wr(df, z_thresh=2.0, stoch_buy=20, wr_buy=-80):
    m = df['close'].rolling(50).mean()
    s = df['close'].rolling(50).std()
    z = (df['close'] - m) / (s + 1e-10)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    buy_v = (z < -z_thresh).astype(int) + (k < stoch_buy).astype(int) + (wr < wr_buy).astype(int)
    sell_v = (z > z_thresh).astype(int) + (k > 100 - stoch_buy).astype(int) + (wr > -100 - wr_buy).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 2] = 1
    sig[sell_v >= 2] = -1
    return sig
reg('Vote_2of3_Z_Stoch_WR', gen_vote_2of3_z_stoch_wr,
    lambda t: {'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'stoch_buy': t.suggest_int('stoch_buy', 8, 30), 'wr_buy': t.suggest_int('wr_buy', -95, -65)})

def gen_vote_2of3_rsi_cci_macd(df, rsi_buy=30, cci_level=100):
    r = rsi(df['close'], 14)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    _, _, hist = macd_calc(df['close'])
    buy_v = (r < rsi_buy).astype(int) + (c < -cci_level).astype(int) + (hist > 0).astype(int)
    sell_v = (r > 100 - rsi_buy).astype(int) + (c > cci_level).astype(int) + (hist < 0).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 2] = 1
    sig[sell_v >= 2] = -1
    return sig
reg('Vote_2of3_RSI_CCI_MACD', gen_vote_2of3_rsi_cci_macd,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'cci_level': t.suggest_int('cci_level', 50, 200)})

def gen_vote_3of5_with_vol(df, rsi_buy=30, z_thresh=2.0, bb_std=2.0, stoch_buy=20, vol_mult=1.3):
    r = rsi(df['close'], 14)
    m = df['close'].rolling(50).mean()
    s = df['close'].rolling(50).std()
    z = (df['close'] - m) / (s + 1e-10)
    mid, upper, lower = bb(df['close'], 20, bb_std)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    vm = df['volume'].rolling(20).mean()
    high_vol = df['volume'] > vol_mult * vm
    buy_v = (r < rsi_buy).astype(int) + (z < -z_thresh).astype(int) + (df['close'] < lower).astype(int) + (k < stoch_buy).astype(int) + high_vol.astype(int)
    sell_v = (r > 100 - rsi_buy).astype(int) + (z > z_thresh).astype(int) + (df['close'] > upper).astype(int) + (k > 100 - stoch_buy).astype(int) + high_vol.astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 3] = 1
    sig[sell_v >= 3] = -1
    return sig
reg('Vote_3of5_Vol', gen_vote_3of5_with_vol,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'stoch_buy': t.suggest_int('stoch_buy', 8, 30), 'vol_mult': t.suggest_float('vol_mult', 1.0, 3.5, step=0.2)})

def gen_vote_adaptive_vol(df, rsi_buy=30, bb_std=2.0, z_thresh=2.0, vol_period=20):
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    high_vol_regime = vol > vol_ma
    r = rsi(df['close'], 14)
    mid, upper, lower = bb(df['close'], 20, bb_std)
    m = df['close'].rolling(50).mean()
    s = df['close'].rolling(50).std()
    z = (df['close'] - m) / (s + 1e-10)
    # In high vol: need 2 of 3, in low vol: need 1 of 3
    buy_v = (r < rsi_buy).astype(int) + (df['close'] < lower).astype(int) + (z < -z_thresh).astype(int)
    sell_v = (r > 100 - rsi_buy).astype(int) + (df['close'] > upper).astype(int) + (z > z_thresh).astype(int)
    thresh_buy = np.where(high_vol_regime, 2, 1)
    thresh_sell = np.where(high_vol_regime, 2, 1)
    sig = pd.Series(0, index=df.index)
    sig[buy_v.values >= thresh_buy] = 1
    sig[sell_v.values >= thresh_sell] = -1
    return sig
reg('Vote_Adaptive_Vol', gen_vote_adaptive_vol,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'vol_period': t.suggest_int('vol_period', 10, 40)})

def gen_vote_2of3_vwap_z_rsi(df, dist_pct=0.01, z_thresh=2.0, rsi_buy=30):
    vw = vwap_calc(df)
    m = df['close'].rolling(50).mean()
    s = df['close'].rolling(50).std()
    z = (df['close'] - m) / (s + 1e-10)
    r = rsi(df['close'], 14)
    buy_v = (df['close'] < vw * (1 - dist_pct)).astype(int) + (z < -z_thresh).astype(int) + (r < rsi_buy).astype(int)
    sell_v = (df['close'] > vw * (1 + dist_pct)).astype(int) + (z > z_thresh).astype(int) + (r > 100 - rsi_buy).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 2] = 1
    sig[sell_v >= 2] = -1
    return sig
reg('Vote_2of3_VWAP_Z_RSI', gen_vote_2of3_vwap_z_rsi,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1), 'rsi_buy': t.suggest_int('rsi_buy', 15, 40)})

def gen_vote_3of5_all_mr(df, rsi_buy=30, z_thresh=2.0, dist_pct=0.01, stoch_buy=20, wr_buy=-80):
    vw = vwap_calc(df)
    r = rsi(df['close'], 14)
    m = df['close'].rolling(50).mean()
    s = df['close'].rolling(50).std()
    z = (df['close'] - m) / (s + 1e-10)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    buy_v = (df['close'] < vw * (1 - dist_pct)).astype(int) + (r < rsi_buy).astype(int) + (z < -z_thresh).astype(int) + (k < stoch_buy).astype(int) + (wr < wr_buy).astype(int)
    sell_v = (df['close'] > vw * (1 + dist_pct)).astype(int) + (r > 100 - rsi_buy).astype(int) + (z > z_thresh).astype(int) + (k > 100 - stoch_buy).astype(int) + (wr > -100 - wr_buy).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 3] = 1
    sig[sell_v >= 3] = -1
    return sig
reg('Vote_3of5_AllMR', gen_vote_3of5_all_mr,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'stoch_buy': t.suggest_int('stoch_buy', 8, 30), 'wr_buy': t.suggest_int('wr_buy', -95, -65)})

def gen_vote_weighted_rsi_z(df, rsi_buy=30, z_thresh=2.0, rsi_weight=2.0, bb_std=2.0):
    r = rsi(df['close'], 14)
    m = df['close'].rolling(50).mean()
    s = df['close'].rolling(50).std()
    z = (df['close'] - m) / (s + 1e-10)
    mid, upper, lower = bb(df['close'], 20, bb_std)
    buy_score = rsi_weight * (r < rsi_buy).astype(float) + (z < -z_thresh).astype(float) + (df['close'] < lower).astype(float)
    sell_score = rsi_weight * (r > 100 - rsi_buy).astype(float) + (z > z_thresh).astype(float) + (df['close'] > upper).astype(float)
    thresh = 1 + rsi_weight
    sig = pd.Series(0, index=df.index)
    sig[buy_score >= thresh] = 1
    sig[sell_score >= thresh] = -1
    return sig
reg('Vote_Weighted_RSI', gen_vote_weighted_rsi_z,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'rsi_weight': t.suggest_float('rsi_weight', 1.0, 4.0, step=0.5), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_vote_2of4_quad(df, rsi_buy=30, z_thresh=2.0, cci_level=100, stoch_buy=20):
    r = rsi(df['close'], 14)
    m = df['close'].rolling(50).mean()
    s = df['close'].rolling(50).std()
    z = (df['close'] - m) / (s + 1e-10)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    buy_v = (r < rsi_buy).astype(int) + (z < -z_thresh).astype(int) + (c < -cci_level).astype(int) + (k < stoch_buy).astype(int)
    sell_v = (r > 100 - rsi_buy).astype(int) + (z > z_thresh).astype(int) + (c > cci_level).astype(int) + (k > 100 - stoch_buy).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 2] = 1
    sig[sell_v >= 2] = -1
    return sig
reg('Vote_2of4_Quad', gen_vote_2of4_quad,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'cci_level': t.suggest_int('cci_level', 50, 200), 'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_vote_3of4_quad(df, rsi_buy=30, z_thresh=2.0, cci_level=100, stoch_buy=20):
    r = rsi(df['close'], 14)
    m = df['close'].rolling(50).mean()
    s = df['close'].rolling(50).std()
    z = (df['close'] - m) / (s + 1e-10)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    buy_v = (r < rsi_buy).astype(int) + (z < -z_thresh).astype(int) + (c < -cci_level).astype(int) + (k < stoch_buy).astype(int)
    sell_v = (r > 100 - rsi_buy).astype(int) + (z > z_thresh).astype(int) + (c > cci_level).astype(int) + (k > 100 - stoch_buy).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 3] = 1
    sig[sell_v >= 3] = -1
    return sig
reg('Vote_3of4_Quad', gen_vote_3of4_quad,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'cci_level': t.suggest_int('cci_level', 50, 200), 'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_vote_consensus_vwap_bb_z_rsi(df, dist_pct=0.01, bb_std=2.0, z_thresh=2.0, rsi_buy=30, min_agree=3):
    vw = vwap_calc(df)
    mid, upper, lower = bb(df['close'], 20, bb_std)
    m = df['close'].rolling(50).mean()
    s = df['close'].rolling(50).std()
    z = (df['close'] - m) / (s + 1e-10)
    r = rsi(df['close'], 14)
    buy_v = (df['close'] < vw * (1 - dist_pct)).astype(int) + (df['close'] < lower).astype(int) + (z < -z_thresh).astype(int) + (r < rsi_buy).astype(int)
    sell_v = (df['close'] > vw * (1 + dist_pct)).astype(int) + (df['close'] > upper).astype(int) + (z > z_thresh).astype(int) + (r > 100 - rsi_buy).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= min_agree] = 1
    sig[sell_v >= min_agree] = -1
    return sig
reg('Vote_Consensus_4', gen_vote_consensus_vwap_bb_z_rsi,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'min_agree': t.suggest_int('min_agree', 2, 4)})

# ==============================================================================
# CATEGORY 7: ADAPTIVE STRATEGIES (20)
# ==============================================================================

def gen_adaptive_rsi(df, rsi_period=14, vol_period=20, base_buy=30, base_sell=70):
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    ratio = (vol / (vol_ma + 1e-10)).clip(0.5, 2.0)
    r = rsi(df['close'], rsi_period)
    buy_level = base_buy * ratio
    sell_level = 100 - (100 - base_sell) * ratio
    sig = pd.Series(0, index=df.index)
    sig[r < buy_level] = 1
    sig[r > sell_level] = -1
    return sig
reg('Adaptive_RSI', gen_adaptive_rsi,
    lambda t: {'rsi_period': t.suggest_int('rsi_period', 7, 21), 'vol_period': t.suggest_int('vol_period', 10, 40),
               'base_buy': t.suggest_int('base_buy', 15, 40), 'base_sell': t.suggest_int('base_sell', 60, 85)})

def gen_adaptive_bb(df, bb_period=20, vol_period=20, base_std=2.0):
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    ratio = (vol / (vol_ma + 1e-10)).clip(0.5, 2.0)
    adaptive_std = base_std * ratio
    mid = df['close'].rolling(bb_period).mean()
    s = df['close'].rolling(bb_period).std()
    upper = mid + adaptive_std * s
    lower = mid - adaptive_std * s
    sig = pd.Series(0, index=df.index)
    sig[df['close'] < lower] = 1
    sig[df['close'] > upper] = -1
    return sig
reg('Adaptive_BB', gen_adaptive_bb,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'vol_period': t.suggest_int('vol_period', 10, 40),
               'base_std': t.suggest_float('base_std', 1.0, 3.5, step=0.25)})

def gen_adaptive_vwap(df, base_dist=0.01, atr_period=14, atr_mult=1.5):
    vw = vwap_calc(df)
    a = atr(df['high'], df['low'], df['close'], atr_period)
    adaptive_dist = base_dist + atr_mult * a / (df['close'] + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - adaptive_dist)) & (df['close'] > df['close'].shift())] = 1
    sig[(df['close'] > vw * (1 + adaptive_dist)) & (df['close'] < df['close'].shift())] = -1
    return sig
reg('Adaptive_VWAP', gen_adaptive_vwap,
    lambda t: {'base_dist': t.suggest_float('base_dist', 0.001, 0.04, step=0.001),
               'atr_period': t.suggest_int('atr_period', 7, 28), 'atr_mult': t.suggest_float('atr_mult', 0.5, 4.0, step=0.25)})

def gen_adaptive_zscore(df, base_lookback=50, z_thresh=2.0, vol_period=20):
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    ratio = (vol / (vol_ma + 1e-10)).clip(0.5, 2.0)
    m = df['close'].rolling(base_lookback).mean()
    s = df['close'].rolling(base_lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    adj_thresh = z_thresh * ratio
    sig = pd.Series(0, index=df.index)
    sig[z < -adj_thresh] = 1
    sig[z > adj_thresh] = -1
    return sig
reg('Adaptive_ZScore', gen_adaptive_zscore,
    lambda t: {'base_lookback': t.suggest_int('base_lookback', 15, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'vol_period': t.suggest_int('vol_period', 10, 40)})

def gen_adaptive_stoch(df, k_period=14, vol_period=20, base_buy=20, base_sell=80):
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    ratio = (vol / (vol_ma + 1e-10)).clip(0.5, 2.0)
    k, _ = stoch(df['high'], df['low'], df['close'], k_period, 3)
    buy_level = base_buy * ratio
    sell_level = 100 - (100 - base_sell) * ratio
    sig = pd.Series(0, index=df.index)
    sig[k < buy_level] = 1
    sig[k > sell_level] = -1
    return sig
reg('Adaptive_Stoch', gen_adaptive_stoch,
    lambda t: {'k_period': t.suggest_int('k_period', 5, 28), 'vol_period': t.suggest_int('vol_period', 10, 40),
               'base_buy': t.suggest_int('base_buy', 8, 30), 'base_sell': t.suggest_int('base_sell', 70, 92)})

def gen_adaptive_cci(df, cci_period=20, vol_period=20, base_level=100):
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    ratio = (vol / (vol_ma + 1e-10)).clip(0.5, 2.0)
    c = cci_calc(df['high'], df['low'], df['close'], cci_period)
    adj_level = base_level * ratio
    sig = pd.Series(0, index=df.index)
    sig[(c < -adj_level) & (c > c.shift())] = 1
    sig[(c > adj_level) & (c < c.shift())] = -1
    return sig
reg('Adaptive_CCI', gen_adaptive_cci,
    lambda t: {'cci_period': t.suggest_int('cci_period', 10, 30), 'vol_period': t.suggest_int('vol_period', 10, 40),
               'base_level': t.suggest_int('base_level', 50, 200)})

def gen_adaptive_wr(df, wr_period=14, vol_period=20, base_buy=-80):
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    ratio = (vol / (vol_ma + 1e-10)).clip(0.5, 2.0)
    wr = williams_r(df['high'], df['low'], df['close'], wr_period)
    adj_buy = base_buy * ratio
    adj_sell = (-100 - base_buy) * ratio
    sig = pd.Series(0, index=df.index)
    sig[wr < adj_buy] = 1
    sig[wr > adj_sell] = -1
    return sig
reg('Adaptive_WilliamsR', gen_adaptive_wr,
    lambda t: {'wr_period': t.suggest_int('wr_period', 5, 28), 'vol_period': t.suggest_int('vol_period', 10, 40),
               'base_buy': t.suggest_int('base_buy', -95, -65)})

def gen_adaptive_keltner(df, kc_period=20, vol_period=20, base_mult=2.0):
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    ratio = (vol / (vol_ma + 1e-10)).clip(0.5, 2.0)
    e = ema(df['close'], kc_period)
    a = atr(df['high'], df['low'], df['close'], kc_period)
    adj_mult = base_mult * ratio
    upper = e + adj_mult * a
    lower = e - adj_mult * a
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (df['close'] > df['close'].shift())] = 1
    sig[(df['close'] > upper) & (df['close'] < df['close'].shift())] = -1
    return sig
reg('Adaptive_Keltner', gen_adaptive_keltner,
    lambda t: {'kc_period': t.suggest_int('kc_period', 10, 40), 'vol_period': t.suggest_int('vol_period', 10, 40),
               'base_mult': t.suggest_float('base_mult', 1.0, 3.5, step=0.25)})

def gen_adaptive_macd(df, fast=12, slow=26, signal=9, vol_period=20):
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    high_vol = vol > vol_ma
    _, _, hist = macd_calc(df['close'], fast, slow, signal)
    sig = pd.Series(0, index=df.index)
    # In high vol require stronger signal
    sig[(hist > 0) & (hist.shift() <= 0) & ~high_vol] = 1
    sig[(hist > hist.rolling(3).mean()) & (hist.shift() <= hist.shift().rolling(3).mean()) & high_vol] = 1
    sig[(hist < 0) & (hist.shift() >= 0) & ~high_vol] = -1
    sig[(hist < hist.rolling(3).mean()) & (hist.shift() >= hist.shift().rolling(3).mean()) & high_vol] = -1
    return sig
reg('Adaptive_MACD', gen_adaptive_macd,
    lambda t: {'fast': t.suggest_int('fast', 4, 18), 'slow': t.suggest_int('slow', 18, 45),
               'signal': t.suggest_int('signal', 4, 18), 'vol_period': t.suggest_int('vol_period', 10, 40)})

def gen_adaptive_rsi_atr(df, rsi_period=14, atr_period=14, atr_mult=1.0, base_buy=30):
    r = rsi(df['close'], rsi_period)
    a = atr(df['high'], df['low'], df['close'], atr_period)
    a_norm = a / (df['close'] + 1e-10)
    a_ma = a_norm.rolling(20).mean()
    ratio = (a_norm / (a_ma + 1e-10)).clip(0.5, 2.0)
    buy_level = base_buy + atr_mult * (ratio - 1) * 20
    sell_level = (100 - base_buy) - atr_mult * (ratio - 1) * 20
    sig = pd.Series(0, index=df.index)
    sig[r < buy_level] = 1
    sig[r > sell_level] = -1
    return sig
reg('Adaptive_RSI_ATR', gen_adaptive_rsi_atr,
    lambda t: {'rsi_period': t.suggest_int('rsi_period', 7, 21), 'atr_period': t.suggest_int('atr_period', 7, 28),
               'atr_mult': t.suggest_float('atr_mult', 0.5, 3.0, step=0.25), 'base_buy': t.suggest_int('base_buy', 15, 40)})

def gen_adaptive_vwap_vol_regime(df, dist_pct=0.01, vol_period=20):
    vw = vwap_calc(df)
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    ratio = (vol / (vol_ma + 1e-10)).clip(0.5, 2.0)
    adaptive_dist = dist_pct * ratio
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - adaptive_dist)) & (df['close'] > df['close'].shift())] = 1
    sig[(df['close'] > vw * (1 + adaptive_dist)) & (df['close'] < df['close'].shift())] = -1
    return sig
reg('Adaptive_VWAP_VolRegime', gen_adaptive_vwap_vol_regime,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'vol_period': t.suggest_int('vol_period', 10, 40)})

def gen_adaptive_bb_vol_regime(df, bb_period=20, vol_period=20, base_std=2.0):
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    high_vol = vol > vol_ma * 1.2
    mid, upper_normal, lower_normal = bb(df['close'], bb_period, base_std)
    _, upper_wide, lower_wide = bb(df['close'], bb_period, base_std * 1.5)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower_normal) & ~high_vol] = 1
    sig[(df['close'] < lower_wide) & high_vol] = 1
    sig[(df['close'] > upper_normal) & ~high_vol] = -1
    sig[(df['close'] > upper_wide) & high_vol] = -1
    return sig
reg('Adaptive_BB_VolRegime', gen_adaptive_bb_vol_regime,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'vol_period': t.suggest_int('vol_period', 10, 40),
               'base_std': t.suggest_float('base_std', 1.0, 3.0, step=0.25)})

def gen_adaptive_trend_range(df, ema_period=50, adx_period=14, adx_thresh=25, rsi_buy=30, dist_pct=0.01):
    a, pdi, mdi = adx_calc(df['high'], df['low'], df['close'], adx_period)
    trending = a > adx_thresh
    trend = ema(df['close'], ema_period)
    r = rsi(df['close'], 14)
    vw = vwap_calc(df)
    sig = pd.Series(0, index=df.index)
    # Trending: follow trend
    sig[trending & (pdi > mdi) & (df['close'] > trend) & (r < 50)] = 1
    sig[trending & (mdi > pdi) & (df['close'] < trend) & (r > 50)] = -1
    # Ranging: mean reversion
    sig[~trending & (df['close'] < vw * (1 - dist_pct)) & (r < rsi_buy)] = 1
    sig[~trending & (df['close'] > vw * (1 + dist_pct)) & (r > 100 - rsi_buy)] = -1
    return sig
reg('Adaptive_TrendRange', gen_adaptive_trend_range,
    lambda t: {'ema_period': t.suggest_int('ema_period', 20, 100), 'adx_period': t.suggest_int('adx_period', 7, 21),
               'adx_thresh': t.suggest_int('adx_thresh', 15, 35), 'rsi_buy': t.suggest_int('rsi_buy', 15, 40),
               'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001)})

def gen_adaptive_lookback_bb(df, min_lb=10, max_lb=60, vol_period=20, bb_std=2.0):
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    ratio = (vol / (vol_ma + 1e-10)).clip(0.5, 2.0)
    # High vol => shorter lookback, low vol => longer
    # Use median lookback as approximation
    lb = int((min_lb + max_lb) / 2)
    mid, upper, lower = bb(df['close'], lb, bb_std)
    # Adjust bands with ratio
    adj_upper = mid + bb_std * ratio * (upper - mid) / bb_std
    adj_lower = mid - bb_std * ratio * (mid - lower) / bb_std
    sig = pd.Series(0, index=df.index)
    sig[df['close'] < adj_lower] = 1
    sig[df['close'] > adj_upper] = -1
    return sig
reg('Adaptive_Lookback_BB', gen_adaptive_lookback_bb,
    lambda t: {'min_lb': t.suggest_int('min_lb', 5, 15), 'max_lb': t.suggest_int('max_lb', 30, 100),
               'vol_period': t.suggest_int('vol_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_adaptive_vol_scale_rsi(df, rsi_period=14, vol_lookback=50):
    r = rsi(df['close'], rsi_period)
    vol_ratio = df['volume'] / (df['volume'].rolling(vol_lookback).mean() + 1e-10)
    vol_ratio = vol_ratio.clip(0.5, 3.0)
    # High volume => tighter levels (more confident signals)
    buy_level = 30 / vol_ratio
    sell_level = 100 - 30 / vol_ratio
    sig = pd.Series(0, index=df.index)
    sig[r < buy_level] = 1
    sig[r > sell_level] = -1
    return sig
reg('Adaptive_VolScale_RSI', gen_adaptive_vol_scale_rsi,
    lambda t: {'rsi_period': t.suggest_int('rsi_period', 7, 21), 'vol_lookback': t.suggest_int('vol_lookback', 20, 100)})

def gen_adaptive_regime_mr(df, lookback=50, z_thresh=2.0, adx_thresh=25, rsi_buy=30):
    a, _, _ = adx_calc(df['high'], df['low'], df['close'], 14)
    ranging = a < adx_thresh
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    # Only MR in ranging
    sig[(z < -z_thresh) & ranging & (r < rsi_buy)] = 1
    sig[(z > z_thresh) & ranging & (r > 100 - rsi_buy)] = -1
    return sig
reg('Adaptive_Regime_MR', gen_adaptive_regime_mr,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'adx_thresh': t.suggest_int('adx_thresh', 15, 35), 'rsi_buy': t.suggest_int('rsi_buy', 15, 40)})

def gen_adaptive_vwap_rsi_atr(df, base_dist=0.005, atr_mult=1.0, rsi_buy=30, atr_period=14):
    vw = vwap_calc(df)
    a = atr(df['high'], df['low'], df['close'], atr_period)
    r = rsi(df['close'], 14)
    adaptive_dist = base_dist + atr_mult * a / (df['close'] + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - adaptive_dist)) & (r < rsi_buy)] = 1
    sig[(df['close'] > vw * (1 + adaptive_dist)) & (r > 100 - rsi_buy)] = -1
    return sig
reg('Adaptive_VWAP_RSI_ATR', gen_adaptive_vwap_rsi_atr,
    lambda t: {'base_dist': t.suggest_float('base_dist', 0.001, 0.03, step=0.001),
               'atr_mult': t.suggest_float('atr_mult', 0.5, 4.0, step=0.25),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'atr_period': t.suggest_int('atr_period', 7, 28)})

def gen_adaptive_bb_rsi_regime(df, bb_period=20, rsi_buy=30, adx_thresh=25):
    a_val, _, _ = adx_calc(df['high'], df['low'], df['close'], 14)
    ranging = a_val < adx_thresh
    vol = df['close'].rolling(20).std() / (df['close'].rolling(20).mean() + 1e-10)
    vol_ma = vol.rolling(20).mean()
    ratio = (vol / (vol_ma + 1e-10)).clip(0.5, 2.0)
    adj_std = 2.0 * ratio
    mid = df['close'].rolling(bb_period).mean()
    s = df['close'].rolling(bb_period).std()
    upper = mid + adj_std * s
    lower = mid - adj_std * s
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (r < rsi_buy) & ranging] = 1
    sig[(df['close'] > upper) & (r > 100 - rsi_buy) & ranging] = -1
    return sig
reg('Adaptive_BB_RSI_Regime', gen_adaptive_bb_rsi_regime,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'rsi_buy': t.suggest_int('rsi_buy', 15, 40),
               'adx_thresh': t.suggest_int('adx_thresh', 15, 35)})

# ==============================================================================
# CATEGORY 8: PATTERN + INDICATOR (15)
# ==============================================================================

def gen_engulfing_rsi(df, rsi_filter=40):
    bull = (df['close'] > df['open']) & (df['close'].shift() < df['open'].shift()) & \
           (df['close'] > df['open'].shift()) & (df['open'] < df['close'].shift())
    bear = (df['close'] < df['open']) & (df['close'].shift() > df['open'].shift()) & \
           (df['close'] < df['open'].shift()) & (df['open'] > df['close'].shift())
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[bull & (r < rsi_filter)] = 1
    sig[bear & (r > 100 - rsi_filter)] = -1
    return sig
reg('Pattern_Engulfing_RSI', gen_engulfing_rsi,
    lambda t: {'rsi_filter': t.suggest_int('rsi_filter', 25, 55)})

def gen_hammer_vwap(df, dist_pct=0.01):
    body = (df['close'] - df['open']).abs()
    lower_wick = pd.concat([df['open'], df['close']], axis=1).min(axis=1) - df['low']
    upper_wick = df['high'] - pd.concat([df['open'], df['close']], axis=1).max(axis=1)
    hammer = (lower_wick > 2 * body) & (upper_wick < body)
    inv_hammer = (upper_wick > 2 * body) & (lower_wick < body)
    vw = vwap_calc(df)
    sig = pd.Series(0, index=df.index)
    sig[hammer & (df['close'] < vw * (1 - dist_pct))] = 1
    sig[inv_hammer & (df['close'] > vw * (1 + dist_pct))] = -1
    return sig
reg('Pattern_Hammer_VWAP', gen_hammer_vwap,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001)})

def gen_doji_zscore(df, lookback=50, z_thresh=2.0):
    body = (df['close'] - df['open']).abs()
    wick = df['high'] - df['low']
    doji = body < wick * 0.1
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[doji & (z < -z_thresh)] = 1
    sig[doji & (z > z_thresh)] = -1
    return sig
reg('Pattern_Doji_ZScore', gen_doji_zscore,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_pinbar_bb(df, bb_period=20, bb_std=2.0, pin_ratio=2.5):
    body = (df['close'] - df['open']).abs()
    lower_wick = pd.concat([df['open'], df['close']], axis=1).min(axis=1) - df['low']
    upper_wick = df['high'] - pd.concat([df['open'], df['close']], axis=1).max(axis=1)
    bull_pin = (lower_wick > pin_ratio * body) & (lower_wick > upper_wick * 2)
    bear_pin = (upper_wick > pin_ratio * body) & (upper_wick > lower_wick * 2)
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[bull_pin & (df['close'] < lower)] = 1
    sig[bear_pin & (df['close'] > upper)] = -1
    return sig
reg('Pattern_Pinbar_BB', gen_pinbar_bb,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'pin_ratio': t.suggest_float('pin_ratio', 1.5, 4.0, step=0.5)})

def gen_engulfing_vwap(df, dist_pct=0.01):
    bull = (df['close'] > df['open']) & (df['close'].shift() < df['open'].shift()) & \
           (df['close'] > df['open'].shift()) & (df['open'] < df['close'].shift())
    bear = (df['close'] < df['open']) & (df['close'].shift() > df['open'].shift()) & \
           (df['close'] < df['open'].shift()) & (df['open'] > df['close'].shift())
    vw = vwap_calc(df)
    sig = pd.Series(0, index=df.index)
    sig[bull & (df['close'] < vw * (1 - dist_pct))] = 1
    sig[bear & (df['close'] > vw * (1 + dist_pct))] = -1
    return sig
reg('Pattern_Engulfing_VWAP', gen_engulfing_vwap,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001)})

def gen_engulfing_bb(df, bb_period=20, bb_std=2.0):
    bull = (df['close'] > df['open']) & (df['close'].shift() < df['open'].shift()) & \
           (df['close'] > df['open'].shift()) & (df['open'] < df['close'].shift())
    bear = (df['close'] < df['open']) & (df['close'].shift() > df['open'].shift()) & \
           (df['close'] < df['open'].shift()) & (df['open'] > df['close'].shift())
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[bull & (df['close'] < lower)] = 1
    sig[bear & (df['close'] > upper)] = -1
    return sig
reg('Pattern_Engulfing_BB', gen_engulfing_bb,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_doji_rsi(df, rsi_level=35):
    body = (df['close'] - df['open']).abs()
    wick = df['high'] - df['low']
    doji = body < wick * 0.1
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[doji & (r < rsi_level)] = 1
    sig[doji & (r > 100 - rsi_level)] = -1
    return sig
reg('Pattern_Doji_RSI', gen_doji_rsi,
    lambda t: {'rsi_level': t.suggest_int('rsi_level', 20, 45)})

def gen_hammer_rsi(df, rsi_filter=40):
    body = (df['close'] - df['open']).abs()
    lower_wick = pd.concat([df['open'], df['close']], axis=1).min(axis=1) - df['low']
    upper_wick = df['high'] - pd.concat([df['open'], df['close']], axis=1).max(axis=1)
    hammer = (lower_wick > 2 * body) & (upper_wick < body)
    shooting = (upper_wick > 2 * body) & (lower_wick < body)
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[hammer & (r < rsi_filter)] = 1
    sig[shooting & (r > 100 - rsi_filter)] = -1
    return sig
reg('Pattern_Hammer_RSI', gen_hammer_rsi,
    lambda t: {'rsi_filter': t.suggest_int('rsi_filter', 25, 55)})

def gen_inside_bar_squeeze(df, bb_period=20, kc_mult=1.5):
    inside = (df['high'] < df['high'].shift()) & (df['low'] > df['low'].shift())
    mid, bb_up, bb_lo = bb(df['close'], bb_period)
    e = ema(df['close'], bb_period)
    a = atr(df['high'], df['low'], df['close'], bb_period)
    sq = (bb_lo > e - kc_mult * a) & (bb_up < e + kc_mult * a)
    sig = pd.Series(0, index=df.index)
    sig[inside & sq & (df['close'] > df['open'])] = 1
    sig[inside & sq & (df['close'] < df['open'])] = -1
    return sig
reg('Pattern_InsideBar_Squeeze', gen_inside_bar_squeeze,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 35), 'kc_mult': t.suggest_float('kc_mult', 1.0, 2.5, step=0.25)})

def gen_three_soldiers_momentum(df, mom_period=10, mom_thresh=0.01):
    bull_candle = df['close'] > df['open']
    three_bull = bull_candle & bull_candle.shift() & bull_candle.shift(2)
    bear_candle = df['close'] < df['open']
    three_bear = bear_candle & bear_candle.shift() & bear_candle.shift(2)
    mom = df['close'] / df['close'].shift(mom_period) - 1
    sig = pd.Series(0, index=df.index)
    sig[three_bull & (mom > mom_thresh)] = 1
    sig[three_bear & (mom < -mom_thresh)] = -1
    return sig
reg('Pattern_ThreeSoldiers_Mom', gen_three_soldiers_momentum,
    lambda t: {'mom_period': t.suggest_int('mom_period', 3, 25),
               'mom_thresh': t.suggest_float('mom_thresh', 0.003, 0.05, step=0.002)})

def gen_pinbar_rsi(df, rsi_filter=40, pin_ratio=2.5):
    body = (df['close'] - df['open']).abs()
    lower_wick = pd.concat([df['open'], df['close']], axis=1).min(axis=1) - df['low']
    upper_wick = df['high'] - pd.concat([df['open'], df['close']], axis=1).max(axis=1)
    bull_pin = (lower_wick > pin_ratio * body) & (lower_wick > upper_wick * 2)
    bear_pin = (upper_wick > pin_ratio * body) & (upper_wick > lower_wick * 2)
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[bull_pin & (r < rsi_filter)] = 1
    sig[bear_pin & (r > 100 - rsi_filter)] = -1
    return sig
reg('Pattern_Pinbar_RSI', gen_pinbar_rsi,
    lambda t: {'rsi_filter': t.suggest_int('rsi_filter', 25, 55),
               'pin_ratio': t.suggest_float('pin_ratio', 1.5, 4.0, step=0.5)})

def gen_doji_bb(df, bb_period=20, bb_std=2.0):
    body = (df['close'] - df['open']).abs()
    wick = df['high'] - df['low']
    doji = body < wick * 0.1
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[doji & (df['close'] < lower)] = 1
    sig[doji & (df['close'] > upper)] = -1
    return sig
reg('Pattern_Doji_BB', gen_doji_bb,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_engulfing_stoch(df, stoch_buy=20):
    bull = (df['close'] > df['open']) & (df['close'].shift() < df['open'].shift()) & \
           (df['close'] > df['open'].shift()) & (df['open'] < df['close'].shift())
    bear = (df['close'] < df['open']) & (df['close'].shift() > df['open'].shift()) & \
           (df['close'] < df['open'].shift()) & (df['open'] > df['close'].shift())
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    sig = pd.Series(0, index=df.index)
    sig[bull & (k < stoch_buy)] = 1
    sig[bear & (k > 100 - stoch_buy)] = -1
    return sig
reg('Pattern_Engulfing_Stoch', gen_engulfing_stoch,
    lambda t: {'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_hammer_bb(df, bb_period=20, bb_std=2.0):
    body = (df['close'] - df['open']).abs()
    lower_wick = pd.concat([df['open'], df['close']], axis=1).min(axis=1) - df['low']
    upper_wick = df['high'] - pd.concat([df['open'], df['close']], axis=1).max(axis=1)
    hammer = (lower_wick > 2 * body) & (upper_wick < body)
    shooting = (upper_wick > 2 * body) & (lower_wick < body)
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[hammer & (df['close'] < lower)] = 1
    sig[shooting & (df['close'] > upper)] = -1
    return sig
reg('Pattern_Hammer_BB', gen_hammer_bb,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

# ==============================================================================
# CATEGORY 9: ADVANCED (10)
# ==============================================================================

def gen_heikin_ashi_trend(df, consec=3):
    ha = heikin_ashi(df)
    bull = ha['close'] > ha['open']
    bear = ha['close'] < ha['open']
    bull_streak = bull.rolling(consec).sum() == consec
    bear_streak = bear.rolling(consec).sum() == consec
    sig = pd.Series(0, index=df.index)
    sig[bull_streak & ~bull_streak.shift().fillna(False)] = 1
    sig[bear_streak & ~bear_streak.shift().fillna(False)] = -1
    return sig
reg('Adv_HeikinAshi_Trend', gen_heikin_ashi_trend,
    lambda t: {'consec': t.suggest_int('consec', 2, 6)})

def gen_momentum_divergence(df, mom_period=14, lookback=10):
    mom = df['close'] / df['close'].shift(mom_period) - 1
    price_lower = df['close'] < df['close'].rolling(lookback).min().shift()
    mom_higher = mom > mom.rolling(lookback).min().shift()
    bull_div = price_lower & mom_higher
    price_higher = df['close'] > df['close'].rolling(lookback).max().shift()
    mom_lower = mom < mom.rolling(lookback).max().shift()
    bear_div = price_higher & mom_lower
    sig = pd.Series(0, index=df.index)
    sig[bull_div] = 1
    sig[bear_div] = -1
    return sig
reg('Adv_Momentum_Divergence', gen_momentum_divergence,
    lambda t: {'mom_period': t.suggest_int('mom_period', 5, 25), 'lookback': t.suggest_int('lookback', 5, 20)})

def gen_volume_imbalance(df, lookback=20, vol_mult=2.0):
    buy_vol = df['volume'].where(df['close'] > df['open'], 0.0)
    sell_vol = df['volume'].where(df['close'] < df['open'], 0.0)
    buy_ma = buy_vol.rolling(lookback).mean()
    sell_ma = sell_vol.rolling(lookback).mean()
    sig = pd.Series(0, index=df.index)
    sig[(buy_vol > vol_mult * buy_ma) & (buy_vol > sell_vol * 2)] = 1
    sig[(sell_vol > vol_mult * sell_ma) & (sell_vol > buy_vol * 2)] = -1
    return sig
reg('Adv_Volume_Imbalance', gen_volume_imbalance,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 50), 'vol_mult': t.suggest_float('vol_mult', 1.2, 4.0, step=0.2)})

def gen_heikin_ashi_rsi(df, rsi_buy=30, consec=2):
    ha = heikin_ashi(df)
    r = rsi(df['close'], 14)
    bull = ha['close'] > ha['open']
    bull_streak = bull.rolling(consec).sum() == consec
    bear = ha['close'] < ha['open']
    bear_streak = bear.rolling(consec).sum() == consec
    sig = pd.Series(0, index=df.index)
    sig[bull_streak & (r < rsi_buy)] = 1
    sig[bear_streak & (r > 100 - rsi_buy)] = -1
    return sig
reg('Adv_HeikinAshi_RSI', gen_heikin_ashi_rsi,
    lambda t: {'rsi_buy': t.suggest_int('rsi_buy', 15, 45), 'consec': t.suggest_int('consec', 2, 5)})

def gen_heikin_ashi_vwap(df, dist_pct=0.01, consec=2):
    ha = heikin_ashi(df)
    vw = vwap_calc(df)
    bull = ha['close'] > ha['open']
    bear = ha['close'] < ha['open']
    bull_streak = bull.rolling(consec).sum() == consec
    bear_streak = bear.rolling(consec).sum() == consec
    sig = pd.Series(0, index=df.index)
    sig[bull_streak & (df['close'] < vw * (1 - dist_pct))] = 1
    sig[bear_streak & (df['close'] > vw * (1 + dist_pct))] = -1
    return sig
reg('Adv_HeikinAshi_VWAP', gen_heikin_ashi_vwap,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'consec': t.suggest_int('consec', 2, 5)})

def gen_volume_breakout(df, vol_mult=2.0, lookback=20):
    vm = df['volume'].rolling(lookback).mean()
    vs = df['volume'].rolling(lookback).std()
    vol_breakout = df['volume'] > vm + vol_mult * vs
    sig = pd.Series(0, index=df.index)
    sig[vol_breakout & (df['close'] > df['open']) & (df['close'] > df['close'].shift())] = 1
    sig[vol_breakout & (df['close'] < df['open']) & (df['close'] < df['close'].shift())] = -1
    return sig
reg('Adv_Volume_Breakout', gen_volume_breakout,
    lambda t: {'vol_mult': t.suggest_float('vol_mult', 1.0, 4.0, step=0.25),
               'lookback': t.suggest_int('lookback', 10, 50)})

def gen_tsi_signal(df, long_p=25, short_p=13, signal_p=7):
    d = df['close'].diff()
    ds = ema(ema(d, long_p), short_p)
    ads = ema(ema(d.abs(), long_p), short_p)
    tsi = 100 * ds / (ads + 1e-10)
    tsi_signal = ema(tsi, signal_p)
    sig = pd.Series(0, index=df.index)
    sig[(tsi > tsi_signal) & (tsi.shift() <= tsi_signal.shift())] = 1
    sig[(tsi < tsi_signal) & (tsi.shift() >= tsi_signal.shift())] = -1
    return sig
reg('Adv_TSI_Signal', gen_tsi_signal,
    lambda t: {'long_p': t.suggest_int('long_p', 15, 40), 'short_p': t.suggest_int('short_p', 5, 18),
               'signal_p': t.suggest_int('signal_p', 4, 12)})

def gen_roc_reversal(df, roc_period=10, roc_thresh=3.0):
    roc = (df['close'] / df['close'].shift(roc_period) - 1) * 100
    sig = pd.Series(0, index=df.index)
    sig[(roc < -roc_thresh) & (roc > roc.shift())] = 1
    sig[(roc > roc_thresh) & (roc < roc.shift())] = -1
    return sig
reg('Adv_ROC_Reversal', gen_roc_reversal,
    lambda t: {'roc_period': t.suggest_int('roc_period', 3, 25),
               'roc_thresh': t.suggest_float('roc_thresh', 1.0, 8.0, step=0.5)})

def gen_cmf_divergence(df, cmf_period=20, lookback=10):
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    mfv = mfm * df['volume']
    cmf = mfv.rolling(cmf_period).sum() / (df['volume'].rolling(cmf_period).sum() + 1e-10)
    price_lower = df['close'] < df['close'].rolling(lookback).min().shift()
    cmf_higher = cmf > cmf.rolling(lookback).min().shift()
    bull_div = price_lower & cmf_higher
    price_higher = df['close'] > df['close'].rolling(lookback).max().shift()
    cmf_lower = cmf < cmf.rolling(lookback).max().shift()
    bear_div = price_higher & cmf_lower
    sig = pd.Series(0, index=df.index)
    sig[bull_div] = 1
    sig[bear_div] = -1
    return sig
reg('Adv_CMF_Divergence', gen_cmf_divergence,
    lambda t: {'cmf_period': t.suggest_int('cmf_period', 10, 40), 'lookback': t.suggest_int('lookback', 5, 20)})

def gen_obv_divergence(df, obv_period=20, lookback=10):
    o = obv_calc(df['close'], df['volume'])
    o_sma = o.rolling(obv_period).mean()
    price_lower = df['close'] < df['close'].rolling(lookback).min().shift()
    obv_higher = o > o.rolling(lookback).min().shift()
    bull_div = price_lower & obv_higher
    price_higher = df['close'] > df['close'].rolling(lookback).max().shift()
    obv_lower = o < o.rolling(lookback).max().shift()
    bear_div = price_higher & obv_lower
    sig = pd.Series(0, index=df.index)
    sig[bull_div] = 1
    sig[bear_div] = -1
    return sig
reg('Adv_OBV_Divergence', gen_obv_divergence,
    lambda t: {'obv_period': t.suggest_int('obv_period', 10, 40), 'lookback': t.suggest_int('lookback', 5, 20)})

# ==============================================================================
# CATEGORY 10: EXTRA COMBOS & VARIATIONS (to reach 215+)
# ==============================================================================

def gen_vwap_atr_stoch(df, atr_mult=1.5, atr_period=14, stoch_buy=20):
    vw = vwap_calc(df)
    a = atr(df['high'], df['low'], df['close'], atr_period)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw - atr_mult * a) & (k < stoch_buy)] = 1
    sig[(df['close'] > vw + atr_mult * a) & (k > 100 - stoch_buy)] = -1
    return sig
reg('VWAP_ATR_Stoch', gen_vwap_atr_stoch,
    lambda t: {'atr_mult': t.suggest_float('atr_mult', 0.5, 4.0, step=0.25),
               'atr_period': t.suggest_int('atr_period', 7, 28), 'stoch_buy': t.suggest_int('stoch_buy', 8, 30)})

def gen_vwap_atr_vol(df, atr_mult=1.5, atr_period=14, vol_mult=1.5):
    vw = vwap_calc(df)
    a = atr(df['high'], df['low'], df['close'], atr_period)
    vm = df['volume'].rolling(20).mean()
    high_vol = df['volume'] > vol_mult * vm
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw - atr_mult * a) & high_vol] = 1
    sig[(df['close'] > vw + atr_mult * a) & high_vol] = -1
    return sig
reg('VWAP_ATR_Vol', gen_vwap_atr_vol,
    lambda t: {'atr_mult': t.suggest_float('atr_mult', 0.5, 4.0, step=0.25),
               'atr_period': t.suggest_int('atr_period', 7, 28),
               'vol_mult': t.suggest_float('vol_mult', 1.0, 4.0, step=0.2)})

def gen_zscore_williams(df, lookback=50, z_thresh=2.0, wr_buy=-80):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (wr < wr_buy)] = 1
    sig[(z > z_thresh) & (wr > -100 - wr_buy)] = -1
    return sig
reg('ZScore_WilliamsR', gen_zscore_williams,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'wr_buy': t.suggest_int('wr_buy', -95, -65)})

def gen_bb_zscore(df, bb_period=20, bb_std=2.0, lookback=50, z_thresh=1.5):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    pctb = (df['close'] - lower) / (upper - lower + 1e-10)
    m = pctb.rolling(lookback).mean()
    s = pctb.rolling(lookback).std()
    z = (pctb - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[z < -z_thresh] = 1
    sig[z > z_thresh] = -1
    return sig
reg('BB_ZScore_PctB', gen_bb_zscore,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'lookback': t.suggest_int('lookback', 10, 100), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.0, step=0.1)})

def gen_adaptive_ema_cross(df, fast=9, slow=21, vol_period=20):
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    high_vol = vol > vol_ma
    ef = ema(df['close'], fast)
    es = ema(df['close'], slow)
    sig = pd.Series(0, index=df.index)
    # Only trade crosses in low vol
    cross_up = (ef > es) & (ef.shift() <= es.shift())
    cross_down = (ef < es) & (ef.shift() >= es.shift())
    sig[cross_up & ~high_vol] = 1
    sig[cross_down & ~high_vol] = -1
    return sig
reg('Adaptive_EMA_Cross', gen_adaptive_ema_cross,
    lambda t: {'fast': t.suggest_int('fast', 3, 15), 'slow': t.suggest_int('slow', 15, 60),
               'vol_period': t.suggest_int('vol_period', 10, 40)})

def gen_vote_2of3_bb_stoch_wr(df, bb_std=2.0, stoch_buy=20, wr_buy=-80):
    mid, upper, lower = bb(df['close'], 20, bb_std)
    k, _ = stoch(df['high'], df['low'], df['close'], 14, 3)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    buy_v = (df['close'] < lower).astype(int) + (k < stoch_buy).astype(int) + (wr < wr_buy).astype(int)
    sell_v = (df['close'] > upper).astype(int) + (k > 100 - stoch_buy).astype(int) + (wr > -100 - wr_buy).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 2] = 1
    sig[sell_v >= 2] = -1
    return sig
reg('Vote_2of3_BB_Stoch_WR', gen_vote_2of3_bb_stoch_wr,
    lambda t: {'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'stoch_buy': t.suggest_int('stoch_buy', 8, 30), 'wr_buy': t.suggest_int('wr_buy', -95, -65)})

def gen_vote_2of3_vwap_cci_macd(df, dist_pct=0.01, cci_level=100):
    vw = vwap_calc(df)
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    _, _, hist = macd_calc(df['close'])
    buy_v = (df['close'] < vw * (1 - dist_pct)).astype(int) + (c < -cci_level).astype(int) + (hist > 0).astype(int)
    sell_v = (df['close'] > vw * (1 + dist_pct)).astype(int) + (c > cci_level).astype(int) + (hist < 0).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 2] = 1
    sig[sell_v >= 2] = -1
    return sig
reg('Vote_2of3_VWAP_CCI_MACD', gen_vote_2of3_vwap_cci_macd,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'cci_level': t.suggest_int('cci_level', 50, 200)})

def gen_combo_vwap_rsi(df, dist_pct=0.01, rsi_buy=30):
    vw = vwap_calc(df)
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (r < rsi_buy)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (r > 100 - rsi_buy)] = -1
    return sig
reg('Combo_VWAP_RSI', gen_combo_vwap_rsi,
    lambda t: {'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40)})

def gen_adaptive_zscore_rsi(df, lookback=50, z_thresh=2.0, rsi_buy=30, vol_period=20):
    vol = df['close'].rolling(vol_period).std() / (df['close'].rolling(vol_period).mean() + 1e-10)
    vol_ma = vol.rolling(vol_period).mean()
    ratio = (vol / (vol_ma + 1e-10)).clip(0.5, 2.0)
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    adj_thresh = z_thresh * ratio
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(z < -adj_thresh) & (r < rsi_buy)] = 1
    sig[(z > adj_thresh) & (r > 100 - rsi_buy)] = -1
    return sig
reg('Adaptive_ZScore_RSI', gen_adaptive_zscore_rsi,
    lambda t: {'lookback': t.suggest_int('lookback', 15, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40), 'vol_period': t.suggest_int('vol_period', 10, 40)})

def gen_pinbar_zscore(df, lookback=50, z_thresh=2.0, pin_ratio=2.5):
    body = (df['close'] - df['open']).abs()
    lower_wick = pd.concat([df['open'], df['close']], axis=1).min(axis=1) - df['low']
    upper_wick = df['high'] - pd.concat([df['open'], df['close']], axis=1).max(axis=1)
    bull_pin = (lower_wick > pin_ratio * body) & (lower_wick > upper_wick * 2)
    bear_pin = (upper_wick > pin_ratio * body) & (upper_wick > lower_wick * 2)
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[bull_pin & (z < -z_thresh)] = 1
    sig[bear_pin & (z > z_thresh)] = -1
    return sig
reg('Pattern_Pinbar_ZScore', gen_pinbar_zscore,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'pin_ratio': t.suggest_float('pin_ratio', 1.5, 4.0, step=0.5)})

def gen_engulfing_zscore(df, lookback=50, z_thresh=2.0):
    bull = (df['close'] > df['open']) & (df['close'].shift() < df['open'].shift()) & \
           (df['close'] > df['open'].shift()) & (df['open'] < df['close'].shift())
    bear = (df['close'] < df['open']) & (df['close'].shift() > df['open'].shift()) & \
           (df['close'] < df['open'].shift()) & (df['open'] > df['close'].shift())
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[bull & (z < -z_thresh)] = 1
    sig[bear & (z > z_thresh)] = -1
    return sig
reg('Pattern_Engulfing_ZScore', gen_engulfing_zscore,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1)})

def gen_heikin_bb(df, bb_period=20, bb_std=2.0, consec=2):
    ha = heikin_ashi(df)
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    bull = ha['close'] > ha['open']
    bull_streak = bull.rolling(consec).sum() == consec
    bear = ha['close'] < ha['open']
    bear_streak = bear.rolling(consec).sum() == consec
    sig = pd.Series(0, index=df.index)
    sig[bull_streak & (df['close'] < lower)] = 1
    sig[bear_streak & (df['close'] > upper)] = -1
    return sig
reg('Adv_HeikinAshi_BB', gen_heikin_bb,
    lambda t: {'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25),
               'consec': t.suggest_int('consec', 2, 5)})

def gen_heikin_zscore(df, lookback=50, z_thresh=2.0, consec=2):
    ha = heikin_ashi(df)
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    bull = ha['close'] > ha['open']
    bull_streak = bull.rolling(consec).sum() == consec
    bear = ha['close'] < ha['open']
    bear_streak = bear.rolling(consec).sum() == consec
    sig = pd.Series(0, index=df.index)
    sig[bull_streak & (z < -z_thresh)] = 1
    sig[bear_streak & (z > z_thresh)] = -1
    return sig
reg('Adv_HeikinAshi_ZScore', gen_heikin_zscore,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 150), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'consec': t.suggest_int('consec', 2, 5)})

def gen_mfi_rsi(df, mfi_period=14, mfi_buy=20, rsi_buy=30):
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos_mf = mf.where(tp > tp.shift(), 0.0).rolling(mfi_period).sum()
    neg_mf = mf.where(tp < tp.shift(), 0.0).rolling(mfi_period).sum()
    mfi = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-10))
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(mfi < mfi_buy) & (r < rsi_buy)] = 1
    sig[(mfi > 100 - mfi_buy) & (r > 100 - rsi_buy)] = -1
    return sig
reg('Combo_MFI_RSI', gen_mfi_rsi,
    lambda t: {'mfi_period': t.suggest_int('mfi_period', 7, 28), 'mfi_buy': t.suggest_int('mfi_buy', 10, 35),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40)})

def gen_mfi_bb(df, mfi_period=14, mfi_buy=20, bb_period=20, bb_std=2.0):
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos_mf = mf.where(tp > tp.shift(), 0.0).rolling(mfi_period).sum()
    neg_mf = mf.where(tp < tp.shift(), 0.0).rolling(mfi_period).sum()
    mfi = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-10))
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[(mfi < mfi_buy) & (df['close'] < lower)] = 1
    sig[(mfi > 100 - mfi_buy) & (df['close'] > upper)] = -1
    return sig
reg('Combo_MFI_BB', gen_mfi_bb,
    lambda t: {'mfi_period': t.suggest_int('mfi_period', 7, 28), 'mfi_buy': t.suggest_int('mfi_buy', 10, 35),
               'bb_period': t.suggest_int('bb_period', 10, 40), 'bb_std': t.suggest_float('bb_std', 1.0, 3.5, step=0.25)})

def gen_mfi_vwap(df, mfi_period=14, mfi_buy=20, dist_pct=0.01):
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos_mf = mf.where(tp > tp.shift(), 0.0).rolling(mfi_period).sum()
    neg_mf = mf.where(tp < tp.shift(), 0.0).rolling(mfi_period).sum()
    mfi = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-10))
    vw = vwap_calc(df)
    sig = pd.Series(0, index=df.index)
    sig[(mfi < mfi_buy) & (df['close'] < vw * (1 - dist_pct))] = 1
    sig[(mfi > 100 - mfi_buy) & (df['close'] > vw * (1 + dist_pct))] = -1
    return sig
reg('Combo_MFI_VWAP', gen_mfi_vwap,
    lambda t: {'mfi_period': t.suggest_int('mfi_period', 7, 28), 'mfi_buy': t.suggest_int('mfi_buy', 10, 35),
               'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001)})

def gen_tsi_rsi(df, long_p=25, short_p=13, rsi_buy=30):
    d = df['close'].diff()
    ds = ema(ema(d, long_p), short_p)
    ads = ema(ema(d.abs(), long_p), short_p)
    tsi = 100 * ds / (ads + 1e-10)
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(tsi > tsi.shift()) & (r < rsi_buy)] = 1
    sig[(tsi < tsi.shift()) & (r > 100 - rsi_buy)] = -1
    return sig
reg('Combo_TSI_RSI', gen_tsi_rsi,
    lambda t: {'long_p': t.suggest_int('long_p', 15, 40), 'short_p': t.suggest_int('short_p', 5, 18),
               'rsi_buy': t.suggest_int('rsi_buy', 15, 40)})

def gen_tsi_vwap(df, long_p=25, short_p=13, dist_pct=0.01):
    d = df['close'].diff()
    ds = ema(ema(d, long_p), short_p)
    ads = ema(ema(d.abs(), long_p), short_p)
    tsi = 100 * ds / (ads + 1e-10)
    vw = vwap_calc(df)
    sig = pd.Series(0, index=df.index)
    sig[(tsi > tsi.shift()) & (df['close'] < vw * (1 - dist_pct))] = 1
    sig[(tsi < tsi.shift()) & (df['close'] > vw * (1 + dist_pct))] = -1
    return sig
reg('Combo_TSI_VWAP', gen_tsi_vwap,
    lambda t: {'long_p': t.suggest_int('long_p', 15, 40), 'short_p': t.suggest_int('short_p', 5, 18),
               'dist_pct': t.suggest_float('dist_pct', 0.002, 0.06, step=0.001)})

def gen_dema_rsi(df, dema_period=20, rsi_buy=30):
    e1 = ema(df['close'], dema_period)
    dema = 2 * e1 - ema(e1, dema_period)
    r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] > dema) & (df['close'].shift() <= dema.shift()) & (r < rsi_buy + 20)] = 1
    sig[(df['close'] < dema) & (df['close'].shift() >= dema.shift()) & (r > 100 - rsi_buy - 20)] = -1
    return sig
reg('Combo_DEMA_RSI', gen_dema_rsi,
    lambda t: {'dema_period': t.suggest_int('dema_period', 10, 60), 'rsi_buy': t.suggest_int('rsi_buy', 15, 40)})

def gen_adv_multi_divergence(df, rsi_period=14, lookback=10):
    r = rsi(df['close'], rsi_period)
    _, _, hist = macd_calc(df['close'])
    # RSI divergence
    p_lo = df['close'] < df['close'].rolling(lookback).min().shift()
    r_hi = r > r.rolling(lookback).min().shift()
    # MACD divergence
    h_hi = hist > hist.rolling(lookback).min().shift()
    bull_div = p_lo & (r_hi | h_hi)
    p_hi = df['close'] > df['close'].rolling(lookback).max().shift()
    r_lo = r < r.rolling(lookback).max().shift()
    h_lo = hist < hist.rolling(lookback).max().shift()
    bear_div = p_hi & (r_lo | h_lo)
    sig = pd.Series(0, index=df.index)
    sig[bull_div] = 1
    sig[bear_div] = -1
    return sig
reg('Adv_Multi_Divergence', gen_adv_multi_divergence,
    lambda t: {'rsi_period': t.suggest_int('rsi_period', 7, 21), 'lookback': t.suggest_int('lookback', 5, 20)})

def gen_adv_vol_price_confirm(df, vol_mult=2.0, lookback=20, rsi_buy=35):
    vm = df['volume'].rolling(lookback).mean()
    high_vol = df['volume'] > vol_mult * vm
    r = rsi(df['close'], 14)
    bull_candle = df['close'] > df['open']
    sig = pd.Series(0, index=df.index)
    sig[high_vol & bull_candle & (r < rsi_buy)] = 1
    sig[high_vol & ~bull_candle & (r > 100 - rsi_buy)] = -1
    return sig
reg('Adv_Vol_Price_Confirm', gen_adv_vol_price_confirm,
    lambda t: {'vol_mult': t.suggest_float('vol_mult', 1.2, 4.0, step=0.2),
               'lookback': t.suggest_int('lookback', 10, 50), 'rsi_buy': t.suggest_int('rsi_buy', 20, 45)})

def gen_adv_range_breakout(df, lookback=20, atr_mult=1.5, atr_period=14):
    hi = df['high'].rolling(lookback).max()
    lo = df['low'].rolling(lookback).min()
    rng = hi - lo
    a = atr(df['high'], df['low'], df['close'], atr_period)
    tight = rng < atr_mult * a
    sig = pd.Series(0, index=df.index)
    sig[tight.shift().fillna(False) & (df['close'] > hi.shift()) & (df['close'] > df['close'].shift())] = 1
    sig[tight.shift().fillna(False) & (df['close'] < lo.shift()) & (df['close'] < df['close'].shift())] = -1
    return sig
reg('Adv_Range_Breakout', gen_adv_range_breakout,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 50), 'atr_mult': t.suggest_float('atr_mult', 0.5, 3.0, step=0.25),
               'atr_period': t.suggest_int('atr_period', 7, 28)})

def gen_zscore_of_obv(df, lookback=50, z_thresh=2.0, obv_period=20):
    o = obv_calc(df['close'], df['volume'])
    o_det = o - o.rolling(obv_period).mean()  # detrended
    m = o_det.rolling(lookback).mean()
    s = o_det.rolling(lookback).std()
    z = (o_det - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & (df['close'] > df['close'].shift())] = 1
    sig[(z > z_thresh) & (df['close'] < df['close'].shift())] = -1
    return sig
reg('ZScore_of_OBV', gen_zscore_of_obv,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 100), 'z_thresh': t.suggest_float('z_thresh', 0.8, 3.5, step=0.1),
               'obv_period': t.suggest_int('obv_period', 10, 40)})

def gen_vote_2of3_cci_wr_macd(df, cci_level=100, wr_buy=-80):
    c = cci_calc(df['high'], df['low'], df['close'], 20)
    wr = williams_r(df['high'], df['low'], df['close'], 14)
    _, _, hist = macd_calc(df['close'])
    buy_v = (c < -cci_level).astype(int) + (wr < wr_buy).astype(int) + (hist > 0).astype(int)
    sell_v = (c > cci_level).astype(int) + (wr > -100 - wr_buy).astype(int) + (hist < 0).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_v >= 2] = 1
    sig[sell_v >= 2] = -1
    return sig
reg('Vote_2of3_CCI_WR_MACD', gen_vote_2of3_cci_wr_macd,
    lambda t: {'cci_level': t.suggest_int('cci_level', 50, 200), 'wr_buy': t.suggest_int('wr_buy', -95, -65)})

# ==============================================================================
# PRINT TOTAL
# ==============================================================================

print(f"FACTORY_STRATS loaded: {len(FACTORY_STRATS)} strategies")
