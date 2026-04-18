#!/usr/bin/env python3
"""
TV2 BATCH 78 — 25 estrategias convertidas de Alorse/pinescript-strategies (v4/v5).
Categorías: Trend Following (10) + Momentum (10)

Pine versions: v4 (18), v5 (7)
TradingView repo: Alorse/pinescript-strategies
"""

import pandas as pd
import numpy as np

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _sma(s, p):
    return s.rolling(p).mean()

def _rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0).ewm(span=p, adjust=False).mean()
    l = (-d.where(d < 0, 0.0)).ewm(span=p, adjust=False).mean()
    rs = g / (l + 1e-10)
    return 100 - 100 / (1 + rs)

def _atr_ema(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/p, adjust=False).mean()

def _bb(c, p=20, std=2.0):
    m = _sma(c, p)
    s = c.rolling(p).std()
    return m, m + std * s, m - std * s

def _macd(c, fast=12, slow=26, sig=9):
    ema_fast = _ema(c, fast)
    ema_slow = _ema(c, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, sig)
    return macd_line, signal_line

def _supertrend(h, l, c, atr_period=10, atr_mult=2.7):
    atr = _atr_ema(h, l, c, atr_period)
    hl2 = (h + l) / 2.0
    st_up_raw = (hl2 - atr_mult * atr).values
    st_dn_raw = (hl2 + atr_mult * atr).values
    c_vals = c.values
    n = len(c_vals)
    trending_up = np.zeros(n)
    trending_dn = np.zeros(n)
    direction = np.ones(n, dtype=int)
    trending_up[0] = st_up_raw[0]
    trending_dn[0] = st_dn_raw[0]
    for i in range(1, n):
        if c_vals[i - 1] > trending_up[i - 1]:
            trending_up[i] = max(st_up_raw[i], trending_up[i - 1])
        else:
            trending_up[i] = st_up_raw[i]
        if c_vals[i - 1] < trending_dn[i - 1]:
            trending_dn[i] = min(st_dn_raw[i], trending_dn[i - 1])
        else:
            trending_dn[i] = st_dn_raw[i]
        if c_vals[i] > trending_dn[i - 1]:
            direction[i] = 1
        elif c_vals[i] < trending_up[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return pd.Series(direction, index=c.index)

def _dmi(h, l, c, period=14):
    up_move = h.diff()
    dn_move = (-l.diff()).abs()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    plus_di = 100 * (up_move.rolling(period).mean()) / (atr_val + 1e-10)
    minus_di = 100 * (dn_move.rolling(period).mean()) / (atr_val + 1e-10)
    return plus_di, minus_di

# TREND FOLLOWING

def gen_3_EMA_SMA_Cross(df, ema1=5, ema2=10, sma1=21):
    c = df['close']
    e1 = _ema(c, ema1)
    e2 = _ema(c, ema2)
    s1 = _sma(c, sma1)
    long_cond = (e1 > e2) & (e2 > s1)
    short_cond = (e1 < e2) & (e2 < s1)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_3_EMA_SMA_Cross(trial):
    return {'ema1': trial.suggest_int('ema1', 3, 15), 'ema2': trial.suggest_int('ema2', 8, 25), 'sma1': trial.suggest_int('sma1', 15, 50)}

def gen_Double_Supertrend(df, atr1=10, mult1=2.5, atr2=13, mult2=3.5):
    h, l, c = df['high'], df['low'], df['close']
    dir1 = _supertrend(h, l, c, atr1, mult1)
    dir2 = _supertrend(h, l, c, atr2, mult2)
    long_cond = (dir1 == 1) & (dir2 == 1)
    short_cond = (dir1 == -1) & (dir2 == -1)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Double_Supertrend(trial):
    return {'atr1': trial.suggest_int('atr1', 7, 15), 'mult1': trial.suggest_float('mult1', 1.5, 3.5, step=0.5), 'atr2': trial.suggest_int('atr2', 10, 20), 'mult2': trial.suggest_float('mult2', 2.5, 4.5, step=0.5)}

def gen_EMA_Moving_away(df, ema_fast=12, ema_slow=26):
    c = df['close']
    fast = _ema(c, ema_fast)
    slow = _ema(c, ema_slow)
    dist = fast - slow
    dist_prev = dist.shift(1)
    long_cond = (dist > dist_prev) & (dist > 0)
    short_cond = (dist < dist_prev) & (dist < 0)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_EMA_Moving_away(trial):
    return {'ema_fast': trial.suggest_int('ema_fast', 7, 20), 'ema_slow': trial.suggest_int('ema_slow', 20, 50)}

def gen_HA_UnivLong_Short(df, atr_period=10, atr_mult=2.5):
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    ha_c = (o + h + l + c) / 4
    ha_o = pd.Series(index=df.index, dtype=float)
    ha_o.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(ha_o)):
        ha_o.iloc[i] = (ha_o.iloc[i - 1] + ha_c.iloc[i - 1]) / 2
    ha_h = pd.concat([h, ha_o, ha_c], axis=1).max(axis=1)
    ha_l = pd.concat([l, ha_o, ha_c], axis=1).min(axis=1)
    dir_ha = _supertrend(ha_h, ha_l, ha_c, atr_period, atr_mult)
    sig = pd.Series(0, index=df.index)
    sig[dir_ha == 1] = 1
    sig[dir_ha == -1] = -1
    return sig

def space_HA_UnivLong_Short(trial):
    return {'atr_period': trial.suggest_int('atr_period', 7, 20), 'atr_mult': trial.suggest_float('atr_mult', 1.5, 3.5, step=0.5)}

def gen_Heikin_Ashi_V2(df, atr_period=10, atr_mult=2.7):
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    ha_c = (o + h + l + c) / 4
    ha_o = pd.Series(index=df.index, dtype=float)
    ha_o.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(ha_o)):
        ha_o.iloc[i] = (ha_o.iloc[i - 1] + ha_c.iloc[i - 1]) / 2
    ha_h = pd.concat([h, ha_o, ha_c], axis=1).max(axis=1)
    ha_l = pd.concat([l, ha_o, ha_c], axis=1).min(axis=1)
    dir_ha = _supertrend(ha_h, ha_l, ha_c, atr_period, atr_mult)
    sig = pd.Series(0, index=df.index)
    sig[dir_ha == 1] = 1
    sig[dir_ha == -1] = -1
    return sig

def space_Heikin_Ashi_V2(trial):
    return {'atr_period': trial.suggest_int('atr_period', 7, 20), 'atr_mult': trial.suggest_float('atr_mult', 2.0, 3.5, step=0.5)}

def gen_MA_Cross_DMI(df, ma_fast=20, ma_slow=50, dmi_period=14):
    c = df['close']
    h, l = df['high'], df['low']
    fast_ma = _sma(c, ma_fast)
    slow_ma = _sma(c, ma_slow)
    plus_di, minus_di = _dmi(h, l, c, dmi_period)
    long_cond = (fast_ma > slow_ma) & (plus_di > minus_di)
    short_cond = (fast_ma < slow_ma) & (minus_di > plus_di)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_MA_Cross_DMI(trial):
    return {'ma_fast': trial.suggest_int('ma_fast', 10, 35), 'ma_slow': trial.suggest_int('ma_slow', 40, 100), 'dmi_period': trial.suggest_int('dmi_period', 10, 25)}

def gen_Supertrend(df, atr_period=10, atr_mult=2.7):
    h, l, c = df['high'], df['low'], df['close']
    dir_st = _supertrend(h, l, c, atr_period, atr_mult)
    sig = pd.Series(0, index=df.index)
    sig[dir_st == 1] = 1
    sig[dir_st == -1] = -1
    return sig

def space_Supertrend(trial):
    return {'atr_period': trial.suggest_int('atr_period', 7, 20), 'atr_mult': trial.suggest_float('atr_mult', 1.5, 4.0, step=0.5)}

def gen_Supertrend_EMA_Reb(df, atr_period=10, atr_mult=2.5, ema_period=50):
    h, l, c = df['high'], df['low'], df['close']
    dir_st = _supertrend(h, l, c, atr_period, atr_mult)
    ema = _ema(c, ema_period)
    near_ema = (c - ema).abs() < (c.rolling(10).std() * 0.5)
    long_cond = (dir_st == 1) & near_ema
    short_cond = (dir_st == -1) & near_ema
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Supertrend_EMA_Reb(trial):
    return {'atr_period': trial.suggest_int('atr_period', 7, 15), 'atr_mult': trial.suggest_float('atr_mult', 2.0, 3.5, step=0.5), 'ema_period': trial.suggest_int('ema_period', 30, 70)}

def gen_Supertrend_RSI(df, atr_period=10, atr_mult=2.5, rsi_period=14):
    h, l, c = df['high'], df['low'], df['close']
    dir_st = _supertrend(h, l, c, atr_period, atr_mult)
    rsi_val = _rsi(c, rsi_period)
    long_cond = (dir_st == 1) & (rsi_val > 50)
    short_cond = (dir_st == -1) & (rsi_val < 50)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Supertrend_RSI(trial):
    return {'atr_period': trial.suggest_int('atr_period', 7, 15), 'atr_mult': trial.suggest_float('atr_mult', 2.0, 3.5, step=0.5), 'rsi_period': trial.suggest_int('rsi_period', 10, 20)}

def gen_Trend_EMA_RSI(df, ema_period=50, rsi_period=14):
    c = df['close']
    ema = _ema(c, ema_period)
    rsi = _rsi(c, rsi_period)
    long_cond = (c > ema) & (rsi > 50)
    short_cond = (c < ema) & (rsi < 50)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Trend_EMA_RSI(trial):
    return {'ema_period': trial.suggest_int('ema_period', 30, 100), 'rsi_period': trial.suggest_int('rsi_period', 10, 21)}

# MOMENTUM

def gen_2_EMA_SMA_RSI(df, ema_fast=9, ema_slow=21, sma_period=50, rsi_period=14):
    c = df['close']
    e_fast = _ema(c, ema_fast)
    e_slow = _ema(c, ema_slow)
    s = _sma(c, sma_period)
    r = _rsi(c, rsi_period)
    long_cond = (e_fast > e_slow) & (e_slow > s) & (r > 50)
    short_cond = (e_fast < e_slow) & (e_slow < s) & (r < 50)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_2_EMA_SMA_RSI(trial):
    return {'ema_fast': trial.suggest_int('ema_fast', 5, 15), 'ema_slow': trial.suggest_int('ema_slow', 15, 50), 'sma_period': trial.suggest_int('sma_period', 30, 80), 'rsi_period': trial.suggest_int('rsi_period', 10, 21)}

def gen_DMI_Winner(df, dmi_period=14):
    h, l, c = df['high'], df['low'], df['close']
    plus_di, minus_di = _dmi(h, l, c, dmi_period)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    adx = (((plus_di - minus_di).abs() / ((plus_di + minus_di).abs() + 1e-10)) * 100).rolling(dmi_period).mean()
    long_cond = (plus_di > minus_di) & (plus_di > adx)
    short_cond = (minus_di > plus_di) & (minus_di > adx)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_DMI_Winner(trial):
    return {'dmi_period': trial.suggest_int('dmi_period', 10, 25)}

def gen_Double_RSI(df, rsi1_period=14, rsi2_period=7):
    c = df['close']
    r1 = _rsi(c, rsi1_period)
    r2 = _rsi(c, rsi2_period)
    long_cond = (r1 > 50) & (r2 > 50)
    short_cond = (r1 < 50) & (r2 < 50)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Double_RSI(trial):
    return {'rsi1_period': trial.suggest_int('rsi1_period', 10, 21), 'rsi2_period': trial.suggest_int('rsi2_period', 5, 12)}

def gen_MACD_BB_RSI(df, fast=12, slow=26, sig=9, bb_len=20, rsi_period=14):
    c = df['close']
    macd_line, signal_line = _macd(c, fast, slow, sig)
    _, bb_upper, bb_lower = _bb(c, bb_len)
    rsi_val = _rsi(c, rsi_period)
    long_cond = (macd_line > signal_line) & (c > bb_lower) & (rsi_val > 50)
    short_cond = (macd_line < signal_line) & (c < bb_upper) & (rsi_val < 50)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_MACD_BB_RSI(trial):
    return {'fast': trial.suggest_int('fast', 8, 16), 'slow': trial.suggest_int('slow', 20, 35), 'sig': trial.suggest_int('sig', 5, 15), 'bb_len': trial.suggest_int('bb_len', 15, 30), 'rsi_period': trial.suggest_int('rsi_period', 10, 21)}

def gen_MACD_DMI(df, macd_fast=12, macd_slow=26, macd_sig=9, dmi_period=14):
    c = df['close']
    h, l = df['high'], df['low']
    macd_line, signal_line = _macd(c, macd_fast, macd_slow, macd_sig)
    plus_di, minus_di = _dmi(h, l, c, dmi_period)
    long_cond = (macd_line > signal_line) & (plus_di > minus_di)
    short_cond = (macd_line < signal_line) & (minus_di > plus_di)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_MACD_DMI(trial):
    return {'macd_fast': trial.suggest_int('macd_fast', 8, 16), 'macd_slow': trial.suggest_int('macd_slow', 20, 35), 'macd_sig': trial.suggest_int('macd_sig', 5, 15), 'dmi_period': trial.suggest_int('dmi_period', 10, 25)}

def gen_MACD_Long_Strategy(df, fast=12, slow=26, sig=9):
    c = df['close']
    macd_line, signal_line = _macd(c, fast, slow, sig)
    co_up = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line >= signal_line)
    sig = pd.Series(0, index=df.index)
    sig[co_up] = 1
    return sig

def space_MACD_Long_Strategy(trial):
    return {'fast': trial.suggest_int('fast', 8, 16), 'slow': trial.suggest_int('slow', 20, 35), 'sig': trial.suggest_int('sig', 5, 15)}

def gen_MACD_RSI(df, macd_fast=12, macd_slow=26, macd_sig=9, rsi_period=14):
    c = df['close']
    macd_line, signal_line = _macd(c, macd_fast, macd_slow, macd_sig)
    rsi_val = _rsi(c, rsi_period)
    long_cond = (macd_line > signal_line) & (rsi_val > 50)
    short_cond = (macd_line < signal_line) & (rsi_val < 50)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_MACD_RSI(trial):
    return {'macd_fast': trial.suggest_int('macd_fast', 8, 16), 'macd_slow': trial.suggest_int('macd_slow', 20, 35), 'macd_sig': trial.suggest_int('macd_sig', 5, 15), 'rsi_period': trial.suggest_int('rsi_period', 10, 21)}

def gen_QQE_signals(df, rsi_period=14, smoothing=5):
    c = df['close']
    rsi_val = _rsi(c, rsi_period)
    qqe = _ema(rsi_val, smoothing)
    co_up = (qqe.shift(1) < 50) & (qqe >= 50)
    co_down = (qqe.shift(1) > 50) & (qqe <= 50)
    sig = pd.Series(0, index=df.index)
    sig[co_up] = 1
    sig[co_down] = -1
    return sig

def space_QQE_signals(trial):
    return {'rsi_period': trial.suggest_int('rsi_period', 10, 21), 'smoothing': trial.suggest_int('smoothing', 3, 10)}

def gen_RSI_1200(df, rsi_period=1200):
    c = df['close']
    rsi_val = _rsi(c, min(rsi_period, len(c) - 1))
    long_cond = rsi_val > 50
    short_cond = rsi_val < 50
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_RSI_1200(trial):
    return {'rsi_period': trial.suggest_int('rsi_period', 200, 500)}

def gen_RSI_EMA(df, rsi_period=14, ema_period=50):
    c = df['close']
    rsi_val = _rsi(c, rsi_period)
    ema_val = _ema(c, ema_period)
    long_cond = (c > ema_val) & (rsi_val > 50)
    short_cond = (c < ema_val) & (rsi_val < 50)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_RSI_EMA(trial):
    return {'rsi_period': trial.suggest_int('rsi_period', 10, 21), 'ema_period': trial.suggest_int('ema_period', 30, 100)}

STRATEGY_EXPORT = {
    '3_EMA_SMA_Cross': {'gen': gen_3_EMA_SMA_Cross, 'space': space_3_EMA_SMA_Cross},
    'Double_Supertrend': {'gen': gen_Double_Supertrend, 'space': space_Double_Supertrend},
    'EMA_Moving_away': {'gen': gen_EMA_Moving_away, 'space': space_EMA_Moving_away},
    'HA_UnivLong_Short': {'gen': gen_HA_UnivLong_Short, 'space': space_HA_UnivLong_Short},
    'Heikin_Ashi_V2': {'gen': gen_Heikin_Ashi_V2, 'space': space_Heikin_Ashi_V2},
    'MA_Cross_DMI': {'gen': gen_MA_Cross_DMI, 'space': space_MA_Cross_DMI},
    'Supertrend': {'gen': gen_Supertrend, 'space': space_Supertrend},
    'Supertrend_EMA_Reb': {'gen': gen_Supertrend_EMA_Reb, 'space': space_Supertrend_EMA_Reb},
    'Supertrend_RSI': {'gen': gen_Supertrend_RSI, 'space': space_Supertrend_RSI},
    'Trend_EMA_RSI': {'gen': gen_Trend_EMA_RSI, 'space': space_Trend_EMA_RSI},
    '2_EMA_SMA_RSI': {'gen': gen_2_EMA_SMA_RSI, 'space': space_2_EMA_SMA_RSI},
    'DMI_Winner': {'gen': gen_DMI_Winner, 'space': space_DMI_Winner},
    'Double_RSI': {'gen': gen_Double_RSI, 'space': space_Double_RSI},
    'MACD_BB_RSI': {'gen': gen_MACD_BB_RSI, 'space': space_MACD_BB_RSI},
    'MACD_DMI': {'gen': gen_MACD_DMI, 'space': space_MACD_DMI},
    'MACD_Long_Strategy': {'gen': gen_MACD_Long_Strategy, 'space': space_MACD_Long_Strategy},
    'MACD_RSI': {'gen': gen_MACD_RSI, 'space': space_MACD_RSI},
    'QQE_signals': {'gen': gen_QQE_signals, 'space': space_QQE_signals},
    'RSI_1200': {'gen': gen_RSI_1200, 'space': space_RSI_1200},
    'RSI_EMA': {'gen': gen_RSI_EMA, 'space': space_RSI_EMA},
}
