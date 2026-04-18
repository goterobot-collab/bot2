#!/usr/bin/env python3
"""
TV2 BATCH 79 — 13 estrategias convertidas de Alorse/pinescript-strategies (v4/v5).
Categorías: Mean Reversion (8) + Grid Trading (1) + Other (4)

Mean Reversion (8):
  1. BB_Aroon              — Bollinger Bands + Aroon oscillator
  2. BB_Divergence         — BB divergence strategy
  3. BB_Winner_LITE        — Lightweight BB winner
  4. BB_Winner_PRO         — Professional BB winner
  5. Bollinger_Breakout    — BB breakout
  6. Exceeded_candle       — Candle exceeding bands
  7. MEMA_BB_RSI           — Modified EMA + BB + RSI
  8. Multi_BB              — Multiple Bollinger Bands

Grid Trading (1):
  9. GridBotDir            — Grid bot directional

Other (4):
  10. Flawless_Victory     — Complex multi-indicator
  11. Full_Candle          — Full candle strategy
  12. Improvising          — Improvised logic
  13. Javo_v1              — Javo version 1

Pine versions: v4 (10), v5 (3)
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

def _bb(c, p=20, std=2.0):
    m = _sma(c, p)
    s = c.rolling(p).std()
    return m, m + std * s, m - std * s

def _atr_ema(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/p, adjust=False).mean()

# MEAN REVERSION

def gen_BB_Aroon(df, bb_len=20, bb_std=2.0, aroon_len=25):
    c = df['close']
    h, l = df['high'], df['low']
    _, bb_upper, bb_lower = _bb(c, bb_len, bb_std)
    
    # Aroon (simplified)
    roll_high = h.rolling(aroon_len).apply(lambda x: (len(x) - 1 - np.argmax(x)), raw=True)
    roll_low = l.rolling(aroon_len).apply(lambda x: (len(x) - 1 - np.argmin(x)), raw=True)
    aroon_up = 100 * (aroon_len - roll_high) / aroon_len
    aroon_dn = 100 * (aroon_len - roll_low) / aroon_len
    aroon_osc = aroon_up - aroon_dn
    
    long_cond = (c < bb_lower) & (aroon_osc > 0)
    short_cond = (c > bb_upper) & (aroon_osc < 0)
    
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_BB_Aroon(trial):
    return {'bb_len': trial.suggest_int('bb_len', 15, 30), 'bb_std': trial.suggest_float('bb_std', 1.5, 3.0, step=0.5), 'aroon_len': trial.suggest_int('aroon_len', 15, 50)}

def gen_BB_Divergence(df, bb_len=20, bb_std=2.0):
    c = df['close']
    _, bb_upper, bb_lower = _bb(c, bb_len, bb_std)
    
    # Divergence: price exceeds band but reverses
    above_upper = c > bb_upper
    below_lower = c < bb_lower
    prev_above = above_upper.shift(1)
    prev_below = below_lower.shift(1)
    
    div_down = prev_above & (c < bb_upper)
    div_up = prev_below & (c > bb_lower)
    
    sig = pd.Series(0, index=df.index)
    sig[div_up] = 1
    sig[div_down] = -1
    return sig

def space_BB_Divergence(trial):
    return {'bb_len': trial.suggest_int('bb_len', 15, 30), 'bb_std': trial.suggest_float('bb_std', 1.5, 3.0, step=0.5)}

def gen_BB_Winner_LITE(df, bb_len=20, bb_std=2.0, rsi_period=14):
    c = df['close']
    _, bb_upper, bb_lower = _bb(c, bb_len, bb_std)
    rsi = _rsi(c, rsi_period)
    
    long_cond = (c < bb_lower) & (rsi < 30)
    short_cond = (c > bb_upper) & (rsi > 70)
    
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_BB_Winner_LITE(trial):
    return {'bb_len': trial.suggest_int('bb_len', 15, 30), 'bb_std': trial.suggest_float('bb_std', 1.5, 3.0, step=0.5), 'rsi_period': trial.suggest_int('rsi_period', 10, 21)}

def gen_BB_Winner_PRO(df, bb_len=20, bb_std=2.0, rsi_period=14, ema_period=50):
    c = df['close']
    _, bb_upper, bb_lower = _bb(c, bb_len, bb_std)
    rsi = _rsi(c, rsi_period)
    ema = _ema(c, ema_period)
    
    long_cond = (c < bb_lower) & (rsi < 30) & (c > ema)
    short_cond = (c > bb_upper) & (rsi > 70) & (c < ema)
    
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_BB_Winner_PRO(trial):
    return {'bb_len': trial.suggest_int('bb_len', 15, 30), 'bb_std': trial.suggest_float('bb_std', 1.5, 3.0, step=0.5), 'rsi_period': trial.suggest_int('rsi_period', 10, 21), 'ema_period': trial.suggest_int('ema_period', 30, 100)}

def gen_Bollinger_Breakout(df, bb_len=20, bb_std=2.0, vol_period=14):
    c = df['close']
    v = df.get('volume', pd.Series(1, index=df.index))
    _, bb_upper, bb_lower = _bb(c, bb_len, bb_std)
    vol_avg = v.rolling(vol_period).mean()
    vol_ratio = v / (vol_avg + 1e-10)
    
    co_above = (c.shift(1) < bb_upper) & (c >= bb_upper) & (vol_ratio > 1.2)
    co_below = (c.shift(1) > bb_lower) & (c <= bb_lower) & (vol_ratio > 1.2)
    
    sig = pd.Series(0, index=df.index)
    sig[co_above] = 1
    sig[co_below] = -1
    return sig

def space_Bollinger_Breakout(trial):
    return {'bb_len': trial.suggest_int('bb_len', 15, 30), 'bb_std': trial.suggest_float('bb_std', 1.5, 3.0, step=0.5), 'vol_period': trial.suggest_int('vol_period', 10, 25)}

def gen_Exceeded_candle(df, bb_len=20, bb_std=2.0):
    c = df['close']
    h = df['high']
    l = df['low']
    _, bb_upper, bb_lower = _bb(c, bb_len, bb_std)
    
    # Candle exceeds above upper band
    high_exceeds = h > bb_upper
    prev_below = c.shift(1) < bb_upper
    
    # Candle exceeds below lower band
    low_exceeds = l < bb_lower
    prev_above = c.shift(1) > bb_lower
    
    long_cond = high_exceeds & prev_below
    short_cond = low_exceeds & prev_above
    
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Exceeded_candle(trial):
    return {'bb_len': trial.suggest_int('bb_len', 15, 30), 'bb_std': trial.suggest_float('bb_std', 1.5, 3.0, step=0.5)}

def gen_MEMA_BB_RSI(df, ema_fast=12, ema_slow=26, bb_len=20, bb_std=2.0, rsi_period=14):
    c = df['close']
    ema_f = _ema(c, ema_fast)
    ema_s = _ema(c, ema_slow)
    _, bb_upper, bb_lower = _bb(c, bb_len, bb_std)
    rsi = _rsi(c, rsi_period)
    
    long_cond = (ema_f > ema_s) & (c > bb_lower) & (rsi > 50)
    short_cond = (ema_f < ema_s) & (c < bb_upper) & (rsi < 50)
    
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_MEMA_BB_RSI(trial):
    return {'ema_fast': trial.suggest_int('ema_fast', 8, 16), 'ema_slow': trial.suggest_int('ema_slow', 20, 35), 'bb_len': trial.suggest_int('bb_len', 15, 30), 'bb_std': trial.suggest_float('bb_std', 1.5, 3.0, step=0.5), 'rsi_period': trial.suggest_int('rsi_period', 10, 21)}

def gen_Multi_BB(df, bb_len=20, bb_std1=1.5, bb_std2=2.5):
    c = df['close']
    _, bb_u1, bb_l1 = _bb(c, bb_len, bb_std1)
    _, bb_u2, bb_l2 = _bb(c, bb_len, bb_std2)
    
    # Multi-level entry
    long_cond = (c < bb_l2) & (c.shift(1) > bb_l2)
    short_cond = (c > bb_u2) & (c.shift(1) < bb_u2)
    
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Multi_BB(trial):
    return {'bb_len': trial.suggest_int('bb_len', 15, 30), 'bb_std1': trial.suggest_float('bb_std1', 1.0, 2.0, step=0.5), 'bb_std2': trial.suggest_float('bb_std2', 2.0, 3.5, step=0.5)}

# GRID TRADING

def gen_GridBotDir(df, grid_step=0.5, grid_levels=5):
    c = df['close']
    o = df['open']
    
    # Simplified grid: long if price breaks below, short if breaks above
    lowest = c.rolling(20).min()
    highest = c.rolling(20).max()
    
    mid = (lowest + highest) / 2
    
    long_cond = c < (mid - grid_step / 100 * mid)
    short_cond = c > (mid + grid_step / 100 * mid)
    
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_GridBotDir(trial):
    return {'grid_step': trial.suggest_float('grid_step', 0.3, 2.0, step=0.1), 'grid_levels': trial.suggest_int('grid_levels', 3, 10)}

# OTHER

def gen_Flawless_Victory(df, ema1=20, ema2=50, rsi_period=14, bb_len=20):
    c = df['close']
    e1 = _ema(c, ema1)
    e2 = _ema(c, ema2)
    rsi = _rsi(c, rsi_period)
    _, bb_u, bb_l = _bb(c, bb_len)
    
    long_cond = (c > e1) & (e1 > e2) & (rsi > 50) & (c > bb_l)
    short_cond = (c < e1) & (e1 < e2) & (rsi < 50) & (c < bb_u)
    
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Flawless_Victory(trial):
    return {'ema1': trial.suggest_int('ema1', 10, 30), 'ema2': trial.suggest_int('ema2', 40, 100), 'rsi_period': trial.suggest_int('rsi_period', 10, 21), 'bb_len': trial.suggest_int('bb_len', 15, 30)}

def gen_Full_Candle(df, atr_mult=2.0):
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    atr = _atr_ema(h, l, c, 14)
    
    # Full candle: entire candle above/below previous
    prev_high = h.shift(1)
    prev_low = l.shift(1)
    
    long_cond = (l > prev_high)
    short_cond = (h < prev_low)
    
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Full_Candle(trial):
    return {'atr_mult': trial.suggest_float('atr_mult', 1.0, 3.0, step=0.5)}

def gen_Improvising(df, ema_len=20, rsi_period=14):
    c = df['close']
    ema = _ema(c, ema_len)
    rsi = _rsi(c, rsi_period)
    
    # Improvised: simple EMA + RSI
    long_cond = (c > ema) & (rsi > 45) & (rsi < 75)
    short_cond = (c < ema) & (rsi < 55) & (rsi > 25)
    
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Improvising(trial):
    return {'ema_len': trial.suggest_int('ema_len', 10, 40), 'rsi_period': trial.suggest_int('rsi_period', 10, 21)}

def gen_Javo_v1(df, ema_fast=9, ema_slow=21, rsi_period=14):
    c = df['close']
    ema_f = _ema(c, ema_fast)
    ema_s = _ema(c, ema_slow)
    rsi = _rsi(c, rsi_period)
    
    long_cond = (ema_f > ema_s) & (rsi > 50)
    short_cond = (ema_f < ema_s) & (rsi < 50)
    
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Javo_v1(trial):
    return {'ema_fast': trial.suggest_int('ema_fast', 5, 15), 'ema_slow': trial.suggest_int('ema_slow', 15, 50), 'rsi_period': trial.suggest_int('rsi_period', 10, 21)}

STRATEGY_EXPORT = {
    'BB_Aroon': {'gen': gen_BB_Aroon, 'space': space_BB_Aroon},
    'BB_Divergence': {'gen': gen_BB_Divergence, 'space': space_BB_Divergence},
    'BB_Winner_LITE': {'gen': gen_BB_Winner_LITE, 'space': space_BB_Winner_LITE},
    'BB_Winner_PRO': {'gen': gen_BB_Winner_PRO, 'space': space_BB_Winner_PRO},
    'Bollinger_Breakout': {'gen': gen_Bollinger_Breakout, 'space': space_Bollinger_Breakout},
    'Exceeded_candle': {'gen': gen_Exceeded_candle, 'space': space_Exceeded_candle},
    'MEMA_BB_RSI': {'gen': gen_MEMA_BB_RSI, 'space': space_MEMA_BB_RSI},
    'Multi_BB': {'gen': gen_Multi_BB, 'space': space_Multi_BB},
    'GridBotDir': {'gen': gen_GridBotDir, 'space': space_GridBotDir},
    'Flawless_Victory': {'gen': gen_Flawless_Victory, 'space': space_Flawless_Victory},
    'Full_Candle': {'gen': gen_Full_Candle, 'space': space_Full_Candle},
    'Improvising': {'gen': gen_Improvising, 'space': space_Improvising},
    'Javo_v1': {'gen': gen_Javo_v1, 'space': space_Javo_v1},
}

# ───── ADDITIONAL OTHER (5 más para completar 38 totales)

def gen_Omar_Edited_WF(df, ema_fast=12, ema_slow=26, rsi_period=14):
    """Omar Edited WF (Waveform). Pine v4 (Alorse)."""
    c = df['close']
    ema_f = _ema(c, ema_fast)
    ema_s = _ema(c, ema_slow)
    rsi = _rsi(c, rsi_period)
    long_cond = (ema_f > ema_s) & (rsi > 55)
    short_cond = (ema_f < ema_s) & (rsi < 45)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Omar_Edited_WF(trial):
    return {'ema_fast': trial.suggest_int('ema_fast', 8, 16), 'ema_slow': trial.suggest_int('ema_slow', 20, 35), 'rsi_period': trial.suggest_int('rsi_period', 10, 21)}

def gen_Omar_MMR(df, bb_len=20, bb_std=2.0, momentum_period=14):
    """Omar MMR (Market Mean Reversal). Pine v4 (Alorse)."""
    c = df['close']
    _, bb_u, bb_l = _bb(c, bb_len, bb_std)
    mom = c.diff(momentum_period)
    long_cond = (c < bb_l) & (mom < 0)
    short_cond = (c > bb_u) & (mom > 0)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Omar_MMR(trial):
    return {'bb_len': trial.suggest_int('bb_len', 15, 30), 'bb_std': trial.suggest_float('bb_std', 1.5, 3.0, step=0.5), 'momentum_period': trial.suggest_int('momentum_period', 10, 25)}

def gen_Pin_Bar_Magic_v1(df, atr_period=14, atr_mult=1.5, sensitivity=0.7):
    """Pin Bar Magic v1. Pine v4 (Alorse)."""
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    atr = _atr_ema(h, l, c, atr_period)
    
    # Pin bar: long wick, small body
    body = (c - o).abs()
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    
    is_pinbar = (upper_wick > body * 2) | (lower_wick > body * 2)
    
    long_cond = is_pinbar & (lower_wick > upper_wick)
    short_cond = is_pinbar & (upper_wick > lower_wick)
    
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Pin_Bar_Magic_v1(trial):
    return {'atr_period': trial.suggest_int('atr_period', 10, 20), 'atr_mult': trial.suggest_float('atr_mult', 1.0, 2.5, step=0.5), 'sensitivity': trial.suggest_float('sensitivity', 0.5, 1.5, step=0.1)}

def gen_StratBase(df, sma_fast=20, sma_slow=50, rsi_period=14):
    """StratBase. Pine v4 (Alorse)."""
    c = df['close']
    sma_f = _sma(c, sma_fast)
    sma_s = _sma(c, sma_slow)
    rsi = _rsi(c, rsi_period)
    long_cond = (sma_f > sma_s) & (rsi > 50)
    short_cond = (sma_f < sma_s) & (rsi < 50)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_StratBase(trial):
    return {'sma_fast': trial.suggest_int('sma_fast', 10, 35), 'sma_slow': trial.suggest_int('sma_slow', 40, 100), 'rsi_period': trial.suggest_int('rsi_period', 10, 21)}

def gen_TTM_Squeeze_EMA_Strategy(df, ema_fast=12, bb_len=20, bb_std=2.0, momentum_period=9):
    """TTM Squeeze EMA Strategy. Pine v5 (Alorse)."""
    c = df['close']
    h, l = df['high'], df['low']
    
    ema_f = _ema(c, ema_fast)
    _, bb_u, bb_l = _bb(c, bb_len, bb_std)
    atr = _atr_ema(h, l, c, 14)
    
    # Squeeze: BB width < ATR/2
    bb_width = bb_u - bb_l
    squeeze = bb_width < (atr / 2)
    
    mom = c.diff(momentum_period)
    
    long_cond = squeeze & (mom > 0) & (c > ema_f)
    short_cond = squeeze & (mom < 0) & (c < ema_f)
    
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_TTM_Squeeze_EMA_Strategy(trial):
    return {'ema_fast': trial.suggest_int('ema_fast', 8, 16), 'bb_len': trial.suggest_int('bb_len', 15, 30), 'bb_std': trial.suggest_float('bb_std', 1.5, 3.0, step=0.5), 'momentum_period': trial.suggest_int('momentum_period', 5, 15)}

# Update STRATEGY_EXPORT
STRATEGY_EXPORT.update({
    'Omar_Edited_WF': {'gen': gen_Omar_Edited_WF, 'space': space_Omar_Edited_WF},
    'Omar_MMR': {'gen': gen_Omar_MMR, 'space': space_Omar_MMR},
    'Pin_Bar_Magic_v1': {'gen': gen_Pin_Bar_Magic_v1, 'space': space_Pin_Bar_Magic_v1},
    'StratBase': {'gen': gen_StratBase, 'space': space_StratBase},
    'TTM_Squeeze_EMA_Strategy': {'gen': gen_TTM_Squeeze_EMA_Strategy, 'space': space_TTM_Squeeze_EMA_Strategy},
})
