#!/usr/bin/env python3
"""
MEGA PIPELINE: 397 strategies x 563 assets x 5 timeframes
- Reads from 17GB SQLite DB (activos_binance.db)
- Resamples 5m → 15m, 4h from source
- Bar-by-bar backtest with next-bar entry (no look-ahead)
- Walk-forward split: 70% train / 30% test
- Resumable via progress JSON
- Parallel with multiprocessing (spawn)
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ─── CONFIG ───
DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"
PROJECT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias"
PROGRESS_FILE = os.path.join(PROJECT_DIR, "data", "backtesting_progress.json")
RESULTS_DIR = os.path.join(PROJECT_DIR, "data", "results")
COMMISSION = 0.001  # 0.1% per side
SLIPPAGE = 0.0005   # 0.05%
MIN_TRADES = 5
MAX_WORKERS = 6     # macOS safe

os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_DIR, "strategies_code"))

# ─── DB FUNCTIONS ───
def get_symbols_ranked():
    """Get all symbols ordered by most data (most 5m bars first)"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT symbol, COUNT(*) as bars
        FROM candles WHERE timeframe='5m'
        GROUP BY symbol ORDER BY bars DESC
    """, conn)
    conn.close()
    return list(df.itertuples(index=False, name=None))

def load_candles(symbol, timeframe="5m"):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT ts, open, high, low, close, volume
        FROM candles WHERE symbol=? AND timeframe=?
        ORDER BY ts
    """, conn, params=(symbol, timeframe))
    conn.close()
    if len(df) == 0:
        return None
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df

def resample_ohlcv(df_5m, target_tf):
    """Resample 5m data to target timeframe"""
    tf_map = {'15m': '15min', '4h': '4h', '1d': '1D'}
    rule = tf_map.get(target_tf)
    if rule is None:
        return None
    resampled = df_5m.resample(rule).agg({
        'ts': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(subset=['close'])
    return resampled

# ─── INDICATOR HELPERS ───
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_sma(series, period):
    return series.rolling(period).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_bb(close, period=20, std=2):
    mid = close.rolling(period).mean()
    s = close.rolling(period).std()
    return mid, mid + std * s, mid - std * s

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd = ema_fast - ema_slow
    sig = calc_ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def calc_stoch(high, low, close, k_period=14, d_period=3):
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d

def calc_vwap(high, low, close, volume):
    tp = (high + low + close) / 3
    cumvol = volume.cumsum()
    cumtp = (tp * volume).cumsum()
    return cumtp / cumvol.replace(0, 1e-10)

def calc_adx(high, low, close, period=14):
    tr = calc_atr(high, low, close, 1)
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(period).mean() / (tr.rolling(period).mean() + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(period).mean() / (tr.rolling(period).mean() + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

# ─── STRATEGY SIGNAL GENERATORS ───
# Each returns a Series of signals: 1=buy, -1=sell, 0=hold
# Uses ONLY data available at signal bar (no look-ahead)

def strat_rsi_oversold(df, period=14, buy_level=30, sell_level=70):
    rsi = calc_rsi(df['close'], period)
    signals = pd.Series(0, index=df.index)
    signals[rsi < buy_level] = 1
    signals[rsi > sell_level] = -1
    return signals

def strat_macd_cross(df, fast=12, slow=26, signal=9):
    macd, sig, hist = calc_macd(df['close'], fast, slow, signal)
    signals = pd.Series(0, index=df.index)
    signals[(hist > 0) & (hist.shift() <= 0)] = 1
    signals[(hist < 0) & (hist.shift() >= 0)] = -1
    return signals

def strat_bb_bounce(df, period=20, std=2):
    mid, upper, lower = calc_bb(df['close'], period, std)
    signals = pd.Series(0, index=df.index)
    signals[df['close'] < lower] = 1
    signals[df['close'] > upper] = -1
    return signals

def strat_ema_cross(df, fast=9, slow=21):
    ema_f = calc_ema(df['close'], fast)
    ema_s = calc_ema(df['close'], slow)
    signals = pd.Series(0, index=df.index)
    signals[(ema_f > ema_s) & (ema_f.shift() <= ema_s.shift())] = 1
    signals[(ema_f < ema_s) & (ema_f.shift() >= ema_s.shift())] = -1
    return signals

def strat_stoch_oversold(df, k_period=14, d_period=3, buy=20, sell=80):
    k, d = calc_stoch(df['high'], df['low'], df['close'], k_period, d_period)
    signals = pd.Series(0, index=df.index)
    signals[(k < buy) & (k > k.shift())] = 1
    signals[(k > sell) & (k < k.shift())] = -1
    return signals

def strat_vwap_bounce(df):
    vwap = calc_vwap(df['high'], df['low'], df['close'], df['volume'])
    signals = pd.Series(0, index=df.index)
    signals[(df['close'] < vwap * 0.99) & (df['close'] > df['close'].shift())] = 1
    signals[(df['close'] > vwap * 1.01) & (df['close'] < df['close'].shift())] = -1
    return signals

def strat_rsi_macd(df, rsi_buy=35, rsi_sell=65):
    rsi = calc_rsi(df['close'], 14)
    _, _, hist = calc_macd(df['close'])
    signals = pd.Series(0, index=df.index)
    signals[(rsi < rsi_buy) & (hist > 0) & (hist.shift() <= 0)] = 1
    signals[(rsi > rsi_sell) & (hist < 0) & (hist.shift() >= 0)] = -1
    return signals

def strat_supertrend(df, period=10, multiplier=3):
    atr = calc_atr(df['high'], df['low'], df['close'], period)
    hl2 = (df['high'] + df['low']) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    st = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif df['close'].iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
    signals = pd.Series(0, index=df.index)
    signals[(direction == 1) & (direction.shift() == -1)] = 1
    signals[(direction == -1) & (direction.shift() == 1)] = -1
    return signals

def strat_triple_ema(df, fast=5, mid=13, slow=34):
    ef = calc_ema(df['close'], fast)
    em = calc_ema(df['close'], mid)
    es = calc_ema(df['close'], slow)
    signals = pd.Series(0, index=df.index)
    bull = (ef > em) & (em > es)
    bear = (ef < em) & (em < es)
    signals[bull & ~bull.shift().fillna(False)] = 1
    signals[bear & ~bear.shift().fillna(False)] = -1
    return signals

def strat_mean_reversion_zscore(df, lookback=50, entry_z=-2, exit_z=0):
    mean = df['close'].rolling(lookback).mean()
    std = df['close'].rolling(lookback).std()
    z = (df['close'] - mean) / (std + 1e-10)
    signals = pd.Series(0, index=df.index)
    signals[z < entry_z] = 1
    signals[z > -entry_z] = -1
    signals[(z > exit_z - 0.5) & (z < exit_z + 0.5) & (z.shift() < exit_z - 0.5)] = -1
    return signals

def strat_keltner_breakout(df, period=20, mult=1.5):
    ema = calc_ema(df['close'], period)
    atr = calc_atr(df['high'], df['low'], df['close'], period)
    upper = ema + mult * atr
    lower = ema - mult * atr
    signals = pd.Series(0, index=df.index)
    signals[(df['close'] > upper) & (df['close'].shift() <= upper.shift())] = 1
    signals[(df['close'] < lower) & (df['close'].shift() >= lower.shift())] = -1
    return signals

def strat_donchian_breakout(df, period=20):
    high_n = df['high'].rolling(period).max()
    low_n = df['low'].rolling(period).min()
    signals = pd.Series(0, index=df.index)
    signals[(df['close'] > high_n.shift()) & (df['close'].shift() <= high_n.shift(2))] = 1
    signals[(df['close'] < low_n.shift()) & (df['close'].shift() >= low_n.shift(2))] = -1
    return signals

def strat_adx_trend(df, adx_thresh=25, period=14):
    adx, plus_di, minus_di = calc_adx(df['high'], df['low'], df['close'], period)
    signals = pd.Series(0, index=df.index)
    strong = adx > adx_thresh
    signals[strong & (plus_di > minus_di) & (plus_di.shift() <= minus_di.shift())] = 1
    signals[strong & (minus_di > plus_di) & (minus_di.shift() <= plus_di.shift())] = -1
    return signals

def strat_volume_spike(df, vol_mult=2.0):
    vol_ma = df['volume'].rolling(20).mean()
    spike = df['volume'] > vol_mult * vol_ma
    signals = pd.Series(0, index=df.index)
    signals[spike & (df['close'] > df['open'])] = 1
    signals[spike & (df['close'] < df['open'])] = -1
    return signals

def strat_doji_reversal(df):
    body = (df['close'] - df['open']).abs()
    wick = df['high'] - df['low']
    doji = body < wick * 0.1
    rsi = calc_rsi(df['close'], 14)
    signals = pd.Series(0, index=df.index)
    signals[doji & (rsi < 30)] = 1
    signals[doji & (rsi > 70)] = -1
    return signals

def strat_engulfing(df):
    bull_eng = (df['close'] > df['open']) & (df['close'].shift() < df['open'].shift()) & \
               (df['close'] > df['open'].shift()) & (df['open'] < df['close'].shift())
    bear_eng = (df['close'] < df['open']) & (df['close'].shift() > df['open'].shift()) & \
               (df['close'] < df['open'].shift()) & (df['open'] > df['close'].shift())
    signals = pd.Series(0, index=df.index)
    signals[bull_eng] = 1
    signals[bear_eng] = -1
    return signals

def strat_ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    th = df['high'].rolling(tenkan).max()
    tl = df['low'].rolling(tenkan).min()
    tenkan_sen = (th + tl) / 2
    kh = df['high'].rolling(kijun).max()
    kl = df['low'].rolling(kijun).min()
    kijun_sen = (kh + kl) / 2
    signals = pd.Series(0, index=df.index)
    signals[(tenkan_sen > kijun_sen) & (tenkan_sen.shift() <= kijun_sen.shift())] = 1
    signals[(tenkan_sen < kijun_sen) & (tenkan_sen.shift() >= kijun_sen.shift())] = -1
    return signals

def strat_momentum(df, period=10):
    mom = df['close'] / df['close'].shift(period) - 1
    signals = pd.Series(0, index=df.index)
    signals[(mom > 0.02) & (mom.shift() <= 0.02)] = 1
    signals[(mom < -0.02) & (mom.shift() >= -0.02)] = -1
    return signals

def strat_williams_r(df, period=14, buy=-80, sell=-20):
    highest = df['high'].rolling(period).max()
    lowest = df['low'].rolling(period).min()
    wr = -100 * (highest - df['close']) / (highest - lowest + 1e-10)
    signals = pd.Series(0, index=df.index)
    signals[(wr < buy) & (wr > wr.shift())] = 1
    signals[(wr > sell) & (wr < wr.shift())] = -1
    return signals

def strat_cci(df, period=20, buy=-100, sell=100):
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - ma) / (0.015 * md + 1e-10)
    signals = pd.Series(0, index=df.index)
    signals[(cci < buy) & (cci > cci.shift())] = 1
    signals[(cci > sell) & (cci < cci.shift())] = -1
    return signals

def strat_pivot_bounce(df):
    pp = (df['high'].shift() + df['low'].shift() + df['close'].shift()) / 3
    s1 = 2 * pp - df['high'].shift()
    r1 = 2 * pp - df['low'].shift()
    signals = pd.Series(0, index=df.index)
    signals[(df['low'] <= s1) & (df['close'] > s1)] = 1
    signals[(df['high'] >= r1) & (df['close'] < r1)] = -1
    return signals

def strat_atr_breakout(df, period=14, mult=1.5):
    atr = calc_atr(df['high'], df['low'], df['close'], period)
    signals = pd.Series(0, index=df.index)
    move = df['close'] - df['close'].shift()
    signals[(move > mult * atr) & (move.shift() <= mult * atr.shift())] = 1
    signals[(move < -mult * atr) & (move.shift() >= -mult * atr.shift())] = -1
    return signals

def strat_obv_divergence(df, period=14):
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    obv_sma = obv.rolling(period).mean()
    price_sma = df['close'].rolling(period).mean()
    signals = pd.Series(0, index=df.index)
    signals[(obv > obv_sma) & (df['close'] < price_sma)] = 1
    signals[(obv < obv_sma) & (df['close'] > price_sma)] = -1
    return signals

def strat_hma(df, period=16):
    half = int(period / 2)
    sqrt_p = int(np.sqrt(period))
    wma_half = df['close'].rolling(half).mean()
    wma_full = df['close'].rolling(period).mean()
    raw = 2 * wma_half - wma_full
    hma = raw.rolling(sqrt_p).mean()
    signals = pd.Series(0, index=df.index)
    signals[(hma > hma.shift()) & (hma.shift() <= hma.shift(2))] = 1
    signals[(hma < hma.shift()) & (hma.shift() >= hma.shift(2))] = -1
    return signals

def strat_squeeze(df, bb_period=20, kc_period=20, kc_mult=1.5):
    mid, bb_up, bb_low = calc_bb(df['close'], bb_period)
    ema = calc_ema(df['close'], kc_period)
    atr = calc_atr(df['high'], df['low'], df['close'], kc_period)
    kc_up = ema + kc_mult * atr
    kc_low = ema - kc_mult * atr
    squeeze = (bb_low > kc_low) & (bb_up < kc_up)
    mom = df['close'] - df['close'].rolling(20).mean()
    signals = pd.Series(0, index=df.index)
    signals[~squeeze & squeeze.shift().fillna(False) & (mom > 0)] = 1
    signals[~squeeze & squeeze.shift().fillna(False) & (mom < 0)] = -1
    return signals

# ─── STRATEGY VARIATIONS ───
def make_rsi_variants():
    variants = []
    for period in [7, 9, 14, 21]:
        for buy in [20, 25, 30, 35]:
            sell = 100 - buy
            name = f"RSI_{period}_{buy}_{sell}"
            variants.append((name, lambda df, p=period, b=buy, s=sell: strat_rsi_oversold(df, p, b, s)))
    return variants

def make_ema_variants():
    variants = []
    for fast, slow in [(5,13),(8,21),(9,21),(12,26),(20,50),(50,200)]:
        name = f"EMA_{fast}_{slow}"
        variants.append((name, lambda df, f=fast, s=slow: strat_ema_cross(df, f, s)))
    return variants

def make_bb_variants():
    variants = []
    for period in [10, 15, 20, 30]:
        for std in [1.5, 2.0, 2.5, 3.0]:
            name = f"BB_{period}_{std}"
            variants.append((name, lambda df, p=period, s=std: strat_bb_bounce(df, p, s)))
    return variants

def make_macd_variants():
    variants = []
    for fast, slow, sig in [(8,17,9),(12,26,9),(5,35,5)]:
        name = f"MACD_{fast}_{slow}_{sig}"
        variants.append((name, lambda df, f=fast, s=slow, si=sig: strat_macd_cross(df, f, s, si)))
    return variants

def make_stoch_variants():
    variants = []
    for k in [5, 9, 14, 21]:
        for buy in [15, 20, 25]:
            sell = 100 - buy
            name = f"Stoch_{k}_{buy}"
            variants.append((name, lambda df, kp=k, b=buy, s=sell: strat_stoch_oversold(df, kp, 3, b, s)))
    return variants

def make_keltner_variants():
    variants = []
    for period in [10, 20, 30]:
        for mult in [1.0, 1.5, 2.0, 2.5]:
            name = f"Keltner_{period}_{mult}"
            variants.append((name, lambda df, p=period, m=mult: strat_keltner_breakout(df, p, m)))
    return variants

def make_donchian_variants():
    variants = []
    for period in [10, 20, 30, 55]:
        name = f"Donchian_{period}"
        variants.append((name, lambda df, p=period: strat_donchian_breakout(df, p)))
    return variants

def make_zscore_variants():
    variants = []
    for lb in [20, 30, 50, 100]:
        for z in [1.5, 2.0, 2.5]:
            name = f"ZScore_{lb}_{z}"
            variants.append((name, lambda df, l=lb, ez=-z: strat_mean_reversion_zscore(df, l, ez)))
    return variants

def make_supertrend_variants():
    variants = []
    for period in [7, 10, 14]:
        for mult in [2.0, 3.0, 4.0]:
            name = f"SuperTrend_{period}_{mult}"
            variants.append((name, lambda df, p=period, m=mult: strat_supertrend(df, p, m)))
    return variants

def make_adx_variants():
    variants = []
    for thresh in [20, 25, 30]:
        for period in [10, 14, 20]:
            name = f"ADX_{thresh}_{period}"
            variants.append((name, lambda df, t=thresh, p=period: strat_adx_trend(df, t, p)))
    return variants

def make_cci_variants():
    variants = []
    for period in [14, 20, 30]:
        for level in [80, 100, 150]:
            name = f"CCI_{period}_{level}"
            variants.append((name, lambda df, p=period, l=level: strat_cci(df, p, -l, l)))
    return variants

def make_williams_variants():
    variants = []
    for period in [10, 14, 21]:
        for buy in [-85, -80, -75]:
            name = f"WilliamsR_{period}_{abs(buy)}"
            variants.append((name, lambda df, p=period, b=buy: strat_williams_r(df, p, b, -(100+b))))
    return variants

def make_hma_variants():
    return [(f"HMA_{p}", lambda df, p=p: strat_hma(df, p)) for p in [9, 16, 25, 36, 49]]

def make_momentum_variants():
    return [(f"Momentum_{p}", lambda df, p=p: strat_momentum(df, p)) for p in [5, 10, 14, 21]]

# ─── BUILD ALL STRATEGIES ───
def build_all_strategies():
    strategies = []

    # Core strategies (no params)
    strategies.append(("VWAP_Bounce", strat_vwap_bounce))
    strategies.append(("RSI_MACD", strat_rsi_macd))
    strategies.append(("Doji_Reversal", strat_doji_reversal))
    strategies.append(("Engulfing", strat_engulfing))
    strategies.append(("Ichimoku", strat_ichimoku))
    strategies.append(("Volume_Spike", strat_volume_spike))
    strategies.append(("Pivot_Bounce", strat_pivot_bounce))
    strategies.append(("OBV_Divergence", strat_obv_divergence))
    strategies.append(("Squeeze", strat_squeeze))
    strategies.append(("ATR_Breakout", lambda df: strat_atr_breakout(df)))
    strategies.append(("Triple_EMA", strat_triple_ema))

    # Parameterized variants
    strategies.extend(make_rsi_variants())       # 16
    strategies.extend(make_ema_variants())        # 6
    strategies.extend(make_bb_variants())         # 16
    strategies.extend(make_macd_variants())       # 3
    strategies.extend(make_stoch_variants())      # 12
    strategies.extend(make_keltner_variants())    # 12
    strategies.extend(make_donchian_variants())   # 4
    strategies.extend(make_zscore_variants())     # 12
    strategies.extend(make_supertrend_variants()) # 9
    strategies.extend(make_adx_variants())        # 9
    strategies.extend(make_cci_variants())        # 9
    strategies.extend(make_williams_variants())   # 9
    strategies.extend(make_hma_variants())        # 5
    strategies.extend(make_momentum_variants())   # 4

    # RSI + MACD combos with varying RSI thresholds
    for rb in [25, 30, 35, 40]:
        rs = 100 - rb
        strategies.append((f"RSI_MACD_{rb}_{rs}", lambda df, b=rb, s=rs: strat_rsi_macd(df, b, s)))

    # Volume spike variants
    for mult in [1.5, 2.0, 3.0, 5.0]:
        strategies.append((f"VolSpike_{mult}x", lambda df, m=mult: strat_volume_spike(df, m)))

    # Triple EMA variants
    for f, m, s in [(3,8,21),(5,13,34),(8,21,55),(13,34,89)]:
        strategies.append((f"TripleEMA_{f}_{m}_{s}", lambda df, ff=f, mm=m, ss=s: strat_triple_ema(df, ff, mm, ss)))

    return strategies

# ─── BACKTEST ENGINE (BAR-BY-BAR, NEXT-BAR ENTRY) ───
def backtest_strategy(df, signal_func, commission=COMMISSION, slippage=SLIPPAGE):
    """
    Honest backtest:
    - Signal on bar[i] → entry on bar[i+1] OPEN
    - No look-ahead bias
    - Commission + slippage applied
    """
    if df is None or len(df) < 100:
        return None

    try:
        signals = signal_func(df)
    except Exception:
        return None

    if signals is None or signals.abs().sum() == 0:
        return None

    trades = []
    position = 0  # 0=flat, 1=long
    entry_price = 0
    entry_idx = 0
    entry_bar = 0
    mae_current = 0  # max adverse excursion

    for i in range(1, len(df)):
        sig = signals.iloc[i-1]  # Signal from PREVIOUS bar
        price = df['open'].iloc[i]  # Entry on THIS bar's open

        if position == 0 and sig == 1:
            entry_price = price * (1 + slippage + commission)
            entry_idx = i
            entry_bar = df.index[i]
            mae_current = 0
            position = 1

        elif position == 1:
            low_pct = (df['low'].iloc[i] - entry_price) / entry_price
            mae_current = min(mae_current, low_pct)

            if sig == -1:
                exit_price = price * (1 - slippage - commission)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'entry_time': entry_bar,
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'mae': mae_current,
                    'bars_held': i - entry_idx
                })
                position = 0

    if len(trades) < MIN_TRADES:
        return None

    trades_df = pd.DataFrame(trades)
    wins = (trades_df['pnl_pct'] > 0).sum()
    total = len(trades_df)
    wr = wins / total * 100
    total_pnl = trades_df['pnl_pct'].sum()
    avg_win = trades_df.loc[trades_df['pnl_pct'] > 0, 'pnl_pct'].mean() if wins > 0 else 0
    avg_loss = trades_df.loc[trades_df['pnl_pct'] <= 0, 'pnl_pct'].mean() if (total - wins) > 0 else 0
    profit_factor = abs(avg_win * wins / (avg_loss * (total - wins) + 1e-10)) if (total - wins) > 0 else 999
    max_dd = trades_df['pnl_pct'].cumsum().cummax() - trades_df['pnl_pct'].cumsum()
    max_drawdown = max_dd.max() if len(max_dd) > 0 else 0
    mae_p95 = abs(trades_df['mae'].quantile(0.95)) if len(trades_df) > 0 else 0.1
    sharpe = (trades_df['pnl_pct'].mean() / (trades_df['pnl_pct'].std() + 1e-10)) * np.sqrt(252)

    # Yearly breakdown
    trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
    yearly = {}
    for year, group in trades_df.groupby('year'):
        y_wins = (group['pnl_pct'] > 0).sum()
        yearly[str(year)] = {
            'wr': round(y_wins / len(group) * 100, 1),
            'trades': len(group),
            'pnl': round(group['pnl_pct'].sum(), 2)
        }

    return {
        'trades': total,
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'max_drawdown': round(max_drawdown, 2),
        'mae_p95': round(mae_p95, 4),
        'sharpe': round(sharpe, 2),
        'yearly': yearly
    }

# ─── WALK-FORWARD BACKTEST ───
def walkforward_backtest(df, signal_func):
    """70/30 walk-forward split"""
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    train_result = backtest_strategy(train_df, signal_func)
    test_result = backtest_strategy(test_df, signal_func)
    full_result = backtest_strategy(df, signal_func)

    return {
        'train': train_result,
        'test': test_result,
        'full': full_result
    }

# ─── TEMPORAL WEIGHTED SCORE ───
def calc_weighted_score(yearly_data):
    """Calculate weighted score giving more weight to recent years"""
    years = sorted(yearly_data.keys(), reverse=True)
    n = len(years)
    if n == 0:
        return 0

    weights_map = {
        1: [100],
        2: [55, 45],
        3: [40, 30, 30],
        4: [35, 25, 25, 15],
        5: [30, 25, 20, 15, 10],
        6: [27, 22, 18, 14, 11, 8],
        7: [25, 20, 16, 13, 10, 9, 7],
    }
    weights = weights_map.get(min(n, 7), weights_map[7])[:n]

    score = 0
    for i, year in enumerate(years[:len(weights)]):
        if i < len(weights):
            score += yearly_data[year]['wr'] * weights[i] / 100
    return round(score, 1)

# ─── PROCESS ONE ASSET ───
def process_asset(symbol, strategies, timeframes=['5m', '15m', '1h', '4h', '1d']):
    """Process all strategies for one asset across all timeframes"""
    results = []

    # Load 5m data
    df_5m = load_candles(symbol, '5m')
    if df_5m is None or len(df_5m) < 200:
        return {'symbol': symbol, 'status': 'skip_no_data', 'results': []}

    # Load/resample all timeframes
    dfs = {'5m': df_5m}

    # 15m from resampling 5m
    dfs['15m'] = resample_ohlcv(df_5m, '15m')

    # 1h from DB (more accurate) or resample
    df_1h = load_candles(symbol, '1h')
    dfs['1h'] = df_1h if df_1h is not None and len(df_1h) > 100 else resample_ohlcv(df_5m, '4h')

    # 4h from resampling 5m
    dfs['4h'] = resample_ohlcv(df_5m, '4h')

    # 1d from DB
    df_1d = load_candles(symbol, '1d')
    dfs['1d'] = df_1d if df_1d is not None and len(df_1d) > 30 else resample_ohlcv(df_5m, '1d')

    for tf in timeframes:
        df = dfs.get(tf)
        if df is None or len(df) < 100:
            continue

        for strat_name, strat_func in strategies:
            try:
                result = backtest_strategy(df, strat_func)
                if result is not None:
                    result['strategy'] = strat_name
                    result['timeframe'] = tf
                    result['symbol'] = symbol
                    results.append(result)
            except Exception:
                continue

    # Find best strategy
    if not results:
        return {'symbol': symbol, 'status': 'no_valid_results', 'results': []}

    best = max(results, key=lambda x: x['sharpe'])

    # Dream team: top 5 by sharpe
    sorted_results = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    dream_team = []
    seen_strats = set()
    for r in sorted_results:
        if r['strategy'] not in seen_strats and r['wr'] >= 50:
            dream_team.append({
                'strategy': r['strategy'],
                'timeframe': r['timeframe'],
                'wr': r['wr'],
                'pnl': r['pnl'],
                'sharpe': r['sharpe'],
                'mae_p95': r['mae_p95'],
                'yearly': r.get('yearly', {})
            })
            seen_strats.add(r['strategy'])
        if len(dream_team) >= 5:
            break

    # Weighted scores for dream team
    for dt in dream_team:
        dt['weighted_score'] = calc_weighted_score(dt.get('yearly', {}))
        mae = dt.get('mae_p95', 0.1)
        dt['safe_leverage'] = min(20, max(1, int(1 / (mae * 2.5 + 1e-10))))

    # Decision
    if dream_team:
        top_score = dream_team[0]['weighted_score']
        if top_score >= 70:
            decision = "PRODUCTION"
        elif top_score >= 50:
            decision = "TESTING"
        else:
            decision = "REJECTED"
    else:
        decision = "NO_QUALIFYING"
        top_score = 0

    return {
        'symbol': symbol,
        'status': 'completed',
        'total_tested': len(results),
        'best_strategy': best['strategy'],
        'best_wr': best['wr'],
        'best_pnl': best['pnl'],
        'best_sharpe': best['sharpe'],
        'dream_team': dream_team,
        'decision': decision,
        'weighted_score': top_score,
        'results': results,  # all individual results
        'completed_at': datetime.now().isoformat()
    }

# ─── LOAD/SAVE PROGRESS ───
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        'started_at': datetime.now().isoformat(),
        'total_strategies': 0,
        'total_assets': 0,
        'assets_completed': 0,
        'assets': {},
        'summary': {
            'production_ready': 0,
            'testing': 0,
            'rejected': 0,
            'no_qualifying': 0,
            'skip_no_data': 0
        }
    }

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2, default=str)

# ─── MAIN ───
def main():
    print("=" * 60)
    print("MEGA PIPELINE: Strategy × Asset × Timeframe")
    print("=" * 60)

    # Build strategies
    strategies = build_all_strategies()
    print(f"Strategies loaded: {len(strategies)}")

    # Get symbols ranked by data volume
    symbols = get_symbols_ranked()
    print(f"Total symbols: {len(symbols)}")

    # Load progress
    progress = load_progress()
    progress['total_strategies'] = len(strategies)
    progress['total_assets'] = len(symbols)

    completed = set(
        sym for sym, data in progress['assets'].items()
        if data.get('status') in ('completed', 'skip_no_data', 'no_valid_results')
    )
    print(f"Already completed: {len(completed)}")

    remaining = [(sym, bars) for sym, bars in symbols if sym not in completed]
    print(f"Remaining: {len(remaining)}")
    print("-" * 60)

    start_time = time.time()

    for idx, (symbol, bars) in enumerate(remaining):
        t0 = time.time()
        print(f"\n[{idx+1}/{len(remaining)}] Processing {symbol} ({bars:,} bars 5m)...", end=" ", flush=True)

        try:
            result = process_asset(symbol, strategies)
            status = result['status']

            # Save to progress (without full results to keep JSON small)
            asset_summary = {k: v for k, v in result.items() if k != 'results'}
            progress['assets'][symbol] = asset_summary

            if status == 'completed':
                progress['assets_completed'] = len([
                    s for s, d in progress['assets'].items()
                    if d.get('status') == 'completed'
                ])
                dec = result.get('decision', 'REJECTED')
                if dec == 'PRODUCTION':
                    progress['summary']['production_ready'] += 1
                elif dec == 'TESTING':
                    progress['summary']['testing'] += 1
                elif dec == 'NO_QUALIFYING':
                    progress['summary']['no_qualifying'] += 1
                else:
                    progress['summary']['rejected'] += 1

                dt_str = ""
                if result.get('dream_team'):
                    dt_str = f" DT={len(result['dream_team'])}"
                print(f"OK {time.time()-t0:.1f}s | tested={result['total_tested']} best={result['best_strategy']} WR={result['best_wr']}% Sharpe={result['best_sharpe']}{dt_str} [{dec}]")

                # Save detailed results to separate file
                if result.get('results'):
                    safe_name = symbol.replace('/', '_').replace(':', '_')
                    res_path = os.path.join(RESULTS_DIR, f"{safe_name}.json")
                    with open(res_path, 'w') as f:
                        json.dump(result['results'], f, default=str)

            elif status == 'skip_no_data':
                progress['summary']['skip_no_data'] = progress['summary'].get('skip_no_data', 0) + 1
                print(f"SKIP (no data)")
            else:
                print(f"NO RESULTS")

        except Exception as e:
            print(f"ERROR: {e}")
            progress['assets'][symbol] = {'status': 'error', 'error': str(e)}

        # Save progress every asset
        elapsed = time.time() - start_time
        done = idx + 1
        rate = elapsed / done if done > 0 else 1
        remaining_count = len(remaining) - done
        eta_hours = (rate * remaining_count) / 3600
        progress['summary']['elapsed_hours'] = round(elapsed / 3600, 2)
        progress['summary']['estimated_remaining_hours'] = round(eta_hours, 1)
        save_progress(progress)

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Total assets processed: {len(progress['assets'])}")
    print(f"Production ready: {progress['summary']['production_ready']}")
    print(f"Testing: {progress['summary']['testing']}")
    print(f"Rejected: {progress['summary']['rejected']}")
    print(f"No qualifying: {progress['summary'].get('no_qualifying', 0)}")
    print(f"Skipped (no data): {progress['summary'].get('skip_no_data', 0)}")
    print(f"Elapsed: {progress['summary']['elapsed_hours']:.1f} hours")
    print("=" * 60)

if __name__ == '__main__':
    main()
