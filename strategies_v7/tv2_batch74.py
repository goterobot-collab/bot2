#!/usr/bin/env python3
"""
TV2 BATCH 74 — Estrategias convertidas de GitHub Pine Script v5
Fecha: 2026-04-01
Total: 46 estrategias nuevas
"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# ESTRATEGIAS
# ─────────────────────────────────────────────────────────────────────────────

def gen_Supertrend_V5_Basic(df, **p):
    period = p.get('period', 10)
    mult = p.get('mult', 3.0)

    hl2 = (df['high'] + df['low']) / 2
    atr = df['high'].rolling(period).mean() - df['low'].rolling(period).mean()

    ub = hl2 + mult * atr
    lb = hl2 - mult * atr

    close = df['close'].values
    trend = np.ones(len(close))

    for i in range(1, len(close)):
        if close[i] > ub.iloc[i-1]:
            trend[i] = 1
        elif close[i] < lb.iloc[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    df['signal'] = trend
    return df

def space_Supertrend_V5_Basic():
    return {
        'period': (5, 20, 1),
        'mult': (1.5, 5.0, 0.5),
    }

def gen_EMA_Crossover_3_Period(df, **p):
    ema1 = p.get('ema1', 5)
    ema2 = p.get('ema2', 20)
    ema3 = p.get('ema3', 50)

    df['ema1'] = df['close'].ewm(span=ema1).mean()
    df['ema2'] = df['close'].ewm(span=ema2).mean()
    df['ema3'] = df['close'].ewm(span=ema3).mean()

    signal = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['ema1'].iloc[i] > df['ema2'].iloc[i] > df['ema3'].iloc[i]:
            signal[i] = 1
        elif df['ema1'].iloc[i] < df['ema2'].iloc[i] < df['ema3'].iloc[i]:
            signal[i] = -1

    df['signal'] = signal
    return df

def space_EMA_Crossover_3_Period():
    return {
        'ema1': (3, 10, 1),
        'ema2': (15, 30, 3),
        'ema3': (40, 80, 10),
    }

def gen_RSI_Oversold_Long(df, **p):
    period = p.get('period', 14)
    oversold = p.get('oversold', 30)
    overbought = p.get('overbought', 70)

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    signal = np.zeros(len(df))
    for i in range(1, len(df)):
        if rsi.iloc[i] < oversold and rsi.iloc[i-1] >= oversold:
            signal[i] = 1
        elif rsi.iloc[i] > overbought:
            signal[i] = -1

    df['signal'] = signal
    return df

def space_RSI_Oversold_Long():
    return {
        'period': (7, 21, 2),
        'oversold': (20, 40, 5),
        'overbought': (60, 80, 5),
    }

def gen_MACD_Crossover(df, **p):
    fast = p.get('fast', 12)
    slow = p.get('slow', 26)
    signal = p.get('signal', 9)

    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    histogram = macd - macd_signal

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if histogram.iloc[i] > 0 and histogram.iloc[i-1] <= 0:
            sig[i] = 1
        elif histogram.iloc[i] < 0 and histogram.iloc[i-1] >= 0:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_MACD_Crossover():
    return {
        'fast': (8, 15, 1),
        'slow': (20, 35, 2),
        'signal': (5, 12, 1),
    }

def gen_Bollinger_Bands_Mean_Reversion(df, **p):
    period = p.get('period', 20)
    std_dev = p.get('std_dev', 2.0)

    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()

    upper = sma + std_dev * std
    lower = sma - std_dev * std

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['close'].iloc[i] < lower.iloc[i]:
            sig[i] = 1
        elif df['close'].iloc[i] > upper.iloc[i]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Bollinger_Bands_Mean_Reversion():
    return {
        'period': (15, 30, 3),
        'std_dev': (1.5, 3.0, 0.3),
    }

def gen_Stochastic_Oversold(df, **p):
    period = p.get('period', 14)
    smooth_k = p.get('smooth_k', 3)
    smooth_d = p.get('smooth_d', 3)
    oversold = p.get('oversold', 20)

    low_min = df['low'].rolling(period).min()
    high_max = df['high'].rolling(period).max()

    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    k_smooth = k.rolling(smooth_k).mean()
    d = k_smooth.rolling(smooth_d).mean()

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if k_smooth.iloc[i] < oversold and k_smooth.iloc[i-1] >= oversold:
            sig[i] = 1
        elif k_smooth.iloc[i] > 80:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Stochastic_Oversold():
    return {
        'period': (10, 20, 2),
        'smooth_k': (2, 5, 1),
        'smooth_d': (2, 5, 1),
        'oversold': (15, 35, 5),
    }

def gen_ADX_Trend_Filter(df, **p):
    period = p.get('period', 14)
    adx_threshold = p.get('adx_threshold', 25)

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    up = df['high'].diff()
    down = -df['low'].diff()

    pos_dm = up.where((up > down) & (up > 0), 0)
    neg_dm = down.where((down > up) & (down > 0), 0)

    pos_di = 100 * pos_dm.rolling(period).mean() / atr
    neg_di = 100 * neg_dm.rolling(period).mean() / atr

    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di)
    adx = dx.rolling(period).mean()

    sig = np.zeros(len(df))
    for i in range(period, len(df)):
        if adx.iloc[i] > adx_threshold and pos_di.iloc[i] > neg_di.iloc[i]:
            sig[i] = 1
        elif adx.iloc[i] > adx_threshold and neg_di.iloc[i] > pos_di.iloc[i]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_ADX_Trend_Filter():
    return {
        'period': (10, 20, 2),
        'adx_threshold': (20, 40, 5),
    }

def gen_VWAP_Breakout(df, **p):
    vwap_window = p.get('vwap_window', 20)

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(vwap_window).sum() / df['volume'].rolling(vwap_window).sum()

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['close'].iloc[i] > vwap.iloc[i] and df['close'].iloc[i-1] <= vwap.iloc[i-1]:
            sig[i] = 1
        elif df['close'].iloc[i] < vwap.iloc[i] and df['close'].iloc[i-1] >= vwap.iloc[i-1]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_VWAP_Breakout():
    return {
        'vwap_window': (10, 30, 5),
    }

def gen_Volume_Profile_Support(df, **p):
    vol_period = p.get('vol_period', 20)

    avg_vol = df['volume'].rolling(vol_period).mean()
    vol_spike = df['volume'] > avg_vol * p.get('spike_mult', 1.5)

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if vol_spike.iloc[i] and df['close'].iloc[i] > df['close'].iloc[i-1]:
            sig[i] = 1
        elif vol_spike.iloc[i] and df['close'].iloc[i] < df['close'].iloc[i-1]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Volume_Profile_Support():
    return {
        'vol_period': (10, 30, 5),
        'spike_mult': (1.2, 2.5, 0.3),
    }

def gen_MACD_BBands_RSI(df, **p):
    fast = p.get('fast', 12)
    slow = p.get('slow', 26)
    macd_sig = p.get('macd_sig', 9)
    bb_len = p.get('bb_len', 20)
    bb_mult = p.get('bb_mult', 2.0)
    rsi_len = p.get('rsi_len', 14)

    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=macd_sig).mean()

    sma = df['close'].rolling(bb_len).mean()
    std = df['close'].rolling(bb_len).std()
    bb_upper = sma + bb_mult * std
    bb_lower = sma - bb_mult * std

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(rsi_len).mean() / loss.rolling(rsi_len).mean()
    rsi = 100 - (100 / (1 + rs))

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if macd.iloc[i] > macd_signal.iloc[i] and df['close'].iloc[i] < bb_lower.iloc[i] and rsi.iloc[i] < 50:
            sig[i] = 1
        elif macd.iloc[i] < macd_signal.iloc[i] and df['close'].iloc[i] > bb_upper.iloc[i] and rsi.iloc[i] > 50:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_MACD_BBands_RSI():
    return {
        'fast': (8, 15, 1),
        'slow': (20, 35, 2),
        'macd_sig': (5, 12, 1),
        'bb_len': (15, 30, 3),
        'bb_mult': (1.5, 3.0, 0.3),
        'rsi_len': (10, 21, 2),
    }

def gen_DMI_Winner(df, **p):
    period = p.get('period', 14)
    threshold = p.get('threshold', 25)

    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()

    pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    pos_di = 100 * pos_dm.rolling(period).mean() / atr
    neg_di = 100 * neg_dm.rolling(period).mean() / atr

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if pos_di.iloc[i] > neg_di.iloc[i] and (pos_di.iloc[i] - neg_di.iloc[i]) > threshold:
            sig[i] = 1
        elif neg_di.iloc[i] > pos_di.iloc[i] and (neg_di.iloc[i] - pos_di.iloc[i]) > threshold:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_DMI_Winner():
    return {
        'period': (10, 20, 2),
        'threshold': (15, 35, 5),
    }

def gen_Williams_Vix_Fix(df, **p):
    len1 = p.get('len1', 22)
    len2 = p.get('len2', 20)

    high_low_range = df['high'] - df['low']
    atr = high_low_range.rolling(len1).mean()

    nvi_fast = df['close'].rolling(len2).std()
    nvi_slow = df['close'].rolling(len1).std()

    ratio = nvi_fast / (nvi_slow + 1e-6)

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if ratio.iloc[i] < p.get('lower', 0.7) and ratio.iloc[i-1] >= p.get('lower', 0.7):
            sig[i] = 1
        elif ratio.iloc[i] > p.get('upper', 1.3):
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Williams_Vix_Fix():
    return {
        'len1': (15, 30, 3),
        'len2': (15, 30, 3),
        'lower': (0.5, 1.0, 0.1),
        'upper': (1.0, 1.5, 0.1),
    }

def gen_KDJ_Stochastic(df, **p):
    period = p.get('period', 9)
    smooth_k = p.get('smooth_k', 3)
    smooth_j = p.get('smooth_j', 3)

    low_min = df['low'].rolling(period).min()
    high_max = df['high'].rolling(period).max()

    fastk = 100 * (df['close'] - low_min) / (high_max - low_min)
    k = fastk.rolling(smooth_k).mean()
    d = k.rolling(smooth_j).mean()
    j = 3 * k - 2 * d

    sig = np.zeros(len(df))
    for i in range(2, len(df)):
        if k.iloc[i] > d.iloc[i] and k.iloc[i-1] <= d.iloc[i-1]:
            sig[i] = 1
        elif k.iloc[i] < d.iloc[i] and k.iloc[i-1] >= d.iloc[i-1]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_KDJ_Stochastic():
    return {
        'period': (7, 14, 2),
        'smooth_k': (2, 5, 1),
        'smooth_j': (2, 5, 1),
    }

def gen_ROC_Rate_Change(df, **p):
    period = p.get('period', 12)
    threshold = p.get('threshold', 0)

    roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

    sig = np.zeros(len(df))
    for i in range(period, len(df)):
        if roc.iloc[i] > threshold and roc.iloc[i-1] <= threshold:
            sig[i] = 1
        elif roc.iloc[i] < threshold and roc.iloc[i-1] >= threshold:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_ROC_Rate_Change():
    return {
        'period': (8, 20, 2),
        'threshold': (-5, 5, 1),
    }

def gen_TRIX_Momentum(df, **p):
    period = p.get('period', 15)

    ema1 = df['close'].ewm(span=period).mean()
    ema2 = ema1.ewm(span=period).mean()
    ema3 = ema2.ewm(span=period).mean()

    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 10000

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if trix.iloc[i] > 0 and trix.iloc[i-1] <= 0:
            sig[i] = 1
        elif trix.iloc[i] < 0 and trix.iloc[i-1] >= 0:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_TRIX_Momentum():
    return {
        'period': (10, 20, 2),
    }

def gen_Ichimoku_Cloud(df, **p):
    tenkan_period = p.get('tenkan_period', 9)
    kijun_period = p.get('kijun_period', 26)
    senkou_span_period = p.get('senkou_span_period', 52)

    high9 = df['high'].rolling(tenkan_period).max()
    low9 = df['low'].rolling(tenkan_period).min()
    tenkan = (high9 + low9) / 2

    high26 = df['high'].rolling(kijun_period).max()
    low26 = df['low'].rolling(kijun_period).min()
    kijun = (high26 + low26) / 2

    sig = np.zeros(len(df))
    for i in range(kijun_period, len(df)):
        if tenkan.iloc[i] > kijun.iloc[i]:
            sig[i] = 1
        elif tenkan.iloc[i] < kijun.iloc[i]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Ichimoku_Cloud():
    return {
        'tenkan_period': (7, 12, 1),
        'kijun_period': (20, 30, 2),
        'senkou_span_period': (45, 60, 5),
    }

def gen_Donchian_Breakout(df, **p):
    period = p.get('period', 20)

    high_max = df['high'].rolling(period).max()
    low_min = df['low'].rolling(period).min()

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['close'].iloc[i] > high_max.iloc[i-1]:
            sig[i] = 1
        elif df['close'].iloc[i] < low_min.iloc[i-1]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Donchian_Breakout():
    return {
        'period': (15, 30, 3),
    }

def gen_Fisher_Transform(df, **p):
    period = p.get('period', 10)

    high_low = (df['high'] + df['low']) / 2
    avg = high_low.rolling(period).mean()
    std = high_low.rolling(period).std()

    val = 2 * ((high_low - avg) / (std + 1e-6)) - 1
    val = val.clip(-0.999, 0.999)

    fisher = 0.5 * np.log((1 + val) / (1 - val))

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if fisher.iloc[i] > 0 and fisher.iloc[i-1] <= 0:
            sig[i] = 1
        elif fisher.iloc[i] < 0 and fisher.iloc[i-1] >= 0:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Fisher_Transform():
    return {
        'period': (8, 15, 1),
    }

def gen_Pivot_Point_SR(df, **p):
    period = p.get('period', 5)

    pivot = (df['high'].rolling(period).max() + df['low'].rolling(period).min()) / 2
    resistance = df['high'].rolling(period).max()
    support = df['low'].rolling(period).min()

    sig = np.zeros(len(df))
    for i in range(period, len(df)):
        if df['close'].iloc[i] > resistance.iloc[i-1]:
            sig[i] = 1
        elif df['close'].iloc[i] < support.iloc[i-1]:
            sig[i] = -1
        elif df['close'].iloc[i] > pivot.iloc[i-1] and df['close'].iloc[i-1] <= pivot.iloc[i-1]:
            sig[i] = 1
        elif df['close'].iloc[i] < pivot.iloc[i-1] and df['close'].iloc[i-1] >= pivot.iloc[i-1]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Pivot_Point_SR():
    return {
        'period': (3, 10, 1),
    }

def gen_Keltner_Channel(df, **p):
    period = p.get('period', 20)
    atr_mult = p.get('atr_mult', 2.0)

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    centerline = typical_price.rolling(period).mean()

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    upper = centerline + atr_mult * atr
    lower = centerline - atr_mult * atr

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i] and df['close'].iloc[i-1] <= upper.iloc[i-1]:
            sig[i] = 1
        elif df['close'].iloc[i] < lower.iloc[i] and df['close'].iloc[i-1] >= lower.iloc[i-1]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Keltner_Channel():
    return {
        'period': (15, 30, 3),
        'atr_mult': (1.5, 3.0, 0.3),
    }

def gen_Linear_Regression_Channel(df, **p):
    period = p.get('period', 20)
    std_dev = p.get('std_dev', 1.5)

    x = np.arange(period)

    sig = np.zeros(len(df))
    for i in range(period, len(df)):
        y = df['close'].iloc[i-period:i].values

        mean_x = x.mean()
        mean_y = y.mean()

        slope = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)
        intercept = mean_y - slope * mean_x

        regression_line = slope * x + intercept
        residuals = y - regression_line
        std = residuals.std()

        upper = regression_line[-1] + std_dev * std
        lower = regression_line[-1] - std_dev * std

        if df['close'].iloc[i] > upper:
            sig[i] = -1
        elif df['close'].iloc[i] < lower:
            sig[i] = 1

    df['signal'] = sig
    return df

def space_Linear_Regression_Channel():
    return {
        'period': (15, 30, 3),
        'std_dev': (1.0, 2.5, 0.3),
    }

def gen_Envelope_Trading(df, **p):
    period = p.get('period', 20)
    percent = p.get('percent', 0.025)

    sma = df['close'].rolling(period).mean()
    upper = sma * (1 + percent)
    lower = sma * (1 - percent)

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i] and df['close'].iloc[i-1] <= upper.iloc[i-1]:
            sig[i] = -1
        elif df['close'].iloc[i] < lower.iloc[i] and df['close'].iloc[i-1] >= lower.iloc[i-1]:
            sig[i] = 1

    df['signal'] = sig
    return df

def space_Envelope_Trading():
    return {
        'period': (15, 30, 3),
        'percent': (0.01, 0.05, 0.005),
    }

def gen_ZScore_Crossover(df, **p):
    period = p.get('period', 20)
    threshold = p.get('threshold', 1.5)

    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    zscore = (df['close'] - sma) / (std + 1e-6)

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if zscore.iloc[i] > threshold and zscore.iloc[i-1] <= threshold:
            sig[i] = -1
        elif zscore.iloc[i] < -threshold and zscore.iloc[i-1] >= -threshold:
            sig[i] = 1

    df['signal'] = sig
    return df

def space_ZScore_Crossover():
    return {
        'period': (15, 30, 3),
        'threshold': (1.0, 2.5, 0.25),
    }

def gen_OBV_Confirmation(df, **p):
    ema_period = p.get('ema_period', 20)

    obv = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv[i] = obv[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv[i] = obv[i-1] - df['volume'].iloc[i]
        else:
            obv[i] = obv[i-1]

    obv_ema = pd.Series(obv).ewm(span=ema_period).mean()

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if obv_ema.iloc[i] > obv_ema.iloc[i-1] and df['close'].iloc[i] > df['close'].iloc[i-1]:
            sig[i] = 1
        elif obv_ema.iloc[i] < obv_ema.iloc[i-1] and df['close'].iloc[i] < df['close'].iloc[i-1]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_OBV_Confirmation():
    return {
        'ema_period': (10, 30, 3),
    }

def gen_Money_Flow_Index(df, **p):
    period = p.get('period', 14)
    oversold = p.get('oversold', 20)
    overbought = p.get('overbought', 80)

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)

    pos_flow_sum = pd.Series(positive_flow).rolling(period).sum()
    neg_flow_sum = pd.Series(negative_flow).rolling(period).sum()

    money_flow_ratio = pos_flow_sum / neg_flow_sum
    mfi = 100 - (100 / (1 + money_flow_ratio))

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if mfi.iloc[i] < oversold and mfi.iloc[i-1] >= oversold:
            sig[i] = 1
        elif mfi.iloc[i] > overbought and mfi.iloc[i-1] <= overbought:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Money_Flow_Index():
    return {
        'period': (10, 20, 2),
        'oversold': (15, 35, 5),
        'overbought': (65, 85, 5),
    }

def gen_Accumulation_Distribution(df, **p):
    ema_period = p.get('ema_period', 21)

    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-6)
    ad = (clv * df['volume']).cumsum()
    ad_ema = pd.Series(ad).ewm(span=ema_period).mean()

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if ad_ema.iloc[i] > ad_ema.iloc[i-1]:
            sig[i] = 1
        elif ad_ema.iloc[i] < ad_ema.iloc[i-1]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Accumulation_Distribution():
    return {
        'ema_period': (15, 30, 3),
    }

def gen_Price_Volume_Trend(df, **p):
    period = p.get('period', 14)

    pct_change = df['close'].pct_change() * 100
    pvt = (pct_change * df['volume']).fillna(0).cumsum()
    pvt_sma = pd.Series(pvt).rolling(period).mean()

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if pvt.iloc[i] > pvt_sma.iloc[i] and pvt.iloc[i-1] <= pvt_sma.iloc[i-1]:
            sig[i] = 1
        elif pvt.iloc[i] < pvt_sma.iloc[i] and pvt.iloc[i-1] >= pvt_sma.iloc[i-1]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Price_Volume_Trend():
    return {
        'period': (10, 20, 2),
    }

def gen_Awesome_Oscillator(df, **p):
    fast = p.get('fast', 5)
    slow = p.get('slow', 34)

    median = (df['high'] + df['low']) / 2
    ao_fast = median.rolling(fast).mean()
    ao_slow = median.rolling(slow).mean()
    ao = ao_fast - ao_slow

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if ao.iloc[i] > 0 and ao.iloc[i-1] <= 0:
            sig[i] = 1
        elif ao.iloc[i] < 0 and ao.iloc[i-1] >= 0:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Awesome_Oscillator():
    return {
        'fast': (3, 8, 1),
        'slow': (30, 40, 2),
    }

def gen_True_Range_Crossover(df, **p):
    period = p.get('period', 14)

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    atr_ema = atr.ewm(span=period).mean()

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if atr.iloc[i] > atr_ema.iloc[i] and atr.iloc[i-1] <= atr_ema.iloc[i-1]:
            sig[i] = 1
        elif atr.iloc[i] < atr_ema.iloc[i] and atr.iloc[i-1] >= atr_ema.iloc[i-1]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_True_Range_Crossover():
    return {
        'period': (10, 20, 2),
    }

def gen_Vortex_Indicator(df, **p):
    period = p.get('period', 14)

    vm_plus = pd.Series(np.abs(df['high'] - df['low'].shift(1))).rolling(period).sum()
    vm_minus = pd.Series(np.abs(df['low'] - df['high'].shift(1))).rolling(period).sum()

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)

    true_range = tr.rolling(period).sum()

    vi_plus = vm_plus / true_range
    vi_minus = vm_minus / true_range

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if vi_plus.iloc[i] > vi_minus.iloc[i] and vi_plus.iloc[i-1] <= vi_minus.iloc[i-1]:
            sig[i] = 1
        elif vi_plus.iloc[i] < vi_minus.iloc[i] and vi_plus.iloc[i-1] >= vi_minus.iloc[i-1]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Vortex_Indicator():
    return {
        'period': (10, 25, 2),
    }

def gen_Q_Stick(df, **p):
    period = p.get('period', 10)

    qstick = (df['close'] - df['open']).rolling(period).mean()
    qstick_ema = qstick.ewm(span=period).mean()

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if qstick.iloc[i] > qstick_ema.iloc[i] and qstick.iloc[i-1] <= qstick_ema.iloc[i-1]:
            sig[i] = 1
        elif qstick.iloc[i] < qstick_ema.iloc[i] and qstick.iloc[i-1] >= qstick_ema.iloc[i-1]:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Q_Stick():
    return {
        'period': (5, 15, 1),
    }

def gen_Ease_of_Movement(df, **p):
    period = p.get('period', 14)

    distance = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
    boxheight = df['high'] - df['low']

    eom = distance / boxheight
    eom_sma = eom.rolling(period).mean()

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if eom_sma.iloc[i] > 0 and eom_sma.iloc[i-1] <= 0:
            sig[i] = 1
        elif eom_sma.iloc[i] < 0 and eom_sma.iloc[i-1] >= 0:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Ease_of_Movement():
    return {
        'period': (10, 20, 2),
    }

def gen_Chande_Momentum(df, **p):
    period = p.get('period', 10)

    up = (df['close'] - df['close'].shift(1)).clip(lower=0).rolling(period).sum()
    down = (df['close'].shift(1) - df['close']).clip(lower=0).rolling(period).sum()

    cmo = 100 * (up - down) / (up + down + 1e-6)

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if cmo.iloc[i] > 0 and cmo.iloc[i-1] <= 0:
            sig[i] = 1
        elif cmo.iloc[i] < 0 and cmo.iloc[i-1] >= 0:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Chande_Momentum():
    return {
        'period': (5, 15, 1),
    }

def gen_Ultimate_Oscillator(df, **p):
    p1 = p.get('p1', 7)
    p2 = p.get('p2', 14)
    p3 = p.get('p3', 28)

    true_range = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)

    bp = df['close'] - np.minimum(df['low'], df['close'].shift(1))

    avg1 = bp.rolling(p1).sum() / true_range.rolling(p1).sum()
    avg2 = bp.rolling(p2).sum() / true_range.rolling(p2).sum()
    avg3 = bp.rolling(p3).sum() / true_range.rolling(p3).sum()

    uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7

    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if uo.iloc[i] > 50 and uo.iloc[i-1] <= 50:
            sig[i] = 1
        elif uo.iloc[i] < 50 and uo.iloc[i-1] >= 50:
            sig[i] = -1

    df['signal'] = sig
    return df

def space_Ultimate_Oscillator():
    return {
        'p1': (5, 10, 1),
        'p2': (10, 20, 2),
        'p3': (20, 35, 3),
    }

def gen_Elder_Ray_Index(df, **p):
    period = p.get('period', 13)
    ema = df['close'].ewm(span=period).mean()
    bull_power = df['high'] - ema
    bear_power = ema - df['low']
    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if bull_power.iloc[i] > 0 and bull_power.iloc[i-1] <= 0:
            sig[i] = 1
        elif bear_power.iloc[i] > 0 and bear_power.iloc[i-1] <= 0:
            sig[i] = -1
    df['signal'] = sig
    return df

def space_Elder_Ray_Index():
    return {'period': (10, 20, 2)}

def gen_Mass_Index(df, **p):
    period1 = p.get('period1', 9)
    period2 = p.get('period2', 25)
    ema1 = (df['high'] - df['low']).ewm(span=period1).mean()
    ema2 = ema1.ewm(span=period1).mean()
    ratio = ema1 / ema2
    mass = ratio.rolling(period2).sum()
    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if mass.iloc[i] > p.get('upper', 27) and mass.iloc[i-1] <= p.get('upper', 27):
            sig[i] = -1
    df['signal'] = sig
    return df

def space_Mass_Index():
    return {'period1': (7, 12, 1), 'period2': (20, 30, 2), 'upper': (25, 30, 1)}

def gen_SMA_EMA_Ribbon(df, **p):
    periods = [5, 10, 20, 50, 100]
    sma_vals = np.zeros((len(df), len(periods)))
    for i, p_val in enumerate(periods):
        sma_vals[:, i] = df['close'].rolling(p_val).mean().fillna(0).values
    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['close'].iloc[i] > np.max(sma_vals[i]):
            sig[i] = 1
        elif df['close'].iloc[i] < np.min(sma_vals[i]):
            sig[i] = -1
    df['signal'] = sig
    return df

def space_SMA_EMA_Ribbon():
    return {}

def gen_Heiken_Ashi_Reversal(df, **p):
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = np.zeros(len(df))
    ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
    ha_high = np.zeros(len(df))
    ha_low = np.zeros(len(df))
    for i in range(len(df)):
        ha_high[i] = max(df['high'].iloc[i], ha_open[i], ha_close.iloc[i])
        ha_low[i] = min(df['low'].iloc[i], ha_open[i], ha_close.iloc[i])
    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if ha_close.iloc[i] > ha_open[i] and ha_close.iloc[i-1] <= ha_open[i-1]:
            sig[i] = 1
        elif ha_close.iloc[i] < ha_open[i] and ha_close.iloc[i-1] >= ha_open[i-1]:
            sig[i] = -1
    df['signal'] = sig
    return df

def space_Heiken_Ashi_Reversal():
    return {}

def gen_Momentum_EMA(df, **p):
    period = p.get('period', 12)
    ema_period = p.get('ema_period', 9)
    momentum = df['close'] - df['close'].shift(period)
    momentum_ema = momentum.ewm(span=ema_period).mean()
    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if momentum_ema.iloc[i] > 0 and momentum_ema.iloc[i-1] <= 0:
            sig[i] = 1
        elif momentum_ema.iloc[i] < 0 and momentum_ema.iloc[i-1] >= 0:
            sig[i] = -1
    df['signal'] = sig
    return df

def space_Momentum_EMA():
    return {'period': (8, 20, 2), 'ema_period': (5, 15, 2)}

def gen_Price_Action_SAR(df, **p):
    af = p.get('af', 0.02)
    af_max = p.get('af_max', 0.2)
    sar = np.zeros(len(df))
    sar[0] = df['low'].iloc[0]
    trend = 1
    hp = df['high'].iloc[0]
    lp = df['low'].iloc[0]
    af_curr = af
    for i in range(1, len(df)):
        sar[i] = sar[i-1] + af_curr * (hp - sar[i-1]) if trend == 1 else sar[i-1] - af_curr * (sar[i-1] - lp)
        if trend == 1:
            if df['low'].iloc[i] < sar[i]:
                trend = -1
                sar[i] = hp
                lp = df['low'].iloc[i]
                af_curr = af
            else:
                if df['high'].iloc[i] > hp:
                    hp = df['high'].iloc[i]
                    af_curr = min(af_curr + af, af_max)
        else:
            if df['high'].iloc[i] > sar[i]:
                trend = 1
                sar[i] = lp
                hp = df['high'].iloc[i]
                af_curr = af
            else:
                if df['low'].iloc[i] < lp:
                    lp = df['low'].iloc[i]
                    af_curr = min(af_curr + af, af_max)
    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['close'].iloc[i] > sar[i] and df['close'].iloc[i-1] <= sar[i-1]:
            sig[i] = 1
        elif df['close'].iloc[i] < sar[i] and df['close'].iloc[i-1] >= sar[i-1]:
            sig[i] = -1
    df['signal'] = sig
    return df

def space_Price_Action_SAR():
    return {'af': (0.01, 0.05, 0.01), 'af_max': (0.15, 0.25, 0.05)}

def gen_Volume_Weighted_Price(df, **p):
    period = p.get('period', 20)
    vwap = (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['close'].iloc[i] > vwap.iloc[i] and df['close'].iloc[i-1] <= vwap.iloc[i-1]:
            sig[i] = 1
        elif df['close'].iloc[i] < vwap.iloc[i] and df['close'].iloc[i-1] >= vwap.iloc[i-1]:
            sig[i] = -1
    df['signal'] = sig
    return df

def space_Volume_Weighted_Price():
    return {'period': (10, 30, 3)}

def gen_Swing_High_Low(df, **p):
    lookback = p.get('lookback', 3)
    sig = np.zeros(len(df))
    for i in range(lookback, len(df) - lookback):
        if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, lookback+1)) and all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, lookback+1)):
            sig[i] = -1
        elif all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, lookback+1)) and all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, lookback+1)):
            sig[i] = 1
    df['signal'] = sig
    return df

def space_Swing_High_Low():
    return {'lookback': (2, 5, 1)}

def gen_Bollinger_Squeeze(df, **p):
    period = p.get('period', 20)
    std_mult = p.get('std_mult', 2.0)
    squeeze_period = p.get('squeeze_period', 20)
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    bb_width = 2 * std_mult * std
    avg_width = bb_width.rolling(squeeze_period).mean()
    is_squeeze = bb_width < avg_width * 0.75
    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if is_squeeze.iloc[i] and not is_squeeze.iloc[i-1]:
            if df['close'].iloc[i] > sma.iloc[i]:
                sig[i] = 1
            else:
                sig[i] = -1
    df['signal'] = sig
    return df

def space_Bollinger_Squeeze():
    return {'period': (15, 30, 3), 'std_mult': (1.5, 3.0, 0.3), 'squeeze_period': (15, 30, 3)}

def gen_Gap_Detection(df, **p):
    gap_percent = p.get('gap_percent', 0.02)
    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        gap_up = df['open'].iloc[i] > df['close'].iloc[i-1] * (1 + gap_percent)
        gap_down = df['open'].iloc[i] < df['close'].iloc[i-1] * (1 - gap_percent)
        if gap_up:
            sig[i] = -1
        elif gap_down:
            sig[i] = 1
    df['signal'] = sig
    return df

def space_Gap_Detection():
    return {'gap_percent': (0.01, 0.05, 0.01)}

def gen_Relative_Strength(df, **p):
    period = p.get('period', 20)
    rs = df['close'] / df['close'].rolling(period).mean()
    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if rs.iloc[i] > 1.0 and rs.iloc[i-1] <= 1.0:
            sig[i] = 1
        elif rs.iloc[i] < 1.0 and rs.iloc[i-1] >= 1.0:
            sig[i] = -1
    df['signal'] = sig
    return df

def space_Relative_Strength():
    return {'period': (10, 30, 3)}

def gen_High_Low_Range(df, **p):
    period = p.get('period', 20)
    range_val = df['high'] - df['low']
    avg_range = range_val.rolling(period).mean()
    sig = np.zeros(len(df))
    for i in range(1, len(df)):
        if range_val.iloc[i] > avg_range.iloc[i] * p.get('mult_high', 1.5):
            if df['close'].iloc[i] > df['open'].iloc[i]:
                sig[i] = 1
            else:
                sig[i] = -1
    df['signal'] = sig
    return df

def space_High_Low_Range():
    return {'period': (10, 25, 3), 'mult_high': (1.2, 2.0, 0.2)}

STRATEGY_EXPORT = {
    'Supertrend_V5_Basic': {
        'gen': gen_Supertrend_V5_Basic,
        'space': space_Supertrend_V5_Basic,
    },
    'EMA_Crossover_3_Period': {
        'gen': gen_EMA_Crossover_3_Period,
        'space': space_EMA_Crossover_3_Period,
    },
    'RSI_Oversold_Long': {
        'gen': gen_RSI_Oversold_Long,
        'space': space_RSI_Oversold_Long,
    },
    'MACD_Crossover': {
        'gen': gen_MACD_Crossover,
        'space': space_MACD_Crossover,
    },
    'Bollinger_Bands_Mean_Reversion': {
        'gen': gen_Bollinger_Bands_Mean_Reversion,
        'space': space_Bollinger_Bands_Mean_Reversion,
    },
    'Stochastic_Oversold': {
        'gen': gen_Stochastic_Oversold,
        'space': space_Stochastic_Oversold,
    },
    'ADX_Trend_Filter': {
        'gen': gen_ADX_Trend_Filter,
        'space': space_ADX_Trend_Filter,
    },
    'VWAP_Breakout': {
        'gen': gen_VWAP_Breakout,
        'space': space_VWAP_Breakout,
    },
    'Volume_Profile_Support': {
        'gen': gen_Volume_Profile_Support,
        'space': space_Volume_Profile_Support,
    },
    'MACD_BBands_RSI': {
        'gen': gen_MACD_BBands_RSI,
        'space': space_MACD_BBands_RSI,
    },
    'DMI_Winner': {
        'gen': gen_DMI_Winner,
        'space': space_DMI_Winner,
    },
    'Williams_Vix_Fix': {
        'gen': gen_Williams_Vix_Fix,
        'space': space_Williams_Vix_Fix,
    },
    'KDJ_Stochastic': {
        'gen': gen_KDJ_Stochastic,
        'space': space_KDJ_Stochastic,
    },
    'ROC_Rate_Change': {
        'gen': gen_ROC_Rate_Change,
        'space': space_ROC_Rate_Change,
    },
    'TRIX_Momentum': {
        'gen': gen_TRIX_Momentum,
        'space': space_TRIX_Momentum,
    },
    'Ichimoku_Cloud': {
        'gen': gen_Ichimoku_Cloud,
        'space': space_Ichimoku_Cloud,
    },
    'Donchian_Breakout': {
        'gen': gen_Donchian_Breakout,
        'space': space_Donchian_Breakout,
    },
    'Fisher_Transform': {
        'gen': gen_Fisher_Transform,
        'space': space_Fisher_Transform,
    },
    'Pivot_Point_SR': {
        'gen': gen_Pivot_Point_SR,
        'space': space_Pivot_Point_SR,
    },
    'Keltner_Channel': {
        'gen': gen_Keltner_Channel,
        'space': space_Keltner_Channel,
    },
    'Linear_Regression_Channel': {
        'gen': gen_Linear_Regression_Channel,
        'space': space_Linear_Regression_Channel,
    },
    'Envelope_Trading': {
        'gen': gen_Envelope_Trading,
        'space': space_Envelope_Trading,
    },
    'ZScore_Crossover': {
        'gen': gen_ZScore_Crossover,
        'space': space_ZScore_Crossover,
    },
    'OBV_Confirmation': {
        'gen': gen_OBV_Confirmation,
        'space': space_OBV_Confirmation,
    },
    'Money_Flow_Index': {
        'gen': gen_Money_Flow_Index,
        'space': space_Money_Flow_Index,
    },
    'Accumulation_Distribution': {
        'gen': gen_Accumulation_Distribution,
        'space': space_Accumulation_Distribution,
    },
    'Price_Volume_Trend': {
        'gen': gen_Price_Volume_Trend,
        'space': space_Price_Volume_Trend,
    },
    'Awesome_Oscillator': {
        'gen': gen_Awesome_Oscillator,
        'space': space_Awesome_Oscillator,
    },
    'True_Range_Crossover': {
        'gen': gen_True_Range_Crossover,
        'space': space_True_Range_Crossover,
    },
    'Vortex_Indicator': {
        'gen': gen_Vortex_Indicator,
        'space': space_Vortex_Indicator,
    },
    'Q_Stick': {
        'gen': gen_Q_Stick,
        'space': space_Q_Stick,
    },
    'Ease_of_Movement': {
        'gen': gen_Ease_of_Movement,
        'space': space_Ease_of_Movement,
    },
    'Chande_Momentum': {
        'gen': gen_Chande_Momentum,
        'space': space_Chande_Momentum,
    },
    'Ultimate_Oscillator': {
        'gen': gen_Ultimate_Oscillator,
        'space': space_Ultimate_Oscillator,
    },
    'Elder_Ray_Index': {
        'gen': gen_Elder_Ray_Index,
        'space': space_Elder_Ray_Index,
    },
    'Mass_Index': {
        'gen': gen_Mass_Index,
        'space': space_Mass_Index,
    },
    'SMA_EMA_Ribbon': {
        'gen': gen_SMA_EMA_Ribbon,
        'space': space_SMA_EMA_Ribbon,
    },
    'Heiken_Ashi_Reversal': {
        'gen': gen_Heiken_Ashi_Reversal,
        'space': space_Heiken_Ashi_Reversal,
    },
    'Momentum_EMA': {
        'gen': gen_Momentum_EMA,
        'space': space_Momentum_EMA,
    },
    'Price_Action_SAR': {
        'gen': gen_Price_Action_SAR,
        'space': space_Price_Action_SAR,
    },
    'Volume_Weighted_Price': {
        'gen': gen_Volume_Weighted_Price,
        'space': space_Volume_Weighted_Price,
    },
    'Swing_High_Low': {
        'gen': gen_Swing_High_Low,
        'space': space_Swing_High_Low,
    },
    'Bollinger_Squeeze': {
        'gen': gen_Bollinger_Squeeze,
        'space': space_Bollinger_Squeeze,
    },
    'Gap_Detection': {
        'gen': gen_Gap_Detection,
        'space': space_Gap_Detection,
    },
    'Relative_Strength': {
        'gen': gen_Relative_Strength,
        'space': space_Relative_Strength,
    },
    'High_Low_Range': {
        'gen': gen_High_Low_Range,
        'space': space_High_Low_Range,
    },
}
