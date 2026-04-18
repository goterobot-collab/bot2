#!/usr/bin/env python3
"""
STRATEGIES ROUND 2: 52 NEW strategy types with Optuna search spaces.
All vectorized (pandas/numpy, no Python for-loops in signal generation).
Each generator: DataFrame → Series of signals (1=buy, -1=sell, 0=hold).

Usage:
    from strategies_round2 import STRATEGY_TYPES_R2
"""

import pandas as pd
import numpy as np


# ════════════════════════════════════════════════════════════════
# INDICATOR HELPERS (standalone — no external dependency)
# ════════════════════════════════════════════════════════════════

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def sma(s, p):
    return s.rolling(p).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0).ewm(span=p, adjust=False).mean()
    l = (-d.where(d < 0, 0.0)).ewm(span=p, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def bb(c, p=20, std=2):
    m = c.rolling(p).mean()
    s = c.rolling(p).std()
    return m, m + std * s, m - std * s

def macd(c, f=12, s=26, sig=9):
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

def wma(s, p):
    w = np.arange(1, p + 1, dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def true_range(h, l, c):
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)


# ════════════════════════════════════════════════════════════════
# REGISTRY
# ════════════════════════════════════════════════════════════════

STRATEGY_TYPES_R2 = {}

def reg(name, gen_func, space_func):
    STRATEGY_TYPES_R2[name] = {'gen': gen_func, 'space': space_func}


# ════════════════════════════════════════════════════════════════
# 1-9  TREND FOLLOWING
# ════════════════════════════════════════════════════════════════

# 1. SuperTrend — vectorized via numpy
def gen_supertrend(df, period, multiplier):
    hl2 = (df['high'] + df['low']) / 2
    atr_val = atr(df['high'], df['low'], df['close'], period)
    up = (hl2 - multiplier * atr_val).values
    dn = (hl2 + multiplier * atr_val).values
    close = df['close'].values
    n = len(close)
    final_up = np.copy(up)
    final_dn = np.copy(dn)
    direction = np.ones(n)
    for i in range(1, n):
        # vectorized-style update with numpy arrays (no pandas overhead)
        if up[i] > final_up[i - 1] or close[i - 1] < final_up[i - 1]:
            pass
        else:
            final_up[i] = final_up[i - 1]
        if dn[i] < final_dn[i - 1] or close[i - 1] > final_dn[i - 1]:
            pass
        else:
            final_dn[i] = final_dn[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < final_up[i] else 1
        else:
            direction[i] = 1 if close[i] > final_dn[i] else -1
    d = pd.Series(direction, index=df.index)
    sig = pd.Series(0, index=df.index)
    sig[(d == 1) & (d.shift() == -1)] = 1
    sig[(d == -1) & (d.shift() == 1)] = -1
    return sig

reg('SuperTrend', gen_supertrend,
    lambda t: {'period': t.suggest_int('period', 5, 30),
               'multiplier': t.suggest_float('multiplier', 1.0, 5.0, step=0.5)})


# 2. Parabolic_SAR — simplified vectorized via numpy arrays
def gen_psar(df, af_start, af_max):
    af_step = af_start
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(c)
    psar = np.zeros(n)
    bull = np.ones(n, dtype=bool)
    af = np.full(n, af_start)
    ep = np.zeros(n)
    psar[0] = l[0]
    ep[0] = h[0]
    for i in range(1, n):
        if bull[i - 1]:
            psar[i] = psar[i - 1] + af[i - 1] * (ep[i - 1] - psar[i - 1])
            psar[i] = min(psar[i], l[i - 1], l[max(0, i - 2)])
            if l[i] < psar[i]:
                bull[i] = False
                psar[i] = ep[i - 1]
                af[i] = af_start
                ep[i] = l[i]
            else:
                bull[i] = True
                ep[i] = max(ep[i - 1], h[i])
                af[i] = min(af[i - 1] + af_step, af_max) if h[i] > ep[i - 1] else af[i - 1]
        else:
            psar[i] = psar[i - 1] + af[i - 1] * (ep[i - 1] - psar[i - 1])
            psar[i] = max(psar[i], h[i - 1], h[max(0, i - 2)])
            if h[i] > psar[i]:
                bull[i] = True
                psar[i] = ep[i - 1]
                af[i] = af_start
                ep[i] = h[i]
            else:
                bull[i] = False
                ep[i] = min(ep[i - 1], l[i])
                af[i] = min(af[i - 1] + af_step, af_max) if l[i] < ep[i - 1] else af[i - 1]
    bs = pd.Series(bull, index=df.index)
    sig = pd.Series(0, index=df.index)
    sig[(bs) & (~bs.shift(fill_value=True))] = 1
    sig[(~bs) & (bs.shift(fill_value=False))] = -1
    return sig

reg('Parabolic_SAR', gen_psar,
    lambda t: {'af_start': t.suggest_float('af_start', 0.01, 0.05, step=0.005),
               'af_max': t.suggest_float('af_max', 0.1, 0.3, step=0.02)})


# 3. Aroon — Aroon Up/Down crossover
def gen_aroon(df, period):
    aup = df['high'].rolling(period + 1).apply(lambda x: x.argmax() / period * 100, raw=True)
    adn = df['low'].rolling(period + 1).apply(lambda x: x.argmin() / period * 100, raw=True)
    sig = pd.Series(0, index=df.index)
    sig[(aup > adn) & (aup.shift() <= adn.shift())] = 1
    sig[(aup < adn) & (aup.shift() >= adn.shift())] = -1
    return sig

reg('Aroon', gen_aroon,
    lambda t: {'period': t.suggest_int('period', 10, 50)})


# 4. TRIX — Triple smoothed EMA rate of change
def gen_trix(df, period, signal_period):
    e1 = ema(df['close'], period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    trix_line = e3.pct_change() * 100
    trix_sig = ema(trix_line, signal_period)
    sig = pd.Series(0, index=df.index)
    sig[(trix_line > trix_sig) & (trix_line.shift() <= trix_sig.shift())] = 1
    sig[(trix_line < trix_sig) & (trix_line.shift() >= trix_sig.shift())] = -1
    return sig

reg('TRIX', gen_trix,
    lambda t: {'period': t.suggest_int('period', 8, 30),
               'signal_period': t.suggest_int('signal_period', 5, 15)})


# 5. DEMA_Cross — Double EMA crossover
def gen_dema_cross(df, fast, slow):
    ef = ema(df['close'], fast)
    dema_f = 2 * ef - ema(ef, fast)
    es = ema(df['close'], slow)
    dema_s = 2 * es - ema(es, slow)
    sig = pd.Series(0, index=df.index)
    sig[(dema_f > dema_s) & (dema_f.shift() <= dema_s.shift())] = 1
    sig[(dema_f < dema_s) & (dema_f.shift() >= dema_s.shift())] = -1
    return sig

reg('DEMA_Cross', gen_dema_cross,
    lambda t: {'fast': t.suggest_int('fast', 3, 25),
               'slow': t.suggest_int('slow', 20, 100)})


# 6. TEMA_Cross — Triple EMA crossover
def gen_tema_cross(df, fast, slow):
    e1f = ema(df['close'], fast)
    e2f = ema(e1f, fast)
    e3f = ema(e2f, fast)
    tema_f = 3 * e1f - 3 * e2f + e3f
    e1s = ema(df['close'], slow)
    e2s = ema(e1s, slow)
    e3s = ema(e2s, slow)
    tema_s = 3 * e1s - 3 * e2s + e3s
    sig = pd.Series(0, index=df.index)
    sig[(tema_f > tema_s) & (tema_f.shift() <= tema_s.shift())] = 1
    sig[(tema_f < tema_s) & (tema_f.shift() >= tema_s.shift())] = -1
    return sig

reg('TEMA_Cross', gen_tema_cross,
    lambda t: {'fast': t.suggest_int('fast', 3, 25),
               'slow': t.suggest_int('slow', 20, 100)})


# 7. ZLEMA — Zero-lag EMA
def gen_zlema(df, fast, slow):
    lag_f = int((fast - 1) / 2)
    lag_s = int((slow - 1) / 2)
    zf = ema(2 * df['close'] - df['close'].shift(lag_f), fast)
    zs = ema(2 * df['close'] - df['close'].shift(lag_s), slow)
    sig = pd.Series(0, index=df.index)
    sig[(zf > zs) & (zf.shift() <= zs.shift())] = 1
    sig[(zf < zs) & (zf.shift() >= zs.shift())] = -1
    return sig

reg('ZLEMA', gen_zlema,
    lambda t: {'fast': t.suggest_int('fast', 3, 25),
               'slow': t.suggest_int('slow', 20, 100)})


# 8. KAMA — Kaufman Adaptive Moving Average (vectorized via cumulative)
def gen_kama(df, period, fast_sc, slow_sc):
    c = df['close']
    er_num = (c - c.shift(period)).abs()
    er_den = c.diff().abs().rolling(period).sum()
    er = er_num / (er_den + 1e-10)
    fast_c = 2 / (fast_sc + 1)
    slow_c = 2 / (slow_sc + 1)
    sc = (er * (fast_c - slow_c) + slow_c) ** 2
    # Must iterate for KAMA itself (recursive formula), use numpy for speed
    vals = c.values.copy()
    sc_v = sc.values
    kama_arr = np.full(len(vals), np.nan)
    start = period
    kama_arr[start] = vals[start]
    for i in range(start + 1, len(vals)):
        if np.isnan(sc_v[i]) or np.isnan(kama_arr[i - 1]):
            kama_arr[i] = vals[i]
        else:
            kama_arr[i] = kama_arr[i - 1] + sc_v[i] * (vals[i] - kama_arr[i - 1])
    k = pd.Series(kama_arr, index=df.index)
    sig = pd.Series(0, index=df.index)
    sig[(c > k) & (c.shift() <= k.shift())] = 1
    sig[(c < k) & (c.shift() >= k.shift())] = -1
    return sig

reg('KAMA', gen_kama,
    lambda t: {'period': t.suggest_int('period', 5, 30),
               'fast_sc': t.suggest_int('fast_sc', 2, 5),
               'slow_sc': t.suggest_int('slow_sc', 20, 40)})


# 9. VIDYA — Variable Index Dynamic Average
def gen_vidya(df, period, cmo_period):
    c = df['close']
    delta = c.diff()
    su = delta.where(delta > 0, 0.0).rolling(cmo_period).sum()
    sd = (-delta.where(delta < 0, 0.0)).rolling(cmo_period).sum()
    cmo_val = ((su - sd) / (su + sd + 1e-10)).abs()
    sc = 2 / (period + 1)
    vals = c.values.copy()
    cmo_v = cmo_val.values
    vid = np.full(len(vals), np.nan)
    start = max(period, cmo_period)
    vid[start] = vals[start]
    for i in range(start + 1, len(vals)):
        if np.isnan(vid[i - 1]) or np.isnan(cmo_v[i]):
            vid[i] = vals[i]
        else:
            vid[i] = vid[i - 1] + sc * cmo_v[i] * (vals[i] - vid[i - 1])
    v = pd.Series(vid, index=df.index)
    sig = pd.Series(0, index=df.index)
    sig[(c > v) & (c.shift() <= v.shift())] = 1
    sig[(c < v) & (c.shift() >= v.shift())] = -1
    return sig

reg('VIDYA', gen_vidya,
    lambda t: {'period': t.suggest_int('period', 8, 30),
               'cmo_period': t.suggest_int('cmo_period', 5, 20)})


# ════════════════════════════════════════════════════════════════
# 10-17  MOMENTUM
# ════════════════════════════════════════════════════════════════

# 10. ROC — Rate of Change
def gen_roc(df, period, buy_thresh, sell_thresh):
    r = (df['close'] - df['close'].shift(period)) / (df['close'].shift(period) + 1e-10) * 100
    sig = pd.Series(0, index=df.index)
    sig[(r > buy_thresh) & (r.shift() <= buy_thresh)] = 1
    sig[(r < sell_thresh) & (r.shift() >= sell_thresh)] = -1
    return sig

reg('ROC', gen_roc,
    lambda t: {'period': t.suggest_int('period', 5, 30),
               'buy_thresh': t.suggest_float('buy_thresh', 0.5, 5.0, step=0.5),
               'sell_thresh': t.suggest_float('sell_thresh', -5.0, -0.5, step=0.5)})


# 11. CMO — Chande Momentum Oscillator
def gen_cmo(df, period, buy, sell):
    delta = df['close'].diff()
    su = delta.where(delta > 0, 0.0).rolling(period).sum()
    sd = (-delta.where(delta < 0, 0.0)).rolling(period).sum()
    cmo_val = 100 * (su - sd) / (su + sd + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(cmo_val < buy) & (cmo_val > cmo_val.shift())] = 1
    sig[(cmo_val > sell) & (cmo_val < cmo_val.shift())] = -1
    return sig

reg('CMO', gen_cmo,
    lambda t: {'period': t.suggest_int('period', 5, 30),
               'buy': t.suggest_int('buy', -70, -20),
               'sell': t.suggest_int('sell', 20, 70)})


# 12. TSI — True Strength Index
def gen_tsi(df, long_period, short_period, signal_period):
    diff = df['close'].diff()
    ds = ema(ema(diff, long_period), short_period)
    ds_abs = ema(ema(diff.abs(), long_period), short_period)
    tsi_val = 100 * ds / (ds_abs + 1e-10)
    tsi_sig = ema(tsi_val, signal_period)
    sig = pd.Series(0, index=df.index)
    sig[(tsi_val > tsi_sig) & (tsi_val.shift() <= tsi_sig.shift())] = 1
    sig[(tsi_val < tsi_sig) & (tsi_val.shift() >= tsi_sig.shift())] = -1
    return sig

reg('TSI', gen_tsi,
    lambda t: {'long_period': t.suggest_int('long_period', 15, 35),
               'short_period': t.suggest_int('short_period', 8, 18),
               'signal_period': t.suggest_int('signal_period', 5, 15)})


# 13. Ultimate_Osc — Ultimate Oscillator
def gen_ultimate_osc(df, p1, p2, p3, buy, sell):
    prev_c = df['close'].shift()
    bp = df['close'] - pd.concat([df['low'], prev_c], axis=1).min(axis=1)
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - prev_c).abs(),
                    (df['low'] - prev_c).abs()], axis=1).max(axis=1)
    avg1 = bp.rolling(p1).sum() / (tr.rolling(p1).sum() + 1e-10)
    avg2 = bp.rolling(p2).sum() / (tr.rolling(p2).sum() + 1e-10)
    avg3 = bp.rolling(p3).sum() / (tr.rolling(p3).sum() + 1e-10)
    uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
    sig = pd.Series(0, index=df.index)
    sig[(uo < buy) & (uo > uo.shift())] = 1
    sig[(uo > sell) & (uo < uo.shift())] = -1
    return sig

reg('Ultimate_Osc', gen_ultimate_osc,
    lambda t: {'p1': t.suggest_int('p1', 4, 10),
               'p2': t.suggest_int('p2', 10, 20),
               'p3': t.suggest_int('p3', 20, 40),
               'buy': t.suggest_int('buy', 20, 40),
               'sell': t.suggest_int('sell', 60, 80)})


# 14. KST — Know Sure Thing
def gen_kst(df, r1, r2, r3, r4, signal_p):
    roc1 = df['close'].pct_change(r1) * 100
    roc2 = df['close'].pct_change(r2) * 100
    roc3 = df['close'].pct_change(r3) * 100
    roc4 = df['close'].pct_change(r4) * 100
    kst_line = sma(roc1, 10) + 2 * sma(roc2, 10) + 3 * sma(roc3, 10) + 4 * sma(roc4, 15)
    kst_sig = sma(kst_line, signal_p)
    sig = pd.Series(0, index=df.index)
    sig[(kst_line > kst_sig) & (kst_line.shift() <= kst_sig.shift())] = 1
    sig[(kst_line < kst_sig) & (kst_line.shift() >= kst_sig.shift())] = -1
    return sig

reg('KST', gen_kst,
    lambda t: {'r1': t.suggest_int('r1', 5, 15),
               'r2': t.suggest_int('r2', 10, 20),
               'r3': t.suggest_int('r3', 15, 30),
               'r4': t.suggest_int('r4', 20, 45),
               'signal_p': t.suggest_int('signal_p', 5, 15)})


# 15. Coppock — Coppock Curve
def gen_coppock(df, roc1, roc2, wma_period):
    r1 = df['close'].pct_change(roc1) * 100
    r2 = df['close'].pct_change(roc2) * 100
    cop = wma(r1 + r2, wma_period)
    sig = pd.Series(0, index=df.index)
    sig[(cop > 0) & (cop.shift() <= 0)] = 1
    sig[(cop < 0) & (cop.shift() >= 0)] = -1
    return sig

reg('Coppock', gen_coppock,
    lambda t: {'roc1': t.suggest_int('roc1', 8, 18),
               'roc2': t.suggest_int('roc2', 8, 15),
               'wma_period': t.suggest_int('wma_period', 5, 15)})


# 16. PPO — Percentage Price Oscillator
def gen_ppo(df, fast, slow, signal_p):
    ef = ema(df['close'], fast)
    es = ema(df['close'], slow)
    ppo_line = (ef - es) / (es + 1e-10) * 100
    ppo_sig = ema(ppo_line, signal_p)
    sig = pd.Series(0, index=df.index)
    sig[(ppo_line > ppo_sig) & (ppo_line.shift() <= ppo_sig.shift())] = 1
    sig[(ppo_line < ppo_sig) & (ppo_line.shift() >= ppo_sig.shift())] = -1
    return sig

reg('PPO', gen_ppo,
    lambda t: {'fast': t.suggest_int('fast', 5, 18),
               'slow': t.suggest_int('slow', 18, 40),
               'signal_p': t.suggest_int('signal_p', 5, 15)})


# 17. DPO — Detrended Price Oscillator
def gen_dpo(df, period, thresh):
    shift = int(period / 2) + 1
    dpo_val = df['close'] - sma(df['close'], period).shift(shift)
    sig = pd.Series(0, index=df.index)
    sig[(dpo_val > thresh) & (dpo_val.shift() <= thresh)] = 1
    sig[(dpo_val < -thresh) & (dpo_val.shift() >= -thresh)] = -1
    return sig

reg('DPO', gen_dpo,
    lambda t: {'period': t.suggest_int('period', 10, 40),
               'thresh': t.suggest_float('thresh', 0.0, 0.5, step=0.05)})


# ════════════════════════════════════════════════════════════════
# 18-21  VOLATILITY
# ════════════════════════════════════════════════════════════════

# 18. ATR_Channel — ATR channel breakout
def gen_atr_channel(df, period, mult):
    mid = ema(df['close'], period)
    atr_val = atr(df['high'], df['low'], df['close'], period)
    upper = mid + mult * atr_val
    lower = mid - mult * atr_val
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] > upper) & (df['close'].shift() <= upper.shift())] = 1
    sig[(df['close'] < lower) & (df['close'].shift() >= lower.shift())] = -1
    return sig

reg('ATR_Channel', gen_atr_channel,
    lambda t: {'period': t.suggest_int('period', 8, 40),
               'mult': t.suggest_float('mult', 1.0, 4.0, step=0.25)})


# 19. Chaikin_Vol — Chaikin Volatility expansion/contraction
def gen_chaikin_vol(df, ema_period, roc_period, thresh):
    hl = df['high'] - df['low']
    ema_hl = ema(hl, ema_period)
    chvol = (ema_hl - ema_hl.shift(roc_period)) / (ema_hl.shift(roc_period) + 1e-10) * 100
    sig = pd.Series(0, index=df.index)
    # High vol expansion + price rising = buy; contraction from high + price falling = sell
    price_up = df['close'] > df['close'].shift()
    price_dn = df['close'] < df['close'].shift()
    sig[(chvol > thresh) & price_up] = 1
    sig[(chvol < -thresh) & price_dn] = -1
    return sig

reg('Chaikin_Vol', gen_chaikin_vol,
    lambda t: {'ema_period': t.suggest_int('ema_period', 5, 20),
               'roc_period': t.suggest_int('roc_period', 5, 20),
               'thresh': t.suggest_float('thresh', 10.0, 50.0, step=5.0)})


# 20. Historical_Vol — Historical vol breakout
def gen_histvol(df, period, lookback, vol_mult):
    ret = df['close'].pct_change()
    hvol = ret.rolling(period).std() * np.sqrt(252) * 100
    hvol_avg = hvol.rolling(lookback).mean()
    sig = pd.Series(0, index=df.index)
    # Vol spike + bullish momentum = buy; vol spike + bearish = sell
    vol_spike = hvol > vol_mult * hvol_avg
    bull = df['close'] > ema(df['close'], period)
    sig[vol_spike & bull] = 1
    sig[vol_spike & ~bull] = -1
    return sig

reg('Historical_Vol', gen_histvol,
    lambda t: {'period': t.suggest_int('period', 10, 30),
               'lookback': t.suggest_int('lookback', 20, 60),
               'vol_mult': t.suggest_float('vol_mult', 1.2, 2.5, step=0.1)})


# 21. Ulcer_Index — Ulcer Index reversal
def gen_ulcer(df, period, thresh):
    roll_max = df['close'].rolling(period).max()
    pct_dd = (df['close'] - roll_max) / (roll_max + 1e-10) * 100
    ui = np.sqrt((pct_dd ** 2).rolling(period).mean())
    sig = pd.Series(0, index=df.index)
    # High ulcer index (pain) + turn up = buy signal (reversal)
    sig[(ui > thresh) & (df['close'] > df['close'].shift())] = 1
    sig[(ui < thresh * 0.3) & (df['close'] < df['close'].shift())] = -1
    return sig

reg('Ulcer_Index', gen_ulcer,
    lambda t: {'period': t.suggest_int('period', 10, 30),
               'thresh': t.suggest_float('thresh', 3.0, 15.0, step=1.0)})


# ════════════════════════════════════════════════════════════════
# 22-27  VOLUME
# ════════════════════════════════════════════════════════════════

# 22. MFI — Money Flow Index
def gen_mfi(df, period, buy, sell):
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos = mf.where(tp > tp.shift(), 0).rolling(period).sum()
    neg = mf.where(tp < tp.shift(), 0).rolling(period).sum()
    mfi_val = 100 - 100 / (1 + pos / (neg + 1e-10))
    sig = pd.Series(0, index=df.index)
    sig[(mfi_val < buy) & (mfi_val > mfi_val.shift())] = 1
    sig[(mfi_val > sell) & (mfi_val < mfi_val.shift())] = -1
    return sig

reg('MFI', gen_mfi,
    lambda t: {'period': t.suggest_int('period', 8, 28),
               'buy': t.suggest_int('buy', 15, 35),
               'sell': t.suggest_int('sell', 65, 85)})


# 23. CMF — Chaikin Money Flow
def gen_cmf(df, period, thresh):
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    cmf_val = (mfm * df['volume']).rolling(period).sum() / (df['volume'].rolling(period).sum() + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(cmf_val > thresh) & (cmf_val.shift() <= thresh)] = 1
    sig[(cmf_val < -thresh) & (cmf_val.shift() >= -thresh)] = -1
    return sig

reg('CMF', gen_cmf,
    lambda t: {'period': t.suggest_int('period', 10, 30),
               'thresh': t.suggest_float('thresh', 0.05, 0.3, step=0.025)})


# 24. VWMA_Cross — Volume Weighted MA crossover
def gen_vwma_cross(df, fast, slow):
    vwma_f = (df['close'] * df['volume']).rolling(fast).sum() / (df['volume'].rolling(fast).sum() + 1e-10)
    vwma_s = (df['close'] * df['volume']).rolling(slow).sum() / (df['volume'].rolling(slow).sum() + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(vwma_f > vwma_s) & (vwma_f.shift() <= vwma_s.shift())] = 1
    sig[(vwma_f < vwma_s) & (vwma_f.shift() >= vwma_s.shift())] = -1
    return sig

reg('VWMA_Cross', gen_vwma_cross,
    lambda t: {'fast': t.suggest_int('fast', 5, 20),
               'slow': t.suggest_int('slow', 20, 80)})


# 25. Force_Index — Elder's Force Index
def gen_force_index(df, period):
    fi = ema(df['close'].diff() * df['volume'], period)
    sig = pd.Series(0, index=df.index)
    sig[(fi > 0) & (fi.shift() <= 0)] = 1
    sig[(fi < 0) & (fi.shift() >= 0)] = -1
    return sig

reg('Force_Index', gen_force_index,
    lambda t: {'period': t.suggest_int('period', 5, 30)})


# 26. EMV — Ease of Movement
def gen_emv(df, period, thresh):
    dm = ((df['high'] + df['low']) / 2) - ((df['high'].shift() + df['low'].shift()) / 2)
    box_ratio = (df['volume'] / 1e6) / (df['high'] - df['low'] + 1e-10)
    emv_raw = dm / (box_ratio + 1e-10)
    emv_val = sma(emv_raw, period)
    sig = pd.Series(0, index=df.index)
    sig[(emv_val > thresh) & (emv_val.shift() <= thresh)] = 1
    sig[(emv_val < -thresh) & (emv_val.shift() >= -thresh)] = -1
    return sig

reg('EMV', gen_emv,
    lambda t: {'period': t.suggest_int('period', 8, 25),
               'thresh': t.suggest_float('thresh', 0.0, 5.0, step=0.5)})


# 27. AD_Line — Accumulation/Distribution Line divergence
def gen_ad_line(df, fast, slow):
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    adl = (mfm * df['volume']).cumsum()
    adl_fast = ema(adl, fast)
    adl_slow = ema(adl, slow)
    sig = pd.Series(0, index=df.index)
    sig[(adl_fast > adl_slow) & (adl_fast.shift() <= adl_slow.shift())] = 1
    sig[(adl_fast < adl_slow) & (adl_fast.shift() >= adl_slow.shift())] = -1
    return sig

reg('AD_Line', gen_ad_line,
    lambda t: {'fast': t.suggest_int('fast', 3, 15),
               'slow': t.suggest_int('slow', 15, 50)})


# ════════════════════════════════════════════════════════════════
# 28-31  MEAN REVERSION
# ════════════════════════════════════════════════════════════════

# 28. RSI_Divergence — RSI bullish/bearish divergence (vectorized approx)
def gen_rsi_divergence(df, rsi_period, lookback, rsi_thresh_low, rsi_thresh_high):
    r = rsi(df['close'], rsi_period)
    price_low = df['close'].rolling(lookback).min()
    rsi_at_low = r.rolling(lookback).min()
    # Bullish divergence: price makes new low but RSI doesn't
    bull_div = (df['close'] <= price_low * 1.01) & (r > rsi_at_low * 1.05) & (r < rsi_thresh_low)
    price_high = df['close'].rolling(lookback).max()
    rsi_at_high = r.rolling(lookback).max()
    # Bearish divergence: price makes new high but RSI doesn't
    bear_div = (df['close'] >= price_high * 0.99) & (r < rsi_at_high * 0.95) & (r > rsi_thresh_high)
    sig = pd.Series(0, index=df.index)
    sig[bull_div] = 1
    sig[bear_div] = -1
    return sig

reg('RSI_Divergence', gen_rsi_divergence,
    lambda t: {'rsi_period': t.suggest_int('rsi_period', 8, 21),
               'lookback': t.suggest_int('lookback', 10, 30),
               'rsi_thresh_low': t.suggest_int('rsi_thresh_low', 25, 45),
               'rsi_thresh_high': t.suggest_int('rsi_thresh_high', 55, 75)})


# 29. BB_Width — Bollinger Band width squeeze then expand
def gen_bb_width(df, period, std_mult, squeeze_pctile):
    mid, upper, lower = bb(df['close'], period, std_mult)
    width = (upper - lower) / (mid + 1e-10)
    width_pctile = width.rolling(100).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x) * 100 if len(x) > 0 else 50, raw=False)
    # Squeeze: low width; expand: width increasing
    was_squeezed = width_pctile.shift() < squeeze_pctile
    expanding = width > width.shift()
    sig = pd.Series(0, index=df.index)
    sig[was_squeezed & expanding & (df['close'] > mid)] = 1
    sig[was_squeezed & expanding & (df['close'] < mid)] = -1
    return sig

reg('BB_Width', gen_bb_width,
    lambda t: {'period': t.suggest_int('period', 10, 30),
               'std_mult': t.suggest_float('std_mult', 1.5, 3.0, step=0.25),
               'squeeze_pctile': t.suggest_int('squeeze_pctile', 10, 35)})


# 30. Regression_Channel — Linear regression channel
def gen_regression_channel(df, period, mult):
    c = df['close']
    # Rolling linear regression using numpy vectorized
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    reg_val = c.rolling(period).apply(
        lambda y: np.polyval(np.polyfit(x, y, 1), period - 1), raw=True)
    reg_std = c.rolling(period).std()
    upper = reg_val + mult * reg_std
    lower = reg_val - mult * reg_std
    sig = pd.Series(0, index=df.index)
    sig[(c < lower) & (c > c.shift())] = 1   # Below lower + turning up
    sig[(c > upper) & (c < c.shift())] = -1   # Above upper + turning down
    return sig

reg('Regression_Channel', gen_regression_channel,
    lambda t: {'period': t.suggest_int('period', 15, 50),
               'mult': t.suggest_float('mult', 1.0, 3.0, step=0.25)})


# 31. Zscore_Volume — Z-score with volume confirmation
def gen_zscore_vol(df, lookback, z_thresh, vol_mult):
    m = df['close'].rolling(lookback).mean()
    s = df['close'].rolling(lookback).std()
    z = (df['close'] - m) / (s + 1e-10)
    vol_avg = df['volume'].rolling(lookback).mean()
    vol_high = df['volume'] > vol_mult * vol_avg
    sig = pd.Series(0, index=df.index)
    sig[(z < -z_thresh) & vol_high] = 1
    sig[(z > z_thresh) & vol_high] = -1
    return sig

reg('Zscore_Volume', gen_zscore_vol,
    lambda t: {'lookback': t.suggest_int('lookback', 10, 60),
               'z_thresh': t.suggest_float('z_thresh', 1.0, 3.0, step=0.25),
               'vol_mult': t.suggest_float('vol_mult', 1.0, 3.0, step=0.25)})


# ════════════════════════════════════════════════════════════════
# 32-36  CANDLESTICK PATTERNS
# ════════════════════════════════════════════════════════════════

# 32. Hammer_Star — Hammer and Shooting Star
def gen_hammer_star(df, body_ratio, shadow_ratio):
    body = (df['close'] - df['open']).abs()
    total = df['high'] - df['low'] + 1e-10
    upper_shadow = df['high'] - pd.concat([df['open'], df['close']], axis=1).max(axis=1)
    lower_shadow = pd.concat([df['open'], df['close']], axis=1).min(axis=1) - df['low']
    hammer = (lower_shadow > shadow_ratio * body) & (upper_shadow < body * body_ratio) & (body / total < 0.4)
    star = (upper_shadow > shadow_ratio * body) & (lower_shadow < body * body_ratio) & (body / total < 0.4)
    sig = pd.Series(0, index=df.index)
    sig[hammer] = 1
    sig[star] = -1
    return sig

reg('Hammer_Star', gen_hammer_star,
    lambda t: {'body_ratio': t.suggest_float('body_ratio', 0.2, 0.6, step=0.1),
               'shadow_ratio': t.suggest_float('shadow_ratio', 1.5, 4.0, step=0.5)})


# 33. Three_Soldiers — Three white soldiers / black crows
def gen_three_soldiers(df, min_body_pct):
    body = df['close'] - df['open']
    total = df['high'] - df['low'] + 1e-10
    body_pct = body.abs() / total
    # Three white soldiers
    bull1 = body.shift(2) > 0
    bull2 = body.shift(1) > 0
    bull3 = body > 0
    big1 = body_pct.shift(2) > min_body_pct
    big2 = body_pct.shift(1) > min_body_pct
    big3 = body_pct > min_body_pct
    higher = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))
    soldiers = bull1 & bull2 & bull3 & big1 & big2 & big3 & higher
    # Three black crows
    bear1 = body.shift(2) < 0
    bear2 = body.shift(1) < 0
    bear3 = body < 0
    lower = (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))
    crows = bear1 & bear2 & bear3 & big1 & big2 & big3 & lower
    sig = pd.Series(0, index=df.index)
    sig[soldiers] = 1
    sig[crows] = -1
    return sig

reg('Three_Soldiers', gen_three_soldiers,
    lambda t: {'min_body_pct': t.suggest_float('min_body_pct', 0.3, 0.7, step=0.05)})


# 34. Morning_Star — Morning/Evening star
def gen_morning_star(df, body_thresh):
    total = df['high'] - df['low'] + 1e-10
    body_pct = (df['close'] - df['open']).abs() / total
    big_bear = (df['open'].shift(2) > df['close'].shift(2)) & (body_pct.shift(2) > body_thresh)
    small_body = body_pct.shift(1) < 0.3
    big_bull = (df['close'] > df['open']) & (body_pct > body_thresh)
    morning = big_bear & small_body & big_bull
    big_bull2 = (df['close'].shift(2) > df['open'].shift(2)) & (body_pct.shift(2) > body_thresh)
    big_bear2 = (df['close'] < df['open']) & (body_pct > body_thresh)
    evening = big_bull2 & small_body & big_bear2
    sig = pd.Series(0, index=df.index)
    sig[morning] = 1
    sig[evening] = -1
    return sig

reg('Morning_Star', gen_morning_star,
    lambda t: {'body_thresh': t.suggest_float('body_thresh', 0.4, 0.7, step=0.05)})


# 35. Harami — Bullish/Bearish harami
def gen_harami(df, min_parent_pct):
    body = df['close'] - df['open']
    total = df['high'] - df['low'] + 1e-10
    body_pct = body.abs() / total
    # Bullish harami: big bearish candle, then small bullish candle inside
    big_bear = (body.shift(1) < 0) & (body_pct.shift(1) > min_parent_pct)
    small_bull = (body > 0) & (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))
    bull_harami = big_bear & small_bull
    # Bearish harami: big bullish candle, then small bearish candle inside
    big_bull = (body.shift(1) > 0) & (body_pct.shift(1) > min_parent_pct)
    small_bear = (body < 0) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
    bear_harami = big_bull & small_bear
    sig = pd.Series(0, index=df.index)
    sig[bull_harami] = 1
    sig[bear_harami] = -1
    return sig

reg('Harami', gen_harami,
    lambda t: {'min_parent_pct': t.suggest_float('min_parent_pct', 0.4, 0.8, step=0.05)})


# 36. Pin_Bar — Pin bar reversal with RSI filter
def gen_pin_bar(df, pin_ratio, rsi_period, rsi_low, rsi_high):
    body = (df['close'] - df['open']).abs()
    upper_wick = df['high'] - pd.concat([df['open'], df['close']], axis=1).max(axis=1)
    lower_wick = pd.concat([df['open'], df['close']], axis=1).min(axis=1) - df['low']
    r = rsi(df['close'], rsi_period)
    bull_pin = (lower_wick > pin_ratio * body) & (upper_wick < body * 0.5) & (r < rsi_low)
    bear_pin = (upper_wick > pin_ratio * body) & (lower_wick < body * 0.5) & (r > rsi_high)
    sig = pd.Series(0, index=df.index)
    sig[bull_pin] = 1
    sig[bear_pin] = -1
    return sig

reg('Pin_Bar', gen_pin_bar,
    lambda t: {'pin_ratio': t.suggest_float('pin_ratio', 1.5, 4.0, step=0.5),
               'rsi_period': t.suggest_int('rsi_period', 8, 21),
               'rsi_low': t.suggest_int('rsi_low', 20, 40),
               'rsi_high': t.suggest_int('rsi_high', 60, 80)})


# ════════════════════════════════════════════════════════════════
# 37-45  MULTI-INDICATOR COMBOS
# ════════════════════════════════════════════════════════════════

# 37. RSI_BB — RSI oversold/overbought + BB touch
def gen_rsi_bb(df, rsi_period, bb_period, bb_std, rsi_buy, rsi_sell):
    r = rsi(df['close'], rsi_period)
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (df['close'] <= lower)] = 1
    sig[(r > rsi_sell) & (df['close'] >= upper)] = -1
    return sig

reg('RSI_BB', gen_rsi_bb,
    lambda t: {'rsi_period': t.suggest_int('rsi_period', 8, 21),
               'bb_period': t.suggest_int('bb_period', 12, 30),
               'bb_std': t.suggest_float('bb_std', 1.5, 3.0, step=0.25),
               'rsi_buy': t.suggest_int('rsi_buy', 20, 40),
               'rsi_sell': t.suggest_int('rsi_sell', 60, 80)})


# 38. MACD_BB — MACD cross + BB position
def gen_macd_bb(df, fast, slow, signal_p, bb_period, bb_std):
    ml, sl, hist = macd(df['close'], fast, slow, signal_p)
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[(hist > 0) & (hist.shift() <= 0) & (df['close'] < mid)] = 1
    sig[(hist < 0) & (hist.shift() >= 0) & (df['close'] > mid)] = -1
    return sig

reg('MACD_BB', gen_macd_bb,
    lambda t: {'fast': t.suggest_int('fast', 6, 16),
               'slow': t.suggest_int('slow', 18, 40),
               'signal_p': t.suggest_int('signal_p', 5, 14),
               'bb_period': t.suggest_int('bb_period', 12, 30),
               'bb_std': t.suggest_float('bb_std', 1.5, 3.0, step=0.25)})


# 39. Stoch_RSI — Stochastic applied to RSI
def gen_stoch_rsi(df, rsi_period, stoch_period, buy, sell):
    r = rsi(df['close'], rsi_period)
    r_min = r.rolling(stoch_period).min()
    r_max = r.rolling(stoch_period).max()
    sr = (r - r_min) / (r_max - r_min + 1e-10) * 100
    sr_k = sr.rolling(3).mean()
    sr_d = sr_k.rolling(3).mean()
    sig = pd.Series(0, index=df.index)
    sig[(sr_k < buy) & (sr_k > sr_d) & (sr_k.shift() <= sr_d.shift())] = 1
    sig[(sr_k > sell) & (sr_k < sr_d) & (sr_k.shift() >= sr_d.shift())] = -1
    return sig

reg('Stoch_RSI', gen_stoch_rsi,
    lambda t: {'rsi_period': t.suggest_int('rsi_period', 8, 21),
               'stoch_period': t.suggest_int('stoch_period', 8, 21),
               'buy': t.suggest_int('buy', 15, 35),
               'sell': t.suggest_int('sell', 65, 85)})


# 40. EMA_RSI_Vol — EMA cross + RSI filter + Volume spike
def gen_ema_rsi_vol(df, ema_fast, ema_slow, rsi_period, rsi_mid, vol_mult):
    ef = ema(df['close'], ema_fast)
    es = ema(df['close'], ema_slow)
    r = rsi(df['close'], rsi_period)
    vol_avg = df['volume'].rolling(20).mean()
    vol_spike = df['volume'] > vol_mult * vol_avg
    sig = pd.Series(0, index=df.index)
    sig[(ef > es) & (ef.shift() <= es.shift()) & (r > rsi_mid) & vol_spike] = 1
    sig[(ef < es) & (ef.shift() >= es.shift()) & (r < (100 - rsi_mid)) & vol_spike] = -1
    return sig

reg('EMA_RSI_Vol', gen_ema_rsi_vol,
    lambda t: {'ema_fast': t.suggest_int('ema_fast', 5, 20),
               'ema_slow': t.suggest_int('ema_slow', 20, 60),
               'rsi_period': t.suggest_int('rsi_period', 8, 21),
               'rsi_mid': t.suggest_int('rsi_mid', 45, 60),
               'vol_mult': t.suggest_float('vol_mult', 1.2, 3.0, step=0.2)})


# 41. ADX_EMA — ADX trending + EMA crossover
def gen_adx_ema(df, adx_period, adx_thresh, ema_fast, ema_slow):
    adx_val, pdi, mdi = adx_calc(df['high'], df['low'], df['close'], adx_period)
    ef = ema(df['close'], ema_fast)
    es = ema(df['close'], ema_slow)
    trending = adx_val > adx_thresh
    sig = pd.Series(0, index=df.index)
    sig[trending & (ef > es) & (ef.shift() <= es.shift())] = 1
    sig[trending & (ef < es) & (ef.shift() >= es.shift())] = -1
    return sig

reg('ADX_EMA', gen_adx_ema,
    lambda t: {'adx_period': t.suggest_int('adx_period', 8, 25),
               'adx_thresh': t.suggest_int('adx_thresh', 15, 35),
               'ema_fast': t.suggest_int('ema_fast', 5, 20),
               'ema_slow': t.suggest_int('ema_slow', 20, 60)})


# 42. BB_RSI_MACD — Triple confirmation
def gen_bb_rsi_macd(df, bb_period, bb_std, rsi_period, rsi_buy, rsi_sell, macd_fast, macd_slow, macd_sig):
    mid, upper, lower = bb(df['close'], bb_period, bb_std)
    r = rsi(df['close'], rsi_period)
    ml, sl, hist = macd(df['close'], macd_fast, macd_slow, macd_sig)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] <= lower) & (r < rsi_buy) & (hist > hist.shift())] = 1
    sig[(df['close'] >= upper) & (r > rsi_sell) & (hist < hist.shift())] = -1
    return sig

reg('BB_RSI_MACD', gen_bb_rsi_macd,
    lambda t: {'bb_period': t.suggest_int('bb_period', 12, 30),
               'bb_std': t.suggest_float('bb_std', 1.5, 3.0, step=0.25),
               'rsi_period': t.suggest_int('rsi_period', 8, 21),
               'rsi_buy': t.suggest_int('rsi_buy', 20, 40),
               'rsi_sell': t.suggest_int('rsi_sell', 60, 80),
               'macd_fast': t.suggest_int('macd_fast', 6, 16),
               'macd_slow': t.suggest_int('macd_slow', 18, 40),
               'macd_sig': t.suggest_int('macd_sig', 5, 14)})


# 43. Ichimoku_RSI — Ichimoku TK cross + RSI filter
def gen_ichimoku_rsi(df, tenkan_p, kijun_p, rsi_period, rsi_buy, rsi_sell):
    tenkan = (df['high'].rolling(tenkan_p).max() + df['low'].rolling(tenkan_p).min()) / 2
    kijun = (df['high'].rolling(kijun_p).max() + df['low'].rolling(kijun_p).min()) / 2
    r = rsi(df['close'], rsi_period)
    sig = pd.Series(0, index=df.index)
    sig[(tenkan > kijun) & (tenkan.shift() <= kijun.shift()) & (r > rsi_buy)] = 1
    sig[(tenkan < kijun) & (tenkan.shift() >= kijun.shift()) & (r < rsi_sell)] = -1
    return sig

reg('Ichimoku_RSI', gen_ichimoku_rsi,
    lambda t: {'tenkan_p': t.suggest_int('tenkan_p', 5, 15),
               'kijun_p': t.suggest_int('kijun_p', 15, 40),
               'rsi_period': t.suggest_int('rsi_period', 8, 21),
               'rsi_buy': t.suggest_int('rsi_buy', 40, 60),
               'rsi_sell': t.suggest_int('rsi_sell', 40, 60)})


# 44. Keltner_RSI — Keltner breakout + RSI confirmation
def gen_keltner_rsi(df, kc_period, kc_mult, rsi_period, rsi_buy, rsi_sell):
    mid = ema(df['close'], kc_period)
    atr_val = atr(df['high'], df['low'], df['close'], kc_period)
    upper = mid + kc_mult * atr_val
    lower = mid - kc_mult * atr_val
    r = rsi(df['close'], rsi_period)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] > upper) & (df['close'].shift() <= upper.shift()) & (r > rsi_buy)] = 1
    sig[(df['close'] < lower) & (df['close'].shift() >= lower.shift()) & (r < rsi_sell)] = -1
    return sig

reg('Keltner_RSI', gen_keltner_rsi,
    lambda t: {'kc_period': t.suggest_int('kc_period', 10, 30),
               'kc_mult': t.suggest_float('kc_mult', 1.0, 3.5, step=0.25),
               'rsi_period': t.suggest_int('rsi_period', 8, 21),
               'rsi_buy': t.suggest_int('rsi_buy', 45, 65),
               'rsi_sell': t.suggest_int('rsi_sell', 35, 55)})


# 45. SuperTrend_MACD — SuperTrend direction + MACD cross
def gen_supertrend_macd(df, st_period, st_mult, macd_fast, macd_slow, macd_sig):
    # Reuse SuperTrend logic
    hl2 = (df['high'] + df['low']) / 2
    atr_val = atr(df['high'], df['low'], df['close'], st_period)
    up = (hl2 - st_mult * atr_val).values
    dn = (hl2 + st_mult * atr_val).values
    close = df['close'].values
    n = len(close)
    final_up = np.copy(up)
    final_dn = np.copy(dn)
    direction = np.ones(n)
    for i in range(1, n):
        if not (up[i] > final_up[i - 1] or close[i - 1] < final_up[i - 1]):
            final_up[i] = final_up[i - 1]
        if not (dn[i] < final_dn[i - 1] or close[i - 1] > final_dn[i - 1]):
            final_dn[i] = final_dn[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < final_up[i] else 1
        else:
            direction[i] = 1 if close[i] > final_dn[i] else -1
    st_dir = pd.Series(direction, index=df.index)
    ml, sl, hist = macd(df['close'], macd_fast, macd_slow, macd_sig)
    sig = pd.Series(0, index=df.index)
    sig[(st_dir == 1) & (hist > 0) & (hist.shift() <= 0)] = 1
    sig[(st_dir == -1) & (hist < 0) & (hist.shift() >= 0)] = -1
    return sig

reg('SuperTrend_MACD', gen_supertrend_macd,
    lambda t: {'st_period': t.suggest_int('st_period', 5, 20),
               'st_mult': t.suggest_float('st_mult', 1.0, 4.0, step=0.5),
               'macd_fast': t.suggest_int('macd_fast', 6, 16),
               'macd_slow': t.suggest_int('macd_slow', 18, 40),
               'macd_sig': t.suggest_int('macd_sig', 5, 14)})


# ════════════════════════════════════════════════════════════════
# 46-52  ADVANCED
# ════════════════════════════════════════════════════════════════

# 46. Heikin_Ashi — Heikin Ashi candle color change
def gen_heikin_ashi(df, confirm_bars):
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    # Vectorized HA open via recursive formula approximation using EMA-like approach
    ha_open = (df['open'] + df['close']) / 2  # initial approx
    # Iterate numpy for HA open (recursive)
    ho = ha_open.values.copy()
    hc = ha_close.values
    ho[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    for i in range(1, len(ho)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    ha_o = pd.Series(ho, index=df.index)
    bull = (ha_close > ha_o).astype(int)
    bear = (ha_close <= ha_o).astype(int)
    # Require confirm_bars consecutive same-color candles
    bull_run = bull.rolling(confirm_bars).sum() == confirm_bars
    bear_run = bear.rolling(confirm_bars).sum() == confirm_bars
    sig = pd.Series(0, index=df.index)
    sig[bull_run & (bear.shift(confirm_bars) == 1)] = 1
    sig[bear_run & (bull.shift(confirm_bars) == 1)] = -1
    return sig

reg('Heikin_Ashi', gen_heikin_ashi,
    lambda t: {'confirm_bars': t.suggest_int('confirm_bars', 2, 5)})


# 47. Renko_Trend — Renko-style ATR brick detection
def gen_renko_trend(df, atr_period, brick_mult, confirm):
    atr_val = atr(df['high'], df['low'], df['close'], atr_period)
    brick_size = (atr_val.rolling(atr_period).mean() * brick_mult)
    # Measure cumulative move in one direction
    ret = df['close'].diff()
    cum_up = ret.where(ret > 0, 0).rolling(confirm).sum()
    cum_dn = ret.where(ret < 0, 0).rolling(confirm).sum().abs()
    sig = pd.Series(0, index=df.index)
    sig[(cum_up > brick_size) & (cum_up.shift() <= brick_size.shift())] = 1
    sig[(cum_dn > brick_size) & (cum_dn.shift() <= brick_size.shift())] = -1
    return sig

reg('Renko_Trend', gen_renko_trend,
    lambda t: {'atr_period': t.suggest_int('atr_period', 8, 25),
               'brick_mult': t.suggest_float('brick_mult', 0.5, 2.0, step=0.25),
               'confirm': t.suggest_int('confirm', 3, 10)})


# 48. Elder_Ray — Bull Power / Bear Power
def gen_elder_ray(df, period):
    ema_val = ema(df['close'], period)
    bull_power = df['high'] - ema_val
    bear_power = df['low'] - ema_val
    sig = pd.Series(0, index=df.index)
    # Buy: bear_power < 0 but rising + bull_power > 0
    sig[(bear_power < 0) & (bear_power > bear_power.shift()) & (bull_power > 0)] = 1
    # Sell: bull_power > 0 but falling + bear_power < 0
    sig[(bull_power > 0) & (bull_power < bull_power.shift()) & (bear_power < 0)] = -1
    return sig

reg('Elder_Ray', gen_elder_ray,
    lambda t: {'period': t.suggest_int('period', 8, 25)})


# 49. Mass_Index — Mass Index reversal bulge
def gen_mass_index(df, period, bulge_thresh):
    ema_range = ema(df['high'] - df['low'], 9)
    double_ema = ema(ema_range, 9)
    ratio = ema_range / (double_ema + 1e-10)
    mi = ratio.rolling(period).sum()
    # Reversal bulge: MI crosses above threshold then back below
    crossed_up = (mi > bulge_thresh) & (mi.shift() <= bulge_thresh)
    crossed_dn = (mi < bulge_thresh) & (mi.shift() >= bulge_thresh)
    # Look for recent bulge (within last 5 bars)
    recent_bulge = crossed_up.rolling(5).sum() > 0
    sig = pd.Series(0, index=df.index)
    sig[crossed_dn & recent_bulge & (df['close'] > ema(df['close'], 9))] = 1
    sig[crossed_dn & recent_bulge & (df['close'] < ema(df['close'], 9))] = -1
    return sig

reg('Mass_Index', gen_mass_index,
    lambda t: {'period': t.suggest_int('period', 20, 30),
               'bulge_thresh': t.suggest_float('bulge_thresh', 26.0, 28.0, step=0.25)})


# 50. Vortex — Vortex Indicator crossover
def gen_vortex(df, period):
    tr = true_range(df['high'], df['low'], df['close'])
    vm_plus = (df['high'] - df['low'].shift()).abs()
    vm_minus = (df['low'] - df['high'].shift()).abs()
    vi_plus = vm_plus.rolling(period).sum() / (tr.rolling(period).sum() + 1e-10)
    vi_minus = vm_minus.rolling(period).sum() / (tr.rolling(period).sum() + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(vi_plus > vi_minus) & (vi_plus.shift() <= vi_minus.shift())] = 1
    sig[(vi_plus < vi_minus) & (vi_plus.shift() >= vi_minus.shift())] = -1
    return sig

reg('Vortex', gen_vortex,
    lambda t: {'period': t.suggest_int('period', 10, 30)})


# 51. Connors_RSI — 3-component RSI (standard + streak + percentile rank)
def gen_connors_rsi(df, rsi_period, streak_period, rank_period, buy, sell):
    c = df['close']
    rsi_val = rsi(c, rsi_period)
    # Streak: consecutive up/down days (numpy loop for speed)
    vals = c.values
    streak = np.zeros(len(vals))
    for i in range(1, len(vals)):
        if vals[i] > vals[i - 1]:
            streak[i] = max(streak[i - 1], 0) + 1
        elif vals[i] < vals[i - 1]:
            streak[i] = min(streak[i - 1], 0) - 1
        else:
            streak[i] = 0
    streak_s = pd.Series(streak, index=df.index)
    streak_rsi = rsi(streak_s, streak_period)
    # Percentile rank of 1-day return
    ret = c.pct_change()
    pct_rank = ret.rolling(rank_period).apply(
        lambda x: (x[:-1] < x[-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50, raw=True)
    crsi = (rsi_val + streak_rsi + pct_rank) / 3
    sig = pd.Series(0, index=df.index)
    sig[(crsi < buy) & (crsi > crsi.shift())] = 1
    sig[(crsi > sell) & (crsi < crsi.shift())] = -1
    return sig

reg('Connors_RSI', gen_connors_rsi,
    lambda t: {'rsi_period': t.suggest_int('rsi_period', 2, 5),
               'streak_period': t.suggest_int('streak_period', 2, 5),
               'rank_period': t.suggest_int('rank_period', 50, 150),
               'buy': t.suggest_int('buy', 10, 30),
               'sell': t.suggest_int('sell', 70, 90)})


# 52. Choppiness — Choppiness Index (trade only when NOT choppy)
def gen_choppiness(df, period, chop_thresh, ema_fast, ema_slow):
    tr = true_range(df['high'], df['low'], df['close'])
    atr_sum = tr.rolling(period).sum()
    hi = df['high'].rolling(period).max()
    lo = df['low'].rolling(period).min()
    chop = 100 * np.log10(atr_sum / (hi - lo + 1e-10)) / np.log10(period)
    ef = ema(df['close'], ema_fast)
    es = ema(df['close'], ema_slow)
    not_choppy = chop < chop_thresh
    sig = pd.Series(0, index=df.index)
    sig[not_choppy & (ef > es) & (ef.shift() <= es.shift())] = 1
    sig[not_choppy & (ef < es) & (ef.shift() >= es.shift())] = -1
    return sig

reg('Choppiness', gen_choppiness,
    lambda t: {'period': t.suggest_int('period', 10, 25),
               'chop_thresh': t.suggest_float('chop_thresh', 38.0, 55.0, step=1.0),
               'ema_fast': t.suggest_int('ema_fast', 5, 20),
               'ema_slow': t.suggest_int('ema_slow', 20, 60)})


# ════════════════════════════════════════════════════════════════
# ROUND 5 — TradingView Strategies (TV_ prefix)
# ════════════════════════════════════════════════════════════════

# --- TV_DOUBLE_RSI: Fast RSI + MTF proxy RSI crossover ---
def gen_tv_double_rsi(df, rsi_len=14, mtf_factor=3):
    c = df['close']
    fast_rsi = rsi(c, rsi_len)
    slow_rsi = rsi(c, rsi_len * mtf_factor)
    sig = pd.Series(0, index=df.index)
    sig[(fast_rsi < 30) & (slow_rsi < 50)] = 1
    sig[(fast_rsi > 70) & (slow_rsi > 50)] = -1
    return sig

reg('TV_DOUBLE_RSI', gen_tv_double_rsi,
    lambda t: {'rsi_len': t.suggest_int('rsi_len', 7, 21),
               'mtf_factor': t.suggest_int('mtf_factor', 2, 5)})


# --- TV_BB_WINNER_PRO: BB + RSI + MA + candle filter ---
def gen_tv_bb_winner_pro(df, bb_len=20, bb_mult=2.0, rsi_len=14,
                          rsi_above=45, ma_len=200, candle_pct=30):
    c, o, h, l_ = df['close'], df['open'], df['high'], df['low']
    mid, upper, lower = bb(c, bb_len, bb_mult)
    r = rsi(c, rsi_len)
    ma = ema(c, ma_len)
    body = (c - o).abs()
    wick_lo = pd.concat([o, c], axis=1).min(axis=1) - l_
    wick_hi = h - pd.concat([o, c], axis=1).max(axis=1)
    buyzone = c - (body * candle_pct / 100)
    sellzone = c + (body * candle_pct / 100)
    sig = pd.Series(0, index=df.index)
    sig[(buyzone < lower) & (c < o) & (r < rsi_above) & (c > ma)] = 1
    sig[(sellzone > upper) & (c > o) & (r > (100 - rsi_above)) & (c < ma)] = -1
    return sig

reg('TV_BB_WINNER_PRO', gen_tv_bb_winner_pro,
    lambda t: {'bb_len': t.suggest_int('bb_len', 10, 40),
               'bb_mult': t.suggest_float('bb_mult', 1.5, 3.0),
               'rsi_len': t.suggest_int('rsi_len', 7, 21),
               'rsi_above': t.suggest_int('rsi_above', 30, 55),
               'ma_len': t.suggest_int('ma_len', 100, 400),
               'candle_pct': t.suggest_int('candle_pct', 10, 50)})


# --- TV_PASSWORD_BB_EMA: BB breakout + EMA trend filter ---
def gen_tv_password_bb_ema(df, bb_len=20, bb_mult=2.0, ema_len=200):
    c = df['close']
    mid, upper, lower = bb(c, bb_len, bb_mult)
    trend = ema(c, ema_len)
    sig = pd.Series(0, index=df.index)
    sig[(c < lower) & (c > trend)] = 1
    sig[(c > upper) & (c < trend)] = -1
    return sig

reg('TV_PASSWORD_BB_EMA', gen_tv_password_bb_ema,
    lambda t: {'bb_len': t.suggest_int('bb_len', 10, 40),
               'bb_mult': t.suggest_float('bb_mult', 1.5, 3.0),
               'ema_len': t.suggest_int('ema_len', 100, 400)})


# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"STRATEGY_TYPES_R2: {len(STRATEGY_TYPES_R2)} strategy types registered")
    for name in STRATEGY_TYPES_R2:
        print(f"  - {name}")
