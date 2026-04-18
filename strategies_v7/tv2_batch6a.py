#!/usr/bin/env python3
"""
TV2 Batch 6a — 8 estrategias Pine → Python
Likes totales: ~35,300

1. RSI_Divergence         (9172 likes) — Pine v4
2. Full_CRYPTO_pack        (4859 likes) — Pine v4
3. Price_Volume_Breakout   (4771 likes) — Pine v5
4. HMA_72s_Adaptive        (4552 likes) — Pine v4
5. Hull_MA_Swing_Trader    (3733 likes) — Pine v4
6. EVWMA_VWAP_MACD         (3575 likes) — Pine v4 (QuantNomad)
7. QuantNomad_HA_PSAR      (3188 likes) — Pine v4
8. WaveTrend_MFI_EMA_PB    (2451 likes) — Pine v4 (TradePro Cipher B+)

Convención: sig = 1 (LONG), -1 (SHORT), 0 (sin señal)
Sin look-ahead: señales basadas en datos hasta la barra actual (shift(1) para prev).
"""

import pandas as pd
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _sma(s, p):
    return s.rolling(p).mean()

def _rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0).ewm(span=p, adjust=False).mean()
    l = (-d.where(d < 0, 0.0)).ewm(span=p, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def _atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def _wma(s, p):
    w = np.arange(1, p + 1, dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def _hma(s, p):
    return _wma(2 * _wma(s, p // 2) - _wma(s, p), max(1, int(p ** 0.5)))


# ─── 1. RSI DIVERGENCE ────────────────────────────────────────────────────────

def gen_RSI_Divergence(df, rsi_len=9, lbL=1, lbR=3, range_lower=5, range_upper=60):
    """
    RSI Divergence — Pine v4 (9172 likes)
    Regular Bullish: precio hace lower-low, RSI hace higher-low → LONG
    Regular Bearish: precio hace higher-high, RSI hace lower-high → SHORT
    Señal confirmada lbR barras después del pivot (sin look-ahead).
    """
    close = df['close']
    rsi   = _rsi(close, rsi_len)

    def pivot_low(series, left, right):
        r = series.values
        n = len(r)
        out = np.zeros(n, dtype=bool)
        for i in range(left, n - right):
            if np.all(r[i] <= r[i - left:i]) and np.all(r[i] <= r[i + 1:i + right + 1]):
                out[i] = True
        return pd.Series(out, index=series.index)

    def pivot_high(series, left, right):
        r = series.values
        n = len(r)
        out = np.zeros(n, dtype=bool)
        for i in range(left, n - right):
            if np.all(r[i] >= r[i - left:i]) and np.all(r[i] >= r[i + 1:i + right + 1]):
                out[i] = True
        return pd.Series(out, index=series.index)

    pl = pivot_low(rsi, lbL, lbR)
    ph = pivot_high(rsi, lbL, lbR)

    c     = close.values
    r     = rsi.values
    pl_v  = pl.values
    ph_v  = ph.values
    n     = len(df)

    sig_arr = np.zeros(n)

    for i in range(lbR + range_upper + 1, n):
        # Bullish divergence: pivot low en RSI
        if pl_v[i]:
            curr_rsi   = r[i]
            curr_close = c[i]
            for j in range(i - range_lower, max(0, i - range_upper - 1), -1):
                if pl_v[j]:
                    # Regular bull: precio lower-low, RSI higher-low
                    if curr_close < c[j] and curr_rsi > r[j]:
                        fire = i + lbR
                        if fire < n:
                            sig_arr[fire] = 1
                    break

        # Bearish divergence: pivot high en RSI
        if ph_v[i]:
            curr_rsi   = r[i]
            curr_close = c[i]
            for j in range(i - range_lower, max(0, i - range_upper - 1), -1):
                if ph_v[j]:
                    # Regular bear: precio higher-high, RSI lower-high
                    if curr_close > c[j] and curr_rsi < r[j]:
                        fire = i + lbR
                        if fire < n:
                            sig_arr[fire] = -1
                    break

    return pd.Series(sig_arr, index=df.index)


def space_RSI_Divergence(trial):
    return {
        'rsi_len':     trial.suggest_int('rsi_len',     5, 21),
        'lbL':         trial.suggest_int('lbL',         1,  5),
        'lbR':         trial.suggest_int('lbR',         1,  5),
        'range_lower': trial.suggest_int('range_lower', 3, 15),
        'range_upper': trial.suggest_int('range_upper', 20, 100),
    }


# ─── 2. FULL CRYPTO PACK ──────────────────────────────────────────────────────

def gen_Full_CRYPTO_pack(df, ema_len=9, macd_fast=12, macd_slow=26,
                          macd_sig=9, rsi_len=14, obv_ema=20):
    """
    Full CRYPTO Pack — Pine v4 (4859 likes)
    Multi-indicator: EMA(hl2,9) + MACD + RSI + OBV EMA cross.
    Trigger: crossover/crossunder de close sobre EMA9 con confirmación de MACD y RSI.
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']

    src  = (high + low) / 2            # hl2
    ema9 = _ema(src, ema_len)

    # MACD
    macd_line   = _ema(close, macd_fast) - _ema(close, macd_slow)
    macd_signal = _ema(macd_line, macd_sig)
    hist        = macd_line - macd_signal

    # RSI
    rsi = _rsi(close, rsi_len)

    # OBV
    obv          = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_ema_line = _ema(obv, obv_ema)

    # Crossovers de close sobre ema9 (sin look-ahead: prev bar via shift(1))
    co_long  = (
        (close.shift(1) < ema9.shift(1)) &
        (close >= ema9) &
        (hist > 0) &
        (rsi > 50) &
        (obv > obv_ema_line)
    )
    co_short = (
        (close.shift(1) > ema9.shift(1)) &
        (close <= ema9) &
        (hist < 0) &
        (rsi < 50) &
        (obv < obv_ema_line)
    )

    sig = pd.Series(0, index=df.index)
    sig[co_long]  = 1
    sig[co_short] = -1
    return sig


def space_Full_CRYPTO_pack(trial):
    return {
        'ema_len':   trial.suggest_int('ema_len',   5,  20),
        'macd_fast': trial.suggest_int('macd_fast', 8,  20),
        'macd_slow': trial.suggest_int('macd_slow', 20, 40),
        'macd_sig':  trial.suggest_int('macd_sig',  5,  15),
        'rsi_len':   trial.suggest_int('rsi_len',   7,  21),
        'obv_ema':   trial.suggest_int('obv_ema',   10, 40),
    }


# ─── 3. PRICE VOLUME BREAKOUT ─────────────────────────────────────────────────

def gen_Price_Volume_Breakout(df, price_period=60, vol_period=60, trend_len=200):
    """
    Price + Volume Dual Breakout — Pine v5 (4771 likes)
    Long:  close > highest(close, 60)[1]  AND  volume > highest(volume, 60)[1]  AND  close > SMA(200)
    Short: close < lowest(close, 60)[1]   AND  volume > highest(volume, 60)[1]  AND  close < SMA(200)
    [1] = shift(1) para evitar look-ahead (breakout del máximo de las anteriores N barras)
    """
    close  = df['close']
    volume = df['volume']

    # Máx/mín de las N barras ANTERIORES (shift(1) → excluye la barra actual)
    highest_close  = close.rolling(price_period).max().shift(1)
    lowest_close   = close.rolling(price_period).min().shift(1)
    highest_volume = volume.rolling(vol_period).max().shift(1)
    trend_sma      = _sma(close, trend_len)

    long_cond  = (close > highest_close)  & (volume > highest_volume) & (close > trend_sma)
    short_cond = (close < lowest_close)   & (volume > highest_volume) & (close < trend_sma)

    # Señal solo en el primer bar que cumple (entrada en breakout, no persistente)
    long_entry  = long_cond  & ~long_cond.shift(1).fillna(False)
    short_entry = short_cond & ~short_cond.fillna(False).shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    sig[long_entry]  = 1
    sig[short_entry] = -1
    return sig


def space_Price_Volume_Breakout(trial):
    return {
        'price_period': trial.suggest_int('price_period', 20,  120),
        'vol_period':   trial.suggest_int('vol_period',   20,  120),
        'trend_len':    trial.suggest_int('trend_len',    50,  300),
    }


# ─── 4. HMA 72s ADAPTIVE ─────────────────────────────────────────────────────

def gen_HMA_72s_Adaptive(df, hma_min=172, hma_max=233, atr_period=21):
    """
    HMA Adaptive — Pine v4 (4552 likes)
    Período adaptativo interpolado entre hma_min y hma_max según ratio ATR.
    Señal: cambio de pendiente del HMA (slope turns positive/negative).
    """
    close = df['close']
    atr   = _atr(df['high'], df['low'], close, atr_period)

    atr_min  = atr.rolling(50).min()
    atr_max  = atr.rolling(50).max()
    atr_norm = (atr - atr_min) / (atr_max - atr_min + 1e-10)

    period_s = (hma_min + atr_norm * (hma_max - hma_min)).fillna(hma_min)
    period_s = period_s.clip(hma_min, hma_max)

    # Usar el período modal (más frecuente) para un HMA estable
    valid = period_s.dropna()
    p = int(valid.mode()[0]) if len(valid) > 0 else hma_min

    hma = _hma(close, p)

    slope_up   = (hma > hma.shift(1)) & (hma.shift(1) <= hma.shift(2))
    slope_down = (hma < hma.shift(1)) & (hma.shift(1) >= hma.shift(2))

    sig = pd.Series(0, index=df.index)
    sig[slope_up]   = 1
    sig[slope_down] = -1
    return sig


def space_HMA_72s_Adaptive(trial):
    return {
        'hma_min':    trial.suggest_int('hma_min',    50,  250),
        'hma_max':    trial.suggest_int('hma_max',    100, 350),
        'atr_period': trial.suggest_int('atr_period', 7,   42),
    }


# ─── 5. HULL MA SWING TRADER ──────────────────────────────────────────────────

def gen_Hull_MA_Swing_Trader(df, period=210):
    """
    Hull MA Swing Trader — Pine v4 (3733 likes)
    HMA sobre open price. Señal en cambio de condición:
    Long:  open > HMA Y open[1] > HMA[1] Y HMA subiendo (entrada al inicio del uptrend)
    Short: open < HMA Y open[1] < HMA[1] Y HMA bajando
    Entrada solo en el primer bar que cumple la condición (no persistente).
    """
    price = df['open']
    hma   = _hma(price, period)

    long_cond  = (price > hma) & (price.shift(1) > hma.shift(1)) & (hma > hma.shift(1))
    short_cond = (price < hma) & (price.shift(1) < hma.shift(1)) & (hma < hma.shift(1))

    long_entry  = long_cond  & ~long_cond.shift(1).fillna(False)
    short_entry = short_cond & ~short_cond.shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    sig[long_entry]  = 1
    sig[short_entry] = -1
    return sig


def space_Hull_MA_Swing_Trader(trial):
    return {
        'period': trial.suggest_int('period', 50, 400),
    }


# ─── 6. EVWMA VWAP MACD ───────────────────────────────────────────────────────

def gen_EVWMA_VWAP_MACD(df, sum_length=30, signal_length1=5, signal_length2=20):
    """
    EVWMA vs VWAP MACD — QuantNomad, Pine v4 (3575 likes)
    EVWMA: Exponential Volume Weighted MA (stateful loop).
    diff = VWAP_rolling - EVWMA → MACD-style con dos EMAs.
    Long:  EMA(diff, 5) crossover  EMA(diff, 20)
    Short: EMA(diff, 5) crossunder EMA(diff, 20)
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']

    n = len(df)
    c_v  = close.values
    h_v  = high.values
    l_v  = low.values
    vol_v = volume.values

    # EVWMA stateful loop
    evwma_arr = np.zeros(n)
    evwma_arr[0] = c_v[0]
    for i in range(1, n):
        vol_sum = max(vol_v[max(0, i - sum_length + 1):i + 1].sum(), 1e-10)
        evwma_arr[i] = ((vol_sum - vol_v[i]) * evwma_arr[i - 1] + vol_v[i] * c_v[i]) / vol_sum

    evwma = pd.Series(evwma_arr, index=df.index)

    # Rolling VWAP sobre ventana sum_length
    tp     = (high + low + close) / 3
    vwap   = (tp * volume).rolling(sum_length).sum() / volume.rolling(sum_length).sum()

    diff    = vwap - evwma
    macd_f  = _ema(diff, signal_length1)
    macd_s  = _ema(diff, signal_length2)

    cross_up   = (macd_f.shift(1) < macd_s.shift(1)) & (macd_f >= macd_s)
    cross_down = (macd_f.shift(1) > macd_s.shift(1)) & (macd_f <= macd_s)

    sig = pd.Series(0, index=df.index)
    sig[cross_up]   = 1
    sig[cross_down] = -1
    return sig


def space_EVWMA_VWAP_MACD(trial):
    return {
        'sum_length':      trial.suggest_int('sum_length',      10, 60),
        'signal_length1':  trial.suggest_int('signal_length1',  3,  15),
        'signal_length2':  trial.suggest_int('signal_length2',  10, 40),
    }


# ─── 7. QUANTNOMAD HA PSAR ────────────────────────────────────────────────────

def gen_QuantNomad_HA_PSAR(df, psar_start=0.02, psar_inc=0.02, psar_max=0.2):
    """
    Heikin Ashi + Parabolic SAR — QuantNomad, Pine v4 (3188 likes)
    PSAR calculado sobre velas HA (stateful loop).
    Long:  HA close crossover PSAR (tendencia cambia a alcista)
    Short: HA close crossunder PSAR (tendencia cambia a bajista)
    """
    c   = df['close'].values
    h   = df['high'].values
    l   = df['low'].values
    o   = df['open'].values
    n   = len(df)

    # Heikin Ashi candles
    ha_close = (o + h + l + c) / 4
    ha_open  = np.zeros(n)
    ha_open[0] = (o[0] + c[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
    ha_low  = np.minimum(l, np.minimum(ha_open, ha_close))

    # PSAR stateful sobre HA
    psar    = np.zeros(n)
    sig_arr = np.zeros(n)

    psar[0] = ha_close[0]
    af      = psar_start
    ep      = ha_high[0]
    bull    = True

    for i in range(1, n):
        if bull:
            psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
            # PSAR en uptrend nunca mayor que los dos mínimos anteriores
            psar[i] = min(psar[i], ha_low[i - 1],
                          ha_low[i - 2] if i > 1 else ha_low[i - 1])
            if ha_close[i] < psar[i]:
                # Reversión a bajista
                bull    = False
                psar[i] = ep
                ep      = ha_low[i]
                af      = psar_start
                sig_arr[i] = -1
            else:
                if ha_high[i] > ep:
                    ep = ha_high[i]
                    af = min(af + psar_inc, psar_max)
        else:
            psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
            # PSAR en downtrend nunca menor que los dos máximos anteriores
            psar[i] = max(psar[i], ha_high[i - 1],
                          ha_high[i - 2] if i > 1 else ha_high[i - 1])
            if ha_close[i] > psar[i]:
                # Reversión a alcista
                bull    = True
                psar[i] = ep
                ep      = ha_high[i]
                af      = psar_start
                sig_arr[i] = 1
            else:
                if ha_low[i] < ep:
                    ep = ha_low[i]
                    af = min(af + psar_inc, psar_max)

    return pd.Series(sig_arr, index=df.index)


def space_QuantNomad_HA_PSAR(trial):
    return {
        'psar_start': trial.suggest_float('psar_start', 0.01, 0.05),
        'psar_inc':   trial.suggest_float('psar_inc',   0.01, 0.05),
        'psar_max':   trial.suggest_float('psar_max',   0.10, 0.40),
    }


# ─── 8. WAVETREND MFI EMA PB ──────────────────────────────────────────────────

def gen_WaveTrend_MFI_EMA(df, wt_ch=14, wt_avg=10, ob=53, os_=-60, ema_len=150):
    """
    WaveTrend + MFI + EMA Pullback — TradePro Cipher B+, Pine v4 (2451 likes)
    WaveTrend: ap=hlc3; esa=EMA(ap,ch); d=EMA(|ap-esa|,ch); ci=(ap-esa)/(0.015*d); wt1=EMA(ci,avg); wt2=SMA(wt1,4)
    Buy:  wt1 crossover  wt2 en zona oversold (wt2 <= os_) + close > EMA(150)
    Sell: wt1 crossunder wt2 en zona overbought (wt2 >= ob)  + close < EMA(150)
    """
    close = df['close']
    high  = df['high']
    low   = df['low']

    ap  = (close + high + low) / 3
    esa = _ema(ap, wt_ch)
    d   = _ema((ap - esa).abs(), wt_ch)
    ci  = (ap - esa) / (0.015 * d + 1e-10)
    wt1 = _ema(ci, wt_avg)
    wt2 = _sma(wt1, 4)

    ema_trend = _ema(close, ema_len)

    wt_cross_up   = (wt1.shift(1) < wt2.shift(1)) & (wt1 >= wt2) & (wt2 <= os_)
    wt_cross_down = (wt1.shift(1) > wt2.shift(1)) & (wt1 <= wt2) & (wt2 >= ob)

    sig = pd.Series(0, index=df.index)
    sig[wt_cross_up   & (close > ema_trend)] = 1
    sig[wt_cross_down & (close < ema_trend)] = -1
    return sig


def space_WaveTrend_MFI_EMA(trial):
    return {
        'wt_ch':   trial.suggest_int('wt_ch',   5,  30),
        'wt_avg':  trial.suggest_int('wt_avg',  5,  20),
        'ob':      trial.suggest_int('ob',       40,  70),
        'os_':     trial.suggest_int('os_',     -80, -40),
        'ema_len': trial.suggest_int('ema_len',  50, 300),
    }


# ─── STRATEGY EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'RSI_Divergence': {
        'gen':   gen_RSI_Divergence,
        'space': space_RSI_Divergence,
        'params': {
            'rsi_len': 9, 'lbL': 1, 'lbR': 3,
            'range_lower': 5, 'range_upper': 60,
        },
        'pine_version': 'v4',
        'likes': 9172,
    },
    'Full_CRYPTO_pack': {
        'gen':   gen_Full_CRYPTO_pack,
        'space': space_Full_CRYPTO_pack,
        'params': {
            'ema_len': 9, 'macd_fast': 12, 'macd_slow': 26,
            'macd_sig': 9, 'rsi_len': 14, 'obv_ema': 20,
        },
        'pine_version': 'v4',
        'likes': 4859,
    },
    'Price_Volume_Breakout': {
        'gen':   gen_Price_Volume_Breakout,
        'space': space_Price_Volume_Breakout,
        'params': {
            'price_period': 60, 'vol_period': 60, 'trend_len': 200,
        },
        'pine_version': 'v5',
        'likes': 4771,
    },
    'HMA_72s_Adaptive': {
        'gen':   gen_HMA_72s_Adaptive,
        'space': space_HMA_72s_Adaptive,
        'params': {
            'hma_min': 172, 'hma_max': 233, 'atr_period': 21,
        },
        'pine_version': 'v4',
        'likes': 4552,
    },
    'Hull_MA_Swing_Trader': {
        'gen':   gen_Hull_MA_Swing_Trader,
        'space': space_Hull_MA_Swing_Trader,
        'params': {
            'period': 210,
        },
        'pine_version': 'v4',
        'likes': 3733,
    },
    'EVWMA_VWAP_MACD': {
        'gen':   gen_EVWMA_VWAP_MACD,
        'space': space_EVWMA_VWAP_MACD,
        'params': {
            'sum_length': 30, 'signal_length1': 5, 'signal_length2': 20,
        },
        'pine_version': 'v4',
        'likes': 3575,
    },
    'QuantNomad_HA_PSAR': {
        'gen':   gen_QuantNomad_HA_PSAR,
        'space': space_QuantNomad_HA_PSAR,
        'params': {
            'psar_start': 0.02, 'psar_inc': 0.02, 'psar_max': 0.2,
        },
        'pine_version': 'v4',
        'likes': 3188,
    },
    'WaveTrend_MFI_EMA': {
        'gen':   gen_WaveTrend_MFI_EMA,
        'space': space_WaveTrend_MFI_EMA,
        'params': {
            'wt_ch': 14, 'wt_avg': 10, 'ob': 53, 'os_': -60, 'ema_len': 150,
        },
        'pine_version': 'v4',
        'likes': 2451,
    },
}
