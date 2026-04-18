#!/usr/bin/env python3
"""
TV2 BATCH 43 — EternaHybridExchange + Rodrigo Brito + kat3samsin (28 estrategias)
Pine v4/v5/v6 → Python. Sources: GitHub repos nuevos.

Familias:
  EternaHybridExchange (v5): AO, CCI, StochRSI, MomentumCombo, RSI_Div, TSI,
    UO, WilliamsR, Chandelier, SMA_5020, ADX_Trend, Keltner, LinReg, Donchian,
    VWAP_Bounce, SAR, EMA_Ribbon, Ichimoku, MFI, Chaikin
  Rodrigo Brito (v4): HiLo_Tranquilo, Scalping_3EMA
  kat3samsin (v4): BIAS_Strategy, Overbought_Swing, Oversold_Bounce,
    RSI_PowerZone, Sandwich_Strategy, Undercut_Strategy
"""
import pandas as pd
import numpy as np

# ─── HELPERS ───────────────────────────────────────────────────────────────────

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
    return tr.ewm(span=p, adjust=False).mean()

def _stoch(h, l, c, kp=14, dp=3, smooth=3):
    lo = l.rolling(kp).min()
    hi = h.rolling(kp).max()
    k = 100 * (c - lo) / (hi - lo + 1e-10)
    k_smooth = _sma(k, smooth)
    d = _sma(k_smooth, dp)
    return k_smooth, d

def _macd(c, f=12, s=26, sig=9):
    m = _ema(c, f) - _ema(c, s)
    si = _ema(m, sig)
    return m, si, m - si

def _cci(h, l, c, p=20):
    tp = (h + l + c) / 3
    ma = _sma(tp, p)
    md = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md + 1e-10)

def _mfi(h, l, c, v, p=14):
    tp = (h + l + c) / 3
    mf = tp * v
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(p).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(p).sum()
    return 100 - 100 / (1 + pos / (neg + 1e-10))

def _williams_r(h, l, c, p=14):
    upper = h.rolling(p).max()
    lower = l.rolling(p).min()
    return -100 * (upper - c) / (upper - lower + 1e-10)

def _tsi(c, long_p=25, short_p=13, sig_p=13):
    pc = c.diff()
    double_smooth = _ema(_ema(pc, long_p), short_p)
    double_smooth_abs = _ema(_ema(pc.abs(), long_p), short_p)
    tsi = 100 * double_smooth / (double_smooth_abs + 1e-10)
    signal = _ema(tsi, sig_p)
    return tsi, signal

def _uo(h, l, c, l1=7, l2=14, l3=28):
    prev_c = c.shift(1)
    true_low = pd.concat([l, prev_c], axis=1).min(axis=1)
    true_high = pd.concat([h, prev_c], axis=1).max(axis=1)
    bp = c - true_low
    tr = true_high - true_low
    avg1 = bp.rolling(l1).sum() / (tr.rolling(l1).sum() + 1e-10)
    avg2 = bp.rolling(l2).sum() / (tr.rolling(l2).sum() + 1e-10)
    avg3 = bp.rolling(l3).sum() / (tr.rolling(l3).sum() + 1e-10)
    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

def _alma(s, window=15, offset=0.85, sigma=6.0):
    m = offset * (window - 1)
    s_param = window / sigma
    weights = np.array([np.exp(-((i - m) ** 2) / (2 * s_param ** 2)) for i in range(window)])
    weights /= weights.sum()
    return s.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)

def _chop(h, l, c, period=28):
    atr1 = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_sum = atr1.rolling(period).sum()
    hh = h.rolling(period).max()
    ll = l.rolling(period).min()
    return 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)

def _ichimoku(h, l, conv=9, base=26, span2=52, disp=26):
    def donchian(period):
        return (h.rolling(period).max() + l.rolling(period).min()) / 2
    conversion = donchian(conv)
    base_line = donchian(base)
    lead1 = (conversion + base_line) / 2
    lead2 = donchian(span2)
    lead1_d = lead1.shift(disp)
    lead2_d = lead2.shift(disp)
    return conversion, base_line, lead1_d, lead2_d

def _parabolic_sar(h, l, start=0.02, inc=0.02, max_af=0.2):
    n = len(h)
    sar = np.full(n, np.nan)
    trend = np.ones(n)  # 1=up, -1=down
    ep = np.full(n, np.nan)
    af = np.full(n, start)
    h_v = h.values
    l_v = l.values
    sar[0] = l_v[0]
    ep[0] = h_v[0]
    for i in range(1, n):
        if np.isnan(h_v[i]) or np.isnan(l_v[i]):
            sar[i] = sar[i-1]
            trend[i] = trend[i-1]
            ep[i] = ep[i-1]
            af[i] = af[i-1]
            continue
        prev_sar = sar[i-1]
        prev_trend = trend[i-1]
        prev_ep = ep[i-1]
        prev_af = af[i-1]
        new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
        if prev_trend == 1:
            new_sar = min(new_sar, l_v[i-1], l_v[max(0, i-2)])
            if l_v[i] < new_sar:
                trend[i] = -1
                new_sar = prev_ep
                ep[i] = l_v[i]
                af[i] = start
            else:
                trend[i] = 1
                if h_v[i] > prev_ep:
                    ep[i] = h_v[i]
                    af[i] = min(prev_af + inc, max_af)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
        else:
            new_sar = max(new_sar, h_v[i-1], h_v[max(0, i-2)])
            if h_v[i] > new_sar:
                trend[i] = 1
                new_sar = prev_ep
                ep[i] = h_v[i]
                af[i] = start
            else:
                trend[i] = -1
                if l_v[i] < prev_ep:
                    ep[i] = l_v[i]
                    af[i] = min(prev_af + inc, max_af)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
        sar[i] = new_sar
    return pd.Series(sar, index=h.index), pd.Series(trend, index=h.index)

def _crossover(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))

def _crossunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


# ─── 1. AO_Eterna — Awesome Oscillator ────────────────────────────────────────

def gen_AO_Eterna(df, fast_p=5, slow_p=34):
    """
    Awesome Oscillator: zero cross + twin peaks + saucer patterns.
    LONG: zero cross up / bullish twin peak / bullish saucer
    SHORT: zero cross down / bearish twin peak / bearish saucer
    """
    h, l = df['high'], df['low']
    median = (h + l) / 2
    ao = _sma(median, fast_p) - _sma(median, slow_p)
    ao_green = ao > ao.shift(1)
    ao_red = ao < ao.shift(1)
    bull_zero_cross = _crossover(ao, pd.Series(0, index=ao.index))
    bear_zero_cross = _crossunder(ao, pd.Series(0, index=ao.index))
    twin_peak_bull = (ao < 0) & (ao > ao.shift(2)) & (ao.shift(2) < ao.shift(4))
    twin_peak_bear = (ao > 0) & (ao < ao.shift(2)) & (ao.shift(2) > ao.shift(4))
    saucer_bull = (ao > 0) & (ao < ao.shift(1)) & (ao.shift(1) < ao.shift(2)) & ao_green
    saucer_bear = (ao < 0) & (ao > ao.shift(1)) & (ao.shift(1) > ao.shift(2)) & ao_red
    consec_green = ao_green & ao_green.shift(1) & (ao > 0)
    consec_red = ao_red & ao_red.shift(1) & (ao < 0)
    buy = bull_zero_cross | twin_peak_bull | (saucer_bull & consec_green)
    sell = bear_zero_cross | twin_peak_bear | (saucer_bear & consec_red)
    sig = pd.Series(0, index=df.index)
    sig[buy] = 1
    sig[sell] = -1
    return sig

def space_AO_Eterna(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 3, 10),
        'slow_p': trial.suggest_int('slow_p', 20, 50),
    }


# ─── 2. CCI_Eterna — CCI Strategy ─────────────────────────────────────────────

def gen_CCI_Eterna(df, cci_len=20, ob=100, os=-100, extreme_ob=200, extreme_os=-200):
    """
    CCI: crossover OS/OB + extreme reversal signals.
    LONG: cross above OS or reverse from extreme_os
    SHORT: cross below OB or reverse from extreme_ob
    """
    h, l, c = df['high'], df['low'], df['close']
    cci = _cci(h, l, c, cci_len)
    bull_std = _crossover(cci, pd.Series(os, index=cci.index))
    bear_std = _crossunder(cci, pd.Series(ob, index=cci.index))
    extreme_bull = (cci.shift(1) < extreme_os) & (cci > extreme_os) & (cci < os)
    extreme_bear = (cci.shift(1) > extreme_ob) & (cci < extreme_ob) & (cci > ob)
    sig = pd.Series(0, index=df.index)
    sig[bull_std | extreme_bull] = 1
    sig[bear_std | extreme_bear] = -1
    return sig

def space_CCI_Eterna(trial):
    return {
        'cci_len': trial.suggest_int('cci_len', 10, 40),
        'ob': trial.suggest_int('ob', 80, 150),
        'os': trial.suggest_int('os', -150, -80),
        'extreme_ob': trial.suggest_int('extreme_ob', 150, 300),
        'extreme_os': trial.suggest_int('extreme_os', -300, -150),
    }


# ─── 3. StochRSI_Eterna — Stochastic RSI ──────────────────────────────────────

def gen_StochRSI_Eterna(df, rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3, upper=80, lower=20):
    """
    Stochastic RSI: K crosses D in oversold/overbought zones.
    LONG: K cross above D while K < lower
    SHORT: K cross below D while K > upper
    """
    c = df['close']
    rsi = _rsi(c, rsi_len)
    lo = rsi.rolling(stoch_len).min()
    hi = rsi.rolling(stoch_len).max()
    stoch_rsi = 100 * (rsi - lo) / (hi - lo + 1e-10)
    k = _sma(stoch_rsi, smooth_k)
    d = _sma(k, smooth_d)
    bull = _crossover(k, d) & (k < lower)
    bear = _crossunder(k, d) & (k > upper)
    sig = pd.Series(0, index=df.index)
    sig[bull] = 1
    sig[bear] = -1
    return sig

def space_StochRSI_Eterna(trial):
    return {
        'rsi_len': trial.suggest_int('rsi_len', 8, 21),
        'stoch_len': trial.suggest_int('stoch_len', 8, 21),
        'smooth_k': trial.suggest_int('smooth_k', 2, 5),
        'smooth_d': trial.suggest_int('smooth_d', 2, 5),
        'upper': trial.suggest_int('upper', 70, 90),
        'lower': trial.suggest_int('lower', 10, 30),
    }


# ─── 4. MomentumCombo_Eterna — Multi-indicator Confluence ─────────────────────

def gen_MomentumCombo_Eterna(df, rsi_len=14, macd_f=12, macd_s=26, macd_sig=9,
                               stoch_len=14, stoch_k=3, stoch_d=3, min_conf=2):
    """
    RSI + MACD + Stoch + MOM confluence.
    LONG: ≥ min_conf bullish indicators agree
    SHORT: ≥ min_conf bearish indicators agree
    """
    c, h, l = df['close'], df['high'], df['low']
    rsi = _rsi(c, rsi_len)
    rsi_bull = (rsi < 30) | (_crossover(rsi, pd.Series(30, index=rsi.index)) & (rsi < 50))
    rsi_bear = (rsi > 70) | (_crossunder(rsi, pd.Series(70, index=rsi.index)) & (rsi > 50))
    macd_line, sig_line, hist = _macd(c, macd_f, macd_s, macd_sig)
    macd_bull = _crossover(macd_line, sig_line) | ((macd_line > sig_line) & (hist > hist.shift(1)))
    macd_bear = _crossunder(macd_line, sig_line) | ((macd_line < sig_line) & (hist < hist.shift(1)))
    stoch_rsi = 100 * (rsi - rsi.rolling(stoch_len).min()) / (rsi.rolling(stoch_len).max() - rsi.rolling(stoch_len).min() + 1e-10)
    k_s = _sma(stoch_rsi, stoch_k)
    d_s = _sma(k_s, stoch_d)
    stoch_bull = (_crossover(k_s, d_s) & (k_s < 20)) | ((k_s > d_s) & (k_s < 30) & (k_s > k_s.shift(1)))
    stoch_bear = (_crossunder(k_s, d_s) & (k_s > 80)) | ((k_s < d_s) & (k_s > 70) & (k_s < k_s.shift(1)))
    mom = c.diff(10)
    mom_bull = (mom > 0) & (mom > mom.shift(1))
    mom_bear = (mom < 0) & (mom < mom.shift(1))
    bull_count = rsi_bull.astype(int) + macd_bull.astype(int) + stoch_bull.astype(int) + mom_bull.astype(int)
    bear_count = rsi_bear.astype(int) + macd_bear.astype(int) + stoch_bear.astype(int) + mom_bear.astype(int)
    sig = pd.Series(0, index=df.index)
    sig[bull_count >= min_conf] = 1
    sig[bear_count >= min_conf] = -1
    return sig

def space_MomentumCombo_Eterna(trial):
    return {
        'rsi_len': trial.suggest_int('rsi_len', 8, 21),
        'macd_f': trial.suggest_int('macd_f', 8, 16),
        'macd_s': trial.suggest_int('macd_s', 20, 34),
        'macd_sig': trial.suggest_int('macd_sig', 7, 13),
        'stoch_len': trial.suggest_int('stoch_len', 8, 21),
        'min_conf': trial.suggest_int('min_conf', 2, 3),
    }


# ─── 5. RSI_Div_Eterna — RSI Divergence ───────────────────────────────────────

def gen_RSI_Div_Eterna(df, rsi_len=14, ob=70, os=30, pivot_lb=5):
    """
    RSI Divergence: price vs RSI pivot comparison.
    LONG: price lower low but RSI higher low (bullish div) + RSI < OS
    SHORT: price higher high but RSI lower high (bearish div) + RSI > OB
    """
    h, l, c = df['high'], df['low'], df['close']
    rsi = _rsi(c, rsi_len)
    # Detect local lows/highs using rolling min/max
    price_lo = l.rolling(pivot_lb * 2 + 1, center=True).min()
    price_hi = h.rolling(pivot_lb * 2 + 1, center=True).max()
    rsi_lo = rsi.rolling(pivot_lb * 2 + 1, center=True).min()
    rsi_hi = rsi.rolling(pivot_lb * 2 + 1, center=True).max()
    is_price_low = (l == price_lo)
    is_price_high = (h == price_hi)
    is_rsi_low = (rsi == rsi_lo)
    is_rsi_high = (rsi == rsi_hi)
    prev_price_low = l.where(is_price_low).ffill().shift(1)
    prev_rsi_low = rsi.where(is_rsi_low).ffill().shift(1)
    prev_price_high = h.where(is_price_high).ffill().shift(1)
    prev_rsi_high = rsi.where(is_rsi_high).ffill().shift(1)
    bull_div = is_price_low & (l < prev_price_low) & (rsi > prev_rsi_low) & (rsi < os)
    bear_div = is_price_high & (h > prev_price_high) & (rsi < prev_rsi_high) & (rsi > ob)
    sig = pd.Series(0, index=df.index)
    sig[bull_div] = 1
    sig[bear_div] = -1
    return sig

def space_RSI_Div_Eterna(trial):
    return {
        'rsi_len': trial.suggest_int('rsi_len', 8, 21),
        'ob': trial.suggest_int('ob', 65, 80),
        'os': trial.suggest_int('os', 20, 35),
        'pivot_lb': trial.suggest_int('pivot_lb', 3, 10),
    }


# ─── 6. TSI_Eterna — True Strength Index ──────────────────────────────────────

def gen_TSI_Eterna(df, long_p=25, short_p=13, sig_p=13, ob=25, os=-25):
    """
    TSI: crossover signal line + zero line + OS/OB.
    LONG: TSI cross above signal while TSI < 0 OR bull in oversold
    SHORT: TSI cross below signal while TSI > 0 OR bear in overbought
    """
    c = df['close']
    tsi, signal = _tsi(c, long_p, short_p, sig_p)
    bull_cross = _crossover(tsi, signal) & (tsi < 0)
    bear_cross = _crossunder(tsi, signal) & (tsi > 0)
    bull_os = (tsi < os) & (tsi > tsi.shift(1)) & _crossover(tsi, signal)
    bear_ob = (tsi > ob) & (tsi < tsi.shift(1)) & _crossunder(tsi, signal)
    sig = pd.Series(0, index=df.index)
    sig[bull_cross | bull_os] = 1
    sig[bear_cross | bear_ob] = -1
    return sig

def space_TSI_Eterna(trial):
    return {
        'long_p': trial.suggest_int('long_p', 15, 35),
        'short_p': trial.suggest_int('short_p', 8, 18),
        'sig_p': trial.suggest_int('sig_p', 8, 18),
        'ob': trial.suggest_int('ob', 15, 40),
        'os': trial.suggest_int('os', -40, -15),
    }


# ─── 7. UO_Eterna — Ultimate Oscillator ───────────────────────────────────────

def gen_UO_Eterna(df, l1=7, l2=14, l3=28, ob=70, os=30):
    """
    Ultimate Oscillator: OS/OB reversal.
    LONG: UO cross above OS while UO < 50
    SHORT: UO cross below OB while UO > 50
    """
    h, l, c = df['high'], df['low'], df['close']
    uo = _uo(h, l, c, l1, l2, l3)
    bull = _crossover(uo, pd.Series(os, index=uo.index)) & (uo < 50)
    bear = _crossunder(uo, pd.Series(ob, index=uo.index)) & (uo > 50)
    sig = pd.Series(0, index=df.index)
    sig[bull] = 1
    sig[bear] = -1
    return sig

def space_UO_Eterna(trial):
    return {
        'l1': trial.suggest_int('l1', 4, 10),
        'l2': trial.suggest_int('l2', 10, 20),
        'l3': trial.suggest_int('l3', 22, 35),
        'ob': trial.suggest_int('ob', 65, 80),
        'os': trial.suggest_int('os', 20, 35),
    }


# ─── 8. WilliamsR_Eterna — Williams %R Reversal ───────────────────────────────

def gen_WilliamsR_Eterna(df, wr_len=14, ob=-20, os=-80):
    """
    Williams %R: strong reversal signals at OS/OB extremes.
    LONG: cross above OS + close > prev close
    SHORT: cross below OB + close < prev close
    """
    h, l, c = df['high'], df['low'], df['close']
    wr = _williams_r(h, l, c, wr_len)
    # cross above OS = bullish; cross below OB = bearish (+ momentum confirmation)
    strong_bull = _crossover(wr, pd.Series(os, index=wr.index)) & (c > c.shift(1))
    strong_bear = _crossunder(wr, pd.Series(ob, index=wr.index)) & (c < c.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[strong_bull] = 1
    sig[strong_bear] = -1
    return sig

def space_WilliamsR_Eterna(trial):
    return {
        'wr_len': trial.suggest_int('wr_len', 8, 21),
        'ob': trial.suggest_int('ob', -30, -10),
        'os': trial.suggest_int('os', -90, -70),
    }


# ─── 9. Chandelier_Eterna — Chandelier Exit ───────────────────────────────────

def gen_Chandelier_Eterna(df, atr_len=22, atr_mult=3.0, trend_ema=50):
    """
    Chandelier Exit with EMA trend filter.
    LONG: price crosses above chandelier long stop (uptrend)
    SHORT: price crosses below chandelier short stop (downtrend)
    """
    h, l, c = df['high'], df['low'], df['close']
    atr_v = _atr(h, l, c, atr_len)
    highest_h = h.rolling(atr_len).max()
    lowest_l = l.rolling(atr_len).min()
    chan_long = highest_h - atr_v * atr_mult
    chan_short = lowest_l + atr_v * atr_mult
    ema_f = _ema(c, trend_ema)
    bull = _crossover(c, chan_long) & (c > ema_f)
    bear = _crossunder(c, chan_short) & (c < ema_f)
    sig = pd.Series(0, index=df.index)
    sig[bull] = 1
    sig[bear] = -1
    return sig

def space_Chandelier_Eterna(trial):
    return {
        'atr_len': trial.suggest_int('atr_len', 14, 30),
        'atr_mult': trial.suggest_float('atr_mult', 2.0, 4.0, step=0.5),
        'trend_ema': trial.suggest_int('trend_ema', 30, 100),
    }


# ─── 10. SMA_5020_Eterna — SMA 50/200 Golden Cross ────────────────────────────

def gen_SMA_5020_Eterna(df, fast_p=50, slow_p=200):
    """
    SMA Golden/Death Cross.
    LONG: fast SMA crosses above slow SMA
    SHORT: fast SMA crosses below slow SMA
    """
    c = df['close']
    fast = _sma(c, fast_p)
    slow = _sma(c, slow_p)
    sig = pd.Series(0, index=df.index)
    sig[_crossover(fast, slow)] = 1
    sig[_crossunder(fast, slow)] = -1
    return sig

def space_SMA_5020_Eterna(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 20, 100),
        'slow_p': trial.suggest_int('slow_p', 100, 300),
    }


# ─── 11. ADX_Trend_Eterna — ADX Trend Strength ───────────────────────────────

def gen_ADX_Trend_Eterna(df, adx_len=14, adx_thresh=25, ema_len=50):
    """
    ADX: strong trend (ADX>thresh) + DI cross + EMA filter.
    LONG: +DI > -DI + ADX strong + close > EMA
    SHORT: -DI > +DI + ADX strong + close < EMA
    """
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    tr_s = tr.ewm(span=adx_len, adjust=False).mean()
    dmp = (h - h.shift(1)).clip(lower=0)
    dmn = (l.shift(1) - l).clip(lower=0)
    dmp_s = dmp.ewm(span=adx_len, adjust=False).mean()
    dmn_s = dmn.ewm(span=adx_len, adjust=False).mean()
    di_plus = 100 * dmp_s / (tr_s + 1e-10)
    di_minus = 100 * dmn_s / (tr_s + 1e-10)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
    adx = dx.ewm(span=adx_len, adjust=False).mean()
    ema_f = _ema(c, ema_len)
    strong = adx > adx_thresh
    bull = strong & (di_plus > di_minus) & (c > ema_f)
    bear = strong & (di_minus > di_plus) & (c < ema_f)
    bull_entry = _crossover(di_plus, di_minus) & strong & (c > ema_f)
    bear_entry = _crossunder(di_plus, di_minus) & strong & (c < ema_f)
    sig = pd.Series(0, index=df.index)
    sig[bull_entry] = 1
    sig[bear_entry] = -1
    return sig

def space_ADX_Trend_Eterna(trial):
    return {
        'adx_len': trial.suggest_int('adx_len', 10, 21),
        'adx_thresh': trial.suggest_int('adx_thresh', 20, 35),
        'ema_len': trial.suggest_int('ema_len', 30, 100),
    }


# ─── 12. Keltner_Eterna — Keltner Channel Trend ──────────────────────────────

def gen_Keltner_Eterna(df, ema_len=20, atr_len=10, atr_mult=2.0, trend_ema=50):
    """
    Keltner Channel: bounce off lower band in uptrend / reject upper band in downtrend.
    LONG: price crosses lower band from below + close > trend EMA
    SHORT: price crosses upper band from above + close < trend EMA
    """
    h, l, c = df['high'], df['low'], df['close']
    basis = _ema(c, ema_len)
    atr_v = _atr(h, l, c, atr_len)
    upper = basis + atr_v * atr_mult
    lower = basis - atr_v * atr_mult
    trend = _ema(c, trend_ema)
    long_cond = _crossover(c, lower) & (c > trend)
    short_cond = _crossunder(c, upper) & (c < trend)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Keltner_Eterna(trial):
    return {
        'ema_len': trial.suggest_int('ema_len', 10, 30),
        'atr_len': trial.suggest_int('atr_len', 7, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 3.0, step=0.5),
        'trend_ema': trial.suggest_int('trend_ema', 30, 100),
    }


# ─── 13. LinReg_Eterna — Linear Regression Channel ───────────────────────────

def gen_LinReg_Eterna(df, lr_len=100, dev_mult=2.0, trend_ema=50):
    """
    Linear Regression Channel: breakout from channel + EMA filter.
    LONG: close breaks above upper channel + close > trend EMA
    SHORT: close breaks below lower channel + close < trend EMA
    """
    c = df['close']
    # Compute rolling linear regression value
    def lr_val(x):
        n = len(x)
        t = np.arange(n, dtype=float)
        t_m = t.mean()
        x_m = x.mean()
        slope = ((t - t_m) * (x - x_m)).sum() / ((t - t_m) ** 2).sum()
        intercept = x_m - slope * t_m
        return slope * (n - 1) + intercept
    lr = c.rolling(lr_len).apply(lr_val, raw=True)
    # Channel width from rolling std
    std = c.rolling(lr_len).std()
    upper = lr + dev_mult * std
    lower = lr - dev_mult * std
    trend = _ema(c, trend_ema)
    long_cond = _crossover(c, upper) & (c > trend)
    short_cond = _crossunder(c, lower) & (c < trend)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_LinReg_Eterna(trial):
    return {
        'lr_len': trial.suggest_int('lr_len', 50, 150),
        'dev_mult': trial.suggest_float('dev_mult', 1.5, 3.0, step=0.5),
        'trend_ema': trial.suggest_int('trend_ema', 30, 100),
    }


# ─── 14. Donchian_Eterna — Donchian Channel Breakout ─────────────────────────

def gen_Donchian_Eterna(df, chan_len=20, confirm=1):
    """
    Donchian Channel Breakout.
    LONG: close breaks above upper band (highest high)
    SHORT: close breaks below lower band (lowest low)
    """
    h, l, c = df['high'], df['low'], df['close']
    upper = h.shift(confirm).rolling(chan_len).max()
    lower = l.shift(confirm).rolling(chan_len).min()
    middle = (upper + lower) / 2
    long_break = c > upper
    short_break = c < lower
    exit_long = _crossunder(c, middle)
    exit_short = _crossover(c, middle)
    # Use state tracking to avoid re-entry
    sig = pd.Series(0, index=df.index)
    sig[long_break] = 1
    sig[short_break] = -1
    return sig

def space_Donchian_Eterna(trial):
    return {
        'chan_len': trial.suggest_int('chan_len', 10, 40),
        'confirm': trial.suggest_int('confirm', 0, 3),
    }


# ─── 15. VWAP_Bounce_Eterna — VWAP Bounce ────────────────────────────────────

def gen_VWAP_Bounce_Eterna(df, trend_ema=50, vwap_dev=0.5):
    """
    VWAP Bounce: price returns to VWAP ± dev% then bounces in trend direction.
    LONG: close near VWAP from below + uptrend
    SHORT: close near VWAP from above + downtrend
    """
    c = df['close']
    v = df['volume']
    # Rolling VWAP (daily reset approximated with 50-bar rolling)
    vwap_p = (df['high'] + df['low'] + c) / 3
    vwap = (vwap_p * v).rolling(50).sum() / (v.rolling(50).sum() + 1e-10)
    dev = vwap * (vwap_dev / 100)
    trend = _ema(c, trend_ema)
    uptrend = c > trend
    downtrend = c < trend
    near_vwap_bull = (c >= vwap - dev) & (c <= vwap + dev) & (c > c.shift(1))
    near_vwap_bear = (c >= vwap - dev) & (c <= vwap + dev) & (c < c.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[near_vwap_bull & uptrend] = 1
    sig[near_vwap_bear & downtrend] = -1
    return sig

def space_VWAP_Bounce_Eterna(trial):
    return {
        'trend_ema': trial.suggest_int('trend_ema', 20, 100),
        'vwap_dev': trial.suggest_float('vwap_dev', 0.3, 1.5, step=0.1),
    }


# ─── 16. SAR_Eterna — Parabolic SAR ──────────────────────────────────────────

def gen_SAR_Eterna(df, sar_start=0.02, sar_inc=0.02, sar_max=0.2, trend_ema=50):
    """
    Parabolic SAR trend following with EMA filter.
    LONG: SAR turns bullish (trend=1) + close > EMA
    SHORT: SAR turns bearish (trend=-1) + close < EMA
    """
    h, l, c = df['high'], df['low'], df['close']
    sar_v, trend = _parabolic_sar(h, l, sar_start, sar_inc, sar_max)
    ema_f = _ema(c, trend_ema)
    bull = _crossover(trend, pd.Series(0, index=trend.index)) & (c > ema_f)
    bear = _crossunder(trend, pd.Series(0, index=trend.index)) & (c < ema_f)
    sig = pd.Series(0, index=df.index)
    sig[bull] = 1
    sig[bear] = -1
    return sig

def space_SAR_Eterna(trial):
    return {
        'sar_start': trial.suggest_float('sar_start', 0.01, 0.05, step=0.01),
        'sar_inc': trial.suggest_float('sar_inc', 0.01, 0.05, step=0.01),
        'sar_max': trial.suggest_float('sar_max', 0.1, 0.3, step=0.05),
        'trend_ema': trial.suggest_int('trend_ema', 20, 100),
    }


# ─── 17. EMA_Ribbon_Eterna — EMA Ribbon 9/21/55 ──────────────────────────────

def gen_EMA_Ribbon_Eterna(df, fast_p=9, mid_p=21, slow_p=55):
    """
    EMA Ribbon: all three EMAs aligned in direction.
    LONG: fast > mid > slow (bullish alignment)
    SHORT: fast < mid < slow (bearish alignment)
    """
    c = df['close']
    e1 = _ema(c, fast_p)
    e2 = _ema(c, mid_p)
    e3 = _ema(c, slow_p)
    bull_align = (e1 > e2) & (e2 > e3)
    bear_align = (e1 < e2) & (e2 < e3)
    # Enter on transition to aligned state
    bull = bull_align & ~bull_align.shift(1).fillna(False)
    bear = bear_align & ~bear_align.shift(1).fillna(False)
    sig = pd.Series(0, index=df.index)
    sig[bull] = 1
    sig[bear] = -1
    return sig

def space_EMA_Ribbon_Eterna(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 5, 15),
        'mid_p': trial.suggest_int('mid_p', 15, 30),
        'slow_p': trial.suggest_int('slow_p', 40, 100),
    }


# ─── 18. Ichimoku_Eterna — Ichimoku Cloud ────────────────────────────────────

def gen_Ichimoku_Eterna(df, conv_p=9, base_p=26, span2_p=52, disp=26):
    """
    Ichimoku Cloud: price above/below cloud + TK cross.
    LONG: price above cloud + bullish TK cross + bullish cloud
    SHORT: price below cloud + bearish TK cross + bearish cloud
    """
    h, l, c = df['high'], df['low'], df['close']
    conv, base, lead1_d, lead2_d = _ichimoku(h, l, conv_p, base_p, span2_p, disp)
    cloud_top = pd.concat([lead1_d, lead2_d], axis=1).max(axis=1)
    cloud_bot = pd.concat([lead1_d, lead2_d], axis=1).min(axis=1)
    price_above = c > cloud_top
    price_below = c < cloud_bot
    bull_tk = _crossover(conv, base)
    bear_tk = _crossunder(conv, base)
    bull_cloud = lead1_d > lead2_d
    bear_cloud = lead1_d < lead2_d
    sig = pd.Series(0, index=df.index)
    sig[price_above & bull_tk & bull_cloud] = 1
    sig[price_below & bear_tk & bear_cloud] = -1
    return sig

def space_Ichimoku_Eterna(trial):
    return {
        'conv_p': trial.suggest_int('conv_p', 7, 13),
        'base_p': trial.suggest_int('base_p', 20, 35),
        'span2_p': trial.suggest_int('span2_p', 44, 60),
        'disp': trial.suggest_int('disp', 20, 35),
    }


# ─── 19. MFI_Eterna — Money Flow Index ───────────────────────────────────────

def gen_MFI_Eterna(df, mfi_len=14, ob=80, os=20, div_lb=5):
    """
    MFI: OS/OB crossover + divergence.
    LONG: MFI cross above OS (with divergence bonus) + MFI < 50
    SHORT: MFI cross below OB (with divergence bonus) + MFI > 50
    """
    h, l, c, v = df['high'], df['low'], df['close'], df['volume']
    mfi = _mfi(h, l, c, v, mfi_len)
    bull_ob = _crossover(mfi, pd.Series(os, index=mfi.index))
    bear_os = _crossunder(mfi, pd.Series(ob, index=mfi.index))
    price_lo = l.rolling(div_lb).min()
    price_hi = h.rolling(div_lb).max()
    mfi_lo = mfi.rolling(div_lb).min()
    mfi_hi = mfi.rolling(div_lb).max()
    bull_div = (l == price_lo) & (mfi > mfi_lo) & (mfi < os)
    bear_div = (h == price_hi) & (mfi < mfi_hi) & (mfi > ob)
    buy = (bull_ob | bull_div) & (mfi < 50)
    sell = (bear_os | bear_div) & (mfi > 50)
    sig = pd.Series(0, index=df.index)
    sig[buy] = 1
    sig[sell] = -1
    return sig

def space_MFI_Eterna(trial):
    return {
        'mfi_len': trial.suggest_int('mfi_len', 8, 21),
        'ob': trial.suggest_int('ob', 70, 90),
        'os': trial.suggest_int('os', 10, 30),
        'div_lb': trial.suggest_int('div_lb', 3, 10),
    }


# ─── 20. Chaikin_Eterna — Chaikin Oscillator ─────────────────────────────────

def gen_Chaikin_Eterna(df, fast_p=3, slow_p=10, div_lb=5):
    """
    Chaikin Oscillator: zero-line crossover with momentum + divergence.
    LONG: Chaikin cross above 0 with momentum OR bullish divergence
    SHORT: Chaikin cross below 0 with momentum OR bearish divergence
    """
    h, l, c, v = df['high'], df['low'], df['close'], df['volume']
    mfm = ((c - l) - (h - c)) / (h - l + 1e-10)
    mfv = mfm * v
    ad = mfv.cumsum()
    chaikin = _ema(ad, fast_p) - _ema(ad, slow_p)
    bull_cross = _crossover(chaikin, pd.Series(0.0, index=chaikin.index))
    bear_cross = _crossunder(chaikin, pd.Series(0.0, index=chaikin.index))
    bull_mom = (chaikin > chaikin.shift(1)) & (chaikin.shift(1) > chaikin.shift(2))
    bear_mom = (chaikin < chaikin.shift(1)) & (chaikin.shift(1) < chaikin.shift(2))
    # Zero-cross + 3-bar momentum confirmation (no loose divergence)
    sig = pd.Series(0, index=df.index)
    sig[bull_cross & bull_mom] = 1
    sig[bear_cross & bear_mom] = -1
    return sig

def space_Chaikin_Eterna(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 2, 6),
        'slow_p': trial.suggest_int('slow_p', 8, 15),
        'div_lb': trial.suggest_int('div_lb', 3, 10),
    }


# ─── 21. HiLo_Tranquilo — Rodrigo Brito ──────────────────────────────────────

def gen_HiLo_Tranquilo(df, period=3, shift=1):
    """
    HiLo Tranquilo (brito.com.br):
    EMA-based high/low direction tracker + MACD confirmation.
    LONG: direction=1 + MACD >= signal + MACD rising
    SHORT: direction=-1 + MACD < signal + MACD falling
    """
    h, l, c = df['high'], df['low'], df['close']
    hi_ema = _ema(h.shift(shift), period)
    lo_ema = _ema(l.shift(shift), period)
    direction = np.ones(len(c))
    lo_v = lo_ema.values
    hi_v = hi_ema.values
    c_v = c.values
    for i in range(1, len(c_v)):
        if direction[i-1] == 1 and c_v[i] < lo_v[i]:
            direction[i] = -1
        elif direction[i-1] == -1 and c_v[i] > hi_v[i]:
            direction[i] = 1
        else:
            direction[i] = direction[i-1]
    dir_s = pd.Series(direction, index=c.index)
    macd_l, macd_sig, _ = _macd(c)
    macd_rising = macd_l > macd_l.shift(1)
    macd_falling = macd_l < macd_l.shift(1)
    bull = _crossover(dir_s, pd.Series(0, index=dir_s.index)) | ((dir_s == 1) & (macd_l >= macd_sig) & macd_rising & (dir_s != dir_s.shift(1)))
    bear = _crossunder(dir_s, pd.Series(0, index=dir_s.index)) | ((dir_s == -1) & (macd_l < macd_sig) & macd_falling & (dir_s != dir_s.shift(1)))
    sig = pd.Series(0, index=df.index)
    sig[_crossover(dir_s, pd.Series(0.5, index=dir_s.index)) & (macd_l >= macd_sig)] = 1
    sig[_crossunder(dir_s, pd.Series(0.5, index=dir_s.index)) & (macd_l < macd_sig)] = -1
    return sig

def space_HiLo_Tranquilo(trial):
    return {
        'period': trial.suggest_int('period', 2, 8),
        'shift': trial.suggest_int('shift', 1, 3),
    }


# ─── 22. Scalping_3EMA — Rodrigo Brito ───────────────────────────────────────

def gen_Scalping_3EMA(df, e1_p=50, e2_p=100, e3_p=150, rsi_len=14, macd_f=12, macd_s=26, macd_sig=9):
    """
    3 EMA trend alignment + MACD confirmation.
    LONG: EMA1 > EMA2 > EMA3 + MACD above signal + both rising
    SHORT: EMA1 < EMA2 < EMA3 + MACD below signal + both falling
    """
    c = df['close']
    e1 = _ema(c, e1_p)
    e2 = _ema(c, e2_p)
    e3 = _ema(c, e3_p)
    macd_l, macd_sig_l, _ = _macd(c, macd_f, macd_s, macd_sig)
    macd_hist = macd_l - macd_sig_l
    ema_bull = (e1 > e2) & (e2 > e3)
    ema_bear = (e1 < e2) & (e2 < e3)
    e1_rising = e1 > e1.shift(1)
    e1_falling = e1 < e1.shift(1)
    macd_rising = macd_hist > macd_hist.shift(1)
    macd_falling = macd_hist < macd_hist.shift(1)
    # Entry on first bar of alignment
    bull_new = ema_bull & ~ema_bull.shift(1).fillna(False)
    bear_new = ema_bear & ~ema_bear.shift(1).fillna(False)
    sig = pd.Series(0, index=df.index)
    sig[bull_new & (macd_l > macd_sig_l) & macd_rising & e1_rising] = 1
    sig[bear_new & (macd_l < macd_sig_l) & macd_falling & e1_falling] = -1
    return sig

def space_Scalping_3EMA(trial):
    return {
        'e1_p': trial.suggest_int('e1_p', 30, 80),
        'e2_p': trial.suggest_int('e2_p', 80, 150),
        'e3_p': trial.suggest_int('e3_p', 120, 200),
    }


# ─── 23. BIAS_Strategy — kat3samsin ──────────────────────────────────────────

def gen_BIAS_Strategy(df, alma_win=15, alma_offset=0.85, alma_sigma=6.0,
                      chop_p=28, stoch_k=14, stoch_d=3, stoch_smooth=3, lb_period=5):
    """
    BIAS: Stoch > RSI + RSI crossover CHOP + buy on Stoch crossover K/D.
    LONG: RSI crosses above CHOP while Stoch > RSI
    SHORT: RSI crosses below CHOP while Stoch < RSI (mirrored)
    """
    h, l, c = df['high'], df['low'], df['close']
    rsi = _rsi(c, 14)
    chop = _chop(h, l, c, chop_p)
    k, d = _stoch(h, l, c, stoch_k, stoch_d, stoch_smooth)
    alma = _alma(c, alma_win, alma_offset, alma_sigma)
    stoch_gt_rsi = k > rsi
    stoch_lt_rsi = k < rsi
    bars_since_rsi_cross_chop = _crossover(rsi, chop).astype(int).rolling(lb_period).sum() > 0
    bars_since_rsi_cross_chop_dn = _crossunder(rsi, chop).astype(int).rolling(lb_period).sum() > 0
    cond1 = stoch_gt_rsi & _crossover(rsi, chop)
    cond2 = stoch_gt_rsi & _crossover(k, d) & bars_since_rsi_cross_chop
    cond1_s = stoch_lt_rsi & _crossunder(rsi, chop)
    cond2_s = stoch_lt_rsi & _crossunder(k, d) & bars_since_rsi_cross_chop_dn
    sig = pd.Series(0, index=df.index)
    sig[cond1 | cond2] = 1
    sig[cond1_s | cond2_s] = -1
    return sig

def space_BIAS_Strategy(trial):
    return {
        'alma_win': trial.suggest_int('alma_win', 10, 25),
        'chop_p': trial.suggest_int('chop_p', 14, 50),
        'stoch_k': trial.suggest_int('stoch_k', 8, 21),
        'lb_period': trial.suggest_int('lb_period', 3, 10),
    }


# ─── 24. Overbought_Swing — kat3samsin ───────────────────────────────────────

def gen_Overbought_Swing(df, alma_win=15, alma_offset=0.85, alma_sigma=6.0,
                          stoch_k=14, stoch_d=3, stoch_smooth=3, sma50_p=50, sma200_p=200):
    """
    Overbought: RSI>=70 in strong uptrend + Stoch K > D, exit below ALMA.
    (LONG momentum strategy — SHORT mirrors for downtrend)
    """
    h, l, c = df['high'], df['low'], df['close']
    rsi = _rsi(c, 14)
    sma50 = _sma(c, sma50_p)
    sma200 = _sma(c, sma200_p)
    alma = _alma(c, alma_win, alma_offset, alma_sigma)
    k, d = _stoch(h, l, c, stoch_k, stoch_d, stoch_smooth)
    long_entry = (rsi >= 70) & (c > sma50) & (sma50 > sma200) & (k > d)
    short_entry = (rsi <= 30) & (c < sma50) & (sma50 < sma200) & (k < d)
    sig = pd.Series(0, index=df.index)
    sig[long_entry & ~long_entry.shift(1).fillna(False)] = 1
    sig[short_entry & ~short_entry.shift(1).fillna(False)] = -1
    return sig

def space_Overbought_Swing(trial):
    return {
        'alma_win': trial.suggest_int('alma_win', 10, 25),
        'stoch_k': trial.suggest_int('stoch_k', 8, 21),
        'sma50_p': trial.suggest_int('sma50_p', 30, 80),
        'sma200_p': trial.suggest_int('sma200_p', 150, 250),
    }


# ─── 25. Oversold_Bounce — kat3samsin ────────────────────────────────────────

def gen_Oversold_Bounce(df, alma_win=15, stoch_k=14, stoch_d=3, stoch_smooth=3, sma20_p=20):
    """
    Oversold: RSI<=30 + close within 15% below SMA20 + Stoch K > D.
    SHORT mirror: RSI>=70 + close within 15% above SMA20.
    """
    h, l, c = df['high'], df['low'], df['close']
    rsi = _rsi(c, 14)
    sma20 = _sma(c, sma20_p)
    k, d = _stoch(h, l, c, stoch_k, stoch_d, stoch_smooth)
    near_sma20_dn = (c >= sma20 * 0.85) & (c < sma20)
    near_sma20_up = (c <= sma20 * 1.15) & (c > sma20)
    long_entry = (rsi <= 30) & near_sma20_dn & (k > d)
    short_entry = (rsi >= 70) & near_sma20_up & (k < d)
    sig = pd.Series(0, index=df.index)
    sig[long_entry & ~long_entry.shift(1).fillna(False)] = 1
    sig[short_entry & ~short_entry.shift(1).fillna(False)] = -1
    return sig

def space_Oversold_Bounce(trial):
    return {
        'alma_win': trial.suggest_int('alma_win', 10, 25),
        'stoch_k': trial.suggest_int('stoch_k', 8, 21),
        'sma20_p': trial.suggest_int('sma20_p', 15, 30),
    }


# ─── 26. RSI_PowerZone — kat3samsin ──────────────────────────────────────────

def gen_RSI_PowerZone(df, rsi_len=4, entry_level=30, exit_level=55,
                       sma50_p=50, sma100_p=100, sma200_p=200):
    """
    RSI PowerZone: RSI(4) <= entry in confirmed uptrend.
    LONG: RSI(4) oversold + 50>100>200 SMA stacked
    SHORT: RSI(4) overbought + SMAs stacked bearish
    """
    c = df['close']
    rsi = _rsi(c, rsi_len)
    sma50 = _sma(c, sma50_p)
    sma100 = _sma(c, sma100_p)
    sma200 = _sma(c, sma200_p)
    uptrend = (c > sma50) & (sma50 > sma100) & (sma100 > sma200)
    downtrend = (c < sma50) & (sma50 < sma100) & (sma100 < sma200)
    long_entry = uptrend & (rsi <= entry_level)
    short_entry = downtrend & (rsi >= 100 - entry_level)
    sig = pd.Series(0, index=df.index)
    sig[long_entry & ~long_entry.shift(1).fillna(False)] = 1
    sig[short_entry & ~short_entry.shift(1).fillna(False)] = -1
    return sig

def space_RSI_PowerZone(trial):
    return {
        'rsi_len': trial.suggest_int('rsi_len', 2, 8),
        'entry_level': trial.suggest_int('entry_level', 20, 40),
        'exit_level': trial.suggest_int('exit_level', 50, 70),
        'sma50_p': trial.suggest_int('sma50_p', 30, 60),
        'sma200_p': trial.suggest_int('sma200_p', 150, 250),
    }


# ─── 27. Sandwich_Strategy — kat3samsin ──────────────────────────────────────

def gen_Sandwich_Strategy(df, chop_p=28, stoch_k=14, stoch_d=3, stoch_smooth=3,
                           alma_win=15, alma_offset=0.85, alma_sigma=6.0):
    """
    Sandwich: RSI > CHOP, Stoch K/D both between RSI and CHOP.
    LONG: sandwich condition met + Stoch rising
    SHORT: inverted sandwich + Stoch falling
    """
    h, l, c = df['high'], df['low'], df['close']
    rsi = _rsi(c, 14)
    chop = _chop(h, l, c, chop_p)
    k, d = _stoch(h, l, c, stoch_k, stoch_d, stoch_smooth)
    alma = _alma(c, alma_win, alma_offset, alma_sigma)
    stoch_k_sandwich = (k <= rsi) & (k >= chop)
    stoch_d_sandwich = (d <= rsi) & (d >= chop)
    stoch_k_sandwich_s = (k >= rsi) & (k <= chop + (chop - rsi + 1e-10).abs())
    stoch_d_sandwich_s = (d >= rsi) & (d <= chop + (chop - rsi + 1e-10).abs())
    stoch_up = k > d
    stoch_down = k < d
    long_cond = (rsi > chop) & stoch_k_sandwich & stoch_d_sandwich & stoch_up
    short_cond = (rsi < chop) & stoch_k_sandwich_s & stoch_d_sandwich_s & stoch_down
    sig = pd.Series(0, index=df.index)
    sig[long_cond & ~long_cond.shift(1).fillna(False)] = 1
    sig[short_cond & ~short_cond.shift(1).fillna(False)] = -1
    return sig

def space_Sandwich_Strategy(trial):
    return {
        'chop_p': trial.suggest_int('chop_p', 14, 50),
        'stoch_k': trial.suggest_int('stoch_k', 8, 21),
        'alma_win': trial.suggest_int('alma_win', 10, 25),
    }


# ─── 28. Undercut_Strategy — kat3samsin ──────────────────────────────────────

def gen_Undercut_Strategy(df, chop_p=28, stoch_k=14, stoch_d=3, stoch_smooth=3, lb_period=5):
    """
    Undercut: RSI stayed above CHOP for period, Stoch crosses above CHOP.
    LONG: RSI>CHOP for lb bars + Stoch K crosses CHOP upward + Stoch rising
    SHORT: RSI<CHOP for lb bars + Stoch K crosses CHOP downward + Stoch falling
    """
    h, l, c = df['high'], df['low'], df['close']
    rsi = _rsi(c, 14)
    chop = _chop(h, l, c, chop_p)
    k, d = _stoch(h, l, c, stoch_k, stoch_d, stoch_smooth)
    rsi_above_chop = (rsi > chop).rolling(lb_period).sum() >= lb_period
    rsi_below_chop = (rsi < chop).rolling(lb_period).sum() >= lb_period
    stoch_cross_chop_up = _crossover(k, chop)
    stoch_cross_chop_dn = _crossunder(k, chop)
    stoch_up = k > d
    stoch_down = k < d
    long_cond = rsi_above_chop & stoch_cross_chop_up & stoch_up & (rsi > chop)
    short_cond = rsi_below_chop & stoch_cross_chop_dn & stoch_down & (rsi < chop)
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

def space_Undercut_Strategy(trial):
    return {
        'chop_p': trial.suggest_int('chop_p', 14, 50),
        'stoch_k': trial.suggest_int('stoch_k', 8, 21),
        'lb_period': trial.suggest_int('lb_period', 3, 10),
    }


# ─── 29. Stella_BB_MeanRev — Bollinger Bands Mean Reversion (v6) ──────────────

def gen_Stella_BB_MeanRev(df, bb_len=20, bb_mult=2.0):
    """
    Stella's BB Mean Reversion: buy lower band touch, sell upper band touch.
    LONG: close <= lower band
    SHORT: close >= upper band
    """
    c = df['close']
    mid = _sma(c, bb_len)
    std = c.rolling(bb_len).std()
    upper = mid + bb_mult * std
    lower = mid - bb_mult * std
    sig = pd.Series(0, index=df.index)
    sig[c <= lower] = 1
    sig[c >= upper] = -1
    return sig

def space_Stella_BB_MeanRev(trial):
    return {
        'bb_len': trial.suggest_int('bb_len', 10, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0, step=0.5),
    }


# ─── 30. Stella_MACD_Cross — MACD Crossover (v6) ─────────────────────────────

def gen_Stella_MACD_Cross(df, fast_p=12, slow_p=26, sig_p=9, trend_ema_p=200, use_trend=False):
    """
    Stella's MACD Crossover with optional EMA trend filter.
    LONG: MACD crosses above signal (+ above trend EMA if use_trend)
    SHORT: MACD crosses below signal (+ below trend EMA if use_trend)
    """
    c = df['close']
    macd_l, macd_sig, _ = _macd(c, fast_p, slow_p, sig_p)
    trend_ema = _ema(c, trend_ema_p)
    bull = _crossover(macd_l, macd_sig)
    bear = _crossunder(macd_l, macd_sig)
    if use_trend:
        bull = bull & (c > trend_ema)
        bear = bear & (c < trend_ema)
    sig = pd.Series(0, index=df.index)
    sig[bull] = 1
    sig[bear] = -1
    return sig

def space_Stella_MACD_Cross(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 8, 16),
        'slow_p': trial.suggest_int('slow_p', 20, 34),
        'sig_p': trial.suggest_int('sig_p', 7, 13),
        'use_trend': trial.suggest_categorical('use_trend', [True, False]),
    }


# ─── 31. Stella_RSI_EMA — RSI + EMA Crossover (v6) ───────────────────────────

def gen_Stella_RSI_EMA(df, rsi_len=14, ob=70, os=30, ema_p=50, trend_ema_p=200):
    """
    Stella's RSI + EMA: RSI OS/OB + price/EMA cross + 200 EMA trend filter.
    LONG: RSI < OS + close > EMA + uptrend (close > trend_ema)
    SHORT: RSI > OB + close < EMA + downtrend (close < trend_ema)
    """
    c = df['close']
    rsi = _rsi(c, rsi_len)
    ema = _ema(c, ema_p)
    trend = _ema(c, trend_ema_p)
    uptrend = c > trend
    downtrend = c < trend
    rsi_bull = (rsi < os) & (rsi > rsi.shift(1))
    rsi_bear = (rsi > ob) & (rsi < rsi.shift(1))
    price_above_ema = _crossover(c, ema)
    price_below_ema = _crossunder(c, ema)
    sig = pd.Series(0, index=df.index)
    sig[(rsi_bull | price_above_ema) & uptrend] = 1
    sig[(rsi_bear | price_below_ema) & downtrend] = -1
    return sig

def space_Stella_RSI_EMA(trial):
    return {
        'rsi_len': trial.suggest_int('rsi_len', 8, 21),
        'ob': trial.suggest_int('ob', 65, 80),
        'os': trial.suggest_int('os', 20, 35),
        'ema_p': trial.suggest_int('ema_p', 20, 100),
        'trend_ema_p': trial.suggest_int('trend_ema_p', 100, 300),
    }


# ─── 32. Stella_SuperTrend — SuperTrend Basic (v6) ────────────────────────────

def gen_Stella_SuperTrend(df, atr_period=10, multiplier=3.0):
    """
    Stella's SuperTrend Basic: trend direction changes.
    LONG: price crosses above SuperTrend (direction turns up)
    SHORT: price crosses below SuperTrend (direction turns down)
    """
    h, l, c = df['high'], df['low'], df['close']
    atr_v = _atr(h, l, c, atr_period)
    hl2 = (h + l) / 2
    up_band = (hl2 - multiplier * atr_v).values
    dn_band = (hl2 + multiplier * atr_v).values
    c_v = c.values
    n = len(c_v)
    final_up = np.copy(up_band)
    final_dn = np.copy(dn_band)
    direction = np.ones(n)
    for i in range(1, n):
        if np.isnan(c_v[i]):
            direction[i] = direction[i-1]
            continue
        final_up[i] = up_band[i] if up_band[i] > final_up[i-1] or c_v[i-1] < final_up[i-1] else final_up[i-1]
        final_dn[i] = dn_band[i] if dn_band[i] < final_dn[i-1] or c_v[i-1] > final_dn[i-1] else final_dn[i-1]
        if direction[i-1] == 1:
            direction[i] = -1 if c_v[i] < final_up[i] else 1
        else:
            direction[i] = 1 if c_v[i] > final_dn[i] else -1
    dir_s = pd.Series(direction, index=c.index)
    bull = _crossover(dir_s, pd.Series(0, index=dir_s.index))
    bear = _crossunder(dir_s, pd.Series(0, index=dir_s.index))
    sig = pd.Series(0, index=df.index)
    sig[bull] = 1
    sig[bear] = -1
    return sig

def space_Stella_SuperTrend(trial):
    return {
        'atr_period': trial.suggest_int('atr_period', 7, 20),
        'multiplier': trial.suggest_float('multiplier', 2.0, 5.0, step=0.5),
    }


# ─── EXPORT ────────────────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'AO_Eterna':            {'gen': gen_AO_Eterna,            'space': space_AO_Eterna},
    'CCI_Eterna':           {'gen': gen_CCI_Eterna,           'space': space_CCI_Eterna},
    'StochRSI_Eterna':      {'gen': gen_StochRSI_Eterna,      'space': space_StochRSI_Eterna},
    'MomentumCombo_Eterna': {'gen': gen_MomentumCombo_Eterna, 'space': space_MomentumCombo_Eterna},
    'RSI_Div_Eterna':       {'gen': gen_RSI_Div_Eterna,       'space': space_RSI_Div_Eterna},
    'TSI_Eterna':           {'gen': gen_TSI_Eterna,           'space': space_TSI_Eterna},
    'UO_Eterna':            {'gen': gen_UO_Eterna,            'space': space_UO_Eterna},
    'WilliamsR_Eterna':     {'gen': gen_WilliamsR_Eterna,     'space': space_WilliamsR_Eterna},
    'Chandelier_Eterna':    {'gen': gen_Chandelier_Eterna,    'space': space_Chandelier_Eterna},
    'SMA_5020_Eterna':      {'gen': gen_SMA_5020_Eterna,      'space': space_SMA_5020_Eterna},
    'ADX_Trend_Eterna':     {'gen': gen_ADX_Trend_Eterna,     'space': space_ADX_Trend_Eterna},
    'Keltner_Eterna':       {'gen': gen_Keltner_Eterna,       'space': space_Keltner_Eterna},
    'LinReg_Eterna':        {'gen': gen_LinReg_Eterna,        'space': space_LinReg_Eterna},
    'Donchian_Eterna':      {'gen': gen_Donchian_Eterna,      'space': space_Donchian_Eterna},
    'VWAP_Bounce_Eterna':   {'gen': gen_VWAP_Bounce_Eterna,   'space': space_VWAP_Bounce_Eterna},
    'SAR_Eterna':           {'gen': gen_SAR_Eterna,           'space': space_SAR_Eterna},
    'EMA_Ribbon_Eterna':    {'gen': gen_EMA_Ribbon_Eterna,    'space': space_EMA_Ribbon_Eterna},
    'Ichimoku_Eterna':      {'gen': gen_Ichimoku_Eterna,      'space': space_Ichimoku_Eterna},
    'MFI_Eterna':           {'gen': gen_MFI_Eterna,           'space': space_MFI_Eterna},
    'Chaikin_Eterna':       {'gen': gen_Chaikin_Eterna,       'space': space_Chaikin_Eterna},
    'HiLo_Tranquilo':       {'gen': gen_HiLo_Tranquilo,       'space': space_HiLo_Tranquilo},
    'Scalping_3EMA':        {'gen': gen_Scalping_3EMA,        'space': space_Scalping_3EMA},
    'BIAS_Strategy':        {'gen': gen_BIAS_Strategy,        'space': space_BIAS_Strategy},
    'Overbought_Swing':     {'gen': gen_Overbought_Swing,     'space': space_Overbought_Swing},
    'Oversold_Bounce':      {'gen': gen_Oversold_Bounce,      'space': space_Oversold_Bounce},
    'RSI_PowerZone':        {'gen': gen_RSI_PowerZone,        'space': space_RSI_PowerZone},
    'Sandwich_Strategy':    {'gen': gen_Sandwich_Strategy,    'space': space_Sandwich_Strategy},
    'Undercut_Strategy':    {'gen': gen_Undercut_Strategy,    'space': space_Undercut_Strategy},
    # Stella Pine v6
    'Stella_BB_MeanRev':    {'gen': gen_Stella_BB_MeanRev,    'space': space_Stella_BB_MeanRev},
    'Stella_MACD_Cross':    {'gen': gen_Stella_MACD_Cross,    'space': space_Stella_MACD_Cross},
    'Stella_RSI_EMA':       {'gen': gen_Stella_RSI_EMA,       'space': space_Stella_RSI_EMA},
    'Stella_SuperTrend':    {'gen': gen_Stella_SuperTrend,    'space': space_Stella_SuperTrend},
}
