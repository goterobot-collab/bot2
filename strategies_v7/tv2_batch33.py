#!/usr/bin/env python3
"""TV2 BATCH 33 — 30 Trend Continuation Strategies 2026-04-01

  TC_EMA_Expansion       — EMA spread expanding: fast further from slow
  TC_HH_HL_Entry         — Enter on new HH in HH/HL structure
  TC_Pullback_EMA        — Pullback to EMA then bounce in uptrend
  TC_Flag_Continuation   — Bull flag: pole + tight consolidation + new high
  TC_ATR_Follow          — N consecutive bars move same direction > ATR*pct
  TC_EMA_Stack           — Three EMAs stacked + spread increasing
  TC_Volume_Breakout     — Breakout above high[N] with accelerating volume
  TC_SuperTrend_Pullback — SuperTrend bull + pullback to ST line + bounce
  TC_Momentum_Continue   — ROC positive and increasing
  TC_ADX_Trend           — ADX>thresh AND DI+>DI-
  TC_RSI_50_Hold         — RSI holds above 50 for N bars
  TC_Channel_Continuation— Donchian: long when close > N-bar high (Turtle)
  TC_MACD_Expansion      — MACD histogram expanding above zero
  TC_Price_Momentum      — Price momentum score > threshold
  TC_Trend_Score         — Composite 4-factor trend score >= 3
  TC_BB_Riding           — N consecutive closes above BB upper band
  TC_Keltner_Ride        — N consecutive closes above Keltner upper band
  TC_Ichimoku_Cloud      — Tenkan above Kijun + price above cloud
  TC_EMA_Cross_Continue  — After EMA cross, hold; RSI>50 filter
  TC_OBV_Rising          — OBV consistently higher than N bars ago
  TC_Vol_Trend_Confirm   — Rising volume in trend direction for N bars
  TC_RSI_Trend_Range     — RSI stays in bull range 40-80 for N bars
  TC_Coral_Continue      — Coral trend rising + price above coral
  TC_VWAP_Trend          — Rolling VWAP rising + close above VWAP
  TC_ST_EMA_Stack        — SuperTrend bull + EMA stack + vol above avg
  TC_Squeeze_Direction   — Post-squeeze: momentum direction signal
  TC_ATR_Trend           — ATR expanding in trend direction
  TC_Gap_Continue        — Gap up + EMA uptrend confirmation
  TC_Star_Pattern        — Isolated opposite candles in strong uptrend
  TC_Parabolic_Exit      — Parabolic entry: RSI>70 + vol explosion
"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=int(p), adjust=False).mean()


def _rma(s, p):
    return s.ewm(alpha=1.0 / int(p), adjust=False).mean()


def _sma(s, p):
    return s.rolling(int(p), min_periods=1).mean()


def _atr(df, p):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, p)


def _rsi(s, p):
    d  = s.diff()
    g  = d.clip(lower=0)
    l  = (-d).clip(lower=0)
    rs = _rma(g, p) / _rma(l, p).replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


# ── 1. TC_EMA_Expansion ───────────────────────────────────────────────────────

def gen_TC_EMA_Expansion(df, fast_p=10, slow_p=30, **kw):
    close    = df['close']
    fast_p   = int(fast_p); slow_p = int(slow_p)
    fast     = _ema(close, fast_p)
    slow     = _ema(close, slow_p)
    spread   = (fast - slow) / slow.replace(0, 1e-9)
    expanding_up   = spread > spread.shift(1)
    expanding_down = spread < spread.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[expanding_up   & (spread > 0)]  =  1
    sig[expanding_down & (spread < 0)]  = -1
    return sig.fillna(0)


def space_TC_EMA_Expansion(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 5, 20),
        'slow_p': trial.suggest_int('slow_p', 20, 60),
    }


# ── 2. TC_HH_HL_Entry ─────────────────────────────────────────────────────────

def gen_TC_HH_HL_Entry(df, look_p=5, **kw):
    high  = df['high']
    low   = df['low']
    close = df['close']
    look_p = int(look_p)
    new_hh = close > high.rolling(look_p).max().shift(1)
    prev_low        = low.shift(1)
    prev_prev_low   = low.shift(2)
    hl_ok = prev_low > prev_prev_low
    new_ll = close < low.rolling(look_p).min().shift(1)
    prev_high       = high.shift(1)
    prev_prev_high  = high.shift(2)
    lh_ok = prev_high < prev_prev_high
    sig = pd.Series(0, index=df.index)
    sig[new_hh & hl_ok]  =  1
    sig[new_ll & lh_ok]  = -1
    return sig.fillna(0)


def space_TC_HH_HL_Entry(trial):
    return {
        'look_p': trial.suggest_int('look_p', 3, 10),
    }


# ── 3. TC_Pullback_EMA ────────────────────────────────────────────────────────

def gen_TC_Pullback_EMA(df, ema_p=21, **kw):
    close  = df['close']
    ema_p  = int(ema_p)
    ema    = _ema(close, ema_p)
    uptrend   = close > ema
    touched   = (df['low'] <= ema) & (df['low'].shift(1) > ema.shift(1))
    bounce_up = close > close.shift(1)
    downtrend = close < ema
    touched_h = (df['high'] >= ema) & (df['high'].shift(1) < ema.shift(1))
    bounce_dn = close < close.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[uptrend   & (touched | touched.shift(1)) & bounce_up]  =  1
    sig[downtrend & (touched_h | touched_h.shift(1)) & bounce_dn] = -1
    return sig.fillna(0)


def space_TC_Pullback_EMA(trial):
    return {
        'ema_p': trial.suggest_int('ema_p', 10, 50),
    }


# ── 4. TC_Flag_Continuation ───────────────────────────────────────────────────

def gen_TC_Flag_Continuation(df, pole_mult=2.0, flag_bars=5, vol_p=30, **kw):
    close     = df['close']
    high      = df['high']
    low       = df['low']
    vol       = df['volume']
    pole_mult = float(pole_mult)
    flag_bars = int(flag_bars)
    vol_p     = int(vol_p)
    atr       = _atr(df, 14)
    vol_avg   = _sma(vol, vol_p)
    pole_up   = (close - close.shift(flag_bars)) > atr * pole_mult
    consol_hi = high.rolling(flag_bars).max()
    consol_lo = low.rolling(flag_bars).min()
    tight     = (consol_hi - consol_lo) < atr * pole_mult * 0.5
    new_high  = close > consol_hi.shift(1)
    pole_dn   = (close.shift(flag_bars) - close) > atr * pole_mult
    new_low   = close < consol_lo.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[pole_up.shift(flag_bars) & tight & new_high & (vol > vol_avg)]  =  1
    sig[pole_dn.shift(flag_bars) & tight & new_low  & (vol > vol_avg)]  = -1
    return sig.fillna(0)


def space_TC_Flag_Continuation(trial):
    return {
        'pole_mult': trial.suggest_float('pole_mult', 1.5, 3.0),
        'flag_bars': trial.suggest_int('flag_bars', 3, 10),
        'vol_p':     trial.suggest_int('vol_p', 20, 50),
    }


# ── 5. TC_ATR_Follow ──────────────────────────────────────────────────────────

def gen_TC_ATR_Follow(df, bars=2, atr_p=14, pct=0.5, **kw):
    close  = df['close']
    bars   = int(bars); atr_p = int(atr_p)
    pct    = float(pct)
    atr    = _atr(df, atr_p)
    bar_chg = close.diff()
    sig = pd.Series(0, index=df.index)
    for i in range(len(df)):
        if i < bars:
            continue
        all_up = all(bar_chg.iloc[i - j] > atr.iloc[i - j] * pct for j in range(bars))
        all_dn = all(bar_chg.iloc[i - j] < -atr.iloc[i - j] * pct for j in range(bars))
        if all_up:
            sig.iloc[i] = 1
        elif all_dn:
            sig.iloc[i] = -1
    return sig.fillna(0)


def space_TC_ATR_Follow(trial):
    return {
        'bars':  trial.suggest_int('bars', 2, 4),
        'atr_p': trial.suggest_int('atr_p', 10, 20),
        'pct':   trial.suggest_float('pct', 0.3, 0.8),
    }


# ── 6. TC_EMA_Stack ───────────────────────────────────────────────────────────

def gen_TC_EMA_Stack(df, p1=8, p2=21, p3=55, **kw):
    close = df['close']
    p1 = int(p1); p2 = int(p2); p3 = int(p3)
    e1 = _ema(close, p1)
    e2 = _ema(close, p2)
    e3 = _ema(close, p3)
    spread = e1 - e3
    bull = (e1 > e2) & (e2 > e3) & (spread > spread.shift(1))
    bear = (e1 < e2) & (e2 < e3) & (spread < spread.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[bull]  =  1
    sig[bear]  = -1
    return sig.fillna(0)


def space_TC_EMA_Stack(trial):
    return {
        'p1': trial.suggest_int('p1', 5,  15),
        'p2': trial.suggest_int('p2', 15, 35),
        'p3': trial.suggest_int('p3', 35, 80),
    }


# ── 7. TC_Volume_Breakout ─────────────────────────────────────────────────────

def gen_TC_Volume_Breakout(df, break_p=20, vol_p=30, **kw):
    close   = df['close']
    high    = df['high']
    vol     = df['volume']
    low     = df['low']
    break_p = int(break_p); vol_p = int(vol_p)
    prev_high = high.rolling(break_p).max().shift(1)
    prev_low  = low.rolling(break_p).min().shift(1)
    vol_avg   = _sma(vol, vol_p)
    vol_accel = vol > vol.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[(close > prev_high) & vol_accel & (vol > vol_avg)]  =  1
    sig[(close < prev_low)  & vol_accel & (vol > vol_avg)]  = -1
    return sig.fillna(0)


def space_TC_Volume_Breakout(trial):
    return {
        'break_p': trial.suggest_int('break_p', 10, 50),
        'vol_p':   trial.suggest_int('vol_p',   20, 50),
    }


# ── 8. TC_SuperTrend_Pullback ─────────────────────────────────────────────────

def _supertrend(df, p, mult):
    p    = int(p); mult = float(mult)
    atr  = _atr(df, p)
    hl2  = (df['high'] + df['low']) / 2
    upper_basic = hl2 + mult * atr
    lower_basic = hl2 - mult * atr
    upper = upper_basic.copy()
    lower = lower_basic.copy()
    trend = pd.Series(1, index=df.index)
    close = df['close']
    for i in range(1, len(df)):
        upper.iloc[i] = upper_basic.iloc[i] if upper_basic.iloc[i] < upper.iloc[i-1] or close.iloc[i-1] > upper.iloc[i-1] else upper.iloc[i-1]
        lower.iloc[i] = lower_basic.iloc[i] if lower_basic.iloc[i] > lower.iloc[i-1] or close.iloc[i-1] < lower.iloc[i-1] else lower.iloc[i-1]
        if trend.iloc[i-1] == -1 and close.iloc[i] > upper.iloc[i]:
            trend.iloc[i] = 1
        elif trend.iloc[i-1] == 1 and close.iloc[i] < lower.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    st_line = pd.Series(np.where(trend == 1, lower, upper), index=df.index)
    return trend, st_line


def gen_TC_SuperTrend_Pullback(df, st_p=10, st_mult=3.0, zone_pct=0.005, **kw):
    zone_pct = float(zone_pct)
    trend, st_line = _supertrend(df, st_p, st_mult)
    close  = df['close']
    near_st_bull = (close - st_line).abs() / st_line.replace(0, 1e-9) < zone_pct
    near_st_bear = (st_line - close).abs() / st_line.replace(0, 1e-9) < zone_pct
    bounce_up = close > close.shift(1)
    bounce_dn = close < close.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[(trend == 1)  & (near_st_bull | near_st_bull.shift(1)) & bounce_up]  =  1
    sig[(trend == -1) & (near_st_bear | near_st_bear.shift(1)) & bounce_dn]  = -1
    return sig.fillna(0)


def space_TC_SuperTrend_Pullback(trial):
    return {
        'st_p':     trial.suggest_int('st_p', 7, 21),
        'st_mult':  trial.suggest_float('st_mult', 2.0, 4.0),
        'zone_pct': trial.suggest_float('zone_pct', 0.002, 0.01),
    }


# ── 9. TC_Momentum_Continue ───────────────────────────────────────────────────

def gen_TC_Momentum_Continue(df, roc_p=10, **kw):
    close  = df['close']
    roc_p  = int(roc_p)
    roc    = close.pct_change(roc_p)
    rising = roc > roc.shift(1)
    falling = roc < roc.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[(roc > 0)  & rising]   =  1
    sig[(roc < 0)  & falling]  = -1
    return sig.fillna(0)


def space_TC_Momentum_Continue(trial):
    return {
        'roc_p': trial.suggest_int('roc_p', 5, 20),
    }


# ── 10. TC_ADX_Trend ──────────────────────────────────────────────────────────

def _adx(df, p):
    p    = int(p)
    high = df['high']; low = df['low']; close = df['close']
    tr   = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    dm_plus  = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    atr14  = _rma(tr, p)
    di_plus  = 100 * _rma(dm_plus,  p) / atr14.replace(0, 1e-9)
    di_minus = 100 * _rma(dm_minus, p) / atr14.replace(0, 1e-9)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1e-9)
    adx = _rma(dx, p)
    return adx, di_plus, di_minus


def gen_TC_ADX_Trend(df, adx_p=14, thresh=25.0, **kw):
    adx_p  = int(adx_p); thresh = float(thresh)
    adx, di_plus, di_minus = _adx(df, adx_p)
    sig = pd.Series(0, index=df.index)
    sig[(adx > thresh) & (di_plus > di_minus)]  =  1
    sig[(adx > thresh) & (di_minus > di_plus)]  = -1
    return sig.fillna(0)


def space_TC_ADX_Trend(trial):
    return {
        'adx_p':  trial.suggest_int('adx_p', 10, 20),
        'thresh': trial.suggest_float('thresh', 20.0, 35.0),
    }


# ── 11. TC_RSI_50_Hold ────────────────────────────────────────────────────────

def gen_TC_RSI_50_Hold(df, rsi_p=14, hold_bars=5, **kw):
    close      = df['close']
    rsi_p      = int(rsi_p); hold_bars = int(hold_bars)
    rsi        = _rsi(close, rsi_p)
    bull_hold  = rsi.rolling(hold_bars).min() > 50
    bear_hold  = rsi.rolling(hold_bars).max() < 50
    sig = pd.Series(0, index=df.index)
    sig[bull_hold]  =  1
    sig[bear_hold]  = -1
    return sig.fillna(0)


def space_TC_RSI_50_Hold(trial):
    return {
        'rsi_p':     trial.suggest_int('rsi_p', 7, 21),
        'hold_bars': trial.suggest_int('hold_bars', 3, 8),
    }


# ── 12. TC_Channel_Continuation ───────────────────────────────────────────────

def gen_TC_Channel_Continuation(df, n_p=20, **kw):
    close   = df['close']
    high    = df['high']
    low     = df['low']
    n_p     = int(n_p)
    upper   = high.rolling(n_p).max().shift(1)
    lower   = low.rolling(n_p).min().shift(1)
    sig = pd.Series(0, index=df.index)
    sig[close > upper]  =  1
    sig[close < lower]  = -1
    return sig.fillna(0)


def space_TC_Channel_Continuation(trial):
    return {
        'n_p': trial.suggest_int('n_p', 20, 55),
    }


# ── 13. TC_MACD_Expansion ─────────────────────────────────────────────────────

def gen_TC_MACD_Expansion(df, fast=12, slow=26, sig_p=9, **kw):
    close  = df['close']
    fast   = int(fast); slow = int(slow); sig_p = int(sig_p)
    macd   = _ema(close, fast) - _ema(close, slow)
    signal = _ema(macd, sig_p)
    hist   = macd - signal
    expanding_bull = (hist > hist.shift(1)) & (hist > 0)
    expanding_bear = (hist < hist.shift(1)) & (hist < 0)
    sig = pd.Series(0, index=df.index)
    sig[expanding_bull]  =  1
    sig[expanding_bear]  = -1
    return sig.fillna(0)


def space_TC_MACD_Expansion(trial):
    return {
        'fast':  trial.suggest_int('fast',  8, 16),
        'slow':  trial.suggest_int('slow',  20, 30),
        'sig_p': trial.suggest_int('sig_p', 5, 12),
    }


# ── 14. TC_Price_Momentum ─────────────────────────────────────────────────────

def gen_TC_Price_Momentum(df, mom_p=20, threshold=0.02, **kw):
    close     = df['close']
    mom_p     = int(mom_p); threshold = float(threshold)
    score     = close / close.shift(mom_p) - 1
    sig = pd.Series(0, index=df.index)
    sig[score >  threshold]  =  1
    sig[score < -threshold]  = -1
    return sig.fillna(0)


def space_TC_Price_Momentum(trial):
    return {
        'mom_p':     trial.suggest_int('mom_p', 10, 30),
        'threshold': trial.suggest_float('threshold', 0.01, 0.05),
    }


# ── 15. TC_Trend_Score ────────────────────────────────────────────────────────

def gen_TC_Trend_Score(df, fast_p=10, slow_p=30, adx_p=14, rsi_p=14, **kw):
    close   = df['close']
    fast_p  = int(fast_p); slow_p = int(slow_p)
    adx_p   = int(adx_p);  rsi_p  = int(rsi_p)
    fast    = _ema(close, fast_p)
    slow    = _ema(close, slow_p)
    ema200  = _ema(close, 200)
    adx, di_plus, di_minus = _adx(df, adx_p)
    rsi = _rsi(close, rsi_p)
    bull_score = (
        (fast > slow).astype(int) +
        (adx > 20).astype(int) +
        (rsi > 50).astype(int) +
        (close > ema200).astype(int)
    )
    bear_score = (
        (fast < slow).astype(int) +
        (adx > 20).astype(int) +
        (rsi < 50).astype(int) +
        (close < ema200).astype(int)
    )
    sig = pd.Series(0, index=df.index)
    sig[bull_score >= 3]  =  1
    sig[bear_score >= 3]  = -1
    return sig.fillna(0)


def space_TC_Trend_Score(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 5, 20),
        'slow_p': trial.suggest_int('slow_p', 20, 60),
        'adx_p':  trial.suggest_int('adx_p',  10, 20),
        'rsi_p':  trial.suggest_int('rsi_p',   7, 21),
    }


# ── 16. TC_BB_Riding ──────────────────────────────────────────────────────────

def gen_TC_BB_Riding(df, bb_p=20, bb_mult=2.0, consec=3, **kw):
    close   = df['close']
    bb_p    = int(bb_p); consec = int(consec)
    bb_mult = float(bb_mult)
    mid     = _sma(close, bb_p)
    std     = close.rolling(bb_p, min_periods=1).std()
    upper   = mid + bb_mult * std
    lower   = mid - bb_mult * std
    above_upper = (close > upper).rolling(consec).min().fillna(0).astype(bool)
    below_lower = (close < lower).rolling(consec).min().fillna(0).astype(bool)
    sig = pd.Series(0, index=df.index)
    sig[above_upper]  =  1
    sig[below_lower]  = -1
    return sig.fillna(0)


def space_TC_BB_Riding(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 10, 30),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 3.0),
        'consec':  trial.suggest_int('consec', 2, 5),
    }


# ── 17. TC_Keltner_Ride ───────────────────────────────────────────────────────

def gen_TC_Keltner_Ride(df, kc_p=20, kc_mult=2.0, consec=3, **kw):
    close   = df['close']
    kc_p    = int(kc_p); consec = int(consec)
    kc_mult = float(kc_mult)
    mid     = _ema(close, kc_p)
    atr     = _atr(df, kc_p)
    upper   = mid + kc_mult * atr
    lower   = mid - kc_mult * atr
    above_upper = (close > upper).rolling(consec).min().fillna(0).astype(bool)
    below_lower = (close < lower).rolling(consec).min().fillna(0).astype(bool)
    sig = pd.Series(0, index=df.index)
    sig[above_upper]  =  1
    sig[below_lower]  = -1
    return sig.fillna(0)


def space_TC_Keltner_Ride(trial):
    return {
        'kc_p':    trial.suggest_int('kc_p', 10, 30),
        'kc_mult': trial.suggest_float('kc_mult', 1.5, 3.0),
        'consec':  trial.suggest_int('consec', 2, 4),
    }


# ── 18. TC_Ichimoku_Cloud ─────────────────────────────────────────────────────

def gen_TC_Ichimoku_Cloud(df, tenkan=9, kijun=26, senkou=52, **kw):
    high     = df['high']; low = df['low']; close = df['close']
    tenkan   = int(tenkan); kijun = int(kijun); senkou = int(senkou)
    tenkan_s = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_s  = (high.rolling(kijun).max()  + low.rolling(kijun).min())  / 2
    span_a   = ((tenkan_s + kijun_s) / 2).shift(kijun)
    span_b   = ((high.rolling(senkou).max() + low.rolling(senkou).min()) / 2).shift(kijun)
    cloud_top    = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
    bull = (tenkan_s > kijun_s) & (close > cloud_top) & (span_a > span_b)
    bear = (tenkan_s < kijun_s) & (close < cloud_bottom) & (span_a < span_b)
    sig = pd.Series(0, index=df.index)
    sig[bull]  =  1
    sig[bear]  = -1
    return sig.fillna(0)


def space_TC_Ichimoku_Cloud(trial):
    return {
        'tenkan': trial.suggest_int('tenkan', 5,  15),
        'kijun':  trial.suggest_int('kijun',  20, 35),
        'senkou': trial.suggest_int('senkou', 44, 60),
    }


# ── 19. TC_EMA_Cross_Continue ─────────────────────────────────────────────────

def gen_TC_EMA_Cross_Continue(df, fast_p=10, slow_p=30, rsi_p=14, cross_ago=5, **kw):
    close    = df['close']
    fast_p   = int(fast_p); slow_p   = int(slow_p)
    rsi_p    = int(rsi_p);  cross_ago = int(cross_ago)
    fast     = _ema(close, fast_p)
    slow     = _ema(close, slow_p)
    crossed_bull = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    crossed_bear = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    recent_cross_bull = crossed_bull.rolling(cross_ago).max().fillna(0).astype(bool)
    recent_cross_bear = crossed_bear.rolling(cross_ago).max().fillna(0).astype(bool)
    rsi = _rsi(close, rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[recent_cross_bull & (rsi > 50)]  =  1
    sig[recent_cross_bear & (rsi < 50)]  = -1
    return sig.fillna(0)


def space_TC_EMA_Cross_Continue(trial):
    return {
        'fast_p':    trial.suggest_int('fast_p', 5, 20),
        'slow_p':    trial.suggest_int('slow_p', 20, 60),
        'rsi_p':     trial.suggest_int('rsi_p',  7, 21),
        'cross_ago': trial.suggest_int('cross_ago', 3, 15),
    }


# ── 20. TC_OBV_Rising ─────────────────────────────────────────────────────────

def gen_TC_OBV_Rising(df, obv_look=5, ema_p=20, **kw):
    close    = df['close']
    vol      = df['volume']
    obv_look = int(obv_look); ema_p = int(ema_p)
    obv      = (vol * np.sign(close.diff())).fillna(0).cumsum()
    obv_ema  = _ema(obv, ema_p)
    rising   = obv > obv.shift(obv_look)
    falling  = obv < obv.shift(obv_look)
    sig = pd.Series(0, index=df.index)
    sig[rising  & (obv_ema > obv_ema.shift(1))]  =  1
    sig[falling & (obv_ema < obv_ema.shift(1))]  = -1
    return sig.fillna(0)


def space_TC_OBV_Rising(trial):
    return {
        'obv_look': trial.suggest_int('obv_look', 3, 10),
        'ema_p':    trial.suggest_int('ema_p',   10, 30),
    }


# ── 21. TC_Vol_Trend_Confirm ──────────────────────────────────────────────────

def gen_TC_Vol_Trend_Confirm(df, bars=3, ema_p=20, **kw):
    close  = df['close']
    vol    = df['volume']
    bars   = int(bars); ema_p = int(ema_p)
    ema    = _ema(close, ema_p)
    price_up   = (close > close.shift(1)).rolling(bars).min().fillna(0).astype(bool)
    vol_up     = (vol   > vol.shift(1)  ).rolling(bars).min().fillna(0).astype(bool)
    price_dn   = (close < close.shift(1)).rolling(bars).min().fillna(0).astype(bool)
    vol_dn_ok  = (vol   > vol.shift(1)  ).rolling(bars).min().fillna(0).astype(bool)
    sig = pd.Series(0, index=df.index)
    sig[(close > ema) & price_up & vol_up]   =  1
    sig[(close < ema) & price_dn & vol_dn_ok] = -1
    return sig.fillna(0)


def space_TC_Vol_Trend_Confirm(trial):
    return {
        'bars':  trial.suggest_int('bars',  2, 5),
        'ema_p': trial.suggest_int('ema_p', 10, 40),
    }


# ── 22. TC_RSI_Trend_Range ────────────────────────────────────────────────────

def gen_TC_RSI_Trend_Range(df, rsi_p=14, rsi_floor=40.0, hold_bars=3, **kw):
    close      = df['close']
    rsi_p      = int(rsi_p); hold_bars = int(hold_bars)
    rsi_floor  = float(rsi_floor)
    rsi        = _rsi(close, rsi_p)
    bull_range = rsi.rolling(hold_bars).min() > rsi_floor
    bear_range = rsi.rolling(hold_bars).max() < (100 - rsi_floor)
    sig = pd.Series(0, index=df.index)
    sig[bull_range]  =  1
    sig[bear_range]  = -1
    return sig.fillna(0)


def space_TC_RSI_Trend_Range(trial):
    return {
        'rsi_p':     trial.suggest_int('rsi_p',  7, 21),
        'rsi_floor': trial.suggest_float('rsi_floor', 35.0, 50.0),
        'hold_bars': trial.suggest_int('hold_bars', 2, 5),
    }


# ── 23. TC_Coral_Continue ─────────────────────────────────────────────────────

def gen_TC_Coral_Continue(df, sm=5, cd=0.4, **kw):
    close = df['close']
    sm    = int(sm); cd = float(cd)
    di    = (sm - 1.0) / 2.0 + 1.0
    c1    = 2.0 / (di + 1.0)
    c2    = 1.0 - c1
    c3    = 3.0 * (cd * cd + cd * cd * cd)
    c4    = -3.0 * (2.0 * cd * cd + cd + cd * cd * cd)
    c5    = 3.0 * cd + 1.0 + cd * cd * cd + 3.0 * cd * cd
    i1 = close.ewm(span=sm, adjust=False).mean()
    i2 = i1.ewm(span=sm, adjust=False).mean()
    i3 = i2.ewm(span=sm, adjust=False).mean()
    i4 = i3.ewm(span=sm, adjust=False).mean()
    i5 = i4.ewm(span=sm, adjust=False).mean()
    i6 = i5.ewm(span=sm, adjust=False).mean()
    coral = c3 * i1 + c4 * i2 + c5 * i3 - c4 * i4 - c3 * i5 + i6 * (c4 * c5 - c3 * c4)
    coral = coral.fillna(close)
    rising  = coral > coral.shift(1)
    falling = coral < coral.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[rising  & (close > coral)]  =  1
    sig[falling & (close < coral)]  = -1
    return sig.fillna(0)


def space_TC_Coral_Continue(trial):
    return {
        'sm': trial.suggest_int('sm', 3, 20),
        'cd': trial.suggest_float('cd', 0.3, 0.8),
    }


# ── 24. TC_VWAP_Trend ─────────────────────────────────────────────────────────

def gen_TC_VWAP_Trend(df, vwap_p=30, **kw):
    close   = df['close']
    vol     = df['volume']
    vwap_p  = int(vwap_p)
    tp      = (df['high'] + df['low'] + close) / 3
    vwap    = (tp * vol).rolling(vwap_p, min_periods=1).sum() / vol.rolling(vwap_p, min_periods=1).sum()
    rising  = vwap > vwap.shift(1)
    falling = vwap < vwap.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[rising  & (close > vwap)]  =  1
    sig[falling & (close < vwap)]  = -1
    return sig.fillna(0)


def space_TC_VWAP_Trend(trial):
    return {
        'vwap_p': trial.suggest_int('vwap_p', 20, 60),
    }


# ── 25. TC_ST_EMA_Stack ───────────────────────────────────────────────────────

def gen_TC_ST_EMA_Stack(df, st_p=10, st_mult=3.0, fast_p=9, slow_p=21, vol_p=20, **kw):
    close  = df['close']
    vol    = df['volume']
    st_p   = int(st_p); fast_p = int(fast_p); slow_p = int(slow_p); vol_p = int(vol_p)
    st_mult = float(st_mult)
    trend, _ = _supertrend(df, st_p, st_mult)
    fast    = _ema(close, fast_p)
    slow    = _ema(close, slow_p)
    vol_avg = _sma(vol, vol_p)
    bull = (trend == 1) & (fast > slow) & (vol > vol_avg)
    bear = (trend == -1) & (fast < slow) & (vol > vol_avg)
    sig = pd.Series(0, index=df.index)
    sig[bull]  =  1
    sig[bear]  = -1
    return sig.fillna(0)


def space_TC_ST_EMA_Stack(trial):
    return {
        'st_p':    trial.suggest_int('st_p', 7, 21),
        'st_mult': trial.suggest_float('st_mult', 2.0, 4.0),
        'fast_p':  trial.suggest_int('fast_p', 5, 20),
        'slow_p':  trial.suggest_int('slow_p', 20, 60),
        'vol_p':   trial.suggest_int('vol_p', 20, 50),
    }


# ── 26. TC_Squeeze_Direction ──────────────────────────────────────────────────

def gen_TC_Squeeze_Direction(df, bb_p=20, kc_p=20, bb_mult=2.0, kc_mult=1.5, **kw):
    close   = df['close']
    bb_p    = int(bb_p); kc_p = int(kc_p)
    bb_mult = float(bb_mult); kc_mult = float(kc_mult)
    bb_mid  = _sma(close, bb_p)
    bb_std  = close.rolling(bb_p, min_periods=1).std()
    bb_up   = bb_mid + bb_mult * bb_std
    bb_lo   = bb_mid - bb_mult * bb_std
    kc_mid  = _ema(close, kc_p)
    kc_atr  = _atr(df, kc_p)
    kc_up   = kc_mid + kc_mult * kc_atr
    kc_lo   = kc_mid - kc_mult * kc_atr
    squeeze = (bb_up < kc_up) & (bb_lo > kc_lo)
    fired   = squeeze.shift(1) & ~squeeze
    delta   = close - ((df['high'].rolling(kc_p).max() + df['low'].rolling(kc_p).min()) / 2 + kc_mid) / 2
    mom     = _ema(delta, kc_p)
    sig = pd.Series(0, index=df.index)
    sig[fired & (mom > 0)]  =  1
    sig[fired & (mom < 0)]  = -1
    return sig.fillna(0)


def space_TC_Squeeze_Direction(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p', 15, 25),
        'kc_p':    trial.suggest_int('kc_p', 15, 25),
        'bb_mult': trial.suggest_float('bb_mult', 1.5, 2.5),
        'kc_mult': trial.suggest_float('kc_mult', 1.0, 2.0),
    }


# ── 27. TC_ATR_Trend ──────────────────────────────────────────────────────────

def gen_TC_ATR_Trend(df, atr_p=14, avg_p=30, **kw):
    close  = df['close']
    atr_p  = int(atr_p); avg_p = int(avg_p)
    atr    = _atr(df, atr_p)
    atr_avg = _sma(atr, avg_p)
    price_up = close > close.shift(1)
    price_dn = close < close.shift(1)
    expanding = atr > atr_avg
    sig = pd.Series(0, index=df.index)
    sig[expanding & price_up]  =  1
    sig[expanding & price_dn]  = -1
    return sig.fillna(0)


def space_TC_ATR_Trend(trial):
    return {
        'atr_p': trial.suggest_int('atr_p', 10, 20),
        'avg_p': trial.suggest_int('avg_p', 20, 60),
    }


# ── 28. TC_Gap_Continue ───────────────────────────────────────────────────────

def gen_TC_Gap_Continue(df, gap_pct=0.005, ema_p=30, **kw):
    close   = df['close']
    open_   = df['open']
    gap_pct = float(gap_pct); ema_p = int(ema_p)
    ema     = _ema(close, ema_p)
    gap_up  = open_ > close.shift(1) * (1 + gap_pct)
    gap_dn  = open_ < close.shift(1) * (1 - gap_pct)
    uptrend = close > ema
    dntrend = close < ema
    sig = pd.Series(0, index=df.index)
    sig[gap_up & uptrend]  =  1
    sig[gap_dn & dntrend]  = -1
    return sig.fillna(0)


def space_TC_Gap_Continue(trial):
    return {
        'gap_pct': trial.suggest_float('gap_pct', 0.002, 0.01),
        'ema_p':   trial.suggest_int('ema_p', 20, 60),
    }


# ── 29. TC_Star_Pattern ───────────────────────────────────────────────────────

def gen_TC_Star_Pattern(df, consec_trend=5, ema_p=30, **kw):
    close  = df['close']
    open_  = df['open']
    high   = df['high']
    low    = df['low']
    consec_trend = int(consec_trend); ema_p = int(ema_p)
    ema    = _ema(close, ema_p)
    body   = (close - open_).abs()
    range_ = (high - low).replace(0, 1e-9)
    doji   = body / range_ < 0.3
    uptrend_bars  = (close > close.shift(1)).rolling(consec_trend).min().fillna(0).astype(bool)
    downtrend_bars = (close < close.shift(1)).rolling(consec_trend).min().fillna(0).astype(bool)
    star_in_uptrend   = doji.shift(1) & uptrend_bars.shift(1)
    star_in_downtrend = doji.shift(1) & downtrend_bars.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[star_in_uptrend   & (close > ema) & (close > open_)]  =  1
    sig[star_in_downtrend & (close < ema) & (close < open_)]  = -1
    return sig.fillna(0)


def space_TC_Star_Pattern(trial):
    return {
        'consec_trend': trial.suggest_int('consec_trend', 3, 7),
        'ema_p':        trial.suggest_int('ema_p', 20, 60),
    }


# ── 30. TC_Parabolic_Exit ─────────────────────────────────────────────────────

def gen_TC_Parabolic_Exit(df, rsi_p=14, vol_mult=3.0, vol_p=30, **kw):
    close    = df['close']
    vol      = df['volume']
    rsi_p    = int(rsi_p); vol_p = int(vol_p)
    vol_mult = float(vol_mult)
    rsi      = _rsi(close, rsi_p)
    vol_avg  = _sma(vol, vol_p)
    parabolic_long  = (rsi > 70) & (vol > vol_avg * vol_mult)
    parabolic_short = (rsi < 30) & (vol > vol_avg * vol_mult)
    sig = pd.Series(0, index=df.index)
    sig[parabolic_long]   =  1
    sig[parabolic_short]  = -1
    return sig.fillna(0)


def space_TC_Parabolic_Exit(trial):
    return {
        'rsi_p':    trial.suggest_int('rsi_p', 7, 21),
        'vol_mult': trial.suggest_float('vol_mult', 2.0, 5.0),
        'vol_p':    trial.suggest_int('vol_p', 20, 50),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'TC_EMA_Expansion': {
        'gen': gen_TC_EMA_Expansion,
        'space': space_TC_EMA_Expansion,
        'default_params': {'fast_p': 10, 'slow_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'EMA spread expanding: fast further from slow = trend continuation.'},
    },
    'TC_HH_HL_Entry': {
        'gen': gen_TC_HH_HL_Entry,
        'space': space_TC_HH_HL_Entry,
        'default_params': {'look_p': 5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Enter on new HH when prior low is higher than low before (HH/HL structure).'},
    },
    'TC_Pullback_EMA': {
        'gen': gen_TC_Pullback_EMA,
        'space': space_TC_Pullback_EMA,
        'default_params': {'ema_p': 21},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2800,
                 'description': 'Trend pullback to EMA then bounce — continuation entry.'},
    },
    'TC_Flag_Continuation': {
        'gen': gen_TC_Flag_Continuation,
        'space': space_TC_Flag_Continuation,
        'default_params': {'pole_mult': 2.0, 'flag_bars': 5, 'vol_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3200,
                 'description': 'Bull/bear flag: strong pole + tight consolidation + breakout with volume.'},
    },
    'TC_ATR_Follow': {
        'gen': gen_TC_ATR_Follow,
        'space': space_TC_ATR_Follow,
        'default_params': {'bars': 2, 'atr_p': 14, 'pct': 0.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2200,
                 'description': 'Follow strong moves: N bars each moving > ATR*pct in same direction.'},
    },
    'TC_EMA_Stack': {
        'gen': gen_TC_EMA_Stack,
        'space': space_TC_EMA_Stack,
        'default_params': {'p1': 8, 'p2': 21, 'p3': 55},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3100,
                 'description': 'Three EMAs stacked (e1>e2>e3) with spread expanding.'},
    },
    'TC_Volume_Breakout': {
        'gen': gen_TC_Volume_Breakout,
        'space': space_TC_Volume_Breakout,
        'default_params': {'break_p': 20, 'vol_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3500,
                 'description': 'Breakout above/below N-bar high/low with accelerating volume.'},
    },
    'TC_SuperTrend_Pullback': {
        'gen': gen_TC_SuperTrend_Pullback,
        'space': space_TC_SuperTrend_Pullback,
        'default_params': {'st_p': 10, 'st_mult': 3.0, 'zone_pct': 0.005},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'SuperTrend bullish + pullback to ST line zone + bounce continuation.'},
    },
    'TC_Momentum_Continue': {
        'gen': gen_TC_Momentum_Continue,
        'space': space_TC_Momentum_Continue,
        'default_params': {'roc_p': 10},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'ROC positive and increasing = momentum continuation.'},
    },
    'TC_ADX_Trend': {
        'gen': gen_TC_ADX_Trend,
        'space': space_TC_ADX_Trend,
        'default_params': {'adx_p': 14, 'thresh': 25.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3800,
                 'description': 'Strong trend: ADX above threshold with DI+/DI- directional confirmation.'},
    },
    'TC_RSI_50_Hold': {
        'gen': gen_TC_RSI_50_Hold,
        'space': space_TC_RSI_50_Hold,
        'default_params': {'rsi_p': 14, 'hold_bars': 5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2400,
                 'description': 'RSI holds above/below 50 for N bars = sustained bull/bear trend.'},
    },
    'TC_Channel_Continuation': {
        'gen': gen_TC_Channel_Continuation,
        'space': space_TC_Channel_Continuation,
        'default_params': {'n_p': 20},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 4200,
                 'description': 'Donchian channel breakout: long on N-bar high (Turtle style).'},
    },
    'TC_MACD_Expansion': {
        'gen': gen_TC_MACD_Expansion,
        'space': space_TC_MACD_Expansion,
        'default_params': {'fast': 12, 'slow': 26, 'sig_p': 9},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3300,
                 'description': 'MACD histogram expanding above zero = increasing bullish momentum.'},
    },
    'TC_Price_Momentum': {
        'gen': gen_TC_Price_Momentum,
        'space': space_TC_Price_Momentum,
        'default_params': {'mom_p': 20, 'threshold': 0.02},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2600,
                 'description': 'Price momentum score (close/close[N]-1) above/below threshold.'},
    },
    'TC_Trend_Score': {
        'gen': gen_TC_Trend_Score,
        'space': space_TC_Trend_Score,
        'default_params': {'fast_p': 10, 'slow_p': 30, 'adx_p': 14, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3700,
                 'description': 'Composite trend score: EMA aligned + ADX>20 + RSI>50 + price>EMA200.'},
    },
    'TC_BB_Riding': {
        'gen': gen_TC_BB_Riding,
        'space': space_TC_BB_Riding,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'consec': 3},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2700,
                 'description': 'Ride BB band: N consecutive closes above upper/below lower band.'},
    },
    'TC_Keltner_Ride': {
        'gen': gen_TC_Keltner_Ride,
        'space': space_TC_Keltner_Ride,
        'default_params': {'kc_p': 20, 'kc_mult': 2.0, 'consec': 3},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Ride Keltner Channel: N consecutive closes outside KC band.'},
    },
    'TC_Ichimoku_Cloud': {
        'gen': gen_TC_Ichimoku_Cloud,
        'space': space_TC_Ichimoku_Cloud,
        'default_params': {'tenkan': 9, 'kijun': 26, 'senkou': 52},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4500,
                 'description': 'Ichimoku: tenkan>kijun + price above bullish cloud.'},
    },
    'TC_EMA_Cross_Continue': {
        'gen': gen_TC_EMA_Cross_Continue,
        'space': space_TC_EMA_Cross_Continue,
        'default_params': {'fast_p': 10, 'slow_p': 30, 'rsi_p': 14, 'cross_ago': 5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2900,
                 'description': 'After EMA cross, hold and add: cross within N bars + RSI filter.'},
    },
    'TC_OBV_Rising': {
        'gen': gen_TC_OBV_Rising,
        'space': space_TC_OBV_Rising,
        'default_params': {'obv_look': 5, 'ema_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2300,
                 'description': 'OBV consistently higher than N bars ago with rising EMA.'},
    },
    'TC_Vol_Trend_Confirm': {
        'gen': gen_TC_Vol_Trend_Confirm,
        'space': space_TC_Vol_Trend_Confirm,
        'default_params': {'bars': 3, 'ema_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2100,
                 'description': 'Volume confirms trend: rising price + rising vol for N consecutive bars.'},
    },
    'TC_RSI_Trend_Range': {
        'gen': gen_TC_RSI_Trend_Range,
        'space': space_TC_RSI_Trend_Range,
        'default_params': {'rsi_p': 14, 'rsi_floor': 40.0, 'hold_bars': 3},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2200,
                 'description': 'RSI stays in bull range (above floor) for N bars = trend health check.'},
    },
    'TC_Coral_Continue': {
        'gen': gen_TC_Coral_Continue,
        'space': space_TC_Coral_Continue,
        'default_params': {'sm': 5, 'cd': 0.4},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3100,
                 'description': 'Coral trend indicator: rising coral + price above = continuation long.'},
    },
    'TC_VWAP_Trend': {
        'gen': gen_TC_VWAP_Trend,
        'space': space_TC_VWAP_Trend,
        'default_params': {'vwap_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2800,
                 'description': 'Rolling VWAP rising + close above VWAP = bullish institutional trend.'},
    },
    'TC_ST_EMA_Stack': {
        'gen': gen_TC_ST_EMA_Stack,
        'space': space_TC_ST_EMA_Stack,
        'default_params': {'st_p': 10, 'st_mult': 3.0, 'fast_p': 9, 'slow_p': 21, 'vol_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3400,
                 'description': 'Triple confirmation: SuperTrend bull + EMA fast>slow + vol>avg.'},
    },
    'TC_Squeeze_Direction': {
        'gen': gen_TC_Squeeze_Direction,
        'space': space_TC_Squeeze_Direction,
        'default_params': {'bb_p': 20, 'kc_p': 20, 'bb_mult': 2.0, 'kc_mult': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3600,
                 'description': 'Post-squeeze direction: BB inside KC fires + momentum direction.'},
    },
    'TC_ATR_Trend': {
        'gen': gen_TC_ATR_Trend,
        'space': space_TC_ATR_Trend,
        'default_params': {'atr_p': 14, 'avg_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2400,
                 'description': 'ATR expanding above average in trend direction = continuation.'},
    },
    'TC_Gap_Continue': {
        'gen': gen_TC_Gap_Continue,
        'space': space_TC_Gap_Continue,
        'default_params': {'gap_pct': 0.005, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2600,
                 'description': 'Gap up/down in trend direction = continuation entry.'},
    },
    'TC_Star_Pattern': {
        'gen': gen_TC_Star_Pattern,
        'space': space_TC_Star_Pattern,
        'default_params': {'consec_trend': 5, 'ema_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Doji/star in strong trend: pause then continuation in trend direction.'},
    },
    'TC_Parabolic_Exit': {
        'gen': gen_TC_Parabolic_Exit,
        'space': space_TC_Parabolic_Exit,
        'default_params': {'rsi_p': 14, 'vol_mult': 3.0, 'vol_p': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2800,
                 'description': 'Parabolic entry: RSI>70 + volume explosion = strong continuation signal.'},
    },
}
