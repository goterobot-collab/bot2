#!/usr/bin/env python3
"""TV2 BATCH 28 — 30 estrategias Crypto-Specific 2026-04-01

  Crypto_Pump_Detector   — Volume spike + ATR expansion pump detect (v5, ~4000L)
  Altcoin_Season         — RSI>65 + rising vol + EMA alignment momentum (v5, ~3000L)
  Crypto_Breakout        — 24-bar breakout with volume confirmation (v4, ~5000L)
  Crypto_Dip_Buy         — EMA uptrend + RSI oversold dip + vol spike (v5, ~4000L)
  Crypto_Scalp_ATR       — ATR contraction then expansion scalp (v4, ~3000L)
  CVDD_Strategy          — OBV divergence bull/bear signals (v5, ~2500L)
  Crypto_Volume_Price    — VPT crosses above EMA(VPT) (v5, ~3000L)
  Crypto_VPVR            — Simplified volume-at-price support/resistance (v5, ~2500L)
  Crypto_Absorption      — High-vol small-body absorption at lows/highs (v5, ~2000L)
  Crypto_Volume_Trend    — Rising volume in trend direction (v4, ~2500L)
  Crypto_Grid_MR         — EMA ± N*ATR grid mean reversion (v5, ~2500L)
  Crypto_BB_MR           — BB lower touch with ATR volatility filter (v4, ~3000L)
  Crypto_Zscore          — Z-score mean reversion entry (v5, ~3000L)
  Crypto_Reversion_RSI   — RSI + BB + ATR triple confirmation MR (v5, ~2500L)
  BTC_Corr_Long          — Regime filter (long EMA) + EMA cross + RSI (v5, ~3000L)
  Alt_Season_Momentum    — Price above 3 EMAs simultaneously (v5, ~2500L)
  Crypto_Trend_Regime    — 200 EMA regime + fast/slow EMA cross (v4, ~2500L)
  Crypto_FOMO            — ROC acceleration + volume FOMO entry (v5, ~2000L)
  Crypto_Accumulation    — Wyckoff: tight range + vol dry-up + expansion (v5, ~2500L)
  Crypto_Distribution    — High vol + small body near BB upper = short (v5, ~2000L)
  Crypto_Liquidity_Run   — Spike below recent low then close above (v5, ~2500L)
  Adaptive_Crypto_EMA    — EMA period adapts to ATR volatility (v5, ~2000L)
  Crypto_ATR_Bands       — EMA ± ATR*mult bounce and breakout (v5, ~2500L)
  Crypto_Momentum_Score  — Composite ROC + vol-change + RSI score (v5, ~2000L)
  High_Vol_Breakout      — ATR-adjusted breakout for volatile assets (v4, ~3000L)
  Small_Cap_Momentum     — Strong RSI + vol surge + ATR increasing (v5, ~2000L)
  Parabolic_Entry        — ROC accelerating 3 bars + vol exploding (v5, ~2500L)
  Capitulation_Buy       — Large red candle + high vol + extreme RSI (v5, ~3000L)
  Relief_Rally           — First green after N red candles + vol surge (v5, ~2000L)
  Crypto_Confluence      — EMA + SuperTrend + RSI + Volume all aligned (v5, ~3000L)
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


# ── 1. Crypto_Pump_Detector ───────────────────────────────────────────────────

def gen_Crypto_Pump_Detector(df, vol_p=20, vol_mult=3.0, atr_p=14, atr_mult=2.0, **kw):
    close  = df['close']
    vol    = df['volume']
    vol_p  = int(vol_p)
    atr_p  = int(atr_p)
    vol_avg = _sma(vol, vol_p)
    atr     = _atr(df, atr_p)
    atr_avg = _sma(atr, vol_p)
    sig = pd.Series(0, index=df.index)
    pump = (vol > vol_avg * vol_mult) & (atr > atr_avg * atr_mult)
    sig[pump & (close > close.shift(1))]  =  1
    sig[pump & (close < close.shift(1))]  = -1
    return sig.fillna(0)


def space_Crypto_Pump_Detector(trial):
    return {
        'vol_p':    trial.suggest_int('vol_p',    20, 50),
        'vol_mult': trial.suggest_float('vol_mult', 2.0, 5.0),
        'atr_p':    trial.suggest_int('atr_p',    10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 3.0),
    }


# ── 2. Altcoin_Season ─────────────────────────────────────────────────────────

def gen_Altcoin_Season(df, rsi_p=14, vol_p=20, fast_p=9, slow_p=21, **kw):
    close    = df['close']
    vol      = df['volume']
    rsi_p    = int(rsi_p);  vol_p = int(vol_p)
    fast_p   = int(fast_p); slow_p = int(slow_p)
    rsi      = _rsi(close, rsi_p)
    vol_avg  = _sma(vol, vol_p)
    ema_fast = _ema(close, fast_p)
    ema_slow = _ema(close, slow_p)
    sig = pd.Series(0, index=df.index)
    bull = (rsi > 65) & (vol > vol_avg * 2.0) & (ema_fast > ema_slow)
    bear = (rsi < 35) & (vol > vol_avg * 2.0) & (ema_fast < ema_slow)
    sig[bull] =  1
    sig[bear] = -1
    return sig.fillna(0)


def space_Altcoin_Season(trial):
    return {
        'rsi_p':  trial.suggest_int('rsi_p',  7,  21),
        'vol_p':  trial.suggest_int('vol_p',  20, 50),
        'fast_p': trial.suggest_int('fast_p', 5,  20),
        'slow_p': trial.suggest_int('slow_p', 20, 60),
    }


# ── 3. Crypto_Breakout ────────────────────────────────────────────────────────

def gen_Crypto_Breakout(df, break_p=20, vol_p=20, vol_mult=2.0, **kw):
    close    = df['close']
    vol      = df['volume']
    break_p  = int(break_p); vol_p = int(vol_p)
    highest  = close.rolling(break_p, min_periods=1).max().shift(1)
    lowest   = close.rolling(break_p, min_periods=1).min().shift(1)
    vol_avg  = _sma(vol, vol_p)
    sig = pd.Series(0, index=df.index)
    vol_ok = vol > vol_avg * vol_mult
    sig[(close > highest) & vol_ok] =  1
    sig[(close < lowest)  & vol_ok] = -1
    return sig.fillna(0)


def space_Crypto_Breakout(trial):
    return {
        'break_p':  trial.suggest_int('break_p',  10, 50),
        'vol_p':    trial.suggest_int('vol_p',     20, 50),
        'vol_mult': trial.suggest_float('vol_mult', 1.5, 4.0),
    }


# ── 4. Crypto_Dip_Buy ─────────────────────────────────────────────────────────

def gen_Crypto_Dip_Buy(df, fast_p=9, slow_p=21, rsi_p=14, rsi_os=30, **kw):
    close    = df['close']
    vol      = df['volume']
    fast_p   = int(fast_p); slow_p = int(slow_p); rsi_p = int(rsi_p)
    ema_fast = _ema(close, fast_p)
    ema_slow = _ema(close, slow_p)
    rsi      = _rsi(close, rsi_p)
    vol_avg  = _sma(vol, 20)
    sig = pd.Series(0, index=df.index)
    uptrend  = ema_fast > ema_slow
    downtrend = ema_fast < ema_slow
    dip_long  = uptrend   & (rsi < rsi_os)    & (vol > vol_avg)
    dip_short = downtrend & (rsi > 100 - rsi_os) & (vol > vol_avg)
    sig[dip_long]  =  1
    sig[dip_short] = -1
    return sig.fillna(0)


def space_Crypto_Dip_Buy(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 5,  20),
        'slow_p': trial.suggest_int('slow_p', 20, 60),
        'rsi_p':  trial.suggest_int('rsi_p',  7,  21),
        'rsi_os': trial.suggest_int('rsi_os', 25, 40),
    }


# ── 5. Crypto_Scalp_ATR ───────────────────────────────────────────────────────

def gen_Crypto_Scalp_ATR(df, atr_p=14, atr_mult=1.8, ema_p=20, contraction_p=10, **kw):
    close         = df['close']
    atr_p         = int(atr_p); ema_p = int(ema_p); contraction_p = int(contraction_p)
    atr           = _atr(df, atr_p)
    atr_contracted = _sma(atr, contraction_p)
    ema           = _ema(close, ema_p)
    expanding     = atr > atr_contracted.shift(1) * atr_mult
    contracted_prev = atr.shift(1) < atr.shift(2)
    sig = pd.Series(0, index=df.index)
    sig[(expanding & contracted_prev) & (close > ema)] =  1
    sig[(expanding & contracted_prev) & (close < ema)] = -1
    return sig.fillna(0)


def space_Crypto_Scalp_ATR(trial):
    return {
        'atr_p':         trial.suggest_int('atr_p',         10, 20),
        'atr_mult':      trial.suggest_float('atr_mult',     1.3, 2.5),
        'ema_p':         trial.suggest_int('ema_p',         10, 40),
        'contraction_p': trial.suggest_int('contraction_p',  5, 15),
    }


# ── 6. CVDD_Strategy ──────────────────────────────────────────────────────────

def gen_CVDD_Strategy(df, lookback=10, vol_p=20, **kw):
    close    = df['close']
    vol      = df['volume']
    lookback = int(lookback); vol_p = int(vol_p)
    obv      = (np.sign(close.diff()).fillna(0) * vol).cumsum()
    vol_avg  = _sma(vol, vol_p)
    price_ll = close < close.rolling(lookback, min_periods=1).min().shift(1)
    price_hh = close > close.rolling(lookback, min_periods=1).max().shift(1)
    obv_hl   = obv > obv.rolling(lookback, min_periods=1).min().shift(1) + obv.rolling(lookback, min_periods=1).std().fillna(0)
    obv_lh   = obv < obv.rolling(lookback, min_periods=1).max().shift(1) - obv.rolling(lookback, min_periods=1).std().fillna(0)
    rising_vol = vol > vol_avg
    sig = pd.Series(0, index=df.index)
    sig[price_ll & obv_hl & rising_vol] =  1
    sig[price_hh & obv_lh & rising_vol] = -1
    return sig.fillna(0)


def space_CVDD_Strategy(trial):
    return {
        'lookback': trial.suggest_int('lookback', 5,  20),
        'vol_p':    trial.suggest_int('vol_p',    20, 50),
    }


# ── 7. Crypto_Volume_Price ────────────────────────────────────────────────────

def gen_Crypto_Volume_Price(df, vpt_ema_p=14, **kw):
    close     = df['close']
    vol       = df['volume']
    vpt_ema_p = int(vpt_ema_p)
    pct_chg   = close.pct_change().fillna(0)
    vpt       = (vol * pct_chg).cumsum()
    vpt_ema   = _ema(vpt, vpt_ema_p)
    sig = pd.Series(0, index=df.index)
    sig[(vpt > vpt_ema) & (vpt.shift(1) <= vpt_ema.shift(1))] =  1
    sig[(vpt < vpt_ema) & (vpt.shift(1) >= vpt_ema.shift(1))] = -1
    return sig.fillna(0)


def space_Crypto_Volume_Price(trial):
    return {
        'vpt_ema_p': trial.suggest_int('vpt_ema_p', 10, 30),
    }


# ── 8. Crypto_VPVR ────────────────────────────────────────────────────────────

def gen_Crypto_VPVR(df, vp_p=50, zone_pct=0.01, **kw):
    close    = df['close']
    vol      = df['volume']
    vp_p     = int(vp_p)
    sig = pd.Series(0, index=df.index)
    for i in range(vp_p, len(df)):
        c_win = close.iloc[i - vp_p:i]
        v_win = vol.iloc[i - vp_p:i]
        if len(c_win) < 2:
            continue
        bins     = np.linspace(c_win.min(), c_win.max(), 20)
        idx_bins = np.digitize(c_win, bins) - 1
        idx_bins = np.clip(idx_bins, 0, len(bins) - 2)
        vol_bins = np.zeros(len(bins) - 1)
        for j, b in enumerate(idx_bins):
            vol_bins[b] += v_win.iloc[j]
        poc_bin  = vol_bins.argmax()
        poc_low  = bins[poc_bin]
        poc_high = bins[poc_bin + 1]
        c_cur    = close.iloc[i]
        zone_lo  = poc_low  * (1 - zone_pct)
        zone_hi  = poc_high * (1 + zone_pct)
        if zone_lo <= c_cur <= zone_hi:
            trend_up = c_cur > close.iloc[i - vp_p:i].mean()
            sig.iloc[i] = 1 if trend_up else -1
    return sig.fillna(0)


def space_Crypto_VPVR(trial):
    return {
        'vp_p':     trial.suggest_int('vp_p',       20, 100),
        'zone_pct': trial.suggest_float('zone_pct', 0.005, 0.02),
    }


# ── 9. Crypto_Absorption ──────────────────────────────────────────────────────

def gen_Crypto_Absorption(df, body_mult=0.3, vol_mult=3.0, ema_p=20, **kw):
    close    = df['close']
    open_    = df['open']
    vol      = df['volume']
    ema_p    = int(ema_p)
    atr      = _atr(df, 14)
    body     = (close - open_).abs()
    vol_avg  = _sma(vol, 20)
    ema      = _ema(close, ema_p)
    small_body = body < atr * body_mult
    high_vol   = vol > vol_avg * vol_mult
    sig = pd.Series(0, index=df.index)
    sig[small_body & high_vol & (close < ema)] =  1
    sig[small_body & high_vol & (close > ema)] = -1
    return sig.fillna(0)


def space_Crypto_Absorption(trial):
    return {
        'body_mult': trial.suggest_float('body_mult', 0.1, 0.5),
        'vol_mult':  trial.suggest_float('vol_mult',  2.0, 5.0),
        'ema_p':     trial.suggest_int('ema_p',       20, 60),
    }


# ── 10. Crypto_Volume_Trend ───────────────────────────────────────────────────

def gen_Crypto_Volume_Trend(df, ema_p=20, vol_trend_bars=3, **kw):
    close          = df['close']
    vol            = df['volume']
    ema_p          = int(ema_p); vol_trend_bars = int(vol_trend_bars)
    ema            = _ema(close, ema_p)
    vol_rising     = pd.Series(True, index=df.index)
    for i in range(1, vol_trend_bars):
        vol_rising = vol_rising & (vol > vol.shift(i).fillna(0))
    sig = pd.Series(0, index=df.index)
    sig[vol_rising & (close > ema)] =  1
    sig[vol_rising & (close < ema)] = -1
    return sig.fillna(0)


def space_Crypto_Volume_Trend(trial):
    return {
        'ema_p':          trial.suggest_int('ema_p',          10, 50),
        'vol_trend_bars': trial.suggest_int('vol_trend_bars',  2,  5),
    }


# ── 11. Crypto_Grid_MR ────────────────────────────────────────────────────────

def gen_Crypto_Grid_MR(df, ema_p=20, atr_p=14, grid_levels=2.0, **kw):
    close       = df['close']
    ema_p       = int(ema_p); atr_p = int(atr_p)
    ema         = _ema(close, ema_p)
    atr         = _atr(df, atr_p)
    upper       = ema + grid_levels * atr
    lower       = ema - grid_levels * atr
    sig = pd.Series(0, index=df.index)
    sig[close < lower] =  1
    sig[close > upper] = -1
    return sig.fillna(0)


def space_Crypto_Grid_MR(trial):
    return {
        'ema_p':       trial.suggest_int('ema_p',         20, 60),
        'atr_p':       trial.suggest_int('atr_p',         10, 20),
        'grid_levels': trial.suggest_float('grid_levels',  1.0, 3.0),
    }


# ── 12. Crypto_BB_MR ──────────────────────────────────────────────────────────

def gen_Crypto_BB_MR(df, bb_p=20, bb_mult=2.0, atr_p=14, **kw):
    close   = df['close']
    bb_p    = int(bb_p); atr_p = int(atr_p)
    mid     = _sma(close, bb_p)
    std     = close.rolling(bb_p, min_periods=1).std().fillna(0)
    lower   = mid - bb_mult * std
    upper   = mid + bb_mult * std
    atr     = _atr(df, atr_p)
    atr_avg = _sma(atr, bb_p)
    calm    = atr < atr_avg * 1.5
    sig = pd.Series(0, index=df.index)
    sig[(close < lower) & calm] =  1
    sig[(close > upper) & calm] = -1
    return sig.fillna(0)


def space_Crypto_BB_MR(trial):
    return {
        'bb_p':    trial.suggest_int('bb_p',      15, 30),
        'bb_mult': trial.suggest_float('bb_mult',  1.5, 3.0),
        'atr_p':   trial.suggest_int('atr_p',     10, 20),
    }


# ── 13. Crypto_Zscore ─────────────────────────────────────────────────────────

def gen_Crypto_Zscore(df, z_p=30, entry_z=1.5, **kw):
    close = df['close']
    z_p   = int(z_p)
    mean  = close.rolling(z_p, min_periods=1).mean()
    std   = close.rolling(z_p, min_periods=1).std().fillna(1e-9).replace(0, 1e-9)
    z     = (close - mean) / std
    sig = pd.Series(0, index=df.index)
    sig[z < -entry_z] =  1
    sig[z >  entry_z] = -1
    return sig.fillna(0)


def space_Crypto_Zscore(trial):
    return {
        'z_p':     trial.suggest_int('z_p',       20, 60),
        'entry_z': trial.suggest_float('entry_z',  1.0, 2.5),
    }


# ── 14. Crypto_Reversion_RSI ──────────────────────────────────────────────────

def gen_Crypto_Reversion_RSI(df, rsi_p=14, bb_p=20, bb_mult=2.0, atr_p=14, max_atr_mult=3.0, **kw):
    close        = df['close']
    rsi_p        = int(rsi_p); bb_p = int(bb_p); atr_p = int(atr_p)
    rsi          = _rsi(close, rsi_p)
    mid          = _sma(close, bb_p)
    std          = close.rolling(bb_p, min_periods=1).std().fillna(0)
    lower        = mid - bb_mult * std
    upper        = mid + bb_mult * std
    atr          = _atr(df, atr_p)
    atr_avg      = _sma(atr, bb_p)
    not_extreme  = atr < atr_avg * max_atr_mult
    sig = pd.Series(0, index=df.index)
    sig[(rsi < 25) & (close < lower) & not_extreme] =  1
    sig[(rsi > 75) & (close > upper) & not_extreme] = -1
    return sig.fillna(0)


def space_Crypto_Reversion_RSI(trial):
    return {
        'rsi_p':        trial.suggest_int('rsi_p',         7,  21),
        'bb_p':         trial.suggest_int('bb_p',          15, 30),
        'bb_mult':      trial.suggest_float('bb_mult',      1.5, 3.0),
        'atr_p':        trial.suggest_int('atr_p',         10, 20),
        'max_atr_mult': trial.suggest_float('max_atr_mult', 2.0, 4.0),
    }


# ── 15. BTC_Corr_Long ─────────────────────────────────────────────────────────

def gen_BTC_Corr_Long(df, regime_p=150, fast_p=9, rsi_p=14, **kw):
    close    = df['close']
    regime_p = int(regime_p); fast_p = int(fast_p); rsi_p = int(rsi_p)
    ema_long = _ema(close, regime_p)
    ema_fast = _ema(close, fast_p)
    rsi      = _rsi(close, rsi_p)
    bull_regime = close > ema_long
    bear_regime = close < ema_long
    sig = pd.Series(0, index=df.index)
    sig[bull_regime & (ema_fast > ema_fast.shift(1)) & (rsi > 50) & (rsi < 70)] =  1
    sig[bear_regime & (ema_fast < ema_fast.shift(1)) & (rsi < 50) & (rsi > 30)] = -1
    return sig.fillna(0)


def space_BTC_Corr_Long(trial):
    return {
        'regime_p': trial.suggest_int('regime_p', 100, 200),
        'fast_p':   trial.suggest_int('fast_p',     5,  20),
        'rsi_p':    trial.suggest_int('rsi_p',      7,  21),
    }


# ── 16. Alt_Season_Momentum ───────────────────────────────────────────────────

def gen_Alt_Season_Momentum(df, p1=20, p2=50, p3=200, **kw):
    close = df['close']
    p1 = int(p1); p2 = int(p2); p3 = int(p3)
    ema1  = _ema(close, p1)
    ema2  = _ema(close, p2)
    ema3  = _ema(close, p3)
    sig = pd.Series(0, index=df.index)
    sig[(close > ema1) & (close > ema2) & (close > ema3)] =  1
    sig[(close < ema1) & (close < ema2) & (close < ema3)] = -1
    return sig.fillna(0)


def space_Alt_Season_Momentum(trial):
    return {
        'p1': trial.suggest_int('p1',  15,  25),
        'p2': trial.suggest_int('p2',  40,  60),
        'p3': trial.suggest_int('p3', 180, 220),
    }


# ── 17. Crypto_Trend_Regime ───────────────────────────────────────────────────

def gen_Crypto_Trend_Regime(df, regime_p=200, fast_p=9, slow_p=21, **kw):
    close    = df['close']
    regime_p = int(regime_p); fast_p = int(fast_p); slow_p = int(slow_p)
    ema_200  = _ema(close, regime_p)
    ema_fast = _ema(close, fast_p)
    ema_slow = _ema(close, slow_p)
    bull     = close > ema_200
    bear     = close < ema_200
    cross_up   = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    cross_down = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[bull & cross_up]   =  1
    sig[bear & cross_down] = -1
    return sig.fillna(0)


def space_Crypto_Trend_Regime(trial):
    return {
        'regime_p': trial.suggest_int('regime_p', 150, 250),
        'fast_p':   trial.suggest_int('fast_p',     5,  20),
        'slow_p':   trial.suggest_int('slow_p',    20,  60),
    }


# ── 18. Crypto_FOMO ───────────────────────────────────────────────────────────

def gen_Crypto_FOMO(df, roc_p=5, vol_p=20, accel_mult=2.0, **kw):
    close   = df['close']
    vol     = df['volume']
    roc_p   = int(roc_p); vol_p = int(vol_p)
    roc     = (close - close.shift(roc_p)) / close.shift(roc_p).replace(0, 1e-9) * 100
    vol_avg = _sma(vol, vol_p)
    accel   = (roc > roc.shift(1)) & (roc.shift(1) > roc.shift(2))
    high_vol = vol > vol_avg * accel_mult
    sig = pd.Series(0, index=df.index)
    sig[accel & high_vol & (roc > 0)] =  1
    sig[accel & high_vol & (roc < 0)] = -1
    return sig.fillna(0)


def space_Crypto_FOMO(trial):
    return {
        'roc_p':      trial.suggest_int('roc_p',       3,   8),
        'vol_p':      trial.suggest_int('vol_p',       20,  50),
        'accel_mult': trial.suggest_float('accel_mult', 1.5, 4.0),
    }


# ── 19. Crypto_Accumulation ───────────────────────────────────────────────────

def gen_Crypto_Accumulation(df, std_p=30, vol_p=20, expand_mult=2.0, **kw):
    close    = df['close']
    vol      = df['volume']
    std_p    = int(std_p); vol_p = int(vol_p)
    std_roll = close.rolling(std_p, min_periods=1).std().fillna(0)
    std_avg  = _sma(std_roll, std_p)
    vol_avg  = _sma(vol, vol_p)
    tight    = std_roll < std_avg * 0.7
    vol_dry  = vol < vol_avg * 0.8
    vol_exp  = vol > vol_avg * expand_mult
    accum    = tight.shift(1).fillna(False) & vol_dry.shift(1).fillna(False)
    sig = pd.Series(0, index=df.index)
    sig[accum & vol_exp & (close > close.shift(1))] =  1
    sig[accum & vol_exp & (close < close.shift(1))] = -1
    return sig.fillna(0)


def space_Crypto_Accumulation(trial):
    return {
        'std_p':       trial.suggest_int('std_p',         20, 50),
        'vol_p':       trial.suggest_int('vol_p',         20, 50),
        'expand_mult': trial.suggest_float('expand_mult',  1.5, 3.0),
    }


# ── 20. Crypto_Distribution ───────────────────────────────────────────────────

def gen_Crypto_Distribution(df, vol_mult=2.0, body_pct=0.25, bb_p=20, bb_mult=2.0, **kw):
    close  = df['close']
    open_  = df['open']
    high   = df['high']
    low    = df['low']
    vol    = df['volume']
    bb_p   = int(bb_p)
    mid    = _sma(close, bb_p)
    std    = close.rolling(bb_p, min_periods=1).std().fillna(0)
    upper  = mid + bb_mult * std
    lower  = mid - bb_mult * std
    vol_avg = _sma(vol, bb_p)
    candle_range = (high - low).replace(0, 1e-9)
    body    = (close - open_).abs()
    body_ratio = body / candle_range
    high_vol    = vol > vol_avg * vol_mult
    small_body  = body_ratio < body_pct
    sig = pd.Series(0, index=df.index)
    sig[high_vol & small_body & (close > upper)] = -1
    sig[high_vol & small_body & (close < lower)] =  1
    return sig.fillna(0)


def space_Crypto_Distribution(trial):
    return {
        'vol_mult': trial.suggest_float('vol_mult',  1.5, 3.0),
        'body_pct': trial.suggest_float('body_pct',  0.1, 0.35),
        'bb_p':     trial.suggest_int('bb_p',        15, 30),
        'bb_mult':  trial.suggest_float('bb_mult',   1.5, 2.5),
    }


# ── 21. Crypto_Liquidity_Run ──────────────────────────────────────────────────

def gen_Crypto_Liquidity_Run(df, lookback=10, spike_pct=0.005, **kw):
    close    = df['close']
    low      = df['low']
    high     = df['high']
    lookback = int(lookback)
    prev_low  = low.rolling(lookback, min_periods=1).min().shift(1)
    prev_high = high.rolling(lookback, min_periods=1).max().shift(1)
    spike_dn = (low < prev_low * (1 - spike_pct)) & (close > prev_low)
    spike_up = (high > prev_high * (1 + spike_pct)) & (close < prev_high)
    sig = pd.Series(0, index=df.index)
    sig[spike_dn] =  1
    sig[spike_up] = -1
    return sig.fillna(0)


def space_Crypto_Liquidity_Run(trial):
    return {
        'lookback':  trial.suggest_int('lookback',     5,  20),
        'spike_pct': trial.suggest_float('spike_pct', 0.002, 0.01),
    }


# ── 22. Adaptive_Crypto_EMA ───────────────────────────────────────────────────

def gen_Adaptive_Crypto_EMA(df, base_p=20, atr_p=14, **kw):
    close  = df['close']
    base_p = int(base_p); atr_p = int(atr_p)
    atr    = _atr(df, atr_p)
    atr_avg = _sma(atr, base_p)
    vol_ratio = (atr / atr_avg.replace(0, 1e-9)).fillna(1.0)
    adapt_p = (base_p / vol_ratio.clip(0.5, 3.0)).clip(lower=3).round().astype(int)
    ema_vals = pd.Series(index=df.index, dtype=float)
    for span in adapt_p.unique():
        mask = adapt_p == span
        ema_vals[mask] = _ema(close, int(span))[mask]
    ema_vals = ema_vals.fillna(_ema(close, base_p))
    ema_rising = ema_vals > ema_vals.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[ema_rising  & (close > ema_vals)] =  1
    sig[~ema_rising & (close < ema_vals)] = -1
    return sig.fillna(0)


def space_Adaptive_Crypto_EMA(trial):
    return {
        'base_p': trial.suggest_int('base_p', 10, 50),
        'atr_p':  trial.suggest_int('atr_p',  10, 20),
    }


# ── 23. Crypto_ATR_Bands ──────────────────────────────────────────────────────

def gen_Crypto_ATR_Bands(df, ema_p=20, atr_p=14, bounce_mult=2.0, break_mult=3.0, **kw):
    close      = df['close']
    ema_p      = int(ema_p); atr_p = int(atr_p)
    ema        = _ema(close, ema_p)
    atr        = _atr(df, atr_p)
    lower_bounce = ema - bounce_mult * atr
    lower_break  = ema - break_mult  * atr
    upper_bounce = ema + bounce_mult * atr
    upper_break  = ema + break_mult  * atr
    sig = pd.Series(0, index=df.index)
    sig[(close < lower_bounce) & (close > lower_break)] =  1
    sig[(close < lower_break)]                           =  1
    sig[(close > upper_bounce) & (close < upper_break)] = -1
    sig[(close > upper_break)]                           = -1
    return sig.fillna(0)


def space_Crypto_ATR_Bands(trial):
    return {
        'ema_p':       trial.suggest_int('ema_p',          20, 60),
        'atr_p':       trial.suggest_int('atr_p',          10, 20),
        'bounce_mult': trial.suggest_float('bounce_mult',   1.5, 3.0),
        'break_mult':  trial.suggest_float('break_mult',    2.0, 4.0),
    }


# ── 24. Crypto_Momentum_Score ─────────────────────────────────────────────────

def gen_Crypto_Momentum_Score(df, roc_p=10, vol_p=20, rsi_p=14, threshold=0.5, **kw):
    close    = df['close']
    vol      = df['volume']
    roc_p    = int(roc_p); vol_p = int(vol_p); rsi_p = int(rsi_p)
    roc      = (close - close.shift(roc_p)) / close.shift(roc_p).replace(0, 1e-9) * 100
    roc_norm = roc / (roc.abs().rolling(vol_p, min_periods=1).mean().replace(0, 1e-9))
    vol_avg  = _sma(vol, vol_p)
    vol_chg  = (vol - vol_avg) / vol_avg.replace(0, 1e-9)
    rsi      = _rsi(close, rsi_p)
    rsi_norm = (rsi - 50) / 50.0
    score    = (roc_norm * 0.4 + vol_chg.clip(-3, 3) * 0.3 + rsi_norm * 0.3)
    sig = pd.Series(0, index=df.index)
    sig[score >  threshold] =  1
    sig[score < -threshold] = -1
    return sig.fillna(0)


def space_Crypto_Momentum_Score(trial):
    return {
        'roc_p':     trial.suggest_int('roc_p',       5,  20),
        'vol_p':     trial.suggest_int('vol_p',       20, 50),
        'rsi_p':     trial.suggest_int('rsi_p',       7,  21),
        'threshold': trial.suggest_float('threshold',  0.3, 0.7),
    }


# ── 25. High_Vol_Breakout ─────────────────────────────────────────────────────

def gen_High_Vol_Breakout(df, break_p=20, atr_p=14, atr_mult=1.0, **kw):
    close   = df['close']
    break_p = int(break_p); atr_p = int(atr_p)
    highest = close.rolling(break_p, min_periods=1).max().shift(1)
    lowest  = close.rolling(break_p, min_periods=1).min().shift(1)
    atr     = _atr(df, atr_p)
    upper_thresh = highest * (1 + atr_mult * atr / close.replace(0, 1e-9))
    lower_thresh = lowest  * (1 - atr_mult * atr / close.replace(0, 1e-9))
    sig = pd.Series(0, index=df.index)
    sig[close > upper_thresh] =  1
    sig[close < lower_thresh] = -1
    return sig.fillna(0)


def space_High_Vol_Breakout(trial):
    return {
        'break_p':  trial.suggest_int('break_p',    10, 30),
        'atr_p':    trial.suggest_int('atr_p',      10, 20),
        'atr_mult': trial.suggest_float('atr_mult',  0.5, 2.0),
    }


# ── 26. Small_Cap_Momentum ────────────────────────────────────────────────────

def gen_Small_Cap_Momentum(df, rsi_p=14, vol_p=20, atr_p=14, **kw):
    close   = df['close']
    vol     = df['volume']
    rsi_p   = int(rsi_p); vol_p = int(vol_p); atr_p = int(atr_p)
    rsi     = _rsi(close, rsi_p)
    vol_avg = _sma(vol, vol_p)
    atr     = _atr(df, atr_p)
    atr_rising = atr > atr.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[(rsi > 70) & (vol > vol_avg * 3.0) & atr_rising]              =  1
    sig[(rsi < 30) & (vol > vol_avg * 3.0) & atr_rising]              = -1
    return sig.fillna(0)


def space_Small_Cap_Momentum(trial):
    return {
        'rsi_p': trial.suggest_int('rsi_p', 7,  21),
        'vol_p': trial.suggest_int('vol_p', 20, 50),
        'atr_p': trial.suggest_int('atr_p', 10, 20),
    }


# ── 27. Parabolic_Entry ───────────────────────────────────────────────────────

def gen_Parabolic_Entry(df, roc_p=5, vol_p=20, **kw):
    close   = df['close']
    vol     = df['volume']
    roc_p   = int(roc_p); vol_p = int(vol_p)
    rsi     = _rsi(close, 14)
    roc     = (close - close.shift(roc_p)) / close.shift(roc_p).replace(0, 1e-9) * 100
    vol_avg = _sma(vol, vol_p)
    accel   = (roc > roc.shift(1)) & (roc.shift(1) > roc.shift(2)) & (roc.shift(2) > roc.shift(3))
    decel   = (roc < roc.shift(1)) & (roc.shift(1) < roc.shift(2)) & (roc.shift(2) < roc.shift(3))
    high_vol = vol > vol_avg * 2.0
    sig = pd.Series(0, index=df.index)
    sig[accel & high_vol & (rsi < 80)] =  1
    sig[decel & high_vol & (rsi > 20)] = -1
    return sig.fillna(0)


def space_Parabolic_Entry(trial):
    return {
        'roc_p': trial.suggest_int('roc_p',  3,  8),
        'vol_p': trial.suggest_int('vol_p', 20, 50),
    }


# ── 28. Capitulation_Buy ──────────────────────────────────────────────────────

def gen_Capitulation_Buy(df, drop_pct=0.05, vol_mult=3.0, rsi_p=14, **kw):
    close    = df['close']
    open_    = df['open']
    vol      = df['volume']
    rsi_p    = int(rsi_p)
    bar_drop = (close - open_) / open_.replace(0, 1e-9)
    vol_avg  = _sma(vol, 20)
    rsi      = _rsi(close, rsi_p)
    cap_long  = (bar_drop < -drop_pct) & (vol > vol_avg * vol_mult) & (rsi < 20)
    cap_short = (bar_drop >  drop_pct) & (vol > vol_avg * vol_mult) & (rsi > 80)
    sig = pd.Series(0, index=df.index)
    sig[cap_long]  =  1
    sig[cap_short] = -1
    return sig.fillna(0)


def space_Capitulation_Buy(trial):
    return {
        'drop_pct':  trial.suggest_float('drop_pct',  0.03, 0.08),
        'vol_mult':  trial.suggest_float('vol_mult',   2.0,  5.0),
        'rsi_p':     trial.suggest_int('rsi_p',        7,   21),
    }


# ── 29. Relief_Rally ──────────────────────────────────────────────────────────

def gen_Relief_Rally(df, n_red=5, vol_mult=2.0, **kw):
    close   = df['close']
    open_   = df['open']
    vol     = df['volume']
    n_red   = int(n_red)
    red_bar  = (close < open_).astype(int)
    green_bar = close > open_
    consec_red = red_bar.copy().astype(float)
    for i in range(1, n_red):
        consec_red = consec_red + red_bar.shift(i).fillna(0)
    enough_red = consec_red >= n_red
    down_then_green = enough_red.shift(1).fillna(False) & green_bar
    consec_green = (1 - red_bar).copy().astype(float)
    for i in range(1, n_red):
        consec_green = consec_green + (1 - red_bar).shift(i).fillna(0)
    enough_green = consec_green >= n_red
    up_then_red  = enough_green.shift(1).fillna(False) & ~green_bar
    vol_avg  = _sma(vol, 20)
    high_vol = vol > vol_avg * vol_mult
    sig = pd.Series(0, index=df.index)
    sig[down_then_green & high_vol] =  1
    sig[up_then_red     & high_vol] = -1
    return sig.fillna(0)


def space_Relief_Rally(trial):
    return {
        'n_red':    trial.suggest_int('n_red',       3,   7),
        'vol_mult': trial.suggest_float('vol_mult',  1.5, 3.0),
    }


# ── 30. Crypto_Confluence ─────────────────────────────────────────────────────

def _supertrend(df, st_p, st_mult):
    atr  = _atr(df, int(st_p))
    hl2  = (df['high'] + df['low']) / 2
    upper = hl2 + st_mult * atr
    lower = hl2 - st_mult * atr
    close = df['close']
    trend = pd.Series(1, index=df.index)
    st    = lower.copy()
    for i in range(1, len(df)):
        if close.iloc[i - 1] > st.iloc[i - 1]:
            st.iloc[i]    = max(lower.iloc[i], st.iloc[i - 1])
            trend.iloc[i] = 1
        else:
            st.iloc[i]    = min(upper.iloc[i], st.iloc[i - 1])
            trend.iloc[i] = -1
    return trend


def gen_Crypto_Confluence(df, ema_fast=9, ema_slow=21, st_p=10, st_mult=3.0, rsi_p=14, vol_p=20, **kw):
    close    = df['close']
    vol      = df['volume']
    ema_fast = int(ema_fast); ema_slow = int(ema_slow)
    st_p     = int(st_p); rsi_p = int(rsi_p); vol_p = int(vol_p)
    ema_f    = _ema(close, ema_fast)
    ema_s    = _ema(close, ema_slow)
    rsi      = _rsi(close, rsi_p)
    vol_avg  = _sma(vol, vol_p)
    trend    = _supertrend(df, st_p, st_mult)
    ema_bull = ema_f > ema_s
    ema_bear = ema_f < ema_s
    rsi_bull = (rsi > 50) & (rsi < 65)
    rsi_bear = (rsi > 35) & (rsi < 50)
    vol_ok   = vol > vol_avg
    sig = pd.Series(0, index=df.index)
    sig[ema_bull & (trend == 1)  & rsi_bull & vol_ok] =  1
    sig[ema_bear & (trend == -1) & rsi_bear & vol_ok] = -1
    return sig.fillna(0)


def space_Crypto_Confluence(trial):
    return {
        'ema_fast': trial.suggest_int('ema_fast',    5,  20),
        'ema_slow': trial.suggest_int('ema_slow',   20,  60),
        'st_p':     trial.suggest_int('st_p',        7,  21),
        'st_mult':  trial.suggest_float('st_mult',   2.0, 4.0),
        'rsi_p':    trial.suggest_int('rsi_p',       7,  21),
        'vol_p':    trial.suggest_int('vol_p',       20, 50),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Crypto_Pump_Detector': {
        'gen': gen_Crypto_Pump_Detector,
        'space': space_Crypto_Pump_Detector,
        'default_params': {'vol_p': 20, 'vol_mult': 3.0, 'atr_p': 14, 'atr_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Detect pump: volume spike + ATR expansion. Bidirectional.'},
    },
    'Altcoin_Season': {
        'gen': gen_Altcoin_Season,
        'space': space_Altcoin_Season,
        'default_params': {'rsi_p': 14, 'vol_p': 20, 'fast_p': 9, 'slow_p': 21},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Altcoin momentum: RSI>65 + vol surge + EMA alignment.'},
    },
    'Crypto_Breakout': {
        'gen': gen_Crypto_Breakout,
        'space': space_Crypto_Breakout,
        'default_params': {'break_p': 20, 'vol_p': 20, 'vol_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 5000,
                 'description': '24-bar breakout with volume confirmation for crypto.'},
    },
    'Crypto_Dip_Buy': {
        'gen': gen_Crypto_Dip_Buy,
        'space': space_Crypto_Dip_Buy,
        'default_params': {'fast_p': 9, 'slow_p': 21, 'rsi_p': 14, 'rsi_os': 30},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 4000,
                 'description': 'Buy the dip: EMA uptrend + RSI oversold + volume spike.'},
    },
    'Crypto_Scalp_ATR': {
        'gen': gen_Crypto_Scalp_ATR,
        'space': space_Crypto_Scalp_ATR,
        'default_params': {'atr_p': 14, 'atr_mult': 1.8, 'ema_p': 20, 'contraction_p': 10},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'ATR contraction then expansion scalp with EMA trend filter.'},
    },
    'CVDD_Strategy': {
        'gen': gen_CVDD_Strategy,
        'space': space_CVDD_Strategy,
        'default_params': {'lookback': 10, 'vol_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Crypto Volatility-Driven Divergence: OBV divergence signals.'},
    },
    'Crypto_Volume_Price': {
        'gen': gen_Crypto_Volume_Price,
        'space': space_Crypto_Volume_Price,
        'default_params': {'vpt_ema_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Volume-Price Trend: VPT crosses EMA(VPT).'},
    },
    'Crypto_VPVR': {
        'gen': gen_Crypto_VPVR,
        'space': space_Crypto_VPVR,
        'default_params': {'vp_p': 50, 'zone_pct': 0.01},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Simplified VPVR: enter at high-volume price zone.'},
    },
    'Crypto_Absorption': {
        'gen': gen_Crypto_Absorption,
        'space': space_Crypto_Absorption,
        'default_params': {'body_mult': 0.3, 'vol_mult': 3.0, 'ema_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'High volume + small body = absorption. Accumulation at lows.'},
    },
    'Crypto_Volume_Trend': {
        'gen': gen_Crypto_Volume_Trend,
        'space': space_Crypto_Volume_Trend,
        'default_params': {'ema_p': 20, 'vol_trend_bars': 3},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': 'Rising volume in trend direction confirms momentum.'},
    },
    'Crypto_Grid_MR': {
        'gen': gen_Crypto_Grid_MR,
        'space': space_Crypto_Grid_MR,
        'default_params': {'ema_p': 20, 'atr_p': 14, 'grid_levels': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Grid mean reversion: EMA ± N*ATR levels. Long at lower, short at upper.'},
    },
    'Crypto_BB_MR': {
        'gen': gen_Crypto_BB_MR,
        'space': space_Crypto_BB_MR,
        'default_params': {'bb_p': 20, 'bb_mult': 2.0, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'BB mean reversion with ATR calm filter (avoid panic sells).'},
    },
    'Crypto_Zscore': {
        'gen': gen_Crypto_Zscore,
        'space': space_Crypto_Zscore,
        'default_params': {'z_p': 30, 'entry_z': 1.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Z-score of price vs rolling mean. Long z<-threshold, short z>threshold.'},
    },
    'Crypto_Reversion_RSI': {
        'gen': gen_Crypto_Reversion_RSI,
        'space': space_Crypto_Reversion_RSI,
        'default_params': {'rsi_p': 14, 'bb_p': 20, 'bb_mult': 2.0, 'atr_p': 14, 'max_atr_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Triple confirmation MR: RSI extreme + BB touch + ATR not extreme.'},
    },
    'BTC_Corr_Long': {
        'gen': gen_BTC_Corr_Long,
        'space': space_BTC_Corr_Long,
        'default_params': {'regime_p': 150, 'fast_p': 9, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Regime filter via long EMA + short EMA direction + RSI zone.'},
    },
    'Alt_Season_Momentum': {
        'gen': gen_Alt_Season_Momentum,
        'space': space_Alt_Season_Momentum,
        'default_params': {'p1': 20, 'p2': 50, 'p3': 200},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Strong altcoin momentum: price above 3 EMAs simultaneously.'},
    },
    'Crypto_Trend_Regime': {
        'gen': gen_Crypto_Trend_Regime,
        'space': space_Crypto_Trend_Regime,
        'default_params': {'regime_p': 200, 'fast_p': 9, 'slow_p': 21},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 2500,
                 'description': '200 EMA regime divide. EMA cross only in correct regime direction.'},
    },
    'Crypto_FOMO': {
        'gen': gen_Crypto_FOMO,
        'space': space_Crypto_FOMO,
        'default_params': {'roc_p': 5, 'vol_p': 20, 'accel_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'FOMO entry: 3-bar ROC acceleration + volume surge.'},
    },
    'Crypto_Accumulation': {
        'gen': gen_Crypto_Accumulation,
        'space': space_Crypto_Accumulation,
        'default_params': {'std_p': 30, 'vol_p': 20, 'expand_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Wyckoff accumulation: tight range + vol dry-up then expansion.'},
    },
    'Crypto_Distribution': {
        'gen': gen_Crypto_Distribution,
        'space': space_Crypto_Distribution,
        'default_params': {'vol_mult': 2.0, 'body_pct': 0.25, 'bb_p': 20, 'bb_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Distribution: high vol + small body at BB extremes.'},
    },
    'Crypto_Liquidity_Run': {
        'gen': gen_Crypto_Liquidity_Run,
        'space': space_Crypto_Liquidity_Run,
        'default_params': {'lookback': 10, 'spike_pct': 0.005},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Liquidity grab: spike beyond recent level then reversal close.'},
    },
    'Adaptive_Crypto_EMA': {
        'gen': gen_Adaptive_Crypto_EMA,
        'space': space_Adaptive_Crypto_EMA,
        'default_params': {'base_p': 20, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'EMA period adapts to ATR volatility. High vol = shorter period.'},
    },
    'Crypto_ATR_Bands': {
        'gen': gen_Crypto_ATR_Bands,
        'space': space_Crypto_ATR_Bands,
        'default_params': {'ema_p': 20, 'atr_p': 14, 'bounce_mult': 2.0, 'break_mult': 3.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'EMA ± ATR*mult bands. Bounce from lower = long, break above upper = strong long.'},
    },
    'Crypto_Momentum_Score': {
        'gen': gen_Crypto_Momentum_Score,
        'space': space_Crypto_Momentum_Score,
        'default_params': {'roc_p': 10, 'vol_p': 20, 'rsi_p': 14, 'threshold': 0.5},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Composite momentum score: ROC + volume change + RSI weighted.'},
    },
    'High_Vol_Breakout': {
        'gen': gen_High_Vol_Breakout,
        'space': space_High_Vol_Breakout,
        'default_params': {'break_p': 20, 'atr_p': 14, 'atr_mult': 1.0},
        'info': {'source': 'TradingView', 'version': 4, 'likes': 3000,
                 'description': 'ATR-adjusted breakout thresholds for high-volatility assets.'},
    },
    'Small_Cap_Momentum': {
        'gen': gen_Small_Cap_Momentum,
        'space': space_Small_Cap_Momentum,
        'default_params': {'rsi_p': 14, 'vol_p': 20, 'atr_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Strong signals for small caps: RSI>70 + vol>3x avg + ATR increasing.'},
    },
    'Parabolic_Entry': {
        'gen': gen_Parabolic_Entry,
        'space': space_Parabolic_Entry,
        'default_params': {'roc_p': 5, 'vol_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2500,
                 'description': 'Enter parabolic moves: 3-bar ROC acceleration + volume exploding + RSI<80.'},
    },
    'Capitulation_Buy': {
        'gen': gen_Capitulation_Buy,
        'space': space_Capitulation_Buy,
        'default_params': {'drop_pct': 0.05, 'vol_mult': 3.0, 'rsi_p': 14},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Buy capitulation: large red candle + high volume + RSI extreme low.'},
    },
    'Relief_Rally': {
        'gen': gen_Relief_Rally,
        'space': space_Relief_Rally,
        'default_params': {'n_red': 5, 'vol_mult': 2.0},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 2000,
                 'description': 'Relief rally: first green candle after N red bars with volume surge.'},
    },
    'Crypto_Confluence': {
        'gen': gen_Crypto_Confluence,
        'space': space_Crypto_Confluence,
        'default_params': {'ema_fast': 9, 'ema_slow': 21, 'st_p': 10, 'st_mult': 3.0, 'rsi_p': 14, 'vol_p': 20},
        'info': {'source': 'TradingView', 'version': 5, 'likes': 3000,
                 'description': 'Full confluence: EMA aligned + SuperTrend bull + RSI 50-65 + Volume above avg.'},
    },
}
