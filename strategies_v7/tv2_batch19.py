#!/usr/bin/env python3
"""TV2 BATCH 19 — 30 estrategias Momentum Oscillators 2026-04-01

  CMO_Strategy              — Chande Momentum Oscillator zero-line cross (v4, ~2500L)
  CMO_EMA                   — CMO + EMA trend filter (v5, ~1500L)
  CMO_Signal                — CMO + its own EMA signal line (v4, ~1200L)
  DPO_Strategy              — Detrended Price Oscillator zero cross (v4, ~2000L)
  DPO_RSI                   — DPO cycle + RSI oversold/overbought (v5, ~1200L)
  Ultimate_Oscillator       — Weighted 3-TF buying pressure oscillator (v4, ~3000L)
  UO_EMA                    — Ultimate Oscillator + EMA trend (v5, ~1500L)
  UO_Divergence             — Ultimate Oscillator bull/bear divergence (v4, ~1200L)
  Coppock_Curve             — WMA of dual ROC zero-line cross (v4, ~2500L)
  Coppock_EMA               — Coppock Curve + EMA filter (v5, ~1200L)
  TRIX_Strategy             — Triple-EMA TRIX zero-line cross (v4, ~2500L)
  TRIX_Signal               — TRIX + EMA signal line cross (v5, ~1500L)
  TRIX_Histogram            — TRIX histogram momentum (v4, ~1000L)
  ROC_Strategy              — Rate of Change zero cross + rising (v4, ~2000L)
  ROC_EMA                   — ROC + EMA trend filter (v5, ~1500L)
  ROC_RSI                   — ROC + RSI band (v4, ~1000L)
  Momentum_OSC              — Momentum oscillator + EMA signal (v4, ~2000L)
  AO_Classic                — Awesome Oscillator zero line + acceleration (v4, ~3000L)
  AO_Divergence             — Awesome Oscillator bull/bear divergence (v5, ~2000L)
  AO_Saucer                 — Awesome Oscillator saucer pattern (v4, ~1500L)
  AO_Twin_Peaks             — Awesome Oscillator twin peaks (v5, ~1200L)
  DeMarker_Strategy         — DeMarker oscillator oversold/overbought (v4, ~2000L)
  DeMarker_EMA              — DeMarker + EMA filter (v5, ~1200L)
  RVI_Strategy              — Relative Vigor Index signal cross (v4, ~2000L)
  RVI_RSI                   — RVI + RSI confirmation (v5, ~1200L)
  APO_Strategy              — Absolute Price Oscillator signal cross below zero (v4, ~1500L)
  PPO_Strategy              — Percentage Price Oscillator zero cross (v5, ~1500L)
  Fisher_Transform          — Ehlers Fisher Transform trigger cross (v4, ~3500L)
  Fisher_ATR                — Fisher Transform + ATR volatility filter (v5, ~1800L)
  Klinger_Signal            — Klinger Volume Oscillator signal cross (v4, ~2000L)
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


def _wma(s, p):
    """Linearly-weighted moving average."""
    p = int(p)
    weights = np.arange(1, p + 1, dtype=float)
    return s.rolling(p, min_periods=p).apply(
        lambda x: (x * weights).sum() / weights.sum(), raw=True
    )


def _cmo(s, p):
    """Chande Momentum Oscillator."""
    d    = s.diff()
    ups  = d.clip(lower=0).rolling(int(p), min_periods=1).sum()
    dns  = (-d).clip(lower=0).rolling(int(p), min_periods=1).sum()
    return 100.0 * (ups - dns) / (ups + dns + 1e-9)


def _roc(s, p):
    """Rate of Change in percent."""
    s_shifted = s.shift(int(p))
    return (s - s_shifted) / s_shifted.replace(0, 1e-9) * 100.0


def _uo(df, p1, p2, p3):
    """Ultimate Oscillator."""
    p1, p2, p3  = int(p1), int(p2), int(p3)
    prev_close  = df['close'].shift(1)
    bp          = df['close'] - pd.concat([df['low'], prev_close], axis=1).min(axis=1)
    tr          = pd.concat([df['high'], prev_close], axis=1).max(axis=1) - \
                  pd.concat([df['low'],  prev_close], axis=1).min(axis=1)
    bp_s1 = bp.rolling(p1, min_periods=1).sum()
    tr_s1 = tr.rolling(p1, min_periods=1).sum().replace(0, 1e-9)
    bp_s2 = bp.rolling(p2, min_periods=1).sum()
    tr_s2 = tr.rolling(p2, min_periods=1).sum().replace(0, 1e-9)
    bp_s3 = bp.rolling(p3, min_periods=1).sum()
    tr_s3 = tr.rolling(p3, min_periods=1).sum().replace(0, 1e-9)
    return 100.0 * (4.0 * bp_s1 / tr_s1 + 2.0 * bp_s2 / tr_s2 + bp_s3 / tr_s3) / 7.0


def _trix(s, p):
    """TRIX: 1-bar pct change of triple EMA."""
    e1 = _ema(s, p)
    e2 = _ema(e1, p)
    e3 = _ema(e2, p)
    return e3.pct_change() * 100.0


def _ao(df, fast_p, slow_p):
    """Awesome Oscillator."""
    hl2 = (df['high'] + df['low']) / 2.0
    return _sma(hl2, fast_p) - _sma(hl2, slow_p)


def _rvi_calc(df, p):
    """Relative Vigor Index."""
    c, o = df['close'], df['open']
    h, l = df['high'],  df['low']
    num_raw = (
        (c - o) +
        2.0 * (c.shift(1) - o.shift(1)) +
        2.0 * (c.shift(2) - o.shift(2)) +
        (c.shift(3) - o.shift(3))
    ) / 6.0
    den_raw = (
        (h - l) +
        2.0 * (h.shift(1) - l.shift(1)) +
        2.0 * (h.shift(2) - l.shift(2)) +
        (h.shift(3) - l.shift(3))
    ) / 6.0
    rvi_val  = _sma(num_raw, p) / _sma(den_raw, p).replace(0, 1e-9)
    # 4-bar symmetrically-weighted signal line (same weights as RVI numerator)
    sig = (rvi_val + 2.0 * rvi_val.shift(1) + 2.0 * rvi_val.shift(2) + rvi_val.shift(3)) / 6.0
    return rvi_val, sig


def _fisher(df, p):
    """Ehlers Fisher Transform."""
    p       = int(p)
    highest = df['high'].rolling(p, min_periods=1).max()
    lowest  = df['low'].rolling(p, min_periods=1).min()
    value   = 2.0 * (df['close'] - lowest) / (highest - lowest + 1e-9) - 1.0
    value   = value.clip(-0.999, 0.999)
    fish    = 0.5 * np.log((1.0 + value) / (1.0 - value + 1e-9))
    trigger = fish.shift(1)
    return fish, trigger


def _klinger(df, fast_p, slow_p, sig_p):
    """Klinger Volume Oscillator."""
    fast_p, slow_p, sig_p = int(fast_p), int(slow_p), int(sig_p)
    hlc3     = (df['high'] + df['low'] + df['close']) / 3.0
    trend    = (hlc3 > hlc3.shift(1)).map({True: 1.0, False: -1.0})
    dm       = df['high'] - df['low']
    cm_raw   = dm.copy().astype(float)
    # cumulative measure: if same trend cm = prev_cm + dm, else cm = prev_hlc_range
    cm_vals  = [0.0] * len(df)
    dm_vals  = dm.values
    tr_vals  = trend.values
    hl_vals  = dm.values
    for i in range(1, len(df)):
        if tr_vals[i] == tr_vals[i - 1]:
            cm_vals[i] = cm_vals[i - 1] + dm_vals[i]
        else:
            cm_vals[i] = dm_vals[i - 1] + dm_vals[i]
    cm_s     = pd.Series(cm_vals, index=df.index)
    vol      = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    vf_num   = (cm_s.replace(0, 1e-9) * 2.0).abs()
    vf_denom = (cm_s.abs() + 1e-9)
    vf       = vol * trend * 100.0 * (vf_num / vf_denom - 1.0).abs()
    ko       = _ema(vf, fast_p) - _ema(vf, slow_p)
    sig      = _ema(ko, sig_p)
    return ko, sig


# ── 1. CMO_Strategy ───────────────────────────────────────────────────────────

def gen_CMO_Strategy(df, cmo_p=14, **kw):
    cmo_p = int(cmo_p)
    cmo   = _cmo(df['close'], cmo_p)
    long  = (cmo > -50) & (cmo.shift(1) <= -50)
    short = (cmo < 50)  & (cmo.shift(1) >= 50)
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_CMO_Strategy(trial):
    return {'cmo_p': trial.suggest_int('cmo_p', 9, 21)}


# ── 2. CMO_EMA ────────────────────────────────────────────────────────────────

def gen_CMO_EMA(df, cmo_p=14, ema_p=34, **kw):
    cmo_p = int(cmo_p)
    ema_p = int(ema_p)
    cmo   = _cmo(df['close'], cmo_p)
    ema   = _ema(df['close'], ema_p)
    long  = (cmo > 0) & (df['close'] > ema)
    short = (cmo < 0) & (df['close'] < ema)
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_CMO_EMA(trial):
    return {
        'cmo_p': trial.suggest_int('cmo_p', 9, 21),
        'ema_p': trial.suggest_int('ema_p', 20, 60),
    }


# ── 3. CMO_Signal ─────────────────────────────────────────────────────────────

def gen_CMO_Signal(df, cmo_p=14, sig_p=7, **kw):
    cmo_p = int(cmo_p)
    sig_p = int(sig_p)
    cmo   = _cmo(df['close'], cmo_p)
    sig   = _ema(cmo, sig_p)
    long  = (cmo > sig) & (cmo.shift(1) <= sig.shift(1))
    short = (cmo < sig) & (cmo.shift(1) >= sig.shift(1))
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_CMO_Signal(trial):
    return {
        'cmo_p': trial.suggest_int('cmo_p', 9, 21),
        'sig_p': trial.suggest_int('sig_p', 5, 10),
    }


# ── 4. DPO_Strategy ───────────────────────────────────────────────────────────

def gen_DPO_Strategy(df, dpo_p=20, **kw):
    dpo_p  = int(dpo_p)
    offset = dpo_p // 2 + 1
    dpo    = df['close'].shift(offset) - df['close'].rolling(dpo_p, min_periods=1).mean()
    long   = (dpo > 0) & (dpo.shift(1) <= 0)
    short  = (dpo < 0) & (dpo.shift(1) >= 0)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_DPO_Strategy(trial):
    return {'dpo_p': trial.suggest_int('dpo_p', 14, 30)}


# ── 5. DPO_RSI ────────────────────────────────────────────────────────────────

def gen_DPO_RSI(df, dpo_p=20, rsi_p=14, **kw):
    dpo_p  = int(dpo_p)
    rsi_p  = int(rsi_p)
    offset = dpo_p // 2 + 1
    dpo    = df['close'].shift(offset) - df['close'].rolling(dpo_p, min_periods=1).mean()
    rsi    = _rsi(df['close'], rsi_p)
    long   = (dpo > 0) & (rsi < 50)
    short  = (dpo < 0) & (rsi > 50)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_DPO_RSI(trial):
    return {
        'dpo_p': trial.suggest_int('dpo_p', 14, 30),
        'rsi_p': trial.suggest_int('rsi_p', 7,  21),
    }


# ── 6. Ultimate_Oscillator ────────────────────────────────────────────────────

def gen_Ultimate_Oscillator(df, p1=7, p2=14, p3=28, **kw):
    uo    = _uo(df, p1, p2, p3)
    long  = uo < 30.0
    short = uo > 70.0
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_Ultimate_Oscillator(trial):
    return {
        'p1': trial.suggest_int('p1', 5,  10),
        'p2': trial.suggest_int('p2', 10, 20),
        'p3': trial.suggest_int('p3', 20, 40),
    }


# ── 7. UO_EMA ─────────────────────────────────────────────────────────────────

def gen_UO_EMA(df, p1=7, p2=14, p3=28, ema_p=50, **kw):
    uo    = _uo(df, p1, p2, p3)
    ema   = _ema(df['close'], int(ema_p))
    long  = (uo < 35.0) & (df['close'] > ema)
    short = (uo > 65.0) & (df['close'] < ema)
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_UO_EMA(trial):
    return {
        'p1':    trial.suggest_int('p1',    5,   10),
        'p2':    trial.suggest_int('p2',    10,  20),
        'p3':    trial.suggest_int('p3',    20,  40),
        'ema_p': trial.suggest_int('ema_p', 20, 100),
    }


# ── 8. UO_Divergence ─────────────────────────────────────────────────────────

def gen_UO_Divergence(df, p1=7, p2=14, p3=28, lookback=10, **kw):
    lookback = int(lookback)
    uo       = _uo(df, p1, p2, p3)
    price    = df['close']
    # Bullish divergence: price makes lower low, UO makes higher low over lookback
    price_ll = price < price.rolling(lookback, min_periods=1).min().shift(1)
    uo_hl    = uo    > uo.rolling(lookback, min_periods=1).min().shift(1)
    # Bearish divergence: price makes higher high, UO makes lower high
    price_hh = price > price.rolling(lookback, min_periods=1).max().shift(1)
    uo_lh    = uo    < uo.rolling(lookback, min_periods=1).max().shift(1)
    out      = pd.Series(0, index=df.index)
    out[price_ll & uo_hl] =  1
    out[price_hh & uo_lh] = -1
    return out.fillna(0).astype(int)


def space_UO_Divergence(trial):
    return {
        'p1':       trial.suggest_int('p1',       5,   10),
        'p2':       trial.suggest_int('p2',       10,  20),
        'p3':       trial.suggest_int('p3',       20,  40),
        'lookback': trial.suggest_int('lookback',  5,  20),
    }


# ── 9. Coppock_Curve ──────────────────────────────────────────────────────────

def gen_Coppock_Curve(df, roc1=14, roc2=11, wma_p=10, **kw):
    roc1  = int(roc1)
    roc2  = int(roc2)
    wma_p = int(wma_p)
    cc    = _wma(_roc(df['close'], roc1) + _roc(df['close'], roc2), wma_p)
    long  = (cc > 0) & (cc.shift(1) <= 0)
    short = (cc < 0) & (cc.shift(1) >= 0)
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_Coppock_Curve(trial):
    return {
        'roc1':  trial.suggest_int('roc1',  10, 16),
        'roc2':  trial.suggest_int('roc2',   7, 13),
        'wma_p': trial.suggest_int('wma_p',  8, 15),
    }


# ── 10. Coppock_EMA ───────────────────────────────────────────────────────────

def gen_Coppock_EMA(df, roc1=14, roc2=11, wma_p=10, ema_p=100, **kw):
    roc1  = int(roc1)
    roc2  = int(roc2)
    wma_p = int(wma_p)
    ema_p = int(ema_p)
    cc    = _wma(_roc(df['close'], roc1) + _roc(df['close'], roc2), wma_p)
    ema   = _ema(df['close'], ema_p)
    long  = (cc > 0) & (df['close'] > ema)
    short = (cc < 0) & (df['close'] < ema)
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_Coppock_EMA(trial):
    return {
        'roc1':  trial.suggest_int('roc1',  10,  16),
        'roc2':  trial.suggest_int('roc2',   7,  13),
        'wma_p': trial.suggest_int('wma_p',  8,  15),
        'ema_p': trial.suggest_int('ema_p', 50, 200),
    }


# ── 11. TRIX_Strategy ─────────────────────────────────────────────────────────

def gen_TRIX_Strategy(df, trix_p=14, sig_p=5, **kw):
    trix_p = int(trix_p)
    trix   = _trix(df['close'], trix_p)
    long   = (trix > 0) & (trix.shift(1) <= 0)
    short  = (trix < 0) & (trix.shift(1) >= 0)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_TRIX_Strategy(trial):
    return {
        'trix_p': trial.suggest_int('trix_p', 9, 20),
        'sig_p':  trial.suggest_int('sig_p',  3,  9),
    }


# ── 12. TRIX_Signal ───────────────────────────────────────────────────────────

def gen_TRIX_Signal(df, trix_p=14, sig_p=5, **kw):
    trix_p = int(trix_p)
    sig_p  = int(sig_p)
    trix   = _trix(df['close'], trix_p)
    sig    = _ema(trix, sig_p)
    long   = (trix > sig) & (trix.shift(1) <= sig.shift(1))
    short  = (trix < sig) & (trix.shift(1) >= sig.shift(1))
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_TRIX_Signal(trial):
    return {
        'trix_p': trial.suggest_int('trix_p', 9, 20),
        'sig_p':  trial.suggest_int('sig_p',  3,  9),
    }


# ── 13. TRIX_Histogram ────────────────────────────────────────────────────────

def gen_TRIX_Histogram(df, trix_p=14, sig_p=5, **kw):
    trix_p = int(trix_p)
    sig_p  = int(sig_p)
    trix   = _trix(df['close'], trix_p)
    sig    = _ema(trix, sig_p)
    hist   = trix - sig
    # Long: histogram turns positive AND rising (hist > 0 and hist > hist[1])
    long   = (hist > 0) & (hist > hist.shift(1))
    short  = (hist < 0) & (hist < hist.shift(1))
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_TRIX_Histogram(trial):
    return {
        'trix_p': trial.suggest_int('trix_p', 9, 20),
        'sig_p':  trial.suggest_int('sig_p',  3,  9),
    }


# ── 14. ROC_Strategy ─────────────────────────────────────────────────────────

def gen_ROC_Strategy(df, roc_p=20, **kw):
    roc_p = int(roc_p)
    roc   = _roc(df['close'], roc_p)
    long  = (roc > 0) & (roc.shift(1) <= 0) & (roc > roc.shift(1))
    short = (roc < 0) & (roc.shift(1) >= 0) & (roc < roc.shift(1))
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_ROC_Strategy(trial):
    return {'roc_p': trial.suggest_int('roc_p', 10, 30)}


# ── 15. ROC_EMA ───────────────────────────────────────────────────────────────

def gen_ROC_EMA(df, roc_p=20, ema_p=50, **kw):
    roc_p = int(roc_p)
    ema_p = int(ema_p)
    roc   = _roc(df['close'], roc_p)
    ema   = _ema(df['close'], ema_p)
    long  = (roc > 0) & (df['close'] > ema)
    short = (roc < 0) & (df['close'] < ema)
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_ROC_EMA(trial):
    return {
        'roc_p': trial.suggest_int('roc_p', 10,  30),
        'ema_p': trial.suggest_int('ema_p', 20, 100),
    }


# ── 16. ROC_RSI ───────────────────────────────────────────────────────────────

def gen_ROC_RSI(df, roc_p=20, rsi_p=14, **kw):
    roc_p = int(roc_p)
    rsi_p = int(rsi_p)
    roc   = _roc(df['close'], roc_p)
    rsi   = _rsi(df['close'], rsi_p)
    long  = (roc > 0) & (rsi >= 40) & (rsi <= 65)
    short = (roc < 0) & (rsi >= 35) & (rsi <= 60)
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_ROC_RSI(trial):
    return {
        'roc_p': trial.suggest_int('roc_p', 10, 30),
        'rsi_p': trial.suggest_int('rsi_p',  7, 21),
    }


# ── 17. Momentum_OSC ─────────────────────────────────────────────────────────

def gen_Momentum_OSC(df, mom_p=20, sig_p=10, **kw):
    mom_p = int(mom_p)
    sig_p = int(sig_p)
    mom   = df['close'] - df['close'].shift(mom_p)
    sig   = _ema(mom, sig_p)
    long  = (mom > 0) & (mom > sig) & (mom.shift(1) <= sig.shift(1))
    short = (mom < 0) & (mom < sig) & (mom.shift(1) >= sig.shift(1))
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_Momentum_OSC(trial):
    return {
        'mom_p': trial.suggest_int('mom_p', 10, 30),
        'sig_p': trial.suggest_int('sig_p',  5, 15),
    }


# ── 18. AO_Classic ────────────────────────────────────────────────────────────

def gen_AO_Classic(df, fast_p=5, slow_p=34, **kw):
    ao    = _ao(df, fast_p, slow_p)
    # Long: AO > 0 and AO > prev (accelerating upward); or zero cross
    long  = (ao > 0) & ((ao > ao.shift(1)) | (ao.shift(1) <= 0))
    short = (ao < 0) & ((ao < ao.shift(1)) | (ao.shift(1) >= 0))
    out   = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_AO_Classic(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',  3,  8),
        'slow_p': trial.suggest_int('slow_p', 25, 40),
    }


# ── 19. AO_Divergence ─────────────────────────────────────────────────────────

def gen_AO_Divergence(df, fast_p=5, slow_p=34, lookback=10, **kw):
    lookback = int(lookback)
    ao       = _ao(df, fast_p, slow_p)
    price    = df['close']
    price_ll = price < price.rolling(lookback, min_periods=1).min().shift(1)
    ao_hl    = ao    > ao.rolling(lookback, min_periods=1).min().shift(1)
    price_hh = price > price.rolling(lookback, min_periods=1).max().shift(1)
    ao_lh    = ao    < ao.rolling(lookback, min_periods=1).max().shift(1)
    out      = pd.Series(0, index=df.index)
    out[price_ll & ao_hl] =  1
    out[price_hh & ao_lh] = -1
    return out.fillna(0).astype(int)


def space_AO_Divergence(trial):
    return {
        'fast_p':   trial.suggest_int('fast_p',   3,  8),
        'slow_p':   trial.suggest_int('slow_p',  25, 40),
        'lookback': trial.suggest_int('lookback',  5, 20),
    }


# ── 20. AO_Saucer ─────────────────────────────────────────────────────────────

def gen_AO_Saucer(df, fast_p=5, slow_p=34, **kw):
    ao    = _ao(df, fast_p, slow_p)
    # Bullish saucer: 3 consecutive bars above zero; middle bar is the lowest
    bull_saucer = (
        (ao > 0) & (ao.shift(1) > 0) & (ao.shift(2) > 0) &
        (ao.shift(1) < ao) & (ao.shift(1) < ao.shift(2))
    )
    # Bearish saucer: 3 consecutive bars below zero; middle bar is the highest
    bear_saucer = (
        (ao < 0) & (ao.shift(1) < 0) & (ao.shift(2) < 0) &
        (ao.shift(1) > ao) & (ao.shift(1) > ao.shift(2))
    )
    out = pd.Series(0, index=df.index)
    out[bull_saucer] =  1
    out[bear_saucer] = -1
    return out.fillna(0).astype(int)


def space_AO_Saucer(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',  3,  8),
        'slow_p': trial.suggest_int('slow_p', 25, 40),
    }


# ── 21. AO_Twin_Peaks ────────────────────────────────────────────────────────

def gen_AO_Twin_Peaks(df, fast_p=5, slow_p=34, lookback=10, **kw):
    lookback = int(lookback)
    ao       = _ao(df, fast_p, slow_p)
    # Bullish twin peaks: two troughs below zero, second trough higher than first
    prev_min = ao.rolling(lookback, min_periods=1).min().shift(1)
    bull     = (ao < 0) & (ao > prev_min) & (prev_min < 0)
    # Bearish twin peaks: two peaks above zero, second peak lower than first
    prev_max = ao.rolling(lookback, min_periods=1).max().shift(1)
    bear     = (ao > 0) & (ao < prev_max) & (prev_max > 0)
    out      = pd.Series(0, index=df.index)
    out[bull] =  1
    out[bear] = -1
    return out.fillna(0).astype(int)


def space_AO_Twin_Peaks(trial):
    return {
        'fast_p':   trial.suggest_int('fast_p',   3,  8),
        'slow_p':   trial.suggest_int('slow_p',  25, 40),
        'lookback': trial.suggest_int('lookback',  5, 15),
    }


# ── 22. DeMarker_Strategy ─────────────────────────────────────────────────────

def gen_DeMarker_Strategy(df, dem_p=14, **kw):
    dem_p  = int(dem_p)
    demax  = (df['high'] - df['high'].shift(1)).clip(lower=0)
    demin  = (df['low'].shift(1) - df['low']).clip(lower=0)
    dem    = _sma(demax, dem_p) / (_sma(demax, dem_p) + _sma(demin, dem_p) + 1e-9)
    long   = dem < 0.3
    short  = dem > 0.7
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_DeMarker_Strategy(trial):
    return {'dem_p': trial.suggest_int('dem_p', 10, 20)}


# ── 23. DeMarker_EMA ──────────────────────────────────────────────────────────

def gen_DeMarker_EMA(df, dem_p=14, ema_p=50, **kw):
    dem_p  = int(dem_p)
    ema_p  = int(ema_p)
    demax  = (df['high'] - df['high'].shift(1)).clip(lower=0)
    demin  = (df['low'].shift(1) - df['low']).clip(lower=0)
    dem    = _sma(demax, dem_p) / (_sma(demax, dem_p) + _sma(demin, dem_p) + 1e-9)
    ema    = _ema(df['close'], ema_p)
    long   = (dem < 0.3) & (df['close'] > ema)
    short  = (dem > 0.7) & (df['close'] < ema)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_DeMarker_EMA(trial):
    return {
        'dem_p': trial.suggest_int('dem_p', 10,  20),
        'ema_p': trial.suggest_int('ema_p', 20, 100),
    }


# ── 24. RVI_Strategy ──────────────────────────────────────────────────────────

def gen_RVI_Strategy(df, rvi_p=10, **kw):
    rvi, sig = _rvi_calc(df, int(rvi_p))
    long     = (rvi > sig) & (rvi.shift(1) <= sig.shift(1))
    short    = (rvi < sig) & (rvi.shift(1) >= sig.shift(1))
    out      = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_RVI_Strategy(trial):
    return {'rvi_p': trial.suggest_int('rvi_p', 7, 20)}


# ── 25. RVI_RSI ───────────────────────────────────────────────────────────────

def gen_RVI_RSI(df, rvi_p=10, rsi_p=14, **kw):
    rvi, sig = _rvi_calc(df, int(rvi_p))
    rsi      = _rsi(df['close'], int(rsi_p))
    long     = (rvi > sig) & (rsi > 50)
    short    = (rvi < sig) & (rsi < 50)
    out      = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_RVI_RSI(trial):
    return {
        'rvi_p': trial.suggest_int('rvi_p', 7, 20),
        'rsi_p': trial.suggest_int('rsi_p', 7, 21),
    }


# ── 26. APO_Strategy ──────────────────────────────────────────────────────────

def gen_APO_Strategy(df, fast_p=12, slow_p=26, sig_p=9, **kw):
    fast_p = int(fast_p)
    slow_p = int(slow_p)
    sig_p  = int(sig_p)
    apo    = _ema(df['close'], fast_p) - _ema(df['close'], slow_p)
    sig    = _ema(apo, sig_p)
    # Long: APO crosses above signal from below 0
    long   = (apo > sig) & (apo.shift(1) <= sig.shift(1)) & (apo < 0)
    short  = (apo < sig) & (apo.shift(1) >= sig.shift(1)) & (apo > 0)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_APO_Strategy(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',  8, 16),
        'slow_p': trial.suggest_int('slow_p', 20, 35),
        'sig_p':  trial.suggest_int('sig_p',   5, 12),
    }


# ── 27. PPO_Strategy ──────────────────────────────────────────────────────────

def gen_PPO_Strategy(df, fast_p=12, slow_p=26, sig_p=9, **kw):
    fast_p = int(fast_p)
    slow_p = int(slow_p)
    sig_p  = int(sig_p)
    fast   = _ema(df['close'], fast_p)
    slow_v = _ema(df['close'], slow_p)
    ppo    = (fast - slow_v) / slow_v.replace(0, 1e-9) * 100.0
    sig    = _ema(ppo, sig_p)
    long   = (ppo > 0) & (ppo.shift(1) <= 0)
    short  = (ppo < 0) & (ppo.shift(1) >= 0)
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_PPO_Strategy(trial):
    return {
        'fast_p': trial.suggest_int('fast_p',  8, 16),
        'slow_p': trial.suggest_int('slow_p', 20, 35),
        'sig_p':  trial.suggest_int('sig_p',   5, 12),
    }


# ── 28. Fisher_Transform ──────────────────────────────────────────────────────

def gen_Fisher_Transform(df, fish_p=14, **kw):
    fish, trigger = _fisher(df, fish_p)
    long          = (fish > trigger) & (fish.shift(1) <= trigger.shift(1))
    short         = (fish < trigger) & (fish.shift(1) >= trigger.shift(1))
    out           = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_Fisher_Transform(trial):
    return {'fish_p': trial.suggest_int('fish_p', 7, 20)}


# ── 29. Fisher_ATR ────────────────────────────────────────────────────────────

def gen_Fisher_ATR(df, fish_p=14, atr_p=14, **kw):
    fish_p = int(fish_p)
    atr_p  = int(atr_p)
    fish, trigger = _fisher(df, fish_p)
    atr    = _atr(df, atr_p)
    atr_ma = _sma(atr, atr_p)
    vol_ok = atr > atr_ma
    long   = (fish > trigger) & (fish.shift(1) <= trigger.shift(1)) & vol_ok
    short  = (fish < trigger) & (fish.shift(1) >= trigger.shift(1)) & vol_ok
    out    = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_Fisher_ATR(trial):
    return {
        'fish_p': trial.suggest_int('fish_p', 7, 20),
        'atr_p':  trial.suggest_int('atr_p', 10, 20),
    }


# ── 30. Klinger_Signal ────────────────────────────────────────────────────────

def gen_Klinger_Signal(df, fast_p=34, slow_p=55, sig_p=13, **kw):
    ko, sig = _klinger(df, fast_p, slow_p, sig_p)
    long    = (ko > sig) & (ko.shift(1) <= sig.shift(1))
    short   = (ko < sig) & (ko.shift(1) >= sig.shift(1))
    out     = pd.Series(0, index=df.index)
    out[long]  =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_Klinger_Signal(trial):
    return {
        'fast_p': trial.suggest_int('fast_p', 30, 40),
        'slow_p': trial.suggest_int('slow_p', 50, 65),
        'sig_p':  trial.suggest_int('sig_p',  10, 15),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'CMO_Strategy': {
        'gen': gen_CMO_Strategy,
        'space': space_CMO_Strategy,
        'default_params': {'cmo_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'Chande Momentum Oscillator. Long: CMO crosses above -50. Short: crosses below 50.',
        },
    },
    'CMO_EMA': {
        'gen': gen_CMO_EMA,
        'space': space_CMO_EMA,
        'default_params': {'cmo_p': 14, 'ema_p': 34},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'CMO + EMA trend filter. Long: CMO > 0 AND close > EMA. Short: CMO < 0 AND close < EMA.',
        },
    },
    'CMO_Signal': {
        'gen': gen_CMO_Signal,
        'space': space_CMO_Signal,
        'default_params': {'cmo_p': 14, 'sig_p': 7},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1200,
            'description': 'CMO with its own EMA signal line. Long: CMO crosses above EMA(CMO). Short: crosses below.',
        },
    },
    'DPO_Strategy': {
        'gen': gen_DPO_Strategy,
        'space': space_DPO_Strategy,
        'default_params': {'dpo_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Detrended Price Oscillator removes trend to expose cycles. Long: DPO crosses above 0. Short: crosses below.',
        },
    },
    'DPO_RSI': {
        'gen': gen_DPO_RSI,
        'space': space_DPO_RSI,
        'default_params': {'dpo_p': 20, 'rsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1200,
            'description': 'DPO cycle direction combined with RSI. Long: DPO > 0 AND RSI < 50. Short: DPO < 0 AND RSI > 50.',
        },
    },
    'Ultimate_Oscillator': {
        'gen': gen_Ultimate_Oscillator,
        'space': space_Ultimate_Oscillator,
        'default_params': {'p1': 7, 'p2': 14, 'p3': 28},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 3000,
            'description': 'Williams Ultimate Oscillator: weighted buying pressure across 3 timeframes. Long: UO < 30. Short: UO > 70.',
        },
    },
    'UO_EMA': {
        'gen': gen_UO_EMA,
        'space': space_UO_EMA,
        'default_params': {'p1': 7, 'p2': 14, 'p3': 28, 'ema_p': 50},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'Ultimate Oscillator + EMA trend. Long: UO < 35 AND close > EMA. Short: UO > 65 AND close < EMA.',
        },
    },
    'UO_Divergence': {
        'gen': gen_UO_Divergence,
        'space': space_UO_Divergence,
        'default_params': {'p1': 7, 'p2': 14, 'p3': 28, 'lookback': 10},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1200,
            'description': 'UO divergence. Bull: price LL + UO HL. Bear: price HH + UO LH.',
        },
    },
    'Coppock_Curve': {
        'gen': gen_Coppock_Curve,
        'space': space_Coppock_Curve,
        'default_params': {'roc1': 14, 'roc2': 11, 'wma_p': 10},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'Coppock Curve = WMA(ROC(14)+ROC(11), 10). Long: CC crosses above 0. Short: crosses below.',
        },
    },
    'Coppock_EMA': {
        'gen': gen_Coppock_EMA,
        'space': space_Coppock_EMA,
        'default_params': {'roc1': 14, 'roc2': 11, 'wma_p': 10, 'ema_p': 100},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1200,
            'description': 'Coppock Curve + EMA trend filter. Long: CC > 0 AND close > EMA. Short: CC < 0 AND close < EMA.',
        },
    },
    'TRIX_Strategy': {
        'gen': gen_TRIX_Strategy,
        'space': space_TRIX_Strategy,
        'default_params': {'trix_p': 14, 'sig_p': 5},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'TRIX: 1-bar pct change of triple EMA. Long: TRIX crosses above 0. Short: crosses below.',
        },
    },
    'TRIX_Signal': {
        'gen': gen_TRIX_Signal,
        'space': space_TRIX_Signal,
        'default_params': {'trix_p': 14, 'sig_p': 5},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'TRIX + EMA signal line. Long: TRIX crosses above signal. Short: crosses below.',
        },
    },
    'TRIX_Histogram': {
        'gen': gen_TRIX_Histogram,
        'space': space_TRIX_Histogram,
        'default_params': {'trix_p': 14, 'sig_p': 5},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1000,
            'description': 'TRIX histogram (TRIX - signal). Long: histogram > 0 AND rising. Short: < 0 AND falling.',
        },
    },
    'ROC_Strategy': {
        'gen': gen_ROC_Strategy,
        'space': space_ROC_Strategy,
        'default_params': {'roc_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Rate of Change zero cross. Long: ROC crosses above 0 AND rising. Short: crosses below 0 AND falling.',
        },
    },
    'ROC_EMA': {
        'gen': gen_ROC_EMA,
        'space': space_ROC_EMA,
        'default_params': {'roc_p': 20, 'ema_p': 50},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'ROC + EMA trend filter. Long: ROC > 0 AND close > EMA. Short: ROC < 0 AND close < EMA.',
        },
    },
    'ROC_RSI': {
        'gen': gen_ROC_RSI,
        'space': space_ROC_RSI,
        'default_params': {'roc_p': 20, 'rsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1000,
            'description': 'ROC direction + RSI momentum band (40-65 for longs). Filters out extremes.',
        },
    },
    'Momentum_OSC': {
        'gen': gen_Momentum_OSC,
        'space': space_Momentum_OSC,
        'default_params': {'mom_p': 20, 'sig_p': 10},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Momentum oscillator with EMA signal. Long: momentum > 0 AND crosses above its EMA signal.',
        },
    },
    'AO_Classic': {
        'gen': gen_AO_Classic,
        'space': space_AO_Classic,
        'default_params': {'fast_p': 5, 'slow_p': 34},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 3000,
            'description': 'Awesome Oscillator classic: SMA(hl2,5) - SMA(hl2,34). Long: AO > 0 and accelerating or zero cross.',
        },
    },
    'AO_Divergence': {
        'gen': gen_AO_Divergence,
        'space': space_AO_Divergence,
        'default_params': {'fast_p': 5, 'slow_p': 34, 'lookback': 10},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'AO divergence. Bull: price LL + AO HL. Bear: price HH + AO LH.',
        },
    },
    'AO_Saucer': {
        'gen': gen_AO_Saucer,
        'space': space_AO_Saucer,
        'default_params': {'fast_p': 5, 'slow_p': 34},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'AO saucer pattern: 3 bars on same side of zero with middle as lowest (bull) or highest (bear).',
        },
    },
    'AO_Twin_Peaks': {
        'gen': gen_AO_Twin_Peaks,
        'space': space_AO_Twin_Peaks,
        'default_params': {'fast_p': 5, 'slow_p': 34, 'lookback': 10},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1200,
            'description': 'AO twin peaks: two troughs below zero with second higher. Long on second trough signal.',
        },
    },
    'DeMarker_Strategy': {
        'gen': gen_DeMarker_Strategy,
        'space': space_DeMarker_Strategy,
        'default_params': {'dem_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'DeMarker oscillator. Long: DeM < 0.3 (oversold). Short: DeM > 0.7 (overbought).',
        },
    },
    'DeMarker_EMA': {
        'gen': gen_DeMarker_EMA,
        'space': space_DeMarker_EMA,
        'default_params': {'dem_p': 14, 'ema_p': 50},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1200,
            'description': 'DeMarker + EMA trend filter. Long: DeM < 0.3 AND close > EMA. Short: DeM > 0.7 AND close < EMA.',
        },
    },
    'RVI_Strategy': {
        'gen': gen_RVI_Strategy,
        'space': space_RVI_Strategy,
        'default_params': {'rvi_p': 10},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Relative Vigor Index signal cross. Long: RVI crosses above 4-bar WMA signal. Short: crosses below.',
        },
    },
    'RVI_RSI': {
        'gen': gen_RVI_RSI,
        'space': space_RVI_RSI,
        'default_params': {'rvi_p': 10, 'rsi_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1200,
            'description': 'RVI + RSI confirmation. Long: RVI > signal AND RSI > 50. Short: RVI < signal AND RSI < 50.',
        },
    },
    'APO_Strategy': {
        'gen': gen_APO_Strategy,
        'space': space_APO_Strategy,
        'default_params': {'fast_p': 12, 'slow_p': 26, 'sig_p': 9},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'Absolute Price Oscillator signal cross from below zero. Long: APO crosses above signal while APO < 0.',
        },
    },
    'PPO_Strategy': {
        'gen': gen_PPO_Strategy,
        'space': space_PPO_Strategy,
        'default_params': {'fast_p': 12, 'slow_p': 26, 'sig_p': 9},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'Percentage Price Oscillator zero-line cross. Long: PPO crosses above 0. Short: crosses below.',
        },
    },
    'Fisher_Transform': {
        'gen': gen_Fisher_Transform,
        'space': space_Fisher_Transform,
        'default_params': {'fish_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 3500,
            'description': 'Ehlers Fisher Transform. Long: Fisher crosses above trigger from below. Short: crosses below trigger.',
        },
    },
    'Fisher_ATR': {
        'gen': gen_Fisher_ATR,
        'space': space_Fisher_ATR,
        'default_params': {'fish_p': 14, 'atr_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1800,
            'description': 'Fisher Transform + ATR volatility filter. Long: Fisher bullish cross AND ATR > ATR_SMA.',
        },
    },
    'Klinger_Signal': {
        'gen': gen_Klinger_Signal,
        'space': space_Klinger_Signal,
        'default_params': {'fast_p': 34, 'slow_p': 55, 'sig_p': 13},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Klinger Volume Oscillator signal cross. Long: KO crosses above signal EMA(13). Short: crosses below.',
        },
    },
}
