#!/usr/bin/env python3
"""
TV2 BATCH 11b — 9 estrategias Elder/DEMA/Chop/Pivot/CMO 2026-03-31

  Elder_Ray_Bull       — Elder Ray Bull Power + EMA (v5, 1425L)
  DEMA_ATR_Dashboard   — DEMA ATR trailing + dashboard (v6, 1438L)
  SR_5min_Intraday     — S/R pivots + EMA-150 ATR trail (v5, 1402L)
  Turtle_Triple_EMA    — Turtle concept + Triple EMA + ADX (v4, 1398L)
  Triple_DEMA_8_20_63  — Triple DEMA alignment (v4, 1341L)
  Chop_DMI_PSAR        — Choppiness + DMI entry + PSAR exit (v4, 1360L)
  Chande_Momentum      — CMO + EMA signal (v5, 1152L)
  Pivot_Of_Pivot       — Pivot-of-pivot reversal (v4, 1146L)
  SR_Ceyhun            — S/R with SMA-10 dynamic (v4, 1306L)
"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def _dema(series, period):
    e = _ema(series, period)
    return 2 * e - _ema(e, period)


def _rma(series, period):
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _atr(df, period):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, period)


def _adx_dmi(df, period):
    tr   = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr  = _rma(tr, period)
    up   = df['high'].diff()
    dn   = -df['low'].diff()
    pdm  = ((up > dn) & (up > 0)) * up
    ndm  = ((dn > up) & (dn > 0)) * dn
    pdi  = 100 * _rma(pdm, period) / atr.replace(0, np.nan)
    ndi  = 100 * _rma(ndm, period) / atr.replace(0, np.nan)
    dx   = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx  = _rma(dx, period)
    return pdi, ndi, adx


def _psar(df, start=0.02, increment=0.02, maximum=0.2):
    hi, lo = df['high'].values, df['low'].values
    n = len(hi)
    sar, bull, af, ep = np.zeros(n), np.ones(n, dtype=bool), np.full(n, start), np.zeros(n)
    sar[0] = lo[0]; ep[0] = hi[0]
    for i in range(1, n):
        if bull[i-1]:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = min(sar[i], lo[i-1], lo[i-2] if i >= 2 else lo[i-1])
            if hi[i] > ep[i-1]:
                ep[i] = hi[i]; af[i] = min(af[i-1] + increment, maximum)
            else:
                ep[i] = ep[i-1]; af[i] = af[i-1]
            bull[i] = lo[i] >= sar[i]
            if not bull[i]:
                sar[i] = ep[i-1]; ep[i] = lo[i]; af[i] = start
        else:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = max(sar[i], hi[i-1], hi[i-2] if i >= 2 else hi[i-1])
            if lo[i] < ep[i-1]:
                ep[i] = lo[i]; af[i] = min(af[i-1] + increment, maximum)
            else:
                ep[i] = ep[i-1]; af[i] = af[i-1]
            bull[i] = hi[i] > sar[i]
            if bull[i]:
                sar[i] = ep[i-1]; ep[i] = hi[i]; af[i] = start
    return pd.Series(bull.astype(float), index=df.index)


def _choppiness(df, period):
    """Choppiness Index: 100 × log10(sum(ATR14, n) / (HH-LL)) / log10(n)."""
    n     = int(period)
    tr    = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_sum = tr.rolling(n).sum()
    hh      = df['high'].rolling(n).max()
    ll      = df['low'].rolling(n).min()
    rang    = (hh - ll).replace(0, np.nan)
    ci      = 100 * np.log10(atr_sum / rang) / np.log10(n)
    return ci


# ── 1. Elder Ray Bull Power ───────────────────────────────────────────────────

def gen_Elder_Ray_Bull(df, ema_period=13, ob_level=0, os_level=0, **kw):
    """Elder Ray: Bull Power = High - EMA(13). Long when BP crosses above 0."""
    ema      = _ema(df['close'], int(ema_period))
    bull_pwr = df['high'] - ema
    bear_pwr = df['low']  - ema

    long_cond  = (bull_pwr > ob_level) & (bull_pwr.shift(1) <= ob_level) & (df['close'] > ema)
    short_cond = (bear_pwr < os_level) & (bear_pwr.shift(1) >= os_level) & (df['close'] < ema)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Elder_Ray_Bull(trial):
    return {
        'ema_period': trial.suggest_int('ema_period', 8, 21),
        'ob_level':   trial.suggest_float('ob_level', -0.5, 0.5),
        'os_level':   trial.suggest_float('os_level', -0.5, 0.5),
    }


# ── 2. DEMA ATR Dashboard [PrimeAutomation] ──────────────────────────────────

def gen_DEMA_ATR_Dashboard(df, dema_period=20, atr_period=14, atr_mult=2.0, **kw):
    """DEMA direction + ATR trailing stop."""
    dema     = _dema(df['close'], int(dema_period))
    atr      = _atr(df, int(atr_period))
    trail    = df['close'] - atr_mult * atr

    long_cond  = (dema > dema.shift(1)) & (dema.shift(1) <= dema.shift(2)) & (df['close'] > trail)
    short_cond = (dema < dema.shift(1)) & (dema.shift(1) >= dema.shift(2)) & (df['close'] < (df['close'] + atr_mult * atr))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_DEMA_ATR_Dashboard(trial):
    return {
        'dema_period': trial.suggest_int('dema_period', 10, 50),
        'atr_period':  trial.suggest_int('atr_period', 7, 21),
        'atr_mult':    trial.suggest_float('atr_mult', 1.0, 4.0),
    }


# ── 3. Support/Resistance 5min Intraday ──────────────────────────────────────

def gen_SR_5min_Intraday(df, pivot_period=5, ema_period=150, atr_period=14, atr_mult=1.5, **kw):
    """Pivot S/R + EMA-150 trend + ATR trailing."""
    pp    = int(pivot_period)
    ema   = _ema(df['close'], int(ema_period))
    atr   = _atr(df, int(atr_period))
    ph    = df['high'].rolling(pp * 2 + 1, center=True).max()
    pl    = df['low'].rolling(pp * 2 + 1, center=True).min()
    resist = df['high'].where(df['high'] == ph).fillna(method='ffill')
    supprt = df['low'].where(df['low'] == pl).fillna(method='ffill')

    long_cond  = (df['close'] > ema) & (df['close'] > supprt) & \
                 (df['close'].shift(1) <= supprt.shift(1))
    short_cond = (df['close'] < ema) & (df['close'] < resist) & \
                 (df['close'].shift(1) >= resist.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_SR_5min_Intraday(trial):
    return {
        'pivot_period': trial.suggest_int('pivot_period', 3, 15),
        'ema_period':   trial.suggest_int('ema_period', 50, 250),
        'atr_period':   trial.suggest_int('atr_period', 7, 21),
        'atr_mult':     trial.suggest_float('atr_mult', 1.0, 3.0),
    }


# ── 4. Turtle Triple EMA + ADX ───────────────────────────────────────────────

def gen_Turtle_Triple_EMA(df, e1=10, e2=20, e3=50, adx_period=14, adx_thresh=20,
                           bo_period=20, **kw):
    """Turtle breakout filtered by triple EMA alignment + ADX."""
    ema1 = _ema(df['close'], int(e1))
    ema2 = _ema(df['close'], int(e2))
    ema3 = _ema(df['close'], int(e3))
    _, _, adx = _adx_dmi(df, int(adx_period))
    hh = df['high'].rolling(int(bo_period)).max().shift(1)
    ll = df['low'].rolling(int(bo_period)).min().shift(1)

    bull_align = (ema1 > ema2) & (ema2 > ema3)
    bear_align = (ema1 < ema2) & (ema2 < ema3)
    trending   = adx > adx_thresh

    sig = pd.Series(0, index=df.index)
    sig[(df['close'] > hh) & bull_align & trending] =  1
    sig[(df['close'] < ll) & bear_align & trending] = -1
    return sig


def space_Turtle_Triple_EMA(trial):
    return {
        'e1':         trial.suggest_int('e1', 5, 15),
        'e2':         trial.suggest_int('e2', 15, 30),
        'e3':         trial.suggest_int('e3', 30, 80),
        'adx_period': trial.suggest_int('adx_period', 10, 21),
        'adx_thresh': trial.suggest_int('adx_thresh', 15, 35),
        'bo_period':  trial.suggest_int('bo_period', 10, 40),
    }


# ── 5. Triple DEMA 8/20/63 ───────────────────────────────────────────────────

def gen_Triple_DEMA_8_20_63(df, d1=8, d2=20, d3=63, **kw):
    """Long when DEMA8 > DEMA20 > DEMA63 (all aligned bullish)."""
    dema1 = _dema(df['close'], int(d1))
    dema2 = _dema(df['close'], int(d2))
    dema3 = _dema(df['close'], int(d3))

    long_cond  = (dema1 > dema2) & (dema2 > dema3)
    short_cond = (dema1 < dema2) & (dema2 < dema3)
    # Only enter on transition
    prev_long  = long_cond.shift(1).fillna(False)
    prev_short = short_cond.shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    sig[long_cond  & ~prev_long]  =  1
    sig[short_cond & ~prev_short] = -1
    return sig


def space_Triple_DEMA_8_20_63(trial):
    return {
        'd1': trial.suggest_int('d1', 5, 15),
        'd2': trial.suggest_int('d2', 15, 30),
        'd3': trial.suggest_int('d3', 40, 100),
    }


# ── 6. Chop Zone + DMI + PSAR ────────────────────────────────────────────────

def gen_Chop_DMI_PSAR(df, chop_period=14, chop_thresh=50, dmi_period=14,
                       sar_start=0.02, sar_inc=0.02, sar_max=0.2, **kw):
    """Choppiness < threshold = trending. DMI for direction, PSAR for exit."""
    ci          = _choppiness(df, int(chop_period))
    pdi, ndi, _ = _adx_dmi(df, int(dmi_period))
    bull_sar    = _psar(df, sar_start, sar_inc, sar_max)
    trending    = ci < chop_thresh

    long_cond  = trending & (pdi > ndi) & (bull_sar == 1) & \
                 ((pdi.shift(1) <= ndi.shift(1)) | (bull_sar.shift(1) == 0))
    short_cond = trending & (ndi > pdi) & (bull_sar == 0) & \
                 ((ndi.shift(1) <= pdi.shift(1)) | (bull_sar.shift(1) == 1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Chop_DMI_PSAR(trial):
    return {
        'chop_period': trial.suggest_int('chop_period', 10, 21),
        'chop_thresh': trial.suggest_int('chop_thresh', 38, 62),
        'dmi_period':  trial.suggest_int('dmi_period', 10, 21),
        'sar_start':   trial.suggest_float('sar_start', 0.01, 0.05),
        'sar_inc':     trial.suggest_float('sar_inc', 0.01, 0.05),
        'sar_max':     trial.suggest_float('sar_max', 0.1, 0.4),
    }


# ── 7. Chande Momentum Oscillator [TradeDots] ────────────────────────────────

def _cmo(close, period):
    delta = close.diff()
    up    = delta.clip(lower=0).rolling(period).sum()
    dn    = (-delta.clip(upper=0)).rolling(period).sum()
    total = (up + dn).replace(0, np.nan)
    return 100 * (up - dn) / total


def gen_Chande_Momentum(df, cmo_period=9, signal_period=3, ob=50, os=-50, **kw):
    """CMO + EMA signal line. Long on CMO crossing above os from below."""
    cmo    = _cmo(df['close'], int(cmo_period))
    signal = _ema(cmo, int(signal_period))

    long_cond  = (cmo > signal) & (cmo.shift(1) <= signal.shift(1)) & (cmo < ob)
    short_cond = (cmo < signal) & (cmo.shift(1) >= signal.shift(1)) & (cmo > os)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Chande_Momentum(trial):
    return {
        'cmo_period':    trial.suggest_int('cmo_period', 5, 21),
        'signal_period': trial.suggest_int('signal_period', 2, 9),
        'ob':            trial.suggest_int('ob', 30, 70),
        'os':            trial.suggest_int('os', -70, -30),
    }


# ── 8. Pivot of Pivot Reversal [QuantNomad] ──────────────────────────────────

def gen_Pivot_Of_Pivot(df, left=5, right=5, level=2, **kw):
    """Second-level pivots (extremes of extremes)."""
    lft, rgt = int(left), int(right)
    w = lft + rgt + 1

    # First-level pivots
    ph1 = df['high'].rolling(w, center=True).max() == df['high']
    pl1 = df['low'].rolling(w, center=True).min()  == df['low']

    # Second-level: pivot high of pivot highs
    pivot_h_vals = df['high'].where(ph1)
    pivot_l_vals = df['low'].where(pl1)
    ph2_vals = pivot_h_vals.rolling(w * level, min_periods=1).max()
    pl2_vals = pivot_l_vals.rolling(w * level, min_periods=1).min()

    is_ph2 = ph1 & (df['high'] == ph2_vals)
    is_pl2 = pl1 & (df['low']  == pl2_vals)

    sig = pd.Series(0, index=df.index)
    sig[is_pl2] =  1   # bounce from double-bottom
    sig[is_ph2] = -1   # reversal from double-top
    return sig


def space_Pivot_Of_Pivot(trial):
    return {
        'left':  trial.suggest_int('left', 3, 10),
        'right': trial.suggest_int('right', 3, 10),
        'level': trial.suggest_int('level', 1, 4),
    }


# ── 9. S/R Ceyhun (SMA-10 dynamic) ───────────────────────────────────────────

def gen_SR_Ceyhun(df, sma_period=10, atr_period=14, bo_pct=0.3, **kw):
    """SMA-based dynamic S/R. Price breakout above/below SMA with ATR confirm."""
    sma   = df['close'].rolling(int(sma_period)).mean()
    atr   = _atr(df, int(atr_period))
    band  = atr * bo_pct

    long_cond  = (df['close'] > sma + band) & (df['close'].shift(1) <= sma.shift(1) + band.shift(1))
    short_cond = (df['close'] < sma - band) & (df['close'].shift(1) >= sma.shift(1) - band.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_SR_Ceyhun(trial):
    return {
        'sma_period': trial.suggest_int('sma_period', 5, 30),
        'atr_period': trial.suggest_int('atr_period', 7, 21),
        'bo_pct':     trial.suggest_float('bo_pct', 0.1, 1.0),
    }


# ── STRATEGY_EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Elder_Ray_Bull': {
        'gen':            gen_Elder_Ray_Bull,
        'space':          space_Elder_Ray_Bull,
        'default_params': {'ema_period': 13, 'ob_level': 0, 'os_level': 0},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1425,
                           'description': 'Elder Ray Bull Power vs EMA'},
    },
    'DEMA_ATR_Dashboard': {
        'gen':            gen_DEMA_ATR_Dashboard,
        'space':          space_DEMA_ATR_Dashboard,
        'default_params': {'dema_period': 20, 'atr_period': 14, 'atr_mult': 2.0},
        'info':           {'source': 'TradingView', 'version': 6, 'likes': 1438,
                           'description': 'DEMA ATR trailing + PnL dashboard'},
    },
    'SR_5min_Intraday': {
        'gen':            gen_SR_5min_Intraday,
        'space':          space_SR_5min_Intraday,
        'default_params': {'pivot_period': 5, 'ema_period': 150, 'atr_period': 14, 'atr_mult': 1.5},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1402,
                           'description': 'S/R pivots + EMA-150 + ATR trail intraday'},
    },
    'Turtle_Triple_EMA': {
        'gen':            gen_Turtle_Triple_EMA,
        'space':          space_Turtle_Triple_EMA,
        'default_params': {'e1': 10, 'e2': 20, 'e3': 50, 'adx_period': 14,
                           'adx_thresh': 20, 'bo_period': 20},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 1398,
                           'description': 'Turtle breakout + Triple EMA + ADX'},
    },
    'Triple_DEMA_8_20_63': {
        'gen':            gen_Triple_DEMA_8_20_63,
        'space':          space_Triple_DEMA_8_20_63,
        'default_params': {'d1': 8, 'd2': 20, 'd3': 63},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 1341,
                           'description': 'Triple DEMA alignment 8/20/63'},
    },
    'Chop_DMI_PSAR': {
        'gen':            gen_Chop_DMI_PSAR,
        'space':          space_Chop_DMI_PSAR,
        'default_params': {'chop_period': 14, 'chop_thresh': 50, 'dmi_period': 14,
                           'sar_start': 0.02, 'sar_inc': 0.02, 'sar_max': 0.2},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 1360,
                           'description': 'Choppiness filter + DMI entry + PSAR exit'},
    },
    'Chande_Momentum': {
        'gen':            gen_Chande_Momentum,
        'space':          space_Chande_Momentum,
        'default_params': {'cmo_period': 9, 'signal_period': 3, 'ob': 50, 'os': -50},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1152,
                           'description': 'Chande Momentum Oscillator + EMA signal [TradeDots]'},
    },
    'Pivot_Of_Pivot': {
        'gen':            gen_Pivot_Of_Pivot,
        'space':          space_Pivot_Of_Pivot,
        'default_params': {'left': 5, 'right': 5, 'level': 2},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 1146,
                           'description': 'Pivot of pivot reversal [QuantNomad]'},
    },
    'SR_Ceyhun': {
        'gen':            gen_SR_Ceyhun,
        'space':          space_SR_Ceyhun,
        'default_params': {'sma_period': 10, 'atr_period': 14, 'bo_pct': 0.3},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 1306,
                           'description': 'S/R with SMA dynamic [ceyhun]'},
    },
}
