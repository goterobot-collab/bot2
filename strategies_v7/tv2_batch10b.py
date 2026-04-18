#!/usr/bin/env python3
"""
TV2 BATCH 10b — 10 VWAP/MACD/Volume estrategias 2026-03-31

  VWAP_Trendfollow        — VWAP SD band breakout (v5, 3842L)
  Price_Volume_Breakout   — Price + Volume breakout (v5, 4771L)
  Full_Crypto_Pack        — EMA+MACD+RSI+OBV (v4, 4860L)
  MACD_Reloaded           — Multi-MA MACD (v5, 7478L)
  MACD_ATR_Trail          — MACD + ATR trailing SL (v5, 2344L)
  MACD_BB_RSI_Alorse      — MACD + BB + RSI triple (v4, 2028L)
  Scalping_Crypto_Stocks  — SMA+EMA+Keltner+Stoch+MACD (v5, 3223L)
  Keltner_ETH_DMI         — Keltner Channel + DMI/ADX (v5, 2390L)
  Squeeze_Momentum_TPSL   — BB + Keltner Squeeze (v5, 1918L)
  VWAP_Momentum_Pullback  — VWAP momentum pullback (v5, 1425L)
"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def _rma(series, period):
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _atr(df, period):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, period)


def _macd(close, fast=12, slow=26, sig=9):
    e_fast = _ema(close, fast)
    e_slow = _ema(close, slow)
    line   = e_fast - e_slow
    signal = _ema(line, sig)
    hist   = line - signal
    return line, signal, hist


def _rsi(close, period):
    d     = close.diff()
    gain  = d.clip(lower=0).rolling(period).mean()
    loss  = (-d.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _bb(close, period, std_mult):
    sma  = close.rolling(period).mean()
    std  = close.rolling(period).std()
    return sma + std_mult * std, sma, sma - std_mult * std


def _keltner(df, period, mult):
    atr     = _atr(df, period)
    mid     = _ema(df['close'], period)
    return mid + mult * atr, mid, mid - mult * atr


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


def _stoch(df, k_period, d_period):
    lo = df['low'].rolling(k_period).min()
    hi = df['high'].rolling(k_period).max()
    k  = 100 * (df['close'] - lo) / (hi - lo).replace(0, np.nan)
    d  = k.rolling(d_period).mean()
    return k, d


# ── 1. VWAP Trendfollow [wbburgin] ───────────────────────────────────────────

def gen_VWAP_Trendfollow(df, vwap_period=20, sd_mult=0.88, **kw):
    """Rolling VWAP + SD bands. Entry on upper/lower band breakout."""
    vp    = int(vwap_period)
    pv    = df['close'] * df['volume']
    vwap  = pv.rolling(vp).sum() / df['volume'].rolling(vp).sum().replace(0, np.nan)
    std   = df['close'].rolling(vp).std()
    upper = vwap + sd_mult * std
    lower = vwap - sd_mult * std

    sig = pd.Series(0, index=df.index)
    sig[df['close'] > upper.shift(1)] =  1
    sig[df['close'] < lower.shift(1)] = -1
    return sig


def space_VWAP_Trendfollow(trial):
    return {
        'vwap_period': trial.suggest_int('vwap_period', 10, 50),
        'sd_mult':     trial.suggest_float('sd_mult', 0.5, 2.5),
    }


# ── 2. Price + Volume Breakout [TradeDots] ───────────────────────────────────

def gen_Price_Volume_Breakout(df, bo_period=60, vol_period=60, trend_period=200, **kw):
    """Price breakout above N-bar high + volume > N-bar avg + above SMA trend filter."""
    bo  = int(bo_period)
    vp  = int(vol_period)
    tp  = int(trend_period)

    price_bo  = df['close'] > df['close'].rolling(bo).max().shift(1)
    price_bd  = df['close'] < df['close'].rolling(bo).min().shift(1)
    vol_bo    = df['volume'] > df['volume'].rolling(vp).mean()
    trend_ma  = df['close'].rolling(tp).mean()

    sig = pd.Series(0, index=df.index)
    sig[price_bo & vol_bo & (df['close'] > trend_ma)]  =  1
    sig[price_bd & vol_bo & (df['close'] < trend_ma)]  = -1
    return sig


def space_Price_Volume_Breakout(trial):
    return {
        'bo_period':    trial.suggest_int('bo_period', 20, 100),
        'vol_period':   trial.suggest_int('vol_period', 20, 100),
        'trend_period': trial.suggest_int('trend_period', 100, 300),
    }


# ── 3. Full CRYPTO pack EMA+MACD+RSI+OBV [SoftKill21] ───────────────────────

def gen_Full_Crypto_Pack(df, ema_period=9, macd_fast=12, macd_slow=26, macd_sig=9,
                         rsi_period=14, rsi_os=35, **kw):
    ema       = _ema((df['high'] + df['low']) / 2, int(ema_period))
    _, _, hist = _macd(df['close'], int(macd_fast), int(macd_slow), int(macd_sig))
    rsi       = _rsi(df['close'], int(rsi_period))
    obv       = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    obv_ma    = _ema(obv, 20)

    long_cond  = (df['close'] > ema) & (hist > 0) & (rsi < 100 - rsi_os) & (obv > obv_ma)
    short_cond = (df['close'] < ema) & (hist < 0) & (rsi > rsi_os) & (obv < obv_ma)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Full_Crypto_Pack(trial):
    return {
        'ema_period': trial.suggest_int('ema_period', 5, 20),
        'macd_fast':  trial.suggest_int('macd_fast', 8, 16),
        'macd_slow':  trial.suggest_int('macd_slow', 20, 34),
        'macd_sig':   trial.suggest_int('macd_sig', 7, 13),
        'rsi_period': trial.suggest_int('rsi_period', 10, 21),
        'rsi_os':     trial.suggest_int('rsi_os', 25, 45),
    }


# ── 4. MACD Reloaded [kivancozbilgic] ────────────────────────────────────────

def _wma(series, period):
    w = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def _hull_ma(series, period):
    half = max(int(period / 2), 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    return _wma(2 * _wma(series, half) - _wma(series, period), sqrt_p)


def _dema(series, period):
    e = _ema(series, period)
    return 2 * e - _ema(e, period)


def gen_MACD_Reloaded(df, ma_type=0, fast=12, slow=26, sig_p=9, **kw):
    """Multi-MA MACD. ma_type: 0=EMA, 1=SMA, 2=WMA, 3=DEMA, 4=HULL."""
    mt = int(ma_type) % 5
    fast, slow, sig_p = int(fast), int(slow), int(sig_p)
    c = df['close']

    def _ma(s, p):
        if   mt == 0: return _ema(s, p)
        elif mt == 1: return s.rolling(p).mean()
        elif mt == 2: return _wma(s, p)
        elif mt == 3: return _dema(s, p)
        else:         return _hull_ma(s, p)

    line   = _ma(c, fast) - _ma(c, slow)
    signal = _ema(line, sig_p)
    hist   = line - signal

    sig = pd.Series(0, index=df.index)
    sig[(hist > 0) & (hist.shift(1) <= 0)] =  1
    sig[(hist < 0) & (hist.shift(1) >= 0)] = -1
    return sig


def space_MACD_Reloaded(trial):
    return {
        'ma_type': trial.suggest_int('ma_type', 0, 4),
        'fast':    trial.suggest_int('fast', 8, 16),
        'slow':    trial.suggest_int('slow', 20, 34),
        'sig_p':   trial.suggest_int('sig_p', 7, 13),
    }


# ── 5. MACD + ATR Trailing SL [Deobald] ─────────────────────────────────────

def gen_MACD_ATR_Trail(df, ma_period=34, macd_fast=12, macd_slow=26, macd_sig=9,
                       atr_period=14, atr_mult=2.2, **kw):
    ma        = _ema(df['close'], int(ma_period))
    line, sig_line, _ = _macd(df['close'], int(macd_fast), int(macd_slow), int(macd_sig))
    atr       = _atr(df, int(atr_period))
    trail_sl  = df['close'] - atr_mult * atr

    long_cond  = (df['close'] > ma) & (line > sig_line) & (line > line.shift(1))
    short_cond = (df['close'] < ma) & (line < sig_line) & (line < line.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_MACD_ATR_Trail(trial):
    return {
        'ma_period':  trial.suggest_int('ma_period', 20, 55),
        'macd_fast':  trial.suggest_int('macd_fast', 8, 16),
        'macd_slow':  trial.suggest_int('macd_slow', 20, 34),
        'macd_sig':   trial.suggest_int('macd_sig', 7, 13),
        'atr_period': trial.suggest_int('atr_period', 10, 21),
        'atr_mult':   trial.suggest_float('atr_mult', 1.0, 4.0),
    }


# ── 6. MACD + BB + RSI [Alorse] ──────────────────────────────────────────────

def gen_MACD_BB_RSI_Alorse(df, macd_fast=12, macd_slow=26, macd_sig=9,
                            bb_period=20, bb_std=2.0, rsi_period=14, **kw):
    line, sig_line, hist = _macd(df['close'], int(macd_fast), int(macd_slow), int(macd_sig))
    upper, mid, lower    = _bb(df['close'], int(bb_period), bb_std)
    rsi                  = _rsi(df['close'], int(rsi_period))

    long_cond  = (hist > 0) & (df['close'] < mid) & (rsi < 50)
    short_cond = (hist < 0) & (df['close'] > mid) & (rsi > 50)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_MACD_BB_RSI_Alorse(trial):
    return {
        'macd_fast':  trial.suggest_int('macd_fast', 8, 16),
        'macd_slow':  trial.suggest_int('macd_slow', 20, 34),
        'macd_sig':   trial.suggest_int('macd_sig', 7, 13),
        'bb_period':  trial.suggest_int('bb_period', 15, 30),
        'bb_std':     trial.suggest_float('bb_std', 1.5, 3.0),
        'rsi_period': trial.suggest_int('rsi_period', 10, 21),
    }


# ── 7. Scalping Crypto + Stocks (SMA+EMA+Keltner+Stoch+MACD) ─────────────────

def gen_Scalping_Crypto_Stocks(df, sma_period=25, ema_period=200,
                                kc_period=10, kc_mult=2.0,
                                stoch_k=14, stoch_d=3,
                                macd_fast=4, macd_slow=34, macd_sig=5, **kw):
    sma       = df['close'].rolling(int(sma_period)).mean()
    ema       = _ema(df['close'], int(ema_period))
    ku, km, kl = _keltner(df, int(kc_period), kc_mult)
    k, d      = _stoch(df, int(stoch_k), int(stoch_d))
    _, _, hist = _macd(df['close'], int(macd_fast), int(macd_slow), int(macd_sig))

    long_cond  = (df['close'] > sma) & (df['close'] > kl) & (df['close'] < ku) \
                 & (hist < 0) & (k < 50) & (df['close'] > ema)
    short_cond = (df['close'] < sma) & (df['close'] < ku) & (df['close'] > kl) \
                 & (hist > 0) & (k > 50) & (df['close'] < ema)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Scalping_Crypto_Stocks(trial):
    return {
        'sma_period': trial.suggest_int('sma_period', 15, 40),
        'ema_period': trial.suggest_int('ema_period', 100, 300),
        'kc_period':  trial.suggest_int('kc_period', 7, 20),
        'kc_mult':    trial.suggest_float('kc_mult', 1.0, 3.5),
        'stoch_k':    trial.suggest_int('stoch_k', 7, 21),
        'stoch_d':    trial.suggest_int('stoch_d', 2, 5),
        'macd_fast':  trial.suggest_int('macd_fast', 3, 8),
        'macd_slow':  trial.suggest_int('macd_slow', 20, 50),
        'macd_sig':   trial.suggest_int('macd_sig', 3, 9),
    }


# ── 8. Keltner Channel ETH/USDT [Wunderbit] ──────────────────────────────────

def gen_Keltner_ETH_DMI(df, kc_period=46, kc_mult=1.5, adx_period=14, adx_thresh=20, **kw):
    ku, km, kl   = _keltner(df, int(kc_period), kc_mult)
    pdi, ndi, adx = _adx_dmi(df, int(adx_period))

    long_cond  = (df['close'] > ku) & (pdi > ndi) & (adx > adx_thresh)
    short_cond = (df['close'] < kl) & (ndi > pdi) & (adx > adx_thresh)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_Keltner_ETH_DMI(trial):
    return {
        'kc_period':  trial.suggest_int('kc_period', 20, 80),
        'kc_mult':    trial.suggest_float('kc_mult', 0.5, 3.0),
        'adx_period': trial.suggest_int('adx_period', 10, 21),
        'adx_thresh': trial.suggest_int('adx_thresh', 15, 35),
    }


# ── 9. Squeeze Momentum BB+Keltner ───────────────────────────────────────────

def gen_Squeeze_Momentum_TPSL(df, bb_period=20, bb_std=2.0, kc_period=20, kc_mult=1.5,
                               mom_period=12, **kw):
    """BB inside Keltner = squeeze. Momentum oscillator for direction."""
    bbu, bbm, bbl   = _bb(df['close'], int(bb_period), bb_std)
    kcu, kcm, kcl   = _keltner(df, int(kc_period), kc_mult)
    squeeze  = (bbl > kcl) & (bbu < kcu)
    no_sq    = ~squeeze

    # Momentum: close - midpoint of highest high / lowest low
    mom_p = int(mom_period)
    delta = df['close'] - (df['high'].rolling(mom_p).max() + df['low'].rolling(mom_p).min()) / 2
    mom   = _ema(delta, mom_p)

    sig = pd.Series(0, index=df.index)
    sig[no_sq & (mom > 0) & (mom > mom.shift(1))]  =  1
    sig[no_sq & (mom < 0) & (mom < mom.shift(1))]  = -1
    return sig


def space_Squeeze_Momentum_TPSL(trial):
    return {
        'bb_period':  trial.suggest_int('bb_period', 15, 30),
        'bb_std':     trial.suggest_float('bb_std', 1.5, 3.0),
        'kc_period':  trial.suggest_int('kc_period', 15, 30),
        'kc_mult':    trial.suggest_float('kc_mult', 0.5, 2.5),
        'mom_period': trial.suggest_int('mom_period', 8, 20),
    }


# ── 10. VWAP Momentum Pullback [JS-TechTrading] ──────────────────────────────

def gen_VWAP_Momentum_Pullback(df, vwap_period=20, mo_period=14, ob_level=70, os_level=30, **kw):
    vp    = int(vwap_period)
    pv    = df['close'] * df['volume']
    vwap  = pv.rolling(vp).sum() / df['volume'].rolling(vp).sum().replace(0, np.nan)
    rsi   = _rsi(df['close'], int(mo_period))

    # Pullback: price retraces to VWAP, RSI confirms momentum
    long_cond  = (df['close'] > vwap) & (rsi.shift(1) < os_level) & (rsi > os_level)
    short_cond = (df['close'] < vwap) & (rsi.shift(1) > ob_level) & (rsi < ob_level)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig


def space_VWAP_Momentum_Pullback(trial):
    return {
        'vwap_period': trial.suggest_int('vwap_period', 10, 50),
        'mo_period':   trial.suggest_int('mo_period', 10, 21),
        'ob_level':    trial.suggest_int('ob_level', 65, 80),
        'os_level':    trial.suggest_int('os_level', 20, 35),
    }


# ── STRATEGY_EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'VWAP_Trendfollow': {
        'gen':            gen_VWAP_Trendfollow,
        'space':          space_VWAP_Trendfollow,
        'default_params': {'vwap_period': 20, 'sd_mult': 0.88},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 3842,
                           'description': 'VWAP SD band breakout'},
    },
    'Price_Volume_Breakout_v2': {
        'gen':            gen_Price_Volume_Breakout,
        'space':          space_Price_Volume_Breakout,
        'default_params': {'bo_period': 60, 'vol_period': 60, 'trend_period': 200},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 4771,
                           'description': 'Price + volume breakout [TradeDots]'},
    },
    'Full_Crypto_Pack': {
        'gen':            gen_Full_Crypto_Pack,
        'space':          space_Full_Crypto_Pack,
        'default_params': {'ema_period': 9, 'macd_fast': 12, 'macd_slow': 26,
                           'macd_sig': 9, 'rsi_period': 14, 'rsi_os': 35},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 4860,
                           'description': 'Full crypto pack: EMA+MACD+RSI+OBV'},
    },
    'MACD_Reloaded': {
        'gen':            gen_MACD_Reloaded,
        'space':          space_MACD_Reloaded,
        'default_params': {'ma_type': 0, 'fast': 12, 'slow': 26, 'sig_p': 9},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 7478,
                           'description': 'MACD with 11 MA types'},
    },
    'MACD_ATR_Trail': {
        'gen':            gen_MACD_ATR_Trail,
        'space':          space_MACD_ATR_Trail,
        'default_params': {'ma_period': 34, 'macd_fast': 12, 'macd_slow': 26,
                           'macd_sig': 9, 'atr_period': 14, 'atr_mult': 2.2},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 2344,
                           'description': 'MACD + ATR trailing SL'},
    },
    'MACD_BB_RSI_Alorse': {
        'gen':            gen_MACD_BB_RSI_Alorse,
        'space':          space_MACD_BB_RSI_Alorse,
        'default_params': {'macd_fast': 12, 'macd_slow': 26, 'macd_sig': 9,
                           'bb_period': 20, 'bb_std': 2.0, 'rsi_period': 14},
        'info':           {'source': 'TradingView', 'version': 4, 'likes': 2028,
                           'description': 'MACD + BB + RSI triple [Alorse]'},
    },
    'Scalping_Crypto_Stocks': {
        'gen':            gen_Scalping_Crypto_Stocks,
        'space':          space_Scalping_Crypto_Stocks,
        'default_params': {'sma_period': 25, 'ema_period': 200, 'kc_period': 10,
                           'kc_mult': 2.0, 'stoch_k': 14, 'stoch_d': 3,
                           'macd_fast': 4, 'macd_slow': 34, 'macd_sig': 5},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 3223,
                           'description': 'Scalping: SMA+EMA+Keltner+Stoch+MACD'},
    },
    'Keltner_ETH_DMI': {
        'gen':            gen_Keltner_ETH_DMI,
        'space':          space_Keltner_ETH_DMI,
        'default_params': {'kc_period': 46, 'kc_mult': 1.5, 'adx_period': 14, 'adx_thresh': 20},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 2390,
                           'description': 'Keltner Channel + DMI/ADX [Wunderbit]'},
    },
    'Squeeze_Momentum_TPSL': {
        'gen':            gen_Squeeze_Momentum_TPSL,
        'space':          space_Squeeze_Momentum_TPSL,
        'default_params': {'bb_period': 20, 'bb_std': 2.0, 'kc_period': 20,
                           'kc_mult': 1.5, 'mom_period': 12},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1918,
                           'description': 'BB + Keltner squeeze momentum'},
    },
    'VWAP_Momentum_Pullback': {
        'gen':            gen_VWAP_Momentum_Pullback,
        'space':          space_VWAP_Momentum_Pullback,
        'default_params': {'vwap_period': 20, 'mo_period': 14, 'ob_level': 70, 'os_level': 30},
        'info':           {'source': 'TradingView', 'version': 5, 'likes': 1425,
                           'description': 'VWAP momentum pullback [JS-TechTrading]'},
    },
}
