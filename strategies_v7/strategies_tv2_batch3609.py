"""
Batch 3609 - 5 classic-but-untested families for H7 wave.

Ideas:
 - TV_RSI_Divergence: RSI divergence vs price (bullish = price LL + RSI HL)
 - TV_BB_Squeeze_Breakout: BB inside Keltner (squeeze) + breakout direction
 - TV_VolumeSpike_Reversal: volume z-score spike fade
 - TV_Double_MACD: fast MACD cross filtered by slow MACD sign
 - TV_OBV_Momentum: OBV crossing its own EMA as trend signal

All pickle-safe, no lambdas, no look-ahead (.shift(1) on signals).
"""
import numpy as np
import pandas as pd


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ----- 1) RSI DIVERGENCE -----
def gen_TV_RSI_Divergence(df, rsi_len=14, lookback=20, **kw):
    c = df['close'].astype(float)
    r = _rsi(c, int(rsi_len))
    L = int(lookback)
    # Rolling argmin/argmax of price and RSI
    p_min_now = c.rolling(L).min()
    p_min_prev = c.shift(L).rolling(L).min()
    p_max_now = c.rolling(L).max()
    p_max_prev = c.shift(L).rolling(L).max()
    r_min_now = r.rolling(L).min()
    r_min_prev = r.shift(L).rolling(L).min()
    r_max_now = r.rolling(L).max()
    r_max_prev = r.shift(L).rolling(L).max()
    bull_div = (p_min_now < p_min_prev) & (r_min_now > r_min_prev)
    bear_div = (p_max_now > p_max_prev) & (r_max_now < r_max_prev)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[bull_div.shift(1).fillna(False)] = 1
    sig[bear_div.shift(1).fillna(False)] = -1
    return sig


def space_TV_RSI_Divergence():
    return {'rsi_len': ('int', 7, 30),
            'lookback': ('int', 10, 60)}


# ----- 2) BB SQUEEZE BREAKOUT -----
def gen_TV_BB_Squeeze_Breakout(df, bb_len=20, bb_mult=2.0, kc_len=20, kc_mult=1.5, **kw):
    c = df['close'].astype(float); h = df['high'].astype(float); l = df['low'].astype(float)
    n = int(bb_len)
    ma = c.rolling(n, min_periods=n).mean()
    sd = c.rolling(n, min_periods=n).std(ddof=0)
    bb_up = ma + float(bb_mult) * sd
    bb_dn = ma - float(bb_mult) * sd
    # Keltner
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(int(kc_len), min_periods=int(kc_len)).mean()
    kc_up = ma + float(kc_mult) * atr
    kc_dn = ma - float(kc_mult) * atr
    squeeze = (bb_up < kc_up) & (bb_dn > kc_dn)
    # Breakout direction: close breaks above/below BB on exit from squeeze
    sq_prev = squeeze.shift(1).fillna(False)
    exit_up = sq_prev & (~squeeze) & (c > bb_up)
    exit_dn = sq_prev & (~squeeze) & (c < bb_dn)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[exit_up.shift(1).fillna(False)] = 1
    sig[exit_dn.shift(1).fillna(False)] = -1
    return sig


def space_TV_BB_Squeeze_Breakout():
    return {'bb_len': ('int', 10, 60),
            'bb_mult': ('float', 1.5, 3.0),
            'kc_len': ('int', 10, 60),
            'kc_mult': ('float', 1.0, 2.5)}


# ----- 3) VOLUME SPIKE REVERSAL -----
def gen_TV_VolumeSpike_Reversal(df, vol_len=40, z_thresh=2.5, body_len=5, **kw):
    c = df['close'].astype(float); v = df['volume'].astype(float)
    n = int(vol_len)
    mu = v.rolling(n, min_periods=n).mean()
    sd = v.rolling(n, min_periods=n).std(ddof=0).replace(0.0, np.nan)
    z = (v - mu) / sd
    spike = (z > float(z_thresh)).shift(1).fillna(False)
    # Direction: N-bar price change at spike
    body = c.pct_change(int(body_len)).shift(1)
    sig = pd.Series(0, index=df.index, dtype=int)
    # High-vol + up-body -> fade short; + down-body -> fade long
    sig[(spike & (body > 0)).fillna(False)] = -1
    sig[(spike & (body < 0)).fillna(False)] = 1
    return sig


def space_TV_VolumeSpike_Reversal():
    return {'vol_len': ('int', 20, 150),
            'z_thresh': ('float', 1.8, 4.0),
            'body_len': ('int', 2, 15)}


# ----- 4) DOUBLE MACD -----
def gen_TV_Double_MACD(df, fast_a=12, slow_a=26, sig_a=9,
                       fast_b=24, slow_b=52, sig_b=18, **kw):
    c = df['close'].astype(float)
    def _macd(c, f, s, sg):
        ef = c.ewm(span=int(f), adjust=False, min_periods=int(f)).mean()
        es = c.ewm(span=int(s), adjust=False, min_periods=int(s)).mean()
        m = ef - es
        sig = m.ewm(span=int(sg), adjust=False, min_periods=int(sg)).mean()
        return m, sig
    m_a, s_a = _macd(c, fast_a, slow_a, sig_a)
    m_b, s_b = _macd(c, fast_b, slow_b, sig_b)
    cross_up = (m_a > s_a) & (m_a.shift(1) <= s_a.shift(1))
    cross_dn = (m_a < s_a) & (m_a.shift(1) >= s_a.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    # Filter by slow MACD sign
    sig[(cross_up & (m_b > 0)).shift(1).fillna(False)] = 1
    sig[(cross_dn & (m_b < 0)).shift(1).fillna(False)] = -1
    return sig


def space_TV_Double_MACD():
    return {'fast_a': ('int', 5, 20),
            'slow_a': ('int', 20, 40),
            'sig_a': ('int', 5, 15),
            'fast_b': ('int', 15, 40),
            'slow_b': ('int', 40, 100),
            'sig_b': ('int', 10, 30)}


# ----- 5) OBV MOMENTUM -----
def gen_TV_OBV_Momentum(df, ema_len=30, smooth=3, **kw):
    c = df['close'].astype(float); v = df['volume'].astype(float)
    r = c.diff().fillna(0.0)
    direction = np.sign(r)
    obv = (direction * v).cumsum()
    obv_s = obv.ewm(span=int(smooth), adjust=False, min_periods=int(smooth)).mean()
    obv_ema = obv_s.ewm(span=int(ema_len), adjust=False, min_periods=int(ema_len)).mean()
    cross_up = (obv_s > obv_ema) & (obv_s.shift(1) <= obv_ema.shift(1))
    cross_dn = (obv_s < obv_ema) & (obv_s.shift(1) >= obv_ema.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False)] = 1
    sig[cross_dn.shift(1).fillna(False)] = -1
    return sig


def space_TV_OBV_Momentum():
    return {'ema_len': ('int', 10, 120),
            'smooth': ('int', 1, 10)}


STRATEGY_EXPORT = {
    'TV_RSI_Divergence': {
        'gen': gen_TV_RSI_Divergence, 'space': space_TV_RSI_Divergence,
        'source': 'RSI vs price divergence (batch 3609)'},
    'TV_BB_Squeeze_Breakout': {
        'gen': gen_TV_BB_Squeeze_Breakout, 'space': space_TV_BB_Squeeze_Breakout,
        'source': 'Bollinger-Keltner squeeze breakout (batch 3609)'},
    'TV_VolumeSpike_Reversal': {
        'gen': gen_TV_VolumeSpike_Reversal, 'space': space_TV_VolumeSpike_Reversal,
        'source': 'Volume z-score spike fade (batch 3609)'},
    'TV_Double_MACD': {
        'gen': gen_TV_Double_MACD, 'space': space_TV_Double_MACD,
        'source': 'Fast MACD cross filtered by slow MACD sign (batch 3609)'},
    'TV_OBV_Momentum': {
        'gen': gen_TV_OBV_Momentum, 'space': space_TV_OBV_Momentum,
        'source': 'OBV-EMA crossover trend (batch 3609)'},
}
