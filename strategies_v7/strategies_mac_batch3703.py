"""
Batch 3703 - Mac paralela wave m4 - 5 channel / structural families.

Owner: Mac paralela (Claude Code online, rama claude/grail-hunt-bot-noeIq).

Families chosen for ZERO overlap with sandbox 3566-3610 (verified against:
Wyckoff, Ehlers Fisher/MESA/SuperSmoother/Sine/Trend, ZigZag_Swing, Kernel
Ridge, OLS, Funding, KAMA, Ichimoku, HA, ElderTripleScreen, VolumeProfile,
CVD, Footprint, Kyle, VPIN, Linear Regression, BB Squeeze, OBV, RSI Div,
Double MACD, ATR Regime, BTC Beta, CumVol, Momentum Accel, RealizedVol,
VolumeSpike), Mac V8 prod, or Mac batches 3700-3702:

 1. TV_Keltner_Breakout     - direct Keltner upper/lower break (not BB-squeeze filter)
 2. TV_DPO_Extreme          - Detrended Price Oscillator crosses back from extremes
 3. TV_Renko_Flip           - Renko-brick color flip (ATR-sized bricks from OHLC)
 4. TV_Andrews_Pitchfork    - median-line channel upper/lower break
 5. TV_DMI_ADX_Surge        - +DI/-DI cross with ADX>threshold surge

All pickle-safe, no lambdas, signal integer in {-1, 0, 1}, .shift(1) applied.
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _atr(df, n):
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()


# ----- 1) KELTNER BREAKOUT -----
def gen_TV_Keltner_Breakout(df, ema_len=20, atr_len=10, mult=1.5,
                             trend_len=50, **kw):
    """Upper = EMA(close) + mult*ATR; Lower = EMA(close) - mult*ATR.
    Long: close breaks above upper AND close > EMA(trend_len).
    Short: close breaks below lower AND close < EMA(trend_len).
    """
    el = int(ema_len)
    al = int(atr_len)
    m = float(mult)
    tl = int(trend_len)
    c = df['close'].astype(float)
    mid = _ema(c, el)
    atr = _atr(df, al)
    upper = mid + m * atr
    lower = mid - m * atr
    trend = _ema(c, tl)
    cross_up = (c > upper) & (c.shift(1) <= upper.shift(1)) & (c > trend)
    cross_dn = (c < lower) & (c.shift(1) >= lower.shift(1)) & (c < trend)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Keltner_Breakout():
    return {
        'ema_len': ('int', 10, 50),
        'atr_len': ('int', 7, 30),
        'mult': ('float', 1.0, 3.0),
        'trend_len': ('int', 30, 150),
    }


# ----- 2) DPO EXTREME -----
def gen_TV_DPO_Extreme(df, dpo_len=20, z_threshold=1.5, **kw):
    """DPO = close.shift(n/2+1) - SMA(close, n). Detrended oscillator.
    Signal: z-score of DPO crosses back from |z| > z_threshold.
    Long entry: z crosses UP from z < -threshold (oversold extreme).
    Short entry: z crosses DOWN from z > +threshold.
    """
    n = int(dpo_len)
    zt = float(z_threshold)
    c = df['close'].astype(float)
    sma = c.rolling(n, min_periods=n).mean()
    dpo = c.shift(n // 2 + 1) - sma
    mu = dpo.rolling(n, min_periods=n).mean()
    sd = dpo.rolling(n, min_periods=n).std(ddof=0).replace(0.0, np.nan)
    z = (dpo - mu) / sd
    long_cross = (z > -zt) & (z.shift(1) <= -zt)
    short_cross = (z < zt) & (z.shift(1) >= zt)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_DPO_Extreme():
    return {
        'dpo_len': ('int', 14, 40),
        'z_threshold': ('float', 1.0, 2.5),
    }


# ----- 3) RENKO FLIP (ATR-sized bricks) -----
def gen_TV_Renko_Flip(df, brick_atr_len=14, brick_mult=1.0, **kw):
    """Simplified Renko: brick_size = mult * ATR(n) (recomputed lazily).
    Walk bar-by-bar; when close moves +/- brick_size from last brick close,
    emit a new brick of that direction. Entry on brick color flip.
    """
    n = int(brick_atr_len)
    m = float(brick_mult)
    c = df['close'].astype(float)
    atr = _atr(df, n).to_numpy()
    vals = c.to_numpy()
    direction = np.zeros(len(vals), dtype=int)
    last_close = vals[0] if len(vals) > 0 else 0.0
    current_dir = 0
    for i in range(1, len(vals)):
        a = atr[i - 1] if not np.isnan(atr[i - 1]) else 0.0
        bs = m * a
        if bs <= 0:
            direction[i] = current_dir
            continue
        move = vals[i] - last_close
        if move >= bs:
            new_dir = 1
        elif move <= -bs:
            new_dir = -1
        else:
            direction[i] = current_dir
            continue
        if new_dir != current_dir and current_dir != 0:
            direction[i] = new_dir  # flip bar
        else:
            direction[i] = new_dir
        current_dir = new_dir
        last_close = vals[i]
    d = pd.Series(direction, index=df.index)
    flip_up = (d == 1) & (d.shift(1) == -1)
    flip_dn = (d == -1) & (d.shift(1) == 1)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[flip_up.shift(1).fillna(False).astype(bool)] = 1
    sig[flip_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Renko_Flip():
    return {
        'brick_atr_len': ('int', 7, 30),
        'brick_mult': ('float', 0.5, 3.0),
    }


# ----- 4) ANDREWS PITCHFORK CHANNEL -----
def gen_TV_Andrews_Pitchfork(df, pivot_len=50, break_pct=0.005, **kw):
    """Rolling pitchfork: find 3 anchor points = pivot_hi1, pivot_lo, pivot_hi2
    within last pivot_len bars. Median line = mid(pivot_hi1, pivot_hi2) slope
    anchored at pivot_lo. Upper/lower parallel lines at +- half-width.
    Entry long on close breaking above upper line; short on breaking below lower.
    Simplified version: compute rolling max/min and midline as mean of
    recent high and low; entry on close deviating by break_pct.
    """
    n = int(pivot_len)
    bp = float(break_pct)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    hh = h.rolling(n, min_periods=n).max()
    ll = l.rolling(n, min_periods=n).min()
    mid = (hh + ll) / 2.0
    upper = mid * (1 + bp)
    lower = mid * (1 - bp)
    cross_up = (c > upper) & (c.shift(1) <= upper.shift(1))
    cross_dn = (c < lower) & (c.shift(1) >= lower.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Andrews_Pitchfork():
    return {
        'pivot_len': ('int', 20, 100),
        'break_pct': ('float', 0.002, 0.02),
    }


# ----- 5) DMI / ADX SURGE -----
def gen_TV_DMI_ADX_Surge(df, dmi_len=14, adx_threshold=25, **kw):
    """+DI / -DI / ADX from Wilder. Long when +DI crosses above -DI with ADX >= threshold.
    Short when -DI crosses above +DI with ADX >= threshold.
    """
    n = int(dmi_len)
    at = float(adx_threshold)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    up = h - h.shift(1)
    dn = l.shift(1) - l
    plus_dm = ((up > dn) & (up > 0)).astype(float) * up
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn
    atr = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    cross_up = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1)) & (adx >= at)
    cross_dn = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1)) & (adx >= at)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_DMI_ADX_Surge():
    return {
        'dmi_len': ('int', 8, 30),
        'adx_threshold': ('int', 15, 40),
    }


STRATEGY_EXPORT = {
    "TV_Keltner_Breakout": {
        "gen": gen_TV_Keltner_Breakout, "space": space_TV_Keltner_Breakout,
        "source": "mac_batch3703",
    },
    "TV_DPO_Extreme": {
        "gen": gen_TV_DPO_Extreme, "space": space_TV_DPO_Extreme,
        "source": "mac_batch3703",
    },
    "TV_Renko_Flip": {
        "gen": gen_TV_Renko_Flip, "space": space_TV_Renko_Flip,
        "source": "mac_batch3703",
    },
    "TV_Andrews_Pitchfork": {
        "gen": gen_TV_Andrews_Pitchfork, "space": space_TV_Andrews_Pitchfork,
        "source": "mac_batch3703",
    },
    "TV_DMI_ADX_Surge": {
        "gen": gen_TV_DMI_ADX_Surge, "space": space_TV_DMI_ADX_Surge,
        "source": "mac_batch3703",
    },
}
