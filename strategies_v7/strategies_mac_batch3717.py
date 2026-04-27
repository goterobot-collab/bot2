"""
Batch 3717 - Mac paralela wave m18 - 5 volume classics / chart-pattern.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3716.

 1. TV_Force_Index_Elder    - Elder's Force Index = (close-prev) * volume,
                               EMA cross zero (NOT in repo as standalone)
 2. TV_NVI_PVI_Cross        - Norman Fosback Negative/Positive Volume Index
                               cross (NEW)
 3. TV_Chaikin_Oscillator   - Chaikin Osc = EMA(AD,3) - EMA(AD,10) zero-cross
                               (NEW; distinct from CMF, AD-Divergence)
 4. TV_Elliott_Impulse5     - 5-bar impulse pattern (5 same-direction
                               with extension > N*ATR). Distinct from
                               sandbox TV_Elliott_Wave_Fib_Scalper.
 5. TV_3LineBreak_Reversal  - 3-line break chart-style: 3 consecutive
                               new highs/lows then break opposite
                               (chart-pattern rule, NEW)
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


# ----- 1) ELDER FORCE INDEX -----
def gen_TV_Force_Index_Elder(df, ema_len=13, **kw):
    """Force Index = (close - prev_close) * volume.
    Smoothed: EMA(FI, ema_len). Cross zero = signal.
    """
    el = int(ema_len)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    fi = (c - c.shift(1)) * v
    fi_ema = _ema(fi, el)
    cross_up = (fi_ema > 0) & (fi_ema.shift(1) <= 0)
    cross_dn = (fi_ema < 0) & (fi_ema.shift(1) >= 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Force_Index_Elder():
    return {
        'ema_len': ('int', 7, 30),
    }


# ----- 2) NVI / PVI CROSS -----
def gen_TV_NVI_PVI_Cross(df, ma_len=255, **kw):
    """Negative Volume Index: cumulates close return ONLY when volume drops.
    Positive Volume Index: cumulates close return ONLY when volume rises.
    Long entry: NVI crosses above its MA (smart money - low-vol moves).
    Short entry: PVI crosses below its MA (retail panic - high-vol moves down).
    """
    ml = int(ma_len)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    ret = c.pct_change()
    vol_drop = v < v.shift(1)
    vol_rise = v > v.shift(1)
    nvi_inc = ret.where(vol_drop, 0.0)
    pvi_inc = ret.where(vol_rise, 0.0)
    nvi = (1 + nvi_inc).cumprod()
    pvi = (1 + pvi_inc).cumprod()
    nvi_ma = nvi.rolling(ml, min_periods=ml).mean()
    pvi_ma = pvi.rolling(ml, min_periods=ml).mean()
    long_cross = (nvi > nvi_ma) & (nvi.shift(1) <= nvi_ma.shift(1))
    short_cross = (pvi < pvi_ma) & (pvi.shift(1) >= pvi_ma.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_NVI_PVI_Cross():
    return {
        'ma_len': ('int', 100, 300),
    }


# ----- 3) CHAIKIN OSCILLATOR -----
def gen_TV_Chaikin_Oscillator(df, fast=3, slow=10, **kw):
    """Accumulation/Distribution Line = cumsum( ((C-L)-(H-C))/(H-L) * V ).
    Chaikin Osc = EMA(AD, fast) - EMA(AD, slow).
    Zero cross = signal.
    """
    f = int(fast)
    s = int(slow)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    rng = (h - l).replace(0.0, np.nan)
    mfm = ((c - l) - (h - c)) / rng
    mfv = mfm * v
    ad = mfv.fillna(0).cumsum()
    osc = _ema(ad, f) - _ema(ad, s)
    cross_up = (osc > 0) & (osc.shift(1) <= 0)
    cross_dn = (osc < 0) & (osc.shift(1) >= 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Chaikin_Oscillator():
    return {
        'fast': ('int', 2, 8),
        'slow': ('int', 8, 25),
    }


# ----- 4) ELLIOTT 5-BAR IMPULSE -----
def gen_TV_Elliott_Impulse5(df, atr_len=14, ext_mult=2.0, **kw):
    """Simplified Elliott impulse: 5 consecutive same-direction bars
    (close > prev close) AND total move > ext_mult * ATR(atr_len).
    Long: 5 up bars + extension. Short: 5 down bars.
    """
    al = int(atr_len)
    em = float(ext_mult)
    c = df['close'].astype(float)
    up_bar = (c > c.shift(1)).astype(int)
    dn_bar = (c < c.shift(1)).astype(int)
    up5 = up_bar.rolling(5, min_periods=5).sum() == 5
    dn5 = dn_bar.rolling(5, min_periods=5).sum() == 5
    move5 = (c - c.shift(5)).abs()
    atr = _atr(df, al)
    big_move = move5 > em * atr
    long_trigger = up5 & big_move
    short_trigger = dn5 & big_move
    long_entry = long_trigger & ~(long_trigger.shift(1).fillna(False))
    short_entry = short_trigger & ~(short_trigger.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Elliott_Impulse5():
    return {
        'atr_len': ('int', 7, 30),
        'ext_mult': ('float', 1.0, 4.0),
    }


# ----- 5) 3-LINE BREAK REVERSAL -----
def gen_TV_3LineBreak_Reversal(df, lookback=3, **kw):
    """Simplified 3-line break: 3 consecutive new highs (close > max of last
    `lookback` highs each bar) then close below first of the 3 = reversal SHORT.
    Symmetric for LONG.
    """
    lb = int(lookback)
    c = df['close'].astype(float)
    new_hi = c > c.rolling(lb, min_periods=lb).max().shift(1)
    new_lo = c < c.rolling(lb, min_periods=lb).min().shift(1)
    streak_hi = new_hi.rolling(3, min_periods=3).sum() == 3
    streak_lo = new_lo.rolling(3, min_periods=3).sum() == 3
    # Reversal: streak_hi 3 bars ago AND now close < close 3 bars ago
    short_trigger = streak_hi.shift(1).fillna(False) & (c < c.shift(3))
    long_trigger = streak_lo.shift(1).fillna(False) & (c > c.shift(3))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_trigger.shift(1).fillna(False).astype(bool)] = 1
    sig[short_trigger.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_3LineBreak_Reversal():
    return {
        'lookback': ('int', 2, 8),
    }


STRATEGY_EXPORT = {
    "TV_Force_Index_Elder": {
        "gen": gen_TV_Force_Index_Elder, "space": space_TV_Force_Index_Elder,
        "source": "mac_batch3717_Elder_ForceIndex",
    },
    "TV_NVI_PVI_Cross": {
        "gen": gen_TV_NVI_PVI_Cross, "space": space_TV_NVI_PVI_Cross,
        "source": "mac_batch3717_Fosback_NVI_PVI",
    },
    "TV_Chaikin_Oscillator": {
        "gen": gen_TV_Chaikin_Oscillator, "space": space_TV_Chaikin_Oscillator,
        "source": "mac_batch3717_Chaikin_Oscillator",
    },
    "TV_Elliott_Impulse5": {
        "gen": gen_TV_Elliott_Impulse5, "space": space_TV_Elliott_Impulse5,
        "source": "mac_batch3717_Elliott_5bar_impulse_heuristic",
    },
    "TV_3LineBreak_Reversal": {
        "gen": gen_TV_3LineBreak_Reversal, "space": space_TV_3LineBreak_Reversal,
        "source": "mac_batch3717_3LineBreak_chart_reversal",
    },
}
