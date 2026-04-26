"""
Batch 3707 - Mac paralela wave m8 - 5 TREND-following / breakout families.

After m7 finding (mean-reversion / contextual strategies suffer 15-27pp gap
between optuna engine and run_forensic engine due to +1bar exit delay),
this batch focuses on PURE TREND / BREAKOUT signals which are less sensitive
to exit-bar timing.

All low-param (1-3 each), pickle-safe, signal int {-1,0,1}, .shift(1).

ZERO overlap verified with sandbox 3566-3610, Mac V8 prod, Mac 3700-3706.

 1. TV_PSAR_Flip                - Wilder Parabolic SAR flip (DISTINCT from
                                   sandbox SuperTrend variants - PSAR uses
                                   acceleration factor, not ATR bands)
 2. TV_Wilder_ATR_Channel_Break - close break of EMA +/- mult*ATR (Wilder
                                   volatility channel, NOT Keltner-style)
 3. TV_LowVol_Compression_Break - stdev(returns) at N-bar minimum, then break
                                   (DISTINCT from BB_Squeeze which uses BB
                                   inside Keltner)
 4. TV_HighClose_Confirm        - close > N-period high AND close in top X%
                                   of bar's range (high+confirmation, distinct
                                   from BreaksAndRetests)
 5. TV_McClellan_AD_Oscillator  - sum signed volume (close>prev: +V else -V),
                                   EMA(fast) - EMA(slow); cross zero (NEW)
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


# ----- 1) WILDER PARABOLIC SAR FLIP -----
def gen_TV_PSAR_Flip(df, af_init=0.02, af_max=0.20, af_step=0.02, **kw):
    """Welles Wilder Parabolic SAR. Flip detection generates entry signal."""
    afi = float(af_init)
    afm = float(af_max)
    afs = float(af_step)
    h = df['high'].astype(float).to_numpy()
    l = df['low'].astype(float).to_numpy()
    n = len(h)
    if n < 5:
        return pd.Series(0, index=df.index, dtype=int)
    psar = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    af = afi
    ep = h[0]  # extreme point
    psar[0] = l[0]
    direction[0] = 1  # start long
    for i in range(1, n):
        prev_psar = psar[i - 1]
        if direction[i - 1] == 1:  # was long
            psar_i = prev_psar + af * (ep - prev_psar)
            psar_i = min(psar_i, l[i - 1])
            if i >= 2:
                psar_i = min(psar_i, l[i - 2])
            if l[i] <= psar_i:
                # flip to short
                direction[i] = -1
                psar[i] = ep
                ep = l[i]
                af = afi
            else:
                direction[i] = 1
                psar[i] = psar_i
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + afs, afm)
        else:  # was short
            psar_i = prev_psar + af * (ep - prev_psar)
            psar_i = max(psar_i, h[i - 1])
            if i >= 2:
                psar_i = max(psar_i, h[i - 2])
            if h[i] >= psar_i:
                direction[i] = 1
                psar[i] = ep
                ep = h[i]
                af = afi
            else:
                direction[i] = -1
                psar[i] = psar_i
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + afs, afm)
    d = pd.Series(direction, index=df.index)
    flip_long = (d == 1) & (d.shift(1) == -1)
    flip_short = (d == -1) & (d.shift(1) == 1)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[flip_long.shift(1).fillna(False).astype(bool)] = 1
    sig[flip_short.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_PSAR_Flip():
    return {
        'af_init': ('float', 0.01, 0.05),
        'af_max': ('float', 0.10, 0.40),
        'af_step': ('float', 0.01, 0.05),
    }


# ----- 2) WILDER ATR CHANNEL BREAK -----
def gen_TV_Wilder_ATR_Channel_Break(df, ema_len=20, atr_len=14, mult=2.0, **kw):
    """close > EMA(N) + mult*ATR -> long; close < EMA(N) - mult*ATR -> short.
    Wilder's volatility channel (NOT Keltner which uses true range).
    """
    el = int(ema_len)
    al = int(atr_len)
    m = float(mult)
    c = df['close'].astype(float)
    ema = _ema(c, el)
    atr = _atr(df, al)
    upper = ema + m * atr
    lower = ema - m * atr
    cross_up = (c > upper) & (c.shift(1) <= upper.shift(1))
    cross_dn = (c < lower) & (c.shift(1) >= lower.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Wilder_ATR_Channel_Break():
    return {
        'ema_len': ('int', 10, 50),
        'atr_len': ('int', 7, 30),
        'mult': ('float', 1.0, 3.5),
    }


# ----- 3) LOW VOL COMPRESSION BREAK -----
def gen_TV_LowVol_Compression_Break(df, vol_window=20, lookback=60,
                                      break_pct=1.5, **kw):
    """stdev(returns, vol_window) at minimum over last lookback bars.
    Then breakout: |return| > break_pct * stdev (norm).
    """
    vw = int(vol_window)
    lb = int(lookback)
    bp = float(break_pct)
    c = df['close'].astype(float)
    ret = c.pct_change()
    sd = ret.rolling(vw, min_periods=vw).std(ddof=0)
    sd_min = sd.rolling(lb, min_periods=lb).min()
    is_compressed = (sd <= sd_min * 1.05)  # within 5% of recent min
    long_break = is_compressed & (ret > bp * sd)
    short_break = is_compressed & (ret < -bp * sd)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_break.shift(1).fillna(False).astype(bool)] = 1
    sig[short_break.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_LowVol_Compression_Break():
    return {
        'vol_window': ('int', 10, 30),
        'lookback': ('int', 30, 120),
        'break_pct': ('float', 1.0, 3.0),
    }


# ----- 4) HIGH CLOSE BREAKOUT CONFIRM -----
def gen_TV_HighClose_Confirm(df, n_period=20, top_pct=25, **kw):
    """close > N-period high (high break) AND close in top X% of bar's
    own range (close confirms). Symmetric for short.
    """
    n = int(n_period)
    tp = float(top_pct) / 100.0
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    prior_high = h.rolling(n, min_periods=n).max().shift(1)
    prior_low = l.rolling(n, min_periods=n).min().shift(1)
    bar_rng = (h - l).replace(0.0, np.nan)
    close_pct = (c - l) / bar_rng
    long_setup = (c > prior_high) & (close_pct > (1 - tp))
    short_setup = (c < prior_low) & (close_pct < tp)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_HighClose_Confirm():
    return {
        'n_period': ('int', 10, 60),
        'top_pct': ('int', 15, 40),
    }


# ----- 5) MCCLELLAN A/D OSCILLATOR (single-asset adaptation) -----
def gen_TV_McClellan_AD_Oscillator(df, ema_fast=19, ema_slow=39, **kw):
    """McClellan Oscillator originally measures market breadth.
    Single-asset adaptation: signed volume = +V if close>prev_close, else -V.
    Oscillator = EMA(signed_vol, fast) - EMA(signed_vol, slow).
    Long entry: oscillator crosses above zero. Short: crosses below.
    """
    ef = int(ema_fast)
    es = int(ema_slow)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    pc = c.shift(1)
    direction = np.where(c > pc, 1.0, np.where(c < pc, -1.0, 0.0))
    signed_vol = pd.Series(direction, index=df.index) * v
    osc = _ema(signed_vol, ef) - _ema(signed_vol, es)
    cross_up = (osc > 0) & (osc.shift(1) <= 0)
    cross_dn = (osc < 0) & (osc.shift(1) >= 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_McClellan_AD_Oscillator():
    return {
        'ema_fast': ('int', 5, 25),
        'ema_slow': ('int', 25, 60),
    }


STRATEGY_EXPORT = {
    "TV_PSAR_Flip": {
        "gen": gen_TV_PSAR_Flip, "space": space_TV_PSAR_Flip,
        "source": "mac_batch3707_Wilder_NewConcepts1978",
    },
    "TV_Wilder_ATR_Channel_Break": {
        "gen": gen_TV_Wilder_ATR_Channel_Break,
        "space": space_TV_Wilder_ATR_Channel_Break,
        "source": "mac_batch3707_Wilder_ATR_channel",
    },
    "TV_LowVol_Compression_Break": {
        "gen": gen_TV_LowVol_Compression_Break,
        "space": space_TV_LowVol_Compression_Break,
        "source": "mac_batch3707_volatility_compression",
    },
    "TV_HighClose_Confirm": {
        "gen": gen_TV_HighClose_Confirm, "space": space_TV_HighClose_Confirm,
        "source": "mac_batch3707_breakout_close_confirm",
    },
    "TV_McClellan_AD_Oscillator": {
        "gen": gen_TV_McClellan_AD_Oscillator,
        "space": space_TV_McClellan_AD_Oscillator,
        "source": "mac_batch3707_McClellan_singleAsset_adapt",
    },
}
