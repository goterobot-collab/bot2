"""
Batch 3705 - Mac paralela wave m6 - 5 LOW-PARAM families (1-3 params each)
designed to maximize plateau pass-rate after the m1-m5 finding that >=4
params nearly always fails plateau >=60% threshold.

Owner: Mac paralela (Claude Code online, rama claude/grail-hunt-bot-noeIq).

All checked for ZERO overlap with sandbox 3566-3610, Mac V8 prod, or Mac
batches 3700-3704.

 1. TV_RVI_Cross                 - Relative Vigor Index cross (Ehlers/Brown
                                    "Investools" 2002, NOT in any existing batch)
 2. TV_AwesomeOscillator_Saucer  - Bill Williams AO saucer entry (variant of
                                    AO, NOT covered by sandbox Bill_Williams_Fractal)
 3. TV_VolumeOscillator_Cross    - VolOsc = (EMA(V,fast) - EMA(V,slow))/EMA(V,slow)
                                    cross zero (NOT in repo)
 4. TV_NormalizedROC             - ROC(n) / SMA(|ROC(n)|, n) z-score-like cross
                                    (distinct from AdaptiveFisherizedROC)
 5. TV_GapFill_Reversal          - bar-to-bar gap reversal (intra-bar, not
                                    weekend-specific like TV_Weekend_Gap_Fade)

All pickle-safe, no lambdas, signal int {-1,0,1}, .shift(1) applied.
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


# ----- 1) RELATIVE VIGOR INDEX CROSS -----
def gen_TV_RVI_Cross(df, rvi_len=10, sig_len=4, **kw):
    """RVI = SMA(close-open, n) / SMA(high-low, n).
    Signal = SMA(RVI, sig_len). Long on RVI cross above signal AND RVI > 0.
    Short on cross below signal AND RVI < 0.
    """
    rl = int(rvi_len)
    sl = int(sig_len)
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    num = _sma(c - o, rl)
    den = _sma(h - l, rl).replace(0.0, np.nan)
    rvi = num / den
    sig_line = _sma(rvi, sl)
    cross_up = (rvi > sig_line) & (rvi.shift(1) <= sig_line.shift(1)) & (rvi > 0)
    cross_dn = (rvi < sig_line) & (rvi.shift(1) >= sig_line.shift(1)) & (rvi < 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_RVI_Cross():
    return {
        'rvi_len': ('int', 5, 20),
        'sig_len': ('int', 3, 10),
    }


# ----- 2) AWESOME OSCILLATOR SAUCER -----
def gen_TV_AwesomeOscillator_Saucer(df, fast=5, slow=34, **kw):
    """AO = SMA(median, fast) - SMA(median, slow), median = (H+L)/2.
    Saucer = 3-bar pattern in same color (above zero=bullish, below=bearish):
      bull saucer: AO[-2]>0, AO[-1]>0, AO[-2]>AO[-1], AO>AO[-1]  (down then up while green)
      bear saucer: AO[-2]<0, AO[-1]<0, AO[-2]<AO[-1], AO<AO[-1]
    """
    f = int(fast)
    s = int(slow)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    median = (h + l) / 2.0
    ao = _sma(median, f) - _sma(median, s)
    ao_1 = ao.shift(1)
    ao_2 = ao.shift(2)
    bull = (ao_2 > 0) & (ao_1 > 0) & (ao_2 > ao_1) & (ao > ao_1)
    bear = (ao_2 < 0) & (ao_1 < 0) & (ao_2 < ao_1) & (ao < ao_1)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[bull.shift(1).fillna(False).astype(bool)] = 1
    sig[bear.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_AwesomeOscillator_Saucer():
    return {
        'fast': ('int', 3, 10),
        'slow': ('int', 20, 50),
    }


# ----- 3) VOLUME OSCILLATOR CROSS -----
def gen_TV_VolumeOscillator_Cross(df, fast=14, slow=28, **kw):
    """VolOsc = (EMA(V, fast) - EMA(V, slow)) / EMA(V, slow) * 100.
    Long: VolOsc crosses above zero AND close > prev close.
    Short: VolOsc crosses above zero AND close < prev close (negative breakout).
    """
    f = int(fast)
    s = int(slow)
    v = df['volume'].astype(float)
    c = df['close'].astype(float)
    ef = _ema(v, f)
    es = _ema(v, s)
    vo = (ef - es) / es.replace(0.0, np.nan) * 100.0
    cross_up = (vo > 0) & (vo.shift(1) <= 0)
    long_dir = cross_up & (c > c.shift(1))
    short_dir = cross_up & (c < c.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_dir.shift(1).fillna(False).astype(bool)] = 1
    sig[short_dir.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_VolumeOscillator_Cross():
    return {
        'fast': ('int', 5, 20),
        'slow': ('int', 20, 60),
    }


# ----- 4) NORMALIZED ROC -----
def gen_TV_NormalizedROC(df, roc_len=14, threshold=1.5, **kw):
    """nROC = ROC(n) / SMA(|ROC(n)|, n) -> z-score-like momentum.
    Long: nROC crosses above +threshold.
    Short: nROC crosses below -threshold.
    """
    n = int(roc_len)
    t = float(threshold)
    c = df['close'].astype(float)
    roc = (c / c.shift(n) - 1.0) * 100.0
    abs_roc = roc.abs()
    avg_abs = _sma(abs_roc, n).replace(0.0, np.nan)
    nroc = roc / avg_abs
    long_cross = (nroc > t) & (nroc.shift(1) <= t)
    short_cross = (nroc < -t) & (nroc.shift(1) >= -t)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cross.shift(1).fillna(False).astype(bool)] = 1
    sig[short_cross.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_NormalizedROC():
    return {
        'roc_len': ('int', 7, 30),
        'threshold': ('float', 0.8, 2.5),
    }


# ----- 5) GAP FILL REVERSAL -----
def gen_TV_GapFill_Reversal(df, gap_pct=1.0, atr_len=14, **kw):
    """Bar-to-bar gap = (open - prev_close) / prev_close * 100.
    Long entry: large negative gap (gap < -gap_pct) - expect fill upward.
    Short entry: large positive gap (gap > +gap_pct) - expect fill downward.
    Confirmation: |gap| > 0.5 * ATR(n)/prev_close (avoid trivial gaps).
    """
    gp = float(gap_pct)
    al = int(atr_len)
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    gap = (o - pc) / pc.replace(0.0, np.nan) * 100.0
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / al, adjust=False, min_periods=al).mean()
    atr_pct = atr / pc.replace(0.0, np.nan) * 100.0
    confirm = gap.abs() > 0.5 * atr_pct
    long_trigger = (gap < -gp) & confirm
    short_trigger = (gap > gp) & confirm
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_trigger.shift(1).fillna(False).astype(bool)] = 1
    sig[short_trigger.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_GapFill_Reversal():
    return {
        'gap_pct': ('float', 0.5, 4.0),
        'atr_len': ('int', 7, 30),
    }


STRATEGY_EXPORT = {
    "TV_RVI_Cross": {
        "gen": gen_TV_RVI_Cross, "space": space_TV_RVI_Cross,
        "source": "mac_batch3705_low_param",
    },
    "TV_AwesomeOscillator_Saucer": {
        "gen": gen_TV_AwesomeOscillator_Saucer,
        "space": space_TV_AwesomeOscillator_Saucer,
        "source": "mac_batch3705_low_param",
    },
    "TV_VolumeOscillator_Cross": {
        "gen": gen_TV_VolumeOscillator_Cross,
        "space": space_TV_VolumeOscillator_Cross,
        "source": "mac_batch3705_low_param",
    },
    "TV_NormalizedROC": {
        "gen": gen_TV_NormalizedROC, "space": space_TV_NormalizedROC,
        "source": "mac_batch3705_low_param",
    },
    "TV_GapFill_Reversal": {
        "gen": gen_TV_GapFill_Reversal, "space": space_TV_GapFill_Reversal,
        "source": "mac_batch3705_low_param",
    },
}
