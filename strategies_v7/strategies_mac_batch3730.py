"""
Batch 3730 - Mac paralela wave m31 - 5 microstructure / cycle novel.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3729.

 1. TV_LastNHours_RangePct      - last N hours bar range as % of daily range
                                   (24h proxy)
 2. TV_ATR_Channel_DeepRevert   - close pierces deep N*ATR channel +
                                   returns (3*ATR pierce, deeper than m4)
 3. TV_DualMA_AlignedBreakout   - short MA aligned with long MA + breakout
                                   (single-TF proxy for MTF alignment)
 4. TV_VRegimeShift_Detection   - sudden variance ratio drop (regime shift)
 5. TV_HalvingCycle_DayProxy    - days % 1460 (BTC halving cycle proxy)
                                   regime indicator
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


def _atr(df, n):
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()


# ----- 1) LAST N HOURS RANGE % -----
def gen_TV_LastNHours_RangePct(df, win=24, n_recent=4, pct_thr=70, **kw):
    """Last n_recent bars range divided by previous (win - n_recent) bars range.
    > pct_thr means recent activity is dominating (breakout setup).
    Direction = recent return sign.
    """
    w = int(win)
    nr = int(n_recent)
    pt = float(pct_thr) / 100.0
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    recent_range = (h.rolling(nr, min_periods=nr).max()
                    - l.rolling(nr, min_periods=nr).min())
    full_range = (h.rolling(w, min_periods=w).max()
                  - l.rolling(w, min_periods=w).min()).replace(0.0, np.nan)
    pct = recent_range / full_range
    short_ret = c.pct_change().rolling(nr, min_periods=nr).sum()
    long_setup = (pct > pt) & (pct.shift(1) <= pt) & (short_ret > 0)
    short_setup = (pct > pt) & (pct.shift(1) <= pt) & (short_ret < 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_LastNHours_RangePct():
    return {
        'win': ('int', 12, 60),
        'n_recent': ('int', 2, 12),
        'pct_thr': ('int', 50, 90),
    }


# ----- 2) DEEP ATR CHANNEL REVERT -----
def gen_TV_ATR_Channel_DeepRevert(df, ema_len=20, atr_len=14, mult=3.0, **kw):
    """Close pierces EMA +/- 3*ATR (deep extreme), returns toward EMA.
    Long: prev_close was BELOW EMA - mult*ATR, now close > EMA - mult*ATR.
    Short: prev_close was ABOVE EMA + mult*ATR, now close < EMA + mult*ATR.
    Deeper than typical (m4 Wilder ATR Channel uses mult ~2).
    """
    el = int(ema_len)
    al = int(atr_len)
    m = float(mult)
    c = df['close'].astype(float)
    ema = _ema(c, el)
    atr = _atr(df, al)
    upper = ema + m * atr
    lower = ema - m * atr
    long_bounce = (c.shift(1) < lower.shift(1)) & (c > lower)
    short_bounce = (c.shift(1) > upper.shift(1)) & (c < upper)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_bounce.shift(1).fillna(False).astype(bool)] = 1
    sig[short_bounce.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_ATR_Channel_DeepRevert():
    return {
        'ema_len': ('int', 10, 50),
        'atr_len': ('int', 7, 25),
        'mult': ('float', 2.0, 4.5),
    }


# ----- 3) DUAL MA ALIGNED BREAKOUT -----
def gen_TV_DualMA_AlignedBreakout(df, short_ma=20, long_ma=100, breakout_len=20, **kw):
    """SMA_short aligned with SMA_long (both above/below close at same time)
    + close breaks N-bar high/low.
    Single-TF proxy for multi-TF alignment.
    """
    sm = int(short_ma)
    lm = int(long_ma)
    bl = int(breakout_len)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    ma_s = _sma(c, sm)
    ma_l = _sma(c, lm)
    bullish_aligned = (c > ma_s) & (ma_s > ma_l)
    bearish_aligned = (c < ma_s) & (ma_s < ma_l)
    prior_high = h.rolling(bl, min_periods=bl).max().shift(1)
    prior_low = l.rolling(bl, min_periods=bl).min().shift(1)
    long_break = bullish_aligned & (c > prior_high)
    short_break = bearish_aligned & (c < prior_low)
    long_entry = long_break & ~(long_break.shift(1).fillna(False))
    short_entry = short_break & ~(short_break.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_DualMA_AlignedBreakout():
    return {
        'short_ma': ('int', 10, 35),
        'long_ma': ('int', 60, 180),
        'breakout_len': ('int', 10, 50),
    }


# ----- 4) VARIANCE REGIME SHIFT DETECTION -----
def gen_TV_VRegimeShift_Detection(df, short_win=20, long_win=100, shift_pct=40, **kw):
    """Short-term variance vs long-term variance.
    ratio = var_short / var_long.
    Big drop (ratio falls below shift_pct% of long-run avg) = regime shifting
    to LOW vol from HIGH vol. Direction = sign of recent return.
    """
    sw = int(short_win)
    lw = int(long_win)
    sp = float(shift_pct) / 100.0
    c = df['close'].astype(float)
    ret = c.pct_change()
    var_s = ret.rolling(sw, min_periods=sw).var(ddof=0)
    var_l = ret.rolling(lw, min_periods=lw).var(ddof=0)
    ratio = var_s / var_l.replace(0.0, np.nan)
    avg_ratio = _sma(ratio, lw)
    drop_below = (ratio < sp * avg_ratio) & (ratio.shift(1) >= sp * avg_ratio.shift(1))
    short_ret = ret.rolling(5, min_periods=5).sum()
    long_setup = drop_below & (short_ret > 0)
    short_setup = drop_below & (short_ret < 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_VRegimeShift_Detection():
    return {
        'short_win': ('int', 10, 40),
        'long_win': ('int', 60, 250),
        'shift_pct': ('int', 20, 70),
    }


# ----- 5) HALVING CYCLE DAY PROXY -----
def gen_TV_HalvingCycle_DayProxy(df, phase_thr=120, ret_len=20, **kw):
    """Approximate BTC halving cycle as 1460 days (4 years).
    Phase = (epoch_days % 1460). Different phases of cycle have different
    behavior.
    Late phase (phase > 1340 = within 120 days of next halving):
       expect mean-revert (already pumped pre-halving narrative).
    Mid phase: follow trend.
    """
    pt = int(phase_thr)
    rl = int(ret_len)
    c = df['close'].astype(float)
    idx = df.index
    # idx dtype is datetime64[ms, UTC] -> int64 gives milliseconds since epoch.
    # Convert via pd.Timestamp epoch difference for robustness regardless of unit.
    epoch_days = ((idx - pd.Timestamp('1970-01-01', tz='UTC')).days)
    phase = pd.Series(np.asarray(epoch_days) % 1460, index=idx)
    late_phase = phase > (1460 - pt)
    mid_phase = (phase > pt) & (phase < (1460 - pt))
    ret = (c / c.shift(rl) - 1.0)
    long_setup = (late_phase & (ret < -0.05)) | (mid_phase & (ret > 0.05) & (ret.shift(1) <= 0.05))
    short_setup = (late_phase & (ret > 0.05)) | (mid_phase & (ret < -0.05) & (ret.shift(1) >= -0.05))
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_HalvingCycle_DayProxy():
    return {
        'phase_thr': ('int', 60, 200),
        'ret_len': ('int', 10, 60),
    }


STRATEGY_EXPORT = {
    "TV_LastNHours_RangePct": {
        "gen": gen_TV_LastNHours_RangePct, "space": space_TV_LastNHours_RangePct,
        "source": "mac_batch3730_recent_vs_full_range_dominance",
    },
    "TV_ATR_Channel_DeepRevert": {
        "gen": gen_TV_ATR_Channel_DeepRevert, "space": space_TV_ATR_Channel_DeepRevert,
        "source": "mac_batch3730_deep_atr_channel_pierce_revert",
    },
    "TV_DualMA_AlignedBreakout": {
        "gen": gen_TV_DualMA_AlignedBreakout, "space": space_TV_DualMA_AlignedBreakout,
        "source": "mac_batch3730_dual_MA_alignment_breakout",
    },
    "TV_VRegimeShift_Detection": {
        "gen": gen_TV_VRegimeShift_Detection, "space": space_TV_VRegimeShift_Detection,
        "source": "mac_batch3730_variance_regime_shift",
    },
    "TV_HalvingCycle_DayProxy": {
        "gen": gen_TV_HalvingCycle_DayProxy, "space": space_TV_HalvingCycle_DayProxy,
        "source": "mac_batch3730_BTC_halving_cycle_phase_proxy",
    },
}
