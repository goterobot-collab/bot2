"""
Batch 3700 - Mac paralela wave m1 - 5 classic-but-untested families.

Owner: Mac paralela (Claude Code online, rama claude/grail-hunt-bot-noeIq).

Families chosen for ZERO overlap with sandbox 3566-3610 batches or
Mac V8 prod coverage (per MAC_V8_COVERED_DEDUP_20260421.json):

 1. TV_Chandelier_Exit_Entry - ATR trailing line flip (Le Beau 1991)
 2. TV_TRIX_Cross            - triple-smoothed EMA momentum + signal cross
 3. TV_Vortex_Cross          - Vortex VI+/VI- cross (Etlaes 2009)
 4. TV_Aroon_Strong          - Aroon Up/Down with 90+ threshold
 5. TV_HullMA_Slope          - Hull MA slope-sign reversal

All pickle-safe, no lambdas, signal integer in {-1, 0, 1}, .shift(1) applied.
"""
import numpy as np
import pandas as pd


def _atr(df, n):
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _wma(s, n):
    n = int(n)
    w = np.arange(1, n + 1, dtype=float)
    w_sum = w.sum()
    vals = s.to_numpy(dtype=float)
    out = np.full_like(vals, np.nan)
    for i in range(n - 1, len(vals)):
        window = vals[i - n + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        out[i] = (window * w).sum() / w_sum
    return pd.Series(out, index=s.index)


# ----- 1) CHANDELIER EXIT / ENTRY -----
def gen_TV_Chandelier_Exit_Entry(df, atr_len=22, mult=3.0, **kw):
    """Chandelier long line = highest_high(N) - mult*ATR; short = lowest_low(N) + mult*ATR.
    Entry long when close crosses ABOVE long chandelier; short when close crosses BELOW
    short chandelier."""
    n = int(atr_len)
    mult = float(mult)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    atr = _atr(df, n)
    hh = h.rolling(n, min_periods=n).max()
    ll = l.rolling(n, min_periods=n).min()
    chand_long = hh - mult * atr
    chand_short = ll + mult * atr
    cross_up = (c > chand_long) & (c.shift(1) <= chand_long.shift(1))
    cross_dn = (c < chand_short) & (c.shift(1) >= chand_short.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Chandelier_Exit_Entry():
    return {
        'atr_len': ('int', 10, 50),
        'mult': ('float', 1.5, 5.0),
    }


# ----- 2) TRIX CROSS -----
def gen_TV_TRIX_Cross(df, trix_len=15, sig_len=9, require_zero=1, **kw):
    """TRIX = rate-of-change of triple-smoothed EMA of close.
    Signal line = EMA(TRIX, sig_len).
    Entry long = TRIX crosses above signal (and TRIX>0 if require_zero=1).
    Entry short = TRIX crosses below signal (and TRIX<0 if require_zero=1)."""
    n = int(trix_len)
    s = int(sig_len)
    rz = int(require_zero)
    c = df['close'].astype(float)
    e1 = _ema(c, n)
    e2 = _ema(e1, n)
    e3 = _ema(e2, n)
    trix = (e3 - e3.shift(1)) / e3.shift(1) * 10000.0  # bps
    sigl = _ema(trix, s)
    cross_up = (trix > sigl) & (trix.shift(1) <= sigl.shift(1))
    cross_dn = (trix < sigl) & (trix.shift(1) >= sigl.shift(1))
    if rz:
        cross_up = cross_up & (trix > 0)
        cross_dn = cross_dn & (trix < 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_TRIX_Cross():
    return {
        'trix_len': ('int', 8, 30),
        'sig_len': ('int', 5, 20),
        'require_zero': ('int', 0, 1),
    }


# ----- 3) VORTEX CROSS -----
def gen_TV_Vortex_Cross(df, vi_len=14, min_strength=0.0, **kw):
    """Vortex VI+ = sum(|high - prev_low|, n) / sum(TR, n).
    VI-  = sum(|low  - prev_high|, n) / sum(TR, n).
    Long entry: VI+ crosses above VI- (with optional min strength diff).
    Short entry: VI- crosses above VI+."""
    n = int(vi_len)
    ms = float(min_strength)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    vm_plus = (h - l.shift(1)).abs()
    vm_minus = (l - h.shift(1)).abs()
    tr_sum = tr.rolling(n, min_periods=n).sum()
    vi_plus = vm_plus.rolling(n, min_periods=n).sum() / tr_sum.replace(0.0, np.nan)
    vi_minus = vm_minus.rolling(n, min_periods=n).sum() / tr_sum.replace(0.0, np.nan)
    diff = (vi_plus - vi_minus).abs()
    cross_up = (vi_plus > vi_minus) & (vi_plus.shift(1) <= vi_minus.shift(1)) & (diff >= ms)
    cross_dn = (vi_minus > vi_plus) & (vi_minus.shift(1) <= vi_plus.shift(1)) & (diff >= ms)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False).astype(bool)] = 1
    sig[cross_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Vortex_Cross():
    return {
        'vi_len': ('int', 7, 40),
        'min_strength': ('float', 0.0, 0.3),
    }


# ----- 4) AROON STRONG -----
def gen_TV_Aroon_Strong(df, aroon_len=25, threshold=90, mode=0, **kw):
    """Aroon Up = (n - periods_since_highest_high(n)) / n * 100.
    Aroon Down = (n - periods_since_lowest_low(n)) / n * 100.
    mode=0 (cross): long when Aroon Up crosses above threshold AND > Down.
                    short when Aroon Down crosses above threshold AND > Up.
    mode=1 (state): long while Up>=thr and Up>Down; short while Down>=thr and Down>Up.
    """
    n = int(aroon_len)
    thr = float(threshold)
    md = int(mode)
    h = df['high'].astype(float)
    l = df['low'].astype(float)

    def _since_max(arr, n):
        out = np.full(len(arr), np.nan)
        a = arr.to_numpy(dtype=float)
        for i in range(n, len(a)):
            window = a[i - n:i + 1]
            idx_max = int(np.nanargmax(window))
            out[i] = (len(window) - 1 - idx_max)
        return pd.Series(out, index=arr.index)

    def _since_min(arr, n):
        out = np.full(len(arr), np.nan)
        a = arr.to_numpy(dtype=float)
        for i in range(n, len(a)):
            window = a[i - n:i + 1]
            idx_min = int(np.nanargmin(window))
            out[i] = (len(window) - 1 - idx_min)
        return pd.Series(out, index=arr.index)

    since_hi = _since_max(h, n)
    since_lo = _since_min(l, n)
    aroon_up = (n - since_hi) / n * 100.0
    aroon_dn = (n - since_lo) / n * 100.0
    if md == 0:
        long_ok = (aroon_up >= thr) & (aroon_up > aroon_dn) & (aroon_up.shift(1) < thr)
        short_ok = (aroon_dn >= thr) & (aroon_dn > aroon_up) & (aroon_dn.shift(1) < thr)
    else:
        long_ok = (aroon_up >= thr) & (aroon_up > aroon_dn)
        short_ok = (aroon_dn >= thr) & (aroon_dn > aroon_up)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_ok.shift(1).fillna(False).astype(bool)] = 1
    sig[short_ok.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Aroon_Strong():
    return {
        'aroon_len': ('int', 10, 40),
        'threshold': ('int', 70, 100),
        'mode': ('int', 0, 1),
    }


# ----- 5) HULL MA SLOPE -----
def gen_TV_HullMA_Slope(df, hma_len=20, slope_bars=1, **kw):
    """Hull MA = WMA(2*WMA(n/2) - WMA(n), sqrt(n)).
    Long entry on slope flip from negative to positive (HMA[t] > HMA[t-slope_bars] after being lower).
    Short entry on slope flip from positive to negative.
    """
    n = int(hma_len)
    sb = int(slope_bars)
    c = df['close'].astype(float)
    half = max(2, n // 2)
    sq = max(2, int(round(np.sqrt(n))))
    w_half = _wma(c, half)
    w_full = _wma(c, n)
    raw = 2.0 * w_half - w_full
    hma = _wma(raw, sq)
    slope = hma - hma.shift(sb)
    slope_prev = hma.shift(sb) - hma.shift(2 * sb)
    flip_up = (slope > 0) & (slope_prev <= 0)
    flip_dn = (slope < 0) & (slope_prev >= 0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[flip_up.shift(1).fillna(False).astype(bool)] = 1
    sig[flip_dn.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_HullMA_Slope():
    return {
        'hma_len': ('int', 8, 60),
        'slope_bars': ('int', 1, 5),
    }


STRATEGY_EXPORT = {
    "TV_Chandelier_Exit_Entry": {
        "gen": gen_TV_Chandelier_Exit_Entry,
        "space": space_TV_Chandelier_Exit_Entry,
        "source": "mac_batch3700",
    },
    "TV_TRIX_Cross": {
        "gen": gen_TV_TRIX_Cross,
        "space": space_TV_TRIX_Cross,
        "source": "mac_batch3700",
    },
    "TV_Vortex_Cross": {
        "gen": gen_TV_Vortex_Cross,
        "space": space_TV_Vortex_Cross,
        "source": "mac_batch3700",
    },
    "TV_Aroon_Strong": {
        "gen": gen_TV_Aroon_Strong,
        "space": space_TV_Aroon_Strong,
        "source": "mac_batch3700",
    },
    "TV_HullMA_Slope": {
        "gen": gen_TV_HullMA_Slope,
        "space": space_TV_HullMA_Slope,
        "source": "mac_batch3700",
    },
}
