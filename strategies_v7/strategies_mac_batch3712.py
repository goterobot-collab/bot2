"""
Batch 3712 - Mac paralela wave m13 - 5 advanced cycle / entropy / fractal.

All low-param (1-3 each), pickle-safe, signal int {-1,0,1}, .shift(1).
ZERO overlap with sandbox 3566-3610 (incl. Hurst/Detrended_Fluctuation/
FractalDimension), Mac V8 prod, Mac 3700-3711.

 1. TV_SampleEntropy_RegimeFlip - Sample Entropy of returns: low entropy
                                   (predictable) -> trade trend, high entropy
                                   -> stay flat (NEW, no entropy in repo)
 2. TV_PermutationEntropy_Switch - Bandt-Pompe permutation entropy threshold
                                   regime indicator (NEW)
 3. TV_LempelZiv_Complexity     - LZ complexity of binary returns string;
                                   compressed = trend, random = stay flat (NEW)
 4. TV_RangeBreakout_Z          - z-score of bar range vs rolling mean range;
                                   surge breakout direction by close pos (NEW)
 5. TV_AutoCorr_Sign_Switch     - 1-lag autocorrelation rolling: positive
                                   (momentum regime) vs negative (revert),
                                   trade with regime-appropriate signal
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


# ----- 1) SAMPLE ENTROPY REGIME FLIP -----
def gen_TV_SampleEntropy_RegimeFlip(df, win=30, m=2, low_thr=0.5, **kw):
    """Sample Entropy (SampEn) of normalized returns over rolling window.
    Low SampEn (< low_thr) -> highly predictable -> trend regime.
    In trend regime: long if return positive direction (last 5d ret > 0).
                     short if negative direction.
    """
    w = int(win)
    md = int(m)
    th = float(low_thr)
    c = df['close'].astype(float)
    ret = c.pct_change()
    sd = ret.rolling(w, min_periods=w).std(ddof=0)

    def _sampen(arr):
        a = arr.to_numpy(dtype=float)
        if np.any(np.isnan(a)) or len(a) < md + 2:
            return np.nan
        a_norm = (a - a.mean()) / (a.std(ddof=0) + 1e-10)
        r = 0.2  # tolerance
        # Count matches
        def _phi(mm):
            xs = np.array([a_norm[i:i + mm] for i in range(len(a_norm) - mm + 1)])
            n = len(xs)
            if n < 2:
                return 0.0
            count = 0
            for i in range(n - 1):
                d = np.max(np.abs(xs[i + 1:] - xs[i]), axis=1)
                count += int((d <= r).sum())
            return count / (n * (n - 1) / 2 + 1e-10)
        b = _phi(md)
        a_v = _phi(md + 1)
        if b <= 0 or a_v <= 0:
            return np.nan
        return -np.log(a_v / b)

    sampen = ret.rolling(w, min_periods=w).apply(_sampen, raw=False)
    low_entropy = sampen < th
    short_dir = ret.rolling(5, min_periods=5).mean() < 0
    long_dir = ret.rolling(5, min_periods=5).mean() > 0
    long_setup = low_entropy & long_dir
    short_setup = low_entropy & short_dir
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_SampleEntropy_RegimeFlip():
    return {
        'win': ('int', 20, 60),
        'm': ('int', 2, 3),
        'low_thr': ('float', 0.3, 1.5),
    }


# ----- 2) BANDT-POMPE PERMUTATION ENTROPY -----
def gen_TV_PermutationEntropy_Switch(df, win=50, order=3, high_thr=0.85, **kw):
    """Bandt-Pompe Permutation Entropy: count distinct ordinal patterns
    of length `order` in window. Normalized to [0,1] by log(order!).
    HIGH PE (>high_thr) = random = mean revert.
    LOW PE = predictable = trend follow.
    Direction by recent SMA slope.
    """
    w = int(win)
    o = int(order)
    th = float(high_thr)
    c = df['close'].astype(float)

    from math import factorial
    norm = np.log(factorial(o))

    def _perm_entropy(arr):
        a = arr.to_numpy(dtype=float)
        if np.any(np.isnan(a)) or len(a) < o:
            return np.nan
        patterns = {}
        for i in range(len(a) - o + 1):
            window = a[i:i + o]
            pat = tuple(np.argsort(window))
            patterns[pat] = patterns.get(pat, 0) + 1
        total = sum(patterns.values())
        probs = np.array([v / total for v in patterns.values()])
        return -np.sum(probs * np.log(probs)) / norm

    pe = c.rolling(w, min_periods=w).apply(_perm_entropy, raw=False)
    sma_short = _sma(c, 10)
    sma_long = _sma(c, 30)
    high_pe = pe > th  # random regime: trade reversion
    low_pe = pe < th  # trend regime: follow trend
    long_trend = low_pe & (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
    short_trend = low_pe & (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))
    long_revert = high_pe & (c < sma_short) & (c.shift(1) >= sma_short.shift(1))
    short_revert = high_pe & (c > sma_short) & (c.shift(1) <= sma_short.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(long_trend | long_revert).shift(1).fillna(False).astype(bool)] = 1
    sig[(short_trend | short_revert).shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_PermutationEntropy_Switch():
    return {
        'win': ('int', 30, 80),
        'order': ('int', 3, 5),
        'high_thr': ('float', 0.75, 0.95),
    }


# ----- 3) LEMPEL-ZIV COMPLEXITY -----
def gen_TV_LempelZiv_Complexity(df, win=60, low_thr=0.5, **kw):
    """LZ complexity of binary returns: 1 if ret>0 else 0.
    Low LZ (compressed) = trend regime.
    Trade direction by recent net return sign.
    """
    w = int(win)
    th = float(low_thr)
    c = df['close'].astype(float)
    ret = c.pct_change()
    binary = (ret > 0).astype(int)

    def _lz(arr):
        a = arr.to_numpy(dtype=int)
        if len(a) < 2:
            return np.nan
        s = ''.join(str(x) for x in a)
        n = len(s)
        i, c_count, l = 0, 1, 1
        while True:
            if i + l > n:
                break
            sub = s[i:i + l]
            prev = s[:i + l - 1]
            if sub in prev:
                l += 1
            else:
                c_count += 1
                i += l
                l = 1
            if i + l > n:
                break
        # Normalize by upper bound n / log2(n)
        upper = n / np.log2(n)
        return c_count / upper

    lz = binary.rolling(w, min_periods=w).apply(_lz, raw=False)
    low_lz = lz < th
    short_dir = ret.rolling(10, min_periods=10).sum() < 0
    long_dir = ret.rolling(10, min_periods=10).sum() > 0
    long_setup = low_lz & long_dir
    short_setup = low_lz & short_dir
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_LempelZiv_Complexity():
    return {
        'win': ('int', 30, 100),
        'low_thr': ('float', 0.9, 1.4),
    }


# ----- 4) RANGE BREAKOUT Z-SCORE -----
def gen_TV_RangeBreakout_Z(df, win=20, z_threshold=2.0, **kw):
    """Bar range = high - low. z = (bar_range - mean_range) / std_range
    over rolling window.
    Surge (z > threshold): direction = sign(close - midpoint of bar).
    """
    w = int(win)
    zt = float(z_threshold)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    rng = h - l
    mu = rng.rolling(w, min_periods=w).mean()
    sd = rng.rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
    z = (rng - mu) / sd
    midpoint = (h + l) / 2.0
    surge = (z > zt) & (z.shift(1) <= zt)
    long_setup = surge & (c > midpoint)
    short_setup = surge & (c < midpoint)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_RangeBreakout_Z():
    return {
        'win': ('int', 10, 50),
        'z_threshold': ('float', 1.5, 3.5),
    }


# ----- 5) AUTOCORR SIGN SWITCH -----
def gen_TV_AutoCorr_Sign_Switch(df, win=50, **kw):
    """1-lag autocorrelation of returns over rolling window.
    Positive AC: momentum regime -> follow recent direction.
    Negative AC: revert regime -> oppose recent direction.
    """
    w = int(win)
    c = df['close'].astype(float)
    ret = c.pct_change()

    def _autocorr(arr):
        a = arr.to_numpy(dtype=float)
        if np.any(np.isnan(a)) or len(a) < 3:
            return np.nan
        return np.corrcoef(a[:-1], a[1:])[0, 1]

    ac = ret.rolling(w, min_periods=w).apply(_autocorr, raw=False)
    short_ret = ret.rolling(5, min_periods=5).sum()
    pos_regime = ac > 0
    neg_regime = ac < 0
    long_setup = (pos_regime & (short_ret > 0)) | (neg_regime & (short_ret < 0))
    short_setup = (pos_regime & (short_ret < 0)) | (neg_regime & (short_ret > 0))
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_AutoCorr_Sign_Switch():
    return {
        'win': ('int', 30, 100),
    }


STRATEGY_EXPORT = {
    "TV_SampleEntropy_RegimeFlip": {
        "gen": gen_TV_SampleEntropy_RegimeFlip,
        "space": space_TV_SampleEntropy_RegimeFlip,
        "source": "mac_batch3712_SampEn_Richman2000",
    },
    "TV_PermutationEntropy_Switch": {
        "gen": gen_TV_PermutationEntropy_Switch,
        "space": space_TV_PermutationEntropy_Switch,
        "source": "mac_batch3712_BandtPompe_PermutationEntropy_2002",
    },
    "TV_LempelZiv_Complexity": {
        "gen": gen_TV_LempelZiv_Complexity,
        "space": space_TV_LempelZiv_Complexity,
        "source": "mac_batch3712_LempelZiv_complexity_1976",
    },
    "TV_RangeBreakout_Z": {
        "gen": gen_TV_RangeBreakout_Z, "space": space_TV_RangeBreakout_Z,
        "source": "mac_batch3712_range_zscore_surge",
    },
    "TV_AutoCorr_Sign_Switch": {
        "gen": gen_TV_AutoCorr_Sign_Switch,
        "space": space_TV_AutoCorr_Sign_Switch,
        "source": "mac_batch3712_AC1_regime_switch",
    },
}
