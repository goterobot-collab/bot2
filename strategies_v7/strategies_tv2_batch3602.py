"""
Batch 3602 - Wyckoff phases
Sources:
  - Wyckoff method overview: https://school.stockcharts.com/doku.php?id=market_analysis:the_wyckoff_method
  - Spring & Upthrust definitions: https://en.wikipedia.org/wiki/Wyckoff_analysis
  - Accumulation / Distribution schematics: https://www.tradingview.com/support/solutions/43000704424-wyckoff-accumulation-distribution/
  - Secondary test: https://stockcharts.com/articles/wyckoff/
5 strategies, pickle-safe, no lambdas, no look-ahead.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1) TV_Wyckoff_Spring
#    "Sweep prior low + recover + volume contraction -> long"
# ---------------------------------------------------------------------------
def gen_TV_Wyckoff_Spring(df, range_len=30, vol_len=20, vol_contract=0.9, **kw):
    """
    Spring = bar pierces the rolling low of the last `range_len` bars
    (excluding current), then closes back inside the range, while
    current volume <= vol_contract * mean volume (contraction).
    """
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    n = int(range_len)
    prior_lo = l.shift(1).rolling(n, min_periods=n).min()
    vmean = v.rolling(int(vol_len), min_periods=int(vol_len)).mean()

    pierced = (l < prior_lo)
    recovered = (c > prior_lo)
    contracted = (v <= float(vol_contract) * vmean)

    cond = pierced & recovered & contracted
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cond.fillna(False)] = 1
    return sig


def space_TV_Wyckoff_Spring():
    return {
        'range_len': ('int', 10, 100),
        'vol_len': ('int', 10, 100),
        'vol_contract': ('float', 0.3, 1.1),
    }


# ---------------------------------------------------------------------------
# 2) TV_Wyckoff_Upthrust
#    "Sweep prior high + reject + low volume -> short"
# ---------------------------------------------------------------------------
def gen_TV_Wyckoff_Upthrust(df, range_len=30, vol_len=20, vol_contract=0.9, **kw):
    """
    Upthrust = bar pierces rolling high (prior N bars), closes back under it,
    on volume <= vol_contract * mean (low-quality breakout).
    """
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    n = int(range_len)
    prior_hi = h.shift(1).rolling(n, min_periods=n).max()
    vmean = v.rolling(int(vol_len), min_periods=int(vol_len)).mean()

    pierced = (h > prior_hi)
    rejected = (c < prior_hi)
    thin_vol = (v <= float(vol_contract) * vmean)

    cond = pierced & rejected & thin_vol
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cond.fillna(False)] = -1
    return sig


def space_TV_Wyckoff_Upthrust():
    return {
        'range_len': ('int', 10, 100),
        'vol_len': ('int', 10, 100),
        'vol_contract': ('float', 0.3, 1.1),
    }


# ---------------------------------------------------------------------------
# 3) TV_Wyckoff_AccumulationBreakout
#    "Range-bound + narrow range + break up on volume -> long"
# ---------------------------------------------------------------------------
def _atr_wilder(df, n):
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    # Wilder RMA
    return tr.ewm(alpha=1.0 / float(n), adjust=False, min_periods=int(n)).mean()


def gen_TV_Wyckoff_AccumulationBreakout(df, range_len=40, narrow_atr_len=14,
                                        narrow_frac=0.9, vol_mult=1.2, **kw):
    """
    Compression filter: ATR over last `narrow_atr_len` bars < narrow_frac *
    ATR over `range_len` bars (volatility contraction).  Breakout: close
    > prior rolling high over range_len bars, on volume >= vol_mult *
    rolling mean volume -> long.
    """
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    n_r = int(range_len)
    prior_hi = h.shift(1).rolling(n_r, min_periods=n_r).max()

    atr_long = _atr_wilder(df, n_r)
    atr_short = _atr_wilder(df, int(narrow_atr_len))
    compression = atr_short < float(narrow_frac) * atr_long

    vmean = v.rolling(n_r, min_periods=n_r).mean()
    breakout = (c > prior_hi) & (v >= float(vol_mult) * vmean)

    cond = compression & breakout
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cond.fillna(False)] = 1
    return sig


def space_TV_Wyckoff_AccumulationBreakout():
    return {
        'range_len': ('int', 20, 120),
        'narrow_atr_len': ('int', 5, 40),
        'narrow_frac': ('float', 0.4, 1.1),
        'vol_mult': ('float', 1.0, 4.0),
    }


# ---------------------------------------------------------------------------
# 4) TV_Wyckoff_DistributionBreakdown
# ---------------------------------------------------------------------------
def gen_TV_Wyckoff_DistributionBreakdown(df, range_len=40, narrow_atr_len=14,
                                        narrow_frac=0.9, vol_mult=1.2, **kw):
    """
    Mirror of accumulation breakout: volatility compression + close below
    prior rolling low on expanding volume -> short.
    """
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    n_r = int(range_len)
    prior_lo = l.shift(1).rolling(n_r, min_periods=n_r).min()
    atr_long = _atr_wilder(df, n_r)
    atr_short = _atr_wilder(df, int(narrow_atr_len))
    compression = atr_short < float(narrow_frac) * atr_long

    vmean = v.rolling(n_r, min_periods=n_r).mean()
    breakdown_ = (c < prior_lo) & (v >= float(vol_mult) * vmean)

    cond = compression & breakdown_
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cond.fillna(False)] = -1
    return sig


def space_TV_Wyckoff_DistributionBreakdown():
    return {
        'range_len': ('int', 20, 120),
        'narrow_atr_len': ('int', 5, 40),
        'narrow_frac': ('float', 0.4, 1.1),
        'vol_mult': ('float', 1.0, 4.0),
    }


# ---------------------------------------------------------------------------
# 5) TV_Wyckoff_Test_After_Spring
#    "Revisit spring low on lower volume -> long (secondary test / LPS)"
# ---------------------------------------------------------------------------
def gen_TV_Wyckoff_Test_After_Spring(df, range_len=30, lookback=10,
                                     vol_len=20, vol_ratio=0.7, **kw):
    """
    Step 1 - detect a spring in the last `lookback` bars: a bar whose low
    pierced the rolling prior-low (over range_len) and closed back above it.
    Step 2 - current bar retests close to that spring low with volume
    <= vol_ratio * vol_at_spring. Long.
    All shifted to avoid look-ahead.
    """
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    n = int(range_len)
    prior_lo = l.shift(1).rolling(n, min_periods=n).min()
    spring_bar = (l < prior_lo) & (c > prior_lo)

    # For each bar, did we see a spring within the prior `lookback` bars?
    lk = int(lookback)
    had_spring = spring_bar.shift(1).rolling(lk, min_periods=1).max().astype(bool)

    # Reference spring low & volume = min low / matched volume over the same window.
    spring_low_ref = l.where(spring_bar).shift(1).rolling(lk, min_periods=1).min()
    spring_vol_ref = v.where(spring_bar).shift(1).rolling(lk, min_periods=1).max()

    vmean = v.rolling(int(vol_len), min_periods=int(vol_len)).mean()
    tol = 0.25 * vmean.std() if False else None  # unused; kept for future

    # Test = current low within 1 ATR of spring low AND current vol lower than spring vol.
    atr = _atr_wilder(df, int(vol_len))
    near_spring = (l <= spring_low_ref + 0.5 * atr) & (l >= spring_low_ref - 1.0 * atr)
    lower_vol = (v <= float(vol_ratio) * spring_vol_ref)

    cond = had_spring & near_spring & lower_vol & (c > l)  # closes green-ish off the retest
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cond.fillna(False)] = 1
    return sig


def space_TV_Wyckoff_Test_After_Spring():
    return {
        'range_len': ('int', 10, 80),
        'lookback': ('int', 3, 30),
        'vol_len': ('int', 10, 60),
        'vol_ratio': ('float', 0.3, 1.0),
    }


STRATEGY_EXPORT = {
    'TV_Wyckoff_Spring': {
        'gen': gen_TV_Wyckoff_Spring,
        'space': space_TV_Wyckoff_Spring,
        'source': 'https://school.stockcharts.com/doku.php?id=market_analysis:the_wyckoff_method',
    },
    'TV_Wyckoff_Upthrust': {
        'gen': gen_TV_Wyckoff_Upthrust,
        'space': space_TV_Wyckoff_Upthrust,
        'source': 'https://en.wikipedia.org/wiki/Wyckoff_analysis',
    },
    'TV_Wyckoff_AccumulationBreakout': {
        'gen': gen_TV_Wyckoff_AccumulationBreakout,
        'space': space_TV_Wyckoff_AccumulationBreakout,
        'source': 'https://www.tradingview.com/support/solutions/43000704424-wyckoff-accumulation-distribution/',
    },
    'TV_Wyckoff_DistributionBreakdown': {
        'gen': gen_TV_Wyckoff_DistributionBreakdown,
        'space': space_TV_Wyckoff_DistributionBreakdown,
        'source': 'https://www.tradingview.com/support/solutions/43000704424-wyckoff-accumulation-distribution/',
    },
    'TV_Wyckoff_Test_After_Spring': {
        'gen': gen_TV_Wyckoff_Test_After_Spring,
        'space': space_TV_Wyckoff_Test_After_Spring,
        'source': 'https://stockcharts.com/articles/wyckoff/',
    },
}
