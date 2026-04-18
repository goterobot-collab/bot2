"""
Batch 3601 - Order flow proxies (OHLCV-only approximations)
Sources:
  - CVD: https://www.tradingview.com/support/solutions/43000589100-cumulative-volume-delta-cvd/
  - Kyle's Lambda: https://en.wikipedia.org/wiki/Market_impact  (Kyle 1985)
  - VPIN: Easley, Lopez de Prado, O'Hara (2012) https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695596
  - BVC: https://www.cmegroup.com/education/files/vpin-toxicity-cme-white-paper.pdf
  - Footprint / absorption: https://www.tradingview.com/support/solutions/43000669022-volume-footprint/
5 strategies, pickle-safe, no lambdas, no look-ahead.
"""
import numpy as np
import pandas as pd


def _safe_range(df):
    rng = (df['high'] - df['low']).astype(float)
    # avoid divide-by-zero on doji/flat bars
    rng = rng.where(rng > 0, np.nan)
    return rng


# ---------------------------------------------------------------------------
# 1) TV_CVD_Crossover
# ---------------------------------------------------------------------------
def gen_TV_CVD_Crossover(df, ma_len=50, **kw):
    """
    Per-bar volume delta proxy = (close - open) / (high - low) * volume.
    CVD = cumulative sum. Signal = CVD cross over/under its rolling mean.
    """
    o = df['open'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    rng = _safe_range(df)
    bar_delta = ((c - o) / rng) * v
    bar_delta = bar_delta.fillna(0.0)
    cvd = bar_delta.cumsum()
    ma = cvd.rolling(int(ma_len), min_periods=int(ma_len)).mean()

    long_cond = (cvd > ma) & (cvd.shift(1) <= ma.shift(1))
    short_cond = (cvd < ma) & (cvd.shift(1) >= ma.shift(1))

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cond.fillna(False)] = 1
    sig[short_cond.fillna(False)] = -1
    return sig


def space_TV_CVD_Crossover():
    return {
        'ma_len': ('int', 10, 200),
    }


# ---------------------------------------------------------------------------
# 2) TV_VolumeDelta_Imbalance
# ---------------------------------------------------------------------------
def gen_TV_VolumeDelta_Imbalance(df, lookback=20, thresh=0.70, **kw):
    """
    Per-bar up-volume if close>=open else down-volume.
    Ratio = up / (up+down) over rolling lookback.
    Extreme high ratio (>thresh) fades short; extreme low (<1-thresh) fades long.
    """
    o = df['open'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    up_v = v.where(c >= o, 0.0)
    dn_v = v.where(c < o, 0.0)
    n = int(lookback)
    up_sum = up_v.rolling(n, min_periods=n).sum()
    dn_sum = dn_v.rolling(n, min_periods=n).sum()
    total = up_sum + dn_sum
    ratio = (up_sum / total.replace(0.0, np.nan))

    t = float(thresh)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(ratio >= t).fillna(False)] = -1        # too much buying -> fade
    sig[(ratio <= (1.0 - t)).fillna(False)] = 1  # too much selling -> fade
    return sig


def space_TV_VolumeDelta_Imbalance():
    return {
        'lookback': ('int', 5, 100),
        'thresh': ('float', 0.55, 0.90),
    }


# ---------------------------------------------------------------------------
# 3) TV_Kyle_Lambda_Fade
# ---------------------------------------------------------------------------
def gen_TV_Kyle_Lambda_Fade(df, lookback=50, z_thresh=2.0, **kw):
    """
    Kyle's lambda proxy per bar: |ret| / signed_volume, where
    signed_volume = (close-open)/range * volume.
    A lambda spike = abnormal price impact per unit of flow, usually
    a move that exhausts itself. We fade in the opposite direction of
    the current bar when the z-scored lambda exceeds z_thresh.
    """
    o = df['open'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    rng = _safe_range(df)
    signed_v = ((c - o) / rng) * v
    # protect against ~0 signed flow
    abs_signed = signed_v.abs().replace(0.0, np.nan)
    ret = (c / c.shift(1) - 1.0).abs()
    lam = ret / abs_signed
    lam = lam.replace([np.inf, -np.inf], np.nan)

    n = int(lookback)
    mu = lam.rolling(n, min_periods=n).mean()
    sd = lam.rolling(n, min_periods=n).std(ddof=0)
    z = (lam - mu) / sd.replace(0.0, np.nan)

    spike = (z >= float(z_thresh)).fillna(False)
    up_bar = (c > o)
    dn_bar = (c < o)

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[spike & up_bar] = -1   # impulsive up with thin flow -> fade short
    sig[spike & dn_bar] = 1    # impulsive down with thin flow -> fade long
    return sig


def space_TV_Kyle_Lambda_Fade():
    return {
        'lookback': ('int', 20, 200),
        'z_thresh': ('float', 1.0, 4.0),
    }


# ---------------------------------------------------------------------------
# 4) TV_VPIN_Toxic_Flow
# ---------------------------------------------------------------------------
def gen_TV_VPIN_Toxic_Flow(df, bucket=50, sigma_len=50, hi=0.75, lo=0.35, **kw):
    """
    Bulk Volume Classification VPIN (Easley, Lopez de Prado, O'Hara 2012).
    We approximate per-bar buy fraction with a normal CDF on standardized
    returns over sigma_len; sell fraction = 1 - buy.
    VPIN = rolling mean over 'bucket' bars of |buy_vol - sell_vol| / total_vol.
    High VPIN => toxic/informed flow active => stay out (signal 0 / exit).
    Low VPIN  => benign tape => take the sign of the last bar (momentum entry).
    """
    from math import erf, sqrt
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    ret = np.log(c / c.shift(1))
    n_sig = int(sigma_len)
    sd = ret.rolling(n_sig, min_periods=n_sig).std(ddof=0)
    z = ret / sd.replace(0.0, np.nan)
    # vectorised Phi via numpy erf
    buy_frac = 0.5 * (1.0 + np.vectorize(erf)(z.to_numpy() / sqrt(2.0)))
    buy_frac = pd.Series(buy_frac, index=df.index).fillna(0.5)
    buy_v = v * buy_frac
    sell_v = v * (1.0 - buy_frac)

    n_b = int(bucket)
    num = (buy_v - sell_v).abs().rolling(n_b, min_periods=n_b).sum()
    den = v.rolling(n_b, min_periods=n_b).sum().replace(0.0, np.nan)
    vpin = num / den

    o = df['open'].astype(float)
    up_bar = (c > o)
    dn_bar = (c < o)

    sig = pd.Series(0, index=df.index, dtype=int)
    # low VPIN: benign -> follow last bar
    low_mask = (vpin <= float(lo)).fillna(False)
    sig[low_mask & up_bar] = 1
    sig[low_mask & dn_bar] = -1
    # high VPIN explicitly flat (already 0) — no entry
    return sig


def space_TV_VPIN_Toxic_Flow():
    return {
        'bucket': ('int', 20, 200),
        'sigma_len': ('int', 20, 200),
        'hi': ('float', 0.60, 0.90),
        'lo': ('float', 0.20, 0.50),
    }


# ---------------------------------------------------------------------------
# 5) TV_Footprint_Volume_Climax
# ---------------------------------------------------------------------------
def gen_TV_Footprint_Volume_Climax(df, swing_len=20, vol_mult=2.0, **kw):
    """
    Absorption / climax proxy: if current bar prints volume >= vol_mult * rolling
    mean volume and the bar low equals the rolling swing low, that's absorption
    at support -> long. Symmetric for swing high -> short.
    """
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float)
    n = int(swing_len)
    swing_lo = l.rolling(n, min_periods=n).min()
    swing_hi = h.rolling(n, min_periods=n).max()
    vmean = v.rolling(n, min_periods=n).mean()
    climax = (v >= float(vol_mult) * vmean)

    at_low = (l <= swing_lo)          # current low is the rolling min
    at_high = (h >= swing_hi)

    long_cond = climax & at_low
    short_cond = climax & at_high

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cond.fillna(False)] = 1
    sig[short_cond.fillna(False)] = -1
    return sig


def space_TV_Footprint_Volume_Climax():
    return {
        'swing_len': ('int', 5, 100),
        'vol_mult': ('float', 1.2, 5.0),
    }


STRATEGY_EXPORT = {
    'TV_CVD_Crossover': {
        'gen': gen_TV_CVD_Crossover,
        'space': space_TV_CVD_Crossover,
        'source': 'https://www.tradingview.com/support/solutions/43000589100-cumulative-volume-delta-cvd/',
    },
    'TV_VolumeDelta_Imbalance': {
        'gen': gen_TV_VolumeDelta_Imbalance,
        'space': space_TV_VolumeDelta_Imbalance,
        'source': 'https://www.tradingview.com/support/solutions/43000589100-cumulative-volume-delta-cvd/',
    },
    'TV_Kyle_Lambda_Fade': {
        'gen': gen_TV_Kyle_Lambda_Fade,
        'space': space_TV_Kyle_Lambda_Fade,
        'source': 'https://en.wikipedia.org/wiki/Market_impact',
    },
    'TV_VPIN_Toxic_Flow': {
        'gen': gen_TV_VPIN_Toxic_Flow,
        'space': space_TV_VPIN_Toxic_Flow,
        'source': 'https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695596',
    },
    'TV_Footprint_Volume_Climax': {
        'gen': gen_TV_Footprint_Volume_Climax,
        'space': space_TV_Footprint_Volume_Climax,
        'source': 'https://www.tradingview.com/support/solutions/43000669022-volume-footprint/',
    },
}
