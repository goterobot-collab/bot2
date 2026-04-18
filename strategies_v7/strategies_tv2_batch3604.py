"""
Batch 3604 - Fractal / multi-scale indicators
Sources:
  - Hurst exponent (R/S): Hurst H.E. (1951)
  - DFA: Peng et al. (1994)
  - Higuchi FD: Higuchi T. (1988)
  - ZigZag, Williams Fractal
5 strategies, pickle-safe, no lambdas, no look-ahead.
"""
import numpy as np
import pandas as pd


def _rs_hurst_window(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 16 or not np.isfinite(x).all():
        return np.nan
    r = np.diff(np.log(np.where(x > 0, x, np.nan)))
    r = r[np.isfinite(r)]
    m = r.size
    if m < 16:
        return np.nan
    sizes = []
    s = 8
    while s <= m // 2:
        sizes.append(s)
        s *= 2
    if len(sizes) < 2:
        return np.nan
    rs_vals = []
    for s in sizes:
        k = m // s
        if k < 1:
            continue
        chunks = r[: k * s].reshape(k, s)
        mean = chunks.mean(axis=1, keepdims=True)
        dev = chunks - mean
        csum = dev.cumsum(axis=1)
        R = csum.max(axis=1) - csum.min(axis=1)
        S = chunks.std(axis=1, ddof=0)
        valid = S > 0
        if not valid.any():
            continue
        rs_vals.append(np.log((R[valid] / S[valid]).mean()))
    if len(rs_vals) < 2:
        return np.nan
    xs = np.log(np.asarray(sizes[: len(rs_vals)], dtype=float))
    ys = np.asarray(rs_vals, dtype=float)
    xm, ym = xs.mean(), ys.mean()
    denom = ((xs - xm) ** 2).sum()
    if denom <= 0:
        return np.nan
    return float(((xs - xm) * (ys - ym)).sum() / denom)


def _rolling_apply(series, window, func):
    vals = series.to_numpy(dtype=float)
    n = vals.size
    w = int(window)
    out = np.full(n, np.nan, dtype=float)
    if w <= 1 or w > n:
        return pd.Series(out, index=series.index)
    for i in range(w - 1, n):
        out[i] = func(vals[i - w + 1: i + 1])
    return pd.Series(out, index=series.index)


def gen_TV_Hurst_Regime_Switch(df, hurst_len=128, rsi_len=14, h_trend=0.55,
                                h_rev=0.45, bo_len=20, rsi_os=30, **kw):
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    hurst = _rolling_apply(c, int(hurst_len), _rs_hurst_window).shift(1)
    diff = c.diff()
    up = diff.clip(lower=0.0)
    dn = (-diff).clip(lower=0.0)
    rn = int(rsi_len)
    ema_up = up.ewm(alpha=1.0 / rn, adjust=False, min_periods=rn).mean()
    ema_dn = dn.ewm(alpha=1.0 / rn, adjust=False, min_periods=rn).mean()
    rs = ema_up / ema_dn.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    prior_hi = h.shift(1).rolling(int(bo_len), min_periods=int(bo_len)).max()
    prior_lo = l.shift(1).rolling(int(bo_len), min_periods=int(bo_len)).min()
    trend_long = (hurst >= float(h_trend)) & (c > prior_hi)
    trend_short = (hurst >= float(h_trend)) & (c < prior_lo)
    rev_long = (hurst <= float(h_rev)) & (rsi <= float(rsi_os))
    rev_short = (hurst <= float(h_rev)) & (rsi >= (100.0 - float(rsi_os)))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(trend_long | rev_long).fillna(False)] = 1
    sig[(trend_short | rev_short).fillna(False)] = -1
    return sig


def space_TV_Hurst_Regime_Switch():
    return {'hurst_len': ('int', 64, 512), 'rsi_len': ('int', 7, 30),
            'h_trend': ('float', 0.52, 0.70), 'h_rev': ('float', 0.30, 0.48),
            'bo_len': ('int', 10, 60), 'rsi_os': ('int', 15, 40)}


def _dfa_alpha(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 32 or not np.isfinite(x).all():
        return np.nan
    y = np.cumsum(x - x.mean())
    sizes = []
    s = 8
    while s <= n // 4:
        sizes.append(s)
        s = int(s * 1.5)
    if len(sizes) < 3:
        return np.nan
    F = []
    for s in sizes:
        k = n // s
        if k < 2:
            continue
        segs = y[: k * s].reshape(k, s)
        t = np.arange(s, dtype=float)
        tm = t.mean()
        denom = ((t - tm) ** 2).sum()
        if denom <= 0:
            continue
        ym = segs.mean(axis=1, keepdims=True)
        num = ((t - tm) * (segs - ym)).sum(axis=1)
        slope = num / denom
        intercept = ym.squeeze(axis=1) - slope * tm
        trend = slope[:, None] * t[None, :] + intercept[:, None]
        resid = segs - trend
        F.append(np.sqrt(float((resid ** 2).mean())))
    if len(F) < 3:
        return np.nan
    xs = np.log(np.asarray(sizes[: len(F)], dtype=float))
    ys = np.log(np.asarray(F, dtype=float))
    if not np.isfinite(ys).all():
        return np.nan
    xm, ym = xs.mean(), ys.mean()
    d = ((xs - xm) ** 2).sum()
    if d <= 0:
        return np.nan
    return float(((xs - xm) * (ys - ym)).sum() / d)


def gen_TV_Detrended_Fluctuation(df, dfa_len=200, ema_fast=20, ema_slow=50,
                                  bb_len=20, bb_mult=2.0, alpha_trend=0.55,
                                  alpha_rev=0.45, **kw):
    c = df['close'].astype(float)
    logp = np.log(c.replace(0.0, np.nan))
    alpha = _rolling_apply(logp, int(dfa_len), _dfa_alpha).shift(1)
    ema_f = c.ewm(span=int(ema_fast), adjust=False, min_periods=int(ema_fast)).mean()
    ema_s = c.ewm(span=int(ema_slow), adjust=False, min_periods=int(ema_slow)).mean()
    n_bb = int(bb_len)
    ma = c.rolling(n_bb, min_periods=n_bb).mean()
    sd = c.rolling(n_bb, min_periods=n_bb).std(ddof=0)
    upper = ma + float(bb_mult) * sd
    lower = ma - float(bb_mult) * sd
    trend_long = (alpha >= float(alpha_trend)) & (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
    trend_short = (alpha >= float(alpha_trend)) & (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))
    rev_long = (alpha <= float(alpha_rev)) & (c < lower)
    rev_short = (alpha <= float(alpha_rev)) & (c > upper)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(trend_long | rev_long).fillna(False)] = 1
    sig[(trend_short | rev_short).fillna(False)] = -1
    return sig


def space_TV_Detrended_Fluctuation():
    return {'dfa_len': ('int', 100, 500), 'ema_fast': ('int', 5, 50),
            'ema_slow': ('int', 20, 200), 'bb_len': ('int', 10, 60),
            'bb_mult': ('float', 1.5, 3.0), 'alpha_trend': ('float', 0.52, 0.70),
            'alpha_rev': ('float', 0.30, 0.48)}


def _higuchi_fd(x, k_max=8):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 16 or not np.isfinite(x).all():
        return np.nan
    k_max = int(min(k_max, n // 4))
    if k_max < 2:
        return np.nan
    L = []
    for k in range(1, k_max + 1):
        Lk = 0.0
        for m in range(k):
            idxs = np.arange(m, n, k)
            if idxs.size < 2:
                continue
            diffs = np.abs(np.diff(x[idxs]))
            norm = (n - 1) / (float(idxs.size - 1) * k)
            Lmk = diffs.sum() * norm / k
            Lk += Lmk
        if k > 0:
            L.append(Lk / k)
    if len(L) < 2:
        return np.nan
    xs = np.log(1.0 / np.arange(1, len(L) + 1, dtype=float))
    ys = np.log(np.asarray(L, dtype=float))
    xm, ym = xs.mean(), ys.mean()
    d = ((xs - xm) ** 2).sum()
    if d <= 0:
        return np.nan
    return float(((xs - xm) * (ys - ym)).sum() / d)


def gen_TV_FractalDimension_Break(df, fd_len=64, fd_thresh=1.45, mom_len=10, **kw):
    c = df['close'].astype(float)
    fd = _rolling_apply(c, int(fd_len), _higuchi_fd).shift(1)
    mom = c - c.shift(int(mom_len))
    break_ = fd < float(fd_thresh)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(break_ & (mom > 0)).fillna(False)] = 1
    sig[(break_ & (mom < 0)).fillna(False)] = -1
    return sig


def space_TV_FractalDimension_Break():
    return {'fd_len': ('int', 32, 200), 'fd_thresh': ('float', 1.20, 1.60),
            'mom_len': ('int', 3, 40)}


def gen_TV_ZigZag_Swing(df, pct=3.0, **kw):
    c = df['close'].to_numpy(dtype=float)
    n = c.size
    sig = np.zeros(n, dtype=int)
    if n < 2 or not np.isfinite(c).all():
        return pd.Series(sig, index=df.index, dtype=int)
    thr = float(pct) / 100.0
    last_pivot_price = c[0]
    direction = 0
    prev_pivot_price = np.nan
    for i in range(1, n):
        ci = c[i]
        if direction == 0:
            if ci >= last_pivot_price * (1.0 + thr):
                direction = 1
                prev_pivot_price = last_pivot_price
                last_pivot_price = ci
            elif ci <= last_pivot_price * (1.0 - thr):
                direction = -1
                prev_pivot_price = last_pivot_price
                last_pivot_price = ci
        elif direction == 1:
            if ci > last_pivot_price:
                last_pivot_price = ci
            elif ci <= last_pivot_price * (1.0 - thr):
                if i + 1 < n and np.isfinite(prev_pivot_price) and last_pivot_price < prev_pivot_price:
                    sig[i + 1] = -1
                prev_pivot_price = last_pivot_price
                last_pivot_price = ci
                direction = -1
        else:
            if ci < last_pivot_price:
                last_pivot_price = ci
            elif ci >= last_pivot_price * (1.0 + thr):
                if i + 1 < n and np.isfinite(prev_pivot_price) and last_pivot_price > prev_pivot_price:
                    sig[i + 1] = 1
                prev_pivot_price = last_pivot_price
                last_pivot_price = ci
                direction = 1
    return pd.Series(sig, index=df.index, dtype=int)


def space_TV_ZigZag_Swing():
    return {'pct': ('float', 0.5, 10.0)}


def gen_TV_Bill_Williams_Fractal(df, confirm_bars=2, **kw):
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    bullish = ((l.shift(2) < l.shift(4)) & (l.shift(2) < l.shift(3)) &
               (l.shift(2) < l.shift(1)) & (l.shift(2) < l))
    bearish = ((h.shift(2) > h.shift(4)) & (h.shift(2) > h.shift(3)) &
               (h.shift(2) > h.shift(1)) & (h.shift(2) > h))
    frac_high = h.shift(2).where(bearish)
    frac_low = l.shift(2).where(bullish)
    cb = int(confirm_bars)
    rolling_high = frac_high.rolling(cb + 1, min_periods=1).max()
    rolling_low = frac_low.rolling(cb + 1, min_periods=1).min()
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(c > rolling_high).fillna(False)] = 1
    sig[(c < rolling_low).fillna(False)] = -1
    return sig


def space_TV_Bill_Williams_Fractal():
    return {'confirm_bars': ('int', 1, 10)}


STRATEGY_EXPORT = {
    'TV_Hurst_Regime_Switch': {'gen': gen_TV_Hurst_Regime_Switch, 'space': space_TV_Hurst_Regime_Switch,
                                'source': 'https://en.wikipedia.org/wiki/Hurst_exponent'},
    'TV_Detrended_Fluctuation': {'gen': gen_TV_Detrended_Fluctuation, 'space': space_TV_Detrended_Fluctuation,
                                  'source': 'https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis'},
    'TV_FractalDimension_Break': {'gen': gen_TV_FractalDimension_Break, 'space': space_TV_FractalDimension_Break,
                                   'source': 'https://en.wikipedia.org/wiki/Higuchi_fractal_dimension'},
    'TV_ZigZag_Swing': {'gen': gen_TV_ZigZag_Swing, 'space': space_TV_ZigZag_Swing,
                        'source': 'https://www.tradingview.com/support/solutions/43000591664-zig-zag/'},
    'TV_Bill_Williams_Fractal': {'gen': gen_TV_Bill_Williams_Fractal, 'space': space_TV_Bill_Williams_Fractal,
                                  'source': 'https://www.tradingview.com/support/solutions/43000591663-williams-fractal/'},
}
